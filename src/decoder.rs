//! JPEG XS pixel decoder — round 5.
//!
//! Wires the rounds 1–4 marker / DWT / entropy / quant pieces together
//! with the round-5 multi-component slice walker, Annex F inverse
//! colour transform, and Annex G output mapping, into a working
//! `Decoder` for the multi-component, single-precinct-row subset of
//! the standard:
//!
//! * `Nc ∈ {1, 2, 3, 4}`, sub-sampling factors `sx, sy ∈ {1, 2}` per
//!   component.
//! * `Cw == 0` (one precinct per row of the picture).
//! * `Cpih ∈ {0, 1}`. `Cpih == 3` (Star-Tetrix) needs CTS+CRG marker
//!   parsing and is round 6.
//! * `Qpih ∈ {0, 1}` (deadzone or uniform inverse quantizer).
//! * `Fq ∈ {0, 8}` per Table A.8 (lossless / regular).
//! * NLT marker present → quadratic / extended output scaling
//!   (Annex G.4 / G.5) is wired but the round-5 fixtures cover the
//!   linear (no-NLT) path. The other paths are unit-tested in
//!   [`crate::output`].
//! * 8-bit output (`B[i] == 8`); higher bit depths return `Unsupported`
//!   from the output mapper.
//!
//! Anything outside this subset returns `Error::Unsupported`.

use std::collections::VecDeque;

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, VideoFrame,
};

use crate::codestream;
use crate::colour_transform::{inverse_rct, inverse_star_tetrix};
use crate::crg::{cfa_pattern_type, parse_crg};
use crate::cts::parse_cts;
use crate::dequant::dequantize_precinct;
use crate::dwt::{inverse_2d, inverse_cascade_2d};
use crate::entropy::packet_body::PrecinctState;
use crate::entropy::{
    decode_packet_body, parse_packet_header, parse_precinct_header, precinct_truncation,
    BandCoefficients, PrecinctHeader,
};
use crate::output::{apply_output_scaling, parse_nlt};
use crate::slice_walker::{build_plan, PicturePlan, PrecinctPlan};

/// Build a JPEG XS decoder. Round 5 accepts the multi-component
/// single-precinct-row subset.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let codec_id = params.codec_id.clone();
    Ok(Box::new(JpegXsDecoder {
        codec_id,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct JpegXsDecoder {
    codec_id: CodecId,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Decoder for JpegXsDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // JPEG XS is intra-only and one packet == one codestream. We
        // simply queue it for `receive_frame` to pop.
        self.pending.push_back(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.pop_front() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_codestream(&pkt.data, pkt.pts)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

/// Decode a single JPEG XS codestream into a [`VideoFrame`].
fn decode_codestream(buf: &[u8], pts: Option<i64>) -> Result<VideoFrame> {
    let cs = codestream::parse(buf)?;

    let pih = cs.pih;
    let cdt = cs.cdt.clone();
    let wgt = cs.wgt.clone();

    if pih.qpih > 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: Qpih == {} reserved for ISO/IEC use (Table A.10)",
            pih.qpih
        )));
    }
    if cs.cwd.is_some() {
        return Err(Error::Unsupported(
            "jpegxs decoder: CWD (component-dependent wavelet decomposition) not supported".into(),
        ));
    }
    // Annex F.2 hard requirements.
    if pih.cpih == 1 {
        if pih.nc < 3 {
            return Err(Error::invalid(
                "jpegxs: Cpih=1 (RCT) requires Nc >= 3".to_string(),
            ));
        }
        for (i, c) in cdt.components.iter().enumerate().take(3) {
            if c.sx != 1 || c.sy != 1 {
                return Err(Error::invalid(format!(
                    "jpegxs: Cpih=1 (RCT) requires sx[i]=sy[i]=1 for i<3, got component {i} sx={} sy={}",
                    c.sx, c.sy
                )));
            }
        }
    }

    // Parse optional NLT body (Annex A.4.6).
    let nlt = match cs.nlt.as_deref() {
        Some(body) => Some(parse_nlt(body)?),
        None => None,
    };

    let (plan, _weights) = build_plan(&pih, &cdt, &wgt)?;

    let multi_level = pih.nlx > 1 || pih.nly > 1;

    // Allocate per-component sample buffers sized at Wc[i] × Hc[i].
    let wf = pih.wf as usize;
    let hf = pih.hf as usize;
    let mut samples: Vec<Vec<i32>> = Vec::with_capacity(plan.nc as usize);
    let mut comp_dims: Vec<(usize, usize)> = Vec::with_capacity(plan.nc as usize);
    for c in &cdt.components {
        let wc = wf / (c.sx as usize);
        let hc = hf / (c.sy as usize);
        samples.push(vec![0i32; wc * hc]);
        comp_dims.push((wc, hc));
    }

    // For multi-level cascade we gather all band coefficients into
    // per-component, per-band picture-level arrays first, then run
    // [`inverse_cascade_2d`] once per component. That avoids any
    // cross-precinct vertical-prediction state because the cascade
    // sees the entire picture's band data at once. Single-level paths
    // still go through the streaming per-precinct synthesis kept as a
    // fast path. `gathered[i][β]` is the picture-level band buffer for
    // component i, filter type β.
    let mut gathered: Vec<Vec<Vec<i32>>> = Vec::with_capacity(plan.nc as usize);
    if multi_level {
        for (i, c) in cdt.components.iter().enumerate() {
            let wc = wf / (c.sx as usize);
            let hc = hf / (c.sy as usize);
            let nlx_i = pih.nlx;
            // For sub-sampled components in multi-level we mirror the
            // single-level path: drop vertical levels by log2(sy[i]).
            let nly_i = pih.nly.saturating_sub(match c.sy {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            let nb = beta_count(nlx_i, nly_i) as usize;
            let mut bands_i: Vec<Vec<i32>> = Vec::with_capacity(nb);
            for beta in 0..nb as u32 {
                let (bw, bh) = band_dims(wc, hc, nlx_i, nly_i, beta);
                bands_i.push(vec![0i32; bw * bh]);
            }
            let _ = i;
            gathered.push(bands_i);
        }
    }

    // Walk slices in order. Each slice contributes a contiguous run of
    // precincts that span the picture width.
    for (slice_idx, slice_plan) in plan.slices.iter().enumerate() {
        let slice = cs.slices.get(slice_idx).ok_or_else(|| {
            Error::invalid(format!(
                "jpegxs decoder: codestream has fewer slices ({}) than the plan expects",
                cs.slices.len()
            ))
        })?;
        let slice_data = buf
            .get(slice.data_offset..slice.data_offset + slice.data_length)
            .ok_or_else(|| Error::invalid("jpegxs decoder: slice data range past buffer end"))?;
        decode_slice(
            slice_data,
            slice_plan,
            &plan,
            &pih,
            &cdt,
            &mut samples,
            if multi_level {
                Some(&mut gathered)
            } else {
                None
            },
        )?;
    }

    if multi_level {
        // Run the inverse-DWT cascade per component now that all band
        // coefficients have been gathered.
        for (i, c) in cdt.components.iter().enumerate() {
            let wc = wf / (c.sx as usize);
            let hc = hf / (c.sy as usize);
            let nlx_i = pih.nlx;
            let nly_i = pih.nly.saturating_sub(match c.sy {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            inverse_cascade_2d(wc, hc, nlx_i, nly_i, &gathered[i], &mut samples[i])?;
        }
    }

    // Annex F inverse colour transform.
    if pih.cpih == 1 {
        let mut refs: Vec<&mut [i32]> = samples.iter_mut().map(|p| p.as_mut_slice()).collect();
        inverse_rct(&mut refs, wf, hf)?;
    } else if pih.cpih == 3 {
        // Star-Tetrix needs the CTS marker (chroma exponents + Cf) and
        // CRG marker (CFA pattern type) per Annex F.5 / Tables F.9 /
        // F.10. The codestream parser already enforced "Cpih=3 → CTS
        // present", but CRG is also mandatory in this case (§A.4.9).
        let cts_body = cs
            .cts
            .as_deref()
            .ok_or_else(|| Error::invalid("jpegxs Cpih=3: CTS marker required (A.4.8)"))?;
        let cts = parse_cts(cts_body)?;
        let crg_body = cs
            .crg
            .as_deref()
            .ok_or_else(|| Error::invalid("jpegxs Cpih=3: CRG marker required (A.4.9)"))?;
        let crg = parse_crg(crg_body, pih.nc)?;
        let ct = cfa_pattern_type(&crg).ok_or_else(|| {
            Error::invalid(
                "jpegxs Cpih=3: CRG entries do not match a Table F.9 CFA pattern (RGGB/BGGR/GRBG/GBRG)",
            )
        })?;
        if pih.nc != 4 {
            return Err(Error::invalid(format!(
                "jpegxs Cpih=3: Star-Tetrix requires Nc=4, got {}",
                pih.nc
            )));
        }
        let mut refs: Vec<&mut [i32]> = samples.iter_mut().map(|p| p.as_mut_slice()).collect();
        inverse_star_tetrix(&mut refs, wf, hf, cts.e1, cts.e2, ct, cts.cf.cf())?;
    }

    // Annex G output scaling, DC level shift, clipping per component.
    let mut planes = Vec::with_capacity(plan.nc as usize);
    for (i, comp) in cdt.components.iter().enumerate() {
        let (wc, hc) = comp_dims[i];
        let bytes = apply_output_scaling(&samples[i], pih.bw, comp.bit_depth, nlt)?;
        let _ = hc;
        planes.push(VideoPlane {
            stride: wc,
            data: bytes,
        });
    }

    Ok(VideoFrame { pts, planes })
}

#[allow(clippy::too_many_arguments)]
fn decode_slice(
    slice_data: &[u8],
    slice_plan: &crate::slice_walker::SlicePlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    cdt: &crate::component_table::ComponentTable,
    samples: &mut [Vec<i32>],
    mut gathered: Option<&mut Vec<Vec<Vec<i32>>>>,
) -> Result<()> {
    let mut cursor = 0usize;
    for precinct_plan in &slice_plan.precincts {
        let pdata = slice_data
            .get(cursor..)
            .ok_or_else(|| Error::invalid("jpegxs decoder: precinct cursor past slice end"))?;

        // Precinct header.
        let precinct_header = parse_precinct_header(pdata, &precinct_plan.geometry)?;
        let header_bytes = precinct_header.header_bytes;
        let entropy_start = cursor + header_bytes;
        let entropy_end = entropy_start + (precinct_header.lprc as usize);
        if entropy_end > slice_data.len() {
            return Err(Error::invalid(format!(
                "jpegxs decoder: precinct lprc={} runs past slice data ({} bytes left)",
                precinct_header.lprc,
                slice_data.len() - entropy_start
            )));
        }
        let mut entropy_cursor = entropy_start;
        let mut state = PrecinctState::default();

        for packet_layout in precinct_plan.packets.iter() {
            if packet_layout.entries.is_empty() {
                continue;
            }
            let pktdata = slice_data
                .get(entropy_cursor..entropy_end)
                .ok_or_else(|| Error::invalid("jpegxs decoder: packet cursor past precinct end"))?;
            let packet_header = parse_packet_header(pktdata, &precinct_plan.geometry)?;
            entropy_cursor += packet_header.header_bytes;

            let body = slice_data
                .get(entropy_cursor..entropy_end)
                .ok_or_else(|| Error::invalid("jpegxs decoder: packet body past precinct end"))?;
            let dec = decode_packet_body(
                body,
                &precinct_plan.geometry,
                &precinct_header,
                &packet_header,
                packet_layout,
                &mut state,
            )?;
            entropy_cursor += dec.bytes_consumed;
        }

        // Skip precinct filler bytes up to Lprc.
        cursor = entropy_end;

        // Inverse-quantize. For single-level pictures, also DWT-
        // synthesise the precinct in place. For multi-level pictures,
        // accumulate band coefficients into the picture-level gather
        // buffer; the cascade runs after all precincts are processed.
        if let Some(g) = gathered.as_deref_mut() {
            gather_precinct(
                precinct_plan,
                plan,
                pih,
                cdt,
                &state.coefficients,
                &precinct_header,
                g,
            )?;
        } else {
            synthesise_precinct(
                precinct_plan,
                plan,
                pih,
                cdt,
                &state.coefficients,
                &precinct_header,
                samples,
            )?;
        }
    }
    Ok(())
}

/// Multi-level path — copy this precinct's dequantized band data into
/// the picture-level gather buffers `gathered[i][β]`. The cascade runs
/// later in [`decode_codestream`] once every precinct has contributed.
#[allow(clippy::too_many_arguments)]
fn gather_precinct(
    precinct_plan: &PrecinctPlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    cdt: &crate::component_table::ComponentTable,
    bands: &[BandCoefficients],
    precinct_header: &PrecinctHeader,
    gathered: &mut [Vec<Vec<i32>>],
) -> Result<()> {
    let trunc = precinct_truncation(&precinct_plan.geometry, precinct_header);
    let dequant = dequantize_precinct(pih.qpih, &precinct_plan.geometry, &trunc, bands);

    let nc = plan.nc as u32;
    let nbeta = plan.n_beta;
    let py = precinct_plan.p as usize; // np_x == 1 → p = py.
    let nly_pic = pih.nly;

    for (i, c) in cdt.components.iter().enumerate().take(nc as usize) {
        let sy_i = c.sy;
        let nly_i = nly_pic.saturating_sub(match sy_i {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 0,
        });
        let nb_i = beta_count(pih.nlx, nly_i) as u32;
        for beta in 0..nbeta.min(nb_i) {
            let b = (nc * beta + i as u32) as usize;
            let band_geom = &precinct_plan.geometry.bands[b];
            if !band_geom.exists {
                continue;
            }
            let lines = (band_geom.l1 - band_geom.l0) as usize;
            if lines == 0 {
                continue;
            }
            let wpb = band_geom.wpb as usize;
            // Picture-level band dimensions for this (β, i).
            let wc = (pih.wf as usize) / (c.sx as usize);
            let hc = (pih.hf as usize) / (sy_i as usize);
            let (pic_bw, pic_bh) = band_dims(wc, hc, pih.nlx, nly_i, beta);
            // Map this precinct's band-line slice [0..lines) into the
            // picture-level band rows. The first picture-band-line for
            // precinct py is `py * (pow)`, where pow = 2^max(NL,y - dy, 0).
            // Equivalently, lines per precinct = (L1 - L0); the row
            // offset in the picture-level band = py * (lines per
            // precinct from L0 alignment), but L0 is constant across
            // precincts so we use py * pow. We recover pow from
            // L1 - L0 + (anything truncated by Hb — handled by saturating
            // the picture row at pic_bh).
            //
            // For the standard square cascade (NL,x = NL,y = N) and
            // NL,y > 0, every band has pow = 2^max(N - dy, 0). The
            // precinct contains exactly `pow` band-lines (or fewer at
            // the picture's bottom edge), so picture-row = py * pow + λ
            // where λ ∈ [0, lines).
            let pow = pic_bh.div_ceil(
                plan.slices
                    .iter()
                    .map(|s| s.n_precincts)
                    .sum::<u32>()
                    .max(1) as usize,
            );
            let _ = pow;
            // Better: derive pow from the non-truncated case — the
            // first precinct should have lines == pow, so we can just
            // use the first precinct's `lines` directly via the band
            // geometry's pow encoded as `min(pow, hb_remaining)`. For
            // py = 0 it's `min(pow, hb)` = pow when hb >= pow, but
            // when hb < pow it's hb. Since we want the offset into the
            // picture-band, we compute it directly from band geometry.
            let pow_h = cascade_band_pow_h(pih.nlx, nly_i, beta, hc);
            let row_offset = py * pow_h;
            let band_buf = &mut gathered[i][beta as usize];
            if band_buf.len() != pic_bw * pic_bh {
                return Err(Error::invalid(format!(
                    "jpegxs decoder gather: band buffer for comp {i} β={beta} sized {} != {}*{}",
                    band_buf.len(),
                    pic_bw,
                    pic_bh
                )));
            }
            for line in 0..lines {
                let pic_row = row_offset + line;
                if pic_row >= pic_bh {
                    break;
                }
                let dst = &mut band_buf[pic_row * pic_bw..pic_row * pic_bw + wpb.min(pic_bw)];
                let src = &dequant[b][line * wpb..line * wpb + wpb.min(pic_bw)];
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(())
}

/// Compute the precinct height in band-lines for filter type `beta`.
/// Mirrors `2^max(NL,y - dy, 0)`. Used to figure out the picture-row
/// offset for a precinct's band slice.
fn cascade_band_pow_h(nlx: u8, nly: u8, beta: u32, _hc: usize) -> usize {
    let key = beta_key_for(beta, nlx, nly);
    let nly_u = nly as u32;
    let dy = key.dy;
    if dy >= nly_u || nly_u == 0 {
        1
    } else {
        1usize << (nly_u - dy)
    }
}

/// Helper: forward the (dx, dy, τx, τy) computation by inlining the
/// same algorithm as [`crate::dwt`] / the slice walker. Kept private to
/// the decoder so we don't add a load-bearing crate-internal API.
struct DecoderBandKey {
    #[allow(dead_code)]
    dx: u32,
    dy: u32,
    #[allow(dead_code)]
    tau_x: bool,
    #[allow(dead_code)]
    tau_y: bool,
}

fn beta_key_for(beta: u32, nlx: u8, nly: u8) -> DecoderBandKey {
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    if nly_u == 0 {
        if beta == 0 {
            return DecoderBandKey {
                dx: nlx_u,
                dy: 0,
                tau_x: false,
                tau_y: false,
            };
        }
        return DecoderBandKey {
            dx: nlx_u + 1 - beta,
            dy: 0,
            tau_x: true,
            tau_y: false,
        };
    }
    let beta1 = nlx_u - nly_u + 1;
    if beta < beta1 {
        if beta == 0 {
            return DecoderBandKey {
                dx: nlx_u,
                dy: nly_u,
                tau_x: false,
                tau_y: false,
            };
        }
        return DecoderBandKey {
            dx: nlx_u + 1 - beta,
            dy: nly_u,
            tau_x: true,
            tau_y: false,
        };
    }
    let group_in = beta - beta1;
    let triple = group_in / 3;
    let within = group_in % 3;
    let dy = nly_u - triple;
    let dx = dy;
    match within {
        0 => DecoderBandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: false,
        },
        1 => DecoderBandKey {
            dx,
            dy,
            tau_x: false,
            tau_y: true,
        },
        _ => DecoderBandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: true,
        },
    }
}

/// Number of filter types `Nβ` for a (NL,x, NL,y) decomposition.
fn beta_count(nlx: u8, nly: u8) -> usize {
    let mn = nlx.min(nly) as usize;
    let mx = nlx.max(nly) as usize;
    2 * mn + mx + 1
}

/// Picture-level dimensions of band β under (NL,x, NL,y) for a
/// component sized `wc × hc`. Mirrors the slice walker formula.
fn band_dims(wc: usize, hc: usize, nlx: u8, nly: u8, beta: u32) -> (usize, usize) {
    let key = beta_key_for(beta, nlx, nly);
    let dx = key.dx;
    let dy = key.dy;
    let tx = key.tau_x;
    let ty = key.tau_y;
    let w = if !tx {
        if dx == 0 {
            wc as u32
        } else {
            ((wc as u32) + (1u32 << dx) - 1) >> dx
        }
    } else {
        let denom_minus1 = if dx == 0 { 1 } else { 1u32 << (dx - 1) };
        (wc as u32).div_ceil(denom_minus1) / 2
    };
    let h = if !ty {
        if dy == 0 {
            hc as u32
        } else {
            ((hc as u32) + (1u32 << dy) - 1) >> dy
        }
    } else {
        let denom_minus1 = if dy == 0 { 1 } else { 1u32 << (dy - 1) };
        (hc as u32).div_ceil(denom_minus1) / 2
    };
    (w as usize, h as usize)
}

fn synthesise_precinct(
    precinct_plan: &PrecinctPlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    cdt: &crate::component_table::ComponentTable,
    bands: &[BandCoefficients],
    precinct_header: &PrecinctHeader,
    samples: &mut [Vec<i32>],
) -> Result<()> {
    let trunc = precinct_truncation(&precinct_plan.geometry, precinct_header);
    let dequant = dequantize_precinct(pih.qpih, &precinct_plan.geometry, &trunc, bands);

    let nlx = plan.nlx as u32;
    let nly = plan.nly as u32;
    let nbeta = plan.n_beta;
    let nc = plan.nc as u32;
    // Round 5: Per-component synthesis. For each component i, gather
    // the (LL, HL, LH, HH) bands of that component and do a single-
    // level inverse 2-D DWT (or the corresponding 1-D variant for
    // NL,y == 0). Multi-level cascade arrives in round 6.

    let py = precinct_plan.p as usize; // np_x == 1
    let wp = precinct_plan.wp as usize;
    let hp = precinct_plan.hp as usize;

    for (i, samples_i) in samples.iter_mut().enumerate().take(nc as usize) {
        let comp = cdt.components[i];
        let sx_i = comp.sx as usize;
        let sy_i = comp.sy as usize;
        let wc_i = (pih.wf as usize) / sx_i;
        let _hc_i = (pih.hf as usize) / sy_i;
        let wp_i = wp / sx_i; // per-component precinct width
        let hp_i = hp / sy_i; // per-component precinct height

        // Gather band-id of each (β, i) pair via the band index formula
        // b = nc * β + i.
        let band_id = |beta: u32| -> usize { (nc * beta + i as u32) as usize };

        if nlx == 0 && nly == 0 {
            // No DWT — band 0 is the raw component samples for this precinct.
            let b = band_id(0);
            let band_geom = &precinct_plan.geometry.bands[b];
            if !band_geom.exists {
                continue;
            }
            let band_samples = &dequant[b];
            let row_offset = py * hp_i;
            let lines = (band_geom.l1 - band_geom.l0) as usize;
            for line in 0..lines {
                let target_row = row_offset + line;
                if target_row >= samples_i.len() / wc_i {
                    break;
                }
                let dst = &mut samples_i[target_row * wc_i..target_row * wc_i + wp_i];
                let src = &band_samples
                    [line * (band_geom.wpb as usize)..line * (band_geom.wpb as usize) + wp_i];
                dst.copy_from_slice(src);
            }
            continue;
        }

        if nlx == 1 && nly == 1 {
            // Per-component 4-band inverse 2-D DWT.
            // For sub-sampled components (e.g. 4:2:0 chroma at sy=2),
            // the per-component effective vertical decomposition level
            // is N'L,y[i] = NL,y - log2(sy[i]) = 0 — i.e. the LH/HH
            // bands are absent. We handle that with the NLY=0 path
            // below.
            let nly_i = if sy_i == 2 { 0 } else { 1 };
            if nly_i == 0 {
                // Only the LL and HL bands exist — single-row 1-D
                // horizontal inverse synthesis (same as the NL,y == 0,
                // NL,x == 1 case).
                inverse_synth_1d(
                    precinct_plan,
                    band_id,
                    nbeta,
                    &dequant,
                    py,
                    hp_i,
                    wp_i,
                    wc_i,
                    samples_i,
                )?;
            } else {
                // Standard 4-band 2-D synthesis.
                let b_ll = band_id(0);
                let b_hl = band_id(1);
                let b_lh = band_id(2);
                let b_hh = band_id(3);
                if !precinct_plan.geometry.bands[b_ll].exists {
                    continue;
                }
                let ll = &dequant[b_ll];
                let hl = &dequant[b_hl];
                let lh = &dequant[b_lh];
                let hh = &dequant[b_hh];
                let mut out = vec![0i32; wp_i * hp_i];
                inverse_2d(wp_i, hp_i, ll, hl, lh, hh, &mut out)?;
                let row_offset = py * hp_i;
                for line in 0..hp_i {
                    let target_row = row_offset + line;
                    if target_row >= samples_i.len() / wc_i {
                        break;
                    }
                    let dst = &mut samples_i[target_row * wc_i..target_row * wc_i + wp_i];
                    let src = &out[line * wp_i..line * wp_i + wp_i];
                    dst.copy_from_slice(src);
                }
            }
            continue;
        }

        if nlx == 1 && nly == 0 {
            inverse_synth_1d(
                precinct_plan,
                band_id,
                nbeta,
                &dequant,
                py,
                hp_i,
                wp_i,
                wc_i,
                samples_i,
            )?;
            continue;
        }

        return Err(Error::Unsupported(format!(
            "jpegxs decoder: NL,x={nlx} NL,y={nly} not implemented in round 5"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inverse_synth_1d(
    precinct_plan: &PrecinctPlan,
    band_id: impl Fn(u32) -> usize,
    _nbeta: u32,
    dequant: &[Vec<i32>],
    py: usize,
    hp_i: usize,
    wp_i: usize,
    wc_i: usize,
    samples_i: &mut [i32],
) -> Result<()> {
    let b_ll = band_id(0);
    let b_hl = band_id(1);
    let ll_band = &precinct_plan.geometry.bands[b_ll];
    let hl_band = &precinct_plan.geometry.bands[b_hl];
    if !ll_band.exists {
        return Ok(());
    }
    let ll = &dequant[b_ll];
    let hl = &dequant[b_hl];
    let lines = (ll_band.l1 - ll_band.l0) as usize;
    let row_offset = py * hp_i;
    let mut row = vec![0i32; wp_i];
    for line in 0..lines {
        let low = &ll[line * (ll_band.wpb as usize)..(line + 1) * (ll_band.wpb as usize)];
        let high = &hl[line * (hl_band.wpb as usize)..(line + 1) * (hl_band.wpb as usize)];
        crate::dwt::inverse_horizontal_1d(low, high, &mut row)?;
        let target_row = row_offset + line;
        if target_row >= samples_i.len() / wc_i {
            break;
        }
        let dst = &mut samples_i[target_row * wc_i..target_row * wc_i + wp_i];
        dst.copy_from_slice(&row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, TimeBase};

    #[test]
    #[ignore]
    fn debug_multilevel_layout() {
        use crate::component_table::{Component, ComponentTable};
        use crate::picture_header::PictureHeader;
        use crate::slice_walker::build_plan;
        let pih = PictureHeader {
            lcod: 0,
            ppih: 0,
            plev: 0,
            wf: 4,
            hf: 4,
            cw: 0,
            hsl: 1,
            nc: 1,
            ng: 4,
            ss: 8,
            bw: 20,
            fq: 8,
            br: 4,
            fslc: 0,
            ppoc: 0,
            cpih: 0,
            nlx: 2,
            nly: 2,
            lh: 0,
            rl: 0,
            qpih: 0,
            fs: 0,
            rm: 0,
        };
        let cdt = ComponentTable {
            components: vec![Component {
                bit_depth: 8,
                sx: 1,
                sy: 1,
            }],
        };
        let wgt = vec![0u8; 14];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).unwrap();
        eprintln!("nbeta={} nbands={}", plan.n_beta, plan.n_bands);
        for s in &plan.slices {
            for p in &s.precincts {
                eprintln!("Precinct p={} packets={}", p.p, p.packets.len());
                for (i, b) in p.geometry.bands.iter().enumerate() {
                    eprintln!(
                        "  band[{i}] wpb={} l0={} l1={} exists={}",
                        b.wpb, b.l0, b.l1, b.exists
                    );
                }
                for (i, pkt) in p.packets.iter().enumerate() {
                    let entries: Vec<_> = pkt
                        .entries
                        .iter()
                        .map(|e| format!("(b={} l={})", e.band, e.line))
                        .collect();
                    eprintln!("  packet[{i}] {}", entries.join(","));
                }
            }
        }
    }

    #[test]
    fn factory_returns_decoder() {
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let dec = make_decoder(&params).expect("decoder factory");
        assert_eq!(dec.codec_id().as_str(), crate::CODEC_ID_STR);
    }

    #[test]
    fn need_more_before_packet() {
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let res = dec.receive_frame();
        assert!(matches!(res, Err(Error::NeedMore)));
    }

    /// Build the minimum-size single-component JPEG XS codestream with
    /// `NL,x=1, NL,y=0`, a 4×1 image, and entropy data that sets every
    /// quantization-index magnitude to zero (M=0 for every code group).
    /// The resulting decoded plane is a single row of mid-grey samples
    /// (`2^(B-1) = 128`), proving the slice walker / precinct loop /
    /// entropy decoder / inverse DWT / output-clip pipeline is wired
    /// end to end.
    fn build_zero_codestream_4x1() -> Vec<u8> {
        let mut v = Vec::new();
        // SOC
        v.extend_from_slice(&[0xff, 0x10]);
        // CAP
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // PIH
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes()); // Lcod
        v.extend_from_slice(&0u16.to_be_bytes()); // Ppih
        v.extend_from_slice(&0u16.to_be_bytes()); // Plev
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(1); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x84); // Fq=8 | Br=4
        v.push(0x00); // Fslc/Ppoc/Cpih=0
        v.push(0x10); // NL,x=1, NL,y=0
        v.push(0x00); // Lh/Rl/Qpih/Fs/Rm
                      // CDT
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&6u16.to_be_bytes());
        v.extend_from_slice(&[0u8, 0, 0, 0]); // 2 bands
                                              // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // Precinct header (6 bytes): Lprc=12.
        v.extend_from_slice(&[0x00, 0x00, 12, 0, 0, 0x00]);
        // 2 packets, each 5-byte header + 1 byte body.
        let mut packet1_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet1_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        v.extend_from_slice(&packet1_hdr);
        v.push(0x00);
        v.extend_from_slice(&packet1_hdr);
        v.push(0x00);
        // EOC
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_zero_4x1_codestream() {
        let buf = build_zero_codestream_4x1();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("send_packet");
        let frame = dec.receive_frame().expect("receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 1);
        assert_eq!(vf.planes[0].stride, 4);
        assert_eq!(vf.planes[0].data.len(), 4);
        for (i, &px) in vf.planes[0].data.iter().enumerate() {
            assert_eq!(
                px, 128,
                "pixel {i}: expected 128 (mid-grey), got {px} (full row {:?})",
                vf.planes[0].data
            );
        }
    }

    fn build_zero_codestream_2x2() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.push(1);
        v.push(4);
        v.push(8);
        v.push(20);
        v.push(0x84);
        v.push(0x00);
        v.push(0x11);
        v.push(0x00);
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&10u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 8]);
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&[0x00, 0x00, 24, 0, 0, 0x00]);
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..4 {
            v.extend_from_slice(&packet_hdr);
            v.push(0x00);
        }
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_zero_2x2_codestream() {
        let buf = build_zero_codestream_2x2();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("send_packet 2x2");
        let frame = dec.receive_frame().expect("receive_frame 2x2");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 1);
        assert_eq!(vf.planes[0].stride, 2);
        assert_eq!(vf.planes[0].data.len(), 4);
        for &px in &vf.planes[0].data {
            assert_eq!(px, 128, "all-zero coefs should give flat 128");
        }
    }

    /// 4x1 image, lossless mode (Fq=0), LL = [1,1] HL = [0,0].
    /// Inverse 5/3 reconstructs [1,1,1,1]. After +DC bias = 129.
    fn build_constant_4x1_lossless() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.push(1);
        v.push(4);
        v.push(8);
        v.push(8);
        v.push(0x04); // Fq=0, Br=4
        v.push(0x00);
        v.push(0x10);
        v.push(0x00);
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&6u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 4]);
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        let mut payload = Vec::new();
        let mut bits1: u64 = 0;
        bits1 = (bits1 << 1) | 0;
        bits1 = (bits1 << 15) | 1;
        bits1 = (bits1 << 13) | 1;
        bits1 = (bits1 << 11) | 0;
        let mut hdr1 = vec![0u8; 5];
        for (i, b) in hdr1.iter_mut().enumerate() {
            *b = ((bits1 >> (8 * (4 - i))) & 0xff) as u8;
        }
        payload.extend_from_slice(&hdr1);
        payload.push(0b10000000);
        payload.push(0x0C);
        let mut bits2: u64 = 0;
        bits2 = (bits2 << 1) | 0;
        bits2 = (bits2 << 15) | 0;
        bits2 = (bits2 << 13) | 1;
        bits2 = (bits2 << 11) | 0;
        let mut hdr2 = vec![0u8; 5];
        for (i, b) in hdr2.iter_mut().enumerate() {
            *b = ((bits2 >> (8 * (4 - i))) & 0xff) as u8;
        }
        payload.extend_from_slice(&hdr2);
        payload.push(0x00);
        let lprc = payload.len() as u32;
        let mut prec_hdr = vec![0u8; 6];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_constant_4x1_lossless() {
        let buf = build_constant_4x1_lossless();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("send_packet const");
        let frame = dec.receive_frame().expect("receive_frame const");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes[0].data.len(), 4);
        assert_eq!(
            vf.planes[0].data,
            vec![129, 129, 129, 129],
            "non-zero LL coefficient should propagate through the inverse 5/3 DWT"
        );
    }

    /// 3-component 4:4:4 4×1 zero codestream — entropy data sets every
    /// magnitude to zero, so every component plane decodes to a flat
    /// row of mid-grey samples. With no inverse colour transform
    /// (Cpih=0), each plane sits at 128.
    fn build_zero_3comp_4x1() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf=4
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf=1
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(3); // Nc=3
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw=20
        v.push(0x84); // Fq=8 | Br=4
        v.push(0x00); // Cpih=0
        v.push(0x10); // NL,x=1 NL,y=0
        v.push(0x00);
        // CDT — 3 components, B[c]=8, sx=sy=1.
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&8u16.to_be_bytes()); // Lcdt = 2*Nc+2 = 8
        v.extend_from_slice(&[8, 0x11, 8, 0x11, 8, 0x11]);
        // WGT — 3 components × 2 bands = 6 bands × 2 = 12, +2 = 14.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&14u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 12]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // Precinct geometry: 3 comps × 2 βs (LL,HL) per comp = 6 bands.
        // For NL,x=1 NL,y=0, β1 = 1, so the first packet groups all 3
        // LL bands of all components on line 0, then 3 separate packets
        // for HL_0, HL_1, HL_2 on line 0. Total 4 packets.
        // Each packet: 5-byte short header + 1-byte body (single VLC '0'
        // for M=0).
        let mut payload = Vec::new();
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..4 {
            payload.extend_from_slice(&packet_hdr);
            payload.push(0x00);
        }
        let lprc = payload.len() as u32;
        // Precinct header bits: 24 (Lprc) + 8 (Q) + 8 (R) + 6×2 (D) =
        // 52 bits → 7 bytes after byte alignment.
        let mut prec_hdr = vec![0u8; 7];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_zero_3comp_4x1() {
        let buf = build_zero_3comp_4x1();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("3-comp send_packet");
        let frame = dec.receive_frame().expect("3-comp receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 3, "3-component output");
        for (i, plane) in vf.planes.iter().enumerate() {
            assert_eq!(plane.stride, 4, "component {i} stride");
            assert_eq!(plane.data.len(), 4);
            for (x, &px) in plane.data.iter().enumerate() {
                assert_eq!(px, 128, "comp {i} pixel {x}: expected mid-grey");
            }
        }
    }

    /// 3-component 4:4:4 4×1 RCT-decoded zero codestream. Every
    /// quantization-index magnitude is zero, so each O[c] plane is flat
    /// 0 in the wavelet domain. The inverse RCT then computes:
    ///   o1 = 0 - ((0 + 0) >> 2) = 0      (green)
    ///   o0 = 0 + 0 = 0                    (red)
    ///   o2 = 0 + 0 = 0                    (blue)
    /// → still flat zero, then DC bias of 128 → mid-grey on every plane.
    fn build_zero_3comp_rct_4x1() -> Vec<u8> {
        let mut v = build_zero_3comp_4x1();
        // Patch Cpih byte (PIH body offset 21 = absolute offset 6 + 4 +
        // 21 = 31). Actually let's locate the byte by searching the PIH
        // we just wrote. Easier: rewrite build for clarity. Instead,
        // patch Cpih = 1 by overwriting the byte at the known offset:
        // after SOC(2)+CAP(4)+PIH header(4)+22 = 32 -> body[21] is at
        // index 32 (0-based). Verify:
        //   v[0..2] = SOC; v[2..6] = CAP marker+len(2)+body(0);
        //   v[6..8] = PIH marker; v[8..10] = Lpih; v[10..34] = body[0..24].
        // body[21] is at v[31].
        v[31] = 0x01;
        v
    }

    #[test]
    fn end_to_end_decode_rct_zero_3comp_4x1() {
        let buf = build_zero_3comp_rct_4x1();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("RCT send_packet");
        let frame = dec.receive_frame().expect("RCT receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 3);
        // RCT of all-zero produces all-zero in all components → mid-grey.
        for (i, plane) in vf.planes.iter().enumerate() {
            for (x, &px) in plane.data.iter().enumerate() {
                assert_eq!(px, 128, "RCT comp {i} pixel {x}: expected mid-grey");
            }
        }
    }

    /// 3-component 4:2:2 4×1 image (luma 4×1, chroma 2×1 each), all
    /// quantization-index magnitudes zero, Cpih=0. Each plane decodes
    /// to mid-grey at its native width.
    fn build_zero_3comp_422_4x1() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf=4
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf=1
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.push(3);
        v.push(4);
        v.push(8);
        v.push(20);
        v.push(0x84);
        v.push(0x00);
        v.push(0x10); // NL,x=1, NL,y=0
        v.push(0x00);
        // CDT: comp 0 sx=1, comp 1/2 sx=2.
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&8u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11, 8, 0x21, 8, 0x21]);
        // WGT: 6 bands × 2 = 12, +2 = 14.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&14u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 12]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // 4 packets (LL all 3 + 3×HL packets), same payload as the 4:4:4
        // case.
        let mut payload = Vec::new();
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..4 {
            payload.extend_from_slice(&packet_hdr);
            payload.push(0x00);
        }
        let lprc = payload.len() as u32;
        let mut prec_hdr = vec![0u8; 7];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_zero_3comp_422_4x1() {
        let buf = build_zero_3comp_422_4x1();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("4:2:2 send_packet");
        let frame = dec.receive_frame().expect("4:2:2 receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 3);
        // Luma plane: 4 samples wide × 1 tall.
        assert_eq!(vf.planes[0].stride, 4);
        assert_eq!(vf.planes[0].data.len(), 4);
        // Chroma planes: 2 samples wide × 1 tall (sx=2 → Wc = 4/2 = 2).
        assert_eq!(vf.planes[1].stride, 2);
        assert_eq!(vf.planes[1].data.len(), 2);
        assert_eq!(vf.planes[2].stride, 2);
        assert_eq!(vf.planes[2].data.len(), 2);
        for (i, plane) in vf.planes.iter().enumerate() {
            for &px in &plane.data {
                assert_eq!(px, 128, "4:2:2 comp {i}: expected mid-grey");
            }
        }
    }

    /// Parse-only smoke test confirming the codestream parser hands an
    /// NLT body to the decoder which then routes through the quadratic
    /// output scaling path. The fixture itself has all-zero coefficients
    /// (Bw=18 is required for NLT per Table A.8); we only check the
    /// decoder accepts the codestream and emits a 4-byte plane.
    fn build_zero_with_nlt_quadratic_4x1() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        // CAP — bit 2 must be set per Table A.5/A.8 for NLT quadratic.
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&3u16.to_be_bytes()); // Lcap=3 → 1 byte cap[]
        v.push(0x20); // bit 2 set (counting MSB-first)
                      // PIH — Bw=18, Fq=6 per Table A.8.
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(1);
        v.push(4);
        v.push(8);
        v.push(18); // Bw=18
        v.push((6 << 4) | 4); // Fq=6, Br=4
        v.push(0x00);
        v.push(0x10); // NL,x=1, NL,y=0
        v.push(0x00);
        // CDT
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // NLT — Tnlt=1, σ=0, α=0 → DCO=0. Lnlt=5.
        v.extend_from_slice(&[0xff, 0x16]);
        v.extend_from_slice(&5u16.to_be_bytes());
        v.push(1);
        v.extend_from_slice(&0u16.to_be_bytes());
        // WGT
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&6u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 4]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // Precinct
        v.extend_from_slice(&[0x00, 0x00, 12, 0, 0, 0x00]);
        let mut packet1_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet1_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        v.extend_from_slice(&packet1_hdr);
        v.push(0x00);
        v.extend_from_slice(&packet1_hdr);
        v.push(0x00);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    /// Build a 4×4 single-component JPEG XS codestream with
    /// `NL,x = NL,y = 2` (multi-level cascade) and entropy data that
    /// sets every quantization-index magnitude to zero. The expected
    /// output is a single 4×4 plane of mid-grey samples (`128`).
    ///
    /// This is the minimum-viable multi-level fixture: 1 slice, 1
    /// precinct, 7 bands (Nβ = 7 for NL,x = NL,y = 2), 10 packets
    /// matching the layout the slice walker emits (verified via the
    /// `debug_multilevel_layout` test).
    fn build_zero_codestream_4x4_nl22() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf
        v.extend_from_slice(&4u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(1); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x84); // Fq=8, Br=4
        v.push(0x00); // Cpih=0
        v.push(0x22); // NL,x=2, NL,y=2
        v.push(0x00);
        // CDT: 1 component, B=8, sx=sy=1.
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT: 7 bands × 2 = 14, +2 = 16.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&16u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 14]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // 10 packets, each 5-byte short header + 1-byte body.
        let mut payload = Vec::new();
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..10 {
            payload.extend_from_slice(&packet_hdr);
            payload.push(0x00);
        }
        let lprc = payload.len() as u32;
        // Precinct header: Lprc(24) + Q(8) + R(8) + D[7](14) = 54 bits → 7 bytes.
        let mut prec_hdr = vec![0u8; 7];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_zero_4x4_nl22() {
        let buf = build_zero_codestream_4x4_nl22();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("multi-level send_packet");
        let frame = dec.receive_frame().expect("multi-level receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 1);
        assert_eq!(vf.planes[0].stride, 4);
        assert_eq!(vf.planes[0].data.len(), 16);
        for (i, &px) in vf.planes[0].data.iter().enumerate() {
            assert_eq!(
                px, 128,
                "pixel {i} should be mid-grey for all-zero coeffs through NL=2 cascade"
            );
        }
    }

    /// Build an 8×8 single-component JPEG XS codestream with
    /// `NL,x = NL,y = 3` (3-level cascade) and entropy data that sets
    /// every quantization-index magnitude to zero. Expected output is
    /// a flat 8×8 plane of mid-grey samples.
    fn build_zero_codestream_8x8_nl33() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&8u16.to_be_bytes()); // Wf
        v.extend_from_slice(&8u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(1); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x84);
        v.push(0x00);
        v.push(0x33); // NL,x=3, NL,y=3
        v.push(0x00);
        // CDT
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT — 10 bands × 2 = 20, +2 = 22.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&22u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 20]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // 22 packets × 6 bytes each.
        let mut payload = Vec::new();
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..22 {
            payload.extend_from_slice(&packet_hdr);
            payload.push(0x00);
        }
        let lprc = payload.len() as u32;
        // Precinct header: Lprc(24)+Q(8)+R(8)+D[10](20) = 60 bits → 8 bytes.
        let mut prec_hdr = vec![0u8; 8];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    /// Build a 4-component, 4×2 Star-Tetrix (`Cpih = 3`) codestream
    /// with single-level wavelet (`NL,x = NL,y = 1`), 4:4:4:4 sampling,
    /// CTS marker (`Cf = 0`, `e1 = 0`, `e2 = 0`), and CRG marker
    /// configured for the RGGB pattern (`Ct = 0`). All entropy data
    /// encodes magnitude zero. The decoder must accept the codestream,
    /// run the inverse Star-Tetrix transform, and emit four 4×2 planes.
    ///
    /// With all-zero wavelet coefficients, the inverse cascade yields
    /// flat-zero per-component planes. The Star-Tetrix lifting
    /// (Tables F.5/F.6/F.7/F.8) then operates on flat zeros: every
    /// average / delta / Y / CbCr lift adds floor(0/8) or floor(0/4)
    /// = 0, so the output stays flat zero. After the +DC bias and
    /// 8-bit clip from `apply_output_scaling`, every plane sits at 128.
    fn build_zero_star_tetrix_4comp_4x2() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]);
        // CAP — bit 1 (Star-Tetrix) set per A.5.
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&3u16.to_be_bytes());
        v.push(0x40); // bit 1 = 0x40
                      // PIH
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf
        v.extend_from_slice(&2u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(4); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x84);
        v.push(0x03); // Fslc=0,Ppoc=0,Cpih=3
        v.push(0x11); // NL,x=1,NL,y=1
        v.push(0x00);
        // CDT — 4 components 8-bit 4:4:4:4
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&10u16.to_be_bytes()); // 2*Nc + 2 = 10
        v.extend_from_slice(&[8, 0x11, 8, 0x11, 8, 0x11, 8, 0x11]);
        // CTS — Lcts=4, Cf=0, e1=0, e2=0
        v.extend_from_slice(&[0xff, 0x18]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[0x00, 0x00]);
        // CRG — Lcrg = 2 + 4*Nc = 18; RGGB (Ct=0).
        v.extend_from_slice(&[0xff, 0x19]);
        v.extend_from_slice(&18u16.to_be_bytes());
        for &(x, y) in &[(0u16, 0u16), (32768, 0), (0, 32768), (32768, 32768)] {
            v.extend_from_slice(&x.to_be_bytes());
            v.extend_from_slice(&y.to_be_bytes());
        }
        // WGT — 16 bands × 2 bytes = 32, +2 = 34.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&34u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 32]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // 13 packets × 6 bytes (5 hdr + 1 body).
        let mut payload = Vec::new();
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0;
        bits = (bits << 15) | 0;
        bits = (bits << 13) | 1;
        bits = (bits << 11) | 0;
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..13 {
            payload.extend_from_slice(&packet_hdr);
            payload.push(0x00);
        }
        let lprc = payload.len() as u32;
        // Precinct hdr: 24 + 8 + 8 + 16*2 = 72 bits → 9 bytes.
        let mut prec_hdr = vec![0u8; 9];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn end_to_end_decode_star_tetrix_4comp_4x2() {
        let buf = build_zero_star_tetrix_4comp_4x2();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("star-tetrix send_packet");
        let frame = dec.receive_frame().expect("star-tetrix receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(
            vf.planes.len(),
            4,
            "Star-Tetrix produces 4 component planes"
        );
        for (i, plane) in vf.planes.iter().enumerate() {
            assert_eq!(plane.stride, 4, "comp {i} stride");
            assert_eq!(plane.data.len(), 8, "comp {i} 4×2 plane");
            for (x, &px) in plane.data.iter().enumerate() {
                assert_eq!(
                    px, 128,
                    "comp {i} pixel {x}: all-zero coeffs through Star-Tetrix should give mid-grey"
                );
            }
        }
    }

    #[test]
    fn end_to_end_decode_zero_8x8_nl33() {
        let buf = build_zero_codestream_8x8_nl33();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("nl=3 send_packet");
        let frame = dec.receive_frame().expect("nl=3 receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes.len(), 1);
        assert_eq!(vf.planes[0].stride, 8);
        assert_eq!(vf.planes[0].data.len(), 64);
        for (i, &px) in vf.planes[0].data.iter().enumerate() {
            assert_eq!(
                px, 128,
                "pixel {i} should be mid-grey for all-zero coeffs through NL=3 cascade"
            );
        }
    }

    #[test]
    fn end_to_end_decode_with_nlt_marker_quadratic() {
        // Confirms the codestream parser captures the NLT marker, the
        // decoder dispatches to apply_output_scaling with the parsed
        // params, and the result is a valid plane.
        let buf = build_zero_with_nlt_quadratic_4x1();
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        dec.send_packet(&pkt).expect("NLT send_packet");
        let frame = dec.receive_frame().expect("NLT receive_frame");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(vf.planes[0].data.len(), 4);
        // For all-zero coefficients with quadratic NLT (Bw=18, B=8,
        // DCO=0): v = 0 + 2^17 = 131072; v² = 17_179_869_184; ζ = 28;
        // (v² >> 28) = 64. So output is 64.
        for &px in &vf.planes[0].data {
            assert_eq!(px, 64, "quadratic NLT all-zero output");
        }
    }
}
