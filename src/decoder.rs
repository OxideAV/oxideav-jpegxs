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
//! * Output `B[i] ∈ 8..=16` (round 118): `B[i] == 8` packs one byte per
//!   sample; `B[i] > 8` packs two little-endian bytes per sample (Annex G
//!   kernels are bit-depth agnostic). `B[i] > 16` returns `Unsupported`.
//!
//! Anything outside this subset returns `Error::Unsupported`.

use crate::codestream;
use crate::colour_transform::{inverse_rct, inverse_star_tetrix};
use crate::crg::cfa_pattern_type;
use crate::dequant::dequantize_precinct;
use crate::dwt::{inverse_2d, inverse_cascade_2d};
use crate::entropy::packet_body::{PrecinctState, PrecinctTop};
use crate::entropy::{
    check_raw_mode_consistency, decode_packet_body, parse_packet_header, parse_precinct_header,
    precinct_filler_bytes, precinct_truncation, BandCoefficients, PacketWireSize, PrecinctHeader,
};
use crate::error::{JpegXsError as Error, Result};
use crate::image::{JpegXsImage, JpegXsPlane as VideoPlane};
use crate::output::apply_output_scaling;
use crate::slice_walker::{PicturePlan, PrecinctPlan};

/// Decode a single JPEG XS codestream into a [`JpegXsImage`].
pub(crate) fn decode_codestream(buf: &[u8], pts: Option<i64>) -> Result<JpegXsImage> {
    let cs = codestream::parse(buf)?;

    let pih = cs.pih;
    let cdt = cs.cdt.clone();
    let wgt = cs.wgt.clone();

    // Profile / level conformance (ISO/IEC 21122-2:2019 Annex A). The
    // picture header carries a `Ppih` profile indicator (Table A.5) and a
    // `Plev` level/sublevel indicator (Tables A.12/A.13). A conforming
    // decoder validates the codestream against whatever it declares:
    //
    //  * `Ppih = 0x0000` is `Profile::Unrestricted` — no structural
    //    constraints, so the check is a no-op.
    //  * A non-zero `Ppih` that maps to a known profile (`from_ppih`)
    //    pins the component count, bit-depth set, chroma format,
    //    decomposition counts, `Qpih`, slice height and column mode the
    //    stream may use; [`crate::profile::check_codestream`] rejects a
    //    stream whose header contradicts its own profile claim.
    //  * A non-zero `Ppih` that maps to no profile is reserved for
    //    ISO/IEC — a value a conforming encoder cannot emit — so we
    //    reject it rather than decode under an unknown profile.
    //
    // [`crate::profile::check_level`] independently bounds the picture's
    // `Wf` / `Hf` / `Wf×Hf` against the declared level's `Wmax` / `Hmax`
    // / `Lmax` (Table A.6); a reserved `Plev` high byte is rejected.
    match crate::profile::Profile::from_ppih(pih.ppih) {
        Some(profile) => crate::profile::check_codestream(&cs, profile)?,
        None => {
            return Err(Error::invalid(format!(
                "jpegxs decoder: Ppih=0x{:04X} is reserved for ISO/IEC use (Table A.5)",
                pih.ppih
            )));
        }
    }
    crate::profile::check_level(&cs)?;
    // Sublevel coded-domain bound: the SOC-to-EOC codestream byte count
    // must not exceed Ssl,max = floor(Lmax × Nbpp / 8) for the declared
    // level + sublevel (§A.4.1, Tables A.8–A.11).
    crate::profile::check_codestream_size(&cs, buf.len())?;

    // Lcod conformance (ISO/IEC 21122-1:2022 Table 11). The picture
    // header's Lcod field is "the size of the entire codestream in bytes
    // from SOC to EOC, including all markers, if constant-bitrate coding
    // is used; 0 if variable-bitrate coding is used." When non-zero it is
    // a CBR self-description that must match the codestream's actual
    // SOC-to-EOC length — the EOC marker ends at `eoc_offset + 2` (the
    // marker is two bytes). A mismatch means the stream's own length
    // field disagrees with its byte layout, so a conforming decoder
    // rejects it rather than decode a stream that mis-describes itself.
    if pih.lcod != 0 {
        if let Some(eoc) = cs.eoc_offset {
            let actual = (eoc + 2) as u64;
            if pih.lcod as u64 != actual {
                return Err(Error::invalid(format!(
                    "jpegxs decoder: Lcod={} (declared CBR codestream length) does not match the \
                     actual SOC-to-EOC length of {actual} bytes (Table 11)",
                    pih.lcod
                )));
            }
        }
    }

    // Picture-dimension conformance (ISO/IEC 21122-1:2022 Table 11). The
    // Wf / Hf rows constrain the picture size against the declared
    // sub-sampling and decomposition depth:
    //
    //   Wf ≥ max_i(sx[i]) × 2^NL,x
    //   Hf ≥ max_i(sy[i]) × 2^NL,y
    //
    // i.e. the picture must be at least one fully-decomposed
    // low-frequency sample wide / tall in every component. A smaller
    // picture cannot carry the LL band the header claims, so the geometry
    // is internally inconsistent and the stream is rejected.
    let max_sx = cdt.components.iter().map(|c| c.sx).max().unwrap_or(1) as u32;
    let max_sy = cdt.components.iter().map(|c| c.sy).max().unwrap_or(1) as u32;
    let wf_min = max_sx << pih.nlx as u32;
    let hf_min = max_sy << pih.nly as u32;
    if (pih.wf as u32) < wf_min {
        return Err(Error::invalid(format!(
            "jpegxs decoder: Wf={} below the minimum max_i(sx)×2^NL,x = {}×2^{} = {} (Table 11)",
            pih.wf, max_sx, pih.nlx, wf_min
        )));
    }
    if (pih.hf as u32) < hf_min {
        return Err(Error::invalid(format!(
            "jpegxs decoder: Hf={} below the minimum max_i(sy)×2^NL,y = {}×2^{} = {} (Table 11)",
            pih.hf, max_sy, pih.nly, hf_min
        )));
    }

    if pih.qpih > 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: Qpih == {} reserved for ISO/IEC use (Table A.10)",
            pih.qpih
        )));
    }
    // ISO/IEC 21122-1:2022 Table A.8 — the only conformant (Bw, Fq)
    // combinations are (B[0], 0) for lossless / integer-transform coding,
    // (18, 6) when a non-linearity (NLT marker) is present, and (20, 8)
    // for the high-precision regular case. Reject everything else: a
    // non-tabulated pair (e.g. the historical Bw=8/Fq=8) is not a stream a
    // conforming encoder can produce, and decoding it under the Annex E.3
    // `c << Fq` scaling would silently corrupt the output.
    match (pih.bw, pih.fq) {
        (bw, 0) if bw == cdt.components[0].bit_depth => {}
        (18, 6) => {}
        (20, 8) => {}
        (bw, fq) => {
            return Err(Error::Unsupported(format!(
                "jpegxs decoder: (Bw={bw}, Fq={fq}) is not a valid ISO/IEC 21122-1:2022 \
                 Table A.8 combination (expected (B[0]={}, 0), (18, 6), or (20, 8))",
                cdt.components[0].bit_depth
            )));
        }
    }
    // Annex A.4.6: the NLT marker "shall not be present if Fq=0".
    if pih.fq == 0 && cs.nlt()?.is_some() {
        return Err(Error::Unsupported(
            "jpegxs decoder: NLT marker present with Fq=0 (Annex A.4.6 forbids this)".to_string(),
        ));
    }
    // CWD body is validated by the codestream parser; here we route
    // the Sd lookup through the typed [`codestream::Codestream::cwd`]
    // accessor (Annex A.4.7 Table A.18). Absent CWD → Sd = 0.
    let sd: u8 = cs.cwd()?.map(|c| c.sd).unwrap_or(0);
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

    // Parse optional NLT body (Annex A.4.6) through the typed
    // [`codestream::Codestream::nlt`] accessor.
    let nlt = cs.nlt()?;

    let (plan, _weights) = crate::slice_walker::build_plan_sd(&pih, &cdt, &wgt, sd)?;

    // Cw > 0 (Np_x > 1) forces the picture-level gather/cascade path
    // because per-precinct DWT is not equivalent to a multi-precinct-per-
    // row layout (precinct boundaries reflect at the band level, not the
    // sample level — only the picture-level cascade DWT is correct).
    // Sd > 0 also forces the gather path because the suppressed-component
    // band data is copied directly into samples there.
    let multi_level = pih.nlx > 1 || pih.nly > 1 || plan.np_x > 1 || sd > 0;

    // Allocate per-component sample buffers sized at Wc[i] × Hc[i].
    let wf = pih.wf as usize;
    let hf = pih.hf as usize;
    let mut samples: Vec<Vec<i32>> = Vec::with_capacity(plan.nc as usize);
    let mut comp_dims: Vec<(usize, usize)> = Vec::with_capacity(plan.nc as usize);
    for c in &cdt.components {
        let wc = wf.div_ceil(c.sx as usize);
        let hc = hf.div_ceil(c.sy as usize);
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
            // Suppressed components (Sd): no wavelet bands. Push an
            // empty per-component slot so indexing by `i` still works.
            if (i as u8) >= plan.nc - plan.sd {
                gathered.push(Vec::new());
                continue;
            }
            let wc = wf.div_ceil(c.sx as usize);
            let hc = hf.div_ceil(c.sy as usize);
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
        // coefficients have been gathered. Skip suppressed (Sd) components:
        // their samples were written directly during gather_precinct.
        for (i, c) in cdt.components.iter().enumerate() {
            if (i as u8) >= plan.nc - plan.sd {
                continue;
            }
            let wc = wf.div_ceil(c.sx as usize);
            let hc = hf.div_ceil(c.sy as usize);
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

    // Annex F inverse colour transform — Table F.1 dispatches by Cpih.
    // The transform reads the wavelet output O[c,x,y] for every operand
    // component (c < 3 for RCT, c < 4 for Star-Tetrix). When Sd > 0 the
    // suppressed tail (c ≥ Nc - Sd) is *raw-coded* rather than wavelet-
    // decomposed; its O[c] values were written directly into `samples[c]`
    // during the gather pass (see the suppressed-component branch above),
    // so the inverse transform consumes them unchanged. Suppressing a
    // transform *output* therefore composes cleanly — Annex B Tables B.10 /
    // B.11 tabulate exactly this for Star-Tetrix (Sd=1, Nc=4 CFA, the case
    // the Annex H weight tables H.9–H.11 target). RCT keeps the stricter
    // Nc-Sd >= 3 guard (no tabulated RCT-with-suppressed-output example).
    if pih.cpih == 1 {
        if (pih.nc - sd) < 3 {
            return Err(Error::invalid(format!(
                "jpegxs Cpih=1 (RCT) + Sd>0: Nc-Sd must be >= 3 so RCT operand window c<3 is wavelet-coded, got Nc={} Sd={}",
                pih.nc, sd
            )));
        }
        let mut refs: Vec<&mut [i32]> = samples.iter_mut().map(|p| p.as_mut_slice()).collect();
        inverse_rct(&mut refs, wf, hf)?;
    } else if pih.cpih == 3 {
        // Star-Tetrix needs the CTS marker (chroma exponents + Cf) and
        // CRG marker (CFA pattern type) per Annex F.5 / Tables F.9 /
        // F.10. The codestream parser already enforced "Cpih=3 → CTS
        // present", but CRG is also mandatory in this case (§A.4.9).
        // We route both through the typed
        // [`codestream::Codestream::cts`] / `crg` accessors so the
        // body-level field checks live in exactly one place.
        let cts = cs
            .cts()?
            .ok_or_else(|| Error::invalid("jpegxs Cpih=3: CTS marker required (A.4.8)"))?;
        let crg = cs
            .crg()?
            .ok_or_else(|| Error::invalid("jpegxs Cpih=3: CRG marker required (A.4.9)"))?;
        let ct = cfa_pattern_type(&crg).ok_or_else(|| {
            Error::invalid(
                "jpegxs Cpih=3: CRG entries do not match a Table F.9 CFA pattern (RGGB/BGGR/GRBG/GBRG)",
            )
        })?;
        if pih.nc < 4 {
            return Err(Error::invalid(format!(
                "jpegxs Cpih=3: Star-Tetrix requires Nc>=4 per Annex F.2, got {}",
                pih.nc
            )));
        }
        // Annex A.4.3 / §F.2: Cpih shall be 0 if any sx[i]/sy[i] for the
        // transform inputs differ from 1. `inverse_star_tetrix` reads all
        // four CFA inputs at full picture resolution (`planes[c][y*Wf+x]`),
        // so a sub-sampled input plane would be the wrong size and the
        // transform would index out of its own component. Reject it.
        for (i, c) in cdt.components.iter().enumerate().take(4) {
            if c.sx != 1 || c.sy != 1 {
                return Err(Error::invalid(format!(
                    "jpegxs Cpih=3: Star-Tetrix requires sx[i]=sy[i]=1 for i<4, got component {i} sx={} sy={} (Annex A.4.3 / F.2)",
                    c.sx, c.sy
                )));
            }
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
        // `JpegXsPlane::stride` is bytes per row: one byte per sample for
        // B[i] == 8, two little-endian bytes per sample for B[i] > 8
        // (round 118 high-bit-depth plane format).
        let bps = if comp.bit_depth > 8 { 2 } else { 1 };
        planes.push(VideoPlane {
            stride: wc * bps,
            data: bytes,
        });
    }

    Ok(JpegXsImage {
        width: pih.wf as u32,
        height: pih.hf as u32,
        num_components: pih.nc,
        cpih: pih.cpih,
        bit_depth: pih.bw,
        planes,
        pts,
    })
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
    // Vertical-prediction predecessor cache, one slot per precinct
    // column `px = p mod Np,x` (Annex C.6.3 Table C.11: the predecessor
    // of a top-line band is `M[p−Np,x,…]` / `T[p−Np,x,b]`, i.e. the
    // precinct directly above in the same column). The cache is local to
    // this slice, so vertical prediction never reaches across a slice
    // boundary (§C.6.1 / §C.6.3) — a fresh `decode_slice` call starts
    // with every column empty, matching the spec's per-slice
    // independence requirement.
    let np_x = plan.np_x.max(1) as usize;
    let mut col_tops: Vec<Option<PrecinctTop>> = vec![None; np_x];
    for precinct_plan in &slice_plan.precincts {
        let px = (precinct_plan.p % plan.np_x.max(1)) as usize;
        // Clone the column predecessor so the immutable borrow of
        // `col_tops` is released before we overwrite this column's slot
        // at the end of the precinct. `PrecinctTop` is small (per-band
        // last-line bitplane counts + truncation bytes).
        let top_above = col_tops.get(px).and_then(|o| o.clone());
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

        // Collect each non-empty packet's wire layout so that, after the
        // precinct is fully decoded, the wire `Lprc[p]` / `Lcnt`+`Lsig`
        // buffer-bound fields can be cross-checked against the values
        // independently reconstructed from the coding state (Annex C.2
        // Table C.1 + Annex C.5.3.4 Table C.6). `entries` references the
        // packet layout owned by `precinct_plan`, which outlives this
        // borrow, so the slices stay valid for the post-loop checks.
        let mut wire_sizes: Vec<PacketWireSize> = Vec::new();

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
                top_above.as_ref(),
            )?;
            entropy_cursor += dec.bytes_consumed;

            wire_sizes.push(PacketWireSize {
                header_bytes: packet_header.header_bytes as u32,
                lcnt: packet_header.lcnt,
                ldat: packet_header.ldat,
                lsgn: packet_header.lsgn,
                dr: packet_header.dr,
                entries: &packet_layout.entries,
            });
        }

        // Annex C.3 (`Rl = 0`): a band's raw-mode flag `Dr[p,s]` must be
        // identical across every packet that includes the band within this
        // precinct — raw and non-raw bitplane-count coding shall not be
        // mixed within one band. Reject a codestream that violates this
        // before the inconsistent decode state reaches the inverse
        // quantizer. The check is a no-op when `Rl = 1`.
        check_raw_mode_consistency(&precinct_plan.geometry, &wire_sizes)?;

        // Annex C.2 (Table C.1): the precinct's `Lprc[p]` field must be at
        // least the summed on-wire size of all its packets (headers +
        // sub-packets, including the inferred significance sub-packet); any
        // surplus is filler. `precinct_filler_bytes` returns `Err` when the
        // packets do not fit — a malformed codestream whose `Lprc[p]` is too
        // small to contain its own packets. Reject it rather than decode
        // past the declared precinct length.
        //
        // (The Annex C.5.3.4 / Table C.6 bitplane-count buffer bound is a
        // *codestream-construction* constraint, not a decode gate: a
        // degenerate all-zero precinct whose single-line bands each occupy
        // a whole byte can legally violate it yet still decode, so it is
        // exposed via [`bitplane_buffer_bound_satisfied`] for strict
        // conformance callers rather than enforced here.)
        let inferred_filler =
            precinct_filler_bytes(&precinct_plan.geometry, &precinct_header, &wire_sizes)?;

        // Cross-check: the filler-byte count `precinct_filler_bytes`
        // reconstructs from the per-packet sizes (header + inferred
        // `Lsig[p,s]` + `Lcnt` + `Ldat` + `Lsgn`) must equal the gap the
        // decoder is about to skip (`entropy_end - entropy_cursor`). The
        // decoder's own `entropy_cursor` advanced by `header_bytes +
        // bytes_consumed`, where `bytes_consumed` already folds in the
        // significance sub-packet it read off the wire, so the two
        // accountings are independent: one infers `Lsig` from the band
        // geometry, the other reads it from the codestream. A mismatch
        // means the codestream's sub-packet lengths are internally
        // inconsistent (e.g. a doctored `Lcnt`/`Ldat` field that still sums
        // to a valid `Lprc`), which the bare `Lprc` bound alone would miss.
        let actual_filler = entropy_end - entropy_cursor;
        if inferred_filler as usize != actual_filler {
            return Err(Error::invalid(format!(
                "jpegxs decoder: precinct p={} sub-packet length fields inconsistent — \
                 inferred {inferred_filler} filler bytes but {actual_filler} remain before Lprc end \
                 (Annex C.2/C.3 Tables C.1/C.3)",
                precinct_plan.p
            )));
        }

        // Capture this precinct's vertical-prediction predecessor for
        // the precinct directly below (same column) — Annex C.6.3
        // Table C.11.
        col_tops[px] = Some(PrecinctTop::capture(
            &precinct_plan.geometry,
            &precinct_header,
            &state,
        ));

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
                samples,
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
///
/// Sd suppressed components (i ≥ Nc - Sd) bypass `gathered`: their
/// band data is the raw component samples and gets copied straight into
/// `samples[i]` at the precinct's row offset.
#[allow(clippy::too_many_arguments)]
fn gather_precinct(
    precinct_plan: &PrecinctPlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    cdt: &crate::component_table::ComponentTable,
    bands: &[BandCoefficients],
    precinct_header: &PrecinctHeader,
    gathered: &mut [Vec<Vec<i32>>],
    samples: &mut [Vec<i32>],
) -> Result<()> {
    let trunc = precinct_truncation(&precinct_plan.geometry, precinct_header);
    let dequant = dequantize_precinct(pih.qpih, &precinct_plan.geometry, &trunc, bands, pih.fq);

    let _nc = plan.nc as u32;
    let n_decomposed = plan.n_decomposed;
    let sd_u = (plan.sd) as u32;
    let nbeta = plan.n_beta;
    let np_x = plan.np_x as usize;
    let py = (precinct_plan.p as usize) / np_x.max(1);
    let px = (precinct_plan.p as usize) % np_x.max(1);
    let nly_pic = pih.nly;

    // Wavelet components (i < Nc - Sd). Iterate over picture-β slots
    // (the bitstream's flat band id `b = (Nc - Sd) × β_pic + i`). For
    // each existing slot the precinct plan records the matching
    // chroma-local β (the index into the component's own DWT cascade
    // output buffer `gathered[i][local_β]`) per Annex B.4 / Figure B.2.
    for (i, c) in cdt
        .components
        .iter()
        .enumerate()
        .take(n_decomposed as usize)
    {
        let sy_i = c.sy;
        let nly_i = nly_pic.saturating_sub(match sy_i {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 0,
        });
        for beta_pic in 0..nbeta {
            let b = (n_decomposed * beta_pic + i as u32) as usize;
            let band_geom = &precinct_plan.geometry.bands[b];
            if !band_geom.exists {
                continue;
            }
            let local_beta = precinct_plan.band_local_beta[b];
            if local_beta == u32::MAX {
                continue;
            }
            let lines = (band_geom.l1 - band_geom.l0) as usize;
            if lines == 0 {
                continue;
            }
            let wpb = band_geom.wpb as usize;
            // Picture-level band dimensions for chroma's *local* β — the
            // chroma plane decomposed at NL,x / N'L,y[i] produces a band
            // sized (pic_bw × pic_bh) in chroma's grid.
            let wc = (pih.wf as usize).div_ceil(c.sx as usize);
            let hc = (pih.hf as usize).div_ceil(sy_i as usize);
            let (pic_bw, pic_bh) = band_dims(wc, hc, pih.nlx, nly_i, local_beta);
            let band_cols_per_uniform_precinct: usize = {
                let cs = plan.cs as usize;
                let sx_i = c.sx as usize;
                let key = beta_key_for(local_beta, pih.nlx, nly_i);
                let dx = key.dx as usize;
                let tx = key.tau_x;
                let denom_low = sx_i * (1usize << dx);
                if !tx {
                    cs.div_ceil(denom_low)
                } else {
                    let denom_high = sx_i * (1usize << dx.saturating_sub(1));
                    cs.div_ceil(denom_high) / 2
                }
            };
            let band_col_offset = px * band_cols_per_uniform_precinct;
            // Row offset in chroma's picture-level band buffer for this
            // precinct. Uses the component's own dy (chroma-local) so
            // 2^max(N'L,y[i] - dy_chroma, 0) is the chroma-band-grid
            // rows per precinct.
            let pow_h = cascade_band_pow_h(pih.nlx, nly_i, local_beta, hc);
            let row_offset = py * pow_h;
            let band_buf = &mut gathered[i][local_beta as usize];
            if band_buf.len() != pic_bw * pic_bh {
                return Err(Error::invalid(format!(
                    "jpegxs decoder gather: band buffer for comp {i} picture-β={beta_pic} local-β={local_beta} sized {} != {}*{}",
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
                let copy_w = wpb.min(pic_bw.saturating_sub(band_col_offset));
                if copy_w == 0 {
                    break;
                }
                let dst_start = pic_row * pic_bw + band_col_offset;
                let dst = &mut band_buf[dst_start..dst_start + copy_w];
                let src = &dequant[b][line * wpb..line * wpb + copy_w];
                dst.copy_from_slice(src);
            }
        }
    }

    // Sd tail: suppressed components have sx=sy=1 (enforced in walker)
    // and bypass the DWT cascade. Copy their dequantized band data
    // directly into the sample plane at this precinct's row offset.
    if sd_u > 0 {
        let wf = pih.wf as usize;
        let hf = pih.hf as usize;
        let cs = plan.cs as usize;
        let col_offset = px * cs;
        let wp = (wf.saturating_sub(col_offset)).min(cs);
        let pow_pic = if pih.nly == 0 {
            1usize
        } else {
            1usize << pih.nly
        };
        let pic_row_offset = py * pow_pic;
        for sd_idx in 0..sd_u as usize {
            let i = (n_decomposed as usize) + sd_idx;
            let b = ((n_decomposed * nbeta) as usize) + sd_idx;
            let band_geom = &precinct_plan.geometry.bands[b];
            if !band_geom.exists {
                continue;
            }
            let lines = (band_geom.l1 - band_geom.l0) as usize;
            let wpb = band_geom.wpb as usize;
            // Pull the dequantized values for this band.
            let band_buf = &dequant[b];
            for line in 0..lines {
                let pic_row = pic_row_offset + line;
                if pic_row >= hf {
                    break;
                }
                let copy_w = wpb.min(wp);
                if copy_w == 0 {
                    break;
                }
                let dst_start = pic_row * wf + col_offset;
                let src_start = line * wpb;
                let dst = &mut samples[i][dst_start..dst_start + copy_w];
                let src = &band_buf[src_start..src_start + copy_w];
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
    let dequant = dequantize_precinct(pih.qpih, &precinct_plan.geometry, &trunc, bands, pih.fq);

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
        let wc_i = (pih.wf as usize).div_ceil(sx_i);
        let _hc_i = (pih.hf as usize).div_ceil(sy_i);
        let wp_i = wp.div_ceil(sx_i); // per-component precinct width
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
                // For partial bottom precincts (e.g. odd-height
                // pictures where the last precinct only covers 1 pixel
                // row), some bands carry fewer rows than `hp_i`
                // expects. Pad them with zero rows so the 2-D
                // synthesis runs at the full precinct height; the
                // post-DWT row-copy clips at `target_row >= Hf / sy_i`
                // so synthesised samples beyond the picture boundary
                // are dropped.
                let ll_w_e = wp_i.div_ceil(2);
                let hl_w_e = wp_i / 2;
                let ll_h_e = hp_i.div_ceil(2);
                let lh_h_e = hp_i / 2;
                let pad_to = |buf: &[i32], want: usize| -> Vec<i32> {
                    if buf.len() == want {
                        buf.to_vec()
                    } else {
                        let mut v = buf.to_vec();
                        v.resize(want, 0);
                        v
                    }
                };
                let ll_p = pad_to(ll, ll_w_e * ll_h_e);
                let hl_p = pad_to(hl, hl_w_e * ll_h_e);
                let lh_p = pad_to(lh, ll_w_e * lh_h_e);
                let hh_p = pad_to(hh, hl_w_e * lh_h_e);
                let mut out = vec![0i32; wp_i * hp_i];
                inverse_2d(wp_i, hp_i, &ll_p, &hl_p, &lh_p, &hh_p, &mut out)?;
                let row_offset = py * hp_i;
                let hf_rows = (pih.hf as usize).div_ceil(sy_i);
                for line in 0..hp_i {
                    let target_row = row_offset + line;
                    if target_row >= hf_rows {
                        break;
                    }
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

// Integration tests live behind the `registry` feature because they
// drive the decoder through `make_decoder` + `Packet`/`Frame`/`TimeBase`
// which are oxideav-core types.
#[cfg(all(test, feature = "registry"))]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::registry::make_decoder;
    use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, TimeBase};

    /// Multi-level plan-shape sanity check: 4×4 luma, NL=2/2 → 7 bands
    /// (LL2 + HL2 + LH2 + HH2 + HL1 + LH1 + HH1), single component,
    /// one slice covering the whole picture, every precinct's band
    /// geometry must exist and have non-zero width.
    #[test]
    fn multilevel_plan_shape_nl_2_2_4x4_luma() {
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
        // Annex B.3: Nβ = 2*min(NL,x,NL,y) + max(NL,x,NL,y) + 1.
        // For NL,x = NL,y = 2 → Nβ = 7.
        assert_eq!(plan.n_beta, 7, "NL=2/2 must give Nβ=7");
        assert_eq!(plan.slices.len(), 1, "Hsl=1 single-slice plan");
        for s in &plan.slices {
            assert!(!s.precincts.is_empty(), "slice must contain >= 1 precinct");
            for p in &s.precincts {
                for b in &p.geometry.bands {
                    if b.exists {
                        assert!(b.wpb > 0, "existing band has non-zero width");
                    }
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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

    /// Byte offset of the PIH `Lcod` field inside a codestream built by
    /// the `build_zero_codestream_*` helpers: SOC(2) + CAP marker(2) +
    /// Lcap(2) + PIH marker(2) + Lpih(2) = 10, then Lcod(4) at 10.
    const LCOD_OFFSET: usize = 10;
    /// Byte offset of the PIH `Ppih` field: Lcod(4) follows at 10..14,
    /// then Ppih(2) at 14, Plev(2) at 16.
    const PPIH_OFFSET: usize = 14;
    const PLEV_OFFSET: usize = 16;

    /// Patch a big-endian u32 in place at `off`.
    fn patch_u32(buf: &mut [u8], off: usize, val: u32) {
        buf[off..off + 4].copy_from_slice(&val.to_be_bytes());
    }

    /// PIH body field offsets (absolute, from the helper layout): the PIH
    /// body begins at byte 10 (after SOC + CAP segment + PIH marker +
    /// Lpih). Wf is body[8..10], Hf body[10..12], the NL,x|NL,y nibble
    /// byte is body[22].
    const WF_OFFSET: usize = 18;
    const HF_OFFSET: usize = 20;

    /// Patch a big-endian u16 in place at `off`.
    fn patch_u16(buf: &mut [u8], off: usize, val: u16) {
        buf[off..off + 2].copy_from_slice(&val.to_be_bytes());
    }

    /// Decode `buf` through the registry decoder, returning the result.
    fn decode_buf(buf: Vec<u8>) -> Result<JpegXsImage> {
        decode_codestream(&buf, None)
    }

    #[test]
    fn ppih_offset_points_at_ppih_field() {
        // Sanity-check the offset constant against the helper layout: the
        // unpatched stream carries Ppih = 0 (Unrestricted) at PPIH_OFFSET.
        let buf = build_zero_codestream_4x1();
        assert_eq!(
            u16::from_be_bytes([buf[PPIH_OFFSET], buf[PPIH_OFFSET + 1]]),
            0
        );
        assert_eq!(
            u16::from_be_bytes([buf[PLEV_OFFSET], buf[PLEV_OFFSET + 1]]),
            0
        );
    }

    #[test]
    fn decode_rejects_reserved_ppih() {
        // 0x9999 maps to no profile (Profile::from_ppih → None), so a
        // conforming decoder rejects it rather than decode under an
        // unknown profile (ISO/IEC 21122-2 Table A.5).
        let mut buf = build_zero_codestream_4x1();
        patch_u16(&mut buf, PPIH_OFFSET, 0x9999);
        let err = decode_buf(buf).unwrap_err();
        assert!(
            format!("{err}").contains("reserved for ISO/IEC"),
            "expected reserved-Ppih rejection, got {err}"
        );
    }

    #[test]
    fn decode_rejects_profile_violating_stream() {
        // The 4×1 luma stream codes a single slice of one image row
        // (Hsl=1, NL,y=0). Every constrained profile fixes the slice
        // height at 16 image rows (Table A.1/A.2/A.3 "Slice height = 16"),
        // so declaring the Light 422.10 profile (0x1500) contradicts the
        // stream's own slice geometry — a decoder that honours the
        // declared profile must reject it (ISO/IEC 21122-2 Annex A).
        let mut buf = build_zero_codestream_4x1();
        patch_u16(&mut buf, PPIH_OFFSET, 0x1500);
        let err = decode_buf(buf).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Light 422.10") && msg.contains("image rows"),
            "expected Light-422.10 slice-height conformance rejection, got {msg}"
        );
    }

    #[test]
    fn decode_accepts_unrestricted_profile_stream() {
        // The baseline stream declares Ppih = 0 (Unrestricted) and decodes
        // to the mid-grey row — the conformance wiring must be a no-op for
        // the unrestricted profile.
        let buf = build_zero_codestream_4x1();
        let img = decode_buf(buf).expect("unrestricted stream decodes");
        assert_eq!(img.planes[0].data, vec![128u8; 4]);
    }

    #[test]
    fn decode_rejects_reserved_plev_high_byte() {
        // Plev high byte 0xFF is reserved (Level::from_plev_high → None),
        // so check_level rejects the stream (Table A.6 / A.12).
        let mut buf = build_zero_codestream_4x1();
        patch_u16(&mut buf, PLEV_OFFSET, 0xFF00);
        let err = decode_buf(buf).unwrap_err();
        assert!(
            format!("{err}").contains("Plev high byte"),
            "expected reserved-Plev rejection, got {err}"
        );
    }

    #[test]
    fn wf_hf_offsets_point_at_dimension_fields() {
        // The 4×1 helper carries Wf=4, Hf=1 (NL,x=1, NL,y=0).
        let buf = build_zero_codestream_4x1();
        assert_eq!(u16::from_be_bytes([buf[WF_OFFSET], buf[WF_OFFSET + 1]]), 4);
        assert_eq!(u16::from_be_bytes([buf[HF_OFFSET], buf[HF_OFFSET + 1]]), 1);
    }

    #[test]
    fn decode_rejects_wf_below_decomposition_minimum() {
        // 4×1 luma at NL,x=1 requires Wf ≥ 1×2^1 = 2. Patch Wf=1 (below
        // the minimum): the picture cannot carry the LL band the header
        // claims, so the stream is rejected (Table 11).
        let mut buf = build_zero_codestream_4x1();
        patch_u16(&mut buf, WF_OFFSET, 1);
        let err = decode_buf(buf).unwrap_err();
        assert!(
            format!("{err}").contains("Wf=1 below the minimum"),
            "expected Wf-minimum rejection, got {err}"
        );
    }

    #[test]
    fn decode_accepts_wf_at_decomposition_minimum() {
        // Wf=2 is exactly the minimum for NL,x=1; the 4×1 stream's entropy
        // covers a 4-wide row, so shrinking the declared Wf below 4 would
        // desync the entropy layout — instead assert the boundary check
        // itself does not fire for the unmodified Wf=4 stream.
        let buf = build_zero_codestream_4x1();
        decode_buf(buf).expect("Wf=4 ≥ minimum 2 decodes");
    }

    #[test]
    fn decode_rejects_hf_below_decomposition_minimum() {
        // The 2×2 luma fixture is NL,x=1 / NL,y=1, so Hf ≥ 1×2^1 = 2.
        // Patch Hf=1 (below the minimum): rejected per Table 11. The 2×2
        // helper shares the PIH-field offsets with the 4×1 helper.
        let mut buf = build_zero_codestream_2x2();
        patch_u16(&mut buf, HF_OFFSET, 1);
        let err = decode_buf(buf).unwrap_err();
        assert!(
            format!("{err}").contains("Hf=1 below the minimum"),
            "expected Hf-minimum rejection, got {err}"
        );
    }

    #[test]
    fn lcod_offset_points_at_lcod_field() {
        let buf = build_zero_codestream_4x1();
        // The helper sets Lcod = 0 (VBR).
        assert_eq!(
            u32::from_be_bytes([
                buf[LCOD_OFFSET],
                buf[LCOD_OFFSET + 1],
                buf[LCOD_OFFSET + 2],
                buf[LCOD_OFFSET + 3]
            ]),
            0
        );
    }

    #[test]
    fn decode_accepts_matching_cbr_lcod() {
        // Patch Lcod to the exact SOC-to-EOC byte count: a CBR stream that
        // truthfully describes its own length decodes normally.
        let mut buf = build_zero_codestream_4x1();
        let len = buf.len() as u32;
        patch_u32(&mut buf, LCOD_OFFSET, len);
        let img = decode_buf(buf).expect("matching Lcod decodes");
        assert_eq!(img.planes[0].data, vec![128u8; 4]);
    }

    #[test]
    fn decode_rejects_mismatched_cbr_lcod() {
        // A non-zero Lcod that disagrees with the actual length is an
        // internally inconsistent CBR self-description (Table 11).
        let mut buf = build_zero_codestream_4x1();
        let wrong = buf.len() as u32 + 7;
        patch_u32(&mut buf, LCOD_OFFSET, wrong);
        let err = decode_buf(buf).unwrap_err();
        assert!(
            format!("{err}").contains("Lcod"),
            "expected Lcod-mismatch rejection, got {err}"
        );
    }

    #[test]
    fn decode_accepts_vbr_lcod_zero() {
        // Lcod = 0 (VBR) imposes no length self-check.
        let buf = build_zero_codestream_4x1();
        let img = decode_buf(buf).expect("VBR Lcod=0 decodes");
        assert_eq!(img.planes[0].data, vec![128u8; 4]);
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
        bits1 <<= 1;
        bits1 = (bits1 << 15) | 1;
        bits1 = (bits1 << 13) | 1;
        bits1 <<= 11;
        let mut hdr1 = vec![0u8; 5];
        for (i, b) in hdr1.iter_mut().enumerate() {
            *b = ((bits1 >> (8 * (4 - i))) & 0xff) as u8;
        }
        payload.extend_from_slice(&hdr1);
        payload.push(0b10000000);
        payload.push(0x0C);
        let mut bits2: u64 = 0;
        bits2 <<= 1;
        bits2 <<= 15;
        bits2 = (bits2 << 13) | 1;
        bits2 <<= 11;
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

    /// Annex C.2 (Table C.1) gate: a precinct whose `Lprc[p]` field is too
    /// small to contain its own packet headers + sub-packets must be
    /// rejected, not silently decoded against a truncated entropy window.
    /// We take the valid `build_constant_4x1_lossless` codestream and
    /// overwrite the 24-bit `Lprc[p]` field at the start of its (only)
    /// precinct header with 1, which is smaller than the two packet
    /// headers the precinct carries, so `precinct_filler_bytes` reports the
    /// packets do not fit.
    #[test]
    fn decode_rejects_undersized_lprc() {
        let mut buf = build_constant_4x1_lossless();
        // Layout (see `build_constant_4x1_lossless`): the buffer ends with
        // the 2-byte EOC, preceded by `payload`, preceded by the 6-byte
        // precinct header. The first 3 bytes of that precinct header are
        // the big-endian `Lprc[p]` field. Recompute the payload length the
        // same way the builder does so the offset is exact.
        let mut payload_len = 0usize;
        // packet 1: 5-byte header + 2 body bytes (0b10000000, 0x0C).
        payload_len += 5 + 2;
        // packet 2: 5-byte header + 1 body byte (0x00).
        payload_len += 5 + 1;
        let eoc = 2usize;
        let prec_hdr_len = 6usize;
        let prec_hdr_off = buf.len() - eoc - payload_len - prec_hdr_len;
        // Sanity: the original Lprc equals payload_len.
        let orig_lprc = ((buf[prec_hdr_off] as u32) << 16)
            | ((buf[prec_hdr_off + 1] as u32) << 8)
            | (buf[prec_hdr_off + 2] as u32);
        assert_eq!(
            orig_lprc as usize, payload_len,
            "test self-check: located the precinct header Lprc field"
        );
        // Corrupt Lprc to 1 (too small for the packet headers).
        buf[prec_hdr_off] = 0;
        buf[prec_hdr_off + 1] = 0;
        buf[prec_hdr_off + 2] = 1;

        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), buf);
        // The undersized Lprc must surface as a decode error (it is caught
        // either by the precinct-length consistency gate added here or by
        // the downstream marker-chain check that finds the following SLH /
        // EOC marker missing because the precinct's declared length stopped
        // short) — never silently mis-decoded.
        let res = dec
            .send_packet(&pkt)
            .and_then(|_| dec.receive_frame().map(|_| ()));
        assert!(
            res.is_err(),
            "undersized Lprc[p] must be rejected, got Ok(())"
        );
    }

    /// Annex C.3 (§4182): "the bitplane count, data and sign subpackets may
    /// contain an arbitrary number of filler bytes at their end". The exact
    /// sub-packet filler cross-check added to the decode path must *tolerate*
    /// such legal filler, not reject it. We take the valid constant-4×1
    /// codestream and grow packet 1's data sub-packet by one filler byte:
    /// bump its short-form `Ldat` field from 1 to 2, insert one 0x00 filler
    /// byte after the data byte, and grow `Lprc[p]` by 1 to match. The
    /// decoder must skip the filler and reconstruct the identical
    /// `[129, 129, 129, 129]` row.
    #[test]
    fn decode_tolerates_data_subpacket_filler() {
        let base = build_constant_4x1_lossless();
        // Locate the precinct header / payload boundaries exactly as in
        // `decode_rejects_undersized_lprc`.
        let payload_len = (5 + 2) + (5 + 1);
        let eoc = 2usize;
        let prec_hdr_len = 6usize;
        let prec_hdr_off = base.len() - eoc - payload_len - prec_hdr_len;
        let payload_off = prec_hdr_off + prec_hdr_len;
        // Packet 1 is `hdr1` (5 bytes) + Lcnt byte (1) + Ldat byte (1). The
        // short header packs Dr(1)|Ldat(15)|Lcnt(13)|Lsgn(11); `Ldat` is the
        // top 15 bits after the Dr bit, i.e. bits [38..=24] of the 40-bit
        // header. With the builder's values (Dr=0, Ldat=1, Lcnt=1, Lsgn=0)
        // those header bytes are byte 0..5 of the payload.
        let hdr1_off = payload_off;
        // Rebuild hdr1 with Ldat = 2 instead of 1.
        let mut bits1: u64 = 0;
        bits1 <<= 1; // Dr = 0
        bits1 = (bits1 << 15) | 2; // Ldat = 2 (was 1)
        bits1 = (bits1 << 13) | 1; // Lcnt = 1
        bits1 <<= 11; // Lsgn = 0
        let mut new_buf = base.clone();
        for i in 0..5 {
            new_buf[hdr1_off + i] = ((bits1 >> (8 * (4 - i))) & 0xff) as u8;
        }
        // Insert one filler byte after packet 1's data byte. Packet 1 data
        // byte is at payload offset 5 (hdr) + 1 (Lcnt) + 1 (Ldat) - 1 = index
        // 6 within the payload; the filler goes immediately after it.
        let filler_at = payload_off + 5 + 1 + 1; // after hdr1 + Lcnt + Ldat byte
        new_buf.insert(filler_at, 0x00);
        // Grow Lprc[p] by 1 to account for the inserted filler byte.
        let lprc = ((new_buf[prec_hdr_off] as u32) << 16)
            | ((new_buf[prec_hdr_off + 1] as u32) << 8)
            | (new_buf[prec_hdr_off + 2] as u32);
        let lprc = lprc + 1;
        new_buf[prec_hdr_off] = ((lprc >> 16) & 0xff) as u8;
        new_buf[prec_hdr_off + 1] = ((lprc >> 8) & 0xff) as u8;
        new_buf[prec_hdr_off + 2] = (lprc & 0xff) as u8;

        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), new_buf);
        dec.send_packet(&pkt).expect("send_packet filler");
        let frame = dec
            .receive_frame()
            .expect("data sub-packet filler must be skipped, not rejected");
        let Frame::Video(vf) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(
            vf.planes[0].data,
            vec![129, 129, 129, 129],
            "legal data sub-packet filler must not perturb the decode"
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
    /// matching the layout the slice walker emits (Nβ = 7 invariant
    /// also asserted by `multilevel_plan_shape_nl_2_2_4x4_luma`).
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
        bits <<= 1;
        bits <<= 15;
        bits = (bits << 13) | 1;
        bits <<= 11;
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
