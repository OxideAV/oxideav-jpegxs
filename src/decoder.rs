//! JPEG XS pixel decoder — round 4.
//!
//! Wires the pieces produced by rounds 1–3 plus the round-4 slice
//! walker and inverse quantizer into a working `Decoder` for the
//! single-component, single-precinct, single-slice subset of the
//! standard:
//!
//! * `Nc == 1`, `sx == sy == 1`, `Cw == 0` (one precinct per row).
//! * `Hsl` arbitrary; the slice walker still groups precincts into
//!   slices per Annex B.10.
//! * `Cpih == 0` (no inverse colour transform — Annex F is round 5).
//! * `Qpih ∈ {0, 1}` (deadzone or uniform inverse quantizer).
//! * `Fq` per Table A.8 controls how the wavelet integer output maps
//!   to sample-domain bytes; for `Fq == 8` the coefficients are scaled
//!   relative to `Bw == 20` and we down-shift by `Fq` before clipping
//!   to `[0, 2^B[0] - 1]`. Annex G's full DC-shift / non-linearity /
//!   clip path is round 5; we do the obvious linear mapping for now.
//!
//! Anything outside this subset returns `Error::Unsupported` from the
//! decoder factory or from `send_packet`.

use std::collections::VecDeque;

use oxideav_core::{
    frame::VideoPlane, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, VideoFrame,
};

use crate::codestream;
use crate::dequant::dequantize_precinct;
use crate::dwt::inverse_2d;
use crate::entropy::packet_body::PrecinctState;
use crate::entropy::{
    decode_packet_body, parse_packet_header, parse_precinct_header, precinct_truncation,
    BandCoefficients,
};
use crate::slice_walker::{build_plan, PicturePlan, PrecinctPlan};

/// Build a JPEG XS decoder. Round 4 accepts the single-component,
/// single-precinct subset.
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

    if pih.cpih != 0 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: Cpih == {} requires Annex F inverse colour transform (round 5)",
            pih.cpih
        )));
    }
    if pih.qpih > 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: Qpih == {} reserved for ISO/IEC use (Table A.10)",
            pih.qpih
        )));
    }
    if cs.nlt.is_some() {
        return Err(Error::Unsupported(
            "jpegxs decoder: NLT (non-linear transform) is round-5 (Annex G)".into(),
        ));
    }
    if cs.cwd.is_some() {
        return Err(Error::Unsupported(
            "jpegxs decoder: CWD (component-dependent wavelet decomposition) not supported".into(),
        ));
    }

    let (plan, _weights) = build_plan(&pih, &cdt, &wgt)?;

    // Allocate the reconstructed sample buffer per component — round 4
    // is single-component.
    let comp = cdt.components[0];
    let wf = pih.wf as usize;
    let hf = pih.hf as usize;
    let mut samples = vec![0i32; wf * hf];

    // Round 4 only handles NL,x ≤ 1 and NL,y ≤ 1. Beyond that the
    // multi-level cascade (Annex E.2) is deferred — `inverse_2d`
    // handles a single level only.
    if pih.nlx > 1 || pih.nly > 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: NL,x={} NL,y={} requires multi-level DWT cascade (round 5)",
            pih.nlx, pih.nly
        )));
    }

    // Walk slices in order; each slice contributes a contiguous run of
    // precincts (rows of size `Hp` lines tall when NL,y > 0).
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
            comp.bit_depth,
            &mut samples,
            wf,
            hf,
        )?;
    }

    // Map int32 wavelet samples to bytes per Annex G "linear path"
    // (round-4 shortcut — Annex G full path lands in round 5). The
    // samples are stored as `Fq`-fractional-bit fixed-point integers
    // relative to the picture's bit depth `B[c]`. For `Fq == 8` and
    // `B[c] == 8` the spec scales by `2^Fq` on encode; we down-shift
    // and clip on decode.
    let plane = pack_to_plane(&samples, wf, hf, comp.bit_depth, pih.fq)?;

    Ok(VideoFrame {
        pts,
        planes: vec![plane],
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_slice(
    slice_data: &[u8],
    slice_plan: &crate::slice_walker::SlicePlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    bit_depth: u8,
    samples: &mut [i32],
    wf: usize,
    hf: usize,
) -> Result<()> {
    let mut cursor = 0usize;
    // Per-precinct state of the previous precinct in the same column —
    // needed for the cross-precinct vertical predictor (Annex C.6.3).
    // For round 4 (single precinct in the only column when Cw == 0,
    // and the topmost precinct of the slice — bands at λ = top of
    // band cannot use vertical prediction per Annex C.6.1), the state
    // is empty for the first precinct of every slice.
    let _ = bit_depth;
    let _ = hf;

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

        for (s, packet_layout) in precinct_plan.packets.iter().enumerate() {
            // Skip empty packets — Annex C.3 emits no header for them.
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
            let _ = s;
        }

        // Skip precinct filler bytes up to Lprc.
        cursor = entropy_end;

        // Inverse-quantize and DWT-synthesise this precinct.
        synthesise_precinct(
            precinct_plan,
            plan,
            pih,
            &state.coefficients,
            &precinct_header,
            samples,
            wf,
        )?;
    }
    Ok(())
}

fn synthesise_precinct(
    precinct_plan: &PrecinctPlan,
    plan: &PicturePlan,
    pih: &crate::picture_header::PictureHeader,
    bands: &[BandCoefficients],
    precinct_header: &crate::entropy::PrecinctHeader,
    samples: &mut [i32],
    wf: usize,
) -> Result<()> {
    let trunc = precinct_truncation(&precinct_plan.geometry, precinct_header);
    let dequant = dequantize_precinct(pih.qpih, &precinct_plan.geometry, &trunc, bands);

    let nlx = plan.nlx as u32;
    let nly = plan.nly as u32;
    let wp = precinct_plan.wp as usize;
    let hp = precinct_plan.hp as usize;

    if nlx == 0 && nly == 0 {
        // No DWT — band 0 is the raw component samples for this precinct.
        let band = &precinct_plan.geometry.bands[0];
        let band_samples = &dequant[0];
        let py = precinct_plan.p as usize; // np_x == 1 in round 4
        let row_offset = py * hp;
        for line in 0..(band.l1 - band.l0) as usize {
            let target_row = row_offset + line;
            if target_row >= samples.len() / wf {
                break;
            }
            let dst = &mut samples[target_row * wf..target_row * wf + wp];
            let src = &band_samples[line * (band.wpb as usize)..line * (band.wpb as usize) + wp];
            dst.copy_from_slice(src);
        }
        return Ok(());
    }

    // Single-level DWT cascade: the picture plan for round 4 limits us
    // to NL,x ≤ 1 and NL,y ≤ 1, so we run at most one inverse_2d call
    // per precinct.
    if nlx == 1 && nly == 1 {
        // 4 bands: LL=β=0, HL=β=1, LH=β=2, HH=β=3.
        let ll = &dequant[0];
        let hl = &dequant[1];
        let lh = &dequant[2];
        let hh = &dequant[3];
        let mut out = vec![0i32; wp * hp];
        inverse_2d(wp, hp, ll, hl, lh, hh, &mut out)?;
        let py = precinct_plan.p as usize; // np_x == 1
        let row_offset = py * hp;
        for line in 0..hp {
            let target_row = row_offset + line;
            if target_row >= samples.len() / wf {
                break;
            }
            let dst = &mut samples[target_row * wf..target_row * wf + wp];
            let src = &out[line * wp..line * wp + wp];
            dst.copy_from_slice(src);
        }
        return Ok(());
    }
    if nlx == 1 && nly == 0 {
        // 2 bands: LL=β=0, HL=β=1. Single-row inverse-horizontal.
        let ll_band = &precinct_plan.geometry.bands[0];
        let hl_band = &precinct_plan.geometry.bands[1];
        let ll = &dequant[0];
        let hl = &dequant[1];
        let lines = (ll_band.l1 - ll_band.l0) as usize;
        let py = precinct_plan.p as usize;
        let row_offset = py * hp;
        let mut row = vec![0i32; wp];
        for line in 0..lines {
            let low = &ll[line * (ll_band.wpb as usize)..(line + 1) * (ll_band.wpb as usize)];
            let high = &hl[line * (hl_band.wpb as usize)..(line + 1) * (hl_band.wpb as usize)];
            crate::dwt::inverse_horizontal_1d(low, high, &mut row)?;
            let target_row = row_offset + line;
            if target_row >= samples.len() / wf {
                break;
            }
            let dst = &mut samples[target_row * wf..target_row * wf + wp];
            dst.copy_from_slice(&row);
        }
        return Ok(());
    }

    Err(Error::Unsupported(format!(
        "jpegxs decoder: NL,x={} NL,y={} not implemented in round 4",
        nlx, nly
    )))
}

/// Convert the int32 wavelet output into an 8-bit sample plane. Annex G
/// is round 5; round 4 does the obvious linear scaling: down-shift by
/// `Fq` (treating coefficients as `Fq`-fractional-bit fixed point), add
/// the DC bias `2^(B-1)`, and clip to `[0, 2^B - 1]`.
fn pack_to_plane(
    samples: &[i32],
    wf: usize,
    hf: usize,
    bit_depth: u8,
    fq: u8,
) -> Result<VideoPlane> {
    if bit_depth != 8 {
        return Err(Error::Unsupported(format!(
            "jpegxs decoder: bit depth {bit_depth} requires Annex G round 5"
        )));
    }
    let stride = wf;
    let mut data = vec![0u8; wf * hf];
    let dc_bias = 1i32 << (bit_depth - 1);
    let max = (1i32 << bit_depth) - 1;
    let shift = fq as u32;
    for (i, v) in samples.iter().enumerate() {
        // Down-shift and round-to-nearest using add-half-shift.
        let scaled = if shift == 0 {
            *v
        } else {
            let half = 1i32 << (shift - 1);
            (*v + half) >> shift
        };
        let with_bias = scaled + dc_bias;
        let clipped = with_bias.clamp(0, max);
        data[i] = clipped as u8;
    }
    Ok(VideoPlane { stride, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, TimeBase};

    /// The factory produces a real decoder (no Unsupported error) for a
    /// generic params block.
    #[test]
    fn factory_returns_decoder() {
        let params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let dec = make_decoder(&params).expect("decoder factory");
        assert_eq!(dec.codec_id().as_str(), crate::CODEC_ID_STR);
    }

    /// `receive_frame` returns `NeedMore` before any packet is sent.
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
        // CAP — Lcap = 2 (no capability bits)
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // PIH — Lpih = 26
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        // PIH body (24 bytes):
        v.extend_from_slice(&0u32.to_be_bytes()); // Lcod = 0 (VBR)
        v.extend_from_slice(&0u16.to_be_bytes()); // Ppih
        v.extend_from_slice(&0u16.to_be_bytes()); // Plev
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf = 4
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf = 1
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw = 0
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl = 1
        v.push(1); // Nc = 1
        v.push(4); // Ng = 4
        v.push(8); // Ss = 8
        v.push(20); // Bw = 20
        v.push(0x80); // Fq = 8 | Br = 0
        v.push(0x00); // Fslc=0, Ppoc=0, Cpih=0
        v.push(0x10); // NL,x = 1, NL,y = 0
        v.push(0x10); // Lh=0, Rl=0, Qpih=0, Fs=0, Rm=0
                      //   bit layout: 0 0 00 00 00 → 0x00. We use 0x00.
                      // Override the previous push because 0x10 wrong:
        let last = v.len() - 1;
        v[last] = 0x00;
        // Set Br to 4 so the raw mode (unused) header field passes
        // checks; bit-plane-count will use VLC instead. Update the
        // packed byte at PIH offset 20 (within body): Fq=8 | Br=4.
        // The PIH body starts after the 4-byte header (FF 12 + Lpih).
        // Body offset 20 is the 21st body byte; absolute offset = 4 +
        // 20 = 24 inside the marker segment, plus the SOC(2) + CAP(4) =
        // 6 bytes earlier. Easier: locate the 0x80 byte we wrote and
        // bump it to 0x84.
        for byte in v.iter_mut().rev() {
            if *byte == 0x80 {
                *byte = 0x84;
                break;
            }
        }
        // CDT — Lcdt = 4, body = [B[0]=8, sx<<4 | sy = 0x11]
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT — Lwgt = 2 (no bands → ?). Two bands actually, both
        // with G=0, P=0. So Lwgt = 2 + 2*2 = 6.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&6u16.to_be_bytes());
        v.extend_from_slice(&[0, 0, 0, 0]); // G[0],P[0],G[1],P[1]
                                            // SLH — Lslh = 4, Yslh = 0
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // --- Entropy coded data for the single precinct ---
        // Precinct header (6 bytes):
        //   Lprc[24] + Q[8] + R[8] + D[0]:2 + D[1]:2 + pad → 48 bits
        //   = 6 bytes. Lprc = 12 (count of the bytes following).
        v.extend_from_slice(&[0x00, 0x00, 12, 0, 0, 0x00]);
        // Packet 1 header (5 bytes, short form):
        //   dr=0, Ldat=0, Lcnt=1, Lsgn=0.
        //   bits: 0 | 000_0000_0000_0000 | 0_0000_0000_0001 | 000_0000_0000
        //   = 40 bits.
        //   Pack: 0 0000000 00000000 0000000 0001 000 0000_0000
        //   → byte 0 = 0x00, byte 1 = 0x00, byte 2 = 0x00,
        //     byte 3 = 0x00, byte 4 = 0x40 (for Lcnt's bit 12 = 0...).
        // Easier: build via u64.
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0; // dr
        bits = (bits << 15) | 0; // Ldat
        bits = (bits << 13) | 1; // Lcnt = 1
        bits = (bits << 11) | 0; // Lsgn
        let mut packet1_hdr = vec![0u8; 5];
        for (i, b) in packet1_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        v.extend_from_slice(&packet1_hdr);
        // Packet 1 body:
        //   sig: D&2==0 → no bits (0 bytes consumed).
        //   bitplane-count (Lcnt=1): one VLC for mtop=T=0 → "0" comma.
        //     1 byte: 0x00 (one 0 bit at MSB, padding zeros).
        v.push(0x00);
        // Packet 2 header (LH/HL band): same as packet 1.
        v.extend_from_slice(&packet1_hdr);
        // Packet 2 body: 1 byte 0x00.
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
        // All-zero coefficients → flat mid-grey (2^(B-1) = 128) after
        // inverse DWT and DC-bias.
        for (i, &px) in vf.planes[0].data.iter().enumerate() {
            assert_eq!(
                px, 128,
                "pixel {i}: expected 128 (mid-grey), got {px} (full row {:?})",
                vf.planes[0].data
            );
        }
    }

    /// 2x2 image with full 1-level 2-D DWT (NL,x = NL,y = 1). All
    /// quantization-index magnitudes zero → 2-D inverse 5/3 produces a
    /// flat zero band → all output pixels are 128 after DC bias.
    fn build_zero_codestream_2x2() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xff, 0x10]); // SOC
        v.extend_from_slice(&[0xff, 0x50]); // CAP
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x12]); // PIH
        v.extend_from_slice(&26u16.to_be_bytes());
        // PIH body — Wf=2, Hf=2, Hsl=1, Nc=1, Ng=4, Ss=8, Bw=20,
        //          Fq=8 Br=4, NL,x=1 NL,y=1, all flags zero.
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&2u16.to_be_bytes()); // Wf
        v.extend_from_slice(&2u16.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(1); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x84); // Fq=8 | Br=4
        v.push(0x00); // Fslc/Ppoc/Cpih
        v.push(0x11); // NL,x=1, NL,y=1
        v.push(0x00); // Lh/Rl/Qpih/Fs/Rm
                      // CDT
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT — 4 bands × 2 bytes = 8 + 2 (Lwgt) = 10
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&10u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 8]); // G/P all zero for 4 bands
                                        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // Precinct header — 24 + 8 + 8 + 4*2 + pad → 48 bits = 6 bytes.
        // Lprc = 4 packets × (5 hdr + 1 cnt) = 24 bytes.
        v.extend_from_slice(&[0x00, 0x00, 24, 0, 0, 0x00]);
        // 4 packets, each with header (Ldat=0, Lcnt=1, Lsgn=0) and
        // body of 1 byte 0x00 (one VLC zero comma bit).
        let mut packet_hdr = vec![0u8; 5];
        let mut bits: u64 = 0;
        bits = (bits << 1) | 0; // dr
        bits = (bits << 15) | 0; // Ldat
        bits = (bits << 13) | 1; // Lcnt
        bits = (bits << 11) | 0; // Lsgn
        for (i, b) in packet_hdr.iter_mut().enumerate() {
            *b = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        for _ in 0..4 {
            v.extend_from_slice(&packet_hdr);
            v.push(0x00); // single 0 bit + padding
        }
        v.extend_from_slice(&[0xff, 0x11]); // EOC
        v
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
        v.extend_from_slice(&4u16.to_be_bytes()); // Wf=4
        v.extend_from_slice(&1u16.to_be_bytes()); // Hf=1
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.push(1);
        v.push(4);
        v.push(8);
        v.push(8); // Bw = B[0] = 8 (lossless)
        v.push(0x04); // Fq = 0 | Br = 4
        v.push(0x00);
        v.push(0x10); // NL,x=1, NL,y=0
        v.push(0x00);
        // CDT
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT (2 bands)
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&6u16.to_be_bytes());
        v.extend_from_slice(&[0u8; 4]);
        // SLH
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // Compute precinct payload first to derive Lprc.
        let mut payload = Vec::new();
        // Packet 1 (LL): Ldat=1, Lcnt=1, Lsgn=0.
        let mut bits1: u64 = 0;
        bits1 = (bits1 << 1) | 0;
        bits1 = (bits1 << 15) | 1; // Ldat = 1
        bits1 = (bits1 << 13) | 1; // Lcnt = 1
        bits1 = (bits1 << 11) | 0;
        let mut hdr1 = vec![0u8; 5];
        for (i, b) in hdr1.iter_mut().enumerate() {
            *b = ((bits1 >> (8 * (4 - i))) & 0xff) as u8;
        }
        payload.extend_from_slice(&hdr1);
        // Bitplane-count: VLC for M=1 with mtop=T=0 → "10" (10000000)
        payload.push(0b10000000);
        // Data: signs "0000" + plane0 "1100" → 0x0C
        payload.push(0x0C);
        // Packet 2 (HL): Ldat=0, Lcnt=1, Lsgn=0. M=0 ("0").
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
        payload.push(0x00); // bitplane-count "0" comma
        let lprc = payload.len() as u32;
        // Precinct header: 24+8+8+4+pad = 44 bits → 6 bytes.
        let mut prec_hdr = vec![0u8; 6];
        prec_hdr[0] = ((lprc >> 16) & 0xff) as u8;
        prec_hdr[1] = ((lprc >> 8) & 0xff) as u8;
        prec_hdr[2] = (lprc & 0xff) as u8;
        // prec_hdr[3] = Q = 0, prec_hdr[4] = R = 0, prec_hdr[5] = D bits = 0.
        v.extend_from_slice(&prec_hdr);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&[0xff, 0x11]); // EOC
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
        // Inverse 5/3 of LL=[1,1], HL=[0,0] is constant 1 across the
        // four output samples. With Fq=0 (lossless) and B=8 → +128 →
        // 129 after clipping.
        assert_eq!(
            vf.planes[0].data,
            vec![129, 129, 129, 129],
            "non-zero LL coefficient should propagate through the inverse 5/3 DWT"
        );
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
}
