//! JPEG XS encoder — round 1 minimum-viable bootstrap.
//!
//! Mirrors the decoder pipeline (rounds 1-6) in reverse for the smallest
//! useful subset of ISO/IEC 21122-1:2022:
//!
//! * Single luma component (`Nc = 1`, `sx = sy = 1`, `B[0] = 8`).
//! * Single decomposition level (`NL,x = NL,y = 1`).
//! * `Cw = 0` — one precinct column spanning the picture width.
//! * `Hsl = Np_y` — a single slice covering every precinct row.
//! * No colour transform (`Cpih = 0`), deadzone inverse quantizer
//!   (`Qpih = 0`), `Q = R = 0` per precinct (lossless inside the
//!   quantizer; truncation position `T = 0`).
//! * Lossless coding mode (`Fq = 0`, `Bw = 8`) — decoder Annex G linear
//!   path with `ζ = 0` so reconstructed samples equal `(c + 2^(B-1))`
//!   clamped.
//! * Raw bitplane-count mode (`Dr = 1`) per packet so the bitplane-count
//!   sub-packet is a fixed-width MSB-first encoding (`Br` bits per code
//!   group); no significance / vertical prediction state is needed for
//!   round 1.
//! * `Fs = 0` — signs interleaved into the data sub-packet alongside
//!   magnitudes, exactly matching the decoder's `decode_packet_body`
//!   data-path branch.
//!
//! Byte stream layout produced (all big-endian):
//!
//! ```text
//! SOC | CAP | PIH | CDT | WGT | SLH | <slice 0 entropy data> | EOC
//! ```
//!
//! Per-precinct entropy data:
//!
//! ```text
//! precinct_header (Lprc | Q | R | D[..] | pad)
//!   packet_header_LL  (5-byte short form)
//!   packet_body_LL    (Lcnt bytes raw bitplane counts | Ldat bytes data)
//!   packet_header_HL ... etc for all four single-level bands
//! ```
//!
//! Round-1 self-roundtrip pipeline:
//!
//! 1. Subtract DC bias `1 << (Bw - 1) = 128` from each input pixel.
//! 2. For every 2-row precinct stripe, run [`crate::dwt::forward_2d`] to
//!    obtain `(LL, HL, LH, HH)` sub-bands of `1 × 16` coefficients each.
//! 3. Convert each coefficient to (sign, magnitude). Magnitudes feed the
//!    raw bitplane counter; signs feed the per-coef sign bit emitted in
//!    the data sub-packet.
//! 4. Per code group of `Ng = 4` adjacent coefficients, compute
//!    `M = ⌈log₂(max_magnitude + 1)⌉`; this is the bitplane count
//!    written to the bitplane-count sub-packet (raw form, `Br` bits).
//! 5. For every group with `M > T = 0`, write `Ng = 4` sign bits then
//!    `M × Ng` magnitude bits MSB-first per Annex C.4 Table C.8.
//! 6. Pad sub-packets to byte boundaries and record the resulting
//!    `Lcnt` / `Ldat` lengths in the packet header. `Lsgn = 0` (`Fs = 0`).
//!
//! The decoder reverses every step exactly. The per-precinct `Lprc` field
//! is tallied as the encoder finishes each precinct and patched back into
//! the precinct header before moving on.

use crate::dwt::forward_2d;
use crate::error::{JpegXsError as Error, Result};
use crate::image::{JpegXsImage, JpegXsPlane};

/// Encoder configuration for round 1. Most fields are derived from the
/// picture geometry — only the input pixels are caller-supplied.
#[derive(Debug, Clone, Copy)]
struct EncodeConfig {
    /// Picture width in pixels (`Wf`).
    width: u16,
    /// Picture height in pixels (`Hf`).
    height: u16,
    /// Component bit depth, fixed at 8 for round 1.
    bit_depth: u8,
    /// Wavelet coefficient precision (`Bw`), fixed at 8 (lossless,
    /// `Fq = 0`).
    bw: u8,
    /// Coefficients per code group (`Ng`), Annex C constant.
    ng: u8,
    /// Code groups per significance group (`Ss`), Annex C constant.
    ss: u8,
    /// Raw bitplane-count width (`Br`), 4 bits in `0..=15` per Table A.7.
    br: u8,
}

impl EncodeConfig {
    /// Round-1 hard-coded configuration sized for the caller-supplied
    /// width / height. Constants match the decoder's lossless single-
    /// component path.
    fn round1(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            bw: 8,
            ng: 4,
            ss: 8,
            // Br = 8 → bitplane counts up to 255, ample headroom for any
            // 8-bit pixel after the 5/3 forward DWT.
            br: 8,
        }
    }
}

/// Encode a single-luma 8-bit image to a JPEG XS codestream.
///
/// `pixels` is a row-major `width * height` byte slice; encoder validates
/// length and dimensions against the round-1 supported subset.
///
/// Returns the complete codestream including SOC / EOC.
pub fn encode_luma_8bit(width: u16, height: u16, pixels: &[u8]) -> Result<Vec<u8>> {
    if width < 2 || height < 2 {
        return Err(Error::invalid(format!(
            "jpegxs encoder round 1: picture dimensions must be >= 2, got {width}x{height}"
        )));
    }
    // Round 1: NL,y = 1 means Hp = 2; we currently require even height
    // (and even width for the per-precinct 2-D DWT to roundtrip via the
    // decoder's per-precinct synthesis path).
    if width % 2 != 0 || height % 2 != 0 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 1: dimensions must be even (got {width}x{height}); odd dims deferred to round 2"
        )));
    }
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(Error::invalid(format!(
            "jpegxs encoder: pixel slice length {} does not match {width}x{height} = {expected}",
            pixels.len()
        )));
    }

    let cfg = EncodeConfig::round1(width, height);
    let mut out = Vec::with_capacity(expected + 256);

    write_main_header(&mut out, &cfg)?;
    write_slice(&mut out, &cfg, pixels)?;
    // EOC.
    out.extend_from_slice(&[0xff, 0x11]);
    Ok(out)
}

/// Encode the JPEG XS codestream out of a [`JpegXsImage`]. Convenience
/// wrapper for callers who already hold the decoder's output type.
/// Round-1 accepts only 1-component, 8-bit, even-dimension inputs.
pub fn encode_image(img: &JpegXsImage) -> Result<Vec<u8>> {
    if img.num_components != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 1: requires Nc = 1, got {}",
            img.num_components
        )));
    }
    if img.bit_depth != 8 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 1: requires Bw = 8, got {}",
            img.bit_depth
        )));
    }
    let plane = img
        .planes
        .first()
        .ok_or_else(|| Error::invalid("jpegxs encoder: image has zero planes"))?;
    if plane.stride != img.width as usize {
        return Err(Error::Unsupported(
            "jpegxs encoder round 1: plane stride must equal width (no padding)".into(),
        ));
    }
    encode_luma_8bit(img.width as u16, img.height as u16, &plane.data)
}

/// Round-1 convenience: build a [`JpegXsImage`] from raw bytes and then
/// encode. Useful for self-roundtrip tests that already have raw pixels.
pub fn encode_raw_luma(width: u16, height: u16, pixels: Vec<u8>) -> Result<Vec<u8>> {
    let img = JpegXsImage {
        width: width as u32,
        height: height as u32,
        num_components: 1,
        cpih: 0,
        bit_depth: 8,
        planes: vec![JpegXsPlane {
            stride: width as usize,
            data: pixels,
        }],
        pts: None,
    };
    encode_image(&img)
}

fn write_main_header(out: &mut Vec<u8>, cfg: &EncodeConfig) -> Result<()> {
    // SOC.
    out.extend_from_slice(&[0xff, 0x10]);
    // CAP — Lcap = 2 (no capability bits).
    out.extend_from_slice(&[0xff, 0x50]);
    out.extend_from_slice(&2u16.to_be_bytes());
    // PIH — Lpih = 26, body = 24 bytes.
    out.extend_from_slice(&[0xff, 0x12]);
    out.extend_from_slice(&26u16.to_be_bytes());
    write_pih_body(out, cfg);
    // CDT — Lcdt = 2 + 2*Nc = 4, body = 2 bytes.
    out.extend_from_slice(&[0xff, 0x13]);
    out.extend_from_slice(&4u16.to_be_bytes());
    out.push(cfg.bit_depth); // B[0] = 8
    out.push(0x11); // sx = 1, sy = 1
                    // WGT — Lwgt = 2 + 2*N_existing. For Nc=1 NL=1/1, Nβ = 4 → 4 bands,
                    // all gain/priority pairs are zero.
    out.extend_from_slice(&[0xff, 0x14]);
    let n_bands_wgt = 4u16; // single component × Nβ=4
    let lwgt = 2 + 2 * n_bands_wgt;
    out.extend_from_slice(&lwgt.to_be_bytes());
    for _ in 0..n_bands_wgt {
        out.push(0); // G[b] = 0
        out.push(0); // P[b] = 0
    }
    Ok(())
}

fn write_pih_body(out: &mut Vec<u8>, cfg: &EncodeConfig) {
    // Lcod = 0 (variable bitrate).
    out.extend_from_slice(&0u32.to_be_bytes());
    // Ppih = 0 (unrestricted profile).
    out.extend_from_slice(&0u16.to_be_bytes());
    // Plev = 0.
    out.extend_from_slice(&0u16.to_be_bytes());
    // Wf, Hf.
    out.extend_from_slice(&cfg.width.to_be_bytes());
    out.extend_from_slice(&cfg.height.to_be_bytes());
    // Cw = 0 → one precinct column per row of the picture.
    out.extend_from_slice(&0u16.to_be_bytes());
    // Hsl = Np_y so all precincts land in a single slice. With NL,y = 1
    // and Cw = 0, Hp = 2 → Np_y = ceil(Hf / 2).
    let np_y = (cfg.height as u32).div_ceil(2);
    out.extend_from_slice(&(np_y as u16).to_be_bytes());
    // Nc, Ng, Ss, Bw.
    out.push(1); // Nc = 1
    out.push(cfg.ng);
    out.push(cfg.ss);
    out.push(cfg.bw);
    // Fq:Br — Fq = 0 (lossless), Br = cfg.br.
    out.push(cfg.br & 0x0f);
    // Fslc:Ppoc:Cpih — all zero.
    out.push(0x00);
    // NL,x:NL,y — 1 / 1.
    out.push(0x11);
    // Lh:Rl:Qpih:Fs:Rm — all zero.
    out.push(0x00);
}

/// Per-precinct working state: the four sub-bands and the raw entropy
/// bytes (precinct header + packet headers + packet bodies).
struct PrecinctEncoded {
    bytes: Vec<u8>,
}

fn write_slice(out: &mut Vec<u8>, cfg: &EncodeConfig, pixels: &[u8]) -> Result<()> {
    // SLH — Lslh = 4, Yslh = 0 (single slice covers the whole picture).
    out.extend_from_slice(&[0xff, 0x20]);
    out.extend_from_slice(&4u16.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes());

    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let dc_bias: i32 = 1 << (cfg.bw - 1);

    // Encode each precinct (2-row strip).
    let np_y = (cfg.height as u32).div_ceil(2) as usize;
    for py in 0..np_y {
        let y0 = py * 2;
        let y1 = (y0 + 2).min(h);
        if y1 - y0 != 2 {
            // Round 1 only supports even heights (validated in
            // `encode_luma_8bit`); a non-2 strip would require odd-height
            // precinct handling, deferred to round 2.
            return Err(Error::Unsupported(
                "jpegxs encoder round 1: terminal precinct with < 2 rows not yet supported".into(),
            ));
        }
        let mut strip: Vec<i32> = Vec::with_capacity(w * (y1 - y0));
        for y in y0..y1 {
            for x in 0..w {
                strip.push(pixels[y * w + x] as i32 - dc_bias);
            }
        }
        let pe = encode_precinct(cfg, &strip, w, 2)?;
        out.extend_from_slice(&pe.bytes);
    }
    Ok(())
}

fn encode_precinct(
    cfg: &EncodeConfig,
    strip: &[i32],
    wp: usize,
    hp: usize,
) -> Result<PrecinctEncoded> {
    // Per-precinct forward 2-D DWT into the four sub-bands.
    let ll_w = wp.div_ceil(2);
    let hl_w = wp / 2;
    let ll_h = hp.div_ceil(2);
    let lh_h = hp / 2;
    let mut ll = vec![0i32; ll_w * ll_h];
    let mut hl = vec![0i32; hl_w * ll_h];
    let mut lh = vec![0i32; ll_w * lh_h];
    let mut hh = vec![0i32; hl_w * lh_h];
    forward_2d(wp, hp, strip, &mut ll, &mut hl, &mut lh, &mut hh)?;

    // Build the four packet bodies + headers in band order LL, HL, LH, HH.
    // The slice walker emits the same packet ordering for NL=1/1.
    let bands_in_order: [&[i32]; 4] = [&ll, &hl, &lh, &hh];

    // The precinct header has fixed layout for 4 existing bands:
    //   Lprc(24) + Q(8) + R(8) + 4 × D(2) = 48 bits → 6 bytes, no
    //   trailing pad needed.
    let precinct_header_len = 6usize;
    let mut precinct_bytes = vec![0u8; precinct_header_len];
    // Q = 0, R = 0, D[..] = 0 — no quantization, no significance, no
    // prediction. Lprc patched below.
    // Bytes 3 = Q, byte 4 = R, byte 5 = packed D bits (all zero).

    let mut entropy: Vec<u8> = Vec::new();
    for band in bands_in_order {
        let (header_bytes, body_bytes) = encode_packet(cfg, band)?;
        entropy.extend_from_slice(&header_bytes);
        entropy.extend_from_slice(&body_bytes);
    }
    let lprc = entropy.len() as u32;
    if lprc == 0 {
        return Err(Error::invalid(
            "jpegxs encoder: empty precinct (Lprc must be >= 1)",
        ));
    }
    if lprc > (1 << 20) - 1 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: precinct exceeds 2^20-1 bytes (Lprc = {lprc})"
        )));
    }
    precinct_bytes[0] = ((lprc >> 16) & 0xff) as u8;
    precinct_bytes[1] = ((lprc >> 8) & 0xff) as u8;
    precinct_bytes[2] = (lprc & 0xff) as u8;
    // Q, R, D bits already zero.
    precinct_bytes.extend_from_slice(&entropy);
    Ok(PrecinctEncoded {
        bytes: precinct_bytes,
    })
}

/// Encode one packet covering one band line. Returns `(header, body)`
/// each as a freshly-allocated `Vec<u8>`.
///
/// `Ncg = ⌈Wpb / Ng⌉` per Annex B.8: short tail groups are allowed when
/// `Wpb` is not a multiple of `Ng`. Per Table C.8 the data sub-packet
/// still walks `Ng` slots per group regardless; the spec emits sign and
/// magnitude bits for the out-of-band slots too. For the encoder side we
/// supply zero-padded slots in those positions so the decoder's
/// `if xpos < band.wpb` guard simply discards them.
fn encode_packet(cfg: &EncodeConfig, band: &[i32]) -> Result<(Vec<u8>, Vec<u8>)> {
    let wpb = band.len();
    let ng = cfg.ng as usize;
    let br = cfg.br;
    let ncg = wpb.div_ceil(ng);

    // Helper: read coefficient at logical group/slot position; out-of-
    // range slots return 0 (matches the decoder's `xpos < wpb` guard).
    let coef = |g: usize, k: usize| -> i32 {
        let xpos = g * ng + k;
        if xpos < wpb {
            band[xpos]
        } else {
            0
        }
    };

    // Compute per-group bitplane counts M = ceil(log2(max_mag + 1)).
    let mut m_per_group = vec![0u8; ncg];
    let m_max_for_br: u32 = if br >= 8 { 255 } else { (1u32 << br) - 1 };
    for (g, slot) in m_per_group.iter_mut().enumerate() {
        let mut max_mag: u32 = 0;
        for k in 0..ng {
            let v = coef(g, k);
            let mag = v.unsigned_abs();
            if mag > max_mag {
                max_mag = mag;
            }
        }
        let m = if max_mag == 0 {
            0u32
        } else {
            32 - max_mag.leading_zeros()
        };
        if m > m_max_for_br {
            return Err(Error::invalid(format!(
                "jpegxs encoder: group {g} bitplane count {m} exceeds Br = {br} (cap {m_max_for_br}). Use a higher Br or quantize the input."
            )));
        }
        *slot = m as u8;
    }

    // ---- Bitplane-count sub-packet (raw mode, Dr = 1): Br bits per group ----
    let mut cnt_writer = BitWriter::default();
    for &m in &m_per_group {
        cnt_writer.write_bits(m as u32, br);
    }
    cnt_writer.align_to_byte();
    let cnt_bytes = cnt_writer.into_bytes();
    let lcnt = cnt_bytes.len() as u32;

    // ---- Data sub-packet ----
    let mut data_writer = BitWriter::default();
    // T = 0 → every group with M > 0 emits Ng signs + M*Ng magnitudes.
    for (g, &m_u8) in m_per_group.iter().enumerate() {
        let m = m_u8 as u32;
        if m == 0 {
            continue;
        }
        // Fs = 0 → write Ng sign bits first.
        for k in 0..ng {
            let v = coef(g, k);
            let sign_bit = if v < 0 { 1 } else { 0 };
            data_writer.write_bit(sign_bit);
        }
        // Then m bitplanes, MSB-first.
        for plane in (0..m).rev() {
            for k in 0..ng {
                let v = coef(g, k);
                let mag = v.unsigned_abs();
                let bit = ((mag >> plane) & 1) as u8;
                data_writer.write_bit(bit);
            }
        }
    }
    data_writer.align_to_byte();
    let data_bytes = data_writer.into_bytes();
    let ldat = data_bytes.len() as u32;

    // Lsgn = 0 (Fs = 0 → sign sub-packet absent).
    let lsgn: u32 = 0;

    // Round-1 always emits short packet headers — Wf*Nc < 32752 holds for
    // any round-1 picture (Nc = 1, Wf <= 65535 but our test fixtures
    // stay well under 32752). Lh = 0 in PIH so the decoder dispatches to
    // the short-form branch.
    if ldat > (1 << 15) - 1 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: Ldat = {ldat} exceeds short packet header capacity (15 bits). Use the long form (round 2)."
        )));
    }
    if lcnt > (1 << 13) - 1 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: Lcnt = {lcnt} exceeds short packet header capacity (13 bits)."
        )));
    }

    // Header layout: 1-bit Dr | 15-bit Ldat | 13-bit Lcnt | 11-bit Lsgn.
    let mut hdr_bits: u64 = 0;
    hdr_bits = (hdr_bits << 1) | 1; // Dr = 1 (raw mode)
    hdr_bits = (hdr_bits << 15) | (ldat as u64 & 0x7fff);
    hdr_bits = (hdr_bits << 13) | (lcnt as u64 & 0x1fff);
    hdr_bits = (hdr_bits << 11) | (lsgn as u64 & 0x07ff);
    let mut header = vec![0u8; 5];
    for (i, b) in header.iter_mut().enumerate() {
        *b = ((hdr_bits >> (8 * (4 - i))) & 0xff) as u8;
    }

    // Body: bitplane-count sub-packet (Lcnt bytes) then data sub-packet
    // (Ldat bytes). No significance sub-packet because Dr = 1 (Annex C.4
    // Table C.5: significance sub-packet absent when Dr = 1). No sign
    // sub-packet because Fs = 0.
    let mut body = Vec::with_capacity(cnt_bytes.len() + data_bytes.len());
    body.extend_from_slice(&cnt_bytes);
    body.extend_from_slice(&data_bytes);
    Ok((header, body))
}

/// Tiny MSB-first bit writer mirroring [`crate::entropy::bits::BitReader`].
#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    /// Number of bits written into the in-progress final byte (0 means
    /// the last byte is full or there is no in-progress byte).
    bit_pos: u8,
}

impl BitWriter {
    fn write_bit(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.bytes.push(0);
        }
        let last = self.bytes.last_mut().unwrap();
        *last |= (bit & 1) << (7 - self.bit_pos);
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::decode_codestream;

    /// PSNR of two byte-arrays of equal length, in dB. `255` is the
    /// peak signal value.
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sse: u64 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x as i32) - (*y as i32);
            sse += (d * d) as u64;
        }
        if sse == 0 {
            return f64::INFINITY;
        }
        let mse = sse as f64 / a.len() as f64;
        20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
    }

    fn make_synthetic_32x32() -> Vec<u8> {
        let mut buf = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                // A mix of low- and high-frequency content + a smooth
                // diagonal gradient so every wavelet sub-band contains
                // non-trivial energy.
                let v = ((x as i32) * 5 + (y as i32) * 7 + ((x ^ y) & 0x0f) as i32 * 3) % 256;
                buf[y * 32 + x] = v as u8;
            }
        }
        buf
    }

    #[test]
    fn bit_writer_packs_msb_first() {
        let mut w = BitWriter::default();
        w.write_bits(0b1010, 4);
        w.write_bits(0b0110, 4);
        w.write_bits(0b1100, 4);
        w.write_bits(0b0011, 4);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0xa6, 0xc3]);
    }

    #[test]
    fn bit_writer_pads_with_zeros() {
        let mut w = BitWriter::default();
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        w.align_to_byte();
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0b1010_0000]);
    }

    #[test]
    fn rejects_odd_dimensions() {
        let pixels = vec![0u8; 31 * 31];
        assert!(encode_luma_8bit(31, 31, &pixels).is_err());
    }

    #[test]
    fn rejects_pixel_buffer_size_mismatch() {
        let pixels = vec![0u8; 4];
        assert!(encode_luma_8bit(32, 32, &pixels).is_err());
    }

    /// Per-precinct (32-col × 2-row) forward 2-D DWT must round-trip
    /// through `inverse_2d` losslessly. Pinned because round-1 relies on
    /// per-precinct synthesis of `Hp = 2`-line bands and the decoder
    /// path through [`crate::dwt::extend_symmetric`] for `z = 2`.
    #[test]
    fn per_precinct_dwt_round_trips_for_hp_2() {
        use crate::dwt::{forward_2d, inverse_2d};
        let pixels = make_synthetic_32x32();
        let w = 32usize;
        let mut strip = Vec::with_capacity(w * 2);
        for y in 0..2 {
            for x in 0..w {
                strip.push(pixels[y * w + x] as i32 - 128);
            }
        }
        let mut ll = vec![0i32; 16];
        let mut hl = vec![0i32; 16];
        let mut lh = vec![0i32; 16];
        let mut hh = vec![0i32; 16];
        forward_2d(w, 2, &strip, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        let mut out = vec![0i32; w * 2];
        inverse_2d(w, 2, &ll, &hl, &lh, &hh, &mut out).unwrap();
        assert_eq!(out, strip, "per-precinct (Hp = 2) 2-D DWT must round-trip");
    }

    #[test]
    fn encode_then_decode_flat_image_is_exact() {
        let pixels = vec![123u8; 32 * 32];
        let codestream = encode_luma_8bit(32, 32, &pixels).expect("encode flat 32x32");
        let img = decode_codestream(&codestream, None).expect("decode flat 32x32");
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.planes.len(), 1);
        assert_eq!(img.planes[0].data.len(), 32 * 32);
        assert_eq!(
            img.planes[0].data, pixels,
            "flat image must round-trip losslessly"
        );
    }

    #[test]
    fn self_roundtrip_synthetic_32x32_is_lossless() {
        let pixels = make_synthetic_32x32();
        let codestream =
            encode_luma_8bit(32, 32, &pixels).expect("encode synthetic 32x32 round-1 image");
        let img = decode_codestream(&codestream, None).expect("decode synthetic 32x32 codestream");
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        assert_eq!(img.num_components, 1);
        let decoded = &img.planes[0].data;
        // Round 1 is the lossless path (Fq = 0, Q = 0, deadzone, T = 0)
        // so we expect a bit-exact reconstruction of the input pixels;
        // PSNR is therefore +∞ but the workspace requirement is
        // ≥ 40 dB. We hard-assert exact match (+∞ dB) to catch any
        // regression of the lossless guarantee.
        assert_eq!(
            decoded, &pixels,
            "lossless round 1 encoder must reconstruct the input exactly (PSNR = inf, requirement >= 40 dB)"
        );
        let p = psnr(&pixels, decoded);
        assert!(
            p >= 40.0,
            "self-roundtrip PSNR {p:.2} dB falls short of the 40 dB round-1 minimum"
        );
    }

    #[test]
    fn self_roundtrip_2x2_minimum_size() {
        let pixels = vec![10u8, 200, 50, 150];
        let codestream = encode_luma_8bit(2, 2, &pixels).expect("encode 2x2");
        let img = decode_codestream(&codestream, None).expect("decode 2x2");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.planes[0].data, pixels);
    }

    /// Round-trip via [`encode_image`] which takes a [`JpegXsImage`] —
    /// the same struct the decoder produces.
    #[test]
    fn encode_image_then_decode_round_trips() {
        let pixels = make_synthetic_32x32();
        let codestream = encode_raw_luma(32, 32, pixels.clone()).expect("encode_raw_luma");
        let img = decode_codestream(&codestream, None).expect("decode after encode_raw_luma");
        assert_eq!(img.planes[0].data, pixels);
    }
}
