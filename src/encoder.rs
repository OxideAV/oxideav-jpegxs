//! JPEG XS encoder — rounds 1-2.
//!
//! Round 1 (commit `95b4e27`) shipped the lossless single-luma 8-bit
//! single-decomposition single-precinct-column case. Round 2 broadens
//! the encoder to cover three additional production-relevant axes:
//!
//! * **Multi-component (`Nc ∈ {1, 3}`).** New entry point
//!   [`encode_rgb_8bit`] takes an interleaved 24-bit-per-pixel RGB
//!   buffer, splits it into three planes, optionally runs the forward
//!   reversible colour transform (Annex F.4 Table F.3) when the caller
//!   asks for `cpih = 1`, then emits a 3-component codestream
//!   (`Cpih ∈ {0, 1}`, sx = sy = 1 for all three components — i.e.
//!   4:4:4 only). The picture-level band ordering matches Annex B.2
//!   (`b = Nc * β + i`).
//! * **Multi-decomposition (`NL,x = NL,y ∈ {1, 2}`).** Higher-level
//!   cascades go through [`crate::dwt::forward_cascade_2d`] which
//!   produces every (β, i) sub-band buffer at picture level. The
//!   encoder slices each band into its per-precinct rows when
//!   serialising packets. The decoder's gather-then-cascade path
//!   synthesises them back exactly.
//! * **Odd dimensions.** Inputs with `Wf` or `Hf` not a multiple of
//!   `2^NL,y` are accepted as-is. The forward DWT already supports
//!   `Wpb < Ng` short tail groups; band heights derived by
//!   `Hb = ⌈Hc / 2^dy⌉` cope with the rounding without explicit
//!   padding. The minimum dimension stays at 2 per Annex E.6.
//!
//! Out-of-scope (deferred to round 3+):
//! * 4:2:2 / 4:2:0 chroma sub-sampling (`sy[i] > 1` or `sx[i] > 1`).
//! * `Fq = 8` regular (lossy) mode.
//! * NLT-aware encoder (linear / quadratic / extended gamma).
//! * VLC bitplane-count modes (`Dr = 0` no-prediction or vertical
//!   prediction) — round 2 still emits raw mode (`Dr = 1`).
//! * Significance coding (`D[p,b] & 2`).
//! * `Cw > 0` (custom precinct widths).
//! * Star-Tetrix encoder (`Cpih = 3`).
//!
//! Byte stream shape (unchanged from round 1, with the segment field
//! values driven by the round-2 config):
//!
//! ```text
//! SOC | CAP | PIH | CDT | WGT | SLH | <slice 0 entropy data> | EOC
//! ```
//!
//! Per-precinct layout for the multi-component / multi-decomp case:
//! the precinct header carries `Lprc` + `Q` + `R` + `D[..]` for every
//! existing band (one bit pair per band = `2 * NL` bits, padded to a
//! byte). The packet stream walks Annex B.7 Table B.4: one packet
//! containing the level-`(NL,x, NL,y)` LL band across all components,
//! then one packet per (line × β × component) for the proxy levels.
//! Each packet body is significance (empty when `D & 2 = 0`),
//! bitplane-count (raw mode `Dr = 1` → `Br` bits per code group),
//! data (sign bit + magnitude bitplanes interleaved with `Fs = 0`),
//! sign (absent when `Fs = 0`).

use crate::colour_transform::forward_rct;
use crate::dwt::{forward_2d, forward_cascade_2d};
use crate::error::{JpegXsError as Error, Result};
use crate::image::{JpegXsImage, JpegXsPlane};

/// Encoder configuration. Most fields are derived from the picture
/// geometry — only the input pixels, layout, and Cpih choice are
/// caller-controlled.
#[derive(Debug, Clone, Copy)]
struct EncodeConfig {
    /// Picture width in pixels (`Wf`).
    width: u16,
    /// Picture height in pixels (`Hf`).
    height: u16,
    /// Number of components (`Nc`).
    nc: u8,
    /// Component bit depth, fixed at 8 for round 2.
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
    /// Horizontal decomposition levels (`NL,x`).
    nlx: u8,
    /// Vertical decomposition levels (`NL,y`).
    nly: u8,
    /// Colour transformation id (`Cpih`).
    cpih: u8,
}

impl EncodeConfig {
    fn validate(&self) -> Result<()> {
        if self.width < 2 || self.height < 2 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: picture dimensions must be >= 2, got {}x{}",
                self.width, self.height
            )));
        }
        if !matches!(self.nc, 1 | 3) {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 2: Nc must be 1 or 3, got {}",
                self.nc
            )));
        }
        if self.cpih == 1 && self.nc != 3 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Cpih=1 (RCT) requires Nc=3, got {}",
                self.nc
            )));
        }
        if self.cpih > 1 {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 2: Cpih must be 0 or 1, got {} (Star-Tetrix not yet supported)",
                self.cpih
            )));
        }
        if !(1..=2).contains(&self.nlx) || self.nlx != self.nly {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 2: only NL,x = NL,y ∈ {{1, 2}} supported, got NL,x={} NL,y={}",
                self.nlx, self.nly
            )));
        }
        Ok(())
    }
}

/// Encode a single-luma 8-bit image to a JPEG XS codestream.
///
/// `pixels` is a row-major `width * height` byte slice. Single-
/// decomposition (`NL,x = NL,y = 1`) bootstrap path retained from
/// round 1 for callers / tests that pin the original geometry.
pub fn encode_luma_8bit(width: u16, height: u16, pixels: &[u8]) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(Error::invalid(format!(
            "jpegxs encoder: pixel slice length {} does not match {width}x{height} = {expected}",
            pixels.len()
        )));
    }
    encode_planar(width, height, 1, 0, 1, 1, &[pixels.to_vec()])
}

/// Encode a 3-component RGB image to a JPEG XS codestream.
///
/// `pixels` is interleaved 24-bit `R,G,B,R,G,B,…` row-major. The encoder
/// splits the input into three planes, optionally applies the forward
/// reversible colour transform (Annex F.4) when `cpih == 1`, and runs
/// the multi-component encoder. Both `cpih == 0` (no transform) and
/// `cpih == 1` (RCT) round-trip losslessly through the decoder.
///
/// Multi-decomposition `nl ∈ {1, 2}` selects the wavelet cascade depth.
pub fn encode_rgb_8bit(
    width: u16,
    height: u16,
    pixels: &[u8],
    cpih: u8,
    nl: u8,
) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() != expected {
        return Err(Error::invalid(format!(
            "jpegxs encoder: RGB pixel slice length {} does not match {width}x{height}*3 = {expected}",
            pixels.len()
        )));
    }
    let n = (width as usize) * (height as usize);
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        r.push(chunk[0]);
        g.push(chunk[1]);
        b.push(chunk[2]);
    }
    encode_planar(width, height, 3, cpih, nl, nl, &[r, g, b])
}

/// Encode the JPEG XS codestream out of a [`JpegXsImage`]. Round 2
/// accepts:
/// * 1 or 3 components, 8-bit, 4:4:4 sub-sampling.
/// * `cpih ∈ {0, 1}` (no transform or forward RCT for 3-component
///   inputs).
///
/// `cpih` is taken from `img.cpih`. Multi-decomposition is taken from
/// the largest power-of-two `nl` that fits — round 2 ranges over
/// `nl ∈ {1, 2}`. For backwards compatibility with the round-1 caller,
/// we always pick `nl = 1` here; use [`encode_planar`] or
/// [`encode_rgb_8bit`] with an explicit `nl` to drive the multi-decomp
/// path.
pub fn encode_image(img: &JpegXsImage) -> Result<Vec<u8>> {
    if img.bit_depth != 8 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 2: requires Bw = 8, got {}",
            img.bit_depth
        )));
    }
    if !matches!(img.num_components, 1 | 3) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 2: Nc must be 1 or 3, got {}",
            img.num_components
        )));
    }
    if img.planes.len() != img.num_components as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: image planes ({}) != num_components ({})",
            img.planes.len(),
            img.num_components
        )));
    }
    let w = img.width as usize;
    let h = img.height as usize;
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(img.planes.len());
    for (i, plane) in img.planes.iter().enumerate() {
        if plane.stride != w {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 2: plane {i} stride {} != width {w} (no padding)",
                plane.stride
            )));
        }
        if plane.data.len() != w * h {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} data length {} != width*height {}",
                plane.data.len(),
                w * h
            )));
        }
        planes.push(plane.data.clone());
    }
    encode_planar(
        img.width as u16,
        img.height as u16,
        img.num_components,
        img.cpih,
        1,
        1,
        &planes,
    )
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

/// Core encoder entry point. `planes` is one byte buffer per component
/// (length `width * height` each). `cpih` selects the colour transform
/// (0 = none, 1 = RCT). `nlx`/`nly` are the wavelet decomposition levels
/// (round 2: must be equal and in `{1, 2}`).
pub fn encode_planar(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let cfg = EncodeConfig {
        width,
        height,
        nc,
        bit_depth: 8,
        bw: 8,
        ng: 4,
        ss: 8,
        br: 8,
        nlx,
        nly,
        cpih,
    };
    cfg.validate()?;
    let n = (width as usize) * (height as usize);
    if planes.len() != nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: expected {nc} component planes, got {}",
            planes.len()
        )));
    }
    for (i, p) in planes.iter().enumerate() {
        if p.len() != n {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} size {} != width*height {n}",
                p.len()
            )));
        }
    }
    let mut out = Vec::with_capacity(n * (nc as usize) + 256);
    write_main_header(&mut out, &cfg)?;
    write_slice(&mut out, &cfg, planes)?;
    // EOC.
    out.extend_from_slice(&[0xff, 0x11]);
    Ok(out)
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
    // CDT — Lcdt = 2 + 2*Nc, body = 2*Nc bytes.
    out.extend_from_slice(&[0xff, 0x13]);
    let lcdt = 2u16 + 2 * (cfg.nc as u16);
    out.extend_from_slice(&lcdt.to_be_bytes());
    for _ in 0..cfg.nc {
        out.push(cfg.bit_depth); // B[i] = 8
        out.push(0x11); // sx = 1, sy = 1 — round 2 is 4:4:4 only
    }
    // WGT — Lwgt = 2 + 2*N_existing. Round 2 has Nc * Nβ existing bands
    // (no Sd, no chroma sub-sampling, all bands exist).
    let nbeta = n_beta(cfg.nlx, cfg.nly) as u16;
    let n_existing = (cfg.nc as u16) * nbeta;
    out.extend_from_slice(&[0xff, 0x14]);
    let lwgt = 2 + 2 * n_existing;
    out.extend_from_slice(&lwgt.to_be_bytes());
    for _ in 0..n_existing {
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
    // Hsl = Np_y so all precincts land in a single slice. Hp = 2^NL,y →
    // Np_y = ceil(Hf / 2^NL,y).
    let hp_pow = 1u32 << cfg.nly;
    let np_y = (cfg.height as u32).div_ceil(hp_pow);
    out.extend_from_slice(&(np_y as u16).to_be_bytes());
    // Nc, Ng, Ss, Bw.
    out.push(cfg.nc);
    out.push(cfg.ng);
    out.push(cfg.ss);
    out.push(cfg.bw);
    // Fq:Br — Fq = 0 (lossless), Br = cfg.br.
    out.push(cfg.br & 0x0f);
    // Fslc:Ppoc:Cpih — Fslc=0 Ppoc=0 Cpih = caller-selected.
    out.push(cfg.cpih & 0x0f);
    // NL,x:NL,y.
    out.push(((cfg.nlx & 0x0f) << 4) | (cfg.nly & 0x0f));
    // Lh:Rl:Qpih:Fs:Rm — all zero.
    out.push(0x00);
}

/// Number of wavelet filter types `Nβ` per Annex B.3.
fn n_beta(nlx: u8, nly: u8) -> u32 {
    let mn = nlx.min(nly) as u32;
    let mx = nlx.max(nly) as u32;
    2 * mn + mx + 1
}

/// Per-(β, i) band geometry needed by the encoder. Mirrors the
/// `(dx, dy, τx, τy)` enumeration in [`crate::slice_walker`].
#[derive(Debug, Clone, Copy)]
struct EncBandKey {
    dx: u32,
    dy: u32,
    tau_x: bool,
    tau_y: bool,
}

fn beta_key(beta: u32, nlx: u8, nly: u8) -> EncBandKey {
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    if nly_u == 0 {
        if beta == 0 {
            return EncBandKey {
                dx: nlx_u,
                dy: 0,
                tau_x: false,
                tau_y: false,
            };
        }
        return EncBandKey {
            dx: nlx_u + 1 - beta,
            dy: 0,
            tau_x: true,
            tau_y: false,
        };
    }
    let beta1 = nlx_u - nly_u + 1;
    if beta < beta1 {
        if beta == 0 {
            return EncBandKey {
                dx: nlx_u,
                dy: nly_u,
                tau_x: false,
                tau_y: false,
            };
        }
        return EncBandKey {
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
        0 => EncBandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: false,
        },
        1 => EncBandKey {
            dx,
            dy,
            tau_x: false,
            tau_y: true,
        },
        _ => EncBandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: true,
        },
    }
}

/// Picture-level dimensions of band β for a `wc × hc` component under
/// (NL,x, NL,y). Mirrors the slice walker / cascade formula.
fn band_dims(wc: usize, hc: usize, nlx: u8, nly: u8, beta: u32) -> (usize, usize) {
    let key = beta_key(beta, nlx, nly);
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

/// Per-precinct band-row offset count `pow_h = 2^max(NL,y - dy, 0)`.
/// Mirrors the decoder's `cascade_band_pow_h` helper.
fn pow_h(nly: u8, dy: u32) -> usize {
    let nly_u = nly as u32;
    if dy >= nly_u || nly_u == 0 {
        1
    } else {
        1usize << (nly_u - dy)
    }
}

fn write_slice(out: &mut Vec<u8>, cfg: &EncodeConfig, planes_u8: &[Vec<u8>]) -> Result<()> {
    // SLH — Lslh = 4, Yslh = 0 (single slice covers the whole picture).
    out.extend_from_slice(&[0xff, 0x20]);
    out.extend_from_slice(&4u16.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes());

    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let nc = cfg.nc as usize;
    let dc_bias: i32 = 1 << (cfg.bw - 1);

    // 1) Convert pixels to i32 with DC level shift, optionally apply
    //    forward RCT (Cpih=1).
    let mut comp_planes: Vec<Vec<i32>> = planes_u8
        .iter()
        .map(|p| p.iter().map(|&v| v as i32 - dc_bias).collect::<Vec<i32>>())
        .collect();
    if cfg.cpih == 1 {
        let mut refs: Vec<&mut [i32]> = comp_planes.iter_mut().map(|p| p.as_mut_slice()).collect();
        forward_rct(&mut refs, w, h)?;
    }

    let nlx = cfg.nlx;
    let nly = cfg.nly;
    let multi_level = nlx > 1 || nly > 1;
    let hp_pow = 1u32 << nly;
    let np_y = (h as u32).div_ceil(hp_pow) as usize;

    if multi_level {
        // Multi-level path: forward cascade on the whole picture per
        // component, then slice each band into per-precinct row ranges.
        // Mirrors the decoder's gather-then-cascade synthesis.
        let nbeta = n_beta(nlx, nly) as usize;
        let mut bands_per_comp: Vec<Vec<Vec<i32>>> = Vec::with_capacity(nc);
        for plane in comp_planes.iter().take(nc) {
            let bands = forward_cascade_2d(w, h, nlx, nly, plane)?;
            if bands.len() != nbeta {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: forward_cascade_2d returned {} bands, expected {}",
                    bands.len(),
                    nbeta
                )));
            }
            bands_per_comp.push(bands);
        }
        for py in 0..np_y {
            let pbytes = encode_precinct_cascade(cfg, &bands_per_comp, py)?;
            out.extend_from_slice(&pbytes);
        }
    } else {
        // Single-level path (NL=1/1): per-precinct forward DWT into 4
        // sub-bands. Mirrors the decoder's `synthesise_precinct`
        // streaming path.
        for py in 0..np_y {
            let y0 = py * (hp_pow as usize);
            let y1 = (y0 + hp_pow as usize).min(h);
            let hp_real = y1 - y0;
            let pbytes = encode_precinct_single_level(cfg, &comp_planes, y0, y1, hp_real)?;
            out.extend_from_slice(&pbytes);
        }
    }
    Ok(())
}

/// Single-level encode (NL=1/1). Mirrors round-1 layout but extended
/// for `Nc ∈ {1, 3}` and odd dimensions. For partial bottom precincts
/// (`hp_real < hp_pow`) the input strip is padded up to `hp_pow` rows
/// using whole-sample symmetric reflection (matches the spec's view
/// that band heights derive from `Hf`, not from per-precinct sample
/// counts).
fn encode_precinct_single_level(
    cfg: &EncodeConfig,
    comp_planes: &[Vec<i32>],
    y0: usize,
    y1: usize,
    hp_real: usize,
) -> Result<Vec<u8>> {
    let w = cfg.width as usize;
    let nc = cfg.nc as usize;
    let hp_pow = 1usize << cfg.nly;

    // Per-(β, i) band buffer.  β order: LL=0, HL=1, LH=2, HH=3.
    // The picture-level band heights for component i derive from
    // `Hc[i] = Hf / sy[i]` (sy=1 here), per Annex B.2:
    //   Hb_LL = ceil(Hf/2),  Hb_LH = Hf/2  (floor division).
    // For each precinct the row-range L1 - L0 is at most pow_h, but at
    // the bottom edge it can be shorter when the precinct's row offset
    // overflows Hb. We compute that per band below.
    let h_full = cfg.height as usize;
    let pic_ll_h = h_full.div_ceil(2);
    let pic_lh_h = h_full / 2;

    // Forward DWT runs at the full precinct height (`hp_pow`). When
    // the real strip is shorter, pad with whole-sample symmetric
    // reflection to fill the missing rows. This way the decoder's
    // inverse_2d operates on a consistent geometry regardless of
    // partial precincts.
    let ll_w = w.div_ceil(2);
    let hl_w = w / 2;
    let ll_h_per_precinct = hp_pow.div_ceil(2);
    let lh_h_per_precinct = hp_pow / 2;

    let mut bands_per_comp: Vec<[Vec<i32>; 4]> = Vec::with_capacity(nc);
    for plane in comp_planes.iter().take(nc) {
        let mut strip: Vec<i32> = Vec::with_capacity(w * hp_pow);
        for y in y0..y1 {
            for x in 0..w {
                strip.push(plane[y * w + x]);
            }
        }
        // Pad with whole-sample symmetric reflection up to hp_pow rows.
        // For hp_real == 1, X[1] = X[1] (only row exists); reflect from
        // the only row.  For hp_real == 0 (shouldn't happen given np_y
        // computation), pad with zeros.
        while strip.len() < w * hp_pow {
            // Reflect: row at offset r where r = strip.len()/w copies
            // from row (2 * hp_real - r - 2) in the original. For
            // hp_real==1, hp_pow==2: r=1 → src row = -1 → folds to 0.
            let target_row = strip.len() / w;
            let src_row = if hp_real >= 2 {
                let mirrored = 2 * hp_real - target_row - 2;
                mirrored.min(hp_real - 1)
            } else {
                0
            };
            let row_start = src_row * w;
            for x in 0..w {
                let src_idx = if hp_real == 0 { 0 } else { row_start + x };
                let val = if hp_real == 0 { 0 } else { strip[src_idx] };
                strip.push(val);
            }
        }
        let mut ll = vec![0i32; ll_w * ll_h_per_precinct];
        let mut hl = vec![0i32; hl_w * ll_h_per_precinct];
        let mut lh = vec![0i32; ll_w * lh_h_per_precinct];
        let mut hh = vec![0i32; hl_w * lh_h_per_precinct];
        forward_2d(w, hp_pow, &strip, &mut ll, &mut hl, &mut lh, &mut hh)?;
        bands_per_comp.push([ll, hl, lh, hh]);
    }

    // Per-band number of lines this precinct contributes (Annex B.6).
    // pow_h = 2^max(NL,y - dy, 0).  For NL=1/1 with β ∈ {1,2,3} dy=1,
    // pow_h = 1, so each precinct contributes at most 1 line per band.
    // For LL (β=0, dy=NL,y=1, ty=false) L0=0; for HL (β=1, dy=1,
    // ty=false) L0=0; for LH/HH (β=2,3, dy=1, ty=true) L0=1 in
    // precinct-local coords. The band's picture-level row offset is
    //   row_offset = py * pow_h
    // and lines = pow_h.min(pic_band_h - row_offset).
    let py = y0 / hp_pow;
    let row_offset_ll = py; // pow_h = 1
    let row_offset_lh = py;
    let lines_ll_real = if row_offset_ll >= pic_ll_h {
        0
    } else {
        1.min(pic_ll_h - row_offset_ll)
    };
    let lines_lh_real = if row_offset_lh >= pic_lh_h {
        0
    } else {
        1.min(pic_lh_h - row_offset_lh)
    };

    // For partial bottom precincts the bands keep their full
    // `ll_h_per_precinct = 1` row, but only emit `lines_*_real` rows
    // into the entropy stream. The decoder's `BandCoefficients` is
    // sized to (L1 - L0) lines, so emitting fewer rows is correct.
    let _ = (lines_ll_real, lines_lh_real); // captured below in band_lines selector

    // Precinct header bits: 24 + 8 + 8 + 2 × N_existing.
    // For NL=1/1 single-level Sy=1 every band exists per Annex B.4
    // ((L0 % sy[i]) == 0), so n_existing = Nc * 4.
    let nbeta = 4u32;
    let n_bands = (nc as u32) * nbeta;
    let n_existing = n_bands as usize;
    let header_bits = 24 + 8 + 8 + 2 * n_existing;
    let header_bytes = header_bits.div_ceil(8);
    let mut precinct_bytes = vec![0u8; header_bytes];

    let mut entropy: Vec<u8> = Vec::new();

    // First packet: β=0, line 0 of LL for each component.
    let mut first_entries: Vec<(usize, usize, usize)> = Vec::new(); // (β_id, comp_i, line_off)
    for i in 0..nc {
        first_entries.push((0, i, 0));
    }

    let emit_packet = |entries: &[(usize, usize, usize)],
                       bands_per_comp: &[[Vec<i32>; 4]],
                       out: &mut Vec<u8>|
     -> Result<()> {
        let mut data_writer = BitWriter::default();
        let mut cnt_writer = BitWriter::default();
        for &(beta_idx, i, line_off) in entries {
            let band_buf = &bands_per_comp[i][beta_idx];
            // Per-band wpb in this precinct.
            let wpb = if beta_idx == 1 || beta_idx == 3 {
                hl_w
            } else {
                ll_w
            };
            // Per-band lines in band buffer (always full per-precinct
            // size). Per-precinct emitted lines (real) is enforced by
            // the caller via the entries list — entries beyond
            // lines_*_real are simply not added.
            let band_lines_in_buf = if beta_idx == 0 || beta_idx == 1 {
                ll_h_per_precinct
            } else {
                lh_h_per_precinct
            };
            if line_off >= band_lines_in_buf {
                continue;
            }
            let row_start = line_off * wpb;
            let row_end = row_start + wpb;
            let band_line: &[i32] = &band_buf[row_start..row_end];

            let ng_u = cfg.ng as usize;
            let ncg = wpb.div_ceil(ng_u);
            let m_max_for_br: u32 = if cfg.br >= 8 {
                255
            } else {
                (1u32 << cfg.br) - 1
            };
            let coef = |g: usize, k: usize| -> i32 {
                let xpos = g * ng_u + k;
                if xpos < wpb {
                    band_line[xpos]
                } else {
                    0
                }
            };
            let mut m_per_group = vec![0u8; ncg];
            for (g, slot) in m_per_group.iter_mut().enumerate() {
                let mut max_mag: u32 = 0;
                for k in 0..ng_u {
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
                        "jpegxs encoder: β={beta_idx} comp {i} group {g} bitplane count {m} exceeds Br = {} (cap {m_max_for_br}). Use a higher Br or quantize the input.",
                        cfg.br
                    )));
                }
                *slot = m as u8;
            }
            for &m in &m_per_group {
                cnt_writer.write_bits(m as u32, cfg.br);
            }
            for (g, &m_u8) in m_per_group.iter().enumerate() {
                let m = m_u8 as u32;
                if m == 0 {
                    continue;
                }
                for k in 0..ng_u {
                    let v = coef(g, k);
                    let sign_bit = if v < 0 { 1 } else { 0 };
                    data_writer.write_bit(sign_bit);
                }
                for plane in (0..m).rev() {
                    for k in 0..ng_u {
                        let v = coef(g, k);
                        let mag = v.unsigned_abs();
                        let bit = ((mag >> plane) & 1) as u8;
                        data_writer.write_bit(bit);
                    }
                }
            }
        }
        cnt_writer.align_to_byte();
        data_writer.align_to_byte();
        let cnt_bytes = cnt_writer.into_bytes();
        let data_bytes = data_writer.into_bytes();
        let lcnt = cnt_bytes.len() as u32;
        let ldat = data_bytes.len() as u32;
        let lsgn: u32 = 0;
        if ldat > (1 << 15) - 1 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Ldat = {ldat} exceeds short packet header capacity (15 bits)."
            )));
        }
        if lcnt > (1 << 13) - 1 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Lcnt = {lcnt} exceeds short packet header capacity (13 bits)."
            )));
        }
        let mut hdr_bits: u64 = 0;
        hdr_bits = (hdr_bits << 1) | 1; // Dr = 1
        hdr_bits = (hdr_bits << 15) | (ldat as u64 & 0x7fff);
        hdr_bits = (hdr_bits << 13) | (lcnt as u64 & 0x1fff);
        hdr_bits = (hdr_bits << 11) | (lsgn as u64 & 0x07ff);
        let mut header = vec![0u8; 5];
        for (k, byte) in header.iter_mut().enumerate() {
            *byte = ((hdr_bits >> (8 * (4 - k))) & 0xff) as u8;
        }
        out.extend_from_slice(&header);
        out.extend_from_slice(&cnt_bytes);
        out.extend_from_slice(&data_bytes);
        Ok(())
    };

    // First packet may itself be skipped if the LL band has no lines
    // for this precinct (`lines_ll_real == 0` for an empty bottom-edge
    // case — though for NL=1/1 with Hf >= 2, LL always has at least 1
    // line for every precinct except the wholly-empty trailing one,
    // which np_y excludes by construction).
    if lines_ll_real > 0 {
        emit_packet(&first_entries, &bands_per_comp, &mut entropy)?;
    }

    // Proxy levels: β = 1 (HL), 2 (LH), 3 (HH).
    for beta_idx in 1usize..4 {
        for i in 0..nc {
            let band_lines_real = if beta_idx == 1 {
                lines_ll_real
            } else {
                lines_lh_real
            };
            if band_lines_real == 0 {
                continue;
            }
            let entries = vec![(beta_idx, i, 0)];
            emit_packet(&entries, &bands_per_comp, &mut entropy)?;
        }
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
    precinct_bytes.extend_from_slice(&entropy);
    Ok(precinct_bytes)
}

/// Encode one precinct using the multi-level cascade band layout.
/// `bands_per_comp[i][β]` is the picture-level band buffer for filter
/// type β of component i. The encoder slices the per-precinct row range
/// out of each band based on `pow_h(β)`.
fn encode_precinct_cascade(
    cfg: &EncodeConfig,
    bands_per_comp: &[Vec<Vec<i32>>],
    py: usize,
) -> Result<Vec<u8>> {
    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let nc = cfg.nc as usize;
    let nlx = cfg.nlx;
    let nly = cfg.nly;
    let nbeta = n_beta(nlx, nly);
    let n_bands = (nc as u32) * nbeta;

    // For each (β, i) compute (lines, pic_band_row_offset, wpb) for
    // this precinct.
    struct Slice {
        wpb: usize,
        lines: usize,
        pic_bw: usize,
        pic_row_offset: usize,
    }
    let mut slices: Vec<Slice> = Vec::with_capacity(n_bands as usize);
    for beta in 0..nbeta {
        let key = beta_key(beta, nlx, nly);
        for _i in 0..nc {
            let (pic_bw, pic_bh) = band_dims(w, h, nlx, nly, beta);
            let pow = pow_h(nly, key.dy);
            let row_offset = py * pow;
            let lines = if row_offset >= pic_bh {
                0
            } else {
                pow.min(pic_bh - row_offset)
            };
            slices.push(Slice {
                wpb: pic_bw,
                lines,
                pic_bw,
                pic_row_offset: row_offset,
            });
        }
    }

    // Precinct header: Lprc(24) + Q(8) + R(8) + N_existing × D(2),
    // padded to byte boundary. Round 2: every band exists (Sd=0, no
    // chroma sub-sampling) → N_existing = n_bands.
    let n_existing = n_bands as usize;
    let header_bits = 24 + 8 + 8 + 2 * n_existing;
    let header_bytes = header_bits.div_ceil(8);
    let mut precinct_bytes = vec![0u8; header_bytes];

    // Build entropy stream: walk packets per Annex B.7 Table B.4.
    // First packet: β=0 .. β1-1, all components, line λ = L0.
    // Then proxy levels: β0 = β1, β1+3, …
    let mut entropy: Vec<u8> = Vec::new();
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let beta1 = nlx_u.max(nly_u) - nlx_u.min(nly_u) + 1;

    // Helper: emit the full packet (entries, body, header) for a list
    // of (band_id, line_offset_within_lines) items.
    let emit_packet = |entries: &[(usize, usize)],
                       bands_per_comp: &[Vec<Vec<i32>>],
                       slices: &[Slice],
                       out: &mut Vec<u8>|
     -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        // Build the (header, body) for the packet covering these entries.
        let mut data_writer = BitWriter::default();
        let mut cnt_writer = BitWriter::default();
        for &(b_id, line_off) in entries {
            let s = &slices[b_id];
            if s.lines == 0 {
                continue;
            }
            let beta = (b_id as u32) / (nc as u32);
            let i = (b_id as u32) % (nc as u32);
            let band_buf = &bands_per_comp[i as usize][beta as usize];
            // Picture row inside the band.
            let pic_row = s.pic_row_offset + line_off;
            let row_start = pic_row * s.pic_bw;
            let row_end = row_start + s.wpb;
            let band_line: &[i32] = &band_buf[row_start..row_end];
            // ---- bitplane-count sub-packet (raw mode): Br bits per group ----
            let ng_u = cfg.ng as usize;
            let ncg = s.wpb.div_ceil(ng_u);
            let m_max_for_br: u32 = if cfg.br >= 8 {
                255
            } else {
                (1u32 << cfg.br) - 1
            };
            let coef = |g: usize, k: usize| -> i32 {
                let xpos = g * ng_u + k;
                if xpos < s.wpb {
                    band_line[xpos]
                } else {
                    0
                }
            };
            // Per-group bitplane counts.
            let mut m_per_group = vec![0u8; ncg];
            for (g, slot) in m_per_group.iter_mut().enumerate() {
                let mut max_mag: u32 = 0;
                for k in 0..ng_u {
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
                        "jpegxs encoder: band {b_id} group {g} bitplane count {m} exceeds Br = {} (cap {m_max_for_br}). Use a higher Br or quantize the input.",
                        cfg.br
                    )));
                }
                *slot = m as u8;
            }
            for &m in &m_per_group {
                cnt_writer.write_bits(m as u32, cfg.br);
            }
            // ---- data sub-packet ----
            for (g, &m_u8) in m_per_group.iter().enumerate() {
                let m = m_u8 as u32;
                if m == 0 {
                    continue;
                }
                // Fs = 0 → write Ng sign bits first.
                for k in 0..ng_u {
                    let v = coef(g, k);
                    let sign_bit = if v < 0 { 1 } else { 0 };
                    data_writer.write_bit(sign_bit);
                }
                for plane in (0..m).rev() {
                    for k in 0..ng_u {
                        let v = coef(g, k);
                        let mag = v.unsigned_abs();
                        let bit = ((mag >> plane) & 1) as u8;
                        data_writer.write_bit(bit);
                    }
                }
            }
        }
        cnt_writer.align_to_byte();
        data_writer.align_to_byte();
        let cnt_bytes = cnt_writer.into_bytes();
        let data_bytes = data_writer.into_bytes();
        let lcnt = cnt_bytes.len() as u32;
        let ldat = data_bytes.len() as u32;
        let lsgn: u32 = 0;

        // Short header (Wf*Nc < 32752 holds for round-2 fixtures).
        if ldat > (1 << 15) - 1 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Ldat = {ldat} exceeds short packet header capacity (15 bits)."
            )));
        }
        if lcnt > (1 << 13) - 1 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Lcnt = {lcnt} exceeds short packet header capacity (13 bits)."
            )));
        }
        let mut hdr_bits: u64 = 0;
        hdr_bits = (hdr_bits << 1) | 1; // Dr = 1
        hdr_bits = (hdr_bits << 15) | (ldat as u64 & 0x7fff);
        hdr_bits = (hdr_bits << 13) | (lcnt as u64 & 0x1fff);
        hdr_bits = (hdr_bits << 11) | (lsgn as u64 & 0x07ff);
        let mut header = vec![0u8; 5];
        for (i, byte) in header.iter_mut().enumerate() {
            *byte = ((hdr_bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        out.extend_from_slice(&header);
        out.extend_from_slice(&cnt_bytes);
        out.extend_from_slice(&data_bytes);
        Ok(())
    };

    // First packet: β = 0 .. β1-1 × Nc components × line 0.
    let mut first_entries: Vec<(usize, usize)> = Vec::new();
    for beta in 0..beta1 {
        for i in 0..(nc as u32) {
            let b_id = ((nc as u32) * beta + i) as usize;
            let s = &slices[b_id];
            if s.lines == 0 {
                continue;
            }
            // Line λ = L0 → first line of the band slice = local index 0
            // when L0 == 0 (true for β < β1, no τy → τy_b = 0). Round 2
            // restricts to NL,x = NL,y, so β1 = 1 always — only the LL
            // band lands in this packet.
            first_entries.push((b_id, 0));
        }
    }
    emit_packet(&first_entries, bands_per_comp, &slices, &mut entropy)?;

    // Proxy levels.
    let mut beta0 = beta1;
    while beta0 < nbeta {
        // Number of band lines per precinct at this proxy level.
        let key0 = beta_key(beta0, nlx, nly);
        let pow = pow_h(nly, key0.dy);
        for lambda_within in 0..pow {
            for beta in beta0..(beta0 + 3).min(nbeta) {
                for i in 0..(nc as u32) {
                    let b_id = ((nc as u32) * beta + i) as usize;
                    let s = &slices[b_id];
                    if lambda_within >= s.lines {
                        continue;
                    }
                    // One packet per (band, line) entry — matches the
                    // per-line emission in `compute_packet_layouts`.
                    let entry = vec![(b_id, lambda_within)];
                    emit_packet(&entry, bands_per_comp, &slices, &mut entropy)?;
                }
            }
        }
        beta0 += 3;
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
    Ok(precinct_bytes)
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
                let v = ((x as i32) * 5 + (y as i32) * 7 + ((x ^ y) & 0x0f) as i32 * 3) % 256;
                buf[y * 32 + x] = v as u8;
            }
        }
        buf
    }

    fn make_synthetic_rgb_32x32() -> Vec<u8> {
        let mut buf = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let off = (y * 32 + x) * 3;
                buf[off] = (((x as i32) * 8 + y as i32) % 256) as u8; // R
                buf[off + 1] = (((y as i32) * 5 + x as i32 * 3) % 256) as u8; // G
                buf[off + 2] = ((x ^ y) as u8).wrapping_mul(13); // B
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

    // === Round 2: multi-component =======================================

    /// 3-component RGB without colour transform. Each plane should
    /// round-trip independently and bit-exactly.
    #[test]
    fn self_roundtrip_rgb_32x32_no_transform() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 0, 1).expect("encode RGB 32x32 (Cpih=0)");
        let img = decode_codestream(&codestream, None).expect("decode RGB Cpih=0");
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        assert_eq!(img.num_components, 3);
        // De-interleave the input for comparison plane-by-plane.
        let n = 32 * 32;
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r, "R plane");
        assert_eq!(img.planes[1].data, g, "G plane");
        assert_eq!(img.planes[2].data, b, "B plane");
    }

    /// 3-component RGB with forward RCT, Cpih=1. Lossless because RCT
    /// is reversible.
    #[test]
    fn self_roundtrip_rgb_32x32_rct() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 1, 1).expect("encode RGB 32x32 (Cpih=1)");
        let img = decode_codestream(&codestream, None).expect("decode RGB Cpih=1");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.cpih, 1);
        let n = 32 * 32;
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r, "R plane after RCT round-trip");
        assert_eq!(img.planes[1].data, g, "G plane after RCT round-trip");
        assert_eq!(img.planes[2].data, b, "B plane after RCT round-trip");
    }

    /// Multi-decomposition NL=2/2, single luma. Cascade + decoder
    /// gather-then-cascade must round-trip losslessly.
    #[test]
    fn self_roundtrip_luma_nl_2_2() {
        let pixels = make_synthetic_32x32();
        let codestream = encode_planar(32, 32, 1, 0, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode luma NL=2/2");
        let img = decode_codestream(&codestream, None).expect("decode luma NL=2/2");
        assert_eq!(img.planes[0].data, pixels, "NL=2/2 lossless luma");
    }

    /// Multi-decomposition NL=2/2 + RGB + RCT — the full bundle.
    #[test]
    fn self_roundtrip_rgb_nl_2_2_rct() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 1, 2).expect("encode RGB NL=2/2 Cpih=1");
        let img = decode_codestream(&codestream, None).expect("decode RGB NL=2/2 Cpih=1");
        let n = 32 * 32;
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

    // === Round 2: odd dimensions ========================================

    /// 31×31 odd dimensions, single luma, NL=1.
    #[test]
    fn self_roundtrip_odd_dimensions_31x31() {
        let mut pixels = vec![0u8; 31 * 31];
        for y in 0..31 {
            for x in 0..31 {
                pixels[y * 31 + x] = ((x * 11 + y * 7) % 256) as u8;
            }
        }
        let codestream = encode_luma_8bit(31, 31, &pixels).expect("encode 31x31");
        let img = decode_codestream(&codestream, None).expect("decode 31x31");
        assert_eq!(img.width, 31);
        assert_eq!(img.height, 31);
        assert_eq!(img.planes[0].data, pixels, "odd-dim 31x31 lossless");
    }

    /// 33×17 odd dimensions, NL=2/2 — exercises both odd width and
    /// odd height with cascaded forward DWT. Pinned: the slice walker
    /// `Wpb` formula is now τx-aware (matches `Wb` from the cascade),
    /// so the encoder + decoder agree on band geometry under odd
    /// dimensions for any `NL,x = NL,y`.
    #[test]
    fn self_roundtrip_odd_dimensions_33x17_nl_2_2() {
        let w = 33usize;
        let h = 17usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 13 + y * 23 + 5) % 256) as u8;
            }
        }
        let codestream = encode_planar(w as u16, h as u16, 1, 0, 2, 2, &[pixels.clone()])
            .expect("encode 33x17 NL=2/2");
        let img = decode_codestream(&codestream, None).expect("decode 33x17 NL=2/2");
        assert_eq!(img.width, w as u32);
        assert_eq!(img.height, h as u32);
        assert_eq!(img.planes[0].data, pixels);
    }

    /// `encode_image` accepts 3-component inputs (round 2 promotion).
    #[test]
    fn encode_image_rgb_round_trips() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let img = JpegXsImage {
            width: 32,
            height: 32,
            num_components: 3,
            cpih: 1,
            bit_depth: 8,
            planes: vec![
                JpegXsPlane {
                    stride: 32,
                    data: r.clone(),
                },
                JpegXsPlane {
                    stride: 32,
                    data: g.clone(),
                },
                JpegXsPlane {
                    stride: 32,
                    data: b.clone(),
                },
            ],
            pts: None,
        };
        let codestream = encode_image(&img).expect("encode_image RGB Cpih=1");
        let decoded = decode_codestream(&codestream, None).expect("decode RGB image");
        assert_eq!(decoded.planes[0].data, r);
        assert_eq!(decoded.planes[1].data, g);
        assert_eq!(decoded.planes[2].data, b);
    }

    /// Compression ratio sanity: round-2 raw-mode encoder produces
    /// codestreams whose size is dominated by per-precinct overhead +
    /// uncompressed magnitude bitplanes. We pin only that the encoder
    /// stays within a generous bound of raw RGB bytes — fine-tuned
    /// compression arrives with VLC bitplane-counts + significance
    /// coding in round 3.
    #[test]
    fn round2_codestream_within_size_bound() {
        let pixels = make_synthetic_rgb_32x32();
        let raw = pixels.len();
        let _cpih0_nl1 = encode_rgb_8bit(32, 32, &pixels, 0, 1).unwrap().len();
        let _cpih1_nl1 = encode_rgb_8bit(32, 32, &pixels, 1, 1).unwrap().len();
        let cpih1_nl2 = encode_rgb_8bit(32, 32, &pixels, 1, 2).unwrap().len();
        // Sanity bound: raw-mode entropy is bound to be larger than
        // raw on this size, but stay within 5x for the best mode.
        assert!(
            cpih1_nl2 < raw * 5,
            "best lossless codestream {cpih1_nl2} blew past 5x raw {raw}"
        );
    }

    /// Round 2 still rejects unsupported configs (Nc=2, NL>2, etc.).
    #[test]
    fn rejects_unsupported_configurations() {
        let pixels = vec![0u8; 32 * 32];
        // NL=3 not yet supported.
        assert!(encode_planar(32, 32, 1, 0, 3, 3, std::slice::from_ref(&pixels)).is_err());
        // Asymmetric NL not yet supported.
        assert!(encode_planar(32, 32, 1, 0, 2, 1, std::slice::from_ref(&pixels)).is_err());
        // Nc=2 not yet supported.
        let two = vec![pixels.clone(), pixels.clone()];
        assert!(encode_planar(32, 32, 2, 0, 1, 1, &two).is_err());
        // Cpih=1 with Nc=1 invalid.
        assert!(encode_planar(32, 32, 1, 1, 1, 1, &[pixels]).is_err());
    }
}
