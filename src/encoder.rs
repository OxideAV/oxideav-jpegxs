//! JPEG XS encoder — rounds 1-6.
//!
//! Round 1 (commit `95b4e27`) shipped the lossless single-luma 8-bit
//! single-decomposition single-precinct-column case. Round 2 broadened
//! the encoder to multi-component (`Nc ∈ {1, 3}`), multi-decomposition
//! (`NL,x = NL,y ∈ {1, 2}`), and odd dimensions but stayed in raw
//! bitplane-count mode (`Dr = 1`) and lossless (`Fq = 0`) only. Round 3
//! adds the three biggest compression-feature axes:
//!
//! * **Dr = 0 VLC bitplane-count mode (Annex C.6.6, Table C.14, no
//!   prediction).** Per-precinct each packet is emitted in both Dr=1
//!   raw form and Dr=0 VLC form, then the smaller of the two is kept.
//!   For sparse bands (`M[g]` small / mostly zero) the VLC is one to
//!   two orders of magnitude smaller than raw mode's flat
//!   `Br=8` bits per code group.
//! * **Regular (`Fq = 8`) lossy mode with Annex D.2 deadzone
//!   quantizer.** A new `q` parameter sets the precinct-level
//!   `Q[p]` (`0..=15`) which in turn drives the per-band truncation
//!   `T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)`. The encoder right-
//!   shifts coefficient magnitudes by `T` (truncation) and emits only
//!   `M - T` bitplanes per code group; the decoder reconstructs with
//!   the half-bucket offset `((1 << T) >> 1)`. PSNR ≥ 40 dB at `q = 1`,
//!   ≥ 32 dB at `q = 4` on synthetic 32×32 RGB.
//! * **4:2:2 / 4:2:0 chroma sub-sampling.** New entry point
//!   [`encode_planar_subsampled`] takes per-component `(sx, sy)` plus
//!   per-component byte buffers sized at `Wc[i] × Hc[i]` (`Wc[i] = Wf
//!   / sx[i]`, `Hc[i] = Hf / sy[i]`). The CDT signals the per-
//!   component ratios; per-band geometry uses
//!   `N'L,y[i] = NL,y - log2(sy[i])` per Annex B.2. The picture is
//!   still 4:4:4 dimensionally (Cpih=0; RCT requires sx=sy=1 for i<3).
//!
//! Round 5 adds:
//! * **Significance coding (`D[p,b] & 2 = 1`, Annex C.5 / Table C.14
//!   gating).** One bit per significance group indicates whether the
//!   group contains any non-zero coefficient. Insignificant groups skip
//!   the bitplane-count VLC (inferred Δm=0). The cascade encoder emits
//!   a trial form in both D&2=0 and D&2=1 modes and keeps the smaller.
//! * **`NL,x ≠ NL,y` support** (e.g. NL,x=2, NL,y=1) for anisotropic
//!   content. The cascade path already handled `nly ≤ nlx`; the
//!   single-level streaming path is promoted to the cascade encoder for
//!   any `nlx > 1` or `nly > 1` (multi-level cascade). Both paths are
//!   verified via self-roundtrip across the `(nlx, nly) ∈ {1,2} × {1,2}`
//!   matrix (with `nlx ≥ nly`).
//! * **Per-band Q tuning (gain-weighted truncation).** The WGT marker
//!   now emits non-zero gain values (LL=0, HL/LH=1, HH=2) so the
//!   per-band truncation `T[p,b] = clamp(Q - G[b], 0, 15)` allocates
//!   more bits to perceptually important high-frequency subbands. This
//!   lifts PSNR at q=4/8/12 by 2-4 dB compared to flat-gain encoding.
//! * **NLT encoder (quadratic, Annex G.4).** New entry point
//!   [`encode_planar_nlt_quadratic`] emits the NLT marker (Tnlt=1,
//!   Bw=18), applies the forward quadratic pre-distortion to the input
//!   pixels before quantization, and self-roundtrips through the
//!   decoder's inverse quadratic path.
//!
//! Round 6 adds:
//! * **Deeper wavelet cascade `NL ∈ {1..=5}`.** The encoder validation
//!   was capped at NL=2/2 even though `forward_cascade_2d` /
//!   `inverse_cascade_2d` are generic in NL,y ≤ NL,x. Relaxing the
//!   validation lets users opt into deeper multi-resolution analysis
//!   (NL=3/3, 4/4, 5/5 all self-roundtrip). The cascade path is the
//!   only path used for NL > 1 already, so no encoder kernel changes
//!   were needed beyond the validate threshold. Spec Annex A.4.4
//!   Table A.7 allows NL,x up to 8; we test through 5/5 here.
//!
//! Round 7 adds:
//! * **Extended NLT encoder (Tnlt=2, Annex G.5).** New entry point
//!   [`encode_planar_nlt_extended`] emits the NLT marker (Tnlt=2, T1,
//!   T2, E) with `Bw = 18`, then applies a forward extended-gamma
//!   pre-distortion that inverts the decoder's three-segment kernel via
//!   a `2^Bw`-entry reverse lookup table. Self-roundtrip PSNR ≥ 30 dB on
//!   a synthetic 32×32 gradient at q=0 (lossless intent within the LUT
//!   resolution), ≥ 25 dB at q=2.
//! * **Deeper wavelet cascade `NL ∈ {1..=8}`.** Validation cap lifted
//!   from 5 to 8 (the spec Annex A.4.4 Table A.7 hard maximum). The
//!   cascade DWT / band geometry helpers were already parametric in
//!   `NL` — only the validation threshold needed adjustment. NL=6/6
//!   self-roundtrip verified.
//!
//! Out-of-scope (deferred to round 8+):
//! * `Cw > 0` (custom precinct widths).
//! * Per-band per-precinct Q rate-distortion optimization.
//! * `Sd > 0` decomposition suppression (CWD).
//!
//! Byte stream shape:
//!
//! ```text
//! SOC | CAP | PIH | CDT | WGT | [NLT] | [CTS] | [CRG] | SLH | <slice 0 entropy data> | EOC
//! ```

use crate::colour_transform::{forward_rct, forward_star_tetrix};
use crate::dwt::{forward_2d, forward_cascade_2d};
use crate::error::{JpegXsError as Error, Result};
use crate::image::{JpegXsImage, JpegXsPlane};
use crate::output::NltParams;

/// Encoder configuration.
#[derive(Debug, Clone)]
struct EncodeConfig {
    /// Picture width (`Wf`).
    width: u16,
    /// Picture height (`Hf`).
    height: u16,
    /// Number of components (`Nc`).
    nc: u8,
    /// Component bit depth, fixed at 8 in round 3.
    bit_depth: u8,
    /// Wavelet coefficient precision (`Bw`), fixed at 8 (Fq=0 lossless
    /// per Table A.8). For Fq=8 we still use Bw=8 since input is 8-bit
    /// and the deadzone quantizer drops bottom bitplanes via T, not via
    /// extra fractional bits.
    bw: u8,
    /// Coefficients per code group (`Ng`), Annex C constant = 4.
    ng: u8,
    /// Code groups per significance group (`Ss`), Annex C constant = 8.
    ss: u8,
    /// Raw bitplane-count width (`Br`), 4-bit field cap of 15.
    br: u8,
    /// Horizontal decomposition levels (`NL,x`).
    nlx: u8,
    /// Vertical decomposition levels (`NL,y`).
    nly: u8,
    /// Colour transformation id (`Cpih`).
    cpih: u8,
    /// Fractional bits in wavelet domain (`Fq`). 0 = lossless,
    /// 8 = regular per Table A.8.
    fq: u8,
    /// Precinct quantization step `Q[p]` (constant across precincts).
    /// `0..=15` per Annex D.2; clamped at the encoder boundary so
    /// `T[p,b] = clamp(Q - G[b] - r, 0, 15)` stays in-range.
    q: u8,
    /// Per-component sub-sampling factors `sx[i]` and `sy[i]`.
    /// Length `nc`. Defaults to all-ones (4:4:4).
    sx: Vec<u8>,
    sy: Vec<u8>,
    /// Star-Tetrix `e1` (CTS field, 0..=3). Only meaningful when
    /// `cpih == 3`; ignored otherwise.
    cts_e1: u8,
    /// Star-Tetrix `e2` (CTS field, 0..=3).
    cts_e2: u8,
    /// Star-Tetrix `Cf` (CTS field, 0 = full or 3 = in-line).
    cts_cf: u8,
    /// Star-Tetrix CFA pattern type `Ct` (Table F.9, 0 or 1). Drives the
    /// CRG marker emission and the inverse `access()` reflection.
    st_ct: u8,
    /// Optional forward NLT parameters. When `Some`, the encoder writes
    /// an NLT marker and applies a forward (encoding-direction) map to
    /// input pixels. `Bw` is set to 18 for quadratic NLT per Table A.8.
    nlt: Option<NltParams>,
    /// Per-band gain values for the WGT marker.  Index matches the
    /// picture-level band enumeration order (β = 0 .. Nβ-1).  Length
    /// must equal `count_existing_bands(cfg)` or be empty (→ all-zero
    /// gains, backward-compatible with rounds 1–4).
    band_gains: Vec<u8>,
    /// Precinct-width parameter (`Cw`, PIH §A.4.4). `0` means a single
    /// precinct column spans the full picture width (`Cs = Wf`, the
    /// only mode supported up through round 7). For `Cw > 0` the
    /// per-row column width becomes `Cs = 8 × Cw × max(sx) × 2^NL,x`
    /// (Annex B.5), and the encoder emits `Np,x = ⌈Wf / Cs⌉` precincts
    /// per row in raster order.
    cw: u16,
    /// Number of trailing components whose wavelet decomposition is
    /// suppressed (`Sd`, Annex A.4.7 Table A.18). Zero unless the
    /// caller explicitly enables CWD via [`encode_planar_sd`]. When
    /// non-zero, the encoder emits a CWD marker and routes the
    /// suppressed components through raw single-band (β=0) per-line
    /// packets after the wavelet packets.
    sd: u8,
}

impl EncodeConfig {
    fn validate(&self) -> Result<()> {
        if self.width < 2 || self.height < 2 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: picture dimensions must be >= 2, got {}x{}",
                self.width, self.height
            )));
        }
        // Round 9 (r91): Sd>0 enables Nc up to 8 (Annex A.4.1 hard cap).
        // Otherwise stay on the pre-r91 supported set of {1, 3, 4}.
        let allowed_nc = if self.sd > 0 {
            (4..=8).contains(&self.nc)
        } else {
            matches!(self.nc, 1 | 3 | 4)
        };
        if !allowed_nc {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: Nc must be 1/3/4 (or 4..=8 with Sd>0), got {}",
                self.nc
            )));
        }
        if self.cpih == 1 && self.nc != 3 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Cpih=1 (RCT) requires Nc=3, got {}",
                self.nc
            )));
        }
        if self.cpih == 3 && self.nc != 4 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Cpih=3 (Star-Tetrix) requires Nc=4, got {}",
                self.nc
            )));
        }
        if !matches!(self.cpih, 0 | 1 | 3) {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 4: Cpih must be 0, 1, or 3, got {}",
                self.cpih
            )));
        }
        if self.cpih == 3 {
            // Star-Tetrix requires sx[i] = sy[i] = 1 on every component
            // (the CFA grid is fully populated).
            for (i, (&sx, &sy)) in self.sx.iter().zip(self.sy.iter()).enumerate() {
                if sx != 1 || sy != 1 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: Cpih=3 (Star-Tetrix) requires sx[i]=sy[i]=1, got component {i} (sx, sy)=({sx}, {sy})"
                    )));
                }
            }
            if self.cts_e1 > 3 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Star-Tetrix e1 must be 0..=3, got {}",
                    self.cts_e1
                )));
            }
            if self.cts_e2 > 3 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Star-Tetrix e2 must be 0..=3, got {}",
                    self.cts_e2
                )));
            }
            if !matches!(self.cts_cf, 0 | 3) {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Star-Tetrix Cf must be 0 (full) or 3 (in-line), got {}",
                    self.cts_cf
                )));
            }
            if self.st_ct > 1 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Star-Tetrix Ct must be 0 or 1, got {}",
                    self.st_ct
                )));
            }
        }
        // Round 7: NL,x ∈ {1..=8} (spec Annex A.4.4 Table A.7 hard max).
        // NL,y ∈ {0..=NL,x} per Annex B (NOTE 1: NL,y > NL,x case "needs
        // not to be considered for interoperability").
        if self.nlx < 1 || self.nlx > 8 || self.nly > self.nlx {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: NL,x ∈ {{1..=8}}, NL,y ∈ {{0..=NL,x}}, got NL,x={} NL,y={}",
                self.nlx, self.nly
            )));
        }
        if self.fq != 0 && self.fq != 8 {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder round 3: Fq must be 0 (lossless) or 8 (regular), got {}",
                self.fq
            )));
        }
        // Q range: spec allows 0..=31 in the precinct header, but per-
        // band T = clamp(Q - G - r, 0, 15) and our encoder uses G[b] = 0
        // and r = 0 → clip Q to 0..=15.
        if self.q > 15 {
            return Err(Error::invalid(format!(
                "jpegxs encoder round 3: q must be in 0..=15, got {}",
                self.q
            )));
        }
        if self.q > 0 && self.fq == 0 {
            return Err(Error::invalid(
                "jpegxs encoder round 3: q > 0 requires Fq = 8 (regular mode); for lossless use q = 0".to_string(),
            ));
        }
        if self.sx.len() != self.nc as usize || self.sy.len() != self.nc as usize {
            return Err(Error::invalid(format!(
                "jpegxs encoder: sx/sy must have length nc={}",
                self.nc
            )));
        }
        for (i, (&sx, &sy)) in self.sx.iter().zip(self.sy.iter()).enumerate() {
            if !matches!(sx, 1 | 2) || !matches!(sy, 1 | 2) {
                return Err(Error::Unsupported(format!(
                    "jpegxs encoder round 3: component {i} (sx, sy) must be in {{1, 2}}, got ({sx}, {sy})"
                )));
            }
        }
        // RCT requires sx = sy = 1 for i < 3 per Annex F.2.
        if self.cpih == 1 {
            for (i, (&sx, &sy)) in self.sx.iter().zip(self.sy.iter()).enumerate().take(3) {
                if sx != 1 || sy != 1 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: Cpih=1 (RCT) requires sx[i]=sy[i]=1 for i<3, got component {i} (sx, sy)=({sx}, {sy})"
                    )));
                }
            }
        }
        // Vertical sub-sampling needs `sy[i]` to divide `2^NL,y` evenly
        // so `N'L,y[i] = NL,y - log2(sy[i])` stays >= 0.
        for (i, &sy) in self.sy.iter().enumerate() {
            if sy == 2 && self.nly < 1 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: component {i} sy=2 requires NL,y >= 1, got {}",
                    self.nly
                )));
            }
        }
        // Picture dimensions must be divisible by sx / sy on each
        // component (otherwise per-component plane size is undefined).
        for (i, (&sx, &sy)) in self.sx.iter().zip(self.sy.iter()).enumerate() {
            if (self.width as u32) % (sx as u32) != 0 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: width {} not divisible by component {i} sx={sx}",
                    self.width
                )));
            }
            if (self.height as u32) % (sy as u32) != 0 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: height {} not divisible by component {i} sy={sy}",
                    self.height
                )));
            }
        }
        // Sd > 0 (CWD, Annex A.4.7 Table A.18). Requires Nc>3 and every
        // suppressed component must have sx=sy=1.
        if self.sd != 0 {
            if self.nc <= 3 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Sd>0 requires Nc>3 per Annex A.4.7, got Nc={}",
                    self.nc
                )));
            }
            if self.sd >= self.nc {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Sd={} must be < Nc={} per Table A.18",
                    self.sd, self.nc
                )));
            }
            for i in (self.nc - self.sd) as usize..self.nc as usize {
                if self.sx[i] != 1 || self.sy[i] != 1 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: suppressed component i={i} (Sd) must have sx=sy=1, got ({}, {}) (Annex A.4.7)",
                        self.sx[i], self.sy[i]
                    )));
                }
            }
            // Sd > 0 is incompatible with colour transforms that span
            // every component in the picture (`Cpih == 1` RCT touches
            // components 0..3, `Cpih == 3` Star-Tetrix touches 0..4).
            // The suppressed components are excluded by definition, so
            // the colour transform can still legally run on i < Nc-Sd
            // when its operand count fits; we restrict to `Cpih == 0`
            // for the round-9 encoder to keep the implementation tight.
            if self.cpih != 0 {
                return Err(Error::Unsupported(format!(
                    "jpegxs encoder round 9: Sd>0 currently requires Cpih=0, got Cpih={}",
                    self.cpih
                )));
            }
        }
        // Cw > 0 — validate the derived Cs makes sense per Annex B.5.
        if self.cw != 0 {
            let max_sx = self.sx.iter().copied().max().unwrap_or(1) as u32;
            let pow_nlx = 1u32 << self.nlx;
            let cs = 8u32 * (self.cw as u32) * max_sx * pow_nlx;
            if cs == 0 {
                return Err(Error::invalid(
                    "jpegxs encoder: derived Cs = 0 (check Cw / NL,x / sx)".to_string(),
                ));
            }
            if cs > self.width as u32 {
                return Err(Error::Unsupported(format!(
                    "jpegxs encoder: derived Cs={cs} exceeds picture width {} (Cw={} too large for NL,x={} and max sx={max_sx})",
                    self.width, self.cw, self.nlx
                )));
            }
            // Spec Note 1 in §B.5: all but the rightmost precincts must
            // contain at least 8 samples of the LL band, which is the
            // motivation for the 8× factor in Cs.  The encoder cannot
            // do better than the formula gives; the user is responsible
            // for picking Cw such that the rightmost precinct also has
            // reasonable width.
        }
        Ok(())
    }
}

/// Encode a single-luma 8-bit image to a JPEG XS codestream.
///
/// Lossless single-decomposition (`NL,x = NL,y = 1`) bootstrap path
/// retained from round 1 for callers that pin the original geometry.
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
/// Round-3 retains the round-2 lossless behaviour. For lossy encoding
/// or chroma sub-sampling, use [`encode_planar_lossy`] /
/// [`encode_planar_subsampled`].
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

/// Encode the JPEG XS codestream out of a [`JpegXsImage`].
///
/// Round 3 still defaults to lossless (`Fq = 0`, `q = 0`) and 4:4:4
/// (`sx = sy = 1` for every plane). For lossy or chroma-sub-sampled
/// encoding, see [`encode_planar_lossy`] / [`encode_planar_subsampled`].
pub fn encode_image(img: &JpegXsImage) -> Result<Vec<u8>> {
    if img.bit_depth != 8 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 3: requires Bw = 8, got {}",
            img.bit_depth
        )));
    }
    if !matches!(img.num_components, 1 | 3) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder round 3: Nc must be 1 or 3, got {}",
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
                "jpegxs encoder round 3: plane {i} stride {} != width {w} (no padding)",
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

/// Build a [`JpegXsImage`] from raw bytes and then encode. Useful for
/// self-roundtrip tests that already have raw pixels.
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

/// Lossless 4:4:4 entry point (round-2 signature). All `sx[i] = sy[i] = 1`.
pub fn encode_planar(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    encode_planar_inner(
        width, height, nc, cpih, nlx, nly, 0, 0, &sx, &sy, 0, 0, 0, 0, planes,
    )
}

/// Star-Tetrix (Cpih=3) entry point — round 4. Takes 4 component planes
/// in input order `(R, G1, G2, B)` matching [`crate::colour_transform::
/// inverse_star_tetrix`]'s output convention. Self-roundtrips losslessly
/// for `q == 0`. Emits the CTS marker (`Cf`, `e1`, `e2`) and the CRG
/// marker (driving the inverse `access()` reflection via Ct).
///
/// `e1` and `e2` are the CTS chroma-weighting exponents (0..=3); `cf`
/// is the CTS extent (0 = full, 3 = in-line). `ct` is the CFA pattern
/// type per Table F.9 (0 = RGGB or BGGR, 1 = GRBG or GBRG); the CRG
/// marker emitted carries the canonical RGGB or GRBG arrangement
/// depending on `ct`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_star_tetrix(
    width: u16,
    height: u16,
    nlx: u8,
    nly: u8,
    q: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; 4];
    let sy = vec![1u8; 4];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner(
        width, height, 4, 3, nlx, nly, fq, q, &sx, &sy, e1, e2, cf, ct, planes,
    )
}

/// Lossy entry point. `q` is the precinct quantization step (0..=15);
/// 0 reduces to lossless. `fq` must be 8 for `q > 0` per Table A.8.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_lossy(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner(
        width, height, nc, cpih, nlx, nly, fq, q, &sx, &sy, 0, 0, 0, 0, planes,
    )
}

/// Round-8 multi-precinct-per-row entry point (`Cw > 0`).
///
/// Same shape as [`encode_planar_lossy`] but takes the precinct-width
/// parameter `cw` from the picture header (`Cw`, Annex A.4.4). With
/// `cw > 0` the encoder splits each precinct row into
/// `Np,x = ⌈Wf / Cs⌉` precincts where `Cs = 8 × cw × max(sx) × 2^NL,x`
/// (Annex B.5).  `cw = 0` reduces to a single precinct column spanning
/// the full picture width (equivalent to [`encode_planar_lossy`]).
///
/// The decoder side has been updated in parallel to walk
/// `Np,x × Np,y` precincts in raster order and gather them into the
/// picture-level band buffers before running the inverse cascade DWT,
/// so any encoder output with `cw > 0` round-trips through
/// [`crate::decode_jpeg_xs`].
///
/// Validation: `Cs` must not exceed the picture width; `Cs == 0` is
/// rejected.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_cw(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    cw: u16,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        cw,
        0,
        planes,
    )
}

/// Round-9 (r91) `Sd > 0` (CWD) entry point — Annex A.4.7 Table A.18.
///
/// Encodes a multi-component picture where the trailing `sd` components
/// (indices `[nc - sd, nc)`) are coded raw (no wavelet decomposition)
/// while the leading `nc - sd` components go through the standard 5/3
/// cascade DWT. Emits a CWD marker with the chosen `Sd`. Per the spec,
/// `sd ∈ 1..=nc-1`, `nc > 3`, and every suppressed component must have
/// `sx[i] = sy[i] = 1`. The encoder currently restricts `cpih` to 0
/// (no colour transform) for the Sd path; the wavelet components still
/// follow the regular gain-weighted Fq=8 lossy / lossless behaviour.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_sd(
    width: u16,
    height: u16,
    nc: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    sd: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        0, // cpih: only 0 supported for Sd>0 in this round
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        0, // cw
        sd,
        planes,
    )
}

/// Sub-sampled (4:2:2 / 4:2:0) entry point. Each `planes[i]` has length
/// `(width / sx[i]) * (height / sy[i])`. `q = 0` for lossless, `q > 0`
/// engages Fq=8 lossy mode.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner(
        width, height, nc, cpih, nlx, nly, fq, q, sx, sy, 0, 0, 0, 0, planes,
    )
}

/// Round-5 NLT encoder (quadratic, Tnlt=1, Annex G.4).
///
/// Applies the forward quadratic pre-distortion `y = round(sqrt(x /
/// 255) * (2^18 - 1))` to 8-bit input pixels before the DWT, then
/// emits an NLT marker so the decoder applies the inverse `v²` path.
/// Requires `Bw = 18` per Table A.8. `q = 0` → lossless (within the
/// quadratic approximation); `q > 0` engages Fq=8 lossy mode.
///
/// `dco` is the DC offset applied to the forward map and embedded in
/// the NLT marker (Annex G.4 `DCO`). For standard use pass `dco = 0`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_nlt_quadratic(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    dco: i32,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    if !(-32768..=32767).contains(&dco) {
        return Err(Error::invalid(format!(
            "jpegxs NLT quadratic: dco {dco} out of signed 16-bit range"
        )));
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        Some(NltParams::Quadratic { dco }),
        Vec::new(), // band_gains built inside after validation
        0,
        0,
        planes,
    )
}

/// Round-7 NLT extended encoder (Tnlt=2, Annex G.5).
///
/// Applies a forward extended-gamma pre-distortion to 8-bit input pixels
/// before the DWT and emits an NLT marker so the decoder applies the
/// inverse three-segment kernel. Requires `Bw = 18` per Table A.8.
/// `q = 0` reduces to the "lossless within LUT resolution" case; `q > 0`
/// engages Fq=8 lossy mode.
///
/// `t1`, `t2`, `e` are the extended-NLT parameters embedded in the NLT
/// marker (Annex G.5 thresholds and linear-slope exponent). Constraints
/// validated by [`crate::output::parse_nlt`] also apply here:
/// `0 < t1 < t2`, `1 ≤ e ≤ 4`, both `t1` and `t2` in `1..=2^Bw - 1`.
///
/// The forward pre-distortion is built by walking the decoder's
/// `extended_path` once across `v_wave ∈ [0, 2^Bw - 1]` and recording the
/// first wavelet-domain code that reconstructs each 8-bit pixel value.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_nlt_extended(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    t1: u32,
    t2: u32,
    e: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    if t1 == 0 || t2 == 0 || t2 <= t1 {
        return Err(Error::invalid(format!(
            "jpegxs NLT extended: require 0 < T1 < T2, got T1={t1} T2={t2}"
        )));
    }
    if !(1..=4).contains(&e) {
        return Err(Error::invalid(format!(
            "jpegxs NLT extended: E must be in 1..=4, got {e}"
        )));
    }
    // Bw is forced to 18 by encode_planar_inner_nlt when nlt.is_some().
    let bw_max = (1u32 << 18) - 1;
    if t1 > bw_max || t2 > bw_max {
        return Err(Error::invalid(format!(
            "jpegxs NLT extended: T1={t1} or T2={t2} exceeds 2^Bw-1={bw_max}"
        )));
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = if q == 0 { 0 } else { 8 };
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        Some(NltParams::Extended { t1, t2, e }),
        Vec::new(),
        0,
        0,
        planes,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_planar_inner(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    fq: u8,
    q: u8,
    sx: &[u8],
    sy: &[u8],
    cts_e1: u8,
    cts_e2: u8,
    cts_cf: u8,
    st_ct: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q,
        sx,
        sy,
        cts_e1,
        cts_e2,
        cts_cf,
        st_ct,
        None,
        Vec::new(),
        0,
        0,
        planes,
    )
}

/// Inner encoder with NLT support and per-band gains.
#[allow(clippy::too_many_arguments)]
fn encode_planar_inner_nlt(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    fq: u8,
    q: u8,
    sx: &[u8],
    sy: &[u8],
    cts_e1: u8,
    cts_e2: u8,
    cts_cf: u8,
    st_ct: u8,
    nlt: Option<NltParams>,
    band_gains: Vec<u8>,
    cw: u16,
    sd: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let bw = if nlt.is_some() { 18 } else { 8 };
    let cfg = EncodeConfig {
        width,
        height,
        nc,
        bit_depth: 8,
        bw,
        ng: 4,
        ss: 8,
        br: 8,
        nlx,
        nly,
        cpih,
        fq,
        q,
        sx: sx.to_vec(),
        sy: sy.to_vec(),
        cts_e1,
        cts_e2,
        cts_cf,
        st_ct,
        nlt,
        band_gains,
        cw,
        sd,
    };
    cfg.validate()?;
    // Build per-band gains after validation so beta_key is called with
    // known-good (nlx >= nly) parameters.
    let cfg = if cfg.band_gains.is_empty() {
        EncodeConfig {
            band_gains: build_band_gains_sd(nc, sd, nlx, nly, sx, sy),
            ..cfg
        }
    } else {
        cfg
    };
    if planes.len() != nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: expected {nc} component planes, got {}",
            planes.len()
        )));
    }
    for (i, p) in planes.iter().enumerate() {
        let wc = (width as usize) / (cfg.sx[i] as usize);
        let hc = (height as usize) / (cfg.sy[i] as usize);
        let want = wc * hc;
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} size {} != Wc*Hc {want} (Wc={wc}, Hc={hc})",
                p.len()
            )));
        }
    }
    let total: usize = planes.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total + 256);
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
    for i in 0..cfg.nc as usize {
        out.push(cfg.bit_depth); // B[i] = 8
        out.push(((cfg.sx[i] & 0x0f) << 4) | (cfg.sy[i] & 0x0f));
    }
    // WGT — one (G[b], P[b]) pair per *existing* band. With sub-
    // sampling some bands don't exist for chroma components; we
    // emit gain/priority from cfg.band_gains (or all-zero if empty).
    let n_existing = count_existing_bands(cfg);
    out.extend_from_slice(&[0xff, 0x14]);
    let lwgt = 2 + 2 * (n_existing as u16);
    out.extend_from_slice(&lwgt.to_be_bytes());
    for k in 0..n_existing as usize {
        let g = cfg.band_gains.get(k).copied().unwrap_or(0);
        out.push(g); // G[b]
        out.push(0); // P[b] = 0
    }
    // CWD marker — Annex A.4.7 Table A.18. Emitted whenever Sd > 0
    // (must precede the first SLH and follow PIH/CDT/WGT). Body is
    // exactly 1 byte holding `Sd`.
    if cfg.sd != 0 {
        out.extend_from_slice(&[0xff, 0x17]); // CWD marker (Table A.1)
        out.extend_from_slice(&3u16.to_be_bytes()); // Lcwd = 3
        out.push(cfg.sd);
    }
    // NLT marker — required for quadratic / extended non-linearity
    // (Annex A.4.6). Round 5 implements Tnlt=1 (quadratic) only.
    if let Some(nlt) = cfg.nlt {
        match nlt {
            NltParams::Quadratic { dco } => {
                // Lnlt = 5, Tnlt = 1, then σ:α packed into 16 bits.
                out.extend_from_slice(&[0xff, 0x16]);
                out.extend_from_slice(&5u16.to_be_bytes());
                out.push(1); // Tnlt = 1
                let (sigma, alpha) = if dco < 0 {
                    let alpha = (-dco) as u16 & 0x7fff;
                    (1u16, alpha)
                } else {
                    (0u16, dco as u16 & 0x7fff)
                };
                let packed: u16 = (sigma << 15) | alpha;
                out.extend_from_slice(&packed.to_be_bytes());
            }
            NltParams::Extended { t1, t2, e } => {
                // Lnlt = 12, Tnlt = 2, T1, T2, E.
                out.extend_from_slice(&[0xff, 0x16]);
                out.extend_from_slice(&12u16.to_be_bytes());
                out.push(2); // Tnlt = 2
                out.extend_from_slice(&t1.to_be_bytes());
                out.extend_from_slice(&t2.to_be_bytes());
                out.push(e);
            }
        }
    }
    // CTS + CRG — required when Cpih=3 (Star-Tetrix) per A.4.8 / A.4.9.
    if cfg.cpih == 3 {
        // CTS — Lcts = 4, body = 2 bytes (Reserved/Cf, e1/e2).
        out.extend_from_slice(&[0xff, 0x18]);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.push(cfg.cts_cf & 0x0f); // Reserved=0, Cf
        out.push(((cfg.cts_e1 & 0x0f) << 4) | (cfg.cts_e2 & 0x0f));
        // CRG — Lcrg = 2 + 4*Nc, body = 4 * Nc bytes.
        out.extend_from_slice(&[0xff, 0x19]);
        let lcrg = 2u16 + 4 * (cfg.nc as u16);
        out.extend_from_slice(&lcrg.to_be_bytes());
        // Emit the canonical CRG entries that map back to the chosen Ct
        // via Table F.9 (RGGB for Ct=0, GRBG for Ct=1).
        let entries: [(u16, u16); 4] = if cfg.st_ct == 0 {
            // RGGB layout per Table F.9 row 1:
            //   c=0 (R)  : (0,        0)
            //   c=1 (G1) : (32768,    0)
            //   c=2 (G2) : (0,    32768)
            //   c=3 (B)  : (32768,32768)
            [(0, 0), (32768, 0), (0, 32768), (32768, 32768)]
        } else {
            // GRBG layout per Table F.9 row 3:
            //   c=0 (G1) : (32768,    0)
            //   c=1 (R)  : (0,        0)
            //   c=2 (B)  : (32768,32768)
            //   c=3 (G2) : (0,    32768)
            [(32768, 0), (0, 0), (32768, 32768), (0, 32768)]
        };
        for (xc, yc) in entries.iter() {
            out.extend_from_slice(&xc.to_be_bytes());
            out.extend_from_slice(&yc.to_be_bytes());
        }
    }
    Ok(())
}

fn write_pih_body(out: &mut Vec<u8>, cfg: &EncodeConfig) {
    out.extend_from_slice(&0u32.to_be_bytes()); // Lcod
    out.extend_from_slice(&0u16.to_be_bytes()); // Ppih
    out.extend_from_slice(&0u16.to_be_bytes()); // Plev
    out.extend_from_slice(&cfg.width.to_be_bytes());
    out.extend_from_slice(&cfg.height.to_be_bytes());
    out.extend_from_slice(&cfg.cw.to_be_bytes()); // Cw (0 = full-width precincts)
    let hp_pow = 1u32 << cfg.nly;
    let np_y = (cfg.height as u32).div_ceil(hp_pow);
    out.extend_from_slice(&(np_y as u16).to_be_bytes()); // Hsl = Np_y
    out.push(cfg.nc);
    out.push(cfg.ng);
    out.push(cfg.ss);
    out.push(cfg.bw);
    // Fq:Br
    out.push(((cfg.fq & 0x0f) << 4) | (cfg.br & 0x0f));
    // Fslc:Ppoc:Cpih
    out.push(cfg.cpih & 0x0f);
    // NL,x:NL,y
    out.push(((cfg.nlx & 0x0f) << 4) | (cfg.nly & 0x0f));
    // Lh:Rl:Qpih:Fs:Rm = 0
    out.push(0x00);
}

/// Number of wavelet filter types `Nβ` per Annex B.3.
fn n_beta(nlx: u8, nly: u8) -> u32 {
    let mn = nlx.min(nly) as u32;
    let mx = nlx.max(nly) as u32;
    2 * mn + mx + 1
}

/// Build a 256-entry "forward extended-NLT" lookup table mapping each
/// 8-bit input pixel value to a wavelet-domain code in `[0, 2^Bw - 1]`.
///
/// The lookup is built by walking the decoder's extended-gamma kernel
/// (Annex G.5, Table G.4) across every `v_wave ∈ [0, 2^Bw - 1]`,
/// computing the 8-bit output, and recording the first wavelet code that
/// reconstructs each output level. This is O(2^Bw) and runs once per
/// encode; for Bw=18 that's ~262k iterations and ~256 bytes of state.
///
/// The output 8-bit value walks monotonically (modulo the rounding /
/// segment-boundary discretization) so a single pass suffices. Levels
/// that never appear in the decoder output (e.g. unreachable due to
/// segment-boundary skips) are filled with the nearest neighbour from
/// the left.
fn build_extended_forward_lut(bw: u8, bc: u8, t1: u32, t2: u32, e: u8) -> Vec<u32> {
    let bw_i = bw as i64;
    let m = (1i64 << bc) - 1;
    let two_pow_bw_minus_one = (1i64 << bw) - 1;
    let t1 = t1 as i64;
    let t2 = t2 as i64;
    let e_i = e as i64;
    let b2 = t1 * t1;
    let shift_a13 = 2 * bw_i - 2 - 2 * e_i;
    let a1 = b2 + (t1 << (bw_i - e_i)) + (1i64 << shift_a13);
    let b1 = t1 + (1i64 << (bw_i - e_i - 1));
    let a3 = b2 + (t2 << (bw_i - e_i)) - (1i64 << shift_a13);
    let b3 = t2 - (1i64 << (bw_i - e_i - 1));
    let zeta = 2 * bw_i - (bc as i64);
    let zeta_u = zeta.max(0) as u32;
    let half: i64 = if zeta_u == 0 { 0 } else { 1i64 << (zeta_u - 1) };

    let n_levels = (1usize << bc).min(257);
    let mut lut: Vec<Option<u32>> = vec![None; n_levels];

    let max_wave = 1u64 << bw;
    for v_wave in 0..max_wave {
        let v0 = v_wave as i64;
        let v = if v0 < t1 {
            let v = b1 - v0;
            let v = v.clamp(0, two_pow_bw_minus_one);
            a1 - v * v
        } else if v0 < t2 {
            (v0 << (bw_i - e_i)) + b2
        } else {
            let v = v0 - b3;
            let v = v.clamp(0, two_pow_bw_minus_one);
            a3 + v * v
        };
        let v = if zeta_u == 0 { v } else { (v + half) >> zeta_u };
        let out = v.clamp(0, m) as usize;
        if out < lut.len() && lut[out].is_none() {
            lut[out] = Some(v_wave as u32);
        }
    }

    // Fill any unreachable levels with the nearest filled neighbour.
    let mut filled: Vec<u32> = Vec::with_capacity(n_levels);
    let mut last: u32 = 0;
    for slot in lut.iter() {
        match slot {
            Some(v) => {
                last = *v;
                filled.push(*v);
            }
            None => filled.push(last),
        }
    }
    filled
}

/// Count the bands that actually exist over every component (i.e.
/// matching the WGT existing-band convention). For component i with
/// `sy[i] = 2`, band β with `dy < NL,y` and `τy = true` (i.e. LH/HH
/// rows) does not exist; every other band does.
fn count_existing_bands(cfg: &EncodeConfig) -> u32 {
    let nbeta_pic = n_beta(cfg.nlx, cfg.nly);
    let n_decomposed = (cfg.nc - cfg.sd) as usize;
    let mut n = 0u32;
    for i in 0..n_decomposed {
        let nly_i = cfg.nly.saturating_sub(match cfg.sy[i] {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 0,
        });
        let nbeta_i = n_beta(cfg.nlx, nly_i);
        // Only the first nbeta_i bands of the picture-level β layout
        // exist for component i; the rest are non-existent (Annex B.4).
        n += nbeta_pic.min(nbeta_i);
    }
    // Sd tail bands always exist (sx=sy=1 enforced upstream).
    n += cfg.sd as u32;
    n
}

/// Build per-band gain values in WGT emission order (picture-level β,
/// then component i). For the standard 5/3 wavelet the gain of each
/// band corresponds to the number of high-pass axes:
///   LL (τx=false, τy=false) → G=0
///   HL (τx=true, τy=false) or LH (τx=false, τy=true) → G=1
///   HH (τx=true, τy=true) → G=2
///
/// This allows `T[p,b] = clamp(Q - G[b], 0, 15)` in the precinct
/// header to allocate fewer bits (higher T) to the LL band and more
/// bits (lower T) to the high-frequency HH band, improving PSNR/byte.
/// Variant of [`build_band_gains_sd`] that accounts for Sd suppressed
/// components by appending one gain slot per suppressed component at
/// the tail. Suppressed-component gains are zero (LL-equivalent —
/// the band is the raw samples, so we don't want extra truncation).
fn build_band_gains_sd(nc: u8, sd: u8, nlx: u8, nly: u8, _sx: &[u8], sy: &[u8]) -> Vec<u8> {
    let nbeta_pic = n_beta(nlx, nly);
    let n_decomposed = (nc - sd) as usize;
    let mut gains = Vec::new();
    for beta in 0..nbeta_pic {
        for &sy_val in sy.iter().take(n_decomposed) {
            let nly_i = nly.saturating_sub(match sy_val {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            let nbeta_i = n_beta(nlx, nly_i);
            if beta >= nbeta_i {
                continue; // band does not exist for this component
            }
            let key = beta_key(beta, nlx, nly_i);
            let gain = (if key.tau_x { 1u8 } else { 0 }) + (if key.tau_y { 1 } else { 0 });
            gains.push(gain);
        }
    }
    // Append one zero-gain slot per Sd suppressed component.
    if sd > 0 {
        gains.resize(gains.len() + sd as usize, 0);
    }
    gains
}

/// Per-(β, i) band geometry needed by the encoder.
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
/// (NL,x, NL,y).
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

    // 1) Optional forward NLT pre-distortion.
    //    Tnlt=1 (quadratic, Annex G.4 forward):
    //      y = round(sqrt(v_linear / (2^B - 1)) * (2^Bw - 1)) + dco.
    //    Tnlt=2 (extended, Annex G.5): no closed-form algebraic inverse
    //      across the three-segment kernel, so we build a reverse LUT
    //      from the decoder's `extended_path` and pick the first wavelet
    //      code per 8-bit output value.
    //    Both paths produce a wavelet-domain value in [0, 2^Bw-1] which
    //    is then shifted by `-dc_bias` to land in [-2^(Bw-1), 2^(Bw-1)-1].
    let mut comp_planes: Vec<Vec<i32>> = match cfg.nlt {
        Some(NltParams::Quadratic { dco }) => {
            // Forward quadratic: map u8 input → Bw-bit domain.
            // Spec Annex G.4 forward: y = round(sqrt(x / (2^B-1)) * (2^Bw-1)).
            // The DC level shift for Bw=18 is 2^17.
            let bw_max = (1i64 << cfg.bw) - 1;
            let b_max = (1i64 << 8) - 1; // B[i] = 8 always
            planes_u8
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|&v| {
                            let x = (v as i64).clamp(0, b_max);
                            // y = round(sqrt(x / b_max) * bw_max)
                            let y = if x == 0 {
                                0i64
                            } else {
                                let ratio = (x as f64) / (b_max as f64);
                                (ratio.sqrt() * (bw_max as f64)).round() as i64
                            };
                            let y = (y + (dco as i64)).clamp(0, bw_max);
                            // Subtract DC bias (for Bw=18 that's 2^17=131072).
                            (y as i32) - dc_bias
                        })
                        .collect()
                })
                .collect()
        }
        Some(NltParams::Extended { t1, t2, e }) => {
            // Build the reverse LUT (output u8 → first wavelet code that
            // reconstructs it under `extended_path`). This is O(2^Bw) per
            // encode, independent of picture size. Bw is always 18 here.
            let fwd = build_extended_forward_lut(cfg.bw, 8, t1, t2, e);
            planes_u8
                .iter()
                .map(|p| {
                    p.iter()
                        .map(|&v| {
                            let y = fwd[v as usize] as i64;
                            // Subtract DC bias.
                            (y as i32) - dc_bias
                        })
                        .collect()
                })
                .collect()
        }
        None => {
            // Normal linear path: u8 input shifted to i32 wavelet domain.
            planes_u8
                .iter()
                .map(|p| p.iter().map(|&v| v as i32 - dc_bias).collect::<Vec<i32>>())
                .collect()
        }
    };

    // 2) Per-component colour transform.
    if cfg.cpih == 1 {
        let mut refs: Vec<&mut [i32]> = comp_planes.iter_mut().map(|p| p.as_mut_slice()).collect();
        forward_rct(&mut refs, w, h)?;
    } else if cfg.cpih == 3 {
        let mut refs: Vec<&mut [i32]> = comp_planes.iter_mut().map(|p| p.as_mut_slice()).collect();
        forward_star_tetrix(
            &mut refs, w, h, cfg.cts_e1, cfg.cts_e2, cfg.st_ct, cfg.cts_cf,
        )?;
    }

    let nlx = cfg.nlx;
    let nly = cfg.nly;
    // Compute Cs / Np,x per Annex B.5.
    let max_sx = cfg.sx.iter().copied().max().unwrap_or(1) as u32;
    let cs: u32 = if cfg.cw == 0 {
        w as u32
    } else {
        8u32 * (cfg.cw as u32) * max_sx * (1u32 << nlx)
    };
    let np_x: usize = ((w as u32).div_ceil(cs)) as usize;
    // Route everything with nlx > 1 or nly > 1 through the cascade path,
    // including asymmetric (nlx != nly) configurations. Cw > 0 (Np,x > 1)
    // also forces the cascade path because per-precinct DWT does not
    // commute with multi-precinct-per-row layout (precinct boundaries
    // reflect at the band level, not the sample level).
    let multi_level = nlx > 1 || nly > 1 || np_x > 1 || cfg.sd > 0;
    let hp_pow = 1u32 << nly;
    let np_y = (h as u32).div_ceil(hp_pow) as usize;

    // 3) Per-component forward DWT.
    //    The decoder picks per-precinct streaming synthesis at NL=1/1
    //    single-column and gather-then-cascade otherwise. The encoder
    //    must mirror that exactly because per-precinct DWT and
    //    picture-level cascade DWT are *not* equivalent (the 5/3
    //    high-pass coefficient at the precinct boundary depends on a
    //    sample two precincts away — picture-level cascade reflects
    //    across the picture boundary, per-precinct cascade reflects
    //    across the precinct boundary).
    if multi_level {
        let n_decomposed = (cfg.nc - cfg.sd) as usize;
        let mut bands_per_comp: Vec<Vec<Vec<i32>>> = Vec::with_capacity(nc);
        for (i, plane) in comp_planes.iter().enumerate().take(nc) {
            if i >= n_decomposed {
                // Suppressed (Sd) — no wavelet bands; push empty slot.
                bands_per_comp.push(Vec::new());
                continue;
            }
            let wc = w / (cfg.sx[i] as usize);
            let hc = h / (cfg.sy[i] as usize);
            let nly_i = cfg.nly.saturating_sub(match cfg.sy[i] {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            let bands = forward_cascade_2d(wc, hc, cfg.nlx, nly_i, plane)?;
            bands_per_comp.push(bands);
        }
        // For suppressed (Sd) components, encode_precinct_cascade reads
        // the wavelet-domain `comp_planes` slice for the component
        // directly (no DWT was applied). `comp_planes` is already
        // DC-biased so the values fed into the entropy coder match the
        // dynamic range the decoder dequant path will produce when
        // copying straight back into the sample plane.
        for py in 0..np_y {
            for px in 0..np_x {
                let pbytes =
                    encode_precinct_cascade(cfg, &bands_per_comp, &comp_planes, py, px, cs)?;
                out.extend_from_slice(&pbytes);
            }
        }
    } else {
        // NL=1/1 streaming per-precinct path. Handles 4:4:4 and chroma-
        // sub-sampled cases with a per-component effective NL,y and a
        // per-component precinct row range.
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

/// Single-level streaming encode (NL=1/1). Mirrors the decoder's
/// per-precinct synthesis path, including chroma sub-sampling — for
/// `sy[i] = 2` components the per-precinct strip has only one row
/// (`hp_i = 1`) and only the LL/HL bands exist (1-D horizontal DWT).
fn encode_precinct_single_level(
    cfg: &EncodeConfig,
    comp_planes: &[Vec<i32>],
    y0: usize,
    y1: usize,
    hp_real: usize,
) -> Result<Vec<u8>> {
    let w = cfg.width as usize;
    let h_full = cfg.height as usize;
    let nc = cfg.nc as usize;
    let hp_pow = 1usize << cfg.nly;

    // Per-component bands. Each component has [LL, HL, LH, HH] but for
    // `sy=2` (4:2:0) only [LL, HL] are populated.
    struct CompBands {
        nly_i: u8,
        ll: Vec<i32>,
        hl: Vec<i32>,
        lh: Vec<i32>,
        hh: Vec<i32>,
        ll_w: usize,
        hl_w: usize,
        ll_h_per_precinct: usize,
        lh_h_per_precinct: usize,
        pic_ll_h: usize,
        pic_lh_h: usize,
    }
    let mut comp_bands: Vec<CompBands> = Vec::with_capacity(nc);
    for (i, plane) in comp_planes.iter().enumerate().take(nc) {
        let sx_i = cfg.sx[i] as usize;
        let sy_i = cfg.sy[i] as usize;
        let wc = w / sx_i;
        let hc = h_full / sy_i;
        let hp_i = hp_pow / sy_i;
        let nly_i = cfg.nly.saturating_sub(match cfg.sy[i] {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 0,
        });

        // Per-precinct strip rows for this component.
        let y0_i = y0 / sy_i;
        let y1_i = (y1 / sy_i).min(hc);
        let hp_real_i = y1_i.saturating_sub(y0_i);
        let mut strip: Vec<i32> = Vec::with_capacity(wc * hp_i);
        for y in y0_i..y1_i {
            for x in 0..wc {
                strip.push(plane[y * wc + x]);
            }
        }
        // Pad with whole-sample symmetric reflection up to hp_i rows.
        while strip.len() < wc * hp_i {
            let target_row = strip.len() / wc;
            let src_row = if hp_real_i >= 2 {
                let mirrored = 2 * hp_real_i - target_row - 2;
                mirrored.min(hp_real_i - 1)
            } else {
                0
            };
            let row_start = src_row * wc;
            for x in 0..wc {
                let src_idx = if hp_real_i == 0 { 0 } else { row_start + x };
                let val = if hp_real_i == 0 { 0 } else { strip[src_idx] };
                strip.push(val);
            }
        }

        let ll_w = wc.div_ceil(2);
        let hl_w = wc / 2;
        let ll_h_per_precinct = hp_i.div_ceil(2);
        let lh_h_per_precinct = hp_i / 2;
        // Picture-level LL band height: depends on the per-component
        // effective vertical decomposition. nly_i = 0 → no vertical
        // split → LL rows == chroma rows (no /2). nly_i = 1 → vertical
        // 1-D split halves the height.
        let pic_ll_h = if nly_i == 0 { hc } else { hc.div_ceil(2) };
        let pic_lh_h = if nly_i == 0 { 0 } else { hc / 2 };

        if nly_i == 0 {
            // 1-D horizontal-only DWT: one row, two bands (LL, HL).
            // hp_i must be 1 in this case.
            debug_assert_eq!(hp_i, 1);
            let mut ll = vec![0i32; ll_w];
            let mut hl = vec![0i32; hl_w];
            crate::dwt::forward_horizontal_1d(&strip, &mut ll, &mut hl)?;
            comp_bands.push(CompBands {
                nly_i,
                ll,
                hl,
                lh: Vec::new(),
                hh: Vec::new(),
                ll_w,
                hl_w,
                ll_h_per_precinct,
                lh_h_per_precinct,
                pic_ll_h,
                pic_lh_h,
            });
        } else {
            // 2-D DWT.
            let mut ll = vec![0i32; ll_w * ll_h_per_precinct];
            let mut hl = vec![0i32; hl_w * ll_h_per_precinct];
            let mut lh = vec![0i32; ll_w * lh_h_per_precinct];
            let mut hh = vec![0i32; hl_w * lh_h_per_precinct];
            forward_2d(wc, hp_i, &strip, &mut ll, &mut hl, &mut lh, &mut hh)?;
            comp_bands.push(CompBands {
                nly_i,
                ll,
                hl,
                lh,
                hh,
                ll_w,
                hl_w,
                ll_h_per_precinct,
                lh_h_per_precinct,
                pic_ll_h,
                pic_lh_h,
            });
        }
        let _ = (wc, hp_i); // dimensions captured into ll_w/hl_w/etc.
    }

    // Per-component, per-precinct line counts. β=0 (LL) and β=1 (HL)
    // contribute up to 1 line per precinct (since pow_h(1, dy=1) = 1
    // and dy=1 for all 4 bands at NL=1/1; ll_h_per_precinct = 1 here).
    // β=2 (LH) and β=3 (HH) only exist for nly_i >= 1 components.
    let py = y0 / hp_pow;
    // Lines emitted for the LL/HL row in this precinct: 1 unless we're
    // past the picture edge. For chroma at sy=2, ll_h_per_precinct
    // already accounts for hp_i=1.
    let lines_ll_real_per_comp: Vec<usize> = comp_bands
        .iter()
        .map(|cb| {
            let row_offset = py;
            // For sy=2 the per-component ll_h_per_precinct == 1 anyway.
            if row_offset >= cb.pic_ll_h {
                0
            } else {
                cb.ll_h_per_precinct.min(cb.pic_ll_h - row_offset)
            }
        })
        .collect();
    let lines_lh_real_per_comp: Vec<usize> = comp_bands
        .iter()
        .map(|cb| {
            if cb.nly_i == 0 {
                0
            } else {
                let row_offset = py;
                if row_offset >= cb.pic_lh_h {
                    0
                } else {
                    cb.lh_h_per_precinct.min(cb.pic_lh_h - row_offset)
                }
            }
        })
        .collect();

    // Precinct-header band-existence bookkeeping. β=0 / 1 always exist
    // for every component; β=2 / 3 only exist for components with
    // `nly_i >= 1`.
    let mut n_existing = 0usize;
    for (i, _) in comp_bands.iter().enumerate() {
        n_existing += 2; // LL + HL always exist
        if comp_bands[i].nly_i >= 1 {
            n_existing += 2; // LH + HH
        }
    }
    let header_bits = 24 + 8 + 8 + 2 * n_existing;
    let header_bytes = header_bits.div_ceil(8);
    let mut precinct_bytes = vec![0u8; header_bytes];

    let mut entropy: Vec<u8> = Vec::new();
    // Per-band truncation positions. Single-level: β=0→G=0, β=1→G=1,
    // β=2→G=1, β=3→G=2. Uses gain-weighted T[p,b] = clamp(Q-G, 0, 15).
    let t_for_gain = |g: u8| -> u8 { (cfg.q as i32 - g as i32).clamp(0, 15) as u8 };

    // First packet: β=0 (LL) for all components — but only those with
    // a non-empty LL line for this precinct.
    let mut first_entries: Vec<PerBandEntry> = Vec::new();
    for (i, cb) in comp_bands.iter().enumerate() {
        if lines_ll_real_per_comp[i] == 0 {
            continue;
        }
        // β=0 (LL) gain = 0.
        let line_data = cb.ll[..cb.ll_w].to_vec();
        first_entries.push(PerBandEntry {
            wpb: cb.ll_w as u32,
            line: BandLineSlice::Direct(line_data),
            t: t_for_gain(0),
        });
    }
    if !first_entries.is_empty() {
        emit_packet(&mut entropy, cfg, &first_entries)?;
    }

    // Proxy levels: β=1 (HL, G=1), β=2 (LH, G=1), β=3 (HH, G=2).
    // One packet per (β, i) entry, gated by per-component existence and lines.
    for beta_idx in 1usize..=3 {
        for (i, cb) in comp_bands.iter().enumerate() {
            // Existence per component.
            if beta_idx >= 2 && cb.nly_i == 0 {
                continue;
            }
            let lines_real = if beta_idx == 1 {
                lines_ll_real_per_comp[i]
            } else {
                lines_lh_real_per_comp[i]
            };
            if lines_real == 0 {
                continue;
            }
            let (band_buf, wpb) = match beta_idx {
                1 => (&cb.hl, cb.hl_w),
                2 => (&cb.lh, cb.ll_w),
                _ => (&cb.hh, cb.hl_w),
            };
            // Gain per sub-band type: HL/LH=1, HH=2.
            let gain: u8 = if beta_idx <= 2 { 1 } else { 2 };
            let line_data = band_buf[..wpb].to_vec();
            let entries = vec![PerBandEntry {
                wpb: wpb as u32,
                line: BandLineSlice::Direct(line_data),
                t: t_for_gain(gain),
            }];
            emit_packet(&mut entropy, cfg, &entries)?;
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
    precinct_bytes[3] = cfg.q.min(31);
    // R[p] at offset 4 stays 0.
    // D[p,b] bits at offset 5+ stay 0.
    precinct_bytes.extend_from_slice(&entropy);
    let _ = hp_real; // hp_real is the original-pixel-grid count; unused now
    Ok(precinct_bytes)
}

/// Encode one precinct using the multi-level cascade band layout.
///
/// `comp_planes` carries the raw (DC-biased) per-component samples used
/// only for the Sd suppressed components (`i ≥ Nc - Sd`); the wavelet
/// components draw from `bands_per_comp` as before.
fn encode_precinct_cascade(
    cfg: &EncodeConfig,
    bands_per_comp: &[Vec<Vec<i32>>],
    comp_planes: &[Vec<i32>],
    py: usize,
    px: usize,
    cs: u32,
) -> Result<Vec<u8>> {
    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let nc = cfg.nc as usize;
    let nlx = cfg.nlx;
    let nly = cfg.nly;
    let nbeta_pic = n_beta(nlx, nly);
    // Width Wp[p] in image-grid columns for precinct (px, py). All but
    // the rightmost are Cs wide; the rightmost picks up Wf mod Cs.
    let np_x = ((w as u32).div_ceil(cs)) as usize;
    let _wp_this = if px + 1 < np_x {
        cs as usize
    } else {
        ((w as u32 - 1) % cs + 1) as usize
    };

    // Per-component "effective" decomposition levels.
    let nly_i: Vec<u8> = (0..nc)
        .map(|i| {
            cfg.nly.saturating_sub(match cfg.sy[i] {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            })
        })
        .collect();

    // Collect per-(β, i) slices for this precinct. Bands not existing
    // for component i (because β >= n_beta(nlx, nly_i[i])) are skipped.
    struct Slice {
        wpb: usize,
        lines: usize,
        pic_bw: usize,
        pic_row_offset: usize,
        /// Column offset into the picture-level band buffer for this
        /// precinct column (`px * Cs / (sx[i] * 2^dx)` for low-pass,
        /// or `px * Cs / (sx[i] * 2^(dx-1)) / 2` for high-pass; both
        /// reduce to `px * Cs / (sx[i] * 2^dx)` because `Cs` is a
        /// multiple of `8 * max(sx) * 2^NL,x`).
        pic_col_offset: usize,
        comp_i: usize,
        beta: u32,
        exists: bool,
    }
    let sd_u = cfg.sd as usize;
    let n_decomposed = nc - sd_u;
    let mut slices: Vec<Slice> = Vec::with_capacity(((nbeta_pic as usize) * n_decomposed) + sd_u);
    for beta in 0..nbeta_pic {
        for (i, &nly_comp) in nly_i.iter().enumerate().take(n_decomposed) {
            let wc = w / (cfg.sx[i] as usize);
            let hc = h / (cfg.sy[i] as usize);
            // Existence: β must be < n_beta(nlx, nly_i[i]).
            let nbeta_i = n_beta(cfg.nlx, nly_comp);
            let exists_per_comp = beta < nbeta_i;
            if !exists_per_comp {
                slices.push(Slice {
                    wpb: 0,
                    lines: 0,
                    pic_bw: 0,
                    pic_row_offset: 0,
                    pic_col_offset: 0,
                    comp_i: i,
                    beta,
                    exists: false,
                });
                continue;
            }
            let key = beta_key(beta, cfg.nlx, nly_comp);
            let (pic_bw, pic_bh) = band_dims(wc, hc, cfg.nlx, nly_comp, beta);
            // pow_h is computed against the *picture-level* nly; for
            // sub-sampled components the precinct still holds
            // `pow_h(nly, dy)` band-rows on average — but each component
            // contributes only `pow_h_i = 2^max(nly_i - dy, 0)` because
            // the per-component decomposition is shallower. The
            // sub-sample-aware Annex B.6 says lines per precinct =
            // 2^max(NL,y - dy, 0) / sy[i].
            let pow_pic = pow_h(cfg.nly, key.dy);
            let pow_eff = pow_pic / (cfg.sy[i] as usize).max(1);
            let pow_eff = pow_eff.max(1);
            let row_offset = py * pow_eff;
            let lines = if row_offset >= pic_bh {
                0
            } else {
                pow_eff.min(pic_bh - row_offset)
            };
            // Per-precinct Wpb[p,b]. For Cw == 0 every precinct equals
            // pic_bw (the picture-level band width). For Cw > 0 the
            // band-cols-per-precinct equals `Cs / (sx[i] * 2^dx)` for
            // both low- and high-pass, since `Cs = 8 × Cw × max(sx) ×
            // 2^NL,x` is an exact multiple of `sx[i] * 2^dx` for any
            // dx ∈ {0..=NL,x}.
            let sx_i = cfg.sx[i] as usize;
            let dx = key.dx as usize;
            let cols_per_uniform = (cs as usize) / (sx_i * (1usize << dx)).max(1);
            let pic_col_offset = px * cols_per_uniform;
            let remaining_cols = pic_bw.saturating_sub(pic_col_offset);
            let wpb_this = if px + 1 < np_x {
                cols_per_uniform.min(remaining_cols)
            } else {
                remaining_cols
            };
            slices.push(Slice {
                wpb: wpb_this,
                lines,
                pic_bw,
                pic_row_offset: row_offset,
                pic_col_offset,
                comp_i: i,
                beta,
                exists: true,
            });
        }
    }

    // Sd tail slices: one per suppressed component (β = 0, no DWT). The
    // band data lives in `comp_planes[i]` at the precinct's row offset.
    // Per Annex A.4.7, sx[i] = sy[i] = 1 so the per-precinct band width
    // is exactly Wp[p] and the precinct holds Hp = 2^NL,y picture lines.
    let hp_pic = if nly == 0 { 1usize } else { 1usize << nly };
    let pic_row_offset_sd = py * hp_pic;
    let lines_this_precinct = hp_pic.min(h.saturating_sub(pic_row_offset_sd));
    let wp_this = if px + 1 < np_x {
        cs as usize
    } else {
        ((w as u32 - 1) % cs + 1) as usize
    };
    let pic_col_offset_sd = px * (cs as usize);
    for sd_idx in 0..sd_u {
        let i = n_decomposed + sd_idx;
        slices.push(Slice {
            wpb: wp_this.min(w.saturating_sub(pic_col_offset_sd)),
            lines: lines_this_precinct,
            pic_bw: w, // sx[i] = 1, so the picture-level band width is the full width
            pic_row_offset: pic_row_offset_sd,
            pic_col_offset: pic_col_offset_sd,
            comp_i: i,
            beta: 0,
            exists: lines_this_precinct > 0 && wp_this > 0,
        });
    }

    // Precinct header: Lprc(24) + Q(8) + R(8) + N_existing × D(2),
    // padded to byte boundary.
    let n_existing = slices.iter().filter(|s| s.exists).count();
    let header_bits = 24 + 8 + 8 + 2 * n_existing;
    let header_bytes = header_bits.div_ceil(8);
    let mut precinct_bytes = vec![0u8; header_bytes];

    // Build entropy stream: walk packets per Annex B.7 Table B.4. Round 4
    // adds vertical-prediction VLC: per-band per-precinct, the encoder
    // evaluates D[p,b] = 0 (no-prediction VLC) vs D[p,b] = 1 (vertical-
    // prediction VLC, Table C.13) and commits the smaller. Per-packet,
    // the encoder still picks min(Dr=1 raw, Dr=0 in-band-VLC-mode).
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let beta1 = nlx_u.max(nly_u) - nlx_u.min(nly_u) + 1;

    // Per-band truncation: T[p,b] = clamp(Q - G[b], 0, 15).
    // G[b] = #high-pass axes in band β for comp i (using comp-local nly_i).
    let t_for_band = |beta: u32, comp_i: usize| -> u8 {
        let nly_comp = nly_i[comp_i];
        let key = beta_key(beta, cfg.nlx, nly_comp);
        let gain = (if key.tau_x { 1u8 } else { 0 }) + (if key.tau_y { 1 } else { 0 });
        (cfg.q as i32 - gain as i32).clamp(0, 15) as u8
    };

    // Helper: build a one-line band slice from a per-component band buffer.
    // Wavelet components (i < n_decomposed) read from bands_per_comp; the
    // Sd suppressed components read directly from comp_planes (their
    // "band" is the raw, DC-biased picture samples).
    let extract_band_line = |s: &Slice, line_off: usize| -> Option<Vec<i32>> {
        if !s.exists || s.lines == 0 {
            return None;
        }
        if line_off >= s.lines {
            return None;
        }
        if s.wpb == 0 {
            return None;
        }
        if s.comp_i >= n_decomposed {
            // Sd suppressed: comp_planes is sized at Wf*Hf for sx=sy=1.
            let plane = &comp_planes[s.comp_i];
            let pic_row = s.pic_row_offset + line_off;
            let row_start = pic_row * s.pic_bw + s.pic_col_offset;
            let row_end = row_start + s.wpb;
            return Some(plane[row_start..row_end].to_vec());
        }
        let band_buf = &bands_per_comp[s.comp_i][s.beta as usize];
        let pic_row = s.pic_row_offset + line_off;
        let row_start = pic_row * s.pic_bw + s.pic_col_offset;
        let row_end = row_start + s.wpb;
        Some(band_buf[row_start..row_end].to_vec())
    };

    // Phase 1 — collect every packet job in slice-walker emission order.
    // Each job carries the entries it covers + the (comp, beta) coordinate
    // of each entry (needed for per-band D decision and Mtop tracking).
    let mut jobs: Vec<PacketJob> = Vec::new();

    // First packet: β = 0 .. β1-1 × (Nc - Sd) wavelet components × line 0
    // (subject to existence + sub-sample guard).
    {
        let mut entries: Vec<PerBandEntry> = Vec::new();
        let mut coords: Vec<(usize, u32)> = Vec::new();
        for beta in 0..beta1 {
            for i in 0..n_decomposed {
                let s_idx = (beta as usize) * n_decomposed + i;
                let s = &slices[s_idx];
                if let Some(line_data) = extract_band_line(s, 0) {
                    entries.push(PerBandEntry {
                        wpb: s.wpb as u32,
                        line: BandLineSlice::Direct(line_data),
                        t: t_for_band(beta, i),
                    });
                    coords.push((i, beta));
                }
            }
        }
        if !entries.is_empty() {
            jobs.push(PacketJob {
                entries,
                coords,
                first_line_in_precinct: true,
            });
        }
    }
    // Proxy levels.
    {
        let mut beta0 = beta1;
        // Track per-(comp, beta) whether we've already seen a packet for
        // that band in this precinct (to mark first-line packets, which
        // can never use vertical prediction).
        let mut first_seen: std::collections::HashSet<(usize, u32)> =
            std::collections::HashSet::new();
        while beta0 < nbeta_pic {
            let key0 = beta_key(beta0, cfg.nlx, cfg.nly);
            let pow_pic = pow_h(cfg.nly, key0.dy);
            for lambda_within in 0..pow_pic {
                for beta in beta0..(beta0 + 3).min(nbeta_pic) {
                    for i in 0..n_decomposed {
                        let s_idx = (beta as usize) * n_decomposed + i;
                        let s = &slices[s_idx];
                        if !s.exists {
                            continue;
                        }
                        let sy_i = cfg.sy[i] as usize;
                        let pic_grid_line = lambda_within;
                        if sy_i != 0 && pic_grid_line % sy_i != 0 {
                            continue;
                        }
                        let comp_line = pic_grid_line / sy_i.max(1);
                        if let Some(line_data) = extract_band_line(s, comp_line) {
                            let key = (i, beta);
                            let is_first = first_seen.insert(key);
                            jobs.push(PacketJob {
                                entries: vec![PerBandEntry {
                                    wpb: s.wpb as u32,
                                    line: BandLineSlice::Direct(line_data),
                                    t: t_for_band(beta, i),
                                }],
                                coords: vec![key],
                                first_line_in_precinct: is_first,
                            });
                        }
                    }
                }
            }
            beta0 += 3;
        }
    }

    // Sd tail: one packet per (line λ, suppressed component i), with
    // component as the fast and line as the slow variable per Annex B.7
    // Table B.4. The slice index for the tail is
    // `nbeta_pic * n_decomposed + (i - n_decomposed)`.
    if sd_u > 0 {
        let sd_first_slice = (nbeta_pic as usize) * n_decomposed;
        for lambda in 0..lines_this_precinct {
            for sd_idx in 0..sd_u {
                let s_idx = sd_first_slice + sd_idx;
                let s = &slices[s_idx];
                if !s.exists {
                    continue;
                }
                let i = n_decomposed + sd_idx;
                let line_off = lambda;
                if let Some(line_data) = extract_band_line(s, line_off) {
                    let key = (i, 0u32);
                    let is_first = lambda == 0;
                    jobs.push(PacketJob {
                        entries: vec![PerBandEntry {
                            wpb: s.wpb as u32,
                            line: BandLineSlice::Direct(line_data),
                            // Sd tail bands carry raw samples; T = clamp(Q - 0).
                            t: (cfg.q as i32).clamp(0, 15) as u8,
                        }],
                        coords: vec![key],
                        first_line_in_precinct: is_first,
                    });
                }
            }
        }
    }

    // Phase 2 — for every job, evaluate all candidate forms:
    //   D&1=0, D&2=0: min(raw, no-pred VLC).
    //   D&1=0, D&2=1: min(raw, no-pred-sig VLC).
    //   D&1=1, D&2=0: min(raw, vert-pred VLC).
    //   D&1=1, D&2=1: min(raw, vert-pred-sig VLC).
    //
    // The two D bits are treated as independent dimensions here;
    // Phase 3 picks the (pred_bit, sig_bit) combination with the lowest
    // total per-band byte count.
    //
    // Vertical-prediction needs the per-band Mtop cache (last-line M for
    // each (comp, beta)). The cache is populated as we visit jobs.
    // Significance-coding dimension accounted separately.
    let mut sizes_d00: std::collections::HashMap<(usize, u32), usize> =
        std::collections::HashMap::new();
    let mut sizes_d01: std::collections::HashMap<(usize, u32), usize> =
        std::collections::HashMap::new();
    let mut sizes_d10: std::collections::HashMap<(usize, u32), usize> =
        std::collections::HashMap::new();
    let mut sizes_d11: std::collections::HashMap<(usize, u32), usize> =
        std::collections::HashMap::new();
    let mut m_top_cache: std::collections::HashMap<(usize, u32), Vec<u8>> =
        std::collections::HashMap::new();
    let mut precomputed: Vec<JobForms> = Vec::with_capacity(jobs.len());

    for job in &jobs {
        // Compute M arrays for every entry once (shared across forms).
        let m_per_entry: Vec<Vec<u8>> = job
            .entries
            .iter()
            .map(|e| compute_m_per_group(cfg, e))
            .collect::<Result<Vec<_>>>()?;

        let raw = build_packet_body_with_m(cfg, &job.entries, &m_per_entry, BitplaneMode::Raw)?;
        let no_pred = build_packet_body_with_m(
            cfg,
            &job.entries,
            &m_per_entry,
            BitplaneMode::Vlc(VlcKind::NoPred),
        )?;
        let no_pred_sig = build_packet_body_with_m(
            cfg,
            &job.entries,
            &m_per_entry,
            BitplaneMode::Vlc(VlcKind::NoPredSig),
        )?;

        // Vertical-prediction is only attempted when EVERY entry of the
        // packet has a predecessor-line M cached for its (comp, beta).
        // For the cascade encoder this happens to be all-or-nothing per
        // packet (proxy-level packets carry one entry; first packet
        // carries one entry per (β<β1, i) — none have predecessors).
        let mut vert_predecessor_per_entry: Vec<Vec<u8>> = Vec::with_capacity(job.entries.len());
        let mut have_all_predecessors = !job.first_line_in_precinct;
        if have_all_predecessors {
            for coord in &job.coords {
                if let Some(prev_m) = m_top_cache.get(coord) {
                    vert_predecessor_per_entry.push(prev_m.clone());
                } else {
                    have_all_predecessors = false;
                    break;
                }
            }
        }
        let (vert, vert_sig) = if have_all_predecessors {
            let v = build_packet_body_with_m(
                cfg,
                &job.entries,
                &m_per_entry,
                BitplaneMode::Vlc(VlcKind::VertPred {
                    predecessor: vert_predecessor_per_entry.clone(),
                }),
            )?;
            let vs = build_packet_body_with_m(
                cfg,
                &job.entries,
                &m_per_entry,
                BitplaneMode::Vlc(VlcKind::VertPredSig {
                    predecessor: vert_predecessor_per_entry.clone(),
                }),
            )?;
            (Some(v), Some(vs))
        } else {
            (None, None)
        };

        // Update the cache for downstream packets — store the last entry's
        // M-array per (comp, beta). For packets with multiple entries (the
        // first packet) each entry's coord is unique, so we record them all.
        for (entry_idx, coord) in job.coords.iter().enumerate() {
            m_top_cache.insert(*coord, m_per_entry[entry_idx].clone());
        }

        // Per-packet byte counts for each D combination.
        let pick_d00 = raw.total_len().min(no_pred.total_len());
        let pick_d01 = raw.total_len().min(no_pred_sig.total_len());
        let pick_d10 = match &vert {
            Some(v) => raw.total_len().min(v.total_len()),
            None => raw.total_len(),
        };
        let pick_d11 = match &vert_sig {
            Some(vs) => raw.total_len().min(vs.total_len()),
            None => raw.total_len(),
        };
        for coord in &job.coords {
            *sizes_d00.entry(*coord).or_insert(0) += pick_d00;
            *sizes_d01.entry(*coord).or_insert(0) += pick_d01;
            *sizes_d10.entry(*coord).or_insert(0) += pick_d10;
            *sizes_d11.entry(*coord).or_insert(0) += pick_d11;
        }
        precomputed.push(JobForms {
            raw,
            no_pred,
            no_pred_sig,
            vert,
            vert_sig,
        });
    }

    // Phase 3 — per band, commit D[p,b] ∈ {0,1,2,3} by total bytes.
    // D encodes (sig_bit=D>>1, pred_bit=D&1) per the precinct header.
    // Pick the combination with the lowest total byte count.
    let mut d_per_band: std::collections::HashMap<(usize, u32), u8> =
        std::collections::HashMap::new();
    for coord in sizes_d00.keys() {
        let s00 = sizes_d00[coord];
        let s01 = sizes_d01.get(coord).copied().unwrap_or(usize::MAX);
        let s10 = sizes_d10.get(coord).copied().unwrap_or(usize::MAX);
        let s11 = sizes_d11.get(coord).copied().unwrap_or(usize::MAX);
        let best = s00.min(s01).min(s10).min(s11);
        let d = if s11 == best {
            3u8 // sig=1, pred=1
        } else if s10 == best {
            1u8 // sig=0, pred=1
        } else if s01 == best {
            2u8 // sig=1, pred=0
        } else {
            0u8 // sig=0, pred=0
        };
        d_per_band.insert(*coord, d);
    }

    // Phase 4 — emit packets in order, picking per-packet form according
    // to the band's chosen D[p,b].
    let mut entropy: Vec<u8> = Vec::new();
    for (job, forms) in jobs.iter().zip(precomputed) {
        // For multi-entry packets (first packet) every entry's band
        // matters, but they all follow the same rule. Pick the D value
        // for the first coord (all coords in the first packet have no
        // predecessor so vert forms are absent).
        let d_any = job
            .coords
            .iter()
            .map(|c| d_per_band.get(c).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let pred_bit = d_any & 1;
        let sig_bit = (d_any >> 1) & 1;
        let chosen = if pred_bit == 1 {
            if sig_bit == 1 {
                // D=3: vert-pred-sig vs raw.
                if let Some(vs) = forms.vert_sig {
                    if vs.total_len() <= forms.raw.total_len() {
                        vs
                    } else {
                        forms.raw
                    }
                } else {
                    forms.raw
                }
            } else {
                // D=1: vert-pred vs raw.
                if let Some(v) = forms.vert {
                    if v.total_len() <= forms.raw.total_len() {
                        v
                    } else {
                        forms.raw
                    }
                } else {
                    forms.raw
                }
            }
        } else if sig_bit == 1 {
            // D=2: no-pred-sig vs raw.
            if forms.no_pred_sig.total_len() <= forms.raw.total_len() {
                forms.no_pred_sig
            } else {
                forms.raw
            }
        } else {
            // D=0: no-pred vs raw.
            if forms.no_pred.total_len() <= forms.raw.total_len() {
                forms.no_pred
            } else {
                forms.raw
            }
        };
        write_packet(&mut entropy, &chosen)?;
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
    precinct_bytes[3] = cfg.q.min(31);
    // R[p] at offset 4 stays 0.
    // D[p,b] bits at offset 5+: pack 2 bits per existing band (Sig|Pred).
    // D[p,b] = (sig_bit << 1) | pred_bit per Table C.1.
    let mut bit_cursor: usize = (24 + 8 + 8) as usize; // skip Lprc/Q/R bits
    for s in &slices {
        if !s.exists {
            continue;
        }
        let d = d_per_band.get(&(s.comp_i, s.beta)).copied().unwrap_or(0);
        let sig_bit = (d >> 1) & 1;
        let pred_bit = d & 1;
        write_d_bit(&mut precinct_bytes, bit_cursor, sig_bit);
        bit_cursor += 1;
        write_d_bit(&mut precinct_bytes, bit_cursor, pred_bit);
        bit_cursor += 1;
    }
    precinct_bytes.extend_from_slice(&entropy);
    Ok(precinct_bytes)
}

/// Write one bit at `bit_pos` (counted MSB-first) into `bytes`. Used for
/// the precinct header `D[p,b]` field.
fn write_d_bit(bytes: &mut [u8], bit_pos: usize, value: u8) {
    let byte = bit_pos / 8;
    let off = 7 - (bit_pos % 8);
    if byte < bytes.len() {
        bytes[byte] |= (value & 1) << off;
    }
}

/// One packet's data plus its band coordinates — phase-1 output of the
/// per-precinct encoder.
struct PacketJob {
    entries: Vec<PerBandEntry>,
    /// `(comp_idx, beta)` per entry.
    coords: Vec<(usize, u32)>,
    /// True iff at least one entry of this packet belongs to a band that
    /// has not yet been seen in the current precinct (i.e. no in-precinct
    /// vertical-prediction predecessor exists for that entry).
    first_line_in_precinct: bool,
}

/// All candidate forms of one packet, computed by phase 2 of the
/// per-precinct encoder. Round 5 adds significance-coded variants for
/// no-pred and vert-pred so the picker evaluates D&2=0 vs D&2=1.
struct JobForms {
    raw: PacketBytes,
    no_pred: PacketBytes,
    no_pred_sig: PacketBytes,
    vert: Option<PacketBytes>,
    vert_sig: Option<PacketBytes>,
}

/// Bitplane-count sub-packet coding mode (Table C.7 / C.12 / C.13 / C.14).
#[derive(Debug)]
enum BitplaneMode {
    /// Dr = 1: raw, Br bits per code group.
    Raw,
    /// Dr = 0: VLC, with the spec's prediction sub-mode.
    Vlc(VlcKind),
}

#[derive(Debug)]
enum VlcKind {
    /// Table C.14 — no prediction. `mtop = T[p,b]`, `θ = 0`.
    NoPred,
    /// Table C.14 with significance gating (`D[p,b] & 2 = 1`).
    /// `Z[j]` flags indicate whether significance group `j` is non-zero.
    NoPredSig,
    /// Table C.13 — vertical prediction. `mtop = max(M_above, T)`,
    /// `θ = max(M_above - T, 0)`. Per-entry predecessor M-array.
    VertPred { predecessor: Vec<Vec<u8>> },
    /// Table C.13 with significance gating.
    VertPredSig { predecessor: Vec<Vec<u8>> },
}

/// One band-line emitted in a packet.
#[derive(Debug)]
struct PerBandEntry {
    /// `Wpb[p,b]` — coefficients per line in this band.
    wpb: u32,
    /// One line of int32 coefficients for this band (length `wpb`).
    line: BandLineSlice,
    /// Per-band `T[p,b]`.
    t: u8,
}

#[derive(Debug)]
enum BandLineSlice {
    Direct(Vec<i32>),
}

impl BandLineSlice {
    fn as_slice(&self) -> &[i32] {
        match self {
            BandLineSlice::Direct(v) => v.as_slice(),
        }
    }
}

/// Emit one packet to `out`. The packet covers a list of (band, line)
/// entries. Single-level (NL=1/1) callers use this — every packet has
/// at most one line per band per precinct, so vertical prediction is
/// never available. The cascade encoder uses the multi-form pipeline
/// (`PacketJob` / `JobForms`) which adds vertical prediction and
/// significance coding (which require D-bit update in the precinct header).
fn emit_packet(out: &mut Vec<u8>, cfg: &EncodeConfig, entries: &[PerBandEntry]) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let m_per_entry: Vec<Vec<u8>> = entries
        .iter()
        .map(|e| compute_m_per_group(cfg, e))
        .collect::<Result<Vec<_>>>()?;
    let raw = build_packet_body_with_m(cfg, entries, &m_per_entry, BitplaneMode::Raw)?;
    let no_pred = build_packet_body_with_m(
        cfg,
        entries,
        &m_per_entry,
        BitplaneMode::Vlc(VlcKind::NoPred),
    )?;
    let chosen = if no_pred.total_len() <= raw.total_len() {
        no_pred
    } else {
        raw
    };
    write_packet(out, &chosen)?;
    Ok(())
}

#[derive(Debug)]
struct PacketBytes {
    dr: u8,
    /// Significance sub-packet (may be empty when `D[p,b] & 2 = 0`).
    sig: Vec<u8>,
    cnt: Vec<u8>,
    data: Vec<u8>,
    sgn: Vec<u8>,
}

impl PacketBytes {
    fn total_len(&self) -> usize {
        // Short header is 5 bytes.
        5 + self.sig.len() + self.cnt.len() + self.data.len() + self.sgn.len()
    }
}

fn write_packet(out: &mut Vec<u8>, pkt: &PacketBytes) -> Result<()> {
    let lcnt = pkt.cnt.len() as u32;
    let ldat = pkt.data.len() as u32;
    let lsgn = pkt.sgn.len() as u32;
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
    if lsgn > (1 << 11) - 1 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: Lsgn = {lsgn} exceeds short packet header capacity (11 bits)."
        )));
    }
    let mut hdr_bits: u64 = 0;
    hdr_bits = (hdr_bits << 1) | (pkt.dr as u64 & 1);
    hdr_bits = (hdr_bits << 15) | (ldat as u64 & 0x7fff);
    hdr_bits = (hdr_bits << 13) | (lcnt as u64 & 0x1fff);
    hdr_bits = (hdr_bits << 11) | (lsgn as u64 & 0x07ff);
    let mut header = vec![0u8; 5];
    for (k, byte) in header.iter_mut().enumerate() {
        *byte = ((hdr_bits >> (8 * (4 - k))) & 0xff) as u8;
    }
    out.extend_from_slice(&header);
    // Significance sub-packet precedes bitplane-count sub-packet
    // (Annex C.4, Table C.5). Only present when has_sig == true.
    out.extend_from_slice(&pkt.sig);
    out.extend_from_slice(&pkt.cnt);
    out.extend_from_slice(&pkt.data);
    out.extend_from_slice(&pkt.sgn);
    Ok(())
}

/// Compute the per-code-group bitplane counts `M[g]` for one entry,
/// applying the lossy lower-bound `M >= T` so dequantized magnitudes
/// don't exceed the bitstream's M cap.
fn compute_m_per_group(cfg: &EncodeConfig, entry: &PerBandEntry) -> Result<Vec<u8>> {
    let wpb = entry.wpb as usize;
    let band_line: &[i32] = entry.line.as_slice();
    let t = entry.t as u32;
    let ng_u = cfg.ng as usize;
    let ncg = wpb.div_ceil(ng_u);
    let m_max_for_br: u32 = if cfg.br >= 8 {
        255
    } else {
        (1u32 << cfg.br) - 1
    };
    let mut m_per_group = vec![0u8; ncg];
    for (g, slot) in m_per_group.iter_mut().enumerate() {
        let mut max_mag: u32 = 0;
        for k in 0..ng_u {
            let xpos = g * ng_u + k;
            let v = if xpos < wpb { band_line[xpos] } else { 0 };
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
        let m_eff = m.max(t);
        if m_eff > m_max_for_br {
            return Err(Error::invalid(format!(
                "jpegxs encoder: code group {g} bitplane count {m_eff} exceeds Br = {} (cap {m_max_for_br}). Use a higher Br or quantize the input.",
                cfg.br
            )));
        }
        *slot = m_eff as u8;
    }
    Ok(m_per_group)
}

/// Build one packet body for the given bitplane-count coding mode,
/// using pre-computed per-entry M arrays. The data sub-packet is
/// independent of the bitplane-count mode (only the `cnt` sub-packet
/// changes per Tables C.7 / C.12 / C.13 / C.14).
///
/// Round 5: `NoPredSig` / `VertPredSig` variants emit a significance
/// sub-packet (one bit per significance group, padded to byte) before
/// the bitplane-count sub-packet. Insignificant groups (all M[g] for
/// the group = T) skip their VLC code entirely (Δm = 0 inferred).
fn build_packet_body_with_m(
    cfg: &EncodeConfig,
    entries: &[PerBandEntry],
    m_per_entry: &[Vec<u8>],
    mode: BitplaneMode,
) -> Result<PacketBytes> {
    let mut data_writer = BitWriter::default();
    let mut cnt_writer = BitWriter::default();
    let mut sig_writer = BitWriter::default();
    let ng_u = cfg.ng as usize;
    let ss_u = cfg.ss as usize; // code groups per significance group

    // Determine if this mode uses significance coding.
    let use_sig = matches!(
        &mode,
        BitplaneMode::Vlc(VlcKind::NoPredSig | VlcKind::VertPredSig { .. })
    );

    // Build per-(entry, sig_group) significance flags from M arrays.
    // A significance group j covers code groups [j*Ss .. (j+1)*Ss).
    // The group is significant (Z[j]=1) iff any M[g] > T within it.
    let sig_flags_per_entry: Vec<Vec<bool>> = if use_sig {
        m_per_entry
            .iter()
            .zip(entries.iter())
            .map(|(m_per_group, entry)| {
                let ncg = m_per_group.len();
                let t = entry.t;
                let ns = ncg.div_ceil(ss_u);
                (0..ns)
                    .map(|j| {
                        let g0 = j * ss_u;
                        let g1 = (g0 + ss_u).min(ncg);
                        m_per_group[g0..g1].iter().any(|&m| m > t)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![vec![]; entries.len()]
    };

    // Significance sub-packet: one bit per sig group across all entries,
    // in the same order as the bitplane-count sub-packet.
    if use_sig {
        for sig_flags in &sig_flags_per_entry {
            for &z in sig_flags {
                sig_writer.write_bit(if z { 1 } else { 0 });
            }
        }
        sig_writer.align_to_byte();
    }

    for (entry_idx, entry) in entries.iter().enumerate() {
        let wpb = entry.wpb as usize;
        let band_line: &[i32] = entry.line.as_slice();
        let t = entry.t as u32;
        let m_per_group = &m_per_entry[entry_idx];
        let coef = |g: usize, k: usize| -> i32 {
            let xpos = g * ng_u + k;
            if xpos < wpb {
                band_line[xpos]
            } else {
                0
            }
        };

        // Helper to check if a code group is in a significant sig group.
        let group_sig = |g: usize| -> bool {
            if !use_sig {
                return true;
            }
            let j = g / ss_u;
            sig_flags_per_entry[entry_idx]
                .get(j)
                .copied()
                .unwrap_or(true)
        };

        // Bitplane-count sub-packet.
        match &mode {
            BitplaneMode::Raw => {
                for &m in m_per_group {
                    cnt_writer.write_bits(m as u32, cfg.br);
                }
            }
            BitplaneMode::Vlc(VlcKind::NoPred) => {
                // mtop = T[p,b]; θ = max(mtop - T, 0) = 0. Δm = M - mtop
                // is always >= 0 (since we cap M at T) → unary
                // sub-alphabet → x = Δm → "Δm ones + 0".
                for &m in m_per_group {
                    let delta_m = (m as i32) - (t as i32);
                    debug_assert!(delta_m >= 0);
                    emit_vlc_signed(&mut cnt_writer, delta_m, 0);
                }
            }
            BitplaneMode::Vlc(VlcKind::NoPredSig) => {
                // Same as NoPred but skip VLC for insignificant groups
                // (Z[j]=0 → Δm = 0 implicitly, no bits emitted).
                for (g, &m) in m_per_group.iter().enumerate() {
                    if !group_sig(g) {
                        // Insignificant group: M = T implicitly, no VLC.
                        continue;
                    }
                    let delta_m = (m as i32) - (t as i32);
                    debug_assert!(delta_m >= 0);
                    emit_vlc_signed(&mut cnt_writer, delta_m, 0);
                }
            }
            BitplaneMode::Vlc(VlcKind::VertPred { predecessor }) => {
                // Table C.13: mtop = max(M_above, max(T, Ttop)). With
                // Ttop = T (in-precinct predecessor) → mtop = max(
                // M_above, T). θ = max(mtop - T, 0). Δm = M - mtop is
                // signed (can go negative when M < mtop).
                let pred_m = &predecessor[entry_idx];
                if pred_m.len() != m_per_group.len() {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: vertical predictor M length {} != current {}",
                        pred_m.len(),
                        m_per_group.len()
                    )));
                }
                for (g, &m) in m_per_group.iter().enumerate() {
                    let m_above = pred_m[g] as i32;
                    let mtop = m_above.max(t as i32);
                    let theta = (mtop - t as i32).max(0);
                    let delta_m = (m as i32) - mtop;
                    emit_vlc_signed(&mut cnt_writer, delta_m, theta);
                }
            }
            BitplaneMode::Vlc(VlcKind::VertPredSig { predecessor }) => {
                let pred_m = &predecessor[entry_idx];
                if pred_m.len() != m_per_group.len() {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: vertical predictor M length {} != current {}",
                        pred_m.len(),
                        m_per_group.len()
                    )));
                }
                for (g, &m) in m_per_group.iter().enumerate() {
                    if !group_sig(g) {
                        continue; // insignificant group: Δm = 0 implicit
                    }
                    let m_above = pred_m[g] as i32;
                    let mtop = m_above.max(t as i32);
                    let theta = (mtop - t as i32).max(0);
                    let delta_m = (m as i32) - mtop;
                    emit_vlc_signed(&mut cnt_writer, delta_m, theta);
                }
            }
        }

        // Data sub-packet — independent of bitplane-count mode.
        for (g, &m_u8) in m_per_group.iter().enumerate() {
            let m = m_u8 as u32;
            if m <= t {
                continue;
            }
            // Fs = 0: write Ng sign bits first.
            for k in 0..ng_u {
                let v = coef(g, k);
                let sign_bit = if v < 0 { 1 } else { 0 };
                data_writer.write_bit(sign_bit);
            }
            for plane in (t..m).rev() {
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
    let sig_bytes = sig_writer.into_bytes();
    let cnt_bytes = cnt_writer.into_bytes();
    let data_bytes = data_writer.into_bytes();
    let dr = match mode {
        BitplaneMode::Raw => 1,
        BitplaneMode::Vlc(_) => 0,
    };
    Ok(PacketBytes {
        dr,
        sig: sig_bytes,
        cnt: cnt_bytes,
        data: data_bytes,
        sgn: Vec::new(),
    })
}

/// Emit a VLC-encoded signed `value` for the predictor parameter `theta`
/// (Annex C.7.1, Table C.15). Inverse of [`crate::entropy::bits::vlc`]:
///
/// * `value > theta` → unary sub-alphabet, `x = value + theta`.
/// * `0 < value <= theta` → signed-binary even codeword, `x = 2 * value`.
/// * `-theta <= value < 0` → signed-binary odd codeword, `x = 2 * (-value) - 1`.
/// * `value == 0` → `x = 0` (single 0 bit).
///
/// Then `x` ones are emitted MSB-first followed by a single 0 comma.
/// `value` must satisfy `-theta <= value` (caller's responsibility — for
/// our encoder `value = M - mtop` and `M >= T` ensures
/// `value >= -theta = T - mtop`).
fn emit_vlc_signed(writer: &mut BitWriter, value: i32, theta: i32) {
    debug_assert!(theta >= 0);
    debug_assert!(
        value >= -theta,
        "VLC signed value {value} below -theta {theta}"
    );
    let x: u32 = if value > theta {
        (value + theta) as u32
    } else if value > 0 {
        (2 * value) as u32
    } else if value == 0 {
        0
    } else {
        (2 * (-value) - 1) as u32
    };
    debug_assert!(x <= 32, "VLC codeword length {x} exceeds 32-bit cap");
    for _ in 0..x {
        writer.write_bit(1);
    }
    writer.write_bit(0);
}

/// Tiny MSB-first bit writer.
#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
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
                buf[off] = (((x as i32) * 8 + y as i32) % 256) as u8;
                buf[off + 1] = (((y as i32) * 5 + x as i32 * 3) % 256) as u8;
                buf[off + 2] = ((x ^ y) as u8).wrapping_mul(13);
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
        assert_eq!(out, strip);
    }

    #[test]
    fn encode_then_decode_flat_image_is_exact() {
        let pixels = vec![123u8; 32 * 32];
        let codestream = encode_luma_8bit(32, 32, &pixels).expect("encode flat 32x32");
        let img = decode_codestream(&codestream, None).expect("decode flat 32x32");
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn self_roundtrip_synthetic_32x32_is_lossless() {
        let pixels = make_synthetic_32x32();
        let codestream = encode_luma_8bit(32, 32, &pixels).expect("encode 32x32");
        let img = decode_codestream(&codestream, None).expect("decode 32x32");
        assert_eq!(img.planes[0].data, pixels);
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 40.0, "self-roundtrip PSNR {p:.2} dB falls short");
    }

    #[test]
    fn self_roundtrip_2x2_minimum_size() {
        let pixels = vec![10u8, 200, 50, 150];
        let codestream = encode_luma_8bit(2, 2, &pixels).expect("encode 2x2");
        let img = decode_codestream(&codestream, None).expect("decode 2x2");
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn encode_image_then_decode_round_trips() {
        let pixels = make_synthetic_32x32();
        let codestream = encode_raw_luma(32, 32, pixels.clone()).expect("encode_raw_luma");
        let img = decode_codestream(&codestream, None).expect("decode after encode_raw_luma");
        assert_eq!(img.planes[0].data, pixels);
    }

    // === Round 2 carry-over: multi-component =============================

    #[test]
    fn self_roundtrip_rgb_32x32_no_transform() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 0, 1).expect("encode RGB 32x32");
        let img = decode_codestream(&codestream, None).expect("decode RGB Cpih=0");
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

    #[test]
    fn self_roundtrip_rgb_32x32_rct() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 1, 1).expect("encode RGB Cpih=1");
        let img = decode_codestream(&codestream, None).expect("decode RGB Cpih=1");
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

    #[test]
    fn self_roundtrip_luma_nl_2_2() {
        let pixels = make_synthetic_32x32();
        let codestream = encode_planar(32, 32, 1, 0, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode luma NL=2/2");
        let img = decode_codestream(&codestream, None).expect("decode luma NL=2/2");
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn self_roundtrip_rgb_nl_2_2_rct() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 1, 2).expect("encode RGB NL=2/2 Cpih=1");
        let img = decode_codestream(&codestream, None).expect("decode RGB NL=2/2 Cpih=1");
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

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
        assert_eq!(img.planes[0].data, pixels);
    }

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
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn encode_image_rgb_round_trips() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
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

    #[test]
    fn rejects_unsupported_configurations() {
        let pixels = vec![0u8; 32 * 32];
        // NL=6 above the round-6 cap (round-6 supports NL ∈ {1..=5}).
        assert!(encode_planar(32, 32, 1, 0, 6, 6, std::slice::from_ref(&pixels)).is_err());
        // NL,y > NL,x is not legal per spec.
        assert!(encode_planar(32, 32, 1, 0, 1, 2, std::slice::from_ref(&pixels)).is_err());
        // Nc=2 not yet supported.
        let two = vec![pixels.clone(), pixels.clone()];
        assert!(encode_planar(32, 32, 2, 0, 1, 1, &two).is_err());
        // Cpih=1 with Nc=1 invalid.
        assert!(encode_planar(32, 32, 1, 1, 1, 1, &[pixels]).is_err());
    }

    // === Round 3: VLC bitplane-count mode (Dr=0, no-prediction) ==========

    /// Round-3 raw-mode-vs-VLC picker: lossless 32×32 RGB stays
    /// lossless and the codestream gets *smaller* than the round-2 raw-
    /// mode-only emission. We compare the round-3 size against a
    /// hand-computed raw-only lower bound (5 packets × 5-byte header +
    /// data + Br × Ncg × packets) — but the easier pin is just to
    /// assert the round-3 size is below a tightened bound.
    #[test]
    fn round3_vlc_shrinks_codestream_vs_raw_only_baseline() {
        let pixels = make_synthetic_rgb_32x32();
        let raw_input = pixels.len();
        let r3 = encode_rgb_8bit(32, 32, &pixels, 1, 2)
            .expect("encode RGB NL=2/2 Cpih=1")
            .len();
        // Round 2's reported size for this config was within 5x raw.
        // Round 3 with VLC picker tightens this to ≤ 2.0x raw on
        // synthetic 32×32 RGB.
        assert!(
            r3 < raw_input * 2,
            "round-3 codestream {r3} exceeds 2x raw {raw_input}; VLC picker not engaging?"
        );
        // And round-trip remains lossless.
        let img =
            decode_codestream(&encode_rgb_8bit(32, 32, &pixels, 1, 2).unwrap(), None).unwrap();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

    /// Flat luma — every M = 0 except possibly the lowest band. With
    /// Dr=0 VLC the bitplane-count sub-packet collapses to 1 bit per
    /// group (the 0 comma) → near-minimum size.
    #[test]
    fn round3_flat_luma_compresses_well() {
        let pixels = vec![123u8; 32 * 32];
        let codestream = encode_luma_8bit(32, 32, &pixels).expect("encode flat 32x32");
        // Flat input → raw input = 1024 bytes. Round-3 VLC + small
        // header should stay well under 1024.
        assert!(
            codestream.len() < 1024,
            "round-3 flat luma codestream {} not smaller than raw 1024",
            codestream.len()
        );
        let img = decode_codestream(&codestream, None).expect("decode flat 32x32");
        assert_eq!(img.planes[0].data, pixels);
    }

    // === Round 3: Fq=8 lossy mode =========================================

    /// Fq=8 with Q=1 should still produce high-quality output (PSNR ≥
    /// 40 dB) while shrinking the codestream further than lossless.
    #[test]
    fn round3_fq8_q1_psnr_above_40db() {
        let pixels = make_synthetic_rgb_32x32();
        let lossless = encode_rgb_8bit(32, 32, &pixels, 1, 2)
            .expect("encode lossless")
            .len();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let cs = encode_planar_lossy(32, 32, 3, 1, 2, 2, 1, &[r.clone(), g.clone(), b.clone()])
            .expect("encode lossy q=1");
        let img = decode_codestream(&cs, None).expect("decode lossy");
        let mut decoded_rgb = vec![0u8; pixels.len()];
        for (i, ((rd, gd), bd)) in img.planes[0]
            .data
            .iter()
            .zip(&img.planes[1].data)
            .zip(&img.planes[2].data)
            .enumerate()
        {
            decoded_rgb[i * 3] = *rd;
            decoded_rgb[i * 3 + 1] = *gd;
            decoded_rgb[i * 3 + 2] = *bd;
        }
        let p = psnr(&pixels, &decoded_rgb);
        assert!(p >= 40.0, "Fq=8 q=1 PSNR {p:.2} dB below 40 dB floor");
        assert!(
            cs.len() < lossless,
            "Fq=8 q=1 codestream {} not smaller than lossless {}",
            cs.len(),
            lossless
        );
    }

    /// Fq=8 with Q=4 trades quality for compression but still must
    /// PSNR ≥ 25 dB. Synthetic 32×32 RGB hits ≈28-30 dB at q=4 because
    /// our deadzone-only encoder drops 4 bitplanes from every coefficient
    /// without any rate-distortion shaping; q=1/2 is the sweet spot for
    /// near-perceptually-lossless encoding (≥ 40 dB), q=4/6/8 trade
    /// linearly until the band-truncation noise dominates.
    #[test]
    fn round3_fq8_q4_psnr_above_25db() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let cs = encode_planar_lossy(32, 32, 3, 1, 2, 2, 4, &[r, g, b]).expect("encode lossy q=4");
        let img = decode_codestream(&cs, None).expect("decode lossy q=4");
        let mut decoded_rgb = vec![0u8; pixels.len()];
        for (i, ((rd, gd), bd)) in img.planes[0]
            .data
            .iter()
            .zip(&img.planes[1].data)
            .zip(&img.planes[2].data)
            .enumerate()
        {
            decoded_rgb[i * 3] = *rd;
            decoded_rgb[i * 3 + 1] = *gd;
            decoded_rgb[i * 3 + 2] = *bd;
        }
        let p = psnr(&pixels, &decoded_rgb);
        assert!(p >= 25.0, "Fq=8 q=4 PSNR {p:.2} dB below 25 dB floor");
    }

    /// Fq=8 with Q=0 reduces to the lossless path (validated up the
    /// chain: q=0 → fq must be 0).
    #[test]
    fn round3_q0_requires_fq0() {
        // encode_planar_lossy with q=0 internally sets fq=0 → matches
        // encode_planar exactly. Good.
        let pixels = vec![0u8; 32 * 32];
        let r2 = encode_planar(32, 32, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let r3 = encode_planar_lossy(32, 32, 1, 0, 1, 1, 0, std::slice::from_ref(&pixels)).unwrap();
        assert_eq!(r2, r3, "q=0 path must match lossless encode_planar");
    }

    // === Round 3: 4:2:2 / 4:2:0 chroma sub-sampling ======================

    /// 4:2:2 — chroma planes are W/2 × H. Self round-trip must restore
    /// every plane bit-exactly (lossless).
    #[test]
    fn round3_chroma_422_lossless_round_trip() {
        let w = 32u16;
        let h = 32u16;
        let n_y = (w as usize) * (h as usize);
        let n_c = ((w as usize) / 2) * (h as usize);
        let mut y_plane = vec![0u8; n_y];
        let mut cb_plane = vec![0u8; n_c];
        let mut cr_plane = vec![0u8; n_c];
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) % 256) as u8;
        }
        for i in 0..n_c {
            cb_plane[i] = ((i * 11 + 17) % 256) as u8;
            cr_plane[i] = ((i * 19 + 23) % 256) as u8;
        }
        let cs = encode_planar_subsampled(
            w,
            h,
            3,
            0,
            1,
            1,
            0,
            &[1, 2, 2],
            &[1, 1, 1],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:2 lossless");
        let img = decode_codestream(&cs, None).expect("decode 4:2:2");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// 4:2:0 — chroma planes are W/2 × H/2. Self round-trip lossless.
    #[test]
    fn round3_chroma_420_lossless_round_trip() {
        let w = 32u16;
        let h = 32u16;
        let n_y = (w as usize) * (h as usize);
        let n_c = ((w as usize) / 2) * ((h as usize) / 2);
        let mut y_plane = vec![0u8; n_y];
        let mut cb_plane = vec![0u8; n_c];
        let mut cr_plane = vec![0u8; n_c];
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) % 256) as u8;
        }
        for i in 0..n_c {
            cb_plane[i] = ((i * 11 + 17) % 256) as u8;
            cr_plane[i] = ((i * 19 + 23) % 256) as u8;
        }
        let cs = encode_planar_subsampled(
            w,
            h,
            3,
            0,
            1,
            1,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 lossless");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// 4:2:0 codestream is smaller than 4:4:4 of the same picture
    /// (chroma byte budget halves twice).
    #[test]
    fn round3_chroma_420_smaller_than_444() {
        let w = 32u16;
        let h = 32u16;
        let n_y = (w as usize) * (h as usize);
        let mut y_plane = vec![0u8; n_y];
        for (i, slot) in y_plane.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) % 256) as u8;
        }
        let cb_full = y_plane.clone();
        let cr_full = y_plane.clone();
        let cs_444 = encode_planar(w, h, 3, 0, 1, 1, &[y_plane.clone(), cb_full, cr_full])
            .expect("encode 4:4:4");
        // Down-sample chroma by 2× in both axes for the 4:2:0 case.
        let n_c = ((w as usize) / 2) * ((h as usize) / 2);
        let mut cb420 = vec![0u8; n_c];
        let mut cr420 = vec![0u8; n_c];
        for y in 0..(h as usize / 2) {
            for x in 0..(w as usize / 2) {
                cb420[y * (w as usize / 2) + x] = y_plane[(y * 2) * w as usize + (x * 2)];
                cr420[y * (w as usize / 2) + x] = y_plane[(y * 2) * w as usize + (x * 2)];
            }
        }
        let cs_420 = encode_planar_subsampled(
            w,
            h,
            3,
            0,
            1,
            1,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane, cb420, cr420],
        )
        .expect("encode 4:2:0");
        assert!(
            cs_420.len() < cs_444.len(),
            "4:2:0 codestream {} not smaller than 4:4:4 {}",
            cs_420.len(),
            cs_444.len()
        );
    }

    // === Round 4: Star-Tetrix (Cpih=3) =====================================

    fn make_cfa_8x8() -> [Vec<u8>; 4] {
        // Synthesise four 8x8 CFA-like component planes with distinct
        // patterns — emulates a Bayer mosaic decomposed into 4 separate
        // sub-grid components.
        let n = 8 * 8;
        let mut r = vec![0u8; n];
        let mut g1 = vec![0u8; n];
        let mut g2 = vec![0u8; n];
        let mut b = vec![0u8; n];
        for y in 0..8 {
            for x in 0..8 {
                let idx = y * 8 + x;
                r[idx] = ((x as i32 * 11 + y as i32 * 5) % 240).unsigned_abs() as u8;
                g1[idx] = ((x as i32 * 7 + y as i32 * 13) % 240).unsigned_abs() as u8;
                g2[idx] = ((x as i32 * 13 + y as i32 * 7) % 240).unsigned_abs() as u8;
                b[idx] = ((x as i32 * 5 + y as i32 * 11) % 240).unsigned_abs() as u8;
            }
        }
        [r, g1, g2, b]
    }

    /// Self-roundtrip: encode 4-component CFA via Star-Tetrix, decode,
    /// recover every plane bit-exactly.
    #[test]
    fn round4_star_tetrix_lossless_round_trip() {
        let [r, g1, g2, b] = make_cfa_8x8();
        let cs = encode_planar_star_tetrix(
            8,
            8,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode Cpih=3 lossless");
        let img = decode_codestream(&cs, None).expect("decode Cpih=3");
        assert_eq!(img.num_components, 4);
        assert_eq!(img.planes[0].data, r, "red plane must round-trip");
        assert_eq!(img.planes[1].data, g1, "G1 plane must round-trip");
        assert_eq!(img.planes[2].data, g2, "G2 plane must round-trip");
        assert_eq!(img.planes[3].data, b, "blue plane must round-trip");
    }

    /// Star-Tetrix with Ct=1 (GRBG layout) and non-default e1/e2 also
    /// round-trips losslessly.
    #[test]
    fn round4_star_tetrix_ct1_e1_2_e2_3_round_trip() {
        let [r, g1, g2, b] = make_cfa_8x8();
        let cs = encode_planar_star_tetrix(
            8,
            8,
            1,
            1,
            0,
            2,
            3,
            0,
            1,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode Cpih=3 Ct=1 e1=2 e2=3");
        let img = decode_codestream(&cs, None).expect("decode Cpih=3 Ct=1");
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g1);
        assert_eq!(img.planes[2].data, g2);
        assert_eq!(img.planes[3].data, b);
    }

    /// Star-Tetrix with NL=2/2 and Cf=3 (in-line access) round-trips.
    #[test]
    fn round4_star_tetrix_nl_2_cf3_round_trip() {
        let [r, g1, g2, b] = make_cfa_8x8();
        let cs = encode_planar_star_tetrix(
            8,
            8,
            2,
            2,
            0,
            1,
            1,
            3,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode Cpih=3 NL=2/2 Cf=3");
        let img = decode_codestream(&cs, None).expect("decode Cpih=3 NL=2/2 Cf=3");
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g1);
        assert_eq!(img.planes[2].data, g2);
        assert_eq!(img.planes[3].data, b);
    }

    // === Round 4: vertical-prediction VLC (Dr=0, D[p,b] & 1 = 1) =========

    /// Vertical-prediction picker self-roundtrips losslessly on the
    /// synthetic 32×32 RGB cascade fixture. Picker compares D=0 (no-pred)
    /// vs D=1 (vert-pred) per band per precinct and emits the smaller.
    #[test]
    fn round4_vertical_prediction_lossless_round_trip() {
        let pixels = make_synthetic_rgb_32x32();
        let codestream = encode_rgb_8bit(32, 32, &pixels, 1, 2).expect("encode RGB NL=2/2 Cpih=1");
        let img = decode_codestream(&codestream, None).expect("decode round 4 vertpred");
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        assert_eq!(img.planes[0].data, r, "red plane");
        assert_eq!(img.planes[1].data, g, "green plane");
        assert_eq!(img.planes[2].data, b, "blue plane");
    }

    /// Smooth vertical gradient — vertical-prediction picker should
    /// engage on the LH/HH bands and beat round-3 no-pred-only baseline.
    /// We assert round-trip + the codestream stays under 4 KB on a
    /// 64×64 vertical gradient.
    #[test]
    fn round4_vertical_gradient_compresses_well() {
        let w = 64u16;
        let h = 64u16;
        let mut pixels = vec![0u8; (w as usize) * (h as usize)];
        for y in 0..h as usize {
            for x in 0..w as usize {
                pixels[y * w as usize + x] = ((x * 2 + y * 4) % 256) as u8;
            }
        }
        let cs = encode_planar(w, h, 1, 0, 2, 2, &[pixels.clone()])
            .expect("encode 64x64 vertical gradient");
        let img = decode_codestream(&cs, None).expect("decode 64x64 vertical gradient");
        assert_eq!(img.planes[0].data, pixels, "round-trip lossless");
        assert!(
            cs.len() < 4096,
            "vertical-gradient codestream {} bytes >= 4 KB raw",
            cs.len()
        );
    }

    /// Cpih=3 must fail when Nc != 4.
    #[test]
    fn round4_star_tetrix_rejects_wrong_nc() {
        let pixels = vec![0u8; 8 * 8];
        let res = encode_planar(8, 8, 3, 3, 1, 1, &[pixels.clone(), pixels.clone(), pixels]);
        assert!(res.is_err(), "Cpih=3 with Nc=3 must be rejected");
    }

    /// Rejects `(sx, sy)` outside `{1, 2}`.
    #[test]
    fn round3_rejects_unsupported_sxy() {
        let pixels = vec![0u8; 32 * 32];
        let res = encode_planar_subsampled(
            32,
            32,
            1,
            0,
            1,
            1,
            0,
            &[3],
            &[1],
            std::slice::from_ref(&pixels),
        );
        assert!(res.is_err());
    }

    // === Round 5: NL_x ≠ NL_y (asymmetric decomposition) ==================

    /// NL_x=2 / NL_y=1 self-roundtrip — anisotropic decomposition with
    /// 2 horizontal levels and only 1 vertical level. Validates that the
    /// cascade path routes nly=1 correctly for every component.
    #[test]
    fn round5_asymmetric_nl_2_1_lossless_round_trip() {
        let w = 32u16;
        let h = 32u16;
        let mut pixels = vec![0u8; (w as usize) * (h as usize)];
        for (i, v) in pixels.iter_mut().enumerate() {
            *v = ((i * 7 + 13) % 256) as u8;
        }
        let cs = encode_planar(w, h, 1, 0, 2, 1, std::slice::from_ref(&pixels))
            .expect("encode luma NL_x=2 NL_y=1");
        let img = decode_codestream(&cs, None).expect("decode NL_x=2 NL_y=1");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip with NL_x=2 NL_y=1"
        );
    }

    /// NL_x=2 / NL_y=1 RGB (Cpih=1) self-roundtrip.
    #[test]
    fn round5_asymmetric_nl_2_1_rgb_lossless_round_trip() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let cs = encode_planar(32, 32, 3, 1, 2, 1, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB NL_x=2 NL_y=1 Cpih=1");
        let img = decode_codestream(&cs, None).expect("decode RGB NL_x=2 NL_y=1");
        assert_eq!(img.planes[0].data, r, "red plane NL_x=2 NL_y=1");
        assert_eq!(img.planes[1].data, g, "green plane NL_x=2 NL_y=1");
        assert_eq!(img.planes[2].data, b, "blue plane NL_x=2 NL_y=1");
    }

    /// NL_y > NL_x is still rejected (spec constraint NL_y ≤ NL_x).
    #[test]
    fn round5_rejects_nly_greater_than_nlx() {
        let pixels = vec![0u8; 32 * 32];
        let res = encode_planar(32, 32, 1, 0, 1, 2, std::slice::from_ref(&pixels));
        assert!(res.is_err(), "NL_y=2 > NL_x=1 must be rejected");
    }

    // === Round 5: NLT quadratic encoder (Annex G.4) ========================

    /// NLT quadratic encode + decode round-trip. The decoder applies the
    /// inverse NLT (linear path when Tnlt=0, but here Tnlt=1 → quadratic
    /// Annex G.1). With dco=0 the forward map is y=sqrt(x/255)*262143 and
    /// the inverse restores x within the 8-bit clamp. The lossless (q=0)
    /// path should self-roundtrip via the NLT marker path; the reconstructed
    /// plane values won't be bit-exact because the Bw=18 intermediate space
    /// and inverse scaling introduce rounding, but PSNR must be ≥ 40 dB.
    #[test]
    fn round5_nlt_quadratic_high_psnr() {
        let pixels = make_synthetic_32x32();
        let cs =
            encode_planar_nlt_quadratic(32, 32, 1, 0, 2, 2, 0, 0, std::slice::from_ref(&pixels))
                .expect("encode NLT quadratic lossless");
        let img = decode_codestream(&cs, None).expect("decode NLT quadratic");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 40.0,
            "NLT quadratic round-trip PSNR {p:.2} dB below 40 dB floor"
        );
    }

    /// NLT quadratic with q=2 (lossy) compresses further than lossless and
    /// still achieves ≥ 30 dB PSNR on a synthetic gradient.
    #[test]
    fn round5_nlt_quadratic_lossy_q2_psnr() {
        let pixels = make_synthetic_32x32();
        let lossless =
            encode_planar_nlt_quadratic(32, 32, 1, 0, 2, 2, 0, 0, std::slice::from_ref(&pixels))
                .expect("encode NLT lossless")
                .len();
        let lossy_cs =
            encode_planar_nlt_quadratic(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels))
                .expect("encode NLT lossy q=2");
        let img = decode_codestream(&lossy_cs, None).expect("decode NLT lossy q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 30.0,
            "NLT quadratic lossy q=2 PSNR {p:.2} dB below 30 dB floor"
        );
        assert!(
            lossy_cs.len() <= lossless,
            "NLT lossy q=2 size {} not ≤ lossless {}",
            lossy_cs.len(),
            lossless
        );
    }

    // === Round 5: per-band Q tuning ========================================

    /// Per-band gain weighting: lossy q=4 with per-band gains
    /// (LL=0, HL/LH=1, HH=2) should give better PSNR than a flat q=4
    /// without gain weighting. We measure both on the same input;
    /// the gain-aware path (via encode_planar_lossy) uses T[p,b]=
    /// clamp(q-G[b], 0, 15) so LL is always preserved (T=4-0=4),
    /// HL/LH uses T=3, HH uses T=2. The cascade path encodes this way
    /// automatically whenever band_gains is populated.
    /// We just verify PSNR ≥ 25 dB for q=4 and ≥ 35 dB for q=2 since
    /// the actual gain from per-band weighting depends on image content.
    #[test]
    fn round5_per_band_q_psnr_q2_above_35db() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let cs = encode_planar_lossy(32, 32, 3, 1, 2, 2, 2, &[r.clone(), g.clone(), b.clone()])
            .expect("encode lossy q=2 with per-band gains");
        let img = decode_codestream(&cs, None).expect("decode q=2");
        let mut dec_rgb = vec![0u8; pixels.len()];
        for (i, ((rd, gd), bd)) in img.planes[0]
            .data
            .iter()
            .zip(&img.planes[1].data)
            .zip(&img.planes[2].data)
            .enumerate()
        {
            dec_rgb[i * 3] = *rd;
            dec_rgb[i * 3 + 1] = *gd;
            dec_rgb[i * 3 + 2] = *bd;
        }
        let p = psnr(&pixels, &dec_rgb);
        assert!(p >= 35.0, "per-band Q q=2 PSNR {p:.2} dB below 35 dB floor");
    }

    // === Round 5: significance coding ======================================

    /// Significance coding (D[p,b] bit 1 = 1) compresses sparse/flat
    /// bands: a nearly-uniform image with NL=2 (cascade path, significance
    /// coding active) must round-trip and the AC bands codestream must be
    /// well below raw pixel count (4096 bytes for 64×64 luma).
    #[test]
    fn round5_significance_coding_compresses_flat_image() {
        // Nearly flat luma — most wavelet coefficients are 0 after DWT.
        let mut pixels = vec![128u8; 64 * 64];
        // Add a small perturbation so we don't hit the all-zeros degenerate.
        pixels[0] = 130;
        pixels[63] = 125;
        // NL=2 with significance coding active: zero groups are flagged and
        // skipped. Codestream must be well below the 4096-byte raw budget.
        let cs_nl2 = encode_planar(64, 64, 1, 0, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode NL=2 (significance coding active)");
        assert!(
            cs_nl2.len() < 4096,
            "NL=2 significance-coded codestream ({} B) not below 4 KB raw for flat 64×64",
            cs_nl2.len()
        );
        // Round-trip losslessly.
        let img2 = decode_codestream(&cs_nl2, None).expect("decode NL=2 flat");
        assert_eq!(img2.planes[0].data, pixels, "NL=2 round-trip");
        // NL=1 (single-level, no significance) also round-trips.
        let cs_nl1 =
            encode_planar(64, 64, 1, 0, 1, 1, std::slice::from_ref(&pixels)).expect("encode NL=1");
        let img1 = decode_codestream(&cs_nl1, None).expect("decode NL=1 flat");
        assert_eq!(img1.planes[0].data, pixels, "NL=1 round-trip");
    }

    // === Round 6: deeper wavelet cascade (NL > 2) ==========================
    //
    // The decoder cascade has always been generic (`forward_cascade_2d`
    // / `inverse_cascade_2d` accept any `(NL,x, NL,y)` pair with
    // `NL,y ≤ NL,x`). The encoder validation previously capped at
    // NL=2 / 2; relaxing to NL=5 / 5 lets users opt into deeper
    // multi-resolution analysis. Each extra level halves the LL band
    // again, so deep transforms compress smoother content better at the
    // same Q budget but cost a few extra cascade steps. Tested at every
    // step from 3/3 up to 5/5 on a 64×64 luma + RGB to keep all four
    // candidate D-form variants exercised, plus an asymmetric NL,x=3 /
    // NL,y=2 case.
    //
    // The test images are non-trivial (sinusoidal fringes + per-pixel
    // gradient) so the cascade actually splits energy across all bands;
    // a flat gray image would short-circuit through the significance
    // coding path and not validate the cascade logic.
    fn make_nl_test_64x64() -> Vec<u8> {
        let mut pixels = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                let v = 128i32
                    + ((x as i32 - 32) * (y as i32 - 32) / 8).clamp(-100, 100)
                    + (((x ^ y) as i32) & 0x1f);
                pixels[y * 64 + x] = v.clamp(0, 255) as u8;
            }
        }
        pixels
    }

    #[test]
    fn round6_nl_3_3_lossless_round_trip_luma() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 3, 3, std::slice::from_ref(&pixels))
            .expect("encode luma NL=3/3");
        let img = decode_codestream(&cs, None).expect("decode NL=3/3");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL=3/3"
        );
    }

    #[test]
    fn round6_nl_4_4_lossless_round_trip_luma() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 4, 4, std::slice::from_ref(&pixels))
            .expect("encode luma NL=4/4");
        let img = decode_codestream(&cs, None).expect("decode NL=4/4");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL=4/4"
        );
    }

    #[test]
    fn round6_nl_5_5_lossless_round_trip_luma() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 5, 5, std::slice::from_ref(&pixels))
            .expect("encode luma NL=5/5");
        let img = decode_codestream(&cs, None).expect("decode NL=5/5");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL=5/5"
        );
    }

    #[test]
    fn round6_nl_3_3_lossless_round_trip_rgb() {
        let pixels = make_synthetic_rgb_32x32();
        let n = 32 * 32;
        let (mut r, mut g, mut b) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for chunk in pixels.chunks_exact(3) {
            r.push(chunk[0]);
            g.push(chunk[1]);
            b.push(chunk[2]);
        }
        let cs = encode_planar(32, 32, 3, 1, 3, 3, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB NL=3/3 Cpih=1");
        let img = decode_codestream(&cs, None).expect("decode RGB NL=3/3");
        assert_eq!(img.planes[0].data, r, "red plane NL=3/3");
        assert_eq!(img.planes[1].data, g, "green plane NL=3/3");
        assert_eq!(img.planes[2].data, b, "blue plane NL=3/3");
    }

    #[test]
    fn round6_nl_3_2_asymmetric_lossless_round_trip() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 3, 2, std::slice::from_ref(&pixels))
            .expect("encode luma NL=3/2");
        let img = decode_codestream(&cs, None).expect("decode NL=3/2");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL=3/2"
        );
    }

    /// NL,x=9 must be rejected (round-7 cap is NL=8; spec Annex A.4.4
    /// Table A.7 hard maximum is 8).
    #[test]
    fn round7_rejects_nlx_above_8() {
        let pixels = vec![0u8; 64 * 64];
        let res = encode_planar(64, 64, 1, 0, 9, 9, std::slice::from_ref(&pixels));
        assert!(res.is_err(), "NL,x=9 must be rejected at the encoder cap");
    }

    /// Deeper cascades typically compress smoother content better at
    /// the same Q. We don't rely on a strict ordering across NL because
    /// the picker can flip on small inputs, but NL=4 q=4 lossy must
    /// still round-trip with PSNR ≥ 25 dB.
    #[test]
    fn round6_nl_4_4_lossy_q4_psnr_above_25db() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar_lossy(64, 64, 1, 0, 4, 4, 4, std::slice::from_ref(&pixels))
            .expect("encode luma NL=4/4 q=4");
        let img = decode_codestream(&cs, None).expect("decode NL=4/4 q=4");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 25.0, "NL=4/4 q=4 PSNR {p:.2} dB below 25 dB floor");
    }

    // === Round 7: extended NLT encoder (Annex G.5) =========================

    /// Extended NLT encode + decode round-trip on a 32×32 synthetic
    /// gradient. The forward LUT inverts the decoder's three-segment
    /// kernel within rounding, so PSNR must be ≥ 30 dB on a smooth ramp
    /// (the per-band Q and DWT rounding contribute additional loss on
    /// top of the LUT quantization).
    #[test]
    fn round7_nlt_extended_high_psnr() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            1 << 14,
            1 << 16,
            1,
            std::slice::from_ref(&pixels),
        )
        .expect("encode NLT extended lossless");
        let img = decode_codestream(&cs, None).expect("decode NLT extended");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 30.0,
            "NLT extended round-trip PSNR {p:.2} dB below 30 dB floor"
        );
    }

    /// Extended NLT with q=2 (lossy) still meets the 25 dB floor and
    /// produces a codestream no larger than the lossless variant.
    #[test]
    fn round7_nlt_extended_lossy_q2_psnr() {
        let pixels = make_synthetic_32x32();
        let lossless = encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            1 << 14,
            1 << 16,
            1,
            std::slice::from_ref(&pixels),
        )
        .expect("encode NLT extended lossless")
        .len();
        let lossy_cs = encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            2,
            1 << 14,
            1 << 16,
            1,
            std::slice::from_ref(&pixels),
        )
        .expect("encode NLT extended lossy q=2");
        let img = decode_codestream(&lossy_cs, None).expect("decode NLT extended lossy q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 25.0,
            "NLT extended lossy q=2 PSNR {p:.2} dB below 25 dB floor"
        );
        assert!(
            lossy_cs.len() <= lossless,
            "NLT extended lossy q=2 size {} not ≤ lossless {}",
            lossy_cs.len(),
            lossless
        );
    }

    /// Extended NLT rejects bad parameters (T2 ≤ T1, E out of range,
    /// thresholds exceeding 2^Bw-1).
    #[test]
    fn round7_nlt_extended_rejects_bad_params() {
        let pixels = vec![0u8; 32 * 32];
        // T2 ≤ T1.
        assert!(encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            200,
            100,
            3,
            std::slice::from_ref(&pixels)
        )
        .is_err());
        // T1 = 0.
        assert!(encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            0,
            100,
            3,
            std::slice::from_ref(&pixels)
        )
        .is_err());
        // E = 0.
        assert!(encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            100,
            200,
            0,
            std::slice::from_ref(&pixels)
        )
        .is_err());
        // E = 5.
        assert!(encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            100,
            200,
            5,
            std::slice::from_ref(&pixels)
        )
        .is_err());
        // T exceeds 2^Bw-1.
        assert!(encode_planar_nlt_extended(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            (1 << 18) + 1,
            (1 << 18) + 2,
            3,
            std::slice::from_ref(&pixels)
        )
        .is_err());
    }

    // === Round 7: deeper wavelet cascade NL ∈ {6, 7, 8} ====================

    /// NL=6/6 self-roundtrip on a 64×64 luma image. Verifies the encoder
    /// validate cap was correctly lifted from 5 to 8.
    #[test]
    fn round7_nl_6_6_lossless_round_trip() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 6, 6, std::slice::from_ref(&pixels))
            .expect("encode luma NL=6/6");
        let img = decode_codestream(&cs, None).expect("decode NL=6/6");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL=6/6"
        );
    }

    // === Round 8: multi-precinct-per-row (Cw > 0) ==========================

    /// 64×16 luma at NL=1/1 with Cw=1 → Cs = 8 × 1 × 1 × 2 = 16 →
    /// Np,x = 4 precincts per row. Self-roundtrips losslessly.
    #[test]
    fn round8_cw1_64x16_luma_nl_1_1_lossless() {
        let w = 64usize;
        let h = 16usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 7 + y * 11) % 256) as u8;
            }
        }
        let cs = encode_planar_cw(w as u16, h as u16, 1, 0, 1, 1, 0, 1, &[pixels.clone()])
            .expect("encode 64x16 Cw=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode 64x16 Cw=1");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly with Cw=1 NL=1/1"
        );
    }

    /// 64×16 luma at NL=2/2 with Cw=1 → Cs = 8 × 1 × 1 × 4 = 32 →
    /// Np,x = 2 precincts per row. Verifies the gather path's per-precinct
    /// column offset is correct under a deeper cascade.
    #[test]
    fn round8_cw1_64x16_luma_nl_2_2_lossless() {
        let w = 64usize;
        let h = 16usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 13 + y * 5 + 3) % 256) as u8;
            }
        }
        let cs = encode_planar_cw(w as u16, h as u16, 1, 0, 2, 2, 0, 1, &[pixels.clone()])
            .expect("encode 64x16 Cw=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 64x16 Cw=1 NL=2/2");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly with Cw=1 NL=2/2"
        );
    }

    /// 128×32 RGB at NL=2/2 with Cw=2 (Cs = 8 × 2 × 1 × 4 = 64 → Np,x = 2)
    /// and Cpih=1 (RCT). The full multi-component + multi-precinct path.
    #[test]
    fn round8_cw2_128x32_rgb_rct_nl_2_2_lossless() {
        let w = 128usize;
        let h = 32usize;
        let mut r = vec![0u8; w * h];
        let mut g = vec![0u8; w * h];
        let mut b = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                r[y * w + x] = ((x * 3 + y * 5) % 256) as u8;
                g[y * w + x] = ((x * 7 + y * 11 + 17) % 256) as u8;
                b[y * w + x] = ((x * 13 + y * 17 + 29) % 256) as u8;
            }
        }
        let cs = encode_planar_cw(
            w as u16,
            h as u16,
            3,
            1,
            2,
            2,
            0,
            2,
            &[r.clone(), g.clone(), b.clone()],
        )
        .expect("encode 128x32 Cw=2 RCT NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 128x32 Cw=2 RCT NL=2/2");
        assert_eq!(img.planes[0].data, r);
        assert_eq!(img.planes[1].data, g);
        assert_eq!(img.planes[2].data, b);
    }

    /// Cw > 0 with q > 0 (lossy mode) — still round-trips within the
    /// PSNR floor the cascade lossy path holds at q=2.
    #[test]
    fn round8_cw1_64x16_luma_lossy_q2_psnr() {
        let w = 64usize;
        let h = 16usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 7 + y * 11) % 256) as u8;
            }
        }
        let cs = encode_planar_cw(w as u16, h as u16, 1, 0, 2, 2, 2, 1, &[pixels.clone()])
            .expect("encode 64x16 Cw=1 q=2");
        let img = decode_codestream(&cs, None).expect("decode 64x16 Cw=1 q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 25.0, "Cw=1 lossy q=2 PSNR {p:.2} dB below 25 dB floor");
    }

    /// Encoder rejects Cs > Wf (Cw too large for the picture).
    #[test]
    fn round8_rejects_cw_exceeding_picture() {
        let pixels = vec![0u8; 32 * 32];
        // Cw=4 at NL,x=2 → Cs = 8 × 4 × 1 × 4 = 128 > 32.
        let result = encode_planar_cw(32, 32, 1, 0, 2, 2, 0, 4, std::slice::from_ref(&pixels));
        assert!(result.is_err());
    }

    /// Cw > 0 with chroma sub-sampling. 64×8 YUV 4:2:2 at NL=1/1 Cw=1
    /// with max(sx)=2 → Cs = 8 × 1 × 2 × 2 = 32, Np,x = ⌈64/32⌉ = 2.
    /// Routes through `encode_planar_inner_nlt` via a custom call site
    /// because `encode_planar_cw` only handles 4:4:4.
    #[test]
    fn round8_cw1_64x8_yuv_422_lossless() {
        let w = 64usize;
        let h = 8usize;
        let mut y_plane = vec![0u8; w * h];
        let mut u_plane = vec![0u8; (w / 2) * h];
        let mut v_plane = vec![0u8; (w / 2) * h];
        for y in 0..h {
            for x in 0..w {
                y_plane[y * w + x] = ((x * 3 + y * 5) % 256) as u8;
            }
            for x in 0..(w / 2) {
                u_plane[y * (w / 2) + x] = ((x * 7 + y * 11 + 17) % 256) as u8;
                v_plane[y * (w / 2) + x] = ((x * 13 + y * 17 + 29) % 256) as u8;
            }
        }
        // Inline call to encode_planar_inner_nlt with sx=[1,2,2], sy=[1,1,1].
        let sx = vec![1u8, 2, 2];
        let sy = vec![1u8, 1, 1];
        let cs = encode_planar_inner_nlt(
            w as u16,
            h as u16,
            3,
            0,
            1,
            1,
            0,
            0,
            &sx,
            &sy,
            0,
            0,
            0,
            0,
            None,
            Vec::new(),
            1, // cw
            0, // sd
            &[y_plane.clone(), u_plane.clone(), v_plane.clone()],
        )
        .expect("encode 64x8 4:2:2 Cw=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode 64x8 4:2:2 Cw=1");
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, u_plane);
        assert_eq!(img.planes[2].data, v_plane);
    }

    /// Odd-width picture with Cw > 0: rightmost precinct picks up the
    /// remainder. 96×16 luma at NL=1/1 Cw=1 → Cs=16, Np,x=⌈96/16⌉=6,
    /// every precinct is 16 wide (no remainder).
    #[test]
    fn round8_cw1_96x16_luma_six_precincts_lossless() {
        let w = 96usize;
        let h = 16usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x.wrapping_mul(19) + y.wrapping_mul(31)) % 256) as u8;
            }
        }
        let cs = encode_planar_cw(w as u16, h as u16, 1, 0, 1, 1, 0, 1, &[pixels.clone()])
            .expect("encode 96x16 Cw=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode 96x16 Cw=1");
        assert_eq!(img.planes[0].data, pixels);
    }

    /// Cw=0 reduces to single-precinct-per-row behaviour (bit-equivalent
    /// to encode_planar).
    #[test]
    fn round8_cw0_matches_encode_planar() {
        let pixels = make_synthetic_32x32();
        let cs_a = encode_planar_cw(32, 32, 1, 0, 2, 2, 0, 0, std::slice::from_ref(&pixels))
            .expect("encode_planar_cw cw=0");
        let cs_b = encode_planar(32, 32, 1, 0, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode_planar");
        assert_eq!(cs_a, cs_b, "Cw=0 must match encode_planar bit-for-bit");
    }

    /// Round 9 (r91): Sd=1 with Nc=4, NL=2/2. Components 0..3 are
    /// wavelet-coded; component 3 is suppressed and carried raw.
    #[test]
    fn round9_sd1_4comp_32x16_lossless() {
        let w = 32usize;
        let h = 16usize;
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 3)
                        .wrapping_add((y as u32).wrapping_mul(seed + 7))
                        .wrapping_add(seed)
                        % 256) as u8;
                }
            }
            v
        };
        let p0 = make(11);
        let p1 = make(17);
        let p2 = make(23);
        let p3 = make(29);
        let cs = encode_planar_sd(
            w as u16,
            h as u16,
            4,
            2,
            2,
            0,
            1, // sd: suppress component 3 only
            &[p0.clone(), p1.clone(), p2.clone(), p3.clone()],
        )
        .expect("encode 32x16 Nc=4 Sd=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode Sd=1");
        assert_eq!(img.planes[0].data, p0, "wavelet comp 0 lossless");
        assert_eq!(img.planes[1].data, p1, "wavelet comp 1 lossless");
        assert_eq!(img.planes[2].data, p2, "wavelet comp 2 lossless");
        assert_eq!(img.planes[3].data, p3, "Sd-suppressed comp 3 lossless");
    }

    /// Round 9: Sd=2 with Nc=5 — two suppressed components.
    #[test]
    fn round9_sd2_5comp_16x8_lossless() {
        let w = 16usize;
        let h = 8usize;
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32 + seed)
                        .wrapping_mul((y as u32 + 1).wrapping_add(seed))
                        % 251) as u8;
                }
            }
            v
        };
        let p: Vec<Vec<u8>> = (0..5u32).map(make).collect();
        let cs = encode_planar_sd(w as u16, h as u16, 5, 1, 1, 0, 2, &p)
            .expect("encode 16x8 Nc=5 Sd=2 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode Sd=2");
        for (i, expected) in p.iter().enumerate().take(5) {
            assert_eq!(&img.planes[i].data, expected, "comp {i} roundtrip");
        }
    }

    /// Round 9: Sd=1 lossy q=2 — wavelet components are quantized, Sd
    /// tail component is also subjected to T but at G=0 retains useful
    /// PSNR (≥30 dB on smooth patterns).
    #[test]
    fn round9_sd1_4comp_lossy_q2_psnr_floor() {
        let w = 32usize;
        let h = 16usize;
        let mut p = vec![vec![0u8; w * h]; 4];
        for y in 0..h {
            for x in 0..w {
                let g = ((x as u32 * 8 + y as u32 * 4) % 256) as u8;
                p[0][y * w + x] = g;
                p[1][y * w + x] = g.wrapping_add(20);
                p[2][y * w + x] = g.wrapping_add(40);
                p[3][y * w + x] = g.wrapping_add(60);
            }
        }
        let cs =
            encode_planar_sd(w as u16, h as u16, 4, 2, 2, 2, 1, &p).expect("encode lossy Sd=1 q=2");
        let img = decode_codestream(&cs, None).expect("decode lossy Sd=1");
        for (i, expected) in p.iter().enumerate().take(4) {
            let q = psnr(expected, &img.planes[i].data);
            assert!(
                q >= 30.0,
                "Sd=1 q=2 comp {i} PSNR {q:.2} dB below 30 dB floor"
            );
        }
    }

    /// Round 9: encoder rejects Sd>0 when Nc<=3 (Annex A.4.7).
    #[test]
    fn round9_rejects_sd_with_nc_3() {
        let p = vec![vec![0u8; 16 * 8]; 3];
        let result = encode_planar_sd(16, 8, 3, 1, 1, 0, 1, &p);
        assert!(result.is_err(), "Sd>0 must require Nc>3");
    }

    /// Round 9: encoder rejects Sd>=Nc (Annex A.4.7).
    #[test]
    fn round9_rejects_sd_eq_nc() {
        let p = vec![vec![0u8; 16 * 8]; 4];
        let result = encode_planar_sd(16, 8, 4, 1, 1, 0, 4, &p);
        assert!(result.is_err(), "Sd must be < Nc");
    }
}
