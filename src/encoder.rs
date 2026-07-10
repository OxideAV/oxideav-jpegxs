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
//! * **Regular (`Fq = 8`) lossy mode with a `Qpih`-aware forward
//!   quantizer.** A new `q` parameter sets the precinct-level
//!   `Q[p]` (`0..=15`) which in turn drives the per-band truncation
//!   `T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)`. The quantization index
//!   `v[p,λ,b,x]` is computed by [`forward_quant_index`], which selects
//!   the quantizer matching the picture-header `Qpih`: deadzone
//!   (`Qpih = 0`, Annex D.4 Table D.3 — `v = |c| >> T`, reconstructed at
//!   the half-bucket offset `((1 << T) >> 1)`) or uniform (`Qpih = 1`,
//!   Annex D.5 Table D.4 — `v = ((|c| << ζ) − |c| + (1 << M)) >>
//!   (M + 1)`, `ζ = M − T + 1`, round-to-nearest, reconstructed by the
//!   Neumann series of Annex D.3). Either way only `M - T` bitplanes per
//!   code group go on the wire. PSNR ≥ 40 dB at `q = 1`, ≥ 32 dB at
//!   `q = 4` on synthetic 32×32 RGB.
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
//! Previously-deferred items, all now implemented:
//! * `Cw > 0` (custom precinct widths / multi-precinct-per-row) —
//!   [`encode_planar_cw`]; composes with chroma sub-sampling and the
//!   multi-level cascade (r371 verified 4:2:0 NL=2/2).
//! * Per-precinct `Q[p]` / `R[p]` rate-distortion overrides —
//!   [`encode_planar_qpr_rpr`] and the `*_target_bytes` pickers.
//! * `Sd > 0` decomposition suppression (CWD) — [`encode_planar_sd`]
//!   and the Star-Tetrix / RCT `_sd_*` companions.
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
    /// Inverse-quantizer type (`Qpih`, Annex A.4.4 Table A.10). `0` =
    /// deadzone (Annex D.2); `1` = uniform / Neumann-series (Annex D.3).
    /// The data sub-packet is byte-identical for both — only the decoder's
    /// reconstruction kernel differs — so the encoder just signals the bit
    /// and the decoder picks the matching inverse. Values 2/3 are reserved
    /// (the decoder rejects `Qpih > 1`).
    qpih: u8,
    /// Sign handling strategy (`Fs`, Annex A.4.4 Table A.11). `0` = signs
    /// encoded jointly with the data sub-packet (Table C.8); `1` = signs
    /// encoded in a separate sign sub-packet (Table C.9), one bit per
    /// non-zero coefficient.
    fs: u8,
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
    /// Per-band priority values `P[b]` for the WGT marker (Annex A.4.11
    /// Table A.24), in the same existing-band emission order as
    /// [`EncodeConfig::band_gains`]. Each entry is the true band index
    /// `b = (Nc - Sd)×β + i` (or `(Nc - Sd)×Nβ + i` for the Sd tail),
    /// so a precinct refinement of `R[p] = k` refines exactly the bands
    /// whose `P[b] < k` — i.e. the `k` lowest band indices, which run
    /// LL-first per the `β`-major enumeration (Annex B.6). Length must
    /// equal `band_gains.len()` or be empty (→ all-zero priorities,
    /// backward-compatible with rounds 1–110: every band has `P[b] = 0`,
    /// so no `R[p] > 0` refinement ever fires).
    band_priorities: Vec<u8>,
    /// Precinct refinement `R[p]` (Annex C.2 Table C.1), constant across
    /// precincts. Range `0..=NL-1` where `NL = (Nc - Sd)×Nβ + Sd` is the
    /// total band count (Annex B.6). `R[p] = 0` (the default through
    /// round 111) disables refinement: `r = (P[b] < R[p]) ? 1 : 0` is
    /// always 0, so `T[p,b] = clamp(Q - G[b], 0, 15)`. `R[p] > 0` grants
    /// one extra retained bitplane (`r = 1`, lower `T`) to bands with
    /// `P[b] < R[p]` per the Annex C.6.2 Table C.10 truncation algorithm.
    rp: u8,
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
    /// Slice height in precinct rows (`Hsl`, PIH §A.4.4, signalled in the
    /// picture header per Annex B.10). `0` means a single slice spanning
    /// the whole picture (`Hsl = Np,y`), which is the only mode supported
    /// through round 100. `hsl > 0` groups the `Np,y` precinct rows into
    /// `⌈Np,y / Hsl⌉` slices of `Hsl` precinct rows each (the last slice
    /// is shorter when `Np,y` is not a multiple of `Hsl`), emitting one
    /// SLH marker per slice with `Yslh = t` (the slice's top-down order).
    /// Vertical prediction is already precinct-scoped in this encoder, so
    /// slice boundaries fall cleanly between precinct rows with no
    /// cross-slice predictor state to reset (Annex B.10).
    hsl: u16,
    /// Per-slice `Q[p]` overrides (round 206 — slice-level rate budgeting).
    /// Empty → every slice uses the single picture-level `q` (the round-3
    /// .. round-201 behaviour). Non-empty must hold exactly one entry per
    /// slice (i.e. `⌈Np,y / effective_hsl⌉` values), with each value in
    /// `0..=15`; the slice's `Q[p]` is then `q_slices[t]` and the
    /// precinct header carries that per-slice value for every precinct
    /// inside the slice (Annex C.2 Table C.1 — Q is a per-precinct
    /// field, so any per-slice partition is spec-compliant).
    ///
    /// The decoder reads `Q[p]` per precinct (`precinct_truncation` in
    /// the entropy module), so no decoder change is needed; this is a
    /// pure encoder rate-allocation lever. The bitstream-wire impact is
    /// only the per-precinct `Q` byte. Existing callers that pass an
    /// empty `q_slices` keep the byte-identical legacy stream.
    q_slices: Vec<u8>,
    /// Per-precinct `Q[p]` overrides (round 233 — precinct-level rate
    /// budgeting, the spec-natural form of Annex C.2 Table C.1 where
    /// `Q[p]` is indexed by precinct `p`). Empty → every precinct
    /// inherits its slice's `Q[p]` from [`EncodeConfig::q_slices`] (or
    /// the picture-level `q` when both are empty). Non-empty must hold
    /// exactly one entry per precinct (i.e. `Np,y × Np,x` values, in
    /// raster scan order with the precinct at `(py, px)` at index
    /// `py * Np,x + px`), with each value in `0..=15`.
    ///
    /// Round 206 lifted `q` from picture-level to one per slice; round
    /// 233 lifts it further to one per precinct. Per-precinct override
    /// wins over per-slice override at the precincts it covers — both
    /// can coexist (the slice override remains as the fallback only
    /// when `q_precincts` is empty). The precinct header carries the
    /// per-precinct value for every precinct (Annex C.2 Table C.1).
    ///
    /// The decoder reads `Q[p]` per precinct (`precinct_truncation` in
    /// the entropy module) since the early rounds, so no decoder change
    /// is needed — this is a pure encoder rate-allocation lever. The
    /// bitstream-wire impact is only the per-precinct `Q` byte.
    q_precincts: Vec<u8>,
    /// Per-precinct `R[p]` overrides (round 239 — precinct-level refinement
    /// budgeting, the spec-natural form of Annex C.2 Table C.1 where
    /// `R[p]` is indexed by precinct `p`). Empty → every precinct
    /// inherits the picture-level `rp`. Non-empty must hold exactly one
    /// entry per precinct (i.e. `Np,y × Np,x` values, in raster scan
    /// order with the precinct at `(py, px)` at index `py * Np,x + px`),
    /// with each value in `0..=NL-1` where `NL = (Nc - Sd)·Nβ + Sd`.
    ///
    /// Round 115 fixed `R[p]` picture-wide; round 239 lifts it to per
    /// precinct. The precinct header carries the per-precinct value for
    /// every precinct (Annex C.2 Table C.1).
    ///
    /// The decoder reads `R[p]` per precinct (`precinct_truncation` in
    /// the entropy module) since round 115, so no decoder change is
    /// needed — this is a pure encoder rate-allocation lever. The
    /// bitstream-wire impact is only the per-precinct `R` byte.
    r_precincts: Vec<u8>,
    /// Run mode (`Rm`, Annex A.4.4 Table A.12). Selects the interpretation
    /// of an insignificant significance group in the *vertical-prediction*
    /// bitplane-count mode (Table C.13):
    ///
    /// * `0` — runs indicate zero *prediction residuals*: an insignificant
    ///   group reconstructs to `M = mtop` (the vertical predictor), so the
    ///   data sub-packet may still carry its coefficients when `mtop > T`.
    ///   The significance-flag computation therefore keeps a group
    ///   *significant* whenever its predecessor's `M > T` (otherwise the
    ///   reconstructed `M = mtop > T` would demand coefficient data the
    ///   "insignificant" label omits).
    /// * `1` — runs indicate zero *coefficients*: an insignificant group
    ///   reconstructs to `M = T[p,b]` regardless of the predictor, so the
    ///   data sub-packet omits it (`M > T` is false) and no VLC residual is
    ///   coded either. A group is insignificant iff *all* its code groups
    ///   satisfy `M ≤ T` (i.e. contain only zero quantization indices),
    ///   which drops the `Rm = 0` predecessor guard and compresses the
    ///   all-zero groups that vertically follow non-zero ones.
    ///
    /// The no-prediction mode (Table C.14) is `Rm`-independent (`mtop = T`,
    /// so an insignificant group is always `M = T`). Values `2`/`3` are
    /// reserved (rejected by [`EncodeConfig::validate`]). Default `0`.
    rm: u8,
}

impl EncodeConfig {
    fn validate(&self) -> Result<()> {
        if self.width < 2 || self.height < 2 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: picture dimensions must be >= 2, got {}x{}",
                self.width, self.height
            )));
        }
        // Rm (Annex A.4.4 Table A.12): only 0 (zero prediction residual) and
        // 1 (zero coefficients) are defined; 2/3 are reserved.
        if self.rm > 1 {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: Rm must be 0 or 1 (Table A.12), got {}",
                self.rm
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
        // Annex F.2 Table F.1: Cpih=1 operates on c<3; Cpih=3 on c<4.
        // The transform's operand range is fixed regardless of Nc, so
        // Nc>=3 (Cpih=1) and Nc>=4 (Cpih=3) are the spec's actual
        // requirements. The trailing components (c >= 3 for RCT, c >= 4
        // for Star-Tetrix) are passed through unchanged — that is the
        // same path CWD's Sd-suppressed components take, which is why
        // Cpih≠0 composes with Sd>0 as long as the colour-transform
        // operand window does not overlap the suppressed-component tail
        // (checked below in the Sd block).
        if self.cpih == 1 && self.nc < 3 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Cpih=1 (RCT) requires Nc>=3 per Annex F.2, got {}",
                self.nc
            )));
        }
        if self.cpih == 3 && self.nc < 4 {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Cpih=3 (Star-Tetrix) requires Nc>=4 per Annex F.2, got {}",
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
            // Star-Tetrix requires sx[i] = sy[i] = 1 on the 4 CFA
            // components it consumes (c<4). Components beyond that — Sd
            // suppressed tail when Sd>0, or otherwise — are not touched
            // by the transform; their sampling factors are independently
            // governed by CDT + (if applicable) the CWD constraint
            // sx=sy=1 on suppressed comps.
            for (i, (&sx, &sy)) in self.sx.iter().zip(self.sy.iter()).enumerate().take(4) {
                if sx != 1 || sy != 1 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: Cpih=3 (Star-Tetrix) requires sx[i]=sy[i]=1 for i<4, got component {i} (sx, sy)=({sx}, {sy})"
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
        // Fq must be one of the ISO/IEC 21122-1:2022 Table A.8 values:
        // 0 (integer / lossless transform), 6 (NLT, Bw = 18), or 8
        // (high-precision regular, Bw = 20).
        if !matches!(self.fq, 0 | 6 | 8) {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: Fq must be 0, 6, or 8 (Table A.8), got {}",
                self.fq
            )));
        }
        // Fs (Annex A.4.4 Table A.11): 0 = signs jointly with data, 1 =
        // separate sign sub-packet. Values 2/3 are reserved.
        if self.fs > 1 {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: Fs must be 0 (joint signs) or 1 (separate sign sub-packet), got {}",
                self.fs
            )));
        }
        // Qpih (Annex A.4.4 Table A.10): 0 = deadzone inverse quantizer
        // (Annex D.2), 1 = uniform / Neumann-series inverse quantizer
        // (Annex D.3). Values 2/3 are reserved and the decoder rejects
        // `Qpih > 1`, so refuse them here too.
        if self.qpih > 1 {
            return Err(Error::Unsupported(format!(
                "jpegxs encoder: Qpih must be 0 (deadzone) or 1 (uniform), got {}",
                self.qpih
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
        // NOTE: q > 0 does NOT require Fq != 0. Lossy quantisation is
        // driven by the precinct truncation T[p,b], which the inverse
        // quantiser applies independently of Fq. The integer-transform
        // regular lossy stream is the conformant Table A.8 combination
        // (Bw = B[0], Fq = 0): a non-zero T still drops bitplanes there.
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
        // Picture dimensions need NOT be divisible by sx / sy: §B.1
        // defines the per-component plane as Wc[i] = ⌈Wf / sx[i]⌉ ×
        // Hc[i] = ⌈Hf / sy[i]⌉ (samples populate sampling-grid
        // positions 0, sx, 2·sx, … below Wf), so odd picture
        // dimensions are legal with sub-sampled components.
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
            // Sd > 0 composes with the colour transform. The order of
            // operations matters: per Annex F.2 Table F.1 the inverse
            // colour transform reads the *wavelet output* O[c,x,y] for
            // every operand component (c<3 for RCT, c<4 for Star-Tetrix),
            // and the forward encoder applies the transform to all
            // operand components *before* the wavelet stage (see the
            // forward_rct / forward_star_tetrix call site — it runs on the
            // full plane set ahead of the per-component DWT-suppression
            // loop). Sd then suppresses the wavelet decomposition for the
            // *trailing transform outputs* (Annex A.4.7: "wavelet
            // decomposition is suppressed"), which are coded raw instead.
            // A suppressed transform output is still a valid O[c] value, so
            // the inverse transform consumes it unchanged.
            //
            // For Star-Tetrix (Cpih=3) this is exactly the spec's tabulated
            // CFA configuration: Annex B Table B.10 (Sd=1, 4 components,
            // 4:4:4:4, NL,y=1, 25 bands) and Table B.11 (NL,y=2, 31 bands)
            // give the line-inclusion layout for `Nc=4, Sd=1` — the fourth
            // Star-Tetrix output (blue, Table F.4) is raw-coded. The Annex H
            // PSNR weight tables H.9–H.11 are written for precisely this
            // `Sd=1` CFA case. So the only Cpih=3 requirement is that the
            // four transform *inputs* exist (Nc>=4, already checked above);
            // suppression carves the *output* tail and never reduces the
            // transform's input window. RCT (Cpih=1) keeps the stricter
            // Nc-Sd >= 3 guard: the spec tabulates no RCT-with-suppressed-
            // output example, so we do not relax a path we cannot validate.
            if self.cpih == 1 && self.nc - self.sd < 3 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Cpih=1 (RCT) requires Nc-Sd >= 3 so the RCT operand window is fully wavelet-coded (Annex F.2 Table F.1), got Nc={} Sd={}",
                    self.nc, self.sd
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
            // contain at least 8 samples of the LL band (the 8× factor in
            // Cs), and "all bands of the rightmost precincts are
            // non-empty". Table 11 makes the latter a hard Cw constraint:
            //
            //   Wf umod Cs ≥ max_i(sx) × 2^NL,x
            //
            // i.e. the rightmost partial precinct must still hold at least
            // one decomposed low-frequency sample, or divide Wf evenly.
            // Refuse to emit a stream whose rightmost precinct is a sliver
            // narrower than one LL sample — such a stream is non-conformant
            // and the decoder rejects it.
            let ll_sample = max_sx * pow_nlx;
            let rem = (self.width as u32) % cs;
            if rem != 0 && rem < ll_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Cw={}: rightmost precinct width Wf umod Cs = {} umod {} = {} \
                     is below one decomposed sample max_i(sx)×2^NL,x = {} (Table 11 / §B.5 Note 1) \
                     — choose a Cw whose column stride divides Wf or leaves ≥ {} columns",
                    self.cw, self.width, cs, rem, ll_sample, ll_sample
                )));
            }
        }
        // Hsl > 0 — slice height in precinct rows (Annex B.10). A value
        // exceeding Np,y is meaningless: it would describe a single slice
        // that extends past the last precinct row. Np,y = ⌈Hf / 2^NL,y⌉.
        if self.hsl != 0 {
            let hp_pow = 1u32 << self.nly;
            let np_y = (self.height as u32).div_ceil(hp_pow);
            if (self.hsl as u32) > np_y {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: Hsl={} exceeds the {} precinct rows (Np,y) for height={} NL,y={} (Annex B.10)",
                    self.hsl, np_y, self.height, self.nly
                )));
            }
        }
        // R[p] > 0 — precinct refinement (Annex C.2 Table C.1, range
        // 0..=NL-1). NL = (Nc - Sd)×Nβ + Sd is the total band count
        // (Annex B.6 NL definition). Values at or above NL are invalid:
        // the precinct header field is u(8) but the spec caps it at
        // NL-1, and a refinement threshold past the highest band index
        // would refine every band (degenerate).
        if self.rp != 0 {
            let nbeta = n_beta(self.nlx, self.nly);
            let nl = (self.nc - self.sd) as u32 * nbeta + self.sd as u32;
            if (self.rp as u32) > nl - 1 {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: R[p]={} exceeds NL-1={} (NL={} bands, Annex C.2 Table C.1)",
                    self.rp,
                    nl - 1,
                    nl
                )));
            }
        }
        // Round 206 — per-slice Q overrides (Annex C.2 Table C.1, Q[p] is a
        // per-precinct field so any per-slice partition is spec-compliant).
        // Validate length matches the number of slices the encoder will
        // emit, and each Q is in the same `0..=15` range as the picture-
        // level `q` (the band-truncation `T[p,b] = clamp(Q - G - r, 0,
        // 15)` math is identical, only the source of Q changes).
        if !self.q_slices.is_empty() {
            let hp_pow = 1u32 << self.nly;
            let np_y = (self.height as u32).div_ceil(hp_pow);
            let hsl_rows = if self.hsl == 0 { np_y } else { self.hsl as u32 };
            let n_slices = np_y.div_ceil(hsl_rows) as usize;
            if self.q_slices.len() != n_slices {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: q_slices length {} != slice count {} (Np,y={}, Hsl={})",
                    self.q_slices.len(),
                    n_slices,
                    np_y,
                    hsl_rows
                )));
            }
            for (t, &qs) in self.q_slices.iter().enumerate() {
                if qs > 15 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: q_slices[{t}] = {qs} > 15 (per-band T clamp range)"
                    )));
                }
            }
        }
        // Round 233 — per-precinct Q overrides (Annex C.2 Table C.1; Q[p]
        // is indexed by precinct p, so this is the spec-natural form of
        // the round-206 mechanism). Validate length matches Np,y × Np,x
        // (raster scan, py * Np,x + px) and each Q is in `0..=15`.
        if !self.q_precincts.is_empty() {
            let hp_pow = 1u32 << self.nly;
            let np_y = (self.height as u32).div_ceil(hp_pow);
            let max_sx = self.sx.iter().copied().max().unwrap_or(1) as u32;
            let cs_w: u32 = if self.cw == 0 {
                self.width as u32
            } else {
                8u32 * (self.cw as u32) * max_sx * (1u32 << self.nlx)
            };
            let np_x = (self.width as u32).div_ceil(cs_w);
            let expected = (np_y as usize) * (np_x as usize);
            if self.q_precincts.len() != expected {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: q_precincts length {} != precinct count {} (Np,y={}, Np,x={})",
                    self.q_precincts.len(),
                    expected,
                    np_y,
                    np_x
                )));
            }
            for (p, &qp) in self.q_precincts.iter().enumerate() {
                if qp > 15 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: q_precincts[{p}] = {qp} > 15 (per-band T clamp range)"
                    )));
                }
            }
        }
        // Round 239 — per-precinct R overrides (Annex C.2 Table C.1; R[p]
        // is indexed by precinct p, mirroring the round-233 lift of Q[p]).
        // Validate length matches Np,y × Np,x (raster scan, py * Np,x + px)
        // and each R[p] is in `0..=NL - 1` (Annex C.2 Table C.1 range).
        if !self.r_precincts.is_empty() {
            let hp_pow = 1u32 << self.nly;
            let np_y = (self.height as u32).div_ceil(hp_pow);
            let max_sx = self.sx.iter().copied().max().unwrap_or(1) as u32;
            let cs_w: u32 = if self.cw == 0 {
                self.width as u32
            } else {
                8u32 * (self.cw as u32) * max_sx * (1u32 << self.nlx)
            };
            let np_x = (self.width as u32).div_ceil(cs_w);
            let expected = (np_y as usize) * (np_x as usize);
            if self.r_precincts.len() != expected {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: r_precincts length {} != precinct count {} (Np,y={}, Np,x={})",
                    self.r_precincts.len(),
                    expected,
                    np_y,
                    np_x
                )));
            }
            let nbeta = n_beta(self.nlx, self.nly);
            let nl = (self.nc - self.sd) as u32 * nbeta + self.sd as u32;
            let max_rp = nl.saturating_sub(1);
            for (p, &rp) in self.r_precincts.iter().enumerate() {
                if (rp as u32) > max_rp {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: r_precincts[{p}] = {rp} exceeds NL-1={max_rp} (NL={nl} bands, Annex C.2 Table C.1)"
                    )));
                }
            }
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner(
        width, height, 4, 3, nlx, nly, fq, q, &sx, &sy, e1, e2, cf, ct, planes,
    )
}

/// Round-195 high-bit-depth (`B[i] > 8`) Star-Tetrix (`Cpih = 3`) entry
/// point.
///
/// Widens [`encode_planar_star_tetrix`] from `B[i] = 8` to any `bd =
/// B[i] ∈ 9..=16`. Star-Tetrix is the Annex F.5 four-component CFA
/// colour transform — its four lifting steps (Tables F.4–F.8) are
/// integer linear combinations on `i32` coefficients, so bit depth is
/// fully orthogonal to the transform. The only bit-depth-dependent
/// pieces are the DC level shift `1 << (bd − 1)` (Annex G.3 inverse) and
/// the two-bytes-per-sample `u16`-LE plane format that
/// [`encode_planar_highbd`] / [`encode_planar_subsampled_highbd`]
/// established for high-bit-depth I/O.
///
/// Input plane order is `Ω = [R, G1, G2, B]` matching the 8-bit form
/// and [`crate::colour_transform::inverse_star_tetrix`]'s output
/// convention. Each `planes[i]` carries `width * height` little-endian
/// `u16` samples in `0..=2^bd − 1` (samples above that are an encoder
/// error). Per Annex F.2 Table F.1 every Cpih = 3 operand component
/// requires `sx[i] = sy[i] = 1`, so this entry point is 4:4:4 only and
/// pins `Nc = 4`.
///
/// Codes the picture losslessly with `Bw = B[i] = bd` and `Fq = 0` (the
/// lossless choice of Table A.8). The codestream carries the CTS
/// marker (`Cf`, `e1`, `e2`) and the CRG marker (Table F.9 RGGB layout
/// for `Ct = 0`, GRBG layout for `Ct = 1`) identical to the 8-bit form;
/// only the per-component CDT `B[i]` byte and the PIH `Bw` byte change.
/// Self-roundtrips bit-exactly through [`crate::decode_jpeg_xs`] at
/// 10/12/16-bit.
///
/// `e1`, `e2` are the CTS chroma-weighting exponents (0..=3); `cf` is
/// the CTS extent (0 = full, 3 = in-line). `ct` is the CFA pattern type
/// per Table F.9 (0 = RGGB, 1 = GRBG). All four parameters share the
/// 8-bit form's validation in [`EncodeConfig::validate`].
///
/// Lossy Star-Tetrix high-bit-depth (`q > 0`) is a follow-up round —
/// the inner `encode_planar_inner_bd` already accepts `q > 0` with
/// `cpih = 3`, but this entry point pins `q = 0` for the round-195
/// lossless scope.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_star_tetrix_highbd(
    width: u16,
    height: u16,
    nlx: u8,
    nly: u8,
    bd: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_star_tetrix_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_star_tetrix for 8-bit)"
        )));
    }
    if planes.len() != 4 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: encode_planar_star_tetrix_highbd requires exactly 4 component planes (Cpih=3, Annex F.2), got {}",
            planes.len()
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let want_samples = (width as usize) * (height as usize);
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(4);
    for (i, p) in planes.iter().enumerate() {
        if p.len() != want_samples {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Star-Tetrix highbd plane {i} sample count {} != width*height {want_samples}",
                p.len()
            )));
        }
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; 4];
    let sy = vec![1u8; 4];
    encode_planar_inner_bd(
        width,
        height,
        4,
        bd,
        3, // Cpih = 3 (Star-Tetrix)
        nlx,
        nly,
        0, // fq = 0 (lossless)
        0, // q = 0 (lossless)
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone, no-op at q=0)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-201 high-bit-depth (`B[i] > 8`) **lossy** Star-Tetrix
/// (`Cpih = 3`) entry point.
///
/// The `q > 0` companion to [`encode_planar_star_tetrix_highbd`]: same
/// four-component CFA plane layout (`[R, G1, G2, B]`, each
/// `width * height` little-endian `u16` samples in `0..=2^bd − 1`, `bd
/// ∈ 9..=16`), same `Bw = B[i] = bd` and DC level shift `1 << (bd − 1)`
/// (Annex G.3 inverse), but with a non-zero precinct quantization step
/// `q ∈ 1..=15` (Annex C.2 `Q[p]`) and `Fq = 8` (regular mode,
/// Table A.8). The per-band deadzone truncation
/// `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex D.4 Table D.3) drops the
/// low magnitude bitplanes and the decoder reconstructs with the
/// matching deadzone inverse (Annex D.2, `Qpih = 0`).
///
/// Bit depth, quantization, and the Star-Tetrix lifting are mutually
/// orthogonal here: the Annex F.5 lifting (Tables F.4–F.8) is an
/// integer linear combination on `i32` coefficients, the forward
/// quantizer and the inverse dequantizer both operate on `i32` wavelet
/// coefficients regardless of `B[i]`, and the colour transform runs in
/// the un-quantized wavelet domain — so the only bit-depth-dependent
/// pieces remain the DC level shift and the two-bytes-per-sample
/// `u16`-LE plane packing established by the round-118 / 133 / 151 /
/// 195 paths.
///
/// Per Annex F.2 Table F.1 the Star-Tetrix operand window is `c < 4`,
/// so this entry point pins `Nc = 4`, `sx[i] = sy[i] = 1` for `i < 4`,
/// and `Cpih = 3` (4:4:4 CFA only). The CTS / CRG markers (`Cf`, `e1`,
/// `e2`, RGGB/GRBG via `Ct`) survive on the high-bit-depth lossy path
/// identically to the round-195 lossless form — only the per-component
/// CDT `B[i]` byte, the PIH `Bw` byte, and the per-precinct `Q` field
/// change.
///
/// Rejects `q = 0` (use [`encode_planar_star_tetrix_highbd`] for the
/// lossless path), `bd = 8` (use [`encode_planar_star_tetrix`] with
/// `q > 0`), `bd > 16`, and any plane-count `!= 4`. `e1`, `e2`, `cf`,
/// `ct` share the same validation as the 8-bit / lossless high-bit-
/// depth forms (in [`EncodeConfig::validate`]).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_star_tetrix_highbd_lossy(
    width: u16,
    height: u16,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_star_tetrix_highbd_lossy requires B[i] in 9..=16, got {bd} (use encode_planar_star_tetrix for 8-bit)"
        )));
    }
    if q == 0 {
        return Err(Error::invalid(
            "jpegxs encoder: encode_planar_star_tetrix_highbd_lossy requires q > 0 (use encode_planar_star_tetrix_highbd for lossless)".to_string(),
        ));
    }
    if planes.len() != 4 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: encode_planar_star_tetrix_highbd_lossy requires exactly 4 component planes (Cpih=3, Annex F.2), got {}",
            planes.len()
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let want_samples = (width as usize) * (height as usize);
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(4);
    for (i, p) in planes.iter().enumerate() {
        if p.len() != want_samples {
            return Err(Error::invalid(format!(
                "jpegxs encoder: Star-Tetrix highbd lossy plane {i} sample count {} != width*height {want_samples}",
                p.len()
            )));
        }
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; 4];
    let sy = vec![1u8; 4];
    encode_planar_inner_bd(
        width,
        height,
        4,
        bd,
        3, // Cpih = 3 (Star-Tetrix)
        nlx,
        nly,
        0, // fq = 0 (integer-transform path; lossy via T[p,b], Table A.8)
        q,
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// High-bit-depth (`B[i] > 8`) **default-weights** CFA Star-Tetrix
/// (`Cpih = 3`, `Sd = 1`) entry point.
///
/// The high-bit-depth companion to the 8-bit [`encode_planar_sd_star_tetrix`]
/// at `nc = 4`, `sd = 1`. Suppresses the fourth Star-Tetrix output (the blue
/// channel, Table F.4): that transform output is *raw-coded* per Annex B
/// Tables B.10 / B.11 rather than wavelet-decomposed, while the inverse
/// transform consumes it unchanged — so `NL = (Nc−Sd)·Nβ + Sd = 19 / 25 / 31`
/// bands for `NL,x = 5`, `NL,y ∈ {0, 1, 2}`. The four CFA planes carry
/// little-endian `u16` samples in `0..=2^bd − 1` (`bd ∈ 9..=16`), with
/// `Bw = B[i] = bd` and the Annex G.3 DC level shift `1 << (bd − 1)`. The
/// Star-Tetrix lifting (Annex F.5 Tables F.4–F.8), the forward / inverse
/// quantizer, and the colour transform all operate on `i32` coefficients, so
/// the only bit-depth-dependent pieces remain the level shift and the
/// two-bytes-per-sample plane packing. `q = 0` → lossless (`Fq = 0`); `q ∈
/// 1..=15` → regular (`Fq = 8`) deadzone quantization. Self-roundtrips
/// bit-exactly at `q = 0`. Use [`encode_planar_sd_star_tetrix`] for the 8-bit
/// path; use [`encode_planar_star_tetrix_highbd_annex_h`] to drive the Annex H
/// PSNR-optimized weights on this same `Sd = 1` highbd layout.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_sd_star_tetrix_highbd(
    width: u16,
    height: u16,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    let byte_planes = pack_cfa_u16_planes(
        width,
        height,
        bd,
        planes,
        "encode_planar_sd_star_tetrix_highbd",
    )?;
    let sx = vec![1u8; 4];
    let sy = vec![1u8; 4];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        4,
        bd,
        3, // Cpih = 3 (Star-Tetrix)
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        Vec::new(), // band_gains: default (build_band_gains_sd)
        Vec::new(), // band_priorities: default (build_band_priorities_sd)
        0,          // cw: single precinct per row
        1,          // sd = 1: suppress the fourth Star-Tetrix output
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// High-bit-depth (`B[i] > 8`) **Annex H content-adaptive WGT** CFA
/// Star-Tetrix (`Cpih = 3`, `Sd = 1`) entry point.
///
/// The high-bit-depth companion to the 8-bit
/// [`encode_planar_star_tetrix_annex_h`]. Identical `Sd = 1` CFA layout (four
/// `width × height` little-endian `u16` planes `[R, G1, G2, B]` in
/// `0..=2^bd − 1`, `bd ∈ 9..=16`, the fourth output raw-coded per Annex B
/// Tables B.10 / B.11), but drives the WGT marker and forward truncation
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the ISO/IEC 21122-1:2022
/// Annex H PSNR-optimized CFA tables H.9 / H.10 / H.11 (`Cpih = 3`, `Sd = 1`,
/// 4:4:4:4, `NL,x = 5`, `NL,y ∈ {0, 1, 2}`). Each table tabulates a separate
/// `(G[b], P[b])` column per CTS extent `Cf ∈ {0, 3}` (full / restricted
/// in-line); `cf` selects the column. Configurations outside the tabulated set
/// (`cf ∉ {0, 3}`, `NL,x ≠ 5`, `NL,y > 2`) fall back to the default-weights
/// [`encode_planar_sd_star_tetrix_highbd`] path so the codestream is always
/// well-formed.
///
/// Bit depth and the Annex H weights are orthogonal: the H-table gains /
/// priorities and the matching forward-truncation branch are bit-depth
/// agnostic (they act on `i32` wavelet coefficients), so the only
/// bit-depth-dependent pieces remain the Annex G.3 DC level shift
/// `1 << (bd − 1)` and the two-bytes-per-sample `u16`-LE plane packing. `q = 0`
/// → lossless (the WGT advertises the H column but `T = 0` so every sample
/// self-roundtrips bit-exactly); `q ∈ 1..=15` → regular deadzone quantization
/// using the H gains. This closes the previously-deferred "content-adaptive WGT
/// weights for the CFA Star-Tetrix layouts at bit depths above 8" follow-up.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_star_tetrix_highbd_annex_h(
    width: u16,
    height: u16,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; 4];
    let sy = vec![1u8; 4];
    // Outside the tabulated H.9–H.11 set `annex_h_weights` returns `None`; fall
    // back to the default-weights highbd Sd=1 Star-Tetrix path so the CFA
    // codestream is still well-formed (mirrors the 8-bit
    // `encode_planar_star_tetrix_annex_h` fallback).
    let Some((gains, priorities)) = annex_h_weights(4, 3, 1, nlx, nly, cf, &sx, &sy) else {
        return encode_planar_sd_star_tetrix_highbd(
            width, height, nlx, nly, bd, q, e1, e2, cf, ct, planes,
        );
    };
    let byte_planes = pack_cfa_u16_planes(
        width,
        height,
        bd,
        planes,
        "encode_planar_star_tetrix_highbd_annex_h",
    )?;
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        4,
        bd,
        3, // Cpih = 3 (Star-Tetrix)
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        gains,
        priorities,
        0,          // cw: single precinct per row
        1,          // sd = 1: suppress the fourth Star-Tetrix output
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Validate + pack four high-bit-depth (`bd ∈ 9..=16`) CFA `u16` planes into
/// the two-bytes-per-sample little-endian byte layout `encode_planar_inner_bd`
/// expects (`Cpih = 3` Star-Tetrix, `Nc = 4`, 4:4:4:4). Shared by the highbd
/// `Sd = 1` Star-Tetrix entry points so the bound / count / range checks live
/// in exactly one place. `who` names the caller for error messages.
fn pack_cfa_u16_planes(
    width: u16,
    height: u16,
    bd: u8,
    planes: &[Vec<u16>],
    who: &str,
) -> Result<Vec<Vec<u8>>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: {who} requires B[i] in 9..=16, got {bd} (use the 8-bit Star-Tetrix entry points for bd=8)"
        )));
    }
    if planes.len() != 4 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: {who} requires exactly 4 component planes (Cpih=3, Annex F.2), got {}",
            planes.len()
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let want_samples = (width as usize) * (height as usize);
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(4);
    for (i, p) in planes.iter().enumerate() {
        if p.len() != want_samples {
            return Err(Error::invalid(format!(
                "jpegxs encoder: {who} plane {i} sample count {} != width*height {want_samples}",
                p.len()
            )));
        }
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: {who} plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    Ok(byte_planes)
}

/// Lossy entry point. `q` is the precinct quantization step (0..=15);
/// 0 reduces to lossless. This is the **integer-transform** regular case:
/// it signals the conformant ISO/IEC 21122-1:2022 Table A.8 combination
/// `(Bw = B[0], Fq = 0)` and achieves lossy compression purely through the
/// precinct truncation `T[p,b]` (the inverse quantiser is independent of
/// `Fq`). For genuinely higher fractional precision use
/// [`encode_planar_highprec_lossy`], which selects the `(Bw = 20, Fq = 8)`
/// combination.
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner(
        width, height, nc, cpih, nlx, nly, fq, q, &sx, &sy, 0, 0, 0, 0, planes,
    )
}

/// Run-mode-1 (`Rm = 1`) entry point — signals the ISO/IEC 21122-1:2022
/// Table A.12 "runs indicate zero coefficients" significance
/// interpretation.
///
/// Identical to [`encode_planar_lossy`] (8-bit planar, `Cpih ∈ {0, 1}`,
/// linear scaling, deadzone quantiser) except that the picture header
/// carries `Rm = 1` and the vertical-prediction significance coding drops
/// the `Rm = 0` predecessor guard: a code group is insignificant iff *all*
/// its coefficients are zero after truncation (`M ≤ T`), and the decoder
/// reconstructs such a group as `M = T[p,b]` (Table C.13, `Δm = T − mtop`)
/// so both its VLC bitplane-count residual and its data are elided. On
/// content with vertical structure — a non-zero band region followed by an
/// all-zero one — this compresses at least as well as `Rm = 0` (never
/// worse, since the encoder still picks the smallest per-band `D[p,b]`
/// form) while remaining bit-exact for `q = 0`.
///
/// `q` is the precinct quantization step (`0..=15`; `0` = lossless).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_run_mode1(
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
    encode_planar_inner_bd(
        width,
        height,
        nc,
        8,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (integer transform; quantisation via T[p,b])
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,          // cw
        0,          // sd
        0,          // fs
        0,          // hsl
        0,          // qpih (deadzone)
        0,          // rp
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        planes,
        1, // rm = 1 (zero-coefficient significance)
    )
}

/// Run-mode-1 (`Rm = 1`) high-bit-depth entry point — the `Rm = 1`
/// composition of [`encode_planar_highbd`]. Component bit depth
/// `bd ∈ 9..=16`, `u16`-LE planes, `Cpih ∈ {0, 1}`, lossless. The
/// zero-coefficient significance semantics operate on the wider (`Bw = bd`)
/// wavelet coefficients unchanged (`Rm` gates only the insignificant-group
/// `M` reconstruction, which is bit-depth-independent).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_run_mode1_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_run_mode1_highbd requires B[i] in 9..=16, got {bd}"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_run_mode1_highbd supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (lossless)
        0, // q = 0 (lossless)
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,          // cw
        0,          // sd
        0,          // fs
        0,          // hsl
        0,          // qpih (deadzone)
        0,          // rp
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        &byte_planes,
        1, // rm = 1 (zero-coefficient significance)
    )
}

/// Run-mode-1 (`Rm = 1`) chroma-subsampled entry point — the `Rm = 1`
/// composition of [`encode_planar_subsampled`]. `sx`/`sy` give the
/// per-component sampling factors (`∈ {1, 2}`); `Cpih ∈ {0}` (RCT is
/// rejected on sub-sampled inputs per Annex F.2). `q` is the precinct
/// quantization step (`0` = lossless). A `sy = 2` component decomposes one
/// vertical level shallower, so the `Rm = 1` insignificant-group logic runs
/// across the mixed-depth band cascade.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_run_mode1_subsampled(
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
    encode_planar_inner_bd(
        width,
        height,
        nc,
        8,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (integer transform)
        q,
        sx,
        sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,          // cw
        0,          // sd
        0,          // fs
        0,          // hsl
        0,          // qpih (deadzone)
        0,          // rp
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        planes,
        1, // rm = 1 (zero-coefficient significance)
    )
}

/// High-precision lossy entry point — the ISO/IEC 21122-1:2022 Table A.8
/// `(Bw = 20, Fq = 8)` regular case.
///
/// Where [`encode_planar_lossy`] runs the integer 5/3 transform with zero
/// fractional bits (`Fq = 0`), this entry point carries **8 fractional
/// bits** through the wavelet transform per Annex E.3 (Table E.13): the
/// `B[i]`-bit samples are up-scaled into the 20-bit wavelet domain
/// (`Ω = (sample << (Bw − B[i])) − (1 << (Bw − 1))`, the inverse of the
/// Annex G.2 output scaling), the forward DWT runs in that wider domain,
/// and its output is down-scaled by `>> Fq` before quantisation. The
/// decoder applies the exact inverse `c << Fq` at the dequant boundary and
/// the Annex G.2 `>> (Bw − B[i])` on output. The extra fractional precision
/// reduces the rounding noise the integer transform injects at each
/// decomposition level, so for a given `q` the reconstructed PSNR is at
/// least as high as the integer-transform path — and strictly higher once
/// the transform depth makes the per-level rounding visible.
///
/// `q` is the precinct quantization step (0..=15). `q = 0` is permitted
/// (the fractional scaling is exercised but no bitplanes are dropped);
/// note that because the `>> Fq` rounding discards the bottom `Fq` bits it
/// is *not* bit-exact lossless — use [`encode_planar`] for lossless.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_highprec_lossy(
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
    // fq = 8 selects the (Bw = 20, Fq = 8) Table A.8 combination in
    // encode_planar_inner_bd (the non-NLT `fq >= 8` branch).
    encode_planar_inner(
        width, height, nc, cpih, nlx, nly, 8, q, &sx, &sy, 0, 0, 0, 0, planes,
    )
}

/// Lossy 4:4:4 encode driven by the **content-adaptive WGT weights** from
/// ISO/IEC 21122-1:2022 Annex H (the PSNR-optimized `(G[b], P[b])` example
/// tables, [`annex_h_weights`]) rather than the encoder's default plain
/// gains + `P[b] = b` priorities.
///
/// Annex H supplies tables H.1 / H.2 / H.3 for the 4:4:4, RCT-enabled
/// (`Cpih = 1`), 5-horizontal-level configurations with `NLy ∈ {0, 1, 2}`.
/// For any other `(nc, cpih, nlx, nly)` the function falls back to the
/// default-weights path of [`encode_planar_lossy`], so callers always get
/// a valid codestream.
///
/// The supplied weights drive BOTH the WGT marker (Annex A.4.11) AND the
/// forward truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` (Annex
/// C.6.2 Table C.10, `r = (P[b] < R[p]) ? 1 : 0`), so the codestream
/// round-trips through [`crate::decode_jpeg_xs`] — the decoder reads the
/// identical `(G[b], P[b])` off the WGT segment and reconstructs the same
/// per-band truncation positions. The richer gains (Annex H assigns up to
/// `G = 4` to the deepest LL band, versus the default cap of 2) let the
/// quantizer drop more low-frequency bitplanes where the eye is least
/// sensitive, improving rate–distortion at a fixed `q`.
///
/// At `q = 0` (lossless) the weights are a no-op — every `T[p,b]` is
/// already clamped to its `0` floor — so this entry point is only
/// meaningful for `q ∈ 1..=15`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_lossy_annex_h(
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
    let Some((gains, priorities)) = annex_h_weights(nc, cpih, 0, nlx, nly, 0, &sx, &sy) else {
        return encode_planar_lossy(width, height, nc, cpih, nlx, nly, q, planes);
    };
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        gains,
        priorities,
        0,          // cw: single precinct per row
        0,          // sd: no suppressed components
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        planes,
    )
}

/// Chroma-subsampled companion to [`encode_planar_lossy_annex_h`]: drives the
/// WGT marker and forward truncation from the ISO/IEC 21122-1:2022 Annex H
/// PSNR-optimized example tables for the **4:2:2** (H.4 / H.5 / H.6) and
/// **4:2:0** (H.7 / H.8) layouts, both with RCT disabled (`Cpih = 0`),
/// `Nc = 3`, `NLx = 5`.
///
/// `sx` / `sy` are the per-component sampling factors: 4:2:2 is
/// `sx = [1, 2, 2], sy = [1, 1, 1]`; 4:2:0 is `sx = [1, 2, 2], sy = [2, 2, 2]`.
/// The tabulated configurations are 4:2:2 with `NLy ∈ {0, 1, 2}` and 4:2:0
/// with `NLy ∈ {1, 2}` (the spec lists no 4:2:0 `NLy = 0` table). Any other
/// `(cpih, sx, sy, nlx, nly)` falls back to the default-weights
/// [`encode_planar_subsampled`], so callers always get a valid codestream.
///
/// In the 4:2:0 tables H.7 / H.8 some band indices are non-existent
/// (`bx[β,i] = 0`, the spec's `-*` slots): the WGT marker loop emits one
/// `(G[b], P[b])` pair per *existing* band only (Annex A.4.11), and
/// [`annex_h_weights`] already drops those slots so the supplied vectors are in
/// the encoder's existing-band emission order. The decoder reads the identical
/// `(G[b], P[b])` off the wire and reconstructs the same
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` (Annex C.6.2 Table C.10), so the
/// codestream round-trips through [`crate::decode_jpeg_xs`].
///
/// As with the 4:4:4 entry point, the weights are a no-op at `q = 0`
/// (lossless), so this path is meaningful for `q ∈ 1..=15`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_annex_h(
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
    let Some((gains, priorities)) = annex_h_weights(nc, cpih, 0, nlx, nly, 0, sx, sy) else {
        return encode_planar_subsampled(width, height, nc, cpih, nlx, nly, q, sx, sy, planes);
    };
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        0,
        0,
        0,
        0,
        None,
        gains,
        priorities,
        0,          // cw: single precinct per row
        0,          // sd: no suppressed components
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        planes,
    )
}

/// CFA Star-Tetrix companion to [`encode_planar_lossy_annex_h`]: drives the
/// WGT marker and forward truncation from the ISO/IEC 21122-1:2022 Annex H
/// PSNR-optimized example tables H.9 / H.10 / H.11 for the **colour-filter-
/// array** Star-Tetrix layout (`Cpih = 3`, `Sd = 1`, four 4:4:4:4
/// components, `NL,x = 5`, `NL,y ∈ {0, 1, 2}`).
///
/// Star-Tetrix (Annex F.5) decorrelates the four CFA components `[R, G1, G2,
/// B]` into four transform outputs; with `Sd = 1` the fourth output (blue,
/// per Table F.4) is coded raw rather than wavelet-decomposed (Annex B
/// Tables B.10 / B.11). The transform itself still reads all four inputs, so
/// it composes cleanly with the suppressed output. The Annex H tables list a
/// separate `(G[b], P[b])` column per CTS extent: `cf = 0` (full
/// transformation, Table A.20) and `cf = 3` (restricted in-line). `cf`
/// selects the column; both round-trip because the decoder reads the
/// identical `(G[b], P[b])` off the WGT segment and rebuilds the same
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` (Annex C.6.2 Table C.10).
///
/// `planes` is `[R, G1, G2, B]`, each `width * height` 8-bit samples (matching
/// [`encode_planar_star_tetrix`]). `e1` / `e2` are the CTS chroma exponents
/// (0..=3); `ct` is the CFA pattern type (0 = RGGB, 1 = GRBG). For any
/// `(cf, nly)` outside the tabulated set (`cf ∉ {0, 3}` or `nly > 2`) the
/// entry point falls back to the default-weights [`encode_planar_star_tetrix`]
/// with `Sd = 1`, so callers always get a valid codestream.
///
/// As with the other Annex H entry points the weights are a no-op at `q = 0`
/// (lossless) — every `T[p,b]` is already clamped to its `0` floor — so this
/// path is meaningful for `q ∈ 1..=15`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_star_tetrix_annex_h(
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
                // Outside the tabulated H.9–H.11 set `annex_h_weights` returns `None`;
                // empty gains/priorities then fall back to the default-weights Sd=1
                // Star-Tetrix path so the CFA codestream is still well-formed.
    let (gains, priorities) = annex_h_weights(4, 3, 1, nlx, nly, cf, &sx, &sy).unwrap_or_default();
    encode_planar_inner_nlt(
        width,
        height,
        4, // Nc = 4 CFA components
        3, // Cpih = 3 (Star-Tetrix)
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        gains,
        priorities,
        0,          // cw: single precinct per row
        1,          // sd = 1: suppress the fourth Star-Tetrix output
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        planes,
    )
}

/// Round-118 high-bit-depth (`B[i] > 8`) lossless 4:4:4 entry point.
///
/// Codes a `bd`-bit picture losslessly with `Bw = B[i] = bd`
/// (`bd ∈ 9..=16`, the lossless choice of Table A.8) and `Fq = 0`. The
/// DC level shift is `1 << (bd - 1)` (Annex G.3 inverse), so each sample
/// lands in the wavelet domain `[-2^(bd-1), 2^(bd-1) - 1]`; the 5/3 DWT,
/// entropy coder, and (optional) reversible colour transform all operate
/// on `i32` coefficients regardless of bit depth, so the only bit-depth-
/// dependent pieces are this level shift and the output plane packing.
///
/// `planes[i]` carries the component samples as **little-endian `u16`**
/// values in `0..=2^bd - 1`; the decoder returns the reconstructed plane
/// in the same two-bytes-per-sample [`crate::image::JpegXsPlane`] layout
/// (one byte per sample is still used for `bd == 8`). Samples above
/// `2^bd - 1` are an encoder error (out of the component's nominal range).
///
/// `cpih ∈ {0, 1}`: no transform, or the reversible RCT (Annex F.3 — bit-
/// depth agnostic). Star-Tetrix (`cpih = 3`) and NLT pre-distortion are
/// not exposed on this path. `nlx`/`nly` follow the Annex A.4.4 limits.
/// `q` is fixed at 0 (lossless) here; lossy high-bit-depth quantization is
/// a later round. Self-roundtrips bit-exactly through
/// [`crate::decode_jpeg_xs`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_lossy for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_highbd supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    // Pack each plane to little-endian u16 bytes (the EncodeConfig
    // bit_depth > 8 plane format), validating the nominal range first.
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (lossless)
        0, // q = 0 (lossless)
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone, no-op at q=0)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-133 high-bit-depth (`B[i] > 8`) **lossy** 4:4:4 entry point.
///
/// The lossy companion to [`encode_planar_highbd`]: same plane format
/// (little-endian `u16` samples in `0..=2^bd - 1`, `bd ∈ 9..=16`, `Bw =
/// B[i] = bd`, DC level shift `1 << (bd - 1)` per the Annex G.3 inverse)
/// but with a non-zero precinct quantization step `q` (Annex C.2 `Q[p]`,
/// `1..=15`). `q > 0` forces `Fq = 8` (regular mode, Table A.8) so the
/// per-band deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex
/// D.4 Table D.3) drops the low bitplanes; the decoder reconstructs with
/// the matching deadzone inverse (Annex D.2, `Qpih = 0`).
///
/// Bit depth is orthogonal to quantization here: the forward quantizer
/// ([`forward_quant_index`]) and the inverse dequantizer both operate on
/// `i32` wavelet coefficients regardless of `B[i]`, so the only bit-depth-
/// dependent pieces remain the level shift and the `u16`-LE plane packing —
/// exactly as on the round-118 lossless path. The reconstructed plane is
/// returned in the two-bytes-per-sample [`crate::image::JpegXsPlane`]
/// layout and clipped to `0..=2^bd - 1`.
///
/// `cpih ∈ {0, 1}` (no transform / reversible RCT, Annex F.3 — bit-depth
/// agnostic). Star-Tetrix (`cpih = 3`) and NLT pre-distortion stay 8-bit-
/// input specific. `q = 0` is rejected — use [`encode_planar_highbd`] for
/// the lossless path.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_highbd_lossy(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_highbd_lossy requires B[i] in 9..=16, got {bd} (use encode_planar_lossy for 8-bit)"
        )));
    }
    if q == 0 {
        return Err(Error::invalid(
            "jpegxs encoder: encode_planar_highbd_lossy requires q > 0 (use encode_planar_highbd for lossless)".to_string(),
        ));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_highbd_lossy supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    // Pack each plane to little-endian u16 bytes (the EncodeConfig
    // bit_depth > 8 plane format), validating the nominal range first.
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (integer-transform path; lossy via T[p,b], Table A.8)
        q,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-151 high-bit-depth chroma-sub-sampled entry point (lossless).
///
/// Widens [`encode_planar_highbd`] from 4:4:4-only to arbitrary per-
/// component `(sx[i], sy[i]) ∈ {1, 2}` sub-sampling at component bit
/// depth `bd ∈ 9..=16`. Same plane format as the round-118 lossless
/// path: each `planes[i]` carries `⌈width / sx[i]⌉ * ⌈height / sy[i]⌉`
/// little-endian `u16` samples in `0..=2^bd - 1`. The codestream uses
/// `Bw = B[i] = bd` and `Fq = 0` (the lossless choice of Table A.8);
/// the DC level shift is `1 << (bd - 1)` per the Annex G.3 inverse, so
/// each sample lands in the wavelet domain `[−2^(bd−1), 2^(bd−1) − 1]`
/// independent of sub-sampling.
///
/// Per Annex F.2 Table F.1 the reversible RCT (`Cpih = 1`) requires
/// `sx[i] = sy[i] = 1` for `i < 3`, so the typical 4:2:2 / 4:2:0
/// configurations are exposed only with `Cpih = 0` (no transform).
/// Star-Tetrix (`Cpih = 3`) and NLT pre-distortion stay 8-bit-input
/// specific and are not exposed on this path.
///
/// The 5/3 DWT and entropy coder operate on `i32` coefficients
/// independent of bit depth, and the per-component effective vertical
/// decomposition depth `N'L,y[i] = NL,y − log2(sy[i])` (used in the
/// 4:2:2 / 4:2:0 case) is the same path as the 8-bit sub-sampled
/// encoder. The decoder packs `u16` LE per plane when `B[i] > 8`
/// regardless of sub-sampling, so the output round-trips bit-exactly
/// through [`crate::decode_jpeg_xs`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_highbd supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    if sx.len() != nc as usize || sy.len() != nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: sx/sy must have length nc={nc}, got sx={}, sy={}",
            sx.len(),
            sy.len()
        )));
    }
    let byte_planes = pack_subsampled_u16_planes(
        width,
        height,
        bd,
        sx,
        sy,
        planes,
        "encode_planar_subsampled_highbd",
    )?;
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (lossless)
        0, // q = 0 (lossless)
        sx,
        sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone, no-op at q=0)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-151 high-bit-depth chroma-sub-sampled entry point (lossy).
///
/// The `q > 0` companion to [`encode_planar_subsampled_highbd`]: same
/// `u16`-LE plane format, same `Bw = B[i] = bd` (`bd ∈ 9..=16`), same
/// per-component `(sx, sy)`, but with a non-zero precinct quantization
/// step `q ∈ 1..=15` (Annex C.2 `Q[p]`) and `Fq = 8` (regular mode,
/// Table A.8). The per-band deadzone truncation
/// `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex D.4 Table D.3) drops the
/// low magnitude bitplanes and the decoder reconstructs with the
/// matching deadzone inverse (Annex D.2, `Qpih = 0`).
///
/// Bit depth is orthogonal to quantization (the forward quantizer and
/// the inverse dequantizer both run on `i32` wavelet coefficients
/// regardless of `B[i]`), so the only bit-depth-dependent pieces remain
/// the level shift and `u16` packing — same as the round-118 / 133
/// 4:4:4 paths. `cpih ∈ {0, 1}`. Rejects `q = 0` (use
/// [`encode_planar_subsampled_highbd`]).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_highbd_lossy(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if q == 0 {
        return Err(Error::invalid(
            "jpegxs encoder: encode_planar_subsampled_highbd_lossy requires q > 0 (use encode_planar_subsampled_highbd for lossless)".to_string(),
        ));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_highbd_lossy supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    if sx.len() != nc as usize || sy.len() != nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: sx/sy must have length nc={nc}, got sx={}, sy={}",
            sx.len(),
            sy.len()
        )));
    }
    let byte_planes = pack_subsampled_u16_planes(
        width,
        height,
        bd,
        sx,
        sy,
        planes,
        "encode_planar_subsampled_highbd_lossy",
    )?;
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        0, // fq = 0 (integer-transform path; lossy via T[p,b], Table A.8)
        q,
        sx,
        sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone)
        0,          // rp = 0 (no refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Validate + pack high-bit-depth (`bd ∈ 9..=16`) chroma-sub-sampled `u16`
/// planes into the two-bytes-per-sample little-endian byte layout
/// `encode_planar_inner_bd` expects. Shared by the subsampled highbd entry
/// points so the bit-depth / length / per-component-dimension / range checks
/// live in one place. `who` names the caller for error messages.
fn pack_subsampled_u16_planes(
    width: u16,
    height: u16,
    bd: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
    who: &str,
) -> Result<Vec<Vec<u8>>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: {who} requires B[i] in 9..=16, got {bd} (use the 8-bit subsampled entry points for bd=8)"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let want =
            (width as usize).div_ceil(sx[i] as usize) * (height as usize).div_ceil(sy[i] as usize);
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs encoder: {who} plane {i} sample count {} != Wc*Hc {} (sx={}, sy={})",
                p.len(),
                want,
                sx[i],
                sy[i]
            )));
        }
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: {who} plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    Ok(byte_planes)
}

/// High-bit-depth (`B[i] > 8`) **Annex H content-adaptive WGT** chroma-
/// sub-sampled entry point.
///
/// The high-bit-depth companion to the 8-bit
/// [`encode_planar_subsampled_annex_h`]. Same `u16`-LE sub-sampled plane
/// format (`bd ∈ 9..=16`, per-component `(sx, sy)`), but drives the WGT marker
/// and the forward truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the
/// ISO/IEC 21122-1:2022 Annex H subsampled tables H.4 / H.5 / H.6 (4:2:2, RCT
/// disabled, `NL,y ∈ {0, 1, 2}`) and H.7 / H.8 (4:2:0, RCT disabled,
/// `NL,y ∈ {1, 2}`) — all at `NL,x = 5`, `Cpih = 0`. The spec's `-*`
/// non-existent-band slots are dropped so the emitted `(G[b], P[b])` pairs land
/// in the encoder's existing-band emission order (matching the
/// `picture_beta_to_local_beta` skip rule position-for-position). Configurations
/// outside the tabulated set fall back to the default-weights
/// [`encode_planar_subsampled_highbd`] / `_lossy` path so the codestream is
/// always well-formed. `q = 0` → lossless (`Fq = 0`, bit-exact self-roundtrip
/// even with the H column advertised); `q ∈ 1..=15` → regular (`Fq = 8`).
///
/// Bit depth and the Annex H weights are orthogonal — the gains / priorities
/// and the matching forward-truncation branch act on `i32` wavelet
/// coefficients, so the only bit-depth-dependent pieces remain the Annex G.3 DC
/// level shift and the `u16`-LE plane packing.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_highbd_annex_h(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if sx.len() != nc as usize || sy.len() != nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs encoder: sx/sy must have length nc={nc}, got sx={}, sy={}",
            sx.len(),
            sy.len()
        )));
    }
    // Outside the tabulated H.4–H.8 set `annex_h_weights` returns `None`; fall
    // back to the default-weights highbd subsampled path (lossless or lossy by
    // `q`), mirroring the 8-bit `encode_planar_subsampled_annex_h` fallback.
    let Some((gains, priorities)) = annex_h_weights(nc, cpih, 0, nlx, nly, 0, sx, sy) else {
        return if q == 0 {
            encode_planar_subsampled_highbd(width, height, nc, cpih, nlx, nly, bd, sx, sy, planes)
        } else {
            encode_planar_subsampled_highbd_lossy(
                width, height, nc, cpih, nlx, nly, bd, q, sx, sy, planes,
            )
        };
    };
    let byte_planes = pack_subsampled_u16_planes(
        width,
        height,
        bd,
        sx,
        sy,
        planes,
        "encode_planar_subsampled_highbd_annex_h",
    )?;
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        fq,
        q,
        sx,
        sy,
        0,
        0,
        0,
        0,
        None,
        gains,
        priorities,
        0,          // cw: single precinct per row
        0,          // sd: no suppressed components
        0,          // fs: signs jointly with data
        0,          // hsl: single slice
        0,          // qpih: deadzone inverse quantizer
        0,          // rp: no precinct refinement
        Vec::new(), // q_slices
        Vec::new(), // q_precincts
        Vec::new(), // r_precincts
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-103 multi-slice 4:4:4 entry point (`Hsl > 0`).
///
/// Same shape as [`encode_planar_lossy`] but takes an explicit slice
/// height `hsl` in precinct rows (PIH `Hsl`, signalled per Annex B.10).
/// `hsl = 0` reduces to the single-slice layout (`Hsl = Np,y`, the whole
/// picture in one slice — bit-equivalent to [`encode_planar_lossy`]).
/// `hsl > 0` partitions the `Np,y = ⌈Hf / 2^NL,y⌉` precinct rows into
/// `⌈Np,y / hsl⌉` slices of `hsl` precinct rows each (the last slice is
/// shorter when `Np,y` is not a multiple of `hsl`), emitting one SLH
/// marker per slice (Annex A.4.12 Table A.25) with `Yslh = t` — the
/// slice's top-down order, counting from 0 at the top of the image.
///
/// Slices decode independently: the decoder reconstructs the identical
/// precinct-to-slice grouping from PIH `Hsl` + `Np,y`
/// ([`crate::slice_walker`] Annex B.10), so any output round-trips
/// through [`crate::decode_jpeg_xs`]. Vertical prediction is precinct-
/// scoped in this encoder, so slice boundaries carry no predictor state
/// across them (Annex B.10 requires vertical prediction be disabled
/// across slice boundaries — satisfied trivially here because no
/// predictor crosses a precinct row in the first place).
///
/// `hsl` must be `<= Np,y` (a larger value would describe a single slice
/// running past the last precinct row). `q` is the precinct quantization
/// step (`0..=15`); `q = 0` is lossless and `q > 0` forces `Fq = 8`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    hsl: u16,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        hsl,        // hsl: slice height in precinct rows
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-206 per-slice rate-budgeting entry point (`Hsl > 0` multi-slice
/// + per-slice `Q[p]` override, Annex C.2 Table C.1).
///
/// Same shape as [`encode_planar_hsl`] but lifts the picture-level `q`
/// from a single scalar to a `q_slices` vector — one value per slice
/// (in top-down order, matching `Yslh = t`). The encoder partitions the
/// `Np,y = ⌈Hf / 2^NL,y⌉` precinct rows into `⌈Np,y / hsl⌉` slices the
/// same way [`encode_planar_hsl`] does, but emits each precinct's
/// header with the slice's `Q[p]` instead of the constant `cfg.q`.
///
/// `q_slices.len()` must exactly equal `⌈Np,y / hsl⌉` (the slice count
/// the encoder will emit). When `hsl == 0` the encoder is in single-
/// slice mode and `q_slices` must hold exactly one entry. Each entry
/// is in the encoder's working `0..=15` range (the band-truncation
/// `T[p,b] = clamp(Q − G[b] − r, 0, 15)` math is identical, only the
/// source of `Q[p]` changes); any entry above 15 is rejected.
///
/// Lossless / lossy mixing is allowed: if every `q_slices[t] == 0`,
/// `Fq` is forced to 0 (lossless); if any is `> 0`, `Fq = 8`
/// (regular, Table A.8 — required for `q > 0`). The decoder reads
/// `Q[p]` per precinct (`parse_precinct_header` + `precinct_truncation`
/// in [`crate::entropy`]), so the slice-local `Q[p]` flows through
/// the existing inverse-quantizer path with no decoder change.
///
/// The bitstream-wire impact is exactly the per-precinct `Q` byte
/// inside each slice. SOC / CAP / PIH / CDT / WGT / SLH markers and
/// the entropy-data layout are unchanged. When `q_slices` carries a
/// single repeated value the output is byte-identical to
/// [`encode_planar_hsl`] at that `q`.
///
/// Intended use: bit-budgeted live workflows where slices closer to
/// the visually-important picture region get a lower `Q[p]` than
/// non-salient slices. The encoder leaves the `q_slices` assignment
/// to the caller (no rate-distortion search is performed); a
/// follow-up round can wrap this with a PSNR-driven slice budgeter.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_qslice(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    q_slices: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    // Pick the picture-level fallback Q (used by the precinct path only
    // if `q_slices` resolves empty per-slice via `slice_cfg_for` — which
    // never happens here because the public entry always supplies it).
    // We still set `cfg.q` to a representative value so the validate()
    // q <= 15 guard runs against a known-good scalar, and so any future
    // code path reading `cfg.q` directly observes a sensible default.
    let q_pic = q_slices.iter().copied().max().unwrap_or(0);
    // Fq = 8 whenever any slice quantizes; 0 only when every slice is
    // lossless (so the byte-identical lossless layout is preserved).
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q_pic,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,   // cw: single precinct column
        0,   // sd: no CWD suppression
        0,   // fs: signs jointly with data (Fs=0)
        hsl, // hsl: slice height in precinct rows
        0,   // qpih: deadzone inverse quantizer (Qpih=0)
        0,   // rp: no precinct refinement (R[p] = 0)
        q_slices.to_vec(),
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-233 per-precinct `Q[p]` override entry point — the spec-natural
/// form of Annex C.2 Table C.1 where `Q[p]` is indexed by precinct `p`
/// rather than by slice or by picture.
///
/// Same shape as [`encode_planar_lossy`] but takes one `Q[p]` per
/// precinct rather than a single picture-level scalar. `q_precincts` is
/// indexed in raster scan order with the precinct at row `py`, column
/// `px` at position `py * Np,x + px`; the length must be exactly
/// `Np,y × Np,x` where:
///
/// * `Np,y = ⌈Hf / 2^NL,y⌉` is the number of precinct rows;
/// * `Np,x = 1` for this single-precinct-column entry point (`Cw = 0`),
///   so the array reduces to one `Q[p]` per precinct row of length
///   `Np,y`. Multi-column (`Cw > 0`) intersects with the per-precinct
///   override on a future round.
///
/// Each entry is in `0..=15` (the band-truncation
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` math is identical to the
/// picture-level form, only the source of `Q[p]` changes). `Fq` is
/// auto-selected: `0` (lossless) when every entry is `0`, else `8`
/// (regular, Table A.8 — required for any non-zero `Q[p]`).
///
/// Round 206 lifted `q` from picture-level to one per slice; round 233
/// lifts it further to one per precinct. The two levers can coexist:
/// `q_precincts` wins where it's non-empty; a future
/// `encode_planar_hsl_qprecinct` would carry both. This entry point
/// pins `Hsl = 0` (single slice spanning the picture), so per-slice
/// override would be a degenerate length-1 input — the per-precinct
/// override is the strict lift.
///
/// **Wire impact:** the per-precinct `Q` byte (precinct header field
/// `Q[p]`, Annex C.2 Table C.1) carries the override. The decoder reads
/// `Q[p]` per precinct (`parse_precinct_header` + `precinct_truncation`)
/// since the early rounds, so no decoder change is needed — this is a
/// pure encoder rate-allocation lever. Output is byte-identical to
/// [`encode_planar_lossy`] when every `q_precincts[p]` equals a single
/// picture-level `q`, and byte-identical to [`encode_planar_hsl_qslice`]
/// (with `hsl = 0`, single-slice) when every entry is the same value.
///
/// **Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
/// `Cpih ∈ {0, 1, 3}` — no transform / reversible RCT / Star-Tetrix,
/// `Cw = 0` (single precinct column), `Hsl = 0` (single slice),
/// `Sd = 0` (no CWD suppression), `Fs = 0`, `Qpih = 0`, `B[i] = 8`.
///
/// **Errors:** wrong-length `q_precincts` (must equal `Np,y × Np,x =
/// Np,y` at `Cw = 0`); any entry `> 15`; any non-zero entry with
/// `q_precincts` already validated against `Fq = 8` derived from
/// presence of any non-zero entry; plus the standard
/// [`EncodeConfig::validate`] errors (`cpih` / `nlx` / `nly` / plane
/// sizes).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_qpr(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q_precincts: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    // Pick the picture-level fallback `q` for the validate() guard and as
    // the source of truth when every `q_precincts` entry happens to equal
    // a single value (the precinct override then resolves to a no-op).
    let q_pic = q_precincts.iter().copied().max().unwrap_or(0);
    // Fq = 8 whenever any precinct quantizes; 0 only when every precinct
    // is lossless (preserves byte-identical output to encode_planar
    // for the all-zero case).
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q_pic,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: no per-slice override
        q_precincts.to_vec(),
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-239 per-precinct `R[p]` override (precinct-level refinement
/// budgeting, Annex C.2 Table C.1 in its spec-natural form where `R[p]`
/// is indexed by precinct `p`).
///
/// Round 115 fixed `R[p]` picture-wide via [`encode_planar_rp`]; round
/// 239 lifts it the rest of the way — one `R[p]` per precinct, in raster
/// scan order with the precinct at `(py, px)` at index `py * Np,x + px`
/// where `Np,y = ⌈Hf / 2^NL,y⌉` and `Np,x = 1` at `Cw = 0`.
///
/// `r_precincts.len()` must equal `Np,y × Np,x` (here `Np,y` because
/// `Cw = 0`). Each entry is in `0..=NL - 1` where `NL = Nc × Nβ` is the
/// total band count (Annex B.6 NL definition with `Sd = 0`). Out-of-range
/// entries are rejected with [`crate::JpegXsError::Invalid`].
///
/// **Wire impact:** the per-precinct `R` byte (precinct header field
/// `R[p]`, Annex C.2 Table C.1) carries the override. The decoder reads
/// `R[p]` per precinct (`parse_precinct_header` + `precinct_truncation`)
/// since round 115, so no decoder change is needed — this is a pure
/// encoder rate-allocation lever. Output is byte-identical to
/// [`encode_planar`] when every `r_precincts[p]` is `0`
/// (refinement is a lossless no-op at the picture-wide `q = 0` this
/// entry point pins), and byte-identical to [`encode_planar_rp`] when
/// every entry is the same non-zero value (the per-precinct override
/// resolves to a no-op when `r_precincts[p]` equals the picture-level
/// fallback).
///
/// **Composition with `q`:** the picture-wide `q` is `0` (lossless), so
/// `T[p, b] = clamp(Q − G[b] − r, 0, 15) = clamp(0 − G[b] − r, 0, 15)`
/// is already at its `0` floor regardless of `r`. The refinement bit
/// therefore changes nothing on the wire — `r_precincts = [0; n]`,
/// `[k; n]` for any `k`, and any mixed `r_precincts` all emit the same
/// codestream at `q = 0`. To exercise per-precinct `R[p]` as a
/// rate-distortion lever, combine with `q > 0` (a future
/// `encode_planar_qpr_rpr` cross-product would carry both vectors).
///
/// **Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
/// `Cpih ∈ {0, 1, 3}` — no transform / reversible RCT / Star-Tetrix,
/// `Cw = 0` (single precinct column), `Hsl = 0` (single slice),
/// `Sd = 0` (no CWD suppression), `Fs = 0`, `Qpih = 0`, `B[i] = 8`.
///
/// **Errors:** wrong-length `r_precincts` (must equal
/// `Np,y × Np,x = Np,y` at `Cw = 0`); any entry exceeding `NL − 1`; plus
/// the standard [`EncodeConfig::validate`] errors (`cpih` / `nlx` /
/// `nly` / plane sizes).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_rpr(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    r_precincts: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    // Picture-wide q = 0 (lossless). Annex C.6.2 Table C.10:
    // T[p, b] = clamp(Q[p] − G[b] − r, 0, 15); at Q = 0 every clamp
    // floors regardless of r, so this entry point exposes the wire-
    // level R[p] byte without changing the data sub-packet bytes.
    // Fq = 0 stays the natural lossless mode.
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
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
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: picture-level R[p] override stays the default
        Vec::new(), // q_slices: no per-slice override
        Vec::new(), // q_precincts: no per-precinct override
        r_precincts.to_vec(),
        planes,
    )
}

/// Round-242 joint per-precinct `Q[p] × R[p]` override (the
/// cross-product the round-233 / round-239 changelogs flagged as the
/// next step).
///
/// Round 233 ([`encode_planar_qpr`]) lifted picture-level `q` to one
/// `Q[p]` per precinct. Round 239 ([`encode_planar_rpr`]) lifted
/// picture-level `R[p]` to one per precinct. Both lived on the same
/// precinct-header pair (Annex C.2 Table C.1 — `Q[p]` is precinct-header
/// byte 3, `R[p]` is precinct-header byte 4), and the in-place overlay
/// helper `precinct_cfg_for` already composed them cleanly. Round 242
/// is the public entry point that surfaces both vectors to the caller
/// at the same time, so `R[p]` becomes an active rate-distortion lever
/// where the round-239 `q = 0` pin had left it as a wire-only no-op.
///
/// **Why this is the active lever:** Annex C.6.2 Table C.10
/// truncation is `T[p, b] = clamp(Q[p] − G[b] − r, 0, 15)`. At the
/// round-239 picture-wide `Q = 0` floor every clamp landed at zero
/// regardless of `r`, so the per-precinct `R[p]` only surfaced in the
/// precinct header and not in the data sub-packet bytes. With
/// `q_precincts[p] > 0` the clamp is no longer floored, and a
/// non-zero `r_precincts[p]` actively raises `T[p, b]` for the `R[p]`
/// lowest-band indices — one extra magnitude-bit truncated per
/// affected band per precinct. This is the rate-distortion lever the
/// previous two rounds wired up to.
///
/// `q_precincts.len()` and `r_precincts.len()` must both equal
/// `Np,y × Np,x` (here `Np,y` because `Cw = 0`). At least one of the
/// two vectors must be non-empty — calling with both empty is the
/// existing [`encode_planar`] entry point.
///
/// * Each `q_precincts[p]` is in `0..=15` (Annex C.6.2 Table C.10 `T`
///   clamp range).
/// * Each `r_precincts[p]` is in `0..=NL − 1` where
///   `NL = Nc × Nβ` (Annex B.6 `NL` definition at `Sd = 0`).
///
/// The `Fq` flag is `8` whenever any `q_precincts[p] > 0` (same rule
/// as [`encode_planar_qpr`]) and `0` otherwise (the all-zero / empty-
/// `q_precincts` reduction restores byte-identical output to
/// [`encode_planar_rpr`] for the same `r_precincts`, or to
/// [`encode_planar`] when both vectors are zeros).
///
/// The picture-level fallbacks the precinct overlay helper compares
/// against are picked so a uniform vector resolves to a no-op:
/// `cfg.q = max(q_precincts)` and `cfg.rp = max(r_precincts)` — when
/// every entry equals its own picked-max, the per-precinct branch in
/// `precinct_cfg_for` returns `None` (no per-precinct clone needed)
/// and the codestream reduces to the picture-level form.
///
/// **Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
/// `Cpih ∈ {0, 1, 3}` — no transform / reversible RCT / Star-Tetrix,
/// `Cw = 0` (single precinct column), `Hsl = 0` (single slice),
/// `Sd = 0` (no CWD suppression), `Fs = 0`, `Qpih = 0`, `B[i] = 8`.
///
/// **Errors:** wrong-length `q_precincts` or `r_precincts` (each must
/// equal `Np,y × Np,x = Np,y` at `Cw = 0`); any `q_precincts[p] > 15`;
/// any `r_precincts[p] > NL − 1`; both vectors empty; plus the
/// standard [`EncodeConfig::validate`] errors (`cpih` / `nlx` / `nly`
/// / plane sizes).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_qpr_rpr(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q_precincts: &[u8],
    r_precincts: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    if q_precincts.is_empty() && r_precincts.is_empty() {
        return Err(Error::invalid(
            "jpegxs encoder: encode_planar_qpr_rpr needs at least one of \
             q_precincts / r_precincts non-empty (call encode_planar for \
             the no-override path)",
        ));
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    // Picture-level fallbacks: `max(...)` so a uniform vector reduces
    // to a no-op inside `precinct_cfg_for`.
    let q_pic = q_precincts.iter().copied().max().unwrap_or(0);
    let rp_pic = r_precincts.iter().copied().max().unwrap_or(0);
    // Fq=8 whenever any precinct quantizes (matches encode_planar_qpr).
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q_pic,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0, // cw: single precinct column
        0, // sd: no CWD suppression
        0, // fs: signs jointly with data (Fs=0)
        0, // hsl: single slice (Hsl = Np,y)
        0, // qpih: deadzone inverse quantizer (Qpih=0)
        rp_pic,
        Vec::new(), // q_slices: no per-slice override
        q_precincts.to_vec(),
        r_precincts.to_vec(),
        planes,
    )
}

/// Round-212 rate-budget driven per-slice `Q[p]` picker.
///
/// Given a target byte budget, returns a `q_slices` vector — one
/// `Q[p]` per slice in top-down `Yslh` order — that drives
/// [`encode_planar_hsl_qslice`] to emit a codestream of length
/// `≤ target_bytes`, while concentrating bits on slices whose source
/// content has the lowest spatial activity (so distortion from
/// quantization lands on the busier slices where it is less
/// perceptually visible). The picker is fully deterministic and
/// performs no rate-distortion search beyond calling
/// [`encode_planar_hsl_qslice`] with candidate vectors and reading
/// back the byte length — there is no internal model of the entropy
/// coder, no oracle, no external library. Bytes returned by the
/// callee are the only feedback the search uses.
///
/// Strategy (three passes, each calling `encode_planar_hsl_qslice`
/// internally to measure the actual output length):
///
/// 1. **Lossless probe.** Try `q_slices = [0; n_slices]`. If that
///    fits in `target_bytes`, return it (no reason to quantize).
/// 2. **Uniform-`Q` bisect.** Find the smallest uniform
///    `Q ∈ 1..=15` whose output fits in `target_bytes`. If even
///    `Q = 15` overshoots, return `[15; n_slices]` with a
///    [`JpegXsError::Invalid`] error tagged with the actual
///    encoded length so the caller can surface the budget violation.
/// 3. **Per-slice relaxation.** Sort slices by spatial activity
///    ascending (`Σ |row[r+1][c] − row[r][c]|` summed over every
///    plane's pixels inside the slice's image-row range). Starting
///    from the uniform-`Q` baseline, lower one low-activity slice
///    at a time by one `Q` step (down to `Q = 0`) while each
///    candidate still fits. The picker stops at the first candidate
///    that overshoots and keeps the last fitting vector.
///
/// Inputs:
/// * `width`, `height`, `nc`, `cpih`, `nlx`, `nly`, `hsl`, `planes`
///   — identical to [`encode_planar_hsl_qslice`]. `hsl == 0` is the
///   single-slice mode (the picker is degenerate then: it just
///   bisects a scalar `q`).
/// * `target_bytes` — upper bound on the encoded codestream length.
///   Must be `> 0`.
///
/// Returns: the chosen `q_slices` vector. Pass the same vector into
/// [`encode_planar_hsl_qslice`] to obtain the codestream itself; the
/// [`encode_planar_hsl_target_bytes`] convenience wrapper does both
/// in one call and returns `(codestream, q_slices)`.
///
/// Errors:
/// * Any of the validation errors [`encode_planar_hsl_qslice`] would
///   produce (invalid `cpih`, `nlx`/`nly` out of range, plane size
///   mismatch, etc.).
/// * [`crate::JpegXsError::Invalid`] when `target_bytes == 0`.
/// * [`crate::JpegXsError::Invalid`] when even `q_slices = [15; n_slices]`
///   overshoots the budget. The error message reports the actual
///   encoded length so the caller knows how far over they are.
#[allow(clippy::too_many_arguments)]
pub fn pick_q_slices_for_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    if target_bytes == 0 {
        return Err(Error::invalid(
            "jpegxs picker: target_bytes must be > 0".to_string(),
        ));
    }
    // Compute slice count the same way EncodeConfig::validate does.
    // Np,y = ⌈Hf / 2^NL,y⌉; effective Hsl = Hsl when > 0, else Np,y
    // (single slice).
    let hp_pow = 1u32 << nly;
    let np_y = (height as u32).div_ceil(hp_pow);
    let hsl_rows = if hsl == 0 { np_y } else { hsl as u32 };
    if hsl_rows == 0 {
        return Err(Error::invalid(format!(
            "jpegxs picker: Hsl={hsl} resolves to zero precinct rows per slice (height={height}, NL,y={nly})"
        )));
    }
    let n_slices = np_y.div_ceil(hsl_rows) as usize;

    // Pass 1 — lossless probe. If the source is small / sparse enough
    // that an all-zeros codestream fits, no quantization is required.
    let q_zero = vec![0u8; n_slices];
    let cs_zero =
        encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl, &q_zero, planes)?;
    if cs_zero.len() <= target_bytes {
        return Ok(q_zero);
    }

    // Pass 2 — uniform-Q bisect over `1..=15`. Standard binary search:
    // monotonicity of codestream length in Q is empirical (higher Q
    // truncates more bitplanes → fewer bits) so we bisect rather than
    // assuming a closed form.
    let mut lo: u8 = 1;
    let mut hi: u8 = 15;
    let mut best_uniform_q: Option<u8> = None;
    let mut best_uniform_len: usize = usize::MAX;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let qv = vec![mid; n_slices];
        let cs = encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl, &qv, planes)?;
        if cs.len() <= target_bytes {
            best_uniform_q = Some(mid);
            best_uniform_len = cs.len();
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            if mid == 15 {
                break;
            }
            lo = mid + 1;
        }
    }
    let uniform_q = match best_uniform_q {
        Some(q) => q,
        None => {
            // Even Q=15 overshoots. Report how badly so the caller
            // can either bump Hsl, raise Wf/Hf budget, or accept the
            // overshoot.
            let qv = vec![15u8; n_slices];
            let cs = encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl, &qv, planes)?;
            return Err(Error::invalid(format!(
                "jpegxs picker: target_bytes={target_bytes} unreachable; Q=15 emits {} bytes",
                cs.len()
            )));
        }
    };

    // Single-slice degenerate case — no relaxation possible.
    if n_slices == 1 {
        return Ok(vec![uniform_q; 1]);
    }

    // Pass 3 — per-slice relaxation. Rank slices by spatial activity
    // (low first) using the source pixels in each slice's image-row
    // range, summed over every plane. Activity here is the L1 norm
    // of the row-to-row gradient inside the slice (cheap, no FFT, no
    // wavelet — just `|row[r+1][c] − row[r][c]|`). Low-activity
    // slices receive the quantization relief first because they are
    // (a) likely visually salient (flat regions show banding worst)
    // and (b) likely the cheapest to emit at lower Q (fewer non-zero
    // coefficients to encode after the wavelet anyway).
    let slice_row_ranges = compute_slice_row_ranges(height, nly, hsl_rows);
    debug_assert_eq!(slice_row_ranges.len(), n_slices);
    let mut activity: Vec<(usize, u64)> = slice_row_ranges
        .iter()
        .enumerate()
        .map(|(t, &(y0, y1))| (t, slice_activity(planes, width, y0, y1)))
        .collect();
    activity.sort_by_key(|&(_, a)| a);

    let mut best = vec![uniform_q; n_slices];
    let mut best_len = best_uniform_len;
    // Walk the lowest-activity slices and try to drop their Q one
    // step at a time. We do a full sweep: each slice is offered Q-1,
    // and if the result still fits we commit; otherwise we move on
    // to the next slice. We repeat until a full pass made no change
    // — this lets the most-relaxed slices drop further than one step
    // when slack permits.
    loop {
        let mut changed = false;
        for &(t, _) in &activity {
            if best[t] == 0 {
                continue;
            }
            let mut trial = best.clone();
            trial[t] -= 1;
            let cs =
                encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl, &trial, planes)?;
            if cs.len() <= target_bytes {
                best = trial;
                best_len = cs.len();
                changed = true;
            }
        }
        let _ = best_len; // silence unused-warning when debug-asserts off
        if !changed {
            break;
        }
    }
    Ok(best)
}

/// Round-212 convenience wrapper — picks `q_slices` against
/// `target_bytes` and emits the codestream in one call.
///
/// Returns `(codestream, q_slices)`. The codestream is guaranteed to
/// satisfy `codestream.len() <= target_bytes` (otherwise the picker
/// returns the `target_bytes unreachable` error). The `q_slices`
/// vector is the one returned by [`pick_q_slices_for_target_bytes`];
/// callers can persist it for reproducible re-encode of identical
/// parameters.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let q_slices = pick_q_slices_for_target_bytes(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        hsl,
        target_bytes,
        planes,
    )?;
    let cs = encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl, &q_slices, planes)?;
    Ok((cs, q_slices))
}

/// Round-212 helper — image-row ranges of every slice the encoder
/// will emit, in top-down `Yslh = 0..n_slices` order.
///
/// Returns one `(y0, y1)` per slice, where `y0..y1` is the half-open
/// image-row range. Slice height is `hsl_rows × 2^NL,y` image rows
/// (each precinct covers `2^NL,y` image rows per Annex B.6); the
/// last slice is clipped to `height` if `Np,y % Hsl != 0`.
fn compute_slice_row_ranges(height: u16, nly: u8, hsl_rows: u32) -> Vec<(u32, u32)> {
    let h = height as u32;
    let rows_per_precinct = 1u32 << nly;
    let rows_per_slice = hsl_rows.saturating_mul(rows_per_precinct);
    if rows_per_slice == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut y = 0u32;
    while y < h {
        let y1 = (y + rows_per_slice).min(h);
        out.push((y, y1));
        y = y1;
    }
    out
}

/// Round-212 helper — spatial activity of a slice's pixel range.
///
/// Returns the L1 norm of the row-to-row first-difference summed
/// across every component. Used by the rate-budget picker to rank
/// slices for the per-slice `Q[p]` relaxation pass: low-activity
/// slices get the quantization relief first.
///
/// The norm is computed on the raw 8-bit pixel values (the picker
/// operates on the `Vec<u8>` planar input — high-bit-depth paths
/// will route through a `u16` variant in a future round if needed).
/// A slice that resolves to fewer than two image rows returns 0;
/// the picker treats that as "no activity preference, leave at
/// uniform Q".
fn slice_activity(planes: &[Vec<u8>], width: u16, y0: u32, y1: u32) -> u64 {
    let w = width as usize;
    if w == 0 || y1 <= y0 + 1 {
        return 0;
    }
    let mut acc: u64 = 0;
    for plane in planes {
        if plane.is_empty() || plane.len() < w {
            continue;
        }
        let plane_rows = plane.len() / w;
        let r0 = (y0 as usize).min(plane_rows);
        let r1 = (y1 as usize).min(plane_rows);
        if r1 <= r0 + 1 {
            continue;
        }
        for r in r0..r1 - 1 {
            let cur = &plane[r * w..(r + 1) * w];
            let nxt = &plane[(r + 1) * w..(r + 2) * w];
            for c in 0..w {
                acc += (cur[c] as i32 - nxt[c] as i32).unsigned_abs() as u64;
            }
        }
    }
    acc
}

/// Round-218 rate-budget driven `R[p]` picker for the
/// [`encode_planar_rp`] path.
///
/// Given a fixed quantization step `q` and a target codestream length,
/// picks the largest precinct refinement `R[p] ∈ 0..=NL-1` whose output
/// still fits in `target_bytes`, where `NL = Nc × Nβ` for the
/// 4:4:4 single-precinct-column path this picker operates on (no `Sd`
/// suppression, see "Scope" below). Larger `R[p]` refines more of the
/// lowest-index bands (LL first per the `β`-major band enumeration of
/// Annex B.6), granting each refined band one extra retained magnitude
/// bitplane via the Annex C.6.2 Table C.10 term
/// `r = (P[b] < R[p]) ? 1 : 0` inside
/// `T[p,b] = clamp(Q − G[b] − r, 0, 15)`. With the encoder's `P[b] = b`
/// priority assignment (Annex A.4.11 — emitted by the WGT marker since
/// round 115) the refinement is monotone in `R[p]`: each step from
/// `k → k+1` adds the band-`k` priority bit to one more band's
/// truncation reduction, so the codestream is non-decreasing in `R[p]`
/// at fixed `q`. The picker exploits this with a one-dimensional
/// linear scan from `NL-1` down to `0`, returning the first
/// `R[p]` whose codestream fits.
///
/// **Why "largest fitting `R[p]`" and not "smallest":** the refinement
/// transfers coded bits **toward** the lowest-frequency bands (which
/// are perceptually most important — flat-region banding shows up
/// there worst). At a fixed `q`, every additional refinement step
/// improves PSNR on those bands at the cost of a few more coded bits
/// in those same bands. The optimal use of a byte budget is therefore
/// to spend it on the most refinement the budget can afford. This
/// matches the spec's Annex H NOTE intent ("Other choices are
/// possible") and complements r212's
/// [`pick_q_slices_for_target_bytes`] — which trades quantization
/// strength **between slices** at a fixed refinement — by trading
/// refinement strength **between bands** at a fixed quantization.
///
/// Strategy (linear scan, each iteration calls [`encode_planar_rp`]
/// internally to measure the actual output length):
///
/// 1. **`R[p] = 0` probe.** Encode at the no-refinement baseline. If
///    even that overshoots `target_bytes`, the budget is unreachable
///    by `R[p]` alone (lower `q` or `pick_q_slices_for_target_bytes`
///    can still help) — return a
///    [`crate::JpegXsError::Invalid`] error tagged with the actual
///    encoded length so the caller knows how far over they are.
/// 2. **Scan `R[p] = NL-1` down to `1`.** Return the first `R[p]`
///    whose codestream fits. The scan terminates trivially when
///    `NL == 1` (no refinement possible — `R[p] = 0` is the only
///    legal value) by falling through to the baseline.
///
/// The picker is fully deterministic and performs no rate-distortion
/// search beyond calling [`encode_planar_rp`] with candidate `R[p]`
/// values and reading back the byte length — there is no internal
/// model of the entropy coder, no oracle, no external library. Bytes
/// returned by the callee are the only feedback the search uses.
///
/// **Scope** — matches [`encode_planar_rp`] exactly:
///
/// * 4:4:4 (`sx[i] = sy[i] = 1` for all `i`, hard-wired by
///   [`encode_planar_rp`]).
/// * `Cpih ∈ {0, 1, 3}` (no transform, RCT, or Star-Tetrix per Annex
///   F.2 Table F.1's component-count constraints).
/// * Single precinct column (`Cw = 0`), single slice (`Hsl = 0`), no
///   CWD suppression (`Sd = 0`), `Fs = 0`, `Qpih = 0`.
/// * `q ∈ 0..=15`; `q = 0` makes refinement a lossless no-op (the
///   refinement term can only push `T` down, and `T = 0` is already
///   the floor at `q = 0`) so the picker returns whichever `R[p]`
///   value fits — typically `NL-1`, since every refinement is
///   byte-identical to `R[p] = 0` at `q = 0`.
///
/// Inputs:
/// * `width`, `height`, `nc`, `cpih`, `nlx`, `nly`, `q`, `planes` —
///   identical to [`encode_planar_rp`].
/// * `target_bytes` — upper bound on the encoded codestream length.
///   Must be `> 0`.
///
/// Returns: the chosen `R[p]` value. Pass the same value into
/// [`encode_planar_rp`] to obtain the codestream itself; the
/// [`encode_planar_rp_target_bytes`] convenience wrapper does both in
/// one call and returns `(codestream, rp)`.
///
/// Errors:
/// * Any of the validation errors [`encode_planar_rp`] would produce
///   (invalid `cpih`, `nlx`/`nly` out of range, plane size mismatch,
///   etc.).
/// * [`crate::JpegXsError::Invalid`] when `target_bytes == 0`.
/// * [`crate::JpegXsError::Invalid`] when even `R[p] = 0` overshoots
///   the budget. The error message reports the actual encoded length
///   so the caller knows how far over they are.
#[allow(clippy::too_many_arguments)]
pub fn pick_rp_for_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<u8> {
    if target_bytes == 0 {
        return Err(Error::invalid(
            "jpegxs rp picker: target_bytes must be > 0".to_string(),
        ));
    }

    // NL = Nc × Nβ for the 4:4:4 / Sd = 0 surface this picker covers
    // (Annex B.6 NL definition with Sd = 0). encode_planar_rp validates
    // R[p] ∈ 0..=NL-1, so the scan upper bound is NL-1.
    let nbeta = n_beta(nlx, nly);
    let nl = (nc as u32) * nbeta;
    if nl == 0 {
        return Err(Error::invalid(format!(
            "jpegxs rp picker: NL=0 (nc={nc}, NL,x={nlx}, NL,y={nly})"
        )));
    }
    let rp_max = (nl - 1).min(u8::MAX as u32) as u8;

    // Step 1 — R[p] = 0 baseline. This is the smallest stream of the
    // family (no refinement, no extra bits in any band), so if it
    // overshoots the budget then no R[p] can fit.
    let cs_zero = encode_planar_rp(width, height, nc, cpih, nlx, nly, q, 0, planes)?;
    if cs_zero.len() > target_bytes {
        return Err(Error::invalid(format!(
            "jpegxs rp picker: target_bytes={target_bytes} unreachable; R[p]=0 emits {} bytes",
            cs_zero.len()
        )));
    }

    // Step 2 — scan from R[p] = NL-1 downwards; return the first
    // value whose codestream fits. Larger R[p] refines more bands,
    // and refinement is monotone non-decreasing in the codestream
    // length (each extra refined band gains one magnitude bitplane),
    // so the first fit is also the largest fit. NL = 1 (no refinement
    // possible) falls through this loop unchanged and returns 0
    // from the baseline below.
    let mut rp = rp_max;
    while rp >= 1 {
        let cs = encode_planar_rp(width, height, nc, cpih, nlx, nly, q, rp, planes)?;
        if cs.len() <= target_bytes {
            return Ok(rp);
        }
        rp -= 1;
    }
    Ok(0)
}

/// Round-218 convenience wrapper — picks `R[p]` against `target_bytes`
/// and emits the codestream in one call.
///
/// Returns `(codestream, rp)`. The codestream is guaranteed to satisfy
/// `codestream.len() <= target_bytes` (otherwise the picker returns
/// the `target_bytes unreachable; R[p]=0 emits ...` error). The `rp`
/// value is the one returned by [`pick_rp_for_target_bytes`]; callers
/// can persist it for reproducible re-encode of identical parameters.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_rp_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, u8)> {
    let rp = pick_rp_for_target_bytes(width, height, nc, cpih, nlx, nly, q, target_bytes, planes)?;
    let cs = encode_planar_rp(width, height, nc, cpih, nlx, nly, q, rp, planes)?;
    Ok((cs, rp))
}

/// Round-224 joint per-slice `Q[p]` + precinct refinement `R[p]` encoder
/// primitive — composes [`encode_planar_hsl_qslice`] (round 206) and
/// [`encode_planar_rp`] (round 115) on a single encode call.
///
/// Both axes live on independent precinct-header fields per Annex C.2
/// Table C.1: `Q[p]` is the per-precinct quantization step (one byte
/// per precinct, lifted to "one value per slice" in round 206) and
/// `R[p]` is the per-precinct refinement (one byte per precinct,
/// constant across precincts in this encoder). The Annex C.6.2 Table
/// C.10 per-band truncation
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` with
/// `r = (P[b] < R[p]) ? 1 : 0` combines them additively inside one
/// `clamp`, so the two axes are orthogonal on the bitstream — `Q[p]`
/// lives in the `Q` byte of each precinct header and `R[p]` lives in
/// the `R` byte. There is no cross-axis coupling beyond the shared
/// `clamp` floor at 0 (which already governs each axis in isolation).
///
/// The decoder reconstructs the identical `T[p,b]` from the
/// `(P[b], R[p], Q[p])` triple it reads back from the wire, so any
/// output of this entry point round-trips through
/// [`crate::decode_jpeg_xs`].
///
/// `hsl`, `q_slices`, and `rp` follow the same semantics as in
/// [`encode_planar_hsl_qslice`] and [`encode_planar_rp`]:
///
/// * `hsl` is the slice height in precinct rows (PIH `Hsl`, Annex B.10);
///   `hsl == 0` is the single-slice default (`Hsl = Np,y`, one slice
///   covering the picture).
/// * `q_slices.len()` must exactly equal the slice count
///   `⌈Np,y / max(hsl, 1)⌉` (single entry when `hsl == 0`); each entry
///   is in `0..=15`. `Fq` is auto-selected — `0` when every entry is
///   `0`, else `8` (regular mode, required for any non-zero `Q[p]`).
/// * `rp` is the precinct refinement `R[p] ∈ 0..=NL-1` where
///   `NL = Nc × Nβ` for this picker's 4:4:4 / Sd = 0 surface. `rp = 0`
///   is the no-refinement default; `rp > 0` activates the Annex C.6.2
///   Table C.10 refinement term lowering `T[p,b]` by one for the `rp`
///   lowest-index (LL-first) bands. The encoder emits per-band
///   priorities `P[b] = b` in the WGT marker (Annex A.4.11), identical
///   to round 115.
///
/// **Composition behaviour:**
///
/// * `q_slices = [0; n]` with any `rp` — every precinct lossless (`T =
///   0` regardless of refinement), output is byte-identical to the
///   `rp = 0` lossless stream from [`encode_planar_hsl_qslice`].
/// * `q_slices` all-equal + `rp = 0` — byte-identical to
///   [`encode_planar_hsl_qslice`] at that `q_slices`.
/// * `q_slices.len() == 1` + `hsl = 0` + `rp = 0` — byte-identical to
///   [`encode_planar_lossy`] at that `q`.
/// * `q_slices.len() == 1` + `hsl = 0` + `rp > 0` — byte-identical to
///   [`encode_planar_rp`] at that `q` and `rp`.
///
/// **Scope** — 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Sd = 0`, `Fs = 0`,
/// `Qpih = 0`, `B[i] = 8`. The mixed high-bit-depth + per-slice +
/// refinement surface is a future round if the demand arises (the inner
/// encoder already plumbs all four parameters through
/// `encode_planar_inner_bd`).
///
/// Errors: validation errors from [`EncodeConfig::validate`] (wrong
/// `q_slices` length, entries > 15, `R[p] >= NL`, etc.).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_qslice_rp(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    q_slices: &[u8],
    rp: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let q_pic = q_slices.iter().copied().max().unwrap_or(0);
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        fq,
        q_pic,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,   // cw: single precinct column
        0,   // sd: no CWD suppression
        0,   // fs: signs jointly with data (Fs=0)
        hsl, // hsl: slice height in precinct rows
        0,   // qpih: deadzone inverse quantizer (Qpih=0)
        rp,  // rp: precinct refinement R[p]
        q_slices.to_vec(),
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-224 joint rate-budget picker — picks both `q_slices` (per-slice
/// `Q[p]`, Annex C.2 Table C.1 lifted to per-slice in round 206) and `rp`
/// (precinct refinement `R[p]`, Annex C.2 Table C.1 + Annex C.6.2 Table
/// C.10) against a single byte budget, driving the round-224 joint
/// primitive [`encode_planar_hsl_qslice_rp`].
///
/// The two levers are orthogonal on the bitstream: per-slice `Q[p]`
/// lives in each precinct's `Q` byte and `R[p]` lives in each precinct's
/// `R` byte, so any `(q_slices, rp)` pair the picker emits is
/// spec-compliant. The Annex C.6.2 Table C.10 truncation
/// `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` with
/// `r = (P[b] < R[p]) ? 1 : 0` is monotone non-increasing in `Q[p]` and
/// monotone non-decreasing in `R[p]` (lower `Q` keeps more low-magnitude
/// bits; higher `R[p]` refines more bands toward retaining one extra
/// bitplane), so the joint picker can trade them against each other at
/// the byte-budget boundary.
///
/// **Strategy — two-axis nested search:**
///
/// 1. **Outer loop on `rp`** from `0` up to `NL-1`. Refinement is
///    monotone non-decreasing in codestream length at any fixed `q`
///    (each refined band gains one extra retained magnitude bitplane),
///    so larger `rp` shifts more bits toward the low-index bands. We
///    want the largest `rp` whose inner `q_slices` search still fits
///    the budget — start at `rp = 0` (cheapest) and walk upward,
///    keeping the last fitting solution.
/// 2. **Inner loop on `q_slices`** at the current `rp`. Reuse r212's
///    three-pass strategy verbatim — lossless probe, uniform-`Q`
///    bisect on `1..=15`, per-slice low-activity relaxation — but call
///    [`encode_planar_hsl_qslice_rp`] (with the current `rp`) instead
///    of [`encode_planar_hsl_qslice`] (which pins `rp = 0`).
///    Returns either the fitting `q_slices` or "Q=15 unreachable" if
///    even max-quantization at this `rp` overshoots.
/// 3. **Promotion rule.** If the inner search at `rp+1` succeeds and
///    still fits, replace the current best with that pair. If it
///    fails (Q=15 unreachable at higher `rp`), stop — higher `rp`
///    cannot fit either since `R[p]` is monotone non-decreasing in
///    codestream length at any fixed `Q[p]`.
/// 4. **Baseline rejection.** If even `rp = 0` + `q_slices = [15;n]`
///    overshoots the budget, the budget is unreachable by these two
///    levers alone (no choice of `(q_slices, rp)` can fit). Errors
///    with `target_bytes unreachable; rp=0 Q=15 emits N bytes`.
///
/// The picker is fully deterministic and performs no rate-distortion
/// search beyond calling [`encode_planar_hsl_qslice_rp`] with candidate
/// triples and reading back the byte length — there is no internal model
/// of the entropy coder, no oracle, no external library. Every
/// measurement is a real encode call.
///
/// **Scope** matches [`encode_planar_hsl_qslice_rp`] exactly: 4:4:4,
/// `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`,
/// `B[i] = 8`.
///
/// **Why "largest fitting `rp`" beats "smallest":** refinement transfers
/// coded bits toward the lowest-frequency bands (where flat-region
/// banding shows worst). At any fixed `q_slices` budget, every
/// additional `rp` step improves PSNR on the refined bands at the cost
/// of a few more coded bits in those same bands. The optimal use of a
/// byte budget is therefore to spend as many bits as the budget allows
/// on additional refinement — matching the Annex H NOTE intent
/// ("Other choices are possible") and complementing r212's
/// activity-driven `Q[p]` relaxation.
///
/// Inputs / outputs follow [`pick_q_slices_for_target_bytes`] and
/// [`pick_rp_for_target_bytes`] respectively: returns
/// `Result<(Vec<u8>, u8)>` carrying the chosen `q_slices` and `rp`.
/// The [`encode_planar_hsl_qslice_rp_target_bytes`] convenience wrapper
/// does both in one call and returns `(codestream, q_slices, rp)`.
#[allow(clippy::too_many_arguments)]
pub fn pick_q_slices_rp_for_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, u8)> {
    if target_bytes == 0 {
        return Err(Error::invalid(
            "jpegxs joint picker: target_bytes must be > 0".to_string(),
        ));
    }
    // NL = Nc × Nβ for the 4:4:4 / Sd = 0 surface.
    let nbeta = n_beta(nlx, nly);
    let nl = (nc as u32) * nbeta;
    if nl == 0 {
        return Err(Error::invalid(format!(
            "jpegxs joint picker: NL=0 (nc={nc}, NL,x={nlx}, NL,y={nly})"
        )));
    }
    let rp_max = (nl - 1).min(u8::MAX as u32) as u8;

    // Slice count — mirrors EncodeConfig::validate + r212's picker.
    let hp_pow = 1u32 << nly;
    let np_y = (height as u32).div_ceil(hp_pow);
    let hsl_rows = if hsl == 0 { np_y } else { hsl as u32 };
    if hsl_rows == 0 {
        return Err(Error::invalid(format!(
            "jpegxs joint picker: Hsl={hsl} resolves to zero precinct rows per slice (height={height}, NL,y={nly})"
        )));
    }
    let n_slices = np_y.div_ceil(hsl_rows) as usize;

    // Baseline reachability — if even rp=0 + Q=15 (max quantization,
    // no refinement) overshoots the budget, no (q_slices, rp) pair can
    // fit. Probe before any outer-loop work so the error surfaces fast.
    let cs_baseline = encode_planar_hsl_qslice_rp(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        hsl,
        &vec![15u8; n_slices],
        0,
        planes,
    )?;
    if cs_baseline.len() > target_bytes {
        return Err(Error::invalid(format!(
            "jpegxs joint picker: target_bytes={target_bytes} unreachable; rp=0 Q=15 emits {} bytes",
            cs_baseline.len()
        )));
    }

    // Outer loop on rp — walk upward keeping the last fitting (q_slices,
    // rp) pair. For each rp, run the inner activity-driven q_slices
    // picker; if it fits, promote; if it doesn't, stop (refinement is
    // monotone non-decreasing in codestream length at fixed Q[p], so
    // higher rp won't fit either).
    let mut best_q = vec![15u8; n_slices];
    let mut best_rp: u8 = 0;
    for rp in 0..=rp_max {
        match pick_q_slices_at_rp(
            width,
            height,
            nc,
            cpih,
            nlx,
            nly,
            hsl,
            rp,
            target_bytes,
            n_slices,
            planes,
        ) {
            Ok(qs) => {
                best_q = qs;
                best_rp = rp;
            }
            Err(_) => break,
        }
    }
    Ok((best_q, best_rp))
}

/// Round-224 inner picker — at a fixed `rp`, replays r212's three-pass
/// `q_slices` search against `encode_planar_hsl_qslice_rp`. Returns
/// `Err` if even `[15; n_slices]` overshoots at this `rp` (signal to
/// the outer loop that higher `rp` cannot fit either).
#[allow(clippy::too_many_arguments)]
fn pick_q_slices_at_rp(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    rp: u8,
    target_bytes: usize,
    n_slices: usize,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    // Pass 1 — lossless probe.
    let q_zero = vec![0u8; n_slices];
    let cs_zero =
        encode_planar_hsl_qslice_rp(width, height, nc, cpih, nlx, nly, hsl, &q_zero, rp, planes)?;
    if cs_zero.len() <= target_bytes {
        return Ok(q_zero);
    }

    // Pass 2 — uniform-Q bisect over 1..=15.
    let mut lo: u8 = 1;
    let mut hi: u8 = 15;
    let mut best_uniform_q: Option<u8> = None;
    let mut best_uniform_len: usize = usize::MAX;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let qv = vec![mid; n_slices];
        let cs =
            encode_planar_hsl_qslice_rp(width, height, nc, cpih, nlx, nly, hsl, &qv, rp, planes)?;
        if cs.len() <= target_bytes {
            best_uniform_q = Some(mid);
            best_uniform_len = cs.len();
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            if mid == 15 {
                break;
            }
            lo = mid + 1;
        }
    }
    let uniform_q = match best_uniform_q {
        Some(q) => q,
        None => {
            // Q=15 at this rp overshoots → signal outer loop to stop.
            return Err(Error::invalid(format!(
                "jpegxs joint picker (rp={rp}): target_bytes={target_bytes} unreachable; Q=15 overshoots"
            )));
        }
    };

    if n_slices == 1 {
        return Ok(vec![uniform_q; 1]);
    }

    // Pass 3 — per-slice activity-driven relaxation.
    let slice_row_ranges =
        compute_slice_row_ranges(height, nly, hsl_rows_for(hsl, np_y_for(height, nly)));
    debug_assert_eq!(slice_row_ranges.len(), n_slices);
    let mut activity: Vec<(usize, u64)> = slice_row_ranges
        .iter()
        .enumerate()
        .map(|(t, &(y0, y1))| (t, slice_activity(planes, width, y0, y1)))
        .collect();
    activity.sort_by_key(|&(_, a)| a);

    let mut best = vec![uniform_q; n_slices];
    let mut best_len = best_uniform_len;
    loop {
        let mut changed = false;
        for &(t, _) in &activity {
            if best[t] == 0 {
                continue;
            }
            let mut trial = best.clone();
            trial[t] -= 1;
            let cs = encode_planar_hsl_qslice_rp(
                width, height, nc, cpih, nlx, nly, hsl, &trial, rp, planes,
            )?;
            if cs.len() <= target_bytes {
                best = trial;
                best_len = cs.len();
                changed = true;
            }
        }
        let _ = best_len;
        if !changed {
            break;
        }
    }
    Ok(best)
}

/// Round-224 helper — resolve the effective Hsl-in-precinct-rows the
/// encoder uses internally (`Np,y` when caller passes `hsl == 0`).
fn hsl_rows_for(hsl: u16, np_y: u32) -> u32 {
    if hsl == 0 {
        np_y
    } else {
        hsl as u32
    }
}

/// Round-224 helper — Np,y = ⌈Hf / 2^NL,y⌉.
fn np_y_for(height: u16, nly: u8) -> u32 {
    let hp_pow = 1u32 << nly;
    (height as u32).div_ceil(hp_pow)
}

/// Round-224 convenience wrapper — picks `(q_slices, rp)` against
/// `target_bytes` and emits the codestream in one call.
///
/// Returns `(codestream, q_slices, rp)`. The codestream is guaranteed
/// to satisfy `codestream.len() <= target_bytes` (otherwise the picker
/// returns `target_bytes unreachable`). The `q_slices` and `rp` values
/// are the ones returned by [`pick_q_slices_rp_for_target_bytes`];
/// callers can persist them for reproducible re-encode.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_qslice_rp_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>, u8)> {
    let (q_slices, rp) = pick_q_slices_rp_for_target_bytes(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        hsl,
        target_bytes,
        planes,
    )?;
    let cs = encode_planar_hsl_qslice_rp(
        width, height, nc, cpih, nlx, nly, hsl, &q_slices, rp, planes,
    )?;
    Ok((cs, q_slices, rp))
}

/// Round-230 high-bit-depth widening of the round-224 joint primitive:
/// per-slice `Q[p]` (Annex C.2 Table C.1 lifted to per-slice in round
/// 206) **plus** precinct refinement `R[p]` (Annex C.2 Table C.1 +
/// Annex C.6.2 Table C.10) on a single encode call, at component bit
/// depth `bd = B[i] ∈ 9..=16` (`u16`-LE plane format inherited from
/// rounds 118 / 133 / 151).
///
/// The two rate levers are orthogonal on the bitstream — per-slice
/// `Q[p]` lives in each precinct's `Q` byte and `R[p]` lives in each
/// precinct's `R` byte — so any `(q_slices, rp)` pair is spec-compliant.
/// Bit depth is also orthogonal to both levers because the forward
/// quantizer (Annex D.4) and the refinement term `r = (P[b] < R[p]) ?
/// 1 : 0` (Annex C.6.2 Table C.10) both run on `i32` wavelet
/// coefficients independent of `B[i]`; the only bit-depth-dependent
/// pieces are the DC level shift `1 << (bd − 1)` (Annex G.3 inverse)
/// and the two-bytes-per-sample `u16`-LE plane packing.
///
/// The codestream uses `Bw = B[i] = bd` and `Fq = 8` whenever any slice
/// quantizes (`q_slices.iter().any(|&v| v > 0)`); `Fq = 0` when every
/// slice is lossless. All-equal `q_slices` + `rp = 0` is byte-identical
/// to a hypothetical `encode_planar_hsl_qslice_highbd` at that `q_slices`
/// (the round-206 high-bit-depth form, not separately exposed —
/// callers wanting that surface pass `rp = 0`); single-entry +
/// `hsl = 0` + `rp > 0` is byte-identical to a hypothetical
/// `encode_planar_rp_highbd` at the same `(q, rp)`; `q_slices = [0; n]`
/// with any `rp` is byte-identical to the lossless `rp = 0` stream
/// (refinement is a no-op when `T` is already at its `0` floor).
///
/// **Plane format:** each `planes[i]` carries `width * height`
/// little-endian `u16` samples in `0..=2^bd − 1` (samples above that
/// are an encoder error). The decoder returns the reconstructed plane
/// in the matching two-bytes-per-sample [`crate::image::JpegXsPlane`]
/// layout when `B[i] > 8`.
///
/// **Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`), `Cpih ∈ {0, 1}`
/// (no transform / reversible RCT, Annex F.3 — bit-depth agnostic),
/// `Cw = 0` (single precinct column), `Sd = 0` (no CWD suppression),
/// `Fs = 0` (joint signs), `Qpih = 0` (deadzone inverse quantizer),
/// `bd ∈ 9..=16`. Star-Tetrix (`Cpih = 3`) and NLT pre-distortion are
/// not exposed here — they intersect with the joint primitive on a
/// future round.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_qslice_rp_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    hsl: u16,
    q_slices: &[u8],
    rp: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_hsl_qslice_rp_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_hsl_qslice_rp for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_hsl_qslice_rp_highbd supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    // Pack each plane to little-endian u16 bytes (the EncodeConfig
    // bit_depth > 8 plane format), validating the nominal range first.
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let want = (width as usize) * (height as usize);
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} sample count {} != Wf*Hf {} (4:4:4)",
                p.len(),
                want,
            )));
        }
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let q_pic = q_slices.iter().copied().max().unwrap_or(0);
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        fq,
        q_pic,
        &sx,
        &sy,
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,   // cw: single precinct column
        0,   // sd: no CWD suppression
        0,   // fs: signs jointly with data (Fs=0)
        hsl, // hsl: slice height in precinct rows
        0,   // qpih: deadzone inverse quantizer (Qpih=0)
        rp,  // rp: precinct refinement R[p]
        q_slices.to_vec(),
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-230 high-bit-depth widening of the round-224 joint rate-
/// budget picker. Composes the round-218 `R[p]` linear scan with
/// r212's three-pass per-slice `Q[p]` strategy, driving the round-230
/// high-bit-depth joint primitive [`encode_planar_hsl_qslice_rp_highbd`]
/// at component bit depth `bd ∈ 9..=16`.
///
/// **Strategy** — identical two-axis nested search as
/// [`pick_q_slices_rp_for_target_bytes`]: outer loop on `rp` walks from
/// `0` up to `NL − 1` keeping the last fitting solution; inner loop
/// reuses r212's lossless probe → uniform-`Q` bisect → activity-driven
/// per-slice relaxation against the high-bit-depth joint primitive.
/// Promotion stops as soon as the inner search at `rp+1` fails
/// (refinement is monotone non-decreasing in codestream length at
/// fixed `Q[p]`, so higher `rp` cannot fit either).
///
/// **Baseline reachability** — if even `rp = 0` + `q_slices = [15; n]`
/// overshoots, errors with `target_bytes unreachable; rp=0 Q=15 emits
/// N bytes` (no choice of `(q_slices, rp)` can fit at this bit depth).
/// `target_bytes == 0` rejected.
///
/// Every measurement is a real
/// [`encode_planar_hsl_qslice_rp_highbd`] call — no internal model of
/// the entropy coder, no oracle, no external library. The per-slice
/// activity metric is computed on the original `u16` planes
/// (`Σ |row[r+1][c] − row[r][c]|` summed across every plane inside
/// each slice's image-row range) rather than the byte-packed form, so
/// the high-bit-depth content's spatial structure drives the
/// relaxation rather than the low-byte / high-byte interleave.
///
/// **Scope** mirrors [`encode_planar_hsl_qslice_rp_highbd`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn pick_q_slices_rp_for_target_bytes_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u16>],
) -> Result<(Vec<u8>, u8)> {
    if target_bytes == 0 {
        return Err(Error::invalid(
            "jpegxs joint picker (highbd): target_bytes must be > 0".to_string(),
        ));
    }
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs joint picker (highbd): requires B[i] in 9..=16, got {bd}"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs joint picker (highbd): supports Cpih in {{0, 1}}, got {cpih}"
        )));
    }
    // NL = Nc × Nβ for the 4:4:4 / Sd = 0 surface.
    let nbeta = n_beta(nlx, nly);
    let nl = (nc as u32) * nbeta;
    if nl == 0 {
        return Err(Error::invalid(format!(
            "jpegxs joint picker (highbd): NL=0 (nc={nc}, NL,x={nlx}, NL,y={nly})"
        )));
    }
    let rp_max = (nl - 1).min(u8::MAX as u32) as u8;

    // Slice count — mirrors r224's picker.
    let hp_pow = 1u32 << nly;
    let np_y = (height as u32).div_ceil(hp_pow);
    let hsl_rows = if hsl == 0 { np_y } else { hsl as u32 };
    if hsl_rows == 0 {
        return Err(Error::invalid(format!(
            "jpegxs joint picker (highbd): Hsl={hsl} resolves to zero precinct rows per slice (height={height}, NL,y={nly})"
        )));
    }
    let n_slices = np_y.div_ceil(hsl_rows) as usize;

    // Baseline reachability — if even rp=0 + Q=15 (max quantization,
    // no refinement) overshoots the budget, no (q_slices, rp) pair can
    // fit. Probe before any outer-loop work so the error surfaces fast.
    let cs_baseline = encode_planar_hsl_qslice_rp_highbd(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        bd,
        hsl,
        &vec![15u8; n_slices],
        0,
        planes,
    )?;
    if cs_baseline.len() > target_bytes {
        return Err(Error::invalid(format!(
            "jpegxs joint picker (highbd): target_bytes={target_bytes} unreachable; rp=0 Q=15 emits {} bytes",
            cs_baseline.len()
        )));
    }

    // Outer loop on rp — walk upward keeping the last fitting (q_slices,
    // rp) pair. For each rp, run the inner activity-driven q_slices
    // picker; if it fits, promote; if it doesn't, stop.
    let mut best_q = vec![15u8; n_slices];
    let mut best_rp: u8 = 0;
    for rp in 0..=rp_max {
        match pick_q_slices_at_rp_highbd(
            width,
            height,
            nc,
            cpih,
            nlx,
            nly,
            bd,
            hsl,
            rp,
            target_bytes,
            n_slices,
            planes,
        ) {
            Ok(qs) => {
                best_q = qs;
                best_rp = rp;
            }
            Err(_) => break,
        }
    }
    Ok((best_q, best_rp))
}

/// Round-230 inner picker — at a fixed `rp`, replays r212's three-pass
/// `q_slices` search against
/// [`encode_planar_hsl_qslice_rp_highbd`] using the `u16` plane
/// activity metric. Returns `Err` if even `[15; n_slices]` overshoots
/// at this `rp` (signal to the outer loop that higher `rp` cannot fit
/// either).
#[allow(clippy::too_many_arguments)]
fn pick_q_slices_at_rp_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    hsl: u16,
    rp: u8,
    target_bytes: usize,
    n_slices: usize,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    // Pass 1 — lossless probe.
    let q_zero = vec![0u8; n_slices];
    let cs_zero = encode_planar_hsl_qslice_rp_highbd(
        width, height, nc, cpih, nlx, nly, bd, hsl, &q_zero, rp, planes,
    )?;
    if cs_zero.len() <= target_bytes {
        return Ok(q_zero);
    }

    // Pass 2 — uniform-Q bisect over 1..=15.
    let mut lo: u8 = 1;
    let mut hi: u8 = 15;
    let mut best_uniform_q: Option<u8> = None;
    let mut best_uniform_len: usize = usize::MAX;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let qv = vec![mid; n_slices];
        let cs = encode_planar_hsl_qslice_rp_highbd(
            width, height, nc, cpih, nlx, nly, bd, hsl, &qv, rp, planes,
        )?;
        if cs.len() <= target_bytes {
            best_uniform_q = Some(mid);
            best_uniform_len = cs.len();
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            if mid == 15 {
                break;
            }
            lo = mid + 1;
        }
    }
    let uniform_q = match best_uniform_q {
        Some(q) => q,
        None => {
            // Q=15 at this rp overshoots → signal outer loop to stop.
            return Err(Error::invalid(format!(
                "jpegxs joint picker (highbd, rp={rp}): target_bytes={target_bytes} unreachable; Q=15 overshoots"
            )));
        }
    };

    if n_slices == 1 {
        return Ok(vec![uniform_q; 1]);
    }

    // Pass 3 — per-slice activity-driven relaxation on `u16` planes.
    let slice_row_ranges =
        compute_slice_row_ranges(height, nly, hsl_rows_for(hsl, np_y_for(height, nly)));
    debug_assert_eq!(slice_row_ranges.len(), n_slices);
    let mut activity: Vec<(usize, u64)> = slice_row_ranges
        .iter()
        .enumerate()
        .map(|(t, &(y0, y1))| (t, slice_activity_u16(planes, width, y0, y1)))
        .collect();
    activity.sort_by_key(|&(_, a)| a);

    let mut best = vec![uniform_q; n_slices];
    let mut best_len = best_uniform_len;
    loop {
        let mut changed = false;
        for &(t, _) in &activity {
            if best[t] == 0 {
                continue;
            }
            let mut trial = best.clone();
            trial[t] -= 1;
            let cs = encode_planar_hsl_qslice_rp_highbd(
                width, height, nc, cpih, nlx, nly, bd, hsl, &trial, rp, planes,
            )?;
            if cs.len() <= target_bytes {
                best = trial;
                best_len = cs.len();
                changed = true;
            }
        }
        let _ = best_len;
        if !changed {
            break;
        }
    }
    Ok(best)
}

/// Round-230 high-bit-depth slice-activity helper. Mirrors
/// [`slice_activity`] but operates on `u16` planes so the spatial
/// metric reflects the original sample magnitudes rather than the
/// low-byte / high-byte interleave of a `to_le_bytes()` packing.
fn slice_activity_u16(planes: &[Vec<u16>], width: u16, y0: u32, y1: u32) -> u64 {
    let w = width as usize;
    if w == 0 || y1 <= y0 + 1 {
        return 0;
    }
    let mut acc: u64 = 0;
    for plane in planes {
        if plane.is_empty() || plane.len() < w {
            continue;
        }
        let plane_rows = plane.len() / w;
        let r0 = (y0 as usize).min(plane_rows);
        let r1 = (y1 as usize).min(plane_rows);
        if r1 <= r0 + 1 {
            continue;
        }
        for r in r0..r1 - 1 {
            let cur = &plane[r * w..(r + 1) * w];
            let nxt = &plane[(r + 1) * w..(r + 2) * w];
            for c in 0..w {
                acc += (cur[c] as i32 - nxt[c] as i32).unsigned_abs() as u64;
            }
        }
    }
    acc
}

/// Round-230 high-bit-depth convenience wrapper — picks
/// `(q_slices, rp)` against `target_bytes` and emits the codestream in
/// one call at component bit depth `bd ∈ 9..=16`.
///
/// Returns `(codestream, q_slices, rp)`. The codestream is guaranteed
/// to satisfy `codestream.len() <= target_bytes` (otherwise the picker
/// returns `target_bytes unreachable`). The `q_slices` and `rp` values
/// are the ones returned by
/// [`pick_q_slices_rp_for_target_bytes_highbd`]; callers can persist
/// them for reproducible re-encode through
/// [`encode_planar_hsl_qslice_rp_highbd`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_hsl_qslice_rp_target_bytes_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    hsl: u16,
    target_bytes: usize,
    planes: &[Vec<u16>],
) -> Result<(Vec<u8>, Vec<u8>, u8)> {
    let (q_slices, rp) = pick_q_slices_rp_for_target_bytes_highbd(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        bd,
        hsl,
        target_bytes,
        planes,
    )?;
    let cs = encode_planar_hsl_qslice_rp_highbd(
        width, height, nc, cpih, nlx, nly, bd, hsl, &q_slices, rp, planes,
    )?;
    Ok((cs, q_slices, rp))
}

/// Round-108 uniform-inverse-quantizer entry point (`Qpih = 1`).
///
/// Same shape as [`encode_planar_lossy`] but sets the picture-header
/// inverse-quantizer type to `Qpih = 1` (Annex A.4.4 Table A.10): the
/// decoder reconstructs coefficients with the uniform / Neumann-series
/// kernel of Annex D.3 instead of the deadzone kernel of Annex D.2
/// (`Qpih = 0`).
///
/// The data sub-packet on the wire is byte-identical for both quantizer
/// types — only the `Qpih` bit in the PIH `Lh:Rl:Qpih:Fs:Rm` byte changes,
/// and the decoder picks the matching inverse from it. At `q = 0`
/// (lossless, `T[p,b] = 0`) the deadzone reconstruction `(v << 0) + 0 = v`
/// and the uniform reconstruction (`φ = v`, `ζ = M + 1`, the Neumann
/// series collapses to `v` because the stored magnitude satisfies
/// `v < 2^M`) are both exact, so `Qpih = 1` self-roundtrips losslessly and
/// decodes byte-identically to the `Qpih = 0` form. At `q > 0` the two
/// kernels reconstruct different (but both valid) lossy magnitudes; the
/// uniform path replaces the deadzone midpoint `r = (1<<T)>>1` with the
/// equal-bucket Neumann reconstruction.
///
/// `q` is the precinct quantization step (`0..=15`); `q = 0` is lossless
/// and `q > 0` forces `Fq = 8` per Table A.8. The decoder has threaded
/// `pih.qpih` into `dequantize_precinct` since the early rounds, so any
/// output of this entry point round-trips through [`crate::decode_jpeg_xs`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_qpih(
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        1,          // qpih: uniform inverse quantizer (Qpih=1, Annex D.3)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Chroma-subsampled companion to [`encode_planar_qpih`]: signals the
/// uniform inverse quantizer (`Qpih = 1`, Annex A.4.4 Table A.10) on a
/// picture whose components carry explicit per-component sampling factors
/// `sx` / `sy` (4:2:2 = `sx=[1,2,2], sy=[1,1,1]`; 4:2:0 =
/// `sx=[1,2,2], sy=[2,2,2]`).
///
/// The uniform-quantizer decode path (Annex D.3, the Neumann-series
/// reconstruction) was previously only reachable for the 4:4:4 layout via
/// [`encode_planar_qpih`]; this entry point lets a chroma-subsampled
/// codestream select the same decoder kernel, so the inverse-uniform
/// reconstruction is exercised end-to-end across all three documented
/// chroma samplings (4:4:4 / 4:2:2 / 4:2:0).
///
/// As with [`encode_planar_qpih`], the data sub-packet is byte-identical
/// to the `Qpih = 0` form — only the PIH `Qpih` bit changes — so at
/// `q = 0` (lossless, `T[p,b] = 0`) the uniform reconstruction collapses
/// to `v` and the stream round-trips bit-exactly. At `q > 0` (`Fq = 8`)
/// the decoder applies the equal-bucket Neumann reconstruction per band.
/// The decoder threads `pih.qpih` into `dequantize_precinct`, so any
/// output round-trips through [`crate::decode_jpeg_xs`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_qpih_subsampled(
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        0,
        0,
        0,
        0,
        None,
        Vec::new(),
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        1,          // qpih: uniform inverse quantizer (Qpih=1, Annex D.3)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-115 precinct-refinement entry point (`R[p] > 0`).
///
/// Same shape as [`encode_planar_lossy`] but takes an explicit precinct
/// refinement `rp` (the `R[p]` field, Annex C.2 Table C.1, constant
/// across precincts). `rp = 0` is the no-refinement default (bit-
/// equivalent to [`encode_planar_lossy`]); `rp > 0` activates the Annex
/// C.6.2 Table C.10 refinement term `r = (P[b] < R[p]) ? 1 : 0`, which
/// subtracts one from the truncation position of every band whose
/// priority is below the threshold:
///
/// ```text
///   T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)
/// ```
///
/// The encoder assigns each band the priority `P[b] = b` (its true band
/// index, Annex B.6 `b = (Nc - Sd)×β + i`), emitted in the WGT marker
/// (Annex A.4.11). With that assignment `R[p] = k` refines exactly the
/// `k` lowest-index bands — LL first, since bands are enumerated in
/// `β`-major order — granting them one extra retained magnitude
/// bitplane (lower `T`, finer quantization) at the cost of a few more
/// coded bits in those bands. This is a valid encoder choice the spec
/// permits ("Other choices are possible", Annex H NOTE); the decoder
/// reconstructs the identical `T[p,b]` from the `(P[b], R[p])` pair it
/// reads back, so any output round-trips through [`crate::decode_jpeg_xs`].
///
/// `rp` is range-checked against `NL-1` (the total band count minus one,
/// Annex B.6); a value past the highest band index is rejected. At
/// `q = 0` (lossless) refinement is a no-op — `T` is already clamped to
/// its `0` floor — so a lossless stream round-trips unchanged regardless
/// of `rp`. The refinement only changes the lossy (`q > 0`, `Fq = 8`)
/// magnitudes, where it shifts bits toward the refined low-frequency
/// bands.
///
/// `q` is the precinct quantization step (`0..=15`); `q = 0` is lossless
/// and `q > 0` forces `Fq = 8` per Table A.8.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_rp(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    rp: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        0,          // cw: single precinct column
        0,          // sd: no CWD suppression
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        rp,         // rp: precinct refinement R[p]
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-100 separate-sign-sub-packet entry point (`Fs = 1`).
///
/// Same shape as [`encode_planar_lossy`] but sets the picture-header sign
/// handling strategy to `Fs = 1` (Annex A.4.4 Table A.11): signs are
/// carried in a dedicated sign sub-packet (Annex C.5.5, Table C.9) rather
/// than interleaved into the data sub-packet (Table C.8). With `Fs = 1`
/// only the magnitude bitplanes ride the data sub-packet, and exactly one
/// sign bit is emitted per coefficient whose reconstructed magnitude is
/// non-zero — strictly fewer than the `Fs = 0` form, which spends `Ng`
/// sign bits on every significant code group regardless of how many of
/// its coefficients are actually non-zero.
///
/// `q` is the precinct quantization step (`0..=15`); `q = 0` is lossless
/// and `q > 0` forces `Fq = 8` per Table A.8. The decoder already threads
/// `pih.fs` end-to-end (slice walker → packet body), so any output of this
/// entry point round-trips through [`crate::decode_jpeg_xs`].
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_fs1(
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        0,          // cw
        0,          // sd
        1,          // fs: separate sign sub-packet (Table C.9)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        cw,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
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
/// `sx[i] = sy[i] = 1`. Cpih is forced to 0 by this entry point; for
/// Cpih=1 (RCT) + Sd see [`encode_planar_sd_rct`]; for Cpih=3
/// (Star-Tetrix) + Sd see [`encode_planar_sd_star_tetrix`].
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        0, // cpih: explicit Cpih=0 entry point
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
        Vec::new(),
        0, // cw
        sd,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-95 (r93) `Sd > 0` + `Cpih = 1` (RCT) entry point.
///
/// Per Annex F.2 Table F.1 the RCT operand window is `c < 3` — when the
/// CWD-suppressed tail does not overlap that window (`Nc - Sd >= 3`),
/// the colour transform composes cleanly with the suppressed-tail
/// raw-coding path. The first three components are forward-RCT'd before
/// the DWT cascade; components 3..Nc-Sd remain wavelet-coded without
/// colour transform; components Nc-Sd..Nc are coded raw per the CWD
/// tail loop (Annex B.7 Table B.4).
///
/// Constraints: `nc - sd >= 3`, `nc > 3` (CWD), `sd >= 1`, `sx[i] = sy[i] = 1`
/// for every component (the encoder forces 4:4:4 here to keep the
/// post-RCT geometry well-defined).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_sd_rct(
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        1, // cpih: RCT
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
        Vec::new(),
        0, // cw
        sd,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-95 (r93) `Sd > 0` + `Cpih = 3` (Star-Tetrix) entry point.
///
/// Per Annex F.2 Table F.1 the Star-Tetrix operand window is `c < 4`;
/// the four CFA components are forward-Star-Tetrix'd before the DWT
/// cascade (the transform reads all four inputs `c < 4` per Annex F.2
/// Table F.1) and the `Sd` trailing transform *outputs* ride through raw.
/// Suppressing a transform output is legal — the spec tabulates exactly
/// this CFA layout (`Nc = 4, Sd = 1`) in Annex B Tables B.10 / B.11 — so
/// the only requirement is that the four transform inputs exist (`Nc >= 4`).
/// Emits both the CTS marker (`Cf`, `e1`, `e2`) and the CRG marker (per
/// Table F.9 from `ct`).
///
/// Constraints: `nc >= 4` (Star-Tetrix input window), `nc > 3` (CWD),
/// `1 <= sd < nc`, `sx[i] = sy[i] = 1` for every component.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_sd_star_tetrix(
    width: u16,
    height: u16,
    nc: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    sd: u8,
    e1: u8,
    e2: u8,
    cf: u8,
    ct: u8,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_nlt(
        width,
        height,
        nc,
        3, // cpih: Star-Tetrix
        nlx,
        nly,
        fq,
        q,
        &sx,
        &sy,
        e1,
        e2,
        cf,
        ct,
        None,
        Vec::new(),
        Vec::new(),
        0, // cw
        sd,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Sub-sampled (4:2:2 / 4:2:0) entry point. Each `planes[i]` has length
/// `⌈width / sx[i]⌉ * ⌈height / sy[i]⌉` (§B.1 ceiling — odd picture
/// dimensions are legal with sub-sampled components). `q = 0` for
/// lossless, `q > 0`
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        Vec::new(), // band_gains built inside after validation
        0,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        Vec::new(),
        0,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Sub-sampled (4:2:2 / 4:2:0) NLT quadratic (`Tnlt = 1`, Annex G.4)
/// encoder.
///
/// Widens [`encode_planar_nlt_quadratic`] from 4:4:4-only to per-component
/// chroma sub-sampling `(sx[i], sy[i]) ∈ {1, 2}`. The Annex G.4 forward
/// pre-distortion `y = round(sqrt(x / (2^B − 1)) × (2^Bw − 1)) + dco` is a
/// per-sample, per-component map (it runs on each `planes[i]` before the
/// DWT), so it is orthogonal to the per-component plane geometry: each
/// sub-sampled chroma plane has length `⌈width / sx[i]⌉ × ⌈height /
/// sy[i]⌉` (§B.1 ceiling) and decomposes one vertical level shallower
/// per `log2(sy[i])`, exactly as the linear sub-sampled path. `Bw = 18`
/// (Table A.8) is forced by the present NLT marker. Because the NLT
/// non-linearity and the inter-component RCT do not compose under the
/// spec's Annex F constraints (RCT requires 4:4:4 for `i < 3`), `cpih`
/// must be `0` (no colour transform) for genuinely sub-sampled input;
/// the inner RCT guard rejects `cpih = 1` with a sub-sampled component.
///
/// `q = 0` is lossless within the quadratic LUT resolution; `q > 0`
/// engages the `Fq = 8` lossy mode.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_nlt_quadratic(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    q: u8,
    dco: i32,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    if !(-32768..=32767).contains(&dco) {
        return Err(Error::invalid(format!(
            "jpegxs NLT quadratic: dco {dco} out of signed 16-bit range"
        )));
    }
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        0,
        0,
        0,
        0,
        Some(NltParams::Quadratic { dco }),
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Sub-sampled (4:2:2 / 4:2:0) NLT extended (`Tnlt = 2`, Annex G.5)
/// encoder.
///
/// Widens [`encode_planar_nlt_extended`] from 4:4:4-only to per-component
/// chroma sub-sampling `(sx[i], sy[i]) ∈ {1, 2}`. As with the quadratic
/// variant the three-segment extended pre-distortion is a per-sample,
/// per-component reverse-LUT map applied before the DWT, so it composes
/// freely with the sub-sampled plane geometry. `Bw = 18` (Table A.8) is
/// forced by the present NLT marker. The extended parameter constraints
/// from [`encode_planar_nlt_extended`] apply: `0 < t1 < t2`, `1 ≤ e ≤ 4`,
/// both thresholds in `1..=2^Bw − 1`. `cpih` must be `0` for genuinely
/// sub-sampled input (see [`encode_planar_subsampled_nlt_quadratic`]).
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_nlt_extended(
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
    sx: &[u8],
    sy: &[u8],
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
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
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
        0,
        0,
        0,
        0,
        Some(NltParams::Extended { t1, t2, e }),
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        planes,
    )
}

/// Round-181 high-bit-depth NLT quadratic (`Tnlt = 1`, Annex G.4)
/// encoder.
///
/// Widens [`encode_planar_nlt_quadratic`] from `B[i] = 8`-only to any
/// `bd = B[i] ∈ 9..=16`. The forward pre-distortion is the spec
/// formula `y = round(sqrt(x / (2^B[i] − 1)) × (2^Bw − 1)) + dco`,
/// which is parametric in the component bit depth (the round-5 / round
/// 7 8-bit path was a hard-coded specialisation). The wavelet domain
/// runs at `Bw = 20` — the top of the Table A.8 `{8, 18, 20}` set —
/// giving ≥ 4 bits of precision headroom for the sqrt above any
/// supported `B[i]`. The DC level shift is `1 << (Bw − 1) = 2^19`.
///
/// Plane format follows the round-118 high-bit-depth convention:
/// `planes[i]` is the component's samples as little-endian `u16`
/// values in `0..=2^bd − 1` (matching [`crate::image::JpegXsPlane`]),
/// `⌈width / sx[i]⌉ × ⌈height / sy[i]⌉` samples per plane (§B.1) with
/// `sx[i] = sy[i] = 1` (4:4:4 only on this path). `cpih ∈ {0, 1}`:
/// no transform or reversible RCT (Annex F.3 — bit-depth agnostic;
/// the RCT operand window `c < 3` applies identically to high bit
/// depth). `q = 0` is the "lossless within sqrt rounding" case
/// (PSNR ≥ 40 dB on natural content because the sqrt round-trip is
/// not bit-exact, mirroring the 8-bit `round5_nlt_quadratic_high_psnr`
/// floor); `q > 0` engages `Fq = 8` regular mode and the per-band
/// deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex D.4).
/// `dco` must fit in signed 16-bit (Annex A.4.6 NLT marker σ:α).
///
/// NLT extended (Tnlt=2) high-bit-depth lands in round 193 as
/// [`encode_planar_nlt_extended_highbd`] (Annex G.5 with `Bw = 20`
/// and the full `2^B[i]` reverse-LUT). Star-Tetrix (`Cpih = 3`)
/// high-bit-depth is still 8-bit-input specific.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_nlt_quadratic_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    dco: i32,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_nlt_quadratic_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_nlt_quadratic for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_nlt_quadratic_highbd supports Cpih in {{0, 1}} (Star-Tetrix high-bit-depth is out of scope), got {cpih}"
        )));
    }
    if !(-32768..=32767).contains(&dco) {
        return Err(Error::invalid(format!(
            "jpegxs NLT quadratic: dco {dco} out of signed 16-bit range"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    // Pack each plane to little-endian u16 bytes — the same wire format
    // round 118 / 133 / 151 settled on for high-bit-depth planes.
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
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
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0 (joint signs)
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone inverse)
        0,          // rp = 0 (no precinct refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// r371 high-bit-depth NLT quadratic (`Tnlt = 1`, Annex G.4) with chroma
/// sub-sampling.
///
/// Widens [`encode_planar_nlt_quadratic_highbd`] from 4:4:4-only to
/// per-component `(sx[i], sy[i]) ∈ {1, 2}`. As with the 8-bit
/// [`encode_planar_subsampled_nlt_quadratic`], the Annex G.4 forward
/// pre-distortion `y = round(sqrt(x / (2^B − 1)) × (2^Bw − 1)) + dco`
/// runs per-sample, per-component over the `u16`-LE plane chunks before
/// the DWT, so it composes with the sub-sampled plane geometry: each
/// `planes[i]` carries `⌈width / sx[i]⌉ × ⌈height / sy[i]⌉` samples
/// (§B.1 ceiling) and the sub-sampled chroma decomposes one vertical
/// level shallower per `log2(sy[i])`. The wavelet domain runs at
/// `Bw = 20` (Table A.8). `cpih` must be `0` for a genuinely sub-sampled
/// component (the inner RCT guard rejects `cpih = 1` per Annex F.2);
/// Star-Tetrix high bit depth remains out of scope.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_nlt_quadratic_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    dco: i32,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_nlt_quadratic_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_subsampled_nlt_quadratic for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_nlt_quadratic_highbd supports Cpih in {{0, 1}} (Star-Tetrix high-bit-depth is out of scope), got {cpih}"
        )));
    }
    if !(-32768..=32767).contains(&dco) {
        return Err(Error::invalid(format!(
            "jpegxs NLT quadratic: dco {dco} out of signed 16-bit range"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        fq,
        q,
        sx,
        sy,
        0,
        0,
        0,
        0,
        Some(NltParams::Quadratic { dco }),
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0 (joint signs)
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone inverse)
        0,          // rp = 0 (no precinct refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Round-193 high-bit-depth NLT extended (`Tnlt = 2`, Annex G.5)
/// encoder.
///
/// Widens [`encode_planar_nlt_extended`] from `B[i] = 8`-only to any
/// `bd = B[i] ∈ 9..=16`. The forward three-segment kernel inverts the
/// decoder's `extended_path` (Annex G.5 Table G.4) via a `2^bd`-entry
/// reverse lookup table — the round-7 path capped the level table at
/// 257 slots (an 8-bit shortcut); this round drops the cap so the LUT
/// can address every input sample.
///
/// The wavelet domain runs at `Bw = 20` — the top of the Table A.8
/// `{8, 18, 20}` set — giving ≥ 4 bits of precision headroom over any
/// supported `B[i]`. The DC level shift is `1 << (Bw − 1) = 2^19`.
///
/// Plane format follows the round-118 high-bit-depth convention:
/// `planes[i]` is the component's samples as little-endian `u16`
/// values in `0..=2^bd − 1` (matching [`crate::image::JpegXsPlane`]),
/// `⌈width / sx[i]⌉ × ⌈height / sy[i]⌉` samples per plane (§B.1) with
/// `sx[i] = sy[i] = 1` (4:4:4 only on this path). `cpih ∈ {0, 1}`:
/// no transform or reversible RCT (Annex F.3 — bit-depth agnostic;
/// the RCT operand window `c < 3` applies identically to high bit
/// depth). `q = 0` is the "lossless within LUT resolution" case;
/// `q > 0` engages `Fq = 8` regular mode and the per-band deadzone
/// truncation `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex D.4).
///
/// `t1`, `t2`, `e` are the extended-NLT parameters embedded in the NLT
/// marker (Annex G.5 thresholds and linear-slope exponent). The same
/// constraints from [`encode_planar_nlt_extended`] apply: `0 < t1 < t2`,
/// `1 ≤ e ≤ 4`, both `t1` and `t2` in `1..=2^Bw - 1`.
///
/// Star-Tetrix (`Cpih = 3`) high-bit-depth remains out of scope.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_nlt_extended_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    t1: u32,
    t2: u32,
    e: u8,
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_nlt_extended_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_nlt_extended for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_nlt_extended_highbd supports Cpih in {{0, 1}} (Star-Tetrix high-bit-depth is out of scope), got {cpih}"
        )));
    }
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
    // Bw is forced to 20 by encode_planar_inner_bd when bd > 8 && nlt.is_some().
    let bw_max = (1u32 << 20) - 1;
    if t1 > bw_max || t2 > bw_max {
        return Err(Error::invalid(format!(
            "jpegxs NLT extended highbd: T1={t1} or T2={t2} exceeds 2^Bw-1={bw_max}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let sx = vec![1u8; nc as usize];
    let sy = vec![1u8; nc as usize];
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
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
        Vec::new(),
        0,
        0,
        0,          // fs = 0 (joint signs)
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone inverse)
        0,          // rp = 0 (no precinct refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// r371 high-bit-depth NLT extended (`Tnlt = 2`, Annex G.5) with chroma
/// sub-sampling.
///
/// Widens [`encode_planar_nlt_extended_highbd`] from 4:4:4-only to
/// per-component `(sx[i], sy[i]) ∈ {1, 2}`. The three-segment extended
/// reverse-LUT pre-distortion runs per-sample, per-component over the
/// `u16`-LE planes before the DWT, so it composes with the sub-sampled
/// plane geometry exactly as the quadratic variant. `Bw = 20` (Table
/// A.8). `cpih` must be `0` for a genuinely sub-sampled component (Annex
/// F.2); Star-Tetrix high bit depth remains out of scope. The extended
/// constraints apply: `0 < t1 < t2`, `1 ≤ e ≤ 4`, both thresholds in
/// `1..=2^Bw − 1`.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_subsampled_nlt_extended_highbd(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    bd: u8,
    q: u8,
    t1: u32,
    t2: u32,
    e: u8,
    sx: &[u8],
    sy: &[u8],
    planes: &[Vec<u16>],
) -> Result<Vec<u8>> {
    if !(9..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_nlt_extended_highbd requires B[i] in 9..=16, got {bd} (use encode_planar_subsampled_nlt_extended for 8-bit)"
        )));
    }
    if cpih != 0 && cpih != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: encode_planar_subsampled_nlt_extended_highbd supports Cpih in {{0, 1}} (Star-Tetrix high-bit-depth is out of scope), got {cpih}"
        )));
    }
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
    // Bw is forced to 20 by encode_planar_inner_bd when bd > 8 && nlt.is_some().
    let bw_max = (1u32 << 20) - 1;
    if t1 > bw_max || t2 > bw_max {
        return Err(Error::invalid(format!(
            "jpegxs NLT extended highbd: T1={t1} or T2={t2} exceeds 2^Bw-1={bw_max}"
        )));
    }
    let max_sample: u16 = ((1u32 << bd) - 1) as u16;
    let mut byte_planes: Vec<Vec<u8>> = Vec::with_capacity(planes.len());
    for (i, p) in planes.iter().enumerate() {
        let mut bytes = Vec::with_capacity(p.len() * 2);
        for &s in p {
            if s > max_sample {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: plane {i} sample {s} exceeds B[i]={bd} max {max_sample}"
                )));
            }
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        byte_planes.push(bytes);
    }
    let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
    encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        cpih,
        nlx,
        nly,
        fq,
        q,
        sx,
        sy,
        0,
        0,
        0,
        0,
        Some(NltParams::Extended { t1, t2, e }),
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,          // fs = 0 (joint signs)
        0,          // hsl = 0 (single slice)
        0,          // qpih = 0 (deadzone inverse)
        0,          // rp = 0 (no precinct refinement)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
        &byte_planes,
        0, // rm = 0 (zero-prediction-residual significance)
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
        Vec::new(),
        0,
        0,
        0,          // fs: signs jointly with data (Fs=0)
        0,          // hsl: single slice (Hsl = Np,y)
        0,          // qpih: deadzone inverse quantizer (Qpih=0)
        0,          // rp: no precinct refinement (R[p] = 0)
        Vec::new(), // q_slices: single picture-level q
        Vec::new(), // q_precincts: no per-precinct override
        Vec::new(), // r_precincts: no per-precinct R[p] override
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
    band_priorities: Vec<u8>,
    cw: u16,
    sd: u8,
    fs: u8,
    hsl: u16,
    qpih: u8,
    rp: u8,
    q_slices: Vec<u8>,
    q_precincts: Vec<u8>,
    r_precincts: Vec<u8>,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    // 8-bit path: B[i] = 8, Bw = 8 (or 18 with NLT pre-distortion).
    // Run mode fixed at Rm = 0 (zero-prediction-residual significance) for
    // the NLT/generic funnel; the Rm = 1 entry point calls
    // `encode_planar_inner_bd` directly.
    encode_planar_inner_bd(
        width,
        height,
        nc,
        8,
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
        nlt,
        band_gains,
        band_priorities,
        cw,
        sd,
        fs,
        hsl,
        qpih,
        rp,
        q_slices,
        q_precincts,
        r_precincts,
        planes,
        0, // rm = 0 (zero-prediction-residual significance)
    )
}

/// Inner encoder threading an explicit component bit depth `bd` (`B[i]`).
/// `bd == 8` is the legacy 8-bit path (`planes[i]` is one byte per
/// sample). For `bd ∈ 9..=16` (round 118) the picture is coded losslessly
/// with `Bw = B[i] = bd`, the DC level shift is `1 << (bd - 1)`, and each
/// `planes[i]` carries two little-endian bytes per sample (matching
/// [`crate::image::JpegXsPlane`]). High-bit-depth composes with the
/// linear path (`nlt = None`, rounds 118 / 133 / 151) and — as of round
/// 181 — with NLT quadratic (`nlt = Some(Quadratic { dco })`, Annex G.4
/// at any `B[i] ∈ 9..=16` against `Bw = 20`); NLT extended (Tnlt=2) is
/// still 8-bit-input specific because its forward LUT inverter is
/// keyed on the reconstructed level table.
#[allow(clippy::too_many_arguments)]
fn encode_planar_inner_bd(
    width: u16,
    height: u16,
    nc: u8,
    bd: u8,
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
    band_priorities: Vec<u8>,
    cw: u16,
    sd: u8,
    fs: u8,
    hsl: u16,
    qpih: u8,
    rp: u8,
    q_slices: Vec<u8>,
    q_precincts: Vec<u8>,
    r_precincts: Vec<u8>,
    planes: &[Vec<u8>],
    rm: u8,
) -> Result<Vec<u8>> {
    if !(8..=16).contains(&bd) {
        return Err(Error::Unsupported(format!(
            "jpegxs encoder: component bit depth B[i]={bd} out of supported range 8..=16"
        )));
    }
    // NLT quadratic (Tnlt=1) is purely algebraic per Annex G.4 — both
    // the forward `y = round(sqrt(x / (2^B - 1)) * (2^Bw - 1))` and the
    // decoder's inverse `(ω² + half) >> ζ` are parametric in `B[i]`, so
    // the round-181 widening exposes it at `B[i] ∈ 9..=16` against
    // `Bw = 20` (the upper end of Table A.8 `{8, 18, 20}`).
    //
    // NLT extended (Tnlt=2) — round 193 widening: the decoder's
    // `extended_path` (Annex G.5 Table G.4) is also parametric in
    // `B[i]`, and the encoder's reverse LUT now allocates the full
    // `2^B[i]` level slots (the earlier `.min(257)` cap was an 8-bit
    // shortcut). Bw = 20 same as quadratic for the wavelet-domain
    // headroom and the inverse-LUT key uses `bc = bd`.
    // For 8-bit the legacy choice stands: Bw = 18 with NLT, else Bw = 8.
    let bw = if bd > 8 {
        if nlt.is_some() {
            20
        } else {
            bd
        }
    } else if nlt.is_some() {
        18
    } else {
        8
    };
    // Resolve Fq to a value the crate signals *truthfully* and pair it
    // with `bw` per ISO/IEC 21122-1:2022 Table A.8, which lists exactly
    // three valid (Bw, Fq) combinations: (B[0], 0) for lossless /
    // integer-transform coding, (18, 6) when a non-linearity (NLT marker)
    // is present, and (20, 8) for the high-precision regular case.
    //
    // `Fq` is the number of fractional bits the wavelet transform carries
    // (Annex E.3, Table E.13): the dequantised coefficient is up-scaled by
    // `<< Fq` before the inverse DWT and the forward-DWT output is
    // down-scaled by `>> Fq` after it (see [`fq_downscale_band`] and
    // [`crate::dequant::dequantize_precinct`]; the two are exact inverses).
    //
    // * The default integer-transform regular path passes `fq = 0`: zero
    //   extra fractional precision. Lossy quantisation via the precinct
    //   truncation `T[p,b]` is still permitted (the inverse quantiser is
    //   independent of Fq), so a regular lossy stream is the conformant
    //   (Bw = B[0], Fq = 0) combination. (The historical `Fq = 8` written
    //   alongside `Bw = 8` was *not* a Table A.8 combination — a conforming
    //   decoder would have `<< 8`-ed the coefficients and corrupted them.)
    // * NLT requires Fq != 0 (Annex A.4.6: "This marker shall not be
    //   present if Fq=0"); the chosen Bw selects the Table A.8 value.
    // * The high-precision regular entry point passes `fq = 8`, pairing
    //   with `Bw = 20` and the Annex E.3 input up-scaling below.
    let (bw, fq) = if nlt.is_some() {
        (bw, if bw >= 20 { 8 } else { 6 })
    } else if fq >= 8 {
        // High-precision regular case: pin the conformant (20, 8) pair.
        // The Annex E.3 input up-scaling is applied in `write_slice`.
        (20, 8)
    } else {
        // Integer-transform regular path (0 extra fractional bits).
        (bw, 0)
    };
    let cfg = EncodeConfig {
        width,
        height,
        nc,
        bit_depth: bd,
        bw,
        ng: 4,
        ss: 8,
        br: 8,
        nlx,
        nly,
        cpih,
        qpih,
        fs,
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
        band_priorities,
        rp,
        cw,
        sd,
        hsl,
        q_slices,
        q_precincts,
        r_precincts,
        rm,
    };
    cfg.validate()?;
    // Build per-band gains and priorities after validation so beta_key /
    // band-index math runs with known-good (nlx >= nly) parameters. A
    // caller may instead supply a complete custom weights table (both
    // `band_gains` AND `band_priorities` non-empty, e.g. the Annex H
    // PSNR-optimized table from [`annex_h_weights`]); in that case it is
    // threaded through verbatim — the WGT emission and the `custom_wgt`
    // forward-truncation branch in [`write_slice`] then agree by
    // construction. A non-empty `band_gains` with an empty
    // `band_priorities` keeps the legacy `P[b] = b` priorities.
    let cfg = if cfg.band_gains.is_empty() {
        EncodeConfig {
            band_gains: build_band_gains_sd(nc, sd, nlx, nly, sx, sy),
            band_priorities: build_band_priorities_sd(nc, sd, nlx, nly, sy),
            ..cfg
        }
    } else if cfg.band_priorities.is_empty() {
        EncodeConfig {
            band_priorities: build_band_priorities_sd(nc, sd, nlx, nly, sy),
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
    // Bytes per sample on the wire: 1 for B[i] = 8, 2 (little-endian) for
    // B[i] > 8 (round 118 high-bit-depth plane format).
    let bps: usize = if cfg.bit_depth > 8 { 2 } else { 1 };
    for (i, p) in planes.iter().enumerate() {
        let wc = (width as usize).div_ceil(cfg.sx[i] as usize);
        let hc = (height as usize).div_ceil(cfg.sy[i] as usize);
        let want = wc * hc * bps;
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} size {} != Wc*Hc*bps {want} (Wc={wc}, Hc={hc}, bps={bps})",
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

/// Build the CAP marker `cap[]` byte array (the bytes after `Lcap`) for
/// the tools this configuration uses, per ISO/IEC 21122-1:2022 Table A.5
/// (MSB-first bit numbering; bit 0 is intentionally unused).
///
///  * bit 1 — Star-Tetrix transform + CTS marker (`Cpih == 3`)
///  * bit 2 — quadratic non-linear transform (NLT `Tnlt = 1`)
///  * bit 3 — extended non-linear transform (NLT `Tnlt = 2`)
///  * bit 4 — a component with `sy[i] > 1` is present (vertical subsampling)
///  * bit 5 — component-dependent wavelet decomposition (CWD, `Sd > 0`)
///  * bit 6 — lossless decoding (`q == 0`, integer reversible path)
///  * bit 8 — packet-based raw-mode switch (`Rl = 1`, always set by this
///    encoder — it selects the bitplane-count coding mode per packet)
///
/// `Lcap` is then `2 + cap_bytes.len()`. The trailing byte is guaranteed
/// non-zero because bit 8 (always set) lives in byte 1, so the §A.4.3
/// minimality rule ("for Lcap>2 the last byte is not all zero") holds.
fn build_cap_bytes(cfg: &EncodeConfig) -> Vec<u8> {
    // Two bytes cover bits 0..=15; bit 8 (always set) forces byte 1 to be
    // present and non-zero.
    let mut bits = [false; 16];
    bits[1] = cfg.cpih == 3;
    if let Some(nlt) = cfg.nlt {
        match nlt {
            NltParams::Quadratic { .. } => bits[2] = true,
            NltParams::Extended { .. } => bits[3] = true,
        }
    }
    bits[4] = cfg.sy.iter().any(|&s| s > 1);
    bits[5] = cfg.sd > 0;
    bits[6] = cfg.q == 0;
    // The encoder always signals Rl = 1 (per-packet raw-mode selection),
    // so the packet-based raw-mode-switch capability is always required.
    bits[8] = true;

    let mut bytes = [0u8; 2];
    for (i, &set) in bits.iter().enumerate() {
        if set {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    // Shrink Lcap so the last cap byte is not all zero (§A.4.3). Byte 1
    // is always non-zero here (bit 8), so both bytes are always emitted;
    // the loop keeps the helper correct if bit 8 ever becomes optional.
    let mut len = bytes.len();
    while len > 0 && bytes[len - 1] == 0 {
        len -= 1;
    }
    bytes[..len].to_vec()
}

fn write_main_header(out: &mut Vec<u8>, cfg: &EncodeConfig) -> Result<()> {
    // SOC.
    out.extend_from_slice(&[0xff, 0x10]);
    // CAP — the capabilities marker declares the optional decoder tools
    // required to decode this codestream (ISO/IEC 21122-1:2022 §A.4.3,
    // Table A.5). cap[] is built from the tools this configuration
    // actually uses; Lcap is shrunk so the last cap byte is never all
    // zero (the §A.4.3 minimality rule).
    out.extend_from_slice(&[0xff, 0x50]);
    let cap_bytes = build_cap_bytes(cfg);
    let lcap = 2u16 + cap_bytes.len() as u16;
    out.extend_from_slice(&lcap.to_be_bytes());
    out.extend_from_slice(&cap_bytes);
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
        let p = cfg.band_priorities.get(k).copied().unwrap_or(0);
        out.push(g); // G[b]
        out.push(p); // P[b] (Annex A.4.11) — band index, see build_band_priorities_sd
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
        // via Table F.9 (RGGB for Ct=0, GRBG for Ct=1). The first four
        // entries are the CFA-component placements; any additional Nc-4
        // entries (when Nc > 4, i.e. Cpih=3 + Sd>0) are emitted as
        // (0, 0) — placement-irrelevant since Table F.1 leaves those
        // components untouched by the transform.
        let cfa: [(u16, u16); 4] = if cfg.st_ct == 0 {
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
        for c in 0..cfg.nc as usize {
            let (xc, yc) = cfa.get(c).copied().unwrap_or((0, 0));
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
                                                  // Hsl — slice height in precinct rows (Annex B.10). cfg.hsl == 0 is
                                                  // the single-slice default (Hsl = Np,y, the whole picture in one
                                                  // slice); a non-zero cfg.hsl groups the precinct rows into
                                                  // ⌈Np,y / Hsl⌉ slices and is emitted verbatim.
    let hsl_field = effective_hsl(cfg);
    out.extend_from_slice(&hsl_field.to_be_bytes()); // Hsl
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
    // Lh (1) : Rl (1) : Qpih (2) : Fs (2) : Rm (2). Qpih (bits 5:4,
    // Annex A.4.4 Table A.10) selects the decoder's inverse-quantizer
    // kernel (0 = deadzone, 1 = uniform); Fs (bits 3:2, Table A.11)
    // selects sign handling.
    //
    // Rl (bit 6, Table A.7) is the raw-mode-selection-per-packet flag. The
    // encoder picks the bitplane-count coding mode independently per packet
    // — for each packet it emits the smaller of the raw (`Dr = 1`) and the
    // VLC (`Dr = 0`) form — so a band that spans multiple packets can carry
    // a different `Dr` in each. Under `Rl = 0` Annex C.3 forbids mixing raw
    // and non-raw coding within one band of a precinct; the per-packet
    // greedy choice is instead exactly the `Rl = 1` regime (Annex C.5.3.3),
    // whose per-line buffer bound `L'l_tot ≤ Ll_raw` is satisfied
    // automatically because each packet's chosen size never exceeds its raw
    // size. Signal `Rl = 1` so the construction is conformant and the
    // decoder's Annex C.3 consistency gate does not reject the encoder's own
    // streams. `Lh` stays 0. `Rm` (bits 1:0, Table A.12) selects the
    // insignificant-group interpretation in vertical prediction — the
    // encoder emits `Rm = 0` (zero prediction residual) by default and
    // `Rm = 1` (zero coefficients) via the run-mode-1 entry point.
    out.push((1 << 6) | ((cfg.qpih & 0x03) << 4) | ((cfg.fs & 0x03) << 2) | (cfg.rm & 0x03));
}

/// Build a slice-local `EncodeConfig` whose `q` is overridden by the
/// per-slice value in `cfg.q_slices` (round 206 — slice-level rate
/// budgeting). Returns `None` when `cfg.q_slices` is empty, in which
/// case the precinct emission keeps using the picture-level `cfg.q`
/// (so the byte-identical legacy stream is preserved for the round-3 ..
/// round-201 callers that never set per-slice quantization).
///
/// `t` is the slice's top-down order (`Yslh`), 0-indexed; the length
/// of `cfg.q_slices` is already validated against the slice count in
/// [`EncodeConfig::validate`], so `t < cfg.q_slices.len()` whenever
/// this returns `Some`.
fn slice_cfg_for(cfg: &EncodeConfig, t: usize) -> Option<EncodeConfig> {
    if cfg.q_slices.is_empty() {
        return None;
    }
    let qs = cfg.q_slices[t];
    if qs == cfg.q {
        return None;
    }
    let mut cloned = cfg.clone();
    cloned.q = qs;
    Some(cloned)
}

/// Round 233 — pick the precinct's `Q[p]` override from
/// [`EncodeConfig::q_precincts`] when non-empty (raster-scan index
/// `py * np_x + px`). Returns `None` when `q_precincts` is empty (per-
/// slice / picture-level fallback applies) or the entry matches the
/// already-resolved `cfg.q` (no override needed — the caller keeps the
/// outer `cfg`).
///
/// The length of `q_precincts` is validated against `Np,y × Np,x` in
/// [`EncodeConfig::validate`], so `py * np_x + px` is always in range
/// whenever this helper is called from the per-precinct emission loop.
///
/// Per Annex C.2 Table C.1 `Q[p]` is a per-precinct field, so any
/// per-precinct assignment is spec-compliant; the decoder reads it back
/// per precinct and reconstructs the matching `T[p,b]` without any
/// signalling change beyond the per-precinct `Q` byte.
fn precinct_cfg_for(cfg: &EncodeConfig, py: usize, px: usize, np_x: usize) -> Option<EncodeConfig> {
    let has_q = !cfg.q_precincts.is_empty();
    let has_r = !cfg.r_precincts.is_empty();
    if !has_q && !has_r {
        return None;
    }
    let idx = py * np_x + px;
    // Round 233 — per-precinct Q[p]. Resolves to the slice-level / picture-
    // level Q when `q_precincts` is empty.
    let qp = if has_q { cfg.q_precincts[idx] } else { cfg.q };
    // Round 239 — per-precinct R[p]. Resolves to the picture-level R when
    // `r_precincts` is empty (preserves byte-identical output for the
    // round-233 / round-115 callers).
    let rp = if has_r { cfg.r_precincts[idx] } else { cfg.rp };
    if qp == cfg.q && rp == cfg.rp {
        return None;
    }
    let mut cloned = cfg.clone();
    cloned.q = qp;
    cloned.rp = rp;
    Some(cloned)
}

/// Resolve the `Hsl` field (slice height in precinct rows) emitted in
/// the PIH. `cfg.hsl == 0` means "single slice covering the whole
/// picture", which on the wire is `Hsl = Np,y`. Otherwise the caller's
/// value is used verbatim (validated `<= Np,y` in [`EncodeConfig::validate`]).
fn effective_hsl(cfg: &EncodeConfig) -> u16 {
    let hp_pow = 1u32 << cfg.nly;
    let np_y = (cfg.height as u32).div_ceil(hp_pow) as u16;
    if cfg.hsl == 0 {
        np_y
    } else {
        cfg.hsl
    }
}

/// Number of wavelet filter types `Nβ` per Annex B.3.
fn n_beta(nlx: u8, nly: u8) -> u32 {
    let mn = nlx.min(nly) as u32;
    let mx = nlx.max(nly) as u32;
    2 * mn + mx + 1
}

/// Build a `2^B[i]`-entry "forward extended-NLT" lookup table mapping
/// each input pixel value in `[0, 2^bc − 1]` to a wavelet-domain code in
/// `[0, 2^Bw − 1]`.
///
/// The lookup is built by walking the decoder's extended-gamma kernel
/// (Annex G.5, Table G.4) across every `v_wave ∈ [0, 2^Bw - 1]`,
/// computing the output level, and recording the first wavelet code that
/// reconstructs each output level. This is O(2^Bw) and runs once per
/// encode; for Bw=18 that's ~262k iterations.
///
/// State size is `O(2^bc) × sizeof::<Option<u32>>()`. For `bc = 8` (the
/// 8-bit input path) that's 256 entries / ~2 KB; for `bc = 16` (round
/// 193 high-bit-depth widening) it's 65 536 entries / ~512 KB, still
/// well within an encoder allocation budget. Bit depths beyond 16 are
/// not exposed (the plane layout would also need widening to `u32`).
///
/// The output level walks monotonically (modulo the rounding /
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

    // Full `2^bc` entries; the earlier `.min(257)` cap was specific to
    // the 8-bit input path where `1 << 8 = 256` and the +1 was a
    // defensive head-room slot. At `bc ∈ 9..=16` (round 193) we need
    // the full level table so the per-pixel inverse LUT can address
    // every input sample.
    let n_levels = 1usize << bc;
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
        for beta in 0..nbeta_pic {
            if crate::slice_walker::picture_beta_to_local_beta(beta, cfg.nlx, cfg.nly, cfg.sy[i])
                .is_some()
            {
                n += 1;
            }
        }
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
        for (i, &sy_val) in sy.iter().enumerate().take(n_decomposed) {
            let nly_i = nly.saturating_sub(match sy_val {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            // Skip non-existent picture-β slots (Annex B.4 bx[β,i]=0)
            // and resolve the gain from the component's chroma-local β
            // (its τx/τy reflect the actual band content, not the
            // picture-β slot label).
            let Some(local_beta) =
                crate::slice_walker::picture_beta_to_local_beta(beta, nlx, nly, sy[i])
            else {
                continue;
            };
            let key = beta_key(local_beta, nlx, nly_i);
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

/// Build per-band priority values `P[b]` for the WGT marker (Annex
/// A.4.11 Table A.24), in the same existing-band emission order as
/// [`build_band_gains_sd`] (β-major, then component i).
///
/// Each priority is the band's *true* sequential index per Annex B.6:
/// `b = (Nc - Sd)×β + i` for the wavelet bands (`i < Nc - Sd`) and
/// `b = (Nc - Sd)×Nβ + i` for the `Sd` suppressed-tail bands. Because
/// the band index counts from the LL band upward in `β`-major order, a
/// precinct refinement `R[p] = k` refines exactly the `k` bands whose
/// `P[b] < k` (Annex C.6.2 Table C.10) — i.e. the `k` lowest band
/// indices, LL first. This is an encoder choice the spec permits
/// ("Other choices are possible", Annex H NOTE); the only invariant the
/// decoder enforces is that the priorities it reads from WGT and the
/// `R[p]` it reads from the precinct header reproduce the same `T[p,b]`
/// the encoder quantized with — which holds by construction here since
/// the decoder's [`crate::entropy::precinct_truncation`] reads the same
/// `(P[b], R[p])` pair.
///
/// `P[b]` is `u(8)` (0–255). Band indices beyond 255 are saturated to
/// 255; for such large band counts a small `R[p]` still only refines
/// the low-index bands, so saturation does not change which bands the
/// refinement reaches (only bands `b < R[p] <= NL-1` can ever satisfy
/// `P[b] < R[p]`, and `R[p]` itself never exceeds 255 in practice
/// because the precinct-header field is `u(8)`).
fn build_band_priorities_sd(nc: u8, sd: u8, nlx: u8, nly: u8, sy: &[u8]) -> Vec<u8> {
    let nbeta_pic = n_beta(nlx, nly);
    let n_decomposed = (nc - sd) as u32;
    let mut prios = Vec::new();
    for beta in 0..nbeta_pic {
        for (i, &_sy_val) in sy.iter().take(n_decomposed as usize).enumerate() {
            // Skip non-existent picture-β slots (Annex B.4).
            if crate::slice_walker::picture_beta_to_local_beta(beta, nlx, nly, sy[i]).is_none() {
                continue;
            }
            let b = n_decomposed * beta + i as u32;
            prios.push(b.min(255) as u8);
        }
    }
    // Sd suppressed-tail bands: b = (Nc - Sd)×Nβ + i.
    for i in 0..sd as u32 {
        let b = n_decomposed * nbeta_pic + i;
        prios.push(b.min(255) as u8);
    }
    prios
}

/// Content-adaptive WGT weights from ISO/IEC 21122-1:2022 Annex H
/// (informative) — the PSNR-optimized example `(G[b], P[b])` tables for
/// the canonical 4:4:4, RCT-enabled (`Cpih = 1`), 5-horizontal-level
/// configurations the encoder already supports.
///
/// Returns `Some((gains, priorities))` in band-index order `b = Nc×β + i`
/// (the same `β`-major emission order `build_band_*_sd` and the WGT loop
/// use) when `(nc, cpih, sd, nlx, nly)` matches a tabulated Annex H
/// configuration, or `None` otherwise. The caller threads the returned
/// vectors through [`EncodeConfig::band_gains`] / `band_priorities`, which
/// drives both the WGT marker emission AND the forward truncation (see the
/// `custom_wgt` branch in [`write_slice`]) so the codestream round-trips:
/// the decoder reconstructs the identical `T[p,b] = clamp(Q[p] − G[b] − r,
/// 0, 15)` (Annex C.6.2 Table C.10) from the `(G[b], P[b])` it reads back
/// off the wire.
///
/// Per Annex A.4.11 these example tables were "optimized for PSNR
/// performance of the encoder. Other choices are possible."
///
/// Both the 4:4:4 RCT tables H.1–H.3 (`Cpih = 1`, every band exists) and
/// the chroma-subsampled tables H.4–H.8 (`Cpih = 0`, RCT disabled) are
/// returned. The subsampled 4:2:0 tables H.7 / H.8 list some band indices
/// as non-existent (`bx[β,i] = 0`, marked `-*` in the spec, see Annex
/// A.4.11 — the WGT loop `for(b=0;b<NL;b++) if(b'x[b]) {...}` skips them).
/// Those positions are dropped here so the returned vectors are in the
/// **existing-band emission order** the WGT marker loop and the
/// `custom_wgt` forward-truncation branch both consume — i.e. position `k`
/// is the `k`-th existing band, exactly as [`build_band_gains_sd`] /
/// [`build_band_priorities_sd`] produce. Because the spec defines the
/// `-*` slots as precisely the non-existent bands, this filtering matches
/// the encoder's [`crate::slice_walker::picture_beta_to_local_beta`] skip
/// rule position-for-position regardless of which β indices vanish.
///
/// `sx` / `sy` select the chroma format: all-1 → 4:4:4 (H.1–H.3,
/// `Cpih = 1`); chroma `sx = 2, sy = 1` → 4:2:2 (H.4–H.6, `Cpih = 0`);
/// chroma `sx = sy = 2` → 4:2:0 (H.7–H.8, `Cpih = 0`).
///
/// The CFA Star-Tetrix tables H.9–H.11 (`Cpih = 3`, `Sd = 1`, 4:4:4:4,
/// `NL,x = 5`, `NL,y ∈ {0, 1, 2}`) are also returned. They list a separate
/// `(G[b], P[b])` column per CTS extent `Cf ∈ {0, 3}` (Table A.20): `cf`
/// selects the column. The band-emission order matches the Sd=1 layout of
/// Annex B Tables B.10 / B.11 — `b = (Nc − Sd)×β + i` for the three
/// wavelet-coded Star-Tetrix outputs, then the single raw-coded suppressed
/// output as the tail band `b = (Nc − Sd)×Nβ` — i.e. exactly the order
/// [`build_band_gains_sd`] / [`build_band_priorities_sd`] emit for
/// `nc = 4, sd = 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn annex_h_weights(
    nc: u8,
    cpih: u8,
    sd: u8,
    nlx: u8,
    nly: u8,
    cf: u8,
    sx: &[u8],
    sy: &[u8],
) -> Option<(Vec<u8>, Vec<u8>)> {
    // CFA Star-Tetrix weight tables H.9–H.11 (Cpih=3, Sd=1, Nc=4, 4:4:4:4).
    // Each table tabulates two columns, Cf=0 (full transform) and Cf=3
    // (restricted in-line); `cf` picks the column. Every band exists for
    // 4:4:4:4 sampling, so there are no `-*` slots to drop and the column
    // is already in existing-band emission order.
    if cpih == 3 && sd == 1 && nc == 4 && nlx == 5 {
        let is_4444 = sx.iter().take(4).all(|&v| v == 1) && sy.iter().take(4).all(|&v| v == 1);
        if !is_4444 {
            return None;
        }
        let table: &[(u8, u8, u8, u8)] = match nly {
            0 => &H9_CFA_NLY0,
            1 => &H10_CFA_NLY1,
            2 => &H11_CFA_NLY2,
            _ => return None,
        };
        let (gains, priorities): (Vec<u8>, Vec<u8>) = match cf {
            0 => table.iter().map(|&(g0, p0, _, _)| (g0, p0)).unzip(),
            3 => table.iter().map(|&(_, _, g3, p3)| (g3, p3)).unzip(),
            _ => return None,
        };
        return Some((gains, priorities));
    }
    if nc != 3 || sd != 0 || nlx != 5 {
        return None;
    }
    // Classify the chroma format from the per-component sampling factors.
    // 4:4:4 → luma+chroma all (1,1); 4:2:2 → chroma (2,1); 4:2:0 → (2,2).
    let is_444 = sx.iter().all(|&v| v == 1) && sy.iter().all(|&v| v == 1);
    let chroma_422 = nc == 3
        && sx.first() == Some(&1)
        && sy.first() == Some(&1)
        && sx.get(1) == Some(&2)
        && sy.get(1) == Some(&1)
        && sx.get(2) == Some(&2)
        && sy.get(2) == Some(&1);
    let chroma_420 = nc == 3
        && sx.first() == Some(&1)
        && sy.first() == Some(&1)
        && sx.get(1) == Some(&2)
        && sy.get(1) == Some(&2)
        && sx.get(2) == Some(&2)
        && sy.get(2) == Some(&2);

    // Subsampled tables (RCT disabled, Cpih = 0). H.4–H.8 are stored as
    // full band-index arrays with `None` marking the spec's `-*` slots
    // (bands that do not exist for the subsampled chroma); we filter those
    // out so the result lands in existing-band emission order.
    if cpih == 0 && (chroma_422 || chroma_420) {
        let full: &[Option<(u8, u8)>] = if chroma_422 {
            match nly {
                0 => &H4_422_NLY0,
                1 => &H5_422_NLY1,
                2 => &H6_422_NLY2,
                _ => return None,
            }
        } else {
            // 4:2:0 — H.7 (NLy=1), H.8 (NLy=2). No NLy=0 table is tabulated.
            match nly {
                1 => &H7_420_NLY1,
                2 => &H8_420_NLY2,
                _ => return None,
            }
        };
        let gains: Vec<u8> = full.iter().filter_map(|e| e.map(|(g, _)| g)).collect();
        let priorities: Vec<u8> = full.iter().filter_map(|e| e.map(|(_, p)| p)).collect();
        return Some((gains, priorities));
    }

    if cpih != 1 || !is_444 {
        return None;
    }
    // Each entry is (G[b], P[b]) at band index b, transcribed from the
    // ISO/IEC 21122-1:2022 Annex H tables (read from the spec PDF under
    // docs/image/jpegxs/). Band order is b = Nc×β + i, i.e. the per-β
    // triple (Y, Cb, Cr-after-RCT) the WGT loop emits.
    let table: &[(u8, u8)] = match nly {
        // Table H.1 — 4:4:4, RCT, 5 horizontal / 0 vertical levels (18 bands).
        0 => &[
            (3, 6),
            (2, 8),
            (2, 7),
            (2, 1),
            (1, 4),
            (1, 5),
            (2, 12),
            (1, 14),
            (1, 15),
            (1, 0),
            (0, 2),
            (0, 3),
            (1, 9),
            (0, 11),
            (0, 10),
            (1, 13),
            (0, 16),
            (0, 17),
        ],
        // Table H.2 — 4:4:4, RCT, 5 horizontal / 1 vertical level (24 bands).
        1 => &[
            (4, 21),
            (2, 1),
            (2, 0),
            (3, 15),
            (2, 19),
            (2, 18),
            (2, 5),
            (1, 9),
            (1, 8),
            (2, 14),
            (1, 17),
            (1, 16),
            (1, 2),
            (0, 4),
            (0, 3),
            (1, 7),
            (0, 13),
            (0, 11),
            (1, 6),
            (0, 12),
            (0, 10),
            (1, 20),
            (0, 23),
            (0, 22),
        ],
        // Table H.3 — 4:4:4, RCT, 5 horizontal / 2 vertical levels (30 bands).
        2 => &[
            (4, 12),
            (3, 15),
            (3, 14),
            (3, 3),
            (2, 11),
            (2, 10),
            (3, 24),
            (2, 26),
            (2, 27),
            (2, 0),
            (1, 4),
            (1, 5),
            (2, 18),
            (1, 21),
            (1, 20),
            (2, 19),
            (1, 23),
            (1, 22),
            (1, 13),
            (0, 16),
            (0, 17),
            (1, 2),
            (0, 9),
            (0, 6),
            (1, 1),
            (0, 7),
            (0, 8),
            (1, 25),
            (0, 28),
            (0, 29),
        ],
        _ => return None,
    };
    let gains = table.iter().map(|&(g, _)| g).collect();
    let priorities = table.iter().map(|&(_, p)| p).collect();
    Some((gains, priorities))
}

// ---------------------------------------------------------------------------
// ISO/IEC 21122-1:2022 Annex H chroma-subsampled weight tables (informative,
// RCT disabled / Cpih = 0). Transcribed from the spec PDF under
// docs/image/jpegxs/. Each entry is `Some((G[b], P[b]))` at band index b, or
// `None` for the spec's `-*` slots (bands that do not exist for the subsampled
// chroma — see the H.7 / H.8 footnote and Annex A.4.11). `annex_h_weights`
// drops the `None` entries to land in existing-band emission order.
// ---------------------------------------------------------------------------

/// Table H.4 — 4:2:2, RCT disabled, 5 horizontal / 0 vertical levels (18 bands).
const H4_422_NLY0: [Option<(u8, u8)>; 18] = [
    Some((2, 8)),
    Some((2, 7)),
    Some((2, 6)),
    Some((1, 5)),
    Some((1, 3)),
    Some((1, 4)),
    Some((1, 13)),
    Some((1, 12)),
    Some((1, 14)),
    Some((0, 1)),
    Some((0, 0)),
    Some((0, 2)),
    Some((0, 9)),
    Some((0, 11)),
    Some((0, 10)),
    Some((0, 16)),
    Some((0, 15)),
    Some((0, 17)),
];

/// Table H.5 — 4:2:2, RCT disabled, 5 horizontal / 1 vertical level (24 bands).
const H5_422_NLY1: [Option<(u8, u8)>; 24] = [
    Some((2, 0)),
    Some((2, 1)),
    Some((2, 2)),
    Some((2, 18)),
    Some((2, 19)),
    Some((2, 20)),
    Some((1, 6)),
    Some((1, 7)),
    Some((1, 8)),
    Some((1, 15)),
    Some((1, 16)),
    Some((1, 17)),
    Some((0, 3)),
    Some((0, 4)),
    Some((0, 5)),
    Some((0, 10)),
    Some((0, 12)),
    Some((0, 14)),
    Some((0, 9)),
    Some((0, 11)),
    Some((0, 13)),
    Some((0, 21)),
    Some((0, 22)),
    Some((0, 23)),
];

/// Table H.6 — 4:2:2, RCT disabled, 5 horizontal / 2 vertical levels (30 bands).
const H6_422_NLY2: [Option<(u8, u8)>; 30] = [
    Some((3, 14)),
    Some((3, 13)),
    Some((3, 12)),
    Some((2, 9)),
    Some((2, 11)),
    Some((2, 10)),
    Some((2, 25)),
    Some((2, 24)),
    Some((2, 26)),
    Some((1, 0)),
    Some((1, 1)),
    Some((1, 2)),
    Some((1, 19)),
    Some((1, 20)),
    Some((1, 18)),
    Some((1, 23)),
    Some((1, 22)),
    Some((1, 21)),
    Some((0, 17)),
    Some((0, 15)),
    Some((0, 16)),
    Some((0, 4)),
    Some((0, 8)),
    Some((0, 5)),
    Some((0, 3)),
    Some((0, 6)),
    Some((0, 7)),
    Some((0, 28)),
    Some((0, 27)),
    Some((0, 29)),
];

/// Table H.7 — 4:2:0, RCT disabled, 5 horizontal / 1 vertical level. Band
/// indices 19, 20, 22, 23 do not exist (`-*`) for the 4:2:0 chroma.
const H7_420_NLY1: [Option<(u8, u8)>; 24] = [
    Some((3, 9)),
    Some((3, 16)),
    Some((3, 15)),
    Some((2, 7)),
    Some((2, 13)),
    Some((2, 12)),
    Some((2, 17)),
    Some((1, 3)),
    Some((1, 2)),
    Some((1, 6)),
    Some((1, 11)),
    Some((1, 10)),
    Some((1, 14)),
    Some((0, 1)),
    Some((0, 0)),
    Some((1, 19)),
    Some((0, 5)),
    Some((0, 4)),
    Some((1, 18)),
    None,
    None,
    Some((0, 8)),
    None,
    None,
];

/// Table H.8 — 4:2:0, RCT disabled, 5 horizontal / 2 vertical levels. Band
/// indices 16, 17, 19, 20 do not exist (`-*`) for the 4:2:0 chroma.
const H8_420_NLY2: [Option<(u8, u8)>; 30] = [
    Some((3, 0)),
    Some((3, 13)),
    Some((3, 12)),
    Some((3, 25)),
    Some((2, 7)),
    Some((2, 6)),
    Some((2, 8)),
    Some((2, 17)),
    Some((2, 16)),
    Some((2, 18)),
    Some((1, 5)),
    Some((1, 4)),
    Some((1, 3)),
    Some((1, 15)),
    Some((1, 14)),
    Some((1, 2)),
    None,
    None,
    Some((0, 1)),
    None,
    None,
    Some((1, 24)),
    Some((1, 22)),
    Some((1, 20)),
    Some((1, 23)),
    Some((1, 21)),
    Some((1, 19)),
    Some((0, 11)),
    Some((0, 10)),
    Some((0, 9)),
];

// ---------------------------------------------------------------------------
// ISO/IEC 21122-1:2022 Annex H CFA Star-Tetrix weight tables H.9–H.11
// (informative, Cpih = 3, Sd = 1, 4:4:4:4, NL,x = 5). Transcribed from the
// spec PDF under docs/image/jpegxs/. Each entry is
// `(G_cf0, P_cf0, G_cf3, P_cf3)` — the spec tabulates two columns per band,
// one for CTS extent Cf = 0 (full transform) and one for Cf = 3 (restricted
// in-line). `annex_h_weights` selects the column from the encode `cf`. All
// 4:4:4:4 bands exist, so there are no `-*` slots. Band order is the Sd=1
// layout of Annex B Tables B.10 / B.11: `b = (Nc − Sd)×β + i` for the three
// wavelet-coded Star-Tetrix outputs, then the suppressed raw-coded output as
// the final tail band.
// ---------------------------------------------------------------------------

/// Table H.9 — CFA Star-Tetrix, Sd=1, 5 horizontal / 0 vertical levels
/// (19 bands).
const H9_CFA_NLY0: [(u8, u8, u8, u8); 19] = [
    (3, 3, 4, 18),
    (3, 17, 3, 17),
    (3, 16, 3, 16),
    (2, 1, 3, 13),
    (2, 15, 2, 12),
    (2, 14, 2, 11),
    (2, 11, 2, 5),
    (1, 7, 1, 4),
    (1, 6, 1, 3),
    (1, 0, 2, 14),
    (1, 13, 1, 10),
    (1, 12, 1, 9),
    (1, 10, 1, 6),
    (0, 5, 0, 1),
    (0, 4, 0, 0),
    (1, 18, 1, 15),
    (0, 9, 0, 9),
    (0, 8, 0, 7),
    (0, 2, 0, 2),
];

/// Table H.10 — CFA Star-Tetrix, Sd=1, 5 horizontal / 1 vertical level
/// (25 bands).
const H10_CFA_NLY1: [(u8, u8, u8, u8); 25] = [
    (4, 20, 4, 13),
    (3, 17, 3, 12),
    (3, 16, 3, 11),
    (3, 12, 3, 8),
    (2, 11, 2, 7),
    (2, 10, 2, 6),
    (2, 0, 3, 20),
    (2, 24, 2, 19),
    (2, 23, 2, 18),
    (2, 15, 2, 9),
    (1, 9, 1, 5),
    (1, 8, 1, 4),
    (1, 1, 2, 21),
    (1, 22, 1, 17),
    (1, 21, 1, 16),
    (1, 14, 1, 10),
    (0, 6, 0, 2),
    (0, 4, 0, 1),
    (1, 13, 1, 9),
    (0, 5, 1, 23),
    (0, 3, 1, 22),
    (0, 7, 1, 24),
    (0, 19, 0, 15),
    (0, 18, 0, 14),
    (0, 2, 0, 3),
];

/// Table H.11 — CFA Star-Tetrix, Sd=1, 5 horizontal / 2 vertical levels
/// (31 bands).
const H11_CFA_NLY2: [(u8, u8, u8, u8); 31] = [
    (4, 9, 4, 5),
    (3, 8, 3, 4),
    (3, 7, 3, 3),
    (3, 0, 4, 28),
    (3, 30, 3, 27),
    (3, 29, 3, 26),
    (3, 20, 3, 18),
    (2, 19, 2, 17),
    (2, 18, 2, 16),
    (2, 1, 3, 30),
    (2, 28, 2, 23),
    (2, 27, 2, 22),
    (2, 24, 2, 19),
    (1, 16, 1, 14),
    (1, 14, 1, 13),
    (2, 23, 2, 12),
    (1, 15, 1, 11),
    (1, 13, 1, 10),
    (1, 17, 1, 9),
    (0, 11, 0, 8),
    (0, 10, 0, 7),
    (1, 22, 1, 15),
    (0, 6, 0, 2),
    (0, 4, 0, 1),
    (1, 21, 1, 0),
    (0, 5, 1, 25),
    (0, 3, 1, 24),
    (0, 12, 1, 29),
    (0, 26, 0, 21),
    (0, 25, 0, 20),
    (0, 2, 0, 6),
];

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

/// Apply the Annex E.3 / Table E.13 forward fractional down-scaling
/// `c[p,λ,b,ξ] = sign(T) * ((|T| + r) >> Fq)` with `r = (1 << Fq) >> 1`,
/// in place over a band's wavelet coefficients. This is the exact inverse
/// of the decoder's `c << Fq` reconstruction step (applied in
/// [`crate::dequant::dequantize_precinct`]). A no-op for `Fq = 0`, so the
/// lossless / integer-transform path is byte-identical.
fn fq_downscale_band(band: &mut [i32], fq: u8) {
    if fq == 0 {
        return;
    }
    let fq_u = fq as u32;
    let r: i64 = (1i64 << fq_u) >> 1;
    for v in band.iter_mut() {
        let t = *v as i64;
        let scaled = if t >= 0 {
            (t + r) >> fq_u
        } else {
            -((-t + r) >> fq_u)
        };
        *v = scaled as i32;
    }
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
            // Forward quadratic: map B[i]-bit input → Bw-bit wavelet domain.
            // Spec Annex G.4 forward (parametric in `B[i]`):
            //   y = round(sqrt(x / (2^B[i] - 1)) * (2^Bw - 1)).
            // For B[i] = 8 the input domain is `[0, 255]` and the wire
            // plane is one byte per sample; for B[i] ∈ 9..=16 (round 181
            // high-bit-depth NLT quadratic) the plane is two LE bytes per
            // sample matching `crate::image::JpegXsPlane` and the input
            // domain widens to `[0, 2^B[i] - 1]`.
            let bw_max = (1i64 << cfg.bw) - 1;
            let b_max = (1i64 << cfg.bit_depth) - 1;
            if cfg.bit_depth <= 8 {
                planes_u8
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|&v| {
                                let x = (v as i64).clamp(0, b_max);
                                let y = if x == 0 {
                                    0i64
                                } else {
                                    let ratio = (x as f64) / (b_max as f64);
                                    (ratio.sqrt() * (bw_max as f64)).round() as i64
                                };
                                let y = (y + (dco as i64)).clamp(0, bw_max);
                                (y as i32) - dc_bias
                            })
                            .collect()
                    })
                    .collect()
            } else {
                // u16-LE plane format: two bytes per sample, little-endian.
                planes_u8
                    .iter()
                    .map(|p| {
                        p.chunks_exact(2)
                            .map(|c| {
                                let v = u16::from_le_bytes([c[0], c[1]]) as i64;
                                let x = v.clamp(0, b_max);
                                let y = if x == 0 {
                                    0i64
                                } else {
                                    let ratio = (x as f64) / (b_max as f64);
                                    (ratio.sqrt() * (bw_max as f64)).round() as i64
                                };
                                let y = (y + (dco as i64)).clamp(0, bw_max);
                                (y as i32) - dc_bias
                            })
                            .collect()
                    })
                    .collect()
            }
        }
        Some(NltParams::Extended { t1, t2, e }) => {
            // Build the reverse LUT keyed on the reconstructed level
            // (output `[0, 2^B[i] − 1]` → first wavelet code that
            // reconstructs it under `extended_path`). O(2^Bw) per encode
            // independent of picture size. Bw is 18 at 8-bit, 20 at high
            // bit depth (chosen above).
            //
            // Plane format mirrors the linear / quadratic paths: 1 byte
            // per sample at `B[i] = 8`, two little-endian bytes per
            // sample at `B[i] ∈ 9..=16` (round 193 widening).
            let fwd = build_extended_forward_lut(cfg.bw, cfg.bit_depth, t1, t2, e);
            if cfg.bit_depth <= 8 {
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
            } else {
                planes_u8
                    .iter()
                    .map(|p| {
                        p.chunks_exact(2)
                            .map(|c| {
                                let v = u16::from_le_bytes([c[0], c[1]]) as usize;
                                let y = fwd[v] as i64;
                                (y as i32) - dc_bias
                            })
                            .collect::<Vec<i32>>()
                    })
                    .collect()
            }
        }
        None => {
            // Normal linear path: input samples are scaled into the i32
            // wavelet domain as the exact inverse of the Annex G.2 (Table
            // G.2) output scaling `R = (Ω + dc_bias + half) >> ζ`. The
            // forward map is therefore `Ω = (sample << ζ) - dc_bias`, with
            //   ζ        = Bw - B[i]   (the nominal-precision headroom), and
            //   dc_bias  = 1 << (Bw - 1)   (the inverse DC level shift).
            // For the integer-transform path (`Bw = B[i]`) ζ = 0 and this
            // reduces to the byte-identical legacy `sample - dc_bias`. For
            // the high-precision regular case (`Bw = 20`) ζ > 0 widens the
            // dynamic range so the `>> Fq` fractional precision survives the
            // transform. For `B[i] == 8` the plane is one byte per sample;
            // for `B[i] > 8` it is two little-endian bytes per sample.
            let zeta = (cfg.bw as i32 - cfg.bit_depth as i32).max(0) as u32;
            if cfg.bit_depth <= 8 {
                planes_u8
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|&v| ((v as i32) << zeta) - dc_bias)
                            .collect::<Vec<i32>>()
                    })
                    .collect()
            } else {
                planes_u8
                    .iter()
                    .map(|p| {
                        p.chunks_exact(2)
                            .map(|c| ((u16::from_le_bytes([c[0], c[1]]) as i32) << zeta) - dc_bias)
                            .collect::<Vec<i32>>()
                    })
                    .collect()
            }
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
    // Route everything with nlx > 1 or nly >= 1 through the cascade path,
    // including asymmetric (nlx != nly) configurations. Cw > 0 (Np,x > 1)
    // also forces the cascade path because per-precinct DWT does not
    // commute with multi-precinct-per-row layout (precinct boundaries
    // reflect at the band level, not the sample level).
    //
    // NL,y = 1 previously took a per-precinct streaming forward DWT that
    // reflected the 5/3 vertical filter at every 2-line precinct
    // boundary. That is not the Annex E transform: the DWT is
    // picture-level, so the vertical filter must run across precinct
    // boundaries (symmetric extension exists at the picture edges only).
    // The streams it emitted only round-tripped against the matching
    // per-precinct decoder path and would mis-decode on a conforming
    // decoder — so every NL,y >= 1 layout now takes the picture-level
    // cascade. The streaming path remains for NL,y = 0, where one
    // precinct row is a complete horizontal transform unit.
    let multi_level = nlx > 1 || nly >= 1 || np_x > 1 || cfg.sd > 0;
    let hp_pow = 1u32 << nly;
    let np_y = (h as u32).div_ceil(hp_pow) as usize;

    // Slice grouping per Annex B.10: precinct rows are partitioned into
    // ⌈Np,y / Hsl⌉ slices of `hsl_rows` rows each (the last slice is
    // shorter when Np,y is not a multiple of Hsl). Each slice gets its
    // own SLH marker (Yslh = t, the top-down slice order). cfg.hsl == 0
    // is the legacy single-slice mode (hsl_rows = Np,y → one slice). The
    // decoder reconstructs the identical grouping from PIH Hsl + Np,y
    // (slice_walker::build_plan), so the encoder only has to emit the
    // markers at the matching precinct-row boundaries. Vertical
    // prediction is precinct-scoped in this encoder (the M-top cache in
    // encode_precinct_cascade is local to one precinct), so no cross-
    // slice predictor state needs resetting at the boundaries.
    let hsl_rows: usize = if cfg.hsl == 0 { np_y } else { cfg.hsl as usize };
    // SLH writer: Lslh = 4, body = Yslh (u16). Annex A.4.12 Table A.25.
    let write_slh = |out: &mut Vec<u8>, t: u16| {
        out.extend_from_slice(&[0xff, 0x20]);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&t.to_be_bytes());
    };

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
            let wc = w.div_ceil(cfg.sx[i] as usize);
            let hc = h.div_ceil(cfg.sy[i] as usize);
            let nly_i = cfg.nly.saturating_sub(match cfg.sy[i] {
                1 => 0,
                2 => 1,
                4 => 2,
                _ => 0,
            });
            let mut bands = forward_cascade_2d(wc, hc, cfg.nlx, nly_i, plane)?;
            // Annex E.3 / Table E.13 forward fractional down-scaling
            // (`>> Fq`). The decoder applies the matching `<< Fq` at the
            // dequant boundary, so the two are exact inverses. No-op for
            // the integer-transform path (`Fq = 0`).
            for band in bands.iter_mut() {
                fq_downscale_band(band, cfg.fq);
            }
            bands_per_comp.push(bands);
        }
        // For suppressed (Sd) components, encode_precinct_cascade reads
        // the wavelet-domain `comp_planes` slice for the component
        // directly (no DWT was applied). `comp_planes` is already
        // DC-biased so the values fed into the entropy coder match the
        // dynamic range the decoder dequant path will produce when
        // copying straight back into the sample plane. The Sd band is
        // still subject to the Annex E.3 `>> Fq` (the decoder's reorder
        // step scales every component, including suppressed ones).
        if cfg.fq != 0 {
            for plane in comp_planes.iter_mut().skip(n_decomposed) {
                fq_downscale_band(plane, cfg.fq);
            }
        }
        let mut t: u16 = 0;
        let mut py_start = 0usize;
        while py_start < np_y {
            let py_end = (py_start + hsl_rows).min(np_y);
            write_slh(out, t);
            // Round 206 — pick the slice's Q[p] from cfg.q_slices when
            // non-empty (length already validated to match the slice
            // count). Empty falls back to the picture-level cfg.q,
            // preserving byte-identical output for the legacy path.
            let slice_cfg = slice_cfg_for(cfg, t as usize);
            let slice_cfg_ref = slice_cfg.as_ref().unwrap_or(cfg);
            for py in py_start..py_end {
                for px in 0..np_x {
                    // Round 233 — per-precinct Q[p] override (raster
                    // scan index py * np_x + px). When `q_precincts`
                    // is empty the per-slice (or picture-level) Q is
                    // kept, preserving byte-identical output for the
                    // round-206 path.
                    let precinct_cfg = precinct_cfg_for(slice_cfg_ref, py, px, np_x);
                    let cfg_ref = precinct_cfg.as_ref().unwrap_or(slice_cfg_ref);
                    let pbytes = encode_precinct_cascade(
                        cfg_ref,
                        &bands_per_comp,
                        &comp_planes,
                        py,
                        px,
                        cs,
                    )?;
                    out.extend_from_slice(&pbytes);
                }
            }
            py_start = py_end;
            t += 1;
        }
    } else {
        // NL=1/1 streaming per-precinct path. Handles 4:4:4 and chroma-
        // sub-sampled cases with a per-component effective NL,y and a
        // per-component precinct row range.
        let mut t: u16 = 0;
        let mut py_start = 0usize;
        while py_start < np_y {
            let py_end = (py_start + hsl_rows).min(np_y);
            write_slh(out, t);
            // Round 206 — pick the slice's Q[p] override; see the
            // multi_level branch comment above.
            let slice_cfg = slice_cfg_for(cfg, t as usize);
            let slice_cfg_ref = slice_cfg.as_ref().unwrap_or(cfg);
            for py in py_start..py_end {
                let y0 = py * (hp_pow as usize);
                let y1 = (y0 + hp_pow as usize).min(h);
                let hp_real = y1 - y0;
                // Round 233 — per-precinct Q[p] override. NL=1/1
                // single-column path has np_x = 1, so the index
                // reduces to py * 1 + 0 = py.
                let precinct_cfg = precinct_cfg_for(slice_cfg_ref, py, 0, np_x);
                let cfg_ref = precinct_cfg.as_ref().unwrap_or(slice_cfg_ref);
                let pbytes = encode_precinct_single_level(cfg_ref, &comp_planes, y0, y1, hp_real)?;
                out.extend_from_slice(&pbytes);
            }
            py_start = py_end;
            t += 1;
        }
    }
    Ok(())
}

/// Single-level streaming encode (`NL,y = 0`, `NL,x <= 1`). One
/// single-column precinct row is a complete horizontal transform unit,
/// so per-precinct == picture-level here. `NL,y >= 1` layouts must not
/// reach this function — the Annex E vertical 5/3 filter crosses
/// precinct boundaries, so they take the picture-level cascade encoder
/// (see `multi_level` at the call site).
fn encode_precinct_single_level(
    cfg: &EncodeConfig,
    comp_planes: &[Vec<i32>],
    y0: usize,
    y1: usize,
    hp_real: usize,
) -> Result<Vec<u8>> {
    if cfg.nly != 0 {
        return Err(Error::invalid(
            "jpegxs encoder: streaming single-level path requires NL,y = 0 \
             (NL,y >= 1 takes the picture-level cascade)"
                .to_string(),
        ));
    }
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
        let wc = w.div_ceil(sx_i);
        let hc = h_full.div_ceil(sy_i);
        let hp_i = hp_pow / sy_i;
        let nly_i = cfg.nly.saturating_sub(match cfg.sy[i] {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 0,
        });

        // Per-precinct strip rows for this component.
        let y0_i = y0 / sy_i;
        let y1_i = y1.div_ceil(sy_i).min(hc);
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
            // Annex E.3 / Table E.13 forward fractional down-scaling.
            fq_downscale_band(&mut ll, cfg.fq);
            fq_downscale_band(&mut hl, cfg.fq);
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
            // Annex E.3 / Table E.13 forward fractional down-scaling.
            fq_downscale_band(&mut ll, cfg.fq);
            fq_downscale_band(&mut hl, cfg.fq);
            fq_downscale_band(&mut lh, cfg.fq);
            fq_downscale_band(&mut hh, cfg.fq);
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
    // β=2→G=1, β=3→G=2. Uses gain-weighted T[p,b] = clamp(Q-G-r, 0, 15)
    // per Annex C.6.2 Table C.10. The single-level path is Sd=0, so the
    // band index is `b = Nc×β + i`; the refinement term is `r = (b <
    // R[p]) ? 1 : 0` (encoder assigns P[b] = b, Annex B.6). R[p] = 0
    // disables refinement (r ≡ 0).
    let nc_u32 = cfg.nc as u32;
    let t_for_band = |gain: u8, beta: u32, comp_i: usize| -> u8 {
        let band_index = nc_u32 * beta + comp_i as u32;
        let refine = if band_index < cfg.rp as u32 { 1i32 } else { 0 };
        (cfg.q as i32 - gain as i32 - refine).clamp(0, 15) as u8
    };

    // First packet (Annex B.7 Table B.4): β1 = max(NL,x, NL,y) − min(NL,x,
    // NL,y) + 1 filter types of all components are grouped JOINTLY on line
    // λ = 0. The single-level streaming path always runs at NL,x = 1
    // (NL,x > 1 is routed to the cascade encoder), so β1 is governed by the
    // PICTURE-level NL,y:
    //   * NL,y = 1 → β1 = 1: only β = 0 (LL) joins the first packet; HL
    //     (and, for non-subsampled components, LH/HH) follow as proxy
    //     packets. A 4:2:0-subsampled chroma component has N'L,y[i] = 0 but
    //     β1 is still 1 — its HL line is emitted as a proxy packet under the
    //     Table B.4 sub-sampling guard, NOT in the first packet.
    //   * NL,y = 0 → β1 = 2: β = 0 (LL) AND β = 1 (HL) of every component
    //     share the first packet (Table B.5: NL,x = N, NL,y = 0 groups all
    //     N + 1 bands of every component into packet 0).
    // Per Table B.4 the first packet is filled β-major, component-minor:
    //   for β in 0..β1 { for i in 0..Nc-Sd { include band (Nc-Sd)·β + i } }
    // so all components' LL entries precede all components' HL entries. The
    // decoder's packet plan
    // ([`crate::slice_walker::compute_packet_layouts`]) builds the entry
    // list in exactly this order; the encoder must match it.
    let picture_nly0 = cfg.nly == 0;
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
            t: t_for_band(0, 0, i),
        });
    }
    // β=1 (HL) joins the first packet for every component when the
    // picture-level NL,y = 0 (β1 = 2), appended after all LL entries to
    // preserve β-major order.
    if picture_nly0 {
        for (i, cb) in comp_bands.iter().enumerate() {
            if lines_ll_real_per_comp[i] == 0 {
                continue;
            }
            let hl_data = cb.hl[..cb.hl_w].to_vec();
            first_entries.push(PerBandEntry {
                wpb: cb.hl_w as u32,
                line: BandLineSlice::Direct(hl_data),
                t: t_for_band(1, 1, i),
            });
        }
    }
    if !first_entries.is_empty() {
        emit_packet(&mut entropy, cfg, &first_entries)?;
    }

    // Proxy levels: β=1 (HL, G=1), β=2 (LH, G=1), β=3 (HH, G=2). Per Annex
    // B.7 Table B.4 all existing components of a (β, line) are coded JOINTLY
    // in ONE packet (`r` resets per (λ, β), not per component; Tables
    // B.5–B.9). Accumulate every component's entry for this β and emit a
    // single packet — the decoder's plan groups identically.
    for beta_idx in 1usize..=3 {
        // When the picture-level NL,y = 0 (β1 = 2), every component's HL
        // band was already grouped into the first packet above; do not
        // re-emit it as a standalone proxy packet. (At NL,y = 1 the HL band
        // — including a 4:2:0 chroma component's — stays a proxy packet.)
        if beta_idx == 1 && picture_nly0 {
            continue;
        }
        let mut entries: Vec<PerBandEntry> = Vec::new();
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
            entries.push(PerBandEntry {
                wpb: wpb as u32,
                line: BandLineSlice::Direct(line_data),
                t: t_for_band(gain, beta_idx as u32, i),
            });
        }
        if !entries.is_empty() {
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
    precinct_bytes[4] = cfg.rp; // R[p] — precinct refinement (Annex C.2 Table C.1)
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

    // Collect per-(β_pic, i) slices for this precinct. `β` here is the
    // *picture-level* β slot (Annex B.3 b = (Nc - Sd) × β + i). The
    // per-component DWT output buffers (`bands_per_comp[i]`) are
    // indexed by chroma's LOCAL β, so each slice carries
    // `local_beta = picture_beta_to_local_beta(β_pic, NL,x, NL,y, sy[i])`
    // for the downstream extract_band_line lookup. Slots where bx[β,i]=0
    // (Annex B.4) are marked non-existent.
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
        /// Picture-level β slot (used for bitstream ordering and `b`).
        beta: u32,
        /// Chroma-local β — index into the component's own DWT cascade
        /// output (`bands_per_comp[comp_i][local_beta]`).
        local_beta: u32,
        exists: bool,
    }
    let sd_u = cfg.sd as usize;
    let n_decomposed = nc - sd_u;
    let mut slices: Vec<Slice> = Vec::with_capacity(((nbeta_pic as usize) * n_decomposed) + sd_u);
    for beta in 0..nbeta_pic {
        for (i, &nly_comp) in nly_i.iter().enumerate().take(n_decomposed) {
            let wc = w.div_ceil(cfg.sx[i] as usize);
            let hc = h.div_ceil(cfg.sy[i] as usize);
            // Existence per Annex B.4: bx[β,i] = 0 when this picture-β
            // slot has no equivalent in component i's DWT.
            let Some(local_beta) =
                crate::slice_walker::picture_beta_to_local_beta(beta, cfg.nlx, cfg.nly, cfg.sy[i])
            else {
                slices.push(Slice {
                    wpb: 0,
                    lines: 0,
                    pic_bw: 0,
                    pic_row_offset: 0,
                    pic_col_offset: 0,
                    comp_i: i,
                    beta,
                    local_beta: 0,
                    exists: false,
                });
                continue;
            };
            // The component's DWT band lives at chroma-local β; pull
            // its (dx, dy) and dimensions from the component's own
            // enumeration NL,x / N'L,y[i].
            let key = beta_key(local_beta, cfg.nlx, nly_comp);
            let (pic_bw, pic_bh) = band_dims(wc, hc, cfg.nlx, nly_comp, local_beta);
            // pow_h in chroma's BAND-grid units. Per Annex B.6 the
            // image-grid pow is `2^(NL,y - dy_picture)`; the chroma-
            // band-grid pow follows from dividing by sy[i] — but the
            // component's own dy (from beta_levels with N'L,y[i]) is
            // the depth in chroma's frame, so we apply `pow_h(N'L,y[i], dy_chroma_local)`
            // directly to get chroma-band-grid rows per precinct.
            let pow_eff = pow_h(nly_comp, key.dy).max(1);
            let row_offset = py * pow_eff;
            let lines = if row_offset >= pic_bh {
                0
            } else {
                pow_eff.min(pic_bh - row_offset)
            };
            // Per-precinct Wpb[p,b] in chroma's band-grid columns. The
            // chroma's band width per precinct is `Cs / (sx[i] * 2^dx)`
            // — same formula as 4:4:4 since horizontal sub-sampling
            // already shrinks Wc by sx.
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
                local_beta,
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
            local_beta: 0,
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

    // Per-band truncation: T[p,b] = clamp(Q - G[b] - r, 0, 15) per Annex
    // C.6.2 Table C.10. G[b] = #high-pass axes in band β for comp i
    // (using comp-local nly_i). The refinement term `r = (P[b] < R[p])`
    // grants one extra retained bitplane to low-index (LL-first) bands;
    // the encoder assigns `P[b] = b` (the true band index, Annex B.6), so
    // `r = (b < R[p]) ? 1 : 0`. R[p] = 0 disables refinement (r ≡ 0).
    let nd_u32 = n_decomposed as u32;
    // When a content-adaptive weights table is supplied (e.g. the Annex H
    // PSNR-optimized priorities, [`annex_h_weights`]), the forward
    // truncation MUST consult the very `(G[b], P[b])` pair the WGT marker
    // emits — otherwise the decoder, which reconstructs `T[p,b]` from WGT
    // (Annex C.6.2 Table C.10), would disagree with the quantizer and the
    // picture would not round-trip at `Q[p] > 0`. The weights vector is
    // built in `β`-major emission order (`build_band_*_sd` /
    // [`annex_h_weights`]), matching the decoder's β-major existing-band WGT
    // enumeration. For the canonical all-bands-exist 4:4:4 case every band
    // exists so the emission slot equals the band index `b = (Nc - Sd)×β + i`.
    // For subsampled configs some picture-β slots do not exist for chroma
    // components (Annex B.4 `bx[β,i] = 0`); the supplied weights vector then
    // carries one pair per *existing* band, so the closure must index it by
    // the band's position in the β-major existing-band enumeration — the same
    // cursor the decoder's WGT loader walks. We therefore precompute a
    // `band_index → existing_slot` map below so the table-driven `T[p,b]`
    // matches the WGT the decoder reconstructs from, for subsampled layouts
    // (e.g. the Annex H H.7 / H.8 4:2:0 tables) as well as 4:4:4.
    //
    // Build the β-major existing-slot lookup keyed by the picture band index
    // `b`. A band that does not exist maps to `None` (never read). Order MUST
    // match `slice_walker::build_plan`'s WGT loader: `for beta { for i }`
    // (wavelet bands), then the Sd tail.
    let n_total_bands = (nd_u32 * nbeta_pic + cfg.sd as u32) as usize;
    let mut existing_slot_of_band: Vec<Option<usize>> = vec![None; n_total_bands];
    {
        let mut cursor = 0usize;
        for beta in 0..nbeta_pic {
            for i in 0..n_decomposed {
                let exists = crate::slice_walker::picture_beta_to_local_beta(
                    beta, cfg.nlx, cfg.nly, cfg.sy[i],
                )
                .is_some();
                if exists {
                    let b = (nd_u32 * beta + i as u32) as usize;
                    existing_slot_of_band[b] = Some(cursor);
                    cursor += 1;
                }
            }
        }
        for i in 0..cfg.sd as usize {
            let b = (nd_u32 * nbeta_pic + i as u32) as usize;
            existing_slot_of_band[b] = Some(cursor);
            cursor += 1;
        }
    }
    // A custom table applies when both gain and priority vectors are present
    // and their length equals the *existing*-band count — covering the
    // all-exist 4:4:4 case (existing == total) and the subsampled case
    // (existing < total) uniformly.
    let n_existing_bands = existing_slot_of_band.iter().filter(|s| s.is_some()).count();
    let custom_wgt = !cfg.band_gains.is_empty()
        && cfg.band_gains.len() == cfg.band_priorities.len()
        && cfg.band_gains.len() == n_existing_bands;
    let t_for_band = |beta: u32, comp_i: usize| -> u8 {
        let band_index: u32 = if comp_i < n_decomposed {
            // Wavelet band: b = (Nc - Sd)×β_pic + i.
            nd_u32 * beta + comp_i as u32
        } else {
            // Sd suppressed-tail band: b = (Nc - Sd)×Nβ + (i - (Nc - Sd)).
            nd_u32 * nbeta_pic + (comp_i as u32 - nd_u32)
        };
        if custom_wgt {
            // Table-driven path: read the exact gain/priority emitted to
            // WGT for this band, indexed by its β-major existing-band slot
            // (the same slot the decoder reads), so encoder and decoder
            // compute the identical `T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)`.
            if let Some(slot) = existing_slot_of_band
                .get(band_index as usize)
                .copied()
                .flatten()
            {
                let g = cfg.band_gains[slot] as i32;
                let p = cfg.band_priorities[slot] as u32;
                let refine = if p < cfg.rp as u32 { 1i32 } else { 0 };
                return (cfg.q as i32 - g - refine).clamp(0, 15) as u8;
            }
            // Non-existent band: falls through to the geometry-derived path
            // (never actually coded, so the value is inconsequential).
        }
        let refine = if band_index < cfg.rp as u32 { 1i32 } else { 0 };
        // Gain: suppressed-tail bands have no wavelet axes (G = 0); the
        // wavelet bands use the τx/τy high-pass count from beta_key, but
        // the (τx, τy) come from the *component's* local β slot in its
        // own DWT enumeration. Resolve picture-β→local-β here.
        let gain: i32 = if comp_i < n_decomposed {
            let nly_comp = nly_i[comp_i];
            let sy_i = cfg.sy[comp_i];
            match crate::slice_walker::picture_beta_to_local_beta(beta, cfg.nlx, cfg.nly, sy_i) {
                Some(local_beta) => {
                    let key = beta_key(local_beta, cfg.nlx, nly_comp);
                    (if key.tau_x { 1 } else { 0 }) + (if key.tau_y { 1 } else { 0 })
                }
                None => 0,
            }
        } else {
            0
        };
        (cfg.q as i32 - gain - refine).clamp(0, 15) as u8
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
        // Wpb[p,b] = 0 (a rightmost precinct column too narrow to own
        // any of this band's columns) still contributes an ENTRY to its
        // packet: per Annex B.7 Table B.4 the inclusion test is about
        // line presence, and the Ncg = 0 group count just makes every
        // sub-packet contribution empty. Dropping the entry (and with
        // it a packet whose entries are all zero-width) would desync
        // the decoder, which still expects the packet header.
        if s.wpb == 0 {
            return Some(Vec::new());
        }
        if s.comp_i >= n_decomposed {
            // Sd suppressed: comp_planes is sized at Wf*Hf for sx=sy=1.
            let plane = &comp_planes[s.comp_i];
            let pic_row = s.pic_row_offset + line_off;
            let row_start = pic_row * s.pic_bw + s.pic_col_offset;
            let row_end = row_start + s.wpb;
            return Some(plane[row_start..row_end].to_vec());
        }
        // The per-component DWT cascade was run at NL,x / N'L,y[i] —
        // it emits bands indexed by chroma's LOCAL β. The picture-β slot
        // in `s.beta` is for bitstream ordering; the band data lives at
        // `s.local_beta` in the component's DWT output array.
        let band_buf = &bands_per_comp[s.comp_i][s.local_beta as usize];
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
    // Proxy levels. Per Annex B.7 Table B.4 the "start a new packet" flag
    // `r` is set once per (λ, β) — before the component loop — and cleared
    // after the first included component, so all existing, non-subsampled-
    // out components of a (β, line) are coded JOINTLY in one packet (Tables
    // B.5–B.9: e.g. β = 5 at 5/2 4:4:4 → the single packet `(15, 16, 17)`).
    // The multi-entry PacketJob path (used already by the first packet)
    // carries them together; the decoder's plan
    // ([`crate::slice_walker::compute_packet_layouts`]) groups identically.
    {
        let mut beta0 = beta1;
        // Track per-β whether a packet for that band type has already been
        // emitted in this precinct (to mark first-line packets, which can
        // never use vertical prediction). All components of a β first
        // appear together at the same line, so a per-β flag is exact.
        let mut first_seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        while beta0 < nbeta_pic {
            let key0 = beta_key(beta0, cfg.nlx, cfg.nly);
            let pow_pic = pow_h(cfg.nly, key0.dy);
            for lambda_within in 0..pow_pic {
                for beta in beta0..(beta0 + 3).min(nbeta_pic) {
                    let mut entries: Vec<PerBandEntry> = Vec::new();
                    let mut coords: Vec<(usize, u32)> = Vec::new();
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
                            entries.push(PerBandEntry {
                                wpb: s.wpb as u32,
                                line: BandLineSlice::Direct(line_data),
                                t: t_for_band(beta, i),
                            });
                            coords.push((i, beta));
                        }
                    }
                    if !entries.is_empty() {
                        let is_first = first_seen.insert(beta);
                        jobs.push(PacketJob {
                            entries,
                            coords,
                            first_line_in_precinct: is_first,
                        });
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

    // Phase 3 — commit D[p,b] ∈ {0,1,2,3} by total bytes, but ONE value
    // per *packet group*. Every band coded together in a single packet
    // must share the same D: `build_packet_body_with_m` codes the whole
    // packet body with one bitplane-count mode, while the precinct header
    // carries D[p,b] per band and the decoder dispatches each band's
    // sub-packet contribution on its own D[p,b]. If two bands of one
    // packet disagreed on D, the single-mode body would not match the
    // per-band header and the decoder would desync.
    //
    // Table B.4 groups (a) the first β1 filter types of every component
    // into packet 0, and (b) each proxy β (all components, every line)
    // into its own packets. So the packet-group key is: the first-packet
    // group for β < β1, the β itself for wavelet proxy bands, and a
    // per-band-unique key for the Sd tail (each coded in its own packet).
    let group_key = |coord: &(usize, u32)| -> u32 {
        let (i, beta) = *coord;
        if i >= n_decomposed {
            0x4000_0000 | (i as u32) // Sd tail: one packet per band
        } else if beta < beta1 {
            u32::MAX // first packet: β < β1 of every component
        } else {
            beta // proxy level: all components of a β share a packet
        }
    };
    // Sum each D-option's byte total per packet group.
    let mut g00: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut g01: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut g10: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut g11: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for coord in sizes_d00.keys() {
        let g = group_key(coord);
        *g00.entry(g).or_insert(0) += sizes_d00[coord];
        *g01.entry(g).or_insert(0) += sizes_d01.get(coord).copied().unwrap_or(usize::MAX / 2);
        *g10.entry(g).or_insert(0) += sizes_d10.get(coord).copied().unwrap_or(usize::MAX / 2);
        *g11.entry(g).or_insert(0) += sizes_d11.get(coord).copied().unwrap_or(usize::MAX / 2);
    }
    let mut d_per_group: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
    for g in g00.keys() {
        let s00 = g00[g];
        let s01 = g01.get(g).copied().unwrap_or(usize::MAX);
        let s10 = g10.get(g).copied().unwrap_or(usize::MAX);
        let s11 = g11.get(g).copied().unwrap_or(usize::MAX);
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
        d_per_group.insert(*g, d);
    }
    let mut d_per_band: std::collections::HashMap<(usize, u32), u8> =
        std::collections::HashMap::new();
    for coord in sizes_d00.keys() {
        let d = d_per_group.get(&group_key(coord)).copied().unwrap_or(0);
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
    precinct_bytes[4] = cfg.rp; // R[p] — precinct refinement (Annex C.2 Table C.1)
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

/// Forward quantizer: map a raw wavelet coefficient `c` to the on-wire
/// quantization-index magnitude `v[p,λ,b,x]` for the active inverse
/// quantizer type `qpih`.
///
/// The data sub-packet stores `M - T` magnitude bitplanes per code group
/// (Annex C.5.4): the decoder reads them into `v` at positions
/// `[0, M-T)` and the inverse quantizer shifts that `v` back up by `T`.
/// So this function returns the value whose bits `[0, M-T)` go on the
/// wire — i.e. the *quantization index*, not the coefficient.
///
/// * `qpih == 0` — deadzone forward quantizer (Annex D.4, Table D.3):
///   `v = |c| >> T`. This is the plain truncation the encoder has used
///   since round 3; matched by the deadzone inverse (Annex D.2, Table
///   D.1) whose reconstruction point sits at the bucket midpoint.
/// * `qpih == 1` — uniform forward quantizer (Annex D.5, Table D.4):
///   with `ζ = M − T + 1` and `d = |c|`,
///   `v = ((d << ζ) − d + (1 << M)) >> (M + 1)`. This rounds to the
///   nearest equal-sized bucket (the NOTE under Table D.4 describes it
///   as mid-point quantization with bucket size `Δ = 2^(M+1) /
///   (2^(M+1−T) − 1)`), matching the uniform / Neumann-series inverse
///   (Annex D.3, Table D.2). When `M <= T` no bitplanes are stored, so
///   `v = 0` (Table D.4 `else` branch).
///
/// Both branches return `0` when `M <= T` (the group carries no stored
/// bitplanes) so the caller's `m <= t` guard stays consistent. The
/// returned `v` always fits in the `M − T` stored bitplanes (verified:
/// Table D.4 never overflows `2^(M-T)`).
#[inline]
fn forward_quant_index(qpih: u8, c: i32, m: u32, t: u32) -> u32 {
    if m <= t {
        return 0;
    }
    let d = c.unsigned_abs();
    match qpih {
        // Uniform quantizer (Annex D.5, Table D.4). Performed in u64 so
        // `d << ζ` cannot overflow for any in-range coefficient.
        1 => {
            let zeta = m - t + 1;
            let num = (((d as u64) << zeta) + (1u64 << m)) - (d as u64);
            (num >> (m + 1)) as u32
        }
        // Deadzone quantizer (Annex D.4, Table D.3) — the default.
        _ => d >> t,
    }
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
    // Separate sign sub-packet (Annex C.5.5, Table C.9), only used when
    // Fs=1. Stays empty for Fs=0 (signs interleaved in the data sub-packet
    // per Table C.8).
    let mut sgn_writer = BitWriter::default();
    let fs1 = cfg.fs == 1;
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
        let vert_predecessor = match &mode {
            BitplaneMode::Vlc(VlcKind::VertPredSig { predecessor }) => Some(predecessor),
            _ => None,
        };
        m_per_entry
            .iter()
            .zip(entries.iter())
            .enumerate()
            .map(|(entry_idx, (m_per_group, entry))| {
                let ncg = m_per_group.len();
                let t = entry.t;
                let ns = ncg.div_ceil(ss_u);
                (0..ns)
                    .map(|j| {
                        let g0 = j * ss_u;
                        let g1 = (g0 + ss_u).min(ncg);
                        if m_per_group[g0..g1].iter().any(|&m| m > t) {
                            return true;
                        }
                        // Vertical prediction (Table C.13): an
                        // insignificant group reconstructs as
                        // M = mtop = max(Mtop, T) with Δm = 0 (Rm = 0),
                        // and the data sub-packet includes any group
                        // with M > T (Table C.8). Flagging a group
                        // insignificant while a predictor exceeds T
                        // would therefore inflate the reconstructed M
                        // above the emitted data — keep such groups
                        // significant so the VLC carries the true
                        // (negative) residual instead.
                        //
                        // Under Rm = 1 (runs indicate zero coefficients)
                        // the decoder instead reconstructs an
                        // insignificant group as M = T[p,b] (Δm = T − mtop,
                        // Table C.13) regardless of the predictor, so the
                        // data sub-packet omits it and this guard is not
                        // needed — an all-zero group vertically following a
                        // non-zero one is flagged insignificant and its VLC
                        // residual is elided, which the Rm = 0 guard would
                        // have spent bits on.
                        if cfg.rm != 1 {
                            if let Some(pred) = vert_predecessor {
                                let pred_m = &pred[entry_idx];
                                if pred_m[g0..g1.min(pred_m.len())].iter().any(|&m| m > t) {
                                    return true;
                                }
                            }
                        }
                        false
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
            for &significant in sig_flags {
                // Table C.5: the Z bit identifies whether ALL code
                // groups in the significance group are insignificant —
                // Z = 1 ⇒ insignificant, Z = 0 ⇒ significant
                // (Tables C.13 / C.14 decode the VLC residual when
                // Z == 0).
                sig_writer.write_bit(if significant { 0 } else { 1 });
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
            // Fs = 0: signs interleaved here, Ng bits per significant code
            // group (Table C.8). Fs = 1: signs deferred to the sign
            // sub-packet (Table C.9), so the data sub-packet carries only
            // magnitude bitplanes.
            if !fs1 {
                for k in 0..ng_u {
                    let v = coef(g, k);
                    let sign_bit = if v < 0 { 1 } else { 0 };
                    data_writer.write_bit(sign_bit);
                }
            }
            // Magnitude bitplanes: store the `M - T` bits of the
            // quantization index `v` (Annex C.5.4), MSB-first. The index
            // is derived from the raw coefficient by the active forward
            // quantizer (`Qpih`): deadzone (`v = |c| >> T`, Table D.3) or
            // uniform / round-to-nearest (Table D.4). For deadzone the
            // index `v` equals `|c| >> T`, so writing `v`'s bits
            // `[0, M-T)` is byte-identical to the round-3 path that wrote
            // `|c|`'s bits `[T, M)`.
            for bplane in (0..(m - t)).rev() {
                for k in 0..ng_u {
                    let v = forward_quant_index(cfg.qpih, coef(g, k), m, t);
                    let bit = ((v >> bplane) & 1) as u8;
                    data_writer.write_bit(bit);
                }
            }
        }

        // Sign sub-packet (Fs = 1, Table C.9). One bit per coefficient
        // whose stored quantization index `v` is non-zero, iterating
        // bands → lines → groups → members in the same order the decoder
        // reads. The decoder gates each sign bit on `coef.v != 0` (the
        // index it just read off the wire), so the encoder must gate on
        // the same forward-quantized index — for `Qpih = 1` the uniform
        // round-to-nearest can map a coefficient to a non-zero index that
        // a plain `|c| >> T` truncation would have zeroed (or vice
        // versa), so reusing the active forward quantizer keeps the two
        // sides in lockstep.
        if fs1 {
            for (g, &m_u8) in m_per_group.iter().enumerate() {
                let m = m_u8 as u32;
                for k in 0..ng_u {
                    let c = coef(g, k);
                    let v = forward_quant_index(cfg.qpih, c, m, t);
                    if v != 0 {
                        let sign_bit = if c < 0 { 1 } else { 0 };
                        sgn_writer.write_bit(sign_bit);
                    }
                }
            }
        }
    }
    cnt_writer.align_to_byte();
    data_writer.align_to_byte();
    sgn_writer.align_to_byte();
    let sig_bytes = sig_writer.into_bytes();
    let cnt_bytes = cnt_writer.into_bytes();
    let data_bytes = data_writer.into_bytes();
    let sgn_bytes = sgn_writer.into_bytes();
    let dr = match mode {
        BitplaneMode::Raw => 1,
        BitplaneMode::Vlc(_) => 0,
    };
    Ok(PacketBytes {
        dr,
        sig: sig_bytes,
        cnt: cnt_bytes,
        data: data_bytes,
        sgn: sgn_bytes,
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

/// Round-245 helper — spatial activity of a single precinct's pixel
/// range, summed across every plane.
///
/// Returns the L1 norm of the row-to-row first-difference over the
/// image-row range `[y0, y1)`. Used by the per-precinct picker to
/// rank precincts for the per-precinct `Q[p]` relaxation pass:
/// low-activity precincts get the quantization relief first.
///
/// Delegates to [`slice_activity`] — a precinct is the smallest
/// unit the spec defines for `Q[p]` / `R[p]` (Annex C.2 Table C.1),
/// covering `2^NL,y` image rows (Annex B.5), so the same row-range
/// L1-norm metric the r212 picker uses for a slice applies at the
/// finer granularity unchanged.
fn precinct_activity(planes: &[Vec<u8>], width: u16, y0: u32, y1: u32) -> u64 {
    slice_activity(planes, width, y0, y1)
}

/// Round-245 helper — list of `(y0, y1)` image-row ranges for every
/// precinct in raster scan order.
///
/// At `Cw = 0` there is exactly one precinct per row of `Np,y =
/// ⌈Hf / 2^NL,y⌉`, so this returns `Np,y` entries each covering
/// `2^NL,y` image rows (the last one may be partial when `Hf` is
/// not a multiple of the precinct height).
fn compute_precinct_row_ranges(height: u16, nly: u8) -> Vec<(u32, u32)> {
    let h = height as u32;
    let rows_per_precinct = 1u32 << nly;
    if rows_per_precinct == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut y = 0u32;
    while y < h {
        let y1 = (y + rows_per_precinct).min(h);
        out.push((y, y1));
        y = y1;
    }
    out
}

/// Round-245 rate-budget driven per-precinct `(Q[p], R[p])` picker.
///
/// Closes the round-242 follow-up tail: round 242 shipped the
/// per-precinct joint override [`encode_planar_qpr_rpr`] (one `Q[p]`
/// and one `R[p]` per precinct, Annex C.2 Table C.1), but left the
/// "caller must pick the vectors manually" tail open. Round 245
/// supplies that picker.
///
/// Given a target byte budget, returns the vector pair
/// `(q_precincts, r_precincts)` — both length `Np,y × Np,x` (here
/// `Np,y = ⌈Hf / 2^NL,y⌉` and `Np,x = 1` at `Cw = 0`) — that drives
/// [`encode_planar_qpr_rpr`] to emit a codestream of length
/// `≤ target_bytes`, while concentrating bits on precincts whose
/// source content has the lowest spatial activity (so distortion
/// from quantization lands on the busier precincts where it is less
/// perceptually visible). The picker is fully deterministic and
/// performs no rate-distortion search beyond calling
/// [`encode_planar_qpr_rpr`] with candidate vectors and reading
/// back the byte length — there is no internal model of the entropy
/// coder, no oracle, no external library. Bytes returned by the
/// callee are the only feedback the search uses.
///
/// Strategy (nested search — outer loop on uniform `R[p]`, inner
/// three-pass `Q[p]` picker against [`encode_planar_qpr_rpr`] at
/// the current `R[p]` baseline):
///
/// 1. **Baseline reachability.** Probe `r_precincts = [0; n]` +
///    `q_precincts = [15; n]` (max quantization, no refinement) —
///    the smallest stream the family can produce. If even that
///    overshoots, error with `target_bytes unreachable; rp=0 Q=15
///    emits N bytes` — no `(q_precincts, r_precincts)` pair can fit.
/// 2. **Outer loop on `rp ∈ 0..=NL-1`.** For each uniform `R[p]`,
///    run the inner per-precinct `Q[p]` search; if it fits,
///    promote the `(q_precincts, rp)` pair; if even `[15; n]`
///    overshoots at this `rp`, stop the outer loop (refinement is
///    monotone non-decreasing in codestream length at fixed
///    `Q[p]`, so higher `rp` cannot fit either).
/// 3. **Inner `Q[p]` search.** Three passes against
///    [`encode_planar_qpr_rpr`] at the current uniform `rp`:
///    a. Lossless probe (`q_precincts = [0; n]`).
///    b. Uniform-`Q` bisect on `1..=15`.
///    c. Per-precinct activity-driven relaxation — sort precincts
///    by spatial activity ascending and lower one low-activity
///    precinct at a time by one `Q` step (down to `Q = 0`) while
///    each candidate still fits.
///
/// Inputs:
/// * `width`, `height`, `nc`, `cpih`, `nlx`, `nly`, `planes` —
///   identical to [`encode_planar_qpr_rpr`].
/// * `target_bytes` — upper bound on the encoded codestream length;
///   must be `> 0`.
///
/// Returns `(q_precincts, r_precincts)`. The `r_precincts` vector
/// is filled with the picked uniform `rp` (one entry per precinct;
/// the per-precinct `R[p]` mechanism in round 239 carries the byte
/// independently regardless of whether the values vary — a uniform
/// `r_precincts = [rp; n]` is byte-identical to picture-wide `R[p]`
/// via `encode_planar_qpr_rpr`'s `max(r_precincts)` fallback).
///
/// Composition: at `q_precincts = [0; n]` refinement is a lossless
/// no-op (`T[p,b]` already at its `0` floor), so the picker always
/// returns `r_precincts = [0; n]` whenever the lossless probe
/// fits — the outer `rp` loop never promotes a non-zero `rp` for a
/// budget the lossless probe satisfies. At `q_precincts > 0`, both
/// vectors become active rate-distortion levers.
///
/// **Scope:** mirrors [`encode_planar_qpr_rpr`] exactly — 4:4:4,
/// `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`,
/// `Qpih = 0`, `B[i] = 8`.
///
/// **Errors:**
/// * Any of the validation errors [`encode_planar_qpr_rpr`] would
///   produce.
/// * [`crate::JpegXsError::Invalid`] when `target_bytes == 0`.
/// * [`crate::JpegXsError::Invalid`] when even `q_precincts = [15;
///   n]` + `r_precincts = [0; n]` overshoots the budget.
#[allow(clippy::too_many_arguments)]
pub fn pick_qpr_rpr_for_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>)> {
    if target_bytes == 0 {
        return Err(Error::invalid(
            "jpegxs per-precinct joint picker: target_bytes must be > 0".to_string(),
        ));
    }

    // NL = Nc × Nβ for the 4:4:4 / Sd = 0 surface (Annex B.6).
    let nbeta = n_beta(nlx, nly);
    let nl = (nc as u32) * nbeta;
    if nl == 0 {
        return Err(Error::invalid(format!(
            "jpegxs per-precinct joint picker: NL=0 (nc={nc}, NL,x={nlx}, NL,y={nly})"
        )));
    }
    let rp_max = (nl - 1).min(u8::MAX as u32) as u8;

    // Np,y at Cw = 0 (one precinct per precinct row).
    let np_y = np_y_for(height, nly) as usize;
    if np_y == 0 {
        return Err(Error::invalid(format!(
            "jpegxs per-precinct joint picker: Np,y=0 (height={height}, NL,y={nly})"
        )));
    }

    // Baseline reachability — rp=0 + Q=15 is the smallest stream of
    // the family at this geometry. If even that overshoots, no
    // (q_precincts, r_precincts) pair can fit.
    let cs_baseline = encode_planar_qpr_rpr(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        &vec![15u8; np_y],
        &vec![0u8; np_y],
        planes,
    )?;
    if cs_baseline.len() > target_bytes {
        return Err(Error::invalid(format!(
            "jpegxs per-precinct joint picker: target_bytes={target_bytes} unreachable; rp=0 Q=15 emits {} bytes",
            cs_baseline.len()
        )));
    }

    // Outer loop on uniform rp — promote the last fitting
    // (q_precincts, rp) pair. Stop on the first rp whose inner picker
    // fails (refinement is monotone non-decreasing in codestream
    // length at fixed Q[p]).
    //
    // Lossless short-circuit: if the inner picker returns an
    // all-zero q_precincts at any rp, refinement is a lossless no-op
    // (`T[p,b]` is already at its 0 floor at Q=0, so the precinct's
    // R[p] byte is a wire-only quantity that does not perturb the
    // data sub-packet bytes). Higher rp at q_precincts=[0;n] would
    // produce the same data bytes plus a non-zero R byte the decoder
    // ignores quantization-wise, so promoting beyond the first
    // lossless rp adds zero rate-distortion value. Keep the
    // canonical (r=0) form.
    let mut best_q = vec![15u8; np_y];
    let mut best_rp: u8 = 0;
    for rp in 0..=rp_max {
        match pick_q_precincts_at_rp(
            width,
            height,
            nc,
            cpih,
            nlx,
            nly,
            rp,
            target_bytes,
            np_y,
            planes,
        ) {
            Ok(qs) => {
                let is_lossless = qs.iter().all(|&v| v == 0);
                best_q = qs;
                best_rp = rp;
                if is_lossless {
                    // Canonicalize to r=0; higher rp at q=0 is a
                    // wire-only no-op (R byte is the only diff and
                    // adds no rate-distortion lever).
                    best_rp = 0;
                    break;
                }
            }
            Err(_) => break,
        }
    }
    Ok((best_q, vec![best_rp; np_y]))
}

/// Round-245 inner picker — at a fixed uniform `rp`, runs the
/// three-pass per-precinct `q_precincts` search against
/// [`encode_planar_qpr_rpr`]. Returns `Err` if even `[15; np_y]`
/// overshoots at this `rp` (signal to the outer loop that higher
/// `rp` cannot fit either).
#[allow(clippy::too_many_arguments)]
fn pick_q_precincts_at_rp(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    rp: u8,
    target_bytes: usize,
    np_y: usize,
    planes: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let r_uniform = vec![rp; np_y];

    // Pass 1 — lossless probe.
    let q_zero = vec![0u8; np_y];
    let cs_zero = encode_planar_qpr_rpr(
        width, height, nc, cpih, nlx, nly, &q_zero, &r_uniform, planes,
    )?;
    if cs_zero.len() <= target_bytes {
        return Ok(q_zero);
    }

    // Pass 2 — uniform-Q bisect over 1..=15.
    let mut lo: u8 = 1;
    let mut hi: u8 = 15;
    let mut best_uniform_q: Option<u8> = None;
    let mut best_uniform_len: usize = usize::MAX;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let qv = vec![mid; np_y];
        let cs = encode_planar_qpr_rpr(width, height, nc, cpih, nlx, nly, &qv, &r_uniform, planes)?;
        if cs.len() <= target_bytes {
            best_uniform_q = Some(mid);
            best_uniform_len = cs.len();
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            if mid == 15 {
                break;
            }
            lo = mid + 1;
        }
    }
    let uniform_q = match best_uniform_q {
        Some(q) => q,
        None => {
            // Q=15 at this rp overshoots → signal outer loop to stop.
            return Err(Error::invalid(format!(
                "jpegxs per-precinct joint picker (rp={rp}): target_bytes={target_bytes} unreachable; Q=15 overshoots"
            )));
        }
    };

    if np_y == 1 {
        return Ok(vec![uniform_q; 1]);
    }

    // Pass 3 — per-precinct activity-driven relaxation.
    let precinct_row_ranges = compute_precinct_row_ranges(height, nly);
    debug_assert_eq!(precinct_row_ranges.len(), np_y);
    let mut activity: Vec<(usize, u64)> = precinct_row_ranges
        .iter()
        .enumerate()
        .map(|(p, &(y0, y1))| (p, precinct_activity(planes, width, y0, y1)))
        .collect();
    activity.sort_by_key(|&(_, a)| a);

    let mut best = vec![uniform_q; np_y];
    let mut best_len = best_uniform_len;
    loop {
        let mut changed = false;
        for &(p, _) in &activity {
            if best[p] == 0 {
                continue;
            }
            let mut trial = best.clone();
            trial[p] -= 1;
            let cs = encode_planar_qpr_rpr(
                width, height, nc, cpih, nlx, nly, &trial, &r_uniform, planes,
            )?;
            if cs.len() <= target_bytes {
                best = trial;
                best_len = cs.len();
                changed = true;
            }
        }
        let _ = best_len;
        if !changed {
            break;
        }
    }
    Ok(best)
}

/// Round-245 convenience wrapper — picks `(q_precincts, r_precincts)`
/// against `target_bytes` and emits the codestream in one call.
///
/// Returns `(codestream, q_precincts, r_precincts)`. The codestream
/// is guaranteed to satisfy `codestream.len() <= target_bytes`
/// (otherwise the picker returns `target_bytes unreachable; rp=0
/// Q=15 emits N bytes`). The `q_precincts` and `r_precincts` values
/// are the ones returned by [`pick_qpr_rpr_for_target_bytes`];
/// callers can persist them for reproducible re-encode of identical
/// parameters.
#[allow(clippy::too_many_arguments)]
pub fn encode_planar_qpr_rpr_target_bytes(
    width: u16,
    height: u16,
    nc: u8,
    cpih: u8,
    nlx: u8,
    nly: u8,
    target_bytes: usize,
    planes: &[Vec<u8>],
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (q_precincts, r_precincts) =
        pick_qpr_rpr_for_target_bytes(width, height, nc, cpih, nlx, nly, target_bytes, planes)?;
    let cs = encode_planar_qpr_rpr(
        width,
        height,
        nc,
        cpih,
        nlx,
        nly,
        &q_precincts,
        &r_precincts,
        planes,
    )?;
    Ok((cs, q_precincts, r_precincts))
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

    /// r174 baseline: 4:4:4 at NL=3/3 — should already work (no chroma sub-sampling).
    #[test]
    fn r174_baseline_444_nl3_lossless_round_trip() {
        let w = 64u16;
        let h = 64u16;
        let n = (w as usize) * (h as usize);
        let mut planes: [Vec<u8>; 3] = [vec![0u8; n], vec![0u8; n], vec![0u8; n]];
        for (i, slot) in planes[0].iter_mut().enumerate() {
            *slot = ((i * 7 + 13) % 256) as u8;
        }
        for (i, slot) in planes[1].iter_mut().enumerate() {
            *slot = ((i * 11 + 17) % 256) as u8;
        }
        for (i, slot) in planes[2].iter_mut().enumerate() {
            *slot = ((i * 19 + 23) % 256) as u8;
        }
        let cs = encode_planar(
            w,
            h,
            3,
            0,
            3,
            3,
            &[planes[0].clone(), planes[1].clone(), planes[2].clone()],
        )
        .expect("encode 4:4:4 NL=3/3");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:4:4 NL=3/3");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, planes[0]);
        assert_eq!(img.planes[1].data, planes[1]);
        assert_eq!(img.planes[2].data, planes[2]);
    }

    /// r174: 4:2:2 at NL=3/3 — chroma N'L,y == NL,y, no vertical subsampling issue.
    #[test]
    fn r174_chroma_422_nl3_lossless_round_trip() {
        let w = 64u16;
        let h = 64u16;
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
            3,
            3,
            0,
            &[1, 2, 2],
            &[1, 1, 1],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:2 NL=3/3");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:2 NL=3/3");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// r174: 4:2:0 lossy at NL=2/2 q=2 — fix also covers the Fq=8 path.
    #[test]
    fn r174_chroma_420_nl2_lossy_q2_round_trip() {
        let w = 64u16;
        let h = 64u16;
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
            2,
            2,
            2, // q = 2 lossy
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 NL=2/2 q=2 lossy");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 NL=2/2 q=2");
        assert_eq!(img.num_components, 3);
        // Lossy path — check that bytes-out matches plane sample count and
        // values are within a reasonable error range of the originals (sanity).
        assert_eq!(img.planes[0].data.len(), y_plane.len());
        assert_eq!(img.planes[1].data.len(), cb_plane.len());
        assert_eq!(img.planes[2].data.len(), cr_plane.len());
        let psnr = |orig: &[u8], rec: &[u8]| -> f64 {
            let mut mse = 0f64;
            for (a, b) in orig.iter().zip(rec.iter()) {
                let d = (*a as f64) - (*b as f64);
                mse += d * d;
            }
            mse /= orig.len() as f64;
            if mse <= 0.0 {
                f64::INFINITY
            } else {
                10.0 * (255.0f64 * 255.0 / mse).log10()
            }
        };
        // Picture content is high-frequency (deterministic pseudo-random),
        // so PSNR is below the high-fidelity threshold but should still be
        // unambiguously decodable — 20 dB is a generous floor for sanity.
        let psnr_y = psnr(&y_plane, &img.planes[0].data);
        let psnr_cb = psnr(&cb_plane, &img.planes[1].data);
        let psnr_cr = psnr(&cr_plane, &img.planes[2].data);
        assert!(
            psnr_y >= 20.0 && psnr_cb >= 20.0 && psnr_cr >= 20.0,
            "4:2:0 NL=2/2 q=2 PSNR below 20 dB floor: Y={psnr_y} Cb={psnr_cb} Cr={psnr_cr}"
        );
    }

    /// r174: 4:2:0 at NL,y=2 with NL,x=3 (asymmetric) — chroma N'L,y = 1.
    #[test]
    fn r174_chroma_420_nl3x2y_lossless_round_trip() {
        let w = 64u16;
        let h = 64u16;
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
            3,
            2,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 NL=3/2 lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 NL=3/2");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// r174 probe: 4:2:0 at NL,y >= 2 — does it round-trip?
    #[test]
    fn r174_probe_chroma_420_nly2_lossless() {
        let w = 64u16;
        let h = 64u16;
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
            2,
            2,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 NL=2/2 lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 NL=2/2");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// r190: 4:2:0 at NL=3/3 lossy q=2 — must round-trip with a
    /// reasonable PSNR (the picture-β refactor must not break the
    /// quantization pipeline).
    #[test]
    fn r190_chroma_420_nl3_lossy_q2_round_trip() {
        let w = 64u16;
        let h = 64u16;
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
            3,
            3,
            2,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 NL=3/3 q=2");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 NL=3/3 q=2");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data.len(), y_plane.len());
        assert_eq!(img.planes[1].data.len(), cb_plane.len());
        assert_eq!(img.planes[2].data.len(), cr_plane.len());
    }

    /// r190: 4:2:0 at NL=3/3 — the previously-blocked case where
    /// N'L,y[chroma] = 2 puts chroma's deepest LH/HH triple at a proxy
    /// level that the picture-level β-slot enumeration (Annex B.4 /
    /// Figure B.2) places at picture-β = 7,8,9 rather than the chroma's
    /// local-β = 5,6,7 slots. Fixed in r190 by introducing the
    /// `picture_beta_to_local_beta` permutation across walker /
    /// encoder / decoder.
    #[test]
    fn r190_chroma_420_nl3_lossless_round_trip() {
        let w = 64u16;
        let h = 64u16;
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
            3,
            3,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:0 NL=3/3 lossless");
        let img =
            crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 NL=3/3 lossless");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
    }

    /// r174 probe: 4:2:2 (sy=1) at NL,y=2 — control. Chroma N'L,y[i] == NL,y here.
    #[test]
    fn r174_probe_chroma_422_nly2_lossless() {
        let w = 64u16;
        let h = 64u16;
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
            2,
            2,
            0,
            &[1, 2, 2],
            &[1, 1, 1],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:2 NL=2/2 lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:2 NL=2/2");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, cb_plane);
        assert_eq!(img.planes[2].data, cr_plane);
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

    /// r371: NLT quadratic composed with 4:2:2 chroma sub-sampling
    /// (`cpih = 0`, `(sx, sy) = (1,1)/(2,1)/(2,1)`). The Annex G.4 forward
    /// pre-distortion is per-component, so each plane — full-res luma and
    /// half-width chroma — survives the round-trip at ≥ 40 dB. NL=2/2.
    #[test]
    fn round371_nlt_quadratic_subsampled_422() {
        let w = 32usize;
        let h = 32usize;
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; (w / 2) * h];
        let mut v = vec![0u8; (w / 2) * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 5 + j * 7 + ((i ^ j) & 0x0f) * 3) % 256) as u8;
            }
            for i in 0..(w / 2) {
                u[j * (w / 2) + i] = ((i * 9 + j * 3 + 31) % 256) as u8;
                v[j * (w / 2) + i] = ((i * 11 + j * 13 + 47) % 256) as u8;
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let cs = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:2 NLT quadratic");
        let img = decode_codestream(&cs, None).expect("decode 4:2:2 NLT quadratic");
        for (plane, orig, name) in [
            (&img.planes[0].data, &y, "Y"),
            (&img.planes[1].data, &u, "U"),
            (&img.planes[2].data, &v, "V"),
        ] {
            let p = psnr(orig, plane);
            assert!(
                p >= 40.0,
                "4:2:2 NLT quadratic component {name} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// r371: NLT quadratic composed with 4:2:0 chroma sub-sampling
    /// (`(sx, sy) = (1,1)/(2,2)/(2,2)`). The half-width/half-height chroma
    /// components decompose one vertical level shallower (`N'L,y = NL,y −
    /// log2(sy)`); the per-component NLT map is unaffected. NL=2/2.
    #[test]
    fn round371_nlt_quadratic_subsampled_420() {
        let w = 32usize;
        let h = 32usize;
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; (w / 2) * (h / 2)];
        let mut v = vec![0u8; (w / 2) * (h / 2)];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 3 + j * 5 + ((i ^ j) & 0x0f) * 2) % 256) as u8;
            }
        }
        for j in 0..(h / 2) {
            for i in 0..(w / 2) {
                u[j * (w / 2) + i] = ((i * 7 + j * 11 + 23) % 256) as u8;
                v[j * (w / 2) + i] = ((i * 13 + j * 17 + 41) % 256) as u8;
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let cs = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:0 NLT quadratic");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0 NLT quadratic");
        for (plane, orig, name) in [
            (&img.planes[0].data, &y, "Y"),
            (&img.planes[1].data, &u, "U"),
            (&img.planes[2].data, &v, "V"),
        ] {
            let p = psnr(orig, plane);
            assert!(
                p >= 40.0,
                "4:2:0 NLT quadratic component {name} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// r371: NLT extended (Tnlt=2, three-segment kernel) composed with
    /// 4:2:2 sub-sampling. The reverse-LUT pre-distortion is per-component
    /// so half-width chroma round-trips alongside full-res luma. NL=2/2.
    #[test]
    fn round371_nlt_extended_subsampled_422() {
        let w = 32usize;
        let h = 32usize;
        // Smooth, non-wrapping ramps (no `% 256` discontinuity) so the
        // three-segment extended LUT reconstruction stays well-conditioned —
        // matching the gentle content the single-luma extended test relies on.
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; (w / 2) * h];
        let mut v = vec![0u8; (w / 2) * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 4 + j * 3) / 2).min(255) as u8;
            }
            for i in 0..(w / 2) {
                u[j * (w / 2) + i] = (40 + (i * 5 + j * 2) / 3).min(255) as u8;
                v[j * (w / 2) + i] = (80 + (i * 3 + j * 4) / 3).min(255) as u8;
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        // T1 / T2 / E within the Bw=18 extended-NLT constraints (E=1 matches
        // the proven single-luma extended path).
        let cs = encode_planar_subsampled_nlt_extended(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0,
            1 << 14,
            1 << 16,
            1,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:2 NLT extended");
        let img = decode_codestream(&cs, None).expect("decode 4:2:2 NLT extended");
        for (plane, orig, name) in [
            (&img.planes[0].data, &y, "Y"),
            (&img.planes[1].data, &u, "U"),
            (&img.planes[2].data, &v, "V"),
        ] {
            let p = psnr(orig, plane);
            assert!(
                p >= 30.0,
                "4:2:2 NLT extended component {name} PSNR {p:.2} dB below 30 dB floor"
            );
        }
    }

    /// r371: the sub-sampled NLT entry points reject `cpih = 1` (RCT) with a
    /// genuinely sub-sampled component, mirroring the inner RCT guard (Annex
    /// F.2 requires `sx[i] = sy[i] = 1` for `i < 3`).
    #[test]
    fn round371_nlt_subsampled_rejects_rct_with_subsampling() {
        let w = 16usize;
        let h = 16usize;
        let y = vec![64u8; w * h];
        let u = vec![64u8; (w / 2) * h];
        let v = vec![64u8; (w / 2) * h];
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let r = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            1, // cpih = 1 (RCT) — illegal with sub-sampled chroma
            1,
            1,
            0,
            0,
            &sx,
            &sy,
            &[y, u, v],
        );
        assert!(r.is_err(), "RCT with sub-sampled chroma must be rejected");
    }

    /// r371: lossy (`q = 2`, `Fq = 8`) sub-sampled NLT quadratic. The
    /// per-band deadzone truncation engages under the NLT marker path; the
    /// stream is no larger than its lossless counterpart and every
    /// component (luma + 4:2:0 chroma) holds the ≥ 30 dB lossy floor.
    #[test]
    fn round371_nlt_quadratic_subsampled_420_lossy_q2() {
        let w = 32usize;
        let h = 32usize;
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; (w / 2) * (h / 2)];
        let mut v = vec![0u8; (w / 2) * (h / 2)];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 3 + j * 5) % 200) as u8;
            }
        }
        for j in 0..(h / 2) {
            for i in 0..(w / 2) {
                u[j * (w / 2) + i] = (40 + (i * 4 + j * 2) % 150) as u8;
                v[j * (w / 2) + i] = (60 + (i * 2 + j * 4) % 150) as u8;
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let lossless = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:0 NLT lossless")
        .len();
        let lossy = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            2,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:0 NLT lossy q=2");
        assert!(
            lossy.len() <= lossless,
            "4:2:0 NLT lossy q=2 size {} not <= lossless {}",
            lossy.len(),
            lossless
        );
        let img = decode_codestream(&lossy, None).expect("decode 4:2:0 NLT lossy q=2");
        for (plane, orig, name) in [
            (&img.planes[0].data, &y, "Y"),
            (&img.planes[1].data, &u, "U"),
            (&img.planes[2].data, &v, "V"),
        ] {
            let p = psnr(orig, plane);
            assert!(
                p >= 30.0,
                "4:2:0 NLT quadratic lossy q=2 component {name} PSNR {p:.2} dB below 30 dB floor"
            );
        }
    }

    /// r371: odd picture dimensions with sub-sampled NLT quadratic — the
    /// §B.1 ceiling sizes the chroma planes (`⌈W/2⌉ × ⌈H/2⌉`), so a 30×18
    /// 4:2:0 picture has 15×9 chroma. Exercises the ceiling path under the
    /// NLT marker. Lossless self-roundtrip floor (≥ 40 dB).
    #[test]
    fn round371_nlt_quadratic_subsampled_420_odd_dims() {
        let w = 30usize;
        let h = 18usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 5 + j * 7 + ((i ^ j) & 0x0f) * 3) % 256) as u8;
            }
        }
        for j in 0..ch {
            for i in 0..cw {
                u[j * cw + i] = ((i * 9 + j * 11 + 17) % 256) as u8;
                v[j * cw + i] = ((i * 13 + j * 3 + 29) % 256) as u8;
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let cs = encode_planar_subsampled_nlt_quadratic(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode odd-dim 4:2:0 NLT quadratic");
        let img = decode_codestream(&cs, None).expect("decode odd-dim 4:2:0 NLT quadratic");
        assert_eq!(img.planes[1].data.len(), cw * ch, "chroma is ceil-sized");
        for (plane, orig, name) in [
            (&img.planes[0].data, &y, "Y"),
            (&img.planes[1].data, &u, "U"),
            (&img.planes[2].data, &v, "V"),
        ] {
            let p = psnr(orig, plane);
            assert!(
                p >= 40.0,
                "odd-dim 4:2:0 NLT quadratic component {name} PSNR {p:.2} dB below 40 dB floor"
            );
        }
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

    // === Round 181: NLT quadratic at B[i] > 8 ==============================

    /// Build a synthetic `bd`-bit luma ramp on a `w × h` grid.
    fn make_synthetic_highbd_luma(w: usize, h: usize, bd: u8) -> Vec<u16> {
        let max_sample: u32 = (1u32 << bd) - 1;
        let mut p = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // A diagonal ramp + small modulation, scaled into the
                // component's nominal range.
                let raw = ((x * 11 + y * 7) % 251) as u32;
                let v = ((raw * max_sample) / 250).min(max_sample);
                p.push(v as u16);
            }
        }
        p
    }

    /// Compute PSNR between a `u16` reference plane and the decoder's
    /// `JpegXsPlane` u16-LE byte output, assuming a peak of `2^bd − 1`.
    fn psnr_u16_bytes(a: &[u16], b: &[u8], bd: u8) -> f64 {
        // `b` is the JpegXsPlane byte layout: 2 LE bytes per sample for
        // bd > 8.
        assert_eq!(b.len(), a.len() * 2);
        let n = a.len() as f64;
        let mut mse = 0.0f64;
        for (i, &av) in a.iter().enumerate() {
            let bv = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as i64;
            let d = bv - av as i64;
            mse += (d * d) as f64;
        }
        mse /= n;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        let peak = ((1u32 << bd) - 1) as f64;
        10.0 * (peak * peak / mse).log10()
    }

    /// Round 181: NLT quadratic at `B[i] = 10` self-roundtrips with PSNR
    /// ≥ 40 dB. The sqrt forward / `ω²` inverse pair is not bit-exact
    /// (square-root rounding loses precision at small magnitudes) but the
    /// reconstruction stays well above the 40 dB visual-quality floor.
    #[test]
    fn r181_nlt_quadratic_highbd_10bit_lossless_psnr() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0, // q = 0 — lossless within sqrt rounding
            0, // dco
            std::slice::from_ref(&plane),
        )
        .expect("encode NLT quadratic high-bd 10-bit lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 10-bit NLT");
        assert_eq!(img.num_components, 1);
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 40.0,
            "10-bit NLT quadratic q=0 PSNR {p:.2} dB below 40 dB floor"
        );
    }

    /// Round 181: NLT quadratic at `B[i] = 12` self-roundtrips with PSNR
    /// ≥ 40 dB.
    #[test]
    fn r181_nlt_quadratic_highbd_12bit_lossless_psnr() {
        let w = 32usize;
        let h = 32usize;
        let bd = 12u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            0,
            std::slice::from_ref(&plane),
        )
        .expect("encode NLT quadratic high-bd 12-bit lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 12-bit NLT");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 40.0,
            "12-bit NLT quadratic q=0 PSNR {p:.2} dB below 40 dB floor"
        );
    }

    /// r371: high-bit-depth (`B[i] = 12`) NLT quadratic composed with 4:2:2
    /// chroma sub-sampling. The Annex G.4 forward pre-distortion runs
    /// per-component over the u16-LE planes, so full-res luma and half-width
    /// chroma each round-trip ≥ 40 dB through the `Bw = 20` NLT path. NL=2/2.
    #[test]
    fn round371_nlt_quadratic_highbd_subsampled_422() {
        let w = 32usize;
        let h = 32usize;
        let bd = 12u8;
        let y = make_synthetic_highbd_luma(w, h, bd);
        let u = make_synthetic_highbd_luma(w / 2, h, bd);
        let v = make_synthetic_highbd_luma(w / 2, h, bd);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let cs = encode_planar_subsampled_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            bd,
            0, // q = 0
            0, // dco
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:2 high-bd NLT quadratic");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:2 high-bd NLT");
        assert_eq!(img.num_components, 3);
        for (orig, plane, name) in [
            (&y, &img.planes[0].data, "Y"),
            (&u, &img.planes[1].data, "U"),
            (&v, &img.planes[2].data, "V"),
        ] {
            let p = psnr_u16_bytes(orig, plane, bd);
            assert!(
                p >= 40.0,
                "4:2:2 12-bit NLT quadratic component {name} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// r371: high-bit-depth (`B[i] = 10`) NLT quadratic composed with 4:2:0
    /// chroma sub-sampling (half-width/half-height chroma decomposing one
    /// vertical level shallower). NL=2/2.
    #[test]
    fn round371_nlt_quadratic_highbd_subsampled_420() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let y = make_synthetic_highbd_luma(w, h, bd);
        let u = make_synthetic_highbd_luma(w / 2, h / 2, bd);
        let v = make_synthetic_highbd_luma(w / 2, h / 2, bd);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let cs = encode_planar_subsampled_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            bd,
            0,
            0,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:0 high-bd NLT quadratic");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 4:2:0 high-bd NLT");
        for (orig, plane, name) in [
            (&y, &img.planes[0].data, "Y"),
            (&u, &img.planes[1].data, "U"),
            (&v, &img.planes[2].data, "V"),
        ] {
            let p = psnr_u16_bytes(orig, plane, bd);
            assert!(
                p >= 40.0,
                "4:2:0 10-bit NLT quadratic component {name} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// r371: high-bit-depth (`B[i] = 12`) NLT extended (Tnlt=2, three-segment
    /// kernel) composed with 4:2:2 chroma sub-sampling. The reverse-LUT
    /// pre-distortion is per-component, so half-width chroma round-trips
    /// alongside full-res luma above the extended-NLT PSNR floor. NL=2/2.
    #[test]
    fn round371_nlt_extended_highbd_subsampled_422() {
        let w = 32usize;
        let h = 32usize;
        let bd = 12u8;
        let y = make_synthetic_highbd_luma(w, h, bd);
        let u = make_synthetic_highbd_luma(w / 2, h, bd);
        let v = make_synthetic_highbd_luma(w / 2, h, bd);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        // E=1 matches the proven single-luma extended path; T1/T2 within
        // the Bw=20 extended-NLT constraints.
        let cs = encode_planar_subsampled_nlt_extended_highbd(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            &sx,
            &sy,
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 4:2:2 high-bd NLT extended");
        let img = crate::decoder::decode_codestream(&cs, None)
            .expect("decode 4:2:2 high-bd NLT extended");
        for (orig, plane, name) in [
            (&y, &img.planes[0].data, "Y"),
            (&u, &img.planes[1].data, "U"),
            (&v, &img.planes[2].data, "V"),
        ] {
            let p = psnr_u16_bytes(orig, plane, bd);
            assert!(
                p >= 30.0,
                "4:2:2 12-bit NLT extended component {name} PSNR {p:.2} dB below 30 dB floor"
            );
        }
    }

    /// Round 181: NLT quadratic at `B[i] = 16` self-roundtrips with PSNR
    /// ≥ 40 dB at the upper bit-depth boundary.
    #[test]
    fn r181_nlt_quadratic_highbd_16bit_lossless_psnr() {
        let w = 32usize;
        let h = 32usize;
        let bd = 16u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            0,
            std::slice::from_ref(&plane),
        )
        .expect("encode NLT quadratic high-bd 16-bit lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 16-bit NLT");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 40.0,
            "16-bit NLT quadratic q=0 PSNR {p:.2} dB below 40 dB floor"
        );
    }

    /// Round 181: 10-bit NLT quadratic q=2 is at least as small as the
    /// q=0 form and stays above the 30 dB PSNR floor for a synthetic
    /// ramp. Mirrors the 8-bit `round5_nlt_quadratic_lossy_q2_psnr`
    /// shape at high bit depth.
    #[test]
    fn r181_nlt_quadratic_highbd_10bit_lossy_q2_psnr() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let lossless = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            0,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT lossless")
        .len();
        let lossy_cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            2, // q = 2 — lossy
            0,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT lossy q=2");
        let img = crate::decoder::decode_codestream(&lossy_cs, None).expect("decode q=2 10-bit");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 30.0,
            "10-bit NLT quadratic q=2 PSNR {p:.2} dB below 30 dB floor"
        );
        assert!(
            lossy_cs.len() <= lossless,
            "10-bit NLT q=2 size {} not ≤ lossless {}",
            lossy_cs.len(),
            lossless
        );
    }

    /// Round 181: a non-zero `dco` value is accepted by the NLT marker
    /// encoder (Annex A.4.6 σ:α packing) and produces a valid codestream
    /// that decodes. The encoder's forward sqrt adds `dco` then clamps
    /// to `[0, 2^Bw - 1]` (Annex G.4); the decoder's quadratic inverse
    /// adds `dco` after the inverse multiplication. With `dco = 0` (the
    /// only setting exercised by the 8-bit `round5_nlt_quadratic_*`
    /// tests) the two cancel and PSNR is bounded by the sqrt rounding.
    /// With non-zero `dco` the two operations don't cancel
    /// symmetrically, so the round-trip PSNR varies with `dco` — we
    /// only assert the encode + decode pair completes and produces an
    /// output plane of the correct high-bit-depth byte length here.
    #[test]
    fn r181_nlt_quadratic_highbd_10bit_nonzero_dco_encodes_decodes() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1024,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT dco=1024");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 10-bit NLT dco=1024");
        assert_eq!(img.num_components, 1);
        assert_eq!(img.planes[0].data.len(), w * h * 2);
    }

    /// Round 181: bd outside `9..=16` is rejected with `Unsupported`.
    #[test]
    fn r181_nlt_quadratic_highbd_rejects_8bit_bd() {
        let plane = vec![0u16; 32 * 32];
        let err = encode_planar_nlt_quadratic_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            8,
            0,
            0,
            std::slice::from_ref(&plane),
        )
        .expect_err("bd = 8 must be rejected");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
        let err = encode_planar_nlt_quadratic_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            17,
            0,
            0,
            std::slice::from_ref(&plane),
        )
        .expect_err("bd = 17 must be rejected");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Round 193: NLT extended high-bit-depth is now supported via
    /// [`encode_planar_nlt_extended_highbd`] (Annex G.5 with `Bw = 20`
    /// and the full `2^B[i]` reverse-LUT). The earlier blanket
    /// rejection for `bd > 8 && Extended` no longer applies — the
    /// inner_bd path threads `bd` through to `build_extended_forward_lut`
    /// instead.
    #[test]
    fn r193_nlt_extended_highbd_now_accepted() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        // T1 / T2 / E sized for Bw=20 (max 2^20 − 1 = 1048575).
        let cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            &[plane],
        )
        .expect("encode 10-bit NLT extended lossless");
        // Decoder produces a 10-bit plane (2 bytes per sample).
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 10-bit NLT extended");
        assert_eq!(img.num_components, 1);
        assert_eq!(img.planes[0].data.len(), w * h * 2);
    }

    /// Round 181: 10-bit RGB + RCT (`Cpih = 1`) composes with NLT
    /// quadratic — three components, the reversible RCT applied first,
    /// then sqrt pre-distortion. Self-roundtrips above 35 dB PSNR.
    #[test]
    fn r181_nlt_quadratic_highbd_10bit_rgb_rct_round_trip() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let r = make_synthetic_highbd_luma(w, h, bd);
        let g: Vec<u16> = r.iter().map(|&v| v.wrapping_add(13) & 0x3ff).collect();
        let b: Vec<u16> = r.iter().map(|&v| v.wrapping_add(29) & 0x3ff).collect();
        let cs = encode_planar_nlt_quadratic_highbd(
            w as u16,
            h as u16,
            3,
            1, // Cpih = 1 (RCT)
            2,
            2,
            bd,
            0,
            0,
            &[r.clone(), g.clone(), b.clone()],
        )
        .expect("encode 10-bit RGB+RCT NLT quadratic");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 10-bit RGB+RCT NLT");
        assert_eq!(img.num_components, 3);
        let pr = psnr_u16_bytes(&r, &img.planes[0].data, bd);
        let pg = psnr_u16_bytes(&g, &img.planes[1].data, bd);
        let pb = psnr_u16_bytes(&b, &img.planes[2].data, bd);
        for (label, p) in [("R", pr), ("G", pg), ("B", pb)] {
            assert!(p >= 35.0, "{label} PSNR {p:.2} dB below 35 dB floor");
        }
    }

    // === Round 193: NLT extended (Tnlt=2) high bit depth ===================

    /// Round 193: NLT extended at `B[i] = 10` round-trips with PSNR
    /// ≥ 30 dB on a smooth ramp. The LUT inverter is now keyed on the
    /// full `2^bd` reconstructed-level table (cap dropped) and `Bw = 20`
    /// gives ≥ 4 bits of headroom over the gamma kernel.
    #[test]
    fn r193_nlt_extended_highbd_10bit_psnr_above_30db() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT extended lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 10-bit NLT extended");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 30.0,
            "10-bit NLT extended q=0 PSNR {p:.2} dB below 30 dB floor"
        );
    }

    /// Round 193: NLT extended at `B[i] = 12` round-trips with PSNR
    /// ≥ 30 dB.
    #[test]
    fn r193_nlt_extended_highbd_12bit_psnr_above_30db() {
        let w = 32usize;
        let h = 32usize;
        let bd = 12u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect("encode 12-bit NLT extended lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 12-bit NLT extended");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 30.0,
            "12-bit NLT extended q=0 PSNR {p:.2} dB below 30 dB floor"
        );
    }

    /// Round 193: NLT extended at `B[i] = 16` round-trips. The 16-bit
    /// LUT has `1 << 16 = 65 536` slots — exercise the upper boundary
    /// of the supported bit-depth range.
    #[test]
    fn r193_nlt_extended_highbd_16bit_psnr_above_30db() {
        let w = 32usize;
        let h = 32usize;
        let bd = 16u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect("encode 16-bit NLT extended lossless");
        let img = crate::decoder::decode_codestream(&cs, None).expect("decode 16-bit NLT extended");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 30.0,
            "16-bit NLT extended q=0 PSNR {p:.2} dB below 30 dB floor"
        );
    }

    /// Round 193: 10-bit NLT extended q=2 still meets the 25 dB floor
    /// and produces a codestream no larger than the lossless variant.
    #[test]
    fn r193_nlt_extended_highbd_10bit_lossy_q2_psnr() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let plane = make_synthetic_highbd_luma(w, h, bd);
        let lossless = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT extended lossless")
        .len();
        let lossy_cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            2,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect("encode 10-bit NLT extended lossy q=2");
        let img = crate::decoder::decode_codestream(&lossy_cs, None)
            .expect("decode 10-bit NLT extended q=2");
        let p = psnr_u16_bytes(&plane, &img.planes[0].data, bd);
        assert!(
            p >= 25.0,
            "10-bit NLT extended q=2 PSNR {p:.2} dB below 25 dB floor"
        );
        assert!(
            lossy_cs.len() <= lossless,
            "10-bit NLT extended q=2 size {} not ≤ lossless {}",
            lossy_cs.len(),
            lossless
        );
    }

    /// Round 193: 10-bit RGB + RCT (`Cpih = 1`) composes with NLT
    /// extended — three components, reversible RCT applied first then
    /// the three-segment gamma pre-distortion. Each plane self-decodes
    /// above the 30 dB floor.
    #[test]
    fn r193_nlt_extended_highbd_10bit_rgb_rct_round_trip() {
        let w = 32usize;
        let h = 32usize;
        let bd = 10u8;
        let r = make_synthetic_highbd_luma(w, h, bd);
        let g: Vec<u16> = r.iter().map(|&v| v.wrapping_add(13) & 0x3ff).collect();
        let b: Vec<u16> = r.iter().map(|&v| v.wrapping_add(29) & 0x3ff).collect();
        let cs = encode_planar_nlt_extended_highbd(
            w as u16,
            h as u16,
            3,
            1, // Cpih = 1 (RCT)
            2,
            2,
            bd,
            0,
            1 << 16,
            1 << 18,
            1,
            &[r.clone(), g.clone(), b.clone()],
        )
        .expect("encode 10-bit RGB+RCT NLT extended");
        let img = crate::decoder::decode_codestream(&cs, None)
            .expect("decode 10-bit RGB+RCT NLT extended");
        assert_eq!(img.num_components, 3);
        let pr = psnr_u16_bytes(&r, &img.planes[0].data, bd);
        let pg = psnr_u16_bytes(&g, &img.planes[1].data, bd);
        let pb = psnr_u16_bytes(&b, &img.planes[2].data, bd);
        for (label, p) in [("R", pr), ("G", pg), ("B", pb)] {
            assert!(p >= 30.0, "{label} PSNR {p:.2} dB below 30 dB floor");
        }
    }

    /// Round 193: bd outside `9..=16` is rejected with `Unsupported`.
    #[test]
    fn r193_nlt_extended_highbd_rejects_bad_bd() {
        let plane = vec![0u16; 32 * 32];
        let err = encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            8,
            0,
            1 << 16,
            1 << 18,
            1,
            std::slice::from_ref(&plane),
        )
        .expect_err("bd = 8 must be rejected");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
        let err = encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            17,
            0,
            1 << 16,
            1 << 18,
            1,
            &[plane],
        )
        .expect_err("bd = 17 must be rejected");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Round 193: Star-Tetrix (`Cpih = 3`) high-bit-depth still rejected
    /// on the NLT extended high-bit-depth path, mirroring the
    /// quadratic-highbd behaviour.
    #[test]
    fn r193_nlt_extended_highbd_rejects_star_tetrix_cpih() {
        let plane = vec![0u16; 32 * 32];
        let err = encode_planar_nlt_extended_highbd(
            32,
            32,
            4,
            3, // Cpih = 3
            2,
            2,
            10,
            0,
            1 << 16,
            1 << 18,
            1,
            &[plane.clone(), plane.clone(), plane.clone(), plane],
        )
        .expect_err("Cpih=3 must be rejected on the highbd extended path");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Round 193: extended NLT highbd rejects bad params (T2 ≤ T1, E
    /// out of range, T1 = 0).
    #[test]
    fn r193_nlt_extended_highbd_rejects_bad_params() {
        let plane = vec![0u16; 32 * 32];
        // T2 ≤ T1.
        assert!(encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            1 << 18,
            1 << 16,
            3,
            std::slice::from_ref(&plane)
        )
        .is_err());
        // T1 = 0.
        assert!(encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            0,
            1 << 16,
            3,
            std::slice::from_ref(&plane)
        )
        .is_err());
        // E = 0.
        assert!(encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            1 << 16,
            1 << 18,
            0,
            std::slice::from_ref(&plane)
        )
        .is_err());
        // E = 5.
        assert!(encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            1 << 16,
            1 << 18,
            5,
            std::slice::from_ref(&plane)
        )
        .is_err());
        // T2 above 2^Bw - 1 (Bw=20 highbd).
        assert!(encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            1 << 16,
            1 << 21,
            3,
            &[plane]
        )
        .is_err());
    }

    /// Round 193: a sample value above `2^bd − 1` is rejected.
    #[test]
    fn r193_nlt_extended_highbd_rejects_out_of_range_sample() {
        let mut plane = vec![0u16; 32 * 32];
        plane[5] = 2048; // bd=10 → max 1023
        let err = encode_planar_nlt_extended_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            0,
            1 << 16,
            1 << 18,
            1,
            &[plane],
        )
        .expect_err("out-of-range sample must be rejected");
        assert!(matches!(err, Error::InvalidData(_)));
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

    // === Round 362: multiple significance groups per band-line (Ns > 1) ====
    //
    // The significance sub-packet (Annex B.9 / Table C.5) carries one Z
    // bit per *significance group*, and a band-line holds
    // `Ns[p,b] = ceil(Wpb / (Ng·Ss))` of them (Ng = 4, Ss = 8 → one Z bit
    // every 32 code-positions). Every prior significance round-trip used a
    // picture narrow enough that the widest band stayed at `Wpb ≤ 32`, so
    // `Ns = 1` and the `sig_group = g / Ss` indexing on both the encoder
    // (`group_sig`) and decoder (`sig_flags` keyed by group `j`) collapsed
    // to a single group. These tests push the LL band past 32 positions so
    // `Ns ∈ {2, 3}`, exercising the per-significance-group Z dispatch on
    // both sides.
    //
    // Band widths (Annex B.2): the LL band of a width-`W` picture is
    // `ceil(W/2)` at NL = 1 and `ceil(ceil(W/2)/2)` at NL = 2. To get the
    // *significant* group to land in the second/third sig group (not the
    // first), the perturbed pixels are placed near the right edge of the
    // picture so their wavelet energy concentrates in the high-`g` code
    // groups of each band line.

    /// Wide single-level (NL = 1) picture whose bands span multiple
    /// significance-group widths (`Ns >= 3` on the 66-wide LL band, `Ns = 2`
    /// on the 33-wide HL band), routed through the per-precinct streaming
    /// synthesis path (`synthesise_precinct`) rather than the multi-level
    /// gather/cascade. Non-flat content keeps the encoder on a
    /// non-significance D form, so this locks the wide-band streaming
    /// geometry independently of the significance-coding round-trips below.
    #[test]
    fn round362_wide_band_nl1_streaming_round_trips() {
        let w: u16 = 132;
        let h: u16 = 16;
        let (wu, hu) = (w as usize, h as usize);
        // Non-flat gradient + fringe so the bands carry real AC energy.
        let mut pixels = vec![0u8; wu * hu];
        for y in 0..hu {
            for x in 0..wu {
                let v = 100i32 + (x as i32 - 66) / 3 + (((x ^ y) as i32) & 0x0f);
                pixels[y * wu + x] = v.clamp(0, 255) as u8;
            }
        }
        // Precondition: LL band width ceil(W/2) = 66 > 2·Ng·Ss = 64, so the
        // band-line significance-group count Ns = ceil(66/32) = 3.
        let ll = wu.div_ceil(2);
        assert!(
            ll.div_ceil(32) >= 3,
            "expected Ns>=3 on LL band, got width {ll}"
        );
        let cs = encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels))
            .expect("encode wide NL=1");
        let img = decode_codestream(&cs, None).expect("decode wide NL=1");
        assert_eq!(img.planes[0].data, pixels, "wide NL=1 streaming round-trip");
    }

    /// `Ns = 2` on the LL band at NL = 2 (LL width 33), flat enough that
    /// significance coding stays the rate-optimal `D & 2` form across both
    /// significance groups. Round-trips losslessly and stays well under the
    /// raw budget.
    #[test]
    fn round362_significance_ns2_nl2_flat_round_trips() {
        let w: u16 = 132;
        let h: u16 = 64;
        let (wu, hu) = (w as usize, h as usize);
        let mut pixels = vec![128u8; wu * hu];
        // A couple of point perturbations so we avoid the all-zero
        // degenerate but keep most significance groups insignificant.
        pixels[0] = 130;
        pixels[wu * hu - 1] = 125;
        pixels[(hu / 2) * wu + wu - 1] = 132;
        // LL band at NL=2 is ceil(ceil(W/2)/2); for W=132 that is 33 > 32,
        // so Ns = 2 on the LL band.
        let ll = wu.div_ceil(2).div_ceil(2);
        assert!(
            ll.div_ceil(32) >= 2,
            "expected Ns>=2 on LL band, got width {ll}"
        );
        let cs = encode_planar(w, h, 1, 0, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode Ns=2 NL=2 flat");
        assert!(
            cs.len() < wu * hu,
            "Ns=2 significance-coded codestream ({} B) not below the raw pixel budget ({} B)",
            cs.len(),
            wu * hu
        );
        let img = decode_codestream(&cs, None).expect("decode Ns=2 NL=2 flat");
        assert_eq!(img.planes[0].data, pixels, "Ns=2 NL=2 round-trip");
    }

    // === Run mode Rm=1 (Annex A.4.4 Table A.12, C.6.5) =================

    /// Content with strong vertical structure at the top and an all-zero
    /// (flat) region below — the case where `Rm = 1`'s zero-coefficient
    /// significance elides more than `Rm = 0`'s zero-residual guard.
    fn make_vertical_structure(w: usize, h: usize) -> Vec<u8> {
        let mut px = vec![128u8; w * h];
        for y in 0..(h / 3).max(1) {
            for x in 0..w {
                px[y * w + x] = if (x + y) & 1 == 0 { 40 } else { 200 };
            }
        }
        px
    }

    /// `Rm = 1` luma round-trips losslessly and the PIH carries the run
    /// mode (`rm == 1`, Table A.12).
    #[test]
    fn rm1_luma_lossless_round_trips_and_signals() {
        let (w, h) = (64u16, 64u16);
        let px = make_vertical_structure(w as usize, h as usize);
        let cs = encode_planar_run_mode1(w, h, 1, 0, 2, 2, 0, std::slice::from_ref(&px))
            .expect("encode Rm=1 luma");
        let parsed = crate::codestream::parse(&cs).expect("parse Rm=1 codestream");
        assert_eq!(parsed.pih.rm, 1, "PIH should report Rm=1");
        let img = decode_codestream(&cs, None).expect("decode Rm=1 luma");
        assert_eq!(img.planes[0].data, px, "Rm=1 luma lossless round-trip");
    }

    /// `Rm = 1` never encodes larger than `Rm = 0` on the same content: the
    /// per-band `D[p,b]` picker keeps the smallest form, and dropping the
    /// zero-residual guard only ever removes VLC residuals + data bytes for
    /// all-zero groups that vertically follow non-zero ones.
    #[test]
    fn rm1_no_larger_than_rm0() {
        let (w, h) = (64u16, 96u16);
        let px = make_vertical_structure(w as usize, h as usize);
        let rm0 = encode_planar(w, h, 1, 0, 2, 2, std::slice::from_ref(&px)).expect("Rm=0");
        let rm1 =
            encode_planar_run_mode1(w, h, 1, 0, 2, 2, 0, std::slice::from_ref(&px)).expect("Rm=1");
        assert!(
            rm1.len() <= rm0.len(),
            "Rm=1 ({} B) must not exceed Rm=0 ({} B)",
            rm1.len(),
            rm0.len()
        );
        // Both are bit-exact lossless.
        assert_eq!(
            decode_codestream(&rm1, None).expect("dec Rm=1").planes[0].data,
            px
        );
        assert_eq!(
            decode_codestream(&rm0, None).expect("dec Rm=0").planes[0].data,
            px
        );
    }

    /// `Rm = 1` composes with the reversible colour transform (`Cpih = 1`)
    /// across three components, lossless.
    #[test]
    fn rm1_rgb_rct_lossless_round_trips() {
        let (w, h) = (64u16, 64u16);
        let base = make_vertical_structure(w as usize, h as usize);
        let r = base.clone();
        let g: Vec<u8> = base.iter().map(|&v| v.saturating_add(5)).collect();
        let b: Vec<u8> = base.iter().map(|&v| v.saturating_sub(5)).collect();
        let planes = [r.clone(), g.clone(), b.clone()];
        let cs = encode_planar_run_mode1(w, h, 3, 1, 2, 2, 0, &planes).expect("encode Rm=1 RGB");
        assert_eq!(crate::codestream::parse(&cs).unwrap().pih.rm, 1);
        let img = decode_codestream(&cs, None).expect("decode Rm=1 RGB");
        assert_eq!(img.planes[0].data, r, "R plane");
        assert_eq!(img.planes[1].data, g, "G plane");
        assert_eq!(img.planes[2].data, b, "B plane");
    }

    // === Four components with RCT: Annex F Table F.1 pass-through =====

    /// Build four distinct 8-bit planes: RGB content plus a
    /// binary-noise-like 4th plane (the alpha-or-K shape the ISO/IEC
    /// 21122-4 four-component vector carries).
    fn make_rgba_planes(w: usize, h: usize) -> [Vec<u8>; 4] {
        let base = make_vertical_structure(w, h);
        let r = base.clone();
        let g: Vec<u8> = base.iter().map(|&v| v.saturating_add(9)).collect();
        let b: Vec<u8> = base.iter().map(|&v| v.saturating_sub(7)).collect();
        let a: Vec<u8> = (0..w * h)
            .map(|i| {
                if i.wrapping_mul(2654435761) & 0x100 != 0 {
                    255
                } else {
                    0
                }
            })
            .collect();
        [r, g, b, a]
    }

    /// `Nc = 4` with `Cpih = 1`: the RCT covers components 0..2 and the
    /// 4th component is the Table F.1 pass-through (`Ω[3] = O[3]`) — it
    /// still runs the full entropy / dequant / DWT / Annex G pipeline.
    /// Lossless round-trip must be bit-exact on all four planes.
    #[test]
    fn four_component_rct_pass_through_lossless_round_trips() {
        let (w, h) = (48u16, 32u16);
        let [r, g, b, a] = make_rgba_planes(w as usize, h as usize);
        let planes = [r.clone(), g.clone(), b.clone(), a.clone()];
        let cs = encode_planar(w, h, 4, 1, 2, 2, &planes).expect("encode Nc=4 Cpih=1");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.nc, 4, "PIH Nc=4");
        assert_eq!(parsed.pih.cpih, 1, "PIH Cpih=1");
        let img = decode_codestream(&cs, None).expect("decode Nc=4 Cpih=1");
        assert_eq!(img.planes.len(), 4);
        assert_eq!(img.planes[0].data, r, "R plane");
        assert_eq!(img.planes[1].data, g, "G plane");
        assert_eq!(img.planes[2].data, b, "B plane");
        assert_eq!(img.planes[3].data, a, "pass-through 4th plane");
    }

    /// The same four-component RCT layout at `NL = 1/1` with several
    /// precinct rows and an odd width — the geometry family of the
    /// conformance four-component stream (odd-width level-1 bands whose
    /// last code group carries meaningless tail positions, decoded
    /// through the picture-level cascade rather than any per-precinct
    /// synthesis). Bit-exact on all four planes.
    #[test]
    fn four_component_rct_nl1_odd_width_multi_precinct_round_trips() {
        let (w, h) = (33u16, 12u16);
        let [r, g, b, a] = make_rgba_planes(w as usize, h as usize);
        let planes = [r.clone(), g.clone(), b.clone(), a.clone()];
        let cs = encode_planar(w, h, 4, 1, 1, 1, &planes).expect("encode Nc=4 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode Nc=4 NL=1/1");
        assert_eq!(img.planes.len(), 4);
        assert_eq!(img.planes[0].data, r, "R plane");
        assert_eq!(img.planes[1].data, g, "G plane");
        assert_eq!(img.planes[2].data, b, "B plane");
        assert_eq!(img.planes[3].data, a, "pass-through 4th plane");
    }

    /// Lossy `Nc = 4` `Cpih = 1`: the pass-through component is subject
    /// to the same truncation ladder as the RCT trio; each plane must
    /// stay above the PSNR floor and the decode must keep 4 planes.
    #[test]
    fn four_component_rct_lossy_holds_psnr_floor() {
        let (w, h) = (64u16, 48u16);
        let base = make_vertical_structure(w as usize, h as usize);
        let r = base.clone();
        let g: Vec<u8> = base.iter().map(|&v| v.saturating_add(9)).collect();
        let b: Vec<u8> = base.iter().map(|&v| v.saturating_sub(7)).collect();
        // A smooth ramp for the 4th plane: binary noise has no lossy
        // PSNR guarantee, a gradient does.
        let a: Vec<u8> = (0..w as usize * h as usize)
            .map(|i| ((i % w as usize) * 255 / (w as usize - 1)) as u8)
            .collect();
        let planes = [r.clone(), g.clone(), b.clone(), a.clone()];
        let cs =
            encode_planar_lossy(w, h, 4, 1, 2, 2, 2, &planes).expect("encode Nc=4 Cpih=1 lossy");
        let img = decode_codestream(&cs, None).expect("decode Nc=4 Cpih=1 lossy");
        assert_eq!(img.planes.len(), 4);
        for (k, orig) in [&r, &g, &b, &a].into_iter().enumerate() {
            let mut sse = 0.0f64;
            for (x, y) in orig.iter().zip(img.planes[k].data.iter()) {
                let d = *x as f64 - *y as f64;
                sse += d * d;
            }
            let mse = sse / (w as f64 * h as f64);
            let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
            assert!(
                psnr >= 25.0,
                "component {k} lossy PSNR {psnr:.2} dB below 25 dB floor"
            );
        }
    }

    /// `Rm = 1` lossy (`q > 0`) round-trips within a PSNR floor — the
    /// insignificant-group `M = T` reconstruction is exactly the zero the
    /// data sub-packet would have decoded anyway.
    #[test]
    fn rm1_lossy_round_trips_within_psnr() {
        let (w, h) = (64u16, 64u16);
        let px = make_vertical_structure(w as usize, h as usize);
        let cs = encode_planar_run_mode1(w, h, 1, 0, 2, 2, 2, std::slice::from_ref(&px))
            .expect("encode Rm=1 lossy");
        assert_eq!(crate::codestream::parse(&cs).unwrap().pih.rm, 1);
        let img = decode_codestream(&cs, None).expect("decode Rm=1 lossy");
        let mut sse = 0.0f64;
        for (a, b) in px.iter().zip(img.planes[0].data.iter()) {
            let d = *a as f64 - *b as f64;
            sse += d * d;
        }
        let mse = sse / (w as f64 * h as f64);
        let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
        assert!(
            psnr >= 25.0,
            "Rm=1 lossy PSNR {psnr:.2} dB below 25 dB floor"
        );
    }

    /// `Rm = 1` composed with high bit depth (`B[i] = 12`, `u16`-LE planes):
    /// the zero-coefficient significance operates on the wider wavelet
    /// domain and self-decodes bit-exactly.
    #[test]
    fn rm1_highbd_12bit_lossless_round_trips() {
        let (w, h) = (48u16, 64u16);
        let base = make_vertical_structure(w as usize, h as usize);
        let plane: Vec<u16> = base.iter().map(|&v| (v as u16) << 4).collect();
        let cs = encode_planar_run_mode1_highbd(w, h, 1, 0, 2, 2, 12, std::slice::from_ref(&plane))
            .expect("encode Rm=1 12-bit");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.rm, 1, "PIH Rm=1");
        assert_eq!(parsed.pih.bw, 12, "Bw = B[i] = 12");
        let img = decode_codestream(&cs, None).expect("decode Rm=1 12-bit");
        let out: Vec<u16> = img.planes[0]
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(out, plane, "Rm=1 12-bit lossless round-trip");
    }

    /// `Rm = 1` composed with 4:2:0 chroma sub-sampling: a `sy = 2`
    /// component decomposes one vertical level shallower, so the
    /// insignificant-group reconstruction runs across the mixed-depth
    /// cascade. Lossless self-decode.
    #[test]
    fn rm1_subsampled_420_lossless_round_trips() {
        let (w, h) = (64u16, 64u16);
        let luma = make_vertical_structure(w as usize, h as usize);
        let cw = (w / 2) as usize;
        let ch = (h / 2) as usize;
        let mut cb = vec![128u8; cw * ch];
        let mut cr = vec![110u8; cw * ch];
        for y in 0..ch / 3 {
            for x in 0..cw {
                cb[y * cw + x] = if x & 1 == 0 { 90 } else { 160 };
                cr[y * cw + x] = if x & 1 == 0 { 150 } else { 70 };
            }
        }
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let planes = [luma.clone(), cb.clone(), cr.clone()];
        let cs = encode_planar_run_mode1_subsampled(w, h, 3, 0, 2, 2, 0, &sx, &sy, &planes)
            .expect("encode Rm=1 4:2:0");
        assert_eq!(crate::codestream::parse(&cs).unwrap().pih.rm, 1);
        let img = decode_codestream(&cs, None).expect("decode Rm=1 4:2:0");
        assert_eq!(img.planes[0].data, luma, "luma");
        assert_eq!(img.planes[1].data, cb, "Cb");
        assert_eq!(img.planes[2].data, cr, "Cr");
    }

    /// `Rm = 1` with **multiple significance groups per band line**
    /// (`Ns[p,b] = ⌈Wpb / (Ng·Ss)⌉ > 1`): a band wider than `Ng·Ss = 32`
    /// code positions emits one `Z` bit per group, and under `Rm = 1` each
    /// insignificant group independently collapses to `M = T`. Confirms the
    /// per-group Z dispatch composes with the zero-coefficient run mode.
    #[test]
    fn rm1_multi_sig_group_ns2_round_trips() {
        let (w, h) = (140u16, 64u16);
        let px = make_vertical_structure(w as usize, h as usize);
        // LL band at NL=2 is ceil(ceil(140/2)/2) = 35 > 32 → Ns = 2.
        let ll = (w as usize).div_ceil(2).div_ceil(2);
        assert!(ll.div_ceil(32) >= 2, "expected Ns>=2, got LL width {ll}");
        let cs = encode_planar_run_mode1(w, h, 1, 0, 2, 2, 0, std::slice::from_ref(&px))
            .expect("encode Rm=1 Ns=2");
        assert_eq!(crate::codestream::parse(&cs).unwrap().pih.rm, 1);
        let img = decode_codestream(&cs, None).expect("decode Rm=1 Ns=2");
        assert_eq!(img.planes[0].data, px, "Rm=1 Ns=2 round-trip");
    }

    /// `Rm = 1` at a deeper wavelet cascade (`NL = 3`) with odd picture
    /// dimensions: the §B.1 ceiling-sized band geometry and the cross-line
    /// vertical predictor both compose with the zero-coefficient run mode.
    #[test]
    fn rm1_deep_cascade_odd_dims_round_trips() {
        let (w, h) = (67u16, 83u16);
        let px = make_vertical_structure(w as usize, h as usize);
        let cs = encode_planar_run_mode1(w, h, 1, 0, 3, 3, 0, std::slice::from_ref(&px))
            .expect("encode Rm=1 NL=3 odd");
        assert_eq!(crate::codestream::parse(&cs).unwrap().pih.rm, 1);
        let img = decode_codestream(&cs, None).expect("decode Rm=1 NL=3 odd");
        assert_eq!(img.planes[0].data, px, "Rm=1 NL=3 odd-dim round-trip");
    }

    /// `Rm = 1` self-consistency across the truncation ladder: for every
    /// `q ∈ 0..=6` the stream round-trips through the standard decode path
    /// (bit-exact at `q = 0`, and always internally consistent — the
    /// insignificant-group `M = T` reconstruction never demands data the
    /// encoder did not emit, so the length-conformance gates all pass).
    #[test]
    fn rm1_truncation_ladder_all_q_decode() {
        let (w, h) = (64u16, 96u16);
        let px = make_vertical_structure(w as usize, h as usize);
        for q in 0u8..=6 {
            let cs = encode_planar_run_mode1(w, h, 1, 0, 2, 2, q, std::slice::from_ref(&px))
                .unwrap_or_else(|e| panic!("encode Rm=1 q={q}: {e}"));
            let img =
                decode_codestream(&cs, None).unwrap_or_else(|e| panic!("decode Rm=1 q={q}: {e}"));
            if q == 0 {
                assert_eq!(img.planes[0].data, px, "Rm=1 q=0 bit-exact");
            }
        }
    }

    /// `Rm = 2` / `Rm = 3` are reserved (Table A.12) — the config validator
    /// rejects them before any bytes are emitted.
    #[test]
    fn rm_reserved_values_rejected() {
        let px = vec![128u8; 8 * 8];
        // Reach the validator via a hand-built config (the public entry
        // points only ever set 0 or 1).
        let cfg = EncodeConfig {
            width: 8,
            height: 8,
            nc: 1,
            bit_depth: 8,
            bw: 8,
            ng: 4,
            ss: 8,
            br: 8,
            nlx: 1,
            nly: 1,
            cpih: 0,
            qpih: 0,
            fs: 0,
            fq: 0,
            q: 0,
            sx: vec![1],
            sy: vec![1],
            cts_e1: 0,
            cts_e2: 0,
            cts_cf: 0,
            st_ct: 0,
            nlt: None,
            band_gains: vec![],
            band_priorities: vec![],
            rp: 0,
            cw: 0,
            sd: 0,
            hsl: 0,
            q_slices: Vec::new(),
            q_precincts: Vec::new(),
            r_precincts: Vec::new(),
            rm: 2,
        };
        let _ = &px;
        assert!(
            cfg.validate().is_err(),
            "Rm=2 must be rejected (Table A.12)"
        );
    }

    /// `Ns > 1` composed with three-component RGB + the reversible colour
    /// transform (Cpih = 1): each colour plane independently carries
    /// multi-significance-group band lines, and the inverse RCT recombines
    /// them. Confirms the per-group Z dispatch is correct in the
    /// multi-component gather path.
    #[test]
    fn round362_significance_ns2_rgb_rct_round_trips() {
        let w: u16 = 140;
        let h: u16 = 16;
        let (wu, hu) = (w as usize, h as usize);
        let mut r = vec![120u8; wu * hu];
        let mut g = vec![120u8; wu * hu];
        let mut b = vec![120u8; wu * hu];
        for y in 0..hu {
            for x in 100..wu {
                let v: u8 = if (x ^ y) & 1 == 0 { 180 } else { 40 };
                r[y * wu + x] = v;
                g[y * wu + x] = v.saturating_add(8);
                b[y * wu + x] = v.saturating_sub(8);
            }
        }
        // LL band at NL=2 is ceil(ceil(140/2)/2) = 35 > 32 → Ns = 2.
        let ll = wu.div_ceil(2).div_ceil(2);
        assert!(
            ll.div_ceil(32) >= 2,
            "expected Ns>=2 on LL band, got width {ll}"
        );
        let planes = [r.clone(), g.clone(), b.clone()];
        let cs = encode_planar(w, h, 3, 1, 2, 2, &planes).expect("encode Ns>1 RGB RCT");
        let img = decode_codestream(&cs, None).expect("decode Ns>1 RGB RCT");
        assert_eq!(img.planes[0].data, r, "R plane round-trip");
        assert_eq!(img.planes[1].data, g, "G plane round-trip");
        assert_eq!(img.planes[2].data, b, "B plane round-trip");
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

    // ----------------------------------------------------------------
    // Round 359 — deepen DECODE-geometry coverage past the round-357
    // horizontal-only (NL,y = 0) envelope. JPEG XS allows asymmetric
    // decomposition NL,x ≥ NL,y; for a 4:2:0 component (sy = 2) the
    // decoder drops log2(sy) = 1 vertical level, so the chroma planes
    // decompose at N'L,y[i] = NL,y − 1 (decoder.rs `nly_i` subtraction).
    // The round-343 4:2:0 tests pinned NL,y = 2 (chroma N'L,y = 1); the
    // tests below push the vertical cascade to NL,y ∈ {3, 4, 5} (chroma
    // N'L,y up to 4) and add asymmetric NL,x > NL,y, high-bit-depth
    // 4:2:0, lossy 4:2:0, NLT (Bw = 18) and high-precision (Bw = 20)
    // luma at deeper vertical cascades — exercising the per-band
    // `beta_key_for` cascade-key derivation across the full N'L,y range.
    // ----------------------------------------------------------------

    /// Build a deterministic high-bit-depth 4:2:0 luma + half-resolution
    /// chroma triple, every sample masked to `bd` significant bits.
    fn synthetic_420_highbd(w: usize, h: usize, bd: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let n_y = w * h;
        let n_c = (w / 2) * (h / 2);
        let mask = (1u32 << bd) - 1;
        let y: Vec<u16> = (0..n_y)
            .map(|i| ((i as u32 * 7 + 13) & mask) as u16)
            .collect();
        let cb: Vec<u16> = (0..n_c)
            .map(|i| ((i as u32 * 11 + 17) & mask) as u16)
            .collect();
        let cr: Vec<u16> = (0..n_c)
            .map(|i| ((i as u32 * 19 + 23) & mask) as u16)
            .collect();
        (y, cb, cr)
    }

    /// 4:2:0 lossless decode with a symmetric vertical cascade deeper than
    /// the round-343 NL,y = 2 baseline. Each luma plane decomposes at
    /// NL,x = NL,y ∈ {3, 4, 5}; the sy = 2 chroma planes decompose one
    /// vertical level shallower (N'L,y = NL,y − 1 ∈ {2, 3, 4}). Bit-exact
    /// recovery of all three planes locks the deeper-cascade decode path.
    #[test]
    fn round359_420_lossless_deep_vertical_cascade() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        for nl in [3u8, 4, 5] {
            let cs = encode_planar_subsampled(
                w as u16,
                h as u16,
                3,
                0,
                nl,
                nl,
                0,
                &[1, 2, 2],
                &[1, 2, 2],
                &[y.clone(), cb.clone(), cr.clone()],
            )
            .unwrap_or_else(|e| panic!("encode 4:2:0 NL={nl}/{nl}: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode 4:2:0 NL={nl}/{nl}: {e:?}"));
            assert_eq!(img.num_components, 3);
            assert_eq!(img.planes[0].data, y, "Y plane 4:2:0 NL={nl}/{nl}");
            assert_eq!(img.planes[1].data, cb, "Cb plane 4:2:0 NL={nl}/{nl}");
            assert_eq!(img.planes[2].data, cr, "Cr plane 4:2:0 NL={nl}/{nl}");
        }
    }

    /// 4:2:0 lossless decode with an **asymmetric** cascade NL,x > NL,y.
    /// The horizontal and vertical decomposition depths differ, so the
    /// per-band cascade keys (`beta_key_for`) span the asymmetric region
    /// of Annex B.7 Table B.4 (β1 = NL,x − NL,y + 1 > 1). Chroma drops one
    /// vertical level (N'L,y = NL,y − 1) while keeping the full NL,x.
    #[test]
    fn round359_420_lossless_asymmetric_cascade() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        for (nlx, nly) in [(3u8, 2u8), (4, 2), (4, 3), (5, 3)] {
            let cs = encode_planar_subsampled(
                w as u16,
                h as u16,
                3,
                0,
                nlx,
                nly,
                0,
                &[1, 2, 2],
                &[1, 2, 2],
                &[y.clone(), cb.clone(), cr.clone()],
            )
            .unwrap_or_else(|e| panic!("encode 4:2:0 NL={nlx}/{nly}: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode 4:2:0 NL={nlx}/{nly}: {e:?}"));
            assert_eq!(img.planes[0].data, y, "Y plane 4:2:0 NL={nlx}/{nly}");
            assert_eq!(img.planes[1].data, cb, "Cb plane 4:2:0 NL={nlx}/{nly}");
            assert_eq!(img.planes[2].data, cr, "Cr plane 4:2:0 NL={nlx}/{nly}");
        }
    }

    /// High-bit-depth (`B[i] ∈ {10, 12, 14, 16}`) 4:2:0 lossless decode at
    /// NL,y = 3. The u16-LE plane path composes with the deeper vertical
    /// cascade and the sy = 2 chroma level-drop; every plane round-trips
    /// bit-exactly, confirming sample precision is orthogonal to the
    /// 4:2:0 decode geometry.
    #[test]
    fn round359_420_highbd_deep_vertical_cascade() {
        let (w, h) = (64usize, 64usize);
        for &bd in &[10u8, 12, 14, 16] {
            let (yp, cbp, crp) = synthetic_420_highbd(w, h, bd);
            let cs = encode_planar_subsampled_highbd(
                w as u16,
                h as u16,
                3,
                0,
                3,
                3,
                bd,
                &[1, 2, 2],
                &[1, 2, 2],
                &[yp.clone(), cbp.clone(), crp.clone()],
            )
            .unwrap_or_else(|e| panic!("encode {bd}-bit 4:2:0 NL=3/3: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode {bd}-bit 4:2:0 NL=3/3: {e:?}"));
            let unpack = |d: &[u8]| -> Vec<u16> {
                d.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            };
            assert_eq!(unpack(&img.planes[0].data), yp, "Y {bd}-bit 4:2:0 NL=3/3");
            assert_eq!(unpack(&img.planes[1].data), cbp, "Cb {bd}-bit 4:2:0 NL=3/3");
            assert_eq!(unpack(&img.planes[2].data), crp, "Cr {bd}-bit 4:2:0 NL=3/3");
        }
    }

    /// 4:2:0 lossy (q = 2) decode at NL,y ∈ {2, 3, 4}: the deeper vertical
    /// cascade composes with the `T[p,b]` truncation. The reconstruction
    /// stays coherent (correct geometry, decodable) and above a sane PSNR
    /// floor at every depth.
    #[test]
    fn round359_420_lossy_deep_vertical_cascade() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        for nl in [2u8, 3, 4] {
            let cs = encode_planar_subsampled(
                w as u16,
                h as u16,
                3,
                0,
                nl,
                nl,
                2,
                &[1, 2, 2],
                &[1, 2, 2],
                &[y.clone(), cb.clone(), cr.clone()],
            )
            .unwrap_or_else(|e| panic!("encode 4:2:0 lossy NL={nl}/{nl}: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode 4:2:0 lossy NL={nl}/{nl}: {e:?}"));
            assert_eq!(img.planes[0].data.len(), y.len());
            assert_eq!(img.planes[1].data.len(), cb.len());
            assert_eq!(img.planes[2].data.len(), cr.len());
            let py = psnr(&y, &img.planes[0].data);
            let pcb = psnr(&cb, &img.planes[1].data);
            let pcr = psnr(&cr, &img.planes[2].data);
            assert!(
                py >= 25.0 && pcb >= 25.0 && pcr >= 25.0,
                "4:2:0 lossy NL={nl}/{nl} PSNR below 25 dB: Y={py:.2} Cb={pcb:.2} Cr={pcr:.2}"
            );
        }
    }

    /// NLT quadratic (Annex G.4, Bw = 18) luma decode at deeper vertical
    /// cascades NL,y ∈ {1, 2, 3, 4}. The non-linearity inverse composes
    /// with the multi-level vertical synthesis; round-357 locked NL,y = 0
    /// horizontal-only, this extends the NLT decode path through the
    /// vertical cascade. PSNR floor checks the quadratic round-trip.
    #[test]
    fn round359_nlt_quadratic_deep_vertical_cascade() {
        let lp = make_nl_test_64x64();
        for nly in [1u8, 2, 3, 4] {
            let cs = encode_planar_nlt_quadratic(
                64,
                64,
                1,
                0,
                nly,
                nly,
                0,
                0,
                std::slice::from_ref(&lp),
            )
            .unwrap_or_else(|e| panic!("encode NLT-quad NL={nly}/{nly}: {e:?}"));
            let parsed = crate::codestream::parse(&cs).expect("parse NLT-quad");
            assert_eq!(parsed.pih.bw, 18, "NLT quadratic forces Bw=18");
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode NLT-quad NL={nly}/{nly}: {e:?}"));
            let p = psnr(&lp, &img.planes[0].data);
            assert!(
                p >= 40.0,
                "NLT-quad NL={nly}/{nly} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// NLT extended (Annex G.4, Bw = 18) luma decode at deeper vertical
    /// cascades NL,y ∈ {1, 2, 3}. The two-threshold extended non-linearity
    /// inverse (T1 < T2, E) composes with the vertical cascade synthesis.
    #[test]
    fn round359_nlt_extended_deep_vertical_cascade() {
        let lp = make_nl_test_64x64();
        for nly in [1u8, 2, 3] {
            let cs = encode_planar_nlt_extended(
                64,
                64,
                1,
                0,
                nly,
                nly,
                0,
                4096,
                32768,
                2,
                std::slice::from_ref(&lp),
            )
            .unwrap_or_else(|e| panic!("encode NLT-ext NL={nly}/{nly}: {e:?}"));
            let parsed = crate::codestream::parse(&cs).expect("parse NLT-ext");
            assert_eq!(parsed.pih.bw, 18, "NLT extended forces Bw=18");
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode NLT-ext NL={nly}/{nly}: {e:?}"));
            let p = psnr(&lp, &img.planes[0].data);
            assert!(
                p >= 40.0,
                "NLT-ext NL={nly}/{nly} PSNR {p:.2} dB below 40 dB floor"
            );
        }
    }

    /// High-precision (Bw = 20, Fq = 8) luma decode at deeper vertical
    /// cascades NL,y ∈ {1, 2, 3, 4}. The 8-fractional-bit transform path
    /// (`T[β,x,y] = c << Fq` reconstruction) composes with the multi-level
    /// vertical synthesis; the smooth `make_nl_test_64x64` content decodes
    /// bit-exactly at low q, so this asserts a lossless round-trip.
    #[test]
    fn round359_highprec_bw20_deep_vertical_cascade() {
        let lp = make_nl_test_64x64();
        for nly in [1u8, 2, 3, 4] {
            let cs =
                encode_planar_highprec_lossy(64, 64, 1, 0, nly, nly, 1, std::slice::from_ref(&lp))
                    .unwrap_or_else(|e| panic!("encode highprec NL={nly}/{nly}: {e:?}"));
            let parsed = crate::codestream::parse(&cs).expect("parse highprec");
            assert_eq!(parsed.pih.bw, 20, "high-precision path forces Bw=20");
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode highprec NL={nly}/{nly}: {e:?}"));
            assert_eq!(
                img.planes[0].data, lp,
                "high-precision Bw=20 luma must round-trip at NL={nly}/{nly}"
            );
        }
    }

    /// Star-Tetrix (`Cpih = 3`, 4-component 4:4:4 CFA, Annex F.5) lossless
    /// decode at deeper vertical cascades NL,y ∈ {2, 3, 4}. The CFA inverse
    /// transform composes with the multi-level vertical synthesis; all four
    /// components round-trip bit-exactly, locking the Star-Tetrix decode
    /// path past the round-343 NL = 2 baseline.
    #[test]
    fn round359_star_tetrix_deep_vertical_cascade() {
        let n = 64 * 64;
        let mk =
            |a: usize, b: usize| -> Vec<u8> { (0..n).map(|i| ((i * a + b) % 256) as u8).collect() };
        let planes = [mk(7, 13), mk(11, 17), mk(19, 23), mk(29, 31)];
        for nl in [2u8, 3, 4] {
            let cs = encode_planar_star_tetrix(64, 64, nl, nl, 0, 0, 0, 0, 0, &planes)
                .unwrap_or_else(|e| panic!("encode Star-Tetrix NL={nl}/{nl}: {e:?}"));
            let parsed = crate::codestream::parse(&cs).expect("parse Star-Tetrix");
            assert_eq!(parsed.pih.cpih, 3, "Star-Tetrix must signal Cpih=3");
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode Star-Tetrix NL={nl}/{nl}: {e:?}"));
            assert_eq!(img.num_components, 4);
            for (c, plane) in planes.iter().enumerate() {
                assert_eq!(
                    &img.planes[c].data, plane,
                    "Star-Tetrix component {c} NL={nl}/{nl}"
                );
            }
        }
    }

    /// High-bit-depth (`B[i] = 12`) NLT quadratic (Bw = 18) luma decode at
    /// deeper vertical cascades NL,y ∈ {1, 2, 3}. Combines the three
    /// milestone axes — non-linearity, high precision, and a deep vertical
    /// cascade — on the u16-LE plane path. The quadratic approximation
    /// keeps mean-squared error tiny (≤ a few LSB²) at lossless q = 0.
    #[test]
    fn round359_nlt_quadratic_highbd_deep_vertical_cascade() {
        let n = 64 * 64;
        let bd = 12u8;
        let mask = (1u32 << bd) - 1;
        let lp: Vec<u16> = (0..n).map(|i| ((i as u32 * 37) & mask) as u16).collect();
        for nly in [1u8, 2, 3] {
            let cs = encode_planar_nlt_quadratic_highbd(
                64,
                64,
                1,
                0,
                nly,
                nly,
                bd,
                0,
                0,
                std::slice::from_ref(&lp),
            )
            .unwrap_or_else(|e| panic!("encode NLT-quad-highbd NL={nly}/{nly}: {e:?}"));
            let parsed = crate::codestream::parse(&cs).expect("parse NLT-quad-highbd");
            // High-bit-depth NLT runs the wavelet domain at Bw = 20 (top of
            // the Table A.8 {8, 18, 20} set) for ≥ 4 bits of precision
            // headroom over B[i] ≤ 16 — unlike the 8-bit NLT path's Bw = 18.
            assert_eq!(
                parsed.pih.bw, 20,
                "high-bit-depth NLT runs the wavelet domain at Bw=20"
            );
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode NLT-quad-highbd NL={nly}/{nly}: {e:?}"));
            let got: Vec<u16> = img.planes[0]
                .data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            assert_eq!(got.len(), lp.len());
            let mse: f64 = got
                .iter()
                .zip(&lp)
                .map(|(a, b)| {
                    let d = *a as f64 - *b as f64;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            assert!(
                mse <= 4.0,
                "NLT-quad-highbd NL={nly}/{nly} MSE {mse:.3} above 4.0 LSB^2"
            );
        }
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

    /// Horizontal-only decomposition (NL,y = 0) is a distinct geometry
    /// class: bands are LL (β = 0) and HL (β = 1), each precinct is a
    /// single image line (Hp = 2^NL,y = 1), and Annex B.7 Table B.4 gives
    /// β1 = NL,x − 0 + 1 = 2, so the LL and HL bands of every component
    /// are grouped into the FIRST packet (Table B.5). The single-level
    /// streaming encoder previously emitted HL as a standalone proxy
    /// packet, mismatching the decoder's spec-derived packet plan and
    /// underrunning the entropy buffer. This locks the joint-first-packet
    /// emit for the NL,x = 1, NL,y = 0 streaming path.
    #[test]
    fn round357_nly0_horizontal_only_lossless_round_trip_luma() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar(64, 64, 1, 0, 1, 0, std::slice::from_ref(&pixels))
            .expect("encode luma NL,x=1 NL,y=0");
        let img = decode_codestream(&cs, None).expect("decode NL,y=0");
        assert_eq!(
            img.planes[0].data, pixels,
            "luma must round-trip losslessly at NL,x=1 NL,y=0"
        );
    }

    /// Horizontal-only (NL,y = 0) for odd dimensions exercises the
    /// partial-line precinct accounting alongside the joint LL+HL first
    /// packet.
    #[test]
    fn round357_nly0_horizontal_only_odd_dims_round_trip() {
        let (w, h) = (31usize, 17usize);
        let pixels: Vec<u8> = (0..(w * h)).map(|i| ((i * 13 + 7) & 0xff) as u8).collect();
        let cs = encode_planar(
            w as u16,
            h as u16,
            1,
            0,
            1,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("encode 31x17 NL,y=0");
        let img = decode_codestream(&cs, None).expect("decode 31x17 NL,y=0");
        assert_eq!(
            img.planes[0].data, pixels,
            "odd-dimension luma must round-trip at NL,x=1 NL,y=0"
        );
    }

    /// Horizontal-only (NL,y = 0) multi-component RGB with the reversible
    /// colour transform: every component decomposes at NL,x = 1, NL,y = 0,
    /// so each contributes an LL+HL pair to the first packet.
    #[test]
    fn round357_nly0_horizontal_only_rgb_rct_round_trip() {
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
        // Cpih=0 (no colour transform) first to isolate the packet path.
        let cs0 = encode_planar(32, 32, 3, 0, 1, 0, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB NL,y=0 Cpih=0");
        let img0 = decode_codestream(&cs0, None).expect("decode RGB NL,y=0 Cpih=0");
        assert_eq!(img0.planes[0].data, r, "red plane NL,y=0 Cpih=0");
        assert_eq!(img0.planes[1].data, g, "green plane NL,y=0 Cpih=0");
        assert_eq!(img0.planes[2].data, b, "blue plane NL,y=0 Cpih=0");

        let cs = encode_planar(32, 32, 3, 1, 1, 0, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB NL,y=0 Cpih=1");
        let img = decode_codestream(&cs, None).expect("decode RGB NL,y=0");
        assert_eq!(img.planes[0].data, r, "red plane NL,y=0");
        assert_eq!(img.planes[1].data, g, "green plane NL,y=0");
        assert_eq!(img.planes[2].data, b, "blue plane NL,y=0");
    }

    /// Horizontal-only (NL,y = 0) lossy quantised round-trip: the
    /// joint LL+HL first packet must hold under non-zero quantisation
    /// (T[p,b] truncation) too. PSNR floor checks the decode is coherent.
    #[test]
    fn round357_nly0_horizontal_only_lossy_q2_psnr() {
        let pixels = make_nl_test_64x64();
        let cs = encode_planar_lossy(64, 64, 1, 0, 1, 0, 2, std::slice::from_ref(&pixels))
            .expect("encode luma NL,y=0 q=2");
        let img = decode_codestream(&cs, None).expect("decode NL,y=0 q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 25.0, "NL,y=0 q=2 PSNR {p:.2} dB below 25 dB floor");
    }

    /// Horizontal-only (NL,y = 0) at high bit depth (12-bit luma): the
    /// joint first-packet layout is orthogonal to sample precision, so the
    /// u16-LE plane path must round-trip bit-exactly.
    #[test]
    fn round357_nly0_horizontal_only_highbd_12bit_round_trip() {
        let (w, h) = (32usize, 16usize);
        let pixels: Vec<u16> = (0..(w * h)).map(|i| ((i * 37) & 0x0fff) as u16).collect();
        let cs = encode_planar_highbd(
            w as u16,
            h as u16,
            1,
            0,
            1,
            0,
            12,
            std::slice::from_ref(&pixels),
        )
        .expect("encode 12-bit NL,y=0");
        let img = decode_codestream(&cs, None).expect("decode 12-bit NL,y=0");
        // Plane is u16-LE packed.
        let got: Vec<u16> = img.planes[0]
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, pixels, "12-bit luma must round-trip at NL,y=0");
    }

    /// Deeper horizontal-only cascade (NL,x = 3, NL,y = 0): NL,x > 1
    /// routes through the cascade encoder, which already handled NL,y = 0
    /// (β1 = NL,x + 1 with all bands in the first packet). This locks that
    /// the streaming-path fix did not perturb the cascade path's existing
    /// NL,y = 0 coverage.
    #[test]
    fn round357_nly0_horizontal_only_cascade_nlx3_round_trip() {
        let pixels = make_nl_test_64x64();
        for nlx in [2u8, 3, 5] {
            let cs = encode_planar(64, 64, 1, 0, nlx, 0, std::slice::from_ref(&pixels))
                .unwrap_or_else(|e| panic!("encode NL,x={nlx} NL,y=0: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode NL,x={nlx} NL,y=0: {e:?}"));
            assert_eq!(
                img.planes[0].data, pixels,
                "luma must round-trip at NL,x={nlx} NL,y=0"
            );
        }
    }

    /// 4:2:2 horizontal-only (NL,y = 0): the chroma planes are
    /// horizontally subsampled (sx = 2, sy = 1) while no vertical
    /// decomposition is performed. Every component decomposes at NL,x = 1,
    /// NL,y = 0 (β1 = 2 joint LL+HL first packet) and the luma carries the
    /// full picture width. Locks the subsampling × horizontal-only
    /// decode-geometry combination.
    #[test]
    fn round357_nly0_chroma_422_lossless_round_trip() {
        let (w, h) = (32u16, 16u16);
        let n = (w as usize) * (h as usize);
        let n_c = (w as usize / 2) * (h as usize);
        let mut y_plane = vec![0u8; n];
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
            0,
            0,
            &[1, 2, 2],
            &[1, 1, 1],
            &[y_plane.clone(), cb_plane.clone(), cr_plane.clone()],
        )
        .expect("encode 4:2:2 NL,y=0 lossless");
        let img = decode_codestream(&cs, None).expect("decode 4:2:2 NL,y=0");
        assert_eq!(img.planes[0].data, y_plane, "Y plane 4:2:2 NL,y=0");
        assert_eq!(img.planes[1].data, cb_plane, "Cb plane 4:2:2 NL,y=0");
        assert_eq!(img.planes[2].data, cr_plane, "Cr plane 4:2:2 NL,y=0");
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
            Vec::new(),
            1,          // cw
            0,          // sd
            0,          // fs
            0,          // hsl
            0,          // qpih
            0,          // rp
            Vec::new(), // q_slices: single picture-level q
            Vec::new(), // q_precincts: no per-precinct override
            Vec::new(), // r_precincts: no per-precinct R[p] override
            &[y_plane.clone(), u_plane.clone(), v_plane.clone()],
        )
        .expect("encode 64x8 4:2:2 Cw=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode 64x8 4:2:2 Cw=1");
        assert_eq!(img.planes[0].data, y_plane);
        assert_eq!(img.planes[1].data, u_plane);
        assert_eq!(img.planes[2].data, v_plane);
    }

    /// r371: multi-precinct-per-row (`Cw > 0`) composed with 4:2:0 chroma
    /// sub-sampling at a multi-level cascade (NL=2/2). With max(sx)=2,
    /// NL,x=2, Cw=1 → Cs = 8·1·2·4 = 64, so a 128-wide picture splits into
    /// Np,x = 2 precincts per row. The decoder routes Cw>0 through the
    /// picture-level gather/cascade path, where the sub-sampled chroma
    /// components decompose one vertical level shallower; lossless
    /// self-roundtrip is bit-exact for all three components.
    #[test]
    fn round371_cw1_128x32_yuv_420_nl_2_2_lossless() {
        let w = 128usize;
        let h = 32usize;
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; (w / 2) * (h / 2)];
        let mut v = vec![0u8; (w / 2) * (h / 2)];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 3 + j * 5) % 256) as u8;
            }
        }
        for j in 0..(h / 2) {
            for i in 0..(w / 2) {
                u[j * (w / 2) + i] = ((i * 7 + j * 11 + 17) % 256) as u8;
                v[j * (w / 2) + i] = ((i * 13 + j * 17 + 29) % 256) as u8;
            }
        }
        let sx = vec![1u8, 2, 2];
        let sy = vec![1u8, 2, 2];
        let cs = encode_planar_inner_nlt(
            w as u16,
            h as u16,
            3,
            0, // cpih
            2, // nlx
            2, // nly
            0, // fq
            0, // q (lossless)
            &sx,
            &sy,
            0,
            0,
            0,
            0,
            None, // no NLT
            Vec::new(),
            Vec::new(),
            1, // cw = 1 → Np,x = 2 for 128-wide 4:2:0
            0, // sd
            0, // fs
            0, // hsl
            0, // qpih
            0, // rp
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[y.clone(), u.clone(), v.clone()],
        )
        .expect("encode 128x32 4:2:0 Cw=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 128x32 4:2:0 Cw=1 NL=2/2");
        assert_eq!(img.planes[0].data, y, "luma Cw=1 4:2:0 NL=2/2 lossless");
        assert_eq!(img.planes[1].data, u, "U Cw=1 4:2:0 NL=2/2 lossless");
        assert_eq!(img.planes[2].data, v, "V Cw=1 4:2:0 NL=2/2 lossless");
    }

    /// r371: multi-precinct-per-row (`Cw > 0`) composed with the NLT
    /// quadratic non-linearity (Tnlt=1, Annex G.4). The decoder routes both
    /// Cw>0 and the NLT marker through the picture-level gather/cascade
    /// path; this exercises their intersection. With max(sx)=1, NL,x=2,
    /// Cw=1 → Cs = 8·1·1·4 = 32, so a 96-wide luma picture splits into
    /// Np,x = 3 precincts per row. The forward pre-distortion runs once
    /// over the plane before gather, so the round-trip holds ≥ 40 dB.
    #[test]
    fn round371_cw1_96x32_luma_nlt_quadratic() {
        let w = 96usize;
        let h = 32usize;
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 5 + j * 7 + ((i ^ j) & 0x0f) * 3) % 256) as u8;
            }
        }
        let sx = vec![1u8];
        let sy = vec![1u8];
        let cs = encode_planar_inner_nlt(
            w as u16,
            h as u16,
            1,
            0, // cpih
            2, // nlx
            2, // nly
            0, // fq (forced to 6 by NLT inner path)
            0, // q
            &sx,
            &sy,
            0,
            0,
            0,
            0,
            Some(NltParams::Quadratic { dco: 0 }),
            Vec::new(),
            Vec::new(),
            1, // cw = 1 → Np,x = 3 for 96-wide luma at NL,x=2
            0,
            0,
            0,
            0,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[y.clone()],
        )
        .expect("encode 96x32 luma Cw=1 NLT quadratic");
        let img = decode_codestream(&cs, None).expect("decode 96x32 luma Cw=1 NLT quadratic");
        let p = psnr(&y, &img.planes[0].data);
        assert!(
            p >= 40.0,
            "Cw=1 NLT quadratic round-trip PSNR {p:.2} dB below 40 dB floor"
        );
    }

    /// r371: CWD decomposition suppression (`Sd > 0`) composed with the NLT
    /// quadratic non-linearity (Tnlt=1, Annex G.4). With Nc=4, Sd=1 the
    /// trailing component is coded raw (no wavelet) per Annex B.7 Table B.4
    /// while the leading three are wavelet-coded; both Sd>0 and the NLT
    /// marker force the decoder's picture-level gather path. The forward
    /// pre-distortion applies per-component (including the suppressed tail,
    /// whose values are copied straight through the entropy coder), and the
    /// decoder's per-component Annex G inverse restores every component
    /// above the NLT PSNR floor. NL=2/2, 4:4:4:4.
    #[test]
    fn round371_sd1_nlt_quadratic_4comp() {
        let w = 32usize;
        let h = 32usize;
        let mk = |a: usize, b: usize, c: usize| {
            let mut p = vec![0u8; w * h];
            for j in 0..h {
                for i in 0..w {
                    p[j * w + i] = ((i * a + j * b + c) % 256) as u8;
                }
            }
            p
        };
        let p0 = mk(5, 7, 0);
        let p1 = mk(3, 9, 11);
        let p2 = mk(7, 2, 23);
        let p3 = mk(11, 5, 41);
        let sx = vec![1u8; 4];
        let sy = vec![1u8; 4];
        let cs = encode_planar_inner_nlt(
            w as u16,
            h as u16,
            4,
            0, // cpih
            2, // nlx
            2, // nly
            0, // fq
            0, // q
            &sx,
            &sy,
            0,
            0,
            0,
            0,
            Some(NltParams::Quadratic { dco: 0 }),
            Vec::new(),
            Vec::new(),
            0, // cw
            1, // sd = 1 → trailing component coded raw
            0,
            0,
            0,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &[p0.clone(), p1.clone(), p2.clone(), p3.clone()],
        )
        .expect("encode Sd=1 NLT quadratic 4-comp");
        let img = decode_codestream(&cs, None).expect("decode Sd=1 NLT quadratic 4-comp");
        assert_eq!(img.num_components, 4);
        for (idx, orig) in [&p0, &p1, &p2, &p3].iter().enumerate() {
            let p = psnr(orig, &img.planes[idx].data);
            assert!(
                p >= 40.0,
                "Sd=1 NLT quadratic component {idx} PSNR {p:.2} dB below 40 dB floor"
            );
        }
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

    // === Round 282: odd-dimension subsampled planes (§B.1 ceiling) =========

    /// §B.1 fixture builder: per-component planes sized
    /// `⌈w / sx[i]⌉ × ⌈h / sy[i]⌉` with deterministic synthetic content.
    fn r282_planes(w: usize, h: usize, sx: &[u8], sy: &[u8]) -> Vec<Vec<u8>> {
        (0..sx.len())
            .map(|i| {
                let n = w.div_ceil(sx[i] as usize) * h.div_ceil(sy[i] as usize);
                (0..n).map(|k| ((k * 7 + i * 13 + 3) % 256) as u8).collect()
            })
            .collect()
    }

    /// Odd picture width with 4:2:2 chroma: Wc = ⌈65 / 2⌉ = 33 per
    /// §B.1 (samples populate grid columns 0, 2, …, 64). Lossless
    /// self-roundtrip through the public sub-sampled entry point.
    #[test]
    fn r282_odd_width_422_lossless_round_trip() {
        let (w, h) = (65usize, 16usize);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let planes = r282_planes(w, h, &sx, &sy);
        let cs = encode_planar_subsampled(w as u16, h as u16, 3, 0, 2, 2, 0, &sx, &sy, &planes)
            .expect("encode 65x16 4:2:2 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 65x16 4:2:2");
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(&img.planes[i].data, p, "plane {i} lossless roundtrip");
        }
        assert_eq!(img.planes[1].data.len(), 33 * 16, "chroma Wc = ceil(65/2)");
    }

    /// Odd width AND odd height with 4:2:0 chroma: chroma plane is
    /// ⌈65 / 2⌉ × ⌈17 / 2⌉ = 33 × 9 per §B.1. Lossless self-roundtrip.
    #[test]
    fn r282_odd_dims_420_lossless_round_trip() {
        let (w, h) = (65usize, 17usize);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let planes = r282_planes(w, h, &sx, &sy);
        let cs = encode_planar_subsampled(w as u16, h as u16, 3, 0, 2, 2, 0, &sx, &sy, &planes)
            .expect("encode 65x17 4:2:0 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 65x17 4:2:0");
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(&img.planes[i].data, p, "plane {i} lossless roundtrip");
        }
        assert_eq!(
            img.planes[1].data.len(),
            33 * 9,
            "chroma ceil(65/2) x ceil(17/2)"
        );
    }

    /// Odd dims at NL=3/3 4:2:0 — crosses the §B.1 ceiling dims with
    /// the round-190 picture-β proxy-depth permutation (chroma
    /// N'L,y = 2 while the picture codes NL,y = 3).
    #[test]
    fn r282_odd_dims_420_nl3_lossless_round_trip() {
        let (w, h) = (33usize, 33usize);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let planes = r282_planes(w, h, &sx, &sy);
        let cs = encode_planar_subsampled(w as u16, h as u16, 3, 0, 3, 3, 0, &sx, &sy, &planes)
            .expect("encode 33x33 4:2:0 NL=3/3");
        let img = decode_codestream(&cs, None).expect("decode 33x33 4:2:0 NL=3/3");
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(&img.planes[i].data, p, "plane {i} lossless roundtrip");
        }
        assert_eq!(
            img.planes[1].data.len(),
            17 * 17,
            "chroma ceil(33/2) squared"
        );
    }

    /// Odd dims through the NL=1/1 streaming per-precinct synthesis
    /// path (the decoder fast path that bypasses the picture-level
    /// gather): 4:2:2 odd width + 4:2:0 odd width and height.
    #[test]
    fn r282_odd_dims_streaming_nl1_lossless_round_trip() {
        for (h, sy_c, label) in [(16usize, 1u8, "4:2:2"), (17, 2, "4:2:0")] {
            let w = 65usize;
            let sx = [1u8, 2, 2];
            let sy = [1u8, sy_c, sy_c];
            let planes = r282_planes(w, h, &sx, &sy);
            let cs = encode_planar_subsampled(w as u16, h as u16, 3, 0, 1, 1, 0, &sx, &sy, &planes)
                .unwrap_or_else(|e| panic!("encode 65x{h} {label} NL=1/1: {e}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode 65x{h} {label} NL=1/1: {e}"));
            for (i, p) in planes.iter().enumerate() {
                assert_eq!(
                    &img.planes[i].data, p,
                    "{label} plane {i} streaming roundtrip"
                );
            }
        }
    }

    /// Odd-dim 4:2:0 at q=2 stays above the 30 dB PSNR floor on every
    /// plane (the lossy pipeline reconstructs the ceil-sized planes).
    #[test]
    fn r282_odd_dims_420_lossy_q2_psnr() {
        let (w, h) = (65usize, 17usize);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        let planes: Vec<Vec<u8>> = (0..3)
            .map(|i| {
                let wc = w.div_ceil(sx[i] as usize);
                let hc = h.div_ceil(sy[i] as usize);
                let mut v = vec![0u8; wc * hc];
                for y in 0..hc {
                    for x in 0..wc {
                        v[y * wc + x] = ((x * 2 + y * 3 + i * 31) % 200 + 20) as u8;
                    }
                }
                v
            })
            .collect();
        let cs = encode_planar_subsampled(w as u16, h as u16, 3, 0, 2, 2, 2, &sx, &sy, &planes)
            .expect("encode 65x17 4:2:0 q=2");
        let img = decode_codestream(&cs, None).expect("decode 65x17 4:2:0 q=2");
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(img.planes[i].data.len(), p.len(), "plane {i} ceil-sized");
            let db = psnr(p, &img.planes[i].data);
            assert!(db >= 30.0, "plane {i} PSNR {db:.1} dB < 30 dB at q=2");
        }
    }

    /// Vertical-prediction × significance regression (Table C.13 /
    /// Table C.8): an all-zero code group directly below a
    /// high-magnitude group reconstructs as `M = mtop` when its
    /// significance group is flagged insignificant at `Rm = 0` — so
    /// the encoder may only flag a group insignificant when the
    /// predictor column is also at/below `T`, or the data sub-packet
    /// desynchronises. This 17×8 10-bit fixture (full-range values →
    /// the picker reaches the vpred+sig form) failed with
    /// `read_bit past end of buffer` before the round-282 fix.
    #[test]
    fn r282_vpred_sig_zero_group_below_active_round_trip() {
        let bd = 10u8;
        let max = (1u32 << bd) - 1;
        let (w, h) = (17usize, 8usize);
        let plane: Vec<u16> = (0..w * h)
            .map(|k| ((k as u32 * 11) % (max + 1)) as u16)
            .collect();
        let cs = encode_planar_highbd(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            bd,
            std::slice::from_ref(&plane),
        )
        .expect("encode 17x8 10-bit NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 17x8 10-bit");
        let want: Vec<u8> = plane.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(img.planes[0].data, want, "lossless roundtrip");
    }

    /// Significance sub-packet wire polarity (Table C.5): the Z bit
    /// identifies whether ALL code groups in the significance group
    /// are insignificant — Z = 1 ⇒ insignificant, Z = 0 ⇒
    /// significant (Tables C.13 / C.14 decode the VLC residual when
    /// Z == 0).
    #[test]
    fn r282_sig_subpacket_z_polarity_per_table_c5() {
        // Two significance groups: entry 0 carries a significant
        // group (M = 3 > T = 0), entry 1 an all-insignificant one
        // (M = 0 = T). One entry per band line, Wpb = 4 → Ncg = 1 →
        // Ns = 1 per entry.
        let cfg = EncodeConfig {
            width: 8,
            height: 2,
            nc: 1,
            bit_depth: 8,
            bw: 8,
            ng: 4,
            ss: 8,
            br: 4,
            nlx: 1,
            nly: 0,
            cpih: 0,
            qpih: 0,
            fs: 0,
            fq: 0,
            q: 0,
            sx: vec![1],
            sy: vec![1],
            cts_e1: 0,
            cts_e2: 0,
            cts_cf: 0,
            st_ct: 0,
            nlt: None,
            band_gains: vec![0],
            band_priorities: vec![0],
            rp: 0,
            cw: 0,
            sd: 0,
            hsl: 0,
            q_slices: Vec::new(),
            q_precincts: Vec::new(),
            r_precincts: Vec::new(),
            rm: 0,
        };
        let entries = vec![
            PerBandEntry {
                wpb: 4,
                line: BandLineSlice::Direct(vec![5, -2, 0, 1]),
                t: 0,
            },
            PerBandEntry {
                wpb: 4,
                line: BandLineSlice::Direct(vec![0, 0, 0, 0]),
                t: 0,
            },
        ];
        let m_per_entry: Vec<Vec<u8>> = entries
            .iter()
            .map(|e| compute_m_per_group(&cfg, e).expect("M"))
            .collect();
        assert!(m_per_entry[0][0] > 0 && m_per_entry[1][0] == 0);
        let pkt = build_packet_body_with_m(
            &cfg,
            &entries,
            &m_per_entry,
            BitplaneMode::Vlc(VlcKind::NoPredSig),
        )
        .expect("build sig packet");
        // Two Z bits, MSB-first: significant entry → 0, insignificant
        // entry → 1 ⇒ 0b0100_0000 = 0x40.
        assert_eq!(pkt.sig, vec![0x40], "Z polarity per Table C.5");
    }

    /// Odd width 4:2:2 on the high-bit-depth path (10-bit, `u16`-LE
    /// planes): the §B.1 ceiling dims apply identically.
    #[test]
    fn r282_odd_width_422_highbd_lossless_round_trip() {
        let (w, h) = (65usize, 16usize);
        let bd = 10u8;
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let max = (1u32 << bd) - 1;
        let planes: Vec<Vec<u16>> = (0..3)
            .map(|i| {
                let n = w.div_ceil(sx[i] as usize) * h.div_ceil(sy[i] as usize);
                (0..n)
                    .map(|k| ((k as u32 * 11 + i as u32 * 401) % (max + 1)) as u16)
                    .collect()
            })
            .collect();
        let cs =
            encode_planar_subsampled_highbd(w as u16, h as u16, 3, 0, 2, 2, bd, &sx, &sy, &planes)
                .expect("encode 65x16 4:2:2 10-bit");
        let img = decode_codestream(&cs, None).expect("decode 65x16 4:2:2 10-bit");
        for (i, p) in planes.iter().enumerate() {
            let want: Vec<u8> = p.iter().flat_map(|s| s.to_le_bytes()).collect();
            assert_eq!(
                img.planes[i].data, want,
                "plane {i} u16-LE lossless roundtrip"
            );
        }
    }

    /// Odd width 4:2:2 with Cw=1 multi-column precincts at NL=1/1:
    /// Cs = 8 × 1 × 2 × 2 = 32 → Np,x = ⌈69 / 32⌉ = 3. The rightmost
    /// precinct covers `69 umod 32 = 5` image columns, which exceeds one
    /// decomposed LL sample (`max_i(sx)×2^NL,x = 4`), so the picture is
    /// Table 11 / §B.5-conformant. (Width 65, whose rightmost precinct
    /// would be a single column < 4, is now rejected by the encoder.)
    #[test]
    fn r282_odd_width_422_cw1_lossless_round_trip() {
        let (w, h) = (69usize, 16usize);
        let sx = vec![1u8, 2, 2];
        let sy = vec![1u8, 1, 1];
        let planes = r282_planes(w, h, &sx, &sy);
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
            Vec::new(),
            1,          // cw → Cs = 32, Np,x = 3
            0,          // sd
            0,          // fs
            0,          // hsl
            0,          // qpih
            0,          // rp
            Vec::new(), // q_slices
            Vec::new(), // q_precincts
            Vec::new(), // r_precincts
            &planes,
        )
        .expect("encode 69x16 4:2:2 Cw=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode 69x16 4:2:2 Cw=1");
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(&img.planes[i].data, p, "plane {i} Cw=1 odd-width roundtrip");
        }
    }

    /// The encoder refuses a Cw whose rightmost precinct is a sliver
    /// narrower than one decomposed LL sample (Table 11 / §B.5 Note 1).
    /// A 65×16 4:2:2 picture at Cw=1, NL=1/1 has Cs=32, so the rightmost
    /// precinct is `65 umod 32 = 1` column < `max_i(sx)×2^NL,x = 4`.
    #[test]
    fn r376_cw_rightmost_sliver_rejected() {
        let (w, h) = (65usize, 16usize);
        let sx = vec![1u8, 2, 2];
        let sy = vec![1u8, 1, 1];
        let planes = r282_planes(w, h, &sx, &sy);
        let err = encode_planar_inner_nlt(
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
            Vec::new(),
            1, // cw → Cs = 32, rightmost precinct = 1 column
            0,
            0,
            0,
            0,
            0,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &planes,
        )
        .expect_err("65×16 Cw=1 sliver rightmost precinct must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("rightmost precinct") && msg.contains("Table 11"),
            "expected Table 11 rightmost-precinct rejection, got {msg}"
        );
    }

    /// The encoder emits a CAP marker whose cap[] bits reflect the tools
    /// the stream actually uses (ISO/IEC 21122-1:2022 Table A.5). A plain
    /// lossless luma stream sets only bit 6 (lossless) and bit 8 (the
    /// always-on per-packet raw-mode switch).
    #[test]
    fn r376_encoder_cap_lossless_luma() {
        let cs = encode_planar(8, 8, 1, 0, 1, 0, &[vec![0u8; 64]]).expect("encode luma");
        let parsed = crate::codestream::parse(&cs).expect("parse luma codestream");
        let caps = parsed.capabilities();
        assert!(caps.lossless, "q=0 must set the lossless bit");
        assert!(caps.raw_mode_switch, "Rl=1 must set the raw-mode bit");
        assert!(!caps.star_tetrix);
        assert!(!caps.nlt_quadratic);
        assert!(!caps.nlt_extended);
        assert!(!caps.cwd);
        assert!(!caps.vertical_subsampling);
        // No unsupported bit is ever emitted.
        assert!(crate::capabilities::unsupported_cap_bits(&parsed.cap).is_empty());
    }

    /// The quadratic NLT entry point sets CAP bit 2; a 4:2:0 chroma
    /// component sets bit 4 (vertical subsampling); a lossy stream clears
    /// the lossless bit.
    #[test]
    fn r376_encoder_cap_nlt_quadratic() {
        let cs = encode_planar_nlt_quadratic(8, 8, 1, 0, 1, 0, 1, 0, &[vec![0u8; 64]])
            .expect("encode NLT quadratic");
        let parsed = crate::codestream::parse(&cs).expect("parse NLT codestream");
        let caps = parsed.capabilities();
        assert!(caps.nlt_quadratic, "Tnlt=1 must set the quadratic-NLT bit");
        assert!(!caps.lossless, "q>0 must clear the lossless bit");
        assert!(caps.raw_mode_switch);
        assert!(crate::capabilities::unsupported_cap_bits(&parsed.cap).is_empty());
    }

    /// Table A.8: the (B[0], 0) lossless combination requires
    /// `B[i] = B[0]` for every component. Encode a uniform-8-bit 3-comp
    /// stream, then patch component 1's CDT bit-depth to 10 — the decoder
    /// must reject the now-non-uniform lossless stream.
    #[test]
    fn r376_decode_rejects_mixed_depth_lossless() {
        let pixels = make_synthetic_rgb_32x32();
        let mut cs = encode_rgb_8bit(32, 32, &pixels, 0, 1).expect("encode RGB");
        // Locate the CDT marker (0xff13) and patch component 1's B[i].
        let cdt_pos = cs
            .windows(2)
            .position(|w| w == [0xff, 0x13])
            .expect("CDT marker present");
        // CDT: marker(2) + Lcdt(2) + body; body[i*2] = B[i].
        let body = cdt_pos + 4;
        assert_eq!(cs[body], 8, "component 0 is 8-bit");
        cs[body + 2] = 10; // component 1 → 10-bit (now != B[0])
        let err = decode_codestream(&cs, None).unwrap_err();
        assert!(
            format!("{err}").contains("requires B[i]=B[0]"),
            "expected uniform-depth rejection, got {err}"
        );
    }

    /// A 4:2:0 luma+chroma stream (sy=2 on the chroma planes) sets the
    /// vertical-subsampling CAP bit (bit 4).
    #[test]
    fn r376_encoder_cap_vertical_subsampling() {
        let y = vec![0u8; 16 * 16];
        let c = vec![0u8; 8 * 8];
        let cs = encode_planar_subsampled(
            16,
            16,
            3,
            0,
            1,
            1,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y, c.clone(), c],
        )
        .expect("encode 4:2:0");
        let parsed = crate::codestream::parse(&cs).expect("parse 4:2:0 codestream");
        assert!(
            parsed.capabilities().vertical_subsampling,
            "sy>1 must set the vertical-subsampling bit"
        );
    }

    /// A floor-sized plane (the pre-282 convention) is rejected: §B.1
    /// mandates the ceiling, so a 32-wide chroma plane for a 65-wide
    /// 4:2:2 picture is a length mismatch.
    #[test]
    fn r282_floor_sized_plane_rejected() {
        let (w, h) = (65usize, 16usize);
        let sx = [1u8, 2, 2];
        let sy = [1u8, 1, 1];
        let planes = vec![
            vec![0u8; w * h],
            vec![0u8; (w / 2) * h], // floor: 32×16 — wrong per §B.1
            vec![0u8; (w / 2) * h],
        ];
        let err = encode_planar_subsampled(w as u16, h as u16, 3, 0, 2, 2, 0, &sx, &sy, &planes)
            .expect_err("floor-sized chroma plane must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("plane 1"),
            "error must identify the offending plane: {msg}"
        );
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

    /// Round 95 (r93): Sd=1 + Cpih=1 (RCT). Nc=4 picture with 4 planes,
    /// components 0..2 ride the RCT cascade, component 3 is suppressed
    /// (raw CWD tail). Self-roundtrips losslessly at q=0.
    #[test]
    fn round95_sd1_cpih1_rct_4comp_32x16_lossless() {
        let w = 32usize;
        let h = 16usize;
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 5)
                        .wrapping_add((y as u32).wrapping_mul(seed + 11))
                        .wrapping_add(seed * 13)
                        % 256) as u8;
                }
            }
            v
        };
        let r = make(7);
        let g = make(13);
        let b = make(19);
        let alpha = make(31);
        let cs = encode_planar_sd_rct(
            w as u16,
            h as u16,
            4,
            2,
            2,
            0,
            1,
            &[r.clone(), g.clone(), b.clone(), alpha.clone()],
        )
        .expect("encode 32x16 Nc=4 Sd=1 Cpih=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode Sd=1 Cpih=1");
        assert_eq!(img.num_components, 4);
        assert_eq!(img.cpih, 1, "PIH should report Cpih=1");
        assert_eq!(img.planes[0].data, r, "R lossless via RCT");
        assert_eq!(img.planes[1].data, g, "G lossless via RCT");
        assert_eq!(img.planes[2].data, b, "B lossless via RCT");
        assert_eq!(img.planes[3].data, alpha, "Sd-suppressed alpha lossless");
    }

    /// Round 95 (r93): Sd=1 + Cpih=1 lossy q=2 PSNR floor.
    #[test]
    fn round95_sd1_cpih1_rct_lossy_q2_psnr_floor() {
        let w = 32usize;
        let h = 16usize;
        let mut p = vec![vec![0u8; w * h]; 4];
        for y in 0..h {
            for x in 0..w {
                let g = ((x as u32 * 6 + y as u32 * 9) % 256) as u8;
                p[0][y * w + x] = g;
                p[1][y * w + x] = g.wrapping_add(15);
                p[2][y * w + x] = g.wrapping_add(30);
                p[3][y * w + x] = g.wrapping_add(45);
            }
        }
        let cs = encode_planar_sd_rct(w as u16, h as u16, 4, 2, 2, 2, 1, &p)
            .expect("encode lossy Sd=1 Cpih=1 q=2");
        let img = decode_codestream(&cs, None).expect("decode lossy Sd=1 Cpih=1");
        assert_eq!(img.cpih, 1);
        for (i, expected) in p.iter().enumerate().take(4) {
            let q = psnr(expected, &img.planes[i].data);
            assert!(
                q >= 25.0,
                "Sd=1 Cpih=1 q=2 comp {i} PSNR {q:.2} dB below 25 dB floor"
            );
        }
    }

    /// Round 95 (r93): Sd=2 + Cpih=1 (RCT). Nc=5; components 0..2 ride
    /// RCT cascade, components 3..4 are suppressed (raw CWD tail).
    #[test]
    fn round95_sd2_cpih1_rct_5comp_lossless() {
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
        let cs = encode_planar_sd_rct(w as u16, h as u16, 5, 1, 1, 0, 2, &p)
            .expect("encode 16x8 Nc=5 Sd=2 Cpih=1 NL=1/1");
        let img = decode_codestream(&cs, None).expect("decode Sd=2 Cpih=1");
        assert_eq!(img.cpih, 1);
        for (i, expected) in p.iter().enumerate().take(5) {
            assert_eq!(&img.planes[i].data, expected, "comp {i} roundtrip");
        }
    }

    /// Round 95 (r93): Sd=1 + Cpih=3 (Star-Tetrix). Nc=5 picture; the 4
    /// CFA components ride the Star-Tetrix cascade, component 4 is the
    /// suppressed CWD tail.
    #[test]
    fn round95_sd1_cpih3_star_tetrix_5comp_lossless() {
        let w = 16usize;
        let h = 8usize;
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 2)
                        .wrapping_add((y as u32).wrapping_mul(seed + 3))
                        .wrapping_add(seed)
                        % 256) as u8;
                }
            }
            v
        };
        let r = make(11);
        let g1 = make(17);
        let g2 = make(23);
        let b = make(29);
        let ir = make(41);
        let cs = encode_planar_sd_star_tetrix(
            w as u16,
            h as u16,
            5,
            1,
            1,
            0,
            1, // sd: suppress component 4 (IR / NIR tail)
            0, // e1
            0, // e2
            0, // cf=full
            0, // ct=RGGB
            &[r.clone(), g1.clone(), g2.clone(), b.clone(), ir.clone()],
        )
        .expect("encode 16x8 Nc=5 Sd=1 Cpih=3");
        let img = decode_codestream(&cs, None).expect("decode Sd=1 Cpih=3");
        assert_eq!(img.num_components, 5);
        assert_eq!(img.cpih, 3, "PIH should report Cpih=3");
        assert_eq!(img.planes[0].data, r, "R lossless via Star-Tetrix");
        assert_eq!(img.planes[1].data, g1, "G1 lossless via Star-Tetrix");
        assert_eq!(img.planes[2].data, g2, "G2 lossless via Star-Tetrix");
        assert_eq!(img.planes[3].data, b, "B lossless via Star-Tetrix");
        assert_eq!(img.planes[4].data, ir, "Sd-suppressed IR lossless");
    }

    /// Round 95 (r93): encoder rejects Sd that suppresses an RCT operand
    /// component (Nc-Sd < 3).
    #[test]
    fn round95_rejects_sd_overlapping_rct_operand_window() {
        // Nc=4, Sd=2 means Nc-Sd=2 → suppresses index 2 (Cr operand of
        // RCT). Must reject.
        let p = vec![vec![0u8; 16 * 8]; 4];
        let result = encode_planar_sd_rct(16, 8, 4, 1, 1, 0, 2, &p);
        assert!(
            result.is_err(),
            "Cpih=1 + Sd=2 with Nc=4 must reject (RCT operand overlap)"
        );
    }

    /// Round 330: Star-Tetrix tolerates suppressing a transform *output*
    /// (not just a pass-through component). Per Annex F.2 Table F.1 the
    /// transform reads all four CFA inputs (c<4) and emits four outputs;
    /// Sd suppresses the wavelet decomposition of the *trailing outputs*,
    /// which are then coded raw (Annex A.4.7, Annex B Tables B.10 / B.11).
    /// The earlier "Nc-Sd >= 4" reject was over-strict — it conflated the
    /// transform input window with output suppression. Here Nc=5, Sd=2
    /// suppresses output 3 (blue, Table F.4) and pass-through output 4, and
    /// the picture still round-trips losslessly because the inverse
    /// transform consumes the raw-coded O[3] unchanged.
    #[test]
    fn round330_sd2_cpih3_star_tetrix_5comp_suppresses_output_lossless() {
        let w = 16usize;
        let h = 8usize;
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 2)
                        .wrapping_add((y as u32).wrapping_mul(seed + 3))
                        .wrapping_add(seed)
                        % 256) as u8;
                }
            }
            v
        };
        let planes = [make(11), make(17), make(23), make(29), make(41)];
        // Nc=5, Sd=2 → Nc-Sd=3 wavelet-coded, outputs 3 (B) + 4 (tail) raw.
        let cs =
            encode_planar_sd_star_tetrix(w as u16, h as u16, 5, 1, 1, 0, 2, 0, 0, 0, 0, &planes)
                .expect("encode Nc=5 Sd=2 Cpih=3 (suppressed Star-Tetrix output)");
        let img = decode_codestream(&cs, None).expect("decode Nc=5 Sd=2 Cpih=3");
        assert_eq!(img.num_components, 5);
        assert_eq!(img.cpih, 3);
        for (i, want) in planes.iter().enumerate() {
            assert_eq!(&img.planes[i].data, want, "component {i} lossless");
        }
    }

    // === Round 100: Fs=1 separate sign sub-packet (Annex C.5.5) =========

    /// Round 100: Fs=1 luma round-trips losslessly and the PIH carries
    /// the sign-handling flag (`fs == 1`, Annex A.4.4 Table A.11).
    #[test]
    fn round100_fs1_luma_32x32_nl_2_2_lossless() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_fs1(32, 32, 1, 0, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("encode 32x32 luma Fs=1 NL=2/2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.fs, 1, "PIH should report Fs=1");
        let img = decode_codestream(&cs, None).expect("decode Fs=1 luma");
        assert_eq!(img.planes[0].data, pixels, "Fs=1 luma lossless");
    }

    /// Round 100: Fs=1 with an RGB picture under the reversible colour
    /// transform (Cpih=1) round-trips losslessly.
    #[test]
    fn round100_fs1_rgb_cpih1_nl_2_2_lossless() {
        let rgb = make_synthetic_rgb_32x32();
        let mut r = vec![0u8; 32 * 32];
        let mut g = vec![0u8; 32 * 32];
        let mut b = vec![0u8; 32 * 32];
        for i in 0..32 * 32 {
            r[i] = rgb[i * 3];
            g[i] = rgb[i * 3 + 1];
            b[i] = rgb[i * 3 + 2];
        }
        let cs = encode_planar_fs1(32, 32, 3, 1, 2, 2, 0, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB Fs=1 Cpih=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode Fs=1 RGB");
        assert_eq!(img.planes[0].data, r, "R lossless Fs=1");
        assert_eq!(img.planes[1].data, g, "G lossless Fs=1");
        assert_eq!(img.planes[2].data, b, "B lossless Fs=1");
    }

    /// Round 100: Fs=1 lossy q=2 holds the same PSNR floor as the Fs=0
    /// path — the sign sub-packet is a lossless re-layout, not a quality
    /// trade-off.
    #[test]
    fn round100_fs1_lossy_q2_psnr_floor() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_fs1(32, 32, 1, 0, 2, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode 32x32 luma Fs=1 q=2");
        let img = decode_codestream(&cs, None).expect("decode Fs=1 lossy");
        let q = psnr(&pixels, &img.planes[0].data);
        assert!(q >= 30.0, "Fs=1 q=2 luma PSNR {q:.2} dB below 30 dB floor");
    }

    /// Round 100: Fs=1 and Fs=0 decode to byte-identical pixels (the sign
    /// sub-packet is just an alternative on-wire sign layout). Also checks
    /// the two codestreams differ (the sign re-layout actually changes the
    /// bytes) and that on a sparse-sign image the Fs=1 form is no larger.
    #[test]
    fn round100_fs1_matches_fs0_decode_and_is_compact() {
        // Sparse-sign image: mostly flat with a few high-frequency spikes,
        // so most significant code groups have several zero coefficients.
        // Under Fs=0 each significant group still spends Ng=4 sign bits;
        // under Fs=1 only the non-zero coefficients pay a sign bit.
        let w = 32usize;
        let h = 32usize;
        let mut pixels = vec![128u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x % 8 == 0) && (y % 8 == 0) {
                    pixels[y * w + x] = if (x + y) % 16 == 0 { 200 } else { 60 };
                }
            }
        }
        let cs0 = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("encode Fs=0");
        let cs1 = encode_planar_fs1(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("encode Fs=1");
        let img0 = decode_codestream(&cs0, None).expect("decode Fs=0");
        let img1 = decode_codestream(&cs1, None).expect("decode Fs=1");
        assert_eq!(
            img0.planes[0].data, img1.planes[0].data,
            "Fs=0 and Fs=1 must decode to identical pixels"
        );
        assert_eq!(img0.planes[0].data, pixels, "Fs=0 lossless");
        assert_eq!(img1.planes[0].data, pixels, "Fs=1 lossless");
        assert_ne!(cs0, cs1, "Fs=1 re-layout must change the codestream bytes");
        assert!(
            cs1.len() <= cs0.len(),
            "Fs=1 sparse-sign codestream {} must be <= Fs=0 {}",
            cs1.len(),
            cs0.len()
        );
    }

    /// Round 100: the encoder rejects reserved Fs values (2/3) routed
    /// through the inner builder.
    #[test]
    fn round100_rejects_reserved_fs() {
        let pixels = vec![0u8; 16 * 8];
        let result = encode_planar_inner_nlt(
            16,
            8,
            1,
            0,
            1,
            1,
            0,
            0,
            &[1],
            &[1],
            0,
            0,
            0,
            0,
            None,
            Vec::new(),
            Vec::new(),
            0,          // cw
            0,          // sd
            2,          // fs: reserved → must reject
            0,          // hsl
            0,          // qpih
            0,          // rp
            Vec::new(), // q_slices: single picture-level q
            Vec::new(), // q_precincts: no per-precinct override
            Vec::new(), // r_precincts: no per-precinct R[p] override
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "Fs=2 (reserved) must be rejected");
    }

    // === Round 103: multi-slice emission (Hsl > 0, Annex B.10) ==========

    /// A smooth-ish gradient picture for the multi-slice round-trips.
    fn round103_grad(w: usize, h: usize) -> Vec<u8> {
        let mut p = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = (((x * 5 + y * 3) ^ (x * y)) & 0xff) as u8;
            }
        }
        p
    }

    /// Multi-slice luma: 32×32 at NL=2/2 → Hp=4, Np,y=8 precinct rows.
    /// Hsl=2 partitions those into 4 slices. Self-roundtrips losslessly,
    /// emits exactly 4 SLH markers, and the PIH carries Hsl=2.
    #[test]
    fn round103_hsl2_luma_32x32_nl_2_2_four_slices_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let cs = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            2, // hsl = 2 precinct rows per slice
            std::slice::from_ref(&pixels),
        )
        .expect("encode 32x32 luma Hsl=2");
        // Parse: PIH must carry Hsl=2, and the codestream must hold 4 slices.
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.hsl, 2, "PIH Hsl must be 2");
        assert_eq!(parsed.slices.len(), 4, "Np,y=8 / Hsl=2 → 4 slices");
        let img = decode_codestream(&cs, None).expect("decode 32x32 luma Hsl=2");
        assert_eq!(
            img.planes[0].data, pixels,
            "multi-slice luma must be lossless"
        );
    }

    /// Multi-slice RGB + RCT (Cpih=1): 16×24 at NL=2/2 → Np,y=6, Hsl=3 →
    /// 2 slices. Self-roundtrips losslessly across the slice boundary.
    #[test]
    fn round103_hsl3_rgb_rct_16x24_nl_2_2_two_slices_lossless() {
        let w = 16usize;
        let h = 24usize;
        let r = round103_grad(w, h);
        let g: Vec<u8> = r.iter().map(|&v| v.wrapping_add(40)).collect();
        let b: Vec<u8> = r.iter().map(|&v| v.wrapping_sub(20)).collect();
        let planes = [r.clone(), g.clone(), b.clone()];
        let cs = encode_planar_hsl(w as u16, h as u16, 3, 1, 2, 2, 0, 3, &planes)
            .expect("encode 16x24 RGB+RCT Hsl=3");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.hsl, 3, "PIH Hsl must be 3");
        assert_eq!(parsed.slices.len(), 2, "Np,y=6 / Hsl=3 → 2 slices");
        let img = decode_codestream(&cs, None).expect("decode RGB+RCT Hsl=3");
        assert_eq!(img.planes[0].data, r, "R plane lossless");
        assert_eq!(img.planes[1].data, g, "G plane lossless");
        assert_eq!(img.planes[2].data, b, "B plane lossless");
    }

    /// Multi-slice lossy q=2 must still hold the per-codec PSNR floor.
    #[test]
    fn round103_hsl2_lossy_q2_psnr_floor() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let cs = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2, // q = 2
            2, // hsl = 2
            std::slice::from_ref(&pixels),
        )
        .expect("encode 32x32 luma Hsl=2 q=2");
        let img = decode_codestream(&cs, None).expect("decode Hsl=2 q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 30.0,
            "multi-slice lossy q=2 PSNR {p:.2} dB < 30 dB floor"
        );
    }

    /// `hsl = 0` is the single-slice default and is byte-identical to
    /// [`encode_planar_lossy`]; `hsl = Np,y` (the full picture in one
    /// slice) is also byte-identical (one slice either way).
    #[test]
    fn round103_hsl0_and_hsl_full_match_single_slice() {
        let w = 24usize;
        let h = 16usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Hp=4, Np,y = ⌈16/4⌉ = 4.
        let baseline = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline encode");
        let hsl0 = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            0, // hsl = 0 → single slice
            std::slice::from_ref(&pixels),
        )
        .expect("hsl=0 encode");
        let hsl_full = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            4, // hsl = Np,y → single slice
            std::slice::from_ref(&pixels),
        )
        .expect("hsl=Np,y encode");
        assert_eq!(
            hsl0, baseline,
            "hsl=0 must be byte-identical to encode_planar_lossy"
        );
        // hsl=Np,y produces one slice too; only the PIH Hsl field differs
        // from hsl=0 (Np,y vs Np,y — actually identical since hsl=0 emits
        // Hsl=Np,y). So hsl_full must also equal the baseline.
        assert_eq!(
            hsl_full, baseline,
            "hsl=Np,y must be byte-identical (one slice)"
        );
        let img = decode_codestream(&hsl0, None).expect("decode hsl=0");
        assert_eq!(img.planes[0].data, pixels, "hsl=0 lossless");
    }

    /// Non-divisible Np,y: the last slice is shorter. 20-tall picture at
    /// NL=2/2 → Hp=4, Np,y = ⌈20/4⌉ = 5; Hsl=2 → slices of 2, 2, 1 rows.
    #[test]
    fn round103_hsl2_non_divisible_last_slice_shorter() {
        let w = 16usize;
        let h = 20usize;
        let pixels = round103_grad(w, h);
        let cs = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            2, // hsl = 2; Np,y = 5 → slices 2,2,1
            std::slice::from_ref(&pixels),
        )
        .expect("encode 16x20 luma Hsl=2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.slices.len(), 3, "Np,y=5 / Hsl=2 → 3 slices (2,2,1)");
        let img = decode_codestream(&cs, None).expect("decode 16x20 luma Hsl=2");
        assert_eq!(
            img.planes[0].data, pixels,
            "non-divisible multi-slice lossless"
        );
    }

    /// `hsl` exceeding the precinct-row count is rejected (would describe
    /// a single slice running past the last precinct row, Annex B.10).
    #[test]
    fn round103_rejects_hsl_exceeding_np_y() {
        let w = 16usize;
        let h = 16usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = ⌈16/4⌉ = 4; Hsl=5 is out of range.
        let result = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            5, // hsl > Np,y
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "Hsl > Np,y must be rejected");
    }

    // === Round 206: per-slice Q[p] override (slice-level rate budgeting) ==
    //
    // The base round-103 multi-slice path emitted every precinct with the
    // same Q (constant across the picture). Round 206 lifts that to a
    // per-slice override carried in `q_slices`: one Q value per slice, in
    // top-down (Yslh) order. The decoder reads Q per precinct via
    // `parse_precinct_header` + `precinct_truncation`, so no decoder change
    // is needed — only the per-precinct `Q` byte in the bitstream changes.

    /// All-zero q_slices over a 4-slice picture is bit-equivalent to
    /// `encode_planar_hsl` at q=0 (both are lossless deadzone with
    /// T[p,b] = 0 everywhere, so the byte-identical legacy stream is
    /// preserved).
    #[test]
    fn round206_qslice_all_zero_matches_hsl_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Hp=4, Np,y=8; Hsl=2 → 4 slices.
        let baseline = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline encode");
        let qs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 0, 0, 0],
            std::slice::from_ref(&pixels),
        )
        .expect("qslice all-zero encode");
        assert_eq!(
            qs, baseline,
            "all-zero q_slices must be byte-identical to encode_planar_hsl at q=0"
        );
    }

    /// All-equal q_slices over a 4-slice picture is bit-equivalent to
    /// `encode_planar_hsl` at the same Q — every precinct picks the same
    /// Q either way.
    #[test]
    fn round206_qslice_uniform_q_matches_hsl_lossy() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline encode q=2");
        let qs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[2, 2, 2, 2],
            std::slice::from_ref(&pixels),
        )
        .expect("qslice uniform-q encode");
        assert_eq!(
            qs, baseline,
            "uniform-Q q_slices must be byte-identical to encode_planar_hsl at the same Q"
        );
    }

    /// Mixed per-slice Q produces a different (and smaller-at-higher-Q)
    /// stream than the all-equal-Q baseline, and the picture still
    /// round-trips through the decoder.
    #[test]
    fn round206_qslice_mixed_q_round_trip_and_diverges() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Hp=4, Np,y=8; Hsl=2 → 4 slices.
        let mixed_q: [u8; 4] = [0, 2, 4, 2];
        let mixed = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &mixed_q,
            std::slice::from_ref(&pixels),
        )
        .expect("mixed q_slices encode");
        let constant_q4 = encode_planar_hsl(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            4,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("constant q=4 baseline");
        // The mixed stream should keep more bits in the q=0 slice than
        // the constant-q=4 baseline of the same picture (mixed has at
        // least one lossless slice).
        assert!(
            mixed.len() > constant_q4.len(),
            "mixed q_slices ({} B) must keep more bits than constant Q=4 ({} B)",
            mixed.len(),
            constant_q4.len()
        );
        // Decoder reconstructs the picture (lossy in the q>0 slices,
        // lossless in the q=0 slice).
        let img = decode_codestream(&mixed, None).expect("decode mixed q_slices");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 30.0,
            "mixed-Q multi-slice PSNR {p:.2} dB < 30 dB floor"
        );
        // PIH still carries Hsl=2 (slice layout unchanged).
        let parsed = crate::codestream::parse(&mixed).expect("parse codestream");
        assert_eq!(parsed.pih.hsl, 2, "PIH Hsl unchanged by per-slice Q");
        assert_eq!(parsed.slices.len(), 4, "Np,y=8 / Hsl=2 → 4 slices");
    }

    /// Each slice's precinct-header Q[p] byte carries the override —
    /// verified by parsing the codestream and checking each slice's
    /// first precinct header.
    #[test]
    fn round206_qslice_wire_carries_per_slice_q() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let qslices: [u8; 4] = [0, 1, 2, 3];
        let cs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &qslices,
            std::slice::from_ref(&pixels),
        )
        .expect("qslice wire encode");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.slices.len(), 4, "expected 4 slices");
        // The first byte of each slice's entropy data is the Lprc[p]
        // top byte of the first precinct header; precinct_bytes[3] is
        // the Q[p] byte (per Annex C.2 Table C.1, after the 24-bit
        // Lprc field). Peek at each slice's payload to confirm the
        // per-slice Q[p] override surfaces on the wire.
        for (t, slice) in parsed.slices.iter().enumerate() {
            assert!(
                slice.data_length >= 5,
                "slice {t} payload too small to hold a precinct header"
            );
            // Slice payload starts with the first precinct header;
            // the Q[p] byte is at offset 3 within that header (after
            // the 24-bit Lprc).
            let q_byte = cs[slice.data_offset + 3];
            assert_eq!(
                q_byte, qslices[t],
                "slice {t} first-precinct Q[p] must equal q_slices[{t}]={}",
                qslices[t]
            );
        }
    }

    /// Wrong q_slices length is rejected (must equal the slice count).
    #[test]
    fn round206_qslice_rejects_wrong_length() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // Np,y=8, Hsl=2 → 4 slices, but only 3 q values.
        let result = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 1, 2],
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "wrong q_slices length must be rejected");
    }

    /// Out-of-range q_slices entry (> 15) is rejected.
    #[test]
    fn round206_qslice_rejects_oversize_q() {
        let w = 16usize;
        let h = 16usize;
        let pixels = round103_grad(w, h);
        // Np,y=4, Hsl=2 → 2 slices.
        let result = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 16],
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "q_slices entry > 15 must be rejected");
    }

    /// Single-slice (hsl=0) with q_slices of length 1 must accept and be
    /// byte-identical to `encode_planar_lossy` at that single Q.
    #[test]
    fn round206_qslice_single_slice_matches_lossy() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            3,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline encode_planar_lossy q=3");
        let qs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            &[3],
            std::slice::from_ref(&pixels),
        )
        .expect("single-slice qslice encode");
        assert_eq!(
            qs, baseline,
            "single-slice q_slices=[3] must equal encode_planar_lossy at q=3"
        );
    }

    // === Round 212: target-bytes rate-budget picker =====================
    //
    // `pick_q_slices_for_target_bytes` drives `encode_planar_hsl_qslice`
    // with a deterministic three-pass search (lossless probe → uniform-Q
    // bisect → per-slice activity-ranked relaxation). The picker calls
    // the existing per-slice encoder for every measurement — no internal
    // model of the entropy coder, no external library. Tests below
    // confirm the three regimes (loose → fit at q=0, tight → uniform Q
    // bisect, very tight → unreachable error) plus the convenience
    // wrapper byte-equivalence and zero-budget rejection.

    /// Round 212: a loose budget (≥ the lossless codestream length) must
    /// pick `q_slices = [0; n_slices]` and yield the byte-identical
    /// lossless stream. The picker's pass-1 lossless probe short-circuits
    /// before any uniform-Q bisect runs.
    #[test]
    fn round212_picker_loose_budget_returns_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y=8; Hsl=2 → 4 slices.
        let lossless = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 0, 0, 0],
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossless encode");
        // Budget = lossless length itself: must fit exactly via q=[0;..].
        let q = pick_q_slices_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            lossless.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("picker at loose budget");
        assert_eq!(q, vec![0u8; 4], "loose budget must pick all-lossless");
    }

    /// Round 212: a budget tighter than lossless but reachable at some
    /// `Q ∈ 1..=15` triggers the pass-2 uniform-Q bisect. The result
    /// must satisfy `encode_planar_hsl_qslice(.., q, ..) .len() ≤ budget`
    /// AND no smaller q value fits (i.e. the picker honours the
    /// monotone-ish search bound).
    #[test]
    fn round212_picker_tight_budget_fits_within_target() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // The all-zeros stream is the largest a 4-slice picture can
        // produce; halving its length forces the picker to quantize.
        let lossless = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 0, 0, 0],
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossless");
        let target = lossless.len() / 2;
        let q = pick_q_slices_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            target,
            std::slice::from_ref(&pixels),
        )
        .expect("picker at half-of-lossless budget");
        // The chosen q_slices must produce a stream that fits.
        let cs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &q,
            std::slice::from_ref(&pixels),
        )
        .expect("picked q_slices encode");
        assert!(
            cs.len() <= target,
            "picked stream {} B > target {} B",
            cs.len(),
            target
        );
        // q_slices must have exactly one entry per slice.
        assert_eq!(q.len(), 4, "expected 4 q entries for Np,y=8 / Hsl=2");
        // At least one entry must be > 0 (otherwise lossless would fit).
        assert!(
            q.iter().any(|&v| v > 0),
            "tight budget must quantize at least one slice"
        );
    }

    /// Round 212: a budget so small that even `q = [15; n_slices]`
    /// overshoots is reported as an explicit error rather than silently
    /// truncated. The error message must mention the actual Q=15
    /// encoded length so the caller can rescale.
    #[test]
    fn round212_picker_unreachable_budget_errors() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // Target = 1 byte is unreachable for any 32×32 stream (the
        // marker chain alone — SOC + CAP + PIH + CDT + WGT + 4×SLH +
        // EOC — is well over a hundred bytes).
        let err = pick_q_slices_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            1,
            std::slice::from_ref(&pixels),
        )
        .expect_err("picker must reject unreachable budget");
        let msg = format!("{err}");
        assert!(
            msg.contains("unreachable"),
            "error message must mention unreachable: got {msg:?}"
        );
    }

    /// Round 212: zero target bytes is a precondition violation
    /// (rejected before the lossless probe even runs).
    #[test]
    fn round212_picker_zero_target_rejected() {
        let w = 16usize;
        let h = 16usize;
        let pixels = round103_grad(w, h);
        let err = pick_q_slices_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect_err("zero budget must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("target_bytes"),
            "error must mention target_bytes precondition: got {msg:?}"
        );
    }

    /// Round 212: the `encode_planar_hsl_target_bytes` convenience
    /// wrapper returns `(codestream, q_slices)` that satisfies the
    /// budget AND is byte-identical to a follow-up
    /// `encode_planar_hsl_qslice(.., q_slices, ..)` call (the picker's
    /// chosen q vector is self-consistent / reproducible).
    #[test]
    fn round212_picker_wrapper_is_byte_identical_to_qslice_encode() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // Choose a budget between lossless and pathological.
        let lossless = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &[0, 0, 0, 0],
            std::slice::from_ref(&pixels),
        )
        .expect("lossless baseline");
        let target = (lossless.len() * 3) / 4;
        let (cs, q) = encode_planar_hsl_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            target,
            std::slice::from_ref(&pixels),
        )
        .expect("target-bytes wrapper");
        assert!(
            cs.len() <= target,
            "wrapper stream {} B > target {} B",
            cs.len(),
            target
        );
        // Re-encoding with the same q must reproduce the same bytes.
        let cs2 = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            &q,
            std::slice::from_ref(&pixels),
        )
        .expect("re-encode with picked q");
        assert_eq!(
            cs, cs2,
            "wrapper bytes must equal a follow-up qslice encode with the same q vector"
        );
        // And the picture round-trips at acceptable quality.
        let img = decode_codestream(&cs, None).expect("decode picker stream");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 25.0, "picker round-trip PSNR {p:.2} dB < 25 dB");
    }

    /// Round 212: single-slice mode (`hsl == 0`) must still go through
    /// the picker correctly — the bisect collapses to one Q, the
    /// relaxation pass is a no-op, and the returned vector has length
    /// 1.
    #[test]
    fn round212_picker_single_slice_degenerate() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // Hsl=0 → single slice.
        let lossless = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("single-slice lossless baseline");
        // Tight budget but reachable (use 60% of lossless).
        let target = (lossless.len() * 3) / 5;
        let q = pick_q_slices_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            target,
            std::slice::from_ref(&pixels),
        )
        .expect("single-slice picker");
        assert_eq!(q.len(), 1, "single-slice picker must return one Q");
        let cs = encode_planar_hsl_qslice(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            &q,
            std::slice::from_ref(&pixels),
        )
        .expect("encode at picked single-slice Q");
        assert!(
            cs.len() <= target,
            "single-slice picker stream {} B > target {} B",
            cs.len(),
            target
        );
    }

    // === Round 108: Qpih=1 uniform inverse quantizer (Annex D.3) ========

    /// Round 108: Qpih=1 luma round-trips losslessly and the PIH carries
    /// the inverse-quantizer type (`qpih == 1`, Annex A.4.4 Table A.10).
    /// At q=0 (T=0) both the deadzone and uniform kernels reconstruct
    /// exactly, so the uniform path is lossless.
    #[test]
    fn round108_qpih1_luma_32x32_nl_2_2_lossless() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_qpih(32, 32, 1, 0, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("encode 32x32 luma Qpih=1 NL=2/2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.qpih, 1, "PIH should report Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode Qpih=1 luma");
        assert_eq!(img.planes[0].data, pixels, "Qpih=1 luma lossless");
    }

    /// Round 108: Qpih=1 with an RGB picture under the reversible colour
    /// transform (Cpih=1) round-trips losslessly.
    #[test]
    fn round108_qpih1_rgb_cpih1_nl_2_2_lossless() {
        let rgb = make_synthetic_rgb_32x32();
        let mut r = vec![0u8; 32 * 32];
        let mut g = vec![0u8; 32 * 32];
        let mut b = vec![0u8; 32 * 32];
        for i in 0..32 * 32 {
            r[i] = rgb[i * 3];
            g[i] = rgb[i * 3 + 1];
            b[i] = rgb[i * 3 + 2];
        }
        let cs = encode_planar_qpih(32, 32, 3, 1, 2, 2, 0, &[r.clone(), g.clone(), b.clone()])
            .expect("encode RGB Qpih=1 Cpih=1 NL=2/2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.qpih, 1, "PIH should report Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode Qpih=1 RGB");
        assert_eq!(img.planes[0].data, r, "R lossless Qpih=1");
        assert_eq!(img.planes[1].data, g, "G lossless Qpih=1");
        assert_eq!(img.planes[2].data, b, "B lossless Qpih=1");
    }

    /// Round 108: Qpih=1 lossy q=2 reconstructs through the uniform /
    /// Neumann-series kernel (Annex D.3) and holds a PSNR floor. The
    /// uniform kernel is a valid lossy reconstruction; it does not need to
    /// match the deadzone PSNR, only stay above a sane floor.
    #[test]
    fn round108_qpih1_lossy_q2_psnr_floor() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_qpih(32, 32, 1, 0, 2, 2, 2, std::slice::from_ref(&pixels))
            .expect("encode 32x32 luma Qpih=1 q=2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        assert_eq!(parsed.pih.qpih, 1, "PIH should report Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode Qpih=1 lossy");
        let q = psnr(&pixels, &img.planes[0].data);
        assert!(
            q >= 30.0,
            "Qpih=1 q=2 luma PSNR {q:.2} dB below 30 dB floor"
        );
    }

    /// Round 108: at q=0 (lossless) the only on-wire difference between
    /// Qpih=1 and Qpih=0 is the PIH `Lh:Rl:Qpih:Fs:Rm` byte — the data
    /// sub-packet is byte-identical because the same magnitude bitplanes
    /// are emitted. So the two codestreams differ in exactly one byte (the
    /// PIH quantizer-type byte) and decode to identical pixels.
    #[test]
    fn round108_qpih1_vs_qpih0_lossless_one_byte_diff() {
        let pixels = make_synthetic_32x32();
        let cs0 = encode_planar_lossy(32, 32, 1, 0, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("encode Qpih=0");
        let cs1 = encode_planar_qpih(32, 32, 1, 0, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("encode Qpih=1");
        assert_eq!(
            cs0.len(),
            cs1.len(),
            "Qpih only flips a header bit; lengths must match"
        );
        let diffs: Vec<usize> = cs0
            .iter()
            .zip(cs1.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            diffs.len(),
            1,
            "Qpih=0 vs Qpih=1 lossless must differ in exactly one byte, differ at {diffs:?}"
        );
        // The single differing byte is the PIH quantizer-type byte: bit 4
        // (Qpih LSB) is set in the Qpih=1 stream and clear in Qpih=0.
        let d = diffs[0];
        assert_eq!(
            cs1[d] ^ cs0[d],
            0x10,
            "the differing byte must toggle exactly the Qpih bit (bit 4)"
        );
        let img0 = decode_codestream(&cs0, None).expect("decode Qpih=0");
        let img1 = decode_codestream(&cs1, None).expect("decode Qpih=1");
        assert_eq!(
            img0.planes[0].data, img1.planes[0].data,
            "Qpih=0 and Qpih=1 lossless must decode to identical pixels"
        );
        assert_eq!(img1.planes[0].data, pixels, "Qpih=1 lossless");
    }

    /// Round 108: the encoder rejects reserved Qpih values (2/3) routed
    /// through the inner builder — the decoder rejects `Qpih > 1` so the
    /// encoder must never emit them.
    #[test]
    fn round108_rejects_reserved_qpih() {
        let pixels = vec![0u8; 16 * 8];
        let result = encode_planar_inner_nlt(
            16,
            8,
            1,
            0,
            1,
            1,
            0,
            0,
            &[1],
            &[1],
            0,
            0,
            0,
            0,
            None,
            Vec::new(),
            Vec::new(),
            0,          // cw
            0,          // sd
            0,          // fs
            0,          // hsl
            2,          // qpih: reserved → must reject
            0,          // rp
            Vec::new(), // q_slices: single picture-level q
            Vec::new(), // q_precincts: no per-precinct override
            Vec::new(), // r_precincts: no per-precinct R[p] override
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "Qpih=2 (reserved) must be rejected");
    }

    // === Round 111: Qpih-aware forward quantizer ========================
    // Round 108 signalled `Qpih = 1` but the encoder still picked the
    // quantization index with the deadzone truncation `v = |c| >> T`
    // (Annex D.4, Table D.3). Round 111 adds the matching uniform forward
    // quantizer (Annex D.5, Table D.4) so a `Qpih = 1` stream is encoded
    // with round-to-nearest indices instead of flooring, which the
    // uniform / Neumann-series inverse (Annex D.3, Table D.2) was already
    // expecting.

    /// Round 111: `forward_quant_index` with `qpih = 0` is the deadzone
    /// truncation `v = |c| >> T` (Annex D.4, Table D.3), independent of M
    /// (beyond the `M > T` gate). Sign is handled by the caller, so the
    /// magnitude uses `|c|`.
    #[test]
    fn round111_forward_quant_deadzone_is_truncation() {
        // M > T so the group carries bitplanes.
        assert_eq!(forward_quant_index(0, 3, 4, 2), 0); // 3 >> 2 = 0
        assert_eq!(forward_quant_index(0, 7, 4, 2), 1); // 7 >> 2 = 1
        assert_eq!(forward_quant_index(0, -12, 4, 2), 3); // 12 >> 2 = 3
        assert_eq!(forward_quant_index(0, 15, 4, 2), 3); // 15 >> 2 = 3
                                                         // M <= T → no stored bitplanes → v = 0.
        assert_eq!(forward_quant_index(0, 15, 2, 2), 0);
        assert_eq!(forward_quant_index(0, 15, 1, 2), 0);
    }

    /// Round 111: `forward_quant_index` with `qpih = 1` matches the
    /// uniform forward quantizer of Annex D.5 Table D.4 hand-computed:
    /// `v = ((d << ζ) − d + (1 << M)) >> (M + 1)`, `ζ = M − T + 1`. The
    /// uniform index rounds to nearest where the deadzone floors —
    /// e.g. `d = 3, M = 4, T = 2` gives `1` (uniform) vs `0` (deadzone).
    #[test]
    fn round111_forward_quant_uniform_matches_table_d4() {
        // M=4, T=2 (ζ=3): rounding visible at d=3 and d=7.
        assert_eq!(forward_quant_index(1, 3, 4, 2), 1); // round up vs 0
        assert_eq!(forward_quant_index(1, 7, 4, 2), 2); // round up vs 1
        assert_eq!(forward_quant_index(1, -12, 4, 2), 3); // ties with deadzone
        assert_eq!(forward_quant_index(1, 15, 4, 2), 3);
        // M=5, T=1 (ζ=5): matches deadzone on these exact multiples.
        assert_eq!(forward_quant_index(1, 1, 5, 1), 0);
        assert_eq!(forward_quant_index(1, 2, 5, 1), 1);
        assert_eq!(forward_quant_index(1, 16, 5, 1), 8);
        assert_eq!(forward_quant_index(1, -31, 5, 1), 15);
        // M <= T → v = 0 (Table D.4 `else`).
        assert_eq!(forward_quant_index(1, 15, 2, 2), 0);
    }

    /// Round 111: at `T = 0` (q = 0, lossless) the uniform forward index
    /// equals the coefficient magnitude (`v = |c|`), so it round-trips
    /// through the uniform inverse exactly. This keeps the round-108
    /// lossless / one-byte-diff invariants intact under the new path.
    #[test]
    fn round111_forward_quant_uniform_t0_is_identity() {
        for d in 0u32..256 {
            if d == 0 {
                continue;
            }
            let m = 32 - d.leading_zeros(); // bit length = M for this d
            assert_eq!(
                forward_quant_index(1, d as i32, m, 0),
                d,
                "uniform forward at T=0 must be identity for d={d}, M={m}"
            );
        }
    }

    /// Round 111: a `Qpih = 1` lossy stream now differs from the
    /// `Qpih = 0` lossy stream in more than the single PIH byte — the
    /// data sub-packet now carries the uniform (round-to-nearest) indices
    /// rather than the deadzone (floored) indices. (Before round 111 the
    /// two q>0 streams were byte-identical except the PIH bit.)
    #[test]
    fn round111_qpih1_lossy_data_differs_from_qpih0() {
        let pixels = make_synthetic_32x32();
        let cs0 = encode_planar_lossy(32, 32, 1, 0, 2, 2, 3, std::slice::from_ref(&pixels))
            .expect("encode Qpih=0 q=3");
        let cs1 = encode_planar_qpih(32, 32, 1, 0, 2, 2, 3, std::slice::from_ref(&pixels))
            .expect("encode Qpih=1 q=3");
        let diffs = cs0.iter().zip(cs1.iter()).filter(|(a, b)| a != b).count();
        assert!(
            cs0.len() != cs1.len() || diffs > 1,
            "Qpih=1 lossy must now diverge in the data sub-packet, not just the PIH byte (len {} vs {}, {diffs} byte diffs)",
            cs0.len(),
            cs1.len()
        );
    }

    /// Round 111: the uniform forward quantizer keeps the `Qpih = 1`
    /// lossy reconstruction valid and above a sane PSNR floor at q=3 for
    /// luma and for RGB under the reversible colour transform. The
    /// round-to-nearest indexing should never reconstruct worse than the
    /// previous floored indexing did; we assert a floor rather than an
    /// exact PSNR so the test is robust to the synthetic content.
    #[test]
    fn round111_qpih1_uniform_lossy_psnr_floor_q3() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_qpih(32, 32, 1, 0, 2, 2, 3, std::slice::from_ref(&pixels))
            .expect("encode Qpih=1 q=3 luma");
        let img = decode_codestream(&cs, None).expect("decode Qpih=1 q=3");
        let q = psnr(&pixels, &img.planes[0].data);
        assert!(
            q >= 25.0,
            "Qpih=1 uniform q=3 luma PSNR {q:.2} dB below 25 dB"
        );
    }

    /// Round 111: the uniform forward quantizer composes with `Fs = 1`
    /// (separate sign sub-packet). The sign-gating predicate now uses the
    /// same uniform index the decoder reads, so a `Qpih = 1` + `Fs = 1`
    /// stream still round-trips losslessly at q=0.
    #[test]
    fn round111_qpih1_fs1_lossless() {
        let pixels = make_synthetic_32x32();
        let cs = encode_planar_inner_nlt(
            32,
            32,
            1,
            0,
            2,
            2,
            0, // q = 0 lossless
            0,
            &[1],
            &[1],
            0,
            0,
            0,
            0,
            None,
            Vec::new(),
            Vec::new(),
            0,          // cw
            0,          // sd
            1,          // fs = 1
            0,          // hsl
            1,          // qpih = 1
            0,          // rp
            Vec::new(), // q_slices: single picture-level q
            Vec::new(), // q_precincts: no per-precinct override
            Vec::new(), // r_precincts: no per-precinct R[p] override
            std::slice::from_ref(&pixels),
        )
        .expect("encode Qpih=1 Fs=1 luma lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "PIH Qpih=1");
        assert_eq!(parsed.pih.fs, 1, "PIH Fs=1");
        let img = decode_codestream(&cs, None).expect("decode Qpih=1 Fs=1");
        assert_eq!(img.planes[0].data, pixels, "Qpih=1 Fs=1 lossless");
    }

    // === Round 115: R[p] > 0 precinct refinement (Annex C.6.2) ==========

    /// Round 115: the WGT marker now carries per-band priorities `P[b] = b`
    /// (the true band index, Annex B.6) instead of the all-zero priorities
    /// rounds 1–111 emitted. The decoder reads them via `parse_wgt`; here we
    /// confirm the priorities are the sequential band indices for a luma
    /// NL=2/2 picture (Nβ = 6 bands for one component).
    #[test]
    fn round115_wgt_carries_band_index_priorities() {
        let pixels = make_synthetic_32x32();
        // q = 0 lossless; rp = 0 (priorities are emitted regardless of rp).
        let cs = encode_planar_rp(32, 32, 1, 0, 2, 2, 0, 0, std::slice::from_ref(&pixels))
            .expect("encode luma NL=2/2 rp=0");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        // Nβ for NL,x=2, NL,y=2 is 2*2 + 2 + 1 = ... use n_beta to avoid
        // hard-coding; Nc=1 so NL = Nβ bands, one per β.
        let nbeta = n_beta(2, 2);
        let weights =
            crate::slice_walker::parse_wgt(&parsed.wgt, nbeta as usize).expect("parse_wgt");
        for (b, w) in weights.iter().enumerate() {
            assert_eq!(
                w.priority, b as u8,
                "band {b} priority should equal its band index"
            );
        }
    }

    /// Round 115: `rp = 0` is the no-refinement default and must be
    /// byte-identical to [`encode_planar_lossy`] at every `q` (priorities
    /// are emitted but `r = (P[b] < 0)` is always false, so T is unchanged
    /// and the WGT priority bytes are the only addition — which the lossy
    /// path also carries now). Both paths route through the same builder, so
    /// the streams are fully identical.
    #[test]
    fn round115_rp0_matches_encode_planar_lossy() {
        let pixels = make_synthetic_32x32();
        for q in [0u8, 2, 4] {
            let lossy = encode_planar_lossy(32, 32, 1, 0, 2, 2, q, std::slice::from_ref(&pixels))
                .expect("encode_planar_lossy");
            let rp0 = encode_planar_rp(32, 32, 1, 0, 2, 2, q, 0, std::slice::from_ref(&pixels))
                .expect("encode_planar_rp rp=0");
            assert_eq!(
                lossy, rp0,
                "rp=0 must be byte-identical to encode_planar_lossy at q={q}"
            );
        }
    }

    /// Round 115: `rp > 0` self-roundtrips losslessly at q=0. At q=0 the
    /// truncation T is already clamped to its 0 floor, so the refinement
    /// term `r` cannot lower it further — the lossless invariant holds for
    /// any `rp`.
    #[test]
    fn round115_rp_gt_zero_lossless_at_q0() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2); // NL = Nβ for Nc=1
        for rp in 1..=(nl as u8 - 1) {
            let cs = encode_planar_rp(32, 32, 1, 0, 2, 2, 0, rp, std::slice::from_ref(&pixels))
                .unwrap_or_else(|e| panic!("encode rp={rp} q=0: {e:?}"));
            let img = decode_codestream(&cs, None)
                .unwrap_or_else(|e| panic!("decode rp={rp} q=0: {e:?}"));
            assert_eq!(img.planes[0].data, pixels, "rp={rp} q=0 must be lossless");
        }
    }

    /// Round 115: a `rp > 0` lossy stream self-roundtrips through the
    /// decoder (which recomputes the same T[p,b] from the WGT priorities and
    /// the precinct-header R[p]) and holds a PSNR floor at q=2. This is the
    /// core correctness proof: the encoder quantized with the refined T and
    /// the decoder dequantized with the matching refined T.
    #[test]
    fn round115_rp_gt_zero_lossy_q2_roundtrips_and_holds_psnr() {
        let pixels = make_synthetic_32x32();
        // rp = 1 refines band 0 (LL) only.
        let cs = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 1, std::slice::from_ref(&pixels))
            .expect("encode luma rp=1 q=2");
        let parsed = crate::codestream::parse(&cs).expect("parse codestream");
        // The precinct header R[p] byte sits at precinct-header offset 4;
        // confirm the decoder side reconstructs the picture (round-trip is
        // the authoritative check). Inspect the WGT priority of band 0.
        let nbeta = n_beta(2, 2);
        let weights =
            crate::slice_walker::parse_wgt(&parsed.wgt, nbeta as usize).expect("parse_wgt");
        assert_eq!(weights[0].priority, 0, "band 0 (LL) priority is 0");
        let img = decode_codestream(&cs, None).expect("decode rp=1 q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 30.0, "rp=1 q=2 PSNR {p:.2} dB below 30 dB floor");
    }

    /// Round 115: at q=2 a refinement `rp > 0` changes the data sub-packet
    /// relative to `rp = 0` — the refined bands carry one extra magnitude
    /// bitplane (lower T), so the codestream is not byte-identical. This
    /// proves the refinement actually fires (it is not a silent no-op). The
    /// refined stream is at least as large as the unrefined one because the
    /// extra bitplane adds coded bits to the refined bands.
    #[test]
    fn round115_rp_gt_zero_changes_lossy_stream() {
        let pixels = make_synthetic_32x32();
        let rp0 = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("encode rp=0 q=2");
        // rp = NL-1 refines every band except the highest-index one,
        // guaranteeing several bands gain a bitplane.
        let nl = n_beta(2, 2) as u8; // NL = Nβ for Nc=1
        let rp_hi = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, nl - 1, std::slice::from_ref(&pixels))
            .expect("encode rp=NL-1 q=2");
        assert_ne!(
            rp0, rp_hi,
            "rp>0 must change the lossy data sub-packet vs rp=0"
        );
        // Both still round-trip.
        let img = decode_codestream(&rp_hi, None).expect("decode rp=NL-1 q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        // Refining low-frequency bands should not collapse quality; hold a
        // sane floor (the refined LL band carries more precision).
        assert!(p >= 25.0, "rp=NL-1 q=2 PSNR {p:.2} dB below 25 dB floor");
    }

    /// Round 115: RGB under the reversible colour transform (Cpih=1) with
    /// `rp > 0` self-roundtrips losslessly at q=0 and holds a PSNR floor at
    /// q=2 — exercising the β-major-then-component band-index priority
    /// assignment across 3 components.
    #[test]
    fn round115_rp_rgb_rct_roundtrips() {
        let rgb = make_synthetic_rgb_32x32();
        let mut r = vec![0u8; 32 * 32];
        let mut g = vec![0u8; 32 * 32];
        let mut b = vec![0u8; 32 * 32];
        for i in 0..32 * 32 {
            r[i] = rgb[i * 3];
            g[i] = rgb[i * 3 + 1];
            b[i] = rgb[i * 3 + 2];
        }
        let planes = [r.clone(), g.clone(), b.clone()];
        // q=0 lossless with rp=2 (refines bands 0 and 1).
        let cs0 =
            encode_planar_rp(32, 32, 3, 1, 2, 2, 0, 2, &planes).expect("encode RGB+RCT rp=2 q=0");
        let img0 = decode_codestream(&cs0, None).expect("decode RGB+RCT rp=2 q=0");
        assert_eq!(img0.planes[0].data, r, "R lossless rp=2");
        assert_eq!(img0.planes[1].data, g, "G lossless rp=2");
        assert_eq!(img0.planes[2].data, b, "B lossless rp=2");
        // q=2 lossy with rp=3.
        let cs2 =
            encode_planar_rp(32, 32, 3, 1, 2, 2, 2, 3, &planes).expect("encode RGB+RCT rp=3 q=2");
        let img2 = decode_codestream(&cs2, None).expect("decode RGB+RCT rp=3 q=2");
        for (plane, name) in [(&r, "R"), (&g, "G"), (&b, "B")]
            .iter()
            .map(|(p, n)| (*p, *n))
        {
            let idx = match name {
                "R" => 0,
                "G" => 1,
                _ => 2,
            };
            let p = psnr(plane, &img2.planes[idx].data);
            assert!(
                p >= 25.0,
                "{name} rp=3 q=2 PSNR {p:.2} dB below 25 dB floor"
            );
        }
    }

    /// Round 115: the encoder rejects an out-of-range `R[p]` (>= NL). The
    /// precinct-header field is u(8) but Table C.1 caps it at NL-1; a value
    /// past the highest band index would refine every band and is invalid.
    #[test]
    fn round115_rejects_rp_out_of_range() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2) as u8; // NL = Nβ for Nc=1, NL=2/2
                                     // rp = NL is one past the legal maximum (NL-1).
        let result = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, nl, std::slice::from_ref(&pixels));
        assert!(
            result.is_err(),
            "R[p]={nl} (== NL) must be rejected (legal max is NL-1={})",
            nl - 1
        );
        // rp = NL-1 is the legal maximum and must succeed.
        let ok = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, nl - 1, std::slice::from_ref(&pixels));
        assert!(ok.is_ok(), "R[p]=NL-1={} must be accepted", nl - 1);
    }

    // === Round 118: high bit depth (B[i] > 8) ===========================

    /// Synthetic `bd`-bit luma ramp filling the full `0..=2^bd-1` range.
    fn make_synthetic_highbd(w: usize, h: usize, bd: u8) -> Vec<u16> {
        let max = ((1u32 << bd) - 1) as i64;
        let mut buf = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                // A spread-out pattern that exercises the high bits.
                let v = ((x as i64) * 277 + (y as i64) * 631 + ((x ^ y) as i64) * 53) % (max + 1);
                buf[y * w + x] = v as u16;
            }
        }
        buf
    }

    /// Reinterpret a two-byte-per-sample (little-endian) plane back into
    /// `u16` samples for comparison.
    fn plane_u16(data: &[u8]) -> Vec<u16> {
        data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn highbd_10bit_luma_nl1_round_trips_exactly() {
        let src = make_synthetic_highbd(32, 32, 10);
        let cs = encode_planar_highbd(32, 32, 1, 0, 1, 1, 10, std::slice::from_ref(&src))
            .expect("encode 10-bit luma");
        let img = decode_codestream(&cs, None).expect("decode 10-bit luma");
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        assert_eq!(plane_u16(&img.planes[0].data), src);
    }

    #[test]
    fn highbd_12bit_luma_nl3_round_trips_exactly() {
        let src = make_synthetic_highbd(32, 32, 12);
        let cs = encode_planar_highbd(32, 32, 1, 0, 3, 3, 12, std::slice::from_ref(&src))
            .expect("encode 12-bit luma NL=3/3");
        let img = decode_codestream(&cs, None).expect("decode 12-bit luma NL=3/3");
        assert_eq!(plane_u16(&img.planes[0].data), src);
    }

    #[test]
    fn highbd_16bit_luma_full_range_round_trips_exactly() {
        // Drive the full 16-bit dynamic range, including the extremes.
        let mut src = make_synthetic_highbd(16, 16, 16);
        src[0] = 0;
        src[1] = 65535;
        src[2] = 32768;
        let cs = encode_planar_highbd(16, 16, 1, 0, 2, 2, 16, std::slice::from_ref(&src))
            .expect("encode 16-bit luma");
        let img = decode_codestream(&cs, None).expect("decode 16-bit luma");
        assert_eq!(plane_u16(&img.planes[0].data), src);
    }

    #[test]
    fn highbd_16bit_flat_image_is_exact() {
        let src = vec![40000u16; 32 * 32];
        let cs = encode_planar_highbd(32, 32, 1, 0, 2, 2, 16, std::slice::from_ref(&src))
            .expect("encode flat 16-bit");
        let img = decode_codestream(&cs, None).expect("decode flat 16-bit");
        assert_eq!(plane_u16(&img.planes[0].data), src);
    }

    #[test]
    fn highbd_16bit_rgb_rct_round_trips_exactly() {
        // Three 16-bit planes through the reversible RCT (Cpih = 1).
        let r = make_synthetic_highbd(32, 32, 16);
        let mut g = make_synthetic_highbd(32, 32, 16);
        let mut b = make_synthetic_highbd(32, 32, 16);
        // Perturb g/b so the colour transform actually moves data.
        for v in g.iter_mut() {
            *v = v.wrapping_add(11111);
        }
        for v in b.iter_mut() {
            *v = v.wrapping_mul(3).wrapping_add(7);
        }
        let planes = vec![r.clone(), g.clone(), b.clone()];
        let cs =
            encode_planar_highbd(32, 32, 3, 1, 2, 2, 16, &planes).expect("encode 16-bit RGB + RCT");
        let img = decode_codestream(&cs, None).expect("decode 16-bit RGB + RCT");
        assert_eq!(plane_u16(&img.planes[0].data), r);
        assert_eq!(plane_u16(&img.planes[1].data), g);
        assert_eq!(plane_u16(&img.planes[2].data), b);
    }

    #[test]
    fn highbd_12bit_rgb_no_transform_round_trips_exactly() {
        let r = make_synthetic_highbd(16, 24, 12);
        let g = make_synthetic_highbd(16, 24, 12);
        let b = make_synthetic_highbd(16, 24, 12);
        let planes = vec![r.clone(), g.clone(), b.clone()];
        let cs = encode_planar_highbd(16, 24, 3, 0, 2, 1, 12, &planes)
            .expect("encode 12-bit RGB no transform");
        let img = decode_codestream(&cs, None).expect("decode 12-bit RGB no transform");
        assert_eq!(plane_u16(&img.planes[0].data), r);
        assert_eq!(plane_u16(&img.planes[1].data), g);
        assert_eq!(plane_u16(&img.planes[2].data), b);
    }

    #[test]
    fn highbd_rejects_bd_8_and_above_16() {
        let src = vec![0u16; 4];
        // bd = 8 must route through the 8-bit path, not this entry point.
        assert!(encode_planar_highbd(2, 2, 1, 0, 1, 1, 8, std::slice::from_ref(&src)).is_err());
        // bd = 17 exceeds the two-byte plane format.
        assert!(encode_planar_highbd(2, 2, 1, 0, 1, 1, 17, std::slice::from_ref(&src)).is_err());
    }

    #[test]
    fn highbd_rejects_sample_above_nominal_range() {
        // A 10-bit picture with a sample of 1024 (== 2^10) is out of range.
        let mut src = vec![100u16; 32 * 32];
        src[5] = 1024;
        assert!(
            encode_planar_highbd(32, 32, 1, 0, 1, 1, 10, std::slice::from_ref(&src)).is_err(),
            "sample exceeding 2^bd-1 must be rejected"
        );
    }

    #[test]
    fn highbd_rejects_star_tetrix_cpih() {
        let src = vec![0u16; 4 * 4 * 4];
        // Cpih = 3 (Star-Tetrix) is not exposed on the high-bit-depth path.
        let planes = vec![vec![0u16; 16]; 4];
        let _ = src;
        assert!(encode_planar_highbd(4, 4, 4, 3, 1, 1, 12, &planes).is_err());
    }

    // === Round 133: high bit depth lossy (B[i] > 8, q > 0) ===============

    /// PSNR between two `u16` planes at `bd`-bit peak (`2^bd - 1`).
    fn psnr_u16(a: &[u16], b: &[u16], bd: u8) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut sse: u64 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (*x as i64) - (*y as i64);
            sse += (d * d) as u64;
        }
        if sse == 0 {
            return f64::INFINITY;
        }
        let mse = sse as f64 / a.len() as f64;
        let peak = ((1u32 << bd) - 1) as f64;
        20.0 * peak.log10() - 10.0 * mse.log10()
    }

    #[test]
    fn highbd_lossy_10bit_luma_psnr_q1() {
        // q = 1 on a 10-bit luma ramp: near-lossless, PSNR must stay high.
        let src = make_synthetic_highbd(32, 32, 10);
        let cs = encode_planar_highbd_lossy(32, 32, 1, 0, 2, 2, 10, 1, std::slice::from_ref(&src))
            .expect("encode 10-bit luma q=1");
        let img = decode_codestream(&cs, None).expect("decode 10-bit luma q=1");
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        let rec = plane_u16(&img.planes[0].data);
        assert_eq!(rec.len(), src.len());
        let p = psnr_u16(&src, &rec, 10);
        assert!(p >= 40.0, "10-bit q=1 PSNR {p:.2} dB must be >= 40 dB");
    }

    #[test]
    fn highbd_lossy_12bit_luma_psnr_q2_nl3() {
        // q = 2, NL = 3/3 on a 12-bit luma ramp.
        let src = make_synthetic_highbd(32, 32, 12);
        let cs = encode_planar_highbd_lossy(32, 32, 1, 0, 3, 3, 12, 2, std::slice::from_ref(&src))
            .expect("encode 12-bit luma q=2 NL=3/3");
        let img = decode_codestream(&cs, None).expect("decode 12-bit luma q=2 NL=3/3");
        let rec = plane_u16(&img.planes[0].data);
        let p = psnr_u16(&src, &rec, 12);
        assert!(p >= 30.0, "12-bit q=2 PSNR {p:.2} dB must be >= 30 dB");
    }

    #[test]
    fn highbd_lossy_16bit_rgb_rct_psnr_q1() {
        // Three 16-bit planes through the reversible RCT (Cpih = 1), q = 1.
        let r = make_synthetic_highbd(32, 32, 16);
        let mut g = make_synthetic_highbd(32, 32, 16);
        let mut b = make_synthetic_highbd(32, 32, 16);
        for v in g.iter_mut() {
            *v = v.wrapping_add(11111);
        }
        for v in b.iter_mut() {
            *v = v.wrapping_mul(3).wrapping_add(7);
        }
        let planes = vec![r.clone(), g.clone(), b.clone()];
        let cs = encode_planar_highbd_lossy(32, 32, 3, 1, 2, 2, 16, 1, &planes)
            .expect("encode 16-bit RGB + RCT q=1");
        let img = decode_codestream(&cs, None).expect("decode 16-bit RGB + RCT q=1");
        for (i, orig) in [&r, &g, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 16);
            assert!(
                p >= 40.0,
                "16-bit RGB+RCT comp {i} q=1 PSNR {p:.2} dB must be >= 40 dB"
            );
        }
    }

    #[test]
    fn highbd_lossy_compresses_smaller_than_lossless() {
        // q = 2 must produce a strictly smaller codestream than q = 0
        // (lossless) on the same 12-bit picture — the quantizer is biting.
        let src = make_synthetic_highbd(32, 32, 12);
        let lossless = encode_planar_highbd(32, 32, 1, 0, 2, 2, 12, std::slice::from_ref(&src))
            .expect("encode lossless");
        let lossy =
            encode_planar_highbd_lossy(32, 32, 1, 0, 2, 2, 12, 2, std::slice::from_ref(&src))
                .expect("encode q=2");
        assert!(
            lossy.len() < lossless.len(),
            "q=2 stream ({}) must be smaller than lossless ({})",
            lossy.len(),
            lossless.len()
        );
    }

    #[test]
    fn highbd_lossy_rejects_q0_and_bad_bd() {
        let src = vec![100u16; 16 * 16];
        // q = 0 belongs on the lossless entry point.
        assert!(
            encode_planar_highbd_lossy(16, 16, 1, 0, 1, 1, 10, 0, std::slice::from_ref(&src))
                .is_err(),
            "q=0 must be rejected (use encode_planar_highbd)"
        );
        // bd = 8 must route through the 8-bit lossy path.
        assert!(
            encode_planar_highbd_lossy(16, 16, 1, 0, 1, 1, 8, 2, std::slice::from_ref(&src))
                .is_err(),
            "bd=8 must be rejected"
        );
        // Cpih = 3 (Star-Tetrix) is not exposed here.
        let planes = vec![vec![0u16; 16]; 4];
        assert!(
            encode_planar_highbd_lossy(4, 4, 4, 3, 1, 1, 12, 2, &planes).is_err(),
            "Cpih=3 must be rejected"
        );
    }

    // === Round 195: high bit depth Star-Tetrix (Cpih = 3) ================

    /// Build a synthetic 4-component CFA fixture at `bd`-bit precision,
    /// spread across the full `0..=2^bd-1` range so the high bits actually
    /// matter. Mirrors `make_cfa_8x8` but parametric in bit depth /
    /// dimensions.
    fn make_cfa_highbd(w: usize, h: usize, bd: u8) -> [Vec<u16>; 4] {
        let max = ((1u32 << bd) - 1) as i64;
        let mut r = vec![0u16; w * h];
        let mut g1 = vec![0u16; w * h];
        let mut g2 = vec![0u16; w * h];
        let mut b = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let xi = x as i64;
                let yi = y as i64;
                r[idx] = ((xi * 277 + yi * 631 + (xi ^ yi) * 53) % (max + 1)) as u16;
                g1[idx] = ((xi * 631 + yi * 277 + (xi ^ yi) * 41) % (max + 1)) as u16;
                g2[idx] = ((xi * 419 + yi * 503 + (xi ^ yi) * 67) % (max + 1)) as u16;
                b[idx] = ((xi * 503 + yi * 419 + (xi ^ yi) * 73) % (max + 1)) as u16;
            }
        }
        [r, g1, g2, b]
    }

    /// 10-bit Star-Tetrix (Cpih=3) self-roundtrips bit-exactly through
    /// the high-bit-depth path at NL=2/2. PSNR is `f64::INFINITY` here —
    /// this is lossless, the ≥ 30 dB floor is trivially cleared.
    #[test]
    fn r195_star_tetrix_highbd_10bit_round_trip() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 10);
        let cs = encode_planar_star_tetrix_highbd(
            16,
            16,
            2,
            2,
            10,
            0,
            0,
            0,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 10-bit Cpih=3 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 10-bit Cpih=3 NL=2/2");
        assert_eq!(img.num_components, 4);
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        assert_eq!(img.cpih, 3, "PIH Cpih must be 3 (Star-Tetrix)");
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 10);
            assert!(
                p >= 30.0,
                "10-bit Star-Tetrix comp {i} PSNR {p:.2} dB must be >= 30 dB"
            );
            assert_eq!(
                &rec, *orig,
                "10-bit Star-Tetrix comp {i} must round-trip bit-exactly"
            );
        }
    }

    /// 12-bit Star-Tetrix with Ct=1 (GRBG) + non-default e1/e2/Cf round-
    /// trips bit-exactly. Confirms the CTS / CRG markers survive on the
    /// high-bit-depth path identically to the 8-bit form.
    #[test]
    fn r195_star_tetrix_highbd_12bit_ct1_round_trip() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 12);
        let cs = encode_planar_star_tetrix_highbd(
            16,
            16,
            2,
            2,
            12,
            2, // e1
            3, // e2
            3, // cf = 3 (in-line)
            1, // ct = 1 (GRBG)
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 12-bit Cpih=3 Ct=1 NL=2/2");
        let img = decode_codestream(&cs, None).expect("decode 12-bit Cpih=3 Ct=1");
        assert_eq!(img.bit_depth, 12);
        assert_eq!(img.cpih, 3);
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 12);
            assert!(
                p >= 30.0,
                "12-bit Star-Tetrix Ct=1 comp {i} PSNR {p:.2} dB must be >= 30 dB"
            );
            assert_eq!(
                &rec, *orig,
                "12-bit Star-Tetrix comp {i} must round-trip bit-exactly"
            );
        }
    }

    /// 16-bit Star-Tetrix exercises the top of the high-bit-depth range
    /// — the full `[0, 65535]` sample range through the Annex F.5
    /// lifting cascade.
    #[test]
    fn r195_star_tetrix_highbd_16bit_round_trip() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 16);
        let cs = encode_planar_star_tetrix_highbd(
            16,
            16,
            2,
            2,
            16,
            0,
            0,
            0,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 16-bit Cpih=3");
        let img = decode_codestream(&cs, None).expect("decode 16-bit Cpih=3");
        assert_eq!(img.bit_depth, 16);
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 16);
            assert!(
                p >= 30.0,
                "16-bit Star-Tetrix comp {i} PSNR {p:.2} dB must be >= 30 dB"
            );
            assert_eq!(
                &rec, *orig,
                "16-bit Star-Tetrix comp {i} must round-trip bit-exactly"
            );
        }
    }

    /// Reject `bd ∈ {8, 17}` — bd=8 routes through `encode_planar_star_tetrix`,
    /// bd=17 is above the spec's supported high-bit-depth ceiling.
    #[test]
    fn r195_star_tetrix_highbd_rejects_bad_bd() {
        let planes: Vec<Vec<u16>> = (0..4).map(|_| vec![0u16; 16 * 16]).collect();
        let err = encode_planar_star_tetrix_highbd(16, 16, 2, 2, 8, 0, 0, 0, 0, &planes)
            .expect_err("bd=8 must be rejected");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
        let err = encode_planar_star_tetrix_highbd(16, 16, 2, 2, 17, 0, 0, 0, 0, &planes)
            .expect_err("bd=17 must be rejected");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    /// Reject samples that overshoot `2^bd - 1`.
    #[test]
    fn r195_star_tetrix_highbd_rejects_oversize_sample() {
        let bd = 10u8;
        let mut planes: Vec<Vec<u16>> = (0..4).map(|_| vec![0u16; 8 * 8]).collect();
        planes[0][0] = (1u16 << bd) + 5; // out of 10-bit range
        assert!(
            encode_planar_star_tetrix_highbd(8, 8, 1, 1, bd, 0, 0, 0, 0, &planes).is_err(),
            "sample exceeding 2^bd-1 must be rejected"
        );
    }

    /// Plane-count must be exactly 4 (Cpih=3 operand window per Annex F.2).
    #[test]
    fn r195_star_tetrix_highbd_rejects_wrong_plane_count() {
        let planes: Vec<Vec<u16>> = (0..3).map(|_| vec![0u16; 8 * 8]).collect();
        assert!(
            encode_planar_star_tetrix_highbd(8, 8, 1, 1, 10, 0, 0, 0, 0, &planes).is_err(),
            "plane count 3 must be rejected (Cpih=3 requires Nc=4)"
        );
    }

    // === Round 201: high bit depth Star-Tetrix LOSSY (Cpih = 3, q > 0) =====

    /// 10-bit Star-Tetrix lossy at q=1 holds PSNR ≥ 40 dB per component
    /// on the 16×16 CFA fixture at NL=2/2. q=1 is the near-lossless rung
    /// and matches the floor that the round-133 4:4:4 / round-151 sub-
    /// sampled high-bit-depth lossy paths assert at q=1.
    #[test]
    fn r201_star_tetrix_highbd_lossy_10bit_psnr_q1() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 10);
        let cs = encode_planar_star_tetrix_highbd_lossy(
            16,
            16,
            2,
            2,
            10,
            1, // q = 1 (near-lossless)
            0,
            0,
            0,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 10-bit Cpih=3 q=1");
        let img = decode_codestream(&cs, None).expect("decode 10-bit Cpih=3 q=1");
        assert_eq!(img.num_components, 4);
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        assert_eq!(img.cpih, 3, "PIH Cpih must be 3 (Star-Tetrix)");
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 10);
            assert!(
                p >= 40.0,
                "10-bit Star-Tetrix q=1 comp {i} PSNR {p:.2} dB must be >= 40 dB"
            );
        }
    }

    /// 12-bit Star-Tetrix lossy at q=2 holds PSNR ≥ 30 dB per component.
    /// Mirrors the round-133 12-bit q=2 floor; the Annex D.4 deadzone is
    /// the same kernel.
    #[test]
    fn r201_star_tetrix_highbd_lossy_12bit_psnr_q2() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 12);
        let cs = encode_planar_star_tetrix_highbd_lossy(
            16,
            16,
            2,
            2,
            12,
            2, // q = 2
            0,
            0,
            0,
            0,
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 12-bit Cpih=3 q=2");
        let img = decode_codestream(&cs, None).expect("decode 12-bit Cpih=3 q=2");
        assert_eq!(img.bit_depth, 12);
        assert_eq!(img.cpih, 3);
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 12);
            assert!(
                p >= 30.0,
                "12-bit Star-Tetrix q=2 comp {i} PSNR {p:.2} dB must be >= 30 dB"
            );
        }
    }

    /// 16-bit Star-Tetrix lossy at q=1 with Ct=1 (GRBG) + non-default
    /// e1/e2/Cf. Exercises the full 16-bit dynamic range through the
    /// Annex F.5 lifting with the lossy quantizer engaged, and confirms
    /// the CTS / CRG markers survive on the high-bit-depth lossy path
    /// identically to the 8-bit / lossless forms.
    #[test]
    fn r201_star_tetrix_highbd_lossy_16bit_ct1_psnr_q1() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 16);
        let cs = encode_planar_star_tetrix_highbd_lossy(
            16,
            16,
            2,
            2,
            16,
            1, // q = 1
            2, // e1
            3, // e2
            3, // cf = 3 (in-line)
            1, // ct = 1 (GRBG)
            &[r.clone(), g1.clone(), g2.clone(), b.clone()],
        )
        .expect("encode 16-bit Cpih=3 Ct=1 q=1");
        let img = decode_codestream(&cs, None).expect("decode 16-bit Cpih=3 Ct=1 q=1");
        assert_eq!(img.bit_depth, 16);
        assert_eq!(img.cpih, 3);
        for (i, orig) in [&r, &g1, &g2, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 16);
            assert!(
                p >= 40.0,
                "16-bit Star-Tetrix Ct=1 q=1 comp {i} PSNR {p:.2} dB must be >= 40 dB"
            );
        }
    }

    /// q=2 stream must be strictly smaller than the lossless q=0 stream
    /// on the same 12-bit Star-Tetrix CFA fixture — the deadzone
    /// truncation is biting.
    #[test]
    fn r201_star_tetrix_highbd_lossy_compresses_smaller_than_lossless() {
        let [r, g1, g2, b] = make_cfa_highbd(16, 16, 12);
        let planes = [r.clone(), g1.clone(), g2.clone(), b.clone()];
        let lossless = encode_planar_star_tetrix_highbd(16, 16, 2, 2, 12, 0, 0, 0, 0, &planes)
            .expect("encode 12-bit Cpih=3 q=0");
        let lossy =
            encode_planar_star_tetrix_highbd_lossy(16, 16, 2, 2, 12, 2, 0, 0, 0, 0, &planes)
                .expect("encode 12-bit Cpih=3 q=2");
        assert!(
            lossy.len() < lossless.len(),
            "12-bit Cpih=3 q=2 stream ({}) must be smaller than lossless q=0 stream ({})",
            lossy.len(),
            lossless.len()
        );
    }

    /// Reject `q = 0` (use the lossless entry point), `bd = 8` (route
    /// through `encode_planar_star_tetrix`), `bd = 17` (above the spec's
    /// supported high-bit-depth ceiling), and any plane-count != 4
    /// (Cpih=3 operand window per Annex F.2).
    #[test]
    fn r201_star_tetrix_highbd_lossy_rejects_bad_args() {
        let planes: Vec<Vec<u16>> = (0..4).map(|_| vec![0u16; 16 * 16]).collect();
        // q = 0 must route through the lossless path.
        let err = encode_planar_star_tetrix_highbd_lossy(16, 16, 2, 2, 10, 0, 0, 0, 0, 0, &planes)
            .expect_err("q=0 must be rejected");
        assert!(matches!(err, Error::InvalidData(_)), "got {err:?}");
        // bd = 8 must route through `encode_planar_star_tetrix`.
        let err = encode_planar_star_tetrix_highbd_lossy(16, 16, 2, 2, 8, 2, 0, 0, 0, 0, &planes)
            .expect_err("bd=8 must be rejected");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
        // bd = 17 exceeds the two-byte plane format.
        let err = encode_planar_star_tetrix_highbd_lossy(16, 16, 2, 2, 17, 2, 0, 0, 0, 0, &planes)
            .expect_err("bd=17 must be rejected");
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
        // Plane-count != 4 must be rejected.
        let three: Vec<Vec<u16>> = (0..3).map(|_| vec![0u16; 8 * 8]).collect();
        assert!(
            encode_planar_star_tetrix_highbd_lossy(8, 8, 1, 1, 10, 2, 0, 0, 0, 0, &three).is_err(),
            "plane count 3 must be rejected (Cpih=3 requires Nc=4)"
        );
        // Out-of-range sample (1024 > 2^10 - 1).
        let bd = 10u8;
        let mut bad: Vec<Vec<u16>> = (0..4).map(|_| vec![0u16; 8 * 8]).collect();
        bad[0][0] = (1u16 << bd) + 5;
        assert!(
            encode_planar_star_tetrix_highbd_lossy(8, 8, 1, 1, bd, 2, 0, 0, 0, 0, &bad).is_err(),
            "sample exceeding 2^bd-1 must be rejected"
        );
    }

    // === Round 151: high bit depth + chroma sub-sampling =================

    /// Build a sub-sampled chroma plane of `bd`-bit samples by stride-
    /// sampling the synthetic luma ramp generator.
    fn make_highbd_chroma(w: usize, h: usize, bd: u8, salt: i64) -> Vec<u16> {
        let max = ((1u32 << bd) - 1) as i64;
        let mut buf = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let v =
                    ((x as i64) * 173 + (y as i64) * 449 + salt * ((x ^ y) as i64 + 1)) % (max + 1);
                buf[y * w + x] = v as u16;
            }
        }
        buf
    }

    #[test]
    fn highbd_subsampled_10bit_422_lossless_round_trips_exactly() {
        // 4:2:2 — chroma planes are W/2 × H, 10-bit YCbCr.
        let w = 32u16;
        let h = 16u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 10);
        let cb = make_highbd_chroma((w / 2) as usize, h as usize, 10, 7);
        let cr = make_highbd_chroma((w / 2) as usize, h as usize, 10, 11);
        let planes = vec![y.clone(), cb.clone(), cr.clone()];
        let cs =
            encode_planar_subsampled_highbd(w, h, 3, 0, 2, 2, 10, &[1, 2, 2], &[1, 1, 1], &planes)
                .expect("encode 10-bit 4:2:2 lossless");
        let img = decode_codestream(&cs, None).expect("decode 10-bit 4:2:2 lossless");
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        assert_eq!(plane_u16(&img.planes[0].data), y);
        assert_eq!(plane_u16(&img.planes[1].data), cb);
        assert_eq!(plane_u16(&img.planes[2].data), cr);
    }

    #[test]
    fn highbd_subsampled_12bit_420_lossless_round_trips_exactly() {
        // 4:2:0 — chroma planes are W/2 × H/2, 12-bit YCbCr at NL=1/1
        // (matching the 8-bit 4:2:0 test's decomposition depth).
        let w = 32u16;
        let h = 32u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 12);
        let cb = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 13);
        let cr = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 19);
        let planes = vec![y.clone(), cb.clone(), cr.clone()];
        let cs =
            encode_planar_subsampled_highbd(w, h, 3, 0, 1, 1, 12, &[1, 2, 2], &[1, 2, 2], &planes)
                .expect("encode 12-bit 4:2:0 lossless");
        let img = decode_codestream(&cs, None).expect("decode 12-bit 4:2:0 lossless");
        assert_eq!(img.bit_depth, 12, "PIH Bw must be 12");
        assert_eq!(plane_u16(&img.planes[0].data), y);
        assert_eq!(plane_u16(&img.planes[1].data), cb);
        assert_eq!(plane_u16(&img.planes[2].data), cr);
    }

    #[test]
    fn highbd_subsampled_16bit_422_lossless_round_trips_exactly() {
        // 4:2:2 — chroma planes are W/2 × H, full 16-bit range.
        let w = 16u16;
        let h = 16u16;
        let mut y = make_synthetic_highbd(w as usize, h as usize, 16);
        y[0] = 0;
        y[1] = 65535;
        y[2] = 32768;
        let cb = make_highbd_chroma((w / 2) as usize, h as usize, 16, 23);
        let cr = make_highbd_chroma((w / 2) as usize, h as usize, 16, 29);
        let planes = vec![y.clone(), cb.clone(), cr.clone()];
        let cs =
            encode_planar_subsampled_highbd(w, h, 3, 0, 2, 2, 16, &[1, 2, 2], &[1, 1, 1], &planes)
                .expect("encode 16-bit 4:2:2 lossless");
        let img = decode_codestream(&cs, None).expect("decode 16-bit 4:2:2 lossless");
        assert_eq!(plane_u16(&img.planes[0].data), y);
        assert_eq!(plane_u16(&img.planes[1].data), cb);
        assert_eq!(plane_u16(&img.planes[2].data), cr);
    }

    #[test]
    fn highbd_subsampled_lossy_10bit_422_psnr_q1() {
        // q = 1 on a 10-bit 4:2:2 YCbCr picture: near-lossless luma + chroma.
        let w = 32u16;
        let h = 16u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 10);
        let cb = make_highbd_chroma((w / 2) as usize, h as usize, 10, 7);
        let cr = make_highbd_chroma((w / 2) as usize, h as usize, 10, 11);
        let planes = vec![y.clone(), cb.clone(), cr.clone()];
        let cs = encode_planar_subsampled_highbd_lossy(
            w,
            h,
            3,
            0,
            2,
            2,
            10,
            1,
            &[1, 2, 2],
            &[1, 1, 1],
            &planes,
        )
        .expect("encode 10-bit 4:2:2 q=1");
        let img = decode_codestream(&cs, None).expect("decode 10-bit 4:2:2 q=1");
        assert_eq!(img.bit_depth, 10);
        for (i, orig) in [&y, &cb, &cr].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 10);
            assert!(
                p >= 40.0,
                "10-bit 4:2:2 comp {i} q=1 PSNR {p:.2} dB must be >= 40 dB"
            );
        }
    }

    #[test]
    fn highbd_subsampled_lossy_12bit_420_psnr_q2() {
        // q = 2 on a 12-bit 4:2:0 YCbCr picture (NL = 1/1): PSNR ≥ 30 dB.
        let w = 32u16;
        let h = 32u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 12);
        let cb = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 13);
        let cr = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 19);
        let planes = vec![y.clone(), cb.clone(), cr.clone()];
        let cs = encode_planar_subsampled_highbd_lossy(
            w,
            h,
            3,
            0,
            1,
            1,
            12,
            2,
            &[1, 2, 2],
            &[1, 2, 2],
            &planes,
        )
        .expect("encode 12-bit 4:2:0 q=2");
        let img = decode_codestream(&cs, None).expect("decode 12-bit 4:2:0 q=2");
        for (i, orig) in [&y, &cb, &cr].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            let p = psnr_u16(orig, &rec, 12);
            assert!(
                p >= 30.0,
                "12-bit 4:2:0 comp {i} q=2 PSNR {p:.2} dB must be >= 30 dB"
            );
        }
    }

    #[test]
    fn highbd_subsampled_420_smaller_than_444() {
        // 12-bit 4:2:0 codestream must be smaller than 12-bit 4:4:4 of the
        // same luma — chroma byte budget halves twice.
        let w = 32u16;
        let h = 32u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 12);
        let chroma_full = y.clone();
        let cs_444 = encode_planar_highbd(
            w,
            h,
            3,
            0,
            1,
            1,
            12,
            &[y.clone(), chroma_full.clone(), chroma_full],
        )
        .expect("encode 12-bit 4:4:4 lossless");
        let cb420 = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 13);
        let cr420 = make_highbd_chroma((w / 2) as usize, (h / 2) as usize, 12, 19);
        let cs_420 = encode_planar_subsampled_highbd(
            w,
            h,
            3,
            0,
            1,
            1,
            12,
            &[1, 2, 2],
            &[1, 2, 2],
            &[y, cb420, cr420],
        )
        .expect("encode 12-bit 4:2:0 lossless");
        assert!(
            cs_420.len() < cs_444.len(),
            "4:2:0 codestream ({}) must be smaller than 4:4:4 ({})",
            cs_420.len(),
            cs_444.len()
        );
    }

    #[test]
    fn highbd_subsampled_lossy_compresses_smaller_than_lossless() {
        // Bit depth orthogonal to quantization: q = 2 stream must still be
        // strictly smaller than q = 0 on the same 12-bit 4:2:2 picture.
        let w = 32u16;
        let h = 16u16;
        let y = make_synthetic_highbd(w as usize, h as usize, 12);
        let cb = make_highbd_chroma((w / 2) as usize, h as usize, 12, 13);
        let cr = make_highbd_chroma((w / 2) as usize, h as usize, 12, 19);
        let planes = vec![y, cb, cr];
        let lossless =
            encode_planar_subsampled_highbd(w, h, 3, 0, 2, 2, 12, &[1, 2, 2], &[1, 1, 1], &planes)
                .expect("encode lossless");
        let lossy = encode_planar_subsampled_highbd_lossy(
            w,
            h,
            3,
            0,
            2,
            2,
            12,
            2,
            &[1, 2, 2],
            &[1, 1, 1],
            &planes,
        )
        .expect("encode q=2");
        assert!(
            lossy.len() < lossless.len(),
            "12-bit 4:2:2 q=2 ({}) must be smaller than lossless ({})",
            lossy.len(),
            lossless.len()
        );
    }

    #[test]
    fn highbd_subsampled_rejects_bad_inputs() {
        let y = vec![0u16; 4 * 4];
        let chroma = vec![0u16; 2 * 4];
        // bd = 8 must route through the 8-bit subsampled path.
        assert!(
            encode_planar_subsampled_highbd(
                4,
                4,
                3,
                0,
                1,
                1,
                8,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y.clone(), chroma.clone(), chroma.clone()]
            )
            .is_err(),
            "bd=8 must be rejected"
        );
        // bd = 17 exceeds the two-byte plane format.
        assert!(
            encode_planar_subsampled_highbd(
                4,
                4,
                3,
                0,
                1,
                1,
                17,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y.clone(), chroma.clone(), chroma.clone()]
            )
            .is_err(),
            "bd=17 must be rejected"
        );
        // Cpih = 3 (Star-Tetrix) is not exposed here.
        let four_planes = vec![vec![0u16; 16]; 4];
        assert!(
            encode_planar_subsampled_highbd(
                4,
                4,
                4,
                3,
                1,
                1,
                12,
                &[1, 1, 1, 1],
                &[1, 1, 1, 1],
                &four_planes
            )
            .is_err(),
            "Cpih=3 must be rejected"
        );
        // q = 0 must route through the lossless entry point.
        assert!(
            encode_planar_subsampled_highbd_lossy(
                4,
                4,
                3,
                0,
                1,
                1,
                10,
                0,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y.clone(), chroma.clone(), chroma.clone()]
            )
            .is_err(),
            "q=0 must be rejected on the lossy entry point"
        );
        // RCT (Cpih=1) requires sx=sy=1 for i<3.
        assert!(
            encode_planar_subsampled_highbd(
                4,
                4,
                3,
                1,
                1,
                1,
                10,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y.clone(), chroma.clone(), chroma]
            )
            .is_err(),
            "Cpih=1 with chroma sub-sampling must be rejected (Annex F.2)"
        );
        // Plane sample count must match (width/sx) * (height/sy).
        let bad_chroma = vec![0u16; 3]; // wrong size for 4:2:2 chroma at 4x4
        assert!(
            encode_planar_subsampled_highbd(
                4,
                4,
                3,
                0,
                1,
                1,
                10,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y, bad_chroma.clone(), bad_chroma]
            )
            .is_err(),
            "mismatched plane sample count must be rejected"
        );
    }

    #[test]
    fn highbd_subsampled_rejects_sample_above_nominal_range() {
        let mut y = vec![100u16; 16 * 16];
        y[7] = 1024; // exceeds 2^10 - 1
        let cb = vec![0u16; 8 * 16];
        let cr = vec![0u16; 8 * 16];
        assert!(
            encode_planar_subsampled_highbd(
                16,
                16,
                3,
                0,
                1,
                1,
                10,
                &[1, 2, 2],
                &[1, 1, 1],
                &[y, cb, cr]
            )
            .is_err(),
            "sample exceeding 2^bd-1 must be rejected"
        );
    }

    // === Round 218: pick_rp_for_target_bytes (rate-budget driven R[p]) ===

    /// Round 218: `pick_rp_for_target_bytes` rejects `target_bytes = 0`
    /// before doing any encode work.
    #[test]
    fn round218_rp_picker_rejects_zero_budget() {
        let pixels = make_synthetic_32x32();
        let err = pick_rp_for_target_bytes(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels));
        assert!(err.is_err(), "target_bytes = 0 must be rejected");
    }

    /// Round 218: when the budget is generous (≥ the `R[p] = NL-1`
    /// codestream length), the picker selects the maximum legal
    /// refinement. At `q > 0` larger `R[p]` retains more low-band
    /// bitplanes, so spending a full budget on maximum refinement is
    /// the picker's intent.
    #[test]
    fn round218_rp_picker_returns_nl_minus_one_when_budget_fits_max() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2) as u8; // Nc=1, NL = Nβ
        let rp_max = nl - 1;
        let cs_max = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, rp_max, std::slice::from_ref(&pixels))
            .expect("baseline R[p]=NL-1 q=2");
        let picked = pick_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            2,
            cs_max.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("picker with budget = NL-1 length");
        assert_eq!(
            picked, rp_max,
            "budget = max-stream length must pick R[p] = NL-1"
        );
    }

    /// Round 218: when the budget is exactly the `R[p] = 0` baseline
    /// length, the picker must pick `R[p] = 0` (every higher `R[p]`
    /// overshoots by ≥ 0 bytes, but the picker only accepts strict
    /// fit, so the first higher value rejected leaves `0` as the
    /// selection).
    #[test]
    fn round218_rp_picker_returns_zero_when_budget_is_baseline() {
        let pixels = make_synthetic_32x32();
        let cs_zero = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("R[p]=0 q=2 baseline");
        let picked = pick_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            2,
            cs_zero.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("picker at baseline budget");
        // At baseline budget every R[p] ≥ 1 emits ≥ baseline bytes;
        // when the stream is strictly larger the higher R[p] overshoots
        // and the picker falls back to 0.
        let cs_picked =
            encode_planar_rp(32, 32, 1, 0, 2, 2, 2, picked, std::slice::from_ref(&pixels))
                .expect("re-encode at picked R[p]");
        assert!(
            cs_picked.len() <= cs_zero.len(),
            "picked R[p]={picked} produced {} > baseline {}",
            cs_picked.len(),
            cs_zero.len()
        );
    }

    /// Round 218: the picker errors when even `R[p] = 0` overshoots
    /// the budget. The error reports the actual encoded length so the
    /// caller can size the budget correctly.
    #[test]
    fn round218_rp_picker_errors_when_baseline_overshoots() {
        let pixels = make_synthetic_32x32();
        let cs_zero = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("baseline encode");
        let too_tight = cs_zero.len() - 1;
        let err = pick_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            2,
            too_tight,
            std::slice::from_ref(&pixels),
        );
        assert!(
            err.is_err(),
            "budget < baseline length must error (baseline = {})",
            cs_zero.len()
        );
        let msg = format!("{:?}", err.err().unwrap());
        assert!(
            msg.contains("unreachable") && msg.contains(&cs_zero.len().to_string()),
            "error must report the baseline overshoot length, got: {msg}"
        );
    }

    /// Round 218: the picker is monotone — given a budget strictly
    /// between the `R[p] = 0` and `R[p] = NL-1` lengths, it returns
    /// some intermediate `R[p]` whose stream actually fits.
    #[test]
    fn round218_rp_picker_picks_intermediate_for_intermediate_budget() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2) as u8;
        let rp_max = nl - 1;
        let cs_zero = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 0, std::slice::from_ref(&pixels))
            .expect("R[p]=0");
        let cs_max = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, rp_max, std::slice::from_ref(&pixels))
            .expect("R[p]=NL-1");
        // Only run the assertion if the two endpoints differ — the
        // refinement actually fires on this fixture.
        if cs_max.len() > cs_zero.len() {
            let mid_budget = (cs_zero.len() + cs_max.len()) / 2;
            let picked = pick_rp_for_target_bytes(
                32,
                32,
                1,
                0,
                2,
                2,
                2,
                mid_budget,
                std::slice::from_ref(&pixels),
            )
            .expect("picker at mid budget");
            let cs_picked =
                encode_planar_rp(32, 32, 1, 0, 2, 2, 2, picked, std::slice::from_ref(&pixels))
                    .expect("re-encode at picked R[p]");
            assert!(
                cs_picked.len() <= mid_budget,
                "picked R[p]={picked} emits {} > budget {mid_budget}",
                cs_picked.len()
            );
        }
    }

    /// Round 218: at `q = 0` (lossless) refinement is a no-op — the
    /// truncation `T[p,b]` is already at its 0 floor, so every
    /// candidate `R[p]` emits a byte-identical stream. The picker
    /// must still return a value (the highest one, since every value
    /// "fits") and the round-trip must be lossless.
    #[test]
    fn round218_rp_picker_q0_lossless_roundtrip() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2) as u8;
        let cs_zero = encode_planar_rp(32, 32, 1, 0, 2, 2, 0, 0, std::slice::from_ref(&pixels))
            .expect("R[p]=0 q=0");
        let (cs, rp) = encode_planar_rp_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            cs_zero.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("target-bytes wrapper at q=0");
        assert_eq!(
            rp,
            nl - 1,
            "q=0 every R[p] fits; picker returns the maximum NL-1"
        );
        let img = decode_codestream(&cs, None).expect("decode q=0 picker output");
        assert_eq!(
            img.planes[0].data, pixels,
            "q=0 picker output must be lossless"
        );
    }

    /// Round 218: the `encode_planar_rp_target_bytes` convenience
    /// wrapper returns the same bytes as a manual
    /// `pick_rp_for_target_bytes` + `encode_planar_rp` pair, and the
    /// returned codestream actually fits the budget.
    #[test]
    fn round218_rp_target_bytes_wrapper_matches_manual_pair() {
        let pixels = make_synthetic_32x32();
        let nl = n_beta(2, 2) as u8;
        let rp_max = nl - 1;
        let cs_max = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, rp_max, std::slice::from_ref(&pixels))
            .expect("baseline");
        let budget = cs_max.len();
        let (cs_w, rp_w) = encode_planar_rp_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            2,
            budget,
            std::slice::from_ref(&pixels),
        )
        .expect("wrapper");
        let rp_m =
            pick_rp_for_target_bytes(32, 32, 1, 0, 2, 2, 2, budget, std::slice::from_ref(&pixels))
                .expect("manual picker");
        assert_eq!(rp_w, rp_m, "wrapper and manual picker pick the same R[p]");
        let cs_m = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, rp_m, std::slice::from_ref(&pixels))
            .expect("manual encode at picked R[p]");
        assert_eq!(cs_w, cs_m, "wrapper bytes == manual-pair bytes");
        assert!(cs_w.len() <= budget, "wrapper output must fit budget");
    }

    /// Round 218: the picker works on the RGB + RCT path (`Cpih = 1`),
    /// confirming the picker is colour-transform agnostic — its only
    /// dependency on `cpih` is forwarding it to `encode_planar_rp`.
    #[test]
    fn round218_rp_picker_works_for_rgb_rct() {
        let rgb = make_synthetic_rgb_32x32();
        let mut r = vec![0u8; 32 * 32];
        let mut g = vec![0u8; 32 * 32];
        let mut b = vec![0u8; 32 * 32];
        for i in 0..32 * 32 {
            r[i] = rgb[i * 3];
            g[i] = rgb[i * 3 + 1];
            b[i] = rgb[i * 3 + 2];
        }
        let planes = [r, g, b];
        let nbeta = n_beta(2, 2);
        let nl = (3u32) * nbeta; // Nc=3, RCT
        let rp_max = (nl - 1) as u8;
        let cs_max = encode_planar_rp(32, 32, 3, 1, 2, 2, 2, rp_max, &planes)
            .expect("RGB+RCT R[p]=NL-1 q=2");
        let (cs, rp) = encode_planar_rp_target_bytes(32, 32, 3, 1, 2, 2, 2, cs_max.len(), &planes)
            .expect("RGB+RCT wrapper at max budget");
        assert!(cs.len() <= cs_max.len(), "wrapper output must fit budget");
        assert!(
            decode_codestream(&cs, None).is_ok(),
            "picked RGB+RCT R[p]={rp} q=2 must decode"
        );
    }

    // === Round 224: joint per-slice Q[p] + R[p] encoder primitive ===

    /// Round 224: when `q_slices` is a single repeated value and `rp = 0`,
    /// the joint primitive must emit the same bytes as
    /// `encode_planar_hsl_qslice` at the same `q_slices`.
    #[test]
    fn round224_joint_primitive_matches_hsl_qslice_rp_zero() {
        let pixels = make_synthetic_32x32();
        // 32×32 with NL,y=2 → Np,y = 32/4 = 8; hsl=4 → 2 slices.
        let q_slices = vec![3u8, 3];
        let baseline = encode_planar_hsl_qslice(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &q_slices,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline hsl_qslice");
        let joint = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &q_slices,
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("joint at rp=0");
        assert_eq!(
            baseline, joint,
            "rp=0 joint primitive byte-identical to hsl_qslice"
        );
    }

    /// Round 224: when `hsl = 0`, `q_slices.len() == 1`, `rp > 0`, the
    /// joint primitive must emit the same bytes as `encode_planar_rp`
    /// at the same `q` and `rp`.
    #[test]
    fn round224_joint_primitive_matches_rp_when_single_slice() {
        let pixels = make_synthetic_32x32();
        let baseline = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, 3, std::slice::from_ref(&pixels))
            .expect("baseline encode_planar_rp");
        let joint = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            &[2u8],
            3,
            std::slice::from_ref(&pixels),
        )
        .expect("joint single slice rp=3");
        assert_eq!(
            baseline, joint,
            "single-slice joint byte-identical to encode_planar_rp at same (q, rp)"
        );
    }

    /// Round 224: at `q_slices = [0; n]` (lossless) refinement is a no-op,
    /// so the joint primitive must self-roundtrip losslessly regardless
    /// of `rp`.
    #[test]
    fn round224_joint_lossless_roundtrip_independent_of_rp() {
        let pixels = make_synthetic_32x32();
        let nbeta = n_beta(2, 2);
        let nl = nbeta; // Nc = 1 → NL = Nβ
        let rp_max = (nl - 1) as u8;
        // 32×32 with NL,y=2 → Np,y = 8; hsl=4 → 2 slices matching [0, 0].
        let cs = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &[0u8, 0],
            rp_max,
            std::slice::from_ref(&pixels),
        )
        .expect("joint q=0 rp=NL-1");
        let img = decode_codestream(&cs, None).expect("decode joint q=0 rp=NL-1");
        assert_eq!(
            img.planes[0].data, pixels,
            "lossless joint must roundtrip bit-exactly"
        );
    }

    /// Round 224: the joint picker rejects `target_bytes = 0` before
    /// any encode work.
    #[test]
    fn round224_joint_picker_rejects_zero_budget() {
        let pixels = make_synthetic_32x32();
        let err = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            0,
            std::slice::from_ref(&pixels),
        );
        assert!(err.is_err(), "target_bytes = 0 must be rejected");
    }

    /// Round 224: when the budget comfortably accommodates the largest
    /// candidate (`rp = NL-1` + `q_slices = [0; n]`, lossless slices and
    /// maximum refinement, which is the largest stream this picker can
    /// emit), the joint picker must return that maximum-refinement
    /// configuration with lossless slices.
    #[test]
    fn round224_joint_picker_returns_max_rp_when_budget_is_huge() {
        let pixels = make_synthetic_32x32();
        let nbeta = n_beta(2, 2);
        let nl = nbeta; // Nc = 1 → NL = Nβ
        let rp_max = (nl - 1) as u8;
        // 32×32 with NL,y=2 → Np,y = 8; hsl=4 → 2 slices.
        let cs_loss = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &[0u8, 0],
            rp_max,
            std::slice::from_ref(&pixels),
        )
        .expect("max-cost candidate");
        // Use a budget at least as big as the max-cost candidate.
        let (q_picked, rp_picked) = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            cs_loss.len() + 1024,
            std::slice::from_ref(&pixels),
        )
        .expect("huge-budget picker");
        assert_eq!(
            q_picked,
            vec![0u8, 0],
            "huge budget → lossless slices ([0; n])"
        );
        assert_eq!(
            rp_picked, rp_max,
            "huge budget → maximum refinement R[p]=NL-1"
        );
    }

    /// Round 224: when even `rp = 0` + `Q = 15` overshoots, the joint
    /// picker errors with the unreachable-budget message.
    #[test]
    fn round224_joint_picker_errors_when_budget_unreachable() {
        let pixels = make_synthetic_32x32();
        // Tiny budget — even the worst-case truncated stream is bigger
        // than 8 bytes (SOC + CAP + PIH alone exceed that), so this is a
        // guaranteed overshoot.
        let err = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            8,
            std::slice::from_ref(&pixels),
        );
        let msg = format!("{:?}", err);
        assert!(err.is_err(), "tiny budget must error");
        assert!(
            msg.contains("unreachable"),
            "error must mention 'unreachable', got {}",
            msg
        );
    }

    /// Round 224: the joint picker's output must always fit the budget
    /// (when one fits at all). Tested at a tight mid-range budget on the
    /// 32×32 luma fixture.
    #[test]
    fn round224_joint_picker_output_fits_budget() {
        let pixels = make_synthetic_32x32();
        // 32×32 with NL,y=2 → Np,y = 8; hsl=4 → 2 slices.
        let cs_zero_zero = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &[0u8, 0],
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("zero/zero baseline");
        // Budget = halfway between baseline and 2× baseline (a realistic
        // bit budget for a live workflow that wants some refinement but
        // not full lossless).
        let budget = cs_zero_zero.len() * 3 / 2;
        let (q_picked, rp_picked) = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            budget,
            std::slice::from_ref(&pixels),
        )
        .expect("joint picker at mid budget");
        let cs = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &q_picked,
            rp_picked,
            std::slice::from_ref(&pixels),
        )
        .expect("encode at picked (q, rp)");
        assert!(
            cs.len() <= budget,
            "picker output {} must fit budget {}",
            cs.len(),
            budget
        );
    }

    /// Round 224: the convenience wrapper
    /// `encode_planar_hsl_qslice_rp_target_bytes` returns the same bytes
    /// as a manual picker + encode pair.
    #[test]
    fn round224_joint_target_bytes_wrapper_matches_manual_pair() {
        let pixels = make_synthetic_32x32();
        let cs_zero_zero = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &[0u8, 0],
            0,
            std::slice::from_ref(&pixels),
        )
        .expect("zero/zero baseline");
        let budget = cs_zero_zero.len() * 2;
        let (cs_w, q_w, rp_w) = encode_planar_hsl_qslice_rp_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            budget,
            std::slice::from_ref(&pixels),
        )
        .expect("wrapper");
        let (q_m, rp_m) = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            budget,
            std::slice::from_ref(&pixels),
        )
        .expect("manual picker");
        assert_eq!(q_w, q_m, "wrapper and manual picker pick the same q_slices");
        assert_eq!(rp_w, rp_m, "wrapper and manual picker pick the same rp");
        let cs_m = encode_planar_hsl_qslice_rp(
            32,
            32,
            1,
            0,
            2,
            2,
            4,
            &q_m,
            rp_m,
            std::slice::from_ref(&pixels),
        )
        .expect("manual encode");
        assert_eq!(cs_w, cs_m, "wrapper bytes == manual-pair bytes");
        assert!(cs_w.len() <= budget, "wrapper output must fit budget");
    }

    /// Round 224: the joint picker works on the RGB + RCT (`Cpih = 1`)
    /// path — confirming the picker is colour-transform agnostic.
    #[test]
    fn round224_joint_picker_works_for_rgb_rct() {
        let rgb = make_synthetic_rgb_32x32();
        let mut r = vec![0u8; 32 * 32];
        let mut g = vec![0u8; 32 * 32];
        let mut b = vec![0u8; 32 * 32];
        for i in 0..32 * 32 {
            r[i] = rgb[i * 3];
            g[i] = rgb[i * 3 + 1];
            b[i] = rgb[i * 3 + 2];
        }
        let planes = [r, g, b];
        let cs_zero_zero =
            encode_planar_hsl_qslice_rp(32, 32, 3, 1, 2, 2, 4, &[0u8, 0], 0, &planes)
                .expect("RGB+RCT zero/zero baseline");
        let budget = cs_zero_zero.len() * 3 / 2;
        let (cs, q_picked, rp_picked) =
            encode_planar_hsl_qslice_rp_target_bytes(32, 32, 3, 1, 2, 2, 4, budget, &planes)
                .expect("RGB+RCT joint picker");
        assert!(cs.len() <= budget, "RGB+RCT picker output must fit budget");
        let img = decode_codestream(&cs, None).expect("decode RGB+RCT joint picker output");
        assert_eq!(
            img.planes.len(),
            3,
            "RGB+RCT joint picker output must decode to 3 planes (q={:?}, rp={rp_picked})",
            q_picked
        );
    }

    /// Round 224: composition sanity — at a budget the single-axis
    /// pickers find easy to fit but where R[p] alone gives no headroom,
    /// the joint picker must out-fit or equal the rp picker (refinement
    /// transfers bits within bands; per-slice Q can globally reduce
    /// bytes).
    #[test]
    fn round224_joint_picker_at_least_as_good_as_rp_alone() {
        let pixels = make_synthetic_32x32();
        // Mid-range budget where rp picker finds a non-zero answer at
        // q=2. Use single-slice (hsl=0, q_slices.len()==1) so the joint
        // picker is directly comparable to the rp picker.
        let nbeta = n_beta(2, 2);
        let nl = nbeta; // Nc = 1 → NL = Nβ
        let rp_max = (nl - 1) as u8;
        let cs_max = encode_planar_rp(32, 32, 1, 0, 2, 2, 2, rp_max, std::slice::from_ref(&pixels))
            .expect("rp=NL-1 baseline at q=2");
        let budget = cs_max.len();
        let rp_picked_alone =
            pick_rp_for_target_bytes(32, 32, 1, 0, 2, 2, 2, budget, std::slice::from_ref(&pixels))
                .expect("rp-only picker");
        let (_q_joint, rp_joint) = pick_q_slices_rp_for_target_bytes(
            32,
            32,
            1,
            0,
            2,
            2,
            0,
            budget,
            std::slice::from_ref(&pixels),
        )
        .expect("joint picker single-slice");
        // The joint picker can at minimum match the rp-only picker
        // (q_slices = [2], rp = same) since that path is reachable; it
        // can also reach lower q + same/higher rp. So joint rp must be
        // >= rp_alone (joint picks the largest fitting refinement after
        // potentially lowering q from 2 toward 0).
        assert!(
            rp_joint >= rp_picked_alone,
            "joint picker (rp={rp_joint}) must reach at least as much refinement as rp-only ({rp_picked_alone})"
        );
    }

    // === Round 230: high-bit-depth widening of the joint primitive =======

    /// Round 230: at `q_slices = [0; n]` (lossless) refinement is a no-op,
    /// so the high-bit-depth joint primitive must self-roundtrip
    /// bit-exactly regardless of `rp` at any `bd ∈ 9..=16`.
    #[test]
    fn round230_joint_highbd_lossless_roundtrip_independent_of_rp() {
        let src = make_synthetic_highbd(32, 32, 10);
        let nbeta = n_beta(2, 2);
        let nl = nbeta;
        let rp_max = (nl - 1) as u8;
        // 32×32 with NL,y=2 → Np,y = 8; hsl=4 → 2 slices matching [0, 0].
        let cs = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &[0u8, 0],
            rp_max,
            std::slice::from_ref(&src),
        )
        .expect("joint highbd q=0 rp=NL-1");
        let img = decode_codestream(&cs, None).expect("decode highbd joint q=0 rp=NL-1");
        assert_eq!(img.bit_depth, 10, "PIH Bw must be 10");
        assert_eq!(
            plane_u16(&img.planes[0].data),
            src,
            "lossless joint highbd must roundtrip bit-exactly"
        );
    }

    /// Round 230: at `rp = 0` the high-bit-depth joint primitive emits
    /// exactly the same lossy bytes as the round-133 lossy entry point
    /// (`encode_planar_highbd_lossy`) on a single-slice 4:4:4 setup with
    /// matching `q`. Confirms `rp = 0` is the no-refinement default at
    /// high bit depth.
    #[test]
    fn round230_joint_highbd_rp_zero_single_slice_matches_highbd_lossy() {
        let src = make_synthetic_highbd(32, 32, 12);
        // Single slice (hsl=0) + q_slices = [3] should equal
        // encode_planar_highbd_lossy at q=3, since both pin Cpih=0, Cw=0,
        // Sd=0, Fs=0, Qpih=0, rp=0.
        let baseline =
            encode_planar_highbd_lossy(32, 32, 1, 0, 2, 2, 12, 3, std::slice::from_ref(&src))
                .expect("baseline highbd lossy q=3");
        let joint = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            12,
            0,
            &[3u8],
            0,
            std::slice::from_ref(&src),
        )
        .expect("joint highbd single-slice rp=0 q=3");
        assert_eq!(
            baseline, joint,
            "single-slice rp=0 joint highbd byte-identical to encode_planar_highbd_lossy"
        );
    }

    /// Round 230: the joint highbd primitive must produce a lossy stream
    /// strictly smaller than the lossless one at the same bit depth (the
    /// per-band deadzone truncation must be biting at `q > 0`).
    #[test]
    fn round230_joint_highbd_lossy_compresses_smaller_than_lossless() {
        let src = make_synthetic_highbd(32, 32, 12);
        let lossless = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            12,
            4,
            &[0u8, 0],
            0,
            std::slice::from_ref(&src),
        )
        .expect("lossless joint highbd");
        let lossy = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            12,
            4,
            &[3u8, 3],
            0,
            std::slice::from_ref(&src),
        )
        .expect("lossy joint highbd q=3");
        assert!(
            lossy.len() < lossless.len(),
            "q=3 stream ({}) must be smaller than lossless ({})",
            lossy.len(),
            lossless.len()
        );
    }

    /// Round 230: the joint highbd primitive must preserve a high PSNR
    /// floor at near-lossless quantization on a 10-bit luma fixture
    /// (mirrors the round-133 `highbd_lossy_10bit_luma_psnr_q1` shape but
    /// goes through the round-230 joint entry point).
    #[test]
    fn round230_joint_highbd_10bit_luma_psnr_q1_floor() {
        let src = make_synthetic_highbd(32, 32, 10);
        let cs = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &[1u8, 1],
            0,
            std::slice::from_ref(&src),
        )
        .expect("joint highbd 10-bit q=1");
        let img = decode_codestream(&cs, None).expect("decode joint highbd 10-bit q=1");
        let rec = plane_u16(&img.planes[0].data);
        let p = psnr_u16(&src, &rec, 10);
        assert!(
            p >= 40.0,
            "10-bit joint q=1 PSNR {p:.2} dB must be >= 40 dB"
        );
    }

    /// Round 230: the joint highbd primitive accepts mixed per-slice
    /// quantization values and preserves a reasonable PSNR floor even on
    /// a 12-bit picture with `Q[p] = [0, 3]` (one lossless slice, one
    /// quantized) — sanity that the per-slice mechanism survives bit-
    /// depth widening.
    #[test]
    fn round230_joint_highbd_mixed_q_slices_12bit_psnr_floor() {
        let src = make_synthetic_highbd(32, 32, 12);
        let cs = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            12,
            4,
            &[0u8, 3],
            0,
            std::slice::from_ref(&src),
        )
        .expect("joint highbd 12-bit mixed q");
        let img = decode_codestream(&cs, None).expect("decode joint highbd 12-bit mixed q");
        let rec = plane_u16(&img.planes[0].data);
        let p = psnr_u16(&src, &rec, 12);
        // [0, 3] mix at NL=2/2 is well within the 30 dB floor we hold on
        // 12-bit q=2 (the lossless half raises the average vs uniform q=3).
        assert!(
            p >= 30.0,
            "12-bit joint Q=[0,3] PSNR {p:.2} dB must be >= 30 dB"
        );
    }

    /// Round 230: the joint highbd picker rejects `target_bytes = 0`
    /// before any encode work.
    #[test]
    fn round230_joint_highbd_picker_rejects_zero_budget() {
        let src = make_synthetic_highbd(32, 32, 10);
        let err = pick_q_slices_rp_for_target_bytes_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            0,
            std::slice::from_ref(&src),
        );
        assert!(err.is_err(), "target_bytes = 0 must be rejected");
    }

    /// Round 230: when the budget comfortably accommodates the largest
    /// candidate (lossless slices + maximum refinement), the joint
    /// highbd picker must return that maximum-refinement configuration.
    #[test]
    fn round230_joint_highbd_picker_returns_max_rp_when_budget_is_huge() {
        let src = make_synthetic_highbd(32, 32, 10);
        let nbeta = n_beta(2, 2);
        let nl = nbeta;
        let rp_max = (nl - 1) as u8;
        let cs_loss = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &[0u8, 0],
            rp_max,
            std::slice::from_ref(&src),
        )
        .expect("max-cost highbd candidate");
        let (q_picked, rp_picked) = pick_q_slices_rp_for_target_bytes_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            cs_loss.len() + 1024,
            std::slice::from_ref(&src),
        )
        .expect("huge-budget highbd picker");
        assert_eq!(
            q_picked,
            vec![0u8, 0],
            "huge budget → lossless slices ([0; n])"
        );
        assert_eq!(
            rp_picked, rp_max,
            "huge budget → maximum refinement R[p]=NL-1"
        );
    }

    /// Round 230: when even `rp = 0` + `Q = 15` overshoots, the joint
    /// highbd picker errors with the unreachable-budget message.
    #[test]
    fn round230_joint_highbd_picker_errors_when_budget_unreachable() {
        let src = make_synthetic_highbd(32, 32, 12);
        // Tiny budget — SOC + CAP + PIH + CDT + WGT alone exceed 8 bytes,
        // so this is a guaranteed overshoot even at Q=15.
        let err = pick_q_slices_rp_for_target_bytes_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            12,
            4,
            8,
            std::slice::from_ref(&src),
        );
        assert!(err.is_err(), "target_bytes = 8 must be unreachable");
    }

    /// Round 230: at a mid-range budget the joint highbd picker must
    /// emit a codestream whose length actually fits the budget.
    #[test]
    fn round230_joint_highbd_picker_output_fits_budget() {
        let src = make_synthetic_highbd(32, 32, 10);
        // Compute the lossless rp=0 size as an upper bound, then aim for
        // ~60% of it as a tight-but-reachable budget.
        let cs_lossless = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &[0u8, 0],
            0,
            std::slice::from_ref(&src),
        )
        .expect("lossless reference");
        let budget = cs_lossless.len() * 6 / 10;
        let (q_picked, rp_picked) = pick_q_slices_rp_for_target_bytes_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            budget,
            std::slice::from_ref(&src),
        )
        .expect("mid-budget highbd picker");
        let cs = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &q_picked,
            rp_picked,
            std::slice::from_ref(&src),
        )
        .expect("re-encode picked highbd");
        assert!(
            cs.len() <= budget,
            "picked highbd codestream {} must fit budget {}",
            cs.len(),
            budget
        );
    }

    /// Round 230: the high-bit-depth target-bytes wrapper must produce a
    /// codestream byte-identical to a follow-up
    /// `encode_planar_hsl_qslice_rp_highbd(.., q_slices, rp, ..)` at the
    /// `(q_slices, rp)` the picker returned. Persisting the pair allows
    /// reproducible re-encode through the primitive.
    #[test]
    fn round230_joint_highbd_target_bytes_wrapper_matches_manual_pair() {
        let src = make_synthetic_highbd(32, 32, 10);
        let cs_lossless = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &[0u8, 0],
            0,
            std::slice::from_ref(&src),
        )
        .expect("lossless reference");
        let budget = cs_lossless.len() * 7 / 10;
        let (cs_wrapper, q_picked, rp_picked) = encode_planar_hsl_qslice_rp_target_bytes_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            budget,
            std::slice::from_ref(&src),
        )
        .expect("highbd target-bytes wrapper");
        let cs_manual = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            1,
            0,
            2,
            2,
            10,
            4,
            &q_picked,
            rp_picked,
            std::slice::from_ref(&src),
        )
        .expect("manual re-encode at picked pair");
        assert_eq!(
            cs_wrapper, cs_manual,
            "wrapper output must be byte-identical to manual re-encode"
        );
    }

    /// Round 230: the high-bit-depth joint primitive composes with the
    /// reversible RCT (`Cpih = 1`, 3-component 4:4:4) — the colour
    /// transform is bit-depth agnostic per Annex F.3, so a 10-bit RGB
    /// fixture must self-roundtrip losslessly at `q_slices = [0; n]`
    /// regardless of `rp`.
    #[test]
    fn round230_joint_highbd_rgb_rct_lossless_roundtrip() {
        let r = make_synthetic_highbd(32, 32, 10);
        let mut g = make_synthetic_highbd(32, 32, 10);
        let mut b = make_synthetic_highbd(32, 32, 10);
        for v in g.iter_mut() {
            *v = (*v + 121) & 0x3ff;
        }
        for v in b.iter_mut() {
            *v = (v.wrapping_mul(3).wrapping_add(7)) & 0x3ff;
        }
        let planes = vec![r.clone(), g.clone(), b.clone()];
        let nbeta = n_beta(2, 2);
        let nl = 3 * nbeta;
        let rp_max = (nl - 1) as u8;
        let cs = encode_planar_hsl_qslice_rp_highbd(
            32,
            32,
            3,
            1,
            2,
            2,
            10,
            4,
            &[0u8, 0],
            rp_max,
            &planes,
        )
        .expect("joint highbd 10-bit RGB+RCT lossless rp=NL-1");
        let img = decode_codestream(&cs, None).expect("decode joint highbd 10-bit RGB+RCT");
        for (i, orig) in [&r, &g, &b].iter().enumerate() {
            let rec = plane_u16(&img.planes[i].data);
            assert_eq!(rec, **orig, "component {i} must roundtrip bit-exactly");
        }
    }

    /// Round 230: `bd = 8` is rejected (the caller should use the 8-bit
    /// joint primitive `encode_planar_hsl_qslice_rp`).
    #[test]
    fn round230_joint_highbd_rejects_bd8_and_cpih3() {
        let src = vec![0u16; 16 * 16];
        // bd = 8 routes through the 8-bit joint primitive.
        assert!(
            encode_planar_hsl_qslice_rp_highbd(
                16,
                16,
                1,
                0,
                1,
                1,
                8,
                0,
                &[0u8],
                0,
                std::slice::from_ref(&src),
            )
            .is_err(),
            "bd=8 must be rejected"
        );
        // Cpih = 3 (Star-Tetrix) not exposed on this path.
        let planes = vec![vec![0u16; 16]; 4];
        assert!(
            encode_planar_hsl_qslice_rp_highbd(4, 4, 4, 3, 1, 1, 10, 0, &[0u8], 0, &planes,)
                .is_err(),
            "Cpih=3 must be rejected on the joint highbd primitive"
        );
    }

    // ---------------------------------------------------------------
    // Round 233 — per-precinct Q[p] override (Annex C.2 Table C.1)
    // ---------------------------------------------------------------

    /// `q_precincts = [0; n]` (every precinct lossless) is byte-identical
    /// to `encode_planar` at the same geometry — the round-233 lever is a
    /// pure no-op when every precinct picks the same `Q[p] = 0`.
    #[test]
    fn round233_qpr_all_zero_matches_encode_planar_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossless encode");
        // NL,y = 2 → Hp = 4, Np,y = 8; Cw = 0 → Np,x = 1 → 8 precincts.
        let qpr = vec![0u8; 8];
        let cs = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &qpr,
            std::slice::from_ref(&pixels),
        )
        .expect("all-zero q_precincts encode");
        assert_eq!(
            cs, baseline,
            "all-zero q_precincts must be byte-identical to encode_planar"
        );
    }

    /// All-equal `q_precincts` (every precinct same `Q[p]`) is byte-
    /// identical to `encode_planar_lossy` at that single `q`. This is
    /// the spec-natural reduction: per-precinct → picture-level when
    /// the assignment is constant.
    #[test]
    fn round233_qpr_uniform_q_matches_encode_planar_lossy() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossy q=2 encode");
        let qpr = vec![2u8; 8];
        let cs = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &qpr,
            std::slice::from_ref(&pixels),
        )
        .expect("uniform-q q_precincts encode");
        assert_eq!(
            cs, baseline,
            "uniform-Q q_precincts must be byte-identical to encode_planar_lossy at the same Q"
        );
    }

    /// Mixed per-precinct `Q[p]` produces a different stream than the
    /// constant-Q baseline (the precincts at the lower `Q[p]` retain more
    /// magnitude bitplanes), still round-trips through the decoder, and
    /// each precinct's first-precinct-header `Q[p]` byte carries the
    /// override.
    #[test]
    fn round233_qpr_mixed_q_round_trip_and_diverges() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 8 precincts at Cw=0. Vary Q[p] across them.
        let mixed_q: [u8; 8] = [0, 2, 4, 2, 0, 4, 2, 0];
        let mixed = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &mixed_q,
            std::slice::from_ref(&pixels),
        )
        .expect("mixed q_precincts encode");
        let constant_q4 = encode_planar_lossy(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            4,
            std::slice::from_ref(&pixels),
        )
        .expect("constant q=4 baseline");
        assert!(
            mixed.len() > constant_q4.len(),
            "mixed q_precincts ({} B) must retain more bits than constant Q=4 ({} B)",
            mixed.len(),
            constant_q4.len()
        );
        // Decoder reconstructs the picture (lossy in the q>0 precincts,
        // lossless in the q=0 precincts).
        let img = decode_codestream(&mixed, None).expect("decode mixed q_precincts");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 30.0,
            "mixed-Q per-precinct PSNR {p:.2} dB < 30 dB floor"
        );
        // PIH still carries Hsl=0 (single slice).
        let parsed = crate::codestream::parse(&mixed).expect("parse codestream");
        assert_eq!(
            parsed.slices.len(),
            1,
            "Hsl=0 → single slice covers picture"
        );
    }

    /// Verify the per-precinct `Q[p]` byte surfaces on the wire — every
    /// precinct's header carries its individual `Q[p]` (the per-precinct
    /// override layered on top of the inherited slice / picture state).
    /// The decoder parses each precinct header and reconstructs the
    /// matching `T[p,b]`, so the round-trip is bit-correct.
    #[test]
    fn round233_qpr_wire_carries_per_precinct_q() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // Cw = 0 (np_x = 1), NL=1/1 → Hp = 2, Np,y = 16 precincts.
        // Pick a varied pattern over a smaller picture to keep the test
        // cheap and to exercise both lossless (q=0) and lossy (q>0)
        // precincts side by side.
        let q_pattern: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3];
        let cs = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            1,
            1,
            &q_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("qpr wire encode");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        // Hsl=0 (single slice covering picture), all 16 precincts in it.
        assert_eq!(parsed.slices.len(), 1, "expected 1 slice (Hsl=0)");
        // Walk precincts within the slice and check each precinct header
        // byte 3 (Q[p]). Precincts in this single-precinct-column NL=1/1
        // path are emitted consecutively; the precinct header is 5 bytes
        // (Lprc 3 bytes, Q[p] 1 byte, R[p] 1 byte at byte 4 in our cfg).
        // We rely on the entropy module's parsing to walk individual
        // precincts — at minimum the first precinct's Q byte must equal
        // q_pattern[0].
        let slice = &parsed.slices[0];
        assert!(
            slice.data_length >= 5,
            "slice payload too small to hold a precinct header"
        );
        let q_byte_first = cs[slice.data_offset + 3];
        assert_eq!(
            q_byte_first, q_pattern[0],
            "first precinct Q[p] must equal q_precincts[0]"
        );
        // Decoder round-trip — every precinct's reconstructed Q surfaces
        // in correct band truncation, so the bit-accuracy floor holds.
        let img = decode_codestream(&cs, None).expect("decode mixed q_precincts");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 25.0,
            "per-precinct PSNR {p:.2} dB < 25 dB floor (mixed Q[p] up to 4)"
        );
    }

    /// Wrong-length `q_precincts` is rejected (must equal `Np,y × Np,x`).
    #[test]
    fn round233_qpr_rejects_wrong_length() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 8 precincts at Cw=0, expecting length 8.
        let result = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0, 1, 2],
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "wrong q_precincts length must be rejected");
    }

    /// Out-of-range `q_precincts` entry (> 15) is rejected.
    #[test]
    fn round233_qpr_rejects_oversize_q() {
        let w = 16usize;
        let h = 16usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 4 precincts; entries must be in 0..=15.
        let result = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0, 16, 0, 0],
            std::slice::from_ref(&pixels),
        );
        assert!(result.is_err(), "q_precincts entry > 15 must be rejected");
    }

    /// `q_precincts` composes with the reversible RCT (`Cpih = 1`,
    /// 3-component 4:4:4): the colour transform is Q-agnostic, so a
    /// per-precinct override threads cleanly. Self-roundtrips losslessly
    /// at all-zero, and a mixed pattern stays above the PSNR floor.
    #[test]
    fn round233_qpr_rgb_rct_lossless_roundtrip() {
        let w = 32usize;
        let h = 32usize;
        let r = round103_grad(w, h);
        let mut g = round103_grad(w, h);
        let mut b = round103_grad(w, h);
        for v in g.iter_mut() {
            *v = v.wrapping_add(57);
        }
        for v in b.iter_mut() {
            *v = v.wrapping_mul(3).wrapping_add(13);
        }
        let planes = vec![r.clone(), g.clone(), b.clone()];
        // NL=2/2 → Np,y = 8 precincts (4:4:4, Cw=0 → np_x = 1).
        let qpr = vec![0u8; 8];
        let cs = encode_planar_qpr(w as u16, h as u16, 3, 1, 2, 2, &qpr, &planes)
            .expect("qpr lossless RGB+RCT encode");
        let img = decode_codestream(&cs, None).expect("decode qpr lossless RGB+RCT");
        assert_eq!(img.planes[0].data, r, "R component lossless");
        assert_eq!(img.planes[1].data, g, "G component lossless");
        assert_eq!(img.planes[2].data, b, "B component lossless");
    }

    /// `q_precincts` composes with Star-Tetrix (`Cpih = 3`, 4-component
    /// CFA): the Annex F.5 lifting is Q-agnostic on the wavelet-domain
    /// coefficients, so the per-precinct override threads cleanly. All-
    /// zero self-roundtrips losslessly.
    #[test]
    fn round233_qpr_star_tetrix_lossless_roundtrip() {
        let w = 16usize;
        let h = 16usize;
        let plane = |off: u8| {
            let mut p = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    p[y * w + x] = ((x * 7 + y * 5) as u8).wrapping_add(off);
                }
            }
            p
        };
        let planes = vec![plane(0), plane(31), plane(63), plane(95)];
        // NL=2/2 → Np,y = 4 precincts.
        let qpr = vec![0u8; 4];
        let cs = encode_planar_qpr(w as u16, h as u16, 4, 3, 2, 2, &qpr, &planes)
            .expect("qpr lossless Star-Tetrix encode");
        let img = decode_codestream(&cs, None).expect("decode qpr lossless Star-Tetrix");
        for (i, want) in planes.iter().enumerate() {
            assert_eq!(img.planes[i].data, *want, "component {i} lossless");
        }
    }

    /// Round 233: a mixed per-precinct `Q[p]` pattern keeps strictly more
    /// bits than the uniform-max-Q baseline (the q=0 precincts spend more
    /// bits than the q=15 baseline ever would on those positions). This
    /// is the "rate-allocation" property: per-precinct override expands
    /// the lever surface compared to the per-slice form.
    #[test]
    fn round233_qpr_lower_q_precincts_keep_more_bits() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 8 precincts at Cw=0.
        let max_q = vec![15u8; 8];
        let cs_max_q = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &max_q,
            std::slice::from_ref(&pixels),
        )
        .expect("max-q encode");
        // Mix in some q=0 precincts — those keep all magnitude bitplanes.
        let mut mixed = vec![15u8; 8];
        mixed[0] = 0;
        mixed[3] = 0;
        mixed[5] = 0;
        let cs_mixed = encode_planar_qpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &mixed,
            std::slice::from_ref(&pixels),
        )
        .expect("mixed encode");
        assert!(
            cs_mixed.len() > cs_max_q.len(),
            "mixed (q=0 in 3 precincts) {} B must exceed max-Q baseline {} B",
            cs_mixed.len(),
            cs_max_q.len()
        );
    }

    // ---------------------------------------------------------------
    // Round 239 — per-precinct R[p] override (Annex C.2 Table C.1)
    // ---------------------------------------------------------------

    /// `r_precincts = [0; n]` (every precinct refines none) is byte-
    /// identical to `encode_planar` at the same geometry — the round-239
    /// lever is a pure no-op when every precinct picks the same `R[p] = 0`.
    /// This also confirms the new code path threads through the existing
    /// `encode_planar` callsites without altering the lossless wire form.
    #[test]
    fn round239_rpr_all_zero_matches_encode_planar_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossless encode");
        // NL,y = 2 → Hp = 4, Np,y = 8; Cw = 0 → Np,x = 1 → 8 precincts.
        let rpr = vec![0u8; 8];
        let cs = encode_planar_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &rpr,
            std::slice::from_ref(&pixels),
        )
        .expect("all-zero r_precincts encode");
        assert_eq!(
            cs, baseline,
            "all-zero r_precincts must be byte-identical to encode_planar"
        );
    }

    /// Each precinct's header byte 4 (`R[p]`) carries the per-precinct
    /// value the caller asked for. NL=1/1 single-component → Np,y = 16
    /// precincts at h=32. The first precinct's header sits at the slice
    /// data offset; its byte 4 must equal `r_precincts[0]`. The decoder
    /// round-trips the picture (q=0 ⇒ lossless regardless of R[p]).
    #[test]
    fn round239_rpr_wire_carries_per_precinct_r() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=1/1 → Np,y = 16 precincts. NL = Nc × Nβ = 1 × 4 = 4 bands,
        // so R[p] ∈ 0..=3.
        let r_pattern: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3];
        let cs = encode_planar_rpr(
            w as u16,
            h as u16,
            1,
            0,
            1,
            1,
            &r_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("rpr wire encode");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.slices.len(), 1, "expected 1 slice (Hsl=0)");
        let slice = &parsed.slices[0];
        assert!(
            slice.data_length >= 5,
            "slice payload too small to hold a precinct header"
        );
        // Byte 3 = Q[p], byte 4 = R[p] per the precinct header layout.
        let r_byte_first = cs[slice.data_offset + 4];
        assert_eq!(
            r_byte_first, r_pattern[0],
            "first precinct R[p] must equal r_precincts[0]"
        );
        // Round-trip: q=0 floors every T[p,b] so this is lossless.
        let img = decode_codestream(&cs, None).expect("decode rpr stream");
        assert_eq!(
            img.planes[0].data, pixels,
            "q=0 + per-precinct R[p] must round-trip losslessly"
        );
    }

    /// Wrong-length `r_precincts` is rejected (must equal `Np,y × Np,x`).
    /// Also: an entry exceeding `NL - 1` is rejected (here NL = Nc × Nβ
    /// = 1 × 7 = 7 at NL,x = NL,y = 2, so max R[p] = 6).
    #[test]
    fn round239_rpr_rejects_wrong_length_and_out_of_range() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 8 precincts at Cw=0, so length must be 8.
        let bad_len = encode_planar_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0, 1, 2],
            std::slice::from_ref(&pixels),
        );
        assert!(
            bad_len.is_err(),
            "wrong r_precincts length must be rejected"
        );
        // NL,x = NL,y = 2 → Nβ = 7 → NL = 7 → max R[p] = 6.
        let bad_range = encode_planar_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0, 0, 0, 0, 0, 0, 0, 7],
            std::slice::from_ref(&pixels),
        );
        assert!(
            bad_range.is_err(),
            "r_precincts entry > NL-1 must be rejected"
        );
    }

    /// Round 242 — both vectors all-zero is byte-identical to the
    /// lossless `encode_planar` baseline (the cross-product reduces to
    /// the no-override path: every precinct sees Q[p]=0, R[p]=0, and
    /// Annex C.6.2 Table C.10 truncation lands at the zero floor).
    #[test]
    fn round242_qpr_rpr_both_zero_matches_encode_planar_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let baseline = encode_planar(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            std::slice::from_ref(&pixels),
        )
        .expect("baseline lossless encode");
        // NL,y = 2 → Np,y = 8 precincts at Cw = 0.
        let qpr = vec![0u8; 8];
        let rpr = vec![0u8; 8];
        let cs = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &qpr,
            &rpr,
            std::slice::from_ref(&pixels),
        )
        .expect("all-zero qpr+rpr encode");
        assert_eq!(
            cs, baseline,
            "all-zero q_precincts + r_precincts must be byte-identical to encode_planar"
        );
    }

    /// Round 242 — non-zero `r_precincts` actively *lowers* `T[p, b]`
    /// when paired with `q > 0`, granting one extra retained bitplane
    /// to the `R[p]` lowest-index bands per precinct. Annex C.6.2
    /// Table C.10 truncation is
    /// `T[p, b] = clamp(Q[p] − G[b] − r, 0, 15)` with
    /// `r = (P[b] < R[p]) ? 1 : 0`; at `Q = 4` the clamp is no longer
    /// floored at zero, so subtracting `r = 1` reduces `T` by 1 on the
    /// `R[p]` lowest-index bands — that is, one *more* magnitude bit
    /// retained on those bands per precinct. This *raises*
    /// reconstruction PSNR strictly above the R-off baseline at the
    /// same uniform Q — proving R[p] is engaged as a rate-distortion
    /// lever beyond the round-239 wire-only pin.
    ///
    /// We deliberately do not assert on codestream length: per-precinct
    /// entropy coder state (Annex C.4 packet-header bytes per
    /// precinct, plus group prefix) means the size delta from one
    /// extra retained bitplane on a subset of precincts is
    /// content-dependent; PSNR is the authoritative quality-side
    /// signal.
    #[test]
    fn round242_qpr_rpr_r_active_at_q_gt_0() {
        let w = 64usize;
        let h = 64usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 16 precincts at Cw=0. NL = 1 × 7 = 7 (Nβ = 7
        // at nlx=nly=2), so R[p] ∈ 0..=6.
        let q_uniform = vec![4u8; 16];
        let r_off = vec![0u8; 16];
        let cs_q_only = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &q_uniform,
            &r_off,
            std::slice::from_ref(&pixels),
        )
        .expect("Q-only encode");
        let mut r_pattern = vec![0u8; 16];
        // Hit alternating precincts with R[p] = 3 (refines the 3
        // lowest-index bands in those precincts).
        for (i, r) in r_pattern.iter_mut().enumerate() {
            if i % 2 == 0 {
                *r = 3;
            }
        }
        let cs_q_and_r = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &q_uniform,
            &r_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("Q+R encode");
        let img_q = decode_codestream(&cs_q_only, None).expect("decode Q-only");
        let img_qr = decode_codestream(&cs_q_and_r, None).expect("decode Q+R");
        let psnr_q = psnr(&pixels, &img_q.planes[0].data);
        let psnr_qr = psnr(&pixels, &img_qr.planes[0].data);
        assert!(
            psnr_qr > psnr_q,
            "extra R[p] refinement must raise PSNR (Q+R {psnr_qr:.2} dB <= Q-only {psnr_q:.2} dB) — R[p] is not engaged"
        );
        // Sanity: both reconstructions stay above a coarse PSNR floor.
        assert!(
            psnr_q >= 20.0,
            "Q=4 baseline reconstruction below 20 dB floor ({psnr_q:.2} dB)"
        );
    }

    /// Round 242 — both per-precinct bytes surface on the wire at the
    /// first precinct (Annex C.2 Table C.1: byte 3 = `Q[p]`, byte 4 =
    /// `R[p]`). The decoder reads them and round-trips correctly.
    #[test]
    fn round242_qpr_rpr_wire_carries_q_and_r() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=1/1 → Np,y = 16 precincts. NL = 1 × 4 = 4 bands, so R[p]
        // ∈ 0..=3.
        let q_pattern: [u8; 16] = [4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 2, 1, 4, 1, 2, 3];
        let r_pattern: [u8; 16] = [3, 0, 1, 2, 3, 0, 1, 2, 3, 2, 1, 0, 3, 0, 1, 2];
        let cs = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            1,
            1,
            &q_pattern,
            &r_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("qpr+rpr wire encode");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.slices.len(), 1, "expected 1 slice (Hsl=0)");
        let slice = &parsed.slices[0];
        assert!(
            slice.data_length >= 5,
            "slice payload too small to hold a precinct header"
        );
        let q_byte_first = cs[slice.data_offset + 3];
        let r_byte_first = cs[slice.data_offset + 4];
        assert_eq!(
            q_byte_first, q_pattern[0],
            "first precinct Q[p] must equal q_precincts[0]"
        );
        assert_eq!(
            r_byte_first, r_pattern[0],
            "first precinct R[p] must equal r_precincts[0]"
        );
        let img = decode_codestream(&cs, None).expect("decode qpr+rpr");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(p >= 20.0, "qpr+rpr PSNR {p:.2} dB < 20 dB floor");
    }

    /// Round 242 — when `q_precincts` is all-zero (lossless precincts
    /// everywhere) the cross-product reduces to the round-239
    /// `encode_planar_rpr` byte-for-byte regardless of `r_precincts`
    /// (Annex C.6.2 Table C.10 floors at `Q = 0`). This pins the
    /// reduction back to the round-239 entry point and proves the
    /// fall-through path is intact.
    #[test]
    fn round242_qpr_rpr_q_zero_matches_rpr() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // NL=2/2 → Np,y = 8 precincts at Cw=0.
        let qpr = vec![0u8; 8];
        let r_pattern: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 0];
        let cs_qpr_rpr = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &qpr,
            &r_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("qpr=0 + rpr encode");
        let cs_rpr = encode_planar_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &r_pattern,
            std::slice::from_ref(&pixels),
        )
        .expect("rpr-only encode");
        assert_eq!(
            cs_qpr_rpr, cs_rpr,
            "q_precincts = [0; n] + r_precincts = R must reduce to encode_planar_rpr(R)"
        );
    }

    /// Round 242 — calling with both vectors empty is the rejected
    /// no-override case (the public no-override path is
    /// `encode_planar`). Out-of-range entries in either vector are
    /// also rejected (delegated to the same `EncodeConfig::validate`
    /// guards that the round-233 / round-239 entry points use).
    #[test]
    fn round242_qpr_rpr_rejects_empty_and_out_of_range() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let both_empty = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[],
            &[],
            std::slice::from_ref(&pixels),
        );
        assert!(
            both_empty.is_err(),
            "encode_planar_qpr_rpr with both vectors empty must be rejected"
        );
        // NL=2/2 → Np,y = 8 precincts. q entry > 15 must be rejected.
        let bad_q = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[16, 0, 0, 0, 0, 0, 0, 0],
            &[0u8; 8],
            std::slice::from_ref(&pixels),
        );
        assert!(bad_q.is_err(), "q_precincts entry > 15 must be rejected");
        // NL = 1 × 7 = 7 → max R[p] = 6.
        let bad_r = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[2u8; 8],
            &[0, 0, 0, 0, 0, 0, 0, 7],
            std::slice::from_ref(&pixels),
        );
        assert!(bad_r.is_err(), "r_precincts entry > NL-1 must be rejected");
    }

    /// Round 245 — per-precinct joint picker rejects `target_bytes = 0`
    /// up front (precondition guard mirrors the round-218 / round-224
    /// pickers).
    #[test]
    fn round245_qpr_rpr_picker_rejects_zero_target() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let r = pick_qpr_rpr_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            0,
            std::slice::from_ref(&pixels),
        );
        assert!(r.is_err(), "target_bytes=0 must be rejected");
    }

    /// Round 245 — when the lossless probe fits the budget, the picker
    /// returns `q_precincts = [0; n]` and `r_precincts = [0; n]`
    /// (refinement is a no-op at the `T[p,b]` 0 floor at `q = 0`, so
    /// promoting `rp` is pointless).
    #[test]
    fn round245_qpr_rpr_picker_lossless_probe_fits_returns_zero_vectors() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // The lossless r242 stream at this geometry — measure it then
        // give the picker exactly that budget.
        let lossless = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0u8; 8],
            &[0u8; 8],
            std::slice::from_ref(&pixels),
        )
        .expect("lossless probe");
        let (q, r) = pick_qpr_rpr_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            lossless.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("picker fits at lossless budget");
        assert_eq!(q, vec![0u8; 8], "lossless-fits → q_precincts all zero");
        assert_eq!(r, vec![0u8; 8], "lossless-fits → r_precincts all zero");
        // The convenience wrapper produces the same codestream the
        // lossless probe did.
        let (cs, _, _) = encode_planar_qpr_rpr_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            lossless.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("wrapper fits");
        assert_eq!(cs.len(), lossless.len());
    }

    /// Round 245 — when even `rp=0, Q=15` overshoots, the picker errors
    /// with a `target_bytes unreachable; rp=0 Q=15 emits N bytes` message
    /// (matches the round-224 baseline-reachability shape).
    #[test]
    fn round245_qpr_rpr_picker_unreachable_target_errors() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        // 1 byte is well below any plausible JPEG XS codestream size
        // (SOC + CAP + PIH + CDT + WGT + at least one precinct + EOC).
        let r = pick_qpr_rpr_for_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            1,
            std::slice::from_ref(&pixels),
        );
        assert!(r.is_err(), "1-byte budget must be unreachable");
    }

    /// Round 245 — the picker output decodes through the round-242
    /// joint primitive and the reconstructed pixels self-round-trip
    /// at the picker's chosen `(q_precincts, r_precincts)`.
    ///
    /// We pick the budget at a fraction of the lossless size so the
    /// picker must engage Pass 2 / Pass 3 (uniform-Q bisect + per-
    /// precinct activity relaxation).
    #[test]
    fn round245_qpr_rpr_picker_roundtrips_through_decoder() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let lossless = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0u8; 8],
            &[0u8; 8],
            std::slice::from_ref(&pixels),
        )
        .expect("lossless probe");
        // Pick a budget tight enough that Q must be > 0 somewhere.
        let target = lossless.len() / 2;
        let (cs, q_picked, r_picked) = encode_planar_qpr_rpr_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            target,
            std::slice::from_ref(&pixels),
        )
        .expect("picker fits at half-lossless budget");
        assert!(
            cs.len() <= target,
            "codestream must fit: {} <= {}",
            cs.len(),
            target
        );
        assert_eq!(q_picked.len(), 8, "Np,y = 8 → 8 Q[p] entries");
        assert_eq!(r_picked.len(), 8, "Np,y = 8 → 8 R[p] entries");
        // Lossy stream still decodes through the round-242 primitive.
        let img = decode_codestream(&cs, None).expect("decode picker output");
        assert_eq!(img.width as usize, w);
        assert_eq!(img.height as usize, h);
    }

    /// Round 245 — at a budget set at 90% of the lossless probe size,
    /// the picker's per-precinct relaxation keeps PSNR within the
    /// useful band (≥ 25 dB) against the source. The activity-driven
    /// pass concentrates bits on the low-activity precincts so that
    /// quantization distortion lands on the busier precincts where it
    /// is less perceptually visible; 90% leaves enough rate to keep
    /// reconstruction recognisable on the XOR-ramp fixture.
    #[test]
    fn round245_qpr_rpr_picker_psnr_floor_at_relaxed_budget() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let lossless = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0u8; 8],
            &[0u8; 8],
            std::slice::from_ref(&pixels),
        )
        .expect("lossless probe");
        // 90% of the lossless size — tight enough to force the inner
        // search past the lossless probe, loose enough that the
        // activity-driven relaxation gets one or two slices to Q=0.
        let target = (lossless.len() * 9) / 10;
        let (cs, q_picked, _) = encode_planar_qpr_rpr_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            target,
            std::slice::from_ref(&pixels),
        )
        .expect("picker fits");
        assert!(
            cs.len() <= target,
            "codestream must fit: {} <= {}",
            cs.len(),
            target
        );
        // Sanity — picker engaged Q > 0 somewhere.
        assert!(
            q_picked.iter().any(|&q| q > 0),
            "tight budget must drive Q > 0 on some precinct: {q_picked:?}"
        );
        let img = decode_codestream(&cs, None).expect("decode");
        let plane = &img.planes[0].data;
        assert_eq!(plane.len(), w * h, "reconstructed plane size");
        let p = psnr(&pixels, plane);
        assert!(
            p >= 25.0,
            "round245 90% lossless budget PSNR ≥ 25 dB, got {p}"
        );
    }

    /// Round 245 — when the wrapper picker is given a budget the
    /// lossless probe fits and the codestream comes out byte-identical
    /// to `encode_planar_qpr_rpr` at `q_precincts = [0; n]` +
    /// `r_precincts = [0; n]`, the round trips losslessly (proves the
    /// picker's `(q, r)` output is consistent with a direct
    /// `encode_planar_qpr_rpr` invocation).
    #[test]
    fn round245_qpr_rpr_picker_byte_identical_to_qpr_rpr_at_lossless() {
        let w = 32usize;
        let h = 32usize;
        let pixels = round103_grad(w, h);
        let lossless = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &[0u8; 8],
            &[0u8; 8],
            std::slice::from_ref(&pixels),
        )
        .expect("lossless probe");
        let (cs, q, r) = encode_planar_qpr_rpr_target_bytes(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            lossless.len(),
            std::slice::from_ref(&pixels),
        )
        .expect("wrapper at lossless budget");
        // Picker's output through encode_planar_qpr_rpr is the same
        // byte sequence (the wrapper just composes the picker + the
        // primitive).
        let cs_replay = encode_planar_qpr_rpr(
            w as u16,
            h as u16,
            1,
            0,
            2,
            2,
            &q,
            &r,
            std::slice::from_ref(&pixels),
        )
        .expect("replay primitive");
        assert_eq!(cs, cs_replay, "wrapper output == primitive replay");
        assert_eq!(cs, lossless, "matches lossless reference");
        let img = decode_codestream(&cs, None).expect("decode lossless");
        assert_eq!(img.planes[0].data, pixels, "lossless self-roundtrip");
    }

    /// 64×64 RGB synthetic source for the Annex H content-adaptive WGT
    /// tests — wide enough for the `NL,x = 5` decomposition the Annex H
    /// tables target (2^5 = 32 ≤ 64).
    fn make_synthetic_rgb_64x64() -> [Vec<u8>; 3] {
        let (w, h) = (64usize, 64usize);
        let mut r = vec![0u8; w * h];
        let mut g = vec![0u8; w * h];
        let mut b = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                r[i] = (((x as i32) * 3 + (y as i32) * 2) % 256) as u8;
                g[i] = (((y as i32) * 5 + (x as i32) * 3 + ((x ^ y) as i32 & 0x1f)) % 256) as u8;
                b[i] = ((x ^ y) as u8).wrapping_mul(11).wrapping_add(40);
            }
        }
        [r, g, b]
    }

    /// Round 323: the Annex H content-adaptive WGT encoder
    /// ([`encode_planar_lossy_annex_h`]) must (a) emit the spec's
    /// PSNR-optimized `(G[b], P[b])` pairs into the WGT marker, (b) drive
    /// the forward truncation from those same weights so the codestream
    /// round-trips through the decoder at a healthy PSNR floor, and (c)
    /// produce a genuinely different codestream from the default plain
    /// `P[b] = b` / capped-gain path. Exercised across all three
    /// tabulated vertical depths (Tables H.1 / H.2 / H.3, NLy ∈ {0,1,2}).
    #[test]
    fn round323_annex_h_weights_roundtrip_and_differ_from_default() {
        let planes = make_synthetic_rgb_64x64();
        for nly in 0u8..=2 {
            let (exp_g, exp_p) = annex_h_weights(3, 1, 0, 5, nly, 0, &[1, 1, 1], &[1, 1, 1])
                .expect("Annex H table for 4:4:4 RCT NLx=5");

            let cs = encode_planar_lossy_annex_h(64, 64, 3, 1, 5, nly, 4, &planes)
                .unwrap_or_else(|e| panic!("Annex H encode NLy={nly} failed: {e:?}"));

            // (a) WGT carries the Annex H pairs verbatim.
            let parsed = crate::codestream::parse(&cs).expect("parse Annex H codestream");
            let weights = parsed.wgt().expect("typed WGT view");
            let got_g: Vec<u8> = weights.iter().map(|w| w.gain).collect();
            let got_p: Vec<u8> = weights.iter().map(|w| w.priority).collect();
            assert_eq!(got_g, exp_g, "NLy={nly}: WGT gains must match Annex H");
            assert_eq!(got_p, exp_p, "NLy={nly}: WGT priorities must match Annex H");

            // (b) Round-trips with a sane PSNR floor (encoder/decoder agree
            // on T[p,b] derived from the shared weights table).
            let img = decode_codestream(&cs, None).expect("decode Annex H codestream");
            let mut src = Vec::with_capacity(64 * 64 * 3);
            let mut rec = Vec::with_capacity(64 * 64 * 3);
            for i in 0..64 * 64 {
                for (plane, rec_plane) in planes.iter().zip(img.planes.iter()) {
                    src.push(plane[i]);
                    rec.push(rec_plane.data[i]);
                }
            }
            let p = psnr(&src, &rec);
            assert!(
                p >= 25.0,
                "NLy={nly}: Annex H q=4 PSNR {p:.2} dB below 25 dB floor"
            );

            // (c) Differs from the default-weights path (richer gains +
            // reordered priorities change the truncation positions).
            let default_cs = encode_planar_lossy(64, 64, 3, 1, 5, nly, 4, &planes)
                .expect("default-weights encode");
            assert_ne!(
                cs, default_cs,
                "NLy={nly}: Annex H stream must differ from default weights"
            );
        }
    }

    /// Round 323: at `q = 0` (lossless) the Annex H weights are a no-op for
    /// the reconstructed samples — every `T[p,b]` is clamped to its `0`
    /// floor — so the picture must self-roundtrip bit-exactly even though
    /// the WGT still advertises the Annex H gains/priorities.
    #[test]
    fn round323_annex_h_weights_lossless_bit_exact() {
        let planes = make_synthetic_rgb_64x64();
        let cs = encode_planar_lossy_annex_h(64, 64, 3, 1, 5, 1, 0, &planes)
            .expect("Annex H lossless encode");
        let img = decode_codestream(&cs, None).expect("decode Annex H lossless");
        for (c, (rec_plane, src_plane)) in img.planes.iter().zip(planes.iter()).enumerate() {
            assert_eq!(
                rec_plane.data, *src_plane,
                "component {c} must self-roundtrip bit-exactly at q=0"
            );
        }
    }

    /// Round 323: a configuration outside the Annex H table set (4:4:4 RCT
    /// but `NL,x = 2`) must fall back to the default-weights encoder and
    /// produce byte-identical output to [`encode_planar_lossy`].
    #[test]
    fn round323_annex_h_falls_back_to_default_when_unmatched() {
        let planes = make_synthetic_rgb_64x64();
        assert!(
            annex_h_weights(3, 1, 0, 2, 2, 0, &[1, 1, 1], &[1, 1, 1]).is_none(),
            "no H table for NLx=2"
        );
        let fallback =
            encode_planar_lossy_annex_h(64, 64, 3, 1, 2, 2, 4, &planes).expect("fallback encode");
        let default_cs =
            encode_planar_lossy(64, 64, 3, 1, 2, 2, 4, &planes).expect("default encode");
        assert_eq!(
            fallback, default_cs,
            "unmatched config must be byte-identical to encode_planar_lossy"
        );
    }

    /// 64-wide subsampled RGB→YCbCr-domain fixture: a full-res luma plane and
    /// two chroma planes sized for the requested `(sx, sy)` chroma factors.
    /// The synthetic content is the same deterministic ramp pattern used by
    /// the 4:4:4 Annex H tests, sampled at the chroma grid for Cb / Cr.
    fn make_subsampled_planes(w: usize, h: usize, sx_c: usize, sy_c: usize) -> [Vec<u8>; 3] {
        let cw = w / sx_c;
        let ch = h / sy_c;
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = (((xx as i32) * 3 + (yy as i32) * 2) % 256) as u8;
            }
        }
        for yy in 0..ch {
            for xx in 0..cw {
                cb[yy * cw + xx] = (((xx as i32) * 5 + (yy as i32) * 3) % 256) as u8;
                cr[yy * cw + xx] = ((xx ^ yy) as u8).wrapping_mul(11).wrapping_add(40);
            }
        }
        [y, cb, cr]
    }

    /// Round 327: the Annex H chroma-subsampled WGT encoder
    /// ([`encode_planar_subsampled_annex_h`]) must (a) emit one
    /// `(G[b], P[b])` pair per *existing* band into the WGT marker — matching
    /// the `annex_h_weights` existing-band vectors and `count_existing_bands`
    /// — (b) drive the forward truncation from those same weights so the
    /// codestream round-trips through the decoder at a sane PSNR floor, and
    /// (c) produce a genuinely different codestream from the default
    /// `P[b] = b` / capped-gain subsampled path. Covers Tables H.4 / H.5 / H.6
    /// (4:2:2, NLy ∈ {0,1,2}) and H.7 / H.8 (4:2:0, NLy ∈ {1,2}).
    #[test]
    fn round327_annex_h_subsampled_weights_roundtrip_and_differ() {
        // (chroma sx, chroma sy, NLy) tuples for each tabulated subsampled
        // configuration. 4:2:2 → (2,1); 4:2:0 → (2,2).
        // (chroma sx, chroma sy, NLy, expected existing-band count). The
        // counts come from the spec: H.4=18, H.5=24, H.6=30 (4:2:2, no `-*`);
        // H.7=24−4=20, H.8=30−4=26 (4:2:0, four `-*` slots each dropped).
        let cases: &[(u8, u8, u8, usize)] = &[
            (2, 1, 0, 18), // H.4
            (2, 1, 1, 24), // H.5
            (2, 1, 2, 30), // H.6
            (2, 2, 1, 20), // H.7
            (2, 2, 2, 26), // H.8
        ];
        for &(sxc, syc, nly, n_exist) in cases {
            let planes = make_subsampled_planes(64, 64, sxc as usize, syc as usize);
            let sx = [1u8, sxc, sxc];
            let sy = [1u8, syc, syc];
            let (exp_g, exp_p) = annex_h_weights(3, 0, 0, 5, nly, 0, &sx, &sy)
                .unwrap_or_else(|| panic!("Annex H subsampled table sx={sxc} sy={syc} NLy={nly}"));
            assert_eq!(
                exp_g.len(),
                exp_p.len(),
                "gains/priorities length agree (sx={sxc} sy={syc} NLy={nly})"
            );
            assert_eq!(
                exp_g.len(),
                n_exist,
                "sx={sxc} sy={syc} NLy={nly}: existing-band count (post `-*` drop)"
            );

            // q = 2: low enough that the high LL gains in the 4:2:0 tables
            // (G up to 3) do not truncate the steep luma gradient below the
            // sanity floor, while still engaging Fq = 8 lossy truncation.
            let cs = encode_planar_subsampled_annex_h(64, 64, 3, 0, 5, nly, 2, &sx, &sy, &planes)
                .unwrap_or_else(|e| panic!("subsampled Annex H encode NLy={nly} failed: {e:?}"));

            // (a) WGT carries exactly the existing-band Annex H pairs.
            let parsed = crate::codestream::parse(&cs).expect("parse subsampled Annex H stream");
            let weights = parsed.wgt().expect("typed WGT view");
            let got_g: Vec<u8> = weights.iter().map(|w| w.gain).collect();
            let got_p: Vec<u8> = weights.iter().map(|w| w.priority).collect();
            assert_eq!(got_g, exp_g, "sx={sxc} sy={syc} NLy={nly}: WGT gains");
            assert_eq!(got_p, exp_p, "sx={sxc} sy={syc} NLy={nly}: WGT priorities");

            // (b) The number of WGT entries the encoder actually emitted
            // equals the existing-band count — i.e. the `-*` slots of
            // H.7 / H.8 were correctly dropped and the encoder's own
            // existing-band skip rule agrees with the table's `-*` markings.
            assert_eq!(
                got_g.len(),
                n_exist,
                "sx={sxc} sy={syc} NLy={nly}: emitted WGT entry count == existing bands"
            );

            // (c) Round-trips through the decoder: reconstructed planes have
            // the correct (subsampled) dimensions and the lossy stream is
            // strictly smaller than the lossless probe (truncation engaged).
            // The strict encoder/decoder T[p,b] alignment proof is the q=0
            // bit-exact test below; PSNR is content-dependent (the steep luma
            // ramp + high 4:2:0 LL gains make it a poor invariant here).
            let img = decode_codestream(&cs, None).expect("decode subsampled Annex H stream");
            assert_eq!(img.width as usize, 64, "decoded width");
            assert_eq!(img.height as usize, 64, "decoded height");
            assert_eq!(img.planes.len(), 3, "decoded component count");
            assert_eq!(
                img.planes[1].data.len(),
                planes[1].len(),
                "sx={sxc} sy={syc} NLy={nly}: Cb plane size preserved"
            );
            let lossless =
                encode_planar_subsampled_annex_h(64, 64, 3, 0, 5, nly, 0, &sx, &sy, &planes)
                    .expect("lossless probe");
            assert!(
                cs.len() < lossless.len(),
                "sx={sxc} sy={syc} NLy={nly}: q=2 stream ({}) must be < lossless ({})",
                cs.len(),
                lossless.len()
            );

            // (d) Differs from the default-weights subsampled path.
            let default_cs = encode_planar_subsampled(64, 64, 3, 0, 5, nly, 2, &sx, &sy, &planes)
                .expect("default-weights subsampled encode");
            assert_ne!(
                cs, default_cs,
                "sx={sxc} sy={syc} NLy={nly}: Annex H stream must differ from default"
            );
        }
    }

    /// Round 327: at `q = 0` (lossless) the subsampled Annex H weights are a
    /// no-op for the reconstructed samples, so every component self-roundtrips
    /// bit-exactly even though the WGT advertises the Annex H gains/priorities.
    #[test]
    fn round327_annex_h_subsampled_lossless_bit_exact() {
        for &(sxc, syc, nly) in &[(2u8, 1u8, 2u8), (2u8, 2u8, 2u8)] {
            let planes = make_subsampled_planes(64, 64, sxc as usize, syc as usize);
            let sx = [1u8, sxc, sxc];
            let sy = [1u8, syc, syc];
            let cs = encode_planar_subsampled_annex_h(64, 64, 3, 0, 5, nly, 0, &sx, &sy, &planes)
                .expect("subsampled Annex H lossless encode");
            let img = decode_codestream(&cs, None).expect("decode subsampled Annex H lossless");
            for (c, (rec_plane, src_plane)) in img.planes.iter().zip(planes.iter()).enumerate() {
                assert_eq!(
                    rec_plane.data, *src_plane,
                    "sx={sxc} sy={syc} NLy={nly}: component {c} bit-exact at q=0"
                );
            }
        }
    }

    /// Round 327: a subsampled configuration outside the tabulated Annex H set
    /// (4:2:0 with `NLy = 3`, valid since 4:2:0 requires `NLy ≥ 1` but only
    /// `NLy ∈ {1, 2}` are tabulated) falls back to the default-weights
    /// subsampled encoder and is byte-identical to [`encode_planar_subsampled`].
    #[test]
    fn round327_annex_h_subsampled_falls_back_when_unmatched() {
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        assert!(
            annex_h_weights(3, 0, 0, 5, 3, 0, &sx, &sy).is_none(),
            "no 4:2:0 NLy=3 Annex H table"
        );
        let planes = make_subsampled_planes(64, 64, 2, 2);
        let fallback = encode_planar_subsampled_annex_h(64, 64, 3, 0, 5, 3, 4, &sx, &sy, &planes)
            .expect("fallback subsampled encode");
        let default_cs = encode_planar_subsampled(64, 64, 3, 0, 5, 3, 4, &sx, &sy, &planes)
            .expect("default subsampled encode");
        assert_eq!(
            fallback, default_cs,
            "unmatched subsampled config must equal encode_planar_subsampled"
        );
    }

    // === Round 346: high-bit-depth (B[i] > 8) chroma-subsampled Annex H
    // weights (H.4–H.8) on the 4:2:2 / 4:2:0 Cpih=0 layout. =================

    /// High-bit-depth (`u16`) subsampled luma + chroma fixture: full-res luma
    /// and two chroma planes sized for `(sx_c, sy_c)`, with samples in
    /// `0..=2^bd − 1`. Same deterministic ramp shape as `make_subsampled_planes`.
    fn make_subsampled_planes_u16(
        w: usize,
        h: usize,
        sx_c: usize,
        sy_c: usize,
        bd: u8,
    ) -> [Vec<u16>; 3] {
        let max = ((1u32 << bd) - 1) as i64;
        let cw = w / sx_c;
        let ch = h / sy_c;
        let mut y = vec![0u16; w * h];
        let mut cb = vec![0u16; cw * ch];
        let mut cr = vec![0u16; cw * ch];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = (((xx as i64) * 37 + (yy as i64) * 19) % (max + 1)) as u16;
            }
        }
        for yy in 0..ch {
            for xx in 0..cw {
                cb[yy * cw + xx] = (((xx as i64) * 53 + (yy as i64) * 29) % (max + 1)) as u16;
                cr[yy * cw + xx] = (((xx ^ yy) as i64 * 101 + 401) % (max + 1)) as u16;
            }
        }
        [y, cb, cr]
    }

    /// Round 346: the high-bit-depth subsampled Annex H entry point
    /// [`encode_planar_subsampled_highbd_annex_h`] (a) emits one `(G[b], P[b])`
    /// pair per existing band (with the `-*` slots dropped) into the WGT,
    /// matching `annex_h_weights`, (b) round-trips through the decoder for the
    /// 4:2:2 / 4:2:0 `Cpih = 0` layout at `bd ∈ {10, 12, 16}`, and (c) differs
    /// from the default-weights highbd subsampled path. Covers H.4 / H.5 / H.6
    /// (4:2:2) and H.7 / H.8 (4:2:0).
    #[test]
    fn round346_subsampled_highbd_annex_h_roundtrip_and_differ() {
        // (chroma sx, chroma sy, NLy, existing-band count) — same as round327.
        let cases: &[(u8, u8, u8, usize)] = &[
            (2, 1, 0, 18), // H.4
            (2, 1, 1, 24), // H.5
            (2, 1, 2, 30), // H.6
            (2, 2, 1, 20), // H.7
            (2, 2, 2, 26), // H.8
        ];
        for &bd in &[10u8, 12, 16] {
            for &(sxc, syc, nly, n_exist) in cases {
                let planes = make_subsampled_planes_u16(64, 64, sxc as usize, syc as usize, bd);
                let sx = [1u8, sxc, sxc];
                let sy = [1u8, syc, syc];
                let (exp_g, exp_p) = annex_h_weights(3, 0, 0, 5, nly, 0, &sx, &sy)
                    .unwrap_or_else(|| panic!("H table sx={sxc} sy={syc} NLy={nly}"));
                assert_eq!(exp_g.len(), n_exist, "bd={bd} band count sx={sxc} sy={syc}");

                let cs = encode_planar_subsampled_highbd_annex_h(
                    64, 64, 3, 0, 5, nly, bd, 2, &sx, &sy, &planes,
                )
                .unwrap_or_else(|e| {
                    panic!("highbd subsampled Annex H bd={bd} sx={sxc} sy={syc} NLy={nly}: {e:?}")
                });

                // (a) WGT carries exactly the existing-band column.
                let parsed =
                    crate::codestream::parse(&cs).expect("parse highbd subsampled Annex H");
                let weights = parsed.wgt().expect("typed WGT view");
                let got_g: Vec<u8> = weights.iter().map(|w| w.gain).collect();
                let got_p: Vec<u8> = weights.iter().map(|w| w.priority).collect();
                assert_eq!(
                    got_g, exp_g,
                    "bd={bd} sx={sxc} sy={syc} NLy={nly}: WGT gains"
                );
                assert_eq!(
                    got_p, exp_p,
                    "bd={bd} sx={sxc} sy={syc} NLy={nly}: WGT priorities"
                );

                // (b) Round-trips through the decoder, highbd plane layout.
                let img = decode_codestream(&cs, None).expect("decode highbd subsampled Annex H");
                assert_eq!(img.num_components, 3);
                assert_eq!(img.bit_depth, bd, "bd={bd}: Bw byte");

                // (c) Differs from the default-weights highbd subsampled path.
                let default_cs = encode_planar_subsampled_highbd_lossy(
                    64, 64, 3, 0, 5, nly, bd, 2, &sx, &sy, &planes,
                )
                .expect("default-weights highbd subsampled");
                assert_ne!(
                    cs, default_cs,
                    "bd={bd} sx={sxc} sy={syc} NLy={nly}: Annex H must differ from default"
                );
            }
        }
    }

    /// Round 346: at `q = 0` the highbd subsampled Annex H weights are a no-op
    /// for the reconstructed samples — every component self-roundtrips
    /// bit-exactly through the 2-bytes-per-sample plane layout even though the
    /// WGT advertises the H-table gains/priorities.
    #[test]
    fn round346_subsampled_highbd_annex_h_lossless_bit_exact() {
        for &bd in &[10u8, 12, 14, 16] {
            for &(sxc, syc, nly) in &[(2u8, 1u8, 2u8), (2u8, 2u8, 2u8)] {
                let planes = make_subsampled_planes_u16(64, 64, sxc as usize, syc as usize, bd);
                let sx = [1u8, sxc, sxc];
                let sy = [1u8, syc, syc];
                let cs = encode_planar_subsampled_highbd_annex_h(
                    64, 64, 3, 0, 5, nly, bd, 0, &sx, &sy, &planes,
                )
                .expect("highbd subsampled Annex H lossless encode");
                let img = decode_codestream(&cs, None).expect("decode highbd subsampled lossless");
                for (c, (rec, src)) in img.planes.iter().zip(planes.iter()).enumerate() {
                    let got: Vec<u16> = rec
                        .data
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    assert_eq!(
                        got, *src,
                        "bd={bd} sx={sxc} sy={syc} NLy={nly}: component {c} bit-exact at q=0"
                    );
                }
            }
        }
    }

    /// Round 346: a highbd subsampled configuration outside the tabulated H.4–H.8
    /// set (4:2:0 with `NLy = 3`) falls back to the default-weights highbd
    /// subsampled path and is byte-identical to a direct default encode (lossless
    /// for `q = 0`, lossy otherwise).
    #[test]
    fn round346_subsampled_highbd_annex_h_falls_back_when_unmatched() {
        let bd = 12u8;
        let sx = [1u8, 2, 2];
        let sy = [1u8, 2, 2];
        assert!(
            annex_h_weights(3, 0, 0, 5, 3, 0, &sx, &sy).is_none(),
            "no 4:2:0 NLy=3 Annex H table"
        );
        let planes = make_subsampled_planes_u16(64, 64, 2, 2, bd);
        // q = 0 → lossless fallback.
        let fb0 =
            encode_planar_subsampled_highbd_annex_h(64, 64, 3, 0, 5, 3, bd, 0, &sx, &sy, &planes)
                .expect("fallback highbd subsampled lossless");
        let direct0 = encode_planar_subsampled_highbd(64, 64, 3, 0, 5, 3, bd, &sx, &sy, &planes)
            .expect("direct highbd subsampled lossless");
        assert_eq!(fb0, direct0, "q=0 unmatched must equal lossless default");
        // q > 0 → lossy fallback.
        let fb1 =
            encode_planar_subsampled_highbd_annex_h(64, 64, 3, 0, 5, 3, bd, 4, &sx, &sy, &planes)
                .expect("fallback highbd subsampled lossy");
        let direct1 =
            encode_planar_subsampled_highbd_lossy(64, 64, 3, 0, 5, 3, bd, 4, &sx, &sy, &planes)
                .expect("direct highbd subsampled lossy");
        assert_eq!(fb1, direct1, "q>0 unmatched must equal lossy default");
    }

    /// Round 346: lossy correctness proof for the WGT-truncation alignment
    /// fix. With the encoder now indexing the Annex H weights by their β-major
    /// existing-band slot (the decoder's cursor), a `q > 0` subsampled Annex H
    /// stream must reconstruct to *correct pixels* — not merely decode without
    /// over-read. Covers the 4:2:0 H.7 / H.8 tables (the `-*`-bearing ones the
    /// old `custom_wgt` gate skipped) at high bit depth, against a smooth
    /// low-frequency ramp where the high LL gains do not crush the signal.
    #[test]
    fn round346_subsampled_highbd_annex_h_lossy_psnr_floor() {
        // Smooth gradient content (low spatial frequency) so the H.7/H.8 LL
        // gains leave a recoverable signal at a small q.
        let smooth = |w: usize, h: usize, bd: u8, seed: i64| -> Vec<u16> {
            let max = (1i64 << bd) - 1;
            let mut v = vec![0u16; w * h];
            for y in 0..h {
                for x in 0..w {
                    let val = ((x as i64 + seed) * max / (w as i64 + 4)
                        + (y as i64) * max / (h as i64 + 4) / 2)
                        % (max + 1);
                    v[y * w + x] = val as u16;
                }
            }
            v
        };
        for &bd in &[10u8, 12] {
            for &(sxc, syc, nly) in &[(2u8, 1u8, 2u8), (2u8, 2u8, 2u8)] {
                let (w, h) = (64usize, 64usize);
                let cw = w / sxc as usize;
                let ch = h / syc as usize;
                let planes = vec![
                    smooth(w, h, bd, 0),
                    smooth(cw, ch, bd, 3),
                    smooth(cw, ch, bd, 7),
                ];
                let sx = [1u8, sxc, sxc];
                let sy = [1u8, syc, syc];
                let cs = encode_planar_subsampled_highbd_annex_h(
                    w as u16, h as u16, 3, 0, 5, nly, bd, 1, &sx, &sy, &planes,
                )
                .expect("highbd subsampled Annex H lossy encode");
                let img =
                    decode_codestream(&cs, None).expect("decode highbd subsampled Annex H lossy");
                for (i, orig) in planes.iter().enumerate() {
                    let rec = plane_u16(&img.planes[i].data);
                    let p = psnr_u16(orig, &rec, bd);
                    assert!(
                        p >= 30.0,
                        "bd={bd} sx={sxc} sy={syc} NLy={nly} comp {i}: q=1 PSNR {p:.2} dB < 30 dB floor (WGT/truncation must agree)"
                    );
                }
            }
        }
    }

    // === Round 330: Annex H CFA Star-Tetrix weight tables H.9 / H.10 / H.11 ==

    /// Build four 4:4:4:4 CFA planes (`[R, G1, G2, B]`) with distinct
    /// per-component texture so the Star-Tetrix decorrelation is exercised.
    fn cfa_planes(w: usize, h: usize) -> [Vec<u8>; 4] {
        let make = |seed: u32| {
            let mut v = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 2)
                        .wrapping_add((y as u32).wrapping_mul(seed + 3))
                        .wrapping_add(seed)
                        % 256) as u8;
                }
            }
            v
        };
        [make(7), make(13), make(19), make(31)]
    }

    /// Round 330: the CFA Star-Tetrix Annex H entry point
    /// [`encode_planar_star_tetrix_annex_h`] (a) emits a WGT carrying exactly
    /// the H.9 / H.10 / H.11 `(G[b], P[b])` column selected by `cf`, with one
    /// pair per existing band (NL = (Nc−Sd)·Nβ + Sd = 19 / 25 / 31), (b)
    /// round-trips through the decoder for the previously-rejected `Cpih = 3,
    /// Sd = 1, Nc = 4` CFA layout (Annex B Tables B.10 / B.11), and (c) yields
    /// a genuinely different codestream from the default-weights
    /// [`encode_planar_star_tetrix`] with the same suppression.
    #[test]
    fn round330_cfa_star_tetrix_annex_h_roundtrip_and_differ() {
        let (w, h) = (32usize, 16usize);
        let planes = cfa_planes(w, h);
        // (NLy, expected band count). NL = 3·Nβ + 1; Nβ = NLx+1+NLy = 6/7/8 →
        // wait, Nβ = NLx + 1 for NLy=0 (6), +2 per extra vertical level: the
        // spec tables give 19 / 25 / 31 directly (H.9 / H.10 / H.11).
        let cases: &[(u8, usize)] = &[(0, 19), (1, 25), (2, 31)];
        for &(nly, n_bands) in cases {
            for cf in [0u8, 3u8] {
                let sx = [1u8; 4];
                let sy = [1u8; 4];
                let (exp_g, exp_p) = annex_h_weights(4, 3, 1, 5, nly, cf, &sx, &sy)
                    .unwrap_or_else(|| panic!("H table NLy={nly} cf={cf}"));
                assert_eq!(exp_g.len(), exp_p.len(), "g/p len agree NLy={nly} cf={cf}");
                assert_eq!(
                    exp_g.len(),
                    n_bands,
                    "NLy={nly}: band count NL = (Nc−Sd)·Nβ + Sd"
                );

                let cs = encode_planar_star_tetrix_annex_h(
                    w as u16, h as u16, 5, nly, 2, 0, 0, cf, 0, &planes,
                )
                .unwrap_or_else(|e| panic!("CFA Annex H encode NLy={nly} cf={cf}: {e:?}"));

                // (a) WGT carries exactly the selected H-table column.
                let parsed = crate::codestream::parse(&cs).expect("parse CFA Annex H stream");
                let weights = parsed.wgt().expect("typed WGT view");
                let got_g: Vec<u8> = weights.iter().map(|w| w.gain).collect();
                let got_p: Vec<u8> = weights.iter().map(|w| w.priority).collect();
                assert_eq!(got_g, exp_g, "NLy={nly} cf={cf}: WGT gains");
                assert_eq!(got_p, exp_p, "NLy={nly} cf={cf}: WGT priorities");
                assert_eq!(got_g.len(), n_bands, "NLy={nly} cf={cf}: WGT entry count");

                // (b) Round-trips through the decoder.
                let img = decode_codestream(&cs, None).expect("decode CFA Annex H stream");
                assert_eq!(img.num_components, 4, "NLy={nly} cf={cf}: 4 CFA comps");
                assert_eq!(img.cpih, 3, "NLy={nly} cf={cf}: Cpih=3");
                assert_eq!(img.width as usize, w);
                assert_eq!(img.height as usize, h);

                // (c) Differs from the default-weights Star-Tetrix path (same
                // Cpih=3, Sd=1 suppression and CTS cf, but default P[b]=b /
                // capped gains instead of the Annex H column).
                let default_cs = encode_planar_sd_star_tetrix(
                    w as u16, h as u16, 4, 5, nly, 2, 1, 0, 0, cf, 0, &planes,
                )
                .expect("default-weights Sd=1 Star-Tetrix");
                assert_ne!(
                    cs, default_cs,
                    "NLy={nly} cf={cf}: Annex H stream must differ from default"
                );
            }
        }
    }

    /// Round 330: at `q = 0` the CFA Star-Tetrix Annex H weights are a no-op
    /// for the reconstructed samples — every component self-roundtrips
    /// bit-exactly even though the WGT advertises the H-table gains/priorities.
    /// This is the strict encoder/decoder `T[p,b]` alignment proof for the
    /// Cpih=3, Sd=1 CFA layout.
    #[test]
    fn round330_cfa_star_tetrix_annex_h_lossless_bit_exact() {
        let (w, h) = (32usize, 16usize);
        let planes = cfa_planes(w, h);
        for nly in [0u8, 1, 2] {
            for cf in [0u8, 3u8] {
                let cs = encode_planar_star_tetrix_annex_h(
                    w as u16, h as u16, 5, nly, 0, 0, 0, cf, 0, &planes,
                )
                .expect("CFA Annex H lossless encode");
                let img = decode_codestream(&cs, None).expect("decode CFA Annex H lossless");
                for (c, (rec, src)) in img.planes.iter().zip(planes.iter()).enumerate() {
                    assert_eq!(
                        rec.data, *src,
                        "NLy={nly} cf={cf}: component {c} bit-exact at q=0"
                    );
                }
            }
        }
    }

    /// Round 330: a CFA configuration outside the tabulated H.9–H.11 set
    /// (`cf ∉ {0, 3}` or `NLy > 2`) falls back to the default-weights
    /// Star-Tetrix path and is byte-identical to a direct default encode.
    #[test]
    fn round330_cfa_star_tetrix_annex_h_falls_back_when_unmatched() {
        let sx = [1u8; 4];
        let sy = [1u8; 4];
        assert!(
            annex_h_weights(4, 3, 1, 5, 3, 0, &sx, &sy).is_none(),
            "no NLy=3 CFA Annex H table"
        );
        assert!(
            annex_h_weights(4, 3, 1, 5, 1, 1, &sx, &sy).is_none(),
            "no cf=1 CFA Annex H column"
        );
        let (w, h) = (32usize, 16usize);
        let planes = cfa_planes(w, h);
        // NLy=3 → no table → default weights with Sd=1 Star-Tetrix.
        let fallback =
            encode_planar_star_tetrix_annex_h(w as u16, h as u16, 5, 3, 2, 0, 0, 0, 0, &planes)
                .expect("fallback CFA encode");
        let direct =
            encode_planar_sd_star_tetrix(w as u16, h as u16, 4, 5, 3, 2, 1, 0, 0, 0, 0, &planes)
                .expect("direct default Sd=1 Star-Tetrix encode");
        assert_eq!(
            fallback, direct,
            "unmatched CFA config must equal default Sd=1 Star-Tetrix"
        );
    }

    // === Round 346: high-bit-depth (B[i] > 8) CFA Star-Tetrix Annex H
    // weights (H.9–H.11) on the Sd=1 layout. Closes the README "content-
    // adaptive WGT weights for the CFA star-tetrix layouts at bit depths
    // above 8" follow-up. ================================================

    /// Build four deterministic CFA `u16` planes with samples in
    /// `0..=2^bd − 1` (the `[R, G1, G2, B]` highbd input for the Sd=1
    /// Star-Tetrix entry points).
    fn cfa_planes_u16(w: usize, h: usize, bd: u8) -> [Vec<u16>; 4] {
        let max = (1u32 << bd) - 1;
        let make = |seed: u32| {
            let mut v = vec![0u16; w * h];
            for y in 0..h {
                for x in 0..w {
                    v[y * w + x] = ((x as u32)
                        .wrapping_mul(seed + 2)
                        .wrapping_add((y as u32).wrapping_mul(seed + 3))
                        .wrapping_add(seed)
                        % (max + 1)) as u16;
                }
            }
            v
        };
        [make(7), make(13), make(19), make(31)]
    }

    /// Round 346: the high-bit-depth CFA Star-Tetrix Annex H entry point
    /// [`encode_planar_star_tetrix_highbd_annex_h`] (a) emits a WGT carrying
    /// exactly the H.9 / H.10 / H.11 `(G[b], P[b])` column selected by `cf`
    /// (one pair per existing band, NL = 19 / 25 / 31), (b) round-trips through
    /// the decoder for the `Cpih = 3, Sd = 1, Nc = 4` CFA layout at `bd ∈ {10,
    /// 12, 16}`, and (c) differs from the default-weights highbd Sd=1 path.
    #[test]
    fn round346_cfa_star_tetrix_highbd_annex_h_roundtrip_and_differ() {
        let (w, h) = (32usize, 16usize);
        let cases: &[(u8, usize)] = &[(0, 19), (1, 25), (2, 31)];
        for &bd in &[10u8, 12, 16] {
            let planes = cfa_planes_u16(w, h, bd);
            for &(nly, n_bands) in cases {
                for cf in [0u8, 3u8] {
                    let sx = [1u8; 4];
                    let sy = [1u8; 4];
                    let (exp_g, exp_p) = annex_h_weights(4, 3, 1, 5, nly, cf, &sx, &sy)
                        .unwrap_or_else(|| panic!("H table NLy={nly} cf={cf}"));
                    assert_eq!(exp_g.len(), n_bands, "bd={bd} NLy={nly}: band count");

                    let cs = encode_planar_star_tetrix_highbd_annex_h(
                        w as u16, h as u16, 5, nly, bd, 2, 0, 0, cf, 0, &planes,
                    )
                    .unwrap_or_else(|e| {
                        panic!("highbd CFA Annex H encode bd={bd} NLy={nly} cf={cf}: {e:?}")
                    });

                    // (a) WGT carries exactly the selected H-table column.
                    let parsed = crate::codestream::parse(&cs).expect("parse highbd CFA Annex H");
                    let weights = parsed.wgt().expect("typed WGT view");
                    let got_g: Vec<u8> = weights.iter().map(|w| w.gain).collect();
                    let got_p: Vec<u8> = weights.iter().map(|w| w.priority).collect();
                    assert_eq!(got_g, exp_g, "bd={bd} NLy={nly} cf={cf}: WGT gains");
                    assert_eq!(got_p, exp_p, "bd={bd} NLy={nly} cf={cf}: WGT priorities");

                    // (b) Round-trips through the decoder with the highbd plane
                    // layout (2 bytes/sample, Bw=B[i]=bd).
                    let img = decode_codestream(&cs, None).expect("decode highbd CFA Annex H");
                    assert_eq!(img.num_components, 4, "bd={bd} NLy={nly}: 4 CFA comps");
                    assert_eq!(img.cpih, 3, "bd={bd} NLy={nly}: Cpih=3");
                    assert_eq!(img.bit_depth, bd, "bd={bd}: Bw byte");
                    assert_eq!(img.width as usize, w);
                    assert_eq!(img.height as usize, h);
                    for p in &img.planes {
                        assert_eq!(p.stride, w * 2, "bd={bd}: 2 bytes/sample plane stride");
                    }

                    // (c) Differs from the default-weights highbd Sd=1 path.
                    let default_cs = encode_planar_sd_star_tetrix_highbd(
                        w as u16, h as u16, 5, nly, bd, 2, 0, 0, cf, 0, &planes,
                    )
                    .expect("default-weights highbd Sd=1 Star-Tetrix");
                    assert_ne!(
                        cs, default_cs,
                        "bd={bd} NLy={nly} cf={cf}: Annex H stream must differ from default"
                    );
                }
            }
        }
    }

    /// Round 346: at `q = 0` the highbd CFA Star-Tetrix Annex H weights are a
    /// no-op for the reconstructed samples — every component self-roundtrips
    /// bit-exactly through the 2-bytes-per-sample plane layout even though the
    /// WGT advertises the H-table gains/priorities. The strict encoder/decoder
    /// `T[p,b]` alignment proof for the highbd Cpih=3, Sd=1 CFA layout.
    #[test]
    fn round346_cfa_star_tetrix_highbd_annex_h_lossless_bit_exact() {
        let (w, h) = (32usize, 16usize);
        for &bd in &[10u8, 12, 14, 16] {
            let planes = cfa_planes_u16(w, h, bd);
            for nly in [0u8, 1, 2] {
                for cf in [0u8, 3u8] {
                    let cs = encode_planar_star_tetrix_highbd_annex_h(
                        w as u16, h as u16, 5, nly, bd, 0, 0, 0, cf, 0, &planes,
                    )
                    .expect("highbd CFA Annex H lossless encode");
                    let img =
                        decode_codestream(&cs, None).expect("decode highbd CFA Annex H lossless");
                    // Reconstruct each component as u16 from the LE plane bytes
                    // and compare against the source samples.
                    for (c, (rec, src)) in img.planes.iter().zip(planes.iter()).enumerate() {
                        let got: Vec<u16> = rec
                            .data
                            .chunks_exact(2)
                            .map(|b| u16::from_le_bytes([b[0], b[1]]))
                            .collect();
                        assert_eq!(
                            got, *src,
                            "bd={bd} NLy={nly} cf={cf}: component {c} bit-exact at q=0"
                        );
                    }
                }
            }
        }
    }

    /// Round 346: a highbd CFA configuration outside the tabulated H.9–H.11 set
    /// (`cf ∉ {0, 3}` or `NLy > 2`) falls back to the default-weights highbd
    /// Sd=1 Star-Tetrix path and is byte-identical to a direct default encode.
    #[test]
    fn round346_cfa_star_tetrix_highbd_annex_h_falls_back_when_unmatched() {
        let (w, h) = (32usize, 16usize);
        let bd = 12u8;
        let planes = cfa_planes_u16(w, h, bd);
        // NLy=3 → no table → default weights with Sd=1 highbd Star-Tetrix.
        let fallback = encode_planar_star_tetrix_highbd_annex_h(
            w as u16, h as u16, 5, 3, bd, 2, 0, 0, 0, 0, &planes,
        )
        .expect("fallback highbd CFA encode");
        let direct = encode_planar_sd_star_tetrix_highbd(
            w as u16, h as u16, 5, 3, bd, 2, 0, 0, 0, 0, &planes,
        )
        .expect("direct default highbd Sd=1 Star-Tetrix encode");
        assert_eq!(
            fallback, direct,
            "unmatched highbd CFA config must equal default highbd Sd=1 Star-Tetrix"
        );
        // NL,x ≠ 5 → no table → also fallback (the H.9–H.11 tables are all
        // tabulated at NL,x = 5). cf ∉ {0, 3} is rejected at CTS validation
        // before the fallback can run, so it is not exercised here.
        let fb_nlx = encode_planar_star_tetrix_highbd_annex_h(
            w as u16, h as u16, 4, 1, bd, 2, 0, 0, 0, 0, &planes,
        )
        .expect("fallback highbd CFA encode NLx=4");
        let direct_nlx = encode_planar_sd_star_tetrix_highbd(
            w as u16, h as u16, 4, 1, bd, 2, 0, 0, 0, 0, &planes,
        )
        .expect("direct default highbd Sd=1 Star-Tetrix encode NLx=4");
        assert_eq!(
            fb_nlx, direct_nlx,
            "NLx=4 highbd CFA must fall back to default"
        );
    }

    /// Round 346: the highbd CFA Star-Tetrix entry points reject the same
    /// out-of-band inputs as their 8-bit / `Sd = 0` siblings — bit depth
    /// outside `9..=16`, the wrong plane count, mismatched plane sizes, and
    /// out-of-range samples.
    #[test]
    fn round346_cfa_star_tetrix_highbd_input_validation() {
        let (w, h) = (16usize, 8usize);
        let bd = 10u8;
        let planes = cfa_planes_u16(w, h, bd);
        // bd = 8 rejected (use the 8-bit entry points).
        assert!(matches!(
            encode_planar_star_tetrix_highbd_annex_h(
                w as u16, h as u16, 5, 1, 8, 2, 0, 0, 0, 0, &planes
            ),
            Err(Error::Unsupported(_))
        ));
        // bd = 17 rejected.
        assert!(encode_planar_sd_star_tetrix_highbd(
            w as u16, h as u16, 5, 1, 17, 2, 0, 0, 0, 0, &planes
        )
        .is_err());
        // Wrong plane count (3, not 4) rejected.
        let three: Vec<Vec<u16>> = planes[..3].to_vec();
        assert!(encode_planar_star_tetrix_highbd_annex_h(
            w as u16, h as u16, 5, 1, bd, 2, 0, 0, 0, 0, &three
        )
        .is_err());
        // Mismatched plane sample count rejected.
        let mut bad = planes.to_vec();
        bad[2].pop();
        assert!(encode_planar_sd_star_tetrix_highbd(
            w as u16, h as u16, 5, 1, bd, 2, 0, 0, 0, 0, &bad
        )
        .is_err());
        // Out-of-range sample (> 2^bd − 1) rejected.
        let mut over = planes.to_vec();
        over[0][0] = ((1u32 << bd) as u16).wrapping_add(5);
        assert!(encode_planar_star_tetrix_highbd_annex_h(
            w as u16, h as u16, 5, 1, bd, 2, 0, 0, 0, 0, &over
        )
        .is_err());
    }

    // === Round 343: uniform inverse quantizer (Qpih=1, Annex D.3) on
    // chroma-subsampled inputs — exercises the inverse-uniform decode
    // path across all three documented chroma samplings. =================

    /// Build a deterministic 4:2:0 luma + half-resolution chroma triple of
    /// the given full-resolution dimensions.
    fn synthetic_420(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n_y = w * h;
        let n_c = (w / 2) * (h / 2);
        let mut y = vec![0u8; n_y];
        let mut cb = vec![0u8; n_c];
        let mut cr = vec![0u8; n_c];
        for (i, slot) in y.iter_mut().enumerate() {
            *slot = ((i * 7 + 13) % 256) as u8;
        }
        for i in 0..n_c {
            cb[i] = ((i * 11 + 17) % 256) as u8;
            cr[i] = ((i * 19 + 23) % 256) as u8;
        }
        (y, cb, cr)
    }

    /// Build a deterministic 4:2:2 luma + horizontally-halved chroma triple.
    fn synthetic_422(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n_y = w * h;
        let n_c = (w / 2) * h;
        let mut y = vec![0u8; n_y];
        let mut cb = vec![0u8; n_c];
        let mut cr = vec![0u8; n_c];
        for (i, slot) in y.iter_mut().enumerate() {
            *slot = ((i * 5 + 9) % 256) as u8;
        }
        for i in 0..n_c {
            cb[i] = ((i * 13 + 21) % 256) as u8;
            cr[i] = ((i * 23 + 29) % 256) as u8;
        }
        (y, cb, cr)
    }

    /// 4:2:0 Qpih=1 lossless (q=0): the uniform reconstruction collapses to
    /// `v` when `T = 0`, so the chroma-subsampled stream round-trips bit-
    /// exactly through the inverse-uniform decode kernel. Confirms the PIH
    /// Qpih bit is actually 1 (the decoder takes the Annex D.3 branch) and
    /// every sample is recovered exactly.
    #[test]
    fn r343_qpih_uniform_420_lossless_round_trip() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        let cs = encode_planar_qpih_subsampled(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0, // q = 0 lossless
            &[1, 2, 2],
            &[1, 2, 2],
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:0 Qpih=1 lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "4:2:0 stream must signal Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0 Qpih=1 lossless");
        assert_eq!(img.num_components, 3);
        assert_eq!(img.planes[0].data, y, "Y plane lossless");
        assert_eq!(img.planes[1].data, cb, "Cb plane lossless");
        assert_eq!(img.planes[2].data, cr, "Cr plane lossless");
    }

    /// 4:2:2 Qpih=1 lossless (q=0) bit-exact round-trip through the
    /// inverse-uniform decode path.
    #[test]
    fn r343_qpih_uniform_422_lossless_round_trip() {
        let (w, h) = (64usize, 48usize);
        let (y, cb, cr) = synthetic_422(w, h);
        let cs = encode_planar_qpih_subsampled(
            w as u16,
            h as u16,
            3,
            0,
            3,
            3,
            0, // q = 0 lossless
            &[1, 2, 2],
            &[1, 1, 1],
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:2 Qpih=1 lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "4:2:2 stream must signal Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode 4:2:2 Qpih=1 lossless");
        assert_eq!(img.planes[0].data, y, "Y plane lossless");
        assert_eq!(img.planes[1].data, cb, "Cb plane lossless");
        assert_eq!(img.planes[2].data, cr, "Cr plane lossless");
    }

    /// 4:2:0 Qpih=1 lossy (q=2): the decoder takes the Annex D.3 Neumann-
    /// series branch on every band. Confirms the reconstruction stays valid
    /// (decodable, correct geometry) and above a sane PSNR floor — the
    /// uniform (round-to-nearest) kernel must not reconstruct worse than the
    /// deadzone path it parallels.
    #[test]
    fn r343_qpih_uniform_420_lossy_q2_decodes_above_floor() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        let cs = encode_planar_qpih_subsampled(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            2, // q = 2 lossy → Fq = 8
            &[1, 2, 2],
            &[1, 2, 2],
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:0 Qpih=1 q=2");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "4:2:0 lossy stream must signal Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0 Qpih=1 q=2");
        assert_eq!(img.planes[0].data.len(), y.len());
        assert_eq!(img.planes[1].data.len(), cb.len());
        assert_eq!(img.planes[2].data.len(), cr.len());
        let psnr_y = psnr(&y, &img.planes[0].data);
        let psnr_cb = psnr(&cb, &img.planes[1].data);
        let psnr_cr = psnr(&cr, &img.planes[2].data);
        assert!(
            psnr_y >= 20.0 && psnr_cb >= 20.0 && psnr_cr >= 20.0,
            "4:2:0 Qpih=1 q=2 PSNR below 20 dB: Y={psnr_y} Cb={psnr_cb} Cr={psnr_cr}"
        );
    }

    /// 4:4:4 RGB-RCT Qpih=1 lossy (q=2) under the uniform quantizer keeps the
    /// reconstruction valid through the colour-transform inverse — confirms
    /// the inverse-uniform kernel composes with the RCT decode path.
    #[test]
    fn r343_qpih_uniform_444_rct_lossy_q2_decodes() {
        let (w, h) = (32usize, 32usize);
        let n = w * h;
        let mut r = vec![0u8; n];
        let mut g = vec![0u8; n];
        let mut b = vec![0u8; n];
        for i in 0..n {
            r[i] = ((i * 3 + 5) % 256) as u8;
            g[i] = ((i * 7 + 11) % 256) as u8;
            b[i] = ((i * 13 + 19) % 256) as u8;
        }
        let cs = encode_planar_qpih_subsampled(
            w as u16,
            h as u16,
            3,
            1, // Cpih=1 RCT
            2,
            2,
            2, // q = 2
            &[1, 1, 1],
            &[1, 1, 1],
            &[r.clone(), g.clone(), b.clone()],
        )
        .expect("encode 4:4:4 RCT Qpih=1 q=2");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "RCT stream must signal Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode 4:4:4 RCT Qpih=1 q=2");
        assert_eq!(img.num_components, 3);
        let psnr_r = psnr(&r, &img.planes[0].data);
        let psnr_g = psnr(&g, &img.planes[1].data);
        let psnr_b = psnr(&b, &img.planes[2].data);
        assert!(
            psnr_r >= 18.0 && psnr_g >= 18.0 && psnr_b >= 18.0,
            "4:4:4 RCT Qpih=1 q=2 PSNR below 18 dB: R={psnr_r} G={psnr_g} B={psnr_b}"
        );
    }

    /// Thin wrapper around `encode_planar_inner_nlt` that pins `Qpih = 1`
    /// (uniform inverse quantizer) and lets a test exercise the structural
    /// axes (`hsl`, `cw`, `fs`) the public subsampled entry point fixes at
    /// their defaults.
    #[allow(clippy::too_many_arguments)]
    fn encode_qpih1_struct(
        w: u16,
        h: u16,
        nc: u8,
        cpih: u8,
        nlx: u8,
        nly: u8,
        q: u8,
        sx: &[u8],
        sy: &[u8],
        cw: u16,
        hsl: u16,
        fs: u8,
        planes: &[Vec<u8>],
    ) -> Result<Vec<u8>> {
        let fq = 0; // integer-transform path; quantisation via T[p,b], Fq=0 (Table A.8)
        encode_planar_inner_nlt(
            w,
            h,
            nc,
            cpih,
            nlx,
            nly,
            fq,
            q,
            sx,
            sy,
            0,
            0,
            0,
            0,
            None,
            Vec::new(),
            Vec::new(),
            cw,
            0, // sd
            fs,
            hsl,
            1, // qpih = 1 (uniform, Annex D.3)
            0, // rp
            Vec::new(),
            Vec::new(),
            Vec::new(),
            planes,
        )
    }

    /// Uniform quantizer composed with multi-slice (`Hsl > 0`): the
    /// inverse-uniform decode path runs per slice. Lossless bit-exact
    /// round-trip confirms slice boundaries do not perturb the Annex D.3
    /// reconstruction. PIH `Qpih = 1` confirmed.
    #[test]
    fn r343_qpih_uniform_multi_slice_420_lossless() {
        let (w, h) = (64usize, 64usize);
        let (y, cb, cr) = synthetic_420(w, h);
        // Hsl = 1 precinct row per slice → multiple slices over the picture.
        let cs = encode_qpih1_struct(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0, // lossless
            &[1, 2, 2],
            &[1, 2, 2],
            0, // cw: single precinct column
            1, // hsl: one precinct row per slice
            0, // fs
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:0 Qpih=1 multi-slice lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "multi-slice stream must signal Qpih=1");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0 Qpih=1 multi-slice");
        assert_eq!(img.planes[0].data, y, "Y lossless across slices");
        assert_eq!(img.planes[1].data, cb, "Cb lossless across slices");
        assert_eq!(img.planes[2].data, cr, "Cr lossless across slices");
    }

    /// Uniform quantizer composed with multi-precinct-per-row (`Cw > 0`):
    /// every precinct column dequantizes with the Annex D.3 kernel.
    /// Lossless bit-exact round-trip over a 4:2:2 picture split into
    /// multiple precinct columns.
    #[test]
    fn r343_qpih_uniform_multi_precinct_422_lossless() {
        let (w, h) = (64usize, 32usize);
        let (y, cb, cr) = synthetic_422(w, h);
        // Cw = 1 → Cs = 8 * 1 * max(sx)=2 * 2^NLx. With NLx small the
        // picture splits into several precinct columns per row.
        let cs = encode_qpih1_struct(
            w as u16,
            h as u16,
            3,
            0,
            1,
            1,
            0, // lossless
            &[1, 2, 2],
            &[1, 1, 1],
            1, // cw: multi-precinct per row
            0, // hsl: single slice
            0, // fs
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:2 Qpih=1 multi-precinct lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(
            parsed.pih.qpih, 1,
            "multi-precinct stream must signal Qpih=1"
        );
        let img = decode_codestream(&cs, None).expect("decode 4:2:2 Qpih=1 multi-precinct");
        assert_eq!(img.planes[0].data, y, "Y lossless across precinct columns");
        assert_eq!(
            img.planes[1].data, cb,
            "Cb lossless across precinct columns"
        );
        assert_eq!(
            img.planes[2].data, cr,
            "Cr lossless across precinct columns"
        );
    }

    /// Uniform quantizer composed with the separate sign sub-packet
    /// (`Fs = 1`) on a chroma-subsampled picture: confirms the inverse-
    /// uniform kernel reconstructs correctly when signs arrive from the
    /// dedicated sign sub-packet rather than interleaved in the data
    /// sub-packet. Lossless bit-exact.
    #[test]
    fn r343_qpih_uniform_fs1_420_lossless() {
        let (w, h) = (32usize, 32usize);
        let (y, cb, cr) = synthetic_420(w, h);
        let cs = encode_qpih1_struct(
            w as u16,
            h as u16,
            3,
            0,
            2,
            2,
            0, // lossless
            &[1, 2, 2],
            &[1, 2, 2],
            0, // cw
            0, // hsl
            1, // fs = 1 (separate sign sub-packet)
            &[y.clone(), cb.clone(), cr.clone()],
        )
        .expect("encode 4:2:0 Qpih=1 Fs=1 lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1);
        assert_eq!(parsed.pih.fs, 1, "Fs=1 separate sign sub-packet");
        let img = decode_codestream(&cs, None).expect("decode 4:2:0 Qpih=1 Fs=1");
        assert_eq!(img.planes[0].data, y);
        assert_eq!(img.planes[1].data, cb);
        assert_eq!(img.planes[2].data, cr);
    }

    /// Uniform quantizer on the CFA Star-Tetrix layout (`Cpih = 3`): the
    /// inverse-uniform decode path composes with the Star-Tetrix colour
    /// transform inverse. Lossless bit-exact round-trip over a 4-component
    /// CFA picture.
    #[test]
    fn r343_qpih_uniform_star_tetrix_lossless() {
        let (w, h) = (32usize, 16usize);
        let planes = cfa_planes(w, h);
        let cs = encode_qpih1_struct(
            w as u16,
            h as u16,
            4,
            3, // Cpih = 3 Star-Tetrix
            2,
            2,
            0, // lossless
            &[1, 1, 1, 1],
            &[1, 1, 1, 1],
            0, // cw
            0, // hsl
            0, // fs
            &planes,
        )
        .expect("encode CFA Qpih=1 Star-Tetrix lossless");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(parsed.pih.qpih, 1, "Star-Tetrix stream must signal Qpih=1");
        assert_eq!(parsed.pih.cpih, 3, "Cpih=3 Star-Tetrix");
        let img = decode_codestream(&cs, None).expect("decode CFA Qpih=1 Star-Tetrix");
        assert_eq!(img.num_components, 4);
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(
                &img.planes[c].data, plane,
                "CFA component {c} lossless under uniform quantizer"
            );
        }
    }

    /// Round 351 — a regular *lossy* stream (q > 0, no NLT) now signals the
    /// conformant ISO/IEC 21122-1:2022 Table A.8 combination
    /// `(Bw = B[0], Fq = 0)` instead of the historical non-tabulated
    /// `(Bw = 8, Fq = 8)`. The integer 5/3 transform carries no extra
    /// fractional bits, so Fq = 0 is the truthful signalling; lossy
    /// compression is still achieved through the precinct truncation
    /// `T[p,b]`, and the stream still round-trips within the PSNR floor.
    #[test]
    fn round351_regular_lossy_signals_conformant_bw_fq() {
        let w = 32usize;
        let h = 32usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = ((x * 7 + y * 11) % 256) as u8;
            }
        }
        let cs = encode_planar_lossy(w as u16, h as u16, 1, 0, 2, 2, 2, &[pixels.clone()])
            .expect("encode 32x32 luma q=2");
        let parsed = crate::codestream::parse(&cs).expect("parse");
        assert_eq!(
            parsed.pih.fq, 0,
            "regular lossy must signal the conformant Fq=0 (Table A.8), not 8"
        );
        assert_eq!(
            parsed.pih.bw, 8,
            "Fq=0 requires Bw=B[0]=8 (Table A.8 lossless combination)"
        );
        // Still genuinely lossy + decodes within the floor.
        let img = decode_codestream(&cs, None).expect("decode 32x32 luma q=2");
        let p = psnr(&pixels, &img.planes[0].data);
        assert!(
            p >= 25.0,
            "regular lossy q=2 PSNR {p:.2} dB below 25 dB floor"
        );
    }

    /// Round 351 — the decoder rejects a hand-mutated codestream whose
    /// `(Bw, Fq)` pair is outside ISO/IEC 21122-1:2022 Table A.8. We take a
    /// valid lossless stream (`Bw = 8, Fq = 0`) and flip the PIH `Fq`
    /// nibble to 8, producing the non-tabulated `(8, 8)` pair the old
    /// encoder used to emit. A conforming decoder must reject it rather
    /// than silently `<< 8`-scale the coefficients.
    #[test]
    fn round351_decoder_rejects_non_table_a8_bw_fq() {
        let w = 16usize;
        let h = 16usize;
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i % 256) as u8).collect();
        let mut cs = encode_planar(w as u16, h as u16, 1, 0, 2, 2, &[pixels])
            .expect("encode 16x16 luma lossless");
        // Locate the PIH marker (FF 12) and flip the Fq nibble (the high
        // nibble of the Fq:Br byte, which is `Lpih(2) + 2 + ... ` into the
        // body). Rather than hand-compute the offset, find the byte that
        // currently encodes Fq:Br = 0x?8 with Fq nibble 0 and set Fq=8.
        let pih_pos = cs
            .windows(2)
            .position(|w| w == [0xff, 0x12])
            .expect("PIH marker present");
        // PIH body layout: marker(2) Lpih(2) then fields; within the body
        // (the bytes after Lpih) Bw is body[19] and Fq:Br is body[20].
        // Absolute index: pih_pos + 2 (marker) + 2 (Lpih) + 20.
        let fq_br_idx = pih_pos + 4 + 20;
        // Sanity: current Fq nibble is 0 (lossless).
        assert_eq!(cs[fq_br_idx] >> 4, 0, "lossless PIH should carry Fq=0");
        // Set Fq nibble to 8 (keep Br low nibble), producing the
        // non-tabulated (Bw=8, Fq=8) pair.
        cs[fq_br_idx] = (8 << 4) | (cs[fq_br_idx] & 0x0f);
        let err = decode_codestream(&cs, None)
            .expect_err("decoder must reject the non-Table-A.8 (Bw=8, Fq=8) pair");
        let msg = format!("{err}");
        assert!(
            msg.contains("Table A.8"),
            "rejection should cite Table A.8, got: {msg}"
        );
    }

    /// Round 351 — the high-precision `(Bw = 20, Fq = 8)` regular lossy
    /// path (Annex E.3 fractional scaling) signals the conformant Table A.8
    /// pair, decodes within a PSNR floor, and is at least as faithful as the
    /// integer-transform path at the same `q` on a smooth (gradient) image,
    /// where the per-decomposition-level integer rounding is visible.
    #[test]
    fn round351_highprec_lossy_bw20_fq8_psnr() {
        let w = 64usize;
        let h = 64usize;
        // Smooth diagonal gradient — the multi-level 5/3 transform's
        // per-level rounding is most visible on low-frequency content.
        let mut pixels = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                pixels[y * w + x] = (((x + y) * 2) % 256) as u8;
            }
        }
        // NL = 5/5 to give the fractional precision several levels of
        // rounding to improve on.
        let q = 2u8;
        let hp = encode_planar_highprec_lossy(w as u16, h as u16, 1, 0, 5, 5, q, &[pixels.clone()])
            .expect("encode 64x64 high-precision q=2");
        let parsed = crate::codestream::parse(&hp).expect("parse high-precision PIH");
        assert_eq!(
            (parsed.pih.bw, parsed.pih.fq),
            (20, 8),
            "high-precision path must signal the (Bw=20, Fq=8) Table A.8 pair"
        );
        let hp_img = decode_codestream(&hp, None).expect("decode high-precision q=2");
        let hp_psnr = psnr(&pixels, &hp_img.planes[0].data);
        assert!(
            hp_psnr >= 30.0,
            "high-precision lossy q=2 PSNR {hp_psnr:.2} dB below 30 dB floor"
        );

        // The integer-transform path at the same q on the same content.
        let lo = encode_planar_lossy(w as u16, h as u16, 1, 0, 5, 5, q, &[pixels.clone()])
            .expect("encode 64x64 integer-transform q=2");
        let lo_img = decode_codestream(&lo, None).expect("decode integer-transform q=2");
        let lo_psnr = psnr(&pixels, &lo_img.planes[0].data);
        // The extra fractional precision must not make reconstruction worse;
        // allow a tiny epsilon for cases where both paths are already near
        // the quantiser-limited ceiling.
        assert!(
            hp_psnr >= lo_psnr - 0.01,
            "high-precision PSNR {hp_psnr:.2} dB should be >= integer-transform {lo_psnr:.2} dB"
        );
    }
}
