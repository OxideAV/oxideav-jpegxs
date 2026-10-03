//! [`EncodeOptions`] — every encoder choice as a field of one record
//! (the workspace image-crate API contract: behaviour variants are
//! fields, never function suffixes).
//!
//! The defaults encode losslessly (`quantization = 0`), 4:4:4 at the
//! image's own layout, with an automatically chosen wavelet
//! decomposition, a single slice, no profile declaration and a bare
//! ISO/IEC 21122-1 codestream as output.

use crate::output::NltParams;
use crate::profile::Profile;

/// Inverse-quantiser type the picture header signals (`Qpih`, ISO/IEC
/// 21122-1 Annex A.4.4 Table A.10). The data on the wire is identical;
/// only the decoder's reconstruction kernel differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Quantizer {
    /// `Qpih = 0` — deadzone quantiser (Annex D.2).
    #[default]
    Deadzone,
    /// `Qpih = 1` — uniform quantiser (Annex D.3).
    Uniform,
}

impl Quantizer {
    /// The `Qpih` field value.
    pub fn qpih(self) -> u8 {
        match self {
            Self::Deadzone => 0,
            Self::Uniform => 1,
        }
    }
}

/// Run mode (`Rm`, Annex A.4.4 Table A.12): how an insignificant
/// significance group is interpreted in vertical-prediction bitplane
/// coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum RunMode {
    /// `Rm = 0` — runs mark zero *prediction residuals*.
    #[default]
    ZeroResiduals,
    /// `Rm = 1` — runs mark zero *coefficients*.
    ZeroCoefficients,
}

impl RunMode {
    /// The `Rm` field value.
    pub fn rm(self) -> u8 {
        match self {
            Self::ZeroResiduals => 0,
            Self::ZeroCoefficients => 1,
        }
    }
}

/// Which band weights (`G[b]` gains, `P[b]` priorities — the WGT marker,
/// Annex A.4.11) the encoder writes and truncates with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Weights {
    /// The crate's structural default: LL `0`, HL / LH `1`, HH `2`
    /// gains, priorities in band order.
    #[default]
    Default,
    /// The ISO/IEC 21122-1:2022 Annex H PSNR-optimised example tables
    /// when one exists for the configuration (`NL,x = 5`, 4:4:4 with or
    /// without RCT, 4:2:2, 4:2:0, CFA); otherwise the default weights.
    AnnexH,
}

/// Star-Tetrix (`Cpih = 3`, Annex F.5) parameters, meaningful only for
/// [`crate::encode_components`] with `cpih = 3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct StarTetrixParams {
    /// CTS chroma-weighting exponent `e1` (`0..=3`).
    pub e1: u8,
    /// CTS chroma-weighting exponent `e2` (`0..=3`).
    pub e2: u8,
    /// CTS extent `Cf` (`0` full transform, `3` in-line).
    pub cf: u8,
    /// CFA pattern type `Ct` (Table F.9: `0` RGGB / BGGR, `1` GRBG / GBRG).
    pub ct: u8,
}

impl StarTetrixParams {
    /// Build from the four CTS / CRG fields.
    pub fn new(e1: u8, e2: u8, cf: u8, ct: u8) -> Self {
        Self { e1, e2, cf, ct }
    }
}

/// Encoder options for [`crate::encode`] / [`crate::encode_components`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Precinct quantisation step `Q[p]` (`0..=15`). `0` is lossless
    /// (integer transform, bit-exact round trip); larger values truncate
    /// more bitplanes. Ignored when `target_bytes` is set. Default `0`.
    pub quantization: u8,
    /// Constant-bitrate target: emit a codestream of exactly this many
    /// bytes (per-slice rate allocation, COM padding, `Lcod` set). With
    /// a `profile` the allocation follows the profile's slice height;
    /// without one it uses `slice_height`. Default `None` (variable
    /// bitrate at `quantization`).
    pub target_bytes: Option<usize>,
    /// Shape the stream to an ISO/IEC 21122-2 profile and declare it:
    /// the profile's mandated slice height is used, and `Ppih` / `Plev`
    /// are set to the profile and the tightest verified level /
    /// sublevel. `None` leaves `Ppih = Plev = 0` (unrestricted,
    /// undeclared). Default `None`.
    pub profile: Option<Profile>,
    /// Horizontal wavelet decomposition levels `NL,x` (`1..=8`). `None`
    /// picks the largest of `1..=5` the picture width admits (Table 11:
    /// `Wf ≥ max(sx) × 2^NL,x`). Default `None`.
    pub levels_x: Option<u8>,
    /// Vertical wavelet decomposition levels `NL,y` (`0..=NL,x`). `None`
    /// picks `min(1, NL,x)` if the height admits it, else `0`. Default
    /// `None`.
    pub levels_y: Option<u8>,
    /// Apply the reversible colour transform (`Cpih = 1`, Annex F.3) to
    /// the RGB layouts. Grey and YCbCr layouts never use it. Default
    /// `true`.
    pub rct: bool,
    /// Inverse-quantiser type to signal. Default [`Quantizer::Deadzone`].
    pub quantizer: Quantizer,
    /// Slice height `Hsl` in precinct rows; `None` = one slice spanning
    /// the picture. Overridden by `profile`. Default `None`.
    pub slice_height: Option<u16>,
    /// Precinct column width parameter `Cw` (`0` = one precinct column
    /// spanning the picture width). Default `0`.
    pub column_width: u16,
    /// Code signs in a separate sign sub-packet (`Fs = 1`) instead of
    /// jointly with the data (`Fs = 0`). Default `false`.
    pub sign_packet: bool,
    /// Run mode `Rm`. Default [`RunMode::ZeroResiduals`].
    pub run_mode: RunMode,
    /// Precinct refinement `R[p]` (`0..=NL−1` bands). Default `0`.
    pub refinement: u8,
    /// High-precision regular path: `Bw = 20`, `Fq = 8` (Table A.8),
    /// wavelet coefficients carry eight fractional bits. Default `false`
    /// (integer transform, `Fq = 0`).
    pub high_precision: bool,
    /// Non-linear transform to signal and pre-distort with (NLT marker,
    /// Annex G.4 / G.5). Default `None`.
    pub nlt: Option<NltParams>,
    /// Band weights. Default [`Weights::Default`].
    pub weights: Weights,
    /// Number of trailing components whose wavelet decomposition is
    /// suppressed (`Sd`, CWD marker, Annex A.4.7). Default `0`.
    pub suppressed_components: u8,
    /// Star-Tetrix parameters for `cpih = 3` component sets. Default
    /// `None` (the Annex F.5 defaults `e1 = e2 = 0`, `Cf = 0`, `Ct = 0`
    /// apply when a Star-Tetrix set is encoded without them).
    pub star_tetrix: Option<StarTetrixParams>,
    /// Per-slice `Q[p]` overrides (one per slice); empty = use
    /// `quantization`. Default empty.
    pub q_slices: Vec<u8>,
    /// Per-precinct `Q[p]` overrides (raster order, one per precinct);
    /// empty = none. Default empty.
    pub q_precincts: Vec<u8>,
    /// Per-precinct `R[p]` overrides; empty = `refinement` everywhere.
    /// Default empty.
    pub r_precincts: Vec<u8>,
    /// Wrap the codestream in a `.jxs` still-image file (ISO/IEC 21122-3
    /// Annex A): Signature, File Type and Header boxes (image header,
    /// CICP colour specification from the image's `color`, channel
    /// definition for the alpha layouts, Exif box when the metadata has
    /// one), then the Contiguous Codestream box. Default `false` (bare
    /// codestream).
    pub boxed: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quantization: 0,
            target_bytes: None,
            profile: None,
            levels_x: None,
            levels_y: None,
            rct: true,
            quantizer: Quantizer::Deadzone,
            slice_height: None,
            column_width: 0,
            sign_packet: false,
            run_mode: RunMode::ZeroResiduals,
            refinement: 0,
            high_precision: false,
            nlt: None,
            weights: Weights::Default,
            suppressed_components: 0,
            star_tetrix: None,
            q_slices: Vec::new(),
            q_precincts: Vec::new(),
            r_precincts: Vec::new(),
            boxed: false,
        }
    }
}

impl EncodeOptions {
    /// The defaults (lossless, auto decomposition, bare codestream).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quantisation step `Q[p]` (`0..=15`; `0` = lossless).
    pub fn with_quantization(mut self, quantization: u8) -> Self {
        self.quantization = quantization;
        self
    }

    /// Set a constant-bitrate byte target (`None` = variable bitrate).
    pub fn with_target_bytes(mut self, target_bytes: Option<usize>) -> Self {
        self.target_bytes = target_bytes;
        self
    }

    /// Shape to and declare an ISO/IEC 21122-2 profile (`None` = none).
    pub fn with_profile(mut self, profile: Option<Profile>) -> Self {
        self.profile = profile;
        self
    }

    /// Set the wavelet decomposition levels (`None` = automatic).
    pub fn with_levels(mut self, levels_x: Option<u8>, levels_y: Option<u8>) -> Self {
        self.levels_x = levels_x;
        self.levels_y = levels_y;
        self
    }

    /// Enable / disable the reversible colour transform for RGB layouts.
    pub fn with_rct(mut self, rct: bool) -> Self {
        self.rct = rct;
        self
    }

    /// Set the inverse-quantiser type to signal.
    pub fn with_quantizer(mut self, quantizer: Quantizer) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Set the slice height in precinct rows (`None` = one slice).
    pub fn with_slice_height(mut self, slice_height: Option<u16>) -> Self {
        self.slice_height = slice_height;
        self
    }

    /// Set the precinct column width parameter `Cw`.
    pub fn with_column_width(mut self, column_width: u16) -> Self {
        self.column_width = column_width;
        self
    }

    /// Code signs in a separate sub-packet.
    pub fn with_sign_packet(mut self, sign_packet: bool) -> Self {
        self.sign_packet = sign_packet;
        self
    }

    /// Set the run mode.
    pub fn with_run_mode(mut self, run_mode: RunMode) -> Self {
        self.run_mode = run_mode;
        self
    }

    /// Set the precinct refinement `R[p]`.
    pub fn with_refinement(mut self, refinement: u8) -> Self {
        self.refinement = refinement;
        self
    }

    /// Select the high-precision (`Bw = 20`, `Fq = 8`) path.
    pub fn with_high_precision(mut self, high_precision: bool) -> Self {
        self.high_precision = high_precision;
        self
    }

    /// Set the non-linear transform.
    pub fn with_nlt(mut self, nlt: Option<NltParams>) -> Self {
        self.nlt = nlt;
        self
    }

    /// Set the band weights.
    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the number of decomposition-suppressed trailing components.
    pub fn with_suppressed_components(mut self, suppressed_components: u8) -> Self {
        self.suppressed_components = suppressed_components;
        self
    }

    /// Set the Star-Tetrix parameters.
    pub fn with_star_tetrix(mut self, star_tetrix: Option<StarTetrixParams>) -> Self {
        self.star_tetrix = star_tetrix;
        self
    }

    /// Set per-slice `Q[p]` overrides.
    pub fn with_q_slices(mut self, q_slices: Vec<u8>) -> Self {
        self.q_slices = q_slices;
        self
    }

    /// Set per-precinct `Q[p]` overrides.
    pub fn with_q_precincts(mut self, q_precincts: Vec<u8>) -> Self {
        self.q_precincts = q_precincts;
        self
    }

    /// Set per-precinct `R[p]` overrides.
    pub fn with_r_precincts(mut self, r_precincts: Vec<u8>) -> Self {
        self.r_precincts = r_precincts;
        self
    }

    /// Emit a `.jxs` box file instead of a bare codestream.
    pub fn with_boxed(mut self, boxed: bool) -> Self {
        self.boxed = boxed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_builders() {
        let d = EncodeOptions::default();
        assert_eq!(d.quantization, 0);
        assert!(d.target_bytes.is_none() && d.profile.is_none());
        assert!(d.levels_x.is_none() && d.levels_y.is_none());
        assert!(d.rct && !d.boxed && !d.high_precision && !d.sign_packet);
        assert_eq!(d.quantizer.qpih(), 0);
        assert_eq!(d.run_mode.rm(), 0);
        let o = EncodeOptions::new()
            .with_quantization(3)
            .with_target_bytes(Some(4096))
            .with_profile(Some(Profile::Main444_12))
            .with_levels(Some(5), Some(1))
            .with_rct(false)
            .with_quantizer(Quantizer::Uniform)
            .with_slice_height(Some(4))
            .with_column_width(1)
            .with_sign_packet(true)
            .with_run_mode(RunMode::ZeroCoefficients)
            .with_refinement(2)
            .with_high_precision(true)
            .with_nlt(Some(NltParams::Quadratic { dco: 0 }))
            .with_weights(Weights::AnnexH)
            .with_suppressed_components(1)
            .with_star_tetrix(Some(StarTetrixParams::new(1, 2, 3, 1)))
            .with_q_slices(vec![1])
            .with_q_precincts(vec![2])
            .with_r_precincts(vec![0])
            .with_boxed(true);
        assert_eq!(o.quantization, 3);
        assert_eq!(o.target_bytes, Some(4096));
        assert_eq!(o.profile, Some(Profile::Main444_12));
        assert_eq!((o.levels_x, o.levels_y), (Some(5), Some(1)));
        assert!(!o.rct && o.boxed && o.high_precision && o.sign_packet);
        assert_eq!(o.quantizer.qpih(), 1);
        assert_eq!(o.run_mode.rm(), 1);
        assert_eq!(o.slice_height, Some(4));
        assert_eq!(o.column_width, 1);
        assert_eq!(o.refinement, 2);
        assert_eq!(o.weights, Weights::AnnexH);
        assert_eq!(o.suppressed_components, 1);
        assert_eq!(o.star_tetrix.unwrap().cf, 3);
        assert_eq!(
            (o.q_slices, o.q_precincts, o.r_precincts),
            (vec![1], vec![2], vec![0])
        );
    }
}
