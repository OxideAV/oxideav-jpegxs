//! Crate-local uncompressed image representation.
//!
//! Defined here (rather than reusing `oxideav_core::VideoFrame`) so the
//! crate can be built with the default `registry` feature off — i.e.
//! without depending on `oxideav-core` at all. When the `registry`
//! feature is on the [`crate::registry`] module provides
//! `From<JpegXsImage> for oxideav_core::Frame`, so the `Decoder` trait
//! surface still interoperates cleanly.
//!
//! Unlike a typical decoder, JPEG XS doesn't pin a `PixelFormat`-style
//! tag on the output: the codestream's `Nc` (component count) plus
//! `Cpih` (colour-transform mode) plus per-component sub-sampling
//! describe the layout. We surface those raw fields and let the caller
//! decide how to interpret them.

/// One image plane: row-major bytes plus the row stride in bytes.
#[derive(Debug, Clone)]
pub struct JpegXsPlane {
    /// Bytes per row in `data` (== logical row width for the planes the
    /// decoder produces today, but kept explicit for forward compat).
    pub stride: usize,
    /// Raw plane bytes (8-bit per sample for `Bw == 8`; 16-bit
    /// little-endian when the decoder ever grows higher bit depths).
    pub data: Vec<u8>,
}

/// One decoded JPEG XS frame.
///
/// All-`std`, no `oxideav-core` types. The gated [`crate::registry`]
/// module provides a `From<JpegXsImage> for oxideav_core::Frame`
/// conversion that drops the geometry / colour-mode fields — the
/// `oxideav_core` `Frame` enum carries that out-of-band via
/// `CodecParameters` instead.
#[derive(Debug, Clone)]
pub struct JpegXsImage {
    /// Picture width in pixels (`Wf`).
    pub width: u32,
    /// Picture height in pixels (`Hf`).
    pub height: u32,
    /// Number of components (`Nc` from the picture header). Matches
    /// `planes.len()`.
    pub num_components: u8,
    /// Inverse-colour-transform mode actually applied — copied verbatim
    /// from the codestream's `Cpih`. `0` = no transform, `1` = inverse
    /// RCT (Annex F.3), `3` = inverse Star-Tetrix (Annex F.5).
    pub cpih: u8,
    /// Output bit depth per sample (`Bw`). Currently always `8`; the
    /// decoder rejects everything else.
    pub bit_depth: u8,
    /// One entry per component. Sub-sampled chroma plates appear at
    /// their downsampled dimensions; the caller can read the per-plane
    /// `stride` to recover the per-component width.
    pub planes: Vec<JpegXsPlane>,
    /// Optional presentation timestamp. The standalone decode path
    /// always leaves this `None`; the registry-backed `Decoder` impl
    /// fills it in from the `Packet` it consumed.
    pub pts: Option<i64>,
}
