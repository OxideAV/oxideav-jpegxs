//! Crate-local image, plane, pixel-format, colour, metadata, header
//! summary and decode-option types — the standalone (no `oxideav-core`)
//! vocabulary of the workspace image-crate API contract, plus the
//! JPEG XS component-plane depth record.
//!
//! Defined here (rather than reusing `oxideav_core::VideoFrame` /
//! `oxideav_core::PixelFormat`) so the crate can be built with the
//! default `registry` feature off — i.e. without depending on
//! `oxideav-core` at all. When the `registry` feature is on the
//! [`crate::registry`] module provides `From<JpegXsImage> for
//! oxideav_core::VideoFrame`, [`JpegXsImage::from_video_frame`] and the
//! 1:1 pixel-format / colour-signal mappings.
//!
//! Two pixel shapes live here:
//!
//! * [`JpegXsImage`] — one picture in a contract layout (`format` says
//!   which; planes are in the *layout's* order, e.g. G, B, R for
//!   `Gbrp8`). This is what [`crate::decode`] returns and what
//!   [`crate::encode`] consumes.
//! * [`Components`] — the picture exactly as the codestream carries it:
//!   `Nc` component planes in codestream order with the component
//!   table's bit depths and sampling factors, plus the colour-transform id. Every JPEG XS
//!   codestream has this view (Star-Tetrix CFA pictures and `Nc ∉ {1, 3,
//!   4}` pictures have *only* this view); [`crate::decode_components`] /
//!   [`crate::encode_components`] work on it.

use crate::error::{JpegXsError, Result};

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// One image plane: row-major bytes plus the row stride in bytes.
/// Layout-compatible with `oxideav_core::frame::VideoPlane`.
///
/// Samples deeper than 8 bits are stored as two little-endian bytes
/// each, value in the low bits (`Gray10Le` / `Gbrp12Le` / … storage).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plane {
    /// Bytes per row in `data`.
    pub stride: usize,
    /// Raw plane bytes, `stride × rows`.
    pub data: Vec<u8>,
}

impl Plane {
    /// Wrap a row-major byte buffer with its row stride.
    pub fn new(stride: usize, data: Vec<u8>) -> Self {
        Self { stride, data }
    }

    /// Number of complete rows in the plane (`data.len() / stride`; `0`
    /// for a zero stride).
    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.stride).unwrap_or(0)
    }
}

/// Historical name of [`Plane`]. Same struct.
pub type JpegXsPlane = Plane;

// ---------------------------------------------------------------------------
// Pixel format
// ---------------------------------------------------------------------------

/// Subset of `oxideav_core::PixelFormat` the JPEG XS decoder produces /
/// the encoder accepts. Variant names mirror the framework enum exactly
/// (the `registry` feature maps them 1:1 by name).
///
/// Every layout is planar (JPEG XS codes components as separate planes;
/// the grey layouts are the one-plane case). Deeper-than-8-bit samples
/// are two little-endian bytes with the value in the low bits. Bit
/// depths without a core label of their own (9, 11, 13, 14, 15 for
/// grey / YCbCr; 9, 11, 13, 15 for GBR) ride the 16-bit carrier with
/// [`JpegXsImage::bit_depth`] naming the significant bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JpegXsPixelFormat {
    /// 8-bit grey, one plane (`Nc = 1`, `B = 8`).
    Gray8,
    /// 10-bit grey in 16-bit LE storage.
    Gray10Le,
    /// 12-bit grey in 16-bit LE storage.
    Gray12Le,
    /// 16-bit grey LE (also the carrier for `B ∈ {9, 11, 13, 14, 15}`).
    Gray16Le,
    /// 8-bit planar RGB, planes ordered G, B, R (`Nc = 3`, RCT or an
    /// identity-matrix CICP signal).
    Gbrp8,
    /// 10-bit planar GBR, 16-bit LE storage.
    Gbrp10Le,
    /// 12-bit planar GBR, 16-bit LE storage.
    Gbrp12Le,
    /// 14-bit planar GBR, 16-bit LE storage.
    Gbrp14Le,
    /// 16-bit planar GBR LE (carrier for `B ∈ {9, 11, 13, 15}`).
    Gbrp16Le,
    /// 8-bit planar RGB + alpha, planes ordered G, B, R, A (`Nc = 4`,
    /// RCT on the first three, the fourth passed through).
    Gbrap8,
    /// 10-bit planar GBRA, 16-bit LE storage.
    Gbrap10Le,
    /// 12-bit planar GBRA, 16-bit LE storage.
    Gbrap12Le,
    /// 14-bit planar GBRA, 16-bit LE storage.
    Gbrap14Le,
    /// 16-bit planar GBRA LE (carrier for `B ∈ {9, 11, 13, 15}`).
    Gbrap16Le,
    /// 8-bit planar YCbCr 4:4:4 (`Nc = 3`, no colour transform).
    Yuv444P,
    /// 8-bit planar YCbCr 4:2:2.
    Yuv422P,
    /// 8-bit planar YCbCr 4:2:0.
    Yuv420P,
    /// 10-bit planar YCbCr 4:4:4, 16-bit LE storage.
    Yuv444P10Le,
    /// 10-bit planar YCbCr 4:2:2, 16-bit LE storage.
    Yuv422P10Le,
    /// 10-bit planar YCbCr 4:2:0, 16-bit LE storage.
    Yuv420P10Le,
    /// 12-bit planar YCbCr 4:4:4, 16-bit LE storage.
    Yuv444P12Le,
    /// 12-bit planar YCbCr 4:2:2, 16-bit LE storage.
    Yuv422P12Le,
    /// 12-bit planar YCbCr 4:2:0, 16-bit LE storage.
    Yuv420P12Le,
    /// 16-bit planar YCbCr 4:4:4 LE (carrier for `B ∈ {9, 11, 13..=15}`).
    Yuv444P16Le,
    /// 16-bit planar YCbCr 4:2:2 LE (carrier for `B ∈ {9, 11, 13..=15}`).
    Yuv422P16Le,
    /// 16-bit planar YCbCr 4:2:0 LE (carrier for `B ∈ {9, 11, 13..=15}`).
    Yuv420P16Le,
    /// 8-bit planar YCbCr 4:4:4 + full-resolution alpha (`Nc = 4`, no
    /// colour transform).
    Yuva444P,
    /// 8-bit planar YCbCr 4:2:2 + alpha.
    Yuva422P,
    /// 8-bit planar YCbCr 4:2:0 + alpha.
    Yuva420P,
    /// 10-bit planar YCbCr 4:4:4 + alpha, 16-bit LE storage.
    Yuva444P10Le,
    /// 10-bit planar YCbCr 4:2:2 + alpha, 16-bit LE storage.
    Yuva422P10Le,
    /// 10-bit planar YCbCr 4:2:0 + alpha, 16-bit LE storage.
    Yuva420P10Le,
    /// 12-bit planar YCbCr 4:4:4 + alpha, 16-bit LE storage.
    Yuva444P12Le,
    /// 12-bit planar YCbCr 4:2:2 + alpha, 16-bit LE storage.
    Yuva422P12Le,
    /// 12-bit planar YCbCr 4:2:0 + alpha, 16-bit LE storage.
    Yuva420P12Le,
    /// 16-bit planar YCbCr 4:4:4 + alpha LE.
    Yuva444P16Le,
    /// 16-bit planar YCbCr 4:2:2 + alpha LE.
    Yuva422P16Le,
    /// 16-bit planar YCbCr 4:2:0 + alpha LE.
    Yuva420P16Le,
}

/// Contract alias: the crate's native pixel-format tag.
pub type PixelFormat = JpegXsPixelFormat;

/// Colour model of a [`JpegXsPixelFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorModel {
    /// One luminance / grey plane.
    Gray,
    /// Planar RGB (G, B, R plane order), optionally with alpha.
    Rgb,
    /// Planar YCbCr, optionally with alpha.
    YCbCr,
}

impl JpegXsPixelFormat {
    /// Every variant, in declaration order.
    pub const ALL: [JpegXsPixelFormat; 38] = [
        Self::Gray8,
        Self::Gray10Le,
        Self::Gray12Le,
        Self::Gray16Le,
        Self::Gbrp8,
        Self::Gbrp10Le,
        Self::Gbrp12Le,
        Self::Gbrp14Le,
        Self::Gbrp16Le,
        Self::Gbrap8,
        Self::Gbrap10Le,
        Self::Gbrap12Le,
        Self::Gbrap14Le,
        Self::Gbrap16Le,
        Self::Yuv444P,
        Self::Yuv422P,
        Self::Yuv420P,
        Self::Yuv444P10Le,
        Self::Yuv422P10Le,
        Self::Yuv420P10Le,
        Self::Yuv444P12Le,
        Self::Yuv422P12Le,
        Self::Yuv420P12Le,
        Self::Yuv444P16Le,
        Self::Yuv422P16Le,
        Self::Yuv420P16Le,
        Self::Yuva444P,
        Self::Yuva422P,
        Self::Yuva420P,
        Self::Yuva444P10Le,
        Self::Yuva422P10Le,
        Self::Yuva420P10Le,
        Self::Yuva444P12Le,
        Self::Yuva422P12Le,
        Self::Yuva420P12Le,
        Self::Yuva444P16Le,
        Self::Yuva422P16Le,
        Self::Yuva420P16Le,
    ];

    /// Colour model of the layout.
    pub fn color_model(self) -> ColorModel {
        match self {
            Self::Gray8 | Self::Gray10Le | Self::Gray12Le | Self::Gray16Le => ColorModel::Gray,
            Self::Gbrp8
            | Self::Gbrp10Le
            | Self::Gbrp12Le
            | Self::Gbrp14Le
            | Self::Gbrp16Le
            | Self::Gbrap8
            | Self::Gbrap10Le
            | Self::Gbrap12Le
            | Self::Gbrap14Le
            | Self::Gbrap16Le => ColorModel::Rgb,
            _ => ColorModel::YCbCr,
        }
    }

    /// Number of planes a [`JpegXsImage`] in this format carries (1, 3
    /// or 4) — also the codestream component count `Nc`.
    pub fn plane_count(self) -> usize {
        match self.color_model() {
            ColorModel::Gray => 1,
            _ => {
                if self.has_alpha() {
                    4
                } else {
                    3
                }
            }
        }
    }

    /// `true` for the four-plane layouts (`Gbrap*`, `Yuva*`).
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::Gbrap8
                | Self::Gbrap10Le
                | Self::Gbrap12Le
                | Self::Gbrap14Le
                | Self::Gbrap16Le
                | Self::Yuva444P
                | Self::Yuva422P
                | Self::Yuva420P
                | Self::Yuva444P10Le
                | Self::Yuva422P10Le
                | Self::Yuva420P10Le
                | Self::Yuva444P12Le
                | Self::Yuva422P12Le
                | Self::Yuva420P12Le
                | Self::Yuva444P16Le
                | Self::Yuva422P16Le
                | Self::Yuva420P16Le
        )
    }

    /// `true` for the single-plane grey layouts (the only "packed"
    /// layouts JPEG XS has: one sample per pixel, one plane).
    pub fn is_packed(self) -> bool {
        self.plane_count() == 1
    }

    /// Bytes per sample in storage: 1 for the 8-bit layouts, 2 for the
    /// little-endian 16-bit carriers.
    pub fn bytes_per_sample(self) -> usize {
        if self.nominal_bits() > 8 {
            2
        } else {
            1
        }
    }

    /// Sample precision the layout nominally carries (8 / 10 / 12 / 14 /
    /// 16 bits). A picture may hold fewer significant bits than its
    /// carrier — see [`JpegXsImage::bit_depth`].
    pub fn nominal_bits(self) -> u8 {
        match self {
            Self::Gray8
            | Self::Gbrp8
            | Self::Gbrap8
            | Self::Yuv444P
            | Self::Yuv422P
            | Self::Yuv420P
            | Self::Yuva444P
            | Self::Yuva422P
            | Self::Yuva420P => 8,
            Self::Gray10Le
            | Self::Gbrp10Le
            | Self::Gbrap10Le
            | Self::Yuv444P10Le
            | Self::Yuv422P10Le
            | Self::Yuv420P10Le
            | Self::Yuva444P10Le
            | Self::Yuva422P10Le
            | Self::Yuva420P10Le => 10,
            Self::Gray12Le
            | Self::Gbrp12Le
            | Self::Gbrap12Le
            | Self::Yuv444P12Le
            | Self::Yuv422P12Le
            | Self::Yuv420P12Le
            | Self::Yuva444P12Le
            | Self::Yuva422P12Le
            | Self::Yuva420P12Le => 12,
            Self::Gbrp14Le | Self::Gbrap14Le => 14,
            _ => 16,
        }
    }

    /// Chroma subsampling divisors `(horizontal, vertical)` of the
    /// YCbCr layouts; `(1, 1)` for everything else.
    pub fn chroma_divisors(self) -> (usize, usize) {
        match self {
            Self::Yuv422P
            | Self::Yuv422P10Le
            | Self::Yuv422P12Le
            | Self::Yuv422P16Le
            | Self::Yuva422P
            | Self::Yuva422P10Le
            | Self::Yuva422P12Le
            | Self::Yuva422P16Le => (2, 1),
            Self::Yuv420P
            | Self::Yuv420P10Le
            | Self::Yuv420P12Le
            | Self::Yuv420P16Le
            | Self::Yuva420P
            | Self::Yuva420P10Le
            | Self::Yuva420P12Le
            | Self::Yuva420P16Le => (2, 2),
            _ => (1, 1),
        }
    }

    /// `true` for the planar YCbCr layouts (any depth, with or without
    /// alpha).
    pub fn is_yuv(self) -> bool {
        self.color_model() == ColorModel::YCbCr
    }

    /// `true` for the single-channel layouts.
    pub fn is_gray(self) -> bool {
        self.color_model() == ColorModel::Gray
    }

    /// `true` for the planar RGB layouts (with or without alpha).
    pub fn is_rgb(self) -> bool {
        self.color_model() == ColorModel::Rgb
    }

    /// Sample dimensions `(samples per row, rows)` of plane `index` for
    /// a `width × height` picture in this format. Chroma planes of the
    /// subsampled layouts are `ceil(width / h) × ceil(height / v)`
    /// (ISO/IEC 21122-1 Annex B.2, `Wc = ⌈Wf / sx⌉`); every other plane
    /// is full size.
    pub fn plane_dimensions(self, width: u32, height: u32, index: usize) -> (usize, usize) {
        let (w, h) = (width as usize, height as usize);
        if index == 1 || index == 2 {
            let (dh, dv) = self.chroma_divisors();
            (w.div_ceil(dh), h.div_ceil(dv))
        } else {
            (w, h)
        }
    }

    /// Bytes per row of plane `index` when tightly packed.
    pub fn tight_stride(self, width: u32, height: u32, index: usize) -> usize {
        let (w, _) = self.plane_dimensions(width, height, index);
        w * self.bytes_per_sample()
    }

    /// Short lower-case name mirroring the framework's spelling
    /// (`"yuv422p10le"`, `"gray8"`, `"gbrp12le"`, …).
    pub fn name(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Gray10Le => "gray10le",
            Self::Gray12Le => "gray12le",
            Self::Gray16Le => "gray16le",
            Self::Gbrp8 => "gbrp8",
            Self::Gbrp10Le => "gbrp10le",
            Self::Gbrp12Le => "gbrp12le",
            Self::Gbrp14Le => "gbrp14le",
            Self::Gbrp16Le => "gbrp16le",
            Self::Gbrap8 => "gbrap8",
            Self::Gbrap10Le => "gbrap10le",
            Self::Gbrap12Le => "gbrap12le",
            Self::Gbrap14Le => "gbrap14le",
            Self::Gbrap16Le => "gbrap16le",
            Self::Yuv444P => "yuv444p",
            Self::Yuv422P => "yuv422p",
            Self::Yuv420P => "yuv420p",
            Self::Yuv444P10Le => "yuv444p10le",
            Self::Yuv422P10Le => "yuv422p10le",
            Self::Yuv420P10Le => "yuv420p10le",
            Self::Yuv444P12Le => "yuv444p12le",
            Self::Yuv422P12Le => "yuv422p12le",
            Self::Yuv420P12Le => "yuv420p12le",
            Self::Yuv444P16Le => "yuv444p16le",
            Self::Yuv422P16Le => "yuv422p16le",
            Self::Yuv420P16Le => "yuv420p16le",
            Self::Yuva444P => "yuva444p",
            Self::Yuva422P => "yuva422p",
            Self::Yuva420P => "yuva420p",
            Self::Yuva444P10Le => "yuva444p10le",
            Self::Yuva422P10Le => "yuva422p10le",
            Self::Yuva420P10Le => "yuva420p10le",
            Self::Yuva444P12Le => "yuva444p12le",
            Self::Yuva422P12Le => "yuva422p12le",
            Self::Yuva420P12Le => "yuva420p12le",
            Self::Yuva444P16Le => "yuva444p16le",
            Self::Yuva422P16Le => "yuva422p16le",
            Self::Yuva420P16Le => "yuva420p16le",
        }
    }

    /// Pick the layout for a component set: colour model, alpha flag,
    /// chroma divisors `(h, v)` and significant bit depth. Depths without
    /// a label of their own ride the 16-bit carrier. `None` when no
    /// contract layout exists (RGB with chroma subsampling, divisors
    /// other than 1 / 2, grey with alpha).
    pub fn for_layout(
        model: ColorModel,
        alpha: bool,
        divisors: (usize, usize),
        bit_depth: u8,
    ) -> Option<Self> {
        if !(8..=16).contains(&bit_depth) {
            return None;
        }
        match model {
            ColorModel::Gray => {
                if alpha || divisors != (1, 1) {
                    return None;
                }
                Some(match bit_depth {
                    8 => Self::Gray8,
                    10 => Self::Gray10Le,
                    12 => Self::Gray12Le,
                    _ => Self::Gray16Le,
                })
            }
            ColorModel::Rgb => {
                if divisors != (1, 1) {
                    return None;
                }
                Some(match (alpha, bit_depth) {
                    (false, 8) => Self::Gbrp8,
                    (false, 10) => Self::Gbrp10Le,
                    (false, 12) => Self::Gbrp12Le,
                    (false, 14) => Self::Gbrp14Le,
                    (false, _) => Self::Gbrp16Le,
                    (true, 8) => Self::Gbrap8,
                    (true, 10) => Self::Gbrap10Le,
                    (true, 12) => Self::Gbrap12Le,
                    (true, 14) => Self::Gbrap14Le,
                    (true, _) => Self::Gbrap16Le,
                })
            }
            ColorModel::YCbCr => {
                let depth_class = match bit_depth {
                    8 => 0,
                    10 => 1,
                    12 => 2,
                    _ => 3,
                };
                let table: [[Self; 4]; 3] = if alpha {
                    [
                        [
                            Self::Yuva444P,
                            Self::Yuva444P10Le,
                            Self::Yuva444P12Le,
                            Self::Yuva444P16Le,
                        ],
                        [
                            Self::Yuva422P,
                            Self::Yuva422P10Le,
                            Self::Yuva422P12Le,
                            Self::Yuva422P16Le,
                        ],
                        [
                            Self::Yuva420P,
                            Self::Yuva420P10Le,
                            Self::Yuva420P12Le,
                            Self::Yuva420P16Le,
                        ],
                    ]
                } else {
                    [
                        [
                            Self::Yuv444P,
                            Self::Yuv444P10Le,
                            Self::Yuv444P12Le,
                            Self::Yuv444P16Le,
                        ],
                        [
                            Self::Yuv422P,
                            Self::Yuv422P10Le,
                            Self::Yuv422P12Le,
                            Self::Yuv422P16Le,
                        ],
                        [
                            Self::Yuv420P,
                            Self::Yuv420P10Le,
                            Self::Yuv420P12Le,
                            Self::Yuv420P16Le,
                        ],
                    ]
                };
                let sub = match divisors {
                    (1, 1) => 0,
                    (2, 1) => 1,
                    (2, 2) => 2,
                    _ => return None,
                };
                Some(table[sub][depth_class])
            }
        }
    }
}

impl core::fmt::Display for JpegXsPixelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Colour description
// ---------------------------------------------------------------------------

/// Nominal sample range (H.273 `VideoFullRangeFlag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorRange {
    /// No range was signalled.
    #[default]
    Unspecified,
    /// Limited (video / studio) range: `VideoFullRangeFlag == 0`.
    Limited,
    /// Full (PC) range: `VideoFullRangeFlag == 1`.
    Full,
}

/// Colour description: sample range plus the H.273 `ColourPrimaries` /
/// `TransferCharacteristics` / `MatrixCoefficients` code points (raw
/// `u8` values; `2` means "unspecified" for each).
///
/// A bare ISO/IEC 21122-1 codestream carries no colour signalling; a
/// `.jxs` file (ISO/IEC 21122-3 Annex A) carries it in the CICP Colour
/// Specification box, from which this record is filled verbatim. See
/// [`ColorInfo::default_for`] for the convention used when nothing is
/// signalled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ColorInfo {
    /// Nominal sample range.
    pub range: ColorRange,
    /// H.273 `ColourPrimaries` code point.
    pub primaries: u8,
    /// H.273 `TransferCharacteristics` code point.
    pub transfer: u8,
    /// H.273 `MatrixCoefficients` code point (`0` = identity / RGB).
    pub matrix: u8,
}

impl ColorInfo {
    /// H.273 "unspecified" code point.
    pub const UNSPECIFIED: u8 = 2;

    /// Build from a range and three H.273 code points.
    pub const fn new(range: ColorRange, primaries: u8, transfer: u8, matrix: u8) -> Self {
        Self {
            range,
            primaries,
            transfer,
            matrix,
        }
    }

    /// Every field unspecified — identical to `Default`.
    pub const fn unspecified() -> Self {
        Self::new(
            ColorRange::Unspecified,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
        )
    }

    /// sRGB (IEC 61966-2-1): BT.709 primaries (1), sRGB transfer (13),
    /// identity matrix (0), full range. What [`JpegXsImage::from_rgb8`]
    /// / [`JpegXsImage::from_rgba8`] stamp.
    pub const fn srgb() -> Self {
        Self::new(ColorRange::Full, 1, 13, 0)
    }

    /// The convention for a codestream without colour signalling, by
    /// layout: RGB layouts are reported as identity matrix (`0`) with
    /// every other field unspecified (the decoder's inverse RCT output
    /// *is* RGB, nothing more is known); grey and YCbCr layouts are
    /// entirely unspecified. ISO/IEC 21122-1 defines no colour
    /// semantics for a bare codestream, so no primaries / transfer /
    /// range are invented.
    pub fn default_for(format: JpegXsPixelFormat) -> Self {
        match format.color_model() {
            ColorModel::Rgb => Self::new(
                ColorRange::Unspecified,
                Self::UNSPECIFIED,
                Self::UNSPECIFIED,
                0,
            ),
            _ => Self::unspecified(),
        }
    }

    /// `true` when the range is [`ColorRange::Full`].
    pub const fn is_full_range(&self) -> bool {
        matches!(self.range, ColorRange::Full)
    }

    /// `true` when every field is unspecified.
    pub const fn is_unspecified(&self) -> bool {
        matches!(self.range, ColorRange::Unspecified)
            && self.primaries == Self::UNSPECIFIED
            && self.transfer == Self::UNSPECIFIED
            && self.matrix == Self::UNSPECIFIED
    }
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self::unspecified()
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Embedded metadata blobs, as found in (or to be written into) the
/// `.jxs` file boxes.
///
/// * `exif` — the payload of the Exif box in the JPEG XS Header superbox
///   (ISO/IEC 21122-3 A.5.4 Table A.2), opaque bytes.
/// * `icc` / `xmp` — the JXS file format defines CICP colour
///   specification (`METH = 5`) and no XMP box, so these are `None` on
///   decode and ignored on encode; present so the shape matches the
///   other image crates.
/// * `gamma` — never set (JPEG XS signals transfer characteristics as
///   an H.273 code point, see [`ColorInfo::transfer`]).
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Metadata {
    /// ICC profile bytes (never produced by this crate).
    pub icc: Option<Vec<u8>>,
    /// Exif box payload.
    pub exif: Option<Vec<u8>>,
    /// XMP packet (never produced by this crate).
    pub xmp: Option<Vec<u8>>,
    /// Encoding gamma; always `None` for JPEG XS.
    pub gamma: Option<f32>,
}

impl Metadata {
    /// No metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ICC profile (kept on the image; not written by this crate).
    pub fn with_icc(mut self, icc: Vec<u8>) -> Self {
        self.icc = Some(icc);
        self
    }

    /// Set the Exif payload to embed / report.
    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }

    /// Set the XMP packet (kept on the image; not written by this crate).
    pub fn with_xmp(mut self, xmp: Vec<u8>) -> Self {
        self.xmp = Some(xmp);
        self
    }

    /// `true` when no field is set.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none() && self.gamma.is_none()
    }
}

// ---------------------------------------------------------------------------
// The image
// ---------------------------------------------------------------------------

/// One JPEG XS picture in a contract layout.
///
/// `planes` holds one plane for the grey layouts, three for `Gbrp*` /
/// `Yuv*`, four for `Gbrap*` / `Yuva*`, in the **layout's** plane order
/// (G, B, R(, A) for the RGB layouts; Y, Cb, Cr(, A) for YCbCr), each
/// `stride` bytes per row with the sample geometry of
/// [`JpegXsPixelFormat::plane_dimensions`]. Multi-byte samples are
/// little-endian with the value in the low [`bit_depth`](Self::bit_depth)
/// bits. [`JpegXsImage::to_rgb8`] / [`JpegXsImage::to_rgba8`] convert
/// any layout to tightly packed 8-bit RGB(A).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct JpegXsImage {
    /// Picture width in pixels (`Wf`, `1..=65535`).
    pub width: u32,
    /// Picture height in pixels (`Hf`, `1..=65535`).
    pub height: u32,
    /// Native pixel layout of `planes`.
    pub format: JpegXsPixelFormat,
    /// The sample planes, in the layout's plane order.
    pub planes: Vec<Plane>,
    /// Colour description (from the `.jxs` CICP box, or the layout's
    /// documented default).
    pub color: ColorInfo,
    /// Exif payload from the `.jxs` header, if any.
    pub metadata: Metadata,
    /// Significant bits per sample (`B[i]` of the component table,
    /// `8..=16`). Equals [`JpegXsPixelFormat::nominal_bits`] except
    /// when a 16-bit carrier holds a 9 / 11 / 13 / 14 / 15-bit picture.
    pub bit_depth: u8,
}

impl JpegXsImage {
    /// Build an image from its geometry and planes, validating that the
    /// plane count matches `format` and every plane holds exactly
    /// `stride × rows` bytes with `stride ≥` the tight row size of
    /// [`JpegXsPixelFormat::plane_dimensions`]. `color` is the layout's
    /// documented default, `metadata` empty, `bit_depth` the format's
    /// nominal depth.
    pub fn new(
        width: u32,
        height: u32,
        format: JpegXsPixelFormat,
        planes: Vec<Plane>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(JpegXsError::invalid(
                "jpegxs image: width and height must be >= 1",
            ));
        }
        if width > 65535 || height > 65535 {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: {width}x{height} exceeds the 16-bit Wf / Hf picture-header fields"
            )));
        }
        if planes.len() != format.plane_count() {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: {} plane(s) do not fit {format} ({} expected)",
                planes.len(),
                format.plane_count()
            )));
        }
        for (i, p) in planes.iter().enumerate() {
            let (w, h) = format.plane_dimensions(width, height, i);
            let tight = w * format.bytes_per_sample();
            if p.stride < tight {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs image: plane {i} stride {} below the row size {tight}",
                    p.stride
                )));
            }
            let want = p.stride.checked_mul(h).ok_or_else(|| {
                JpegXsError::invalid("jpegxs image: plane byte size overflows usize")
            })?;
            if p.data.len() != want {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs image: plane {i} holds {} bytes, expected stride {} x {h} rows = {want}",
                    p.data.len(),
                    p.stride
                )));
            }
        }
        Ok(Self {
            width,
            height,
            format,
            planes,
            color: ColorInfo::default_for(format),
            metadata: Metadata::new(),
            bit_depth: format.nominal_bits(),
        })
    }

    /// Wrap tightly packed 8-bit RGB (`3 × width` bytes per row) as a
    /// `Gbrp8` image (the planar RGB layout JPEG XS codes natively
    /// through the reversible colour transform) with the sRGB colour
    /// description. `InvalidData` when `data.len() != 3 × width ×
    /// height`.
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let n = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| JpegXsError::invalid("jpegxs image: pixel count overflows usize"))?;
        if data.len() != n * 3 {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: {} bytes do not fit {width}x{height} RGB ({} expected)",
                data.len(),
                n * 3
            )));
        }
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for px in data.chunks_exact(3) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
        }
        let stride = width as usize;
        Ok(Self::new(
            width,
            height,
            JpegXsPixelFormat::Gbrp8,
            vec![
                Plane::new(stride, g),
                Plane::new(stride, b),
                Plane::new(stride, r),
            ],
        )?
        .with_color(ColorInfo::srgb()))
    }

    /// Wrap tightly packed 8-bit RGBA (`4 × width` bytes per row) as a
    /// `Gbrap8` image (planar RGB through the reversible colour
    /// transform plus the alpha plane as the pass-through fourth
    /// component) with the sRGB colour description. `InvalidData` when
    /// `data.len() != 4 × width × height`.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let n = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| JpegXsError::invalid("jpegxs image: pixel count overflows usize"))?;
        if data.len() != n * 4 {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: {} bytes do not fit {width}x{height} RGBA ({} expected)",
                data.len(),
                n * 4
            )));
        }
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        let mut a = Vec::with_capacity(n);
        for px in data.chunks_exact(4) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
            a.push(px[3]);
        }
        let stride = width as usize;
        Ok(Self::new(
            width,
            height,
            JpegXsPixelFormat::Gbrap8,
            vec![
                Plane::new(stride, g),
                Plane::new(stride, b),
                Plane::new(stride, r),
                Plane::new(stride, a),
            ],
        )?
        .with_color(ColorInfo::srgb()))
    }

    /// Replace the colour description.
    pub fn with_color(mut self, color: ColorInfo) -> Self {
        self.color = color;
        self
    }

    /// Replace the metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Override the significant bit depth (`8..=16`) carried by the
    /// planes — needed only when a 16-bit carrier holds fewer
    /// significant bits (e.g. a 14-bit YCbCr picture in `Yuv444P16Le`).
    /// `InvalidData` when the depth is outside `8..=16` or does not fit
    /// the carrier (`> 8` on an 8-bit layout, `≤ 8` on a 16-bit one).
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Result<Self> {
        if !(8..=16).contains(&bit_depth) {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: bit depth {bit_depth} outside 8..=16"
            )));
        }
        let carrier = self.format.bytes_per_sample() * 8;
        if (carrier == 8) != (bit_depth == 8) {
            return Err(JpegXsError::invalid(format!(
                "jpegxs image: bit depth {bit_depth} does not fit the {}-bit storage of {}",
                carrier, self.format
            )));
        }
        self.bit_depth = bit_depth;
        Ok(self)
    }

    /// Picture width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Picture height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Native pixel layout.
    pub fn format(&self) -> JpegXsPixelFormat {
        self.format
    }

    /// The single plane's bytes for the grey layouts; `None` for the
    /// three- and four-plane layouts — use [`into_raw`](Self::into_raw)
    /// or index `planes` directly.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if self.format.is_packed() && self.planes.len() == 1 {
            Some(&self.planes[0].data)
        } else {
            None
        }
    }

    /// Consume the image and return its plane bytes: the single plane
    /// for grey layouts; the planes concatenated in order (each at its
    /// own stride, as reported in `planes[i].stride`) otherwise.
    pub fn into_raw(self) -> Vec<u8> {
        let mut planes = self.planes.into_iter();
        let Some(first) = planes.next() else {
            return Vec::new();
        };
        let mut out = first.data;
        for p in planes {
            out.extend_from_slice(&p.data);
        }
        out
    }

    /// Convert to tightly packed 8-bit RGB (`3 × width` bytes per row),
    /// whatever the native layout — see [`crate::convert`] for the
    /// kernels (deep samples rescaled to 8 bits by rounding, chroma
    /// replicated, YCbCr → RGB with the signalled or default matrix).
    pub fn to_rgb8(&self) -> Vec<u8> {
        crate::convert::to_rgb8(self)
    }

    /// Convert to tightly packed 8-bit RGBA; alpha from the fourth plane
    /// of the `Gbrap*` / `Yuva*` layouts, opaque (`255`) otherwise.
    pub fn to_rgba8(&self) -> Vec<u8> {
        crate::convert::to_rgba8(self)
    }

    /// Sample at `(x, y)` of plane `index` as an integer (any depth).
    /// Out-of-range coordinates read as `0`.
    pub fn sample(&self, index: usize, x: usize, y: usize) -> u32 {
        let Some(p) = self.planes.get(index) else {
            return 0;
        };
        let bps = self.format.bytes_per_sample();
        let off = y * p.stride + x * bps;
        match bps {
            1 => p.data.get(off).copied().map_or(0, u32::from),
            _ => match (p.data.get(off), p.data.get(off + 1)) {
                (Some(&lo), Some(&hi)) => u32::from(lo) | (u32::from(hi) << 8),
                _ => 0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Component planes — the codestream-order depth record
// ---------------------------------------------------------------------------

/// A JPEG XS picture exactly as the codestream carries it: `Nc`
/// component planes in codestream order with the component table's
/// bit depths and sampling factors (ISO/IEC 21122-1 Annex A.4.5) plus
/// the colour-transform id `Cpih`. This is the view every codestream
/// has; [`JpegXsImage`] is derived from it for the layouts the
/// contract names, and the ISO/IEC 21122-4 conformance comparison is
/// made on it (reference images are per-component planes).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Components {
    /// Picture width `Wf`.
    pub width: u32,
    /// Picture height `Hf`.
    pub height: u32,
    /// Colour transformation applied / to apply (Table A.9): `0` none,
    /// `1` reversible RGB↔YCbCr (RCT, Annex F.3) on components 0..3,
    /// `3` Star-Tetrix (Annex F.5) on components 0..4.
    pub cpih: u8,
    /// Bit precision `B[i]` per component (`8..=16`). One byte per
    /// sample when `8`, two little-endian bytes otherwise.
    pub bit_depths: Vec<u8>,
    /// Sampling factors `(sx[i], sy[i])` per component.
    pub sampling: Vec<(u8, u8)>,
    /// The `⌈Wf / sx⌉ × ⌈Hf / sy⌉` sample planes, codestream order,
    /// tight stride.
    pub planes: Vec<Plane>,
}

impl Components {
    /// Build a component set, validating geometry: `1..=8` components,
    /// `Wf` / `Hf` in `1..=65535`, one bit depth and sampling pair per
    /// plane, each plane exactly `⌈Wf / sx⌉ × ⌈Hf / sy⌉` samples at its
    /// bit depth with the tight stride.
    pub fn new(
        width: u32,
        height: u32,
        cpih: u8,
        bit_depths: Vec<u8>,
        sampling: Vec<(u8, u8)>,
        planes: Vec<Plane>,
    ) -> Result<Self> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(JpegXsError::invalid(format!(
                "jpegxs components: {width}x{height} outside the 1..=65535 Wf / Hf range"
            )));
        }
        if !(1..=8).contains(&planes.len()) {
            return Err(JpegXsError::invalid(format!(
                "jpegxs components: Nc must be 1..=8 (Annex A.4.3), got {}",
                planes.len()
            )));
        }
        if bit_depths.len() != planes.len() || sampling.len() != planes.len() {
            return Err(JpegXsError::invalid(format!(
                "jpegxs components: {} bit depth(s) and {} sampling pair(s) for {} plane(s)",
                bit_depths.len(),
                sampling.len(),
                planes.len()
            )));
        }
        for (i, p) in planes.iter().enumerate() {
            let bd = bit_depths[i];
            let (sx, sy) = sampling[i];
            if !(8..=16).contains(&bd) {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs components: component {i} bit depth {bd} outside 8..=16"
                )));
            }
            if sx == 0 || sy == 0 {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs components: component {i} sampling factors must be >= 1"
                )));
            }
            let wc = (width as usize).div_ceil(sx as usize);
            let hc = (height as usize).div_ceil(sy as usize);
            let tight = wc * if bd > 8 { 2 } else { 1 };
            if p.stride != tight {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs components: component {i} stride {} != tight row size {tight}",
                    p.stride
                )));
            }
            if p.data.len() != tight * hc {
                return Err(JpegXsError::invalid(format!(
                    "jpegxs components: component {i} holds {} bytes, expected {}",
                    p.data.len(),
                    tight * hc
                )));
            }
        }
        Ok(Self {
            width,
            height,
            cpih,
            bit_depths,
            sampling,
            planes,
        })
    }

    /// Number of components `Nc`.
    pub fn len(&self) -> usize {
        self.planes.len()
    }

    /// `true` when there are no components (never for a decoded set).
    pub fn is_empty(&self) -> bool {
        self.planes.is_empty()
    }

    /// Component count as the picture header's `Nc` byte.
    pub fn num_components(&self) -> u8 {
        self.planes.len() as u8
    }

    /// Largest component bit depth.
    pub fn max_bit_depth(&self) -> u8 {
        self.bit_depths.iter().copied().max().unwrap_or(0)
    }

    /// `true` when every component shares one bit depth.
    pub fn uniform_bit_depth(&self) -> bool {
        self.bit_depths.windows(2).all(|w| w[0] == w[1])
    }
}

// ---------------------------------------------------------------------------
// Raw RGB / RGBA results
// ---------------------------------------------------------------------------

/// Tightly packed 8-bit RGB: `3 × width` bytes per row, row-major, no
/// padding. Identical definition in every OxideAV image crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 3` bytes, R G B per pixel.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Wrap a buffer (expected `width × height × 3` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

/// Tightly packed 8-bit RGBA: `4 × width` bytes per row, row-major, no
/// padding. Identical definition in every OxideAV image crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width × height × 4` bytes, R G B A per pixel.
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Wrap a buffer (expected `width × height × 4` bytes).
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume into the pixel buffer.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Header summary
// ---------------------------------------------------------------------------

/// What [`crate::info`] reports without decoding any sample: the
/// picture header, component table and (for a `.jxs` file) the header
/// boxes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageInfo {
    /// Picture width (`Wf`).
    pub width: u32,
    /// Picture height (`Hf`).
    pub height: u32,
    /// The layout [`crate::decode`] will produce for this stream.
    pub format: JpegXsPixelFormat,
    /// Number of pictures; always `1` (a codestream is one picture).
    pub frames: u32,
    /// `true` for the four-component layouts (`Gbrap*` / `Yuva*`).
    pub has_alpha: bool,
    /// Colour description as the decoder will report it.
    pub color: ColorInfo,
    /// Always `false`: the JXS file format carries CICP, not ICC.
    pub has_icc: bool,
    /// An Exif box is present in the `.jxs` header.
    pub has_exif: bool,
    /// Always `false`: the JXS file format has no XMP box.
    pub has_xmp: bool,
    /// Significant bits per sample (`B[i]`, `8..=16`).
    pub bit_depth: u8,
    /// Component count `Nc`.
    pub components: u8,
    /// Colour transformation id `Cpih` (Table A.9).
    pub cpih: u8,
    /// Declared profile `Ppih` (`0` = unrestricted / undeclared;
    /// ISO/IEC 21122-2 Table A.5).
    pub profile: u16,
    /// Declared level + sublevel `Plev` (`0` = undeclared).
    pub level: u16,
    /// `true` when the picture is coded losslessly (`Fq == 0`, Table
    /// A.8, integer transform path with `q = 0` quantisation).
    pub lossless: bool,
    /// `true` when the input is a box-wrapped `.jxs` file (ISO/IEC
    /// 21122-3 Annex A) rather than a bare codestream.
    pub boxed: bool,
}

// ---------------------------------------------------------------------------
// Decode options
// ---------------------------------------------------------------------------

/// Limits and strictness for [`crate::decode_with`].
///
/// Limits are checked against the picture header **before** any sample
/// buffer is allocated; a breach is [`JpegXsError::LimitExceeded`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Largest accepted width; `None` = unlimited (default `65535`, the
    /// `Wf` field maximum).
    pub max_width: Option<u32>,
    /// Largest accepted height; `None` = unlimited (default `65535`).
    pub max_height: Option<u32>,
    /// Largest accepted `width × height`; `None` = unlimited (default
    /// `1 << 28`, 268 Mpixel — about 1 GiB of decoded 16-bit 4:4:4
    /// samples).
    pub max_pixels: Option<u64>,
    /// Largest accepted input length in bytes; `None` = unlimited (the
    /// default).
    pub max_bytes: Option<u64>,
    /// Strict mode: reject bytes after the `EOC` marker of a bare
    /// codestream and a missing `EOC`. Default `false` (trailing bytes
    /// are ignored).
    pub strict: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(65535),
            max_height: Some(65535),
            max_pixels: Some(1 << 28),
            max_bytes: None,
            strict: false,
        }
    }
}

impl DecodeOptions {
    /// The defaults (see the field docs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the accepted width (`None` = unlimited).
    pub fn with_max_width(mut self, max_width: Option<u32>) -> Self {
        self.max_width = max_width;
        self
    }

    /// Cap the accepted height (`None` = unlimited).
    pub fn with_max_height(mut self, max_height: Option<u32>) -> Self {
        self.max_height = max_height;
        self
    }

    /// Cap the accepted pixel count (`None` = unlimited).
    pub fn with_max_pixels(mut self, max_pixels: Option<u64>) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    /// Cap the accepted input length (`None` = unlimited).
    pub fn with_max_bytes(mut self, max_bytes: Option<u64>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Enable / disable strict mode.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_dimensions_follow_annex_b2() {
        assert_eq!(
            JpegXsPixelFormat::Yuv420P.plane_dimensions(33, 17, 1),
            (17, 9)
        );
        assert_eq!(
            JpegXsPixelFormat::Yuv422P10Le.plane_dimensions(33, 17, 2),
            (17, 17)
        );
        assert_eq!(
            JpegXsPixelFormat::Yuva422P.plane_dimensions(33, 17, 3),
            (33, 17)
        );
        assert_eq!(
            JpegXsPixelFormat::Gbrp12Le.plane_dimensions(33, 17, 1),
            (33, 17)
        );
        assert_eq!(JpegXsPixelFormat::Gray16Le.tight_stride(33, 17, 0), 66);
        assert_eq!(JpegXsPixelFormat::Yuv420P12Le.tight_stride(33, 17, 1), 34);
    }

    #[test]
    fn format_table_is_consistent() {
        for f in JpegXsPixelFormat::ALL {
            assert_eq!(f.to_string(), f.name());
            let model = f.color_model();
            let alpha = f.has_alpha();
            let div = f.chroma_divisors();
            assert_eq!(
                JpegXsPixelFormat::for_layout(model, alpha, div, f.nominal_bits()),
                Some(f),
                "{f} round-trips through for_layout"
            );
            assert_eq!(
                f.plane_count(),
                if model == ColorModel::Gray {
                    1
                } else if alpha {
                    4
                } else {
                    3
                }
            );
            assert_eq!(
                f.bytes_per_sample(),
                if f.nominal_bits() > 8 { 2 } else { 1 }
            );
        }
        // Odd depths ride the 16-bit carrier.
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::Gray, false, (1, 1), 9),
            Some(JpegXsPixelFormat::Gray16Le)
        );
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::Rgb, true, (1, 1), 13),
            Some(JpegXsPixelFormat::Gbrap16Le)
        );
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::YCbCr, false, (2, 1), 14),
            Some(JpegXsPixelFormat::Yuv422P16Le)
        );
        // No contract view.
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::Rgb, false, (2, 1), 8),
            None
        );
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::YCbCr, false, (1, 2), 8),
            None
        );
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::Gray, true, (1, 1), 8),
            None
        );
        assert_eq!(
            JpegXsPixelFormat::for_layout(ColorModel::Gray, false, (1, 1), 7),
            None
        );
    }

    #[test]
    fn constructors_validate_geometry() {
        let img = JpegXsImage::from_rgb8(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(img.format, JpegXsPixelFormat::Gbrp8);
        assert_eq!(img.planes[0].data, vec![2, 5]); // G
        assert_eq!(img.planes[1].data, vec![3, 6]); // B
        assert_eq!(img.planes[2].data, vec![1, 4]); // R
        assert_eq!(img.color, ColorInfo::srgb());
        assert_eq!(img.bit_depth, 8);
        assert!(img.as_bytes().is_none());
        assert!(JpegXsImage::from_rgb8(2, 1, vec![0; 5]).is_err());
        let rgba = JpegXsImage::from_rgba8(1, 1, vec![9, 8, 7, 6]).unwrap();
        assert_eq!(rgba.format, JpegXsPixelFormat::Gbrap8);
        assert_eq!(rgba.planes[3].data, vec![6]);
        assert!(JpegXsImage::from_rgba8(1, 1, vec![0; 3]).is_err());
        // new(): plane count, stride, size.
        assert!(JpegXsImage::new(2, 2, JpegXsPixelFormat::Gray8, vec![]).is_err());
        assert!(JpegXsImage::new(
            2,
            2,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(1, vec![0; 2])]
        )
        .is_err());
        assert!(JpegXsImage::new(
            2,
            2,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(2, vec![0; 3])]
        )
        .is_err());
        assert!(JpegXsImage::new(0, 2, JpegXsPixelFormat::Gray8, vec![]).is_err());
        assert!(JpegXsImage::new(70000, 2, JpegXsPixelFormat::Gray8, vec![]).is_err());
        let gray = JpegXsImage::new(
            2,
            2,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(3, vec![0; 6])],
        )
        .unwrap();
        assert_eq!(gray.as_bytes().map(<[u8]>::len), Some(6));
        assert_eq!(gray.color, ColorInfo::unspecified());
        // Chroma planes at their subsampled size.
        let yuv = JpegXsImage::new(
            3,
            3,
            JpegXsPixelFormat::Yuv420P,
            vec![
                Plane::new(3, vec![0; 9]),
                Plane::new(2, vec![0; 4]),
                Plane::new(2, vec![0; 4]),
            ],
        )
        .unwrap();
        assert_eq!(yuv.clone().into_raw().len(), 17);
        assert_eq!(yuv.color, ColorInfo::unspecified());
        let rgb16 = JpegXsImage::new(
            1,
            1,
            JpegXsPixelFormat::Gbrp16Le,
            vec![
                Plane::new(2, vec![0; 2]),
                Plane::new(2, vec![0; 2]),
                Plane::new(2, vec![0; 2]),
            ],
        )
        .unwrap();
        assert_eq!(rgb16.color.matrix, 0);
        assert!(rgb16.clone().with_bit_depth(9).is_ok());
        assert!(rgb16.clone().with_bit_depth(8).is_err());
        assert!(rgb16.with_bit_depth(17).is_err());
        assert!(gray.clone().with_bit_depth(10).is_err());
        assert!(gray.with_bit_depth(8).is_ok());
    }

    #[test]
    fn components_validate() {
        let c = Components::new(
            3,
            2,
            0,
            vec![8, 10],
            vec![(1, 1), (2, 1)],
            vec![Plane::new(3, vec![0; 6]), Plane::new(4, vec![0; 8])],
        )
        .unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.num_components(), 2);
        assert_eq!(c.max_bit_depth(), 10);
        assert!(!c.uniform_bit_depth());
        assert!(Components::new(3, 2, 0, vec![], vec![], vec![]).is_err());
        assert!(Components::new(
            3,
            2,
            0,
            vec![8],
            vec![(1, 1)],
            vec![Plane::new(4, vec![0; 8])]
        )
        .is_err());
        assert!(Components::new(
            3,
            2,
            0,
            vec![7],
            vec![(1, 1)],
            vec![Plane::new(3, vec![0; 6])]
        )
        .is_err());
        assert!(Components::new(
            3,
            2,
            0,
            vec![8, 8],
            vec![(1, 1)],
            vec![Plane::new(3, vec![0; 6])]
        )
        .is_err());
    }

    #[test]
    fn decode_options_defaults_and_builders() {
        let d = DecodeOptions::default();
        assert_eq!(d.max_width, Some(65535));
        assert_eq!(d.max_pixels, Some(1 << 28));
        assert_eq!(d.max_bytes, None);
        assert!(!d.strict);
        let o = DecodeOptions::new()
            .with_max_width(Some(10))
            .with_max_height(None)
            .with_max_pixels(Some(12))
            .with_max_bytes(Some(13))
            .with_strict(true);
        assert_eq!(
            (
                o.max_width,
                o.max_height,
                o.max_pixels,
                o.max_bytes,
                o.strict
            ),
            (Some(10), None, Some(12), Some(13), true)
        );
    }

    #[test]
    fn color_presets() {
        assert_eq!(ColorInfo::default(), ColorInfo::unspecified());
        assert!(ColorInfo::srgb().is_full_range());
        assert!(ColorInfo::unspecified().is_unspecified());
        assert!(!ColorInfo::default_for(JpegXsPixelFormat::Gbrp8).is_unspecified());
        assert!(ColorInfo::default_for(JpegXsPixelFormat::Yuv422P10Le).is_unspecified());
        assert!(Metadata::new().is_empty());
        assert!(!Metadata::new().with_exif(vec![0]).is_empty());
        assert!(!Metadata::new().with_icc(vec![0]).is_empty());
        assert!(!Metadata::new().with_xmp(vec![0]).is_empty());
    }
}
