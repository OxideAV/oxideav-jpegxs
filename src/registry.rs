//! `oxideav-core` integration: the framework `Decoder` / `Encoder`
//! adapters, the [`make_decoder`] / [`make_encoder`] factories, the
//! frame bridge (`From<JpegXsImage> for VideoFrame`,
//! [`JpegXsImage::from_video_frame`]), the pixel-format / colour-signal
//! / error mappings, and the [`register`] / [`register_codecs`] /
//! [`register_containers`] entry points.
//!
//! Gated behind the default-on `registry` Cargo feature. The adapters
//! call the standalone functions ([`crate::decode_with`],
//! [`crate::encode`]) — one implementation, two entry layers. With the
//! feature off the whole standalone surface is still available with no
//! `oxideav-core` in the dependency tree.
//!
//! The framework boundary is the **native layout**: the `Decoder` emits
//! grey / planar GBR(A) / planar YCbCr(A) frames in the picture's own
//! bit depth, with the colour-signal side-channel stamped only when the
//! input is a `.jxs` file carrying a CICP box (a bare codestream has no
//! colour signalling, so none is invented). The `Encoder` accepts every
//! layout the pixel-format mapping names plus packed `Rgb24` / `Rgba`
//! frames (deplaned to `Gbrp8` / `Gbrap8`, the raw-path rule).

use std::collections::VecDeque;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters, CodecRegistry,
    ColorPrimaries, ColorSignal, ContainerRegistry, Decoder, Encoder, Error, Frame,
    MatrixCoefficients, OptionField, OptionKind, OptionValue, Packet, PixelFormat, Result,
    RuntimeContext, TimeBase, TransferCharacteristics, VideoFrame, VideoPlane,
};

use crate::error::JpegXsError;
use crate::image::{ColorInfo, ColorRange, JpegXsImage, JpegXsPixelFormat, Plane};
use crate::options::{EncodeOptions, Quantizer, RunMode, Weights};
use crate::profile::Profile;
use crate::CODEC_ID_STR;

// ---- error mapping --------------------------------------------------------

impl From<JpegXsError> for Error {
    fn from(e: JpegXsError) -> Self {
        match e {
            JpegXsError::InvalidData(s) => Error::InvalidData(s),
            JpegXsError::Unsupported(s) => Error::Unsupported(s),
            JpegXsError::LimitExceeded(s) => Error::ResourceExhausted(s),
            JpegXsError::Io(e) => Error::Io(e),
        }
    }
}

// ---- pixel formats --------------------------------------------------------

/// The 1:1 name mapping from [`JpegXsPixelFormat`] to the framework enum.
pub fn to_core_pixel_format(pf: JpegXsPixelFormat) -> PixelFormat {
    use JpegXsPixelFormat as J;
    match pf {
        J::Gray8 => PixelFormat::Gray8,
        J::Gray10Le => PixelFormat::Gray10Le,
        J::Gray12Le => PixelFormat::Gray12Le,
        J::Gray16Le => PixelFormat::Gray16Le,
        J::Gbrp8 => PixelFormat::Gbrp8,
        J::Gbrp10Le => PixelFormat::Gbrp10Le,
        J::Gbrp12Le => PixelFormat::Gbrp12Le,
        J::Gbrp14Le => PixelFormat::Gbrp14Le,
        J::Gbrp16Le => PixelFormat::Gbrp16Le,
        J::Gbrap8 => PixelFormat::Gbrap8,
        J::Gbrap10Le => PixelFormat::Gbrap10Le,
        J::Gbrap12Le => PixelFormat::Gbrap12Le,
        J::Gbrap14Le => PixelFormat::Gbrap14Le,
        J::Gbrap16Le => PixelFormat::Gbrap16Le,
        J::Yuv444P => PixelFormat::Yuv444P,
        J::Yuv422P => PixelFormat::Yuv422P,
        J::Yuv420P => PixelFormat::Yuv420P,
        J::Yuv444P10Le => PixelFormat::Yuv444P10Le,
        J::Yuv422P10Le => PixelFormat::Yuv422P10Le,
        J::Yuv420P10Le => PixelFormat::Yuv420P10Le,
        J::Yuv444P12Le => PixelFormat::Yuv444P12Le,
        J::Yuv422P12Le => PixelFormat::Yuv422P12Le,
        J::Yuv420P12Le => PixelFormat::Yuv420P12Le,
        J::Yuv444P16Le => PixelFormat::Yuv444P16Le,
        J::Yuv422P16Le => PixelFormat::Yuv422P16Le,
        J::Yuv420P16Le => PixelFormat::Yuv420P16Le,
        J::Yuva444P => PixelFormat::Yuva444P,
        J::Yuva422P => PixelFormat::Yuva422P,
        J::Yuva420P => PixelFormat::Yuva420P,
        J::Yuva444P10Le => PixelFormat::Yuva444P10Le,
        J::Yuva422P10Le => PixelFormat::Yuva422P10Le,
        J::Yuva420P10Le => PixelFormat::Yuva420P10Le,
        J::Yuva444P12Le => PixelFormat::Yuva444P12Le,
        J::Yuva422P12Le => PixelFormat::Yuva422P12Le,
        J::Yuva420P12Le => PixelFormat::Yuva420P12Le,
        J::Yuva444P16Le => PixelFormat::Yuva444P16Le,
        J::Yuva422P16Le => PixelFormat::Yuva422P16Le,
        J::Yuva420P16Le => PixelFormat::Yuva420P16Le,
    }
}

/// Map a framework pixel format to [`JpegXsPixelFormat`]; `Err` for
/// layouts JPEG XS does not carry (packed RGB, interleaved YUV, palette,
/// float, …).
pub fn from_core_pixel_format(
    pf: PixelFormat,
) -> std::result::Result<JpegXsPixelFormat, JpegXsError> {
    JpegXsPixelFormat::ALL
        .iter()
        .copied()
        .find(|&j| to_core_pixel_format(j) == pf)
        .ok_or_else(|| {
            JpegXsError::unsupported(format!(
                "jpegxs: pixel format {pf:?} has no JPEG XS layout (grey, planar GBR(A) and \
                 planar YCbCr(A) at 8..16 bits)"
            ))
        })
}

impl From<JpegXsPixelFormat> for PixelFormat {
    fn from(pf: JpegXsPixelFormat) -> Self {
        to_core_pixel_format(pf)
    }
}

impl TryFrom<PixelFormat> for JpegXsPixelFormat {
    type Error = JpegXsError;
    fn try_from(pf: PixelFormat) -> std::result::Result<Self, JpegXsError> {
        from_core_pixel_format(pf)
    }
}

// ---- colour signalling ----------------------------------------------------

/// [`ColorInfo`] as the framework's [`ColorSignal`] (code points map 1:1).
pub fn to_color_signal(c: &ColorInfo) -> ColorSignal {
    let range = match c.range {
        ColorRange::Unspecified => oxideav_core::ColorRange::Unspecified,
        ColorRange::Limited => oxideav_core::ColorRange::Limited,
        ColorRange::Full => oxideav_core::ColorRange::Full,
    };
    ColorSignal::new(
        range,
        ColorPrimaries(c.primaries),
        TransferCharacteristics(c.transfer),
        MatrixCoefficients(c.matrix),
    )
}

/// The inverse of [`to_color_signal`].
pub fn from_color_signal(s: &ColorSignal) -> ColorInfo {
    let range = match s.range {
        oxideav_core::ColorRange::Limited => ColorRange::Limited,
        oxideav_core::ColorRange::Full => ColorRange::Full,
        _ => ColorRange::Unspecified,
    };
    ColorInfo::new(range, s.primaries.0, s.transfer.0, s.matrix.0)
}

// ---- frame bridge ---------------------------------------------------------

/// [`JpegXsImage`] → `VideoFrame`, moving the planes out of the image.
/// The colour signal is stamped only when `stamp_color` is set (the
/// decoder passes `true` for a `.jxs` file with a CICP box).
pub(crate) fn image_into_video_frame(
    image: JpegXsImage,
    pts: Option<i64>,
    stamp_color: bool,
) -> VideoFrame {
    let color = image.color;
    let planes = image
        .planes
        .into_iter()
        .map(|p| VideoPlane {
            stride: p.stride,
            data: p.data,
        })
        .collect();
    let mut frame = VideoFrame { pts, planes };
    if stamp_color {
        frame.set_color_signal(to_color_signal(&color));
    }
    frame
}

impl From<JpegXsImage> for VideoFrame {
    /// The planes (`pts` `None`), colour signal stamped when the image's
    /// [`ColorInfo`] specifies anything.
    fn from(image: JpegXsImage) -> Self {
        let stamp = !image.color.is_unspecified();
        image_into_video_frame(image, None, stamp)
    }
}

impl From<&JpegXsImage> for VideoFrame {
    fn from(image: &JpegXsImage) -> Self {
        VideoFrame::from(image.clone())
    }
}

/// Copy a frame plane into a tight-stride buffer of `rows × tight` bytes.
fn tight_plane(
    plane: &VideoPlane,
    tight: usize,
    rows: usize,
) -> std::result::Result<Plane, JpegXsError> {
    if plane.stride < tight {
        return Err(JpegXsError::invalid(format!(
            "jpegxs: frame plane stride {} below the row size {tight}",
            plane.stride
        )));
    }
    let mut data = Vec::with_capacity(tight * rows);
    for y in 0..rows {
        let start = y * plane.stride;
        let row = plane.data.get(start..start + tight).ok_or_else(|| {
            JpegXsError::invalid("jpegxs: frame plane too short for its geometry")
        })?;
        data.extend_from_slice(row);
    }
    Ok(Plane::new(tight, data))
}

impl JpegXsImage {
    /// Rebuild an image from a framework frame and the stream parameters
    /// that describe it: `width`, `height` and `pixel_format` are
    /// required. Frames in any [`JpegXsPixelFormat`] layout are taken as
    /// is (padded strides repacked); packed `Rgb24` / `Rgba` frames are
    /// deplaned to `Gbrp8` / `Gbrap8` (the raw-path rule); any other
    /// layout is `Unsupported`. The frame's colour-signal side-channel,
    /// refined over `params.color_signal`, becomes `color` when it
    /// specifies anything; otherwise the layout's documented default.
    pub fn from_video_frame(
        frame: &VideoFrame,
        params: &CodecParameters,
    ) -> std::result::Result<Self, JpegXsError> {
        let width = params
            .width
            .ok_or_else(|| JpegXsError::invalid("jpegxs: width missing in CodecParameters"))?;
        let height = params
            .height
            .ok_or_else(|| JpegXsError::invalid("jpegxs: height missing in CodecParameters"))?;
        let pf = params.pixel_format.ok_or_else(|| {
            JpegXsError::invalid("jpegxs: pixel_format missing in CodecParameters")
        })?;
        let planes = frame.image_planes();
        let mut img = match pf {
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let bpp = if pf == PixelFormat::Rgb24 { 3 } else { 4 };
                let plane = planes
                    .first()
                    .ok_or_else(|| JpegXsError::invalid("jpegxs: frame has no planes"))?;
                let tight = tight_plane(plane, width as usize * bpp, height as usize)?;
                if bpp == 3 {
                    JpegXsImage::from_rgb8(width, height, tight.data)?
                } else {
                    JpegXsImage::from_rgba8(width, height, tight.data)?
                }
            }
            other => {
                let format = from_core_pixel_format(other)?;
                if planes.len() != format.plane_count() {
                    return Err(JpegXsError::invalid(format!(
                        "jpegxs: {} frame plane(s) do not fit {format} ({} expected)",
                        planes.len(),
                        format.plane_count()
                    )));
                }
                let mut tight = Vec::with_capacity(planes.len());
                for (i, p) in planes.iter().enumerate() {
                    let (w, h) = format.plane_dimensions(width, height, i);
                    tight.push(tight_plane(p, w * format.bytes_per_sample(), h)?);
                }
                JpegXsImage::new(width, height, format, tight)?
            }
        };
        let sig = frame
            .color_signal()
            .unwrap_or_default()
            .or(params.color_signal);
        if !sig.is_unspecified() {
            img.color = from_color_signal(&sig);
        }
        Ok(img)
    }
}

impl TryFrom<(&VideoFrame, &CodecParameters)> for JpegXsImage {
    type Error = JpegXsError;
    fn try_from(
        (frame, params): (&VideoFrame, &CodecParameters),
    ) -> std::result::Result<Self, JpegXsError> {
        JpegXsImage::from_video_frame(frame, params)
    }
}

// ---- options schema -------------------------------------------------------

const PROFILE_NAMES: &[&str] = &[
    "light422_10",
    "light444_12",
    "light_subline422_10",
    "main422_10",
    "main444_12",
    "main4444_12",
    "high444_12",
    "high4444_12",
];

fn profile_from_name(name: &str) -> Option<Profile> {
    Some(match name {
        "light422_10" => Profile::Light422_10,
        "light444_12" => Profile::Light444_12,
        "light_subline422_10" => Profile::LightSubline422_10,
        "main422_10" => Profile::Main422_10,
        "main444_12" => Profile::Main444_12,
        "main4444_12" => Profile::Main4444_12,
        "high444_12" => Profile::High444_12,
        "high4444_12" => Profile::High4444_12,
        _ => return None,
    })
}

/// The framework's options schema for the JPEG XS encoder — what makes
/// the knobs discoverable to `oxideav list`, validatable by the
/// pipeline's JSON-options checker, and parsed with uniform error
/// messages.
///
/// Keys: `quantization` (u32 `0..=15`), `target_bytes` (u32, CBR size),
/// `profile` (enum of ISO/IEC 21122-2 profile names), `levels_x` /
/// `levels_y` (u32), `rct` (bool), `quantizer` (`deadzone` /
/// `uniform`), `slice_height` (u32 precinct rows), `run_mode`
/// (`zero_residuals` / `zero_coefficients`), `weights` (`default` /
/// `annex_h`), `high_precision` (bool), `boxed` (bool).
impl CodecOptionsStruct for EncodeOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "quantization",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Precinct quantisation step Q[p] 0..=15 (0 = lossless).",
        },
        OptionField {
            name: "target_bytes",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Constant-bitrate target: exact codestream size in bytes (0 = variable bitrate).",
        },
        OptionField {
            name: "profile",
            kind: OptionKind::Enum(PROFILE_NAMES),
            default: OptionValue::String(String::new()),
            help: "ISO/IEC 21122-2 profile to shape to and declare (Ppih / Plev).",
        },
        OptionField {
            name: "levels_x",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Horizontal wavelet decomposition levels NL,x 1..=8 (0 = automatic).",
        },
        OptionField {
            name: "levels_y",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Vertical wavelet decomposition levels NL,y 0..=NL,x (absent = automatic).",
        },
        OptionField {
            name: "rct",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "Apply the reversible colour transform to RGB layouts.",
        },
        OptionField {
            name: "quantizer",
            kind: OptionKind::Enum(&["deadzone", "uniform"]),
            default: OptionValue::String(String::new()),
            help: "Inverse-quantiser type to signal (Qpih): deadzone (default) or uniform.",
        },
        OptionField {
            name: "slice_height",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Slice height Hsl in precinct rows (0 = one slice).",
        },
        OptionField {
            name: "run_mode",
            kind: OptionKind::Enum(&["zero_residuals", "zero_coefficients"]),
            default: OptionValue::String(String::new()),
            help: "Run mode Rm: zero_residuals (default) or zero_coefficients.",
        },
        OptionField {
            name: "weights",
            kind: OptionKind::Enum(&["default", "annex_h"]),
            default: OptionValue::String(String::new()),
            help: "Band weights: default or the Annex H PSNR-optimised tables.",
        },
        OptionField {
            name: "high_precision",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "High-precision regular path (Bw = 20, Fq = 8).",
        },
        OptionField {
            name: "boxed",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "Emit a .jxs still-image file instead of a bare codestream.",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "quantization" => {
                let q = value.as_u32()?;
                if q > 15 {
                    return Err(Error::invalid(format!(
                        "jpegxs encoder: quantization {q} outside 0..=15"
                    )));
                }
                self.quantization = q as u8;
            }
            "target_bytes" => {
                let t = value.as_u32()?;
                self.target_bytes = (t != 0).then_some(t as usize);
            }
            "profile" => {
                let name = value.as_str()?;
                self.profile = Some(profile_from_name(name).ok_or_else(|| {
                    Error::invalid(format!("jpegxs encoder: unknown profile {name:?}"))
                })?);
            }
            "levels_x" => {
                let n = value.as_u32()?;
                self.levels_x = (n != 0).then_some(n.min(255) as u8);
            }
            "levels_y" => {
                self.levels_y = Some(value.as_u32()?.min(255) as u8);
            }
            "rct" => self.rct = value.as_bool()?,
            "quantizer" => {
                self.quantizer = match value.as_str()? {
                    "deadzone" => Quantizer::Deadzone,
                    "uniform" => Quantizer::Uniform,
                    other => {
                        return Err(Error::invalid(format!(
                            "jpegxs encoder: invalid quantizer {other:?}"
                        )))
                    }
                }
            }
            "slice_height" => {
                let h = value.as_u32()?;
                self.slice_height = (h != 0).then_some(h.min(65535) as u16);
            }
            "run_mode" => {
                self.run_mode = match value.as_str()? {
                    "zero_residuals" => RunMode::ZeroResiduals,
                    "zero_coefficients" => RunMode::ZeroCoefficients,
                    other => {
                        return Err(Error::invalid(format!(
                            "jpegxs encoder: invalid run_mode {other:?}"
                        )))
                    }
                }
            }
            "weights" => {
                self.weights = match value.as_str()? {
                    "default" => Weights::Default,
                    "annex_h" => Weights::AnnexH,
                    other => {
                        return Err(Error::invalid(format!(
                            "jpegxs encoder: invalid weights {other:?}"
                        )))
                    }
                }
            }
            "high_precision" => self.high_precision = value.as_bool()?,
            "boxed" => self.boxed = value.as_bool()?,
            other => {
                return Err(Error::invalid(format!(
                    "jpegxs encoder: unknown option {other:?}"
                )))
            }
        }
        Ok(())
    }
}

// ---- registration ---------------------------------------------------------

/// Register the JPEG XS codec (decoder + encoder) into the supplied
/// [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxs_sw")
        .with_lossy(true)
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_size(65535, 65535)
        .with_pixel_formats(
            JpegXsPixelFormat::ALL
                .iter()
                .map(|&f| to_core_pixel_format(f))
                .chain([PixelFormat::Rgb24, PixelFormat::Rgba])
                .collect(),
        );
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<EncodeOptions>(),
    );
}

/// Register JPEG XS file extensions into the supplied [`ContainerRegistry`].
///
/// A `.jxs` file may be either a bare ISO/IEC 21122-1 codestream (SOC
/// marker first) or the box-based JXS still-image file format of ISO/IEC
/// 21122-3 Annex A (JPEG XS Signature box first); the decoder accepts
/// both, routing on the leading signature. No standalone demuxer is
/// registered — the codec's `Decoder` unwraps the box layer itself. We
/// register the canonical `.jxs` extension against the codec id
/// `"jpegxs"` so a caller resolving a path hint via
/// [`ContainerRegistry::container_for_extension`] still gets a useful
/// answer (lookups are case-insensitive).
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("jxs", CODEC_ID_STR);
}

/// Register codecs and containers into two separate registries.
pub fn register_registries(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-jpegxs` into a [`RuntimeContext`]. Also wired into
/// `oxideav_meta::register_all` via the [`oxideav_core::register!`]
/// macro.
pub fn register(ctx: &mut RuntimeContext) {
    register_registries(&mut ctx.codecs, &mut ctx.containers);
}

oxideav_core::register!("jpegxs", register);

// ---- decoder --------------------------------------------------------------

/// Decoder factory: one packet = one codestream (or `.jxs` file), one
/// frame in the native layout.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(JpegXsDecoder {
        codec_id: params.codec_id.clone(),
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
        // Only a `.jxs` file carries colour signalling (its CICP box);
        // a bare codestream gets no colour signal stamped.
        let stamp = crate::fileformat::is_jxs_file(&pkt.data);
        let img = crate::decode_with(&pkt.data, &crate::DecodeOptions::default())?;
        Ok(Frame::Video(image_into_video_frame(img, pkt.pts, stamp)))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- encoder --------------------------------------------------------------

/// Encoder factory: `params.width` / `height` / `pixel_format` describe
/// the incoming frames; `params.options` map onto [`EncodeOptions`]
/// (see the schema). One frame in, one keyframe packet out.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let options: EncodeOptions = oxideav_core::parse_options(&params.options)?;
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = params.width;
    out_params.height = params.height;
    out_params.pixel_format = params.pixel_format;
    out_params.color_signal = params.color_signal;
    Ok(Box::new(JpegXsEncoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        in_params: params.clone(),
        out_params,
        options,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct JpegXsEncoder {
    codec_id: CodecId,
    in_params: CodecParameters,
    out_params: CodecParameters,
    options: EncodeOptions,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for JpegXsEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let vf = match frame {
            Frame::Video(v) => v,
            _ => return Err(Error::invalid("jpegxs encoder: expected a video frame")),
        };
        let img = JpegXsImage::from_video_frame(vf, &self.in_params)?;
        let bytes = crate::encode(&img, &self.options)?;
        let mut pkt = Packet::new(0, TimeBase::new(1, 1), bytes);
        pkt.pts = vf.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        match self.pending.pop_front() {
            Some(pkt) => Ok(pkt),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fileformat::{
        BRAND_JXS, COLR_METH_CICP, COMPRESSION_JPEG_XS, SIGNATURE_BOX, TBOX_CODESTREAM,
        TBOX_COLOUR, TBOX_FILETYPE, TBOX_HEADER, TBOX_IMAGE_HEADER,
    };

    fn boxed(tbox: u32, body: &[u8]) -> Vec<u8> {
        let len = 8 + body.len();
        let mut v = Vec::with_capacity(len);
        v.extend_from_slice(&(len as u32).to_be_bytes());
        v.extend_from_slice(&tbox.to_be_bytes());
        v.extend_from_slice(body);
        v
    }

    /// Hand-built minimal `.jxs` wrapper with a CICP box (sRGB-ish,
    /// identity matrix, limited-range flag).
    fn wrap(cs: &[u8], w: u32, h: u32, nc: u16) -> Vec<u8> {
        let mut file = Vec::new();
        file.extend_from_slice(&SIGNATURE_BOX);
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        ftyp.extend_from_slice(&0u32.to_be_bytes());
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        file.extend_from_slice(&boxed(TBOX_FILETYPE, &ftyp));
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&nc.to_be_bytes());
        ihdr.push(7); // BPC: 8-bit unsigned
        ihdr.push(COMPRESSION_JPEG_XS);
        ihdr.push(0);
        ihdr.push(0);
        let mut colr = vec![COLR_METH_CICP, 0, 0];
        colr.extend_from_slice(&1u16.to_be_bytes());
        colr.extend_from_slice(&13u16.to_be_bytes());
        colr.extend_from_slice(&0u16.to_be_bytes());
        colr.push(0);
        let mut jp2h = Vec::new();
        jp2h.extend_from_slice(&boxed(TBOX_IMAGE_HEADER, &ihdr));
        jp2h.extend_from_slice(&boxed(TBOX_COLOUR, &colr));
        file.extend_from_slice(&boxed(TBOX_HEADER, &jp2h));
        file.extend_from_slice(&boxed(TBOX_CODESTREAM, cs));
        file
    }

    fn gray(w: u32, h: u32, k: usize) -> JpegXsImage {
        let pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i * k) as u8).collect();
        JpegXsImage::new(
            w,
            h,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(w as usize, pixels)],
        )
        .unwrap()
    }

    #[test]
    fn decoder_accepts_box_wrapped_jxs_file_and_stamps_colour() {
        let (w, h) = (8u32, 4u32);
        let img = gray(w, h, 5);
        let cs = crate::encode(&img, &EncodeOptions::default()).unwrap();
        let file = wrap(&cs, w, h, 1);

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), file).with_pts(42);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.pts, Some(42));
                assert_eq!(v.planes[0].data, img.planes[0].data);
                let sig = v.color_signal().expect("CICP box → colour signal");
                assert_eq!(sig.primaries.0, 1);
                assert_eq!(sig.transfer.0, 13);
                assert_eq!(sig.range, oxideav_core::ColorRange::Limited);
            }
            _ => panic!("expected a video frame"),
        }
    }

    #[test]
    fn decoder_bare_codestream_has_no_colour_signal() {
        let (w, h) = (8u32, 4u32);
        let img = gray(w, h, 3);
        let cs = crate::encode(&img, &EncodeOptions::default()).unwrap();
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), cs))
            .unwrap();
        let frame = dec.receive_frame().unwrap();
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.planes[0].data, img.planes[0].data);
                assert!(v.color_signal().is_none());
            }
            _ => panic!("expected a video frame"),
        }
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn frame_bridge_round_trips_every_layout() {
        for f in JpegXsPixelFormat::ALL {
            let (w, h) = (6u32, 4u32);
            let planes: Vec<Plane> = (0..f.plane_count())
                .map(|i| {
                    let (pw, ph) = f.plane_dimensions(w, h, i);
                    let stride = pw * f.bytes_per_sample();
                    Plane::new(
                        stride,
                        (0..stride * ph).map(|k| (k * 3 + i) as u8).collect(),
                    )
                })
                .collect();
            let img = JpegXsImage::new(w, h, f, planes).unwrap();
            let frame = VideoFrame::from(img.clone());
            assert_eq!(frame.image_planes().len(), f.plane_count());
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.width = Some(w);
            params.height = Some(h);
            params.pixel_format = Some(to_core_pixel_format(f));
            let back = JpegXsImage::from_video_frame(&frame, &params).unwrap();
            assert_eq!(back.planes, img.planes, "{f}");
            assert_eq!(back.format, f);
            assert_eq!(
                JpegXsPixelFormat::try_from(PixelFormat::from(f)).unwrap(),
                f
            );
            let back2 = JpegXsImage::try_from((&frame, &params)).unwrap();
            assert_eq!(back2, back);
        }
        // Colour signal round trip and the unspecified default.
        let img = gray(4, 2, 1).with_color(ColorInfo::new(ColorRange::Full, 1, 13, 2));
        let frame = VideoFrame::from(img.clone());
        assert!(frame.color_signal().is_some());
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(4);
        params.height = Some(2);
        params.pixel_format = Some(PixelFormat::Gray8);
        let back = JpegXsImage::from_video_frame(&frame, &params).unwrap();
        assert_eq!(back.color, img.color);
        let plain = VideoFrame::from(gray(4, 2, 1));
        assert!(plain.color_signal().is_none());
        // Unsupported / missing params are crate errors.
        params.pixel_format = Some(PixelFormat::Yuyv422);
        assert!(matches!(
            JpegXsImage::from_video_frame(&frame, &params),
            Err(JpegXsError::Unsupported(_))
        ));
        params.pixel_format = None;
        assert!(matches!(
            JpegXsImage::from_video_frame(&frame, &params),
            Err(JpegXsError::InvalidData(_))
        ));
        assert!(JpegXsPixelFormat::try_from(PixelFormat::Pal8).is_err());
    }

    #[test]
    fn frame_bridge_deplanes_packed_rgb_frames() {
        // Rgba with a padded stride.
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 12,
                data: vec![1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
            }],
        };
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(2);
        params.height = Some(1);
        params.pixel_format = Some(PixelFormat::Rgba);
        let img = JpegXsImage::from_video_frame(&frame, &params).unwrap();
        assert_eq!(img.format, JpegXsPixelFormat::Gbrap8);
        assert_eq!(img.to_rgba8(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        params.pixel_format = Some(PixelFormat::Rgb24);
        let frame = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 6,
                data: vec![1, 2, 3, 4, 5, 6],
            }],
        };
        let img = JpegXsImage::from_video_frame(&frame, &params).unwrap();
        assert_eq!(img.format, JpegXsPixelFormat::Gbrp8);
        assert_eq!(img.to_rgb8(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn framework_encoder_round_trips_through_the_decoder() {
        let (w, h) = (32u32, 16u32);
        let rgb: Vec<u8> = (0..(w * h * 3) as usize)
            .map(|i| (i * 7 % 256) as u8)
            .collect();
        let src = JpegXsImage::from_rgb8(w, h, rgb.clone()).unwrap();
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(PixelFormat::Gbrp8);
        params.options.insert("quantization", "0");
        params.options.insert("levels_x", "3");
        params.options.insert("boxed", "true");
        let mut enc = make_encoder(&params).unwrap();
        assert_eq!(enc.output_params().width, Some(w));
        let mut vf = VideoFrame::from(src.clone());
        vf.pts = Some(9);
        enc.send_frame(&Frame::Video(vf)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert!(pkt.flags.keyframe);
        assert_eq!(pkt.pts, Some(9));
        assert!(crate::fileformat::is_jxs_file(&pkt.data));
        enc.flush().unwrap();
        assert!(matches!(enc.receive_packet(), Err(Error::Eof)));
        let back = crate::decode(&pkt.data).unwrap();
        assert_eq!(back.to_rgb8(), rgb);
        assert_eq!(back.color, ColorInfo::srgb());
        // Through the framework decoder, the boxed file stamps the signal.
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(frame) = dec.receive_frame().unwrap() else {
            panic!("video frame");
        };
        assert_eq!(frame.image_planes().len(), 3);
        assert_eq!(frame.color_signal().unwrap().matrix.0, 0);
        // Option validation.
        params.options.insert("quantization", "16");
        assert!(make_encoder(&params).is_err());
        params.options.insert("quantization", "2");
        params.options.insert("profile", "main444_12");
        let mut bogus = params.clone();
        bogus.options.insert("bogus", "1");
        assert!(make_encoder(&bogus).is_err());
        let mut enc = make_encoder(&params).unwrap();
        enc.send_frame(&Frame::Video(VideoFrame::from(src)))
            .unwrap();
        let pkt = enc.receive_packet().unwrap();
        let i = crate::info(&pkt.data).unwrap();
        assert_eq!(i.profile, Profile::Main444_12.ppih());
    }

    #[test]
    fn options_schema_applies_every_key() {
        let mut o = EncodeOptions::default();
        for (k, v) in [
            ("target_bytes", OptionValue::U32(4096)),
            ("levels_y", OptionValue::U32(1)),
            ("rct", OptionValue::Bool(false)),
            ("quantizer", OptionValue::String("uniform".into())),
            ("slice_height", OptionValue::U32(2)),
            ("run_mode", OptionValue::String("zero_coefficients".into())),
            ("weights", OptionValue::String("annex_h".into())),
            ("high_precision", OptionValue::Bool(true)),
        ] {
            o.apply(k, &v).unwrap();
        }
        assert_eq!(o.target_bytes, Some(4096));
        assert_eq!(o.levels_y, Some(1));
        assert!(!o.rct && o.high_precision);
        assert_eq!(o.quantizer, Quantizer::Uniform);
        assert_eq!(o.slice_height, Some(2));
        assert_eq!(o.run_mode, RunMode::ZeroCoefficients);
        assert_eq!(o.weights, Weights::AnnexH);
        assert!(o
            .apply("quantizer", &OptionValue::String("x".into()))
            .is_err());
        assert!(o
            .apply("profile", &OptionValue::String("x".into()))
            .is_err());
        assert!(o.apply("nope", &OptionValue::Bool(true)).is_err());
        for name in PROFILE_NAMES {
            assert!(profile_from_name(name).is_some());
        }
    }

    #[test]
    fn registration_and_error_mapping() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(ctx.codecs.decoder_ids().next().is_some());
        assert_eq!(
            ctx.containers.container_for_extension("JXS"),
            Some(CODEC_ID_STR)
        );
        let mut ctx2 = RuntimeContext::new();
        __oxideav_entry(&mut ctx2);
        assert!(ctx2.codecs.decoder_ids().next().is_some());
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        assert!(ctx.codecs.first_decoder(&params).is_ok());
        let e: Error = JpegXsError::limit("x").into();
        assert!(matches!(e, Error::ResourceExhausted(_)));
        let e: Error = JpegXsError::Io(std::io::Error::other("y")).into();
        assert!(matches!(e, Error::Io(_)));
        let e: Error = JpegXsError::unsupported("z").into();
        assert!(matches!(e, Error::Unsupported(_)));
        let e: Error = JpegXsError::invalid("w").into();
        assert!(matches!(e, Error::InvalidData(_)));
    }
}
