//! JPEG XS containers for the framework: the bare ISO/IEC 21122-1
//! codestream (`jpegxs`) and the ISO/IEC 21122-3 Annex A box file
//! (`jxs`, extension `.jxs`). Both are single-image containers in the
//! same shape as `oxideav-farbfeld`: one video stream, one packet
//! holding the whole file, the codec sniffing the framing.
//!
//! The demuxer declares the stream the way [`crate::info`] describes
//! the picture — width / height and the **native** contract layout as
//! `pixel_format` — and stamps `color_signal` only when the `.jxs`
//! header carries a CICP Colour Specification box (round-470 ruling: a
//! bare codestream, or a wrapper that merely restates the layout
//! default, signals nothing). A picture with no contract layout
//! (Star-Tetrix CFA, two- or five-plus-component sets, mixed depths,
//! odd sampling) still opens with `pixel_format = None`; the decoder
//! then reports `Unsupported`. `metadata()` carries
//! `("exif", "present")` when the header has an Exif box (the blob has
//! no framework carriage yet).
//!
//! The muxers take the registry encoder's packets (a bare codestream
//! unless `boxed = true`) and write what their name says: `jpegxs`
//! strips a `.jxs` wrapper to the codestream, `jxs` wraps a bare
//! codestream the way `encode(.., boxed)` does — CICP from the stream's
//! `color_signal` (code points `2` when unspecified), the Channel
//! Definition box for alpha layouts. One picture per file.
//!
//! Lives behind the `registry` feature.

use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, MediaType, Muxer,
    Packet, PixelFormat, ProbeData, ProbeScore, ReadSeek, Result, StreamInfo, TimeBase, WriteSeek,
    MAX_PROBE_SCORE, PROBE_SCORE_EXTENSION,
};

use crate::api::{bare_codestream, describe, wrap_in_jxs};
use crate::fileformat::{is_jxs_file, CODESTREAM_MAGIC};
use crate::image::{ColorInfo, JpegXsPixelFormat};
use crate::registry::{from_color_signal, to_color_signal};
use crate::CODEC_ID_STR;

/// Container name of the bare codestream.
pub const CONTAINER_JPEGXS: &str = "jpegxs";
/// Container name of the ISO/IEC 21122-3 box file (`.jxs`).
pub const CONTAINER_JXS: &str = "jxs";

/// Register both containers: the shared demuxer under each name, the
/// two muxers, the `.jxs` extension and the probes.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_JPEGXS, open_demuxer);
    reg.register_demuxer(CONTAINER_JXS, open_demuxer);
    reg.register_muxer(CONTAINER_JPEGXS, open_muxer_jpegxs);
    reg.register_muxer(CONTAINER_JXS, open_muxer_jxs);
    // `.jxs` is the box file format's extension (21122-3 §A.7); a bare
    // codestream has none.
    reg.register_extension("jxs", CONTAINER_JXS);
    reg.register_probe(CONTAINER_JPEGXS, probe_jpegxs);
    reg.register_probe(CONTAINER_JXS, probe_jxs);
}

/// Bare-codestream probe: the `SOC` + `CAP` marker pair scores full
/// marks; there is no extension of its own.
pub fn probe_jpegxs(data: &ProbeData) -> ProbeScore {
    if data.buf.starts_with(&CODESTREAM_MAGIC) {
        MAX_PROBE_SCORE
    } else {
        0
    }
}

/// `.jxs` probe: the JPEG XS Signature box scores full marks, the
/// `.jxs` extension the conventional weak score.
pub fn probe_jxs(data: &ProbeData) -> ProbeScore {
    if is_jxs_file(data.buf) {
        return MAX_PROBE_SCORE;
    }
    if data.ext == Some("jxs") {
        PROBE_SCORE_EXTENSION
    } else {
        0
    }
}

// ---- Demuxer ----------------------------------------------------------------

/// Open a bare codestream or a `.jxs` file as a one-stream, one-packet
/// container. The picture header (and the box headers) are walked
/// eagerly so the stream carries accurate geometry before the decoder
/// runs.
pub fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);
    if !is_jxs_file(&buf) && !buf.starts_with(&CODESTREAM_MAGIC) {
        return Err(Error::invalid(
            "jpegxs demuxer: neither a JPEG XS Signature box nor an SOC / CAP codestream",
        ));
    }
    let d = describe(&buf)?;
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(d.width);
    params.height = Some(d.height);
    params.pixel_format = d.format.map(PixelFormat::from);
    if let Some(c) = &d.cicp {
        params.color_signal = to_color_signal(c);
    }
    let mut metadata = Vec::new();
    if d.has_exif {
        metadata.push(("exif".to_owned(), "present".to_owned()));
    }
    let stream = StreamInfo {
        index: 0,
        params,
        time_base: TimeBase::new(1, 1),
        start_time: Some(0),
        duration: None,
    };
    Ok(Box::new(JpegXsDemuxer {
        name: if d.boxed {
            CONTAINER_JXS
        } else {
            CONTAINER_JPEGXS
        },
        streams: vec![stream],
        metadata,
        data: Some(buf),
    }))
}

struct JpegXsDemuxer {
    name: &'static str,
    streams: Vec<StreamInfo>,
    metadata: Vec<(String, String)>,
    data: Option<Vec<u8>>,
}

impl Demuxer for JpegXsDemuxer {
    fn format_name(&self) -> &str {
        self.name
    }
    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
    fn next_packet(&mut self) -> Result<Packet> {
        match self.data.take() {
            Some(bytes) => {
                let mut pkt = Packet::new(0, TimeBase::new(1, 1), bytes);
                pkt.pts = Some(0);
                pkt.dts = Some(0);
                pkt.flags.keyframe = true;
                Ok(pkt)
            }
            None => Err(Error::Eof),
        }
    }
}

// ---- Muxers -----------------------------------------------------------------

/// Muxer for the bare codestream: the encoder's packet is written as
/// is, or unwrapped when it is a `.jxs` file.
pub fn open_muxer_jpegxs(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
) -> Result<Box<dyn Muxer>> {
    JpegXsMuxer::open(output, streams, false)
}

/// Muxer for the `.jxs` file: a boxed packet is written as is; a bare
/// codestream is wrapped with a CICP box from the stream's
/// `color_signal` (unspecified code points when the stream signals
/// none) and a Channel Definition box for alpha layouts.
pub fn open_muxer_jxs(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
) -> Result<Box<dyn Muxer>> {
    JpegXsMuxer::open(output, streams, true)
}

struct JpegXsMuxer {
    output: Box<dyn WriteSeek>,
    wrap: bool,
    format: Option<JpegXsPixelFormat>,
    color: ColorInfo,
    written: bool,
}

impl JpegXsMuxer {
    fn open(
        output: Box<dyn WriteSeek>,
        streams: &[StreamInfo],
        wrap: bool,
    ) -> Result<Box<dyn Muxer>> {
        let name = if wrap {
            CONTAINER_JXS
        } else {
            CONTAINER_JPEGXS
        };
        let [stream] = streams else {
            return Err(Error::invalid(format!(
                "{name} muxer: expected exactly one video stream, got {}",
                streams.len()
            )));
        };
        let p = &stream.params;
        if p.media_type != MediaType::Video {
            return Err(Error::invalid(format!(
                "{name} muxer: stream must be video"
            )));
        }
        if p.codec_id.as_str() != CODEC_ID_STR {
            return Err(Error::unsupported(format!(
                "{name} muxer: stream codec {} is not {CODEC_ID_STR}",
                p.codec_id
            )));
        }
        let format = p.pixel_format.and_then(|f| match f {
            // The registry encoder deplanes these (the raw-path rule).
            PixelFormat::Rgb24 => Some(JpegXsPixelFormat::Gbrp8),
            PixelFormat::Rgba => Some(JpegXsPixelFormat::Gbrap8),
            other => JpegXsPixelFormat::try_from(other).ok(),
        });
        let color = if p.color_signal.is_unspecified() {
            ColorInfo::unspecified()
        } else {
            from_color_signal(&p.color_signal)
        };
        Ok(Box::new(JpegXsMuxer {
            output,
            wrap,
            format,
            color,
            written: false,
        }))
    }
}

impl Muxer for JpegXsMuxer {
    fn format_name(&self) -> &str {
        if self.wrap {
            CONTAINER_JXS
        } else {
            CONTAINER_JPEGXS
        }
    }
    fn write_header(&mut self) -> Result<()> {
        Ok(())
    }
    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let name = self.format_name();
        if self.written {
            return Err(Error::unsupported(format!(
                "{name} muxer: a JPEG XS file holds one picture; second packet refused"
            )));
        }
        let data = packet.data.as_slice();
        let out: std::borrow::Cow<'_, [u8]> = if is_jxs_file(data) {
            if self.wrap {
                data.into()
            } else {
                bare_codestream(data)?.into()
            }
        } else if data.starts_with(&CODESTREAM_MAGIC) {
            if self.wrap {
                // Alpha from the stream's layout, else from the
                // codestream's own component set.
                let has_alpha = match self.format {
                    Some(f) => f.has_alpha(),
                    None => describe(data).is_ok_and(|d| d.format.is_some_and(|f| f.has_alpha())),
                };
                wrap_in_jxs(data, &self.color, has_alpha, None)?.into()
            } else {
                data.into()
            }
        } else {
            return Err(Error::invalid(format!(
                "{name} muxer: packet is neither a JPEG XS codestream nor a .jxs file"
            )));
        };
        self.output.write_all(&out)?;
        self.written = true;
        Ok(())
    }
    fn write_trailer(&mut self) -> Result<()> {
        self.output.flush()?;
        Ok(())
    }
}
