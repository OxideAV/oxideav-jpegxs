//! `oxideav-core` integration: `Decoder` trait impl, `Frame` / `Error`
//! conversions, and the [`register`] entry point.
//!
//! Gated behind the default-on `registry` Cargo feature. With the
//! feature off the rest of the crate still exposes the standalone
//! [`crate::decode_jpeg_xs`] API plus the underlying `codestream` /
//! `decoder` / `dwt` / etc. modules and [`crate::JpegXsImage`] /
//! [`crate::JpegXsError`] types — none of which depend on
//! `oxideav-core`.

use std::collections::VecDeque;

use oxideav_core::{
    frame::VideoPlane, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry,
    Decoder, Error, Frame, Packet, Result, VideoFrame,
};

use crate::decoder::decode_codestream;
use crate::error::JpegXsError;
use crate::image::JpegXsImage;
use crate::CODEC_ID_STR;

impl From<JpegXsError> for Error {
    fn from(e: JpegXsError) -> Self {
        match e {
            JpegXsError::InvalidData(s) => Error::InvalidData(s),
            JpegXsError::Unsupported(s) => Error::Unsupported(s),
        }
    }
}

impl From<JpegXsImage> for Frame {
    fn from(img: JpegXsImage) -> Self {
        let planes = img
            .planes
            .into_iter()
            .map(|p| VideoPlane {
                stride: p.stride,
                data: p.data,
            })
            .collect();
        Frame::Video(VideoFrame {
            pts: img.pts,
            planes,
        })
    }
}

/// Register the JPEG XS decoder factory.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxs_sw")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Decoder factory. Round 6 accepts the multi-component
/// single-precinct-row subset.
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
        let img = decode_codestream(&pkt.data, pkt.pts)?;
        Ok(img.into())
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}
