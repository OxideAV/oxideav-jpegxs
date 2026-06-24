//! `oxideav-core` integration: `Decoder` trait impl, `Frame` / `Error`
//! conversions, and the [`register`] / [`register_codecs`] /
//! [`register_containers`] entry points.
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
    ContainerRegistry, Decoder, Error, Frame, Packet, Result, RuntimeContext, VideoFrame,
};

use crate::decoder::decode_codestream;
use crate::error::JpegXsError;
use crate::fileformat::{decode_jxs_file, is_jxs_file};
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
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxs_sw")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
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
/// answer. Lookups are case-insensitive (handled by
/// [`ContainerRegistry::register_extension`] / `container_for_extension`,
/// which lowercase both sides).
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("jxs", CODEC_ID_STR);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-jpegxs` into a [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("jpegxs", register);

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
        // A packet may carry either a bare ISO/IEC 21122-1 codestream
        // (SOC marker first) or a box-wrapped JXS file (ISO/IEC 21122-3
        // Annex A, JPEG XS Signature box first). Route on the signature.
        let img = if is_jxs_file(&pkt.data) {
            let mut img = decode_jxs_file(&pkt.data)?;
            img.pts = pkt.pts;
            img
        } else {
            decode_codestream(&pkt.data, pkt.pts)?
        };
        Ok(img.into())
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
    use oxideav_core::TimeBase;

    fn boxed(tbox: u32, body: &[u8]) -> Vec<u8> {
        let len = 8 + body.len();
        let mut v = Vec::with_capacity(len);
        v.extend_from_slice(&(len as u32).to_be_bytes());
        v.extend_from_slice(&tbox.to_be_bytes());
        v.extend_from_slice(body);
        v
    }

    fn wrap(cs: &[u8], w: u32, h: u32) -> Vec<u8> {
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
        ihdr.extend_from_slice(&1u16.to_be_bytes());
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

    #[test]
    fn decoder_accepts_box_wrapped_jxs_file() {
        let (w, h) = (8u16, 4u16);
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 5) as u8)
            .collect();
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let file = wrap(&cs, w as u32, h as u32);

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 25), file).with_pts(42);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        match frame {
            Frame::Video(v) => {
                assert_eq!(v.pts, Some(42));
                assert_eq!(v.planes[0].data, pixels);
            }
            _ => panic!("expected a video frame"),
        }
    }

    #[test]
    fn decoder_still_accepts_bare_codestream() {
        let (w, h) = (8u16, 4u16);
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 3) as u8)
            .collect();
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();

        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), cs))
            .unwrap();
        let frame = dec.receive_frame().unwrap();
        match frame {
            Frame::Video(v) => assert_eq!(v.planes[0].data, pixels),
            _ => panic!("expected a video frame"),
        }
    }
}
