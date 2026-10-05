//! The framework containers (`jpegxs` bare codestream, `jxs` box file):
//! probe → demuxer → decoder yields the Layer 1 planes byte for byte on
//! every encoder-produced layout, the muxers write files Layer 1 reads
//! back, and the registry round trip `demux(mux(frames)) == frames`
//! holds. The crate ships no `.jxs` fixtures (the conformance vectors
//! are external), so every input here comes from `encode`.

#![cfg(feature = "registry")]

use std::io::Cursor;

use oxideav_core::{
    CodecId, CodecParameters, DecoderLimits, Error, Frame, NullCodecResolver, Packet, PixelFormat,
    ProbeData, ReadSeek, RuntimeContext, StreamInfo, TimeBase, VideoFrame, WriteSeek,
};
use oxideav_jpegxs::container::{
    open_demuxer, open_muxer_jpegxs, open_muxer_jxs, probe_jpegxs, probe_jxs, CONTAINER_JPEGXS,
    CONTAINER_JXS,
};
use oxideav_jpegxs::registry::to_color_signal;
use oxideav_jpegxs::{
    decode, encode, encode_components, info, make_encoder, register, ColorInfo, ColorRange,
    Components, EncodeOptions, JpegXsError, JpegXsImage, Metadata, PixelFormat as Jxs, Plane,
    CODEC_ID_STR,
};

/// A `WriteSeek` sink whose bytes stay reachable after the muxer took
/// ownership of the box.
#[derive(Clone, Default)]
struct SharedSink {
    buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    pos: u64,
}

impl SharedSink {
    fn bytes(&self) -> Vec<u8> {
        self.buf.lock().unwrap().clone()
    }
}

impl std::io::Write for SharedSink {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let mut v = self.buf.lock().unwrap();
        let at = self.pos as usize;
        if v.len() < at + b.len() {
            v.resize(at + b.len(), 0);
        }
        v[at..at + b.len()].copy_from_slice(b);
        self.pos += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::io::Seek for SharedSink {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let len = self.buf.lock().unwrap().len() as i64;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i64,
            std::io::SeekFrom::End(n) => len + n,
            std::io::SeekFrom::Current(n) => self.pos as i64 + n,
        };
        self.pos = u64::try_from(new).map_err(|_| std::io::ErrorKind::InvalidInput)?;
        Ok(self.pos)
    }
}

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    ctx
}

fn reader(bytes: &[u8]) -> Box<dyn ReadSeek> {
    Box::new(Cursor::new(bytes.to_vec()))
}

fn probe(ctx: &RuntimeContext, bytes: &[u8], ext: Option<&str>) -> oxideav_core::Result<String> {
    let mut cur = Cursor::new(bytes.to_vec());
    ctx.containers.probe_input(&mut cur, ext)
}

/// Demux `bytes` through the registry and decode its packet with the
/// registry decoder.
fn demux_decode(ctx: &RuntimeContext, bytes: &[u8]) -> (StreamInfo, VideoFrame) {
    let name = probe(ctx, bytes, None).expect("probe");
    let mut demux = ctx
        .containers
        .open_demuxer(&name, reader(bytes), &ctx.codecs)
        .expect("open_demuxer");
    assert_eq!(demux.format_name(), name);
    let stream = demux.streams()[0].clone();
    let pkt = demux.next_packet().expect("one packet");
    assert_eq!(pkt.stream_index, 0);
    assert!(pkt.flags.keyframe);
    assert_eq!(pkt.pts, Some(0));
    assert!(matches!(demux.next_packet(), Err(Error::Eof)));
    let mut dec = ctx
        .codecs
        .first_decoder(&stream.params)
        .expect("first_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("video frame expected");
    };
    (stream, vf)
}

fn mux(
    ctx: &RuntimeContext,
    name: &str,
    stream: &StreamInfo,
    pkt: &Packet,
) -> oxideav_core::Result<Vec<u8>> {
    let sink = SharedSink::default();
    let out: Box<dyn WriteSeek> = Box::new(sink.clone());
    let mut m = ctx
        .containers
        .open_muxer(name, out, std::slice::from_ref(stream))?;
    assert_eq!(m.format_name(), name);
    m.write_header()?;
    m.write_packet(pkt)?;
    m.write_trailer()?;
    Ok(sink.bytes())
}

/// Planar image in layout `f` with deterministic samples.
fn planar(w: u32, h: u32, f: Jxs, seed: usize) -> JpegXsImage {
    let bps = f.bytes_per_sample();
    let max = if bps == 1 {
        256
    } else {
        1usize << u32::from(f.nominal_bits())
    };
    let planes: Vec<Plane> = (0..f.plane_count())
        .map(|i| {
            let (pw, ph) = f.plane_dimensions(w, h, i);
            let mut data = Vec::with_capacity(pw * ph * bps);
            for k in 0..pw * ph {
                let v = (k * (seed + 7) + i * 31) % max;
                if bps == 1 {
                    data.push(v as u8);
                } else {
                    data.extend_from_slice(&(v as u16).to_le_bytes());
                }
            }
            Plane::new(pw * bps, data)
        })
        .collect();
    JpegXsImage::new(w, h, f, planes).unwrap()
}

/// The layouts under test; the first three carry a colour description,
/// the rest none.
fn sample_images() -> Vec<JpegXsImage> {
    vec![
        JpegXsImage::from_rgb8(
            16,
            8,
            (0..16 * 8 * 3).map(|i| (i * 7 % 256) as u8).collect(),
        )
        .unwrap()
        .with_color(ColorInfo::srgb()),
        JpegXsImage::from_rgba8(
            16,
            8,
            (0..16 * 8 * 4).map(|i| (i * 5 % 256) as u8).collect(),
        )
        .unwrap()
        .with_color(ColorInfo::srgb()),
        planar(16, 8, Jxs::Yuv422P10Le, 3).with_color(ColorInfo::new(ColorRange::Limited, 1, 1, 1)),
        planar(16, 8, Jxs::Gray8, 1),
        planar(16, 8, Jxs::Gray12Le, 2),
        planar(16, 8, Jxs::Yuv420P, 4),
        planar(16, 8, Jxs::Gbrp16Le, 5),
    ]
}

/// `(name, bytes, container)` for every sample image, bare and boxed.
fn fixtures() -> Vec<(String, Vec<u8>, &'static str)> {
    let mut out = Vec::new();
    for img in sample_images() {
        let bare = encode(&img, &EncodeOptions::default()).unwrap();
        out.push((format!("{} bare", img.format), bare, CONTAINER_JPEGXS));
        let boxed = encode(&img, &EncodeOptions::default().with_boxed(true)).unwrap();
        out.push((format!("{} boxed", img.format), boxed, CONTAINER_JXS));
    }
    out
}

#[test]
fn probe_names_the_container_from_magic_and_extension() {
    let ctx = ctx();
    for (name, bytes, container) in fixtures() {
        assert_eq!(
            probe(&ctx, &bytes, None).unwrap(),
            container,
            "{name} magic"
        );
        assert_eq!(
            probe(&ctx, &bytes, Some("jxs")).unwrap(),
            container,
            "{name} magic + extension"
        );
    }
    // Extension alone resolves to the box container; the demuxer then
    // rejects the bytes.
    assert_eq!(
        probe(&ctx, b"not a picture at all", Some("jxs")).unwrap(),
        CONTAINER_JXS
    );
    assert!(ctx
        .containers
        .open_demuxer(CONTAINER_JXS, reader(b"not a picture at all"), &ctx.codecs)
        .is_err());
    for foreign in [
        &b"farbfeld\0\0\0\x01\0\0\0\x01\0\0\0\0\0\0\0\0"[..],
        &b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"[..],
        &b"\xff\xd8\xff\xe0\0\x10JFIF"[..],
        &b"\xff\x4f\xff\x51"[..], // JPEG 2000, not XS
        &[][..],
    ] {
        assert!(
            matches!(probe(&ctx, foreign, None), Err(Error::FormatNotFound(_))),
            "{foreign:?}"
        );
    }
    let boxed = &fixtures()[1].1;
    for n in 0..16 {
        let data = ProbeData {
            buf: &boxed[..n],
            ext: None,
        };
        let _ = probe_jxs(&data);
        let _ = probe_jpegxs(&data);
    }
}

#[test]
fn demuxer_declares_the_layer1_stream() {
    let ctx = ctx();
    let mut formats = Vec::new();
    for (name, bytes, _) in fixtures() {
        let i = info(&bytes).unwrap();
        formats.push(i.format);
        let (stream, _) = demux_decode(&ctx, &bytes);
        let p = &stream.params;
        assert_eq!(p.codec_id, CodecId::new(CODEC_ID_STR), "{name}");
        assert_eq!(p.width, Some(i.width), "{name} width");
        assert_eq!(p.height, Some(i.height), "{name} height");
        assert_eq!(
            p.pixel_format,
            Some(PixelFormat::from(i.format)),
            "{name} format"
        );
        // Only a `.jxs` file with a CICP box signals colour; `encode`
        // writes one on every boxed file.
        assert_eq!(
            !p.color_signal.is_unspecified(),
            i.boxed,
            "{name}: colour signal iff the file carries a CICP box"
        );
        if i.boxed {
            assert_eq!(p.color_signal.primaries.0, i.color.primaries, "{name}");
            assert_eq!(p.color_signal.transfer.0, i.color.transfer, "{name}");
            assert_eq!(p.color_signal.matrix.0, i.color.matrix, "{name}");
        }
        assert!(p.extradata.is_empty());
        assert_eq!(stream.time_base, TimeBase::new(1, 1));
        assert_eq!(stream.start_time, Some(0));
    }
    for want in [
        Jxs::Gray8,
        Jxs::Gray12Le,
        Jxs::Gbrp8,
        Jxs::Gbrap8,
        Jxs::Gbrp16Le,
        Jxs::Yuv420P,
        Jxs::Yuv422P10Le,
    ] {
        assert!(formats.contains(&want), "{want} not covered");
    }
}

#[test]
fn registry_frames_are_byte_identical_to_layer1() {
    let ctx = ctx();
    for (name, bytes, _) in fixtures() {
        let img = decode(&bytes).unwrap();
        let (stream, vf) = demux_decode(&ctx, &bytes);
        assert_eq!(vf.pts, Some(0), "{name}");
        let planes = vf.image_planes();
        assert_eq!(planes.len(), img.planes.len(), "{name} plane count");
        for (k, (a, b)) in planes.iter().zip(&img.planes).enumerate() {
            assert_eq!(a.stride, b.stride, "{name} plane {k} stride");
            assert_eq!(a.data, b.data, "{name} plane {k} samples");
        }
        assert_eq!(
            vf.color_signal().is_some(),
            !stream.params.color_signal.is_unspecified(),
            "{name}: frame and stream agree on colour"
        );
        if let Some(sig) = vf.color_signal() {
            assert_eq!(sig, stream.params.color_signal, "{name}");
        }
    }
}

#[test]
fn boxed_file_without_cicp_signals_no_colour() {
    // A `.jxs` wrapper whose `colr` box uses another METH restates
    // nothing the layout default does not already say — the stream
    // stays unstamped (round-470 ruling).
    let ctx = ctx();
    let img = planar(8, 4, Jxs::Gray8, 9);
    let mut file = encode(&img, &EncodeOptions::default().with_boxed(true)).unwrap();
    let colr = u32::from_be_bytes(*b"colr").to_be_bytes();
    let at = file
        .windows(4)
        .position(|w| w == colr)
        .expect("colr box in the encoder's wrapper");
    assert_eq!(file[at + 4], 5, "METH = 5 (CICP) as encode wrote it");
    file[at + 4] = 1;
    let (stream, vf) = demux_decode(&ctx, &file);
    assert!(stream.params.color_signal.is_unspecified());
    assert!(vf.color_signal().is_none());
    assert_eq!(vf.planes[0].data, img.planes[0].data);
}

#[test]
fn unsupported_component_set_opens_without_a_layout() {
    // Two components have no contract layout: `info` is `Unsupported`,
    // the demuxer still names the stream and its geometry, and the
    // decoder refuses the packet cleanly.
    let ctx = ctx();
    let comps = Components::new(
        8,
        4,
        0,
        vec![8, 8],
        vec![(1, 1), (1, 1)],
        vec![
            Plane::new(8, (0..32).map(|i| i as u8).collect()),
            Plane::new(8, (0..32).map(|i| (255 - i) as u8).collect()),
        ],
    )
    .unwrap();
    let bytes = encode_components(&comps, &EncodeOptions::default()).unwrap();
    assert!(matches!(info(&bytes), Err(JpegXsError::Unsupported(_))));
    let name = probe(&ctx, &bytes, None).unwrap();
    assert_eq!(name, CONTAINER_JPEGXS);
    let mut demux = ctx
        .containers
        .open_demuxer(&name, reader(&bytes), &ctx.codecs)
        .unwrap();
    let stream = demux.streams()[0].clone();
    assert_eq!(
        (stream.params.width, stream.params.height),
        (Some(8), Some(4))
    );
    assert_eq!(stream.params.pixel_format, None);
    let pkt = demux.next_packet().unwrap();
    let mut dec = ctx.codecs.first_decoder(&stream.params).unwrap();
    dec.send_packet(&pkt).unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::Unsupported(_))));
    // The `jxs` muxer still wraps it (component sets carry no colour).
    let wrapped = mux(&ctx, CONTAINER_JXS, &stream, &pkt).unwrap();
    assert!(oxideav_jpegxs::parse_jxs_file(&wrapped).is_ok());
}

#[test]
fn demuxer_reports_exif_presence_in_metadata() {
    let img = planar(8, 4, Jxs::Gray8, 2)
        .with_metadata(Metadata::new().with_exif(vec![0x4d, 0x4d, 0, 0x2a, 0, 0, 0, 8]));
    let file = encode(&img, &EncodeOptions::default().with_boxed(true)).unwrap();
    let demux = open_demuxer(reader(&file), &NullCodecResolver).unwrap();
    assert_eq!(
        demux.metadata(),
        &[("exif".to_owned(), "present".to_owned())]
    );
    let plain = encode(&img, &EncodeOptions::default()).unwrap();
    let demux = open_demuxer(reader(&plain), &NullCodecResolver).unwrap();
    assert!(demux.metadata().is_empty());
}

// ---- muxers -----------------------------------------------------------------

/// Encode `img` through the registry encoder and return the stream it
/// declares plus its packet.
fn encode_via_registry(img: &JpegXsImage, boxed: bool) -> (StreamInfo, Packet) {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(img.width);
    params.height = Some(img.height);
    params.pixel_format = Some(img.format.into());
    if !img.color.is_unspecified() {
        params.color_signal = to_color_signal(&img.color);
    }
    if boxed {
        params.options.insert("boxed", "true");
    }
    let mut enc = make_encoder(&params).unwrap();
    let frame = VideoFrame::from(img.clone());
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    let stream = StreamInfo {
        index: 0,
        params: enc.output_params().clone(),
        time_base: TimeBase::new(1, 1),
        start_time: Some(0),
        duration: None,
    };
    (stream, pkt)
}

#[test]
fn muxers_write_files_layer1_reads_back_and_the_registry_round_trips() {
    let ctx = ctx();
    for img in sample_images() {
        let f = img.format;
        for boxed in [false, true] {
            let (stream, pkt) = encode_via_registry(&img, boxed);
            assert_eq!(oxideav_jpegxs::parse_jxs_file(&pkt.data).is_ok(), boxed);
            // --- jxs: a bare packet gets wrapped, a boxed one passes through.
            let jxs = mux(&ctx, CONTAINER_JXS, &stream, &pkt).unwrap();
            let i = info(&jxs).unwrap();
            assert!(i.boxed, "{f} boxed={boxed}");
            let back = decode(&jxs).unwrap();
            assert_eq!(back.format, f, "{f} boxed={boxed} layout");
            assert_eq!(back.planes, img.planes, "{f} boxed={boxed} planes");
            // CICP has no "unspecified" range (an unsignalled range is
            // written as the limited flag), so colour is exact only for
            // images whose range is signalled — the README's statement.
            if img.color.range != ColorRange::Unspecified {
                assert_eq!(back.color, img.color, "{f} boxed={boxed} colour");
            }
            // --- jpegxs: a bare packet passes through, a boxed one unwraps.
            let bare = mux(&ctx, CONTAINER_JPEGXS, &stream, &pkt).unwrap();
            assert_eq!(&bare[..2], &[0xff, 0x10], "{f} boxed={boxed}");
            assert_eq!(
                decode(&bare).unwrap().planes,
                img.planes,
                "{f} boxed={boxed}"
            );
            // --- registry round trip: demux(mux(frame)) == frame.
            for file in [&jxs, &bare] {
                let (stream2, vf) = demux_decode(&ctx, file);
                assert_eq!(stream2.params.pixel_format, Some(PixelFormat::from(f)));
                let planes = vf.image_planes();
                assert_eq!(planes.len(), img.planes.len(), "{f}");
                for (a, b) in planes.iter().zip(&img.planes) {
                    assert_eq!(a.data, b.data, "{f} round-trip samples");
                }
            }
        }
    }
}

#[test]
fn jxs_muxer_wraps_from_the_codestream_when_the_stream_names_no_layout() {
    // Before the first frame a pipeline's stream may carry no usable
    // pixel format: the alpha channel definition comes from the
    // codestream's own component count.
    let ctx = ctx();
    let img = sample_images().swap_remove(1); // Gbrap8
    let (mut stream, pkt) = encode_via_registry(&img, false);
    stream.params.pixel_format = None;
    let jxs = mux(&ctx, CONTAINER_JXS, &stream, &pkt).unwrap();
    let i = info(&jxs).unwrap();
    assert!(i.boxed && i.has_alpha && i.format == Jxs::Gbrap8);
    assert_eq!(decode(&jxs).unwrap().planes, img.planes);
    // Packed `Rgba` on the stream (the encoder's input layout) maps the
    // same way.
    stream.params.pixel_format = Some(PixelFormat::Rgba);
    let jxs = mux(&ctx, CONTAINER_JXS, &stream, &pkt).unwrap();
    assert!(info(&jxs).unwrap().has_alpha);
}

#[test]
fn muxers_refuse_a_second_picture_and_foreign_packets() {
    let ctx = ctx();
    let img = sample_images().swap_remove(3);
    let (stream, pkt) = encode_via_registry(&img, false);
    for name in [CONTAINER_JPEGXS, CONTAINER_JXS] {
        let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
        let mut m = ctx
            .containers
            .open_muxer(name, out, std::slice::from_ref(&stream))
            .unwrap();
        m.write_header().unwrap();
        m.write_packet(&pkt).unwrap();
        assert!(
            matches!(m.write_packet(&pkt), Err(Error::Unsupported(_))),
            "{name}"
        );
        let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
        let mut m = ctx
            .containers
            .open_muxer(name, out, std::slice::from_ref(&stream))
            .unwrap();
        assert!(m
            .write_packet(&Packet::new(0, TimeBase::new(1, 1), Vec::new()))
            .is_err());
        assert!(m
            .write_packet(&Packet::new(0, TimeBase::new(1, 1), b"farbfeld".to_vec()))
            .is_err());
    }
    let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    assert!(ctx.containers.open_muxer(CONTAINER_JXS, out, &[]).is_err());
    let mut other = stream.clone();
    other.params.codec_id = CodecId::new("png");
    let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
    assert!(ctx
        .containers
        .open_muxer(CONTAINER_JPEGXS, out, std::slice::from_ref(&other))
        .is_err());
}

// ---- registration ---------------------------------------------------------------

#[test]
fn register_installs_codec_and_containers() {
    let ctx = ctx();
    let id = CodecId::new(CODEC_ID_STR);
    assert!(ctx.codecs.has_decoder(&id) && ctx.codecs.has_encoder(&id));
    for name in [CONTAINER_JPEGXS, CONTAINER_JXS] {
        assert!(ctx.containers.demuxer_names().any(|n| n == name), "{name}");
        assert!(ctx.containers.muxer_names().any(|n| n == name), "{name}");
    }
    assert_eq!(
        ctx.containers.container_for_extension("JXS"),
        Some(CONTAINER_JXS)
    );
    let mut ctx2 = RuntimeContext::new();
    oxideav_jpegxs::__oxideav_entry(&mut ctx2);
    assert!(ctx2.codecs.has_decoder(&id));
    let fx = fixtures();
    assert_eq!(probe(&ctx2, &fx[0].1, None).unwrap(), CONTAINER_JPEGXS);
    assert_eq!(probe(&ctx2, &fx[1].1, None).unwrap(), CONTAINER_JXS);
}

// ---- hostile input ----------------------------------------------------------------

#[test]
fn hostile_input_never_panics() {
    let ctx = ctx();
    for (_, bytes, _) in fixtures() {
        for cut in [0usize, 1, 2, 3, 4, 11, 12, 13, 20, 40, 60, bytes.len() / 2] {
            let cut = cut.min(bytes.len());
            if let Ok(mut d) = open_demuxer(reader(&bytes[..cut]), &NullCodecResolver) {
                let pkt = d.next_packet().unwrap();
                let mut p = d.streams()[0].params.clone();
                p.limits = DecoderLimits::default()
                    .with_max_pixels_per_frame(1 << 20)
                    .with_max_alloc_bytes_per_frame(64 << 20);
                if let Ok(mut dec) = ctx.codecs.first_decoder(&p) {
                    let _ = dec.send_packet(&pkt);
                    let _ = dec.receive_frame();
                }
                let _ = mux(&ctx, CONTAINER_JXS, &d.streams()[0], &pkt);
                let _ = mux(&ctx, CONTAINER_JPEGXS, &d.streams()[0], &pkt);
            }
        }
    }
    // Absurd geometry in the picture header: Wf / Hf follow the PIH
    // marker (`FF 12`), Lpih (2), Lcod (4), Ppih (2), Plev (2).
    let mut huge = fixtures()[0].1.clone();
    let pih = huge
        .windows(2)
        .position(|w| w == [0xff, 0x12])
        .expect("PIH marker");
    let at = pih + 2 + 2 + 4 + 2 + 2;
    huge[at..at + 4].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    huge[at + 4..at + 8].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    if let Ok(mut d) = open_demuxer(reader(&huge), &NullCodecResolver) {
        let pkt = d.next_packet().unwrap();
        let mut p = d.streams()[0].params.clone();
        p.limits = DecoderLimits::default()
            .with_max_pixels_per_frame(1 << 20)
            .with_max_alloc_bytes_per_frame(64 << 20);
        let mut dec = ctx.codecs.first_decoder(&p).unwrap();
        dec.send_packet(&pkt).unwrap();
        assert!(dec.receive_frame().is_err());
    }
    assert!(open_demuxer(reader(&[]), &NullCodecResolver).is_err());
    let (stream, _) = encode_via_registry(&sample_images()[3], false);
    for open in [open_muxer_jxs, open_muxer_jpegxs] {
        let out: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::new()));
        let mut m = open(out, std::slice::from_ref(&stream)).unwrap();
        assert!(m
            .write_packet(&Packet::new(0, TimeBase::new(1, 1), Vec::new()))
            .is_err());
    }
}
