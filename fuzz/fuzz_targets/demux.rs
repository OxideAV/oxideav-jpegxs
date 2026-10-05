//! Framework-container robustness target: arbitrary bytes through
//! `open_demuxer` (a `.jxs` file or a bare codestream, the sniff
//! decides), the packet through the registry decoder under tight
//! limits, and both muxers over the demuxed packet. Every step must
//! return a `Result`; nothing may panic, overflow or allocate an
//! attacker-scaled buffer.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxideav_core::{DecoderLimits, Error, NullCodecResolver, ReadSeek, WriteSeek};
use oxideav_jpegxs::container::{open_demuxer, open_muxer_jpegxs, open_muxer_jxs};

fuzz_target!(|data: &[u8]| {
    let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(data.to_vec()));
    let Ok(mut demux) = open_demuxer(input, &NullCodecResolver) else {
        return;
    };
    let stream = demux.streams()[0].clone();
    assert!(stream.params.width.is_some_and(|w| w > 0));
    assert!(stream.params.height.is_some_and(|h| h > 0));
    let Ok(pkt) = demux.next_packet() else {
        return;
    };
    assert_eq!(pkt.data.len(), data.len());
    assert!(matches!(demux.next_packet(), Err(Error::Eof)));

    let mut params = stream.params.clone();
    params.limits = DecoderLimits::default()
        .with_max_pixels_per_frame(1 << 20)
        .with_max_alloc_bytes_per_frame(64 << 20);
    if let Ok(mut dec) = oxideav_jpegxs::make_decoder(&params) {
        if dec.send_packet(&pkt).is_ok() {
            let _ = dec.receive_frame();
        }
    }

    for open in [open_muxer_jpegxs, open_muxer_jxs] {
        let out: Box<dyn WriteSeek> = Box::new(std::io::Cursor::new(Vec::new()));
        if let Ok(mut mux) = open(out, std::slice::from_ref(&stream)) {
            let _ = mux.write_header();
            let _ = mux.write_packet(&pkt);
            let _ = mux.write_trailer();
        }
    }
});
