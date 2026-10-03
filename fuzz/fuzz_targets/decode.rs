//! Bare-codestream robustness target: arbitrary bytes through the
//! contract surface (`probe`, `info`, `decode`, `decode_rgba8`), the
//! component view, the media-type / declaration-verification helpers and
//! the header inspector. The invariant is "no panic, no unbounded
//! allocation" — every malformed input must surface as an `Err`, never
//! as UB or an abort.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Cheap classifiers first — must never panic on anything.
    let sniff = oxideav_jpegxs::probe(data);
    let _ = oxideav_jpegxs::media_type(data);
    let _ = oxideav_jpegxs::is_jxs_file(data);
    let _ = oxideav_jpegxs::inspect(data);
    let _ = oxideav_jpegxs::verify_declarations(data);
    // Header-only summary.
    let info = oxideav_jpegxs::info(data);
    if info.is_ok() {
        assert!(sniff, "info() succeeded on bytes probe() rejected");
    }
    // Component view (every decodable stream).
    if let Ok(c) = oxideav_jpegxs::decode_components(data) {
        assert_eq!(c.planes.len(), c.bit_depths.len());
        assert_eq!(c.planes.len(), c.sampling.len());
        for p in &c.planes {
            assert!(p.stride > 0 && p.data.len() % p.stride == 0);
        }
    }
    // Contract decode in the default limits, plus the raw RGBA path.
    if let Ok(img) = oxideav_jpegxs::decode(data) {
        let i = info.expect("decode() succeeded, info() must too");
        assert_eq!((i.width, i.height, i.format), (img.width, img.height, img.format));
        assert_eq!(img.planes.len(), img.format.plane_count());
        let rgba = img.to_rgba8();
        assert_eq!(rgba.len(), img.width as usize * img.height as usize * 4);
    }
    // Tight limits must fail cleanly, never allocate past them.
    let tight = oxideav_jpegxs::DecodeOptions::default()
        .with_max_pixels(Some(4096))
        .with_strict(true);
    let _ = oxideav_jpegxs::decode_with(data, &tight);
});
