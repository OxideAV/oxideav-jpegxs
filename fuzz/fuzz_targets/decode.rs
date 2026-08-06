//! Bare-codestream robustness target: arbitrary bytes through the
//! probe, media-type, declaration-verification and full-decode paths.
//! The invariant is "no panic, no unbounded allocation" — every
//! malformed input must surface as an `Err`, never as UB or an abort.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Cheap classifiers first — must never panic on anything.
    let _ = oxideav_jpegxs::media_type(data);
    let _ = oxideav_jpegxs::is_jxs_file(data);
    let _ = oxideav_jpegxs::probe(data);
    let _ = oxideav_jpegxs::verify_declarations(data);
    // Full decode (marker chain + entropy + DWT + output scaling).
    if let Ok(img) = oxideav_jpegxs::decode_jpeg_xs(data) {
        // A successful decode must be internally consistent.
        assert_eq!(img.planes.len(), img.num_components as usize);
        for p in &img.planes {
            assert!(p.stride > 0 && p.data.len() % p.stride == 0);
        }
    }
});
