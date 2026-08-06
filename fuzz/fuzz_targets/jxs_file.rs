//! Part-3 box-file robustness target: arbitrary bytes through the JXS
//! still-image file format parser (ISO/IEC 21122-3 Annex A box syntax,
//! typed box bodies, ihdr/codestream cross-checks) and the wrapped
//! decode path. Malformed files must error, never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = oxideav_jpegxs::parse_jxs_file(data) {
        // Typed accessors over a structurally valid file must not panic,
        // and the extracted codestream span must lie inside the buffer.
        let cs = file.codestream(data);
        assert!(cs.len() <= data.len());
        let _ = &file.file_type;
        let _ = &file.header.image_header;
        let _ = &file.header.colour_specs;
        let _ = &file.header.channel_def;
        let _ = &file.header.exif;
        let _ = &file.profile_level;
        let _ = &file.video_info;
        let _ = &file.buffer_model;
        let _ = &file.mastering_display;
        let _ = &file.transport_params;
    }
    let _ = oxideav_jpegxs::decode_jxs_file(data);
});
