//! Corpus seed generator (not a fuzz target): writes a handful of
//! valid codestreams / box files spanning the encoder's feature axes
//! into `corpus/decode` and `corpus/jxs_file`, so the byte-level fuzz
//! targets start from structurally deep inputs instead of rediscovering
//! the marker chain from scratch.
//!
//! Run from the `fuzz/` directory: `cargo run --bin seed_gen`.

use std::fs;
use std::path::Path;

fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h);
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(97);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(((x * 3 + y * 5) as u32 ^ (s >> 24)) as u8);
        }
    }
    v
}

fn plane16(w: usize, h: usize, bd: u8, seed: u32) -> Vec<u16> {
    let mask = ((1u32 << bd) - 1) as u16;
    let mut v = Vec::with_capacity(w * h);
    let mut s = seed.wrapping_mul(2246822519).wrapping_add(31);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((((x * 7 + y * 11) as u32 ^ (s >> 20)) as u16) & mask);
        }
    }
    v
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    fs::create_dir_all(dir).expect("mkdir corpus dir");
    fs::write(dir.join(name), bytes).expect("write seed");
}

fn main() {
    use oxideav_jpegxs::encoder;
    let decode_dir = Path::new("corpus/decode");
    let file_dir = Path::new("corpus/jxs_file");
    let (w, h) = (24u16, 20u16);

    // Luma lossless.
    let luma = vec![plane(w as usize, h as usize, 1)];
    let cs = encoder::encode_planar(w, h, 1, 0, 2, 1, &luma).unwrap();
    write(decode_dir, "luma_lossless", &cs);
    write(file_dir, "luma_wrapped", &oxideav_jpegxs::write_jxs_file(&cs).unwrap());

    // RGB with RCT, lossy.
    let rgb = vec![
        plane(w as usize, h as usize, 2),
        plane(w as usize, h as usize, 3),
        plane(w as usize, h as usize, 4),
    ];
    let cs = encoder::encode_planar_lossy(w, h, 3, 1, 3, 1, 2, &rgb).unwrap();
    write(decode_dir, "rgb_rct_lossy", &cs);

    // 4:2:0 sub-sampled.
    let yuv = vec![
        plane(w as usize, h as usize, 5),
        plane(w as usize / 2, h as usize / 2, 6),
        plane(w as usize / 2, h as usize / 2, 7),
    ];
    let cs = encoder::encode_planar_subsampled(w, h, 3, 0, 2, 1, 0, &[1, 2, 2], &[1, 2, 2], &yuv)
        .unwrap();
    write(decode_dir, "yuv420_lossless", &cs);
    write(file_dir, "yuv420_wrapped", &oxideav_jpegxs::write_jxs_file(&cs).unwrap());

    // High bit depth (B[i] = 12).
    let hb: Vec<Vec<u16>> = (0..3)
        .map(|i| plane16(w as usize, h as usize, 12, i + 8))
        .collect();
    let cs = encoder::encode_planar_highbd(w, h, 3, 1, 2, 1, 12, &hb).unwrap();
    write(decode_dir, "highbd12", &cs);

    // Profile-signed CBR (Ppih / Plev / Lcod all non-zero).
    let planes: Vec<Vec<u16>> = vec![
        plane16(64, 64, 10, 20),
        plane16(32, 64, 10, 21),
        plane16(32, 64, 10, 22),
    ];
    let (cs, _, _, _) = encoder::encode_planar_for_profile_cbr_target_bytes(
        oxideav_jpegxs::Profile::Main422_10,
        64,
        64,
        3,
        0,
        2,
        1,
        10,
        1,
        &[1, 2, 2],
        &[1, 1, 1],
        7000,
        &planes,
    )
    .unwrap();
    write(decode_dir, "profile_cbr_main422_10", &cs);
    write(file_dir, "profile_cbr_wrapped", &oxideav_jpegxs::write_jxs_file(&cs).unwrap());

    println!("seeds written to {decode_dir:?} and {file_dir:?}");
}
