//! Decoder robustness: mutated codestreams must be rejected or decoded,
//! never panic. Deterministic companion to the `fuzz/` targets that runs
//! in the ordinary test suite (and with overflow checks in debug builds).
#![allow(deprecated)]

use oxideav_jpegxs::encoder;

fn xorshift(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..w * h)
        .map(|i| ((i % w) as u32 * 3 + (xorshift(&mut s) >> 26)) as u8)
        .collect()
}

#[test]
fn mutated_streams_never_panic() {
    let (w, h) = (48usize, 24usize);
    let planes: Vec<Vec<u8>> = (0..3).map(|c| plane(w, h, 5 + c)).collect();
    let cfa: Vec<Vec<u8>> = (0..4).map(|c| plane(w, h, 50 + c)).collect();
    let yuv420 = vec![
        plane(w, h, 60),
        plane(w / 2, h / 2, 61),
        plane(w / 2, h / 2, 62),
    ];
    let seeds: Vec<Vec<u8>> = vec![
        encoder::encode_planar(w as u16, h as u16, 3, 1, 5, 2, &planes).unwrap(),
        encoder::encode_planar(w as u16, h as u16, 3, 0, 3, 0, &planes).unwrap(),
        encoder::encode_planar(w as u16, h as u16, 1, 0, 2, 1, &planes[..1]).unwrap(),
        encoder::encode_planar_highprec_lossy(w as u16, h as u16, 3, 1, 4, 1, 3, &planes).unwrap(),
        encoder::encode_planar_lossy(w as u16, h as u16, 3, 1, 5, 2, 5, &planes).unwrap(),
        // Star-Tetrix (four components) and a quadratic non-linearity.
        encoder::encode_planar_star_tetrix(w as u16, h as u16, 2, 2, 0, 1, 1, 0, 0, &cfa).unwrap(),
        encoder::encode_planar_nlt_quadratic(w as u16, h as u16, 1, 0, 2, 1, 0, 0, &planes[..1])
            .unwrap(),
        // 4:2:0.
        encoder::encode_planar_subsampled(
            w as u16,
            h as u16,
            3,
            0,
            3,
            1,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &yuv420,
        )
        .unwrap(),
    ];
    // `OXIDEAV_JXS_ROBUSTNESS_ROUNDS` deepens the sweep for a local soak.
    let rounds: u32 = std::env::var("OXIDEAV_JXS_ROBUSTNESS_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let mut s = 0x2545_F491u32;
    let (mut ok, mut rejected) = (0u32, 0u32);
    for seed in &seeds {
        assert!(oxideav_jpegxs::decode_components(seed).is_ok());
        for round in 0..rounds {
            let mut buf = seed.clone();
            match round % 4 {
                // Flip one to three bits, biased to the entropy-coded part.
                0 | 1 => {
                    for _ in 0..1 + xorshift(&mut s) % 3 {
                        let at = xorshift(&mut s) as usize % buf.len();
                        buf[at] ^= 1 << (xorshift(&mut s) % 8);
                    }
                }
                // Overwrite a short run.
                2 => {
                    let at = xorshift(&mut s) as usize % buf.len();
                    let n = (1 + xorshift(&mut s) as usize % 6).min(buf.len() - at);
                    for b in &mut buf[at..at + n] {
                        *b = xorshift(&mut s) as u8;
                    }
                }
                // Truncate.
                _ => {
                    let keep = xorshift(&mut s) as usize % buf.len();
                    buf.truncate(keep);
                }
            }
            match oxideav_jpegxs::decode_components(&buf) {
                Ok(_) => ok += 1,
                Err(_) => rejected += 1,
            }
        }
    }
    // Both outcomes occur: header damage is rejected, most payload
    // damage still decodes.
    assert!(ok > 0 && rejected > 0, "ok={ok} rejected={rejected}");
}

/// Fuzz-found input (decode target): a picture header whose column width
/// `Cw` makes `Cs = 8 × Cw × max(sx) × 2^NL,x` exceed 32 bits. The plan
/// builder multiplied in `u32` and tripped the overflow check; any such
/// `Cs` simply means one precinct per row.
#[test]
fn oversized_column_width_is_handled() {
    const INPUT: &[u8] = &[
        0xff, 0x10, 0xff, 0x50, 0x00, 0x04, 0x00, 0x80, 0xff, 0x12, 0x00, 0x1a, 0x00, 0x00, 0x1b,
        0x58, 0x32, 0x40, 0x10, 0x04, 0x00, 0x40, 0x00, 0x40, 0x80, 0xff, 0x11, 0x08, 0x03, 0x04,
        0x08, 0x0a, 0x08, 0x00, 0xed, 0x50, 0xff, 0x13, 0x00, 0x08, 0x0a, 0x11, 0x0a, 0x21, 0x0a,
        0x21, 0xff, 0x14, 0x00, 0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x01, 0x03, 0x01, 0x04,
        0x01, 0x05, 0x01, 0x06, 0x01, 0x07, 0x1c, 0x08, 0x01, 0x09, 0x01, 0x0a, 0x01, 0x0b, 0x02,
        0x0c, 0x02, 0x0d, 0x02, 0x0e, 0xff, 0x15, 0x00, 0x07, 0x80, 0x00, 0x00, 0x00, 0x00, 0xff,
        0x20, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0xc9, 0x07, 0x00, 0x00, 0x00, 0x96, 0xff, 0x12,
        0x00, 0x1a, 0x00, 0x00, 0x1b, 0x58, 0x35, 0x40, 0x10, 0x04, 0x00, 0x40, 0x00, 0x40, 0x00,
        0x00, 0x00, 0x08, 0x03, 0x04, 0x08, 0x48, 0xec, 0x89, 0xf2, 0xde, 0xef, 0x27, 0xe2, 0x16,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x87, 0x18, 0xf9, 0xd2, 0x2a, 0xe7, 0xf6, 0x41,
        0x19, 0x42, 0x70, 0x00, 0x25, 0x00, 0x50, 0x00, 0xef, 0x77, 0x7b, 0xbd, 0xef, 0x7b, 0xbd,
        0xee, 0xf7, 0x00, 0x56, 0xff, 0x10, 0xff, 0xf8, 0xf2, 0x1e, 0x3f, 0xf7, 0x94, 0x98, 0x71,
        0xc1, 0x2a, 0xd2, 0x14, 0x24, 0x26, 0x34, 0x87, 0x50, 0x00, 0x04, 0x00, 0x80, 0xff, 0x12,
        0x22, 0xc8, 0x3f, 0x42, 0xa4, 0x40, 0x84, 0xc1, 0x2a, 0xd2, 0x75, 0x00, 0x1a, 0x00,
    ];
    // Must return (accepting or rejecting), not panic.
    let _ = oxideav_jpegxs::decode_components(INPUT);
    let _ = oxideav_jpegxs::inspect(INPUT);
}
