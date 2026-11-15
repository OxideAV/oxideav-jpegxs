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
        .unwrap_or(1500);
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
