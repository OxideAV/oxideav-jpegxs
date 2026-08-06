//! Structured encode → decode invariant target: fuzzer bytes pick an
//! encoder configuration (dimensions, components, colour transform,
//! cascade depths, sub-sampling, quantization) plus the plane samples.
//! Whenever the encoder accepts the configuration, the emitted
//! codestream must decode — and bit-exactly at `q = 0`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (hdr, rest) = data.split_at(8);
    let width = 1 + (hdr[0] as u16) % 48;
    let height = 1 + (hdr[1] as u16) % 48;
    let nc = 1 + hdr[2] % 4;
    // Colour transform: RCT only with >= 3 full-resolution components.
    let cpih = if hdr[3] & 1 == 1 && nc >= 3 { 1u8 } else { 0u8 };
    let nlx = hdr[4] % 6;
    let nly = (hdr[5] % 3).min(nlx);
    let q = hdr[6] % 16;
    // Per-component sub-sampling from one bit pair each (Cpih = 0 only).
    let mut sx = Vec::with_capacity(nc as usize);
    let mut sy = Vec::with_capacity(nc as usize);
    for i in 0..nc {
        if cpih == 0 && i > 0 {
            sx.push(1 + ((hdr[7] >> (2 * (i - 1))) & 1));
            sy.push(1 + ((hdr[7] >> (2 * (i - 1) + 1)) & 1));
        } else {
            sx.push(1);
            sy.push(1);
        }
    }
    // Deterministic plane fill from the remaining fuzzer bytes.
    let planes: Vec<Vec<u8>> = (0..nc as usize)
        .map(|i| {
            let wc = (width as usize).div_ceil(sx[i] as usize);
            let hc = (height as usize).div_ceil(sy[i] as usize);
            (0..wc * hc)
                .map(|k| {
                    if rest.is_empty() {
                        (k as u8).wrapping_mul(31)
                    } else {
                        rest[(k + i * 7) % rest.len()]
                    }
                })
                .collect()
        })
        .collect();
    let Ok(buf) = oxideav_jpegxs::encoder::encode_planar_subsampled(
        width, height, nc, cpih, nlx, nly, q, &sx, &sy, &planes,
    ) else {
        // A rejected configuration is fine; an accepted one must
        // round-trip below.
        return;
    };
    let img = oxideav_jpegxs::decode_jpeg_xs(&buf)
        .expect("every emitted codestream must decode");
    assert_eq!(img.planes.len(), nc as usize);
    if q == 0 {
        for (i, p) in img.planes.iter().enumerate() {
            assert_eq!(p.data, planes[i], "plane {i} lossless round-trip");
        }
    }
});
