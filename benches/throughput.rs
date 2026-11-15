//! Throughput probe (plain `main`, no harness dependency).
//!
//! `cargo bench --bench throughput` prints megapixels per second
//! (picture pixels, all components counted once) for:
//!
//! * the wavelet cascades alone, forward and inverse, on one plane;
//! * a full encode and a full decode of a synthetic 1920×1080 picture
//!   (three 8-bit components, reversible colour transform, `NL = 5/2`,
//!   lossless, and one 4:2:2 10-bit `Fq = 8` lossy variant);
//! * decode of ISO/IEC 21122-4 codestreams when
//!   `OXIDEAV_JXS_CONFORMANCE_DIR` points at the unpacked vectors
//!   (`OXIDEAV_JXS_BENCH_STREAMS=5,29,60` picks the streams).
//!
//! Each figure is the median of `OXIDEAV_JXS_BENCH_RUNS` (default 5)
//! runs. `OXIDEAV_JXS_BENCH_FILTER=decode` times only the rows whose
//! label contains the given text (handy under a sampling profiler).
#![allow(deprecated)]

use std::time::Instant;

use oxideav_jpegxs::{dwt, encoder};

fn runs() -> usize {
    std::env::var("OXIDEAV_JXS_BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
}

/// Whether the row `label` is selected by `OXIDEAV_JXS_BENCH_FILTER`.
fn selected(label: &str) -> bool {
    std::env::var("OXIDEAV_JXS_BENCH_FILTER").map_or(true, |f| label.contains(&f))
}

/// Median wall time in seconds of `f` over the configured run count
/// (a single untimed-quality run when the row is filtered out, so later
/// rows still get their inputs).
fn median_secs(label: &str, mut f: impl FnMut()) -> f64 {
    let n = if selected(label) { runs() } else { 1 };
    let mut t: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(|a, b| a.total_cmp(b));
    t[t.len() / 2]
}

fn report(label: &str, pixels: usize, secs: f64) {
    if !selected(label) {
        return;
    }
    println!(
        "{label:<46} {:>8.1} MP/s  ({:.1} ms)",
        pixels as f64 / 1e6 / secs,
        secs * 1e3
    );
}

/// Smooth picture content with texture: gradients, a few edges, noise.
fn picture(w: usize, h: usize, seed: u32, bits: u8) -> Vec<u16> {
    let max = (1u32 << bits) - 1;
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let base = (x * 3 + y * 2) as u32 * max / ((w * 3 + h * 2) as u32);
            let edge = if (x / 97 + y / 61) % 2 == 0 {
                max / 6
            } else {
                0
            };
            let n = (s >> 24) * max / 2048;
            v.push((base / 2 + edge + n).min(max) as u16);
        }
    }
    v
}

/// Time one row and print it.
fn row(label: &str, pixels: usize, f: impl FnMut()) {
    let t = median_secs(label, f);
    report(label, pixels, t);
}

fn main() {
    let (w, h) = (1920usize, 1080usize);
    let px = w * h;

    // --- Wavelet cascades alone. ---
    let plane: Vec<i32> = picture(w, h, 7, 8)
        .iter()
        .map(|&v| v as i32 - 128)
        .collect();
    for (nlx, nly) in [(5u8, 0u8), (5, 1), (5, 2)] {
        let mut bands = Vec::new();
        row(
            &format!("forward cascade NL={nlx}/{nly} (1 plane)"),
            px,
            || bands = dwt::forward_cascade_2d(w, h, nlx, nly, &plane).unwrap(),
        );
        let mut out = vec![0i32; px];
        row(
            &format!("inverse cascade NL={nlx}/{nly} (1 plane)"),
            px,
            || dwt::inverse_cascade_2d(w, h, nlx, nly, &bands, &mut out).unwrap(),
        );
        assert_eq!(out, plane);
    }

    // --- Full codec, 4:4:4 8-bit RCT. ---
    let rgb: Vec<Vec<u8>> = (0..3)
        .map(|c| picture(w, h, 11 + c, 8).iter().map(|&v| v as u8).collect())
        .collect();
    let mut buf = Vec::new();
    row("encode 1080p 4:4:4 8-bit RCT lossless NL=5/2", px, || {
        buf = encoder::encode_planar(w as u16, h as u16, 3, 1, 5, 2, &rgb).unwrap()
    });
    println!(
        "    {} bytes ({:.2} bpp)",
        buf.len(),
        buf.len() as f64 * 8.0 / px as f64
    );
    row("decode 1080p 4:4:4 8-bit RCT lossless NL=5/2", px, || {
        oxideav_jpegxs::decode_components(&buf).unwrap();
    });
    row("encode 1080p 4:4:4 8-bit RCT Fq=8 q=4 NL=5/2", px, || {
        buf =
            encoder::encode_planar_highprec_lossy(w as u16, h as u16, 3, 1, 5, 2, 4, &rgb).unwrap()
    });
    println!(
        "    {} bytes ({:.2} bpp)",
        buf.len(),
        buf.len() as f64 * 8.0 / px as f64
    );
    row("decode 1080p 4:4:4 8-bit RCT Fq=8 q=4 NL=5/2", px, || {
        oxideav_jpegxs::decode_components(&buf).unwrap();
    });

    // --- Part 4 vectors. ---
    let Some(dir) = std::env::var_os("OXIDEAV_JXS_CONFORMANCE_DIR") else {
        println!("(set OXIDEAV_JXS_CONFORMANCE_DIR to time ISO/IEC 21122-4 codestreams)");
        return;
    };
    let pick = std::env::var("OXIDEAV_JXS_BENCH_STREAMS").unwrap_or_else(|_| "5,29,60".into());
    for n in pick.split(',').filter_map(|s| s.trim().parse::<u32>().ok()) {
        let path = std::path::Path::new(&dir).join(format!("{n}.jxs"));
        let Ok(jxs) = std::fs::read(&path) else {
            println!("stream {n}: not found");
            continue;
        };
        let info = oxideav_jpegxs::codestream::parse(&jxs).unwrap().pih;
        let pixels = info.wf as usize * info.hf as usize;
        row(
            &format!(
                "decode Part 4 stream {n} ({}x{} NL={}/{})",
                info.wf, info.hf, info.nlx, info.nly
            ),
            pixels,
            || {
                oxideav_jpegxs::decode_components(&jxs).unwrap();
            },
        );
    }
}
