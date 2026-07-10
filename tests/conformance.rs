//! ISO/IEC 21122-4:2020 decoder conformance harness.
//!
//! The reference vectors are ~1.4 GB of codestreams paired with
//! sample-exact reference decoded images, catalogued (with per-file
//! SHA-256) under `docs/image/jpegxs/conformance/` but *not* committed to
//! git. This harness therefore has two modes:
//!
//! * **Data present** — when the environment variable
//!   `OXIDEAV_JXS_CONFORMANCE_DIR` points at a directory holding the
//!   unpacked attachment files (`N.jxs`, `N.pgx`, `N_k.h`, `N_k.raw`),
//!   [`run_conformance_dir`] decodes each codestream and compares every
//!   reconstructed component plane sample-by-sample against its reference
//!   (ISO/IEC 21122-4 §B.8: a pass is exact equality on every sample of
//!   every component). Streams that exercise a tool this decoder does not
//!   yet implement are counted separately (`Unsupported`) rather than
//!   failed, so the harness reports true conformance coverage.
//!
//! * **Data absent** (the CI default) — the `pgx` reader and the
//!   sample-comparison logic are exercised end-to-end against a synthetic
//!   fixture built by our own encoder, plus literal byte-layout unit
//!   tests pinned to the §B.7 format description, so this file always has
//!   live test coverage without the external attachments.
//!
//! The reference-image container is the ISO `pgx` format (§B.7):
//! a directory file listing one raw plane per component, a one-line ASCII
//! header per plane (`P<fmt> <endian> <sign><precision> <width> <height>`),
//! and a raw raster-order plane (1 byte/sample if precision ≤ 8, else 2
//! bytes/sample in the header's endianness).

use std::path::{Path, PathBuf};

/// One reference component plane parsed from a `pgx` header + raw file
/// (ISO/IEC 21122-4:2020 §B.7.3 / §B.7.4).
#[derive(Debug, Clone)]
struct PgxComponent {
    width: u32,
    height: u32,
    precision: u8,
    samples: Vec<u32>,
}

/// Parse a `pgx` component header line (§B.7.3):
/// `P<data-format> <endianness> <signedness><precision> <width> <height>`,
/// e.g. `PG ML +8 4064 2704`. Returns `(big_endian, precision, width,
/// height)`. Only the spec-defined field values are accepted.
fn parse_pgx_header(line: &str) -> Result<(bool, u8, u32, u32), String> {
    let line = line.trim_end_matches(['\n', '\r']);
    let mut fields = line.split(' ');
    let fmt = fields
        .next()
        .ok_or("pgx header: missing data-format field")?;
    // "P" identifies the file; "G" is the only defined data format
    // (integer samples). §B.7.3.
    if fmt != "PG" {
        return Err(format!(
            "pgx header: unsupported data format {fmt:?} (only PG)"
        ));
    }
    let endian = fields
        .next()
        .ok_or("pgx header: missing endianness field")?;
    let big_endian = match endian {
        "ML" => true,
        "LM" => false,
        other => return Err(format!("pgx header: bad endianness {other:?}")),
    };
    let signprec = fields
        .next()
        .ok_or("pgx header: missing signedness/precision")?;
    // §B.7.3: ISO/IEC 21122 covers unsigned samples only ('+').
    let prec_str = signprec
        .strip_prefix('+')
        .ok_or_else(|| format!("pgx header: signedness must be '+', got {signprec:?}"))?;
    let precision: u8 = prec_str
        .parse()
        .map_err(|_| format!("pgx header: bad precision {prec_str:?}"))?;
    let width: u32 = fields
        .next()
        .ok_or("pgx header: missing width")?
        .parse()
        .map_err(|_| "pgx header: bad width".to_string())?;
    let height: u32 = fields
        .next()
        .ok_or("pgx header: missing height")?
        .parse()
        .map_err(|_| "pgx header: bad height".to_string())?;
    Ok((big_endian, precision, width, height))
}

/// Decode a raw plane (§B.7.4) into per-sample integer values. `≤ 8`-bit
/// precision packs one byte per sample; higher precisions pack two bytes
/// in the header's endianness. Both are right-aligned (high bits zero).
fn decode_pgx_raw(
    raw: &[u8],
    big_endian: bool,
    precision: u8,
    width: u32,
    height: u32,
) -> Result<Vec<u32>, String> {
    let count = width as usize * height as usize;
    if precision <= 8 {
        if raw.len() != count {
            return Err(format!(
                "pgx raw: expected {count} bytes ({width}x{height}), got {}",
                raw.len()
            ));
        }
        Ok(raw.iter().map(|&b| b as u32).collect())
    } else {
        if raw.len() != count * 2 {
            return Err(format!(
                "pgx raw: expected {} bytes ({width}x{height} @16b), got {}",
                count * 2,
                raw.len()
            ));
        }
        Ok(raw
            .chunks_exact(2)
            .map(|c| {
                if big_endian {
                    ((c[0] as u32) << 8) | c[1] as u32
                } else {
                    ((c[1] as u32) << 8) | c[0] as u32
                }
            })
            .collect())
    }
}

/// Read the reference component set for stream `n` from `dir`
/// (`n.pgx` directory file → one `n_k.h` + `n_k.raw` per component).
fn read_reference(dir: &Path, n: u32) -> Result<Vec<PgxComponent>, String> {
    let pgx_path = dir.join(format!("{n}.pgx"));
    let dir_file = std::fs::read_to_string(&pgx_path)
        .map_err(|e| format!("read {}: {e}", pgx_path.display()))?;
    let mut comps = Vec::new();
    for name in dir_file.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let raw_path = dir.join(name);
        let h_name = name
            .strip_suffix(".raw")
            .map(|stem| format!("{stem}.h"))
            .ok_or_else(|| format!("pgx directory entry {name:?} lacks .raw suffix"))?;
        let h_path = dir.join(&h_name);
        let header = std::fs::read_to_string(&h_path)
            .map_err(|e| format!("read {}: {e}", h_path.display()))?;
        let first = header.lines().next().unwrap_or("");
        let (big_endian, precision, width, height) = parse_pgx_header(first)?;
        let raw =
            std::fs::read(&raw_path).map_err(|e| format!("read {}: {e}", raw_path.display()))?;
        let samples = decode_pgx_raw(&raw, big_endian, precision, width, height)?;
        comps.push(PgxComponent {
            width,
            height,
            precision,
            samples,
        });
    }
    if comps.is_empty() {
        return Err("pgx directory file listed no components".into());
    }
    Ok(comps)
}

/// Compare one decoded [`oxideav_jpegxs::JpegXsImage`] against the
/// reference component set (§B.8 — exact sample equality). Returns the
/// index of the first mismatch on failure.
fn compare_image(
    img: &oxideav_jpegxs::JpegXsImage,
    reference: &[PgxComponent],
) -> Result<(), String> {
    if img.planes.len() != reference.len() {
        return Err(format!(
            "component count: decoded {} vs reference {}",
            img.planes.len(),
            reference.len()
        ));
    }
    for (k, (plane, refc)) in img.planes.iter().zip(reference.iter()).enumerate() {
        // The decoder packs each plane at the *component* precision B[i]
        // (`decoder.rs`: one byte/sample when B[i] ≤ 8, else two
        // little-endian bytes) — not at the picture-wide `Bw`
        // (`img.bit_depth`), which can be 20 even for an 8-bit component.
        // The reference `.h` precision equals B[i], so use it per plane.
        let bytes_per_sample = if refc.precision <= 8 { 1 } else { 2 };
        let stride_samples = plane.stride / bytes_per_sample;
        let plane_w = stride_samples;
        let plane_h = if stride_samples == 0 {
            0
        } else {
            plane.data.len() / (stride_samples * bytes_per_sample)
        };
        if plane_w as u32 != refc.width || plane_h as u32 != refc.height {
            return Err(format!(
                "component {k}: geometry decoded {plane_w}x{plane_h} vs reference {}x{}",
                refc.width, refc.height
            ));
        }
        for y in 0..refc.height as usize {
            for x in 0..refc.width as usize {
                let off = y * plane.stride + x * bytes_per_sample;
                let got = if bytes_per_sample == 1 {
                    plane.data[off] as u32
                } else {
                    (plane.data[off] as u32) | ((plane.data[off + 1] as u32) << 8)
                };
                let want = refc.samples[y * refc.width as usize + x];
                if got != want {
                    return Err(format!(
                        "component {k}: sample ({x},{y}) decoded {got} vs reference {want} \
                         (precision {})",
                        refc.precision
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Discover every stream index `n` for which both `n.jxs` and `n.pgx`
/// exist in `dir`, sorted ascending.
fn discover_streams(dir: &Path) -> Vec<u32> {
    let mut streams = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return streams;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(stem) = name.strip_suffix(".jxs") {
            if let Ok(n) = stem.parse::<u32>() {
                if dir.join(format!("{n}.pgx")).exists() {
                    streams.push(n);
                }
            }
        }
    }
    streams.sort_unstable();
    streams
}

enum Outcome {
    Pass,
    Unsupported(String),
    Fail(String),
}

/// Decode + compare one stream. A codestream that uses a not-yet-decodable
/// tool surfaces as `Unsupported` (not `Fail`).
fn run_stream(dir: &Path, n: u32) -> Outcome {
    let jxs = match std::fs::read(dir.join(format!("{n}.jxs"))) {
        Ok(b) => b,
        Err(e) => return Outcome::Fail(format!("read {n}.jxs: {e}")),
    };
    let decoded = if oxideav_jpegxs::is_jxs_file(&jxs) {
        oxideav_jpegxs::decode_jxs_file(&jxs)
    } else {
        oxideav_jpegxs::decode_jpeg_xs(&jxs)
    };
    let img = match decoded {
        Ok(img) => img,
        Err(oxideav_jpegxs::JpegXsError::Unsupported(m)) => return Outcome::Unsupported(m),
        Err(e) => return Outcome::Fail(format!("decode error: {e}")),
    };
    let reference = match read_reference(dir, n) {
        Ok(r) => r,
        Err(e) => return Outcome::Fail(format!("reference: {e}")),
    };
    match compare_image(&img, &reference) {
        Ok(()) => Outcome::Pass,
        Err(e) => Outcome::Fail(e),
    }
}

fn conformance_dir() -> Option<PathBuf> {
    std::env::var_os("OXIDEAV_JXS_CONFORMANCE_DIR").map(PathBuf::from)
}

#[test]
fn iso_21122_4_conformance_vectors() {
    let Some(dir) = conformance_dir() else {
        eprintln!(
            "OXIDEAV_JXS_CONFORMANCE_DIR unset — skipping ISO/IEC 21122-4 vector run \
             (catalog: docs/image/jpegxs/conformance/). pgx reader is covered by the \
             synthetic + unit tests below."
        );
        return;
    };
    let streams = discover_streams(&dir);
    assert!(
        !streams.is_empty(),
        "OXIDEAV_JXS_CONFORMANCE_DIR={} holds no N.jxs/N.pgx pairs",
        dir.display()
    );
    let (mut pass, mut unsup, mut fail) = (0u32, 0u32, 0u32);
    let mut failures = Vec::new();
    for n in &streams {
        match run_stream(&dir, *n) {
            Outcome::Pass => {
                pass += 1;
                eprintln!("  stream {n}: PASS");
            }
            Outcome::Unsupported(m) => {
                unsup += 1;
                eprintln!("  stream {n}: UNSUPPORTED — {m}");
            }
            Outcome::Fail(m) => {
                fail += 1;
                eprintln!("  stream {n}: FAIL — {m}");
                failures.push((*n, m));
            }
        }
    }
    eprintln!(
        "ISO/IEC 21122-4: {pass} pass, {unsup} unsupported, {fail} fail / {} streams",
        streams.len()
    );
    assert!(fail == 0, "conformance failures: {failures:?}");
}

// ------------------------------------------------------------------
// pgx-reader coverage that runs with no external data (CI default).
// ------------------------------------------------------------------

#[test]
fn pgx_header_parses_spec_example() {
    // §B.7.3 worked example.
    let (big_endian, precision, width, height) = parse_pgx_header("PG ML +8 4064 2704").unwrap();
    assert!(big_endian);
    assert_eq!(precision, 8);
    assert_eq!(width, 4064);
    assert_eq!(height, 2704);
    // Trailing line feed (§B.7.3 terminates the line with 0x0a) tolerated.
    let (be, p, w, h) = parse_pgx_header("PG LM +12 12287 1743\n").unwrap();
    assert!(!be);
    assert_eq!((p, w, h), (12, 12287, 1743));
}

#[test]
fn pgx_header_rejects_malformed() {
    assert!(parse_pgx_header("PX ML +8 4 4").is_err()); // bad data format
    assert!(parse_pgx_header("PG XX +8 4 4").is_err()); // bad endianness
    assert!(parse_pgx_header("PG ML -8 4 4").is_err()); // signed not allowed
    assert!(parse_pgx_header("PG ML +8 4").is_err()); // missing height
}

#[test]
fn pgx_raw_byte_layout_8bit_and_16bit() {
    // 8-bit: one byte per sample.
    let s = decode_pgx_raw(&[0, 1, 254, 255], true, 8, 2, 2).unwrap();
    assert_eq!(s, [0, 1, 254, 255]);
    // 12-bit big-endian (ML): most-significant byte first, right-aligned.
    let be = decode_pgx_raw(&[0x0A, 0xBC, 0x00, 0x01], true, 12, 2, 1).unwrap();
    assert_eq!(be, [0x0ABC, 0x0001]);
    // 12-bit little-endian (LM): least-significant byte first.
    let le = decode_pgx_raw(&[0xBC, 0x0A, 0x01, 0x00], false, 12, 2, 1).unwrap();
    assert_eq!(le, [0x0ABC, 0x0001]);
    // Wrong length is an error, not a panic.
    assert!(decode_pgx_raw(&[0, 1, 2], true, 8, 2, 2).is_err());
}

#[test]
fn synthetic_encode_decode_matches_pgx_roundtrip() {
    // Build a small 8-bit luma image, encode losslessly, decode, and drive
    // it through the same reference-comparison path the vector harness uses
    // — writing the decoded plane out as a pgx component and reading it
    // back, so the reader/comparator are exercised with no external data.
    let (w, h) = (16u16, 12u16);
    let pixels: Vec<u8> = (0..(w as usize * h as usize))
        .map(|i| ((i * 7 + 3) & 0xff) as u8)
        .collect();
    let codestream = oxideav_jpegxs::encoder::encode_luma_8bit(w, h, &pixels).unwrap();
    let img = oxideav_jpegxs::decode_jpeg_xs(&codestream).unwrap();
    assert_eq!(img.planes.len(), 1);

    // Serialize the decoded plane to a pgx raw buffer (8-bit path) and
    // parse it back through the reader.
    let plane = &img.planes[0];
    let mut raw = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h as usize {
        for x in 0..w as usize {
            raw.push(plane.data[y * plane.stride + x]);
        }
    }
    let header = format!("PG ML +8 {w} {h}");
    let (be, prec, rw, rh) = parse_pgx_header(&header).unwrap();
    let samples = decode_pgx_raw(&raw, be, prec, rw, rh).unwrap();
    let reference = vec![PgxComponent {
        width: rw,
        height: rh,
        precision: prec,
        samples,
    }];
    compare_image(&img, &reference).expect("decoded image matches its own pgx serialization");

    // A deliberately corrupted reference must be detected (§B.8: any
    // deviation fails).
    let mut bad = reference;
    bad[0].samples[0] ^= 1;
    assert!(compare_image(&img, &bad).is_err());
}
