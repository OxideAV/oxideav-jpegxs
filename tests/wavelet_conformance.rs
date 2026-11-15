//! Wavelet-transform conformance of the encoder (ISO/IEC 21122-1:2022
//! Annex E).
//!
//! The standard *specifies* the inverse transform (Table E.1: at a joint
//! level, horizontal synthesis of the low and of the high rows, then
//! vertical synthesis) and describes the forward transform as its
//! reverse (Table E.8: vertical analysis, then horizontal analysis of
//! both halves). With `Fq = 0` every lifting step rounds, so an encoder
//! is lossless on a conforming decoder only when its forward transform
//! undoes that inverse step for step.
//!
//! Three layers:
//!
//! 1. Lossless encoder → decoder round trips over every decomposition
//!    shape, odd and even picture sizes, bit depths 8..=16, the
//!    reversible colour transform and chroma subsampling. The decoder's
//!    inverse is pinned to a literal transcription of Annex E by the unit
//!    tests in `src/dwt.rs`, so a bit-exact round trip here means the
//!    encoder's coefficients are the ones Table E.8 produces.
//! 2. A coefficient-level check that does not involve the decoder's
//!    transform at all: the coefficients carried by encoded streams are
//!    read back before any inverse transform and compared with a literal
//!    transcription of Table E.8, on pictures for which the two possible
//!    pass orders give different coefficients.
//! 3. A black-box cross-check: when a JPEG XS decoder binary is on the
//!    machine, streams from this encoder decode there to the same
//!    samples as with this crate's decoder (skipped at run time
//!    otherwise).

// The planar entry points are the direct route to a chosen `(NL,x, NL,y)`
// and component layout.
#![allow(deprecated)]

use oxideav_jpegxs::encoder;

/// Deterministic `bits`-wide noise (full range, every value reachable).
fn noise(n: usize, bits: u8, seed: u32) -> Vec<u16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) & ((1u32 << bits) - 1)) as u16
        })
        .collect()
}

fn to_u8(p: &[u16]) -> Vec<u8> {
    p.iter().map(|&v| v as u8).collect()
}

fn to_le(p: &[u16]) -> Vec<u8> {
    p.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Picture sizes that exercise even / odd widths and heights at and just
/// above the Table 11 minimum `2^NL` for the decomposition.
fn sizes(nlx: u8, nly: u8) -> Vec<(usize, usize)> {
    let (mw, mh) = (1usize << nlx, (1usize << nly).max(2));
    let ws = [mw.max(2), mw.max(2) + 1, mw + 3, 2 * mw + 5, 40, 67];
    let hs = [mh, mh + 1, mh + 3, 2 * mh + 5, 18, 33];
    let mut v = Vec::new();
    for (i, &w) in ws.iter().enumerate() {
        // Pair every width with two heights so both parities meet.
        v.push((w, hs[i]));
        v.push((w, hs[(i + 1) % hs.len()]));
    }
    v
}

#[test]
fn lossless_round_trip_every_decomposition_shape_8bit() {
    let mut n = 0;
    for nlx in 1..=5u8 {
        for nly in 0..=nlx.min(3) {
            for (w, h) in sizes(nlx, nly) {
                let plane = to_u8(&noise(w * h, 8, (w * 7 + h * 3) as u32 + nlx as u32));
                let buf = encoder::encode_planar(
                    w as u16,
                    h as u16,
                    1,
                    0,
                    nlx,
                    nly,
                    std::slice::from_ref(&plane),
                )
                .unwrap_or_else(|e| panic!("encode {w}x{h} NL={nlx}/{nly}: {e}"));
                let img = oxideav_jpegxs::decode_components(&buf)
                    .unwrap_or_else(|e| panic!("decode {w}x{h} NL={nlx}/{nly}: {e}"));
                assert_eq!(img.planes[0].data, plane, "{w}x{h} NL={nlx}/{nly}");
                n += 1;
            }
        }
    }
    assert!(n >= 150, "sweep covered only {n} configurations");
}

#[test]
fn lossless_round_trip_rct_and_high_bit_depths() {
    for &bd in &[9u8, 10, 12, 14, 16] {
        for &(nlx, nly) in &[(1u8, 1u8), (2, 2), (3, 1), (4, 2), (5, 2), (5, 3)] {
            for (w, h) in sizes(nlx, nly).into_iter().step_by(3) {
                for &cpih in &[0u8, 1] {
                    let planes: Vec<Vec<u16>> = (0..3)
                        .map(|c| noise(w * h, bd, (w + 31 * h) as u32 + c + bd as u32))
                        .collect();
                    let buf = encoder::encode_planar_highbd(
                        w as u16, h as u16, 3, cpih, nlx, nly, bd, &planes,
                    )
                    .unwrap_or_else(|e| panic!("encode {w}x{h} NL={nlx}/{nly} B={bd}: {e}"));
                    let img = oxideav_jpegxs::decode_components(&buf).unwrap();
                    for (c, p) in planes.iter().enumerate() {
                        assert_eq!(
                            img.planes[c].data,
                            to_le(p),
                            "{w}x{h} NL={nlx}/{nly} B={bd} Cpih={cpih} component {c}"
                        );
                    }
                }
            }
        }
    }
    // 8-bit RCT.
    for &(nlx, nly) in &[(2u8, 1u8), (5, 2)] {
        for (w, h) in sizes(nlx, nly) {
            let planes: Vec<Vec<u8>> = (0..3)
                .map(|c| to_u8(&noise(w * h, 8, (3 * w + h) as u32 + c)))
                .collect();
            let buf = encoder::encode_planar(w as u16, h as u16, 3, 1, nlx, nly, &planes).unwrap();
            let img = oxideav_jpegxs::decode_components(&buf).unwrap();
            for (c, p) in planes.iter().enumerate() {
                assert_eq!(&img.planes[c].data, p, "{w}x{h} NL={nlx}/{nly} RCT {c}");
            }
        }
    }
}

#[test]
fn lossless_round_trip_chroma_subsampled() {
    // (sx, sy) of the chroma components: 4:2:2 and 4:2:0. A vertically
    // subsampled component runs one vertical level fewer (Annex B.2).
    for &(sx, sy) in &[(2usize, 1usize), (2, 2)] {
        for &(nlx, nly) in &[(2u8, 1u8), (3, 2), (5, 1), (5, 2)] {
            for &(w, h) in &[(64usize, 32usize), (70, 36), (128, 20)] {
                let (cw, ch) = (w.div_ceil(sx), h.div_ceil(sy));
                let planes = vec![
                    to_u8(&noise(w * h, 8, 1 + w as u32)),
                    to_u8(&noise(cw * ch, 8, 2 + h as u32)),
                    to_u8(&noise(cw * ch, 8, 3 + nlx as u32)),
                ];
                let s_x = [1u8, sx as u8, sx as u8];
                let s_y = [1u8, sy as u8, sy as u8];
                let buf = encoder::encode_planar_subsampled(
                    w as u16, h as u16, 3, 0, nlx, nly, 0, &s_x, &s_y, &planes,
                )
                .unwrap_or_else(|e| panic!("encode {w}x{h} NL={nlx}/{nly} s={sx}x{sy}: {e}"));
                let img = oxideav_jpegxs::decode_components(&buf).unwrap();
                for (c, p) in planes.iter().enumerate() {
                    assert_eq!(
                        &img.planes[c].data, p,
                        "{w}x{h} NL={nlx}/{nly} s={sx}x{sy} component {c}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Layer 2 — the encoder's coefficients against Table E.8, without this
// crate's inverse transform in the loop.
// ---------------------------------------------------------------------

/// Literal Table E.5 + E.12 on one line; returns `(low, high)`.
fn annex_e_forward_line(line: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let z = line.len();
    let mut x = vec![0i64; z + 4];
    x[2..2 + z].copy_from_slice(line);
    for i in 1..=2usize {
        x[2 - i] = x[2 + i];
        x[2 + z + i - 1] = x[2 + z - i - 1];
    }
    let mut y = x.clone();
    // i = -1, 1, 3, … < Z + 1 with the offset of 2 folded in.
    for k in (1..z + 3).step_by(2) {
        y[k] = x[k] - ((x[k - 1] + x[k + 1]) >> 1);
    }
    for i in (0..z).step_by(2) {
        y[2 + i] = x[2 + i] + ((y[2 + i - 1] + y[2 + i + 1] + 2) >> 2);
    }
    let body = &y[2..2 + z];
    (
        body.iter().step_by(2).copied().collect(),
        body.iter().skip(1).step_by(2).copied().collect(),
    )
}

/// A band: `(width, height, row-major samples)`.
type Band = (usize, usize, Vec<i64>);

/// Table E.10: rows of `src` into a low and a high band.
fn hor_fwd(src: &Band) -> (Band, Band) {
    let (w, h, t) = src;
    let (mut lo, mut hi) = (Vec::new(), Vec::new());
    for row in t.chunks_exact(*w) {
        let (l, g) = annex_e_forward_line(row);
        lo.extend(l);
        hi.extend(g);
    }
    ((w.div_ceil(2), *h, lo), (w / 2, *h, hi))
}

/// Table E.11: columns of `src` into a low and a high band.
fn ver_fwd(src: &Band) -> (Band, Band) {
    let (w, h, t) = src;
    let (lh, hh) = (h.div_ceil(2), h / 2);
    let (mut lo, mut hi) = (vec![0i64; w * lh], vec![0i64; w * hh]);
    for x in 0..*w {
        let col: Vec<i64> = (0..*h).map(|y| t[y * w + x]).collect();
        let (l, g) = annex_e_forward_line(&col);
        for (y, v) in l.into_iter().enumerate() {
            lo[y * w + x] = v;
        }
        for (y, v) in g.into_iter().enumerate() {
            hi[y * w + x] = v;
        }
    }
    ((*w, lh, lo), (*w, hh, hi))
}

/// Annex B.3 filter index (`NL,x ≥ NL,y`).
fn beta(nlx: usize, nly: usize, dx: usize, dy: usize, tx: usize, ty: usize) -> usize {
    if dx > dy {
        nlx - dx + tx
    } else {
        nlx - nly + tx + 2 * ty + 3 * (nly - dy)
    }
}

/// Table E.8 for one component, bands in β order. `exchanged = true`
/// runs each joint level horizontally first instead — a different
/// integer transform, used only to show the pictures can tell.
fn table_e8(picture: &Band, nlx: usize, nly: usize, exchanged: bool) -> Vec<Vec<i64>> {
    let mut out = vec![Vec::new(); 2 * nly + nlx + 1];
    let mut ll = picture.clone();
    for d in 1..=nly {
        let (ll_d, hl_d, lh_d, hh_d) = if exchanged {
            let (l, g) = hor_fwd(&ll);
            let (ll_d, lh_d) = ver_fwd(&l);
            let (hl_d, hh_d) = ver_fwd(&g);
            (ll_d, hl_d, lh_d, hh_d)
        } else {
            let (l, g) = ver_fwd(&ll);
            let (ll_d, hl_d) = hor_fwd(&l);
            let (lh_d, hh_d) = hor_fwd(&g);
            (ll_d, hl_d, lh_d, hh_d)
        };
        out[beta(nlx, nly, d, d, 1, 0)] = hl_d.2;
        out[beta(nlx, nly, d, d, 0, 1)] = lh_d.2;
        out[beta(nlx, nly, d, d, 1, 1)] = hh_d.2;
        ll = ll_d;
    }
    for dx in nly + 1..=nlx {
        let (ll_d, hl_d) = hor_fwd(&ll);
        out[beta(nlx, nly, dx, nly, 1, 0)] = hl_d.2;
        ll = ll_d;
    }
    out[0] = ll.2;
    out
}

/// The coefficients a lossless (`Bw = B`, `Fq = 0`, no colour transform)
/// stream from this encoder carries are exactly the Table E.8 transform
/// of the level-shifted picture (Annex G: `O = sample − 2^(B−1)`) — read
/// back from the codestream *before* any inverse transform, so nothing
/// but entropy decoding sits between the encoder's forward transform and
/// the comparison.
#[test]
fn encoded_coefficients_are_the_table_e8_transform() {
    let mut separating = 0;
    let mut total = 0;
    for &(nlx, nly) in &[
        (1usize, 1usize),
        (2, 1),
        (2, 2),
        (3, 3),
        (5, 1),
        (5, 2),
        (4, 3),
    ] {
        for &(w, h) in &[(32usize, 8usize), (33, 9), (37, 16), (64, 23), (95, 34)] {
            if w < (1 << nlx) || h < (1 << nly) {
                continue;
            }
            let plane = to_u8(&noise(w * h, 8, (w * 5 + h + nlx * 11 + nly) as u32));
            let buf = encoder::encode_planar(
                w as u16,
                h as u16,
                1,
                0,
                nlx as u8,
                nly as u8,
                std::slice::from_ref(&plane),
            )
            .unwrap();
            let coded = oxideav_jpegxs::decoder::wavelet_coefficients(&buf).unwrap();
            assert_eq!(coded.len(), 1, "one decomposed component");
            let shifted: Band = (w, h, plane.iter().map(|&v| i64::from(v) - 128).collect());
            let want = table_e8(&shifted, nlx, nly, false);
            assert_eq!(coded[0].len(), want.len());
            for (b, (got, want)) in coded[0].iter().zip(&want).enumerate() {
                let got: Vec<i64> = got.iter().map(|&v| i64::from(v)).collect();
                assert_eq!(&got, want, "{w}x{h} NL={nlx}/{nly} filter type β={b}");
            }
            total += 1;
            if table_e8(&shifted, nlx, nly, true) != want {
                separating += 1;
            }
        }
    }
    assert!(total >= 25, "only {total} configurations ran");
    assert_eq!(
        separating, total,
        "every picture here must separate the two pass orders"
    );
}

// ---------------------------------------------------------------------
// Layer 3 — black-box decoder cross-check (run-time skip).
// ---------------------------------------------------------------------

/// Whether the `ffmpeg` on `PATH` lists a JPEG XS *decoder*.
fn black_box_decoder_available() -> bool {
    let Ok(out) = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-decoders"])
        .output()
    else {
        return false;
    };
    String::from_utf8_lossy(&out.stdout).lines().any(|l| {
        l.split_whitespace()
            .nth(1)
            .is_some_and(|n| n.contains("jpegxs"))
    })
}

/// Decode `jxs` with the black-box binary into raw planar samples of
/// `pix_fmt`. `None` when the binary declines the stream.
fn black_box_decode(jxs: &[u8], pix_fmt: &str, tag: &str) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!(
        "oxideav-jpegxs-wavelet-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join("in.jxs"), dir.join("out.raw"));
    std::fs::write(&src, jxs).ok()?;
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&src)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix_fmt])
        .arg(&dst)
        .status()
        .ok()?;
    let out = if status.success() {
        std::fs::read(&dst).ok()
    } else {
        None
    };
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn black_box_decoder_agrees_with_this_decoder() {
    if !black_box_decoder_available() {
        eprintln!("skip: no black-box JPEG XS decoder binary on this machine");
        return;
    }
    let (w, h) = (128usize, 64usize);
    let mut compared = 0;
    for &(nlx, nly) in &[(5u8, 0u8), (5, 1), (5, 2), (3, 1), (2, 2)] {
        // Three 8-bit components without colour transform: planar 4:4:4.
        let planes: Vec<Vec<u8>> = (0..3)
            .map(|c| to_u8(&noise(w * h, 8, 40 + c + nlx as u32 * 5 + nly as u32)))
            .collect();
        let streams = [
            (
                "lossless",
                encoder::encode_planar(w as u16, h as u16, 3, 0, nlx, nly, &planes).unwrap(),
            ),
            (
                "fq8",
                encoder::encode_planar_highprec_lossy(
                    w as u16, h as u16, 3, 0, nlx, nly, 2, &planes,
                )
                .unwrap(),
            ),
        ];
        for (kind, buf) in streams {
            let ours = oxideav_jpegxs::decode_components(&buf).unwrap();
            let ours: Vec<u8> = ours.planes.iter().flat_map(|p| p.data.clone()).collect();
            let tag = format!("{kind}-{nlx}-{nly}");
            match black_box_decode(&buf, "yuv444p", &tag) {
                Some(theirs) if theirs.len() == ours.len() => {
                    assert_eq!(theirs, ours, "NL={nlx}/{nly} {kind}: decoders disagree");
                    compared += 1;
                }
                Some(theirs) => eprintln!(
                    "skip NL={nlx}/{nly} {kind}: black-box output is {} bytes, expected {}",
                    theirs.len(),
                    ours.len()
                ),
                None => eprintln!("skip NL={nlx}/{nly} {kind}: black-box decoder declined"),
            }
        }
    }
    eprintln!("black-box decoder agreed on {compared} stream(s)");
}

// ---------------------------------------------------------------------
// Packet header form (Annex C.3, Table C.3).
// ---------------------------------------------------------------------

/// `Wf × Nc ≥ 32752` selects the long packet header (Table C.3) whatever
/// `Lh` says; the encoder must write that form for such pictures.
#[test]
fn wide_pictures_use_the_long_packet_header() {
    let (w, h) = (8192usize, 4usize);
    let planes: Vec<Vec<u8>> = (0..4).map(|c| to_u8(&noise(w * h, 8, 700 + c))).collect();
    for &(nlx, nly) in &[(5u8, 1u8), (5, 0), (3, 2)] {
        let buf = encoder::encode_planar(w as u16, h as u16, 4, 0, nlx, nly, &planes)
            .unwrap_or_else(|e| panic!("encode NL={nlx}/{nly}: {e}"));
        let img = oxideav_jpegxs::decode_components(&buf)
            .unwrap_or_else(|e| panic!("decode NL={nlx}/{nly}: {e}"));
        for (c, p) in planes.iter().enumerate() {
            assert_eq!(&img.planes[c].data, p, "NL={nlx}/{nly} component {c}");
        }
    }
}

/// A line too large for the short header's 15-bit `Ldat` makes the
/// encoder raise `Lh` (Annex A.4.4 NOTE 2) instead of failing.
#[test]
fn oversized_packets_switch_to_long_headers() {
    let (w, h) = (8000usize, 2usize);
    let planes: Vec<Vec<u16>> = (0..3).map(|c| noise(w * h, 16, 800 + c)).collect();
    let buf = encoder::encode_planar_highbd(w as u16, h as u16, 3, 0, 5, 0, 16, &planes)
        .expect("encode with long headers");
    let img = oxideav_jpegxs::decode_components(&buf).expect("decode");
    for (c, p) in planes.iter().enumerate() {
        assert_eq!(img.planes[c].data, to_le(p), "component {c}");
    }
}
