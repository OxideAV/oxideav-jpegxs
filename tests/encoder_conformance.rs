//! Encoder conformance pinning — deterministic encoder invocations
//! whose emitted codestreams are pinned by SHA-256.
//!
//! The decoder side of this crate is pinned against the official
//! ISO/IEC 21122-4:2020 vector set (see `tests/conformance.rs`). The
//! encoder has no published vector set — 21122-4 §4 defines *decoder*
//! conformance, and encoder conformance is exactly "every emitted
//! codestream decodes correctly on a conforming decoder" — so this
//! harness pins the next best thing, in three layers per stream:
//!
//! 1. **Byte stability**: the SHA-256 of the emitted codestream is
//!    pinned. Any wire-format change (marker layout, entropy coding,
//!    rate allocation, signalling) trips the hash and must be a
//!    deliberate, reviewed change of this table.
//! 2. **Self-decode**: the stream decodes through the crate's own
//!    conformance-gated decoder (the same gates that hold 65/65
//!    published vectors bit-exact), reconstructing the input planes
//!    bit-exactly for every lossless case.
//! 3. **Declaration truth**: streams that claim a profile / level /
//!    sublevel / CBR size pass `verify_declarations` — the decoder-side
//!    checks of ISO/IEC 21122-2 Annex A + 21122-1 Table 11.
//!
//! The pinned matrix spans the encoder's feature axes: colour
//! transforms (`Cpih ∈ {0, 1, 3}`), chroma sub-sampling, high bit
//! depth, NLT non-linearity, run mode 1, multi-slice + rate-allocated
//! CBR, and all eight 21122-2:2019 profile targets.

use oxideav_jpegxs::encoder;
use oxideav_jpegxs::profile::Profile;
use oxideav_jpegxs::signalling;

// ---------------------------------------------------------------------
// SHA-256 (FIPS 180-4) — implemented locally so the crate keeps its
// zero-dependency policy. Test-only code.
// ---------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // Padding: 0x80, zeros, 64-bit big-endian bit length.
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (t, wt) in w.iter_mut().take(16).enumerate() {
            *wt = u32::from_be_bytes([
                block[4 * t],
                block[4 * t + 1],
                block[4 * t + 2],
                block[4 * t + 3],
            ]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn sha256_kat() {
    // FIPS 180-4 known-answer tests pin the local implementation.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

// ---------------------------------------------------------------------
// Deterministic input planes.
// ---------------------------------------------------------------------

fn plane8(w: usize, h: usize, seed: u32) -> Vec<u8> {
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

fn pack16(p: &[u16], bd: u8) -> Vec<u8> {
    if bd == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// Decode `buf` and assert plane `i` equals `expect[i]` byte-for-byte.
fn assert_decodes_to(name: &str, buf: &[u8], expect: &[Vec<u8>]) {
    let img = oxideav_jpegxs::decode_jpeg_xs(buf)
        .unwrap_or_else(|e| panic!("{name}: self-decode failed: {e}"));
    assert_eq!(img.planes.len(), expect.len(), "{name}: plane count");
    for (i, p) in img.planes.iter().enumerate() {
        assert_eq!(p.data, expect[i], "{name}: plane {i} bit-exact");
    }
}

// ---------------------------------------------------------------------
// The pinned matrix.
// ---------------------------------------------------------------------

const W: usize = 64;
const H: usize = 64;

/// Pinned SHA-256 per matrix row. Regenerating this table is a
/// deliberate act: run with `OXIDEAV_JXS_PRINT_HASHES=1` to print the
/// computed hashes, review the wire-format change that moved them, and
/// update the constants.
const PINNED: &[(&str, &str)] = &[
    (
        "rgb_rct_lossless_nl51",
        "2e9113193a1e229c86c15b2f9d8ba1a2a1d94b1393a0421d5cc6e3739969e245",
    ),
    (
        "yuv420_lossless_nl31",
        "745d28c8329569214c679115325edadfcf93156653d430050f56a8b8d4363f0d",
    ),
    (
        "star_tetrix_lossless_nl22",
        "f6ed3025b62befc2fc9e08d7ace7be03fb8377abdfd88d367a7f1d2bca1a5271",
    ),
    (
        "nlt_quadratic_lossless_nl21",
        "81fa204281e44b044befb8cec5f1e706685c03aa58f21a229c6517146d608ac4",
    ),
    (
        "highbd12_lossless_nl21",
        "b7821ae6afa15e73e3b10bff5605655bb25c125b99664cebaa43c8adcc24b751",
    ),
    (
        "run_mode1_lossless_nl33",
        "2bc2989076bf076ef4237f620e303372c6f6500e16d9f9115fab52529429cbeb",
    ),
    (
        "cbr_exact_target",
        "87cfebb1174c0c1fbd61c11a7f7750413ca8c1cd3f484e4998cbb7f3aacf621a",
    ),
    (
        "profile_light422_10",
        "9ee447c537b65b3ea2a3c5729e58dda26abcfda4c349ee3c9ba95a0b95e3fe6b",
    ),
    (
        "profile_light_subline422_10",
        "940be105e6c570a85d113d95fd6a27b03f3991f46c61029566b010732755ae4b",
    ),
    (
        "profile_light444_12",
        "90c1a5c221c5e2ebe023b0c12416d61bc64df837776af8ce39d02a37a97bc918",
    ),
    (
        "profile_main422_10",
        "c8cba2c34ff2a427a4866a926e9119fd02e436bc01be632017731613a1cd85f0",
    ),
    (
        "profile_main444_12",
        "9e29e2c7f4b46099f6e2a48b07af3da24063e306ee14e8f319e03f30f3e7ecac",
    ),
    (
        "profile_main4444_12",
        "0926c2c912104a95e001c6791bd87a310831d963f9d0cba5cd0c2637f78e38e4",
    ),
    (
        "profile_high444_12",
        "493178b6aae2259328d62b9a66c182d93421075dfc42722f5313fcf9a5fda30d",
    ),
    (
        "profile_high4444_12",
        "4840fe9964afa18dd5db6c9f4e9ad855fed4b78f839cf644e9db7937597fb30b",
    ),
];

fn pinned_hash(name: &str) -> &'static str {
    PINNED
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no pinned hash for {name}"))
        .1
}

fn check_pin(name: &str, buf: &[u8]) {
    let got = sha256_hex(buf);
    if std::env::var_os("OXIDEAV_JXS_PRINT_HASHES").is_some() {
        println!("    (\n        \"{name}\",\n        \"{got}\",\n    ),");
        return;
    }
    assert_eq!(
        got,
        pinned_hash(name),
        "{name}: emitted codestream hash moved — wire-format change must be deliberate"
    );
}

#[test]
fn pinned_encoder_streams() {
    // --- Cpih = 1 RCT, 4:4:4 8-bit, NL = 5/1, lossless. ---
    let rgb = vec![plane8(W, H, 1), plane8(W, H, 2), plane8(W, H, 3)];
    let buf = encoder::encode_planar(W as u16, H as u16, 3, 1, 5, 1, &rgb).unwrap();
    check_pin("rgb_rct_lossless_nl51", &buf);
    assert_decodes_to("rgb_rct_lossless_nl51", &buf, &rgb);

    // --- 4:2:0 chroma, Cpih = 0, NL = 3/1, lossless. ---
    let yuv = vec![
        plane8(W, H, 4),
        plane8(W / 2, H / 2, 5),
        plane8(W / 2, H / 2, 6),
    ];
    let buf = encoder::encode_planar_subsampled(
        W as u16,
        H as u16,
        3,
        0,
        3,
        1,
        0,
        &[1, 2, 2],
        &[1, 2, 2],
        &yuv,
    )
    .unwrap();
    check_pin("yuv420_lossless_nl31", &buf);
    assert_decodes_to("yuv420_lossless_nl31", &buf, &yuv);

    // --- Star-Tetrix CFA (Cpih = 3), 4 components, NL = 2/2. ---
    let cfa = vec![
        plane8(W, H, 7),
        plane8(W, H, 8),
        plane8(W, H, 9),
        plane8(W, H, 10),
    ];
    let buf =
        encoder::encode_planar_star_tetrix(W as u16, H as u16, 2, 2, 0, 1, 1, 0, 0, &cfa).unwrap();
    check_pin("star_tetrix_lossless_nl22", &buf);
    assert_decodes_to("star_tetrix_lossless_nl22", &buf, &cfa);

    // --- NLT quadratic (Tnlt = 1), luma, NL = 2/1, q = 0. The
    // quadratic forward map is a rounded square root, so q = 0 is
    // lossless *within the quadratic approximation* (a few LSB), not
    // bit-exact — pin the hash and a tight reconstruction bound.
    let luma = vec![plane8(W, H, 11)];
    let buf =
        encoder::encode_planar_nlt_quadratic(W as u16, H as u16, 1, 0, 2, 1, 0, 0, &luma).unwrap();
    check_pin("nlt_quadratic_lossless_nl21", &buf);
    let img = oxideav_jpegxs::decode_jpeg_xs(&buf).expect("nlt quadratic decodes");
    assert_eq!(img.planes.len(), 1);
    let max_err = img.planes[0]
        .data
        .iter()
        .zip(luma[0].iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap();
    assert!(
        max_err <= 2,
        "nlt quadratic q=0 reconstruction error {max_err} exceeds the approximation bound"
    );

    // --- High bit depth (B[i] = 12), 4:4:4, NL = 2/1, lossless. ---
    let hb: Vec<Vec<u16>> = (0..3).map(|i| plane16(W, H, 12, i + 20)).collect();
    let buf = encoder::encode_planar_highbd(W as u16, H as u16, 3, 1, 2, 1, 12, &hb).unwrap();
    check_pin("highbd12_lossless_nl21", &buf);
    let hb_bytes: Vec<Vec<u8>> = hb.iter().map(|p| pack16(p, 12)).collect();
    assert_decodes_to("highbd12_lossless_nl21", &buf, &hb_bytes);

    // --- Run mode 1 (Rm = 1), RGB RCT, NL = 3/3, lossless. ---
    let buf = encoder::encode_planar_run_mode1(W as u16, H as u16, 3, 1, 3, 3, 0, &rgb).unwrap();
    check_pin("run_mode1_lossless_nl33", &buf);
    assert_decodes_to("run_mode1_lossless_nl33", &buf, &rgb);

    // --- Exact-size CBR: rate-allocated + COM-padded + Lcod. ---
    let target = 9000usize;
    let (buf, _q) =
        encoder::encode_planar_cbr_target_bytes(W as u16, H as u16, 3, 1, 5, 1, 8, target, &rgb)
            .unwrap();
    assert_eq!(buf.len(), target);
    signalling::verify_declarations(&buf).unwrap();
    check_pin("cbr_exact_target", &buf);
    oxideav_jpegxs::decode_jpeg_xs(&buf).unwrap();

    // --- The eight 21122-2:2019 profile targets (CBR-signed). ---
    struct P {
        name: &'static str,
        profile: Profile,
        nc: u8,
        cpih: u8,
        nlx: u8,
        nly: u8,
        bd: u8,
        qpih: u8,
        sx: &'static [u8],
        sy: &'static [u8],
    }
    let cases = [
        P {
            name: "profile_light422_10",
            profile: Profile::Light422_10,
            nc: 3,
            cpih: 0,
            nlx: 2,
            nly: 1,
            bd: 10,
            qpih: 0,
            sx: &[1, 2, 2],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_light_subline422_10",
            profile: Profile::LightSubline422_10,
            nc: 3,
            cpih: 0,
            nlx: 3,
            nly: 0,
            bd: 8,
            qpih: 1,
            sx: &[1, 2, 2],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_light444_12",
            profile: Profile::Light444_12,
            nc: 3,
            cpih: 1,
            nlx: 2,
            nly: 1,
            bd: 12,
            qpih: 0,
            sx: &[1, 1, 1],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_main422_10",
            profile: Profile::Main422_10,
            nc: 3,
            cpih: 0,
            nlx: 2,
            nly: 1,
            bd: 10,
            qpih: 1,
            sx: &[1, 2, 2],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_main444_12",
            profile: Profile::Main444_12,
            nc: 3,
            cpih: 1,
            nlx: 2,
            nly: 1,
            bd: 12,
            qpih: 0,
            sx: &[1, 1, 1],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_main4444_12",
            profile: Profile::Main4444_12,
            nc: 4,
            cpih: 0,
            nlx: 2,
            nly: 1,
            bd: 12,
            qpih: 0,
            sx: &[1, 1, 1, 1],
            sy: &[1, 1, 1, 1],
        },
        P {
            name: "profile_high444_12",
            profile: Profile::High444_12,
            nc: 3,
            cpih: 1,
            nlx: 2,
            nly: 2,
            bd: 12,
            qpih: 1,
            sx: &[1, 1, 1],
            sy: &[1, 1, 1],
        },
        P {
            name: "profile_high4444_12",
            profile: Profile::High4444_12,
            nc: 4,
            cpih: 0,
            nlx: 2,
            nly: 2,
            bd: 12,
            qpih: 1,
            sx: &[1, 1, 1, 1],
            sy: &[1, 1, 1, 1],
        },
    ];
    for case in &cases {
        let planes: Vec<Vec<u16>> = (0..case.nc as usize)
            .map(|i| {
                plane16(
                    W.div_ceil(case.sx[i] as usize),
                    H.div_ceil(case.sy[i] as usize),
                    case.bd,
                    i as u32 + 1,
                )
            })
            .collect();
        let (buf, _level, _sublevel) = encoder::encode_planar_for_profile(
            case.profile,
            W as u16,
            H as u16,
            case.nc,
            case.cpih,
            case.nlx,
            case.nly,
            case.bd,
            case.qpih,
            0,
            case.sx,
            case.sy,
            true,
            &planes,
        )
        .unwrap_or_else(|e| panic!("{}: encode failed: {e}", case.name));
        signalling::verify_declarations(&buf)
            .unwrap_or_else(|e| panic!("{}: verify failed: {e}", case.name));
        check_pin(case.name, &buf);
        let expect: Vec<Vec<u8>> = planes.iter().map(|p| pack16(p, case.bd)).collect();
        assert_decodes_to(case.name, &buf, &expect);
    }
}
