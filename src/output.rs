//! Output scaling, DC level shift, and clipping — ISO/IEC 21122-1:2022
//! Annex G.
//!
//! This module owns the back-end of the decoder pipeline: it converts
//! the post-colour-transform sample plane Ω[i, x, y] (int32, with the
//! wavelet-domain DC bias still in place) into bytes in the
//! component's nominal range `[0, 2^B[i] - 1]`. Three reconstruction
//! paths are normative, selected by the presence and `Tnlt` of an NLT
//! marker (Table G.1):
//!
//! * **No NLT** → linear output scaling (Annex G.3, Table G.2).
//! * **NLT with Tnlt = 1** → quadratic output scaling
//!   (Annex G.4, Table G.3). Inverse gamma `v ← v²`.
//! * **NLT with Tnlt = 2** → extended output scaling
//!   (Annex G.5, Table G.4). Three-segment inverse gamma with
//!   thresholds `T1 < T2`.
//!
//! The NLT body is parsed by [`parse_nlt`] into [`NltParams`]; the
//! decoder hands the variant to [`apply_output_scaling`] which
//! dispatches to the right kernel.
//!
//! Note on bit precision: the linear path requires `ζ = Bw - B[i] >= 0`
//! (i.e. wavelet precision at least as wide as the component
//! precision). Per Table A.8 this always holds — `Bw ∈ {8, 18, 20}`
//! and `B[i] ∈ {8, ..., 16}`.

use oxideav_core::{Error, Result};

/// Decoded NLT marker segment body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NltParams {
    /// Tnlt = 1 — quadratic non-linearity.
    /// `dco` is the DC offset (signed 16-bit, two's-complement encoded
    /// as σ:α with σ being the sign bit and α the 15 magnitude bits).
    Quadratic { dco: i32 },
    /// Tnlt = 2 — extended non-linearity.
    /// `t1` is the upper threshold of the black region, `t2` the upper
    /// threshold of the linear region (both in `1..=2^Bw - 1`),
    /// `e` is the linear-slope exponent (1..=4).
    Extended { t1: u32, t2: u32, e: u8 },
}

/// Parse the NLT marker body. `body` is the bytes following the `Lnlt`
/// length field (i.e. `Lnlt - 2` bytes total).
///
/// Body layout per Table A.16:
/// * `Tnlt: u8` — type selector (1 = quadratic, 2 = extended).
/// * If `Tnlt == 1`: `σ:1 | α:15` packed → `DCO = α - σ * 2^15`. (Lnlt = 5.)
/// * If `Tnlt == 2`: `T1: u32`, `T2: u32`, `E: u8`. (Lnlt = 12.)
pub fn parse_nlt(body: &[u8]) -> Result<NltParams> {
    if body.is_empty() {
        return Err(Error::invalid("jpegxs NLT body must contain at least Tnlt"));
    }
    let tnlt = body[0];
    match tnlt {
        1 => {
            // 1 byte Tnlt + 2 bytes σ:α → 3 bytes total → Lnlt = 5.
            if body.len() != 3 {
                return Err(Error::invalid(format!(
                    "jpegxs NLT Tnlt=1 body must be 3 bytes (got {})",
                    body.len()
                )));
            }
            let packed = u16::from_be_bytes([body[1], body[2]]);
            let sigma = (packed >> 15) & 0x01;
            let alpha = (packed & 0x7fff) as i32;
            let dco = alpha - (sigma as i32) * (1 << 15);
            Ok(NltParams::Quadratic { dco })
        }
        2 => {
            // 1 byte Tnlt + 4 bytes T1 + 4 bytes T2 + 1 byte E → 10 bytes
            // total → Lnlt = 12.
            if body.len() != 10 {
                return Err(Error::invalid(format!(
                    "jpegxs NLT Tnlt=2 body must be 10 bytes (got {})",
                    body.len()
                )));
            }
            let t1 = u32::from_be_bytes([body[1], body[2], body[3], body[4]]);
            let t2 = u32::from_be_bytes([body[5], body[6], body[7], body[8]]);
            let e = body[9];
            if t1 == 0 || t2 == 0 || t2 <= t1 {
                return Err(Error::invalid(format!(
                    "jpegxs NLT Tnlt=2 thresholds T1={t1} T2={t2}: require 0 < T1 < T2"
                )));
            }
            if !(1..=4).contains(&e) {
                return Err(Error::invalid(format!(
                    "jpegxs NLT Tnlt=2 exponent E={e} out of range 1..=4"
                )));
            }
            Ok(NltParams::Extended { t1, t2, e })
        }
        _ => Err(Error::Unsupported(format!(
            "jpegxs NLT Tnlt={tnlt}: reserved or unknown"
        ))),
    }
}

/// Apply Annex G output scaling to one component's sample plane in
/// place. `omega` is the int32 plane (length `wc * hc`), `bw` is the
/// picture-header `Bw`, `bc` is the per-component `B[i]`, and `nlt`
/// is the parsed NLT body (`None` → linear path).
///
/// Returns a `Vec<u8>` for `B[i] == 8` paths; for higher bit depths the
/// caller would need a `Vec<u16>` packing helper. Round-5 wires only the
/// 8-bit output side (the only one exercised by the round-5 fixtures
/// plus round-4 regression tests) but the kernels themselves run on
/// arbitrary `B[i]`.
pub fn apply_output_scaling(
    omega: &[i32],
    bw: u8,
    bc: u8,
    nlt: Option<NltParams>,
) -> Result<Vec<u8>> {
    if bc != 8 {
        return Err(Error::Unsupported(format!(
            "jpegxs output: B[i]={bc} non-8-bit packing is round-6 (need u16/u32 plane format)"
        )));
    }
    if !(8..=20).contains(&bw) {
        return Err(Error::invalid(format!(
            "jpegxs output: Bw={bw} out of supported range 8..=20"
        )));
    }
    let mut out = vec![0u8; omega.len()];
    match nlt {
        None => linear_path(omega, bw, bc, &mut out),
        Some(NltParams::Quadratic { dco }) => quadratic_path(omega, bw, bc, dco, &mut out),
        Some(NltParams::Extended { t1, t2, e }) => {
            extended_path(omega, bw, bc, t1, t2, e, &mut out)
        }
    }
    Ok(out)
}

/// Annex G.3, Table G.2 — linear output scaling and clipping.
fn linear_path(omega: &[i32], bw: u8, bc: u8, out: &mut [u8]) {
    let zeta = (bw as i32) - (bc as i32);
    let m = (1i32 << bc) - 1;
    let dc_bias = 1i32 << (bw - 1);
    let zeta_u = zeta.max(0) as u32;
    let half = if zeta_u == 0 { 0 } else { 1i32 << (zeta_u - 1) };
    for (i, &v) in omega.iter().enumerate() {
        let v = v.saturating_add(dc_bias);
        let v = if zeta_u == 0 {
            v
        } else {
            v.saturating_add(half) >> zeta_u
        };
        let v = v.clamp(0, m);
        out[i] = v as u8;
    }
}

/// Annex G.4, Table G.3 — quadratic output scaling and clipping.
fn quadratic_path(omega: &[i32], bw: u8, bc: u8, dco: i32, out: &mut [u8]) {
    let zeta = 2 * (bw as i32) - (bc as i32);
    let m = (1i32 << bc) - 1;
    let dc_bias = 1i32 << (bw - 1);
    let bw_max = (1i64 << bw) - 1;
    let zeta_u = zeta.max(0) as u32;
    // The intermediate v² needs i64 to avoid overflow at Bw=18 / 20.
    let half: i64 = if zeta_u == 0 { 0 } else { 1i64 << (zeta_u - 1) };
    for (i, &v) in omega.iter().enumerate() {
        let v = v.saturating_add(dc_bias);
        let v = (v as i64).clamp(0, bw_max);
        let v = v * v;
        let v = if zeta_u == 0 { v } else { (v + half) >> zeta_u };
        let v = (v + dco as i64).clamp(0, m as i64);
        out[i] = v as u8;
    }
}

/// Annex G.5, Table G.4 — extended (three-segment) output scaling.
fn extended_path(omega: &[i32], bw: u8, bc: u8, t1: u32, t2: u32, e: u8, out: &mut [u8]) {
    let bw_i = bw as i64;
    let m = (1i64 << bc) - 1;
    let dc_bias = 1i64 << (bw - 1);
    let two_pow_bw_minus_one = (1i64 << bw) - 1;
    let t1 = t1 as i64;
    let t2 = t2 as i64;
    let e_i = e as i64;
    // Pre-compute spec constants from Table G.4.
    let b2 = t1 * t1;
    let shift_a13 = 2 * bw_i - 2 - 2 * e_i;
    let a1 = b2 + (t1 << (bw_i - e_i)) + (1i64 << shift_a13);
    let b1 = t1 + (1i64 << (bw_i - e_i - 1));
    let a3 = b2 + (t2 << (bw_i - e_i)) - (1i64 << shift_a13);
    let b3 = t2 - (1i64 << (bw_i - e_i - 1));
    let zeta = 2 * bw_i - (bc as i64);
    let zeta_u = zeta.max(0) as u32;
    let half: i64 = if zeta_u == 0 { 0 } else { 1i64 << (zeta_u - 1) };

    for (i, &v) in omega.iter().enumerate() {
        let mut v = (v as i64) + dc_bias;
        if v < t1 {
            // Black region: v = B1 - v; clamp; v = A1 - v*v.
            v = b1 - v;
            v = v.clamp(0, two_pow_bw_minus_one);
            v = a1 - v * v;
        } else if v < t2 {
            // Linear region: v = (v << ε) + B2.  ε = bw - e.
            v = (v << (bw_i - e_i)) + b2;
        } else {
            // Regular region: v = v - B3; clamp; v = A3 + v*v.
            v -= b3;
            v = v.clamp(0, two_pow_bw_minus_one);
            v = a3 + v * v;
        }
        let v = if zeta_u == 0 { v } else { (v + half) >> zeta_u };
        let v = v.clamp(0, m);
        out[i] = v as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nlt_quadratic_round_trips() {
        // Tnlt=1, σ=1, α=300 → DCO = 300 - 32768 = -32468.
        let mut body = vec![1u8];
        let packed: u16 = (1 << 15) | 300;
        body.extend_from_slice(&packed.to_be_bytes());
        let p = parse_nlt(&body).unwrap();
        assert_eq!(p, NltParams::Quadratic { dco: -32468 });

        // σ=0, α=12345 → DCO = 12345.
        let mut body = vec![1u8];
        body.extend_from_slice(&12345u16.to_be_bytes());
        let p = parse_nlt(&body).unwrap();
        assert_eq!(p, NltParams::Quadratic { dco: 12345 });
    }

    #[test]
    fn parse_nlt_extended_round_trips() {
        let mut body = vec![2u8];
        body.extend_from_slice(&100u32.to_be_bytes());
        body.extend_from_slice(&200u32.to_be_bytes());
        body.push(3);
        let p = parse_nlt(&body).unwrap();
        assert_eq!(
            p,
            NltParams::Extended {
                t1: 100,
                t2: 200,
                e: 3
            }
        );
    }

    #[test]
    fn parse_nlt_rejects_bad_thresholds() {
        let mut body = vec![2u8];
        body.extend_from_slice(&200u32.to_be_bytes());
        body.extend_from_slice(&100u32.to_be_bytes());
        body.push(3);
        assert!(parse_nlt(&body).is_err());

        let mut body = vec![2u8];
        body.extend_from_slice(&0u32.to_be_bytes());
        body.extend_from_slice(&100u32.to_be_bytes());
        body.push(3);
        assert!(parse_nlt(&body).is_err());
    }

    #[test]
    fn parse_nlt_rejects_bad_exponent() {
        let mut body = vec![2u8];
        body.extend_from_slice(&100u32.to_be_bytes());
        body.extend_from_slice(&200u32.to_be_bytes());
        body.push(0);
        assert!(parse_nlt(&body).is_err());
        let mut body = vec![2u8];
        body.extend_from_slice(&100u32.to_be_bytes());
        body.extend_from_slice(&200u32.to_be_bytes());
        body.push(5);
        assert!(parse_nlt(&body).is_err());
    }

    #[test]
    fn parse_nlt_rejects_unknown_tnlt() {
        let body = vec![3u8, 0, 0];
        assert!(parse_nlt(&body).is_err());
    }

    #[test]
    fn linear_path_zero_yields_mid_grey_8bit() {
        // All-zero coefficients with Bw=20 B=8 → DC bias 524288, >> 12
        // = 128.
        let omega = vec![0i32; 4];
        let out = apply_output_scaling(&omega, 20, 8, None).unwrap();
        assert_eq!(out, vec![128u8; 4]);
    }

    #[test]
    fn linear_path_lossless_path_8bit() {
        // Lossless Bw=B=8, ζ=0. ω=1 → 1 + 128 = 129.
        let omega = vec![1i32, 1, 1, 1];
        let out = apply_output_scaling(&omega, 8, 8, None).unwrap();
        assert_eq!(out, vec![129u8; 4]);
    }

    #[test]
    fn linear_path_clips_to_bound() {
        // Bw=8, B=8: max representable is (255 - 128) = 127.
        // ω = 1000 → (1000 + 128) clamped to 255.
        let omega = vec![1000i32, -1000];
        let out = apply_output_scaling(&omega, 8, 8, None).unwrap();
        assert_eq!(out, vec![255u8, 0u8]);
    }

    #[test]
    fn quadratic_path_runs() {
        // Smoke test: ω = 0 with Bw=18, B=8, DCO=0.
        // ζ = 2*18 - 8 = 28, m=255, dc_bias = 2^17 = 131072.
        // v = 131072, v² = 17179869184. >> 28 = 64. Plus DCO=0 → 64.
        let omega = vec![0i32; 4];
        let out =
            apply_output_scaling(&omega, 18, 8, Some(NltParams::Quadratic { dco: 0 })).unwrap();
        assert_eq!(out, vec![64u8; 4]);
    }

    #[test]
    fn extended_path_smoke() {
        // Bw = 18, E = 3, T1 = 1<<15, T2 = 1<<16. ω = 0 (mid-grey region).
        // dc_bias = 1<<17 = 131072. v = 131072. T2 = 65536 → v >= T2 →
        // regular region: v = v - B3 = v - (T2 - 1<<(Bw-E-1)) = 131072 -
        // (65536 - (1<<14)) = 131072 - 49152 = 81920. clamped, then
        // v = A3 + v*v.
        let omega = vec![0i32; 1];
        let out = apply_output_scaling(
            &omega,
            18,
            8,
            Some(NltParams::Extended {
                t1: 1 << 15,
                t2: 1 << 16,
                e: 3,
            }),
        )
        .unwrap();
        // We don't pin a specific output value here — the goal is to
        // prove the path runs and produces a clamped u8 without panic.
        // (The full extended-NLT regression lives in the round-trip
        // fixture once an extended-NLT codestream lands.) Length check
        // proves the path returned successfully.
        assert_eq!(out.len(), 1);
    }
}
