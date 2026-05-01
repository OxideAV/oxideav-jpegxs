//! Inverse quantization (ISO/IEC 21122-1:2022, Annex D).
//!
//! The decoder turns the per-band quantization-index magnitudes
//! `v[p,λ,b,x]` and signs `s[p,λ,b,x]` (from the entropy decoder) plus
//! the truncation positions `T[p,b]` and bitplane counts `M[p,λ,b,g]`
//! (also entropy-decoded) into wavelet coefficients `c[p,λ,b,x]`.
//!
//! Two inverse quantizer kernels are specified by the standard:
//!
//! * [`inverse_deadzone`] — Annex D.2, Table D.1 (`Qpih == 0`).
//!   Reconstruction point is the middle of each bucket; a zero bucket
//!   that is twice the size of all others is implied by the predicate
//!   `M > T && v != 0`.
//! * [`inverse_uniform`] — Annex D.3, Table D.2 (`Qpih == 1`).
//!   All buckets equal-sized; the reconstruction is the Neumann
//!   series shown in the table (a fixed-point multiplication with
//!   `Δ = 2^(M+1) / (2^(M+1-T) - 1)` carried out as repeated shifts).
//!
//! Round-4 wires the deadzone path; the uniform path is implemented in
//! the same module so a `Qpih == 1` codestream decodes correctly the
//! moment the slice walker hands one in.

use crate::entropy::{BandCoefficients, PrecinctGeometry};

/// Inverse deadzone quantizer (Annex D.2, Table D.1) applied to one
/// band. Writes reconstructed coefficients into `out`, which must have
/// `band.num_lines * band.wpb` entries (one per coefficient in the
/// band's footprint inside this precinct).
///
/// `t` is `T[p,b]` from [`crate::entropy::truncation_position`]. `ng` is
/// the picture-header coefficients-per-code-group constant.
pub fn inverse_deadzone(band: &BandCoefficients, t: u8, ng: u8, out: &mut [i32]) {
    let wpb = band.wpb as usize;
    let ng_u = ng as usize;
    let num_lines = band.num_lines as usize;
    let t_u = t as u32;
    debug_assert_eq!(out.len(), wpb * num_lines);

    for line in 0..num_lines {
        let mline = if ng_u == 0 { 0 } else { wpb.div_ceil(ng_u) };
        for x in 0..wpb {
            let g = x.checked_div(ng_u).unwrap_or(0);
            let v = band.v[line * wpb + x];
            let m = band.m.get(line * mline + g).copied().unwrap_or(0) as u32;
            let sign = band.s[line * wpb + x];
            let c: i32 = if m > t_u && v != 0 {
                // r = (1 << T) >> 1
                let r: u32 = (1u32 << t_u) >> 1;
                // σ = 1 - 2s  →  +1 for s == 0, -1 for s == 1.
                let sigma: i32 = 1 - 2 * (sign as i32);
                let mag = (v << t_u).wrapping_add(r) as i32;
                sigma * mag
            } else {
                0
            };
            out[line * wpb + x] = c;
        }
    }
}

/// Inverse uniform quantizer (Annex D.3, Table D.2). Same signature as
/// [`inverse_deadzone`].
pub fn inverse_uniform(band: &BandCoefficients, t: u8, ng: u8, out: &mut [i32]) {
    let wpb = band.wpb as usize;
    let ng_u = ng as usize;
    let num_lines = band.num_lines as usize;
    let t_u = t as u32;
    debug_assert_eq!(out.len(), wpb * num_lines);

    for line in 0..num_lines {
        let mline = if ng_u == 0 { 0 } else { wpb.div_ceil(ng_u) };
        for x in 0..wpb {
            let g = x.checked_div(ng_u).unwrap_or(0);
            let v = band.v[line * wpb + x];
            let m = band.m.get(line * mline + g).copied().unwrap_or(0) as u32;
            let sign = band.s[line * wpb + x];
            let c: i32 = if m > t_u && v != 0 {
                // φ = v << T
                let mut phi: u64 = (v as u64) << t_u;
                // ζ = M - T + 1
                let zeta = m - t_u + 1;
                let mut rho: u64 = 0;
                if zeta > 0 {
                    while phi > 0 {
                        rho = rho.saturating_add(phi);
                        // Avoid panic on shift overflow when zeta is
                        // u32::MAX (impossible by header constraints
                        // but guard cheaply).
                        if zeta >= 64 {
                            break;
                        }
                        phi >>= zeta;
                    }
                }
                let sigma: i32 = 1 - 2 * (sign as i32);
                sigma * (rho as i32)
            } else {
                0
            };
            out[line * wpb + x] = c;
        }
    }
}

/// Dispatch wrapper that picks the kernel from `qpih`.
///
/// `qpih` follows Annex A.4.4 Table A.10:
/// * `0` → inverse deadzone (Annex D.2).
/// * `1` → inverse uniform (Annex D.3).
/// * other values are reserved; the caller must reject them upstream.
pub fn inverse_quantize(qpih: u8, band: &BandCoefficients, t: u8, ng: u8, out: &mut [i32]) {
    match qpih {
        1 => inverse_uniform(band, t, ng, out),
        // 0 is the spec default; round 4 hits this path.
        _ => inverse_deadzone(band, t, ng, out),
    }
}

/// Convenience: dequantize every band of a precinct using a freshly-
/// computed truncation table. Returns one `Vec<i32>` per band, each
/// row-major `(num_lines × wpb)`.
pub fn dequantize_precinct(
    qpih: u8,
    geom: &PrecinctGeometry,
    truncation: &[u8],
    bands: &[BandCoefficients],
) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(geom.bands.len());
    for (b, coef) in bands.iter().enumerate() {
        let band_geom = &geom.bands[b];
        if !band_geom.exists {
            out.push(Vec::new());
            continue;
        }
        let len = (coef.num_lines as usize) * (coef.wpb as usize);
        let mut buf = vec![0i32; len];
        let t = truncation.get(b).copied().unwrap_or(0);
        inverse_quantize(qpih, coef, t, geom.ng, &mut buf);
        out.push(buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BandCoefficients;

    fn band_one_line(wpb: u32, v: Vec<u32>, s: Vec<u8>, m: Vec<u8>) -> BandCoefficients {
        BandCoefficients {
            wpb,
            num_lines: 1,
            v,
            s,
            m,
        }
    }

    #[test]
    fn deadzone_reconstructs_from_v_and_t() {
        // wpb=4, Ng=4 → 1 group. T=2, M=4 (M>T → reconstruct).
        // v = [3, 0, 1, 5], s = [0, 0, 1, 0].
        // r = (1<<2)>>1 = 2.
        // x=0: 3<<2 + 2 = 14, +1 → +14.
        // x=1: v=0 → 0.
        // x=2: 1<<2 + 2 = 6, -1 → -6.
        // x=3: 5<<2 + 2 = 22, +1 → +22.
        let band = band_one_line(4, vec![3, 0, 1, 5], vec![0, 0, 1, 0], vec![4]);
        let mut out = vec![0i32; 4];
        inverse_deadzone(&band, 2, 4, &mut out);
        assert_eq!(out, vec![14, 0, -6, 22]);
    }

    #[test]
    fn deadzone_zero_when_m_le_t() {
        // M = 2, T = 2 → M <= T, so coefficient is 0 regardless.
        let band = band_one_line(4, vec![3, 0, 1, 5], vec![0, 0, 1, 0], vec![2]);
        let mut out = vec![0i32; 4];
        inverse_deadzone(&band, 2, 4, &mut out);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn uniform_matches_neumann_series() {
        // T=0, M=1. v=[1,0,2,3]. ζ = 2.
        // x=0: φ=1, rho=1, φ=0 → c = +1.
        // x=2: φ=2, rho=2; φ=2>>2=0 → c = +2.
        // x=3: φ=3, rho=3; φ=3>>2=0 → c = +3.
        let band = band_one_line(4, vec![1, 0, 2, 3], vec![0, 0, 0, 1], vec![1]);
        let mut out = vec![0i32; 4];
        inverse_uniform(&band, 0, 4, &mut out);
        assert_eq!(out, vec![1, 0, 2, -3]);
    }

    #[test]
    fn deadzone_t_zero_passes_through() {
        // T=0, M=1, r=0. v=[7,0,1,2], s=[0,0,1,1].
        // c[x] = ±(v<<0 + 0) = ±v.
        let band = band_one_line(4, vec![7, 0, 1, 2], vec![0, 0, 1, 1], vec![1]);
        let mut out = vec![0i32; 4];
        inverse_deadzone(&band, 0, 4, &mut out);
        assert_eq!(out, vec![7, 0, -1, -2]);
    }
}
