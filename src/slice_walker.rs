//! Slice / precinct / packet geometry walker — ISO/IEC 21122-1:2022,
//! Annex B.5 / B.6 / B.7 / B.8 / B.9 / B.10.
//!
//! Round-4 scope (#106): build the per-precinct geometry the entropy
//! decoder needs from the picture header / component table / weights
//! table, including the per-precinct list of [`PacketLayout`]s computed
//! by the algorithm in Annex B.7, Table B.4.
//!
//! Round-4 limits the configurations the walker accepts to keep the
//! initial end-to-end decode tractable: single component (`Nc == 1`),
//! `Sd == 0` (no decomposition suppression), `sx == sy == 1` (no
//! subsampling), `Cw == 0` (single-precinct rows). Multi-component,
//! 4:2:2/4:2:0, and CWD-driven `Sd > 0` configurations follow in later
//! rounds. The walker still computes the spec-accurate quantities — the
//! restrictions are guarded at the entry point so the decoder fails
//! cleanly rather than silently producing garbage on out-of-scope
//! codestreams.
//!
//! Derived quantities:
//!
//! * `Wc[i] = ⌊Wf / sx[i]⌋`, `Hc[i] = ⌊Hf / sy[i]⌋` (Annex B.1).
//! * `Wb[β,i]`, `Hb[β,i]` — band dimensions (Annex B.2).
//! * `dx[β,i]`, `dy[β,i]`, `τx[β]`, `τy[β]` — decomposition depths and
//!   high-pass selector flags (Annex B.3).
//! * `b'x[b]` — band-existence flag (Annex B.4).
//! * `Cs`, `Wp[p]`, `Hp`, `Np_x`, `Np_y` — precinct grid (Annex B.5).
//! * `L0[p,b]`, `L1[p,b]`, `Wpb[p,b]` — per-precinct band geometry
//!   (Annex B.5 / B.6).
//! * `I[p,b,λ,s]` — line inclusion flags and `Npc[p]` (Annex B.7).
//! * `Np[t]` — slice-precinct count (Annex B.10).
//!
//! The output is one [`PrecinctPlan`] per precinct in the slice, holding
//! the [`PrecinctGeometry`] (consumed by the entropy decoder) and the
//! ordered list of [`PacketLayout`]s.

use oxideav_core::{Error, Result};

use crate::component_table::ComponentTable;
use crate::entropy::{BandGeometry, PacketEntry, PacketLayout, PrecinctGeometry};
use crate::picture_header::PictureHeader;

/// Hard cap on band dimensions to bound allocation regardless of
/// arithmetic mistakes downstream. Picture-header `Wf`/`Hf` are u16, so
/// 65536 is already an upper bound on either dimension; we restate it
/// here so a corrupt `Cw` or `Hsl` field cannot blow the heap.
const MAX_DIM: usize = 1 << 17;

/// Per-band parameters from the WGT marker (Annex A.4.11, Table A.24).
/// `Sd == 0` configurations have one (gain, priority) pair per existing
/// band index; absent bands are skipped in the WGT body, mirroring the
/// `if (b'x[b])` guard in Table A.24.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandWeight {
    pub gain: u8,
    pub priority: u8,
}

/// Plan for a single precinct: the [`PrecinctGeometry`] for the entropy
/// decoder plus the ordered [`PacketLayout`]s.
#[derive(Debug, Clone)]
pub struct PrecinctPlan {
    /// Precinct geometry consumed by [`crate::entropy::decode_packet_body`].
    pub geometry: PrecinctGeometry,
    /// Ordered packet layouts (`s = 0 .. Npc[p] - 1`).
    pub packets: Vec<PacketLayout>,
    /// Precinct index `p` within the picture (raster scan).
    pub p: u32,
    /// Precinct height `Hp = 2^NL,y` in sample-grid lines.
    pub hp: u32,
    /// Precinct width `Wp[p]` in sample-grid columns.
    pub wp: u32,
    /// `Cs` — column width of all but the rightmost precinct, in
    /// sample-grid columns (Annex B.5).
    pub cs: u32,
}

/// Plan for a single slice: a contiguous run of precincts.
#[derive(Debug, Clone)]
pub struct SlicePlan {
    /// Slice index `t` (matches SLH `Yslh`).
    pub t: u32,
    /// First precinct `p_first` in raster order.
    pub p_first: u32,
    /// Number of precincts in the slice (`Np[t]`, Annex B.10).
    pub n_precincts: u32,
    /// Per-precinct plans, ordered by `p` ascending.
    pub precincts: Vec<PrecinctPlan>,
}

/// Plan for the entire picture: every slice in order.
#[derive(Debug, Clone)]
pub struct PicturePlan {
    pub slices: Vec<SlicePlan>,
    /// Number of horizontal decomposition levels (`NL,x`).
    pub nlx: u8,
    /// Number of vertical decomposition levels (`NL,y`).
    pub nly: u8,
    /// Number of bands `NL = (Nc - Sd) × Nβ + Sd`.
    pub n_bands: u32,
    /// Number of wavelet filter types `Nβ`.
    pub n_beta: u32,
    /// Picture width and height (sample grid).
    pub wf: u32,
    pub hf: u32,
}

/// Parse the WGT body into `(gain, priority)` pairs, one per existing
/// band. The walker calls this with `n_existing` derived from the
/// picture/component geometry.
pub fn parse_wgt(wgt_body: &[u8], n_existing: usize) -> Result<Vec<BandWeight>> {
    if wgt_body.len() != n_existing * 2 {
        return Err(Error::invalid(format!(
            "jpegxs WGT body must be 2*N_existing = {} bytes, got {}",
            n_existing * 2,
            wgt_body.len()
        )));
    }
    let mut out = Vec::with_capacity(n_existing);
    for i in 0..n_existing {
        let gain = wgt_body[i * 2];
        let priority = wgt_body[i * 2 + 1];
        if gain > 15 {
            return Err(Error::invalid(format!(
                "jpegxs WGT G[{i}] = {gain} exceeds Annex A.4.11 cap of 15"
            )));
        }
        out.push(BandWeight { gain, priority });
    }
    Ok(out)
}

/// Compute `Nβ = 2 × min(NL,x, NL,y) + max(NL,x, NL,y) + 1` per
/// Annex B.3.
pub fn n_beta(nlx: u8, nly: u8) -> u32 {
    let mn = nlx.min(nly) as u32;
    let mx = nlx.max(nly) as u32;
    2 * mn + mx + 1
}

/// Compute the (dx, dy, τx, τy) for filter type β under decomposition
/// `(NL,x, NL,y)` for a vertically-decomposed component (`i < Nc - Sd`).
///
/// Annex B.3 (Table B.1 / B.2 / B.3): for `NL,x ≥ NL,y`,
/// * if `dx > dy`: β = NL,x − dx + τx
/// * else: β = (NL,x − NL,y + τx + 2τy + 3) × NL,y − dy
///
/// We invert that mapping by direct enumeration since the decoder side
/// always knows β and needs (dx, dy).
fn beta_levels(beta: u32, nlx: u8, nly: u8) -> (u32, u32, bool, bool) {
    let nlx = nlx as u32;
    let nly = nly as u32;
    debug_assert!(nlx >= nly, "round 4 walker assumes NL,x >= NL,y");

    // Case 1 — wholly horizontal decomposition (nly == 0, or β within
    // the first nlx + 1 indices). Spec: β = NL,x − dx + τx where
    // τx ∈ {0, 1}; β = 0 → dx = nlx, τx = 0 (LL band).
    // β = 1 → dx = nlx, τx = 1 (HL_nlx).
    // β = 2 → dx = nlx − 1, τx = 1 (HL_{nlx-1}). … and so on.
    if nly == 0 {
        // LL_nlx is β = 0.
        if beta == 0 {
            return (nlx, 0, false, false);
        }
        // β = 1 .. nlx maps to HL_{nlx + 1 - β}.
        let dx = nlx + 1 - beta;
        return (dx, 0, true, false);
    }

    // β1 = nlx − nly + 1 — number of bands in the first packet (the
    // shaded area in Figure B.3). For 5/2 this is 4: LL_5, HL_5, HL_4,
    // HL_3.
    let beta1 = nlx - nly + 1;
    if beta < beta1 {
        // First-packet bands. β = 0 is LL_nlx, β = 1..beta1 is HL_{nlx
        // + 1 - β}.
        if beta == 0 {
            return (nlx, nly, false, false);
        }
        let dx = nlx + 1 - beta;
        return (dx, nly, true, false);
    }

    // For β >= β1, bands come in groups of three at decreasing dy:
    // (HL, LH, HH)_{level}, with level = nly, nly-1, ..., 1.
    let group_in = beta - beta1; // 0 .. 3*nly
    let triple = group_in / 3;
    let within = group_in % 3;
    let dy = nly - triple;
    let dx = dy; // diagonal: HL_dy, LH_dy, HH_dy
    match within {
        0 => (dx, dy, true, false), // HL
        1 => (dx, dy, false, true), // LH
        2 => (dx, dy, true, true),  // HH
        _ => unreachable!(),
    }
}

/// Compute `b'x[b]` per Annex B.4.
fn band_exists(beta: u32, i: usize, nly: u8, dy: u32, sy: u8) -> bool {
    if beta > 0 && i >= 1 {
        // Round-4 assumes Nc - Sd == Nc (Sd == 0). For multi-component
        // round 5 the proper guard is `i >= Nc - Sd`.
        // Single-component round-4: this guard never fires.
    }
    if sy == 0 {
        return false;
    }
    // Test: 2^max(NL,y - dy) × τy[β] mod sy[i] != 0 → not exists.
    // For sy == 1, the modulus is always 0 → always exists.
    let _ = (nly, dy);
    true
}

/// Build a [`PicturePlan`] from the picture header / component table /
/// WGT body. Returns an error if the configuration is outside the
/// round-4 supported subset.
pub fn build_plan(
    pih: &PictureHeader,
    cdt: &ComponentTable,
    wgt_body: &[u8],
) -> Result<(PicturePlan, Vec<BandWeight>)> {
    // Round-4 restrictions.
    if pih.nc != 1 {
        return Err(Error::Unsupported(format!(
            "jpegxs round-4 walker only supports Nc == 1 (got {})",
            pih.nc
        )));
    }
    if pih.cw != 0 {
        return Err(Error::Unsupported(
            "jpegxs round-4 walker only supports Cw == 0 (full-width precincts)".into(),
        ));
    }
    if cdt.components.len() != 1 || cdt.components[0].sx != 1 || cdt.components[0].sy != 1 {
        return Err(Error::Unsupported(
            "jpegxs round-4 walker only supports single-component, sx==sy==1".into(),
        ));
    }
    let comp = cdt.components[0];
    let nlx = pih.nlx;
    let nly = pih.nly;
    if nlx == 0 {
        return Err(Error::invalid(
            "jpegxs: PIH NL,x must be >= 1 per Table A.7",
        ));
    }
    if nly > nlx {
        return Err(Error::Unsupported(
            "jpegxs round-4 walker assumes NL,x >= NL,y".into(),
        ));
    }
    let wf = pih.wf as u32;
    let hf = pih.hf as u32;
    if (wf as usize) > MAX_DIM || (hf as usize) > MAX_DIM {
        return Err(Error::invalid(format!(
            "jpegxs: picture dimensions {wf}x{hf} exceed walker cap {MAX_DIM}"
        )));
    }
    let nbeta = n_beta(nlx, nly);
    // Sd == 0 → NL = Nc × Nβ.
    let n_bands = (pih.nc as u32) * nbeta;

    // Component-level dimensions (Annex B.1).
    let wc = wf / (comp.sx as u32);
    let hc = hf / (comp.sy as u32);

    // Pre-compute per-(β, i) band geometry (round 4 → i is always 0).
    let mut wb = vec![0u32; nbeta as usize];
    let mut hb = vec![0u32; nbeta as usize];
    let mut dx_arr = vec![0u32; nbeta as usize];
    let mut dy_arr = vec![0u32; nbeta as usize];
    let mut tau_x = vec![false; nbeta as usize];
    let mut tau_y = vec![false; nbeta as usize];
    for beta in 0..nbeta {
        let (dx, dy, tx, ty) = beta_levels(beta, nlx, nly);
        dx_arr[beta as usize] = dx;
        dy_arr[beta as usize] = dy;
        tau_x[beta as usize] = tx;
        tau_y[beta as usize] = ty;
        // Wb[β,i] = ⌈Wc / 2^dx⌉ for low-pass-horizontal,
        //          ⌈Wc / 2^(dx-1)⌉ / 2 for high-pass-horizontal.
        // Equivalent to ⌈Wc / 2^dx⌉ in both cases (the high-pass form
        // halves a doubled denominator). The spec writes it differently
        // to expose the τx parity. For 5/3 dyadic decomposition the
        // result is identical: ⌈Wc / 2^dx⌉.
        let wb_b = if !tx {
            (wc + (1u32 << dx) - 1) >> dx
        } else {
            // ⌈Wc / 2^(dx-1)⌉ / 2 (integer division). This is the
            // high-pass dimension: spec literal form, kept distinct for
            // traceability.
            let denom_minus1 = if dx == 0 { 1 } else { 1u32 << (dx - 1) };
            wc.div_ceil(denom_minus1) / 2
        };
        let hb_b = if !ty {
            if dy == 0 {
                hc
            } else {
                (hc + (1u32 << dy) - 1) >> dy
            }
        } else {
            let denom_minus1 = if dy == 0 { 1 } else { 1u32 << (dy - 1) };
            hc.div_ceil(denom_minus1) / 2
        };
        wb[beta as usize] = wb_b;
        hb[beta as usize] = hb_b;
    }

    // Precinct grid (Annex B.5). Cw == 0 → Cs = Wf, Np_x = 1.
    let cs = wf;
    let np_x: u32 = 1;
    // Hp = 2^NL,y. NL,y == 0 is a degenerate case where every component
    // line forms its own precinct. The spec writes Np_y = ⌈Hf / 2^NL,y⌉.
    let hp_pow = if nly == 0 { 1u32 } else { 1u32 << nly };
    let hp = hp_pow;
    let np_y = hf.div_ceil(hp_pow);

    // Per-band gain/priority from WGT (round 4 — every band exists).
    let weights = parse_wgt(wgt_body, nbeta as usize)?;

    // Per-precinct plans.
    let mut precincts: Vec<PrecinctPlan> = Vec::with_capacity(np_y as usize);
    let mut precinct_index_y: Vec<u32> = (0..np_y).collect();
    let _ = &mut precinct_index_y;
    for py in 0..np_y {
        for px in 0..np_x {
            let p = py * np_x + px;
            let plan = build_precinct_plan(
                p, px, py, nlx, nly, nbeta, comp.sx, comp.sy, cs, hp, np_x, hf, &dx_arr, &dy_arr,
                &tau_y, &wb, &hb, &weights, pih,
            )?;
            precincts.push(plan);
        }
    }

    // Group precincts into slices per Annex B.10.
    let hsl = pih.hsl as u32;
    if hsl == 0 {
        return Err(Error::invalid("jpegxs: PIH Hsl must be >= 1"));
    }
    let mut slices: Vec<SlicePlan> = Vec::new();
    let mut p_cursor = 0u32;
    let mut t = 0u32;
    while p_cursor < precincts.len() as u32 {
        // n_per_slice from Annex B.10:
        // If (t+1)*Hsl > ⌈Hf / Hp⌉ → Np[t] = Np_x × (⌈Hf/Hp⌉ mod Hsl)
        // else Np[t] = Np_x × Hsl
        let total_rows = np_y;
        let next_row = (t + 1) * hsl;
        let rows_in_slice = if next_row > total_rows {
            total_rows.saturating_sub(t * hsl)
        } else {
            hsl
        };
        let np_t = np_x * rows_in_slice;
        let p_end = p_cursor + np_t;
        if p_end > precincts.len() as u32 {
            return Err(Error::invalid(
                "jpegxs slice walker: slice extends past last precinct",
            ));
        }
        let slice_precincts: Vec<PrecinctPlan> =
            precincts[p_cursor as usize..p_end as usize].to_vec();
        slices.push(SlicePlan {
            t,
            p_first: p_cursor,
            n_precincts: np_t,
            precincts: slice_precincts,
        });
        p_cursor = p_end;
        t += 1;
    }

    Ok((
        PicturePlan {
            slices,
            nlx,
            nly,
            n_bands,
            n_beta: nbeta,
            wf,
            hf,
        },
        weights,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_precinct_plan(
    p: u32,
    _px: u32,
    py: u32,
    nlx: u8,
    nly: u8,
    nbeta: u32,
    sx: u8,
    sy: u8,
    cs: u32,
    hp: u32,
    np_x: u32,
    hf: u32,
    dx: &[u32],
    dy: &[u32],
    tau_y: &[bool],
    _wb: &[u32],
    hb: &[u32],
    weights: &[BandWeight],
    pih: &PictureHeader,
) -> Result<PrecinctPlan> {
    // Wp[p] (Annex B.5): all but the rightmost precinct are Cs wide,
    // last precinct picks up the remainder. Round-4 has Np_x == 1, so
    // the last is always the only.
    let wp = if (p % np_x) < np_x - 1 {
        cs
    } else {
        ((pih.wf as u32 - 1) % cs) + 1
    };

    // Per-band geometry within this precinct.
    let mut bands: Vec<BandGeometry> = Vec::with_capacity(nbeta as usize);
    for beta in 0..nbeta {
        let beta_i = beta as usize;
        let dx_b = dx[beta_i];
        let dy_b = dy[beta_i];
        let tau_y_b = if tau_y[beta_i] { 1u32 } else { 0u32 };
        let exists = band_exists(beta, 0, nly, dy_b, sy);
        // Wpb[p,b] = ⌈Wp[p] / (sx[i] × 2^dx[β,i])⌉ for low-pass H,
        //          = ⌈Wp[p] / (sx[i] × 2^(dx-1))⌉ / 2 for high-pass H.
        let denom = (sx as u32) * (1u32 << dx_b);
        let wpb = if denom == 0 { 0 } else { wp.div_ceil(denom) };
        // L0[p,b] (Annex B.6):
        //   L0 = 2^max(NL,y - dy[i,β], 0) × τy[β]
        let pow = if dy_b > nly as u32 {
            1u32
        } else {
            1u32 << (nly as u32 - dy_b)
        };
        let l0 = pow * tau_y_b;
        // L1 = L0 + min(Hb[β,i] − ⌊p / Np_x⌋ × 2^max(NL,y - dy, 0),
        //                  2^max(NL,y - dy, 0))
        let pow_dy = pow; // both maxes use NL,y - dy with floor 0
        let row_offset = py * pow_dy;
        let band_h_remaining = (hb[beta_i]).saturating_sub(row_offset);
        let l1_extent = band_h_remaining.min(pow_dy);
        let l1 = l0 + l1_extent;
        let weight = weights[beta_i];
        bands.push(BandGeometry {
            wpb,
            gain: weight.gain,
            priority: weight.priority,
            l0: l0 as u16,
            l1: l1 as u16,
            exists,
        });
    }

    let geometry = PrecinctGeometry {
        bands: bands.clone(),
        ng: pih.ng,
        ss: pih.ss,
        br: pih.br,
        fs: pih.fs,
        rm: pih.rm,
        rl: pih.rl,
        lh: pih.lh,
        // Per Table C.3 short header threshold.
        short_packet_header: (pih.wf as u32) * (pih.nc as u32) < 32752,
    };

    let packets = compute_packet_layouts(nlx, nly, &bands, dy, hp);

    Ok(PrecinctPlan {
        geometry,
        packets,
        p,
        hp,
        wp,
        cs,
    })
    .and_then(|plan| {
        // Sanity: the rightmost precinct's Wp is non-empty.
        if plan.wp == 0 {
            return Err(Error::invalid(
                "jpegxs slice walker: precinct width Wp[p] computed as zero",
            ));
        }
        // Hf parameter is unused in the formula above (Hb already
        // encodes it); keep it referenced for clarity.
        let _ = hf;
        Ok(plan)
    })
}

/// Annex B.7 / Table B.4 — compute `I[p,b,λ,s]` and `Npc[p]` for one
/// precinct's bands. Returns one [`PacketLayout`] per packet `s`.
///
/// Round-4 simplifications: `Sd == 0` (no `i ≥ Nc − Sd` tail loop),
/// single component (`Nc - Sd == 1`), `sy[0] == 1` (the subsampling
/// guard `(λ + L0[p,b]) umod sy[i] == 0` is always true).
fn compute_packet_layouts(
    nlx: u8,
    nly: u8,
    bands: &[BandGeometry],
    dy: &[u32],
    _hp: u32,
) -> Vec<PacketLayout> {
    let mut layouts: Vec<Vec<PacketEntry>> = Vec::new();

    // Step 1 — first packet covers β1 = max(NL,x, NL,y) − min(NL,x,
    // NL,y) + 1 bands, all on line λ = 0.
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let beta1 = nlx_u.max(nly_u) - nlx_u.min(nly_u) + 1;
    let mut first_pkt: Vec<PacketEntry> = Vec::new();
    for beta in 0..beta1 {
        let bi = beta as usize;
        if bi < bands.len() && bands[bi].exists {
            // Single-component → b = β.
            first_pkt.push(PacketEntry {
                band: bi as u16,
                line: 0,
            });
        }
    }
    if !first_pkt.is_empty() {
        layouts.push(first_pkt);
    }

    // Step 2 — proxy levels: for β0 = β1, β1+3, ..., < Nβ:
    //   lines_in_level = 2^(NL,y - dy[0,β0])    (Table B.4)
    //   for λ = 0 .. lines_in_level - 1:
    //     for β = β0 .. β0+2:
    //       if (λ + L0[p,b] < L1[p,b]) → new packet (r=1) per band.
    let nbeta = bands.len() as u32;
    let mut beta0 = beta1;
    while beta0 < nbeta {
        let bi0 = beta0 as usize;
        if bi0 >= dy.len() {
            break;
        }
        let dy_b0 = dy[bi0];
        let pow = if dy_b0 > nly_u {
            1u32
        } else {
            1u32 << (nly_u - dy_b0)
        };
        let lines_in_level = pow;
        for lambda_within in 0..lines_in_level {
            for beta in beta0..(beta0 + 3).min(nbeta) {
                let bi = beta as usize;
                if !bands[bi].exists {
                    continue;
                }
                let l0 = bands[bi].l0 as u32;
                let l1 = bands[bi].l1 as u32;
                let line_in_precinct = l0 + lambda_within;
                if line_in_precinct >= l1 {
                    continue;
                }
                // Each band starts a new packet (r = 1 in Table B.4
                // resets per band; the `r = 0` after the first
                // included component within a band is the
                // multi-component aggregation we don't have here).
                layouts.push(vec![PacketEntry {
                    band: bi as u16,
                    line: line_in_precinct as u16,
                }]);
            }
        }
        beta0 += 3;
    }

    // Round-4 ignores the `Sd > 0` tail loop (no non-decomposed
    // components present).

    layouts
        .into_iter()
        .map(|entries| PacketLayout { entries })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_table::Component;

    fn pih_min(nlx: u8, nly: u8, wf: u16, hf: u16) -> PictureHeader {
        PictureHeader {
            lcod: 0,
            ppih: 0,
            plev: 0,
            wf,
            hf,
            cw: 0,
            hsl: 1,
            nc: 1,
            ng: 4,
            ss: 8,
            bw: 8,
            fq: 8,
            br: 4,
            fslc: 0,
            ppoc: 0,
            cpih: 0,
            nlx,
            nly,
            lh: 0,
            rl: 0,
            qpih: 0,
            fs: 0,
            rm: 0,
        }
    }

    fn cdt_one(bd: u8) -> ComponentTable {
        ComponentTable {
            components: vec![Component {
                bit_depth: bd,
                sx: 1,
                sy: 1,
            }],
        }
    }

    #[test]
    fn n_beta_matches_spec() {
        assert_eq!(n_beta(5, 0), 6);
        assert_eq!(n_beta(5, 1), 8);
        assert_eq!(n_beta(5, 2), 10);
        assert_eq!(n_beta(1, 1), 4);
        assert_eq!(n_beta(2, 1), 5);
    }

    #[test]
    fn beta_levels_5_0() {
        // NL,x=5 NL,y=0 → 6 bands β = 0..5
        // β=0 → LL5 (dx=5, dy=0, τx=0)
        // β=1 → HL5 (dx=5, τx=1)
        // β=5 → HL1 (dx=1, τx=1)
        let (dx, dy, tx, ty) = beta_levels(0, 5, 0);
        assert_eq!((dx, dy, tx, ty), (5, 0, false, false));
        let (dx, _, tx, _) = beta_levels(1, 5, 0);
        assert_eq!((dx, tx), (5, true));
        let (dx, _, tx, _) = beta_levels(5, 5, 0);
        assert_eq!((dx, tx), (1, true));
    }

    #[test]
    fn beta_levels_1_1() {
        // NL,x=1 NL,y=1 → Nβ = 4. β1 = 1.
        // β=0 → LL1 (dx=1, dy=1, τx=0, τy=0)
        // β=1 → HL1 (dx=1, dy=1, τx=1, τy=0)
        // β=2 → LH1 (τx=0, τy=1)
        // β=3 → HH1 (τx=1, τy=1)
        assert_eq!(beta_levels(0, 1, 1), (1, 1, false, false));
        assert_eq!(beta_levels(1, 1, 1), (1, 1, true, false));
        assert_eq!(beta_levels(2, 1, 1), (1, 1, false, true));
        assert_eq!(beta_levels(3, 1, 1), (1, 1, true, true));
    }

    #[test]
    fn build_plan_minimum_1x1_decomp() {
        // 4x4 image, NL,x=1 NL,y=1 → 4 bands, 1 precinct (since Hp=2,
        // ⌈4/2⌉ = 2 precincts vertically).
        let pih = pih_min(1, 1, 4, 4);
        let cdt = cdt_one(8);
        // WGT body: 4 bands × 2 bytes.
        let wgt = vec![0u8, 0, 0, 0, 0, 0, 0, 0];
        let (plan, weights) = build_plan(&pih, &cdt, &wgt).expect("build plan");
        assert_eq!(plan.n_bands, 4);
        assert_eq!(plan.n_beta, 4);
        assert_eq!(weights.len(), 4);
        // Hp = 2^NL,y = 2 → Np_y = ⌈4/2⌉ = 2 precincts vertically.
        assert_eq!(plan.slices.len(), 2);
        assert_eq!(plan.slices[0].n_precincts, 1);
        let p0 = &plan.slices[0].precincts[0];
        // Wp = 4. LL band Wpb = ⌈4 / 2^1⌉ = 2.
        assert_eq!(p0.geometry.bands.len(), 4);
        assert_eq!(p0.geometry.bands[0].wpb, 2);
        // L0 / L1 for LL band (τy = 0, dy=1, NL,y=1): L0 = 0,
        // L1 = min(Hb_LL=2, 2^(NL,y-dy)=1) = 1. (LL has one line in
        // the precinct because the precinct holds Hp = 2^NL,y = 2
        // image rows but only one LL coefficient row.)
        assert_eq!(p0.geometry.bands[0].l0, 0);
        assert_eq!(p0.geometry.bands[0].l1, 1);
        // LH band (β=2, τy=1): L0=1, L1=2.
        assert_eq!(p0.geometry.bands[2].l0, 1);
        assert_eq!(p0.geometry.bands[2].l1, 2);
    }

    #[test]
    fn build_plan_horizontal_only() {
        // 8x4 image, NL,x=2 NL,y=0. Nβ = 3. Hp = 1 → 4 precincts
        // vertically.
        let pih = pih_min(2, 0, 8, 4);
        let cdt = cdt_one(8);
        let wgt = vec![0u8; 6];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).expect("build plan");
        assert_eq!(plan.n_beta, 3);
        // Each precinct is 1 line tall. 4 precincts in 4 slices (Hsl=1).
        assert_eq!(plan.slices.len(), 4);
        let p0 = &plan.slices[0].precincts[0];
        // LL2 width = ⌈8 / 4⌉ = 2.
        assert_eq!(p0.geometry.bands[0].wpb, 2);
        // HL2 width = ⌈8 / 4⌉ = 2 (high-pass form: ⌈8/2⌉/2 = 2).
        // HL1 width = ⌈8 / 2⌉ = 4 (high-pass form: ⌈8/1⌉/2 = 4).
        assert_eq!(p0.geometry.bands[1].wpb, 2);
        assert_eq!(p0.geometry.bands[2].wpb, 4);
    }

    #[test]
    fn rejects_multi_component_for_round_4() {
        let mut pih = pih_min(1, 1, 4, 4);
        pih.nc = 3;
        let cdt = ComponentTable {
            components: vec![
                Component {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1
                };
                3
            ],
        };
        let wgt = vec![0u8; 8];
        assert!(build_plan(&pih, &cdt, &wgt).is_err());
    }

    #[test]
    fn parse_wgt_round_trip() {
        let body = vec![5u8, 100, 7, 200];
        let w = parse_wgt(&body, 2).unwrap();
        assert_eq!(
            w[0],
            BandWeight {
                gain: 5,
                priority: 100
            }
        );
        assert_eq!(
            w[1],
            BandWeight {
                gain: 7,
                priority: 200
            }
        );
    }

    #[test]
    fn parse_wgt_rejects_oversized_gain() {
        let body = vec![16u8, 0];
        assert!(parse_wgt(&body, 1).is_err());
    }

    #[test]
    fn packet_layouts_for_1x1_decomp() {
        // NL,x=1 NL,y=1 → β1 = 1, so first packet has 1 band (LL).
        // Then proxy level β0=1 with 3 bands (HL, LH, HH), each on
        // its own packet, on every line of the band.
        let pih = pih_min(1, 1, 4, 4);
        let cdt = cdt_one(8);
        let wgt = vec![0u8; 8];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).unwrap();
        let p0 = &plan.slices[0].precincts[0];
        // For NL,y=1 → Hp=2, lines_in_level for β0=1 = 1 (since dy=1,
        // 2^(NL,y - dy) = 1). So 3 packets after the first.
        // Total packets = 1 (LL on line 0) + 3 (HL, LH, HH on line 1).
        assert_eq!(p0.packets.len(), 4);
        // Packet 0: LL band (β=0) on line 0.
        assert_eq!(p0.packets[0].entries.len(), 1);
        assert_eq!(p0.packets[0].entries[0].band, 0);
        assert_eq!(p0.packets[0].entries[0].line, 0);
        // Packets 1..3: HL, LH, HH on line 1.
        // β=1 (HL_1) τy=0 → L0=0, line=0. So HL is on line 0.
        // β=2 (LH_1) τy=1 → L0=1, line=1. So LH is on line 1.
        // β=3 (HH_1) τy=1 → L0=1, line=1. So HH is on line 1.
        assert_eq!(p0.packets[1].entries[0].band, 1);
        assert_eq!(p0.packets[1].entries[0].line, 0);
        assert_eq!(p0.packets[2].entries[0].band, 2);
        assert_eq!(p0.packets[2].entries[0].line, 1);
        assert_eq!(p0.packets[3].entries[0].band, 3);
        assert_eq!(p0.packets[3].entries[0].line, 1);
    }
}
