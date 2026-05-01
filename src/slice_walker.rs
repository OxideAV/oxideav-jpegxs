//! Slice / precinct / packet geometry walker — ISO/IEC 21122-1:2022,
//! Annex B.5 / B.6 / B.7 / B.8 / B.9 / B.10.
//!
//! Round-5 scope (#129): build the per-precinct geometry the entropy
//! decoder needs from the picture header / component table / weights
//! table for the multi-component (`Nc ≥ 1`), 4:4:4 / 4:2:2 / 4:2:0
//! sub-sampled cases. `Sd == 0` only — `Sd > 0` (CWD-driven
//! decomposition suppression) is still deferred.
//!
//! Spec band-index layout (Annex B.2): for `i < Nc - Sd`, the band id
//! is `b = (Nc - Sd) * β + i`. So bands are *interleaved* by component
//! within each β level — for 3 components and 4 βs the order is
//! (β=0, i=0), (β=0, i=1), (β=0, i=2), (β=1, i=0), … (β=3, i=2).
//! Annex B.7 Table B.4 also walks them in that order, which is why
//! the first packet in 5/0 / 4:4:4 contains all 18 bands.
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
    /// Per-band component index `i[b]`, parallel to `geometry.bands`.
    /// Round-5: every band is associated with exactly one component;
    /// for `Sd == 0`, `i[b] = b % (Nc - Sd)`.
    pub band_component: Vec<u8>,
    /// Per-band β (filter type) index, parallel to `geometry.bands`.
    pub band_beta: Vec<u32>,
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
    /// Number of decomposed components (`Nc - Sd`). For `Sd == 0` this
    /// equals `Nc`. The walker uses this everywhere it computes a band
    /// index `b = (Nc - Sd) × β + i`.
    pub n_decomposed: u32,
    /// Number of total components.
    pub nc: u8,
    /// Per-component sampling factors (`sx[i]`, `sy[i]`), parallel to
    /// the component table.
    pub sx: Vec<u8>,
    pub sy: Vec<u8>,
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

    if nly == 0 {
        if beta == 0 {
            return (nlx, 0, false, false);
        }
        let dx = nlx + 1 - beta;
        return (dx, 0, true, false);
    }

    let beta1 = nlx - nly + 1;
    if beta < beta1 {
        if beta == 0 {
            return (nlx, nly, false, false);
        }
        let dx = nlx + 1 - beta;
        return (dx, nly, true, false);
    }

    let group_in = beta - beta1;
    let triple = group_in / 3;
    let within = group_in % 3;
    let dy = nly - triple;
    let dx = dy;
    match within {
        0 => (dx, dy, true, false), // HL
        1 => (dx, dy, false, true), // LH
        2 => (dx, dy, true, true),  // HH
        _ => unreachable!(),
    }
}

/// Compute `b'x[b]` per Annex B.4.
fn band_exists(beta: u32, _i_in_decomposed: usize, nly: u8, dy: u32, sy: u8, tau_y: bool) -> bool {
    if sy == 0 {
        return false;
    }
    let _ = beta;
    // Test: 2^max(NL,y - dy) × τy[β] mod sy[i] != 0 → not exists.
    let pow = if dy > nly as u32 {
        1u32
    } else {
        1u32 << (nly as u32 - dy)
    };
    let l0 = if tau_y { pow } else { 0 };
    let sy_u = sy as u32;
    if sy_u == 0 {
        return false;
    }
    (l0 % sy_u) == 0
}

/// Build a [`PicturePlan`] from the picture header / component table /
/// WGT body. Returns an error if the configuration is outside the
/// round-5 supported subset.
pub fn build_plan(
    pih: &PictureHeader,
    cdt: &ComponentTable,
    wgt_body: &[u8],
) -> Result<(PicturePlan, Vec<BandWeight>)> {
    if pih.cw != 0 {
        return Err(Error::Unsupported(
            "jpegxs walker: Cw != 0 (custom precinct width) is round-6".into(),
        ));
    }
    if cdt.components.len() != pih.nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs walker: CDT has {} components but PIH says Nc={}",
            cdt.components.len(),
            pih.nc
        )));
    }
    let nlx = pih.nlx;
    let nly = pih.nly;
    if nlx == 0 {
        return Err(Error::invalid(
            "jpegxs: PIH NL,x must be >= 1 per Table A.7",
        ));
    }
    if nly > nlx {
        return Err(Error::Unsupported(
            "jpegxs walker assumes NL,x >= NL,y".into(),
        ));
    }
    let wf = pih.wf as u32;
    let hf = pih.hf as u32;
    if (wf as usize) > MAX_DIM || (hf as usize) > MAX_DIM {
        return Err(Error::invalid(format!(
            "jpegxs: picture dimensions {wf}x{hf} exceed walker cap {MAX_DIM}"
        )));
    }

    // Annex F.2 mandates Cpih == 0 unless Nc >= 3 and sx[i]=sy[i]=1 for
    // i < 3. For Cpih == 1, all three sub-sampled components have to be
    // 1:1. Cpih == 3 needs Nc >= 4. The walker doesn't enforce these
    // (the decoder does); it only validates the geometry it sees.

    let nbeta = n_beta(nlx, nly);
    let nc = pih.nc as u32;
    let n_decomposed = nc; // Sd == 0 always for round 5.
    let n_bands = n_decomposed * nbeta; // Sd == 0 → no tail term.

    // Per-component sampling factors.
    let sx: Vec<u8> = cdt.components.iter().map(|c| c.sx).collect();
    let sy: Vec<u8> = cdt.components.iter().map(|c| c.sy).collect();
    for (i, &s) in sx.iter().enumerate() {
        if s == 0 {
            return Err(Error::invalid(format!(
                "jpegxs: component {i} sx must be >= 1, got 0"
            )));
        }
    }
    for (i, &s) in sy.iter().enumerate() {
        if s == 0 {
            return Err(Error::invalid(format!(
                "jpegxs: component {i} sy must be >= 1, got 0"
            )));
        }
    }
    // Per-component effective decomposition levels (Annex B.2):
    // N'L,y[i] = NL,y - log2(sy[i]) for i < Nc - Sd, else 0.
    // For round 5 we restrict to sy[i] in {1, 2} (4:2:0 only, as per the
    // CDT validation) — log2 is then 0 or 1.
    let nly_per_component: Vec<u8> = sy
        .iter()
        .map(|&s| {
            let log2 = match s as u32 {
                1 => 0u8,
                2 => 1u8,
                4 => 2u8,
                _ => 0u8, // anything else falls to 0 — caller should reject upstream
            };
            nly.saturating_sub(log2)
        })
        .collect();

    // Pre-compute per-(β, i) band geometry. Index into `wb` / `hb` /
    // `dx_arr` / `dy_arr` / `tau_y` is `i * nbeta + beta`.
    let arr_size = (nbeta as usize) * (nc as usize);
    let mut wb = vec![0u32; arr_size];
    let mut hb = vec![0u32; arr_size];
    let mut dx_arr = vec![0u32; arr_size];
    let mut dy_arr = vec![0u32; arr_size];
    let mut tau_x = vec![false; arr_size];
    let mut tau_y = vec![false; arr_size];
    let mut exists_arr = vec![false; arr_size];
    for (i, comp) in cdt.components.iter().enumerate() {
        let wc = wf / (comp.sx as u32);
        let hc = hf / (comp.sy as u32);
        let nlx_i = nlx; // Annex B.2: N'L,x[i] = NL,x for i < Nc - Sd
        let nly_i = nly_per_component[i];
        for beta in 0..nbeta {
            let (dx, dy, tx, ty) = beta_levels(beta, nlx_i, nly_i);
            let idx = i * (nbeta as usize) + beta as usize;
            dx_arr[idx] = dx;
            dy_arr[idx] = dy;
            tau_x[idx] = tx;
            tau_y[idx] = ty;
            // Cap β by the per-component number of filter types: if a
            // β is not defined for component i (because nly_i < nly),
            // mark the band non-existent. n_beta(nlx_i, nly_i) gives the
            // per-component count.
            let nbeta_i = n_beta(nlx_i, nly_i);
            let exists_per_comp = beta < nbeta_i;
            // Band geometry per Annex B.2.
            let wb_b = if !tx {
                if dx == 0 {
                    wc
                } else {
                    (wc + (1u32 << dx) - 1) >> dx
                }
            } else {
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
            wb[idx] = wb_b;
            hb[idx] = hb_b;
            exists_arr[idx] = exists_per_comp && band_exists(beta, i, nly, dy, comp.sy, ty);
        }
    }

    // Precinct grid (Annex B.5). Cw == 0 → Cs = Wf, Np_x = 1.
    let cs = wf;
    let np_x: u32 = 1;
    // Hp = 2^NL,y. NL,y == 0 is a degenerate case where every component
    // line forms its own precinct. The spec writes Np_y = ⌈Hf / 2^NL,y⌉.
    let hp_pow = if nly == 0 { 1u32 } else { 1u32 << nly };
    let hp = hp_pow;
    let np_y = hf.div_ceil(hp_pow);

    // Per-band gain/priority from WGT. Annex A.4.11 Table A.24 lists
    // a (G[b], P[b]) pair only for existing bands (`if (b'x[b])`); we
    // therefore feed `parse_wgt` the count of existing bands.
    let n_existing: usize = exists_arr.iter().filter(|e| **e).count();
    let weights_existing = parse_wgt(wgt_body, n_existing)?;
    // Build a band-indexed weights array (size `n_bands`); non-existent
    // bands get a placeholder zero pair that the walker never reads.
    let mut weights_by_band = vec![
        BandWeight {
            gain: 0,
            priority: 0
        };
        n_bands as usize
    ];
    {
        let mut wgt_cursor = 0;
        for beta in 0..nbeta {
            for i in 0..nc as usize {
                let idx = i * (nbeta as usize) + beta as usize;
                if !exists_arr[idx] {
                    continue;
                }
                let b = (n_decomposed * beta + i as u32) as usize;
                weights_by_band[b] = weights_existing[wgt_cursor];
                wgt_cursor += 1;
            }
        }
        debug_assert_eq!(wgt_cursor, n_existing);
    }

    // Per-precinct plans.
    let mut precincts: Vec<PrecinctPlan> = Vec::with_capacity(np_y as usize);
    for py in 0..np_y {
        for px in 0..np_x {
            let p = py * np_x + px;
            let plan = build_precinct_plan(
                p,
                px,
                py,
                nlx,
                nly,
                nbeta,
                nc,
                &sx,
                &sy,
                &nly_per_component,
                cs,
                hp,
                np_x,
                hf,
                &dx_arr,
                &dy_arr,
                &tau_y,
                &exists_arr,
                &wb,
                &hb,
                &weights_by_band,
                pih,
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
            n_decomposed,
            nc: pih.nc,
            sx,
            sy,
        },
        weights_existing,
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
    nc: u32,
    sx: &[u8],
    sy: &[u8],
    nly_per_component: &[u8],
    cs: u32,
    hp: u32,
    np_x: u32,
    hf: u32,
    dx: &[u32],
    dy: &[u32],
    tau_y: &[bool],
    exists_arr: &[bool],
    _wb: &[u32],
    hb: &[u32],
    weights_by_band: &[BandWeight],
    pih: &PictureHeader,
) -> Result<PrecinctPlan> {
    // Wp[p] (Annex B.5): all but the rightmost precinct are Cs wide,
    // last precinct picks up the remainder.
    let wp = if (p % np_x) < np_x - 1 {
        cs
    } else {
        ((pih.wf as u32 - 1) % cs) + 1
    };

    let n_bands = nc * nbeta;
    let mut bands: Vec<BandGeometry> = Vec::with_capacity(n_bands as usize);
    let mut band_component: Vec<u8> = Vec::with_capacity(n_bands as usize);
    let mut band_beta: Vec<u32> = Vec::with_capacity(n_bands as usize);
    // Fill per-band geometry in band-id order: b = nc * β + i.
    for beta in 0..nbeta {
        for i in 0..nc as usize {
            let arr_idx = i * (nbeta as usize) + beta as usize;
            let dx_b = dx[arr_idx];
            let dy_b = dy[arr_idx];
            let tau_y_b = if tau_y[arr_idx] { 1u32 } else { 0u32 };
            let exists = exists_arr[arr_idx];

            // Per-component precinct width: Wp / sx[i].
            let denom = (sx[i] as u32) * (1u32 << dx_b);
            let wpb = if denom == 0 { 0 } else { wp.div_ceil(denom) };

            // L0[p,b] = 2^max(NL,y - dy[i,β], 0) × τy[β]
            let nly_i = nly_per_component[i] as u32;
            let dy_eff = if nly == 0 { 0 } else { dy_b };
            let pow = if dy_eff > nly_i || nly_i == 0 {
                1u32
            } else {
                1u32 << (nly_i - dy_eff)
            };
            let l0 = pow * tau_y_b;

            // L1 — see spec subclause B.6:
            //   L1 = L0 + min(Hb[β,i] − ⌊p / Np_x⌋ × 2^max(NL,y - dy, 0),
            //                   2^max(NL,y - dy, 0))
            // ⌊p / Np_x⌋ is the precinct row index py.
            let row_offset = py * pow;
            let band_h_remaining = (hb[arr_idx]).saturating_sub(row_offset);
            let l1_extent = band_h_remaining.min(pow);
            let l1 = l0 + l1_extent;

            let weight = if exists {
                let b = (nc * beta + i as u32) as usize;
                weights_by_band[b]
            } else {
                BandWeight {
                    gain: 0,
                    priority: 0,
                }
            };
            bands.push(BandGeometry {
                wpb,
                gain: weight.gain,
                priority: weight.priority,
                l0: l0 as u16,
                l1: l1 as u16,
                exists,
            });
            band_component.push(i as u8);
            band_beta.push(beta);
        }
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

    let packets = compute_packet_layouts(nlx, nly, nc, &bands, dy, &band_component, sy);

    // Sanity: the rightmost precinct's Wp is non-empty.
    if wp == 0 {
        return Err(Error::invalid(
            "jpegxs slice walker: precinct width Wp[p] computed as zero",
        ));
    }
    // Hf parameter is unused in the formula above (Hb already encodes it);
    // keep it referenced for clarity.
    let _ = (hf, hp);
    Ok(PrecinctPlan {
        geometry,
        packets,
        p,
        hp,
        wp,
        cs,
        band_component,
        band_beta,
    })
}

/// Annex B.7 / Table B.4 — compute `I[p,b,λ,s]` and `Npc[p]` for one
/// precinct's bands. Returns one [`PacketLayout`] per packet `s`.
///
/// Round-5 multi-component handling: bands are interleaved by component
/// per the spec band-id rule `b = (Nc - Sd) × β + i`. The first packet
/// covers β1 bands × Nc components on line 0; subsequent packets group
/// 3 βs × Nc components on each line of the proxy level.
#[allow(clippy::too_many_arguments)]
fn compute_packet_layouts(
    nlx: u8,
    nly: u8,
    nc: u32,
    bands: &[BandGeometry],
    dy: &[u32],
    band_component: &[u8],
    sy: &[u8],
) -> Vec<PacketLayout> {
    let mut layouts: Vec<Vec<PacketEntry>> = Vec::new();

    // Step 1 — first packet: β1 = max(NL,x, NL,y) − min(NL,x, NL,y) + 1
    // bands × all components, all on line λ = 0.
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let beta1 = nlx_u.max(nly_u) - nlx_u.min(nly_u) + 1;
    let mut first_pkt: Vec<PacketEntry> = Vec::new();
    for beta in 0..beta1 {
        for i in 0..nc {
            let b = (nc * beta + i) as usize;
            if b < bands.len() && bands[b].exists {
                let l0 = bands[b].l0 as u32;
                // Subsampling guard from Table B.4: (λ + L0) umod sy[i] == 0.
                let sy_i = sy[i as usize] as u32;
                if sy_i != 0 && (l0 % sy_i) != 0 {
                    continue;
                }
                first_pkt.push(PacketEntry {
                    band: b as u16,
                    line: l0 as u16,
                });
            }
        }
    }
    if !first_pkt.is_empty() {
        layouts.push(first_pkt);
    }

    // Step 2 — proxy levels: for β0 = β1, β1+3, ..., < Nβ:
    //   lines_in_level = 2^(NL,y - dy[β0])    (Table B.4)
    //   for λ within level (in image-grid lines):
    //     for β = β0 .. β0+2:
    //       for i = 0 .. Nc-1:
    //         if exists && (λ + L0[p,b]) umod sy[i] == 0:
    //           start a new packet (r = 1) per band per component
    let nbeta_u = (bands.len() as u32) / nc;
    let mut beta0 = beta1;
    while beta0 < nbeta_u {
        // Use β0's dy for the loop bound — should match across all
        // components since dy depends on β only (not i). Look up via
        // component 0's array index.
        let arr_idx0 = beta0 as usize;
        if arr_idx0 >= dy.len() {
            break;
        }
        let dy_b0 = dy[arr_idx0];
        let pow = if dy_b0 > nly_u || nly_u == 0 {
            1u32
        } else {
            1u32 << (nly_u - dy_b0)
        };
        let lines_in_level = pow;
        for lambda_within in 0..lines_in_level {
            for beta in beta0..(beta0 + 3).min(nbeta_u) {
                for i in 0..nc {
                    let b = (nc * beta + i) as usize;
                    if b >= bands.len() || !bands[b].exists {
                        continue;
                    }
                    let l0 = bands[b].l0 as u32;
                    let l1 = bands[b].l1 as u32;
                    let line_in_precinct = l0 + lambda_within;
                    if line_in_precinct >= l1 {
                        continue;
                    }
                    let sy_i = sy[i as usize] as u32;
                    if sy_i != 0 && (line_in_precinct % sy_i) != 0 {
                        continue;
                    }
                    layouts.push(vec![PacketEntry {
                        band: b as u16,
                        line: line_in_precinct as u16,
                    }]);
                }
            }
        }
        beta0 += 3;
    }
    let _ = band_component;

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

    fn cdt_three_444() -> ComponentTable {
        ComponentTable {
            components: vec![
                Component {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
                Component {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
                Component {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
            ],
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
        let (dx, dy, tx, ty) = beta_levels(0, 5, 0);
        assert_eq!((dx, dy, tx, ty), (5, 0, false, false));
        let (dx, _, tx, _) = beta_levels(1, 5, 0);
        assert_eq!((dx, tx), (5, true));
        let (dx, _, tx, _) = beta_levels(5, 5, 0);
        assert_eq!((dx, tx), (1, true));
    }

    #[test]
    fn beta_levels_1_1() {
        assert_eq!(beta_levels(0, 1, 1), (1, 1, false, false));
        assert_eq!(beta_levels(1, 1, 1), (1, 1, true, false));
        assert_eq!(beta_levels(2, 1, 1), (1, 1, false, true));
        assert_eq!(beta_levels(3, 1, 1), (1, 1, true, true));
    }

    #[test]
    fn build_plan_minimum_1x1_decomp() {
        let pih = pih_min(1, 1, 4, 4);
        let cdt = cdt_one(8);
        let wgt = vec![0u8, 0, 0, 0, 0, 0, 0, 0];
        let (plan, weights) = build_plan(&pih, &cdt, &wgt).expect("build plan");
        assert_eq!(plan.n_bands, 4);
        assert_eq!(plan.n_beta, 4);
        assert_eq!(weights.len(), 4);
        assert_eq!(plan.slices.len(), 2);
        assert_eq!(plan.slices[0].n_precincts, 1);
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.geometry.bands.len(), 4);
        assert_eq!(p0.geometry.bands[0].wpb, 2);
        assert_eq!(p0.geometry.bands[0].l0, 0);
        assert_eq!(p0.geometry.bands[0].l1, 1);
        assert_eq!(p0.geometry.bands[2].l0, 1);
        assert_eq!(p0.geometry.bands[2].l1, 2);
    }

    #[test]
    fn build_plan_horizontal_only() {
        let pih = pih_min(2, 0, 8, 4);
        let cdt = cdt_one(8);
        let wgt = vec![0u8; 6];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).expect("build plan");
        assert_eq!(plan.n_beta, 3);
        assert_eq!(plan.slices.len(), 4);
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.geometry.bands[0].wpb, 2);
        assert_eq!(p0.geometry.bands[1].wpb, 2);
        assert_eq!(p0.geometry.bands[2].wpb, 4);
    }

    #[test]
    fn build_plan_three_components_4x4_1x1() {
        // 4x4 image, NL,x = NL,y = 1, 3 components 4:4:4. Total bands =
        // 3 * 4 = 12. WGT body has 12 (gain, priority) pairs.
        let mut pih = pih_min(1, 1, 4, 4);
        pih.nc = 3;
        let cdt = cdt_three_444();
        let wgt = vec![0u8; 24];
        let (plan, weights) = build_plan(&pih, &cdt, &wgt).expect("3-comp plan");
        assert_eq!(plan.n_bands, 12);
        assert_eq!(plan.n_beta, 4);
        assert_eq!(weights.len(), 12);
        assert_eq!(plan.nc, 3);
        // Verify band[0] is component 0 of β=0 (LL); band[1] is comp 1
        // of β=0; band[2] is comp 2 of β=0.
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.band_component[0], 0);
        assert_eq!(p0.band_beta[0], 0);
        assert_eq!(p0.band_component[1], 1);
        assert_eq!(p0.band_beta[1], 0);
        assert_eq!(p0.band_component[3], 0);
        assert_eq!(p0.band_beta[3], 1); // β=1 starts at band index 3
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
        let pih = pih_min(1, 1, 4, 4);
        let cdt = cdt_one(8);
        let wgt = vec![0u8; 8];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).unwrap();
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.packets.len(), 4);
        assert_eq!(p0.packets[0].entries.len(), 1);
        assert_eq!(p0.packets[0].entries[0].band, 0);
        assert_eq!(p0.packets[0].entries[0].line, 0);
        assert_eq!(p0.packets[1].entries[0].band, 1);
        assert_eq!(p0.packets[1].entries[0].line, 0);
        assert_eq!(p0.packets[2].entries[0].band, 2);
        assert_eq!(p0.packets[2].entries[0].line, 1);
        assert_eq!(p0.packets[3].entries[0].band, 3);
        assert_eq!(p0.packets[3].entries[0].line, 1);
    }

    #[test]
    fn packet_layouts_3_components_5_0() {
        // 3 components, NL,x=5 NL,y=0 → Table B.5: one packet with 18
        // bands on line 0.
        let mut pih = pih_min(5, 0, 32, 1);
        pih.nc = 3;
        let cdt = cdt_three_444();
        let wgt = vec![0u8; 18 * 2];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).expect("3-comp 5/0 plan");
        // n_beta = 6. 6 bands × 3 components = 18 total.
        assert_eq!(plan.n_beta, 6);
        assert_eq!(plan.n_bands, 18);
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.packets.len(), 1, "5/0 → 1 packet");
        assert_eq!(p0.packets[0].entries.len(), 18, "all 18 bands grouped");
    }
}
