//! Slice / precinct / packet geometry walker — ISO/IEC 21122-1:2022,
//! Annex B.5 / B.6 / B.7 / B.8 / B.9 / B.10.
//!
//! Round-5 scope (#129): build the per-precinct geometry the entropy
//! decoder needs from the picture header / component table / weights
//! table for the multi-component (`Nc ≥ 1`), 4:4:4 / 4:2:2 / 4:2:0
//! sub-sampled cases.
//!
//! Round-9 (r91) adds `Sd > 0` (CWD-driven decomposition suppression
//! per Annex A.4.7 Table A.18). For `i ≥ Nc - Sd` the wavelet
//! decomposition is suppressed: only filter type β = 0 carries data
//! and the band index becomes `b = (Nc - Sd) × Nβ + i`. Each such
//! component contributes one packet per picture line (per Table B.4
//! tail loop) and Annex A.4.7 mandates `sx[c] = sy[c] = 1` for
//! suppressed components.
//!
//! Spec band-index layout (Annex B.2): for `i < Nc - Sd`, the band id
//! is `b = (Nc - Sd) * β + i`. So bands are *interleaved* by component
//! within each β level — for 3 components and 4 βs the order is
//! (β=0, i=0), (β=0, i=1), (β=0, i=2), (β=1, i=0), … (β=3, i=2).
//! Annex B.7 Table B.4 also walks them in that order, which is why
//! the first packet in 5/0 / 4:4:4 contains all 18 bands. The Sd
//! suppressed bands live at the tail starting at
//! `(Nc - Sd) × Nβ` and run for `Sd` slots, one per suppressed
//! component (always β = 0).
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

use crate::error::{JpegXsError as Error, Result};

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
    /// This is the *picture-level* β slot as used by the bitstream's
    /// flat band id `b = (Nc − Sd) × β + i` (Annex B.3).
    pub band_beta: Vec<u32>,
    /// Per-band chroma-local-β: the index into the component's own DWT
    /// cascade output (`oxideav_jpegxs::dwt::forward_cascade_2d` etc.)
    /// that supplies this band's coefficients. For 4:4:4 / 4:2:2 (sy=1)
    /// this equals `band_beta[k]`; for vertically sub-sampled
    /// components (sy=2 / 4) the picture-β slot can map to a different
    /// chroma-local-β per Annex B.4 (e.g. picture β=7 = HL1,1 maps to
    /// chroma's local β=5 for NL=5/2 4:2:0 per Figure B.2). `u32::MAX`
    /// for slots that don't exist for the component (bx[β,i] = 0).
    pub band_local_beta: Vec<u32>,
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
    /// Number of precinct columns per row (`Np,x`, Annex B.5). For
    /// `Cw == 0` this is 1; for `Cw > 0` it is `Wf / Cs`.
    pub np_x: u32,
    /// Number of precinct rows (`Np,y`, Annex B.5).
    pub np_y: u32,
    /// Column width `Cs` of all but the rightmost precinct, in sample-
    /// grid columns. `Cs = Wf` when `Cw == 0`; otherwise
    /// `Cs = 8 × Cw × max(sx) × 2^NL,x`.
    pub cs: u32,
    /// Number of components whose wavelet decomposition is suppressed
    /// (Annex A.4.7 Table A.18 `Sd`). Zero unless the codestream
    /// carried a CWD marker. The last `sd` components (indices
    /// `[Nc-Sd, Nc)`) are raw-coded; their band id is
    /// `b = (Nc - Sd) × Nβ + i` and only β = 0 carries data.
    pub sd: u8,
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

/// Map a picture-level filter index `β_pic` to the equivalent
/// chroma-local filter index used by the per-component DWT cascade.
///
/// Background. Annex B.3 enumerates the wavelet filter types β with
/// the *picture-level* (`NL,x`, `NL,y`) — see Table B.3. The
/// per-component band index then comes from `b = (Nc − Sd) × β_pic + i`,
/// so the picture's β indexing is what the slice walker and packet
/// inclusion rule (Table B.4) operate on.
///
/// However the per-component forward / inverse DWT cascade (in
/// `oxideav_jpegxs::dwt`) decomposes each component at its own
/// effective vertical depth `N′L,y[i] = NL,y − log2(sy[i])`, so the
/// cascade emits / consumes bands at *chroma-local* β indexing
/// `(0..n_beta(NL,x, N′L,y[i]))` not the picture-level enumeration.
///
/// For the 4:4:4 / 4:2:2 case (sy[i] == 1) the two indexings coincide
/// because chroma's effective NL,y equals the picture's NL,y. For
/// 4:2:0 (sy[i] == 2) chroma loses one vertical level, and the
/// picture's β slots split into:
///
/// * Picture-β slots that don't exist for chroma — Annex B.4
///   bx[β,i] = 0 — these are the picture's LH/HH triples at the
///   deepest vertical depth(s). The bitstream has no entries for them.
/// * Picture-β slots that map 1:1 to a chroma-local β with matching
///   (dx, τx, τy) and `dy_chroma = dy_pic − log2(sy[i])` for proxy
///   bands, or matching dx for pure-horizontal bands (where dy is
///   irrelevant because τy = 0).
///
/// Returns `None` when bx[β_pic, i] = 0 (band doesn't exist for this
/// component). Returns `Some(β_local)` when the picture-β slot maps
/// to a chroma-local-β in the component's own DWT cascade output.
pub(crate) fn picture_beta_to_local_beta(
    beta_pic: u32,
    nlx: u8,
    nly_pic: u8,
    sy_i: u8,
) -> Option<u32> {
    if sy_i == 0 {
        return None;
    }
    // Compute chroma's effective NL,y per Annex B.2.
    let log2_sy: u8 = match sy_i {
        1 => 0,
        2 => 1,
        4 => 2,
        _ => return None,
    };
    let nly_i = nly_pic.saturating_sub(log2_sy);
    // 4:4:4 / 4:2:2 vertical (sy=1) — identity mapping.
    if log2_sy == 0 {
        return Some(beta_pic);
    }
    // Decode picture-β to (dx, dy_pic, tx, ty) under picture-level NL,y.
    let (dx_pic, dy_pic, tx, ty) = beta_levels(beta_pic, nlx, nly_pic);
    let nly_pic_u = nly_pic as u32;
    let nly_i_u = nly_i as u32;
    let nlx_u = nlx as u32;
    // Annex B.4 existence check: bx[β,i] = 0 iff 2^max(NL,y − dy)·τy[β] mod sy[i] != 0.
    let pow = if dy_pic > nly_pic_u {
        1u32
    } else {
        1u32 << (nly_pic_u - dy_pic)
    };
    let l0_image = if ty { pow } else { 0 };
    if (l0_image % sy_i as u32) != 0 {
        return None;
    }
    // LL — both picture and chroma have β=0 as LL.
    if !ty && !tx {
        return Some(0);
    }
    if !ty {
        // Picture's pure-horizontal HL (τy=0, τx=1) at picture's level
        // dy_pic = NL,y (always, since pure-H zone uses dy=NL,y). The
        // matching chroma band is chroma's HL at the same dx in chroma's
        // own enumeration.
        let beta1_chroma = nlx_u - nly_i_u + 1;
        if dx_pic > nly_i_u {
            // Chroma's pure-horizontal zone — same formula as picture's.
            return Some(nlx_u + 1 - dx_pic);
        }
        // Chroma's proxy zone HL: triple is `nly_i_u - dx_pic` (since
        // chroma's proxy goes dx = nly_i, nly_i - 1, ..., 1).
        let triple = nly_i_u - dx_pic;
        return Some(beta1_chroma + 3 * triple); // within=0 → HL
    }
    // τy = 1 (LH or HH). The band lives in the proxy zone of both
    // picture and chroma. Picture's (dx_pic, dy_pic) are equal (per
    // proxy structure); the same applies to chroma. The chroma matching
    // band has the same dy (since τy=1 bands track the picture's
    // dy_pic — chroma can only carry it when dy_pic ≤ nly_i, which the
    // bx check above already enforced).
    if dy_pic > nly_i_u {
        return None;
    }
    let beta1_chroma = nlx_u - nly_i_u + 1;
    let triple = nly_i_u - dy_pic;
    let within: u32 = if tx { 2 } else { 1 }; // HH=2, LH=1
    Some(beta1_chroma + 3 * triple + within)
}

/// Build a [`PicturePlan`] from the picture header / component table /
/// WGT body. Sd == 0 (no decomposition suppression). Equivalent to
/// [`build_plan_sd(pih, cdt, wgt_body, 0)`].
pub fn build_plan(
    pih: &PictureHeader,
    cdt: &ComponentTable,
    wgt_body: &[u8],
) -> Result<(PicturePlan, Vec<BandWeight>)> {
    build_plan_sd(pih, cdt, wgt_body, 0)
}

/// Build a [`PicturePlan`] from the picture header / component table /
/// WGT body with an explicit Sd (CWD, Annex A.4.7).
///
/// `sd` is the number of trailing components whose wavelet decomposition
/// is suppressed. Per Annex A.4.7 Table A.18, `sd ∈ 1..=Nc-1` when
/// present (and the spec further requires `Nc > 3`), and every
/// suppressed component must have `sx[c] = sy[c] = 1`. `sd = 0` reduces
/// to the original round-5 path. Returns an error if the configuration
/// is outside the supported subset.
pub fn build_plan_sd(
    pih: &PictureHeader,
    cdt: &ComponentTable,
    wgt_body: &[u8],
    sd: u8,
) -> Result<(PicturePlan, Vec<BandWeight>)> {
    if cdt.components.len() != pih.nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs walker: CDT has {} components but PIH says Nc={}",
            cdt.components.len(),
            pih.nc
        )));
    }
    if sd > 0 {
        if pih.nc <= 3 {
            return Err(Error::invalid(format!(
                "jpegxs walker: Sd>0 requires Nc>3 per Annex A.4.7, got Nc={}",
                pih.nc
            )));
        }
        if (sd as u16) >= pih.nc as u16 {
            return Err(Error::invalid(format!(
                "jpegxs walker: Sd={sd} must be < Nc={} per Table A.18",
                pih.nc
            )));
        }
        // Suppressed components (i >= Nc - Sd) must have sx[i]=sy[i]=1.
        let n_decomposed_u = (pih.nc - sd) as usize;
        for (i, c) in cdt.components.iter().enumerate().skip(n_decomposed_u) {
            if c.sx != 1 || c.sy != 1 {
                return Err(Error::invalid(format!(
                    "jpegxs walker: suppressed component i={i} (Sd) must have sx=sy=1, got sx={} sy={} (Annex A.4.7)",
                    c.sx, c.sy
                )));
            }
        }
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
    let sd_u = sd as u32;
    let n_decomposed = nc.saturating_sub(sd_u);
    // Annex B.3 NL = (Nc - Sd) × Nβ + Sd. The Sd tail bands all live at
    // β = 0 and carry the suppressed components' raw samples.
    let n_bands = n_decomposed * nbeta + sd_u;

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

    // Pre-compute per-(β_pic, i) band geometry. Index into `wb` / `hb` /
    // `dx_arr` / `dy_arr` / `tau_y` is `i * nbeta + beta_pic` where
    // `beta_pic` is the picture-level β slot used by the bitstream's
    // band index `b = (Nc - Sd) × β_pic + i` (Annex B.3). For each
    // (i, β_pic) we resolve the chroma-local-β via
    // [`picture_beta_to_local_beta`] then take that band's (dx, dy)
    // from the component's own DWT enumeration (NL,x, N'L,y[i]). When
    // the picture-β slot has no chroma equivalent (bx[β,i] = 0), the
    // slot is marked non-existent. `local_beta_arr` stores the
    // chroma-local-β for each existing slot so the encoder and decoder
    // can read the per-component DWT output buffers.
    let arr_size = (nbeta as usize) * (nc as usize);
    let mut wb = vec![0u32; arr_size];
    let mut hb = vec![0u32; arr_size];
    let mut dx_arr = vec![0u32; arr_size];
    let mut dy_arr = vec![0u32; arr_size];
    let mut tau_x = vec![false; arr_size];
    let mut tau_y = vec![false; arr_size];
    let mut exists_arr = vec![false; arr_size];
    let mut local_beta_arr = vec![u32::MAX; arr_size];
    for (i, comp) in cdt.components.iter().enumerate() {
        // Suppressed components (Sd): the picture-level wavelet array is
        // skipped entirely. Their data lives in the Sd tail bands of the
        // weights_by_band / per-precinct band layout instead.
        if (i as u32) >= n_decomposed {
            continue;
        }
        let wc = wf / (comp.sx as u32);
        let hc = hf / (comp.sy as u32);
        let nly_i = nly_per_component[i];
        for beta_pic in 0..nbeta {
            let idx = i * (nbeta as usize) + beta_pic as usize;
            let Some(local_beta) = picture_beta_to_local_beta(beta_pic, nlx, nly, comp.sy) else {
                // Annex B.4 bx[β,i] = 0 — slot has no band for this
                // component. Leave exists = false.
                exists_arr[idx] = false;
                continue;
            };
            // (dx, dy, τx, τy) of the chroma's actual band — taken from
            // chroma's own DWT enumeration NL,x / N'L,y[i].
            let (dx, dy, tx, ty) = beta_levels(local_beta, nlx, nly_i);
            dx_arr[idx] = dx;
            dy_arr[idx] = dy;
            tau_x[idx] = tx;
            tau_y[idx] = ty;
            local_beta_arr[idx] = local_beta;
            // Band geometry per Annex B.2 — uses chroma's plane dims (wc, hc).
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
            exists_arr[idx] = true;
        }
    }

    // Precinct grid (Annex B.5):
    //   Cs = 8 × Cw × max(sx[i]) × 2^NL,x        if Cw > 0
    //      = Wf                                   otherwise
    //   Np_x = ⌈Wf / Cs⌉, Np_y = ⌈Hf / 2^NL,y⌉
    let cs: u32 = if pih.cw == 0 {
        wf
    } else {
        let max_sx = sx.iter().copied().max().unwrap_or(1) as u32;
        let pow = 1u32 << nlx;
        8u32 * (pih.cw as u32) * max_sx * pow
    };
    if cs == 0 {
        return Err(Error::invalid(
            "jpegxs walker: derived Cs == 0 (check Cw / NL,x / sx)".to_string(),
        ));
    }
    // Spec Note 1: all but the rightmost precincts must contain at least
    // 8 samples of the LL band — i.e. Wf umod Cs <= Wf is automatic, but
    // we still require the rightmost Wp[p] > 0.
    let np_x: u32 = wf.div_ceil(cs);
    // Hp = 2^NL,y. NL,y == 0 is a degenerate case where every component
    // line forms its own precinct. The spec writes Np_y = ⌈Hf / 2^NL,y⌉.
    let hp_pow = if nly == 0 { 1u32 } else { 1u32 << nly };
    let hp = hp_pow;
    let np_y = hf.div_ceil(hp_pow);

    // Per-band gain/priority from WGT. Annex A.4.11 Table A.24 lists
    // a (G[b], P[b]) pair only for existing bands (`if (b'x[b])`); we
    // therefore feed `parse_wgt` the count of existing bands. The Sd
    // tail bands always exist (suppressed components carry one β=0
    // band per component, sx[i]=sy[i]=1).
    let n_existing_wavelet: usize = exists_arr.iter().filter(|e| **e).count();
    let n_existing: usize = n_existing_wavelet + (sd_u as usize);
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
            for i in 0..n_decomposed as usize {
                let idx = i * (nbeta as usize) + beta as usize;
                if !exists_arr[idx] {
                    continue;
                }
                let b = (n_decomposed * beta + i as u32) as usize;
                weights_by_band[b] = weights_existing[wgt_cursor];
                wgt_cursor += 1;
            }
        }
        // Sd tail bands: one per suppressed component, β = 0.
        for i in 0..sd_u as usize {
            let b = (n_decomposed * nbeta + i as u32) as usize;
            weights_by_band[b] = weights_existing[wgt_cursor];
            wgt_cursor += 1;
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
                n_decomposed,
                &sx,
                &sy,
                &nly_per_component,
                cs,
                hp,
                np_x,
                hf,
                &dx_arr,
                &dy_arr,
                &tau_x,
                &tau_y,
                &exists_arr,
                &local_beta_arr,
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
            np_x,
            np_y,
            cs,
            sd,
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
    n_decomposed: u32,
    sx: &[u8],
    sy: &[u8],
    nly_per_component: &[u8],
    cs: u32,
    hp: u32,
    np_x: u32,
    hf: u32,
    dx: &[u32],
    dy: &[u32],
    tau_x: &[bool],
    tau_y: &[bool],
    exists_arr: &[bool],
    local_beta: &[u32],
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

    let sd_u = nc - n_decomposed;
    let n_bands = n_decomposed * nbeta + sd_u;
    let mut bands: Vec<BandGeometry> = Vec::with_capacity(n_bands as usize);
    let mut band_component: Vec<u8> = Vec::with_capacity(n_bands as usize);
    let mut band_beta: Vec<u32> = Vec::with_capacity(n_bands as usize);
    let mut band_local_beta: Vec<u32> = Vec::with_capacity(n_bands as usize);
    // Fill per-band geometry in band-id order: b = (Nc - Sd) * β_pic + i
    // for i ∈ [0, Nc - Sd). `β` here is the *picture-level* β slot as
    // used by the bitstream; the (dx, dy, τx, τy) read from dx/dy/
    // tau_x/tau_y arrays are the **component's local** values (chroma's
    // own decomposition at NL,x / N'L,y[i]). Sd suppressed bands are
    // appended afterward.
    for beta in 0..nbeta {
        for i in 0..n_decomposed as usize {
            let arr_idx = i * (nbeta as usize) + beta as usize;
            let dx_b = dx[arr_idx];
            let dy_b = dy[arr_idx];
            let tx_b = tau_x[arr_idx];
            let tau_y_b = if tau_y[arr_idx] { 1u32 } else { 0u32 };
            let exists = exists_arr[arr_idx];
            let local_b = local_beta[arr_idx];

            // Per-band Wpb: matches the picture-level band width Wb[β,i]
            // when Cw == 0 (single precinct column). For τx = false the
            // band has ⌈Wc / 2^dx⌉ coefficients per row; for τx = true
            // the band has ⌈Wc / 2^(dx-1)⌉ / 2 coefficients per row.
            // Both dx and τx come from the *component's local* β
            // enumeration so the formula matches each component's own
            // DWT band dimensions.
            let wpb = if exists {
                let wc_p = (wp / sx[i] as u32).max(1);
                if !tx_b {
                    if dx_b == 0 {
                        wc_p
                    } else {
                        (wc_p + (1u32 << dx_b) - 1) >> dx_b
                    }
                } else {
                    let denom_minus1 = if dx_b == 0 { 1 } else { 1u32 << (dx_b - 1) };
                    wc_p.div_ceil(denom_minus1) / 2
                }
            } else {
                0
            };

            // L0[p,b] = 2^max(N'L,y[i] - dy[i,β], 0) × τy[β] (band-grid
            // for component i). The image-grid equivalent (Annex B.6) is
            // L0 × sy[i]. We carry the band-grid value in the geometry
            // because the entropy decoder indexes by band-grid lines.
            let nly_i = nly_per_component[i] as u32;
            let dy_eff = if nly == 0 { 0 } else { dy_b };
            let pow = if dy_eff > nly_i || nly_i == 0 {
                1u32
            } else {
                1u32 << (nly_i - dy_eff)
            };
            let l0 = pow * tau_y_b;

            // L1 (band-grid): L1 = L0 + min(Hb − py·pow, pow). Hb is the
            // per-component picture-level band height (chroma's own DWT
            // produces a band Hb_chroma rows tall).
            let row_offset = py * pow;
            let band_h_remaining = (hb[arr_idx]).saturating_sub(row_offset);
            let l1_extent = band_h_remaining.min(pow);
            let l1 = l0 + l1_extent;

            let weight = if exists {
                let b = (n_decomposed * beta + i as u32) as usize;
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
            band_local_beta.push(local_b);
        }
    }

    // Sd tail bands: one per suppressed component (i ∈ [Nc-Sd, Nc)).
    // Filter type β = 0, sx[i] = sy[i] = 1 enforced upstream. The band
    // is the raw component samples — picture width Wf, precinct height
    // Hp = 2^NL,y rows (clamped to Hf at the bottom). The per-precinct
    // band width matches Wp[p] (the standard precinct width formula).
    // L0/L1 enumerate the precinct's line range within the picture.
    let hp_pic_rows = if nly == 0 { 1u32 } else { 1u32 << nly };
    let row_offset_pic = py * hp_pic_rows;
    let lines_this_precinct = hp_pic_rows.min((pih.hf as u32).saturating_sub(row_offset_pic));
    for i in 0..sd_u {
        let comp_idx = n_decomposed + i;
        // Per Annex A.4.7, the suppressed component has sx=sy=1, so its
        // per-precinct band footprint is exactly Wp[p] columns × `lines_this_precinct` rows.
        let wpb = wp; // sx[i] = 1 implies Wpb = Wp[p].
        let b_tail = (n_decomposed * nbeta + i) as usize;
        let weight = weights_by_band[b_tail];
        bands.push(BandGeometry {
            wpb,
            gain: weight.gain,
            priority: weight.priority,
            l0: 0u16,
            l1: lines_this_precinct as u16,
            exists: lines_this_precinct > 0,
        });
        band_component.push(comp_idx as u8);
        band_beta.push(0);
        band_local_beta.push(0);
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

    let packets = compute_packet_layouts(
        nlx,
        nly,
        nc,
        n_decomposed,
        &bands,
        dy,
        &band_component,
        sy,
        lines_this_precinct,
    );

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
        band_local_beta,
    })
}

/// Annex B.7 / Table B.4 — compute `I[p,b,λ,s]` and `Npc[p]` for one
/// precinct's bands. Returns one [`PacketLayout`] per packet `s`.
///
/// Round-5 multi-component handling: bands are interleaved by component
/// per the spec band-id rule `b = (Nc - Sd) × β + i`. The first packet
/// covers β1 bands × (Nc - Sd) components on line 0; subsequent packets group
/// 3 βs × (Nc - Sd) components on each line of the proxy level.
///
/// Round-9 (r91): when `Sd > 0`, the suppressed components are appended
/// at the end (band ids `(Nc-Sd)·Nβ .. (Nc-Sd)·Nβ + Sd`). Per the
/// "tail loop" of Table B.4, each line of each suppressed component
/// gets its own packet, walked with the component index as the fast
/// variable and the line as the slow variable.
#[allow(clippy::too_many_arguments)]
fn compute_packet_layouts(
    nlx: u8,
    nly: u8,
    nc: u32,
    n_decomposed: u32,
    bands: &[BandGeometry],
    dy: &[u32],
    band_component: &[u8],
    sy: &[u8],
    lines_this_precinct: u32,
) -> Vec<PacketLayout> {
    let mut layouts: Vec<Vec<PacketEntry>> = Vec::new();

    // Step 1 — first packet: β1 = max(NL,x, NL,y) − min(NL,x, NL,y) + 1
    // bands × Nc-Sd wavelet components, all on line λ = 0.
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let beta1 = nlx_u.max(nly_u) - nlx_u.min(nly_u) + 1;
    let mut first_pkt: Vec<PacketEntry> = Vec::new();
    for beta in 0..beta1 {
        for i in 0..n_decomposed {
            let b = (n_decomposed * beta + i) as usize;
            if b < bands.len() && bands[b].exists {
                let sy_i = sy[i as usize] as u32;
                let sy_i_safe = sy_i.max(1);
                // Subsampling guard from Table B.4 with λ=0 and image-
                // grid L0 (= band-grid L0 × sy[i]):
                //   (0 + L0_image) umod sy[i] == 0
                let l0_band = bands[b].l0 as u32;
                let l0_image = l0_band * sy_i_safe;
                if sy_i != 0 && (l0_image % sy_i) != 0 {
                    continue;
                }
                first_pkt.push(PacketEntry {
                    band: b as u16,
                    line: l0_band as u16,
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
    //       for i = 0 .. Nc-Sd-1:
    //         if exists && (λ + L0[p,b]) umod sy[i] == 0:
    //           start a new packet (r = 1) per band per component
    let nbeta_u = if n_decomposed > 0 {
        // Wavelet bands occupy `n_decomposed * nβ` slots; the Sd tail
        // adds `Sd` more. Deriving Nβ from the wavelet sub-array.
        let sd_u = nc - n_decomposed;
        ((bands.len() as u32) - sd_u) / n_decomposed
    } else {
        0
    };
    let mut beta0 = beta1;
    while beta0 < nbeta_u {
        // Per Annex B.7 Table B.4 proxy-level outer loop:
        //   for(λ=0; λ < 2^(NL,y - dy[0,β0]); λ=λ+1)
        // λ runs over IMAGE-grid lines using the FIRST (luma) component's
        // dy[0,β0]. dy stored in this walker is indexed `i * nbeta + β`,
        // so luma (i=0) is at `arr_idx0 = β0`.
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
                for i in 0..n_decomposed {
                    let b = (n_decomposed * beta + i) as usize;
                    if b >= bands.len() || !bands[b].exists {
                        continue;
                    }
                    let sy_i = sy[i as usize] as u32;
                    let sy_i_safe = sy_i.max(1);
                    // L0[p,b] and L1[p,b] are stored in per-component
                    // band-grid units (component-effective N'L,y[i]).
                    // Spec B.6 defines L0 in image-grid units with the
                    // PICTURE-level NL,y; the band-grid value scales by
                    // 1 / sy[i]. So the spec image-grid L0 / L1 are:
                    let l0_band = bands[b].l0 as u32;
                    let l1_band = bands[b].l1 as u32;
                    let l0_image = l0_band * sy_i_safe;
                    let l1_image = l1_band * sy_i_safe;
                    let line_image = l0_image + lambda_within;
                    // Spec B.7 Table B.4 line-in-precinct check, image-grid:
                    if line_image >= l1_image {
                        continue;
                    }
                    // Spec B.7 Table B.4 sub-sampling guard, image-grid:
                    //   (λ + L0[p,b]) umod sy[i] == 0
                    if sy_i != 0 && (line_image % sy_i) != 0 {
                        continue;
                    }
                    // Convert back to band-grid for the entry.line field
                    // so `entry.line - band.l0` yields a band-grid row
                    // index when consumed in entropy / packet_body.
                    let line_band = line_image / sy_i_safe;
                    layouts.push(vec![PacketEntry {
                        band: b as u16,
                        line: line_band as u16,
                    }]);
                }
            }
        }
        beta0 += 3;
    }

    // Step 3 — Sd tail: one packet per (line λ, suppressed component i),
    // with component as the fast and line as the slow variable per Annex
    // B.7 Table B.4 NOTE / final loop.
    let sd_u = nc - n_decomposed;
    if sd_u > 0 {
        let sd_band_base = n_decomposed * nbeta_u;
        for lambda in 0..lines_this_precinct {
            for i in 0..sd_u {
                let b = (sd_band_base + i) as usize;
                if b >= bands.len() || !bands[b].exists {
                    continue;
                }
                let l1 = bands[b].l1 as u32;
                if lambda >= l1 {
                    continue;
                }
                layouts.push(vec![PacketEntry {
                    band: b as u16,
                    line: lambda as u16,
                }]);
            }
        }
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
    fn picture_beta_to_local_beta_444() {
        // 4:4:4 (sy=1) — identity mapping for any NL.
        for nlx in 1u8..=5 {
            for nly in 0u8..=nlx {
                for beta_pic in 0..n_beta(nlx, nly) {
                    assert_eq!(
                        picture_beta_to_local_beta(beta_pic, nlx, nly, 1),
                        Some(beta_pic),
                        "4:4:4 identity broken at NL={nlx}/{nly}, β={beta_pic}"
                    );
                }
            }
        }
    }

    #[test]
    fn picture_beta_to_local_beta_420_nl5_2_figure_b2() {
        // Annex B.3 Figure B.2 worked example: NL,x=5, NL,y=2, 4:2:0.
        // Chroma (Cb / Cr) appears at band b=1,4,7,10,13,22,25,28
        // (i.e. picture-β = 0,1,2,3,4,7,8,9; gap at β=5,6) per the
        // shaded cells in the figure.
        let cases = [
            (0u32, Some(0u32)),
            (1, Some(1)),
            (2, Some(2)),
            (3, Some(3)),
            (4, Some(4)),
            (5, None),
            (6, None),
            (7, Some(5)),
            (8, Some(6)),
            (9, Some(7)),
        ];
        for (bp, want) in cases {
            assert_eq!(
                picture_beta_to_local_beta(bp, 5, 2, 2),
                want,
                "NL=5/2 4:2:0 picture-β={bp}"
            );
        }
    }

    #[test]
    fn picture_beta_to_local_beta_420_nl3() {
        // NL=3/3, sy=2 (4:2:0 vertical). Per Table B.3-equivalent at
        // (NL,x=3, NL,y=3) the picture β enumeration is:
        //   0: LL3,3  1: HL3,3  2: LH3,3  3: HH3,3
        //   4: HL2,2  5: LH2,2  6: HH2,2
        //   7: HL1,1  8: LH1,1  9: HH1,1
        // For chroma (sy=2, N'L,y=2) the LH3/HH3 slots (β=2,3) have
        // L0 = 2^(3-3)·1 = 1 image-grid line → 1 mod 2 != 0 → bx = 0.
        // The remaining picture slots map to chroma-local β as:
        //   0 → 0 (LL)
        //   1 → 1 (HL3)
        //   2 → None (LH3 — vertically incompatible with sy=2)
        //   3 → None (HH3)
        //   4 → 2 (HL2)
        //   5 → 3 (LH2)
        //   6 → 4 (HH2)
        //   7 → 5 (HL1)
        //   8 → 6 (LH1)
        //   9 → 7 (HH1)
        let cases = [
            (0u32, Some(0u32)),
            (1, Some(1)),
            (2, None),
            (3, None),
            (4, Some(2)),
            (5, Some(3)),
            (6, Some(4)),
            (7, Some(5)),
            (8, Some(6)),
            (9, Some(7)),
        ];
        for (bp, want) in cases {
            assert_eq!(
                picture_beta_to_local_beta(bp, 3, 3, 2),
                want,
                "NL=3/3 4:2:0 picture-β={bp}"
            );
        }
    }

    #[test]
    fn picture_beta_to_local_beta_420_nl2() {
        // NL=2/2, sy=2 (4:2:0). Chroma N'L,y = 1, nbeta_chroma = 5.
        // Picture β enumeration at NL=2/2:
        //   0: LL2,2  1: HL2,2  2: LH2,2  3: HH2,2  4: HL1,1  5: LH1,1  6: HH1,1
        // Chroma at NL=2/1: 0:LL  1:HL2  2:HL1,1  3:LH1,1  4:HH1,1
        // bx for picture-β at sy=2: β=2 LH2,2 dy=2 ty=1 → l0=1, skip; β=3 same.
        // β=4 HL1,1 ty=0 → exists; β=5 LH1,1 dy=1 ty=1 → l0=2, exists; β=6 same.
        let cases = [
            (0u32, Some(0u32)),
            (1, Some(1)),
            (2, None),
            (3, None),
            (4, Some(2)),
            (5, Some(3)),
            (6, Some(4)),
        ];
        for (bp, want) in cases {
            assert_eq!(
                picture_beta_to_local_beta(bp, 2, 2, 2),
                want,
                "NL=2/2 4:2:0 picture-β={bp}"
            );
        }
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

    /// Round-8: Cw > 0 split a 32-wide picture into 2 precincts per row.
    /// Cw=1, NL,x=1, max(sx)=1 → Cs = 8 × 1 × 1 × 2 = 16, Np,x = 2.
    #[test]
    fn build_plan_cw_gt_zero_64x4_luma_nl_1_1() {
        let mut pih = pih_min(1, 1, 32, 4);
        pih.cw = 1;
        let cdt = cdt_one(8);
        // n_existing = 4 bands × 1 component = 4 → WGT 8 bytes.
        let wgt = vec![0u8; 8];
        let (plan, _) = build_plan(&pih, &cdt, &wgt).expect("Cw=1 plan");
        assert_eq!(plan.np_x, 2);
        assert_eq!(plan.cs, 16);
        // Np,y = ⌈4 / 2⌉ = 2 → total 4 precincts.
        assert_eq!(plan.np_y, 2);
        assert_eq!(
            plan.slices.iter().map(|s| s.precincts.len()).sum::<usize>(),
            4
        );
        // First precinct band 0 (LL): wpb = ⌈(Cs/sx) / 2^dx⌉ = ⌈16/2⌉ = 8.
        let p0 = &plan.slices[0].precincts[0];
        assert_eq!(p0.geometry.bands[0].wpb, 8);
        assert_eq!(p0.wp, 16);
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

    /// Round 9 (r91): Sd=1 with Nc=4 at NL=1/1, 4×4 picture. Annex
    /// A.4.7: NL = (Nc-Sd) × Nβ + Sd = 3*4 + 1 = 13. The 13th band is
    /// the suppressed component's raw single-band.
    #[test]
    fn build_plan_sd1_4comp_4x4_nl_1_1() {
        let mut pih = pih_min(1, 1, 4, 4);
        pih.nc = 4;
        let cdt = ComponentTable {
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
                Component {
                    bit_depth: 8,
                    sx: 1,
                    sy: 1,
                },
            ],
        };
        // n_existing = 3 wavelet comps × 4 bands + 1 Sd tail = 13.
        let wgt = vec![0u8; 13 * 2];
        let (plan, weights) = build_plan_sd(&pih, &cdt, &wgt, 1).expect("Sd=1 4-comp plan");
        assert_eq!(plan.n_bands, 13);
        assert_eq!(plan.n_decomposed, 3);
        assert_eq!(plan.sd, 1);
        assert_eq!(weights.len(), 13);
        let p0 = &plan.slices[0].precincts[0];
        // Wavelet bands: 12 (β = 0..3 × 3 components).
        // Sd tail band: 1 at index 12, component 3, β=0.
        assert_eq!(p0.geometry.bands.len(), 13);
        assert_eq!(p0.band_component[12], 3);
        assert_eq!(p0.band_beta[12], 0);
        // Sd-tail band has the full per-precinct width.
        assert_eq!(p0.geometry.bands[12].wpb, plan.cs);
    }

    /// Round 9: walker rejects suppressed components with sub-sampling.
    #[test]
    fn build_plan_sd_rejects_subsampled_tail() {
        let mut pih = pih_min(1, 1, 4, 4);
        pih.nc = 4;
        let cdt = ComponentTable {
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
                Component {
                    bit_depth: 8,
                    sx: 2,
                    sy: 1,
                },
            ],
        };
        let wgt = vec![0u8; 13 * 2];
        let err = build_plan_sd(&pih, &cdt, &wgt, 1)
            .expect_err("Sd suppressed component with sx=2 must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("Sd") || msg.contains("sx=sy=1"),
            "unexpected error: {msg}"
        );
    }
}
