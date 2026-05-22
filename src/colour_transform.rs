//! Inverse multi-component (colour) transformations — ISO/IEC 21122-1:
//! 2022 Annex F.
//!
//! Two transforms are normative:
//!
//! * [`inverse_rct`] — Annex F.3, Table F.2. Cpih == 1. Three-component
//!   reversible colour transform that maps the encoder-side
//!   (Y, Cb, Cr) decorrelation back to (R, G, B). The forward direction
//!   (Annex F.4 / Table F.3) is `Y = (R + 2G + B) >> 2`,
//!   `Cb = B - G`, `Cr = R - G`; the inverse reconstructs
//!   `G = Y - ((Cb + Cr) >> 2)`, `R = G + Cr`, `B = G + Cb`.
//!
//!   Component layout per Table F.2: input component 0 is luma (Y),
//!   1 is Cb, 2 is Cr; output component 0 is red, 1 is green, 2 is
//!   blue. Components ≥ 3 are passed through unchanged (Annex F.2,
//!   Table F.1: "Set Ω[c,x,y] = O[c,x,y] for all components c≥3").
//!
//! * [`inverse_star_tetrix`] — Annex F.5, Tables F.4 / F.5 / F.6 / F.7
//!   / F.8. Cpih == 3. Four-component CFA-pixel decoder that combines
//!   inverse average, delta, Y, and CbCr lifting steps. The chroma
//!   weighting exponents `e1` and `e2` come from the CTS marker
//!   (Annex A.4.8), and the per-component sub-pixel displacement
//!   vector `(δx, δy)` from the CFA-pattern type `Ct` derived from CRG
//!   (Annex A.4.9, Tables F.9 / F.10).
//!
//! Both transforms operate **post-DWT, pre-output-mapping**: their
//! input is the int32 sample plane the inverse DWT produced for each
//! component (one element per (x, y) on the sample grid `Wf × Hf`),
//! and their output overwrites the same buffers in place. The decoder
//! pipeline runs Annex F → Annex G in that order (Table A.1 puts the
//! NLT step after the colour transform).
//!
//! Allocation. RCT and Star-Tetrix both run in-place on the caller-
//! owned per-component buffers — no heap traffic per transform call.

use crate::error::{JpegXsError as Error, Result};

/// Apply the inverse reversible colour transform (Cpih == 1) to the
/// first three component planes in place.
///
/// `planes` is a slice of mutable per-component plane buffers, each
/// laid out row-major `wf × hf`. Components beyond index 2 are not
/// touched.
///
/// Per Annex F.2: Cpih == 1 requires `Nc >= 3` and `sx[i] == sy[i] == 1`
/// for `i < 3`. The caller must enforce that — `inverse_rct` only checks
/// the buffer count and length here.
pub fn inverse_rct(planes: &mut [&mut [i32]], wf: usize, hf: usize) -> Result<()> {
    if planes.len() < 3 {
        return Err(Error::invalid(format!(
            "jpegxs Cpih=1 (RCT) requires at least 3 component planes, got {}",
            planes.len()
        )));
    }
    let want = wf
        .checked_mul(hf)
        .ok_or_else(|| Error::invalid("jpegxs RCT: wf * hf overflow"))?;
    for (i, p) in planes.iter().enumerate().take(3) {
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs RCT: component {i} has {} samples, expected {want}",
                p.len()
            )));
        }
    }

    // Annex F.3 Table F.2:
    //   o1 = i0 - ((i1 + i2) >> 2)   // green
    //   o0 = o1 + i2                  // red
    //   o2 = o1 + i1                  // blue
    //
    // i0 = O[0] (luma), i1 = O[1] (Cb), i2 = O[2] (Cr).
    // Output Ω[0] = red, Ω[1] = green, Ω[2] = blue.
    //
    // We split the slice borrows so we can write components 0/1/2 in
    // turn without aliasing (each loop iteration touches one (x, y)
    // across all three planes).
    let (p0_p1, p2_rest) = planes.split_at_mut(2);
    let (p0_slot, p1_slot) = p0_p1.split_at_mut(1);
    let p0 = &mut **p0_slot.get_mut(0).expect("checked above");
    let p1 = &mut **p1_slot.get_mut(0).expect("checked above");
    let p2 = &mut **p2_rest.get_mut(0).expect("checked above");

    for x in 0..want {
        let i0 = p0[x];
        let i1 = p1[x];
        let i2 = p2[x];
        let o1 = i0 - ((i1 + i2) >> 2);
        let o0 = o1 + i2;
        let o2 = o1 + i1;
        p0[x] = o0;
        p1[x] = o1;
        p2[x] = o2;
    }
    Ok(())
}

/// Apply the **forward** reversible colour transform (Cpih == 1) — the
/// inverse of [`inverse_rct`]. Maps three RGB component planes in place
/// to (Y, Cb, Cr) per Annex F.4 Table F.3:
///
/// * `Y  = (R + 2G + B) >> 2`
/// * `Cb = B - G`
/// * `Cr = R - G`
///
/// Component layout on input (matching [`inverse_rct`]'s output): plane
/// 0 = R, plane 1 = G, plane 2 = B. On exit: plane 0 = Y, plane 1 = Cb,
/// plane 2 = Cr. Components ≥ 3 are untouched. The transform is exactly
/// invertible for any signed 32-bit input and is the encoder pre-DWT
/// step for `Cpih == 1`.
pub fn forward_rct(planes: &mut [&mut [i32]], wf: usize, hf: usize) -> Result<()> {
    if planes.len() < 3 {
        return Err(Error::invalid(format!(
            "jpegxs Cpih=1 (forward RCT) requires at least 3 component planes, got {}",
            planes.len()
        )));
    }
    let want = wf
        .checked_mul(hf)
        .ok_or_else(|| Error::invalid("jpegxs forward RCT: wf * hf overflow"))?;
    for (i, p) in planes.iter().enumerate().take(3) {
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs forward RCT: component {i} has {} samples, expected {want}",
                p.len()
            )));
        }
    }
    let (p0_p1, p2_rest) = planes.split_at_mut(2);
    let (p0_slot, p1_slot) = p0_p1.split_at_mut(1);
    let p0 = &mut **p0_slot.get_mut(0).expect("checked above");
    let p1 = &mut **p1_slot.get_mut(0).expect("checked above");
    let p2 = &mut **p2_rest.get_mut(0).expect("checked above");

    for x in 0..want {
        let r = p0[x];
        let g = p1[x];
        let b = p2[x];
        let y = (r + 2 * g + b) >> 2;
        let cb = b - g;
        let cr = r - g;
        p0[x] = y;
        p1[x] = cb;
        p2[x] = cr;
    }
    Ok(())
}

/// Inverse Star-Tetrix transform (Cpih == 3) — Annex F.5, Tables F.4 /
/// F.5 / F.6 / F.7 / F.8 plus the [`access`] reflection from Table
/// F.12 and the super-pixel look-up tables in Tables F.9 / F.10 /
/// F.11.
///
/// `planes` must hold 4 component buffers, each laid out row-major
/// `wf × hf`. `e1` and `e2` come from the CTS marker (§A.4.8); `ct` is
/// the CFA pattern type derived from the CRG marker per Table F.9
/// (`Ct ∈ {0, 1}`); `cf` is the CTS marker `Cf` field (0 = full,
/// 3 = restricted in-line access).
///
/// The implementation runs the four lifting steps (`inv_avg_step` →
/// `inv_delta_step` → `inv_Y_step` → `inv_CbCr_step`) in order,
/// double-buffering the intermediate band arrays `ω1`, `ω2`, `ω3`,
/// `ω4`, then assigns the final outputs `Ω[c, x, y]` per Table F.4
/// (the component re-order: Ω[0]=ω4[2] red, Ω[1]=ω4[3] G1,
/// Ω[2]=ω4[0] G2, Ω[3]=ω4[1] blue).
///
/// The lifting filters are bit-exact with the floor-divides
/// (arithmetic right shifts) the spec specifies — note that for
/// negative integers the spec's `⌊·/8⌋` (floor) differs from C's `/`
/// (truncation toward zero); we use `floor_div8` / `floor_div4` to
/// honour the spec.
pub fn inverse_star_tetrix(
    planes: &mut [&mut [i32]],
    wf: usize,
    hf: usize,
    e1: u8,
    e2: u8,
    ct: u8,
    cf: u8,
) -> Result<()> {
    // Per Annex F.2 Table F.1, Cpih=3 operates on components c<4 and
    // sets Ω[c,x,y] = O[c,x,y] for c≥4. Accept any plane count ≥ 4 so
    // CWD-suppressed trailing components (Sd>0, Annex A.4.7) ride through
    // unchanged.
    if planes.len() < 4 {
        return Err(Error::invalid(format!(
            "jpegxs Cpih=3 (Star-Tetrix) requires at least 4 component planes, got {}",
            planes.len()
        )));
    }
    if ct > 1 {
        return Err(Error::invalid(format!(
            "jpegxs Star-Tetrix: Ct={ct} reserved for ISO/IEC use (Table F.9)"
        )));
    }
    if cf != 0 && cf != 3 {
        return Err(Error::invalid(format!(
            "jpegxs Star-Tetrix: Cf={cf} reserved for ISO/IEC use (Table A.20)"
        )));
    }
    let want = wf
        .checked_mul(hf)
        .ok_or_else(|| Error::invalid("jpegxs Star-Tetrix: wf * hf overflow"))?;
    // Only the first 4 component planes participate in the transform.
    // Components ≥ 4 (e.g. CWD-suppressed tail) are left untouched by the
    // caller; their plane size may differ in principle (though in practice
    // CWD mandates sx=sy=1 on suppressed comps, matching the same size).
    for (i, p) in planes.iter().enumerate().take(4) {
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs Star-Tetrix: component {i} has {} samples, expected {want}",
                p.len()
            )));
        }
    }

    // Per-component (δx, δy) from Table F.10.
    let mut delta = [(0u8, 0u8); 4];
    for (c, slot) in delta.iter_mut().enumerate() {
        *slot = crate::crg::displacement(ct, c).ok_or_else(|| {
            Error::invalid(format!(
                "jpegxs Star-Tetrix: no displacement for c={c}, Ct={ct}"
            ))
        })?;
    }

    let st = StarTetrix {
        wf,
        hf,
        cf,
        ct,
        delta,
        e1,
        e2,
    };

    // Snapshot O[c,x,y] (the inverse-DWT output) before we mutate the
    // planes — every step reads neighbouring samples from the *previous*
    // step's array.
    let o0 = planes[0].to_vec();
    let o1 = planes[1].to_vec();
    let o2 = planes[2].to_vec();
    let o3 = planes[3].to_vec();
    let o = [o0.as_slice(), o1.as_slice(), o2.as_slice(), o3.as_slice()];

    // Step 1 — inverse average (Table F.5):
    //   ω1[0,x,y] = O[0] − ⌊(Δlt + Δrt + Δlb + Δrb) / 8⌋
    //   ω1[1..3] = O[1..3] (copy).
    let mut w1_0 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let dlt = read(&o, &st, 0, x, y, -1, -1);
            let drt = read(&o, &st, 0, x, y, 1, -1);
            let dlb = read(&o, &st, 0, x, y, -1, 1);
            let drb = read(&o, &st, 0, x, y, 1, 1);
            w1_0[y * wf + x] = o[0][y * wf + x] - floor_div(dlt + drt + dlb + drb, 8);
        }
    }
    let w1 = [w1_0.as_slice(), o[1], o[2], o[3]];

    // Step 2 — inverse delta (Table F.6):
    //   ω2[3,x,y] = ω1[3] + ⌊(Ylt + Yrt + Ylb + Yrb) / 4⌋  (reads ω1[3])
    //   ω2[0..2] = ω1[0..2] (copy).
    let mut w2_3 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let ylt = read(&w1, &st, 3, x, y, -1, -1);
            let yrt = read(&w1, &st, 3, x, y, 1, -1);
            let ylb = read(&w1, &st, 3, x, y, -1, 1);
            let yrb = read(&w1, &st, 3, x, y, 1, 1);
            w2_3[y * wf + x] = w1[3][y * wf + x] + floor_div(ylt + yrt + ylb + yrb, 4);
        }
    }
    let w2 = [w1[0], w1[1], w1[2], w2_3.as_slice()];

    // Step 3 — inverse Y step (Table F.7):
    //   ω3[0,x,y] = ω2[0] − ⌊(2^e2 (Bl+Br) + 2^e1 (Rt+Rb)) / 8⌋  (G2)
    //     where Bl = ω2[0, access(0, x, y, ±1, 0)]   [Cb neighbours of G2]
    //           Rt = ω2[0, access(0, x, y, 0, ±1)]   [Cr neighbours of G2]
    //   ω3[3,x,y] = ω2[3] − ⌊(2^e2 (Bt+Bb) + 2^e1 (Rl+Rr)) / 8⌋  (G1)
    //     where Bt = ω2[3, access(3, x, y, 0, ±1)]   [Cb neighbours of G1]
    //           Rl = ω2[3, access(3, x, y, ±1, 0)]   [Cr neighbours of G1]
    //   ω3[1] = ω2[1] (copy), ω3[2] = ω2[2] (copy).
    //
    // Note: the spec's "Read the Cb component" / "Read the Cr component"
    // notation refers to the *value* at the access position, but the
    // first component index argument to `access` selects which CFA
    // sub-pixel acts as the reference for the displacement vector;
    // the sample value read is from the same band the formula puts on
    // the LHS. This matches the Table F.5/F.6 pattern where the formula
    // reads the same component (index 0 / 3) it writes back to.
    let mut w3_0 = vec![0i32; want];
    let mut w3_3 = vec![0i32; want];
    let two_e1 = 1i32 << e1;
    let two_e2 = 1i32 << e2;
    for y in 0..hf {
        for x in 0..wf {
            let bl = read(&w2, &st, 0, x, y, -1, 0);
            let br = read(&w2, &st, 0, x, y, 1, 0);
            let rt = read(&w2, &st, 0, x, y, 0, -1);
            let rb = read(&w2, &st, 0, x, y, 0, 1);
            let num = two_e2 * (bl + br) + two_e1 * (rt + rb);
            w3_0[y * wf + x] = w2[0][y * wf + x] - floor_div(num, 8);

            let bt = read(&w2, &st, 3, x, y, 0, -1);
            let bb = read(&w2, &st, 3, x, y, 0, 1);
            let rl = read(&w2, &st, 3, x, y, -1, 0);
            let rr = read(&w2, &st, 3, x, y, 1, 0);
            let num = two_e2 * (bt + bb) + two_e1 * (rl + rr);
            w3_3[y * wf + x] = w2[3][y * wf + x] - floor_div(num, 8);
        }
    }
    let w3 = [w3_0.as_slice(), w2[1], w2[2], w3_3.as_slice()];

    // Step 4 — inverse CbCr (Table F.8):
    //   ω4[1,x,y] = ω3[1] + ⌊(Gl + Gr + Gt + Gb) / 4⌋     (B)
    //     where neighbours of Cb come from ω3, accessed via c=1
    //   ω4[2,x,y] = ω3[2] + ⌊(Gl + Gr + Gt + Gb) / 4⌋     (R)
    //     where neighbours of Cr come from ω3, accessed via c=2
    //   ω4[0] = ω3[0] (copy), ω4[3] = ω3[3] (copy).
    let mut w4_1 = vec![0i32; want];
    let mut w4_2 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let gl = read(&w3, &st, 1, x, y, -1, 0);
            let gr = read(&w3, &st, 1, x, y, 1, 0);
            let gt = read(&w3, &st, 1, x, y, 0, -1);
            let gb = read(&w3, &st, 1, x, y, 0, 1);
            w4_1[y * wf + x] = w3[1][y * wf + x] + floor_div(gl + gr + gt + gb, 4);

            let gl = read(&w3, &st, 2, x, y, -1, 0);
            let gr = read(&w3, &st, 2, x, y, 1, 0);
            let gt = read(&w3, &st, 2, x, y, 0, -1);
            let gb = read(&w3, &st, 2, x, y, 0, 1);
            w4_2[y * wf + x] = w3[2][y * wf + x] + floor_div(gl + gr + gt + gb, 4);
        }
    }
    // Component re-order per Table F.4:
    //   Ω[0,x,y] = ω4[2] (red)
    //   Ω[1,x,y] = ω4[3] (first green / G1)
    //   Ω[2,x,y] = ω4[0] (second green / G2)
    //   Ω[3,x,y] = ω4[1] (blue)
    planes[0].copy_from_slice(&w4_2);
    planes[1].copy_from_slice(&w3_3);
    planes[2].copy_from_slice(&w3_0);
    planes[3].copy_from_slice(&w4_1);
    Ok(())
}

/// Apply the **forward** Star-Tetrix transform (Cpih == 3) — the
/// inverse of [`inverse_star_tetrix`]. Maps four CFA-layout component
/// planes in place: input order is `Ω = [R, G1, G2, B]` (matching the
/// inverse's output convention from Annex F.4 Table F.4); on exit, each
/// plane holds `O[c, x, y]` (the pre-DWT band-domain coefficients) per
/// the band-order `[ω1[0]=Ya, ω1[1]=Cb, ω1[2]=Cr, ω1[3]=Δ]` after the
/// four forward lifting steps.
///
/// The forward cascade reverses the order and sign of each lift step:
///
/// 1. Re-order input to band-position layout
///    `w4 = [G2, B, R, G1]` (matches the inverse's `w4` snapshot at the
///    end of inverse step 4).
/// 2. Forward inverse-of-CbCr-step (Table F.8 inverted): subtract the
///    green-neighbour average from blue (band 1) and red (band 2).
/// 3. Forward inverse-of-Y-step (Table F.7 inverted): add the chroma
///    weighted neighbour sum back into G2 (band 0) and G1 (band 3).
/// 4. Forward inverse-of-delta-step (Table F.6 inverted): subtract the Y
///    corner-neighbour average from the delta band (band 3).
/// 5. Forward inverse-of-average-step (Table F.5 inverted): add the Δ
///    corner-neighbour average back into the average luma band (band 0).
///
/// `e1`, `e2`, `ct`, `cf` come from CTS / CRG markers (see
/// [`inverse_star_tetrix`] for the parameter semantics).
///
/// Round-4: lossless self-roundtrip with the inverse on flat-zero,
/// flat-luma, and synthetic 8×8 four-component fixtures.
pub fn forward_star_tetrix(
    planes: &mut [&mut [i32]],
    wf: usize,
    hf: usize,
    e1: u8,
    e2: u8,
    ct: u8,
    cf: u8,
) -> Result<()> {
    // Accept any plane count ≥ 4 so CWD-suppressed trailing components
    // ride through unchanged (Annex A.4.7 / Annex F.2 Table F.1 — Ω = O
    // for c ≥ 4 when Cpih=3).
    if planes.len() < 4 {
        return Err(Error::invalid(format!(
            "jpegxs Cpih=3 (forward Star-Tetrix) requires at least 4 component planes, got {}",
            planes.len()
        )));
    }
    if ct > 1 {
        return Err(Error::invalid(format!(
            "jpegxs forward Star-Tetrix: Ct={ct} reserved for ISO/IEC use (Table F.9)"
        )));
    }
    if cf != 0 && cf != 3 {
        return Err(Error::invalid(format!(
            "jpegxs forward Star-Tetrix: Cf={cf} reserved for ISO/IEC use (Table A.20)"
        )));
    }
    let want = wf
        .checked_mul(hf)
        .ok_or_else(|| Error::invalid("jpegxs forward Star-Tetrix: wf * hf overflow"))?;
    // Only the first 4 component planes participate in the transform.
    for (i, p) in planes.iter().enumerate().take(4) {
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs forward Star-Tetrix: component {i} has {} samples, expected {want}",
                p.len()
            )));
        }
    }

    // Per-component (δx, δy) from Table F.10 (matches the inverse).
    let mut delta = [(0u8, 0u8); 4];
    for (c, slot) in delta.iter_mut().enumerate() {
        *slot = crate::crg::displacement(ct, c).ok_or_else(|| {
            Error::invalid(format!(
                "jpegxs forward Star-Tetrix: no displacement for c={c}, Ct={ct}"
            ))
        })?;
    }
    let st = StarTetrix {
        wf,
        hf,
        cf,
        ct,
        delta,
        e1,
        e2,
    };

    // 1) Re-order input planes [R, G1, G2, B] into band-position layout
    //    [G2, B, R, G1] (this is `w4` at the end of the inverse cascade).
    let r_in = planes[0].to_vec();
    let g1_in = planes[1].to_vec();
    let g2_in = planes[2].to_vec();
    let b_in = planes[3].to_vec();
    let w4_0 = g2_in;
    let w4_1 = b_in;
    let w4_2 = r_in;
    let w4_3 = g1_in;
    let w4 = [
        w4_0.as_slice(),
        w4_1.as_slice(),
        w4_2.as_slice(),
        w4_3.as_slice(),
    ];

    // 2) Forward inverse-of-CbCr-step: subtract green-neighbour averages
    //    from blue (band 1) and red (band 2).
    let mut w3_1 = vec![0i32; want];
    let mut w3_2 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let gl = read(&w4, &st, 1, x, y, -1, 0);
            let gr = read(&w4, &st, 1, x, y, 1, 0);
            let gt = read(&w4, &st, 1, x, y, 0, -1);
            let gb = read(&w4, &st, 1, x, y, 0, 1);
            w3_1[y * wf + x] = w4[1][y * wf + x] - floor_div(gl + gr + gt + gb, 4);

            let gl = read(&w4, &st, 2, x, y, -1, 0);
            let gr = read(&w4, &st, 2, x, y, 1, 0);
            let gt = read(&w4, &st, 2, x, y, 0, -1);
            let gb = read(&w4, &st, 2, x, y, 0, 1);
            w3_2[y * wf + x] = w4[2][y * wf + x] - floor_div(gl + gr + gt + gb, 4);
        }
    }
    // After step 2, the bands are:
    //   w3 = [G2 (band 0), Cb (band 1), Cr (band 2), G1 (band 3)].
    let w3 = [w4[0], w3_1.as_slice(), w3_2.as_slice(), w4[3]];

    // 3) Forward inverse-of-Y-step: add the chroma weighted neighbour
    //    sum back into G2 (band 0) and G1 (band 3).
    let mut w2_0 = vec![0i32; want];
    let mut w2_3 = vec![0i32; want];
    let two_e1 = 1i32 << e1;
    let two_e2 = 1i32 << e2;
    for y in 0..hf {
        for x in 0..wf {
            let bl = read(&w3, &st, 0, x, y, -1, 0);
            let br = read(&w3, &st, 0, x, y, 1, 0);
            let rt = read(&w3, &st, 0, x, y, 0, -1);
            let rb = read(&w3, &st, 0, x, y, 0, 1);
            let num = two_e2 * (bl + br) + two_e1 * (rt + rb);
            w2_0[y * wf + x] = w3[0][y * wf + x] + floor_div(num, 8);

            let bt = read(&w3, &st, 3, x, y, 0, -1);
            let bb = read(&w3, &st, 3, x, y, 0, 1);
            let rl = read(&w3, &st, 3, x, y, -1, 0);
            let rr = read(&w3, &st, 3, x, y, 1, 0);
            let num = two_e2 * (bt + bb) + two_e1 * (rl + rr);
            w2_3[y * wf + x] = w3[3][y * wf + x] + floor_div(num, 8);
        }
    }
    // After step 3, w2 = [Ya-pre-step1 (band 0), Cb, Cr, Δ-pre-step2 (band 3)].
    let w2 = [w2_0.as_slice(), w3[1], w3[2], w2_3.as_slice()];

    // 4) Forward inverse-of-delta-step: subtract Y corner-neighbour
    //    average from the delta band (band 3).
    let mut w1_3 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let ylt = read(&w2, &st, 3, x, y, -1, -1);
            let yrt = read(&w2, &st, 3, x, y, 1, -1);
            let ylb = read(&w2, &st, 3, x, y, -1, 1);
            let yrb = read(&w2, &st, 3, x, y, 1, 1);
            w1_3[y * wf + x] = w2[3][y * wf + x] - floor_div(ylt + yrt + ylb + yrb, 4);
        }
    }
    // After step 4, w1 = [Ya, Cb, Cr, Δ (final)].
    let w1 = [w2[0], w2[1], w2[2], w1_3.as_slice()];

    // 5) Forward inverse-of-average-step: add the Δ corner-neighbour
    //    average back into the average luma band (band 0). Reads of Δ
    //    come from `w1` (band index 3 lookup via access).
    let mut o0 = vec![0i32; want];
    for y in 0..hf {
        for x in 0..wf {
            let dlt = read(&w1, &st, 0, x, y, -1, -1);
            let drt = read(&w1, &st, 0, x, y, 1, -1);
            let dlb = read(&w1, &st, 0, x, y, -1, 1);
            let drb = read(&w1, &st, 0, x, y, 1, 1);
            o0[y * wf + x] = w1[0][y * wf + x] + floor_div(dlt + drt + dlb + drb, 8);
        }
    }

    // O[0] = Ya (the average-luma band — input to the DWT).
    // O[1] = Cb (band 1, untouched after steps 1/2).
    // O[2] = Cr (band 2, untouched after steps 1/2).
    // O[3] = Δ (the differential luma band — input to the DWT).
    planes[0].copy_from_slice(&o0);
    planes[1].copy_from_slice(w1[1]);
    planes[2].copy_from_slice(w1[2]);
    planes[3].copy_from_slice(&w1_3);
    Ok(())
}

/// Per-image Star-Tetrix configuration cached for the inner `access`
/// helper so the lifting kernels only have to thread one struct.
struct StarTetrix {
    wf: usize,
    hf: usize,
    cf: u8,
    ct: u8,
    delta: [(u8, u8); 4],
    #[allow(dead_code)]
    e1: u8,
    #[allow(dead_code)]
    e2: u8,
}

/// Annex F.5.7 Table F.12 — `access(c, x, y, rx, ry)` returns the
/// triple `(c', x', y')` of the sub-pixel offset `(rx, ry)` from
/// component `c` at sample position `(x, y)`.
fn access(
    c: usize,
    x: usize,
    y: usize,
    rx: i32,
    ry: i32,
    st: &StarTetrix,
) -> (usize, usize, usize) {
    let dx = st.delta[c].0 as i32;
    let dy = st.delta[c].1 as i32;
    let mut rx = rx;
    let mut ry = ry;
    let two_wf = 2 * st.wf as i32;
    let two_hf = 2 * st.hf as i32;
    let two_x = 2 * x as i32;
    let two_y = 2 * y as i32;

    // Horizontal reflection guard (always-on).
    if two_x + rx + dx < 0 || two_x + rx + dx >= two_wf {
        rx = -rx;
    }
    // Vertical reflection guard. Two flavours:
    //   * Cf == 3 — restrict to the same line (reflect any out-of-line
    //     access back into the line).
    //   * Otherwise — only reflect if going past the sample grid edges.
    let in_line_violation = st.cf == 3 && (ry + dy < 0 || ry + dy > 1);
    let grid_violation = two_y + ry + dy < 0 || two_y + ry + dy >= two_hf;
    if in_line_violation || grid_violation {
        ry = -ry;
    }

    let x_out = ((two_x + rx + dx) / 2) as usize;
    let y_out = ((two_y + ry + dy) / 2) as usize;
    let qx = (rx + dx).rem_euclid(2) as u8;
    let qy = (ry + dy).rem_euclid(2) as u8;
    let c_out = crate::crg::component_at(st.ct, qx, qy)
        .expect("k[δx,δy] defined for all 0/1 displacements under Ct ∈ {0,1}");
    (c_out as usize, x_out, y_out)
}

/// Read the sample at `access(c, x, y, rx, ry)` from the per-component
/// plane array `planes`.
fn read(
    planes: &[&[i32]; 4],
    st: &StarTetrix,
    c: usize,
    x: usize,
    y: usize,
    rx: i32,
    ry: i32,
) -> i32 {
    let (c_out, x_out, y_out) = access(c, x, y, rx, ry, st);
    planes[c_out][y_out * st.wf + x_out]
}

/// Floor-division of `n` by a positive integer `d`. Differs from
/// Rust's `n / d` only for negative `n`, where `n / d` truncates toward
/// zero but the spec's `⌊n/d⌋` rounds toward `-∞`. The Star-Tetrix
/// formulas use the floor convention.
fn floor_div(n: i32, d: i32) -> i32 {
    debug_assert!(d > 0);
    let q = n / d;
    if (n % d) != 0 && ((n < 0) != (d < 0)) {
        q - 1
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward RCT companion (Annex F.4, Table F.3) used only by tests
    /// to round-trip the inverse. `o0 = (R + 2G + B) >> 2`,
    /// `o1 = B - G`, `o2 = R - G`.
    fn forward_rct(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        let o0 = (r + 2 * g + b) >> 2;
        let o1 = b - g;
        let o2 = r - g;
        (o0, o1, o2)
    }

    #[test]
    fn rct_round_trips_constant_pixel() {
        // RGB (100, 200, 50). Forward → (Y, Cb, Cr).
        let (y, cb, cr) = forward_rct(100, 200, 50);
        let mut p0 = vec![y; 4];
        let mut p1 = vec![cb; 4];
        let mut p2 = vec![cr; 4];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        inverse_rct(&mut planes, 2, 2).unwrap();
        // Output order is R, G, B per Annex F.3 (the wires assignment in
        // Table F.2 puts red on Ω[0]).
        assert_eq!(p0, vec![100, 100, 100, 100]);
        assert_eq!(p1, vec![200, 200, 200, 200]);
        assert_eq!(p2, vec![50, 50, 50, 50]);
    }

    #[test]
    fn rct_round_trips_arbitrary_pixels() {
        let pix = [
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 200),
        ];
        let mut p0: Vec<i32> = Vec::new();
        let mut p1: Vec<i32> = Vec::new();
        let mut p2: Vec<i32> = Vec::new();
        for &(r, g, b) in &pix {
            let (y, cb, cr) = forward_rct(r, g, b);
            p0.push(y);
            p1.push(cb);
            p2.push(cr);
        }
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        inverse_rct(&mut planes, pix.len(), 1).unwrap();
        for (i, &(r, g, b)) in pix.iter().enumerate() {
            assert_eq!((p0[i], p1[i], p2[i]), (r, g, b), "pixel {i}");
        }
    }

    #[test]
    fn rct_rejects_wrong_buffer_size() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 3];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        assert!(inverse_rct(&mut planes, 2, 2).is_err());
    }

    #[test]
    fn rct_rejects_too_few_components() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 2] = [&mut p0, &mut p1];
        assert!(inverse_rct(&mut planes, 2, 2).is_err());
    }

    #[test]
    fn star_tetrix_rejects_wrong_component_count() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        let err = inverse_star_tetrix(&mut planes, 2, 2, 0, 0, 0, 0).unwrap_err();
        assert!(format!("{err}").contains("4 component"));
    }

    #[test]
    fn star_tetrix_rejects_reserved_ct() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 4];
        let mut p3 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        let err = inverse_star_tetrix(&mut planes, 2, 2, 0, 0, 2, 0).unwrap_err();
        assert!(format!("{err}").contains("Ct=2"));
    }

    #[test]
    fn star_tetrix_rejects_reserved_cf() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 4];
        let mut p3 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        let err = inverse_star_tetrix(&mut planes, 2, 2, 0, 0, 0, 1).unwrap_err();
        assert!(format!("{err}").contains("Cf="));
    }

    /// All-zero input → all-zero output. Trivial sanity check that the
    /// four lifting steps each preserve a flat-zero CFA pattern.
    #[test]
    fn star_tetrix_flat_zero_in_zero_out() {
        let n = 4 * 4;
        let mut p0 = vec![0i32; n];
        let mut p1 = vec![0i32; n];
        let mut p2 = vec![0i32; n];
        let mut p3 = vec![0i32; n];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        inverse_star_tetrix(&mut planes, 4, 4, 0, 0, 0, 0).unwrap();
        for &v in p0.iter().chain(&p1).chain(&p2).chain(&p3) {
            assert_eq!(v, 0, "flat zero must round-trip");
        }
    }

    /// All-zero input should also round-trip with `Cf=3` (in-line
    /// access) and the alternate CFA pattern type.
    #[test]
    fn star_tetrix_zero_inline_ct1() {
        let n = 4 * 4;
        let mut p0 = vec![0i32; n];
        let mut p1 = vec![0i32; n];
        let mut p2 = vec![0i32; n];
        let mut p3 = vec![0i32; n];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        inverse_star_tetrix(&mut planes, 4, 4, 0, 0, 1, 3).unwrap();
        for &v in p0.iter().chain(&p1).chain(&p2).chain(&p3) {
            assert_eq!(v, 0);
        }
    }

    /// A flat-but-non-zero input where component 0 is the average luma
    /// `Ya`, component 3 is the differential luma `Δ`, and Cb/Cr are
    /// zero. With Δ=0 the inverse-average step is a no-op, and with
    /// Cb=Cr=0 the inverse-Y/CbCr steps don't add anything either, so
    /// the average luma should propagate to G1 and G2 (Ω[1] and Ω[2]).
    #[test]
    fn star_tetrix_flat_luma_propagates_to_green() {
        let n = 4 * 4;
        // O[0] = Ya = 50 everywhere. O[1] = O[2] = O[3] = 0.
        let mut p0 = vec![50i32; n];
        let mut p1 = vec![0i32; n];
        let mut p2 = vec![0i32; n];
        let mut p3 = vec![0i32; n];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        inverse_star_tetrix(&mut planes, 4, 4, 0, 0, 0, 0).unwrap();
        // After the four steps, G2 = ω4[0] (which inherits Ya through
        // the avg step) → Ω[2] = 50. G1 = ω4[3] (which inherits the
        // delta-step result) → Ω[1] = 50. Red and Blue come from the
        // CbCr step which adds the green-neighbour average; with green
        // = 50 everywhere, the +⌊(50+50+50+50)/4⌋ = +50 lift gives
        // Ω[0] = 50 + 50 = 50? Let's compute carefully:
        //   ω4[2] = ω3[2] + ⌊(Gl + Gr + Gt + Gb) / 4⌋
        //   ω3[2] = ω2[2] = ω1[2] = O[2] = 0.
        //   The G value at access(2, x, y, ...) is taken from ω3[g],
        //   which is component 0 (G2) for displacement (0,0) and
        //   component 3 (G1) for displacement (1,0) under Ct=0.
        //   In any case all four neighbours are 50 → ⌊200/4⌋ = 50.
        //   So Ω[0] = 0 + 50 = 50.  Likewise Ω[3] = 0 + 50 = 50.
        for v in &p0 {
            assert_eq!(*v, 50, "red plane should be 50");
        }
        for v in &p1 {
            assert_eq!(*v, 50, "G1 plane should be 50");
        }
        for v in &p2 {
            assert_eq!(*v, 50, "G2 plane should be 50");
        }
        for v in &p3 {
            assert_eq!(*v, 50, "blue plane should be 50");
        }
    }

    /// Forward Star-Tetrix followed by inverse must round-trip
    /// every (R, G1, G2, B) plane bit-exactly under flat-zero input.
    #[test]
    fn star_tetrix_forward_inverse_round_trip_zero() {
        let n = 8 * 8;
        let mut p0 = vec![0i32; n];
        let mut p1 = vec![0i32; n];
        let mut p2 = vec![0i32; n];
        let mut p3 = vec![0i32; n];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        forward_star_tetrix(&mut planes, 8, 8, 0, 0, 0, 0).unwrap();
        inverse_star_tetrix(&mut planes, 8, 8, 0, 0, 0, 0).unwrap();
        for &v in p0.iter().chain(&p1).chain(&p2).chain(&p3) {
            assert_eq!(v, 0);
        }
    }

    /// Forward → inverse over a non-trivial CFA-ish input: 4 components,
    /// 8×8 each, with distinct per-plane patterns. Bit-exact recovery.
    #[test]
    fn star_tetrix_forward_inverse_round_trip_nontrivial() {
        let n = 8 * 8;
        let mk = |seed: i32| -> Vec<i32> {
            (0..n)
                .map(|i| {
                    let x = i % 8;
                    let y = i / 8;
                    (x * seed + y * (seed + 1) + 50) % 240 - 120
                })
                .collect()
        };
        let r0 = mk(7);
        let g1_0 = mk(11);
        let g2_0 = mk(13);
        let b0 = mk(17);
        let mut p0 = r0.clone();
        let mut p1 = g1_0.clone();
        let mut p2 = g2_0.clone();
        let mut p3 = b0.clone();
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        forward_star_tetrix(&mut planes, 8, 8, 0, 0, 0, 0).unwrap();
        // Apply inverse — must recover (R, G1, G2, B) exactly.
        inverse_star_tetrix(&mut planes, 8, 8, 0, 0, 0, 0).unwrap();
        assert_eq!(p0, r0, "red plane must round-trip");
        assert_eq!(p1, g1_0, "G1 plane must round-trip");
        assert_eq!(p2, g2_0, "G2 plane must round-trip");
        assert_eq!(p3, b0, "blue plane must round-trip");
    }

    /// Same nontrivial round-trip with `e1=2`, `e2=3` (non-default
    /// chroma weights from the CTS marker).
    #[test]
    fn star_tetrix_round_trip_nondefault_e1_e2() {
        let n = 8 * 8;
        let mk = |seed: i32| -> Vec<i32> {
            (0..n)
                .map(|i| {
                    let x = i % 8;
                    let y = i / 8;
                    (x * seed - y * (seed + 1) + 80) % 200 - 100
                })
                .collect()
        };
        let r0 = mk(3);
        let g1_0 = mk(5);
        let g2_0 = mk(2);
        let b0 = mk(9);
        let mut p0 = r0.clone();
        let mut p1 = g1_0.clone();
        let mut p2 = g2_0.clone();
        let mut p3 = b0.clone();
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        forward_star_tetrix(&mut planes, 8, 8, 2, 3, 0, 0).unwrap();
        inverse_star_tetrix(&mut planes, 8, 8, 2, 3, 0, 0).unwrap();
        assert_eq!(p0, r0);
        assert_eq!(p1, g1_0);
        assert_eq!(p2, g2_0);
        assert_eq!(p3, b0);
    }

    /// Round-trip with the alternate CFA pattern type (Ct=1) and
    /// in-line access mode (Cf=3).
    #[test]
    fn star_tetrix_round_trip_ct1_cf3() {
        let n = 8 * 8;
        let mk = |seed: i32| -> Vec<i32> {
            (0..n)
                .map(|i| {
                    let x = i % 8;
                    let y = i / 8;
                    (x * seed + y + 30) % 200 - 100
                })
                .collect()
        };
        let r0 = mk(7);
        let g1_0 = mk(11);
        let g2_0 = mk(13);
        let b0 = mk(17);
        let mut p0 = r0.clone();
        let mut p1 = g1_0.clone();
        let mut p2 = g2_0.clone();
        let mut p3 = b0.clone();
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        forward_star_tetrix(&mut planes, 8, 8, 0, 0, 1, 3).unwrap();
        inverse_star_tetrix(&mut planes, 8, 8, 0, 0, 1, 3).unwrap();
        assert_eq!(p0, r0);
        assert_eq!(p1, g1_0);
        assert_eq!(p2, g2_0);
        assert_eq!(p3, b0);
    }

    /// Forward Star-Tetrix rejects wrong component count.
    #[test]
    fn forward_star_tetrix_rejects_wrong_component_count() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        let err = forward_star_tetrix(&mut planes, 2, 2, 0, 0, 0, 0).unwrap_err();
        assert!(format!("{err}").contains("4 component"));
    }

    /// Round 95 (r93): Star-Tetrix accepts >=4 planes — Annex F.2
    /// Table F.1 leaves components c≥4 untouched. Verifies the trailing
    /// plane(s) round-trip bit-exactly when the first 4 carry the CFA
    /// data.
    #[test]
    fn star_tetrix_passes_extra_components_through() {
        // 4×4 picture, 5 components: 4 CFA + 1 suppressed tail.
        let wf = 4usize;
        let hf = 4usize;
        let size = wf * hf;
        let r0: Vec<i32> = (0..size as i32).map(|x| 100 + x * 2).collect();
        let g1_0: Vec<i32> = (0..size as i32).map(|x| 120 + x).collect();
        let g2_0: Vec<i32> = (0..size as i32).map(|x| 130 - x).collect();
        let b0: Vec<i32> = (0..size as i32).map(|x| 90 + x * 3).collect();
        let tail0: Vec<i32> = (0..size as i32).map(|x| 200 + x * 5).collect();
        let mut p0 = r0.clone();
        let mut p1 = g1_0.clone();
        let mut p2 = g2_0.clone();
        let mut p3 = b0.clone();
        let mut p4 = tail0.clone();
        {
            let mut planes: [&mut [i32]; 5] = [&mut p0, &mut p1, &mut p2, &mut p3, &mut p4];
            forward_star_tetrix(&mut planes, wf, hf, 0, 0, 0, 0).unwrap();
            inverse_star_tetrix(&mut planes, wf, hf, 0, 0, 0, 0).unwrap();
        }
        assert_eq!(p0, r0, "R round-trips");
        assert_eq!(p1, g1_0, "G1 round-trips");
        assert_eq!(p2, g2_0, "G2 round-trips");
        assert_eq!(p3, b0, "B round-trips");
        // The 5th plane must not be touched by either direction.
        assert_eq!(p4, tail0, "trailing tail plane must pass through unchanged");
    }

    /// Round 95 (r93): RCT accepts >=3 planes — Annex F.2 Table F.1
    /// leaves components c≥3 untouched. Verifies the trailing plane(s)
    /// round-trip bit-exactly when the first 3 carry the RGB data.
    #[test]
    fn rct_passes_extra_components_through() {
        let wf = 4usize;
        let hf = 4usize;
        let size = wf * hf;
        let r0: Vec<i32> = (0..size as i32).map(|x| 60 + x * 2).collect();
        let g0: Vec<i32> = (0..size as i32).map(|x| 80 + x).collect();
        let b0: Vec<i32> = (0..size as i32).map(|x| 50 + x * 3).collect();
        let alpha0: Vec<i32> = (0..size as i32).map(|x| 200 - x).collect();
        let mut p0 = r0.clone();
        let mut p1 = g0.clone();
        let mut p2 = b0.clone();
        let mut p3 = alpha0.clone();
        {
            let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
            super::forward_rct(&mut planes, wf, hf).unwrap();
            inverse_rct(&mut planes, wf, hf).unwrap();
        }
        assert_eq!(p0, r0, "R round-trips");
        assert_eq!(p1, g0, "G round-trips");
        assert_eq!(p2, b0, "B round-trips");
        assert_eq!(
            p3, alpha0,
            "trailing alpha plane must pass through unchanged"
        );
    }

    /// Direct check of the `floor_div` helper — the spec's lifting
    /// uses floor division (i.e. floors towards `-∞`) which differs
    /// from Rust's `/` for negative numerators.
    #[test]
    fn floor_div_matches_spec_floor_semantics() {
        assert_eq!(floor_div(8, 4), 2);
        assert_eq!(floor_div(7, 4), 1);
        assert_eq!(floor_div(-1, 4), -1);
        assert_eq!(floor_div(-4, 4), -1);
        assert_eq!(floor_div(-5, 4), -2);
        assert_eq!(floor_div(0, 8), 0);
    }
}
