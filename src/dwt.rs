//! Reversible 5/3 discrete wavelet transform (ISO/IEC 21122-1:2022,
//! Annex E).
//!
//! Forward and inverse 5/3 transform of one component: the 1-D lifting
//! kernels, one joint (horizontal + vertical) level, and the multi-level
//! cascades of Tables E.1 / E.8.
//!
//! Spec mapping:
//!
//! * [`extend_symmetric`] — Annex E.6, Table E.5 (whole-sample symmetric
//!   reflection: `X[-i] = X[i]`, `X[Z+i-1] = X[Z-i-1]` for `i = 1, 2`),
//!   kept as the literal statement of the boundary rule. The production
//!   kernels fold the reflection into two mirror rules on de-interleaved
//!   data (derivation next to the kernels) and never materialise the
//!   margin.
//! * Annex E.7, Table E.6 (inverse lifting: even samples
//!   `Y[i] = X[i] - ((X[i-1] + X[i+1] + 2) >> 2)`, then odd samples
//!   `Y[i] = X[i] + ((Y[i-1] + Y[i+1]) >> 1)`) and Annex E.13, Table E.12
//!   (forward lifting: odd samples `Y[i] = X[i] - ((X[i-1] + X[i+1]) >> 1)`,
//!   then even samples `Y[i] = X[i] + ((Y[i-1] + Y[i+1] + 2) >> 2)`).
//! * [`inverse_horizontal_1d`] / [`forward_horizontal_1d`] — Annex E.4
//!   / E.11, one row between interleaved samples and the de-interleaved
//!   low-pass / high-pass bands.
//! * [`inverse_vertical_1d`] / [`forward_vertical_1d`] — Annex E.5 /
//!   E.12, the column equivalent. Inside a joint level the vertical pass
//!   runs on whole rows at a time (the table's column loop becomes the
//!   inner contiguous loop; same arithmetic per sample).
//! * [`inverse_2d`] / [`forward_2d`] — one joint (horizontal + vertical)
//!   decomposition level in the pass order of Annex E.2 Table E.1 and
//!   Annex E.9 Table E.8. The inverse filters **horizontally first**
//!   (`LL_{d,d}` + `HL_{d,d}` → `LL_{d-1,d}`, `LH_{d,d}` + `HH_{d,d}` →
//!   `LH_{d-1,d}`) and **then vertically** (`LL_{d-1,d}` + `LH_{d-1,d}` →
//!   `LL_{d-1,d-1}`); the forward transform is the exact reverse,
//!   **vertically first** and then horizontally on both halves. The 5/3
//!   lifting steps round (`>> 1`, `+ 2 >> 2`), so the two possible pass
//!   orders are different integer transforms: a 2×2 block is already
//!   enough to tell them apart (see the unit tests).
//! * [`inverse_cascade_2d`] / [`forward_cascade_2d`] — Tables E.1 / E.8
//!   for one component.
//!
//! Boundary handling. Annex E.6 NOTE guarantees every band is at
//! least two coefficients wide / tall, because of the picture-header
//! constraints on `Wf`, `Hf`, `Cw`. Shorter inputs are rejected.
//!
//! Allocation. A joint level allocates its output block and nothing per
//! row or per column; every size is bounded by the picture-header
//! geometry the codestream parser already validated.

use crate::error::{JpegXsError as Error, Result};

/// Number of pad samples needed on each side of the working buffer
/// for the 5/3 filter. Annex E.6 reflects two samples; the inverse
/// even-sample step needs `X[i-1]` and `X[i+1]` for `i = 0` and
/// `i = Z`, and the forward odd-sample step starts at `i = -1`, so
/// two pad slots on each side are sufficient.
const PAD: usize = 2;

/// Hard cap on a single 1-D DWT pass length. Picture-header
/// width/height fields are 16-bit (Annex A.4.3), so 65 536 samples is
/// already an upper bound; we keep the same cap explicit in the DWT
/// path so a corrupt caller can never request a huge working buffer.
const MAX_DIM: usize = 1 << 17;

/// Apply Annex E.6 whole-sample symmetric extension in place on a
/// working buffer of length `Z + 2*PAD`. The caller writes the `Z`
/// real samples into `buf[PAD .. PAD+Z]`; this function fills the
/// `PAD` slots on either side via `X[-i] = X[i]` and `X[Z+i-1] =
/// X[Z-i-1]`, mapped through the `+PAD` index shift.
///
/// Returns `Error::Decode` if `Z < 2` (Annex E.6 NOTE forbids this) or
/// if the buffer is the wrong size.
pub fn extend_symmetric(buf: &mut [i32], z: usize) -> Result<()> {
    if z < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: band dimension {z} below the spec minimum of 2"
        )));
    }
    if buf.len() != z + 2 * PAD {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: working buffer length {} does not match Z+4 = {}",
            buf.len(),
            z + 2 * PAD
        )));
    }
    // Right edge: X[Z+i-1] = X[Z-i-1] for i = 1, 2. PAD-relative:
    //   buf[PAD+Z]   = buf[PAD+Z-2]    (i = 1)
    //   buf[PAD+Z+1] = buf[PAD+Z-3]    (i = 2)
    // For Z = 2, the i=2 step indexes buf[PAD-1] which is part of the
    // *left* extension. The spec's reflection rules are well defined:
    // we reflect through the left boundary using X[-1] = X[1] (already
    // a real sample at PAD+1). The correct value at PAD-1 must
    // therefore be known before we read it for the right pad.
    //
    // We compute right pad first so its source slots are still real
    // data, then the left pad — this is safe because the left pad's
    // sources (buf[PAD+1] and buf[PAD+2]) live inside the real-data
    // region for Z >= 2 only when Z >= 3. For Z == 2, X[2] is out of
    // range and reflects to X[0] = buf[PAD]: the spec's reflection
    // chain X[-2] -> X[2] -> X[0] collapses to a single hop.
    buf[PAD + z] = buf[PAD + z - 2];
    if z >= 3 {
        buf[PAD + z + 1] = buf[PAD + z - 3];
    } else {
        // Z == 2 case: PAD+Z+1 = PAD+3 which is the right pad slot
        // beyond [PAD+Z]. The reflection chain X[Z+1] = X[Z-3] = X[-1]
        // = X[1] gives buf[PAD+1] (a real sample).
        buf[PAD + z + 1] = buf[PAD + 1];
    }
    // Left edge: X[-i] = X[i] for i = 1, 2. PAD-relative:
    //   buf[PAD-1] = buf[PAD+1]   (real for Z >= 2)
    //   buf[PAD-2] = buf[PAD+2]   (real for Z >= 3; reflects to PAD for Z == 2)
    buf[PAD - 1] = buf[PAD + 1];
    if z >= 3 {
        buf[PAD - 2] = buf[PAD + 2];
    } else {
        // Z == 2 case: X[-2] = X[2] = X[0] (reflection through the
        // right boundary collapses since X[2] = X[Z] which we just set
        // to X[Z-2] = X[0]).
        buf[PAD - 2] = buf[PAD];
    }
    Ok(())
}

/// Check a 1-D pass length against the Annex E.6 NOTE minimum and the
/// working cap.
fn check_pass_len(z: usize) -> Result<()> {
    if z < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: row width {z} below the spec minimum of 2"
        )));
    }
    if z > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: band dimension {z} out of range (1..={MAX_DIM})"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Lifting kernels on de-interleaved data.
//
// Tables E.6 / E.12 are written on an interleaved array `X` extended by
// two samples (Table E.5). With `L[n] = X[2n]`, `H[n] = X[2n+1]`,
// `NL = ⌈Z/2⌉` low and `NH = ⌊Z/2⌋` high samples, the extension reduces to
// two mirror rules on the *other* phase, so no margin is materialised:
//
//   inverse  E[n] = L[n] − ((H[n−1] + H[n] + 2) >> 2)   H[−1] := H[0]
//                                                        H[NH] := H[NH−1]   (Z odd)
//            O[n] = H[n] + ((E[n] + E[n+1]) >> 1)       E[NL] := E[NL−1]   (Z even)
//
//   forward  H[n] = X[2n+1] − ((X[2n] + X[2n+2]) >> 1)  X[Z]  := X[Z−2]    (Z even)
//            L[n] = X[2n] + ((H[n−1] + H[n] + 2) >> 2)  H[−1] := H[0]
//                                                        H[NH] := H[NH−1]   (Z odd)
//
// Derivation of the mirrors from Table E.5 (`X[−i] = X[i]`,
// `X[Z+i−1] = X[Z−i−1]`, `i = 1, 2`):
//   * `X[−1] = X[1]` is `H[−1] = H[0]` on the inverse; on the forward,
//     `Y[−1] = X[−1] − ((X[−2] + X[0]) >> 1) = X[1] − ((X[2] + X[0]) >> 1)`
//     is the already-computed `H[0]`.
//   * Z odd: `X[Z] = X[Z−2]` is `H[NH] = H[NH−1]` on the inverse; on the
//     forward `Y[Z] = X[Z−2] − ((X[Z−1] + X[Z−3]) >> 1) = H[NH−1]`.
//   * Z even: the inverse reads `Y[Z] = X[Z−2] − ((X[Z−1] + X[Z−3] + 2) >> 2)`
//     `= E[NL−1]`; the forward reads `X[Z] = X[Z−2]` directly.
// The unit tests compare these kernels with a literal transcription of
// the tables for every length from 2 up.
// ---------------------------------------------------------------------

/// Inverse lifting of one line: `low` / `high` → interleaved `out`
/// (`out.len() == low.len() + high.len() ≥ 2`).
#[inline]
fn synth_line(low: &[i32], high: &[i32], out: &mut [i32]) {
    let (nl, nh) = (low.len(), high.len());
    debug_assert!(nh >= 1 && (nl == nh || nl == nh + 1) && out.len() == nl + nh);
    // Even phase.
    out[0] = synth_even(low[0], high[0], high[0]);
    for n in 1..nh {
        out[2 * n] = synth_even(low[n], high[n - 1], high[n]);
    }
    if nl > nh {
        out[2 * nh] = synth_even(low[nh], high[nh - 1], high[nh - 1]);
    }
    // Odd phase.
    for n in 0..nh - 1 {
        out[2 * n + 1] = synth_odd(high[n], out[2 * n], out[2 * n + 2]);
    }
    let last = nh - 1;
    let right = if nl > nh {
        out[2 * last + 2]
    } else {
        out[2 * last]
    };
    out[2 * last + 1] = synth_odd(high[last], out[2 * last], right);
}

/// Table E.6 even step, `X − ((a + b + 2) >> 2)`.
///
/// The inverse runs on whatever a codestream carries. Coefficients of a
/// conforming stream stay far inside `i32` (at most `Bw = 20` bits plus
/// the filter gain), where wrapping and plain arithmetic agree; a
/// corrupt stream can carry magnitudes up to 31 bitplanes, and wrapping
/// keeps the transform total (no overflow trap) on such input.
#[inline(always)]
fn synth_even(x: i32, a: i32, b: i32) -> i32 {
    x.wrapping_sub(a.wrapping_add(b).wrapping_add(2) >> 2)
}

/// Table E.6 odd step, `X + ((a + b) >> 1)`; wrapping as [`synth_even`].
#[inline(always)]
fn synth_odd(x: i32, a: i32, b: i32) -> i32 {
    x.wrapping_add(a.wrapping_add(b) >> 1)
}

/// Forward lifting of one line: interleaved `x` → `low` / `high`
/// (`x.len() == low.len() + high.len() ≥ 2`).
#[inline]
fn analyse_line(x: &[i32], low: &mut [i32], high: &mut [i32]) {
    let (nl, nh) = (low.len(), high.len());
    debug_assert!(nh >= 1 && (nl == nh || nl == nh + 1) && x.len() == nl + nh);
    // High phase.
    for n in 0..nh - 1 {
        high[n] = x[2 * n + 1] - ((x[2 * n] + x[2 * n + 2]) >> 1);
    }
    let last = nh - 1;
    let right = if nl > nh {
        x[2 * last + 2]
    } else {
        x[2 * last]
    };
    high[last] = x[2 * last + 1] - ((x[2 * last] + right) >> 1);
    // Low phase.
    low[0] = x[0] + ((high[0] + high[0] + 2) >> 2);
    for n in 1..nh {
        low[n] = x[2 * n] + ((high[n - 1] + high[n] + 2) >> 2);
    }
    if nl > nh {
        low[nh] = x[2 * nh] + ((high[nh - 1] + high[nh - 1] + 2) >> 2);
    }
}

/// Split `buf` (rows of `w`) around row `y`: `(row y−1, row y, row y+1)`,
/// the neighbours being `None` outside the buffer.
#[inline]
fn row_and_neighbours(
    buf: &mut [i32],
    w: usize,
    y: usize,
) -> (Option<&[i32]>, &mut [i32], Option<&[i32]>) {
    let (before, rest) = buf.split_at_mut(y * w);
    let (row, after) = rest.split_at_mut(w);
    let above = (y > 0).then(|| &before[(y - 1) * w..]);
    let below = (after.len() >= w).then(|| &after[..w]);
    (above, row, below)
}

/// Inverse vertical lifting in place on a block whose even rows hold the
/// vertically low-pass lines and whose odd rows the high-pass lines
/// (Table E.4 with whole rows as the unit — the column loop of the table
/// becomes the inner, contiguous loop).
fn synth_rows_in_place(buf: &mut [i32], w: usize, h: usize) {
    debug_assert!(h >= 2 && buf.len() == w * h);
    // Even rows: E[n] = L[n] − ((H[n−1] + H[n] + 2) >> 2).
    for y in (0..h).step_by(2) {
        let (above, row, below) = row_and_neighbours(buf, w, y);
        let (a, b) = match (above, below) {
            (Some(a), Some(b)) => (a, b),
            (None, Some(b)) => (b, b),
            (Some(a), None) => (a, a),
            (None, None) => unreachable!("h >= 2"),
        };
        for ((t, &p), &q) in row.iter_mut().zip(a).zip(b) {
            *t = synth_even(*t, p, q);
        }
    }
    // Odd rows: O[n] = H[n] + ((E[n] + E[n+1]) >> 1).
    for y in (1..h).step_by(2) {
        let (above, row, below) = row_and_neighbours(buf, w, y);
        let a = above.expect("odd row has a row above");
        let b = below.unwrap_or(a);
        for ((t, &p), &q) in row.iter_mut().zip(a).zip(b) {
            *t = synth_odd(*t, p, q);
        }
    }
}

/// Forward vertical lifting in place: on return even rows hold the
/// vertically low-pass lines, odd rows the high-pass lines (Table E.11
/// with whole rows as the unit).
fn analyse_rows_in_place(buf: &mut [i32], w: usize, h: usize) {
    debug_assert!(h >= 2 && buf.len() == w * h);
    // Odd rows: H[n] = X[2n+1] − ((X[2n] + X[2n+2]) >> 1).
    for y in (1..h).step_by(2) {
        let (above, row, below) = row_and_neighbours(buf, w, y);
        let a = above.expect("odd row has a row above");
        let b = below.unwrap_or(a);
        for ((t, &p), &q) in row.iter_mut().zip(a).zip(b) {
            *t -= (p + q) >> 1;
        }
    }
    // Even rows: L[n] = X[2n] + ((H[n−1] + H[n] + 2) >> 2).
    for y in (0..h).step_by(2) {
        let (above, row, below) = row_and_neighbours(buf, w, y);
        let (a, b) = match (above, below) {
            (Some(a), Some(b)) => (a, b),
            (None, Some(b)) => (b, b),
            (Some(a), None) => (a, a),
            (None, None) => unreachable!("h >= 2"),
        };
        for ((t, &p), &q) in row.iter_mut().zip(a).zip(b) {
            *t += (p + q + 2) >> 2;
        }
    }
}

/// Annex E.4, Table E.3 — single-row inverse horizontal filter.
///
/// `low` and `high` hold one row of the low-pass and high-pass
/// horizontal sub-bands respectively; their lengths must satisfy the
/// 5/3 even-input convention `low.len() == (w + 1) / 2` and
/// `high.len() == w / 2` where `w` is the reconstructed row width.
/// `out.len()` must equal `w`. The result is the reconstructed row.
pub fn inverse_horizontal_1d(low: &[i32], high: &[i32], out: &mut [i32]) -> Result<()> {
    let w = out.len();
    check_pass_len(w)?;
    let low_len = w.div_ceil(2);
    let high_len = w / 2;
    if low.len() != low_len || high.len() != high_len {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: inverse horizontal expected low.len()={low_len} high.len()={high_len}, got low.len()={} high.len()={}",
            low.len(),
            high.len(),
        )));
    }
    synth_line(low, high, out);
    Ok(())
}

/// Annex E.11, Table E.10 — single-row forward horizontal filter.
///
/// `input.len()` is the row width `w`. `low` and `high` are the
/// low-pass / high-pass output sub-bands of lengths `(w+1)/2` and `w/2`
/// respectively.
pub fn forward_horizontal_1d(input: &[i32], low: &mut [i32], high: &mut [i32]) -> Result<()> {
    let w = input.len();
    check_pass_len(w)?;
    let low_len = w.div_ceil(2);
    let high_len = w / 2;
    if low.len() != low_len || high.len() != high_len {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: forward horizontal expected low.len()={low_len} high.len()={high_len}, got low.len()={} high.len()={}",
            low.len(),
            high.len(),
        )));
    }
    analyse_line(input, low, high);
    Ok(())
}

/// Annex E.5, Table E.4 — single-column inverse vertical filter.
/// Same conventions as [`inverse_horizontal_1d`] but along the
/// vertical axis: `low` carries even rows, `high` carries odd rows.
pub fn inverse_vertical_1d(low: &[i32], high: &[i32], out: &mut [i32]) -> Result<()> {
    // The 1-D step is axis-agnostic — the heavy lifting is in the
    // 2-D wrapper that walks columns. Delegate to the horizontal
    // routine to keep the spec-traceable code in one place.
    inverse_horizontal_1d(low, high, out)
}

/// Annex E.12, Table E.11 — single-column forward vertical filter.
pub fn forward_vertical_1d(input: &[i32], low: &mut [i32], high: &mut [i32]) -> Result<()> {
    forward_horizontal_1d(input, low, high)
}

/// Single-level 2-D inverse 5/3 DWT.
///
/// Inputs are the four sub-bands of a single decomposition level,
/// each laid out in row-major order:
///
/// * `ll[ll_h × ll_w]` — low-low coefficients
/// * `hl[ll_h × hl_w]` — high-low (horizontal-pass high)
/// * `lh[lh_h × ll_w]` — low-high (vertical-pass high)
/// * `hh[lh_h × hl_w]` — high-high
///
/// where `ll_w = (w + 1) / 2`, `hl_w = w / 2`, `ll_h = (h + 1) / 2`,
/// `lh_h = h / 2`. `out` is the reconstructed `h × w` band, also
/// row-major.
///
/// Pass order is the one Annex E.2 Table E.1 writes for a joint level
/// `d`: `hor_transform(LL_{d-1,d} ← LL_{d,d}, HL_{d,d})`,
/// `hor_transform(LH_{d-1,d} ← LH_{d,d}, HH_{d,d})`, then
/// `ver_transform(LL_{d-1,d-1} ← LL_{d-1,d}, LH_{d-1,d})` — horizontal
/// synthesis on the low rows and on the high rows first, vertical
/// synthesis last.
pub fn inverse_2d(
    w: usize,
    h: usize,
    ll: &[i32],
    hl: &[i32],
    lh: &[i32],
    hh: &[i32],
    out: &mut [i32],
) -> Result<()> {
    if w < 2 || h < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: 2-D band dimensions ({w}x{h}) below the spec minimum of 2x2"
        )));
    }
    if w > MAX_DIM || h > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: 2-D band dimensions ({w}x{h}) exceed cap {MAX_DIM}"
        )));
    }
    let ll_w = w.div_ceil(2);
    let hl_w = w / 2;
    let ll_h = h.div_ceil(2);
    let lh_h = h / 2;
    if ll.len() != ll_w * ll_h
        || hl.len() != hl_w * ll_h
        || lh.len() != ll_w * lh_h
        || hh.len() != hl_w * lh_h
        || out.len() != w * h
    {
        return Err(Error::invalid(
            "JPEG XS DWT: inverse_2d sub-band size mismatch",
        ));
    }

    synth_joint_level(w, h, ll, hl, lh, hh, out);
    Ok(())
}

/// One joint level of Table E.1 with validated geometry. The two
/// horizontal calls write straight into the rows of `out` they belong to
/// (low rows at even, high rows at odd positions — the interleaving
/// Table E.4 asks for), then the vertical call lifts `out` in place.
fn synth_joint_level(
    w: usize,
    h: usize,
    ll: &[i32],
    hl: &[i32],
    lh: &[i32],
    hh: &[i32],
    out: &mut [i32],
) {
    let (ll_w, hl_w) = (w.div_ceil(2), w / 2);
    for (y, dst) in out.chunks_exact_mut(w).enumerate() {
        let n = y / 2;
        let (low, high) = if y % 2 == 0 { (ll, hl) } else { (lh, hh) };
        synth_line(
            &low[n * ll_w..(n + 1) * ll_w],
            &high[n * hl_w..(n + 1) * hl_w],
            dst,
        );
    }
    synth_rows_in_place(out, w, h);
}

/// One joint level of Table E.8 with validated geometry; `block` (the
/// `h × w` input) is used as the workspace of the vertical call and is
/// left holding its interleaved output.
fn analyse_joint_level(
    w: usize,
    h: usize,
    block: &mut [i32],
    ll: &mut [i32],
    hl: &mut [i32],
    lh: &mut [i32],
    hh: &mut [i32],
) {
    let (ll_w, hl_w) = (w.div_ceil(2), w / 2);
    analyse_rows_in_place(block, w, h);
    for (y, src) in block.chunks_exact(w).enumerate() {
        let n = y / 2;
        let (low, high) = if y % 2 == 0 {
            (&mut *ll, &mut *hl)
        } else {
            (&mut *lh, &mut *hh)
        };
        analyse_line(
            src,
            &mut low[n * ll_w..(n + 1) * ll_w],
            &mut high[n * hl_w..(n + 1) * hl_w],
        );
    }
}

/// Multi-level 2-D inverse 5/3 DWT cascade (Annex E.2, Table E.1).
///
/// `wc`, `hc` are the per-component sample-grid dimensions
/// (`Wc[i] × Hc[i]`). `nlx`, `nly` are the number of horizontal /
/// vertical decomposition levels; for `nlx == 1 && nly == 1` this
/// degenerates to a single call to [`inverse_2d`].
///
/// `bands` is indexed by the spec's per-component filter index β
/// (`0 .. Nβ - 1`, where `Nβ = 2 × min(nlx, nly) + max(nlx, nly) + 1`).
/// Each `bands[β]` is the dequantized coefficient array for that
/// (β, component) cell, laid out row-major with the band geometry
/// derived from Annex B.2 (`Wb[β], Hb[β]`).
///
/// For `nlx >= nly`, the spec runs the cascade in two phases:
///
/// 1. Pure horizontal levels `dx = nlx .. dx > min(nlx, nly)`:
///    combine `LL_{dx, nly}` and `HL_{dx, nly}` to obtain
///    `LL_{dx-1, nly}`. (Skipped when `nlx == nly`.)
/// 2. Joint horizontal+vertical levels `d = min(nlx, nly) .. 1`:
///    combine `LL_{d,d}, HL_{d,d}, LH_{d,d}, HH_{d,d}` to obtain
///    `LL_{d-1, d-1}` — horizontally first, then vertically
///    ([`inverse_2d`]).
///
/// The output `out` is the reconstructed `wc × hc` plane (row-major).
///
/// `nly > nlx` is rejected — the spec NOTE 1 in B.3 says "for
/// `NL,x < NL,y` interchange `NL,x` with `NL,y`", but we do not need
/// that path for round-6 scope.
pub fn inverse_cascade_2d(
    wc: usize,
    hc: usize,
    nlx: u8,
    nly: u8,
    bands: &[Vec<i32>],
    out: &mut [i32],
) -> Result<()> {
    if nlx == 0 && nly == 0 {
        // No DWT — band 0 is the raw component samples, copied through.
        if bands.is_empty() {
            return Err(Error::invalid(
                "JPEG XS DWT cascade: no bands but expected at least 1",
            ));
        }
        if bands[0].len() != wc * hc || out.len() != wc * hc {
            return Err(Error::invalid(format!(
                "JPEG XS DWT cascade: size mismatch (wc*hc={} band0={} out={})",
                wc * hc,
                bands[0].len(),
                out.len()
            )));
        }
        out.copy_from_slice(&bands[0]);
        return Ok(());
    }
    if nly > nlx {
        return Err(Error::Unsupported(format!(
            "JPEG XS DWT cascade: NL,y={nly} > NL,x={nlx} not supported"
        )));
    }
    if wc < 2 || hc < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: component dims ({wc}x{hc}) below the spec minimum of 2x2"
        )));
    }
    if wc > MAX_DIM || hc > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: component dims ({wc}x{hc}) exceed cap {MAX_DIM}"
        )));
    }
    let nbeta = beta_count(nlx, nly);
    if bands.len() != nbeta {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: expected {nbeta} bands for NL,x={nlx} NL,y={nly}, got {}",
            bands.len()
        )));
    }
    if out.len() != wc * hc {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: output buffer {} != wc*hc {}",
            out.len(),
            wc * hc
        )));
    }

    // Map each filter type β → (dx, dy, τx, τy).
    let layout: Vec<BandKey> = (0..nbeta as u32)
        .map(|beta| beta_to_band_key(beta, nlx, nly))
        .collect();

    // Index by (dx, dy, τx, τy) → band id β.
    let find_band = |dx: u32, dy: u32, tx: bool, ty: bool| -> Option<usize> {
        layout
            .iter()
            .position(|k| k.dx == dx && k.dy == dy && k.tau_x == tx && k.tau_y == ty)
    };

    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let dxy_min = nlx_u.min(nly_u);

    // Helper: dimensions of a band at decomposition (dx, dy) with
    // high-pass selectors (tx, ty). Mirrors the slice walker formula.
    let band_dims = |dx: u32, dy: u32, tx: bool, ty: bool| -> (usize, usize) {
        let w = if !tx {
            if dx == 0 {
                wc as u32
            } else {
                ((wc as u32) + (1u32 << dx) - 1) >> dx
            }
        } else {
            let denom_minus1 = if dx == 0 { 1 } else { 1u32 << (dx - 1) };
            (wc as u32).div_ceil(denom_minus1) / 2
        };
        let h = if !ty {
            if dy == 0 {
                hc as u32
            } else {
                ((hc as u32) + (1u32 << dy) - 1) >> dy
            }
        } else {
            let denom_minus1 = if dy == 0 { 1 } else { 1u32 << (dy - 1) };
            (hc as u32).div_ceil(denom_minus1) / 2
        };
        (w as usize, h as usize)
    };

    // Working storage for the temporary LL band as it grows from level
    // (nlx, nly) up to (0, 0). At every iteration of the outer loop,
    // `current_ll` holds the latest reconstructed LL band.
    let ll_b = find_band(nlx_u, nly_u, false, false).ok_or_else(|| {
        Error::invalid(format!(
            "JPEG XS DWT cascade: no LL band at level (NL,x={nlx}, NL,y={nly})"
        ))
    })?;
    let (mut cur_w, mut cur_h) = band_dims(nlx_u, nly_u, false, false);
    if bands[ll_b].len() != cur_w * cur_h {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: LL band {} has {} samples, expected {}x{}={}",
            ll_b,
            bands[ll_b].len(),
            cur_w,
            cur_h,
            cur_w * cur_h
        )));
    }
    let mut current_ll = bands[ll_b].clone();

    // Phase 1 — pure horizontal levels (only when nlx > nly).
    // dx walks from nlx down to dxy_min + 1, performing horizontal-only
    // synthesis with LL_{dx, nly} and HL_{dx, nly} → LL_{dx-1, nly}.
    let mut dx_cur = nlx_u;
    while dx_cur > dxy_min {
        let hl_b = find_band(dx_cur, nly_u, true, false).ok_or_else(|| {
            Error::invalid(format!(
                "JPEG XS DWT cascade: no HL band at level (dx={dx_cur}, dy={nly_u})"
            ))
        })?;
        let (hl_w, hl_h) = band_dims(dx_cur, nly_u, true, false);
        if bands[hl_b].len() != hl_w * hl_h {
            return Err(Error::invalid(format!(
                "JPEG XS DWT cascade: HL band {} has {} samples, expected {}x{}",
                hl_b,
                bands[hl_b].len(),
                hl_w,
                hl_h
            )));
        }
        if hl_h != cur_h {
            return Err(Error::invalid(format!(
                "JPEG XS DWT cascade: HL height {hl_h} != current LL height {cur_h}"
            )));
        }
        // Output LL band dimensions at level (dx-1, nly).
        let (next_w, next_h) = band_dims(dx_cur - 1, nly_u, false, false);
        debug_assert_eq!(next_h, cur_h);
        let mut next_ll = vec![0i32; next_w * next_h];
        // Per-row inverse horizontal filter combining LL row with HL row.
        for y in 0..cur_h {
            let low = &current_ll[y * cur_w..(y + 1) * cur_w];
            let high = &bands[hl_b][y * hl_w..(y + 1) * hl_w];
            inverse_horizontal_1d(low, high, &mut next_ll[y * next_w..(y + 1) * next_w])?;
        }
        current_ll = next_ll;
        cur_w = next_w;
        cur_h = next_h;
        dx_cur -= 1;
    }

    // Phase 2 — joint levels d = dxy_min down to 1.
    let mut d = dxy_min;
    while d > 0 {
        let ll_dims = band_dims(d, d, false, false);
        if (cur_w, cur_h) != ll_dims {
            return Err(Error::invalid(format!(
                "JPEG XS DWT cascade: pre-step dims ({cur_w}x{cur_h}) != LL_{{d,d}} dims ({}x{}) at d={d}",
                ll_dims.0, ll_dims.1
            )));
        }
        let hl_b = find_band(d, d, true, false)
            .ok_or_else(|| Error::invalid(format!("JPEG XS DWT cascade: no HL band at (d={d})")))?;
        let lh_b = find_band(d, d, false, true)
            .ok_or_else(|| Error::invalid(format!("JPEG XS DWT cascade: no LH band at (d={d})")))?;
        let hh_b = find_band(d, d, true, true)
            .ok_or_else(|| Error::invalid(format!("JPEG XS DWT cascade: no HH band at (d={d})")))?;
        let (next_w, next_h) = band_dims(d - 1, d - 1, false, false);
        let mut next_ll = vec![0i32; next_w * next_h];
        inverse_2d(
            next_w,
            next_h,
            &current_ll,
            &bands[hl_b],
            &bands[lh_b],
            &bands[hh_b],
            &mut next_ll,
        )?;
        current_ll = next_ll;
        cur_w = next_w;
        cur_h = next_h;
        d -= 1;
    }

    if (cur_w, cur_h) != (wc, hc) {
        return Err(Error::invalid(format!(
            "JPEG XS DWT cascade: final LL dims ({cur_w}x{cur_h}) != component dims ({wc}x{hc})"
        )));
    }
    out.copy_from_slice(&current_ll);
    Ok(())
}

/// Number of wavelet filter types `Nβ` for a (`NL,x`, `NL,y`)
/// decomposition (Annex B.3).
fn beta_count(nlx: u8, nly: u8) -> usize {
    let mn = nlx.min(nly) as usize;
    let mx = nlx.max(nly) as usize;
    2 * mn + mx + 1
}

/// Per-band filter-type metadata used by [`inverse_cascade_2d`].
#[derive(Debug, Clone, Copy)]
struct BandKey {
    dx: u32,
    dy: u32,
    tau_x: bool,
    tau_y: bool,
}

/// Map a filter index β to its (dx, dy, τx, τy) per Annex B.3.
/// Mirrors the inverse direction of the slice walker's `beta_levels`,
/// kept inline to avoid creating a load-bearing crate-internal API.
fn beta_to_band_key(beta: u32, nlx: u8, nly: u8) -> BandKey {
    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    debug_assert!(nlx_u >= nly_u, "cascade assumes NL,x >= NL,y");

    if nly_u == 0 {
        if beta == 0 {
            return BandKey {
                dx: nlx_u,
                dy: 0,
                tau_x: false,
                tau_y: false,
            };
        }
        return BandKey {
            dx: nlx_u + 1 - beta,
            dy: 0,
            tau_x: true,
            tau_y: false,
        };
    }

    let beta1 = nlx_u - nly_u + 1;
    if beta < beta1 {
        if beta == 0 {
            return BandKey {
                dx: nlx_u,
                dy: nly_u,
                tau_x: false,
                tau_y: false,
            };
        }
        return BandKey {
            dx: nlx_u + 1 - beta,
            dy: nly_u,
            tau_x: true,
            tau_y: false,
        };
    }

    let group_in = beta - beta1;
    let triple = group_in / 3;
    let within = group_in % 3;
    let dy = nly_u - triple;
    let dx = dy;
    match within {
        0 => BandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: false,
        },
        1 => BandKey {
            dx,
            dy,
            tau_x: false,
            tau_y: true,
        },
        2 => BandKey {
            dx,
            dy,
            tau_x: true,
            tau_y: true,
        },
        _ => unreachable!(),
    }
}

/// Single-level 2-D forward 5/3 DWT (companion to [`inverse_2d`]).
///
/// `input[h × w]` is decomposed into the four sub-bands
/// `ll`, `hl`, `lh`, `hh`. Layout matches [`inverse_2d`].
///
/// Pass order is the one Annex E.9 Table E.8 writes for a joint level
/// `d`: `ver_fwd_transform(LL_{d-1,d-1} → LL_{d-1,d}, LH_{d-1,d})`, then
/// `hor_fwd_transform(LL_{d-1,d} → LL_{d,d}, HL_{d,d})` and
/// `hor_fwd_transform(LH_{d-1,d} → LH_{d,d}, HH_{d,d})` — vertical
/// analysis first, horizontal analysis of both halves last, which undoes
/// [`inverse_2d`] step for step.
pub fn forward_2d(
    w: usize,
    h: usize,
    input: &[i32],
    ll: &mut [i32],
    hl: &mut [i32],
    lh: &mut [i32],
    hh: &mut [i32],
) -> Result<()> {
    if w < 2 || h < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: 2-D band dimensions ({w}x{h}) below the spec minimum of 2x2"
        )));
    }
    if w > MAX_DIM || h > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: 2-D band dimensions ({w}x{h}) exceed cap {MAX_DIM}"
        )));
    }
    let ll_w = w.div_ceil(2);
    let hl_w = w / 2;
    let ll_h = h.div_ceil(2);
    let lh_h = h / 2;
    if input.len() != w * h
        || ll.len() != ll_w * ll_h
        || hl.len() != hl_w * ll_h
        || lh.len() != ll_w * lh_h
        || hh.len() != hl_w * lh_h
    {
        return Err(Error::invalid(
            "JPEG XS DWT: forward_2d sub-band size mismatch",
        ));
    }

    let mut block = input.to_vec();
    analyse_joint_level(w, h, &mut block, ll, hl, lh, hh);
    Ok(())
}

/// Multi-level 2-D forward 5/3 DWT cascade — encoder-side companion of
/// [`inverse_cascade_2d`]. Produces a `Vec<Vec<i32>>` of band buffers
/// indexed by the same per-component filter index `β` the inverse uses
/// (`0 .. Nβ - 1`); each entry is sized to match the band geometry the
/// inverse expects (`Wb[β] × Hb[β]`).
///
/// Pipeline mirrors the inverse two-phase decomposition in reverse:
///
/// 1. Joint horizontal+vertical levels `d = 1 .. min(nlx, nly)`:
///    forward 2-D DWT splits the running LL band into
///    `(LL_{d,d}, HL_{d,d}, LH_{d,d}, HH_{d,d})`. Output bands are
///    written into the band buffers; LL becomes the input of the next
///    level.
/// 2. Pure horizontal levels `dx = min(nlx, nly) + 1 .. nlx` (only when
///    `nlx > nly`): forward 1-D horizontal split per row turns
///    `LL_{dx-1, nly}` into `(LL_{dx, nly}, HL_{dx, nly})`.
///
/// `nly > nlx` is rejected (mirrors the inverse). `nlx == 0 && nly == 0`
/// returns one band == raw component samples.
pub fn forward_cascade_2d(
    wc: usize,
    hc: usize,
    nlx: u8,
    nly: u8,
    input: &[i32],
) -> Result<Vec<Vec<i32>>> {
    if nlx == 0 && nly == 0 {
        if input.len() != wc * hc {
            return Err(Error::invalid(format!(
                "JPEG XS DWT forward cascade: input {} != wc*hc {}",
                input.len(),
                wc * hc
            )));
        }
        return Ok(vec![input.to_vec()]);
    }
    if nly > nlx {
        return Err(Error::Unsupported(format!(
            "JPEG XS DWT forward cascade: NL,y={nly} > NL,x={nlx} not supported"
        )));
    }
    if wc < 2 || hc < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT forward cascade: component dims ({wc}x{hc}) below the spec minimum of 2x2"
        )));
    }
    if wc > MAX_DIM || hc > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT forward cascade: component dims ({wc}x{hc}) exceed cap {MAX_DIM}"
        )));
    }
    if input.len() != wc * hc {
        return Err(Error::invalid(format!(
            "JPEG XS DWT forward cascade: input {} != wc*hc {}",
            input.len(),
            wc * hc
        )));
    }

    let nbeta = beta_count(nlx, nly);
    let layout: Vec<BandKey> = (0..nbeta as u32)
        .map(|beta| beta_to_band_key(beta, nlx, nly))
        .collect();
    let find_band = |dx: u32, dy: u32, tx: bool, ty: bool| -> Option<usize> {
        layout
            .iter()
            .position(|k| k.dx == dx && k.dy == dy && k.tau_x == tx && k.tau_y == ty)
    };

    // Same band-dim formula as the inverse cascade.
    let band_dims = |dx: u32, dy: u32, tx: bool, ty: bool| -> (usize, usize) {
        let w = if !tx {
            if dx == 0 {
                wc as u32
            } else {
                ((wc as u32) + (1u32 << dx) - 1) >> dx
            }
        } else {
            let denom_minus1 = if dx == 0 { 1 } else { 1u32 << (dx - 1) };
            (wc as u32).div_ceil(denom_minus1) / 2
        };
        let h = if !ty {
            if dy == 0 {
                hc as u32
            } else {
                ((hc as u32) + (1u32 << dy) - 1) >> dy
            }
        } else {
            let denom_minus1 = if dy == 0 { 1 } else { 1u32 << (dy - 1) };
            (hc as u32).div_ceil(denom_minus1) / 2
        };
        (w as usize, h as usize)
    };

    // Pre-allocate every output band buffer.
    let mut bands_out: Vec<Vec<i32>> = (0..nbeta as u32)
        .map(|beta| {
            let k = layout[beta as usize];
            let (bw, bh) = band_dims(k.dx, k.dy, k.tau_x, k.tau_y);
            vec![0i32; bw * bh]
        })
        .collect();

    let nlx_u = nlx as u32;
    let nly_u = nly as u32;
    let dxy_min = nlx_u.min(nly_u);

    // The cascade band layout only carries the LL band for the deepest
    // level (NL,x, NL,y). Intermediate LL_{d,d} values (for d < NL,x or
    // dx < NL,x) are temporary and consumed by the next forward step.
    //
    // The inverse runs in reverse:
    //   1. Pure horizontal levels (nlx > nly): synthesise LL_{nlx-1,nly}
    //      from LL_{nlx,nly} + HL_{nlx,nly}, then LL_{nlx-2,nly} from
    //      LL_{nlx-1,nly} + HL_{nlx-1,nly}, … down to LL_{nly,nly}.
    //   2. Joint levels: synthesise LL_{d-1,d-1} from
    //      (LL_{d,d}, HL_{d,d}, LH_{d,d}, HH_{d,d}), d going from
    //      `dxy_min` down to 1.
    //
    // Forward direction inverts: phase 2 (joint) must split FIRST from
    // the picture down to LL_{dxy_min, dxy_min} producing
    // (HL/LH/HH)_{1..dxy_min}; then phase 1 (horizontal) splits LL_{dxy_min,nly}
    // (which equals LL_{nly,nly} when dxy_min == nly) into the
    // (LL, HL)_{nly+1..nlx} sequence.
    //
    // Strategy: we recursively split. Phase A walks d = 1 .. dxy_min,
    // each step taking the *current LL* and producing 4 bands; we keep
    // only HL/LH/HH (band slots) and feed LL forward as the next input.
    // Phase B walks dx = dxy_min+1 .. nlx, splitting horizontally; HL
    // gets a band slot, LL feeds forward. Final LL after both phases
    // lands in the LL_{nlx, nly} band slot.
    let mut current_ll = input.to_vec();
    let mut cur_w = wc;
    let mut cur_h = hc;

    // Phase A — joint levels.
    let mut d: u32 = 1;
    while d <= dxy_min {
        let hl_b = find_band(d, d, true, false).ok_or_else(|| {
            Error::invalid(format!(
                "JPEG XS DWT forward cascade: no HL_{{d,d}} band at d={d}"
            ))
        })?;
        let lh_b = find_band(d, d, false, true).ok_or_else(|| {
            Error::invalid(format!(
                "JPEG XS DWT forward cascade: no LH_{{d,d}} band at d={d}"
            ))
        })?;
        let hh_b = find_band(d, d, true, true).ok_or_else(|| {
            Error::invalid(format!(
                "JPEG XS DWT forward cascade: no HH_{{d,d}} band at d={d}"
            ))
        })?;
        // The split halves cur_w/cur_h with the standard ceil/floor rule.
        let ll_w = cur_w.div_ceil(2);
        let hl_w_d = cur_w / 2;
        let ll_h = cur_h.div_ceil(2);
        let lh_h = cur_h / 2;

        let mut ll_buf = vec![0i32; ll_w * ll_h];
        let mut hl_buf = vec![0i32; hl_w_d * ll_h];
        let mut lh_buf = vec![0i32; ll_w * lh_h];
        let mut hh_buf = vec![0i32; hl_w_d * lh_h];
        if cur_w < 2 || cur_h < 2 {
            return Err(Error::invalid(format!(
                "JPEG XS DWT: 2-D band dimensions ({cur_w}x{cur_h}) below the spec minimum of 2x2"
            )));
        }
        analyse_joint_level(
            cur_w,
            cur_h,
            &mut current_ll,
            &mut ll_buf,
            &mut hl_buf,
            &mut lh_buf,
            &mut hh_buf,
        );
        bands_out[hl_b] = hl_buf;
        bands_out[lh_b] = lh_buf;
        bands_out[hh_b] = hh_buf;
        current_ll = ll_buf;
        cur_w = ll_w;
        cur_h = ll_h;
        d += 1;
    }

    // Phase B — pure horizontal levels (only when nlx > nly).
    let mut dx_cur = dxy_min + 1;
    while dx_cur <= nlx_u {
        let hl_b = find_band(dx_cur, nly_u, true, false).ok_or_else(|| {
            Error::invalid(format!(
                "JPEG XS DWT forward cascade: no HL_{{dx={dx_cur},dy={nly_u}}} band"
            ))
        })?;
        let next_w = cur_w.div_ceil(2);
        let hl_w_d = cur_w / 2;
        let mut next_ll = vec![0i32; next_w * cur_h];
        let mut hl_buf = vec![0i32; hl_w_d * cur_h];
        for y in 0..cur_h {
            forward_horizontal_1d(
                &current_ll[y * cur_w..(y + 1) * cur_w],
                &mut next_ll[y * next_w..(y + 1) * next_w],
                &mut hl_buf[y * hl_w_d..(y + 1) * hl_w_d],
            )?;
        }
        bands_out[hl_b] = hl_buf;
        current_ll = next_ll;
        cur_w = next_w;
        dx_cur += 1;
    }

    // Final LL → LL_{nlx, nly} band slot.
    let ll_b = find_band(nlx_u, nly_u, false, false).ok_or_else(|| {
        Error::invalid(format!(
            "JPEG XS DWT forward cascade: no LL_{{NL,x={nlx},NL,y={nly}}} band"
        ))
    })?;
    bands_out[ll_b] = current_ll;
    Ok(bands_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Annex E.6: extension reflects through the boundary samples.
    #[test]
    fn extend_symmetric_reflects_through_boundary() {
        // Z = 6 real samples [10, 20, 30, 40, 50, 60].
        let z = 6;
        let mut buf = vec![0i32; z + 2 * PAD];
        for (i, v) in [10, 20, 30, 40, 50, 60].iter().enumerate() {
            buf[PAD + i] = *v;
        }
        extend_symmetric(&mut buf, z).unwrap();
        // Spec Annex E.6 Table E.5 (i = 1, 2):
        //   Left side: X[-1] = X[1] = 20, X[-2] = X[2] = 30.
        //   Right side: X[Z] = X[Z-2] = X[4] = 50,
        //               X[Z+1] = X[Z-3] = X[3] = 40.
        assert_eq!(buf[PAD - 1], 20);
        assert_eq!(buf[PAD - 2], 30);
        assert_eq!(buf[PAD + z], 50);
        assert_eq!(buf[PAD + z + 1], 40);
    }

    #[test]
    fn extend_symmetric_rejects_too_short() {
        let mut buf = vec![0i32; 1 + 2 * PAD];
        assert!(extend_symmetric(&mut buf, 1).is_err());
    }

    /// Hand-computed inverse on an 8-tap interleaved buffer of all-
    /// equal coefficients. With L = 100 at even indices and H = 0 at
    /// odd indices, the inverse 5/3 must reconstruct constant 100s
    /// (a flat low-pass with zero detail represents a flat signal).
    #[test]
    fn inverse_5_3_recovers_constant_signal() {
        let w: usize = 8;
        let low = vec![100i32; w.div_ceil(2)];
        let high = vec![0i32; w / 2];
        let mut out = vec![0i32; w];
        inverse_horizontal_1d(&low, &high, &mut out).unwrap();
        for v in &out {
            assert_eq!(
                *v, 100,
                "flat low-pass + zero high-pass should be flat: {out:?}"
            );
        }
    }

    /// Hand-computed inverse on a ramp: encode [0,1,2,3,4,5,6,7] with
    /// the forward 5/3 then check the inverse round-trip restores the
    /// ramp. Also pin one intermediate value to make sure the lifting
    /// signs are right.
    #[test]
    fn forward_inverse_round_trip_ramp_8() {
        let input: Vec<i32> = (0..8).collect();
        let w = input.len();
        let mut low = vec![0i32; w.div_ceil(2)];
        let mut high = vec![0i32; w / 2];
        forward_horizontal_1d(&input, &mut low, &mut high).unwrap();
        // Inside-the-ramp positions vanish (5/3 predicts odd samples
        // as the average of neighbours, which equals the centre on a
        // ramp). The last odd sample sees the right-edge symmetric
        // reflection X[8] = X[6] = 6 instead of X[8] = 8, so the
        // residual H[3] = 7 - (6+6)/2 = 1 — non-zero.
        assert_eq!(high, vec![0, 0, 0, 1]);
        let mut out = vec![0i32; w];
        inverse_horizontal_1d(&low, &high, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Round-trip a non-trivial 1-D signal of odd length to exercise
    /// the (W+1)/2 / W/2 split and the right-edge symmetric extension.
    #[test]
    fn forward_inverse_round_trip_odd_length() {
        let input = vec![5, 12, -7, 33, 100, -50, 4, 17, 22];
        let w = input.len();
        let mut low = vec![0i32; w.div_ceil(2)];
        let mut high = vec![0i32; w / 2];
        forward_horizontal_1d(&input, &mut low, &mut high).unwrap();
        let mut out = vec![0i32; w];
        inverse_horizontal_1d(&low, &high, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Sub-band sizing for an even-width row.
    #[test]
    fn forward_horizontal_rejects_wrong_split() {
        let input = vec![1, 2, 3, 4];
        let mut low = vec![0i32; 1]; // wrong: should be 2
        let mut high = vec![0i32; 2];
        assert!(forward_horizontal_1d(&input, &mut low, &mut high).is_err());
    }

    /// 2-D round-trip on an 8x8 ramp.
    #[test]
    fn forward_inverse_2d_round_trip_8x8_ramp() {
        let w = 8usize;
        let h = 8usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32) * 3 + (y as i32) * 5;
            }
        }
        let ll_w = w.div_ceil(2);
        let hl_w = w / 2;
        let ll_h = h.div_ceil(2);
        let lh_h = h / 2;
        let mut ll = vec![0i32; ll_w * ll_h];
        let mut hl = vec![0i32; hl_w * ll_h];
        let mut lh = vec![0i32; ll_w * lh_h];
        let mut hh = vec![0i32; hl_w * lh_h];
        forward_2d(w, h, &input, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        let mut out = vec![0i32; w * h];
        inverse_2d(w, h, &ll, &hl, &lh, &hh, &mut out).unwrap();
        assert_eq!(
            out, input,
            "2-D 5/3 DWT must be reversible on integer input"
        );
    }

    /// 2-D round-trip on a 16x16 image with both spatial frequencies
    /// and a discontinuity, which exercises non-zero high-pass bands
    /// in both directions.
    #[test]
    fn forward_inverse_2d_round_trip_16x16_pattern() {
        let w = 16usize;
        let h = 16usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                let base = (x as i32 * 7 + y as i32 * 11) % 256;
                let edge = if x == 8 { 1000 } else { 0 };
                input[y * w + x] = base + edge;
            }
        }
        let ll_w = w.div_ceil(2);
        let hl_w = w / 2;
        let ll_h = h.div_ceil(2);
        let lh_h = h / 2;
        let mut ll = vec![0i32; ll_w * ll_h];
        let mut hl = vec![0i32; hl_w * ll_h];
        let mut lh = vec![0i32; ll_w * lh_h];
        let mut hh = vec![0i32; hl_w * lh_h];
        forward_2d(w, h, &input, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        // Sanity: at least one high-pass band must be non-zero given
        // the discontinuity at x == 8.
        assert!(
            hl.iter().any(|v| *v != 0) || hh.iter().any(|v| *v != 0),
            "patterned input must produce non-trivial high-pass bands",
        );
        let mut out = vec![0i32; w * h];
        inverse_2d(w, h, &ll, &hl, &lh, &hh, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// 2-D round-trip on an asymmetric (non-square, odd dimensions)
    /// shape — exercises the edge case where the LL band has more
    /// rows than the LH band.
    #[test]
    fn forward_inverse_2d_round_trip_odd_dims() {
        let w = 7usize;
        let h = 5usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32 - 3) * (y as i32 - 2) + 50;
            }
        }
        let ll_w = w.div_ceil(2);
        let hl_w = w / 2;
        let ll_h = h.div_ceil(2);
        let lh_h = h / 2;
        let mut ll = vec![0i32; ll_w * ll_h];
        let mut hl = vec![0i32; hl_w * ll_h];
        let mut lh = vec![0i32; ll_w * lh_h];
        let mut hh = vec![0i32; hl_w * lh_h];
        forward_2d(w, h, &input, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        let mut out = vec![0i32; w * h];
        inverse_2d(w, h, &ll, &hl, &lh, &hh, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Multi-level cascade: round-trip an 8x8 ramp via NL,x=NL,y=2.
    /// We build the 7 sub-bands by running `forward_2d` twice (once on
    /// the original, then again on the LL band of that), then feed
    /// `inverse_cascade_2d` and expect the original ramp back.
    #[test]
    fn cascade_2_2_round_trip_8x8() {
        let w = 8usize;
        let h = 8usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32) * 3 + (y as i32) * 5 + 7;
            }
        }
        // Level-1 forward.
        let l1_w = w.div_ceil(2);
        let h1_w = w / 2;
        let l1_h = h.div_ceil(2);
        let h1_h = h / 2;
        let mut ll1 = vec![0i32; l1_w * l1_h];
        let mut hl1 = vec![0i32; h1_w * l1_h];
        let mut lh1 = vec![0i32; l1_w * h1_h];
        let mut hh1 = vec![0i32; h1_w * h1_h];
        forward_2d(w, h, &input, &mut ll1, &mut hl1, &mut lh1, &mut hh1).unwrap();
        // Level-2 forward on LL1.
        let l2_w = l1_w.div_ceil(2);
        let h2_w = l1_w / 2;
        let l2_h = l1_h.div_ceil(2);
        let h2_h = l1_h / 2;
        let mut ll2 = vec![0i32; l2_w * l2_h];
        let mut hl2 = vec![0i32; h2_w * l2_h];
        let mut lh2 = vec![0i32; l2_w * h2_h];
        let mut hh2 = vec![0i32; h2_w * h2_h];
        forward_2d(l1_w, l1_h, &ll1, &mut ll2, &mut hl2, &mut lh2, &mut hh2).unwrap();

        // Cascade input: 7 bands in β order (LL2,2, HL2,2, LH2,2, HH2,2,
        // HL1,1, LH1,1, HH1,1).
        let bands = vec![ll2, hl2, lh2, hh2, hl1, lh1, hh1];
        let mut out = vec![0i32; w * h];
        inverse_cascade_2d(w, h, 2, 2, &bands, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Multi-level cascade: round-trip a 16x16 image via NL,x=NL,y=3.
    #[test]
    fn cascade_3_3_round_trip_16x16() {
        let w = 16usize;
        let h = 16usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32 * 7 + y as i32 * 11) % 256;
            }
        }
        // Recursively forward-decompose 3 times.
        let mut prev = input.clone();
        let mut prev_w = w;
        let mut prev_h = h;
        let mut bands_high: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)> = Vec::new();
        for _ in 0..3 {
            let lw = prev_w.div_ceil(2);
            let hw = prev_w / 2;
            let lh = prev_h.div_ceil(2);
            let hh_ = prev_h / 2;
            let mut ll = vec![0i32; lw * lh];
            let mut hl = vec![0i32; hw * lh];
            let mut lh_b = vec![0i32; lw * hh_];
            let mut hh_b = vec![0i32; hw * hh_];
            forward_2d(
                prev_w, prev_h, &prev, &mut ll, &mut hl, &mut lh_b, &mut hh_b,
            )
            .unwrap();
            bands_high.push((hl, lh_b, hh_b));
            prev = ll;
            prev_w = lw;
            prev_h = lh;
        }
        // bands_high is in level-1, level-2, level-3 order; the cascade
        // expects β order: LL3,3 first, then HL3 LH3 HH3 (level 3),
        // then HL2 LH2 HH2 (level 2), then HL1 LH1 HH1 (level 1).
        let mut bands: Vec<Vec<i32>> = Vec::with_capacity(10);
        bands.push(prev.clone()); // LL3,3
        for level in (0..3).rev() {
            let (ref hl, ref lh, ref hh) = bands_high[level];
            bands.push(hl.clone());
            bands.push(lh.clone());
            bands.push(hh.clone());
        }
        let mut out = vec![0i32; w * h];
        inverse_cascade_2d(w, h, 3, 3, &bands, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Asymmetric NL,x=2 NL,y=1: 1 horizontal-only level, then 1
    /// joint level. Round-trip on an 8×4 image.
    #[test]
    fn cascade_2_1_round_trip_8x4() {
        let w = 8usize;
        let h = 4usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32 - 4) * (y as i32 - 2) + 100;
            }
        }
        // Joint level 1 (NL,x=NL,y=1).
        let l1_w = w.div_ceil(2);
        let h1_w = w / 2;
        let l1_h = h.div_ceil(2);
        let h1_h = h / 2;
        let mut ll1 = vec![0i32; l1_w * l1_h];
        let mut hl1 = vec![0i32; h1_w * l1_h];
        let mut lh1 = vec![0i32; l1_w * h1_h];
        let mut hh1 = vec![0i32; h1_w * h1_h];
        forward_2d(w, h, &input, &mut ll1, &mut hl1, &mut lh1, &mut hh1).unwrap();
        // Horizontal-only level on LL1 (in horizontal direction only).
        // Result LL2 has width l1_w.div_ceil(2) and height l1_h.
        let l2_w = l1_w.div_ceil(2);
        let h2_w = l1_w / 2;
        let mut ll2 = vec![0i32; l2_w * l1_h];
        let mut hl2 = vec![0i32; h2_w * l1_h];
        for y in 0..l1_h {
            forward_horizontal_1d(
                &ll1[y * l1_w..(y + 1) * l1_w],
                &mut ll2[y * l2_w..(y + 1) * l2_w],
                &mut hl2[y * h2_w..(y + 1) * h2_w],
            )
            .unwrap();
        }
        // β order for NL,x=2 NL,y=1: per beta_to_band_key,
        //   β=0: (dx=2, dy=1, false, false) → LL2,1
        //   β=1: (dx=2, dy=1, true,  false) → HL2,1
        //   β=2: (dx=1, dy=1, true,  false) → HL1,1
        //   β=3: (dx=1, dy=1, false, true ) → LH1,1
        //   β=4: (dx=1, dy=1, true,  true ) → HH1,1
        let bands = vec![ll2, hl2, hl1, lh1, hh1];
        let mut out = vec![0i32; w * h];
        inverse_cascade_2d(w, h, 2, 1, &bands, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Cascade with NL,x=NL,y=1 must agree with the single-level
    /// `inverse_2d` directly. Sanity check that the orchestrator
    /// degrades to the round-2 path on the (1, 1) configuration.
    #[test]
    fn cascade_1_1_matches_inverse_2d() {
        let w = 8usize;
        let h = 8usize;
        let mut input = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                input[y * w + x] = (x as i32 + y as i32) * 13;
            }
        }
        let l1_w = w.div_ceil(2);
        let h1_w = w / 2;
        let l1_h = h.div_ceil(2);
        let h1_h = h / 2;
        let mut ll = vec![0i32; l1_w * l1_h];
        let mut hl = vec![0i32; h1_w * l1_h];
        let mut lh = vec![0i32; l1_w * h1_h];
        let mut hh = vec![0i32; h1_w * h1_h];
        forward_2d(w, h, &input, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        let bands = vec![ll, hl, lh, hh];
        let mut out = vec![0i32; w * h];
        inverse_cascade_2d(w, h, 1, 1, &bands, &mut out).unwrap();
        assert_eq!(out, input);
    }

    /// Hand-built fixture: 8 sample row [0, 0, 0, 100, 0, 0, 0, 0]
    /// produces specific known values from the forward 5/3, recorded
    /// here to detect accidental reorderings of the lifting steps.
    /// The impulse sits at sample index 3, so it stays in the
    /// high-pass band (high[1]) and feeds the adjacent low-pass
    /// samples through the even-update step.
    #[test]
    fn forward_5_3_impulse_pin() {
        let input = vec![0, 0, 0, 100, 0, 0, 0, 0];
        let w = input.len();
        let mut low = vec![0i32; w.div_ceil(2)];
        let mut high = vec![0i32; w / 2];
        forward_horizontal_1d(&input, &mut low, &mut high).unwrap();
        // Hand calculation with symmetric extension X[-1]=X[1]=0,
        // X[-2]=X[2]=0, X[8]=X[6]=0, X[9]=X[5]=0.
        //
        // Step 1 (odd) Y[i] = X[i] - (X[i-1]+X[i+1])>>1:
        //   Y[-1] = 0, Y[1] = 0, Y[3] = 100, Y[5] = 0, Y[7] = 0.
        // Step 2 (even) Y[i] = X[i] + (Y[i-1]+Y[i+1]+2)>>2:
        //   Y[0] = 0 + (0+0+2)>>2     = 0
        //   Y[2] = 0 + (0+100+2)>>2   = 25
        //   Y[4] = 0 + (100+0+2)>>2   = 25
        //   Y[6] = 0 + (0+0+2)>>2     = 0
        //
        // Even indices are de-interleaved into `low`, odd into `high`.
        assert_eq!(low, vec![0, 25, 25, 0]);
        assert_eq!(high, vec![0, 100, 0, 0]);
        // Round-trip back to the impulse.
        let mut out = vec![0i32; w];
        inverse_horizontal_1d(&low, &high, &mut out).unwrap();
        assert_eq!(out, input);
    }

    // -----------------------------------------------------------------
    // Pass order of a joint level (Annex E.2 Table E.1 / E.9 Table E.8).
    // -----------------------------------------------------------------

    /// Split a row-major `h × w` block with [`forward_2d`].
    fn split_2d(w: usize, h: usize, input: &[i32]) -> [Vec<i32>; 4] {
        let (ll_w, hl_w, ll_h, lh_h) = (w.div_ceil(2), w / 2, h.div_ceil(2), h / 2);
        let mut ll = vec![0i32; ll_w * ll_h];
        let mut hl = vec![0i32; hl_w * ll_h];
        let mut lh = vec![0i32; ll_w * lh_h];
        let mut hh = vec![0i32; hl_w * lh_h];
        forward_2d(w, h, input, &mut ll, &mut hl, &mut lh, &mut hh).unwrap();
        [ll, hl, lh, hh]
    }

    fn merge_2d(w: usize, h: usize, b: &[Vec<i32>; 4]) -> Vec<i32> {
        let mut out = vec![0i32; w * h];
        inverse_2d(w, h, &b[0], &b[1], &b[2], &b[3], &mut out).unwrap();
        out
    }

    /// 2×2 block worked by hand from Tables E.8 / E.11 / E.10 / E.12.
    ///
    /// Input rows `[15, 14]`, `[15, 12]`. For `Z = 2` the extension of
    /// Table E.5 gives `X[-1] = X[1]`, `X[2] = X[0]`, so the filter of
    /// Table E.12 reduces to `H = X[1] - X[0]`, `L = X[0] + ((2H + 2) >> 2)`.
    ///
    /// Vertical first (Table E.8 call 1):
    ///   column 0 `(15, 15)`: `H = 0`,  `L = 15 + (2 >> 2)  = 15`
    ///   column 1 `(14, 12)`: `H = -2`, `L = 14 + (-2 >> 2) = 13`
    ///   low row `[15, 13]`, high row `[0, -2]`.
    /// Horizontal (calls 2 and 3):
    ///   low row:  `HL = 13 - 15 = -2`, `LL = 15 + (-2 >> 2) = 14`
    ///   high row: `HH = -2 - 0  = -2`, `LH = 0  + (-2 >> 2) = -1`
    ///
    /// The other pass order (rows first) would give `LL = 15`: rows
    /// `(15, 14) → L 15, H -1` and `(15, 12) → L 14, H -3`, then column
    /// `(15, 14) → L = 15 + (-2 + 2 >> 2) = 15`. The block therefore pins
    /// the order.
    #[test]
    fn forward_2x2_hand_computed_is_vertical_then_horizontal() {
        let b = split_2d(2, 2, &[15, 14, 15, 12]);
        assert_eq!(b, [vec![14], vec![-2], vec![-1], vec![-2]]);
    }

    /// Inverse of the block above worked by hand from Tables E.1 / E.3 /
    /// E.4 / E.6. For `Z = 2` Table E.6 reduces to
    /// `Y[0] = X[0] - ((2 X[1] + 2) >> 2)`, `Y[1] = X[1] + Y[0]`.
    ///
    /// Horizontal first (Table E.1 calls 1 and 2):
    ///   `(LL, HL) = (14, -2)`: `Y0 = 14 - (-2 >> 2) = 15`, `Y1 = -2 + 15 = 13`
    ///   `(LH, HH) = (-1, -2)`: `Y0 = -1 - (-2 >> 2) = 0`,  `Y1 = -2 + 0  = -2`
    /// Vertical (call 3):
    ///   column 0 `(15, 0)`:  `Y0 = 15 - (2 >> 2)  = 15`, `Y1 = 0 + 15  = 15`
    ///   column 1 `(13, -2)`: `Y0 = 13 - (-2 >> 2) = 14`, `Y1 = -2 + 14 = 12`
    #[test]
    fn inverse_2x2_hand_computed_is_horizontal_then_vertical() {
        let b = [vec![14], vec![-2], vec![-1], vec![-2]];
        assert_eq!(merge_2d(2, 2, &b), vec![15, 14, 15, 12]);
    }

    /// The inverse order is observable on its own: these four
    /// coefficients reconstruct differently under the two orders.
    ///
    /// Table E.1 order — rows `(1, 0) → (1, 1)` and `(1, -1) → (1, 0)`
    /// (`Y0 = 1 - (2 >> 2) = 1`, `Y1 = 0 + 1`; `Y0 = 1 - (0 >> 2) = 1`,
    /// `Y1 = -1 + 1`), then columns `(1, 1) → (0, 1)`
    /// (`Y0 = 1 - (4 >> 2) = 0`, `Y1 = 1 + 0`) and `(1, 0) → (1, 1)`.
    /// Columns first would end in `[0, 0, 1, 0]` instead.
    #[test]
    fn inverse_2x2_order_is_observable() {
        let b = [vec![1], vec![0], vec![1], vec![-1]];
        assert_eq!(merge_2d(2, 2, &b), vec![0, 1, 1, 1]);
    }

    /// 3×3 block worked by hand (odd sizes: `Z = 3`, Table E.5 gives
    /// `X[-1] = X[1]`, `X[3] = X[1]`, so `H = X[1] - ((X[0] + X[2]) >> 1)`,
    /// `L0 = X[0] + ((2H + 2) >> 2)`, `L1 = X[2] + ((2H + 2) >> 2)`).
    ///
    /// Input rows `[12, 3, 0]`, `[15, 5, 11]`, `[13, 6, 15]`.
    /// Vertical first:
    ///   column `(12, 15, 13)`: `H = 15 - 12 = 3`, `L = 12 + 2, 13 + 2 = 14, 15`
    ///   column `(3, 5, 6)`:    `H = 5 - 4  = 1`, `L = 3 + 1,  6 + 1  = 4, 7`
    ///   column `(0, 11, 15)`:  `H = 11 - 7 = 4`, `L = 0 + 2,  15 + 2 = 2, 17`
    ///   low rows `[14, 4, 2]`, `[15, 7, 17]`; high row `[3, 1, 4]`.
    /// Horizontal:
    ///   `[14, 4, 2]`:  `H = 4 - 8  = -4`, `L = 14 - 2, 2 - 2   = 12, 0`
    ///   `[15, 7, 17]`: `H = 7 - 16 = -9`, `L = 15 - 4, 17 - 4  = 11, 13`
    ///   `[3, 1, 4]`:   `H = 1 - 3  = -2`, `L = 3 - 1,  4 - 1   = 2, 3`
    #[test]
    fn forward_and_inverse_3x3_hand_computed() {
        let input = [12, 3, 0, 15, 5, 11, 13, 6, 15];
        let b = split_2d(3, 3, &input);
        assert_eq!(b, [vec![12, 0, 11, 13], vec![-4, -9], vec![2, 3], vec![-2]]);
        assert_eq!(merge_2d(3, 3, &b), input.to_vec());
    }

    // -----------------------------------------------------------------
    // Literal transcription of Annex E as an independent model: one
    // function per table, `X` / `Y` scratch arrays with the two-sample
    // margin of Table E.5, bands addressed by `(dx, dy, τx, τy)`.
    // -----------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    struct ModelBand {
        w: usize,
        h: usize,
        t: Vec<i64>,
    }

    /// Table E.5 on an array whose index 0 sits at offset 2.
    fn model_extend(x: &mut [i64], z: usize) {
        for i in 1..=2usize {
            x[2 - i] = x[2 + i];
            x[2 + z + i - 1] = x[2 + z - i - 1];
        }
    }

    /// Table E.6. Returns `Y[0 .. Z]`.
    fn model_inverse_filter(mut x: Vec<i64>, z: usize) -> Vec<i64> {
        model_extend(&mut x, z);
        let mut y = x.clone();
        for i in (0..z + 1).step_by(2) {
            y[2 + i] = x[2 + i] - ((x[2 + i - 1] + x[2 + i + 1] + 2) >> 2);
        }
        for i in (1..z).step_by(2) {
            y[2 + i] = x[2 + i] + ((y[2 + i - 1] + y[2 + i + 1]) >> 1);
        }
        y[2..2 + z].to_vec()
    }

    /// Table E.12. Returns `Y[0 .. Z]`.
    fn model_forward_filter(mut x: Vec<i64>, z: usize) -> Vec<i64> {
        model_extend(&mut x, z);
        let mut y = x.clone();
        // i = -1, 1, 3, … < Z + 1, written with the offset folded in.
        for k in (1..z + 3).step_by(2) {
            y[k] = x[k] - ((x[k - 1] + x[k + 1]) >> 1);
        }
        for i in (0..z).step_by(2) {
            y[2 + i] = x[2 + i] + ((y[2 + i - 1] + y[2 + i + 1] + 2) >> 2);
        }
        y[2..2 + z].to_vec()
    }

    /// Table E.3.
    fn model_hor_inverse(low: &ModelBand, high: &ModelBand) -> ModelBand {
        let (w, h) = (low.w + high.w, low.h);
        assert_eq!(high.h, h);
        let mut out = ModelBand {
            w,
            h,
            t: vec![0; w * h],
        };
        for y in 0..h {
            let mut x = vec![0i64; w + 4];
            for c in 0..w {
                x[2 + c] = if c % 2 == 0 {
                    low.t[y * low.w + c / 2]
                } else {
                    high.t[y * high.w + c / 2]
                };
            }
            let r = model_inverse_filter(x, w);
            out.t[y * w..(y + 1) * w].copy_from_slice(&r);
        }
        out
    }

    /// Table E.4.
    fn model_ver_inverse(low: &ModelBand, high: &ModelBand) -> ModelBand {
        let (w, h) = (low.w, low.h + high.h);
        assert_eq!(high.w, w);
        let mut out = ModelBand {
            w,
            h,
            t: vec![0; w * h],
        };
        for c in 0..w {
            let mut x = vec![0i64; h + 4];
            for y in 0..h {
                x[2 + y] = if y % 2 == 0 {
                    low.t[(y / 2) * w + c]
                } else {
                    high.t[(y / 2) * w + c]
                };
            }
            let r = model_inverse_filter(x, h);
            for (y, v) in r.into_iter().enumerate() {
                out.t[y * w + c] = v;
            }
        }
        out
    }

    /// Table E.10.
    fn model_hor_forward(src: &ModelBand) -> (ModelBand, ModelBand) {
        let (w, h) = (src.w, src.h);
        let (lw, hw) = (w.div_ceil(2), w / 2);
        let mut low = ModelBand {
            w: lw,
            h,
            t: vec![0; lw * h],
        };
        let mut high = ModelBand {
            w: hw,
            h,
            t: vec![0; hw * h],
        };
        for y in 0..h {
            let mut x = vec![0i64; w + 4];
            x[2..2 + w].copy_from_slice(&src.t[y * w..(y + 1) * w]);
            let r = model_forward_filter(x, w);
            for (c, v) in r.into_iter().enumerate() {
                if c % 2 == 0 {
                    low.t[y * lw + c / 2] = v;
                } else {
                    high.t[y * hw + c / 2] = v;
                }
            }
        }
        (low, high)
    }

    /// Table E.11.
    fn model_ver_forward(src: &ModelBand) -> (ModelBand, ModelBand) {
        let (w, h) = (src.w, src.h);
        let (lh, hh) = (h.div_ceil(2), h / 2);
        let mut low = ModelBand {
            w,
            h: lh,
            t: vec![0; w * lh],
        };
        let mut high = ModelBand {
            w,
            h: hh,
            t: vec![0; w * hh],
        };
        for c in 0..w {
            let mut x = vec![0i64; h + 4];
            for y in 0..h {
                x[2 + y] = src.t[y * w + c];
            }
            let r = model_forward_filter(x, h);
            for (y, v) in r.into_iter().enumerate() {
                if y % 2 == 0 {
                    low.t[(y / 2) * w + c] = v;
                } else {
                    high.t[(y / 2) * w + c] = v;
                }
            }
        }
        (low, high)
    }

    /// Annex B.3 filter index for `NL,x ≥ NL,y`.
    fn model_beta(nlx: usize, nly: usize, dx: usize, dy: usize, tx: usize, ty: usize) -> usize {
        if dx > dy {
            nlx - dx + tx
        } else {
            nlx - nly + tx + 2 * ty + 3 * (nly - dy)
        }
    }

    /// Table E.8 for one component. Bands come back in β order.
    fn model_forward(wc: usize, hc: usize, nlx: usize, nly: usize, o: &[i32]) -> Vec<ModelBand> {
        let nbeta = 2 * nlx.min(nly) + nlx.max(nly) + 1;
        let mut bands: Vec<Option<ModelBand>> = vec![None; nbeta];
        let mut ll = ModelBand {
            w: wc,
            h: hc,
            t: o.iter().map(|&v| v as i64).collect(),
        };
        let dmin = nlx.min(nly);
        for d in 1..=dmin {
            let (ll_v, lh_v) = model_ver_forward(&ll);
            let (ll_d, hl_d) = model_hor_forward(&ll_v);
            let (lh_d, hh_d) = model_hor_forward(&lh_v);
            bands[model_beta(nlx, nly, d, d, 1, 0)] = Some(hl_d);
            bands[model_beta(nlx, nly, d, d, 0, 1)] = Some(lh_d);
            bands[model_beta(nlx, nly, d, d, 1, 1)] = Some(hh_d);
            ll = ll_d;
        }
        for dx in dmin + 1..=nlx {
            let (ll_d, hl_d) = model_hor_forward(&ll);
            bands[model_beta(nlx, nly, dx, nly, 1, 0)] = Some(hl_d);
            ll = ll_d;
        }
        bands[0] = Some(ll);
        bands
            .into_iter()
            .map(|b| b.expect("every β assigned"))
            .collect()
    }

    /// Table E.1 for one component.
    fn model_inverse(nlx: usize, nly: usize, bands: &[ModelBand]) -> ModelBand {
        let dmin = nlx.min(nly);
        let mut ll = bands[0].clone();
        for dx in (dmin + 1..=nlx).rev() {
            ll = model_hor_inverse(&ll, &bands[model_beta(nlx, nly, dx, nly, 1, 0)]);
        }
        for d in (1..=dmin).rev() {
            let ll_v = model_hor_inverse(&ll, &bands[model_beta(nlx, nly, d, d, 1, 0)]);
            let lh_v = model_hor_inverse(
                &bands[model_beta(nlx, nly, d, d, 0, 1)],
                &bands[model_beta(nlx, nly, d, d, 1, 1)],
            );
            ll = model_ver_inverse(&ll_v, &lh_v);
        }
        ll
    }

    /// Deterministic sample generator: `bits`-wide signed values.
    fn model_samples(n: usize, bits: u32, seed: u64) -> Vec<i32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let span = 1i64 << bits;
                ((s >> 11) as i64 % span - span / 2) as i32
            })
            .collect()
    }

    /// Whether every 1-D pass of the `(nlx, nly)` cascade sees at least
    /// two samples (Annex E.6 NOTE).
    fn cascade_is_defined(wc: usize, hc: usize, nlx: usize, nly: usize) -> bool {
        let (mut w, mut h) = (wc, hc);
        for _ in 0..nly {
            if h < 2 {
                return false;
            }
            h = h.div_ceil(2);
        }
        for _ in 0..nlx {
            if w < 2 {
                return false;
            }
            w = w.div_ceil(2);
        }
        wc >= 2 && hc >= 2
    }

    /// The production cascades equal the literal Annex E model, band for
    /// band and sample for sample, for every decomposition shape
    /// `NL,y ≤ NL,x ≤ 5`, every size 2..=17 in both directions (odd and
    /// even), and sample magnitudes from 1 bit to the 20-bit wavelet
    /// precision; and the inverse undoes the forward exactly.
    #[test]
    fn cascades_match_literal_annex_e_model() {
        let mut checked = 0usize;
        for nlx in 0..=5usize {
            for nly in 0..=nlx.min(3) {
                for wc in 2..=17usize {
                    for hc in 2..=17usize {
                        if !cascade_is_defined(wc, hc, nlx, nly) {
                            continue;
                        }
                        let bits = [1u32, 8, 12, 16, 20][(wc + 3 * hc + nlx) % 5];
                        let seed = (wc * 131 + hc * 17 + nlx * 7 + nly) as u64;
                        let input = model_samples(wc * hc, bits, seed);
                        let model = model_forward(wc, hc, nlx, nly, &input);
                        let ours =
                            forward_cascade_2d(wc, hc, nlx as u8, nly as u8, &input).unwrap();
                        assert_eq!(ours.len(), model.len());
                        for (beta, (a, m)) in ours.iter().zip(&model).enumerate() {
                            let m32: Vec<i32> = m.t.iter().map(|&v| v as i32).collect();
                            assert_eq!(
                                a, &m32,
                                "forward β={beta} {wc}x{hc} NL={nlx}/{nly} bits={bits}"
                            );
                        }
                        // Inverse on coefficients that are not the image
                        // of any picture (independent random bands), so
                        // the inverse is checked on its own.
                        let free: Vec<ModelBand> = model
                            .iter()
                            .enumerate()
                            .map(|(beta, m)| ModelBand {
                                w: m.w,
                                h: m.h,
                                t: model_samples(m.t.len(), bits, seed * 31 + beta as u64)
                                    .into_iter()
                                    .map(i64::from)
                                    .collect(),
                            })
                            .collect();
                        let free32: Vec<Vec<i32>> = free
                            .iter()
                            .map(|m| m.t.iter().map(|&v| v as i32).collect())
                            .collect();
                        let want = model_inverse(nlx, nly, &free);
                        let mut got = vec![0i32; wc * hc];
                        inverse_cascade_2d(wc, hc, nlx as u8, nly as u8, &free32, &mut got)
                            .unwrap();
                        let want32: Vec<i32> = want.t.iter().map(|&v| v as i32).collect();
                        assert_eq!(got, want32, "inverse {wc}x{hc} NL={nlx}/{nly} bits={bits}");
                        // Identity: inverse ∘ forward.
                        let mut back = vec![0i32; wc * hc];
                        inverse_cascade_2d(wc, hc, nlx as u8, nly as u8, &ours, &mut back).unwrap();
                        assert_eq!(back, input, "identity {wc}x{hc} NL={nlx}/{nly}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 2000, "sweep covered only {checked} shapes");
    }

    /// With `Fq` fractional bits every coefficient entering the inverse
    /// is a multiple of `2^Fq`, each horizontal-only level consumes one
    /// bit of that headroom and each joint level two, so for
    /// `NL,x + NL,y ≤ Fq` no lifting step ever rounds and the inverse is
    /// linear — the pass order cannot be observed. This is why decoding
    /// the regular `Fq = 8` configurations never depended on the order,
    /// while `Fq = 0` (lossless) and `Fq = 6` with seven levels do.
    #[test]
    fn inverse_is_exactly_linear_under_fractional_headroom() {
        let (wc, hc, nlx, nly) = (24usize, 12usize, 5usize, 2usize);
        let shape = model_forward(wc, hc, nlx, nly, &vec![0i32; wc * hc]);
        let rand_bands = |seed: u64, fq: u32| -> Vec<ModelBand> {
            shape
                .iter()
                .enumerate()
                .map(|(beta, m)| ModelBand {
                    w: m.w,
                    h: m.h,
                    t: model_samples(m.t.len(), 10, seed + beta as u64)
                        .into_iter()
                        .map(|v| i64::from(v) << fq)
                        .collect(),
                })
                .collect()
        };
        let add = |a: &[ModelBand], b: &[ModelBand]| -> Vec<ModelBand> {
            a.iter()
                .zip(b)
                .map(|(p, q)| ModelBand {
                    w: p.w,
                    h: p.h,
                    t: p.t.iter().zip(&q.t).map(|(x, y)| x + y).collect(),
                })
                .collect()
        };
        // Fq = 8 ≥ NL,x + NL,y = 7: additive.
        let (a, b) = (rand_bands(1, 8), rand_bands(100, 8));
        let (ra, rb) = (model_inverse(nlx, nly, &a), model_inverse(nlx, nly, &b));
        let rs = model_inverse(nlx, nly, &add(&a, &b));
        let sum: Vec<i64> = ra.t.iter().zip(&rb.t).map(|(x, y)| x + y).collect();
        assert_eq!(rs.t, sum);
        // Fq = 0: rounding makes it non-additive.
        let (a, b) = (rand_bands(1, 0), rand_bands(100, 0));
        let (ra, rb) = (model_inverse(nlx, nly, &a), model_inverse(nlx, nly, &b));
        let rs = model_inverse(nlx, nly, &add(&a, &b));
        let sum: Vec<i64> = ra.t.iter().zip(&rb.t).map(|(x, y)| x + y).collect();
        assert_ne!(rs.t, sum);
    }
}
