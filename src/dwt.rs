//! Reversible 5/3 discrete wavelet transform (ISO/IEC 21122-1:2022,
//! Annex E).
//!
//! This module implements the single-level forward and inverse 2-D
//! reversible 5/3 DWT used by every JPEG XS profile. The decomposition
//! itself (which bands feed which bands across multiple levels) lives
//! in the slice walker — that walker isn't ready yet, so this round
//! exposes only the per-band 1-D and 2-D primitives.
//!
//! Spec mapping:
//!
//! * [`extend_symmetric`] — Annex E.6, Table E.5 (whole-sample even
//!   symmetric reflection: `X[-i] = X[i]`, `X[Z+i-1] = X[Z-i-1]` for
//!   `i = 1, 2`).
//! * [`inverse_filter_1d`] — Annex E.7, Table E.6 (inverse 5/3 lifting:
//!   even samples reconstructed from low-pass via `Y[i] = X[i] -
//!   ((X[i-1] + X[i+1] + 2) >> 2)`, then odd samples via `Y[i] = X[i]
//!   + ((Y[i-1] + Y[i+1]) >> 1)`).
//! * [`forward_filter_1d`] — Annex E.13, Table E.12 (forward 5/3
//!   lifting: odd samples first via `Y[i] = X[i] - ((X[i-1] + X[i+1])
//!   >> 1)`, then even samples via `Y[i] = X[i] + ((Y[i-1] + Y[i+1] +
//!   2) >> 2)`).
//! * [`inverse_horizontal_1d`] / [`forward_horizontal_1d`] — Annex E.4
//!   / E.11, single-row bridge between the interleaved L|H|L|H ordering
//!   used by the filter and the de-interleaved low-pass / high-pass
//!   bands stored in coefficient memory.
//! * [`inverse_vertical_1d`] / [`forward_vertical_1d`] — Annex E.5 /
//!   E.12, the column equivalent.
//! * [`inverse_2d`] / [`forward_2d`] — single-level 2-D composition
//!   (vertical first on the inverse to mirror the encoder's
//!   horizontal-first decomposition, per Annex E.2 / E.9).
//!
//! Boundary handling. Annex E.6 NOTE guarantees every band is at
//! least two coefficients wide / tall, because of the picture-header
//! constraints on `Wf`, `Hf`, `Cw`. The extension code rejects shorter
//! inputs. The pad of two samples on each side of the working buffer
//! is exactly what the 5/3 filter taps reach.
//!
//! Allocation. The DWT works on caller-owned slices. The only
//! internal heap use is the `Vec<i32>` working buffer for one row /
//! column at a time; its capacity is `len + 4` where `len` is the band
//! dimension supplied by the caller. The per-call allocation is
//! therefore bounded by the picture-header geometry the codestream
//! parser already validated, never by an unbounded length on the
//! wire.

use oxideav_core::{Error, Result};

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
    // Left edge: X[-1] = X[1], X[-2] = X[2].
    // PAD-relative indices: buf[PAD-1] = buf[PAD+1], buf[PAD-2] = buf[PAD+2].
    buf[PAD - 1] = buf[PAD + 1];
    buf[PAD - 2] = buf[PAD + 2];
    // Right edge: X[Z] = X[Z-2], X[Z+1] = X[Z-3].
    // PAD-relative: buf[PAD+Z] = buf[PAD+Z-2], buf[PAD+Z+1] = buf[PAD+Z-3].
    buf[PAD + z] = buf[PAD + z - 2];
    // For Z == 2 the second reflection step would read X[-1] which we
    // just wrote above, and the spec's loop guard `i <= 2` covers that
    // case identically; the `z >= 3` guard is therefore only on the
    // index computation.
    if z >= 3 {
        buf[PAD + z + 1] = buf[PAD + z - 3];
    } else {
        // Mirror the left-extension policy: reflect through the
        // already-extended sample, equivalent to the spec loop running
        // once more after the first reflection.
        buf[PAD + z + 1] = buf[PAD - 1];
    }
    Ok(())
}

/// Annex E.7, Table E.6 — inverse 5/3 reversible filter. The input
/// `buf` holds interleaved low-pass / high-pass samples in
/// `buf[PAD .. PAD+Z]` (low-pass at even indices, high-pass at odd),
/// with `extend_symmetric` already applied to the pad slots. The
/// output overwrites the same range with reconstructed samples.
fn inverse_filter_1d(buf: &mut [i32], z: usize) {
    // Index helper: read sample at logical position `i` (which may be
    // `-2 ..= Z+1`) by adding the PAD shift.
    let pad = PAD as isize;
    let z_i = z as isize;

    // Step 1 — even samples: Y[i] = X[i] - ((X[i-1] + X[i+1] + 2) >> 2)
    // for i = 0, 2, ..., last even index < Z+1. The spec loop runs
    // `for(i=0; i<Z+1; i+=2)`, so when Z is even the last update is at
    // i = Z (in the right pad slot). That synthetic write is required:
    // step 2 below reads Y[Z] when Z is even (e.g. for Z=8, the i=7
    // odd-sample update reads Y[8]).
    let mut i: isize = 0;
    while i < z_i + 1 {
        let left = buf[(pad + i - 1) as usize];
        let right = buf[(pad + i + 1) as usize];
        let centre_idx = (pad + i) as usize;
        buf[centre_idx] -= (left + right + 2) >> 2;
        i += 2;
    }

    // Step 2 — odd samples: Y[i] = X[i] + ((Y[i-1] + Y[i+1]) >> 1)
    // for i = 1, 3, ..., Z-1.
    let mut i: isize = 1;
    while i < z_i {
        let left = buf[(pad + i - 1) as usize];
        let right = buf[(pad + i + 1) as usize];
        buf[(pad + i) as usize] += (left + right) >> 1;
        i += 2;
    }
}

/// Annex E.13, Table E.12 — forward 5/3 reversible filter. The input
/// `buf` holds raw samples in `buf[PAD .. PAD+Z]` with
/// `extend_symmetric` already applied to the pad slots. The output
/// overwrites the same range with interleaved low-pass / high-pass
/// coefficients (low-pass at even indices, high-pass at odd).
fn forward_filter_1d(buf: &mut [i32], z: usize) {
    let pad = PAD as isize;
    let z_i = z as isize;

    // Step 1 — odd samples first: Y[i] = X[i] - ((X[i-1] + X[i+1]) >> 1)
    // for i = -1, 1, 3, ..., last odd <= Z. The spec loop is
    // `for(i=-1; i<Z+1; i+=2)`.
    let mut i: isize = -1;
    while i < z_i + 1 {
        let left = buf[(pad + i - 1) as usize];
        let right = buf[(pad + i + 1) as usize];
        let centre_idx = (pad + i) as usize;
        // Only write into real or right-pad slots that are valid as
        // sources for the next step. The spec writes into the
        // extension area too because the next step reads it, so allow
        // writes across `[-1 ..= Z]`.
        if i >= -1 && i <= z_i {
            buf[centre_idx] -= (left + right) >> 1;
        }
        i += 2;
    }

    // Step 2 — even samples: Y[i] = X[i] + ((Y[i-1] + Y[i+1] + 2) >> 2)
    // for i = 0, 2, ..., Z-1. Spec loop `for(i=0; i<Z; i+=2)`.
    let mut i: isize = 0;
    while i < z_i {
        let left = buf[(pad + i - 1) as usize];
        let right = buf[(pad + i + 1) as usize];
        buf[(pad + i) as usize] += (left + right + 2) >> 2;
        i += 2;
    }
}

/// Helper: build a working buffer of length `z + 2*PAD` with the band
/// data placed at the central `[PAD .. PAD+z]` slots and the pad slots
/// zero-initialised. Caller is expected to call `extend_symmetric`
/// before invoking the lifting filter.
fn make_work_buf(z: usize) -> Result<Vec<i32>> {
    if z == 0 || z > MAX_DIM {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: band dimension {z} out of range (1..={MAX_DIM})"
        )));
    }
    Ok(vec![0i32; z + 2 * PAD])
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
    if w < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: row width {w} below the spec minimum of 2"
        )));
    }
    let low_len = w.div_ceil(2);
    let high_len = w / 2;
    if low.len() != low_len || high.len() != high_len {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: inverse horizontal expected low.len()={low_len} high.len()={high_len}, got low.len()={} high.len()={}",
            low.len(),
            high.len(),
        )));
    }
    let mut buf = make_work_buf(w)?;
    // Interleave: even indices ← low, odd indices ← high.
    for x in 0..w {
        let i = x / 2;
        if x % 2 == 0 {
            buf[PAD + x] = low[i];
        } else {
            buf[PAD + x] = high[i];
        }
    }
    extend_symmetric(&mut buf, w)?;
    inverse_filter_1d(&mut buf, w);
    out.copy_from_slice(&buf[PAD..PAD + w]);
    Ok(())
}

/// Annex E.11, Table E.10 — single-row forward horizontal filter.
///
/// `input.len()` is the row width `w`. `low` and `high` are the
/// interleaved low-pass / high-pass output sub-bands of lengths
/// `(w+1)/2` and `w/2` respectively.
pub fn forward_horizontal_1d(input: &[i32], low: &mut [i32], high: &mut [i32]) -> Result<()> {
    let w = input.len();
    if w < 2 {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: row width {w} below the spec minimum of 2"
        )));
    }
    let low_len = w.div_ceil(2);
    let high_len = w / 2;
    if low.len() != low_len || high.len() != high_len {
        return Err(Error::invalid(format!(
            "JPEG XS DWT: forward horizontal expected low.len()={low_len} high.len()={high_len}, got low.len()={} high.len()={}",
            low.len(),
            high.len(),
        )));
    }
    let mut buf = make_work_buf(w)?;
    buf[PAD..PAD + w].copy_from_slice(input);
    extend_symmetric(&mut buf, w)?;
    forward_filter_1d(&mut buf, w);
    // De-interleave: even indices → low, odd indices → high.
    for x in 0..w {
        let i = x / 2;
        if x % 2 == 0 {
            low[i] = buf[PAD + x];
        } else {
            high[i] = buf[PAD + x];
        }
    }
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
/// row-major. Vertical synthesis runs first (LL+LH → temporary low
/// column, HL+HH → temporary high column), followed by horizontal
/// synthesis on each row. This mirrors the encoder side, where Annex
/// E.9 / Table E.8 performs vertical decomposition before horizontal
/// at every level.
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

    // Vertical synthesis: independently reconstruct the left
    // (low-pass-horizontal) column-set and the right
    // (high-pass-horizontal) column-set into temporaries `tmp_l`
    // (h x ll_w) and `tmp_h` (h x hl_w).
    let mut tmp_l = vec![0i32; h * ll_w];
    let mut tmp_h = vec![0i32; h * hl_w];

    let mut col_low = vec![0i32; ll_h];
    let mut col_high = vec![0i32; lh_h];
    let mut col_out = vec![0i32; h];

    for x in 0..ll_w {
        for y in 0..ll_h {
            col_low[y] = ll[y * ll_w + x];
        }
        for y in 0..lh_h {
            col_high[y] = lh[y * ll_w + x];
        }
        inverse_vertical_1d(&col_low, &col_high, &mut col_out)?;
        for y in 0..h {
            tmp_l[y * ll_w + x] = col_out[y];
        }
    }
    for x in 0..hl_w {
        for y in 0..ll_h {
            col_low[y] = hl[y * hl_w + x];
        }
        for y in 0..lh_h {
            col_high[y] = hh[y * hl_w + x];
        }
        inverse_vertical_1d(&col_low, &col_high, &mut col_out)?;
        for y in 0..h {
            tmp_h[y * hl_w + x] = col_out[y];
        }
    }

    // Horizontal synthesis: for each row, combine the matched columns
    // of `tmp_l` (low-pass) and `tmp_h` (high-pass) into the final
    // row of `out`.
    let mut row_low = vec![0i32; ll_w];
    let mut row_high = vec![0i32; hl_w];
    let mut row_out = vec![0i32; w];
    for y in 0..h {
        row_low.copy_from_slice(&tmp_l[y * ll_w..(y + 1) * ll_w]);
        row_high.copy_from_slice(&tmp_h[y * hl_w..(y + 1) * hl_w]);
        inverse_horizontal_1d(&row_low, &row_high, &mut row_out)?;
        out[y * w..(y + 1) * w].copy_from_slice(&row_out);
    }

    Ok(())
}

/// Single-level 2-D forward 5/3 DWT (companion to [`inverse_2d`]).
///
/// `input[h × w]` is decomposed into the four sub-bands
/// `ll`, `hl`, `lh`, `hh`. Layout matches [`inverse_2d`].
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

    // Horizontal pass: split each row into low/high column-sets,
    // stored in temporaries `tmp_l` (h x ll_w) and `tmp_h` (h x hl_w).
    let mut tmp_l = vec![0i32; h * ll_w];
    let mut tmp_h = vec![0i32; h * hl_w];
    let mut row_in = vec![0i32; w];
    let mut row_low = vec![0i32; ll_w];
    let mut row_high = vec![0i32; hl_w];
    for y in 0..h {
        row_in.copy_from_slice(&input[y * w..(y + 1) * w]);
        forward_horizontal_1d(&row_in, &mut row_low, &mut row_high)?;
        tmp_l[y * ll_w..(y + 1) * ll_w].copy_from_slice(&row_low);
        tmp_h[y * hl_w..(y + 1) * hl_w].copy_from_slice(&row_high);
    }

    // Vertical pass: split each column of `tmp_l` into LL/LH and each
    // column of `tmp_h` into HL/HH.
    let mut col_in = vec![0i32; h];
    let mut col_low = vec![0i32; ll_h];
    let mut col_high = vec![0i32; lh_h];
    for x in 0..ll_w {
        for y in 0..h {
            col_in[y] = tmp_l[y * ll_w + x];
        }
        forward_vertical_1d(&col_in, &mut col_low, &mut col_high)?;
        for y in 0..ll_h {
            ll[y * ll_w + x] = col_low[y];
        }
        for y in 0..lh_h {
            lh[y * ll_w + x] = col_high[y];
        }
    }
    for x in 0..hl_w {
        for y in 0..h {
            col_in[y] = tmp_h[y * hl_w + x];
        }
        forward_vertical_1d(&col_in, &mut col_low, &mut col_high)?;
        for y in 0..ll_h {
            hl[y * hl_w + x] = col_low[y];
        }
        for y in 0..lh_h {
            hh[y * hl_w + x] = col_high[y];
        }
    }

    Ok(())
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
}
