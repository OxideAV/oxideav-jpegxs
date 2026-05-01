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
///    `LL_{d-1, d-1}`.
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
}
