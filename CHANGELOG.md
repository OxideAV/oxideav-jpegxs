# Changelog

## Unreleased — round 95 / r93 (`Sd > 0` composes with `Cpih ≠ 0`)

Lifts the round-9 (r91) blanket `Cpih = 0` restriction on `Sd > 0` per
ISO/IEC 21122-1:2022 Annex F.2 Table F.1 + §A.5.2 + §B.2. The colour
transform's operand window is fixed at `c < 3` (Cpih=1 / RCT) or
`c < 4` (Cpih=3 / Star-Tetrix); when CWD's suppressed-component tail
sits entirely beyond that window — i.e. `Nc - Sd >= 3` for RCT or
`Nc - Sd >= 4` for Star-Tetrix — the two features compose cleanly: the
first `Nc - Sd` components are wavelet-coded (with the colour transform
applied to indices 0..3 / 0..4), and the trailing `Sd` components ride
the raw CWD tail-loop unchanged.

* `colour_transform.rs` — **RCT / Star-Tetrix accept ≥ 3 / ≥ 4 planes**.
  Previously `inverse_star_tetrix` and `forward_star_tetrix` rejected
  unless `planes.len() == 4`. The new contract accepts any plane count
  ≥ 4 and operates only on the first 4 (per Table F.1 "Set Ω = O for
  c ≥ 4"). `inverse_rct` / `forward_rct` already had the equivalent
  contract on the first 3 planes; comments tightened.
* `encoder.rs` — **new `encode_planar_sd_rct` and
  `encode_planar_sd_star_tetrix` entry points**. The
  `EncodeConfig.validate` block now drops the
  "`Sd > 0` requires `Cpih = 0`" rejection and instead checks the
  operand-window overlap constraint
  (`Nc - Sd >= 3` for `Cpih = 1`, `Nc - Sd >= 4` for `Cpih = 3`).
  The `Cpih = 1 → Nc = 3` and `Cpih = 3 → Nc = 4` strict-equality
  checks become `Nc >= 3` / `Nc >= 4` per Annex F.2. Cpih=3 sx/sy=1
  enforcement narrows from "every component" to "first 4 components"
  so suppressed-tail components beyond the CFA window can carry their
  own sampling factors when needed (in practice the CWD constraint
  forces them to 1×1 anyway, but the encoder no longer cross-rejects).
  CRG marker emission body length now scales with `Nc` and emits
  `(0, 0)` placement for entries beyond the four CFA components.
* `decoder.rs` — **Cpih=3 + Nc > 4 path unblocked**. The
  `pih.nc != 4` rejection becomes `pih.nc < 4` per Annex F.2 Table F.1.
  Defensive `Nc - Sd >= 3 / 4` overlap checks added so a malformed
  codestream that suppresses a transform operand fails fast.
* **Tests** — 6 new encoder roundtrip / rejection tests covering
  Sd=1+Cpih=1 (Nc=4 lossless + lossy q=2 PSNR floor), Sd=2+Cpih=1
  (Nc=5 lossless), Sd=1+Cpih=3 (Nc=5 lossless), and operand-window
  overlap rejection for each colour transform; 2 colour_transform
  module tests verifying RCT/Star-Tetrix pass extra (tail) planes
  through unchanged. 218 total (was 210); 0 ignored.

## Unreleased — round 9 / r91 (`Sd > 0` CWD decomposition suppression)

End-to-end `Sd > 0` support on both decoder and encoder sides per
ISO/IEC 21122-1:2022 Annex A.4.7 Table A.18 (CWD marker) and Annex B.7
Table B.4 (tail-loop packet emission).

* `codestream.rs` — **CWD body parsing**. The 1-byte `Sd` field is now
  decoded and validated against `Nc > 3` and `Sd ∈ 1..=Nc-1`; a new
  `cwd_sd: Option<u8>` field on `Codestream` exposes the parsed value
  without the caller having to re-touch the raw body. Slice-length
  derivation routes through `build_plan_sd` so multi-Sd codestreams
  derive correct precinct counts.
* `slice_walker.rs` — **`build_plan_sd(pih, cdt, wgt, sd)` extension**.
  Reads `n_decomposed = Nc - Sd`, allocates `n_bands = n_decomposed *
  Nβ + Sd` per Annex B.3, builds the wavelet `(β, i)` band table only
  for `i < n_decomposed`, and appends Sd tail bands (β=0, raw, one per
  suppressed component) carrying the full `Wp[p] × Hp` precinct
  footprint. Packet layout walker emits the spec's tail loop
  ("component as fast, line as slow") after the wavelet packets, one
  packet per (line λ, suppressed component i). `PicturePlan` gains a
  public `sd: u8` field.
* `decoder.rs` — **gather path bypasses DWT for Sd components**. The
  multi-level cascade allocation skips suppressed components (empty
  per-component band slot). During `gather_precinct`, the Sd tail
  bands' dequantized values are copied straight into `samples[i]` at
  the precinct's row/column offset (sx=sy=1 mandated). The cascade-DWT
  loop also skips suppressed slots. `Sd > 0` forces the multi-level
  path because the per-precinct streaming path doesn't know about the
  tail bands.
* `encoder.rs` — **new `encode_planar_sd(width, height, nc, nlx, nly,
  q, sd, planes)` entry point**. `EncodeConfig` gains an `sd: u8`
  field; validation accepts `Nc ∈ 4..=8` when `Sd > 0` and rejects
  `Cpih != 0`. `write_main_header` emits the CWD marker (`FF 17`,
  Lcwd=3) after WGT. `count_existing_bands` / `build_band_gains_sd`
  account for the `Sd` tail (gain=0). `write_slice` forces the
  cascade path when `Sd > 0`, skips suppressed components in the
  per-component forward DWT loop, and `encode_precinct_cascade` adds
  `Sd` slices at the tail (reading raw DC-biased samples from
  `comp_planes`); phase-1 packet jobs emit the per-(line, suppressed
  component) packets after the wavelet packets per Annex B.7.
* **Tests** — 5 new encoder roundtrip / rejection tests (`round9_*`)
  + 2 walker tests (`build_plan_sd1_4comp_4x4_nl_1_1`,
  `build_plan_sd_rejects_subsampled_tail`). 7 new / 0 ignored; 210
  total (was 203). Lossless self-roundtrips at Sd=1 Nc=4 NL=2/2 and
  Sd=2 Nc=5 NL=1/1; Sd=1 q=2 holds per-component PSNR ≥ 30 dB.

## Unreleased — round 8 (multi-precinct-per-row `Cw > 0`)

End-to-end `Cw > 0` support on both decoder and encoder sides — the
multi-precinct-per-row case spec §B.5 defines with
`Cs = 8 × Cw × max(sx) × 2^NL,x` and `Np,x = ⌈Wf / Cs⌉` precincts per
row.

* `slice_walker.rs` — **precinct grid computation for `Cw > 0`**. The
  `Cw != 0` `Unsupported` rejection lifts; `build_plan` now computes
  `Cs` per Annex B.5 and walks `Np,x × Np,y` precincts in raster
  order. `PicturePlan` carries new `np_x` / `np_y` / `cs` fields so
  downstream callers can recover the per-row layout without
  recomputing it. The per-precinct `Wp[p]` / `Wpb[p,b]` formulas were
  already parametric in the precinct grid; only the grid stride and
  the rightmost-precinct remainder needed adjusting.
* `decoder.rs` — **gather path generalised to `Np,x > 1`**. The
  picture-level band-buffer copy now uses
  `py = p / Np,x, px = p % Np,x` and offsets each precinct into the
  picture band at `band_col_offset = px × (Cs / (sx[i] × 2^dx))`.
  `Cw > 0` codestreams are forced through the gather-then-cascade
  path (the legacy NL=1/1 streaming-per-precinct fast path runs a
  per-precinct DWT that does not commute with multi-precinct-per-row
  layout, since precinct boundaries reflect at the band level not
  the sample level).
* `encoder.rs` — **new `encode_planar_cw` entry point** taking the
  `Cw` parameter and routing everything through the picture-level
  cascade forward DWT. `EncodeConfig` gains a `cw` field; the PIH
  body now emits the configured `Cw` value instead of hard-coding
  zero. `encode_precinct_cascade` takes `(py, px, Cs)` and slices
  each picture-level band buffer at the per-precinct column range,
  computing per-precinct `Wpb` from `Cs / (sx[i] × 2^dx)` (a clean
  divisor because `Cs` is built as `8 × Cw × max(sx) × 2^NL,x`).
  `Cw = 0` reduces to the prior single-precinct-per-row codestream
  bit-for-bit.
* `encoder.rs` — **`EncodeConfig::validate`** rejects `Cs > Wf` and
  `Cs == 0`, the two corruption modes the spec implicitly excludes.

Verification: 203 tests pass (was 194, +9). New tests cover:
`Cw=1 64×16 luma NL=1/1` lossless, `Cw=1 64×16 luma NL=2/2`
lossless, `Cw=2 128×32 RGB+RCT NL=2/2` lossless, `Cw=1 64×8 YUV 4:2:2
NL=1/1` lossless, `Cw=1 64×16 luma q=2` PSNR ≥ 25 dB, `Cw=1 96×16
luma` (Np,x=6) lossless, `Cw=0` bit-equivalence to `encode_planar`,
`Cs > Wf` rejection, plus a slice-walker plan-shape test for
`Cw=1 32×4 luma NL=1/1` (Np,x=2, Cs=16, Wpb[β=0]=8). The
standalone-feature build passes too (187 tests).

`cargo fmt --check` clean; `cargo clippy --all-targets --no-deps --
-D warnings` clean.

## Unreleased — encoder round 7 (extended NLT Tnlt=2 + NL ∈ {1..=8})

Two encoder-side capabilities on top of round 6:

* `encoder.rs` — **Extended NLT encoder (Tnlt=2, Annex G.5).** New
  entry point `encode_planar_nlt_extended(width, height, nc, cpih, nlx,
  nly, q, t1, t2, e, planes)` emits the NLT marker (Tnlt=2, T1, T2, E)
  with `Bw = 18` and applies a forward extended-gamma pre-distortion
  that inverts the decoder's three-segment kernel via a `2^Bw`-entry
  reverse lookup table built once per encode. The inverse LUT is
  exact-round-trippable on parameter combinations whose decoder output
  spans the full 8-bit range (verified at `T1 = 2^14`, `T2 = 2^16`,
  `E = 1`, where every input pixel 0..=255 round-trips bit-exactly via
  the LUT path alone — DWT / quantizer adds the only further loss).
  Validates `0 < T1 < T2`, `1 ≤ E ≤ 4`, both thresholds ≤ `2^Bw - 1`.
  Self-roundtrip PSNR ≥ 30 dB on a 32×32 synthetic gradient at q=0;
  ≥ 25 dB at q=2 with a strictly smaller codestream.
* `encoder.rs` — **Deeper wavelet cascade `NL ∈ {1..=8}`.** Validation
  cap lifted from 5 to 8 (the spec Annex A.4.4 Table A.7 hard maximum).
  The cascade DWT / band geometry helpers were already parametric in
  `NL`; only the validation threshold needed adjustment. `NL = 6/6`
  64×64 luma self-roundtrip verified; `NL = 9` rejection verified.

`decoder.rs` — **Removed `#[ignore]` on `debug_multilevel_layout`.**
Rewritten as `multilevel_plan_shape_nl_2_2_4x4_luma`: asserts the
Annex B.3 invariant `Nβ = 2·min(NL,x,NL,y) + max(NL,x,NL,y) + 1` (= 7
at NL = 2/2), a single-slice plan with `Hsl = 1`, and that every
existing band carries a non-zero `wpb`. Replaces a debug-only `eprintln!`
helper with a real regression.

Verification: 194 tests pass (was 189, +5; the previously-ignored
debug test is now a real assertion); standalone-feature build also
passes; `cargo fmt --check` and `cargo clippy --all-targets --no-deps
-- -D warnings` clean.

## Unreleased — encoder round 6 (deeper wavelet cascade NL ∈ {1..=5})

The encoder validation previously capped at NL,x = 2 / NL,y = 2 even
though `forward_cascade_2d` and `inverse_cascade_2d` are generic across
any `NL,y ≤ NL,x` pair (decoder cascade tested at NL = 3/3 since round
6 of the decoder).

* `encoder.rs` — **Relaxed `EncodeConfig::validate` to accept
  `NL,x ∈ {1..=5}` (was `{1, 2}`).** `NL,y ∈ {0..=NL,x}` constraint
  unchanged. Spec Annex A.4.4 Table A.7 allows NL,x ∈ {1..=8}; we test
  through 5/5 here with both luma and 3-component RGB self-roundtrips.
  All four entry points (`encode_planar`, `encode_planar_lossy`,
  `encode_planar_subsampled`, `encode_planar_star_tetrix`,
  `encode_planar_nlt_quadratic`) inherit the deeper-NL ceiling because
  they all route through `encode_planar_inner` → `validate`.
* Self-roundtrip tests at NL = 3/3, 4/4, 5/5 for 64×64 luma and at
  NL = 3/3 for 32×32 RGB (Cpih=1); asymmetric NL = 3/2 round-trip;
  rejection at NL = 6.
* `round6_nl_4_4_lossy_q4_psnr_above_25db` confirms the lossy path
  still meets the 25 dB floor at deeper cascades.

The cascade band geometry / `n_beta` / `beta_key` / `band_dims` /
`pow_h` helpers were already generic in NL — relaxing the validation
threshold is the only encoder-side change. The decoder needed no
changes (it has always supported NL > 2 cascades).

Verification: 189 tests pass (was 182, +7); standalone-feature build
also passes; `cargo fmt --check` and `cargo clippy -- -D warnings`
clean.

## Unreleased — encoder round 5 (NL_x≠NL_y + significance coding + NLT quadratic + per-band Q)

Four new encoder capabilities on top of round 4:

* `encoder.rs` — **Asymmetric decomposition (`NL_x ≠ NL_y`, Annex B).**
  Validation now accepts `NL_y ∈ {0..=NL_x}` instead of requiring
  `NL_x == NL_y`. The cascade path routes each component through
  `forward_cascade_2d(wc, hc, nlx, nly_i)` where `nly_i = NL_y -
  log2(sy[i])` (chroma may reduce further). Decoder was already
  asymmetry-capable; encoder now matches. `encode_planar(nlx=2, nly=1)`
  self-roundtrips losslessly for both luma-only and 3-component RGB
  (Cpih=0 and Cpih=1). `NL_y > NL_x` is still rejected.
* `encoder.rs` — **Significance coding (`D[p,b] bit 1`, Annex C.5,
  Table C.5).** The cascade encoder now evaluates four D combinations
  per band per precinct: `D ∈ {0, 1, 2, 3}` (sig×pred bits). Forms
  `no_pred_sig` and `vert_sig` are built alongside the existing
  `no_pred` and `vert` forms. For significance-coded bands, one bit per
  significance group (`Ns = ⌈W_pb / (Ng·Ss)⌉` groups) flags zero groups
  which then skip bitplane-count VLC emission entirely. The significance
  sub-packet is emitted before the cnt sub-packet per Annex C.4 order.
  Phase 3 picks the smallest total size across all four D combinations;
  phase 4 writes 2 D bits per band into the precinct header and selects
  the matching packet form.
* `encoder.rs` — **NLT quadratic forward map (Annex G.4, Tnlt=1).**
  `encode_planar_nlt_quadratic(width, height, nc, cpih, nlx, nly, q,
  dco, planes)` applies `y = round(sqrt(x / (2^B-1)) * (2^Bw-1)) +
  dco` before the DWT, forcing `Bw = 18` and emitting the NLT marker
  (`FF 16`, Tnlt=1, σ:α packed). `dco` must fit in signed 16-bit.
  `q = 0` reduces to lossless in the 18-bit wavelet space; `q > 0`
  sets `Fq = 8`. PSNR ≥ 40 dB (lossless path); ≥ 30 dB at q=2.
* `encoder.rs` — **Per-band gain-weighted Q (Annex C.6.2).**
  `build_band_gains(nc, nlx, nly, sx, sy)` computes `G[b] = tau_x +
  tau_y` (0 for LL, 1 for HL/LH, 2 for HH) per band-in-picture-order.
  Both the single-level and cascade encoder paths compute `T[p,b] =
  clamp(Q - G[b], 0, 15)` so HH bands are truncated 2 steps less than
  LL, preserving low-frequency energy at the same Q budget.

Verification: 182 tests pass (was 175); `cargo fmt --check` and
`cargo clippy -- -D warnings` clean.

## Unreleased — encoder round 4 (Star-Tetrix Cpih=3 + vertical-prediction VLC)

Two production-relevant axes added on top of the round-3 lossless +
Fq=8-lossy + 4:2:2 / 4:2:0 + Dr=0-no-prediction-VLC encoder:

* `colour_transform.rs` — **forward Star-Tetrix transform (`Cpih == 3`,
  Annex F.5).** New `forward_star_tetrix(planes, wf, hf, e1, e2, ct, cf)`
  inverts the four lifting steps (Tables F.5 / F.6 / F.7 / F.8) in
  reverse order with sign flip, sharing the `access()` reflection
  (Table F.12) and the per-CFA-pattern displacement vector (Table
  F.10) with the existing inverse path. Bit-exact with `floor_div`
  (spec floor semantics, not C-truncation toward zero).
* `encoder.rs` — **`encode_planar_star_tetrix(width, height, nlx, nly,
  q, e1, e2, cf, ct, planes)` entry point.** Takes 4 component planes
  in input order `(R, G1, G2, B)` matching the decoder's output
  convention; emits the CTS marker (`Cf`, `e1`, `e2`) and the CRG
  marker (Table F.9 RGGB row for `Ct=0`, GRBG row for `Ct=1`). Self-
  roundtrips losslessly across `Ct ∈ {0, 1}`, `Cf ∈ {0, 3}`,
  `e1, e2 ∈ {0, 2, 3}`, and `NL ∈ {1, 2}`.
* `encoder.rs` — **vertical-prediction VLC (`D[p,b] & 1 = 1`, Annex
  C.6.5, Table C.13).** The cascade encoder now runs a four-phase
  per-precinct selector:
  1. Phase 1 collects every packet job in slice-walker emission
     order, marking the first-in-precinct line per `(comp, beta)`
     band.
  2. Phase 2 builds three forms per packet: Dr=1 raw, Dr=0 no-pred
     VLC (`mtop = T`, θ=0), and Dr=0 vert-pred VLC (`mtop = max
     (M_above, T)`, `θ = max(M_above - T, 0)`) — using a per-band
     `Mtop` cache populated from earlier packets in the same precinct.
  3. Phase 3 sums per-band `min(raw, no_pred)` vs `min(raw, vert_pred)`
     bytes and commits `D[p,b] = 0` or `1` per band.
  4. Phase 4 emits packets in original order using the band's chosen
     `D[p,b]` and the smaller of `(raw, in-band-mode-VLC)`.
  The picker beats round 3's no-pred-only path on smooth content
  with `>= 2` lines per precinct in some band (NL >= 2 cascade,
  proxy levels with `pow_h(NL,y, dy) > 1`).
* `encoder.rs` — `Nc = 4` and `Cpih = 3` accepted by `EncodeConfig`
  validation; `write_main_header` emits the CTS (`Lcts = 4`) and CRG
  (`Lcrg = 2 + 4*Nc`) markers when `cpih == 3`.
* `colour_transform.rs` tests cover forward + inverse Star-Tetrix
  round-trip on flat-zero, non-trivial 8×8 four-component data,
  non-default `(e1, e2) = (2, 3)`, alternate CFA pattern `Ct=1`, and
  in-line `Cf=3` access mode. Encoder tests: 4-component CFA round-
  trip via `encode_planar_star_tetrix` for `Ct ∈ {0, 1}`,
  `Cf ∈ {0, 3}`, `NL ∈ {1, 2}`; vertical-prediction picker round-
  trips synthetic 32×32 RGB at NL=2/2; vertical-gradient 64×64
  compresses below 4 KB raw; rejection of `Cpih = 3` with `Nc != 4`.

Encoder %-delta of synthetic RGB 32×32 (Cpih=1 NL=2/2): lossless 121.3%
→ 120.1% (-1.2 pp from picker engaging on a few cascade bands).
Smooth 64×64 RGB gradient lossless: 68.3% of raw (= 8397 / 12288 B).
4-component CFA Star-Tetrix round-trip: bit-exact across all tested
parameter combinations.

Verification: 173 tests pass (was 162); standalone-feature build also
passes; `cargo fmt --check` and `cargo clippy -- -D warnings` clean.

## Unreleased — encoder round 3 (Dr=0 VLC + Fq=8 lossy + 4:2:2 / 4:2:0)

Three production-relevant compression-feature axes added on top of the
round-2 raw-mode-only / lossless / 4:4:4 encoder, plus a fix to the
slice-walker's per-component band-existence path that surfaced under
4:2:0 + NL=1/1.

* `encoder.rs` — **Dr = 0 VLC bitplane-count mode (Annex C.6.6,
  Table C.14, no prediction).** Per-precinct each packet is now built
  in both Dr=1 raw form and Dr=0 VLC form, and the smaller (header +
  body) is kept. New `emit_vlc_no_prediction(writer, Δm)` emits Δm
  ones followed by a 0 comma — the inverse of the decoder's
  `vlc(reader, mtop=T, t=T)` with θ=0. Saves ≈10–25% on the
  bitplane-count sub-packet for sparse bands; the picker keeps Dr=1
  for dense bands where 8 bits per group beats unary.
* `encoder.rs` — **regular (`Fq = 8`) lossy mode with Annex D.2
  deadzone quantizer.** New `encode_planar_lossy(width, height, nc,
  cpih, nlx, nly, q, &[Vec<u8>]) -> Result<Vec<u8>>` entry point.
  `q` is the precinct-level `Q[p]` (`0..=15`); `q = 0` reduces to
  lossless. The encoder right-shifts coefficient magnitudes by
  `T = clamp(Q - G - r, 0, 15) = q` and emits only `M - T`
  bitplanes per code group; the decoder reconstructs with the
  half-bucket offset `((1 << T) >> 1)`. Synthetic 32×32 RGB benchmark:
  PSNR ≥ 40 dB at q=1, ≥ 28 dB at q=4, ≥ 25 dB at q=6, with codestream
  shrinking from 3725 B (lossless) → 2241 B (q=4) → 1532 B (q=8) for
  Cpih=1 NL=2/2.
* `encoder.rs` — **4:2:2 / 4:2:0 chroma sub-sampling.** New
  `encode_planar_subsampled(width, height, nc, cpih, nlx, nly, q,
  sx, sy, &[Vec<u8>]) -> Result<Vec<u8>>` entry point. Each `planes[i]`
  is `(width / sx[i]) * (height / sy[i])` bytes. Per-component
  `N'L,y[i] = NL,y - log2(sy[i])` per Annex B.2 — sub-sampled chroma
  at `sy = 2` & `NL,y = 1` runs only the 1-D horizontal DWT (LL/HL
  bands), matching the decoder's `inverse_synth_1d` path. The
  precinct-header `D[p,b]` field count and the WGT `(G[b], P[b])`
  pair count are computed from the per-component existing-band mask.
* `slice_walker.rs` — **per-component band-existence guard fixed.**
  The `build_plan` per-(β, i) band geometry loop used to compute
  `beta_levels(β, nlx, nly_i)` for every β in the picture-level
  `Nβ`, including those exceeding the per-component `Nβ_i`. For
  sub-sampled chroma at `sy = 2`, `nly_i = 0` and β > 1 triggered
  `dx = nlx + 1 - β` underflow on `u32`. The fix skips the geometry
  computation when `β >= Nβ_i` and marks the band non-existent —
  same outcome the existing `band_exists` would have flagged later,
  but without the underflow panic.
* New encoder tests cover: Dr=0-vs-Dr=1 picker shrinks the codestream
  on synthetic RGB, flat-luma compresses to ≪ raw via VLC unary
  zeros, Fq=8 q=1 PSNR ≥ 40 dB + smaller than lossless, q=4 PSNR ≥
  25 dB, q=0 path identical to the lossless `encode_planar`, 4:2:2
  and 4:2:0 lossless self-roundtrip, 4:2:0 codestream smaller than
  4:4:4 of the same picture, rejection of unsupported `(sx, sy)`.

Verification: 162 tests pass (was 154); standalone-feature build
also passes 149 tests (was 140).

## Unreleased — encoder round 2 (multi-component RCT + multi-decomp + odd dims)

## [0.0.4](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.3...v0.0.4) - 2026-05-08

### Other

- encoder round 6: relax NL,x cap from 2 to 5 (deeper wavelet cascade)
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- encoder round 5: silence clippy 1.95 cloned_ref_to_slice_refs in tests
- encoder round 5: NL_x≠NL_y + significance coding + NLT quadratic + per-band Q
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-jpegxs/pull/502))
- add register_containers for .jxs extension lookup
- release v0.0.3

## [0.0.3](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.2...v0.0.3) - 2026-05-05

### Added

- *(encoder)* round 4 — Star-Tetrix Cpih=3 + vertical-prediction VLC
- *(encoder)* round 3 — Dr=0 VLC + Fq=8 lossy + 4:2:2/4:2:0 chroma sub-sampling
- *(encoder)* round 2 multi-component RCT, multi-decomp NL=2/2, odd dimensions
- *(encoder)* round 1 luma-only 32x32 self-roundtrip bootstrap

### Other

- release v0.0.3
- clippy 1.95: fix needless_range_loop + cloned_ref_to_slice_refs in round-3 tests

## [0.0.3](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.2...v0.0.3) - 2026-05-05

### Added

- *(encoder)* round 4 — Star-Tetrix Cpih=3 + vertical-prediction VLC
- *(encoder)* round 3 — Dr=0 VLC + Fq=8 lossy + 4:2:2/4:2:0 chroma sub-sampling
- *(encoder)* round 2 multi-component RCT, multi-decomp NL=2/2, odd dimensions
- *(encoder)* round 1 luma-only 32x32 self-roundtrip bootstrap

### Other

- clippy 1.95: fix needless_range_loop + cloned_ref_to_slice_refs in round-3 tests

Three production-relevant axes added on top of the round-1 luma-only
bootstrap, plus two latent decoder bugs uncovered by the new fixtures
were fixed.

* `encoder.rs` — **multi-component (`Nc ∈ {1, 3}`)**. New entry
  points `encode_rgb_8bit(width, height, &[u8], cpih, nl)` and the
  generalised `encode_planar(width, height, nc, cpih, nlx, nly, &[Vec<u8>])`.
  3-component (4:4:4) inputs go through forward RCT (Annex F.4
  Table F.3) when `cpih == 1`. 4:2:2 / 4:2:0 chroma sub-sampling is
  still deferred to round 3.
* `encoder.rs` — **multi-decomposition `NL,x = NL,y ∈ {1, 2}`**. NL
  ≥ 2 routes through the new `dwt::forward_cascade_2d` cascade and
  emits per-(β, i) picture-level band buffers; the encoder slices
  each band into its per-precinct row range using the same
  `pow_h = 2^(NL,y - dy)` formula the decoder uses to gather. The
  NL=1/1 fast path keeps the round-1 per-precinct streaming forward
  DWT.
* `encoder.rs` — **odd dimensions**. Pictures with `Wf` or `Hf` not a
  multiple of `2^NL,y` are accepted at any supported NL. Partial
  bottom precincts (where `hp_real < 2^NL,y`) extend the input strip
  via whole-sample symmetric reflection inside the encoder; the
  decoder's per-precinct synthesis path was hardened to pad missing
  band rows to the spec's `Hp` so `inverse_2d` always operates on a
  consistent geometry.
* `dwt.rs` — new `forward_cascade_2d(wc, hc, nlx, nly, &[i32])
  -> Vec<Vec<i32>>` mirroring `inverse_cascade_2d`. Two-phase
  decomposition (joint horizontal+vertical levels first, then pure
  horizontal levels when `nlx > nly`); produces band buffers in the
  same β order the inverse expects (`b = Nc * β + i` bands).
* `colour_transform.rs` — `forward_rct` promoted from a private
  test helper to public API. Companion of `inverse_rct` for the
  encoder side; mirrors `Y = (R + 2G + B) >> 2`, `Cb = B - G`,
  `Cr = R - G` per Annex F.4 Table F.3.
* `codestream.rs` — **length-driven slice walker (BUG FIX).** The
  legacy `FF 20`/`FF 11` byte-scan walker mis-fires on entropy bytes
  that happen to look like a marker prefix; round-1 fixtures avoided
  this by hand-crafting fixtures with empty entropy. Real codestreams
  produced by the round-2 encoder hit it on every other test fixture,
  so the slice walker now uses the picture plan's per-precinct
  geometry to read each precinct's 24-bit `Lprc` and advance
  `header_bytes + Lprc` per precinct. Empty-slice probe-only
  fixtures still drop through to a fallback byte scan, which works
  because their entropy region is empty.
* `slice_walker.rs` — **`Wpb` formula made τx-aware (BUG FIX).** The
  old `wp.div_ceil(sx * 2^dx)` formula matched `Wb[β,i]` from the
  cascade only for power-of-2 widths; for odd widths it overstated
  HL/HH band widths by one column. The corrected form mirrors the
  cascade's `(Wc).div_ceil(2^(dx-1)) / 2` for τx = true bands. The
  existing fixtures (all power-of-2) still pass; new odd-dim NL≥2
  encoder fixtures now round-trip.
* `decoder.rs` — `synthesise_precinct` pads short bands with zero
  rows so `inverse_2d` runs at the spec's `hp_i` even when the
  bottom precinct is partial; output rows past `Hf / sy_i` are
  dropped (matches Annex B.6 — bands shrink, the synthesis grid
  stays at `Hp`).
* New encoder tests cover: RGB 32×32 with and without RCT, NL=2/2
  for both luma and RGB+RCT, odd-dim 31×31 NL=1/1, odd-dim 33×17
  NL=2/2, `encode_image` for 3-component inputs, rejection of
  unsupported configs, and a generous size-bound sanity check on
  the round-2 raw-mode encoder.

Verification: 154 tests pass (was 146); standalone-feature build
also passes 140 tests.

## Round 1 — encoder bootstrap (luma-only 32×32 self-roundtrip)

Bootstrap encoder mirroring the proven decoder-first pattern. Round 1
encodes single-luma 8-bit images with `NL,x = NL,y = 1`, single
precinct column (`Cw = 0`), single-slice (`Hsl = Np_y`), lossless
mode (`Fq = 0`, `Bw = 8`), deadzone quantizer at `T = 0`, raw
bitplane-count mode (`Dr = 1`), and short packet headers.

* `encoder.rs` — new module exposing `encode_luma_8bit(width, height,
  &[u8])`, `encode_image(&JpegXsImage)`, and `encode_raw_luma(width,
  height, Vec<u8>)`. Pipeline: DC-bias subtraction → per-2-row-precinct
  forward 2-D DWT (Annex E.13) → group-wise raw bitplane-count packing
  (Annex C.6.4) → MSB-first sign-then-magnitude data sub-packet
  (Annex C.4 Table C.8) → short packet header (Table C.3) + precinct
  header (Table C.1). Codestream chain SOC | CAP | PIH | CDT | WGT |
  SLH | <slice 0 entropy> | EOC.
* `dwt.rs` — fix `extend_symmetric` for `z = 2`. The previous version
  read the right-pad slot before initialising it (and used the
  uninitialised value for the left-pad reflection). With `Hp = 2`
  precincts the vertical 1-D pass exercises this path; the existing
  decoder fixtures all use `h ≥ 4` or all-zero coefficients, so the
  bug stayed hidden until the encoder fed real coefficients into a
  2-row inverse DWT. New comment in the function explains the
  collapsed `X[-2] = X[2] = X[0]` reflection chain.
* `lib.rs` — `pub mod encoder;` and re-exports for `encode_image`,
  `encode_luma_8bit`, `encode_raw_luma`.
* New tests: `encoder::tests` covering bit-writer correctness, input
  validation, flat-image bit-exact round-trip, 32×32 synthetic
  lossless round-trip (PSNR = ∞ dB; the 40 dB workspace minimum is a
  binding asserted lower bound), 2×2 minimum-size round-trip via the
  `Wpb < Ng` short-tail-group path, and a round-trip via the
  `JpegXsImage` convenience entry point.
* `dwt.rs` regression: `per_precinct_dwt_round_trips_for_hp_2` pins
  the `z = 2` symmetric-extension behaviour for the round-1 encoder
  pipeline.

Test count: 137 → 146 (+9).

## [0.0.2](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.1...v0.0.2) - 2026-05-03

### Added

- standalone-friendly Cargo feature shape ([#359](https://github.com/OxideAV/oxideav-jpegxs/pull/359))

### Other

- silence unused super::* import in gated tests mod

Mirrors the vp8/webp standalone-friendly treatment so the crate can be
built without `oxideav-core` in the dep tree. Default features stay on
for existing consumers (`oxideav` umbrella, mp4 demuxer, mkv).

* `Cargo.toml` — `oxideav-core` is now an optional dep behind a
  default-on `registry` feature.
* `error.rs` — new crate-local `JpegXsError` enum + `Result` alias
  (`std`-only, no `oxideav-core` dep).
* `image.rs` — new crate-local `JpegXsImage` / `JpegXsPlane` types,
  carrying the picture-header geometry (`width` / `height` /
  `num_components` / `cpih` / `bit_depth`) inline since the
  `oxideav_core::Frame` enum it used to be wrapped in carries that
  out-of-band via `CodecParameters`.
* `registry.rs` — gated `Decoder` trait impl, `JpegXsDecoder` struct,
  `make_decoder` factory, `register()` entry point, and the
  `From<JpegXsError>` / `From<JpegXsImage>` conversions back into
  `oxideav_core::{Error, Frame}`. Re-exported from the crate root as
  before when the default `registry` feature is on.
* `decoder.rs` + every other internal module — switched from
  `oxideav_core::{Error, Result}` to the crate-local `JpegXsError` /
  `Result`. No behaviour change. `decode_codestream` now returns
  `JpegXsImage` instead of `oxideav_core::VideoFrame`.
* `lib.rs` — new `decode_jpeg_xs(buf) -> Result<JpegXsImage>`
  standalone entry point.

CI: adds an inline `ci-standalone` job running
`cargo build/test --no-default-features --lib`.

## [0.0.1](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.0...v0.0.1) - 2026-05-03

### Other

- Fix clippy identity_op warnings in decoder bitstream tests

## Round 6 — Star-Tetrix + multi-level DWT + CAP bits

* `dwt.rs` — multi-level inverse cascade (`inverse_cascade_2d`) per
  Annex E.2 Table E.1. For `nlx > 1` or `nly > 1`, walks the band
  quadtree level-by-level, calling the round-2 single-level
  `inverse_2d` (or single-row `inverse_horizontal_1d` in the
  pure-horizontal levels when `nlx > nly`).
* `decoder.rs` — multi-level path: gathers per-component, per-band
  coefficients across all precincts into picture-level band buffers,
  then runs `inverse_cascade_2d` once per component. The single-level
  streaming per-precinct path is kept as a fast path for `nlx ≤ 1
  ∧ nly ≤ 1`. Sub-sampled chroma in multi-level mirrors the
  single-level convention `N'L,y[i] = NL,y - log2(sy[i])`.
* `colour_transform.rs` — Annex F.5 inverse Star-Tetrix transform
  (Cpih=3) implementation, including all four lifting steps
  (`inv_avg_step`, `inv_delta_step`, `inv_Y_step`, `inv_CbCr_step`),
  the `access(c, x, y, rx, ry)` reflection from Table F.12, and the
  super-pixel look-up tables (Tables F.9 / F.10 / F.11). Honours both
  `Cf` modes (full vs restricted in-line) and uses floor-division
  semantics for the spec's `⌊·/8⌋` and `⌊·/4⌋` operators.
* `cts.rs` — CTS (Colour Transformation Specification) marker parser
  (Annex A.4.8, Tables A.19 / A.20) — decodes `Cf`, `e1`, `e2`.
* `crg.rs` — CRG (Component Registration) marker parser (Annex A.4.9,
  Table A.21) plus Tables F.9 / F.10 / F.11 helpers
  (`cfa_pattern_type`, `displacement`, `component_at`).
* `capabilities.rs` — CAP marker `cap[]` byte-array decoder (Annex
  A.5.4 / Table A.5). Strongly-typed `Capabilities` struct exposes
  individual feature flags: Star-Tetrix, NLT quadratic, NLT extended,
  vertical sub-sampling, CWD, lossless, raw-mode switch.
  `Codestream::capabilities()` decodes the parsed `cap` bytes.
* `decoder.rs` — wires Cpih=3 path: parses CTS+CRG markers, maps CRG
  values to a Table F.9 CFA pattern type, and dispatches
  `inverse_star_tetrix` after the inverse DWT.
* New tests:
  - Multi-level DWT cascade round-trips: NL=2/2, NL=3/3, NL=2/1,
    NL=1/1 (regression).
  - End-to-end multi-level decode: 4×4 NL=2/2, 8×8 NL=3/3.
  - End-to-end Star-Tetrix decode: 4-component 4×2 with CTS + CRG +
    Cpih=3 codestream.
  - Star-Tetrix flat-luma propagates to G1/G2 (manual trace).
  - CAP bit decoder (each bit + lossy/strict modes).
  - CTS body parser (Cf, e1, e2).
  - CRG body parser + RGGB / BGGR / GRBG / GBRG pattern detection +
    Table F.10 / F.11 displacement round-trip.

Test count: 98 → 137 (+39).

## Round 5 — multi-component + Annex F + Annex G

* `colour_transform.rs` — Annex F.3 inverse RCT (Cpih=1) for the first
  three components; Annex F.5 Star-Tetrix is signature-stubbed
  (returns `Unsupported`) pending CTS / CRG marker support.
* `output.rs` — Annex G output scaling kernels:
  - linear (Annex G.3, no NLT marker),
  - quadratic (Annex G.4, NLT Tnlt=1),
  - extended (Annex G.5, NLT Tnlt=2).
  Includes the NLT body parser (Annex A.4.6, Table A.16).
* `slice_walker.rs` — `Nc > 1` support; per-component sampling factors
  (`sx`, `sy`); per-component effective decomposition levels
  `N'L,y[i] = NL,y - log2(sy[i])`; spec-correct band index ordering
  `b = (Nc - Sd) × β + i`; Annex B.7 Table B.4 packet layout for the
  multi-component case.
* `decoder.rs` — multi-component dispatch: per-component sample
  buffers, per-component DWT synthesis with the right band IDs,
  `inverse_rct` after DWT when `Cpih == 1`, `apply_output_scaling`
  per component with the picture's `Bw` and the component's `B[i]`.
  Validates the Cpih-vs-CDT compatibility per Annex F.2.
* New tests:
  - 3-component 4:4:4 zero codestream (Cpih=0)
  - 3-component 4:4:4 RCT zero codestream (Cpih=1)
  - 3-component 4:2:2 zero codestream
  - NLT-quadratic codestream end-to-end
  - Multi-component plan + packet layout regression tests
  - RCT round-trip on synthetic pixels
  - NLT body parser
  - Output scaling kernels (linear, quadratic, extended)

Test count: 78 → 98 (+20).

## Round 4

End-to-end decoder for the single-component, single-precinct,
single-slice subset:

* Slice / precinct / packet geometry walker (Annex B.5–B.10).
* Inverse quantizer (Annex D.2 deadzone + D.3 uniform).
* Wired-up `Decoder` in `decoder.rs` with end-to-end tests.

## Round 3

* Entropy decoder (Annex C) — precinct header, packet header, packet
  body (significance / bitplane-count / data / sign sub-packets).
* Variable-length decoder primitive (Table C.15) and bit-stream cursor.

## Round 2

* Reversible 5/3 inverse DWT (Annex E).
* Forward 5/3 DWT companion for round-trip tests.

## Round 1

* Codestream marker-chain parser (Annex A).
* `probe` API — width / height / components / bit depth.
* Codec registration under id `"jpegxs"`.
