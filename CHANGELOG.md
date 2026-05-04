# Changelog

## Unreleased — encoder round 2 (multi-component RCT + multi-decomp + odd dims)

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
