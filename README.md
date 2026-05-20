# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

| Direction | Status |
| --- | --- |
| Decoder | working — multi-component, **multi-precinct-per-row (Cw ≥ 0)** subset (rounds 1–8) |
| Encoder | Round 8 — luma + RGB 4:4:4 / 4:2:2 / 4:2:0 + 4-component CFA Star-Tetrix, Cpih ∈ {0, 1, 3}, **NL_x ∈ {1..=8} / NL_y ∈ {0..=NL_x}** (spec Annex A.4.4 Table A.7 hard max), **Cw ≥ 0** (`Cs = 8 × Cw × max(sx) × 2^NL,x` per Annex B.5, Np,x = ⌈Wf / Cs⌉ precincts per row), odd dims, Dr ∈ {0, 1} VLC + raw picker with no-prediction (Table C.14) **and vertical-prediction (Table C.13)** sub-modes, **significance coding (D[p,b] bit 1, Annex C.5)** gating zero significance groups, **per-band gain-weighted Q** (`T[p,b] = clamp(Q−G[b], 0, 15)`, G ∈ {0,1,2}), **NLT quadratic forward map** (Annex G.4, Tnlt=1, Bw=18) via `encode_planar_nlt_quadratic`, **NLT extended forward map** (Annex G.5, Tnlt=2, three-segment gamma, Bw=18) via `encode_planar_nlt_extended` with reverse LUT inverter, Fq ∈ {0, 8} lossy with Q ∈ 0..=15. Self-roundtrip ∞ dB lossless at NL=3/3, 4/4, 5/5, 6/6; PSNR ≥ 40 dB at q=1, ≥ 25 dB at q=4; NLT extended PSNR ≥ 30 dB at q=0, ≥ 25 dB at q=2; Cw=1 64×16 luma at NL=1/1 and NL=2/2 + Cw=2 128×32 RGB+RCT NL=2/2 + Cw=1 4:2:2 round-trip bit-exact |

End-to-end decoder for the multi-component, single-precinct-row
subset of ISO/IEC 21122-1:2022. Supports:

* `Nc` ∈ {1, 2, 3, 4} components.
* `sx`, `sy` ∈ {1, 2} per component (4:4:4, 4:2:2, 4:2:0).
* `Cw ≥ 0` — `Cw = 0` keeps the single-precinct-per-row layout; `Cw > 0`
  splits each row into `Np,x = ⌈Wf / Cs⌉` precincts with
  `Cs = 8 × Cw × max(sx) × 2^NL,x` per Annex B.5. Tested at Cw=1 64×16
  luma (NL=1/1, NL=2/2), Cw=1 64×8 YUV 4:2:2 NL=1/1, Cw=2 128×32
  RGB+RCT NL=2/2.
* `Cpih ∈ {0, 1, 3}` — no transform, RGB↔YCbCr reversible
  (Annex F.3), or Star-Tetrix (Annex F.5) for 4-component CFA images.
* `Qpih ∈ {0, 1}` — deadzone or uniform inverse quantizer.
* `Fq ∈ {0, 8}` — lossless or regular per Table A.8.
  `Bw ∈ {8, 18, 20}`.
* Multi-level wavelet cascade (`NL,x ≥ NL,y`, both up to typical
  spec maxima — tested at `NL = 3/3`).
* Annex G linear / quadratic / extended output scaling, dispatched
  from the NLT marker (Annex A.4.6).

Codestream marker-chain parser per ISO/IEC 21122-1:2022 Annex A:

* `SOC` (`FF 10`), `EOC` (`FF 11`)
* `CAP` (`FF 50`) — capability bits decoded into a strongly-typed
  `Capabilities` view (Star-Tetrix, NLT quadratic / extended,
  vertical sub-sampling, CWD, lossless, raw-mode switch)
* `PIH` (`FF 12`) — picture header decoded
* `CDT` (`FF 13`) — per-component bit depth + sampling factors
* `WGT` (`FF 14`) — per-band gain + priority bytes
* `NLT` (`FF 16`) — quadratic / extended non-linearity parameters
* `CTS` (`FF 18`) — colour transformation specification (Cf, e1, e2)
* `CRG` (`FF 19`) — component registration → CFA pattern type Ct
* `COM` / `CWD` — optional segments (raw payload)
* `SLH` (`FF 20`) — slice header

Public API:

* `oxideav_jpegxs::probe(&[u8]) -> Option<JpegXsFileInfo>` —
  width / height / components / bit depth / profile / level / Cpih /
  lossless flag.
* `oxideav_jpegxs::encode_luma_8bit(width, height, &[u8]) -> Result<Vec<u8>>`
  — single-luma 8-bit, NL=1/1 path retained from round 1.
* `oxideav_jpegxs::encode_rgb_8bit(width, height, &[u8], cpih, nl)
  -> Result<Vec<u8>>` — round-2 multi-component path: 3-component
  4:4:4, `cpih ∈ {0, 1}` (no transform / forward RCT), `nl ∈ {1, 2}`.
  Self-roundtrips losslessly. Pixels are interleaved
  `R, G, B, R, G, B, …`.
* `oxideav_jpegxs::encode_planar(width, height, nc, cpih, nlx, nly,
  &[Vec<u8>]) -> Result<Vec<u8>>` — round-2 generalised planar entry
  point covering both Nc=1 and Nc=3, NL,x = NL,y ∈ {1, 2}, any
  dimensions ≥ 2 (odd dims included).
* `oxideav_jpegxs::encoder::encode_planar_lossy(width, height, nc,
  cpih, nlx, nly, q, &[Vec<u8>]) -> Result<Vec<u8>>` — round-3 lossy
  4:4:4 entry point. `q ∈ 0..=15` is the precinct quantization step
  (Annex C.2 `Q[p]`); `q = 0` reduces to lossless. Forces `Fq = 8`
  per Table A.8 when `q > 0`.
* `oxideav_jpegxs::encoder::encode_planar_subsampled(width, height,
  nc, cpih, nlx, nly, q, sx, sy, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-3 chroma-sub-sampled entry point. Each `planes[i]` has length
  `(width / sx[i]) * (height / sy[i])`. Supports 4:4:4 / 4:2:2 / 4:2:0
  with `(sx, sy) ∈ {1, 2}`. `q = 0` lossless / `q > 0` lossy.
* `oxideav_jpegxs::encoder::encode_planar_star_tetrix(width, height,
  nlx, nly, q, e1, e2, cf, ct, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-4 4-component CFA entry point (`Cpih = 3`, Star-Tetrix per
  Annex F.5). Component plane order is `[R, G1, G2, B]`. Emits the
  CTS marker (`Cf`, `e1`, `e2`) and the CRG marker (Table F.9 RGGB
  layout for `Ct=0`, GRBG layout for `Ct=1`).
* `oxideav_jpegxs::encoder::encode_planar_nlt_quadratic(width, height,
  nc, cpih, nlx, nly, q, dco, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-5 NLT quadratic entry point. Applies forward quadratic pre-
  distortion (`y = sqrt(x/255) * 262143 + dco`, Annex G.4, Tnlt=1)
  before the DWT, forces `Bw = 18`, and emits the NLT marker. `dco`
  must fit in signed 16-bit. `q = 0` lossless; `q > 0` Fq=8 lossy.
* `oxideav_jpegxs::encoder::encode_planar_nlt_extended(width, height,
  nc, cpih, nlx, nly, q, t1, t2, e, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-7 NLT extended entry point. Applies a forward extended-gamma
  pre-distortion (Annex G.5, Tnlt=2, three-segment kernel with
  thresholds `0 < T1 < T2 ≤ 2^Bw - 1` and slope exponent `E ∈ 1..=4`)
  built from a `2^Bw`-entry reverse LUT inverting the decoder's
  `extended_path`. Forces `Bw = 18` and emits the NLT marker with
  `(T1, T2, E)`. `q = 0` lossless within LUT resolution; `q > 0` Fq=8
  lossy.
* `oxideav_jpegxs::encode_planar_cw(width, height, nc, cpih, nlx, nly,
  q, cw, &[Vec<u8>]) -> Result<Vec<u8>>` — round-8 multi-precinct-per-
  row 4:4:4 entry point. `cw` controls the precinct-width parameter
  `Cw` (PIH §A.4.4); `cw = 0` reduces to a single precinct column
  spanning the full picture width (bit-equivalent to `encode_planar`).
  `cw > 0` splits each precinct row into `Np,x = ⌈Wf / Cs⌉` precincts
  with `Cs = 8 × cw × max(sx) × 2^NL,x`. Rejects `Cs > Wf`. Routes
  everything through the picture-level cascade DWT so per-precinct
  columns commute with the wavelet boundaries.
* `oxideav_jpegxs::parse_capabilities(&[u8]) -> Result<Capabilities>`
  — decode CAP body bits into individual feature flags.
* `oxideav_jpegxs::parse_cts(&[u8]) -> Result<CtsMarker>`,
  `parse_crg(&[u8], nc) -> Result<CrgMarker>`,
  `cfa_pattern_type(&CrgMarker) -> Option<u8>` — CTS / CRG marker
  parsers and Table F.9 lookup.
* `oxideav_jpegxs::register(&mut RuntimeContext)` — unified entry point
  that installs the codec under id `"jpegxs"` plus the `.jxs` extension
  hint into a `RuntimeContext`. Use `register_codecs(&mut CodecRegistry)`
  and `register_containers(&mut ContainerRegistry)` for the split form.
  The decoder factory returns a working `Decoder` that produces
  multi-plane `VideoFrame`s.

Modules:

* `codestream` — marker-chain parser; `Codestream::capabilities()`
  decodes the parsed CAP bytes
* `picture_header`, `component_table`, `slice_header` — segment parsers
* `capabilities` — CAP `cap[]` decoder (Annex A.5.4)
* `cts` — CTS marker parser (Annex A.4.8)
* `crg` — CRG marker parser (Annex A.4.9) + Tables F.9 / F.10 / F.11
* `slice_walker` — per-precinct geometry + packet layout (Annex B)
* `entropy` — packet body decoder (Annex C)
* `dequant` — inverse quantizer (Annex D)
* `dwt` — reversible 5/3 inverse DWT (Annex E), single-level
  `inverse_2d` and multi-level cascade `inverse_cascade_2d`
* `colour_transform` — inverse RCT (Annex F.3) and inverse
  Star-Tetrix (Annex F.5, Tables F.4–F.8) with Table F.12 access
* `output` — Annex G linear / quadratic / extended output scaling
  + DC level shift + clipping; NLT body parser
* `encoder` — rounds 1-7: forward 5/3 DWT (Annex E.13) per
  precinct (NL=1/1) or via picture-level cascade
  `dwt::forward_cascade_2d` (NL ∈ {1..=8} or asymmetric NL_x≠NL_y);
  forward RCT (`colour_transform::forward_rct`, Annex F.4 Table F.3)
  when `Cpih == 1`; forward Star-Tetrix (`colour_transform::
  forward_star_tetrix`, Annex F.5) when `Cpih == 3`; per-packet
  picker between raw-mode bitplane counts (Annex C.6.4, `Dr = 1`),
  Dr=0 no-prediction VLC (Annex C.6.6, Table C.14), Dr=0
  vertical-prediction VLC (Annex C.6.5, Table C.13), and significance-
  coded variants (`D[p,b] & 2`, Annex C.5) committed per-band-per-
  precinct via 2-bit `D[p,b] ∈ {0,1,2,3}`; significance sub-packet
  emitted before cnt per Annex C.4 order; Fq=8 lossy mode (Annex D.2
  deadzone) with per-band gain-weighted truncation `T[p,b] =
  clamp(Q−G[b], 0, 15)` (G=0 LL, G=1 HL/LH, G=2 HH); forward NLT
  quadratic pre-distortion (Annex G.4, Tnlt=1, Bw=18) and forward NLT
  extended pre-distortion (Annex G.5, Tnlt=2, Bw=18, reverse-LUT
  inverter) with NLT marker emission; chroma sub-sampling (`sx, sy ∈
  {1, 2}`) via per-component effective `N'L,y[i] = NL,y - log2(sy[i])`
  and 1-D horizontal DWT for nly_i=0 chroma; asymmetric decomposition
  `NL_y ≤ NL_x`; Fs=0 data sub-packet (Table C.8); short packet
  headers; symmetric reflection for partial bottom precincts (odd
  heights); CTS / CRG marker emission for `Cpih = 3`

## Out of scope (next round)

* `Sd > 0` (CWD-driven decomposition suppression for components 4..7) —
  the CWD marker parser (Annex A.4.7) needs wiring + the slice
  walker's band-index formula needs the `b = (Nc - Sd) × Nβ + i`
  tail term for suppressed components.
* Output bit depths > 8 — Annex G kernels are bit-depth agnostic but
  the pack-to-plane helper currently emits `Vec<u8>` only.
* Encoder round 9+: per-band per-precinct Q rate-distortion
  optimization, `Sd > 0` decomposition suppression, multi-slice
  emission with `Hsl > 1` precinct rows per slice. (Round 8 lands
  `Cw > 0` multi-precinct-per-row on both decode and encode sides.)
