# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

| Direction | Status |
| --- | --- |
| Decoder | working — multi-component, single-precinct-row subset (rounds 1–6) |
| Encoder | Round 4 — luma + RGB 4:4:4 / 4:2:2 / 4:2:0 + 4-component CFA Star-Tetrix, Cpih ∈ {0, 1, 3}, NL ∈ {1, 2}, odd dims, Dr ∈ {0, 1} VLC + raw picker with no-prediction (Table C.14) **and vertical-prediction (Table C.13)** sub-modes, Fq ∈ {0, 8} lossy with Q ∈ 0..=15. Self-roundtrip ∞ dB lossless; PSNR ≥ 40 dB at q=1, ≥ 25 dB at q=4 |

End-to-end decoder for the multi-component, single-precinct-row
subset of ISO/IEC 21122-1:2022. Supports:

* `Nc` ∈ {1, 2, 3, 4} components.
* `sx`, `sy` ∈ {1, 2} per component (4:4:4, 4:2:2, 4:2:0).
* `Cw == 0` (one precinct per row of the picture).
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
* `encoder` — rounds 1-4: forward 5/3 DWT (Annex E.13) per
  precinct (NL=1/1) or via picture-level cascade
  `dwt::forward_cascade_2d` (NL=2/2); forward RCT
  (`colour_transform::forward_rct`, Annex F.4 Table F.3) when
  `Cpih == 1`; forward Star-Tetrix (`colour_transform::
  forward_star_tetrix`, Annex F.5) when `Cpih == 3`; per-packet
  picker between raw-mode bitplane counts (Annex C.6.4, `Dr = 1`),
  Dr=0 no-prediction VLC (Annex C.6.6, Table C.14), and Dr=0
  vertical-prediction VLC (Annex C.6.5, Table C.13) committed
  per-band-per-precinct via `D[p,b] & 1`; Fq=8 lossy mode (Annex D.2
  deadzone) with precinct-constant `Q[p]`; chroma sub-sampling
  (`sx, sy ∈ {1, 2}`) via per-component effective `N'L,y[i] =
  NL,y - log2(sy[i])` and 1-D horizontal DWT for nly_i=0 chroma;
  Fs=0 data sub-packet (Table C.8); short packet headers; symmetric
  reflection for partial bottom precincts (odd heights); CTS / CRG
  marker emission for `Cpih = 3`

## Out of scope (next round)

* `Cw > 0` (custom precinct widths) and the multi-precinct-per-row
  case.
* `Sd > 0` (CWD-driven decomposition suppression for components 4..7).
* Output bit depths > 8 — Annex G kernels are bit-depth agnostic but
  the pack-to-plane helper currently emits `Vec<u8>` only.
* Encoder rounds 5+: NL,x ≠ NL,y and NL > 2, significance coding
  (Table C.5 / C.14 gating), NLT-aware encoder (linear / quadratic /
  extended gamma per Annex G), `Cw > 0` custom precinct widths on the
  encoder side, per-band per-precinct Q optimization. Round 4 already
  covers Star-Tetrix (`Cpih = 3`) and vertical-prediction VLC
  (Table C.13).
