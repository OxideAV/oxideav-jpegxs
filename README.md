# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

End-to-end decoder for the multi-component, single-precinct-row
subset of ISO/IEC 21122-1:2022. Supports:

* `Nc` ∈ {1, 2, 3, 4} components.
* `sx`, `sy` ∈ {1, 2} per component (4:4:4, 4:2:2, 4:2:0).
* `Cw == 0` (one precinct per row of the picture).
* `Cpih ∈ {0, 1}` — no transform or RGB↔YCbCr reversible
  (Annex F.3).
* `Qpih ∈ {0, 1}` — deadzone or uniform inverse quantizer.
* `Fq ∈ {0, 8}` — lossless or regular per Table A.8.
  `Bw ∈ {8, 18, 20}`.
* `NL,x` ≤ 1 and `NL,y` ≤ 1 (single-level wavelet cascade).
* Annex G linear / quadratic / extended output scaling, dispatched
  from the NLT marker (Annex A.4.6).

Codestream marker-chain parser per ISO/IEC 21122-1:2022 Annex A:

* `SOC` (`FF 10`), `EOC` (`FF 11`)
* `CAP` (`FF 50`) — capability bits captured raw
* `PIH` (`FF 12`) — picture header decoded
* `CDT` (`FF 13`) — per-component bit depth + sampling factors
* `WGT` (`FF 14`) — per-band gain + priority bytes
* `NLT` (`FF 16`) — quadratic / extended non-linearity parameters
* `COM` / `CWD` / `CTS` / `CRG` — optional segments (raw payload)
* `SLH` (`FF 20`) — slice header

Public API:

* `oxideav_jpegxs::probe(&[u8]) -> Option<JpegXsFileInfo>` —
  width / height / components / bit depth / profile / level / Cpih /
  lossless flag.
* `oxideav_jpegxs::register(&mut CodecRegistry)` — registers the
  codec under id `"jpegxs"`. The decoder factory returns a
  working `Decoder` that produces multi-plane `VideoFrame`s.

Modules:

* `codestream` — marker-chain parser
* `picture_header`, `component_table`, `slice_header` — segment parsers
* `slice_walker` — per-precinct geometry + packet layout (Annex B)
* `entropy` — packet body decoder (Annex C)
* `dequant` — inverse quantizer (Annex D)
* `dwt` — reversible 5/3 inverse DWT (Annex E)
* `colour_transform` — inverse RCT (Annex F.3); Star-Tetrix is
  guarded as `Unsupported` until CTS+CRG marker parsers land
* `output` — Annex G linear / quadratic / extended output scaling
  + DC level shift + clipping; NLT body parser

## Out of scope (next round)

* Multi-level wavelet cascade (`NL,x > 1` or `NL,y > 1`) and the
  per-step LL recursion of Annex E.
* `Cw > 0` (custom precinct widths) and the multi-precinct-per-row
  case.
* `Cpih == 3` (Star-Tetrix) needs the CTS marker (Annex A.4.8) and
  CRG marker (Annex A.4.9) parsers.
* `Sd > 0` (CWD-driven decomposition suppression for components 4..7).
* Output bit depths > 8 — Annex G kernels are bit-depth agnostic but
  the pack-to-plane helper currently emits `Vec<u8>` only.
* Encoder side (forward DWT, forward colour transform, entropy
  encoder, quantization).
