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
| Encoder | Round 2 — luma + RGB (Cpih ∈ {0, 1}) 4:4:4, NL ∈ {1, 2}, odd dims, lossless 32×32 self-roundtrip ∞ dB |

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
* `oxideav_jpegxs::parse_capabilities(&[u8]) -> Result<Capabilities>`
  — decode CAP body bits into individual feature flags.
* `oxideav_jpegxs::parse_cts(&[u8]) -> Result<CtsMarker>`,
  `parse_crg(&[u8], nc) -> Result<CrgMarker>`,
  `cfa_pattern_type(&CrgMarker) -> Option<u8>` — CTS / CRG marker
  parsers and Table F.9 lookup.
* `oxideav_jpegxs::register(&mut CodecRegistry)` — registers the
  codec under id `"jpegxs"`. The decoder factory returns a
  working `Decoder` that produces multi-plane `VideoFrame`s.

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
* `encoder` — rounds 1-2: forward 5/3 DWT (Annex E.13) per
  precinct (NL=1/1) or via picture-level cascade
  `dwt::forward_cascade_2d` (NL=2/2); forward RCT
  (`colour_transform::forward_rct`, Annex F.4 Table F.3) when
  `Cpih == 1`; raw-mode bitplane counts (Annex C.6.4, `Dr = 1`) +
  Fs = 0 data sub-packet (Table C.8); short packet headers; symmetric
  reflection for partial bottom precincts (odd heights)

## Out of scope (next round)

* `Cw > 0` (custom precinct widths) and the multi-precinct-per-row
  case.
* `Sd > 0` (CWD-driven decomposition suppression for components 4..7).
* Output bit depths > 8 — Annex G kernels are bit-depth agnostic but
  the pack-to-plane helper currently emits `Vec<u8>` only.
* Encoder rounds 3+: 4:2:2 / 4:2:0 chroma sub-sampling, NL,x ≠ NL,y
  and NL > 2, regular (lossy) `Fq = 8` mode, NLT-aware encoder
  (linear / quadratic / extended gamma), VLC bitplane-count modes
  (no-prediction / vertical), significance coding, Star-Tetrix
  (`Cpih = 3`) encoder, `Cw > 0` custom precinct widths on the
  encoder side.
