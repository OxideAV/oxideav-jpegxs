# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Round 1 (current)

Codestream marker-chain parser per ISO/IEC 21122-1:2022 Annex A:

* `SOC` (`FF 10`), `EOC` (`FF 11`)
* `CAP` (`FF 50`) — capability bits captured raw
* `PIH` (`FF 12`) — picture header decoded into width × height ×
  components × bit-depth-related fields, profile, level, colour
  transform id, lossless flag
* `CDT` (`FF 13`) — per-component bit depth + sampling factors
* `WGT` (`FF 14`) — per-band gain + priority bytes (raw)
* `COM` (`FF 15`) — extension marker (zero or more, raw payload)
* `NLT` / `CWD` / `CTS` / `CRG` — optional segments (raw payload)
* `SLH` (`FF 20`) — slice header; entropy-coded body length recovered
  by forward marker scan (round-1 caveat)

Public API:

* `oxideav_jpegxs::probe(&[u8]) -> Option<JpegXsFileInfo>` —
  width / height / components / bit depth / profile / level / Cpih /
  lossless flag
* `oxideav_jpegxs::register(&mut CodecRegistry)` — registers the codec
  under id `"jpegxs"`. The decoder factory currently returns
  `Error::Unsupported("JPEG XS pixel decode not yet implemented")`.

## Round 2 (planned)

* Inverse DWT (Annex E)
* Entropy decode (Annex C: precinct header, packet header, packet body
  / raw mode)
* Length-driven slice walker (replaces the round-1 marker scan)
* Inverse colour transform (Annex F: RGB↔YCbCr; Star-Tetrix)
* Inverse non-linearity (Annex G; only when NLT present)
