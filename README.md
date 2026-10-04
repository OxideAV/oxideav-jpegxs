# oxideav-jpegxs

[![CI](https://github.com/OxideAV/oxideav-jpegxs/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-jpegxs/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-jpegxs.svg)](https://crates.io/crates/oxideav-jpegxs) [![docs.rs](https://docs.rs/oxideav-jpegxs/badge.svg)](https://docs.rs/oxideav-jpegxs) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Built clean-room from the ISO/IEC 21122 specification documents under
`docs/image/jpegxs/` only. Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone (`default-features = false`, no
`oxideav-core`).

## Standalone use

`oxideav-jpegxs` follows the OxideAV image-crate contract
(`IMAGE_CRATE_API`): the same small root vocabulary every
`oxideav-<format>` image crate exposes, usable with
`default-features = false` and no `oxideav-core`, returning pixels as
plain `Vec<u8>`. Every decode function accepts either a bare ISO/IEC
21122-1 codestream (`SOC` marker first) or a `.jxs` still-image file
(ISO/IEC 21122-3 Annex A, JPEG XS Signature box first).

```toml
[dependencies]
oxideav-jpegxs = { version = "0.0", default-features = false }
```

```rust
let bytes = std::fs::read("in.jxs")?;
if oxideav_jpegxs::probe(&bytes) {
    let info  = oxideav_jpegxs::info(&bytes)?;       // header only: width, height, format, bit depth, profile
    let img   = oxideav_jpegxs::decode(&bytes)?;     // JpegXsImage, native layout (grey / GBR(A) / YCbCr(A) planes)
    let rgba: Vec<u8> = img.to_rgba8();              // tightly packed RGBA, 4 * width bytes per row
    let (w, h) = (img.width(), img.height());

    let opts = oxideav_jpegxs::EncodeOptions::default().with_quantization(2);
    let out: Vec<u8> = oxideav_jpegxs::encode_rgba8(w, h, &rgba, &opts)?;   // planar RGBA through the RCT
    std::fs::write("out.jxs", out)?;
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

| Item | Signature |
|---|---|
| `probe` | `fn(&[u8]) -> bool` — `FF 10 FF 50` (SOC + CAP) or the JPEG XS Signature box; allocation-free |
| `info` | `fn(&[u8]) -> Result<ImageInfo, Error>` — `width`, `height`, `format`, `frames` (1), `has_alpha`, `color`, `has_icc` / `has_exif` / `has_xmp`, plus `bit_depth`, `components`, `cpih`, `profile` (`Ppih`), `level` (`Plev`), `lossless`, `boxed` |
| `decode` / `decode_with` | `fn(&[u8][, &DecodeOptions]) -> Result<JpegXsImage, Error>` — native layout, colour + metadata filled |
| `decode_rgb8` / `decode_rgba8` | `-> Result<RgbImage / RgbaImage, Error>` — `{ width, height, data }`, tightly packed, 3 / 4 bytes per pixel |
| `decode_from` | `fn<R: Read>(R) -> Result<JpegXsImage, Error>` |
| `encode` | `fn(&JpegXsImage, &EncodeOptions) -> Result<Vec<u8>, Error>` — the image's own layout; never a silent conversion |
| `encode_rgb8` / `encode_rgba8` | `fn(w, h, &[u8], &EncodeOptions)` — `Gbrp8` / `Gbrap8` through the reversible colour transform (alpha is the pass-through fourth component) |
| `encode_to` | `fn<W: Write>(&JpegXsImage, &EncodeOptions, W) -> Result<(), Error>` |
| `decode_components` / `decode_components_with` | `-> Result<Components, Error>` — the planes exactly as the codestream carries them (codestream order, per-component `bit_depths` / `sampling`, `cpih`); works for every decodable stream |
| `encode_components` | `fn(&Components, &EncodeOptions) -> Result<Vec<u8>, Error>` — the universal encoder funnel (any `Nc ∈ 1..=8`, any `Cpih`, Star-Tetrix) |
| `inspect` | `fn(&[u8]) -> Option<JpegXsFileInfo>` — raw header summary for any parseable stream (the pre-contract `probe`) |
| `JpegXsImage` | `{ width, height, format: PixelFormat, planes: Vec<Plane>, color: ColorInfo, metadata: Metadata, bit_depth }` with `new` / `from_rgb8` / `from_rgba8` (all `Result`), `with_color` / `with_metadata` / `with_bit_depth`, `width()` / `height()` / `format()`, `as_bytes()` (grey only) / `into_raw()`, `to_rgb8()` / `to_rgba8()`, `sample(plane, x, y)` |
| `PixelFormat` | `= JpegXsPixelFormat`: `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le`, `Gbrp8` / `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` / `Gbrp16Le`, `Gbrap8` … `Gbrap16Le`, `Yuv444P` / `Yuv422P` / `Yuv420P` (+ `10Le` / `12Le` / `16Le`), `Yuva444P` / `Yuva422P` / `Yuva420P` (+ `10Le` / `12Le` / `16Le`) — names mirror `oxideav_core::PixelFormat` |
| `Error` | `= JpegXsError`: `InvalidData`, `Unsupported`, `LimitExceeded`, `Io(std::io::Error)` |

Every layout is planar (JPEG XS codes components as separate planes);
`planes` are in the **layout's** order — G, B, R(, A) for the RGB
layouts, Y, Cb, Cr(, A) for YCbCr — each at its tight stride, chroma
planes at `⌈W / 2⌉ × ⌈H / 2⌉` for 4:2:0 and `⌈W / 2⌉ × H` for 4:2:2.
Samples deeper than 8 bits are two little-endian bytes with the value
in the low `bit_depth` bits; depths without a label of their own (9, 11,
13, 14, 15 — and 14 for YCbCr) ride the 16-bit carrier with
`JpegXsImage::bit_depth` / `ImageInfo::bit_depth` naming the significant
bits.

`to_rgb8` / `to_rgba8` are exact per layout: deep samples rescaled by
rounding `v × 255 / (2^B − 1)`, grey replicated, GBR(A) reordered,
YCbCr converted with the H.273 matrix the image's `color` names (`1`
BT.709, `5` / `6` BT.601, `9` BT.2020; anything else — including
"unspecified" — as BT.709) and its range (`Limited` or `Unspecified` →
studio range, the broadcast default; `Full` → full swing), chroma
replicated without interpolation; alpha from the fourth plane, `255`
when there is none.

The pre-contract names stay for one release as deprecated wrappers:
`decode_jpeg_xs` (→ `decode_components`), `decode_jxs_file` (→
`decode`), `encode_image` / `encode_raw_luma` (→ `encode`), the whole
`encoder::encode_planar_*` / `encode_luma_8bit` / `encode_rgb_8bit` /
`pick_*` family (→ `encode` / `encode_components` with `EncodeOptions`
fields). The old `probe(&[u8]) -> Option<JpegXsFileInfo>` is now
`inspect`; `probe` is the contract's boolean sniff. `JpegXsImage` changed
shape (format-tagged contract image; the raw component view is
`Components`), and `JpegXsPlane` is an alias of `Plane`.

## Framework use

With the default `registry` feature the crate depends on `oxideav-core`
and adds:

- `register(&mut RuntimeContext)` — the codec (decoder + encoder) and the
  `.jxs` extension; `register_codecs` / `register_containers` /
  `register_registries` for the split registries. Wired into
  `oxideav_meta::register_all` through `oxideav_core::register!`.
- `make_decoder(&CodecParameters)` — one packet (bare codestream or
  `.jxs` file) → one `VideoFrame` in the **native layout**
  (`Gray*` / `Gbrp*` / `Gbrap*` / `Yuv*` / `Yuva*`, the picture's own
  depth). The colour-signal side-channel is stamped only when the packet
  is a `.jxs` file carrying a CICP box — a bare codestream has no colour
  signalling and none is invented.
- `make_encoder(&CodecParameters)` — `width` / `height` / `pixel_format`
  describe the frames; `options` map onto `EncodeOptions` (`quantization`,
  `target_bytes`, `profile`, `levels_x` / `levels_y`, `rct`, `quantizer`,
  `slice_height`, `run_mode`, `weights`, `high_precision`, `boxed`). Any
  JPEG XS layout is accepted as is; packed `Rgb24` / `Rgba` frames are
  deplaned to `Gbrp8` / `Gbrap8`. One frame in, one keyframe packet out.
- The frame bridge: `From<JpegXsImage> for VideoFrame`,
  `JpegXsImage::from_video_frame(&VideoFrame, &CodecParameters)` and
  `TryFrom<(&VideoFrame, &CodecParameters)>`; `JpegXsPixelFormat` ↔
  `PixelFormat` 1:1 by name (`From` / `TryFrom`); `ColorInfo` ↔
  `ColorSignal`; `JpegXsError` → `oxideav_core::Error`
  (`LimitExceeded` → `ResourceExhausted`).

The framework `Decoder` / `Encoder` call the standalone functions above
— one implementation. Pictures without a contract layout (Star-Tetrix
CFA, two- or five-plus-component sets) are `Unsupported` on the
framework path and on `decode`; `decode_components` still decodes them.

## Supported layouts

Decode — derived from the component table and `Cpih`:

| Codestream | Layout | Notes |
|---|---|---|
| `Nc = 1`, `B ∈ 8..=16` | `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` | 9 / 11 / 13 / 14 / 15-bit in `Gray16Le` |
| `Nc = 3`, `Cpih = 1` (RCT) | `Gbrp8` / `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le` / `Gbrp16Le` | components R, G, B → planes G, B, R |
| `Nc = 4`, `Cpih = 1` | `Gbrap8` … `Gbrap16Le` | fourth component passed through as alpha |
| `Nc = 3`, `Cpih = 0`, chroma 1:1 / 2:1 / 2:2 | `Yuv444P` / `Yuv422P` / `Yuv420P` (+ `10Le` / `12Le` / `16Le`) | a `.jxs` CICP box with matrix `0` at 4:4:4 → `Gbrp*` instead |
| `Nc = 4`, `Cpih = 0` | `Yuva444P` / `Yuva422P` / `Yuva420P` (+ `10Le` / `12Le` / `16Le`) | fourth component as alpha (full rate) |
| `Cpih = 3` (Star-Tetrix CFA), `Nc ∈ {2, 5..=8}`, mixed `B[i]`, other sampling | — (`Unsupported`) | `decode_components` / `inspect` cover them |

Encode — `encode` writes the image's layout; every contract layout has a
JPEG XS representation, so no input is converted:

| Layout | Codestream |
|---|---|
| `Gray*` | `Nc = 1`, `Cpih = 0` |
| `Gbrp*` / `Gbrap*` | `Nc = 3 / 4`, `Cpih = 1` (reversible colour transform; `Cpih = 0` with `EncodeOptions::rct = false`) |
| `Yuv*` / `Yuva*` | `Nc = 3 / 4`, `Cpih = 0`, `sx` / `sy` from the layout |
| any `Components` set | `encode_components`: `Nc ∈ 1..=8`, `Cpih ∈ {0, 1, 3}`, one `B[i]` per stream |

`encode_rgb8` / `encode_rgba8` go through `Gbrp8` / `Gbrap8`. Note that
RGB coded **without** the RCT in a bare codestream decodes as
`Yuv444P` — the codestream cannot say otherwise; keep `rct` on or write
a `.jxs` file (`boxed`), whose CICP matrix `0` brings it back as RGB.

## Options

`EncodeOptions` (`#[non_exhaustive]`, `Default`, `with_*` builders) —
every encoder choice is a field:

| Field | Default | Meaning |
|---|---|---|
| `quantization` | `0` | `Q[p]` `0..=15`; `0` = lossless (bit-exact round trip) |
| `target_bytes` | `None` | constant bitrate: exact codestream size (rate allocation + COM padding + `Lcod`) |
| `profile` | `None` | shape to an ISO/IEC 21122-2 `Profile` and declare `Ppih` / `Plev` (verified) |
| `levels_x` / `levels_y` | `None` | `NL,x` (`1..=8`) / `NL,y` (`0..=NL,x`); `None` = largest `NL,x ≤ 5` the width admits, `NL,y = 1` when the height admits it |
| `rct` | `true` | reversible colour transform on RGB layouts |
| `quantizer` | `Deadzone` | `Qpih`: `Deadzone` / `Uniform` |
| `slice_height` | `None` | `Hsl` in precinct rows; `None` = one slice |
| `column_width` | `0` | `Cw` precinct column width |
| `sign_packet` | `false` | `Fs = 1`, signs in a separate sub-packet |
| `run_mode` | `ZeroResiduals` | `Rm`: `ZeroResiduals` / `ZeroCoefficients` |
| `refinement` | `0` | `R[p]` |
| `high_precision` | `false` | `Bw = 20`, `Fq = 8` path |
| `nlt` | `None` | `NltParams::Quadratic { dco }` / `Extended { t1, t2, e }` |
| `weights` | `Default` | band weights: `Default` / `AnnexH` (Annex H PSNR-optimised tables when one exists) |
| `suppressed_components` | `0` | `Sd` (CWD marker) |
| `star_tetrix` | `None` | `StarTetrixParams { e1, e2, cf, ct }` for `Cpih = 3` component sets |
| `q_slices` / `q_precincts` / `r_precincts` | empty | per-slice / per-precinct `Q[p]` / `R[p]` overrides |
| `boxed` | `false` | emit a `.jxs` file (CICP from `color`, `cdef` for alpha layouts, Exif from `metadata`) |

Compositions the encoder cannot honour are refused with
`Error::Unsupported`, never silently dropped: `profile` composes with
`quantization` / `quantizer` / `levels_*` / `target_bytes` only;
`target_bytes` without a profile needs 4:4:4 sampling and the deadzone
quantiser (set a profile for sub-sampled CBR); mixed component bit
depths need one `B[i]` per stream.

`DecodeOptions` — `max_width` / `max_height` (default `65535`, the
`Wf` / `Hf` field maximum), `max_pixels` (default `1 << 28`),
`max_bytes` (default unlimited), `strict` (default `false`: reject bytes
after `EOC` and a missing `EOC`). `None` = unlimited. Limits are checked
on the picture header before any sample buffer is allocated
(`Error::LimitExceeded`).

## Metadata and colour

A bare ISO/IEC 21122-1 codestream carries no colour description and no
metadata. A `.jxs` file (ISO/IEC 21122-3 Annex A) carries:

- **Colour** — the CICP Colour Specification box (`METH = 5`):
  `ColourPrimaries` / `TransferCharacteristics` / `MatrixCoefficients`
  (H.273 code points) and the full-range flag, read verbatim into
  `ColorInfo { range, primaries, transfer, matrix }`. Without a box the
  layout's documented default applies: RGB layouts report matrix `0`
  (identity — the inverse RCT output *is* RGB) with range, primaries and
  transfer unspecified; grey and YCbCr layouts are entirely
  unspecified. No primaries, transfer or range is invented.
- **Metadata** — the Exif box payload (`Metadata::exif`). `icc` / `xmp`
  are always `None` (the JXS file format defines neither) and `gamma` is
  `None` (transfer is an H.273 code point).

`encode(.., boxed = true)` writes the image's `ColorInfo` as the CICP box
(`Unspecified` range becomes the limited-range flag — the CICP `V` byte
has no "unspecified"), a Channel Definition box for the alpha layouts
(channel 3 = whole-image opacity) and the Exif payload. Lossless
round trip (`quantization = 0`) is pinned: planes exact for every
layout through a bare codestream; planes, colour and metadata exact
through a `.jxs` file for images whose range is signalled.

## Limits

- `Wf` / `Hf` are 16-bit fields: pictures up to 65535 × 65535; the
  encoder needs `Wf ≥ max(sx) × 2^NL,x` and `Hf ≥ max(sy) × 2^NL,y`
  (ISO/IEC 21122-1 Table 11) and at least 2 × 2.
- `DecodeOptions` defaults: 65535 × 65535, 268 Mpixel, unlimited input
  size; all enforced before allocation.
- The decoder rejects CAP bits it does not implement, reserved `Ppih` /
  `Plev` values, profile / level / sublevel claims the stream does not
  satisfy, `Lcod` mismatches and `.jxs` headers that contradict their
  codestream; every malformed input is an `Error`, never a panic (fuzzed:
  `probe` / `info` / `decode` / `decode_components` / `.jxs` parsing /
  encode → decode round trip).

## JPEG XS specifics

The sections below describe the codec's coverage of ISO/IEC 21122 per
tool; the `encode_planar_*` names they cite are the historical per-axis
entry points (now deprecated wrappers — every axis is an `EncodeOptions`
field, see *Options*).

### Status

Both directions are **working for a substantial subset** of ISO/IEC
21122-1:2022 and self-roundtrip losslessly across the supported feature
matrix. JPEG XS has no inter-frame state, so each picture is independent.

#### Decoder

End-to-end decode of the multi-component subset:

- `Nc ∈ 1..=8` components (the full Annex A.4.3 range, both directions —
  e.g. luma+alpha, RGB+alpha, or eight independent planes); `(sx, sy) ∈
  {1, 2}` per component (4:4:4 / 4:2:2 / 4:2:0), with §B.1 ceiling-sized
  planes so odd picture dimensions are legal.
- `Cw ≥ 0` precincts per row (`Cw = 0` single-precinct-per-row;
  `Cw > 0` splits each row into `⌈Wf / Cs⌉` precincts).
- `Cpih ∈ {0, 1, 3}` — no transform, reversible RGB↔YCbCr (Annex F.3),
  or Star-Tetrix (Annex F.5) for 4-component CFA images. Per Table F.1
  the transforms act on a fixed operand window (`c < 3` for RCT, `c < 4`
  for Star-Tetrix); every trailing component is a pass-through
  (`Ω[c] = O[c]`) that still runs the full entropy / dequant / DWT /
  Annex G pipeline — pinned end-to-end by `Nc = 4` and `Nc = 5`
  RCT-plus-alpha round-trips and the 4-component conformance vectors.
- `Qpih ∈ {0, 1}` deadzone / uniform inverse quantizer; `Fq ∈ {0, 8}`
  lossless / regular; `Bw ∈ {8, 18, 20}`; `B[i]` up to 16-bit. The
  uniform inverse quantizer (Annex D.3 Neumann-series reconstruction) is
  exercised end-to-end across all three chroma samplings (4:4:4 / 4:2:2 /
  4:2:0) and composed with multi-slice, multi-precinct-per-row, the
  separate sign sub-packet, the reversible colour transform, and the CFA
  Star-Tetrix transform.
- Multi-level wavelet cascade (`NL,x ≥ NL,y`), multi-slice (`Hsl ≥ 0`),
  precinct refinement (`R[p]`), per-precinct `Q[p]`, separate sign
  sub-packet (`Fs = 1`), and the entropy-decode loss-of-synchronisation
  guard. The horizontal-only decomposition (`NL,y = 0`) is covered across
  the single-level streaming and multi-level cascade encoders: Annex B.7
  Table B.4 gives `β1 = NL,x + 1`, so the LL and HL bands of every
  component share the first packet (Table B.5), and that joint-packet
  layout self-roundtrips for luma, multi-component RGB (`Cpih ∈ {0, 1}`),
  4:2:2 subsampling, high bit depth, and lossy (`q > 0`) modes.
- **4:2:0 with a deep vertical cascade** (`NL,y ∈ {2, 3, 4, 5}`): a
  `sy = 2` chroma component decomposes one vertical level shallower than
  luma (`N'L,y[i] = NL,y − log2(sy[i])`), so the per-band cascade-key
  derivation (`beta_key_for`) is exercised across the full `N'L,y` range
  up to 4. Decode is bit-exact for lossless 4:2:0 (symmetric `NL,x = NL,y`
  and asymmetric `NL,x > NL,y`), composes with high bit depth
  (`B[i] ∈ {10, 12, 14, 16}`), holds a ≥ 25 dB PSNR floor for lossy
  (`q = 2`) at every depth, and round-trips the 4-component Star-Tetrix
  (`Cpih = 3`) CFA inverse across `NL ∈ {2, 3, 4}`.
- Annex G linear / quadratic / extended (NLT) output scaling, including
  decode at deeper vertical cascades: the quadratic and extended (`Bw =
  18`) non-linearity inverses and the high-precision (`Bw = 20, Fq = 8`)
  fractional-coefficient path each round-trip luma across `NL,y ∈ {1, 2,
  3, 4}`, and the high-bit-depth (`B[i] = 12`) NLT path (which runs the
  wavelet domain at `Bw = 20` for precision headroom) reconstructs within
  a few LSB² across `NL,y ∈ {1, 2, 3}`. The NLT non-linearities also
  compose with **chroma sub-sampling**: the Annex G.4 quadratic and G.5
  extended forward pre-distortions are per-sample, per-component maps, so
  `encode_planar_subsampled_nlt_quadratic` / `_subsampled_nlt_extended`
  drive a luma + 4:2:2 / 4:2:0 chroma picture (`(sx, sy) ∈ {1, 2}`,
  `cpih = 0`) through the NLT marker path — each component (full-res luma,
  half-resolution chroma decomposing one vertical level shallower)
  round-trips above the NLT PSNR floor, while RCT (`cpih = 1`) with a
  sub-sampled component is rejected per Annex F.2. The same NLT × chroma
  sub-sampling composition is exposed at **high bit depth**
  (`B[i] ∈ 9..=16`, `Bw = 20`, `u16`-LE planes):
  `encode_planar_subsampled_nlt_quadratic_highbd` /
  `_subsampled_nlt_extended_highbd` drive a 4:2:2 / 4:2:0 luma + chroma
  picture through the high-precision NLT path. The NLT marker also
  composes with the two other gather-path features: multi-precinct-per-row
  (`Cw > 0`, `Np,x > 1`) and CWD decomposition suppression (`Sd > 0`, the
  suppressed raw-coded tail flowing through the per-component NLT inverse
  unchanged) both round-trip alongside the quadratic non-linearity.

Every bitplane-count decode mode (raw, no-prediction, vertical
prediction — Tables C.12 / C.13 / C.14) enforces the spec range
`0 ≤ M[p,λ,b,g] ≤ (2^Br − 1)`, rejecting out-of-range counts in the
variable-length-code paths as well as the raw path.

The entropy path also exposes structural consistency predicates
(precinct-length `Lprc[p]`, bitplane-count-subpacket size `Lcnt[p,s]`,
data-subpacket size `Ldat[p,s]`, sign-subpacket size `Lsgn[p,s]`,
significance-subpacket size `Lsig[p,s]`, buffer-bound conformance) used
to validate codestream construction. Every subpacket byte count is
reconstructible from the precinct's coding state, so each can be
cross-checked against — or inferred in place of — its packet-header
field. The `Lcnt[p,s]` inference covers both layouts of the count
subpacket: the raw mode (`Dr = 1`, `Br` bits per code group, Annex C.6.4
Table C.12) and the two VLC modes (no-prediction / vertical, Tables
C.14 / C.13), the latter summing per-codeword bit lengths via the exact
inverse of the Table C.15 unary VLC.

These predicates are also **wired into the live decode path as
conformance gates**: every precinct cross-checks its declared `Lprc[p]`
against the summed on-wire size of its packets (Annex C.2 Table C.1),
rejecting a length field too small to contain its own packets, and
verifies that the filler-byte count reconstructed from the per-packet
sizes (header + inferred `Lsig[p,s]` + `Lcnt` + `Ldat` + `Lsgn`) matches
the gap the decoder actually skips — catching internally inconsistent
sub-packet length fields that still sum to a valid `Lprc`. Legal
trailing filler inside the data/sign/count sub-packets (Annex C.3) is
tolerated. The decoder additionally rejects out-of-range header fields a
conforming codestream cannot carry: `R[p] ∉ [0, NL−1]` (Annex C.2
Table C.1); the reserved code points of `Cpih` / `Qpih` / `Fs` / `Rm` /
`Ppoc` (Annex A.4.4 Tables A.9–A.13); the single-value `Ng = 4` / `Ss = 8`
fields (Table A.7 lists no range for either, so any other value would
mis-group the code-group / significance-group geometry); and `Cpih = 3`
Star-Tetrix with any sub-sampled CFA input (Annex A.4.3 / F.2, mirroring
the existing `Cpih = 1` RCT guard).

Significance coding (`D[p,b] & 2`, Table C.5) is exercised end-to-end with
**multiple significance groups per band line** (`Ns[p,b] = ⌈Wpb / (Ng·Ss)⌉
> 1`, Annex B.9): for bands wider than `Ng·Ss = 32` code-positions the
encoder emits one `Z` bit per group and the decoder dispatches each to its
`g / Ss` group span, round-tripping luma and RGB-with-RCT at `Ns ∈ {2, 3}`.

Both **run modes** (`Rm`, Annex A.4.4 Table A.12) are emitted and decoded.
`Rm = 0` (runs indicate zero prediction residuals) is the default: an
insignificant significance group in vertical prediction reconstructs to
`M = mtop` (the predictor), so the significance-flag encoder keeps a group
significant whenever its predecessor `M > T`. `Rm = 1` (runs indicate zero
coefficients) instead reconstructs an insignificant group to `M = T[p,b]`
(`Δm = T − mtop`, Table C.13) regardless of the predictor, so both its
bitplane-count VLC residual and its data are elided; a group is
insignificant iff *all* its code groups satisfy `M ≤ T`. The
`encode_planar_run_mode1` entry point (with `_highbd` / `_subsampled`
compositions) drives `Rm = 1` across luma, RGB-with-RCT, high bit depth
(`B[i] = 12`), 4:2:0 chroma, multi-significance-group (`Ns > 1`), deep
odd-dimension cascades (`NL = 3`), and the `q = 0..=6` truncation ladder,
each self-decoding bit-exactly at `q = 0`. The vertical-prediction
`Rm = 1` decode branch is pinned by hand-built packet-body tests
(`Mtop = (5, 4)` insignificant → `M = (0, 0)` under `Rm = 1` versus
`M = (5, 4)` under `Rm = 0`). The reserved `Rm ∈ {2, 3}` are rejected on
both the encode (`EncodeConfig::validate`) and decode (PIH parse) sides.

When the picture-header `Rl = 0`, the decoder also enforces the Annex C.3
raw-mode-consistency rule: a band's raw-mode flag `Dr[p,s]` must be
identical across every packet that includes the band within a precinct
(raw and non-raw bitplane-count coding shall not be mixed within one
band). The encoder, which selects the bitplane-count coding mode
independently per packet, signals `Rl = 1` — the per-packet raw-selection
regime (Annex C.5.3.3) whose per-line buffer bound holds by construction —
so its streams pass the gate while a malformed `Rl = 0` stream that mixes
raw and non-raw within a band is rejected.

The decoder also enforces a layer of **profile / level / sublevel
conformance** drawn from the codestream's own declarations (ISO/IEC
21122-2 Annex A + 21122-1 Tables 11 / A.5 / A.8):

- The `Ppih` profile indicator (Table A.5) is validated: a non-zero value
  mapping to a known profile runs the full `check_codestream` constraint
  set (component count, bit-depth set, chroma format, `NL,x` / `NL,y`,
  `Qpih`, slice height, column mode), and a reserved (unmappable) `Ppih`
  is rejected rather than decoded under an unknown profile.
- The `Plev` level/sublevel indicator bounds the picture against the
  level's `Wmax` / `Hmax` / `Lmax` (Table A.6) and the sublevel's coded
  size `Ssl,max = ⌊Lmax × Nbpp / 8⌋` (§A.4.1, verified against every entry
  of Tables A.8–A.11), rejecting a reserved `Plev` high byte and the
  `Full` sublevel paired with an unrestricted profile (§A.4.2).
- A non-zero CBR `Lcod` must match the actual SOC-to-EOC byte count
  (Table 11); the picture dimensions must satisfy `Wf ≥ max_i(sx)·2^NL,x`
  and `Hf ≥ max_i(sy)·2^NL,y`; and a `Cw > 0` precinct grid whose
  rightmost column would be a sub-LL-sample sliver (`Wf umod Cs` in
  `(0, max_i(sx)·2^NL,x)`) is rejected — the encoder likewise refuses to
  emit one.
- The CAP capabilities marker (§A.4.3, Table A.5) is now both **emitted**
  by the encoder (cap[] bits reflecting Star-Tetrix / NLT / vertical
  subsampling / CWD / lossless / raw-mode-switch usage) and **validated**
  by the decoder, which aborts on any set capability bit it does not
  implement (bit 0, bit 7, reserved bits ≥ 9).
- Table A.8's "additional constraints" beyond the `(Bw, Fq)` pairing: the
  `(B[0], 0)` lossless case requires a uniform component bit depth, and
  `(18, 6)` requires a NLT marker plus CAP bit 2 or 3. (The `(20, 8)`
  CAP-bits-must-be-0 rule is deliberately not enforced — the
  high-bit-depth NLT path runs the wavelet domain at `Bw = 20` for
  precision headroom, a combination Table A.8 does not tabulate; see the
  NLT@Bw=20 docs gap.)
- The CTS marker presence is fully Cpih-determined: mandatory for
  `Cpih = 3` and rejected for any other `Cpih` (Table A.2).

#### Encoder

A planar encoder covering the same feature matrix, lossless (`q = 0`)
and lossy (`q ∈ 1..=15`), with rate-budget pickers that drive per-slice
`Q[p]` and per-precinct `(Q[p], R[p])` against a target byte budget.
8-bit and high-bit-depth (`B[i] ∈ 9..=16`, little-endian `u16` planes)
paths exist for all three colour-transform modes and both NLT modes.

Every encode path signals a **valid ISO/IEC 21122-1:2022 Table A.8
`(Bw, Fq)` combination** and applies the Annex E.3 (Table E.13) wavelet
fractional scaling that the pair implies: `(Bw = B[0], Fq = 0)` for the
default integer-transform regular case (lossy via the precinct truncation
`T[p,b]`, which the inverse quantizer applies independently of `Fq`),
`(Bw = 18, Fq = 6)` / `(Bw = 20, Fq = 8)` when an NLT non-linearity is
present, and `(Bw = 20, Fq = 8)` for the high-precision regular case
exposed by `encode_planar_highprec_lossy`. The decoder reconstructs
`T[β,x,y] = c[p,λ,b,ξ] << Fq` before the inverse DWT and the encoder
applies the exact inverse `c = sign(T)·((|T| + ((1<<Fq)>>1)) >> Fq)` after
the forward DWT; the linear input/output scaling is the Annex G.2
`ζ = Bw − B[i]` up/down shift. The high-precision path carries 8
fractional bits through the transform, removing the per-level integer
rounding the `Fq = 0` path injects (bit-exact on smooth content at low
`q`). The decoder rejects any `(Bw, Fq)` outside Table A.8 and an NLT
marker present with `Fq = 0` (Annex A.4.6).

The encoder also supports **content-adaptive WGT weights**: the
`encode_planar_lossy_annex_h` entry point drives both the WGT marker and
the forward truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the
ISO/IEC 21122-1:2022 Annex H PSNR-optimized `(G[b], P[b])` example tables
(H.1 / H.2 / H.3 — 4:4:4, RCT, `NL,x = 5`, `NL,y ∈ {0, 1, 2}`), replacing
the default plain band-index priorities `P[b] = b` with the spec's richer
gains (LL up to `G = 4`) and reordered priorities. The companion
`encode_planar_subsampled_annex_h` entry point extends this to the
**chroma-subsampled** Annex H tables: H.4 / H.5 / H.6 (4:2:2, RCT
disabled, `NL,y ∈ {0, 1, 2}`) and H.7 / H.8 (4:2:0, RCT disabled,
`NL,y ∈ {1, 2}`). For the 4:2:0 tables the spec marks some band indices as
non-existent (`bx[β,i] = 0`, the `-*` slots); the encoder emits one
`(G[b], P[b])` pair per *existing* band only (Annex A.4.11 WGT loop), and
those `-*` positions are dropped so the supplied weights land in the
encoder's existing-band emission order — matching the
`picture_beta_to_local_beta` skip rule position-for-position. The
`encode_planar_star_tetrix_annex_h` entry point completes the set with the
**CFA Star-Tetrix** tables H.9 / H.10 / H.11 (`Cpih = 3`, `Sd = 1`,
4:4:4:4, `NL,x = 5`, `NL,y ∈ {0, 1, 2}`): the Star-Tetrix transform reads
all four CFA inputs and the fourth output (blue, Table F.4) is raw-coded
per Annex B Tables B.10 / B.11, so `NL = (Nc−Sd)·Nβ + Sd = 19 / 25 / 31`
bands. Each table tabulates a separate `(G[b], P[b])` column per CTS extent
`Cf ∈ {0, 3}` (full / restricted in-line); the encode `cf` selects the
column. Configurations outside any tabulated set fall back to the
default-weights path.

The Annex H weights are now also wired through the **high-bit-depth**
(`B[i] ∈ 9..=16`, little-endian `u16` planes) paths. For the CFA Star-Tetrix
layout: `encode_planar_star_tetrix_highbd_annex_h` drives the H.9–H.11
`(G[b], P[b])` columns over the two-bytes-per-sample CFA layout, and
`encode_planar_sd_star_tetrix_highbd` is its default-weights companion (the
high-bit-depth `Sd = 1` Star-Tetrix entry point). For the chroma-subsampled
4:2:2 / 4:2:0 layouts: `encode_planar_subsampled_highbd_annex_h` drives the
H.4–H.8 tables (including the `-*` non-existent-band drops) at high bit depth.
Bit depth and the Annex H weights are orthogonal — the gains / priorities and
the matching forward truncation act on `i32` wavelet coefficients, so the only
bit-depth-dependent pieces remain the Annex G.3 DC level shift and the `u16`-LE
plane packing. At `q = 0` every component self-roundtrips bit-exactly through
the highbd decode path even though the WGT advertises the H column.

The subsampled Annex H forward-truncation now indexes the supplied weights by
each band's position in the **β-major existing-band enumeration** (the same
cursor the decoder walks when loading WGT), rather than by the full picture
band index. This makes the encoder's `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)`
agree with the WGT the decoder reconstructs from for layouts whose chroma bands
do not all exist (4:2:0, where H.7 / H.8 carry `-*` slots) — previously such
streams advertised gains the encoder had not actually used for truncation, a
latent inconsistency a conforming decoder would have mis-decoded (it surfaces
as an entropy buffer over-read at high bit depth).

Wiring up the CFA tables required correcting an over-strict `Cpih = 3`
constraint: the encoder and decoder previously rejected `Nc − Sd < 4`,
conflating the Star-Tetrix *input* window (`c < 4`, Annex F.2 Table F.1)
with output suppression. Per Annex A.4.7 + Tables B.10 / B.11, `Sd`
suppresses the wavelet decomposition of trailing transform *outputs*
(coded raw), which the inverse transform consumes unchanged — so the only
requirement is `Nc ≥ 4` (four transform inputs). RCT (`Cpih = 1`) keeps the
stricter `Nc − Sd ≥ 3` guard (no tabulated RCT-with-suppressed-output
example).

The encoder also emits **verified conformance signalling** (the
`signalling` module). Every entry point historically wrote `Ppih = 0` /
`Plev = 0` / `Lcod = 0` — conforming but claiming nothing (ISO/IEC
21122-2 §A.2.2 / §A.5 exclude the unrestricted profile / level as
conformance points). The signalling layer patches the three PIH fields
in place in any already-encoded codestream and **verifies each claim
through the decoder's own gates** (`check_codestream` / `check_level` /
`check_codestream_size` / the Table-11 `Lcod` match) before keeping it,
so a false declaration cannot be emitted: `declare_profile` (Table A.5
`Ppih`), `declare_level_sublevel` (Tables A.12 / A.13 `Plev`, incl. the
§A.4.2 Full-sublevel-requires-profile rule), `declare_cbr` /
`declare_vbr` (Table 11 `Lcod`), the tightest-fit pickers
`pick_profile` / `pick_level` / `pick_sublevel`, and `declare_auto`.
`insert_com` writes COM extension segments (§A.4.10 — encoder-vendor /
copyright strings, vendor-specific data), completing write coverage of
every Annex A marker an encoder may emit.

On top of that sit two conformance-grade entry points:

- **`encode_planar_for_profile`** targets a named ISO/IEC 21122-2:2019
  profile. Every non-unrestricted profile mandates 16-image-row slices
  — a constraint no other entry point family composed with chroma
  sub-sampling and high bit depth at once — so it derives
  `Hsl = 16 / 2^NL,y` from the profile row, funnels the full
  `(sx, sy) × B[i] × Hsl × Qpih` composition through the common core,
  and signs the result (`Ppih` + tightest `Plev` + optional CBR
  `Lcod`). All eight 2019 profiles are pinned end-to-end: encode →
  verified declarations → gated decode → bit-exact plane compare.
- **`encode_planar_cbr_target_bytes`** emits a stream of *exactly*
  `target_bytes` bytes: the per-slice `Q[p]` rate allocation lands at
  or under the budget, COM padding closes the gap (re-allocating
  against `target − 6` when the residue is smaller than the 6-byte
  minimum segment), and `Lcod` truthfully declares the size.
- **`encode_planar_for_profile_cbr_target_bytes`** composes the two in
  one call: an exactly-`target_bytes` CBR stream that also claims — and
  provably satisfies — a named profile. The rate allocation
  (`pick_q_slices_for_profile_target_bytes`) runs the same three-pass
  ladder search generalised over the profile composition (per-component
  `(sx, sy)` sub-sampling, `bd ∈ 8..=16`, `Qpih`, the mandated
  16-image-row slices, with the spatial-activity ranking mapped through
  each component's own sub-sampling grid), and the level / sublevel are
  picked against the final CBR size — the byte count a constant-bitrate
  channel actually carries.

The Part-3 wrapper keeps the declarations consistent: the
`JxsFileBuilder` synthesises its `jxpl` Profile/Level box from the
wrapped codestream's own `Ppih` / `Plev` (A.5.3.3 defines the box as a
redundant early-parse copy of the PIH fields), and both `build` and
`decode_jxs_file` reject a box that contradicts a non-zero codestream
declaration, verify a known-profile box claim over an undeclared stream
through the Part-2 gates, and tolerate unknown (future-profile) code
points as advisory metadata.

**Encoder conformance pinning** (`tests/encoder_conformance.rs`):
ISO/IEC 21122-4 defines decoder conformance only, so the encoder side
pins the strongest checkable equivalent per stream — a SHA-256 of the
emitted bytes (wire-format changes must deliberately update the table),
a self-decode through the conformance-gated decoder (bit-exact for
every lossless case), and declaration truth for every claimed `Ppih` /
`Plev` / `Lcod`. The 16-stream matrix spans `Cpih ∈ {0, 1, 3}`, 4:2:0
sub-sampling, `B[i] = 12`, NLT quadratic, `Rm = 1`, exact-size CBR, all
eight profile targets, and the CBR × profile one-call composition.

#### Codestream parser

The marker-chain parser per ISO/IEC 21122-1:2022 Annex A recognises:

- `SOC` (`FF 10`), `EOC` (`FF 11`), `CAP` (`FF 50`, decoded into a
  typed `Capabilities` view), `PIH` (`FF 12`), `CDT` (`FF 13`),
  `WGT` (`FF 14`), `NLT` (`FF 16`), `CTS` (`FF 18`), `CRG` (`FF 19`),
  `COM` / `CWD`, and `SLH` (`FF 20`).
- Each header marker has a typed body accessor (`cts()`, `crg()`,
  `nlt()`, `wgt()`, `cwd()`, `com()`) surfacing field-level errors.

#### JXS still-image file format

The `fileformat` module parses the box-based **JXS still-image file
format** (ISO/IEC 21122-3:2019 Annex A) that optionally wraps a raw
ISO/IEC 21122-1 codestream. It walks the JPEG 2000-family box syntax
(A.3 Table A.1 — `LBox | TBox | [XLBox] | DBox`, all three `LBox` length
forms), validates the mandatory ordering (Signature box first, File Type
box next, Header box before the first Contiguous Codestream box), skips
unknown boxes (A.6), and surfaces typed bodies for the File Type box,
Image Header box (`ihdr`, with the Table A.17 sign-flag / varying-depth /
bit-depth decoding), Colour Specification box (`colr` CICP code points),
Channel Definition box (`cdef`), the Exif box (`exif`, opaque payload),
Profile/Level box (`jxpl`), and the **Video Information box** (`jpvi`,
A.5.3.2) from the Video Support superbox. Every box the 21122-3 Annex A
file format defines is recognised. The `jpvi` decode is a fully typed `VideoInformation` view of
Table A.5: `brat` max bit rate, the `frat` frame-rate rational with its
`InterlaceMode` (Table A.7) / `FrameRateDenominator` (1.000 / 1.001,
Table A.8) / numerator sub-fields and the `frat = 0` unknown sentinel,
the `schar` sample characteristics (`Valid_Flag`, 4-bit `Sample_Bitdepth`,
`SamplingStructure` Table A.10), and the `tcod` `HH:MM:SS:FF` time code —
each with the spec's reserved-value and range rejections. The two
remaining optional Video Support boxes this document defines are also
typed: the **Mastering Display Metadata box** (`dmon`, A.5.3.5 — SMPTE
ST 2086 primaries + white point, `Lmin`/`Lmax` luminance and the
CTA-861-G `MCLL`/`MFALL` content light levels, with the 0…50000
chromaticity and `Lmin < Lmax` range checks) and the **Video Transport
Parameter box** (`jptp`, A.5.3.6 — `Slgs` slice-group size / `Rsync`
parallel units, with the reserved `Tseq` / `MTU` fields fixed at 0). The
**Buffer Model Description box** (`bmdm`, A.5.3.4 — `Tbmd` model type +
`Ncg,hz` / `Ncg,vt` blanking-period coefficient-group counts) completes
the five A.5.3 Video Support boxes; only the buffer-model *bounds* (which
need a transmission-channel rate) stay out of scope, not the metadata.

`decode_jxs_file` extracts the embedded codestream, cross-checks the
`ihdr` geometry against the codestream picture header (A.5.4.2 rejects
contradictory files), and decodes through the standard path. The
registry `Decoder` accepts either a bare codestream or a box-wrapped
`.jxs` file, routing on the leading signature; `is_jxs_file`
discriminates the two.

The four 21122-3 **Media Type registrations** (RFC 6838) are also
surfaced: `media_type` classifies a buffer by the two registered magics
— `image/jxs` (§A.7.2, the 12-byte Signature box) and `image/jxsc`
(§D.2.2, the `FF10 FF50` SOC + CAP prefix of a bare codestream carried
outside any file format, e.g. as an RTP payload) — and the HEIF-side
`image/jxsi` / `image/jxss` (§C.5.2 / §C.6.2, no registered magic;
HEIF parsing itself is out of this crate's scope) are exposed as
constants.

A matching writer round-trips the wrapper: `JxsFileBuilder` /
`write_jxs_file` serialize a conforming box file around an encoded
codestream — deriving the `ihdr` geometry from the picture header,
carrying a CICP colour specification, and optionally emitting a Channel
Definition box and a JPEG XS Video Support superbox (`jpvs`). When a
`jpvs` is emitted it is **always conforming**: A.5.3.1 makes both the
`jpvi` Video Information box and the `jxpl` Profile/Level box mandatory,
so the builder emits them in the A.5.3 order (`jpvi` first, `jxpl`
second), synthesising a minimal-conforming default (`VideoInformation::
unknown` / `Unrestricted` `jxpl`) for whichever the caller omits via
`video_information(...)` / `profile_level(...)`. The optional `dmon` /
`jptp` boxes (`mastering_display(...)` / `transport_parameters(...)`)
follow `jxpl` in the Figure A.7 order.

#### Fuzzing

`fuzz/` is a cargo-fuzz harness (its own workspace) with three targets:
`decode` (arbitrary bytes through media-type / probe /
`verify_declarations` / full decode, asserting decoded-geometry
consistency on success), `jxs_file` (the Part-3 box parser plus the
wrapped decode path), and `roundtrip` (a structured target — the fuzzer
bytes pick an encoder configuration and the plane samples across four
entry-point axes: generic sub-sampled, `Qpih = 1`, `Rm = 1`, and
explicit multi-slice; every configuration the encoder accepts must
decode, bit-exactly at `q = 0`).
`cargo run --bin seed_gen` (from `fuzz/`) writes feature-spanning
corpus seeds. Two hardening fixes came out of the initial campaigns
(encoder-side Table 11 minimum dimensions; the 32-bit bitplane-count
representability cap).

#### Profile / level surface

The `profile` module implements the ISO/IEC 21122-2:2019 Annex A
profile / level / sublevel tables: `Profile::from_ppih`,
`Level::from_plev_high`, `Sublevel::from_plev_low_byte`, and
`check_profile` / `check_level` enforcing every codestream-observable
constraint (component count, bit depth, chroma format, decomposition
depths, `Qpih`, slice-height, column-mode caps). Buffer-model bounds
(Annexes B/C/D) are out of scope — they require a transmission-channel
rate not observable from the codestream.

#### ISO/IEC 21122-4 conformance

The decoder is exercised against the official **ISO/IEC 21122-4:2020**
decoder conformance codestreams (the 65-stream, 1.4 GB reference-vector
set — codestreams paired with sample-exact reference decoded images). The
`tests/conformance.rs` harness is an opt-in gate: point
`OXIDEAV_JXS_CONFORMANCE_DIR` at the unpacked attachment set and it decodes
each `N.jxs`, then compares every reconstructed component plane
sample-by-sample against its `pgx` reference (§B.8 — a pass is exact
equality on every sample; per §B.6 the normative RCT / Star-Tetrix inverse
is applied while non-normative YCbCr→RGB and 4:2:2→4:4:4 upsampling are
not, so planes are compared at native subsampled resolution). Without the
directory the harness self-tests the `pgx` reader against synthetic
fixtures, so CI stays green without the archives (catalogued, with
per-file SHA-256, in `docs/image/jpegxs/conformance/`).

**The decoder reconstructs all 65 published conformance codestreams
(streams 2–66) bit-exact — 65 pass / 0 unsupported / 0 fail** — covering
every profile / level / sublevel in the vector set, including the
4-component (`Nc = 4`, `Cpih = 1`) stream 64 whose 4th component is the
RCT's pass-through partner (Annex F Table F.1: `Ω[3] = O[3]`, no colour
math, but the full entropy / dequant / DWT / Annex G scaling pipeline).
The harness also reports per-profile **ETS verdicts** (Annex C / §B.8: a
profile conforms only if every stream of its test codestream set is
sample-exact) — all nine sets pass, including the two four-component
ones (C.6 Main 4444.12 at 10/10, C.8 High 4444.12 at 10/10) and C.10
Unrestricted (4/4).

Wiring these vectors in surfaced — and fixed — three structural bugs the
crate's self-roundtrips could never catch:

- **Joint multi-component packets**: Annex B.7 Table B.4 codes all
  components of a `(β, line)` jointly in one packet (the new-packet flag
  `r` resets per `(λ, β)`, not per component), where the planner had
  emitted one packet per band. Both directions now group jointly, and the
  encoder commits one bitplane-count `D[p,b]` mode per packet group.
- **`Fs = 1` sign bits for meaningless tail coefficients** (Table C.9
  NOTE 2): a band whose `Wpb` is not a multiple of `Ng` transmits
  "meaningless" tail coefficients in its last code group, and each
  non-zero tail magnitude carries a sign bit in the sign sub-packet.
  Skipping them desynchronised every following sign in the packet —
  invisible until the next negative coefficient, which is exactly the
  stream-64 "4th component diverges at an interior pixel" signature (two
  displaced sign bits in its 2047-wide level-1 band) and the failure mode
  of 12 further odd-width `Fs = 1` vectors.
- **`NL,y ≥ 1` is picture-level DWT territory**: the `NL = 1/1` layout
  used a per-precinct streaming DWT (both directions) that reflected the
  5/3 vertical filter at every interior 2-line precinct boundary. Annex E
  defines a picture-level transform — symmetric extension exists at the
  picture edges only — so every real multi-precinct `NL,y = 1` stream
  (conformance 29 / 30 / 37–41) diverged from output row 1 down. All
  `NL,y ≥ 1` layouts now take the gather/cascade path; the streaming fast
  path remains for `NL,y = 0` where one single-column precinct row is a
  complete horizontal transform unit.

#### Not yet covered

- Bit depths above 16 (would need a `u32` plane format; B > 16 also has
  no published decode vector to validate against).
- All Annex H example tables are transcribed and wired through both the 8-bit
  and the high-bit-depth (`B[i] ∈ 9..=16`) encode paths: the 4:4:4 RCT tables
  (H.1–H.3, `encode_planar_lossy_annex_h`), the subsampled 4:2:2 / 4:2:0 tables
  (H.4–H.8, `encode_planar_subsampled_annex_h` / `_subsampled_highbd_annex_h`,
  including the `-*` non-existent-band handling), and the CFA Star-Tetrix tables
  (H.9–H.11, `encode_planar_star_tetrix_annex_h` /
  `_star_tetrix_highbd_annex_h`, `Cf = 0` / `Cf = 3` columns).

### Depth API

The contract surface above is the common floor; the depth modules stay
public for callers who need the marker chain or the Annex-level kernels:

- `codestream::parse` → `Codestream` (CAP / PIH / CDT / WGT / NLT / CWD /
  CTS / CRG / COM / slices), `inspect` → `JpegXsFileInfo`.
- `encoder::encode_planar_*` — the historical per-axis entry points,
  deprecated in favour of `encode` / `encode_components` +
  `EncodeOptions` (every axis is a field); their byte output is pinned
  by `tests/encoder_conformance.rs`.
- `signalling` (`declare_profile` / `declare_level_sublevel` /
  `declare_cbr` / `declare_auto` / `pick_*` / `pad_to_size` /
  `insert_com` / `verify_declarations`) upgrades any codestream to a
  verified self-describing one.
- `fileformat` (`parse_jxs_file` → `JxsFile`, `JxsFileBuilder`,
  `write_jxs_file`, `media_type`) — the ISO/IEC 21122-3 Annex A box
  layer, including the Video Support superbox records.
- `profile` (`Profile` / `Level` / `Sublevel` / `check_profile` /
  `check_level` / `check_codestream_size`) — ISO/IEC 21122-2.

## License

MIT — see [LICENSE](LICENSE).
