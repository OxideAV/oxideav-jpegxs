# Changelog

## Unreleased — round 398 (transport: Exif box in the JPEG XS Header box)

Recognises the last defined box that was previously silently dropped: the
`exif` box (ISO/IEC 21122-3:2019 A.5.4 Table A.2) inside the JPEG XS
Header superbox (`jp2h`).

* **Reader** surfaces the opaque EXIF payload as `HeaderBox::exif`
  (`Option<Vec<u8>>`); the EXIF contents themselves stay opaque.
* **Writer** gains `JxsFileBuilder::exif(...)`, emitting the box inside
  `jp2h` after the Channel Definition box.

* +2 tests (616 → 618). Every box the 21122-3 Annex A file format defines
  is now recognised by the parser.

## Unreleased — round 398 (transport: `bmdm` Buffer Model Description box)

Completes all five ISO/IEC 21122-3:2019 A.5.3 boxes of the JPEG XS Video
Support superbox (`jpvi`, `jxpl`, `bmdm`, `dmon`, `jptp`).

* **Buffer Model Description box** (`bmdm`, A.5.3.4, Table A.12 — 6
  bytes): typed `BufferModelDescription` (`Tbmd` model type 0/1/2,
  `Ncg,hz` / `Ncg,vt` blanking-period coefficient-group counts). The
  `Tbmd` domain and the `Rd == 0` reserved field are enforced on both
  parse and serialize.
* **Reader** extracts it into `JxsFile::buffer_model`. **Writer** gains
  `buffer_model(...)`; the box is emitted after `jxpl` and before `dmon`
  per the Figure A.7 order (`jpvi`, `jxpl`, `bmdm`, `dmon`, `jptp`).
* Public API adds `BufferModelDescription`.

* +2 tests (614 → 616).

## Unreleased — round 398 (transport: `dmon` + `jptp` Video Support boxes)

Completes the four ISO/IEC 21122-3:2019 A.5.3 boxes contained in the JPEG
XS Video Support superbox that this document defines: after `jpvi` /
`jxpl`, the two remaining optional boxes are now typed.

* **Mastering Display Metadata box** (`dmon`, A.5.3.5, Figure A.11 — 28
  bytes): typed `MasteringDisplayMetadata` covering the SMPTE ST 2086
  primaries (green `c0` / blue `c1` / red `c2`), white point, `Lmin` /
  `Lmax` display luminance and the CTA-861-G `MCLL` / `MFALL` content
  light levels. Enforces the spec ranges — chromaticity 0…50000, `Lmin <
  Lmax`. Verified against the Table A.13 Rec. BT.709 defaults.
* **Video Transport Parameter box** (`jptp`, A.5.3.6, Table A.15 — 6
  bytes): typed `VideoTransportParameters` (`Slgs` slice-group size,
  `Rsync` parallel units); the `Tseq` (Table A.14) and `MTU` fields are
  fixed at 0 and any other value is rejected as reserved.
* **Reader** extracts both into `JxsFile::mastering_display` /
  `transport_params`. **Writer** gains `mastering_display(...)` /
  `transport_parameters(...)`; both boxes are emitted after `jxpl` in the
  Figure A.7 order (`jpvi`, `jxpl`, `dmon`, `jptp`), and either one alone
  still triggers a conforming `jpvs` (mandatory `jpvi` / `jxpl`
  synthesised).
* Public API adds `MasteringDisplayMetadata`, `VideoTransportParameters`.

* +6 tests (608 → 614).

## Unreleased — round 398 (transport: JPEG XS Video Information box `jpvi`)

Closes the mandatory-but-missing `jpvi` box inside the JPEG XS Video
Support superbox (ISO/IEC 21122-3:2019 A.5.3.2). Previously the `jpvs`
superbox carried only the `jxpl` Profile/Level box — a non-conforming
layout, since A.5.3.1 makes both `jpvi` and `jxpl` mandatory whenever a
`jpvs` is present.

* **New typed `VideoInformation` box** (Table A.5 — `brat` / `frat` /
  `schar` / `tcod`, 14 bytes) with full sub-field decode:
  * `frat` (Table A.6): `InterlaceMode` (Table A.7, reserved value 3
    rejected), `FrameRateDenominator` (Table A.8, 1.000 / 1.001, other
    values rejected), 16-bit numerator, and the `frat == 0` "unknown"
    sentinel modelled as `FrameRate::Unknown`.
  * `schar` (Table A.9): `Valid_Flag` gating, 4-bit `Sample_Bitdepth`,
    and `SamplingStructure` (Table A.10 — reserved 3 / 7–15 rejected;
    `Valid_Flag == 0` requires all sub-fields zero).
  * `tcod`: `HH:MM:SS:FF` with range enforcement (HH 0–23, MM 0–59,
    SS 0–59, FF 1–60); `brat == 0` rejected (Table A.5 range 1…2³²−1).
* **Reader** (`parse_jxs_file`) now extracts the `jpvi` box from the
  `jpvs` superbox into `JxsFile::video_info` (order-tolerant per A.6).
* **Writer** (`JxsFileBuilder`) gains `video_information(...)`. Emitting a
  `jpvs` now always produces a conforming superbox — `jpvi` first, `jxpl`
  second (A.5.3 ordering) — synthesising a minimal-conforming default
  (`VideoInformation::unknown` / `Unrestricted` `jxpl`) for whichever the
  caller omits. This fixes the prior `jxpl`-only `jpvs` output.
* Public API adds `VideoInformation`, `FrameRate`, `InterlaceMode`,
  `FrameRateDenominator`, `SampleCharacteristics`, `SamplingStructure`,
  `TimeCode`.

* +7 tests (601 → 608). README documents the `jpvi` transport surface.

## Unreleased — round 382 (run mode Rm=1 — significance-dispatch + geometry coverage)

Coverage deepening the `Rm = 1` significance path across the axes it must
compose with:

* **Multiple significance groups** (`Ns > 1`) — a 140-wide band at
  `NL = 2` (LL width 35 `> Ng·Ss = 32`) exercises the per-group `Z`
  dispatch under the zero-coefficient run mode.
* **Deep odd-dimension cascade** (`NL = 3`, 67×83) — the §B.1
  ceiling-sized band geometry and the cross-line vertical predictor
  compose with `Rm = 1`.
* **Truncation ladder** (`q = 0..=6`) — every quantization step decodes
  through the standard path (the `M = T` insignificant reconstruction
  never demands data the encoder omitted, so all length-conformance
  gates pass), bit-exact at `q = 0`.

* +3 tests (598 → 601). README documents the run-mode surface.

## Unreleased — round 382 (run mode Rm=1 — high-bit-depth + chroma-subsampled compositions)

`Rm = 1` now composes with the two orthogonal feature axes the rest of
the crate carries, matching the established composition pattern:

* `encode_planar_run_mode1_highbd(w, h, nc, cpih, nlx, nly, bd, planes)`
  — `B[i] ∈ 9..=16`, `u16`-LE planes. `Rm` gates only the
  insignificant-group `M` reconstruction, which is bit-depth-independent,
  so the zero-coefficient significance runs over the wider `Bw = bd`
  wavelet domain unchanged (verified bit-exact at `B[i] = 12`).
* `encode_planar_run_mode1_subsampled(w, h, nc, cpih, nlx, nly, q, sx,
  sy, planes)` — 4:2:2 / 4:2:0 sampling. A `sy = 2` component decomposes
  one vertical level shallower, so the insignificant-group logic runs
  across the mixed-depth band cascade (verified bit-exact for 4:2:0).

* +2 tests (596 → 598).

## Unreleased — round 382 (run mode Rm=1 — encoder emission + decode-path coverage)

The decoder has long carried the `Rm = 1` (Table A.12, "runs indicate
zero coefficients") branch of the vertical-prediction bitplane-count
decoder (C.6.5 Table C.13, `Δm = T − mtop` for an insignificant group),
but no encoder ever produced an `Rm = 1` stream, so that branch was
never exercised end-to-end. This round closes the loop:

* **Encoder** — `EncodeConfig` gains an `rm` field (default 0), emitted
  in the PIH `Lh:Rl:Qpih:Fs:Rm` byte. When `rm == 1` the significance
  computation drops the `Rm = 0` predecessor guard: a code group is
  insignificant iff *all* its coefficients are zero after truncation
  (`M ≤ T`), and the decoder then reconstructs it to `M = T[p,b]`
  (data + VLC residual both elided). The new
  `encode_planar_run_mode1(w, h, nc, cpih, nlx, nly, q, planes)` entry
  point (8-bit planar, `Cpih ∈ {0, 1}`, lossless / lossy) selects it.
  `EncodeConfig::validate` rejects the reserved `Rm ∈ {2, 3}`.
* **Coverage** — round-trip tests (luma, RGB+RCT, lossy PSNR) confirm
  `Rm = 1` self-decodes bit-exactly at `q = 0`, plus two hand-built
  packet-body decode tests that pin the branch as behaviourally
  load-bearing: the same `Mtop = (5, 4)` insignificant group
  reconstructs to `M = (0, 0)` under `Rm = 1` versus `M = (5, 4)` under
  `Rm = 0`. `Rm = 1` never encodes larger than `Rm = 0` on the same
  content (the per-band `D[p,b]` picker keeps the smallest form).

* +7 tests (589 → 596).

## Unreleased — round 376 (Full sublevel requires a non-unrestricted profile)

Enforces §A.4.2: "The Full sublevel shall only be used if the profile
value is not unrestricted." A stream whose `Plev` low byte selects the
Full sublevel (0x80) while `Ppih = 0` (Unrestricted) has no defined
`Nbpp` — Table A.7 defers Full's bits-per-pixel to the profile's
max-decoded-bpp — so the combination is non-conformant and
`check_codestream_size` now rejects it.

* +2 tests (587 → 589).

## Unreleased — round 376 (Table A.8 "additional constraints" enforced at decode)

The decoder enforced the (Bw, Fq) pairing of Table A.8 but not the
table's "Additional constraints" column. Now also enforced:

* **(B[0], 0)** — the lossless integer-transform case requires
  `B[i] = B[0]` for every component (uniform bit depth). A stream with a
  non-uniform CDT bit depth at Fq=0 is rejected.
* **(18, 6)** — requires a NLT marker present *and* CAP bit 2 or bit 3
  set (quadratic / extended NLT capability).

Deliberately **not** enforced: the (20, 8) "CAP bits 2 and 3 shall be 0"
half. This crate's high-bit-depth NLT path runs the wavelet domain at
Bw=20 for precision headroom and emits (Bw=20, Fq=8) *with* an NLT marker
— a combination Table A.8 does not tabulate (it offers only (18, 6) for
NLT, which lacks headroom for `B[i]` up to 16). This is the known
**NLT@Bw=20 docs gap**; enforcing the rule would reject our own
high-bd NLT streams.

* +2 tests (585 → 587).

## Unreleased — round 376 (CTS-marker presence is fully determined by Cpih)

The codestream parser already required a CTS marker when `Cpih=3`; it now
also enforces the converse half of the Table A.2 rule — "the CTS marker
shall not be present otherwise" — rejecting a CTS segment carried by any
`Cpih ≠ 3` stream. Reworked `cts_method_surfaces_body_errors` (which had
relied on a now-illegal Cpih=0 + CTS layout) onto a valid Cpih=3
Star-Tetrix shell with a malformed CTS body.

* +1 test (584 → 585).

## Unreleased — round 376 (CAP capabilities marker — emission + conformance)

The CAP marker declares which optional decoder tools are required to
decode a codestream (ISO/IEC 21122-1:2022 §A.4.3, Table A.5). Two halves:

* **Encoder** now builds a real `cap[]` bit array reflecting the tools it
  uses, replacing the previous always-empty `Lcap = 2`: bit 1
  (Star-Tetrix, `Cpih=3`), bit 2 / 3 (quadratic / extended NLT), bit 4
  (a component with `sy>1`), bit 5 (CWD, `Sd>0`), bit 6 (lossless,
  `q=0`), bit 8 (per-packet raw-mode switch, always — the encoder signals
  `Rl=1`). `Lcap` is shrunk so the last byte is never all-zero.
* **Decoder** rejects a stream whose CAP sets a bit it does not implement
  — bit 0 (unused), bit 7, or any reserved bit ≥ 9 — per §A.4.3 ("abort
  on a capability it does not implement"). New
  `capabilities::unsupported_cap_bits` (re-exported).

* +6 tests (578 → 584).

## Unreleased — round 376 (Cw rightmost-precinct conformance, both directions)

Enforces the Table 11 / §B.5-Note-1 `Cw` constraint that "all bands of
the rightmost precincts are non-empty":

```text
  Wf umod (8 × Cw × max_i(sx) × 2^NL,x) ≥ max_i(sx) × 2^NL,x   (Cw ≠ 0)
```

i.e. the rightmost partial precinct must hold at least one decomposed
low-frequency sample, or divide `Wf` evenly. The decoder rejects a stream
whose rightmost precinct is a sub-LL-sample sliver; the encoder refuses to
emit one (it previously left this to the caller).

Corrects `r282_odd_width_422_cw1_lossless_round_trip`, which used a 65×16
4:2:2 Cw=1 picture whose rightmost precinct is a single column (`65 umod
32 = 1 < 4`) — a non-conformant geometry. Re-pinned to width 69 (`69 umod
32 = 5 ≥ 4`), preserving the odd-width Np,x=3 intent. The rejected 65-wide
case is now an explicit encoder-refusal test.

* net +1 test (577 → 578: +1 encoder rejection, 1 fixture re-pinned).

## Unreleased — round 376 (picture-dimension vs decomposition conformance)

The decoder now enforces the Wf / Hf picture-dimension ranges of
ISO/IEC 21122-1:2022 Table 11:

```text
  Wf ≥ max_i(sx[i]) × 2^NL,x
  Hf ≥ max_i(sy[i]) × 2^NL,y
```

A picture smaller than one fully-decomposed low-frequency sample in any
component cannot carry the LL band the header declares, so the geometry
is internally inconsistent and the stream is rejected.

* +4 tests (573 → 577).

## Unreleased — round 376 (CBR Lcod self-length conformance enforced at decode)

The decoder now validates the picture header's `Lcod` field — "the size
of the entire codestream in bytes from SOC to EOC, including all markers,
if constant-bitrate coding is used; 0 if variable-bitrate coding is used"
(ISO/IEC 21122-1:2022 Table 11). A non-zero `Lcod` is a CBR
self-description that must equal the codestream's actual SOC-to-EOC byte
count (the EOC marker ends at `eoc_offset + 2`); a mismatch is rejected.
`Lcod = 0` (VBR, what this crate's encoder emits) imposes no check.

* +4 tests (569 → 573).

## Unreleased — round 376 (sublevel coded-size bound Ssl,max enforced at decode)

The decoder now enforces the sublevel coded-domain bound: the SOC-to-EOC
codestream byte count must not exceed

```text
  Ssl,max = floor(Lmax × Nbpp / 8)
```

(ISO/IEC 21122-2:2019 §A.4.1), where `Lmax` is the declared level's
maximum sampling-grid-point count and `Nbpp` is the declared sublevel's
nominal bits-per-pixel (Table A.7; profile-dependent for the `Full`
sublevel via Table A.4). New `profile::max_codestream_size` /
`check_codestream_size` and `Sublevel::effective_nbpp`; the closed form
is reproduced bit-exactly against every numeric entry of Tables A.8–A.11
(3 / 6 / 9 / 12 bpp × eight levels). Unrestricted level or sublevel
imposes no bound.

* +5 tests (564 → 569).

## Unreleased — round 376 (profile / level conformance enforced at decode)

The decoder now validates every codestream against the profile (`Ppih`,
Table A.5) and level/sublevel (`Plev`, Tables A.12/A.13) it declares,
per ISO/IEC 21122-2:2019 Annex A. Previously `Ppih` / `Plev` were parsed
and surfaced through `probe()` but never enforced during `decode`:

* A non-zero `Ppih` mapping to a known profile runs
  `profile::check_codestream`, rejecting a stream whose header
  contradicts its own profile claim (component count, bit-depth set,
  chroma format, `NL,x` / `NL,y`, `Qpih`, slice height, column mode).
* A non-zero `Ppih` mapping to no profile is reserved for ISO/IEC and is
  rejected rather than decoded under an unknown profile.
* `Ppih = 0` (Unrestricted) keeps the check a no-op.
* `profile::check_level` bounds `Wf` / `Hf` / `Wf×Hf` against the
  declared level's `Wmax` / `Hmax` / `Lmax` (Table A.6) and rejects a
  reserved `Plev` high byte.

* +5 tests (559 → 564).

## Unreleased — round 371 (Sd>0 CWD × NLT path-intersection coverage)

Adds a round-trip exercising CWD decomposition suppression (`Sd > 0`)
composed with the NLT quadratic non-linearity. With Nc=4, Sd=1 the
trailing component is coded raw (Annex B.7 Table B.4) while the leading
three are wavelet-coded; both force the picture-level gather path. The
forward pre-distortion applies per-component including the suppressed
tail (copied straight through the entropy coder), and the decoder's
per-component Annex G inverse restores every component ≥ 40 dB —
confirming the non-obvious raw-tail-through-NLT-inverse composition.

* +1 test (558 → 559).

## Unreleased — round 371 (Cw>0 × NLT path-intersection coverage)

Adds a round-trip exercising the intersection of the two decode paths
that both force the picture-level gather/cascade: multi-precinct-per-row
(`Cw > 0`) and the NLT quadratic non-linearity (Tnlt=1). A 96-wide luma
picture at NL,x=2 Cw=1 splits into Np,x=3 precincts per row while the
Annex G.4 forward pre-distortion runs over the plane before gather;
round-trip holds ≥ 40 dB.

* +1 test (557 → 558).

## Unreleased — round 371 (Cw>0 × 4:2:0 multi-level coverage + stale-doc fix)

Adds a bit-exact lossless round-trip for multi-precinct-per-row
(`Cw > 0`) composed with 4:2:0 chroma sub-sampling at a multi-level
cascade (NL=2/2): a 128-wide picture with Cw=1 splits into Np,x=2
precincts per row, routed through the picture-level gather/cascade path
where the sub-sampled chroma decomposes one vertical level shallower.

Also corrects the stale encoder module-doc "out-of-scope" list — `Cw>0`,
per-precinct `Q[p]`/`R[p]`, and `Sd>0` (CWD) are all long-implemented.

* +1 test (556 → 557).

## Unreleased — round 371 (sub-sampled NLT lossy + odd-dimension coverage)

Adds round-trip coverage for the new sub-sampled NLT paths under the
lossy regime and odd picture dimensions:

* lossy `q = 2` (`Fq = 8` per-band deadzone truncation) 4:2:0 NLT
  quadratic — stream ≤ lossless, all components ≥ 30 dB.
* odd-dimension (30×18) 4:2:0 NLT quadratic — the §B.1 ceiling sizes the
  chroma planes to 15×9; lossless self-roundtrip ≥ 40 dB.

* +2 tests (554 → 556).

## Unreleased — round 371 (high-bit-depth NLT + chroma sub-sampling)

Extends the round-371 NLT × chroma sub-sampling composition to the
high-bit-depth (`B[i] ∈ 9..=16`, `Bw = 20`, `u16`-LE plane) paths:
`encode_planar_subsampled_nlt_quadratic_highbd` (Tnlt=1) and
`encode_planar_subsampled_nlt_extended_highbd` (Tnlt=2) take per-component
`(sx[i], sy[i]) ∈ {1, 2}`, driving a 4:2:2 / 4:2:0 luma + chroma picture
through the high-precision NLT path.

* +3 tests (551 → 554): 12-bit 4:2:2 and 10-bit 4:2:0 NLT-quadratic
  round-trip (all components ≥ 40 dB), 12-bit 4:2:2 NLT-extended
  round-trip (≥ 30 dB).

## Unreleased — round 371 (NLT + chroma sub-sampling encode paths)

Adds two public encoder entry points composing the Annex G non-linear
transforms (NLT) with chroma sub-sampling: `encode_planar_subsampled_
nlt_quadratic` (Tnlt=1, Annex G.4) and `encode_planar_subsampled_nlt_
extended` (Tnlt=2, Annex G.5). Both take per-component `(sx[i], sy[i]) ∈
{1, 2}` and drive a luma + 4:2:2 / 4:2:0 chroma picture through the NLT
marker path. The forward pre-distortions are per-sample, per-component
maps, so they are orthogonal to the sub-sampled plane geometry (each
chroma plane is §B.1-ceiling-sized and decomposes one vertical level
shallower per `log2(sy[i])`). RCT (`cpih = 1`) with a sub-sampled
component is rejected (Annex F.2).

* +4 tests (547 → 551): 4:2:2 and 4:2:0 NLT-quadratic round-trip (all
  three components ≥ 40 dB), 4:2:2 NLT-extended round-trip (≥ 30 dB), and
  the RCT-with-sub-sampling rejection guard.

## Unreleased — round 366 (probe accepts box-wrapped JXS files)

The crate-root `probe` now transparently accepts a box-wrapped JXS
still-image file (ISO/IEC 21122-3 Annex A) as well as a bare ISO/IEC
21122-1 codestream: when the buffer leads with the JPEG XS Signature
box it locates the embedded codestream and probes that, so a routing
layer handed a real `.jxs` file gets the same `JpegXsFileInfo` it would
from the raw codestream. Bare-codestream probing (`FF 10` lead) is
unchanged.

* +2 tests (545 → 547): a wrapped file and its bare codestream probe to
  identical geometry; garbage / truncated-box buffers return `None`.

Also refines `JxsFileBuilder`: when the codestream's components do not
all share one bit depth it now emits the Table A.17 `0xFF`
"components vary in bit depth" `BPC` code instead of the maximum depth,
keeping the emitted `ihdr` spec-correct for any conforming codestream
(uniform-depth streams — every stream this crate's encoder produces —
are byte-identical to before).

## Unreleased — round 366 (JXS file-format writer + builder)

Completes the JXS still-image file-format round with a **writer** that
serializes a conforming ISO/IEC 21122-3 Annex A box wrapper around an
encoded codestream, giving a real encode → wrap → decode round-trip.

* `fileformat::JxsFileBuilder` — emits the Signature, File Type, JPEG XS
  Header (ihdr + CICP colr + optional cdef), optional `jpvs`/`jxpl`
  Profile-Level, and Contiguous Codestream boxes. The `ihdr`
  `WIDTH`/`HEIGHT`/`NC`/`BPC` are derived from the codestream picture
  header (A.5.4.2), so the file is internally consistent by
  construction; the builder takes a CICP colour spec and optional
  channel definitions / profile-level / `UnkC` flag.
* `fileformat::write_jxs_file` — convenience wrapper emitting a minimal
  CICP-sRGB file with no auxiliary boxes.
* Both stay `oxideav-core`-free (standalone build unaffected) and are
  re-exported at the crate root.
* +5 tests (540 → 545): luma write → decode round-trip; 3-component
  RGB with a Channel Definition box + a Profile/Level box, all
  round-tripping; 12-bit high-bit-depth `BPC` derivation + decode;
  invalid-codestream rejection; full-range CICP carriage.

## Unreleased — round 366 (JXS still-image file format — ISO/IEC 21122-3 Annex A box parser)

New decode subsystem: the box-based **JXS still-image file format**
(ISO/IEC 21122-3:2019 Annex A) that optionally wraps a raw ISO/IEC
21122-1 codestream. Until now the crate decoded only the bare codestream
(`SOC … EOC`); a `.jxs` file produced by a conforming writer carries the
codestream inside a JPEG 2000-family box structure, which this round
parses end-to-end and round-trips through the existing codestream
decoder.

* New `fileformat` module (`src/fileformat.rs`):
  * `parse_box_header` / `parse_boxes` — the A.3 Table A.1 box syntax
    (`LBox | TBox | [XLBox] | DBox`, big-endian). Handles all three
    `LBox` length encodings: literal (`>= 8`), the `LBox = 1` 8-byte
    `XLBox` extended length, and the `LBox = 0` to-end-of-buffer form;
    rejects the reserved `2..=7` values and any box that overruns its
    container.
  * Typed box bodies with body-level validation: `FileType` (A.5.2,
    brand + minor-version + compatibility list, `'jxs\040'`-compat
    check), `ImageHeader` (A.5.4.2 / Table A.16–A.17, with the BPC
    sign-flag / varying-depth (`0xFF`) / 38-bit-max decoding),
    `ColourSpec` + `Cicp` (A.5.4.3, the CICP `METH = 5` code points),
    `ChannelDefinition` (A.5.4.4 Table A.25), `ProfileLevel` (A.5.3.3
    Table A.11).
  * `parse_jxs_file` — enforces the mandatory ordering (Signature box
    first, File Type box immediately after, Header box before the first
    Contiguous Codestream box; A.2.3 / Figure A.1), skips unknown boxes
    (A.6), and pulls the optional `jxpl` Profile/Level box out of a
    `jpvs` Video Support superbox.
  * `is_jxs_file` — signature-box discriminator between a `.jxs` file
    and a bare codestream.
  * `decode_jxs_file` — extracts the first codestream and decodes it,
    cross-checking the `ihdr` `NC` / width / height against the
    codestream picture header (A.5.4.2 declares contradictory files
    non-conforming).
* The `registry` `Decoder` now routes each packet on its leading
  signature: a box-wrapped JXS file is unwrapped via `decode_jxs_file`,
  a bare codestream takes the existing path. No new container crate —
  the box layer is intrinsic to the `.jxs` still-image format. The
  module stays `oxideav-core`-free, so the standalone (no-`registry`)
  build is unaffected.
* +17 tests (523 → 540): box-chain parse + codestream extraction, full
  luma round-trip through the wrapper, `XLBox` / `LBox = 0` length
  forms, reserved-`LBox` rejection, missing-signature / misplaced-ftyp
  rejection, `ihdr`↔codestream `NC` and dimension mismatch rejection,
  unknown + XML box skipping, per-box-body parse/validation units, and
  two `Decoder`-trait tests (box-wrapped and bare codestream).

## Unreleased — round 362 (multi-significance-group coverage + Ng/Ss conformance gate)

Two threads, both anchored in Annex A.4.4 Table A.7 / Annex B.8–B.9.

**Multiple significance groups per band-line (`Ns > 1`).** The significance
sub-packet (Table C.5) carries one `Z` bit per significance group, and a
band-line holds `Ns[p,b] = ⌈Wpb / (Ng·Ss)⌉` of them (`Ng = 4`, `Ss = 8` ⇒ a
fresh `Z` bit every 32 code-positions). Every prior significance round-trip
used a picture narrow enough that the widest band stayed at `Wpb ≤ 32`, so
`Ns = 1` and the `sig_group = g / Ss` group dispatch on both the encoder
(`group_sig`) and decoder (`sig_flags` keyed by group `j`) collapsed to a
single group. The new flat-field NL=2 tests push the LL/HL bands past 32
positions so the encoder selects the significance-coded `D & 2` form with
`Ns ∈ {2, 3}`, and the decode path resolves each `Z` bit to the correct
group span — verified to carry both insignificant (`Z = 1`) and significant
(`Z = 0`) bits in groups `j > 0`.

* +3 tests (520 → 523):
  * `round362_significance_ns2_nl2_flat_round_trips` — flat 132×64 luma, NL=2;
    LL band width 33 (`Ns = 2`), HL bands width 66 (`Ns = 3`), significance
    coding active across all bands.
  * `round362_significance_ns2_rgb_rct_round_trips` — `Ns = 2` composed with
    3-component RGB + the reversible colour transform (`Cpih = 1`).
  * `round362_wide_band_nl1_streaming_round_trips` — wide single-level (NL=1)
    picture whose LL band spans `Ns = 3` group widths, routed through the
    per-precinct streaming synthesis path.

**`Ng = 4` / `Ss = 8` conformance gate (Table A.7).** Table A.7 lists a
single conformant value for both `Ng` (4) and `Ss` (8) — with no range like
the `1—8` given for `Nc`. Every code-group / significance-group geometry
computation presumes these values, so the PIH parser now rejects any other
`Ng` or `Ss` rather than silently mis-grouping coefficients. (Previously
`Ss` was checked against an over-permissive `1..=8` and `Ng` was unchecked.)

## Unreleased — round 359 (deepen decode-geometry coverage: 4:2:0 + NLT + Bw=20 at deeper vertical cascades)

Extends the regression-locked decode-geometry envelope past the round-357
horizontal-only (`NL,y = 0`) baseline into deeper symmetric and asymmetric
vertical cascades, composed with the chroma-subsampled, high-bit-depth, NLT,
and high-precision paths. JPEG XS allows `NL,x ≥ NL,y`; for a `sy = 2`
component the decoder drops `log2(sy) = 1` vertical level, so a 4:2:0 stream
at `NL,y` decomposes chroma at `N'L,y = NL,y − 1`. The round-343 4:2:0 tests
pinned `NL,y = 2` (chroma `N'L,y = 1`); these push the vertical cascade to
`NL,y ∈ {3, 4, 5}` (chroma `N'L,y` up to 4), exercising the per-band
`beta_key_for` cascade-key derivation across the full vertical range. No
decoder logic changes were required — the cascade synthesis already handled
these geometries; this round locks them against regression.

* +9 tests (511 → 520):
  * `round359_420_lossless_deep_vertical_cascade` — bit-exact 4:2:0 lossless
    at `NL = NL,y ∈ {3, 4, 5}`.
  * `round359_420_lossless_asymmetric_cascade` — bit-exact 4:2:0 lossless at
    asymmetric `(NL,x, NL,y) ∈ {(3,2), (4,2), (4,3), (5,3)}`
    (`β1 = NL,x − NL,y + 1 > 1`).
  * `round359_420_highbd_deep_vertical_cascade` — bit-exact 4:2:0 lossless at
    `NL,y = 3` for `B[i] ∈ {10, 12, 14, 16}` (u16-LE planes).
  * `round359_420_lossy_deep_vertical_cascade` — 4:2:0 lossy (`q = 2`) at
    `NL,y ∈ {2, 3, 4}` above a 25 dB PSNR floor on all three planes.
  * `round359_nlt_quadratic_deep_vertical_cascade` /
    `round359_nlt_extended_deep_vertical_cascade` — NLT (`Bw = 18`) luma
    decode across `NL,y ∈ {1..4}` / `{1..3}`.
  * `round359_highprec_bw20_deep_vertical_cascade` — high-precision
    (`Bw = 20, Fq = 8`) luma round-trips bit-exactly across `NL,y ∈ {1..4}`.
  * `round359_star_tetrix_deep_vertical_cascade` — 4-component Star-Tetrix
    (`Cpih = 3`) CFA lossless across `NL ∈ {2, 3, 4}`.
  * `round359_nlt_quadratic_highbd_deep_vertical_cascade` — high-bit-depth
    (`B[i] = 12`) NLT quadratic across `NL,y ∈ {1, 2, 3}`; confirms the
    high-bit-depth NLT path runs the wavelet domain at `Bw = 20` (not the
    8-bit path's `Bw = 18`) for precision headroom, with MSE ≤ 4 LSB².

## Unreleased — round 351 (high-precision (Bw=20, Fq=8) regular lossy entry point)

Adds a public entry point for the ISO/IEC 21122-1:2022 Table A.8
`(Bw = 20, Fq = 8)` high-precision regular case, built on the Annex E.3
fractional-coefficient scaling landed earlier this round.

* `encoder::encode_planar_highprec_lossy(width, height, nc, cpih, nlx, nly, q,
  planes)` — carries 8 fractional bits through the wavelet transform: the
  `B[i]`-bit samples are up-scaled into the 20-bit wavelet domain
  (`Ω = (sample << (Bw − B[i])) − (1 << (Bw − 1))`, the inverse of the Annex G.2
  output scaling), the forward DWT runs in that wider domain, and its output is
  down-scaled by `>> Fq` before quantisation. The decoder applies `c << Fq` and
  the Annex G.2 `>> (Bw − B[i])` symmetrically. The extra precision removes the
  per-decomposition-level integer rounding the `(Bw = B[0], Fq = 0)` path
  injects, so for a fixed `q` the reconstructed PSNR is at least as high.
* +1 test: the high-precision path signals `(Bw = 20, Fq = 8)`, decodes a 64×64
  NL = 5/5 gradient within the 30 dB floor at `q = 2`, and reconstructs it at
  least as faithfully as the integer-transform path (in fact bit-exactly —
  PSNR = ∞ — where the integer path loses ≈ 49 dB to per-level rounding).

## Unreleased — round 351 (Annex E.3 Fq fractional-coefficient scaling + Table A.8 (Bw, Fq) conformance)

Implements the ISO/IEC 21122-1:2022 Annex E.3 (Table E.13) wavelet-coefficient
fractional scaling — `T[β,x,y] = c[p,λ,b,ξ] << Fq` on decode and the exact
inverse `c = sign(T)·((|T| + ((1<<Fq)>>1)) >> Fq)` on encode — and aligns the
signalled `(Bw, Fq)` pair to a valid Table A.8 combination on every path.

* **Fix (conformance):** the regular lossy path used to write the
  non-tabulated `(Bw = 8, Fq = 8)` pair while applying *no* fractional
  scaling. A conforming decoder reconstructing per Annex E.3 would have
  left-shifted the coefficients by 8 and corrupted the output; the streams
  only round-tripped here because this crate's decoder also ignored `Fq`.
  Table A.8 lists exactly three valid combinations: `(B[0], 0)` (integer /
  lossless transform), `(18, 6)` (NLT present), `(20, 8)` (high-precision
  regular). The integer 5/3 transform carries zero extra fractional bits, so
  a regular lossy stream now truthfully signals `(B[0], 0)` — lossy
  compression is still driven by the precinct truncation `T[p,b]`, which the
  inverse quantiser applies independently of `Fq`. The bogus
  `q > 0 ⇒ Fq = 8` encoder invariant is removed.
* **Decode:** [`dequant::dequantize_precinct`] now takes `fq` and applies the
  Annex E.3 `<< Fq` up-scaling to every band of every component (including the
  suppressed `Sd` components copied straight to the sample plane), centralised
  at the dequant boundary so both the cascade and streaming synthesis paths
  inherit it. No-op for `Fq = 0`, so the lossless path is byte-identical.
* **Encode:** [`fq_downscale_band`] applies the matching `>> Fq` rounding to
  the forward-DWT output (cascade, streaming 1-D / 2-D, and `Sd` bands) before
  quantisation. The linear input scaling now performs the full Annex G.2
  inverse `Ω = (sample << ζ) − dc_bias` with `ζ = Bw − B[i]`, so the
  high-precision `(Bw = 20, Fq = 8)` regular case has the dynamic-range
  headroom for the fractional bits to survive the transform (ζ = 0 reduces to
  the byte-identical legacy shift for the integer path).
* **Decode conformance gate:** `decode_codestream` rejects any `(Bw, Fq)` pair
  outside Table A.8, and an NLT marker present with `Fq = 0` (Annex A.4.6).
* The NLT paths now signal the Table A.8 `(18, 6)` / `(20, 8)` value for their
  `Bw` and carry the matching `>> Fq` / `<< Fq` scaling, making them
  `Fq`-truthful (previously they wrote `Fq = 8` with no scaling).
* +2 tests: a regular lossy stream signals the conformant `(Bw = B[0], Fq = 0)`
  and still decodes within the PSNR floor; the decoder rejects a hand-mutated
  `(Bw = 8, Fq = 8)` stream citing Table A.8.

## Unreleased — round 346 (high-bit-depth subsampled Annex H weights + subsampled WGT-truncation alignment fix)

Wires the ISO/IEC 21122-1:2022 Annex H subsampled tables (H.4–H.8) through
the high-bit-depth (`B[i] ∈ 9..=16`) chroma-subsampled path and corrects a
latent encoder/decoder truncation mismatch for subsampled layouts whose
chroma bands do not all exist.

* `encoder::encode_planar_subsampled_highbd_annex_h(width, height, nc, cpih,
  nlx, nly, bd, q, sx, sy, planes)` — the high-bit-depth companion to
  [`encode_planar_subsampled_annex_h`]. Drives the WGT + forward truncation
  `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the H.4 / H.5 / H.6 (4:2:2)
  and H.7 / H.8 (4:2:0, `-*` slots dropped) tables over the `u16`-LE plane
  layout. Falls back to the default-weights highbd subsampled path for
  untabulated configs.
* **Fix:** the subsampled Annex H forward truncation now indexes the supplied
  weights by each band's position in the **β-major existing-band
  enumeration** — the same cursor the decoder's WGT loader
  (`slice_walker::build_plan`) walks — instead of by the full picture band
  index. Previously the `custom_wgt` branch only engaged when the weights
  vector length equalled the *full* band count, so for 4:2:0 (where H.7 / H.8
  carry `-*` non-existent-band slots that shrink the vector) the encoder
  silently truncated with *geometry-derived* gains while the WGT marker
  advertised the *Annex H* gains. A conforming decoder reconstructing `T[p,b]`
  from WGT therefore disagreed with the encoder; at high bit depth this
  surfaced as an entropy buffer over-read. Encoder and decoder now compute the
  identical `T[p,b]` for every chroma layout. This also makes the 8-bit
  subsampled Annex H streams genuinely WGT-consistent (they previously decoded
  only because the low-magnitude 8-bit coefficients masked the structural
  inconsistency).
* `pack_subsampled_u16_planes` helper centralises the highbd subsampled `u16`
  plane validation / packing (shared with `encode_planar_subsampled_highbd`
  and `_lossy`, deduplicating their inline copies).
* +4 tests: WGT carries the existing-band column + decodes + differs from
  default at `bd ∈ {10, 12, 16}` across all five H.4–H.8 configs; lossless
  bit-exact self-roundtrip at `bd ∈ {10, 12, 14, 16}`; fallback (lossless and
  lossy) for untabulated configs; and a `q = 1` lossy PSNR-floor roundtrip on
  the 4:2:0 H.7 / H.8 tables proving the truncation-alignment fix reconstructs
  *correct pixels* (≥ 30 dB), not merely a non-crashing decode.

## Unreleased — round 346 (high-bit-depth CFA Star-Tetrix Annex H weights)

Closes the README "content-adaptive WGT weights for the CFA Star-Tetrix
layouts at bit depths above 8" follow-up by wiring the ISO/IEC
21122-1:2022 Annex H PSNR-optimized CFA tables (H.9 / H.10 / H.11) through
the high-bit-depth (`B[i] ∈ 9..=16`, little-endian `u16` planes) `Sd = 1`
Star-Tetrix path.

* `encoder::encode_planar_sd_star_tetrix_highbd(width, height, nlx, nly,
  bd, q, e1, e2, cf, ct, planes)` — the high-bit-depth companion to
  [`encode_planar_sd_star_tetrix`] at `Nc = 4`, `Sd = 1`. Suppresses the
  fourth Star-Tetrix output (raw-coded per Annex B Tables B.10 / B.11),
  `NL = (Nc−Sd)·Nβ + Sd = 19 / 25 / 31` bands for `NL,x = 5`,
  `NL,y ∈ {0, 1, 2}`. `Bw = B[i] = bd`, Annex G.3 DC level shift
  `1 << (bd − 1)`, two-bytes-per-sample plane packing. This is the first
  high-bit-depth `Sd = 1` Star-Tetrix entry point.
* `encoder::encode_planar_star_tetrix_highbd_annex_h(width, height, nlx,
  nly, bd, q, e1, e2, cf, ct, planes)` — the high-bit-depth companion to
  [`encode_planar_star_tetrix_annex_h`]. Drives the WGT marker and the
  forward truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the
  H.9–H.11 `(G[b], P[b])` column selected by `cf ∈ {0, 3}`. Configurations
  outside the tabulated set (`NL,x ≠ 5`, `NL,y > 2`) fall back to the
  default-weights highbd path. Bit depth and the Annex H weights are
  orthogonal (the gains / priorities act on `i32` coefficients), so the
  reconstruction is bit-exact at `q = 0` even with the H column advertised.
* A shared `pack_cfa_u16_planes` helper centralises the highbd CFA
  validation (bit-depth `9..=16`, exactly 4 planes, plane-size and
  per-sample range checks) for both new entry points.
* +4 tests: WGT carries exactly the selected H column + decodes + differs
  from default at `bd ∈ {10, 12, 16}` and `NL,y ∈ {0, 1, 2}` × `Cf ∈ {0, 3}`;
  lossless bit-exact self-roundtrip at `bd ∈ {10, 12, 14, 16}`; fallback to
  the default-weights highbd path for untabulated configs; and input
  validation (bd outside `9..=16`, wrong plane count, mismatched sizes,
  out-of-range samples).

## Unreleased — round 343 (uniform inverse quantizer `Qpih = 1` on chroma-subsampled inputs)

Widens the uniform-quantizer (`Qpih = 1`, Annex D.3) decode path to the
chroma-subsampled layouts, so the inverse-uniform Neumann-series
reconstruction is now exercised end-to-end across all three documented
chroma samplings (4:4:4 / 4:2:2 / 4:2:0) rather than only the 4:4:4 path
[`encode_planar_qpih`] reached.

* `encoder::encode_planar_qpih_subsampled(width, height, nc, cpih, nlx,
  nly, q, sx, sy, planes)` — chroma-subsampled companion to
  [`encode_planar_qpih`]. Signals the PIH `Qpih = 1` bit so the decoder
  takes the Annex D.3 inverse-uniform branch; takes explicit
  per-component `sx` / `sy` sampling factors. The data sub-packet is
  byte-identical to the `Qpih = 0` form, so the lossless (`q = 0`,
  `T = 0`) stream round-trips bit-exactly while `q > 0` engages the
  equal-bucket Neumann reconstruction per band.
* +4 round-trip tests: 4:2:0 + 4:2:2 lossless bit-exact through the
  inverse-uniform decode kernel (every sample recovered, PIH `Qpih = 1`
  confirmed), 4:2:0 lossy `q = 2` above a 20 dB floor, and 4:4:4 RGB-RCT
  lossy `q = 2` confirming the uniform kernel composes with the
  reversible colour-transform inverse.

Adds the Annex C.3 raw-mode-consistency conformance gate to the decode
path and corrects the encoder's raw-mode-selection flag.

* `entropy::check_raw_mode_consistency(geom, packets)` — when the
  picture-header `Rl = 0`, rejects a precinct in which a band's raw-mode
  flag `Dr[p,s]` differs across the packets that include the band (raw
  and non-raw bitplane-count coding mixed within one band, forbidden by
  Annex C.3). No-op when `Rl = 1`, which lifts the restriction. Wired
  into the live decode loop before the inverse quantizer. +4 unit tests.
* Encoder now signals `Rl = 1` in the PIH `Lh:Rl:Qpih:Fs:Rm` byte. The
  encoder selects the bitplane-count coding mode independently per packet
  (the smaller of the raw and VLC form), so a multi-packet band can carry
  a different `Dr` per packet — exactly the `Rl = 1` regime (Annex
  C.5.3.3), whose per-line buffer bound holds automatically because each
  packet's chosen size never exceeds its raw size. Previously the encoder
  emitted `Rl = 0` while making per-packet raw choices, a non-conformant
  construction the new decode gate surfaces.

Drives the uniform inverse-quantizer (`Qpih = 1`, Annex D.3) decode path
through the structural-feature axes it had never been exercised against.

* +4 lossless bit-exact round-trip tests composing `Qpih = 1` with
  multi-slice (`Hsl > 0`, 4:2:0), multi-precinct-per-row (`Cw > 0`,
  4:2:2), the separate sign sub-packet (`Fs = 1`, 4:2:0), and the CFA
  Star-Tetrix colour transform (`Cpih = 3`, 4-component). Each confirms
  the PIH `Qpih = 1` bit and recovers every sample exactly through the
  inverse-uniform kernel composed with the respective decode path.

## Unreleased — round 334 (bitplane-count-subpacket size `Lcnt[p,s]` inference + consistency predicate)

Adds the bitplane-count subpacket to the family of subpacket-size
consistency predicates (alongside `Lprc[p]`, `Ldat[p,s]`, `Lsgn[p,s]`,
`Lsig[p,s]`), completing the set of signalled subpacket lengths a decoder
can reconstruct and cross-check against the packet header (ISO/IEC
21122-1:2022 Annex C.6, Tables C.12 / C.13 / C.14).

* `entropy::bits::vlc_codeword_bits(value, theta)` — the exact inverse of
  the Table C.15 unary VLC: the wire bit length (`x + 1`) of the codeword
  decoding to signed `value` under predictor parameter
  `theta = max(r − t, 0)`. Round-trip tested against `vlc`'s consumption
  across a `(r, t)` × value sweep.
* `entropy::PacketCountInfo` + `entropy::infer_lcnt(geom, pkt)` — infers
  `Lcnt[p,s]` from the packet's decode mode. Raw mode (`Dr = 1`,
  Table C.12) is `Σ Ncg[p,b] × Br` bits over existing bands, independent
  of the values; the two VLC modes (`Dr = 0`, Tables C.13 / C.14) sum
  `vlc_codeword_bits(Δm, θ)` over the coded code groups, with `None`
  entries for significance-coded groups (`Z = 1`) that emit no codeword.
  Byte-padded the same way `decode_packet_body`'s `align_to_byte()`
  consumes the subpacket.
* `entropy::count_subpacket_filler_bytes(geom, pkt, lcnt)` — cross-checks
  a wire `Lcnt[p,s]` against the inferred minimum and returns the implied
  trailing-filler-byte count, erroring when `lcnt` is smaller than the
  decode mode requires (mirrors `data_subpacket_filler_bytes` /
  `sign_subpacket_filler_bytes`).
* +8 unit tests (2 for the VLC-length inverse, 6 for the `Lcnt`
  inference + filler/overflow predicate).

## Unreleased — round 330 (content-adaptive WGT weights — Annex H CFA Star-Tetrix tables H.9–H.11)

Completes the Annex H example-weight-table coverage with the CFA
Star-Tetrix tables H.9 / H.10 / H.11 (`Cpih = 3`, `Sd = 1`, four 4:4:4:4
components, `NL,x = 5`, `NL,y ∈ {0, 1, 2}`).

* `encoder::annex_h_weights` gained a `cf` parameter and now returns the
  ISO/IEC 21122-1:2022 Annex H CFA Star-Tetrix tables H.9 / H.10 / H.11
  (19 / 25 / 31 bands). Each table tabulates a separate `(G[b], P[b])`
  column per CTS extent `Cf ∈ {0, 3}` (full / restricted in-line, Table
  A.20); `cf` selects the column. Band order follows the `Sd = 1` layout
  of Annex B Tables B.10 / B.11 (`b = (Nc − Sd)·β + i` for the three
  wavelet-coded outputs, then the suppressed raw-coded output tail).
* `encoder::encode_planar_star_tetrix_annex_h(width, height, nlx, nly, q,
  e1, e2, cf, ct, planes)` — public entry point (re-exported at the crate
  root) that drives both the WGT marker and the forward truncation
  `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` from the CFA Annex H weights, so
  the codestream round-trips through `decode_jpeg_xs`. Falls back to the
  default-weights `Sd = 1` Star-Tetrix path for any non-tabulated
  `(cf, nly)`.
* Corrected an over-strict `Cpih = 3` constraint in both the encoder
  (`EncodeConfig::validate`) and decoder: the previous `Nc − Sd ≥ 4` guard
  conflated the Star-Tetrix *input* window (`c < 4`, Annex F.2 Table F.1)
  with `Sd` output suppression. Per Annex A.4.7 + Tables B.10 / B.11, `Sd`
  suppresses the wavelet decomposition of trailing transform *outputs*
  (coded raw, then consumed unchanged by the inverse transform), so the
  requirement is just `Nc ≥ 4`. RCT (`Cpih = 1`) keeps the stricter
  `Nc − Sd ≥ 3` guard. The CFA layout `Nc = 4, Sd = 1` (Nc−Sd = 3) now
  encodes and decodes; `Nc = 5, Sd = 2` (suppress a Star-Tetrix output)
  round-trips losslessly.
* Tests: WGT-readback (both `Cf` columns) + band-count checks for H.9 /
  H.10 / H.11, q=0 bit-exact self-roundtrip across all `(nly, cf)`,
  unmatched-config fallback equivalence, and the previously-rejecting
  `Nc = 5, Sd = 2` case rewritten to a lossless round-trip.

## Unreleased — round 327 (content-adaptive WGT weights — Annex H subsampled tables H.4–H.8)

Extends the round-323 content-adaptive WGT path to the chroma-subsampled
Annex H configurations (RCT disabled, `Cpih = 0`, `Nc = 3`, `NL,x = 5`).

* `encoder::annex_h_weights` gained `sx` / `sy` parameters and now also
  returns the ISO/IEC 21122-1:2022 Annex H subsampled tables: H.4 / H.5 /
  H.6 (4:2:2, `NL,y ∈ {0, 1, 2}`) and H.7 / H.8 (4:2:0, `NL,y ∈ {1, 2}`).
  The 4:2:0 tables list some band indices as non-existent (`bx[β,i] = 0`,
  the spec's `-*` slots); those positions are dropped so the returned
  `(gains, priorities)` vectors are in the encoder's existing-band
  emission order (the WGT loop `for(b=0;b<NL;b++) if(b'x[b])` skips them,
  Annex A.4.11). Because the spec defines `-*` as precisely the
  non-existent bands, this filtering matches the encoder's
  `slice_walker::picture_beta_to_local_beta` skip rule position-for-
  position regardless of which β indices vanish.
* `encoder::encode_planar_subsampled_annex_h(width, height, nc, cpih,
  nlx, nly, q, sx, sy, planes)` — public entry point that threads the
  subsampled Annex H weights through both the WGT marker and the forward
  truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)`, so the codestream
  round-trips through `decode_jpeg_xs`. Falls back to
  `encode_planar_subsampled` for any non-tabulated configuration.
* Tests: WGT-readback + existing-band-count (post-`-*`-drop) checks for
  all five tables, q=0 bit-exact self-roundtrip (the strict
  encoder/decoder `T[p,b]` alignment proof), lossy-path roundtrip +
  size-reduction, and the untabulated-config fallback.

## Unreleased — round 323 (content-adaptive WGT weights — Annex H PSNR-optimized tables H.1–H.3)

New encoder feature: content-adaptive WGT weight assignment, replacing
the long-standing default of plain band-index priorities `P[b] = b` and
geometry-capped gains (LL=0 / HL=LH=1 / HH=2) for the canonical
4:4:4 RCT configurations.

* `encoder::annex_h_weights(nc, cpih, sd, nlx, nly) -> Option<(Vec<u8>,
  Vec<u8>)>` — returns the ISO/IEC 21122-1:2022 Annex H (informative)
  PSNR-optimized `(G[b], P[b])` example tables for 4:4:4, RCT-enabled
  (`Cpih = 1`), `NL,x = 5` with `NL,y ∈ {0, 1, 2}` (Tables H.1 / H.2 /
  H.3), in the WGT emission / band-index order `b = Nc×β + i`. Other
  configurations return `None`.
* `encoder::encode_planar_lossy_annex_h(width, height, nc, cpih, nlx,
  nly, q, planes)` — public entry point that threads the Annex H table
  through both the WGT marker (Annex A.4.11) **and** the forward
  truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` (Annex C.6.2
  Table C.10, `r = (P[b] < R[p]) ? 1 : 0`). Falls back to
  `encode_planar_lossy` for any non-tabulated configuration.
* The multi-level forward-truncation closure in `write_slice` gained a
  `custom_wgt` branch: when a complete weights table (both `band_gains`
  and `band_priorities` non-empty, length == `NL`) is supplied, it reads
  the exact `(G[b], P[b])` emitted to WGT per band index, so the encoder
  and decoder compute the identical `T[p,b]` and the codestream
  round-trips at `Q[p] > 0`. Absent / partial tables keep every prior
  round's behaviour (default gains + `P[b] = b`).
* The inner `encode_planar_inner_bd` / `encode_planar_inner_nlt` carry a
  new `band_priorities` parameter so a caller-supplied priority table is
  no longer overwritten by the built-in `build_band_priorities_sd`.

Tests: `round323_annex_h_weights_roundtrip_and_differ_from_default`
(WGT verbatim + decode round-trip ≥ 25 dB + differs from default across
`NL,y ∈ {0,1,2}`), `round323_annex_h_weights_lossless_bit_exact`,
`round323_annex_h_falls_back_to_default_when_unmatched`.

## Unreleased — round 319 (sign-subpacket size Lsgn[p,s] inference + consistency predicate, Annex C.5.5 Table C.9)

New decode-side conformance feature: the sign subpacket of a packet
(ISO/IEC 21122-1:2022 Annex C.5.5, Table C.9) is present only when the
picture-header sign-packing flag `Fs == 1`, and carries no length on the
wire other than the `Lsgn[p,s]` field of the packet header — yet its
exact bit count is fully determined by the decoded coefficient
magnitudes `v[p,λ,b,x]`, the code-group size `Ng`, and the per-band
coefficient count `Wpb[p,b]`. Round 319 infers it the same way round 309
inferred `Ldat[p,s]` (data subpacket) and round 295/308 inferred
`Lsig[p,s]` (significance subpacket), completing the set: every
subpacket byte count is now reconstructible from the precinct's coding
state, so `Lsgn[p,s]` is no longer only a *given* packet-header input to
the precinct-length predicate.

* `entropy::infer_lsgn(geom, pkt) -> u64` — the inferred sign-subpacket
  byte count. Per Table C.9 the sign subpacket emits exactly **one sign
  bit per non-zero quantization-index magnitude** `v[p,λ,b,Ng×g+k] != 0`,
  iterating every member `k ∈ 0..Ng` of every code group
  `g ∈ 0..Ncg[p,b]` of every included `(band, line)` entry. Positions
  past the band width `Wpb[p,b]` carry no sign bit — the "meaningless
  coefficients near the right edge" of Table C.9 NOTE 2 that the
  magnitude loop already skips (`xpos >= wpb`). The accumulated bits are
  padded up to the next byte boundary (`pad(8)`), giving `Lsgn[p,s]`
  exclusive of any optional trailing filler bytes. Returns `0` when
  `Fs == 0` (the subpacket does not exist and signs ride the data
  subpacket, counted by `infer_ldat`). Bits accumulate in `u64` so an
  adversarial coefficient set cannot overflow before the byte rounding.
* `PacketSignInfo { entries, v }` carries the per-packet inputs:
  the inclusion list `I[p,b,λ,s]` and the decoded magnitudes
  `v[p,λ,b,x]` laid out one slice per entry (the same
  `coef.v[line_offset + xpos]` layout the sign decoder reads). A slice
  shorter than the band width is treated as zero-padded — positions past
  its end carry no sign bit, matching the decoder reading an absent
  coefficient as 0. Bands that do not exist are skipped, matching the
  decode loop's `if !band.exists { continue; }`.
* `entropy::sign_subpacket_filler_bytes(geom, pkt, lsgn) -> Result<u32>`
  — cross-checks a wire `Lsgn[p,s]` against the inferred minimum and
  returns the implied trailing-filler-byte count, mirroring
  `data_subpacket_filler_bytes`. A wire `Lsgn[p,s]` smaller than the
  sign bits the magnitudes require is rejected as malformed. When
  `Fs == 0` the inferred size is `0` and any field value is accepted as
  pure filler (Table C.4 omits the subpacket entirely).
* Seven new tests covering: one-bit-per-non-zero counting with byte
  padding, `Fs == 0 → 0`, past-`Wpb` skip (Table C.9 NOTE 2), short-slice
  zero-padding, multi-entry summation with absent-band skip, and the
  filler/overflow cross-check in both `Fs` modes.

No public-API removal; the decoder's sign sub-packet loop is unchanged.

## Unreleased — round 315 (bitplane-count range conformance across all decode modes, Annex C.6.4/C.6.5/C.6.6 Tables C.12/C.13/C.14)

Decode-side conformance fix: every bitplane-count decode mode must
produce `0 ≤ M[p,λ,b,g] ≤ (2^Br − 1)`. Tables C.13 (vertical) and C.14
(no-prediction) state this bound verbatim ("the codestream shall always
be constructed in such a way that the bitplane counts M[p,λ,b,g] are
between 0 and (2^Br − 1)"), and Table C.12 (raw) gives the same range in
its `Values` column. Previously only the raw path enforced `2^Br − 1`;
the two variable-length-code paths (no-prediction, vertical) merely
clamped the decoded `M` to the byte range `0..=255`, silently accepting
out-of-range counts a conformant decoder must reject.

* `entropy::packet_body::decode_packet_body` now derives `m_max =
  2^Br − 1` once per band (`Br` is the `u(4)` raw-coding field of the
  picture header, Table A.7, with sole conformant value 4 → `m_max =
  15`) and validates the decoded `M` against `0..=m_max` in all three
  modes. The lower bound 0 also rejects an underflowing `mtop + Δm` in
  the VLC paths, where `vlc()` can return a negative residual.
* New test `nopred_vlc_bitplane_count_above_2pow_br_minus_1_rejected`:
  a unary VLC codeword decoding to `M = 16` (one above the `Br = 4`
  maximum) is now rejected as malformed.

No public-API change; the encoder already keeps `M` within range.

## Released as v0.0.7 — round 309 (data-subpacket size Ldat[p,s] inference + consistency predicate, Annex C.5.4 Table C.8)

New decode-side conformance feature: the data subpacket of a packet
(ISO/IEC 21122-1:2022 Annex C.5.4, Table C.8) carries no length on the
wire other than the `Ldat[p,s]` field of the packet header, yet its exact
bit count is fully determined by the decoded bitplane counts
`M[p,λ,b,g]`, the per-band truncation positions `T[p,b]`, the sign-
packing flag `Fs`, and the code-group size `Ng`. Round 309 infers it the
same way the round-295 / round-308 predicates infer `Lsig[p,s]` —
extending the round-308 `precinct_filler_bytes` story where `Ldat[p,s]`
was previously taken as a *given* packet-header input rather than
*inferred* from the precinct's coding state.

* `entropy::infer_ldat(geom, precinct, pkt) -> u64` — the inferred
  data-subpacket byte count. Per Table C.8, for every included
  `(band, line)` entry and every code group `g` with
  `M[p,λ,b,g] > T[p,b]`, the data subpacket carries `Ng` sign bits (only
  when `Fs == 0` — when `Fs == 1` the signs ride the separate sign
  subpacket of Table C.9) plus `Ng × (M[p,λ,b,g] − T[p,b])` magnitude
  bits (the `M − T` retained bitplanes, `Ng` coefficients each). Code
  groups with `M ≤ T` contribute nothing. The accumulated bits are
  padded up to the next byte boundary (`pad(8)`), giving `Ldat[p,s]`
  exclusive of any optional trailing filler bytes. Bits accumulate in
  `u64` so an adversarial count set cannot overflow before the byte
  rounding. The per-band `T[p,b]` is resolved from `Q[p]` / `R[p]` /
  `D[p,b]` + WGT gains/priorities via the existing `precinct_truncation`.
* `PacketDataInfo { entries, m }` carries the per-packet inputs:
  the inclusion list `I[p,b,λ,s]` and the bitplane counts `M[p,λ,b,g]`
  laid out one slice per entry (each slice length `Ncg[p,b]`, the same
  `coef.m[line_index * ncg + g]` layout the decoder reads). Bands that
  do not exist are skipped, matching the decode loop's
  `if !band.exists { continue; }`.
* `entropy::data_subpacket_filler_bytes(geom, precinct, pkt, ldat) ->
  Result<u32>` — cross-checks a wire `Ldat[p,s]` against the inferred
  minimum and returns the implied trailing-filler-byte count (Annex
  C.5.4 NOTE: the filler count is inferred from the `Ldat[p,s]` field).
  Mirrors round-308 `precinct_filler_bytes` at the data-subpacket level:
  `Ok(filler_bytes)` when `ldat` covers the inferred data, `Err(_)` when
  `ldat` is smaller than the bitplane counts require (a malformed /
  inconsistent packet header).

No reconstruction-path change; all existing roundtrips decode
byte-identically. +5 entropy unit tests (Fs=0 sign+magnitude accounting,
Fs=1 sign omission, truncation gating, the filler/overflow cross-check,
multi-entry summation with an absent band skipped); 451 tests total.

## Unreleased — round 308 (precinct-length Lprc[p] consistency predicate, Annex C.2 Table C.1)

New decode-side conformance feature: the
`entropy::precinct_filler_bytes(geom, precinct, packets)` predicate
verifies that a precinct's `Lprc[p]` field (ISO/IEC 21122-1:2022
Annex C.2, Table C.1) is consistent with the actual on-wire size of its
packets, and returns the number of trailing filler bytes the field
implies.

Table C.1 defines `Lprc[p]` as the length of the precinct's entropy-coded
data **including filler bytes**, counted "from the end of the precinct
header of this precinct up to, but not including the first byte of the
next precinct header, slice header or EOC". The precinct header bytes
themselves are not counted; every packet header and subpacket is. The
summed packet size is, per the packet syntax (Annex C.3, Table C.3) and
the packet body (Table C.4), `Σ_s (header_bytes + Lsig[p,s] + Lcnt[p,s]
+ Ldat[p,s] + (Fs==1 ? Lsgn[p,s] : 0))`, where the un-signalled
`Lsig[p,s]` is inferred via the same `significance_subpacket_bytes`
routine the bitplane-buffer-bound predicate and the significance
subpacket decoder both use (raw-mode packets with `Dr = 1` contribute
zero `Lsig`).

* `Lprc[p]` must be **at least** that sum; the difference is the count of
  optional filler bytes the precinct ends with (Annex C.2: "the amount of
  filler bytes following the precinct can be inferred from the `Lprc[p]`
  field"). Returns `Ok(filler_bytes)` when the packets fit, or `Err(_)`
  when the summed packet size exceeds `Lprc[p]` (a malformed codestream
  whose declared length is too small to contain its own packets).
* `PacketWireSize { header_bytes, lcnt, ldat, lsgn, dr, entries }`
  carries each packet's wire-size inputs (5/7-byte short/long header,
  the three signalled subpacket byte counts, the raw-mode flag, and the
  inclusion list for `Lsig` inference). The sign subpacket counts only
  when `Fs == 1` (Table C.4 omits it when sign coding is disabled).
* The packet sizes are accumulated in `u64` so an adversarial set of
  per-packet counts cannot overflow before the comparison against the
  24-bit `Lprc[p]`.

No reconstruction-path change; all existing roundtrips decode
byte-identically. +4 entropy unit tests (exact-fit + padding + overflow,
`Fs == 1` sign accounting, inferred-`Lsig` inclusion with raw-mode
suppression, multi-packet summation); 446 tests total.

## Unreleased — round 302 (vlc loss-of-synchronisation guard, Annex C.7.1 Table C.15)

Decode-side conformance fix to the elementary variable-length decoder
`entropy::vlc(reader, r, t)` (ISO/IEC 21122-1:2022 Annex C.7.1,
Table C.15). The unary-count loop is `do { b; if (b) x++; } while (b &&
x < 32)` followed by `if (x >= 32) error()`: the loop stops counting the
instant `x` reaches 32 (the `x < 32` guard fails) **without consuming a
further bit**, and `x >= 32` is then the hard loss-of-synchronisation
error. The previous implementation:

* errored only on a **33rd** consecutive 1-bit (off-by-one against the
  spec, which errors at the 32nd); and
* read **one extra bit** past the 32nd 1-bit to "confirm" the comma,
  desynchronising the bit reader from the source on a malformed stream
  (the spec never reads that bit).

The decoder now matches Table C.15 exactly: the largest legal codeword
is 31 consecutive 1-bits closed by a 0 comma bit (`x == 31`); a 32nd
consecutive 1-bit is the error condition and the reader does not advance
past it. `vlc` is the primitive every bitplane-count residual
(Tables C.13 / C.14) flows through, so the fix makes malformed-stream
detection bit-exact across the whole entropy decode path.

* +2 boundary unit tests: `vlc_accepts_31_ones_then_comma` (the maximal
  legal codeword decodes to `x − θ = 31` and consumes exactly 4 bytes)
  and `vlc_errors_at_exactly_32_ones_without_overreading` (32 ones with
  no further input errors as loss-of-sync, consuming exactly 4 bytes —
  an over-reading decoder would instead hit a truncated-stream error
  and mask the real condition). Existing valid-alphabet and gross-case
  (40-ones) tests are unchanged and still pass.

## Unreleased — round 295 (bitplane-count buffer-bound conformance predicate, Annex C.5.3.4 Table C.6)

New decode-side conformance feature: the
`entropy::bitplane_buffer_bound_satisfied(geom, precinct, packets)`
predicate evaluates ISO/IEC 21122-1:2022 Annex C.5.3.4 (Table C.6,
`is_encoding_valid`). The standard bounds the buffer an encoder may
spend on entropy-coded bitplane-count data: the coded size of the
bitplane-count + significance subpackets must not exceed the size the
same data would occupy in raw mode (`Br` bits per code group). The
comparison is band-based when the picture-header `Rl` flag is 0
(§C.5.3.2) and packet-based when `Rl` is 1 (§C.5.3.3).

* `significance_subpacket_bytes` infers `Lsig[p,s]` (Annex C.5.3.2):
  one bit per significance group `Ns[p,b]` of every present band whose
  mode enables significance coding (`D[p,b] & 2`) and that is not raw-
  overridden (`Dr[p,s] == 0`), rounded up to whole bytes — matching
  exactly the bits the significance subpacket decoder consumes, so the
  size is exact without the encoder signalling it on the wire.
* `PacketBufferInfo<'a> { lcnt, dr, entries }` carries one packet's
  contribution (`Lcnt[p,s]`, `Dr[p,s]`, the `I[p,b,λ,s]` inclusion
  list) as Table C.6 reads it.
* The predicate returns Table C.6's `valid` indicator as a `bool`
  rather than a hard decode-rejection: §C.5.3 frames the bound as a
  **codestream-construction** constraint ("the codestream shall be
  constructed in such a way that…"), and a decoder that reserves the
  raw-mode buffer can still reconstruct a precinct that does not
  satisfy it (e.g. a degenerate all-zero picture whose tiny single-
  line bands each round up to a whole byte while the summed contiguous
  raw bound is smaller). Strict-conformance callers reject `false`.

+4 entropy unit tests (Lsig inference, Rl=0 pass/fail, the degenerate
tiny-band two-line overshoot + its grouped-packet remedy, Rl=1 per-
packet form), 440 total. No change to the reconstruction path; all
existing roundtrips decode byte-identically.

## Unreleased — round 286 (cross-precinct vertical-prediction decode, Annex C.6.3 Table C.11)

Decoder gap close: the bitplane-count vertical-prediction VLC path
(`D[p,b] & 1 == 1`, Annex C.6.5 Table C.13) now reconstructs the
predictor at the **first line of a band in a precinct** from the
precinct directly above, instead of rejecting it as "not implemented".

Per ISO/IEC 21122-1:2022 Annex C.6.3 Table C.11, when `λ − sy <
L0[p,b]` the predictor comes from precinct `p − Np,x`:

* `Mtop[p,λ,b,g] = M[p−Np,x, L1[p,b]−sy, b, g]` — the bitplane counts
  of the **last decoded line** of band `b` in the precinct above.
* `Ttop[p,b] = T[p−Np,x, b]` — that precinct's per-band truncation
  position, which feeds the effective truncation `teff =
  max(T[p,b], Ttop)` and predictor `mtop = max(Mtop, teff)` (Table
  C.13).

Implementation:

* `entropy::packet_body::PrecinctTop` — new predecessor struct
  carrying per-band last-line bitplane counts (`last_m`) and per-band
  truncation positions (`t`). `PrecinctTop::capture` builds it from a
  just-decoded precinct. The walker emits `entry.line` in band-grid
  units stepping by one (the spec image-grid `λ` divided by `sy[i]`),
  so the spec's `L1[p,b]−sy` predecessor is the band's last stored
  line `L1−1−L0`.
* `decode_packet_body` gains an `Option<&PrecinctTop>` parameter. At a
  first-line band it uses the predecessor's `Mtop`/`Ttop`; inside a
  precinct it keeps the existing line-above predictor. A first-line
  vertical-prediction packet with **no** predecessor is now a precise
  malformed-codestream error — §C.6.1 / §C.6.3 forbid vertical
  prediction at the topmost precinct of a slice, so this can only mean
  a non-conforming stream rather than the previous blanket
  "not implemented" refusal.
* `decoder::decode_slice` maintains a per-precinct-column predecessor
  cache (`Vec<Option<PrecinctTop>>` of length `Np,x`), populated after
  each precinct and consumed by the precinct directly below in the
  same column. The cache is local to `decode_slice`, so vertical
  prediction never reaches across a slice boundary (matching §C.6.1's
  per-slice independence requirement).

Conformance: three hand-built `decode_packet_body` fixtures pin the
Table C.11 predictor — a basic cross-precinct prediction
(`Mtop=[2,0]`, `Ttop=0`), the `teff = max(T, Ttop)` interaction with a
non-zero `Ttop=3`, and the malformed-stream rejection when `top =
None`. No change to encoder output (our encoder is precinct-scoped and
never selects cross-precinct vertical prediction); all existing
multi-precinct-row lossless roundtrips continue to pass through the
now-populated cache. +3 tests, 436 total.

## Unreleased — round 282 (§B.1 odd-dimension sub-sampled planes + significance-coding conformance)

Decoder-side gap close in two parts, both grounded in ISO/IEC
21122-1:2022 and pinned by encoder→decoder roundtrips.

### §B.1 ceiling geometry — odd picture dimensions with sub-sampling

§B.1 defines the per-component plane as

```
Wc[i] = ⌈Wf / sx[i]⌉      Hc[i] = ⌈Hf / sy[i]⌉
```

(samples populate sampling-grid positions `0, sx, 2·sx, …` below
`Wf`). The implementation floored both divisions — the decoder
mis-sized chroma planes on any conforming stream with odd `Wf` / `Hf`
and 4:2:2 / 4:2:0 sampling, and the encoder rejected such pictures
outright ("width not divisible by component i sx").

* `slice_walker::build_plan_sd` — picture-level `Wc` / `Hc` and the
  per-precinct `wc_p = ⌈Wp / sx[i]⌉` now use the ceiling.
* `decoder::decode_codestream` — sample-buffer dims, gather-path band
  dims, streaming-path `wc_i` / `wp_i`, and the `hf_rows` clip all
  moved to the ceiling.
* `encoder` — the `EncodeConfig::validate` divisibility rejection is
  removed; plane-length validation (8-bit, high-bit-depth, and both
  sub-sampled variants), the cascade and streaming component dims,
  and the band-slice extraction all use the ceiling. The streaming
  path's per-precinct row range upper bound is `⌈y1 / sy⌉`.
* `Wpb[p,b] = 0` packets: a rightmost precinct column too narrow to
  own any of a band's columns still **emits its packet** (and the
  entry contributes zero bits to every sub-packet) — Annex B.7
  Table B.4's inclusion test is line presence, not band width. The
  encoder previously dropped the entry (and with it any packet whose
  entries were all zero-width), desynchronising the decoder, which
  correctly still expects the packet header.

### Significance-coding conformance (Tables C.5 / C.13 / C.14)

* **Wire `Z`-bit polarity** — Table C.5: the significance sub-packet
  bit "identifies whether all code groups in the significance group
  are insignificant", i.e. `Z = 1` ⇒ insignificant, and Tables C.13 /
  C.14 decode the VLC residual when `Z == 0`. Both our encoder and
  decoder used the inverted polarity (self-consistent, so
  self-roundtrips passed, but the wire was non-conforming). Both
  sides are flipped; the absent-significance default is now `Z = 0`.
* **Vertical-prediction × significance emission** — Table C.13
  reconstructs an insignificant group as `M = mtop + Δm` with
  `Δm = 0` at `Rm = 0` (so `M = max(Mtop, T)`), and Table C.8
  includes data for **any** group with `M > T`. The encoder flagged a
  group insignificant whenever every `M ≤ T`, ignoring the predictor
  — when the line above carried a high-magnitude group, the decoder
  reconstructed `M = Mtop > T` and read sign + magnitude bits the
  encoder never wrote (`read_bit past end of buffer`). The encoder
  now keeps such groups significant (the VLC carries the true
  negative residual instead), which is both conforming and
  rate-cheaper than emitting padding bitplanes. This was the
  long-standing cause of content-dependent roundtrip failures on the
  high-bit-depth path (large coefficient magnitudes make the
  vpred+sig form win the picker), independent of this round's
  geometry work.

Tests: +10 (423 → 433).

* `r282_odd_width_422_lossless_round_trip` — 65×16 4:2:2 NL=2/2,
  chroma `Wc = ⌈65/2⌉ = 33`.
* `r282_odd_dims_420_lossless_round_trip` — 65×17 4:2:0, chroma
  33×9.
* `r282_odd_dims_420_nl3_lossless_round_trip` — 33×33 4:2:0 NL=3/3
  (crosses the round-190 picture-β proxy-depth permutation).
* `r282_odd_dims_streaming_nl1_lossless_round_trip` — 65×{16,17}
  4:2:2 / 4:2:0 at NL=1/1 (the streaming per-precinct synthesis
  path).
* `r282_odd_dims_420_lossy_q2_psnr` — 65×17 4:2:0 q=2 holds ≥ 30 dB
  on every (ceil-sized) plane.
* `r282_odd_width_422_highbd_lossless_round_trip` — 10-bit 65×16
  4:2:2 `u16`-LE roundtrip.
* `r282_odd_width_422_cw1_lossless_round_trip` — 65×16 4:2:2 with
  `Cw = 1` → `Np,x = 3`; the last precinct column is one image
  column wide (`Wpb = 0` for the high-pass bands — the empty-packet
  emission pin).
* `r282_floor_sized_plane_rejected` — a floor-sized chroma plane
  (the pre-282 convention) is rejected with a plane-length error.
* `r282_vpred_sig_zero_group_below_active_round_trip` — 17×8 10-bit
  NL=2/2 luma with full-range values: an all-zero code group directly
  below a high-magnitude group; failed with `read_bit past end of
  buffer` before the Table C.13 emission fix.
* `r282_sig_subpacket_z_polarity_per_table_c5` — wire-level pin:
  the significance sub-packet carries `Z = 0` for the significant
  group and `Z = 1` for the all-insignificant group (one decoder-side
  unit test updated to the same polarity).

## Unreleased — round 273 (typed `Codestream::com` accessor)

Decoder-side ergonomic surface: a sixth typed accessor on
[`Codestream`] continues the round-251 / round-254 / round-266
pattern (decode a raw marker body into a strongly-typed view,
surface body-level errors as `Result`). This one covers the COM
extension marker — the last header marker that was still exposed
only as a raw `Vec<Vec<u8>>` byte buffer.

* New `com` module (`src/com.rs`) defining
  `ComMarker { tcom: u16, dcom: Vec<u8> }` +
  `parse_com(body) -> Result<ComMarker>` for the Extension marker
  (§A.4.10, Tables A.22 / A.23). The COM body after the `Lcom`
  length field is a big-endian `Tcom` `u(16)` (the extension type)
  followed by the variable `Dcom` user-defined data (which may be
  empty). The body-level parser enforces only what the body itself
  observes:
  * `body.len() >= 2` — the body must carry the mandatory two-byte
    `Tcom` field (`Lcom >= 4`, of which two bytes are the length
    field, leaving at least the two `Tcom` bytes in the body).
  Reserved `Tcom` values (the "all other values" row of Table A.23)
  are surfaced verbatim, *not* rejected: an extension marker is
  advisory metadata and a conforming decoder skips unknown
  extension types, so a reserved-range `Tcom` is a
  tolerate-and-pass-through case rather than a malformed body.
* `ComMarker` convenience predicates `is_encoder_vendor()` /
  `is_copyright()` / `is_vendor_specific()` classify `Tcom` against
  the Table A.23 rows (`0x0000` encoder vendor, `0x0001` copyright
  statement, `0x8000`–`0xffff` vendor-specific). The
  `TCOM_ENCODER_VENDOR` / `TCOM_COPYRIGHT` /
  `TCOM_VENDOR_SPECIFIC_MIN` constants are re-exported from the
  crate root alongside `ComMarker` / `parse_com`.
* `Codestream::com() -> Result<Vec<ComMarker>>` — decodes every
  COM segment body into the strongly-typed view, preserving on-wire
  order. Returns `Ok(Vec::new())` when no COM segment was present;
  body-level errors (a body too short to hold `Tcom`) surface as
  `Err(_)`. Mirrors the round-251 `cts` / `crg` / `nlt`, round-254
  `wgt`, and round-266 `cwd` shape: iterate `self.com`, run the
  body-level parser per segment, collect.

+11 tests (8 body-level in `com::tests`, 3 codestream-level in
`codestream::tests`), 423 total.

## Round 266 (typed `Codestream::cwd` accessor)

Decoder-side ergonomic surface: a fifth typed accessor on
[`Codestream`] continues the round-251 / round-254 pattern
(decode a raw marker body into a strongly-typed view, surface
body-level errors as `Result`).

* New `cwd` module (`src/cwd.rs`) defining
  `CwdMarker { sd: u8 }` + `parse_cwd(body) -> Result<CwdMarker>`
  for the Component Wavelet Decomposition marker (§A.4.7,
  Table A.18). Body-level parser enforces the constraints
  observable from the CWD body alone:
  * `body.len() == 1` — the marker carries exactly one byte
    after the `Lcwd` length field.
  * `Sd != 0` — `Sd = 0` is the no-suppression default and
    the spec forbids emitting CWD in that case
    (`Sd ∈ 1..=Nc-1`).
  The geometry-dependent upper bound `Sd ≤ Nc-1` and the
  forbidden-unless-`Nc>3` rule require the picture-header
  `Nc` and stay at the codestream marker-chain parser (already
  enforced since the early rounds).
* `Codestream::cwd() -> Result<Option<CwdMarker>>` —
  decodes the optional CWD marker body into the strongly-typed
  view. Returns `Ok(None)` when no CWD segment was present
  (decoder treats this as `Sd = 0`); returns `Ok(Some(cwd))`
  with the parsed `Sd` field when present. Mirrors the
  round-251 [`Codestream::cts`] / `crg` / `nlt` shape and the
  round-254 `wgt` shape: borrow `self.cwd.as_deref()`, run the
  body-level parser, surface field-level errors as `Err(_)`.
* `CwdMarker` and `parse_cwd` are re-exported from the crate
  root.
* The decoder's CWD lookup now routes through the typed
  accessor (`cs.cwd()?.map(|c| c.sd).unwrap_or(0)`) instead of
  the raw `cs.cwd_sd` byte field — the existing accessor stays
  for backward compatibility, but the typed view becomes the
  preferred entry point (matches the round-251 NLT decoder
  routing pattern).

+7 tests (4 body-level in `cwd::tests`, 3 codestream-level in
`codestream::tests`), 412 total.

## Round 254 (typed `Codestream::wgt` accessor)

Decoder-side ergonomic surface: a fourth typed accessor on
[`Codestream`] continues the round-251 pattern (decode a raw
marker body into a strongly-typed view, surface body-level
errors as `Result`).

* `Codestream::wgt() -> Result<Vec<BandWeight>>` — decodes the
  mandatory WGT marker body (§A.4.11, Table A.24) into a flat
  vector of `(gain, priority)` pairs in on-wire order. Routes
  through the existing [`parse_wgt`] body parser with
  `n_pairs = body.len() / 2`. Body-level errors surface as
  `Err(_)`:
  * total body length not a multiple of two (each pair is
    two bytes),
  * any `G[b] > 15` (Annex A.4.11 hard cap).

Geometry independence: the typed accessor does not bind the
returned pair count to the picture / component geometry. The
mapping from pair index to flat band id `b` requires `Np,y`,
`Nβ`, `Nc`, the per-component `bx[β,i]` existence flags, and
the optional CWD `Sd` value; the slice walker resolves that
mapping (and catches a count mismatch against the WGT body) at
decode time. The accessor's contract is the on-wire pair list,
not the band-indexed weights array — the same separation of
concerns that `Codestream::cts()` / `crg()` / `nlt()` already
follow with respect to the higher-level decode pipeline.

[`BandWeight`] and [`parse_wgt`] are now re-exported from the
crate root (previously only reachable through
`slice_walker::`).

Scope: decoder-side public API only. No bitstream-wire change,
no encoder change, no behaviour change to existing
`decode_codestream` paths. Test count 405 (was 401 at round
251). Four new tests cover the empty / pair-decoded /
odd-length / oversized-gain matrix:
* `wgt_method_returns_empty_for_zero_band_codestream` — the
  hand-built tiny 4×3 single-component fixture carries
  `Lwgt = 2` (zero-pair body); the accessor returns an empty
  `Vec` rather than erroring (geometry mismatch is the slice
  walker's job).
* `wgt_method_decodes_pair_body` — two-pair body
  `[2, 0, 15, 3]` round-trips to
  `[BandWeight { gain: 2, priority: 0 }, BandWeight { gain: 15,
  priority: 3 }]` in wire order.
* `wgt_method_surfaces_odd_length_body` — three-byte body
  `[2, 0, 4]` surfaces a "multiple of 2" error from the typed
  accessor (the top-level marker-chain parser accepts any
  byte sequence as the WGT body).
* `wgt_method_surfaces_oversized_gain` — two-byte body
  `[16, 0]` surfaces a gain-cap error (gain = 16 > 15 ⇒ illegal
  per §A.4.11).

## Unreleased — round 251 (typed `Codestream::cts` / `Codestream::crg` / `Codestream::nlt` accessors)

Decoder-side ergonomic surface: three typed accessors on
[`Codestream`] mirror the existing
[`Codestream::capabilities`] pattern (decode an optional raw
marker body into a strongly-typed view, surface body-level errors
as `Result`).

* `Codestream::cts() -> Result<Option<CtsMarker>>` — runs the
  existing [`parse_cts`] (§A.4.8, Tables A.19 / A.20) against
  `self.cts.as_deref()`. Returns `Ok(None)` when no CTS segment
  was present (legal for `Cpih ∈ {0, 1, 2}`), `Ok(Some(cts))`
  with the parsed `Cf` / `e1` / `e2` fields when CTS was present,
  and surfaces body-level errors (reserved nibble non-zero, `Cf`
  outside `{0, 3}`, `e1` / `e2` exceeding 3) as `Err(_)`. The
  top-level marker-chain parser only enforces ordering and the
  "Cpih=3 ⇒ CTS present" rule, not the field-level constraints.
* `Codestream::crg() -> Result<Option<CrgMarker>>` — runs
  [`parse_crg`] (§A.4.9, Table A.21) against the optional CRG
  body, passing `self.pih.nc` as the component count. Returns
  `Ok(None)` when absent, `Ok(Some(crg))` with one
  [`CrgEntry`] per component otherwise. Body-level errors (wrong
  byte count for `4 * Nc`) surface as `Err(_)`.
* `Codestream::nlt() -> Result<Option<NltParams>>` — runs
  [`parse_nlt`] (§A.4.6, Table A.16) against the optional NLT
  body. Returns `Ok(None)` when absent (decoder uses the linear
  Annex G.3 output path), `Ok(Some(NltParams::Quadratic { dco }))`
  for `Tnlt = 1` or `Ok(Some(NltParams::Extended { t1, t2, e }))`
  for `Tnlt = 2`. Body-level errors (wrong body length for the
  declared `Tnlt`, unknown `Tnlt`, out-of-range exponent /
  thresholds) surface as `Err(_)`.

[`NltParams`] / [`parse_nlt`] are now re-exported from the crate
root (previously only reachable through `output::`).

The decoder's [`decode_codestream`] is refactored to route through
the new typed accessors instead of re-parsing the raw bodies:
* The NLT body path drops the explicit `parse_nlt` match on
  `cs.nlt.as_deref()` and uses `cs.nlt()?` directly.
* The Star-Tetrix branch (`Cpih == 3`) drops the `parse_cts` /
  `parse_crg` re-parses of `cs.cts` / `cs.crg` and uses `cs.cts()?
  .ok_or_else(...)` / `cs.crg()?.ok_or_else(...)`. The
  `Cpih=3 ⇒ CTS/CRG present` requirement and the "CRG must
  describe one of the four Table F.9 CFA arrangements" check
  still fire from the decoder.

The accessors are non-allocating relative to the existing wire
format: each one borrows `self.{cts,crg,nlt}.as_deref()` and the
parsed view is owned by the caller. They eliminate the
double-decode pattern (`Codestream` had already validated the
length envelope at marker-chain parse time; the typed accessor
finishes the job by running the field-level body parser on the
borrowed bytes).

Scope: decoder-side public API only. No bitstream-wire change,
no encoder change. Test count 401 (was 394 at round 245). New
tests cover the present / absent / body-error matrix for each
accessor (RGGB CRG via the [`cfa_pattern_type`] round-trip,
Tnlt = 1 `dco = -1234` quadratic NLT, Cf = 3 / e1 = 1 / e2 = 2
in-line CTS, and a reserved-nibble-non-zero CTS body wired into
a Cpih = 0 codestream so the top-level parser keeps it while the
typed accessor rejects it).

## Unreleased — round 245 (rate-budget driven per-precinct `(Q[p], R[p])` picker, encoder side)

Closes the round-242 "caller must pick the vectors manually"
follow-up tail. Round 242 shipped the per-precinct joint override
[`encode_planar_qpr_rpr`] (one `Q[p]` and one `R[p]` per precinct,
Annex C.2 Table C.1), and round 245 supplies the picker that drives
it against a byte budget.

* `pick_qpr_rpr_for_target_bytes(width, height, nc, cpih, nlx, nly,
  target_bytes, &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u8>)>` — picks
  the per-precinct `(q_precincts, r_precincts)` pair (each of length
  `Np,y × Np,x` at `Cw = 0`) that drives
  [`encode_planar_qpr_rpr`] to emit a codestream `≤ target_bytes`.
  Nested search — outer loop on uniform `rp ∈ 0..=NL-1`, inner
  three-pass per-precinct `q_precincts` search at the current `rp`:
  (1) lossless probe at `q_precincts = [0; n]`, (2) uniform-`Q`
  bisect on `1..=15`, (3) per-precinct activity-driven relaxation
  walking the lowest-activity precincts first (L1 norm of
  row-to-row first-difference summed across every plane within
  each precinct's `Hp = 2^NL,y` image-row window from Annex B.5,
  computed by the `precinct_activity` helper). Promotion stops on
  the first `rp` whose inner search fails (refinement is monotone
  non-decreasing in codestream length at fixed `Q[p]`, so higher
  `rp` cannot fit either); lossless short-circuit canonicalises to
  `r_precincts = [0; n]` whenever the inner picker returns the
  all-zero `q_precincts` vector (refinement is a wire-only no-op
  at the `T[p,b]` 0 floor, so promoting `rp` adds zero
  rate-distortion value).
* `encode_planar_qpr_rpr_target_bytes(width, height, nc, cpih, nlx,
  nly, target_bytes, &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u8>,
  Vec<u8>)>` — convenience wrapper: returns
  `(codestream, q_precincts, r_precincts)` in one call.

The round-218 / round-224 / round-230 pickers all worked at
per-slice granularity (one `Q` per slice, one uniform `rp` for the
picture); round 245 closes the granularity gap by giving each
precinct its own `Q[p]` while keeping `R[p]` as a uniform
picture-wide knob the outer loop walks. Every measurement is a real
[`encode_planar_qpr_rpr`] call — there is no internal model of the
entropy coder.

Composition behaviour:

* The wrapper output through [`encode_planar_qpr_rpr`] is
  byte-identical to a direct [`encode_planar_qpr_rpr`] invocation
  at the picker's chosen `(q_precincts, r_precincts)`.
* At a budget the lossless probe satisfies, the picker returns
  `(q_precincts, r_precincts) = ([0; n], [0; n])` and the stream
  is byte-identical to [`encode_planar`].
* PSNR ≥ 25 dB at a 90% lossless budget on the 32×32 luma XOR-ramp
  fixture (the activity-driven relaxation concentrates bits on the
  low-activity precincts).

Errors:

* `target_bytes unreachable; rp=0 Q=15 emits N bytes` when even
  `q_precincts = [15; n]` + `r_precincts = [0; n]` overshoots the
  budget. Matches the round-224 / round-230 baseline-reachability
  error shape.
* `target_bytes must be > 0` precondition guard.

Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`,
`Fs = 0`, `Qpih = 0`, `B[i] = 8`.

Out-of-scope (next round):

* High-bit-depth widening for the per-precinct picker (`B[i] ∈
  9..=16` against the round-118 `u16`-LE plane format; mirrors the
  round-230 widening of the round-224 per-slice picker).
* Non-uniform `r_precincts` search — round 245's outer loop walks
  a uniform `rp`. A future round could lift the inner picker to
  also relax `R[p]` per precinct (e.g. priority-aware band ranking)
  while keeping the activity-driven `Q[p]` relaxation in place.
* `Cw > 0` multi-precinct-column widening (the picker's
  `compute_precinct_row_ranges` helper returns one range per
  precinct row, which already maps correctly to `Np,y × 1` at
  `Cw = 0`; widening to `Np,y × Np,x` requires per-column slicing).

+6 tests, 394 total.

## round 242 (joint per-precinct `Q[p] × R[p]` override, encoder side)

Closes the round-239 "Out-of-scope (next round)" tail: the joint
per-precinct `Q[p] × R[p]` cross-product. Round 233 lifted
picture-level `q` to one `Q[p]` per precinct ([`encode_planar_qpr`]);
round 239 lifted picture-level `R[p]` to one per precinct
([`encode_planar_rpr`]). Both lived on the same precinct-header pair
(Annex C.2 Table C.1 — `Q[p]` is precinct-header byte 3, `R[p]` is
byte 4), and the in-place overlay helper `precinct_cfg_for` already
composed them cleanly. Round 242 is the public entry point that
surfaces both vectors to the caller at the same time, so `R[p]`
becomes an active rate-distortion lever where the round-239
`q = 0` pin had left it as a wire-only no-op.

The bitstream-wire impact is the per-precinct `Q` byte AND the
per-precinct `R` byte in each precinct header. The decoder reads both
per precinct (`parse_precinct_header` + `precinct_truncation`) since
the early rounds, so no decoder change is needed — round 242 is pure
encoder rate-allocation plumbing.

* `encode_planar_qpr_rpr(width, height, nc, cpih, nlx, nly,
  q_precincts, r_precincts, &[Vec<u8>]) -> Result<Vec<u8>>` — round-242
  joint per-precinct entry point. Both vectors are indexed in raster
  scan order with the precinct at row `py`, column `px` at position
  `py * Np,x + px`. Each must be empty-or-length `Np,y × Np,x`, where
  `Np,y = ⌈Hf / 2^NL,y⌉` and `Np,x = 1` at `Cw = 0`. At least one of
  the two must be non-empty — the both-empty path is `encode_planar`.
* Each `q_precincts[p]` in `0..=15` (Annex C.6.2 Table C.10 `T` clamp
  range). Each `r_precincts[p]` in `0..=NL − 1` where
  `NL = Nc × Nβ` (Annex B.6 `NL` definition at `Sd = 0`).
* `Fq` is auto-selected (`8` whenever any `q_precincts[p] > 0`, else
  `0`) — same rule as `encode_planar_qpr`, preserves the
  byte-identical-to-`encode_planar` all-zero reduction.
* The picture-level fallbacks the precinct overlay compares against
  are picked so a uniform vector resolves to a no-op:
  `cfg.q = max(q_precincts)` and `cfg.rp = max(r_precincts)` — when
  every entry equals its picked-max, the per-precinct branch in
  `precinct_cfg_for` returns `None` and the codestream reduces to the
  picture-level form.

**Why this is the active lever:** Annex C.6.2 Table C.10 truncation
is `T[p, b] = clamp(Q[p] − G[b] − r, 0, 15)` with `r = (P[b] < R[p]) ?
1 : 0`. At the round-239 picture-wide `Q = 0` floor every clamp
landed at zero regardless of `r`, so per-precinct `R[p]` only
surfaced in the precinct header and not in the data sub-packet bytes.
With `q_precincts[p] > 0` the clamp is no longer floored, and a
non-zero `r_precincts[p]` subtracts `1` from `T[p, b]` on the `R[p]`
lowest-index bands — one *extra retained* magnitude bitplane per
affected band per precinct. PSNR consequently rises on the
alternating-`R = 3` pattern at uniform `Q = 4` strictly above the
same-`Q` R-off baseline.

**Composition behaviour:**

* `q_precincts = [0; n]` + `r_precincts = [0; n]` is byte-identical
  to `encode_planar` at the same geometry (both levers collapse to
  no-op).
* `q_precincts = [0; n]` + any `r_precincts = R` is byte-identical
  to `encode_planar_rpr(R)` — the `Q = 0` floor restores round 239's
  wire-only `R[p]` behaviour exactly.
* Picture-level chain (PIH `Hsl = 0` single slice; CDT 8-bit `B[i] =
  8`; WGT carries the per-band gains and priorities) is identical to
  the round-233 / round-239 outputs.

**Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
`Cpih ∈ {0, 1, 3}`, `Cw = 0` (single precinct column → `Np,x = 1`),
`Hsl = 0` (single slice), `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`.

Tests: +5 (388 total).
* `round242_qpr_rpr_both_zero_matches_encode_planar_lossless` — both
  vectors all-zero reproduces the `encode_planar` baseline
  byte-for-byte at NL=2/2, 32×32 luma.
* `round242_qpr_rpr_r_active_at_q_gt_0` — at uniform `Q[p] = 4`,
  alternating `R[p] = 3` raises PSNR strictly above the same-`Q`
  `R[p] = 0` baseline on the 64×64 luma ramp (round 239's
  `q = 0` no-op pin is lifted).
* `round242_qpr_rpr_wire_carries_q_and_r` — the first precinct's
  header bytes 3 (`Q[p]`) and 4 (`R[p]`) carry the caller's values
  on a 16-precinct NL=1/1 32×32 luma encode; round-trips through the
  decoder with PSNR ≥ 20 dB.
* `round242_qpr_rpr_q_zero_matches_rpr` — `q_precincts = [0; n]` +
  any `r_precincts = R` is byte-identical to `encode_planar_rpr(R)`
  at NL=2/2 (round-239 reduction pin).
* `round242_qpr_rpr_rejects_empty_and_out_of_range` — calling with
  both vectors empty is rejected (the both-empty path is
  `encode_planar`); `q_precincts[p] > 15` and `r_precincts[p] > NL −
  1` are rejected by `EncodeConfig::validate` before any encode work
  runs.

Out-of-scope (next round): high-bit-depth widening of the joint
primitive at `bd = B[i] ∈ 9..=16`, `Cw > 0` multi-column, and
`Hsl > 0` multi-slice × per-precinct intersections.

## Unreleased — round 239 (per-precinct `R[p]` override, encoder side)

Lifts the round-115 picture-wide `R[p]` mechanism (Annex C.2 Table C.1,
the precinct-refinement byte) to the spec-natural per-precinct form
where `R[p]` is indexed by precinct `p`. Round 233 took `Q[p]` from
picture-level to per-precinct; round 239 takes the second precinct-
header rate lever the same distance — one `R[p]` per precinct.

The bitstream-wire impact is exactly the per-precinct `R` byte in
each precinct's header (Annex C.2 Table C.1). The decoder reads
`R[p]` per precinct (`parse_precinct_header` + `precinct_truncation`)
since round 115, so no decoder change is needed — the per-precinct
override is a pure encoder rate-allocation lever, symmetric with the
round-233 `Q[p]` override.

* `encode_planar_rpr(width, height, nc, cpih, nlx, nly, r_precincts,
  &[Vec<u8>]) -> Result<Vec<u8>>` — round-239 per-precinct `R[p]`
  override entry point. `r_precincts` is indexed in raster scan order
  with the precinct at row `py`, column `px` at position
  `py * Np,x + px`; length must equal `Np,y × Np,x` where
  `Np,y = ⌈Hf / 2^NL,y⌉`. Each entry is in `0..=NL − 1` where
  `NL = (Nc − Sd) × Nβ + Sd` is the total band count (Annex B.6 NL
  definition; at the round-239 `Sd = 0` scope `NL = Nc × Nβ`).
* `EncodeConfig` gains an `r_precincts: Vec<u8>` field. Empty → every
  precinct inherits the picture-level `rp` (round-115 behaviour) —
  preserves byte-identical output for every existing callsite.
  Non-empty validates length = `Np,y × Np,x` and each entry `≤ NL − 1`.
* The existing `precinct_cfg_for(cfg, py, px, np_x)` helper, which the
  round-233 work introduced to overlay per-precinct `Q[p]`, is
  extended to also overlay per-precinct `R[p]` — the helper now clones
  the cfg when either `q_precincts` or `r_precincts` resolves to a
  value differing from the outer cfg. Both per-precinct vectors compose
  cleanly: the precinct sees its own `(Q[p], R[p])` pair at the same
  emission site.

**Composition behaviour:**

* `r_precincts = [0; n]` is byte-identical to `encode_planar` at the
  same geometry (every precinct's `R[p] = 0` reduces to the picture-
  level lossless stream).
* Picture-wide `q = 0` is pinned by `encode_planar_rpr`: the
  Annex C.6.2 Table C.10 truncation
  `T[p, b] = clamp(Q[p] − G[b] − r, 0, 15)` floors at the `0` clamp
  regardless of `r`, so any `r_precincts` pattern emits the same data
  sub-packet bytes (the wire-level change is the per-precinct `R[p]`
  byte alone). The lever exposes the precinct-header field without
  yet engaging it as a rate-distortion lever — that requires a
  future `encode_planar_qpr_rpr` cross-product at `q > 0`.
* The picture-header chain (PIH `Hsl = 0` single slice; CDT 8-bit
  `B[i] = 8`; WGT carries the per-band gains and priorities) is
  identical to the round-115 `encode_planar_rp` output.

**Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
`Cpih ∈ {0, 1, 3}`, `Cw = 0` (single precinct column → `Np,x = 1`,
so `r_precincts` reduces to one entry per precinct row), `Hsl = 0`
(single slice), `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`.

Tests: +3 (383 total).
* `round239_rpr_all_zero_matches_encode_planar_lossless` — every
  precinct at `R[p] = 0` reproduces the `encode_planar` baseline
  byte-for-byte at NL=2/2, 32×32 luma.
* `round239_rpr_wire_carries_per_precinct_r` — every precinct's
  header byte 4 (`R[p]`, the round-115 byte position) carries the
  caller's value; the decoder round-trips a 32×32 luma at NL=1/1
  with a mixed `R[p] ∈ 0..=3` pattern across 16 precincts.
* `round239_rpr_rejects_wrong_length_and_out_of_range` — wrong
  `r_precincts.len()` (expected `Np,y × Np,x`) and any entry exceeding
  `NL − 1` are rejected with `JpegXsError::Invalid` before any
  encode work runs.

Out-of-scope (next round): the `q > 0` × `r_precincts` cross-product
(`encode_planar_qpr_rpr`) carrying both per-precinct vectors so
`R[p]` becomes an active rate-distortion lever — granting one extra
retained bitplane to the `R[p]` lowest-index bands on a per-precinct
basis. Multi-column (`Cw > 0`) and multi-slice (`Hsl > 0`) per-
precinct overrides also intersect on a future round.

## Unreleased — round 233 (per-precinct `Q[p]` override, encoder side)

Lifts the round-206 per-slice `Q[p]` mechanism (one `Q[p]` per slice
in top-down `Yslh` order) to the spec-natural per-precinct form of
Annex C.2 Table C.1, where `Q[p]` is indexed by precinct `p`. Round
206 took `q` from picture-level (round 3) to one per slice; round 233
takes it the rest of the way — one `Q[p]` per precinct.

The bitstream-wire impact is exactly the per-precinct `Q` byte in
each precinct's header (Annex C.2 Table C.1). The decoder reads
`Q[p]` per precinct (`parse_precinct_header` + `precinct_truncation`)
since the early rounds, so no decoder change is needed — the per-
precinct override is a pure encoder rate-allocation lever.

* `encode_planar_qpr(width, height, nc, cpih, nlx, nly, q_precincts,
  &[Vec<u8>]) -> Result<Vec<u8>>` — round-233 per-precinct `Q[p]`
  override entry point. `q_precincts` is indexed in raster scan order
  with the precinct at row `py`, column `px` at position
  `py * Np,x + px`; length must equal `Np,y × Np,x` where
  `Np,y = ⌈Hf / 2^NL,y⌉`. Each entry is in `0..=15` (the band-
  truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` math is
  identical, only the source of `Q[p]` changes). `Fq` is
  auto-selected: `0` (lossless) when every entry is `0`, else `8`
  (regular per Table A.8 — required for any non-zero `Q[p]`).
* `EncodeConfig` gains a `q_precincts: Vec<u8>` field. Empty → every
  precinct inherits its slice's `Q[p]` (from `q_slices`) or the
  picture-level `q` when both are empty — preserves byte-identical
  output for every existing callsite. Non-empty validates length =
  `Np,y × Np,x` and each entry `≤ 15` (and `> 0` requires `Fq = 8`).
* `precinct_cfg_for(cfg, py, px, np_x)` helper layers on top of
  `slice_cfg_for(cfg, t)` inside `write_slice`: per-precinct override
  wins where it's non-empty, then per-slice, then picture-level `q`.

**Composition behaviour** (the four byte-identity properties — every
combination has a test against its expected baseline):

* `q_precincts = [0; n]` is byte-identical to `encode_planar` (every
  precinct lossless reduces to the picture-level lossless stream).
* `q_precincts = [q; n]` (every precinct same `Q[p]`) is byte-
  identical to `encode_planar_lossy` at that single `q` (the per-
  precinct override resolves to a no-op when `q_precincts[idx]`
  equals `cfg.q`, the picture-level fallback).
* Mixed `q_precincts` produces a strictly larger codestream than
  `[15; n]` whenever any entry is `< 15` (lower `Q[p]` retains more
  magnitude bitplanes).
* Composes with the reversible RCT (`Cpih = 1`, Annex F.3 — bit-
  depth agnostic operand window `c < 3`) and Star-Tetrix (`Cpih = 3`,
  Annex F.5 — `i32` integer linear combinations on the wavelet-
  domain coefficients, Q-agnostic on the lifting). 4-component CFA
  `Cpih = 3` self-roundtrips losslessly at `q_precincts = [0; n]`.

**Scope:** 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`),
`Cpih ∈ {0, 1, 3}`, `Cw = 0` (single precinct column → `Np,x = 1`,
so `q_precincts` reduces to one entry per precinct row), `Hsl = 0`
(single slice), `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`.
High-bit-depth widening, multi-column (`Cw > 0`) and the
`Hsl > 0` × `q_precincts` cross-product (each slice carrying its own
sub-array of per-precinct overrides) intersect with the lever on
future rounds.

Tests: +9 (380 total). Covers lossless byte-identity, uniform-Q
byte-identity to `encode_planar_lossy`, mixed-Q PSNR ≥ 30 dB +
round-trip, the per-precinct `Q[p]` byte surfacing on the wire,
length / range rejection, RCT + Star-Tetrix lossless self-
roundtrip, and the "lower Q keeps more bits" rate-allocation
property against a uniform-max-Q baseline.

## Round 230 (high-bit-depth widening of the joint primitive, encoder side)

Closes the round-224 "Out-of-scope (next round)" tail: the
high-bit-depth widening of the joint per-slice `Q[p]` + precinct
refinement `R[p]` primitive at `bd = B[i] ∈ 9..=16`. The round-224
note observed that `encode_planar_inner_bd` already plumbs all four
parameters (`hsl`, `q_slices`, `rp`, `bd`) through to the precinct
emitter, so the round-230 widening is the public-API plumbing around
the `u16`-LE per-plane format inherited from rounds 118 / 133 / 151
plus a `u16`-aware slice-activity metric for the picker's relaxation
pass.

Bit depth is orthogonal to both rate levers:

* The forward quantizer (Annex D.4 Table D.3) runs on `i32` wavelet
  coefficients regardless of `B[i]`.
* The Annex C.6.2 Table C.10 refinement term
  `r = (P[b] < R[p]) ? 1 : 0` operates on the band priority `P[b]`
  and the precinct's `R[p]` — neither carries `B[i]`.
* The reversible RCT (Annex F.3, `Cpih = 1`) is bit-depth agnostic
  on the operand window `c < 3`.

So the only bit-depth-dependent pieces remain the DC level shift
`1 << (bd − 1)` (Annex G.3 inverse) and the two-bytes-per-sample
plane packing — both inherited from rounds 118 / 133 / 151.

* `encode_planar_hsl_qslice_rp_highbd(width, height, nc, cpih, nlx,
  nly, bd, hsl, q_slices, rp, &[Vec<u16>]) -> Result<Vec<u8>>` —
  high-bit-depth joint primitive. Validates `bd ∈ 9..=16` and
  `cpih ∈ {0, 1}`, validates the per-plane `width * height` sample
  count and the `0..=2^bd − 1` nominal range, packs each plane to
  `to_le_bytes()` form and dispatches to `encode_planar_inner_bd` with
  `Bw = bd`, `Fq = 8` if any slice quantizes else `0`, `Cw = 0`,
  `Sd = 0`, `Fs = 0`, `Qpih = 0`. The codestream uses the same
  precinct `Q[p]` / `R[p]` layout as the 8-bit joint primitive.
* `pick_q_slices_rp_for_target_bytes_highbd(width, height, nc, cpih,
  nlx, nly, bd, hsl, target_bytes, &[Vec<u16>]) -> Result<(Vec<u8>, u8)>`
  — high-bit-depth picker. Two-axis nested search: outer loop on `rp`
  walks from `0` up to `NL − 1` keeping the last fitting solution;
  inner loop replays r212's lossless probe → uniform-`Q` bisect →
  activity-driven per-slice relaxation against the round-230
  high-bit-depth primitive. Promotion stops as soon as the inner
  search at `rp+1` fails (refinement is monotone non-decreasing in
  codestream length at fixed `Q[p]`). Baseline reachability: if even
  `rp = 0` + `q_slices = [15; n]` overshoots, errors with
  `target_bytes unreachable; rp=0 Q=15 emits N bytes`.
  `target_bytes == 0` rejected. Every measurement is a real
  `encode_planar_hsl_qslice_rp_highbd` call — no internal model of
  the entropy coder, no oracle, no external library.
* The picker's per-slice relaxation pass uses a new
  `slice_activity_u16` helper rather than reusing the byte-plane
  variant on the packed `to_le_bytes()` form: a high-bit-depth `u16`
  picture's spatial structure lives in the original sample values, so
  computing the L1 row-to-row first-difference on the original `u16`
  planes is the right metric (the byte-packed form would mix the low
  and high bytes of adjacent rows into the activity sum and bias the
  relaxation toward arbitrary slices).
* `encode_planar_hsl_qslice_rp_target_bytes_highbd(.., target_bytes,
  &[Vec<u16>]) -> Result<(Vec<u8>, Vec<u8>, u8)>` — convenience
  wrapper returning `(codestream, q_slices, rp)`. The codestream is
  byte-identical to a follow-up
  `encode_planar_hsl_qslice_rp_highbd(.., q_slices, rp, ..)` call, so
  callers can persist `(q_slices, rp)` for reproducible re-encode.
* Scope: 4:4:4 (`sx[i] = sy[i] = 1` for `i < nc`), `Cpih ∈ {0, 1}`
  (no transform / reversible RCT), `Cw = 0` (single precinct
  column), `Sd = 0` (no CWD suppression), `Fs = 0` (joint signs),
  `Qpih = 0` (deadzone), `bd ∈ 9..=16`. Star-Tetrix (`Cpih = 3`) and
  NLT pre-distortion intersect the joint primitive on a future
  round.

12 new tests (359 → 371 total):
* `round230_joint_highbd_lossless_roundtrip_independent_of_rp` —
  `q_slices = [0; n]` at any `rp` self-roundtrips bit-exactly through
  the decoder at 10-bit.
* `round230_joint_highbd_rp_zero_single_slice_matches_highbd_lossy`
  — single-slice + `rp = 0` joint output byte-identical to
  `encode_planar_highbd_lossy` at the same `q`.
* `round230_joint_highbd_lossy_compresses_smaller_than_lossless` —
  `q_slices = [3, 3]` stream strictly smaller than `[0, 0]` at the
  same bit depth (the deadzone truncation is biting).
* `round230_joint_highbd_10bit_luma_psnr_q1_floor` — 10-bit q=1 PSNR
  ≥ 40 dB through the joint primitive.
* `round230_joint_highbd_mixed_q_slices_12bit_psnr_floor` — mixed
  `Q[p] = [0, 3]` at 12-bit holds ≥ 30 dB PSNR.
* `round230_joint_highbd_picker_rejects_zero_budget` —
  `target_bytes = 0` rejected before any encode work.
* `round230_joint_highbd_picker_returns_max_rp_when_budget_is_huge` —
  huge budget selects max-refinement + lossless slices.
* `round230_joint_highbd_picker_errors_when_budget_unreachable` —
  tiny budget triggers the `unreachable` error.
* `round230_joint_highbd_picker_output_fits_budget` — mid-range
  budget produces an output `<= target_bytes`.
* `round230_joint_highbd_target_bytes_wrapper_matches_manual_pair`
  — wrapper output byte-identical to manual re-encode at picked
  pair.
* `round230_joint_highbd_rgb_rct_lossless_roundtrip` — 10-bit
  3-component RGB + RCT (`Cpih = 1`) self-roundtrips bit-exactly at
  `q_slices = [0; n]` and any `rp`.
* `round230_joint_highbd_rejects_bd8_and_cpih3` — `bd = 8` and
  `Cpih = 3` rejected on the high-bit-depth joint primitive (use the
  8-bit joint primitive or wait for the Star-Tetrix follow-up).

Out-of-scope (next round): a Star-Tetrix-aware high-bit-depth joint
primitive (`Cpih = 3` requires the 4-component CFA plane layout and
CTS / CRG markers — the joint picker would also need to know about
the Star-Tetrix gain table); high-bit-depth chroma sub-sampling on
the joint primitive (`encode_planar_subsampled_highbd_lossy` already
exposes the per-component `(sx, sy)` lossy path, and the joint
primitive could be widened similarly).

## Unreleased — round 224 (joint per-slice `Q[p]` + `R[p]` rate-budget picker, encoder side)

Closes the round-218 "next round" tail: the **joint** picker that
simultaneously picks `q_slices` (per-slice `Q[p]`, Annex C.2 Table C.1
lifted to per-slice in round 206) and `rp` (precinct refinement
`R[p]`, Annex C.2 Table C.1 + Annex C.6.2 Table C.10) against a single
byte budget.

The two levers are orthogonal on the bitstream: per-slice `Q[p]`
lives in each precinct's `Q` byte and `R[p]` lives in each precinct's
`R` byte, so any `(q_slices, rp)` pair the picker emits is
spec-compliant. The Annex C.6.2 Table C.10 truncation
`T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` with
`r = (P[b] < R[p]) ? 1 : 0` is monotone non-increasing in `Q[p]` and
monotone non-decreasing in `R[p]`, so the joint picker walks both
axes in a nested search.

* `encode_planar_hsl_qslice_rp(width, height, nc, cpih, nlx, nly,
  hsl, q_slices, rp, &[Vec<u8>]) -> Result<Vec<u8>>` — the joint
  primitive. Composes [`encode_planar_hsl_qslice`] (round 206 per-
  slice `Q[p]`) and [`encode_planar_rp`] (round 115 precinct
  refinement) on a single encode call. All-equal `q_slices` + `rp =
  0` is byte-identical to `encode_planar_hsl_qslice` at that
  `q_slices`; single-entry + `hsl = 0` + `rp > 0` is byte-identical
  to `encode_planar_rp` at the same `(q, rp)`; `q_slices = [0; n]`
  with any `rp` is byte-identical to the lossless `rp = 0` stream
  (refinement is a no-op when `T` is already at its 0 floor).
* `pick_q_slices_rp_for_target_bytes(width, height, nc, cpih, nlx,
  nly, hsl, target_bytes, &[Vec<u8>]) -> Result<(Vec<u8>, u8)>` —
  two-axis nested picker. Outer loop on `rp` from `0` up to `NL-1`,
  keeping the last fitting solution; inner loop reuses r212's three-
  pass `q_slices` strategy (lossless probe, uniform-`Q` bisect,
  activity-driven per-slice relaxation) against
  `encode_planar_hsl_qslice_rp` at the current `rp`. Promotion rule:
  if the inner search succeeds at `rp+1`, replace the best;
  otherwise stop (refinement is monotone non-decreasing in
  codestream length at fixed `Q[p]`, so higher `rp` cannot fit
  either). Baseline reachability: if even `rp = 0` + `Q = 15`
  overshoots, errors with `target_bytes unreachable; rp=0 Q=15 emits
  N bytes` (no choice of `(q_slices, rp)` can fit). `target_bytes ==
  0` rejected. Every measurement is a real call into
  `encode_planar_hsl_qslice_rp` — no internal model of the entropy
  coder, no oracle, no external library.
* `encode_planar_hsl_qslice_rp_target_bytes(.., target_bytes,
  &[Vec<u8>]) -> Result<(Vec<u8>, Vec<u8>, u8)>` — convenience
  wrapper returning `(codestream, q_slices, rp)`. The codestream is
  byte-identical to a follow-up `encode_planar_hsl_qslice_rp(..,
  q_slices, rp, ..)` call, so callers can persist `(q_slices, rp)`
  for reproducible re-encode.
* Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Sd = 0`, `Fs = 0`,
  `Qpih = 0`, `B[i] = 8`. The inner encoder already plumbs all four
  parameters through `encode_planar_inner_bd`, so a future round
  could widen to high-bit-depth without changing the picker's
  contract.

10 new tests (349 → 359 total):
* `round224_joint_primitive_matches_hsl_qslice_rp_zero` — all-equal
  `q_slices` + `rp = 0` byte-identical to `encode_planar_hsl_qslice`.
* `round224_joint_primitive_matches_rp_when_single_slice` —
  single-slice + `hsl = 0` + `rp > 0` byte-identical to
  `encode_planar_rp` at the same `(q, rp)`.
* `round224_joint_lossless_roundtrip_independent_of_rp` —
  `q_slices = [0; n]` at any `rp` self-roundtrips losslessly.
* `round224_joint_picker_rejects_zero_budget` — `target_bytes = 0`
  rejected before any encode work.
* `round224_joint_picker_returns_max_rp_when_budget_is_huge` —
  budget ≥ `(rp = NL-1, q = [0; n])` stream selects the maximum
  refinement with lossless slices.
* `round224_joint_picker_errors_when_budget_unreachable` — tiny
  budget triggers the `unreachable` error.
* `round224_joint_picker_output_fits_budget` — mid-range budget
  produces an output `<= target_bytes`.
* `round224_joint_target_bytes_wrapper_matches_manual_pair` — the
  wrapper output equals `encode_planar_hsl_qslice_rp(
  pick_q_slices_rp_for_target_bytes(..), ..)` byte-for-byte.
* `round224_joint_picker_works_for_rgb_rct` — picker is colour-
  transform agnostic (`Cpih = 1`, Nc=3, RGB+RCT).
* `round224_joint_picker_at_least_as_good_as_rp_alone` — at a
  budget where `pick_rp_for_target_bytes` finds a non-zero answer,
  the joint picker matches or exceeds that refinement level (it
  can additionally lower `Q[p]` to free up bytes for more `R[p]`).

Out-of-scope (next round): an `R[p]`-driven PSNR-optimizing
**priority assignment** that replaces the plain band-index priorities
`P[b] = b` with a content-adaptive ordering; high-bit-depth widening
of the joint primitive (`encode_planar_inner_bd` already plumbs all
four parameters).

## Unreleased — round 218 (rate-budget driven `R[p]` picker, encoder side)

Builds on the round-115 `R[p]` precinct refinement primitive. Round
115 gave the caller the *mechanism* to refine the lowest-index bands
(LL first per the `β`-major band enumeration of Annex B.6) by lowering
`T[p,b]` via the Annex C.6.2 Table C.10 term `r = (P[b] < R[p]) ? 1 :
0`; round 218 adds the *policy* — a deterministic, library-free
picker that chooses an `R[p]` against a caller-supplied byte budget
and spends every available byte on additional refinement.

Complements r212's `pick_q_slices_for_target_bytes`: r212 trades
quantization strength **between slices** at a fixed refinement; r218
trades refinement strength **between bands** at a fixed quantization.
Together they cover both axes of the encoder's rate-distortion lever.

* `pick_rp_for_target_bytes(width, height, nc, cpih, nlx, nly, q,
  target_bytes, &[Vec<u8>]) -> Result<u8>` — linear scan from `R[p] =
  NL-1` down to `0`, returning the first `R[p]` whose codestream fits
  the budget. Refinement is monotone non-decreasing in codestream
  length (each refined band gains one extra retained magnitude
  bitplane), so the first fit is also the largest fit. Step 1 probes
  `R[p] = 0` as the baseline — if even that overshoots the budget,
  the picker errors with `target_bytes unreachable; R[p]=0 emits N
  bytes` (the budget is unreachable by `R[p]` alone; lower `q` or
  `pick_q_slices_for_target_bytes` can still help). `target_bytes ==
  0` is rejected as a precondition. Every measurement is a real call
  into `encode_planar_rp` — no internal model of the entropy coder,
  no oracle, no external library.
* `encode_planar_rp_target_bytes(.., target_bytes, &[Vec<u8>]) ->
  Result<(Vec<u8>, u8)>` — convenience wrapper that returns
  `(codestream, rp)`. The codestream is byte-identical to a follow-up
  `encode_planar_rp(.., rp, ..)` call, so callers can persist the
  picked `rp` for reproducible re-encode.
* Scope mirrors `encode_planar_rp` exactly: 4:4:4, `Cpih ∈ {0, 1, 3}`,
  single precinct column (`Cw = 0`), single slice (`Hsl = 0`), no CWD
  suppression (`Sd = 0`), `Fs = 0`, `Qpih = 0`, `q ∈ 0..=15`. At `q =
  0` refinement is a lossless no-op (the truncation `T[p,b]` is
  already at its 0 floor regardless of the refinement term), so the
  picker returns `NL-1` — every refinement is byte-identical to
  `R[p] = 0` at `q = 0`.

8 new tests (341 → 349 total):
* `round218_rp_picker_rejects_zero_budget` — `target_bytes = 0` is
  rejected before any encode work.
* `round218_rp_picker_returns_nl_minus_one_when_budget_fits_max` —
  budget ≥ `R[p]=NL-1` stream length selects the maximum legal
  refinement (luma NL=2/2 q=2).
* `round218_rp_picker_returns_zero_when_budget_is_baseline` —
  budget = baseline length falls back to `R[p] = 0` when every
  higher value overshoots (output is monotone-non-decreasing).
* `round218_rp_picker_errors_when_baseline_overshoots` — budget <
  baseline triggers the `unreachable` error tagged with the actual
  baseline length.
* `round218_rp_picker_picks_intermediate_for_intermediate_budget` —
  budget strictly between baseline and max stream picks an
  intermediate `R[p]` whose stream fits.
* `round218_rp_picker_q0_lossless_roundtrip` — at `q = 0` the picker
  returns `NL-1` (every R[p] fits) and the codestream decodes
  losslessly.
* `round218_rp_target_bytes_wrapper_matches_manual_pair` — the
  wrapper output equals
  `encode_planar_rp(pick_rp_for_target_bytes(..), ..)` byte-for-byte.
* `round218_rp_picker_works_for_rgb_rct` — picker is colour-
  transform agnostic (`Cpih = 1`, Nc=3, RGB+RCT, NL=2/2 q=2 budget =
  max-stream length).

Out-of-scope (next round): a unified picker that joints r212's per-
slice `Q[p]` policy with this round's `R[p]` policy — pick both
`q_slices` and `rp` against a single budget. The two levers are
orthogonal on the bitstream (per-slice `Q[p]` lives in each
precinct's `Q[p]` byte; `R[p]` lives in each precinct's `R[p]` byte),
so a joint picker is a natural next step.

## Unreleased — round 212 (rate-budget driven per-slice `Q[p]` picker, encoder side)

Builds directly on the round-206 per-slice `Q[p]` slice. Round 206
gave the caller the *mechanism* to set `Q[p]` per slice; round 212
adds the *policy* — a deterministic, library-free picker that
chooses a `q_slices` vector against a caller-supplied byte budget
and concentrates bits on visually-salient (low-activity) slices.

* `pick_q_slices_for_target_bytes(width, height, nc, cpih, nlx, nly,
  hsl, target_bytes, &[Vec<u8>]) -> Result<Vec<u8>>` — three-pass
  search: (1) lossless probe at `q = [0; n_slices]`, return it if
  it fits; (2) uniform-Q binary search on `1..=15`; (3) per-slice
  relaxation walks the lowest-activity slices first, dropping their
  Q one step at a time while the candidate still fits the budget.
  Every measurement is a real call into `encode_planar_hsl_qslice`
  — the picker has no internal model of the entropy coder. Errors
  with `target_bytes unreachable; Q=15 emits N bytes` if even the
  most aggressive uniform-Q overshoots, and rejects `target_bytes == 0`
  as a precondition.
* `encode_planar_hsl_target_bytes(.., target_bytes, &[Vec<u8>]) ->
  Result<(Vec<u8>, Vec<u8>)>` — convenience wrapper that returns
  `(codestream, q_slices)`. The codestream is byte-identical to a
  follow-up `encode_planar_hsl_qslice(.., q_slices, ..)` call, so
  callers can persist the picked vector for reproducible re-encode.
* Activity ranking is the L1 norm of the row-to-row first-difference
  summed across every plane within each slice's image-row range —
  cheap (no FFT, no wavelet), spec-orthogonal, and reused as the
  per-slice priority key for the pass-3 relaxation. Single-slice
  mode (`hsl == 0`) collapses to a scalar bisect with the relaxation
  pass as a no-op.

6 new tests (335 → 341 total):
* `round212_picker_loose_budget_returns_lossless` — budget ≥
  lossless length picks `q = [0; n_slices]`.
* `round212_picker_tight_budget_fits_within_target` — half-of-
  lossless budget triggers the uniform-Q bisect, picker output fits
  and at least one slice is quantized.
* `round212_picker_unreachable_budget_errors` — `target_bytes = 1`
  errors with `unreachable` in the message.
* `round212_picker_zero_target_rejected` — explicit precondition
  rejection.
* `round212_picker_wrapper_is_byte_identical_to_qslice_encode` —
  `encode_planar_hsl_target_bytes` returns `(cs, q)` where re-
  encoding with `q` reproduces `cs` byte-for-byte, and the decoded
  picture clears the 25 dB PSNR floor.
* `round212_picker_single_slice_degenerate` — `hsl == 0` returns a
  length-1 vector and the picked codestream still fits the budget.

## Unreleased — round 206 (per-slice `Q[p]` override / slice-level rate budgeting)

Closes the "single constant `Q` across every slice" tail called out in
the round-103 / 195 / 201 Out-of-scope sections of the README. Before
this round the multi-slice path emitted every precinct with the same
picture-level `Q[p]`, which made `encode_planar_hsl` a layout-only
multi-slice primitive (one SLH per slice + a precinct-row partition)
without any per-slice rate-allocation lever. Round 206 lifts that to
an explicit per-slice `Q[p]` slice — one value per slice, in top-down
`Yslh` order.

Per Annex C.2 Table C.1, `Q[p]` is a per-precinct field — the
bitstream model already supports a different `Q[p]` for every
precinct, and the decoder reads `Q[p]` per precinct (via
`parse_precinct_header` + `precinct_truncation` in
`crate::entropy`). The encoder was the only side enforcing a
constant `Q[p]`; this round drops that synthetic constraint and lets
the caller pick `Q[p]` per slice. The bitstream-wire impact is
exactly the per-precinct `Q` byte inside each slice — SOC / CAP /
PIH / CDT / WGT / SLH markers and the entropy-data layout are
unchanged.

* `encode_planar_hsl_qslice(width, height, nc, cpih, nlx, nly, hsl,
  q_slices: &[u8], &[Vec<u8>]) -> Result<Vec<u8>>` — new public
  entry point. `hsl` is the slice height in precinct rows (PIH
  `Hsl`, identical semantics to `encode_planar_hsl`); `q_slices.len()`
  must exactly equal the slice count `⌈Np,y / max(hsl, 1)⌉` and each
  entry is in `0..=15` (the band-truncation
  `T[p,b] = clamp(Q − G[b] − r, 0, 15)` math is identical, only the
  source of `Q[p]` changes). The encoder picks `Fq` automatically:
  `Fq = 0` (lossless) when every entry is `0`, else `Fq = 8`
  (regular, Table A.8 — required for any non-zero `Q[p]`). Rejects
  wrong-length `q_slices` and entries `> 15`.
* `EncodeConfig::q_slices: Vec<u8>` — new field (empty preserves the
  byte-identical legacy single-`Q[p]` path; non-empty is validated
  against the slice count + per-entry range in
  `EncodeConfig::validate`).
* `slice_cfg_for(cfg, t)` — new internal helper. Returns
  `Some(clone)` with `cfg.q = q_slices[t]` when an override fires,
  otherwise `None`. The slice-emission loop in `write_slice` falls
  back to the picture-level `cfg.q` when this returns `None`, so the
  pre-round-206 callers stay on the byte-identical legacy path.
* The slice-emission loop in `write_slice` clones the config to the
  slice-local `Q[p]` once per slice and passes the slice-local
  config to every precinct emission inside that slice (both the
  multi-level `encode_precinct_cascade` path and the single-level
  `encode_precinct_single_level` path).

Lossless / lossy mixing is supported: a salient picture region's
slice can carry `Q[p] = 0` (lossless inside that slice's `T[p,b] = 0`
clamp) while edge slices carry `Q[p] = 4` (lossy). The `Fq = 8`
picture-level setting plus per-slice `Q[p] = 0` is a benign
configuration — the band-truncation `T[p,b] = clamp(0 − G[b] − 0, 0,
15) = 0` clamps to zero, so the deadzone quantizer kernel becomes
the identity and the slice is reconstructed exactly.

Tests landed (335 total → +7 vs round 201's 328):
* `round206_qslice_all_zero_matches_hsl_lossless` — `q_slices = [0;
  4]` over a 4-slice 32×32 luma picture is byte-identical to
  `encode_planar_hsl(.., q = 0, hsl = 2, ..)`.
* `round206_qslice_uniform_q_matches_hsl_lossy` — `q_slices = [2;
  4]` is byte-identical to `encode_planar_hsl(.., q = 2, hsl = 2,
  ..)`.
* `round206_qslice_mixed_q_round_trip_and_diverges` — mixed
  `q_slices = [0, 2, 4, 2]` keeps more bits than a constant-`Q = 4`
  baseline, the picture still decodes (PSNR ≥ 30 dB), and PIH still
  carries `Hsl = 2` + 4 slices.
* `round206_qslice_wire_carries_per_slice_q` — `q_slices = [0, 1, 2,
  3]` lands one distinct `Q[p]` byte per slice on the wire (verified
  by parsing the codestream + peeking at each slice's first
  precinct-header `Q[p]` byte).
* `round206_qslice_rejects_wrong_length` — `q_slices.len() != Np,y /
  Hsl` is rejected.
* `round206_qslice_rejects_oversize_q` — any `Q[p] > 15` is
  rejected.
* `round206_qslice_single_slice_matches_lossy` — `q_slices = [3]`
  with `hsl = 0` is byte-identical to `encode_planar_lossy(.., q =
  3, ..)`.

Out of scope (next round): a PSNR-driven slice budgeter — given a
rate budget, compute the per-slice `Q[p]` assignment that minimizes
distortion (or maximises PSNR). Round 206 lands only the primitive;
the picker is a follow-on.

## Unreleased — round 201 (high-bit-depth Star-Tetrix LOSSY)

Closes the last remaining high-bit-depth gap in the encoder surface
called out as the round-195 "out of scope" tail: lossy Star-Tetrix
(`Cpih = 3`, Annex F.5) at `B[i] ∈ 9..=16`. The inner
`encode_planar_inner_bd` already accepted `q > 0` with `cfg.cpih == 3`
since the original round-4 Star-Tetrix landing — round 195 wired up the
high-bit-depth plumbing but pinned `q = 0` for its lossless scope, so
the only delta here is a new public entry point that lifts that pin.

Bit depth, quantization, and the Star-Tetrix lifting are mutually
orthogonal: the Annex F.5 lifting (Tables F.4–F.8) is an integer linear
combination on `i32` coefficients, the forward quantizer
(`forward_quant_index`) and the inverse dequantizer both operate on
`i32` wavelet coefficients regardless of `B[i]`, and the colour
transform runs in the un-quantized wavelet domain — so the same Annex
D.4 Table D.3 deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)`
that drives the round-133 / 151 high-bit-depth lossy paths drives this
one as well. The decoder reconstructs with the matching Annex D.2
inverse (`Qpih = 0`).

* `encode_planar_star_tetrix_highbd_lossy(width, height, nlx, nly, bd,
  q, e1, e2, cf, ct, &[Vec<u16>]) -> Result<Vec<u8>>` — new public
  entry point. `bd ∈ 9..=16`; `q ∈ 1..=15`; pins `Nc = 4`,
  `sx[i] = sy[i] = 1` for `i < 4`, `Cpih = 3`, `Fq = 8` (regular,
  Table A.8 — required for `q > 0`). Input plane order is `Ω = [R, G1,
  G2, B]` matching the 8-bit `encode_planar_star_tetrix` and the
  round-195 lossless high-bit-depth form. Same little-endian `u16`
  plane format (`width * height` samples per plane in `0..=2^bd − 1`)
  and same DC level shift `1 << (bd − 1)` (Annex G.3 inverse) as the
  round-118 / 133 / 151 / 195 high-bit-depth paths. Emits the CTS
  marker (`Cf`, `e1`, `e2`) and the CRG marker (Table F.9 RGGB layout
  for `Ct = 0`, GRBG layout for `Ct = 1`) identical to the 8-bit /
  round-195 forms; only the per-component CDT `B[i]` byte, the PIH
  `Bw` byte, and the per-precinct `Q` field change on the wire.
  Rejects `q = 0` (use `encode_planar_star_tetrix_highbd`), `bd = 8`
  (use `encode_planar_star_tetrix` with `q > 0`), `bd > 16`, wrong
  plane count, and oversize samples.

Validation: PSNR ≥ 40 dB at 10-bit q=1 and at 16-bit Ct=1 q=1, ≥ 30 dB
at 12-bit q=2 on the synthetic 16×16 CFA fixture at NL=2/2. The q=2
codestream is strictly smaller than the lossless q=0 stream on the
same 12-bit picture — the Annex D.4 deadzone is biting.

## Unreleased — round 195 (high-bit-depth Star-Tetrix)

Widens the round-4 Star-Tetrix encoder (`Cpih = 3`, Annex F.5) from
`B[i] = 8`-only to any `bd = B[i] ∈ 9..=16`. Closes the last
high-bit-depth gap called out as the round-181 / 193 "out of scope"
tail (along with the round-118 / 133 / 151 `encode_planar_highbd*`
docstrings).

Star-Tetrix is the four-component CFA colour transform from Annex F.5
(Tables F.4–F.8). Its four lifting steps are integer linear
combinations on `i32` coefficients, identical for any input bit depth —
the existing `forward_star_tetrix` in `colour_transform.rs` already
accepts `&mut [i32]` slices and `encode_planar_inner_bd` already
dispatches to it when `cfg.cpih == 3`. The round-195 widening is
therefore pure plumbing: a new public entry point that packs
`u16`-LE planes (matching the round-118 high-bit-depth plane format)
and routes through `encode_planar_inner_bd` with `Bw = B[i] = bd` and
`Fq = 0` (the lossless choice of Table A.8). The DC level shift is
`1 << (bd − 1)` per the Annex G.3 inverse, identical to the
no-transform / RCT high-bit-depth paths.

* `encode_planar_star_tetrix_highbd(width, height, nlx, nly, bd, e1,
  e2, cf, ct, &[Vec<u16>]) -> Result<Vec<u8>>` — new public entry
  point. `bd ∈ 9..=16`; pins `Nc = 4` (Annex F.2 Table F.1
  Star-Tetrix operand window), `sx[i] = sy[i] = 1` for `i < 4`, `Cpih
  = 3`, `Fq = 0`, `q = 0`. Input plane order is `Ω = [R, G1, G2, B]`
  matching the 8-bit `encode_planar_star_tetrix`; each plane is
  `width * height` little-endian `u16` samples in `0..=2^bd − 1`.
  Emits the CTS marker (`Cf`, `e1`, `e2`) and the CRG marker
  (Table F.9 RGGB layout for `Ct = 0`, GRBG layout for `Ct = 1`)
  identical to the 8-bit form; only the per-component CDT `B[i]`
  byte and the PIH `Bw` byte change on the wire. Rejects `bd = 8`
  (use `encode_planar_star_tetrix`), `bd > 16`, wrong plane count,
  oversize samples; the rest of the parameter validation
  (`e1, e2 ≤ 3`, `cf ∈ {0, 3}`, `ct ≤ 1`, `nlx / nly` per Annex
  A.4.4) flows through the shared `EncodeConfig::validate`.

The matching `(rounds 118 / 133 / 151 / 181 / 193)
encode_planar_highbd*` rejection guards on `cpih = 3` stay in place —
those entry points remain scoped to `cpih ∈ {0, 1}` so callers reach
Star-Tetrix high-bit-depth through the dedicated
`encode_planar_star_tetrix_highbd` instead.

Tests landed (323 total → +6 vs round 193's 317):
* `r195_star_tetrix_highbd_10bit_round_trip` — 16×16 four-component
  CFA at NL=2/2 self-round-trips bit-exactly (PSNR `INFINITY` ≥ 30 dB
  floor).
* `r195_star_tetrix_highbd_12bit_ct1_round_trip` — 12-bit cover with
  Ct=1 (GRBG layout) + non-default `e1=2 / e2=3 / cf=3` (in-line),
  confirms the CTS / CRG markers survive on the high-bit-depth path.
* `r195_star_tetrix_highbd_16bit_round_trip` — 16-bit upper boundary,
  exercises the full `[0, 65535]` sample range through the Annex F.5
  lifting cascade.
* `r195_star_tetrix_highbd_rejects_bad_bd` — `bd ∈ {8, 17}` raise
  `Unsupported`.
* `r195_star_tetrix_highbd_rejects_oversize_sample` — a sample
  exceeding `2^bd − 1` is rejected.
* `r195_star_tetrix_highbd_rejects_wrong_plane_count` — a 3-plane
  input is rejected (Cpih=3 pins Nc=4).

## Round 193 — high-bit-depth NLT extended

Widens the round-7 NLT extended encoder (`Tnlt = 2`, Annex G.5) from
`B[i] = 8`-only to any `bd = B[i] ∈ 9..=16` against `Bw = 20`
(matching the round-181 NLT quadratic high-bit-depth choice — the top
of the Table A.8 `{8, 18, 20}` set, giving ≥ 4 bits of headroom for
the three-segment kernel above any supported component bit depth).

The blocker called out in round 181's "Out of scope" tail was the
`(1usize << bc).min(257)` cap on `build_extended_forward_lut`'s
reconstructed-level table. The cap was an 8-bit shortcut: at `bc = 8`
the table is 256 entries and the +1 is a defensive head-room slot, but
at `bc ∈ 9..=16` the cap dropped the table to a stale 257-entry size,
so the LUT could not address every input sample. This round drops the
cap; the LUT is now exactly `1 << bd` entries (max 65 536 at
`bd = 16`, ~512 KB of transient state, allocated once per encode).

The matching `bd > 8 && nlt = Some(Extended { .. })` rejection in
`encode_planar_inner_bd` is removed; the high-bit-depth Extended path
now flows through the same `Bw = 20` inner-bd routine as quadratic.

* `encode_planar_nlt_extended_highbd(width, height, nc, cpih, nlx,
  nly, bd, q, t1, t2, e, &[Vec<u16>]) -> Result<Vec<u8>>` — new
  public entry point. `bd ∈ 9..=16`, `cpih ∈ {0, 1}` (no transform or
  reversible RCT — bit-depth agnostic Annex F.3), plane format is
  little-endian `u16` per `JpegXsPlane` (`(width / sx[i]) × (height /
  sy[i])` samples with `sx[i] = sy[i] = 1`), `q = 0` is the
  lossless-within-LUT-resolution case and `q > 0` engages `Fq = 8`
  + per-band deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)`
  (Annex D.4). Rejects `bd = 8` (use 8-bit `encode_planar_nlt_extended`),
  Star-Tetrix `cpih = 3`, out-of-range samples, and the same
  `(t1, t2, e)` constraints as the 8-bit form (`0 < t1 < t2 ≤
  2^Bw − 1`, `1 ≤ e ≤ 4`).
* `build_extended_forward_lut` — drops `.min(257)` on the level-table
  size; now allocates `1 << bc` slots. The 8-bit path (`bc = 8`)
  remains byte-identical (`1 << 8 = 256` was already below the cap).
* `encode_planar_inner_bd` — removes the `bd > 8 && Extended`
  rejection guard; high-bit-depth Extended now lands on the same
  `bw = 20` branch the round-181 quadratic widening introduced. The
  inline LUT consumer reads `chunks_exact(2)` little-endian `u16`
  for `bd > 8` (mirroring the linear / quadratic paths in the same
  match) and the legacy single-byte path for `bd = 8`.
* `encode_planar_nlt_quadratic_highbd` doc — refreshed: the
  "NLT extended Tnlt=2 stays 8-bit-input specific" tail now points at
  the round-193 `encode_planar_nlt_extended_highbd` instead.

Tests landed (317 total → +10 vs round 190's 307):
* `r193_nlt_extended_highbd_10bit_psnr_above_30db` — 32×32 10-bit
  synthetic ramp self-round-trips at PSNR ≥ 30 dB.
* `r193_nlt_extended_highbd_12bit_psnr_above_30db` — 12-bit cover.
* `r193_nlt_extended_highbd_16bit_psnr_above_30db` — 16-bit upper
  boundary, exercising the full 65 536-entry LUT.
* `r193_nlt_extended_highbd_10bit_lossy_q2_psnr` — q=2 lossy
  ≥ 25 dB, stream ≤ q=0 stream.
* `r193_nlt_extended_highbd_10bit_rgb_rct_round_trip` — 10-bit
  RGB+RCT (`Cpih = 1`) composes: each plane self-decodes ≥ 30 dB.
* `r193_nlt_extended_highbd_rejects_bad_bd` — `bd ∈ {8, 17}` raise
  `Unsupported`.
* `r193_nlt_extended_highbd_rejects_star_tetrix_cpih` — `Cpih = 3`
  on this path raises `Unsupported`.
* `r193_nlt_extended_highbd_rejects_bad_params` — `t2 ≤ t1`,
  `t1 = 0`, `e ∈ {0, 5}`, `t2 > 2^Bw − 1` (Bw=20).
* `r193_nlt_extended_highbd_rejects_out_of_range_sample` — sample
  above `2^bd − 1` raises `InvalidData`.
* `r193_nlt_extended_highbd_now_accepted` — replaces the deleted
  round-181 `r181_nlt_extended_highbd_still_rejected` (which asserted
  the now-removed `bd > 8 && Extended` guard); confirms the path
  encodes and decodes to the expected plane byte count.

The Annex G.5 decoder (`output::extended_path`) was already parametric
in `B[i]` (round-118 widened `apply_output_scaling` to pack `u16` for
any `bc ∈ 9..=16`), so the decoder side needs no changes — every
round-193 codestream parses with the existing high-bit-depth pipeline.

## Unreleased — round 190 (picture-β slot indexing for 4:2:0 chroma)

Closes the `NL,y ≥ 3` 4:2:0 round-trip blocker tracked as task #1139 by
reconciling the band-index convention shared by the slice walker, the
forward cascade encoder and the inverse cascade decoder with the spec's
**picture-level** β slot enumeration (Annex B.3 Figure B.2, Annex B.4
`bx[β,i]`).

Prior to this round the codebase used a packed *per-component* β index
for the bitstream band id `b = (Nc − Sd) × β + i`: chroma's local
`n_beta(NL,x, N'L,y[i])` slots were placed at `β_pic ∈ [0, n_beta_local)`
with the tail slots `[n_beta_local, n_beta_pic)` marked non-existent. At
`NL,y = 2` (`NL=2/2` 4:2:0) the convention happened to round-trip
because chroma's deepest proxy level had `pow_chroma_band = 1` and
luma's level-1 outer-loop pow matched, but at `NL,y ≥ 3` the chroma
proxy depth diverged from luma's: chroma's deepest local-β triple
(`HL1,1 / LH1,1 / HH1,1`) lived inside luma's `β0 = 4` proxy iteration
(luma's `HL2,2 / LH2,2 / HH2,2` group), and the umod-`sy[i]` line-
inclusion guard dropped every odd image-grid λ — producing fewer
packets than the decoder's `BandCoefficients` buffer expected, leaving
half of chroma's deeper-band rows at zero. Result: value-corrupted
4:2:0 output at `NL=3/3` even though the bitstream parsed cleanly.

Per Figure B.2 the picture-β enumeration is *common to all components*
and shaded slots indicate chroma's missing bands (`β = 5, 6` for
`NL=5/2` 4:2:0; `β = 2, 3` for `NL=3/3` 4:2:0). Chroma's local DWT
output is then placed at the surviving picture-β slots via a
permutation that this round records explicitly.

* `slice_walker::picture_beta_to_local_beta(β_pic, NL,x, NL,y, sy[i])
  -> Option<u32>` — the new helper. Returns the chroma-local-β whose
  DWT output (from `forward_cascade_2d(wc, hc, NL,x, N'L,y[i])`)
  belongs at the picture-β slot, or `None` when the slot is missing
  for the component (`bx[β,i] = 0`). Identity for `sy[i] = 1`. For
  `sy[i] = 2` the mapping has the form
  - LL (β_pic = 0) → 0
  - Pure-horizontal `HL_k,NL,y` (β_pic ∈ [1, β1_pic)): chroma-local at
    `NL,x − dx + 1` (chroma's pure-H zone if `dx > N'L,y[i]`) or
    `β1_chroma + 3·(N'L,y[i] − dx)` (chroma's proxy zone if
    `dx ≤ N'L,y[i]`).
  - Proxy (τy = 1) with `dy_pic > N'L,y[i]` → `None` (the level is
    below chroma's plane resolution).
  - Proxy with `dy_pic ≤ N'L,y[i]`: chroma-local at
    `β1_chroma + 3·(N'L,y[i] − dy_pic) + within` (within 1 = LH,
    2 = HH).
* `PrecinctPlan::band_local_beta` — new field carrying the chroma-
  local-β per band slot. `u32::MAX` for non-existent slots; otherwise
  the index into `gathered[i][local_β]` (decoder) and
  `bands_per_comp[i][local_β]` (encoder).
* `build_plan_sd` populates `dx_arr / dy_arr / tau_x / tau_y / wb /
  hb / exists_arr` indexed by picture-β slot, with the (dx, dy, τx,
  τy) read from the component's *local* `beta_levels(local_β,
  NL,x, N'L,y[i])`. The Annex B.4 existence check is the
  `picture_beta_to_local_beta` `None` case.
* `encode_precinct_cascade` resolves the picture-β slot to chroma's
  local-β at slice-collection time and indexes
  `bands_per_comp[s.comp_i][s.local_beta]` in `extract_band_line`.
* `gather_precinct` walks picture-β slots and uses
  `precinct_plan.band_local_beta[b]` to address `gathered[i][local_β]`.
* `count_existing_bands / build_band_gains_sd /
  build_band_priorities_sd` walk picture-β slots and use
  `picture_beta_to_local_beta` to (a) decide existence and (b)
  resolve the τx/τy of the gain for the WGT body. The forward
  encoder's `t_for_band` does the same lookup so encoder T and
  decoder `precinct_truncation` agree on the band's gain.

Tests landed (306 total → 308, +6 vs round 181):
* `slice_walker::picture_beta_to_local_beta_444` — sy=1 identity over
  every `NL=k/m` with `k ≤ 5, m ≤ k`.
* `slice_walker::picture_beta_to_local_beta_420_nl5_2_figure_b2` —
  worked example from Figure B.2 (NL=5/2 4:2:0); chroma exists at
  picture-β ∈ {0,1,2,3,4,7,8,9}, mapping to chroma-local
  {0,1,2,3,4,5,6,7}.
* `slice_walker::picture_beta_to_local_beta_420_nl3` — NL=3/3 4:2:0.
* `slice_walker::picture_beta_to_local_beta_420_nl2` — NL=2/2 4:2:0
  (regression cover for the pre-existing case).
* `encoder::r190_chroma_420_nl3_lossless_round_trip` — 64×64 4:2:0
  NL=3/3 lossless self-round-trip is bit-exact across all three
  planes. This is task #1139 closed.
* `encoder::r190_chroma_420_nl3_lossy_q2_round_trip` — same shape
  at `q = 2` self-decodes to the correct plane byte counts (the
  quantization pipeline now finds the matching T on both sides
  via the corrected WGT-gain enumeration).

The `band_exists` Annex-B.4 standalone helper is removed; existence is
now decided by `picture_beta_to_local_beta` returning `None`.

## Unreleased — round 181 (high-bit-depth NLT quadratic)

Widens the round-5 NLT quadratic encoder (`Tnlt = 1`, Annex G.4) from
`B[i] = 8`-only to any `bd = B[i] ∈ 9..=16` against `Bw = 20` (the top
of the Table A.8 `{8, 18, 20}` set, giving ≥ 4 bits of precision
headroom for the sqrt above any supported component bit depth). The
forward pre-distortion `y = round(sqrt(x / (2^B[i] − 1)) × (2^Bw − 1))
+ dco` is parametric in `B[i]` — the round-5 implementation was a
hard-coded specialisation at `b_max = (1 << 8) − 1` for the input
domain only. The decoder side (`output::quadratic_path`) was already
parametric in `B[i]`, so the matching forward kernel completes the
round-trip.

* `oxideav_jpegxs::encoder::encode_planar_nlt_quadratic_highbd(width,
  height, nc, cpih, nlx, nly, bd, q, dco, &[Vec<u16>]) -> Result<Vec<u8>>`
  — high-bit-depth NLT quadratic entry point. `bd ∈ 9..=16` codes the
  picture with `Bw = 20`, DC level shift `1 << 19`. Plane format
  follows the round-118 convention: little-endian `u16` samples in
  `0..=2^bd − 1` per the `crate::image::JpegXsPlane` layout, 4:4:4
  only (`sx[i] = sy[i] = 1`). `cpih ∈ {0, 1}` (no transform / RCT —
  the Annex F.3 RCT operand window `c < 3` is bit-depth agnostic).
  `q = 0` is the lossless-within-sqrt-rounding mode (the round-5
  8-bit `round5_nlt_quadratic_high_psnr` already proves PSNR ≥ 40 dB
  is the floor for the algebraic-inverse round-trip rather than
  bit-exact); `q > 0` engages `Fq = 8` regular mode and per-band
  deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex D.4).
  `dco` is validated to signed-16-bit (Annex A.4.6 NLT marker σ:α).

* `src/encoder.rs` — `encode_planar_inner_bd` allows
  `NltParams::Quadratic` at `bd > 8` (rejecting only
  `NltParams::Extended` because its forward LUT inverter caps the
  reconstructed-level table at the 8-bit slot). The forward quadratic
  pre-distortion block in `write_slice` now reads the picture's input
  domain from `cfg.bit_depth` and switches between the u8 plane layout
  (legacy 8-bit) and the u16-LE plane layout (round 118 / 133 / 151 /
  181 high-bit-depth) on the same axis the linear path already
  switches on. `Bw = 20` is selected automatically for high-bit-depth
  NLT (replacing the round-118 / 133 / 151 `Bw = B[i]` choice for the
  NLT case only — the linear path remains `Bw = B[i]`).

Tests landed (302 total, +8 vs round 174):
* `r181_nlt_quadratic_highbd_10bit_lossless_psnr` — 10-bit luma at q=0
  self-roundtrips with PSNR ≥ 40 dB.
* `r181_nlt_quadratic_highbd_12bit_lossless_psnr` — 12-bit at q=0 PSNR
  ≥ 40 dB.
* `r181_nlt_quadratic_highbd_16bit_lossless_psnr` — 16-bit at q=0 PSNR
  ≥ 40 dB (upper bit-depth boundary).
* `r181_nlt_quadratic_highbd_10bit_lossy_q2_psnr` — 10-bit q=2 PSNR
  ≥ 30 dB and the codestream is no larger than the q=0 form.
* `r181_nlt_quadratic_highbd_10bit_nonzero_dco_encodes_decodes` — a
  positive `dco` value packs into the NLT σ:α field correctly and the
  resulting codestream decodes (the dco round-trip PSNR is bounded by
  the same forward / inverse asymmetry the 8-bit path inherits at
  non-zero dco; the test asserts the encode + decode pair completes
  with the correct output plane byte length).
* `r181_nlt_quadratic_highbd_rejects_8bit_bd` — `bd = 8` and `bd = 17`
  both return `Unsupported` so callers route to
  `encode_planar_nlt_quadratic` for 8-bit input.
* `r181_nlt_extended_highbd_still_rejected` — `bd > 8` with
  `NltParams::Extended` is still rejected with `Unsupported` (the
  forward LUT inverter is keyed on the 8-bit reconstructed-level
  table and is out of scope for round 181).
* `r181_nlt_quadratic_highbd_10bit_rgb_rct_round_trip` — three-
  component 10-bit RGB + RCT (`Cpih = 1`) composes with NLT quadratic
  pre-distortion and self-roundtrips above 35 dB PSNR per component.

Follow-up (deferred to a later round): NLT extended (`Tnlt = 2`,
Annex G.5) high-bit-depth — requires widening
`build_extended_forward_lut`'s reconstructed-level table from
`(1 << bc).min(257)` to the full `1 << bc` so the per-pixel inverse
LUT can be built at `bc ∈ 9..=16` (at `bc = 16` that's a 65 536-entry
`Vec<Option<u32>>`, ~512 KB peak — well within bounds for a single
encode but worth measuring). Star-Tetrix (`Cpih = 3`) high-bit-depth
also stays 8-bit-input specific. 4:2:0 chroma at `NL,y >= 3` (the
round-174 deferred case, the deeper β-enumeration mismatch) is
unrelated and still pending.

## Unreleased — round 174 (4:2:0 chroma decoder packet-layout fix at NL,y >= 2)

Fixes a latent decoder bug where 4:2:0 sub-sampled chroma at `NL,y >= 2`
failed with `InvalidData("jpegxs entropy: read_bit past end of buffer")`.
The decoder's `compute_packet_layouts` proxy-level and first-packet loops
applied the Annex B.7 Table B.4 sub-sampling guard `(λ + L0) umod sy[i]
== 0` and the line-in-precinct check `λ + L0 < L1` against the per-
component **band-grid** `L0[p,b]` / `L1[p,b]` stored in the slice walker,
but the spec defines both checks in **image-grid** units (Annex B.6:
`L0[p,b] = 2^max(NL,y - dy[i,β], 0) · τy[β]`, where `NL,y` is picture-
level and `dy[i,β]` is per-component). For chroma at `sy=2`, image-grid
`L0` is `2×` the band-grid value; the band-grid mod check then dropped
valid τy=1 chroma packets that the encoder had emitted, producing
fewer decoded packets than written → buffer underflow on subsequent
packets in the same slice.

* `src/slice_walker.rs` — `compute_packet_layouts` first-packet and
  proxy-level loops now compute `L0_image = L0_band × sy[i]` and
  `L1_image = L1_band × sy[i]` and apply the Table B.4 checks in image-
  grid coordinates, converting back to band-grid for the
  `PacketEntry.line` field so consumers (`packet_body::decode_packet_body`,
  `entropy::*`) that index by `entry.line - band.l0` keep their semantics.
  The 4:4:4 / 4:2:2 paths are unaffected (`sy[i] = 1` makes image-grid
  equal to band-grid).

Tests landed (294 total, +6 vs round 151): 4:2:0 NL=2/2 lossless round-
trip bit-exact (previously crashed); 4:2:2 NL=2/2 control; 4:2:0 NL=3/2
asymmetric (chroma N'L,y=1) lossless bit-exact; 4:4:4 NL=3/3 and 4:2:2
NL=3/3 baselines; 4:2:0 NL=2/2 q=2 lossy PSNR >= 20 dB on a synthetic
high-frequency picture.

Follow-up (deferred to a later round): 4:2:0 at `NL,y >= 3` (chroma's
N'L,y >= 2) still produces value-corrupted output rather than crashing.
The root cause is a deeper structural mismatch between the codec's non-
spec internal β-enumeration convention (chroma's bands packed
contiguously at picture-β slots `0..nbeta_chroma` using chroma's
effective NL for the β formula) and the spec's β-enumeration (chroma
bands at picture-β values computed with **picture** NL,x/NL,y, leaving
gaps where chroma has no equivalent band). The luma-driven outer-loop
lambda count under-iterates chroma's deeper-level proxy clusters in the
non-spec convention, so chroma packets for the second row of a 2-row
band are never coded. Repairing this requires reconciling the band
indexing across encoder + decoder + DWT cascade and is out of scope for
this round's fix.

## Unreleased — round 151 (high bit depth + chroma sub-sampling)

Widens the round-118 / round-133 high-bit-depth (`B[i] ∈ 9..=16`) paths
from 4:4:4-only to arbitrary per-component `(sx, sy) ∈ {1, 2}` sub-
sampling. The existing inner pipeline (`encode_planar_inner_bd`) already
threaded both `bd` and `(sx[i], sy[i])` independently, so this round
exposes two new entry points that wrap it for the high-bit-depth 4:2:2 /
4:2:0 case.

* `oxideav_jpegxs::encoder::encode_planar_subsampled_highbd(width,
  height, nc, cpih, nlx, nly, bd, sx, sy, &[Vec<u16>])` — lossless
  high-bit-depth chroma-sub-sampled entry point. `bd ∈ 9..=16` codes the
  picture with `Bw = B[i] = bd`, DC level shift `1 << (bd − 1)` per the
  Annex G.3 inverse, and `Fq = 0` (lossless, Table A.8). Per-component
  `(sx[i], sy[i]) ∈ {1, 2}` sub-sampling uses the existing 8-bit
  per-component effective vertical decomposition depth `N'L,y[i] = NL,y
  − log2(sy[i])` path. `Cpih ∈ {0, 1}` — per Annex F.2 Table F.1 the
  reversible RCT requires `sx[i] = sy[i] = 1` for `i < 3`, so the
  typical 4:2:2 / 4:2:0 configurations are exposed only with `Cpih = 0`.
  Star-Tetrix (`Cpih = 3`) and NLT pre-distortion stay 8-bit-input
  specific.
* `oxideav_jpegxs::encoder::encode_planar_subsampled_highbd_lossy(width,
  height, nc, cpih, nlx, nly, bd, q, sx, sy, &[Vec<u16>])` — the
  `q > 0` companion. Same plane format and constraints, but with
  `q ∈ 1..=15` and `Fq = 8` (regular mode, Table A.8 — required for
  `q > 0`). The per-band deadzone truncation `T[p,b] = clamp(Q − G[b],
  0, 15)` (Annex D.4 Table D.3) drops low magnitude bitplanes; the
  decoder reconstructs with the matching deadzone inverse (Annex D.2).
  Bit depth is orthogonal to quantization (both run on `i32` wavelet
  coefficients), so the only bit-depth-dependent pieces remain the
  level shift and `u16` packing.

Tests landed (288 total, +9 vs round 143): 10-bit 4:2:2 + 12-bit 4:2:0
+ 16-bit 4:2:2 lossless self-roundtrip bit-exact; 4:2:0 codestream
strictly smaller than 4:4:4 of the same luma; q=2 lossy stream strictly
smaller than q=0 lossless; PSNR ≥ 40 dB at 10-bit 4:2:2 q=1 and ≥ 30 dB
at 12-bit 4:2:0 q=2; full rejection sweep (`bd = 8`, `bd = 17`, `cpih =
3`, `q = 0` on the lossy entry point, RCT with chroma sub-sampling,
mismatched plane sample counts, samples above `2^bd − 1`).

## Unreleased — round 143 (Part-2 profile / level / sublevel conformance)

Adds a normative parse + conformance-check surface for ISO/IEC 21122-2:2019
Annex A (profiles, levels and sublevels). The decoder has carried the
underlying `ppih` / `plev` fields since round 1; this round names them and
checks them.

* **New `profile` module** — exposed at the crate root as
  `oxideav_jpegxs::{Profile, Level, Sublevel, ProfileLimits, ChromaFormat,
  ColumnMode, QpihAllowed, classify_chroma, column_width, check_profile,
  check_level}`.
* **Profile ↔ `Ppih` mapping** per Table A.5 — `Profile::from_ppih(u16) ->
  Option<Profile>` / `Profile::ppih(self) -> u16` for the nine documented
  rows (`Unrestricted`, `Light{422.10,444.12}`, `Light-Subline 422.10`,
  `Main{422.10,444.12,4444.12}`, `High{444.12,4444.12}`). Reserved values
  return `None`.
* **Level ↔ `Plev` high byte** per Table A.12 — `Level::from_plev_high` for
  `{2k-1, 4k-1, 4k-2, 4k-3, 8k-1, 8k-2, 8k-3, 10k-1}` plus `Unrestricted`,
  with `Level::{max_width, max_height, max_samples}` per Table A.6.
* **Sublevel ↔ `Plev` low byte** per Table A.13 —
  `Sublevel::from_plev_low_byte` for `{Full, Sublev12bpp, Sublev9bpp,
  Sublev6bpp, Sublev3bpp}` plus `Unrestricted`, with
  `Sublevel::nominal_bpp` per Table A.7.
* **`ProfileLimits` constraint set** — per-profile static table encoding
  the Tables A.1 / A.2 / A.3 columns (`allowed_bit_depths`,
  `allowed_chroma`, `nlx_range`, `max_nly`, `allowed_qpih`, `column_mode`,
  `max_column_width`, `slice_height`, `max_components`). `Light` family
  permits `Qpih = 0` only (DZQ); `Main` / `High` / `Light-Subline` permit
  `Qpih ∈ {0, 1}`. `High` family raises `NL,y` to 2; `Light-Subline` caps
  `Cs ≤ 2048` per formula A.2.
* **`check_profile(&Codestream, Profile)`** enforces every observable
  constraint: `Nc ≤ max_components`, per-component `B[i] ∈
  allowed_bit_depths`, chroma format (classified from CDT `(sx, sy)`
  factors) `∈ allowed_chroma`, `NL,x ∈ nlx_range`, `NL,y ≤ max_nly`,
  `NL,x ≥ NL,y` (Table A.1 footnote c), `Qpih` ∈ allowed set, slice height
  `Hsl × 2^NL,y == 16` image rows, and column-mode rules including the
  `Cs ≤ 2048` Light-Subline cap (formula A.3). `Unrestricted` always
  succeeds (§A.2.2 — not a conformance point).
* **`check_level(&Codestream)`** enforces level-derived `Wmax` / `Hmax` /
  `Lmax` bounds (Table A.6) against `Wf` / `Hf`.
* **Buffer-model bounds (Annexes B / C / D) are explicitly out of scope**
  for this module — they require a transmission-channel rate that is a
  host-side quantity, not a codestream-observable property.
* **15 new unit tests** — `Ppih` round-trip across all 9 profile rows,
  chroma classification for mono / 4:2:2 / 4:2:0 / 4:4:4 / 4:2:2:4 /
  4:4:4:4, profile-conformance acceptance + rejection for every documented
  constraint (Light + 12-bit reject, Light + uniform Qpih reject, Light +
  4:4:4 reject, Main + `Cw > 0` with `NL,y != 0` reject, Light-Subline
  `Cs ≤ 2048` cap, High `NL,y = 2` accept + Main reject, Main4444 4-comp
  accept + Main444 reject, Unrestricted permissive), Level / Sublevel
  decoder spot-checks, level `Wmax` and `Lmax` enforcement.

No encoder / decoder hot-path bytes change in this round. The new
module is read-only against the existing `Codestream` and
`ComponentTable` parser surfaces.

## Unreleased — round 133 (encoder high-bit-depth lossy, `B[i] > 8` + `q > 0`)

Adds the lossy companion to the round-118 high-bit-depth (`B[i] ∈ 9..=16`)
encoder path. Round 118 coded high-bit-depth pictures losslessly only
(`q = 0`, `Fq = 0`); the "Out of scope (next round)" note flagged lossy
`q > 0` as the natural follow-on. Bit depth is orthogonal to quantization
— the forward quantizer (`forward_quant_index`) and the inverse
dequantizer both operate on `i32` wavelet coefficients regardless of
`B[i]`, so the only bit-depth-dependent pieces remain the DC level shift
(`1 << (bd − 1)`, Annex G.3 inverse) and the two-bytes-per-sample `u16`-LE
plane packing — already in place since round 118. This round just exposes
the path.

* `encoder.rs` — **new public `encode_planar_highbd_lossy(width, height,
  nc, cpih, nlx, nly, bd, q, planes)`** entry point. Same plane format as
  `encode_planar_highbd` (little-endian `u16` samples in `0..=2^bd − 1`,
  `Bw = B[i] = bd`) but threads a non-zero precinct quantization step
  `q ∈ 1..=15` (Annex C.2 `Q[p]`) with `Fq = 8` (regular mode, Table A.8).
  The per-band deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)`
  (Annex D.4 Table D.3) drops low bitplanes; the decoder reconstructs with
  the matching deadzone inverse (Annex D.2, `Qpih = 0`).
* `cpih ∈ {0, 1}` (no transform / reversible RCT). `q = 0` is rejected
  (use `encode_planar_highbd` for lossless); `bd = 8` and `cpih = 3`
  (Star-Tetrix) are rejected, mirroring the lossless entry point.
* **Tests** — 10-bit luma q=1 ≥ 40 dB PSNR; 12-bit luma q=2 NL=3/3 ≥ 30
  dB; 16-bit RGB+RCT q=1 ≥ 40 dB per component; q=2 produces a strictly
  smaller stream than the lossless q=0 path (quantizer bites); reject
  cases for q=0 / bd=8 / Cpih=3.

## Unreleased — round 115 (encoder `R[p] > 0` precinct refinement)

Adds encoder support for the precinct-refinement parameter `R[p]` per
ISO/IEC 21122-1:2022 Annex C.2 Table C.1 (precinct-header field) + Annex
A.4.11 Table A.24 (WGT band priorities `P[b]`) + Annex C.6.2 Table C.10
(truncation-position algorithm). The decoder has computed the refined
truncation since the early rounds (`entropy::truncation_position` already
implements `r = (P[b] < R[p]) ? 1 : 0`, `T[p,b] = clamp(Q − G[b] − r, 0,
15)`), but the encoder always emitted `R[p] = 0` and `P[b] = 0`, so the
refinement term `r` was permanently 0. This round wires the encoder side.

* `encoder.rs` — **new public `encode_planar_rp(width, height, nc, cpih,
  nlx, nly, q, rp, planes)`** entry point. `rp = 0` is the no-refinement
  default (byte-identical to `encode_planar_lossy`); `rp > 0` activates
  the Annex C.6.2 refinement.
* **WGT now emits per-band priorities `P[b] = b`** (the true band index,
  Annex B.6 `b = (Nc − Sd)×β + i`, or `(Nc − Sd)×Nβ + i` for the Sd
  tail), built by the new `build_band_priorities_sd` in the same
  existing-band emission order as `build_band_gains_sd`. With this
  assignment a refinement `R[p] = k` refines exactly the `k` lowest-index
  bands — LL first, since bands are enumerated in `β`-major order. This
  replaces the all-zero `P[b]` rounds 1–111 emitted. The spec permits an
  encoder-chosen priority ("Other choices are possible", Annex H NOTE);
  the only invariant is that the WGT `P[b]` and precinct-header `R[p]`
  reproduce the same `T[p,b]` the encoder quantized with, which the
  decoder's `precinct_truncation` reads back identically.
* **The precinct header `R[p]` byte** (offset 4) now carries `cfg.rp` in
  both `encode_precinct_single_level` and `encode_precinct_cascade`
  (previously hard-zero).
* **Truncation in both precinct encoders** now applies the refinement:
  `T[p,b] = clamp(Q − G[b] − r, 0, 15)`, `r = (b < R[p]) ? 1 : 0`. The
  cascade path computes the band index for both wavelet bands and the Sd
  suppressed tail; the single-level path uses `b = Nc×β + i`.
* **`EncodeConfig`** gains `rp: u8` and `band_priorities: Vec<u8>`
  fields, range-checked in `validate()`: `R[p] ∈ 0..=NL−1` where
  `NL = (Nc − Sd)×Nβ + Sd` (Annex B.6). Out-of-range `R[p]` is rejected.
* **At `q = 0`** the refinement is a no-op (T is already clamped to its 0
  floor), so a lossless stream round-trips unchanged for any `rp`. The
  refinement only shifts bits toward the refined low-frequency bands in
  the lossy (`q > 0`, `Fq = 8`) path.
* **Tests** — 7 new (`round115_*`): WGT priorities equal band indices,
  `rp = 0` byte-identical to `encode_planar_lossy` at q∈{0,2,4}, `rp > 0`
  lossless at q=0 across the full `R[p]` range, `rp > 0` lossy q=2
  round-trip + PSNR floor, `rp > 0` changing the lossy stream vs `rp = 0`
  (proving the refinement fires), RGB+RCT round-trip at q=0/q=2, and the
  out-of-range `R[p]` rejection. 240 → 247 tests.

## Unreleased — round 111 (`Qpih`-aware forward quantizer)

Adds a `Qpih`-selected forward quantizer so a `Qpih = 1` codestream is
now encoded with the spec's uniform (round-to-nearest) quantization
index instead of the deadzone (floored) one. Round 108 signalled
`Qpih = 1` in the PIH but the encoder still picked the quantization
index by deadzone truncation (`v = |c| >> T`, Annex D.4 Table D.3) — a
valid stream, but it leaves the bottom of each uniform bucket as the
implied reconstruction target, so lossy magnitudes were biased low. The
uniform inverse (Annex D.3 Table D.2) was already wired on the decode
side and expects the matching uniform forward index (Annex D.5 Table
D.4).

* `encoder.rs` — **new `forward_quant_index(qpih, c, m, t)`** returns the
  on-wire quantization index `v[p,λ,b,x]` for the active quantizer:
  * `Qpih = 0` (deadzone, Table D.3): `v = |c| >> T` — unchanged from the
    round-3 path; bit-identical output.
  * `Qpih = 1` (uniform, Table D.4): with `ζ = M − T + 1` and `d = |c|`,
    `v = ((d << ζ) − d + (1 << M)) >> (M + 1)` — round-to-nearest into
    equal-sized buckets (`Δ = 2^(M+1) / (2^(M+1−T) − 1)`). Returns 0 when
    `M ≤ T` (no stored bitplanes). Computed in `u64` so `d << ζ` cannot
    overflow.
* **The data sub-packet** now writes the `M − T` bits of `v` (MSB-first)
  rather than reusing `|c|`'s bits `[T, M)`. For `Qpih = 0` the two are
  identical (`v`'s bits `[0, M−T)` equal `|c|`'s bits `[T, M)`), so every
  deadzone stream stays byte-for-byte the same. For `Qpih = 1` the data
  sub-packet now diverges from the deadzone form at `q > 0`.
* **The `Fs = 1` sign sub-packet** gating predicate now tests the active
  forward index `v != 0` (matching the decoder's `coef.v != 0`), keeping
  the uniform path consistent with separate-sign streams.
* **At `q = 0` (`T = 0`)** the uniform forward index is the identity
  (`v = |c|`), so all round-108 lossless / one-byte-diff invariants hold.
* **Tests** — 6 new (`round111_*`): `forward_quant_index` deadzone +
  uniform unit cases hand-checked against Tables D.3 / D.4, the `T = 0`
  identity, the `Qpih = 1` lossy data sub-packet now diverging from
  `Qpih = 0`, a uniform-path PSNR floor at q=3, and `Qpih = 1` + `Fs = 1`
  lossless round-trip. 234 → 240 tests.

## Unreleased — round 108 (encoder `Qpih = 1` uniform inverse quantizer)

Adds encoder support for the `Qpih = 1` inverse-quantizer type per
ISO/IEC 21122-1:2022 Annex A.4.4 Table A.10 + Annex D.3 Table D.2. The
decoder has supported `Qpih = 1` (the uniform / Neumann-series inverse
quantizer) end-to-end since the early rounds (`pih.qpih` threads into
`dequant::dequantize_precinct`, which dispatches `inverse_uniform`; the
decoder rejects `Qpih > 1`); the encoder previously hard-coded
`Qpih = 0` (the deadzone quantizer, Annex D.2). This round closes that
gap on the encode side.

* `encoder.rs` — **`EncodeConfig` gains a `qpih: u8` field** (default 0),
  validated as `0` (deadzone) or `1` (uniform); reserved values 2/3 are
  rejected to mirror the decoder's `Qpih > 1` rejection. `write_pih_body`
  now emits `(qpih & 0x03) << 4` into the `Lh:Rl:Qpih:Fs:Rm` byte
  (Qpih is bits 5:4) instead of leaving the field at 0.
* **The data sub-packet is unchanged.** Both quantizer types read the
  identical magnitude bitplanes off the wire — only the decoder's
  reconstruction kernel differs (deadzone midpoint `(v << T) + r` vs the
  uniform Neumann series). The encoder therefore only signals the bit;
  no forward-quantization change is needed. At `q = 0` (`T = 0`) both
  kernels reconstruct exactly (the Neumann series collapses to `v`
  because the stored magnitude satisfies `v < 2^M`), so `Qpih = 1`
  self-roundtrips losslessly and decodes byte-identically to the
  `Qpih = 0` form (the two codestreams differ in exactly one byte — the
  PIH quantizer-type byte). At `q > 0` the two kernels reconstruct
  different (both valid) lossy magnitudes.
* **New public entry point `encode_planar_qpih(width, height, nc, cpih,
  nlx, nly, q, planes)`** — same shape as `encode_planar_lossy` but sets
  `Qpih = 1`.
* **Tests** — 5 new (`round108_*`): Qpih=1 luma lossless + PIH
  `qpih == 1` assertion, Qpih=1 RGB+RCT (Cpih=1) lossless, Qpih=1 lossy
  q=2 PSNR ≥ 30 dB floor, Qpih=1-vs-Qpih=0 lossless exactly-one-byte
  diff (the PIH bit-4 toggle) + identical decode, and reserved-`Qpih`
  rejection. 234 total (was 229); 0 ignored.

## Unreleased — round 103 (encoder multi-slice emission, `Hsl > 0`)

Adds encoder support for multi-slice codestreams per ISO/IEC
21122-1:2022 Annex B.10 (grouping of precincts into slices) + Annex
A.4.12 Table A.25 (slice header). The decoder has reconstructed the
precinct-to-slice grouping from the PIH `Hsl` field since the early
rounds (`slice_walker::build_plan` walks `⌈Np,y / Hsl⌉` slices, and the
codestream parser already loops over every SLH marker); the encoder
previously hard-coded a single slice spanning the whole picture
(`Hsl = Np,y`, one SLH with `Yslh = 0`). This round closes that gap on
the encode side.

* `encoder.rs` — **`EncodeConfig` gains an `hsl: u16` field** (default
  0 = single slice). `validate` rejects `Hsl > Np,y` (a value that
  would describe a slice running past the last precinct row, where
  `Np,y = ⌈Hf / 2^NL,y⌉`). A new `effective_hsl` helper resolves the
  on-wire `Hsl` field (`Np,y` when `cfg.hsl == 0`, else the caller's
  value); `write_pih_body` emits it instead of the unconditional
  `Np,y`.
* `write_slice` — **emits one SLH marker per slice**. The two precinct
  loops (multi-level cascade and NL=1/1 streaming) are now wrapped in a
  slice partition: precinct rows are grouped into runs of `hsl_rows`
  rows (the last run is shorter when `Np,y` is not a multiple of
  `Hsl`), and each run is preceded by `SLH | Lslh=4 | Yslh=t` with `t`
  the top-down slice order. The per-precinct coding is unchanged —
  vertical prediction is already precinct-scoped in this encoder (the
  M-top cache in `encode_precinct_cascade` is local to one precinct),
  so the Annex B.10 requirement that vertical prediction be disabled
  across slice boundaries is satisfied trivially (no predictor crosses
  a precinct row in the first place).
* **New public entry point `encode_planar_hsl(width, height, nc, cpih,
  nlx, nly, q, hsl, planes)`** — same shape as `encode_planar_lossy`
  with an explicit slice height. `hsl = 0` is byte-identical to
  `encode_planar_lossy` (single slice).
* **Tests** — 6 new (`round103_*`): Hsl=2 luma 4-slice lossless +
  slice-count + PIH `Hsl` assertions, Hsl=3 RGB+RCT 2-slice lossless,
  Hsl=2 lossy q=2 PSNR ≥ 30 dB floor, `hsl=0` / `hsl=Np,y`
  single-slice byte-equivalence to `encode_planar_lossy`,
  non-divisible `Np,y` last-slice-shorter (2,2,1) roundtrip, and
  `Hsl > Np,y` rejection. 229 total (was 223); 0 ignored.

## Unreleased — round 100 (encoder `Fs = 1` separate sign sub-packet)

Adds encoder support for the `Fs = 1` sign-handling strategy per
ISO/IEC 21122-1:2022 Annex A.4.4 Table A.11 + Annex C.4 + Annex C.5.5
Table C.9. The decoder has supported `Fs = 1` end-to-end since the
initial rounds (`pih.fs` threads slice walker → packet body, which reads
the separate sign sub-packet); the encoder previously hard-coded
`Fs = 0` (signs interleaved into the data sub-packet, Table C.8). This
round closes that gap on the encode side.

* `encoder.rs` — **`EncodeConfig` gains an `fs: u8` field** (default 0),
  validated as `0` (joint signs) or `1` (separate sign sub-packet);
  reserved values 2/3 are rejected. `write_pih_body` now emits
  `(fs & 0x03) << 2` into the `Lh:Rl:Qpih:Fs:Rm` byte instead of a
  hard-coded `0x00`.
* `build_packet_body_with_m` — **emits the sign sub-packet when
  `Fs = 1`**. The data sub-packet (Table C.8) drops its interleaved
  `Ng`-per-significant-group sign bits and carries magnitude bitplanes
  only; a new `sgn` bit-writer (Table C.9) walks bands → lines → groups
  → members and emits one sign bit per coefficient whose reconstructed
  magnitude is non-zero (gated on `(|v| >> T) != 0`, matching the
  decoder's `coef.v != 0` test after the magnitude planes `[T, M)` are
  laid down). The `sgn` writer is byte-aligned and routed into the
  packet header's existing `Lsgn` field. Works with both VLC (`Dr = 0`)
  and raw (`Dr = 1`) bitplane-count modes.
* **New public entry point `encode_planar_fs1(width, height, nc, cpih,
  nlx, nly, q, planes)`** — same shape as `encode_planar_lossy` but with
  `Fs = 1`. For sparse-sign content it is strictly more compact than the
  `Fs = 0` form (which spends `Ng = 4` sign bits on every significant
  code group regardless of how many coefficients are non-zero).
* **Tests** — 5 new (`round100_*`): Fs=1 luma lossless + PIH `fs == 1`
  assertion, Fs=1 RGB+RCT (Cpih=1) lossless, Fs=1 lossy q=2 PSNR ≥ 30 dB
  floor, Fs=1-vs-Fs=0 identical-decode + compactness on a sparse-sign
  image, and reserved-`Fs` rejection. 223 total (was 218); 0 ignored.

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

## [0.0.7](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.6...v0.0.7) - 2026-06-15

### Other

- round 309 — data-subpacket size Ldat[p,s] inference + consistency predicate (Annex C.5.4 Table C.8)
- precinct-length Lprc[p] consistency predicate (Annex C.2 Table C.1)
- vlc loss-of-synchronisation guard (Annex C.7.1 Table C.15)
- bitplane-count buffer-bound conformance predicate (Annex C.5.3.4 Table C.6)
- decode cross-precinct vertical prediction (Annex C.6.3 Table C.11)
- decoder round 282: §B.1 odd-dimension sub-sampled planes + significance-coding conformance
- decoder round 273: typed Codestream::com accessor
- decoder round 266: typed Codestream::cwd accessor
- decoder round 254: typed Codestream::wgt accessor
- decoder round 251: typed Codestream::cts / crg / nlt accessors
- encoder round 245: rate-budget driven per-precinct (Q[p], R[p]) picker
- encoder round 242: joint per-precinct Q[p] × R[p] override
- scrub README enumerated-denial sentences
- encoder round 239: per-precinct R[p] override (Annex C.2 Table C.1)
- encoder round 233: per-precinct Q[p] override (Annex C.2 Table C.1)
- encoder round 230: high-bit-depth widening of the joint Q[p] + R[p] primitive
- encoder round 224: joint per-slice Q[p] + R[p] rate-budget picker
- round 218 — rate-budget driven R[p] picker
- round-212 rate-budget driven Q[p] picker for the multi-slice path
- encoder round 206: per-slice Q[p] override (slice-level rate budgeting)
- encoder round 201: high-bit-depth Star-Tetrix lossy (Cpih=3, q>0)
- encoder round 195: high-bit-depth Star-Tetrix (Cpih=3, B[i] in 9..=16)
- encoder round 193: high-bit-depth NLT extended (Tnlt=2, Annex G.5)

## [0.0.6](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.5...v0.0.6) - 2026-05-30

### Other

- picture-β slot indexing for 4:2:0 chroma (Annex B.3 Figure B.2)
- encoder round 181: high-bit-depth NLT quadratic (Tnlt=1, Annex G.4)
- fix 4:2:0 chroma decoder packet layout at NL,y >= 2
- encoder round 151: high bit depth + chroma sub-sampling (4:2:2 / 4:2:0)
- `Sublevel::from_plev_low` delegate to byte form (round 143 polish)
- round 143: Part-2 profile / level / sublevel conformance
- encoder round 133: high-bit-depth lossy (B[i] > 8, q > 0)
- report byte stride for high-bit-depth planes (round 118)
- encoder round 118: high bit depth B[i] in 9..=16 (lossless, Annex G.3)

## [0.0.5](https://github.com/OxideAV/oxideav-jpegxs/compare/v0.0.4...v0.0.5) - 2026-05-24

### Other

- encoder round 115: R[p] > 0 precinct refinement (Annex C.2 + C.6.2)
- Qpih-aware forward quantizer (round 111)
- encoder round 108: Qpih=1 uniform inverse quantizer (Annex A.4.4 Table A.10, Annex D.3)
- multi-slice emission (Hsl > 0, Annex B.10)
- encoder round 100: Fs=1 separate sign sub-packet (Annex C.5.5, Table C.9)
- round 95 / r93: Sd > 0 composes with Cpih ∈ {1, 3}
- round 9 / r91: Sd > 0 (CWD) decomposition suppression — encoder + decoder
- round 8: end-to-end Cw > 0 (multi-precinct-per-row)
- update stale comment referencing removed debug helper
- encoder round 7: extended NLT (Tnlt=2) + NL cap raised to 8

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
