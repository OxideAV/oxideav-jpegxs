# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Built clean-room from the ISO/IEC 21122 specification documents under
`docs/image/jpegxs/` only. Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

Both directions are **working for a substantial subset** of ISO/IEC
21122-1:2022 and self-roundtrip losslessly across the supported feature
matrix. JPEG XS has no inter-frame state, so each picture is independent.

### Decoder

End-to-end decode of the multi-component subset:

- `Nc ∈ {1, 2, 3, 4}` components; `(sx, sy) ∈ {1, 2}` per component
  (4:4:4 / 4:2:2 / 4:2:0), with §B.1 ceiling-sized planes so odd picture
  dimensions are legal.
- `Cw ≥ 0` precincts per row (`Cw = 0` single-precinct-per-row;
  `Cw > 0` splits each row into `⌈Wf / Cs⌉` precincts).
- `Cpih ∈ {0, 1, 3}` — no transform, reversible RGB↔YCbCr (Annex F.3),
  or Star-Tetrix (Annex F.5) for 4-component CFA images.
- `Qpih ∈ {0, 1}` deadzone / uniform inverse quantizer; `Fq ∈ {0, 8}`
  lossless / regular; `Bw ∈ {8, 18, 20}`; `B[i]` up to 16-bit.
- Multi-level wavelet cascade (`NL,x ≥ NL,y`), multi-slice (`Hsl ≥ 0`),
  precinct refinement (`R[p]`), per-precinct `Q[p]`, separate sign
  sub-packet (`Fs = 1`), and the entropy-decode loss-of-synchronisation
  guard.
- Annex G linear / quadratic / extended (NLT) output scaling.

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
`Ppoc` and `Ss ∉ 1..=8` (Annex A.4.4 Tables A.9–A.13); and `Cpih = 3`
Star-Tetrix with any sub-sampled CFA input (Annex A.4.3 / F.2, mirroring
the existing `Cpih = 1` RCT guard).

### Encoder

A planar encoder covering the same feature matrix, lossless (`q = 0`)
and lossy (`q ∈ 1..=15`), with rate-budget pickers that drive per-slice
`Q[p]` and per-precinct `(Q[p], R[p])` against a target byte budget.
8-bit and high-bit-depth (`B[i] ∈ 9..=16`, little-endian `u16` planes)
paths exist for all three colour-transform modes and both NLT modes.

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

Wiring up the CFA tables required correcting an over-strict `Cpih = 3`
constraint: the encoder and decoder previously rejected `Nc − Sd < 4`,
conflating the Star-Tetrix *input* window (`c < 4`, Annex F.2 Table F.1)
with output suppression. Per Annex A.4.7 + Tables B.10 / B.11, `Sd`
suppresses the wavelet decomposition of trailing transform *outputs*
(coded raw), which the inverse transform consumes unchanged — so the only
requirement is `Nc ≥ 4` (four transform inputs). RCT (`Cpih = 1`) keeps the
stricter `Nc − Sd ≥ 3` guard (no tabulated RCT-with-suppressed-output
example).

### Codestream parser

The marker-chain parser per ISO/IEC 21122-1:2022 Annex A recognises:

- `SOC` (`FF 10`), `EOC` (`FF 11`), `CAP` (`FF 50`, decoded into a
  typed `Capabilities` view), `PIH` (`FF 12`), `CDT` (`FF 13`),
  `WGT` (`FF 14`), `NLT` (`FF 16`), `CTS` (`FF 18`), `CRG` (`FF 19`),
  `COM` / `CWD`, and `SLH` (`FF 20`).
- Each header marker has a typed body accessor (`cts()`, `crg()`,
  `nlt()`, `wgt()`, `cwd()`, `com()`) surfacing field-level errors.

### Profile / level surface

The `profile` module implements the ISO/IEC 21122-2:2019 Annex A
profile / level / sublevel tables: `Profile::from_ppih`,
`Level::from_plev_high`, `Sublevel::from_plev_low_byte`, and
`check_profile` / `check_level` enforcing every codestream-observable
constraint (component count, bit depth, chroma format, decomposition
depths, `Qpih`, slice-height, column-mode caps). Buffer-model bounds
(Annexes B/C/D) are out of scope — they require a transmission-channel
rate not observable from the codestream.

### Not yet covered

- Bit depths above 16 (would need a `u32` plane format).
- Content-adaptive WGT weights for the CFA star-tetrix layouts at bit
  depths above 8 — the H.9–H.11 tables are wired through
  `encode_planar_star_tetrix_annex_h` for the 8-bit CFA path; a
  high-bit-depth companion is a follow-up. All Annex H example tables are
  now transcribed: the 4:4:4 RCT tables (H.1–H.3,
  `encode_planar_lossy_annex_h`), the subsampled 4:2:2 / 4:2:0 tables
  (H.4–H.8, `encode_planar_subsampled_annex_h`, including the `-*`
  non-existent-band handling), and the CFA Star-Tetrix tables (H.9–H.11,
  `encode_planar_star_tetrix_annex_h`, `Cf = 0` / `Cf = 3` columns).

## Public API

```rust
// Probe: width / height / components / bit depth / profile / level /
// Cpih / lossless flag.
let info = oxideav_jpegxs::probe(bytes);

// Decode a codestream.
let picture = oxideav_jpegxs::decode_jpeg_xs(bytes)?;
# Ok::<(), oxideav_jpegxs::Error>(())
```

Encoder entry points (in `oxideav_jpegxs::encoder`) cover single-luma,
interleaved RGB, and generalised planar input, with `_lossy`,
`_lossy_annex_h`, `_subsampled_annex_h`, `_star_tetrix_annex_h`, `_highbd`,
`_subsampled`, `_star_tetrix`, `_nlt_quadratic`, `_nlt_extended`,
`_hsl_qslice`,
`_qpr_rpr`, and `*_target_bytes` variants for the feature axes above. See
the module docs for the exact signatures and scope per entry point.

The crate also registers a software decoder through the standard
`oxideav-core` registry path.

## License

MIT — see [LICENSE](LICENSE).
