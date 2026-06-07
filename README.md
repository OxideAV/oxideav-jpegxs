# oxideav-jpegxs

Pure-Rust **JPEG XS** — ISO/IEC 21122 low-latency image codec for
production / IP video (SMPTE ST 2110-22, AES67-style live workflows).
Zero C dependencies, zero FFI, zero `*-sys`.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

| Direction | Status |
| --- | --- |
| Decoder | working — multi-component, **multi-precinct-per-row (Cw ≥ 0)** subset (rounds 1–8) + **Sd > 0 (CWD, Annex A.4.7)** decomposition suppression (round 9) + **high bit depth B[i] ∈ 8..=16** (round 118 — Annex G.3 linear output packs `u16` little-endian per `JpegXsPlane` when `B[i] > 8`) + **4:2:0 chroma packet-layout fix (round 174 — image-grid `L0`/`L1` per Annex B.6 in Table B.4 guards at `NL,y ≥ 2`)** + **picture-β slot indexing for 4:2:0 chroma (round 190, Annex B.3 Figure B.2)** + **typed `Codestream::cts` / `crg` / `nlt` accessors (round 251)** — `cts() -> Result<Option<CtsMarker>>` (Annex A.4.8 Tables A.19 / A.20), `crg() -> Result<Option<CrgMarker>>` (Annex A.4.9 Table A.21, component count taken from `pih.nc`), `nlt() -> Result<Option<NltParams>>` (Annex A.4.6 Table A.16). Each accessor mirrors the round-15 `capabilities()` pattern: borrow `self.{cts,crg,nlt}.as_deref()`, run the existing body-level parser, surface field-level errors as `Err(_)`. `NltParams` / `parse_nlt` are now re-exported from the crate root (previously only reachable through `output::`). The decoder routes the NLT body and the Star-Tetrix Cpih=3 CTS / CRG lookups through these accessors instead of re-parsing the raw bytes, eliminating the double-decode pattern. Decoder rounds 1–9 / 118 / 174 / 190 — the walker / encoder / decoder now share a `picture_beta_to_local_beta(β_pic, NL,x, NL,y, sy[i])` permutation that places chroma's DWT output at the picture-β slots the spec defines (with gaps at `β = NL,x + 2 − NL,y` and `β = NL,x + 3 − NL,y` per Annex B.4 / NOTE 3), fixing the value-corruption at `NL,y ≥ 3` 4:2:0 where the prior per-component packed β convention conflated luma's and chroma's spectral bands at the same slot id. 4:2:2 / 4:4:4 unaffected (sy=1 → identity mapping). |
| Encoder | Round 245 — **rate-budget driven per-precinct `(Q[p], R[p])` picker** via `pick_qpr_rpr_for_target_bytes(.., target_bytes, planes) -> Result<(Vec<u8>, Vec<u8>)>` (returns the picked `(q_precincts, r_precincts)` pair) + `encode_planar_qpr_rpr_target_bytes(.., target_bytes, planes) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)>` (convenience: returns `(codestream, q_precincts, r_precincts)`). Closes the round-242 "caller must pick the vectors manually" follow-up tail: round 242 shipped the per-precinct joint override `encode_planar_qpr_rpr` (one `Q[p]` and one `R[p]` per precinct, Annex C.2 Table C.1) and round 245 supplies the picker that drives it against a byte budget. Nested search — outer loop on uniform `rp ∈ 0..=NL-1` (`NL = Nc × Nβ`, Annex B.6), inner three-pass per-precinct `q_precincts` search against `encode_planar_qpr_rpr` at the current `rp`: (1) lossless probe at `q_precincts = [0; n]`, (2) uniform-`Q` bisect on `1..=15`, (3) per-precinct activity-driven relaxation walking the lowest-activity precincts first (L1 norm of row-to-row first-difference summed across every plane within each precinct's image-row range — `Hp = 2^NL,y` image rows per precinct from Annex B.5, sliced by the `compute_precinct_row_ranges` helper that returns `Np,y` ranges) and dropping their `Q` one step at a time while each candidate still fits. Promotion stops as soon as the inner search at `rp+1` fails (refinement is monotone non-decreasing in codestream length at fixed `Q[p]`); lossless short-circuit canonicalises to `r_precincts = [0; n]` whenever the inner picker returns the all-zero `q_precincts` vector (refinement is a wire-only no-op at the `T[p,b]` 0 floor, so promoting `rp` adds zero rate-distortion value). Every measurement is a real `encode_planar_qpr_rpr` call — no internal model of the entropy coder, no oracle, no external library. Errors with `target_bytes unreachable; rp=0 Q=15 emits N bytes` when even `q_precincts = [15; n]` + `r_precincts = [0; n]` overshoots the budget; rejects `target_bytes == 0`. The round-218 / round-224 / round-230 pickers all worked at per-slice granularity (one `Q` per slice, one uniform `rp` for the picture); round 245 closes the granularity gap by giving each precinct its own `Q[p]` while keeping `R[p]` as a uniform picture-wide knob the outer loop walks. Composition behaviour: the wrapper output through `encode_planar_qpr_rpr` is byte-identical to a direct `encode_planar_qpr_rpr` invocation at the picker's chosen `(q_precincts, r_precincts)`; at a budget the lossless probe satisfies the picker returns `(q_precincts, r_precincts) = ([0; n], [0; n])` and the stream matches the lossless reference; PSNR ≥ 25 dB at a 90% lossless budget on the 32×32 luma XOR-ramp fixture. Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. +6 tests, 394 total. Round 242 — **joint per-precinct `Q[p] × R[p]` override** (the cross-product the round-233 / round-239 changelogs flagged as the next step) via `encode_planar_qpr_rpr(.., q_precincts: &[u8], r_precincts: &[u8], &[Vec<u8>]) -> Result<Vec<u8>>`. Round 233 lifted picture-level `q` to one `Q[p]` per precinct; round 239 lifted picture-level `R[p]` to one per precinct (Annex C.2 Table C.1 — `Q[p]` is precinct-header byte 3, `R[p]` is byte 4); round 242 is the public entry point that carries both vectors at the same time, so `R[p]` becomes an active rate-distortion lever where the round-239 `q = 0` pin had left it as a wire-only no-op. Annex C.6.2 Table C.10 truncation `T[p, b] = clamp(Q[p] − G[b] − r, 0, 15)` with `r = (P[b] < R[p]) ? 1 : 0` no longer floors at zero when `Q[p] > 0`, so per-precinct `R[p] > 0` grants one extra retained magnitude bitplane to the `R[p]` lowest-index bands — raising PSNR strictly above the same-`Q` R-off baseline on the alternating-`R = 3` pattern. Both vectors must have length `Np,y × Np,x` (each is empty-or-full; at least one must be non-empty — the both-empty path is `encode_planar`). Picture-level fallbacks pick `max(q_precincts)` / `max(r_precincts)` so a uniform vector reduces to a no-op via the existing `precinct_cfg_for` overlay. `q_precincts = [0; n] + r_precincts = R` is byte-identical to `encode_planar_rpr(R)`; both-zero is byte-identical to `encode_planar`. Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. +5 tests, 388 total. Round 239 — **per-precinct `R[p]` override** (Annex C.2 Table C.1 in its spec-natural form: `R[p]` is indexed by precinct `p`) via `encode_planar_rpr(.., r_precincts: &[u8], &[Vec<u8>]) -> Result<Vec<u8>>`. Lifts the round-115 picture-wide `R[p]` mechanism to one `R[p]` per precinct (raster scan order: the precinct at row `py`, column `px` lives at `py * Np,x + px`); length must equal `Np,y × Np,x` and each entry is in `0..=NL − 1` where `NL = Nc × Nβ` (Annex B.6 NL definition with `Sd = 0`). The decoder reads `R[p]` per precinct (`parse_precinct_header` + `precinct_truncation`) since round 115, so no decoder change is needed — the bitstream-wire impact is exactly the per-precinct `R` byte. This entry point pins picture-wide `q = 0` so the `T[p,b] = clamp(Q − G[b] − r, 0, 15)` clamp floors regardless of `r`; the lever exposes the wire-level `R[p]` byte without changing the data sub-packet bytes (lossless self-roundtrip at any `r_precincts`). A future `encode_planar_qpr_rpr` cross-product would carry both per-precinct vectors and engage `R[p]` as a rate-distortion lever at `q > 0`. Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. +3 tests, 383 total. Round 233 — **per-precinct `Q[p]` override** (Annex C.2 Table C.1 in its spec-natural form: `Q[p]` is indexed by precinct `p`) via `encode_planar_qpr(.., q_precincts: &[u8], &[Vec<u8>]) -> Result<Vec<u8>>`. Lifts the round-206 per-slice `Q[p]` mechanism to one `Q[p]` per precinct (raster scan order: the precinct at row `py`, column `px` lives at `py * Np,x + px`); length must equal `Np,y × Np,x` and each entry is in `0..=15`. `Fq` is auto-selected (`0` when every entry is `0`, else `8`). The decoder reads `Q[p]` per precinct (`parse_precinct_header` + `precinct_truncation`) since the early rounds, so no decoder change is needed — the bitstream-wire impact is exactly the per-precinct `Q` byte. Composition behaviour: `q_precincts = [0; n]` is byte-identical to `encode_planar` at the same geometry; `q_precincts = [q; n]` (every precinct same `Q[p]`) is byte-identical to `encode_planar_lossy` at that single `q`; mixed `q_precincts` keeps strictly more bits than `[15; n]` whenever any entry is `< 15` (lower `Q[p]` retains more magnitude bitplanes). Composes with the reversible RCT (`Cpih = 1`, Annex F.3 — bit-depth agnostic) and Star-Tetrix (`Cpih = 3`, Annex F.5 — `i32` integer linear combinations, Q-agnostic on the lifting); RGB+RCT and 4-component CFA self-roundtrip losslessly at `q_precincts = [0; n]`. Mixed `q_precincts` PSNR ≥ 30 dB at the 32×32 luma fixture with `Q[p] ∈ {0, 2, 4}`. Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. High-bit-depth widening, `Cw > 0` multi-column and `Hsl > 0` × per-precinct cross-product intersect on future rounds. +9 tests, 380 total. Round 230 — **high-bit-depth widening of the round-224 joint primitive** via `encode_planar_hsl_qslice_rp_highbd(.., bd, hsl, q_slices, rp, &[Vec<u16>]) -> Result<Vec<u8>>` + `pick_q_slices_rp_for_target_bytes_highbd(.., bd, hsl, target_bytes, &[Vec<u16>]) -> Result<(Vec<u8>, u8)>` + `encode_planar_hsl_qslice_rp_target_bytes_highbd(.., bd, hsl, target_bytes, &[Vec<u16>]) -> Result<(Vec<u8>, Vec<u8>, u8)>` — lifts the joint per-slice `Q[p]` + precinct refinement `R[p]` axes to `bd = B[i] ∈ 9..=16` against the `u16`-LE per-plane format inherited from rounds 118 / 133 / 151. The forward quantizer (Annex D.4) and the refinement term `r = (P[b] < R[p]) ? 1 : 0` (Annex C.6.2 Table C.10) both run on `i32` wavelet coefficients independent of `B[i]`, so the only bit-depth-dependent pieces remain the DC level shift `1 << (bd − 1)` (Annex G.3 inverse) and the two-bytes-per-sample plane packing — the `encode_planar_inner_bd` plumbing already threaded `(hsl, q_slices, rp)` through to the precinct emitter at r118+. The high-bit-depth picker uses a `u16`-aware slice-activity metric so the per-slice relaxation pass reflects the original sample magnitudes rather than the low-byte / high-byte interleave of a `to_le_bytes()` packing. At `q_slices = [0; n]` refinement is a lossless no-op (`T[p,b]` already at its `0` floor) so the codestream self-roundtrips bit-exactly at any `(rp, bd)`; at `rp = 0` single-slice the bytes are identical to `encode_planar_highbd_lossy` at the matching `q`. Composes with the reversible RCT (`Cpih ∈ {0, 1}`, Annex F.3 bit-depth-agnostic operand window `c < 3`). 10-bit luma q=1 PSNR ≥ 40 dB and 12-bit mixed `Q = [0, 3]` PSNR ≥ 30 dB on the synthetic ramp. Scope: 4:4:4, `Cpih ∈ {0, 1}`, `Cw = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `bd ∈ 9..=16`. Star-Tetrix (`Cpih = 3`) and NLT pre-distortion intersect the joint primitive on a future round. +12 tests, 371 total. Round 224 — **joint per-slice `Q[p]` + precinct refinement `R[p]` rate-budget picker** via `pick_q_slices_rp_for_target_bytes(.., hsl, target_bytes, planes) -> Result<(Vec<u8>, u8)>` (returns `(q_slices, rp)`) + `encode_planar_hsl_qslice_rp_target_bytes(.., hsl, target_bytes, planes) -> Result<(Vec<u8>, Vec<u8>, u8)>` (convenience: returns `(codestream, q_slices, rp)`) + the underlying primitive `encode_planar_hsl_qslice_rp(.., hsl, q_slices, rp, planes)` composing round-206 per-slice `Q[p]` with round-115 precinct refinement on a single encode. The two levers are orthogonal on the bitstream — per-slice `Q[p]` lives in each precinct's `Q` byte, `R[p]` lives in each precinct's `R` byte — and the Annex C.6.2 Table C.10 truncation `T[p,b] = clamp(Q[p] − G[b] − r, 0, 15)` with `r = (P[b] < R[p]) ? 1 : 0` is monotone non-increasing in `Q[p]` and monotone non-decreasing in `R[p]`, so the joint picker walks both axes in a nested search: outer loop on `rp` from `0` upward keeping the last fitting solution, inner loop reuses r212's three-pass `q_slices` strategy (lossless probe → uniform-`Q` bisect → activity-driven per-slice relaxation) against the joint primitive at the current `rp`. Promotion stops as soon as the inner search at `rp+1` fails (refinement is monotone non-decreasing in codestream length at fixed `Q[p]`, so higher `rp` cannot fit either). Baseline reachability: if even `rp = 0` + `Q = 15` overshoots, errors with `target_bytes unreachable; rp=0 Q=15 emits N bytes` (no choice of `(q_slices, rp)` can fit). `target_bytes == 0` rejected. Every measurement is a real `encode_planar_hsl_qslice_rp` call. Composition behaviour: all-equal `q_slices` + `rp = 0` byte-identical to `encode_planar_hsl_qslice` at that `q_slices`; single-entry + `hsl = 0` + `rp > 0` byte-identical to `encode_planar_rp` at the same `(q, rp)`; `q_slices = [0; n]` with any `rp` byte-identical to the lossless `rp = 0` stream (refinement is a no-op when `T` is already at its 0 floor). Closes the round-218 "joint picker" follow-up tail. Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. +10 tests, 359 total. Round 218 — **rate-budget driven `R[p]` picker for the precinct-refinement path** via `pick_rp_for_target_bytes(.., target_bytes, planes) -> Result<u8>` (returns the picked `R[p]`) + `encode_planar_rp_target_bytes(.., target_bytes, planes) -> Result<(Vec<u8>, u8)>` (convenience: returns `(codestream, rp)`). Linear scan from `R[p] = NL-1` down to `0`, returning the first `R[p]` whose `encode_planar_rp` codestream fits the budget — refinement is monotone non-decreasing in codestream length (each refined band gains one extra retained magnitude bitplane via the Annex C.6.2 Table C.10 term `r = (P[b] < R[p]) ? 1 : 0` lowering `T[p,b]`), so the first fit is also the largest fit. Step 1 probes `R[p] = 0` as the baseline; if even that overshoots, errors with `target_bytes unreachable; R[p]=0 emits N bytes` (the budget is unreachable by `R[p]` alone — lower `q` or `pick_q_slices_for_target_bytes` can still help). `target_bytes == 0` is rejected as a precondition. Every measurement is a real `encode_planar_rp` call. Complements r212's `pick_q_slices_for_target_bytes` (trades quantization strength **between slices** at fixed refinement); r218 trades refinement strength **between bands** at fixed quantization — together they cover both axes of the rate-distortion lever. Scope mirrors `encode_planar_rp` exactly: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih = 0`, `q ∈ 0..=15`. At `q = 0` refinement is a lossless no-op (the truncation is already at its 0 floor) so the picker returns `NL-1` and the codestream decodes losslessly. Closes the round-115 "PSNR-optimizing priority assignment" tail (the mechanism shipped in 115; the picker is the policy). +8 tests, 349 total. Round 212 — **rate-budget driven `Q[p]` picker for the multi-slice path** via `pick_q_slices_for_target_bytes(.., target_bytes, planes) -> Result<Vec<u8>>` (returns the picked `q_slices`) + `encode_planar_hsl_target_bytes(.., target_bytes, planes) -> Result<(Vec<u8>, Vec<u8>)>` (convenience: returns `(codestream, q_slices)`). Three-pass deterministic search — (1) lossless probe at `q = [0; n_slices]`, (2) uniform-`Q` binary search on `1..=15`, (3) per-slice relaxation walks the lowest-activity slices first (L1 norm of row-to-row first-difference summed across every plane within each slice's image-row range) and drops their `Q` one step at a time while the candidate still fits the budget. Every measurement is a real `encode_planar_hsl_qslice` call. Errors with `target_bytes unreachable; Q=15 emits N bytes` if even the most aggressive uniform-`Q` overshoots; rejects `target_bytes == 0`. Closes the round-206 "caller must pick `q_slices` manually" tail (the mechanism shipped in 206; the picker is the policy). Round 206 — **per-slice `Q[p]` override** (slice-level rate budgeting, Annex C.2 Table C.1) via `encode_planar_hsl_qslice(.., hsl, q_slices: &[u8], ..)` — lifts the round-103 multi-slice path's single picture-level `q` to one `Q[p]` value per slice (in top-down `Yslh` order). The decoder reads `Q[p]` per precinct (`parse_precinct_header` + `precinct_truncation`), so no decoder change is needed; the bitstream-wire impact is exactly the per-precinct `Q` byte inside each slice. All-equal `q_slices` is byte-identical to `encode_planar_hsl` at that `q`, single-entry + `hsl = 0` byte-identical to `encode_planar_lossy`; mixed `Q[p]` (e.g. `[0, 2, 4, 2]`) keeps more bits in the lower-Q slices than a constant-Q baseline (≥ 30 dB PSNR floor across the 4-slice 32×32 luma fixture). `q_slices.len()` must exactly match `⌈Np,y / max(hsl, 1)⌉` and each entry is in `0..=15`; `Fq` is auto-selected (0 when every slice is lossless, 8 otherwise). Round 201 — high-bit-depth **Star-Tetrix lossy (`Cpih = 3`, Annex F.5, `q > 0`)** at `B[i] ∈ 9..=16` via `encode_planar_star_tetrix_highbd_lossy` (`Bw = B[i] = bd`, `Fq = 8`, Annex D.4 deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)`; `u16`-LE input planes; 16×16 4-component CFA at NL=2/2 holds PSNR ≥ 40 dB at 10-bit q=1 + 16-bit Ct=1 GRBG q=1, ≥ 30 dB at 12-bit q=2; q=2 stream strictly smaller than the round-195 lossless q=0 stream on the same 12-bit fixture, +5 tests, 328 total). Closes the last `B[i] > 8` gap (the round-195 docstring's "follow-up round" tail): bit depth, quantization, and the Annex F.5 lifting are mutually orthogonal — `encode_planar_inner_bd` already dispatched to `forward_star_tetrix` (`&mut [i32]`) with `cpih == 3` regardless of `q`, and `forward_quant_index` already drove the deadzone truncation regardless of `B[i]`, so the round-201 widening is the high-bit-depth `u16`-LE plane plumbing that lifts the round-195 entry point's pinned `q = 0`. Round 195 — high-bit-depth **Star-Tetrix lossless (`Cpih = 3`, Annex F.5)** at `B[i] ∈ 9..=16` via `encode_planar_star_tetrix_highbd` (lossless, `Bw = B[i] = bd`, `Fq = 0`; `u16`-LE input planes; 16×16 4-component CFA at NL=2/2 self-roundtrips bit-exactly at 10/12/16-bit including Ct=1 GRBG + non-default e1/e2/Cf=3, +6 tests). Star-Tetrix's Annex F.5 lifting is a pure-`i32` integer linear combination so bit depth is fully orthogonal — the new entry point is plumbing around `encode_planar_inner_bd`'s existing `cpih = 3` dispatch (`forward_star_tetrix` was already operating on `i32` slices since round 4). DC level shift `1 << (bd − 1)` per the Annex G.3 inverse, identical to the round-118 / 133 / 151 RCT high-bit-depth paths. The CTS / CRG markers (Table F.9 RGGB/GRBG) survive on the high-bit-depth path identical to the 8-bit form — only the CDT `B[i]` byte and the PIH `Bw` byte change on the wire. Round 193 — high-bit-depth **NLT extended (`Tnlt = 2`, Annex G.5)** at `B[i] ∈ 9..=16`, `Bw = 20`, via `encode_planar_nlt_extended_highbd` (PSNR ≥ 30 dB at 10/12/16-bit q=0, ≥ 25 dB at 10-bit q=2, ≥ 30 dB at 10-bit RGB+RCT q=0). The reverse LUT inverter now allocates the full `2^B[i]` reconstructed-level table (the round-7 `.min(257)` cap was an 8-bit shortcut), and the `encode_planar_inner_bd` `bd>8` extended-NLT rejection is gone. Round 181 — high-bit-depth NLT quadratic (`Tnlt = 1`, Annex G.4) at `B[i] ∈ 9..=16`, `Bw = 20`, via `encode_planar_nlt_quadratic_highbd` (PSNR ≥ 40 dB at 10/12/16-bit q=0, ≥ 30 dB at 10-bit q=2, ≥ 35 dB at 10-bit RGB+RCT q=0). Round 151 — luma + RGB 4:4:4 / 4:2:2 / 4:2:0 + 4-component CFA Star-Tetrix, Cpih ∈ {0, 1, 3}, **high bit depth B[i] ∈ 9..=16 (lossless via `encode_planar_highbd`, Bw = B[i], linear path; lossy `q > 0` via `encode_planar_highbd_lossy`, round 133 — same `u16`-LE plane format, `Fq = 8`, deadzone truncation `T[p,b] = clamp(Q−G[b], 0, 15)`, PSNR ≥ 40 dB at 10-bit q=1 / 16-bit RGB+RCT q=1 and ≥ 30 dB at 12-bit q=2; and high-bit-depth chroma sub-sampling 4:2:2 / 4:2:0 via `encode_planar_subsampled_highbd` (lossless) + `encode_planar_subsampled_highbd_lossy` (round 151 — `u16`-LE per-component planes, `(sx, sy) ∈ {1, 2}` per Annex F.2 with `Cpih ∈ {0, 1}` — RCT keeps sx=sy=1 for i<3, 10-bit 4:2:2 / 12-bit 4:2:0 / 16-bit 4:2:2 self-roundtrip losslessly, 4:2:0 stream is strictly smaller than 4:4:4 of the same luma, PSNR ≥ 40 dB at 10-bit 4:2:2 q=1 and ≥ 30 dB at 12-bit 4:2:0 q=2)** — `u16`-LE input planes, DC level shift `1 << (B[i]−1)`, self-roundtrips bit-exactly at 10/12/16-bit luma and 16-bit RGB+RCT, **NL_x ∈ {1..=8} / NL_y ∈ {0..=NL_x}** (spec Annex A.4.4 Table A.7 hard max), **Cw ≥ 0** (`Cs = 8 × Cw × max(sx) × 2^NL,x` per Annex B.5, Np,x = ⌈Wf / Cs⌉ precincts per row), **multi-slice Hsl ≥ 0** (Annex B.10 — `Hsl > 0` groups the `Np,y` precinct rows into `⌈Np,y / Hsl⌉` slices, one SLH per slice with `Yslh = t`, via `encode_planar_hsl`; `Hsl = 0` is the single-slice default), **Sd ∈ 0..Nc-1 (CWD, Annex A.4.7) with Nc up to 8 when Sd>0, composes with Cpih ∈ {1, 3}** (Annex F.2 Table F.1: RCT operand window `c < 3`, Star-Tetrix operand window `c < 4`; encoder validates `Nc - Sd >= 3 / 4`), odd dims, Dr ∈ {0, 1} VLC + raw picker with no-prediction (Table C.14) **and vertical-prediction (Table C.13)** sub-modes, **Fs ∈ {0, 1} sign handling (Annex A.4.4 Table A.11)** — joint signs in the data sub-packet (Table C.8) or a **separate sign sub-packet (Annex C.5.5, Table C.9, one bit per non-zero coefficient)** via `encode_planar_fs1`, **Qpih ∈ {0, 1} inverse-quantizer type (Annex A.4.4 Table A.10)** — deadzone (Annex D.2) or **uniform / Neumann-series (Annex D.3)** via `encode_planar_qpih`, with a **`Qpih`-aware forward quantizer** (round 111): `Qpih = 0` uses the deadzone truncation `v = |c| >> T` (Annex D.4 Table D.3), `Qpih = 1` uses the uniform round-to-nearest index `v = ((|c| << ζ) − |c| + (1 << M)) >> (M+1)`, `ζ = M − T + 1` (Annex D.5 Table D.4); at `q = 0` both reduce to the lossless identity (deadzone stream byte-identical, uniform stream one-byte-diff vs deadzone) and at `q > 0` the uniform data sub-packet diverges, **precinct refinement Rp ∈ 0..=NL-1 (Annex C.2 Table C.1 + Annex C.6.2 Table C.10)** via `encode_planar_rp` — the WGT carries per-band priorities `P[b] = b` (band index, Annex B.6) and the precinct header carries `R[p]`, so `T[p,b] = clamp(Q − G[b] − r, 0, 15)` with `r = (P[b] < R[p]) ? 1 : 0` grants one extra retained bitplane to the `R[p]` lowest-index (LL-first) bands; `R[p] = 0` is the no-refinement default (byte-identical to `encode_planar_lossy`), at `q = 0` refinement is a lossless no-op, at `q > 0` it shifts bits toward the refined low-frequency bands, **significance coding (D[p,b] bit 1, Annex C.5)** gating zero significance groups, **per-band gain-weighted Q** (`T[p,b] = clamp(Q−G[b], 0, 15)`, G ∈ {0,1,2}), **NLT quadratic forward map** (Annex G.4, Tnlt=1, Bw=18) via `encode_planar_nlt_quadratic`, **NLT extended forward map** (Annex G.5, Tnlt=2, three-segment gamma, Bw=18) via `encode_planar_nlt_extended` with reverse LUT inverter, Fq ∈ {0, 8} lossy with Q ∈ 0..=15. Self-roundtrip ∞ dB lossless at NL=3/3, 4/4, 5/5, 6/6 and Sd=1 Nc=4 / Sd=2 Nc=5 lossless; **Sd=1 Nc=4 + RCT** and **Sd=2 Nc=5 + RCT** and **Sd=1 Nc=5 + Star-Tetrix** self-roundtrip losslessly; **Fs=1 luma + RGB+RCT self-roundtrip losslessly** (decodes byte-identical to the Fs=0 layout, no larger on sparse-sign content); **multi-slice Hsl=2 luma (4 slices) + Hsl=3 RGB+RCT (2 slices) + non-divisible Np,y (2,2,1) self-roundtrip losslessly**; **Qpih=1 luma + RGB+RCT self-roundtrip losslessly** (one-byte-diff vs Qpih=0, decodes identically); **Rp>0 luma + RGB+RCT self-roundtrip losslessly at q=0 across the full R[p] range** (rp=0 byte-identical to `encode_planar_lossy`, rp>0 changes the lossy q>0 stream — refinement fires); PSNR ≥ 40 dB at q=1, ≥ 30 dB at Sd=1 q=2 and at Fs=1 q=2 and at Hsl=2 q=2 and at Qpih=1 q=2 and at Rp=1 q=2, ≥ 25 dB at Sd=1+Cpih=1 q=2 and at q=4 and at Rp=NL-1 q=2; NLT extended PSNR ≥ 30 dB at q=0, ≥ 25 dB at q=2; Cw=1 64×16 luma at NL=1/1 and NL=2/2 + Cw=2 128×32 RGB+RCT NL=2/2 + Cw=1 4:2:2 round-trip bit-exact |

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

ISO/IEC 21122-2:2019 Annex A profile / level / sublevel surface
(round 143, `profile` module):

* `Profile::from_ppih(u16)` / `Profile::ppih(self) -> u16` — Table A.5
  bidirectional map across the nine documented `Ppih` rows
  (`Unrestricted`, `Light{422.10,444.12}`, `Light-Subline 422.10`,
  `Main{422.10,444.12,4444.12}`, `High{444.12,4444.12}`). Reserved
  values → `None`.
* `Level::from_plev_high(u16)` per Table A.12 (`{2k-1, 4k-1, 4k-2, 4k-3,
  8k-1, 8k-2, 8k-3, 10k-1}` + `Unrestricted`) with `max_width` /
  `max_height` / `max_samples` per Table A.6.
* `Sublevel::from_plev_low_byte(u8)` per Table A.13 (`{Full, Sublev12bpp,
  Sublev9bpp, Sublev6bpp, Sublev3bpp}` + `Unrestricted`) with
  `nominal_bpp` per Table A.7.
* `check_profile(&Codestream, Profile)` — enforces every observable
  constraint from Tables A.1 / A.2 / A.3: `Nc ≤ max_components`, per-
  component `B[i] ∈ allowed_bit_depths`, chroma format (classified from
  CDT `(sx, sy)`) ∈ `allowed_chroma`, `NL,x ∈ nlx_range`, `NL,y ≤
  max_nly`, `NL,x ≥ NL,y` (Table A.1 footnote c), `Qpih` ∈ allowed set,
  slice height `Hsl × 2^NL,y == 16` image rows, column-mode rules
  including the `Cs ≤ 2048` Light-Subline cap (formula A.3). Buffer-
  model bounds (Annexes B/C/D) are out of scope — they require a
  transmission-channel rate that isn't observable from the codestream.
* `check_level(&Codestream)` — enforces level `Wmax` / `Hmax` / `Lmax`
  bounds against `Wf` / `Hf`.

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
* `oxideav_jpegxs::encoder::encode_planar_highbd(width, height, nc, cpih,
  nlx, nly, bd, &[Vec<u16>]) -> Result<Vec<u8>>` — round-118 high-bit-
  depth lossless 4:4:4 entry point for `bd = B[i] ∈ 9..=16`. Codes the
  picture with `Bw = B[i] = bd` and `Fq = 0` (the lossless choice of
  Table A.8); the DC level shift is `1 << (bd − 1)` (Annex G.3 inverse)
  so each sample lands in the wavelet domain `[−2^(bd−1), 2^(bd−1) − 1]`.
  The 5/3 DWT, entropy coder, and reversible colour transform all run on
  `i32` coefficients independent of bit depth, so the only bit-depth-
  dependent pieces are the level shift and the output plane packing.
  `planes[i]` carries the component samples as **little-endian `u16`**
  values in `0..=2^bd − 1` (samples above that are an encoder error); the
  decoder returns the reconstructed plane in the matching two-bytes-per-
  sample `JpegXsPlane` layout. `cpih ∈ {0, 1}` (no transform / reversible
  RCT, Annex F.3 — bit-depth agnostic). Self-roundtrips bit-exactly
  through `decode_jpeg_xs`. Star-Tetrix, NLT pre-distortion and
  sub-sampling are not exposed on this path (later rounds).
* `oxideav_jpegxs::encoder::encode_planar_highbd_lossy(width, height, nc,
  cpih, nlx, nly, bd, q, &[Vec<u16>]) -> Result<Vec<u8>>` — round-133
  high-bit-depth **lossy** 4:4:4 entry point: the `q > 0` companion to
  `encode_planar_highbd`. Same `u16`-LE plane format and `Bw = B[i] = bd`
  (`bd ∈ 9..=16`), but with a non-zero precinct quantization step
  `q ∈ 1..=15` (Annex C.2 `Q[p]`) and `Fq = 8` (regular mode, Table A.8),
  so the per-band deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)`
  (Annex D.4) drops low bitplanes and the decoder reconstructs with the
  matching deadzone inverse (Annex D.2). Bit depth is orthogonal to the
  quantizer (both run on `i32` coefficients), so the only bit-depth-
  dependent pieces remain the level shift and `u16` packing. `cpih ∈
  {0, 1}`. Rejects `q = 0` (use `encode_planar_highbd`), `bd = 8` and
  `cpih = 3`. PSNR ≥ 40 dB at 10-bit q=1 and 16-bit RGB+RCT q=1, ≥ 30 dB
  at 12-bit q=2 NL=3/3; q=2 stream is strictly smaller than the lossless
  q=0 stream.
* `oxideav_jpegxs::encoder::encode_planar_subsampled(width, height,
  nc, cpih, nlx, nly, q, sx, sy, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-3 chroma-sub-sampled entry point. Each `planes[i]` has length
  `(width / sx[i]) * (height / sy[i])`. Supports 4:4:4 / 4:2:2 / 4:2:0
  with `(sx, sy) ∈ {1, 2}`. `q = 0` lossless / `q > 0` lossy.
* `oxideav_jpegxs::encoder::encode_planar_subsampled_highbd(width, height,
  nc, cpih, nlx, nly, bd, sx, sy, &[Vec<u16>]) -> Result<Vec<u8>>` —
  round-151 high-bit-depth chroma-sub-sampled lossless entry point.
  Widens `encode_planar_highbd` from 4:4:4-only to arbitrary per-
  component `(sx[i], sy[i]) ∈ {1, 2}` sampling at `bd = B[i] ∈ 9..=16`,
  using the same `u16`-LE plane format (`(width/sx[i])*(height/sy[i])`
  samples per plane) and the same DC level shift `1 << (bd − 1)`. Per
  Annex F.2 Table F.1 the reversible RCT (`Cpih = 1`) keeps `sx[i] =
  sy[i] = 1` for `i < 3`, so the typical 4:2:2 / 4:2:0 configurations
  are exposed only with `Cpih = 0`. Star-Tetrix (`Cpih = 3`) and NLT
  pre-distortion stay 8-bit-input specific and are not exposed here.
* `oxideav_jpegxs::encoder::encode_planar_subsampled_highbd_lossy(width,
  height, nc, cpih, nlx, nly, bd, q, sx, sy, &[Vec<u16>]) -> Result<Vec<u8>>`
  — round-151 high-bit-depth chroma-sub-sampled **lossy** companion to
  `encode_planar_subsampled_highbd`. Same per-component `(sx, sy)` and
  `u16`-LE plane format, but with `q ∈ 1..=15` (Annex C.2 `Q[p]`) and
  `Fq = 8` (regular mode, Table A.8 — required for `q > 0`); the
  per-band deadzone truncation `T[p,b] = clamp(Q − G[b], 0, 15)` (Annex
  D.4) drops low bitplanes and the decoder reconstructs with the
  matching deadzone inverse (Annex D.2). PSNR ≥ 40 dB at 10-bit 4:2:2
  q=1 and ≥ 30 dB at 12-bit 4:2:0 q=2 against the synthetic
  high-bit-depth ramp + chroma. Rejects `q = 0` (use
  `encode_planar_subsampled_highbd`), `bd = 8`, and `cpih = 3`.
* `oxideav_jpegxs::encoder::encode_planar_star_tetrix(width, height,
  nlx, nly, q, e1, e2, cf, ct, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-4 4-component CFA entry point (`Cpih = 3`, Star-Tetrix per
  Annex F.5). Component plane order is `[R, G1, G2, B]`. Emits the
  CTS marker (`Cf`, `e1`, `e2`) and the CRG marker (Table F.9 RGGB
  layout for `Ct=0`, GRBG layout for `Ct=1`).
* `oxideav_jpegxs::encoder::encode_planar_star_tetrix_highbd(width,
  height, nlx, nly, bd, e1, e2, cf, ct, &[Vec<u16>]) -> Result<Vec<u8>>`
  — round-195 high-bit-depth Star-Tetrix entry point. Widens
  `encode_planar_star_tetrix` from `B[i] = 8` to any `bd = B[i] ∈
  9..=16`. The Annex F.5 lifting (Tables F.4–F.8) is a pure-`i32`
  integer linear combination, so bit depth is fully orthogonal to the
  transform — the only bit-depth-dependent pieces are the DC level
  shift `1 << (bd − 1)` (Annex G.3 inverse) and the two-bytes-per-
  sample `u16`-LE plane format from rounds 118 / 133 / 151 (each
  `planes[i]` is `width * height` little-endian `u16` samples in
  `0..=2^bd − 1`). Codes losslessly with `Bw = B[i] = bd` and `Fq =
  0` (Table A.8). Pins `Nc = 4`, `sx[i] = sy[i] = 1` for `i < 4`,
  `Cpih = 3`, `q = 0` (Annex F.2 Table F.1 Star-Tetrix operand
  window). The CTS / CRG markers survive on the high-bit-depth path
  identical to the 8-bit form (only the CDT `B[i]` byte and the PIH
  `Bw` byte change on the wire). Self-roundtrips bit-exactly at
  10/12/16-bit including Ct=1 GRBG + non-default `e1/e2/Cf=3` (PSNR
  `INFINITY` ≥ 30 dB floor).
* `oxideav_jpegxs::encoder::encode_planar_star_tetrix_highbd_lossy(
  width, height, nlx, nly, bd, q, e1, e2, cf, ct, &[Vec<u16>]) ->
  Result<Vec<u8>>` — round-201 high-bit-depth Star-Tetrix **lossy**
  entry point: the `q > 0` companion to
  `encode_planar_star_tetrix_highbd`. Same four-component CFA plane
  layout (`Ω = [R, G1, G2, B]`, each `width * height` little-endian
  `u16` samples in `0..=2^bd − 1`, `bd ∈ 9..=16`), same `Bw = B[i] =
  bd` and DC level shift `1 << (bd − 1)`, but with `q ∈ 1..=15`
  (Annex C.2 `Q[p]`) and `Fq = 8` (regular mode, Table A.8 — required
  for `q > 0`). The Annex D.4 Table D.3 deadzone truncation
  `T[p,b] = clamp(Q − G[b], 0, 15)` drops the low magnitude
  bitplanes and the decoder reconstructs with the matching Annex D.2
  deadzone inverse (`Qpih = 0`). Bit depth, quantization, and the
  Annex F.5 lifting are mutually orthogonal (the lifting is `i32`
  integer linear, the forward / inverse quantizer runs on `i32`
  wavelet coefficients, and the colour transform runs in the
  un-quantized wavelet domain), so the only bit-depth-dependent
  pieces remain the level shift and the `u16`-LE plane packing.
  Pins `Nc = 4`, `sx[i] = sy[i] = 1` for `i < 4`, `Cpih = 3`. The
  CTS / CRG markers (`Cf`, `e1`, `e2`, RGGB/GRBG via `Ct`) survive on
  the high-bit-depth lossy path identical to the round-195 form.
  Rejects `q = 0` (use `encode_planar_star_tetrix_highbd`), `bd = 8`
  (use `encode_planar_star_tetrix`), `bd > 16`, and any
  plane-count != 4. PSNR ≥ 40 dB at 10-bit q=1 and at 16-bit Ct=1
  q=1, ≥ 30 dB at 12-bit q=2; q=2 stream is strictly smaller than
  the round-195 lossless q=0 stream on the same 12-bit fixture.
* `oxideav_jpegxs::encoder::encode_planar_nlt_quadratic(width, height,
  nc, cpih, nlx, nly, q, dco, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-5 NLT quadratic entry point. Applies forward quadratic pre-
  distortion (`y = sqrt(x/255) * 262143 + dco`, Annex G.4, Tnlt=1)
  before the DWT, forces `Bw = 18`, and emits the NLT marker. `dco`
  must fit in signed 16-bit. `q = 0` lossless; `q > 0` Fq=8 lossy.
* `oxideav_jpegxs::encoder::encode_planar_nlt_quadratic_highbd(width,
  height, nc, cpih, nlx, nly, bd, q, dco, &[Vec<u16>]) -> Result<Vec<u8>>`
  — round-181 high-bit-depth NLT quadratic entry point. Widens
  `encode_planar_nlt_quadratic` from `B[i] = 8` to any `bd = B[i] ∈
  9..=16` by parameterising the forward sqrt pre-distortion
  `y = round(sqrt(x / (2^B[i] − 1)) × (2^Bw − 1)) + dco` (Annex G.4)
  in the input domain. Uses `Bw = 20` (the top of the Table A.8
  `{8, 18, 20}` set, giving ≥ 4 bits of precision headroom over any
  supported `B[i]`); DC level shift is `1 << 19`. Plane format
  matches the round-118 convention: little-endian `u16` samples in
  `0..=2^bd − 1` per `JpegXsPlane`, 4:4:4 only (`sx[i] = sy[i] = 1`).
  `cpih ∈ {0, 1}` — no transform or reversible RCT (the Annex F.3
  RCT operand window `c < 3` is bit-depth agnostic). `q = 0` is the
  lossless-within-sqrt-rounding mode (PSNR ≥ 40 dB on synthetic
  ramps at 10/12/16-bit); `q > 0` engages `Fq = 8` and the Annex D.4
  deadzone truncation. `dco` validated to signed 16-bit. NLT
  extended (`Tnlt = 2`) and Star-Tetrix (`Cpih = 3`) stay 8-bit-input
  specific.
* `oxideav_jpegxs::encoder::encode_planar_nlt_extended(width, height,
  nc, cpih, nlx, nly, q, t1, t2, e, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-7 NLT extended entry point. Applies a forward extended-gamma
  pre-distortion (Annex G.5, Tnlt=2, three-segment kernel with
  thresholds `0 < T1 < T2 ≤ 2^Bw - 1` and slope exponent `E ∈ 1..=4`)
  built from a `2^Bw`-entry reverse LUT inverting the decoder's
  `extended_path`. Forces `Bw = 18` and emits the NLT marker with
  `(T1, T2, E)`. `q = 0` lossless within LUT resolution; `q > 0` Fq=8
  lossy.
* `oxideav_jpegxs::encoder::encode_planar_nlt_extended_highbd(width,
  height, nc, cpih, nlx, nly, bd, q, t1, t2, e, &[Vec<u16>]) ->
  Result<Vec<u8>>` — round-193 high-bit-depth NLT extended entry
  point. Widens `encode_planar_nlt_extended` from `B[i] = 8` to any
  `bd = B[i] ∈ 9..=16` by dropping the `.min(257)` cap on the reverse
  LUT (the round-7 8-bit shortcut) and addressing the full `1 << bd`
  reconstructed-level table. Uses `Bw = 20` (the top of the Table A.8
  `{8, 18, 20}` set, giving ≥ 4 bits of headroom over the gamma
  kernel) and the DC level shift `1 << 19`. Plane format matches the
  round-118 / 181 convention: little-endian `u16` samples in
  `0..=2^bd − 1` per `JpegXsPlane`, 4:4:4 only (`sx[i] = sy[i] = 1`).
  `cpih ∈ {0, 1}` — no transform or reversible RCT (the Annex F.3
  RCT operand window `c < 3` is bit-depth agnostic). `q = 0` is the
  lossless-within-LUT-resolution mode (PSNR ≥ 30 dB on synthetic
  ramps at 10/12/16-bit); `q > 0` engages `Fq = 8` and the Annex D.4
  deadzone truncation. `t1`, `t2`, `e` validated identically to the
  8-bit path (`0 < t1 < t2 ≤ 2^Bw − 1`, `1 ≤ e ≤ 4`). Star-Tetrix
  (`Cpih = 3`) high-bit-depth stays out of scope.
* `oxideav_jpegxs::encoder::encode_planar_fs1(width, height, nc, cpih,
  nlx, nly, q, &[Vec<u8>]) -> Result<Vec<u8>>` — round-100 `Fs = 1`
  entry point. Same shape as `encode_planar_lossy` but sets the PIH
  sign-handling flag to `Fs = 1` (Annex A.4.4 Table A.11): signs ride a
  dedicated sign sub-packet (Annex C.5.5, Table C.9, one bit per
  non-zero coefficient) instead of being interleaved into the data
  sub-packet (Table C.8). Decodes byte-identically to the `Fs = 0`
  layout and is no larger on sparse-sign content (where the `Fs = 0`
  form wastes `Ng = 4` sign bits per significant code group regardless
  of how many coefficients are non-zero). The decoder has threaded
  `pih.fs` end-to-end since the early rounds.
* `oxideav_jpegxs::encoder::encode_planar_hsl(width, height, nc, cpih,
  nlx, nly, q, hsl, &[Vec<u8>]) -> Result<Vec<u8>>` — round-103
  multi-slice entry point. `hsl` is the slice height in precinct rows
  (PIH `Hsl`, Annex B.10); `hsl = 0` is the single-slice default
  (`Hsl = Np,y`, byte-identical to `encode_planar_lossy`). `hsl > 0`
  partitions the `Np,y = ⌈Hf / 2^NL,y⌉` precinct rows into
  `⌈Np,y / hsl⌉` slices of `hsl` rows each (the last slice is shorter
  when `Np,y` is not a multiple of `hsl`), emitting one SLH marker per
  slice (Annex A.4.12 Table A.25) with `Yslh = t` (top-down slice
  order). Rejects `hsl > Np,y`. The decoder reconstructs the identical
  grouping from PIH `Hsl` + `Np,y` (slice walker, Annex B.10), so the
  output round-trips. Vertical prediction is precinct-scoped in this
  encoder, so slice boundaries carry no cross-slice predictor state
  (Annex B.10 disable-across-boundaries requirement is satisfied
  trivially).
* `oxideav_jpegxs::encoder::encode_planar_hsl_qslice(width, height, nc,
  cpih, nlx, nly, hsl, &[u8], &[Vec<u8>]) -> Result<Vec<u8>>` — round-206
  per-slice rate-budgeting entry point. Lifts `encode_planar_hsl`'s
  single picture-level `q` scalar to a `q_slices` slice — one `Q[p]`
  value per slice, in top-down (`Yslh`) order. `q_slices.len()` must
  exactly equal the slice count `⌈Np,y / max(hsl, 1)⌉`; each entry is
  in `0..=15` (the band-truncation `T[p,b] = clamp(Q − G[b] − r, 0,
  15)` math is identical, only the source of `Q[p]` changes). `Q[p]`
  is a per-precinct field per Annex C.2 Table C.1, so any per-slice
  partition is spec-compliant; the encoder emits each precinct inside
  slice `t` with `Q[p] = q_slices[t]`, leaving the SOC / CAP / PIH /
  CDT / WGT / SLH markers unchanged. `Fq` is auto-selected: `0`
  (lossless) when every entry is `0`, else `8` (regular, Table A.8 —
  required for any non-zero `Q[p]`). Lossless / lossy mixing is
  supported (e.g. a salient mid-picture slice gets `Q[p] = 0` while
  edge slices get `Q[p] = 4`). When `q_slices` carries a single
  repeated value the output is byte-identical to `encode_planar_hsl`
  at that `q`; `&[3]` with `hsl = 0` is byte-identical to
  `encode_planar_lossy` at `q = 3`. Rejects wrong-length `q_slices`
  and entries `> 15`. The decoder reads `Q[p]` per precinct
  (`parse_precinct_header` + `precinct_truncation`), so no decoder
  change is needed — this is a pure encoder rate-allocation lever.
  Round 206 leaves the assignment of `q_slices` to the caller; a
  follow-up round can wrap this with a PSNR-driven slice budgeter.
* `oxideav_jpegxs::encoder::encode_planar_qpr(width, height, nc, cpih,
  nlx, nly, &[u8], &[Vec<u8>]) -> Result<Vec<u8>>` — round-233 per-
  precinct `Q[p]` override entry point — the spec-natural form of
  Annex C.2 Table C.1 where `Q[p]` is indexed by precinct `p`. Round
  206 took `q` from picture-level to one per slice; round 233 takes
  it the rest of the way — one `Q[p]` per precinct. `q_precincts` is
  indexed in raster scan order with the precinct at row `py`, column
  `px` at position `py * Np,x + px`; length must equal `Np,y × Np,x`
  where `Np,y = ⌈Hf / 2^NL,y⌉` and `Np,x = 1` for this single-
  precinct-column entry point (`Cw = 0`), so the array reduces to one
  `Q[p]` per precinct row. Each entry is in `0..=15`. `Fq` is
  auto-selected: `0` (lossless) when every entry is `0`, else `8`
  (regular, Table A.8 — required for any non-zero `Q[p]`). All-zero
  `q_precincts` is byte-identical to `encode_planar` at the same
  geometry; all-equal `q_precincts = [q; n]` is byte-identical to
  `encode_planar_lossy` at that single `q`; mixed `q_precincts`
  keeps strictly more bits than the uniform max-Q baseline whenever
  any entry is `< 15` (lower `Q[p]` retains more magnitude bitplanes).
  Composes with the reversible RCT (`Cpih = 1`, Annex F.3) and Star-
  Tetrix (`Cpih = 3`, Annex F.5) — both Q-agnostic on the wavelet-
  domain coefficients. The decoder reads `Q[p]` per precinct
  (`parse_precinct_header` + `precinct_truncation`) since the early
  rounds, so no decoder change is needed — the bitstream-wire impact
  is exactly the per-precinct `Q` byte. Scope: 4:4:4, `Cpih ∈ {0, 1,
  3}`, `Cw = 0` (single precinct column), `Hsl = 0` (single slice),
  `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. High-bit-depth widening,
  `Cw > 0` multi-column and the `Hsl > 0` × per-precinct cross-
  product intersect on future rounds.
* `oxideav_jpegxs::encoder::encode_planar_qpih(width, height, nc, cpih,
  nlx, nly, q, &[Vec<u8>]) -> Result<Vec<u8>>` — round-108
  uniform-inverse-quantizer entry point. Same shape as
  `encode_planar_lossy` but sets the PIH inverse-quantizer type to
  `Qpih = 1` (Annex A.4.4 Table A.10): the decoder reconstructs with the
  uniform / Neumann-series kernel (Annex D.3) instead of the deadzone
  kernel (Annex D.2, `Qpih = 0`). The data sub-packet on the wire is
  byte-identical for both quantizer types — only the `Qpih` field in the
  PIH `Lh:Rl:Qpih:Fs:Rm` byte changes, and the decoder picks the
  matching inverse. At `q = 0` (`T = 0`) both kernels reconstruct
  exactly, so `Qpih = 1` self-roundtrips losslessly and decodes
  byte-identically to the `Qpih = 0` form (the two codestreams differ in
  exactly one byte — the PIH quantizer-type byte). At `q > 0` the two
  kernels reconstruct different (both valid) lossy magnitudes. Rejects
  reserved `Qpih` values (2/3), mirroring the decoder's `Qpih > 1`
  rejection. The decoder has threaded `pih.qpih` into
  `dequantize_precinct` since the early rounds.
* `oxideav_jpegxs::encoder::encode_planar_rp(width, height, nc, cpih,
  nlx, nly, q, rp, &[Vec<u8>]) -> Result<Vec<u8>>` — round-115
  precinct-refinement entry point. `rp` is the precinct refinement `R[p]`
  (Annex C.2 Table C.1, constant across precincts, range `0..=NL-1` where
  `NL = (Nc-Sd)×Nβ + Sd`). `rp = 0` is the no-refinement default,
  byte-identical to `encode_planar_lossy`. `rp > 0` activates the Annex
  C.6.2 Table C.10 truncation refinement `T[p,b] = clamp(Q − G[b] − r, 0,
  15)` with `r = (P[b] < R[p]) ? 1 : 0`. The encoder emits per-band
  priorities `P[b] = b` (the true band index, Annex B.6) in the WGT
  marker (Annex A.4.11), so `R[p] = k` refines exactly the `k`
  lowest-index bands — LL first, per the `β`-major band enumeration —
  granting them one extra retained magnitude bitplane (lower `T`, finer
  quantization). The decoder reconstructs the identical `T[p,b]` from the
  `(P[b], R[p])` pair (`entropy::truncation_position`, supported since the
  early rounds), so any output round-trips. At `q = 0` refinement is a
  lossless no-op (T already at its 0 floor); at `q > 0` it shifts coded
  bits toward the refined low-frequency bands. Rejects `R[p] >= NL`.
* `oxideav_jpegxs::pick_rp_for_target_bytes(width, height, nc, cpih,
  nlx, nly, q, target_bytes, &[Vec<u8>]) -> Result<u8>` — round-218
  rate-budget driven `R[p]` picker for the
  [`encode_planar_rp`] path. Linear scan from `R[p] = NL-1` down to
  `0`; returns the first `R[p]` whose `encode_planar_rp` codestream
  fits `target_bytes`. Refinement is monotone non-decreasing in
  codestream length (each refined band gains one extra retained
  magnitude bitplane via the Annex C.6.2 Table C.10 term `r = (P[b] <
  R[p]) ? 1 : 0` lowering `T[p,b]`), so the first fit is also the
  largest fit — the picker spends every available byte on additional
  refinement. Step 1 probes `R[p] = 0` as the baseline; if even that
  overshoots, errors with `target_bytes unreachable; R[p]=0 emits N
  bytes` (the budget is unreachable by `R[p]` alone; lower `q` or
  `pick_q_slices_for_target_bytes` can still help). Rejects
  `target_bytes == 0`. Every measurement is a real `encode_planar_rp`
  call. Scope mirrors `encode_planar_rp` exactly: 4:4:4,
  `Cpih ∈ {0, 1, 3}`, `Cw = 0`, `Hsl = 0`, `Sd = 0`, `Fs = 0`, `Qpih
  = 0`, `q ∈ 0..=15`. At `q = 0` refinement is a lossless no-op so
  the picker returns `NL-1` and every candidate is byte-identical.
* `oxideav_jpegxs::encode_planar_rp_target_bytes(width, height, nc,
  cpih, nlx, nly, q, target_bytes, &[Vec<u8>]) -> Result<(Vec<u8>,
  u8)>` — round-218 convenience wrapper around
  `pick_rp_for_target_bytes` + `encode_planar_rp` returning
  `(codestream, rp)`. The codestream is guaranteed to satisfy
  `codestream.len() <= target_bytes` (otherwise the picker returns
  the `unreachable` error). The returned `rp` is the one
  `pick_rp_for_target_bytes` selected; callers can persist it for
  reproducible re-encode of identical parameters.
* `oxideav_jpegxs::encode_planar_hsl_qslice_rp(width, height, nc, cpih,
  nlx, nly, hsl, q_slices, rp, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-224 joint per-slice `Q[p]` + precinct refinement `R[p]`
  primitive. Composes the round-206 per-slice `Q[p]` mechanism
  (`q_slices` carries one `Q[p]` per slice in top-down `Yslh` order,
  validated `len() == ⌈Np,y / max(hsl, 1)⌉`, each entry in `0..=15`)
  with the round-115 precinct refinement mechanism (`rp` is the
  picture-constant `R[p] ∈ 0..=NL-1` driving the Annex C.6.2 Table
  C.10 term `r = (P[b] < rp) ? 1 : 0` inside `T[p,b] = clamp(Q[p] −
  G[b] − r, 0, 15)`). The two axes are orthogonal on the wire (per-
  slice `Q[p]` lives in each precinct's `Q` byte, `R[p]` lives in
  each precinct's `R` byte). All-equal `q_slices` + `rp = 0` is
  byte-identical to `encode_planar_hsl_qslice`; single-entry + `hsl =
  0` + `rp > 0` byte-identical to `encode_planar_rp` at the same
  `(q, rp)`; `q_slices = [0; n]` with any `rp` byte-identical to the
  lossless `rp = 0` stream (refinement is a no-op when `T` is already
  at its 0 floor). `Fq` is auto-selected (0 when every `q_slices`
  entry is 0, else 8). Scope: 4:4:4, `Cpih ∈ {0, 1, 3}`, `Cw = 0`,
  `Sd = 0`, `Fs = 0`, `Qpih = 0`, `B[i] = 8`. The decoder needs no
  change — it reads `Q[p]` and `R[p]` per precinct from the existing
  precinct-header path, so any output round-trips.
* `oxideav_jpegxs::pick_q_slices_rp_for_target_bytes(width, height,
  nc, cpih, nlx, nly, hsl, target_bytes, &[Vec<u8>]) -> Result<(
  Vec<u8>, u8)>` — round-224 joint rate-budget picker. Returns
  `(q_slices, rp)` driving `encode_planar_hsl_qslice_rp` to emit a
  codestream of length `<= target_bytes`. Two-axis nested search:
  outer loop on `rp` from `0` up to `NL-1`, keeping the last fitting
  solution; inner loop reuses r212's three-pass `q_slices` strategy
  (lossless probe at `[0; n]`, uniform-`Q` bisect on `1..=15`,
  activity-driven per-slice relaxation walking the lowest-L1-row-
  gradient slices first) against the joint primitive at the current
  `rp`. Promotion stops at the first `rp+1` whose inner search
  fails (refinement is monotone non-decreasing in codestream length
  at fixed `Q[p]`). Errors with `target_bytes unreachable; rp=0
  Q=15 emits N bytes` when even the most-aggressive (`rp = 0` +
  `Q = 15`) configuration overshoots. Rejects `target_bytes == 0`.
  Every measurement is a real `encode_planar_hsl_qslice_rp` call.
* `oxideav_jpegxs::encode_planar_hsl_qslice_rp_target_bytes(width,
  height, nc, cpih, nlx, nly, hsl, target_bytes, &[Vec<u8>]) ->
  Result<(Vec<u8>, Vec<u8>, u8)>` — round-224 convenience wrapper
  around `pick_q_slices_rp_for_target_bytes` +
  `encode_planar_hsl_qslice_rp` returning `(codestream, q_slices,
  rp)`. The codestream is guaranteed to satisfy `codestream.len() <=
  target_bytes` (otherwise the picker returns the `unreachable`
  error). The returned `(q_slices, rp)` is the pair the picker
  selected; callers can persist them for reproducible re-encode of
  identical parameters.
* `oxideav_jpegxs::encode_planar_hsl_qslice_rp_highbd(width, height,
  nc, cpih, nlx, nly, bd, hsl, q_slices, rp, &[Vec<u16>]) ->
  Result<Vec<u8>>` — round-230 **high-bit-depth widening** of the
  round-224 joint primitive. Composes per-slice `Q[p]` (round-206,
  Annex C.2 Table C.1 lifted to per-slice) and precinct refinement
  `R[p]` (round-115, Annex C.6.2 Table C.10) at component bit depth
  `bd = B[i] ∈ 9..=16` against the `u16`-LE per-plane format inherited
  from rounds 118 / 133 / 151. Each `planes[i]` carries
  `width * height` little-endian `u16` samples in `0..=2^bd − 1`. The
  forward quantizer (Annex D.4) and the refinement term
  `r = (P[b] < R[p]) ? 1 : 0` (Annex C.6.2 Table C.10) both run on
  `i32` wavelet coefficients independent of `B[i]`, so the only
  bit-depth-dependent pieces remain the DC level shift `1 << (bd − 1)`
  (Annex G.3 inverse) and the `to_le_bytes()` plane packing. The
  codestream uses `Bw = B[i] = bd`; `Fq` is auto-selected (`0` when
  every slice is lossless, else `8`). At `q_slices = [0; n]`
  refinement is a lossless no-op regardless of `rp`; at `rp = 0`
  single-slice with `q_slices = [q]` the bytes are identical to
  `encode_planar_highbd_lossy` at the same `q`. Composes with the
  reversible RCT (`Cpih ∈ {0, 1}`, Annex F.3 bit-depth-agnostic);
  Star-Tetrix and NLT pre-distortion stay out of scope on this path.
* `oxideav_jpegxs::pick_q_slices_rp_for_target_bytes_highbd(width,
  height, nc, cpih, nlx, nly, bd, hsl, target_bytes, &[Vec<u16>]) ->
  Result<(Vec<u8>, u8)>` — round-230 high-bit-depth picker. Same
  two-axis nested search as `pick_q_slices_rp_for_target_bytes` —
  outer loop on `rp` from `0` upward keeping the last fitting
  solution, inner loop replays r212's three-pass `q_slices` strategy
  (lossless probe → uniform-`Q` bisect → activity-driven per-slice
  relaxation) — but the relaxation pass uses a `u16`-aware
  slice-activity metric so the per-plane spatial structure reflects
  original sample magnitudes rather than the low-byte / high-byte
  interleave of a `to_le_bytes()` packing. Baseline reachability and
  `target_bytes == 0` handling mirror the 8-bit picker. Every
  measurement is a real `encode_planar_hsl_qslice_rp_highbd` call.
* `oxideav_jpegxs::encode_planar_hsl_qslice_rp_target_bytes_highbd(width,
  height, nc, cpih, nlx, nly, bd, hsl, target_bytes, &[Vec<u16>]) ->
  Result<(Vec<u8>, Vec<u8>, u8)>` — round-230 high-bit-depth
  convenience wrapper, returns `(codestream, q_slices, rp)`. The
  codestream is byte-identical to a follow-up
  `encode_planar_hsl_qslice_rp_highbd(.., q_slices, rp, ..)` call, so
  callers can persist `(q_slices, rp)` for reproducible re-encode.
* `oxideav_jpegxs::encode_planar_cw(width, height, nc, cpih, nlx, nly,
  q, cw, &[Vec<u8>]) -> Result<Vec<u8>>` — round-8 multi-precinct-per-
  row 4:4:4 entry point. `cw` controls the precinct-width parameter
  `Cw` (PIH §A.4.4); `cw = 0` reduces to a single precinct column
  spanning the full picture width (bit-equivalent to `encode_planar`).
  `cw > 0` splits each precinct row into `Np,x = ⌈Wf / Cs⌉` precincts
  with `Cs = 8 × cw × max(sx) × 2^NL,x`. Rejects `Cs > Wf`. Routes
  everything through the picture-level cascade DWT so per-precinct
  columns commute with the wavelet boundaries.
* `oxideav_jpegxs::encoder::encode_planar_sd(width, height, nc, nlx,
  nly, q, sd, &[Vec<u8>]) -> Result<Vec<u8>>` — round-9 (r91) `Sd > 0`
  (CWD) entry point per Annex A.4.7 Table A.18. The leading `nc - sd`
  components are wavelet-coded as usual; the trailing `sd` components
  are coded raw (no DWT) and follow the wavelet packets with the
  component as fast and line as slow variable per Annex B.7 Table B.4.
  Emits a CWD marker (`FF 17`, Lcwd=3) carrying `Sd`. Constraints:
  `1 ≤ sd ≤ nc-1`, `nc > 3` (spec hard requirement), every suppressed
  component must have `sx[i] = sy[i] = 1`, `cpih = 0` (no colour
  transform). Self-roundtrips losslessly at Sd=1 Nc=4 / Sd=2 Nc=5 and
  holds ≥ 30 dB per-component PSNR at Sd=1 q=2.
* `oxideav_jpegxs::encoder::encode_planar_sd_rct(width, height, nc,
  nlx, nly, q, sd, &[Vec<u8>]) -> Result<Vec<u8>>` — round-95 (r93)
  `Sd > 0` + `Cpih = 1` (RCT) entry point. Constraints add
  `nc - sd >= 3` so the RCT operand window (`c < 3` per Annex F.2
  Table F.1) is wavelet-coded. Self-roundtrips losslessly at Sd=1
  Nc=4 and Sd=2 Nc=5; holds ≥ 25 dB per-component PSNR at q=2.
* `oxideav_jpegxs::encoder::encode_planar_sd_star_tetrix(width, height,
  nc, nlx, nly, q, sd, e1, e2, cf, ct, &[Vec<u8>]) -> Result<Vec<u8>>` —
  round-95 (r93) `Sd > 0` + `Cpih = 3` (Star-Tetrix) entry point. The
  first 4 components carry the CFA-laid-out Star-Tetrix data and ride
  through the lifting cascade; components 4..Nc are suppressed CWD
  tail. Constraints add `nc - sd >= 4` so the Star-Tetrix operand
  window (`c < 4`) is wavelet-coded. Emits CTS + CRG markers (the CRG
  body length scales with `Nc`, with `(0, 0)` placement for entries
  beyond the first four CFA components). Self-roundtrips losslessly
  at Sd=1 Nc=5.
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
  `NL_y ≤ NL_x`; Fs=0 joint-sign data sub-packet (Table C.8) or Fs=1
  separate sign sub-packet (Annex C.5.5, Table C.9, one bit per non-zero
  coefficient) selectable via `encode_planar_fs1`; multi-slice emission
  (Annex B.10, one SLH per slice with `Yslh = t`, precinct rows grouped
  `⌈Np,y / Hsl⌉`-ways) via `encode_planar_hsl`; Qpih-aware forward
  quantizer (`forward_quant_index`) — Qpih=0 deadzone truncation
  `v = |c| >> T` (Annex D.4 Table D.3, matched by the Annex D.2 inverse)
  or Qpih=1 uniform round-to-nearest index
  `v = ((|c| << ζ) − |c| + (1 << M)) >> (M+1)`, `ζ = M − T + 1`
  (Annex D.5 Table D.4, matched by the Annex D.3 Neumann-series inverse)
  via `encode_planar_qpih` (PIH `Qpih` field, Annex A.4.4 Table A.10 — at
  q=0 both reduce to the lossless identity, so the deadzone stream is
  byte-identical and the uniform stream differs only in the PIH byte; at
  q>0 the uniform data sub-packet diverges); precinct refinement
  `R[p] > 0` (Annex C.2 Table C.1 + Annex C.6.2 Table C.10) via
  `encode_planar_rp` — WGT carries per-band priorities `P[b] = b` (Annex
  B.6 band index) and the precinct header carries `R[p]`, so the
  truncation gains the refinement term `r = (P[b] < R[p]) ? 1 : 0`,
  lowering `T[p,b]` by one for the `R[p]` lowest-index (LL-first) bands;
  `build_band_priorities_sd` emits priorities in the same order as
  `build_band_gains_sd`; short packet headers;
  symmetric reflection for
  partial bottom precincts (odd heights); CTS / CRG marker emission for
  `Cpih = 3`

## Out of scope (next round)

* High bit depth (`B[i] > 8`) beyond the round-118 lossless +
  round-133 lossy 4:4:4 paths, the round-151 4:2:2 / 4:2:0 sub-
  sampled paths, the round-181 NLT quadratic path
  (`encode_planar_nlt_quadratic_highbd`), the round-193 NLT extended
  path (`encode_planar_nlt_extended_highbd`), the round-195
  Star-Tetrix lossless path (`encode_planar_star_tetrix_highbd`) and
  the round-201 Star-Tetrix lossy path
  (`encode_planar_star_tetrix_highbd_lossy`): the encoder's
  high-bit-depth surface is now complete for the round-supported
  feature set across all four colour-transform modes (`Cpih ∈
  {0, 1, 3}`, no transform / RCT / Star-Tetrix) and both NLT modes
  (`Tnlt ∈ {1, 2}`), lossless + lossy. Bit depths above 16 need a
  wider plane format (`u32`).
* Encoder round 225+: an `R[p]`-driven PSNR-optimizing **priority
  assignment** that replaces the plain band-index priorities `P[b] =
  b` with a content-adaptive ordering — the joint rate picker lands
  in r224 (`pick_q_slices_rp_for_target_bytes` +
  `encode_planar_hsl_qslice_rp_target_bytes`, outer-loop on `rp`
  with the inner r212 `q_slices` search) and closes the r218 "next
  round" follow-up tail. With both rate-axis levers (`q_slices` per
  slice and `R[p]` per precinct) now exposed and the joint picker
  walking them in a nested search, the remaining lever is the WGT
  priority bytes themselves: r115/r218/r224 all emit `P[b] = b` (the
  plain band-index priorities), which is spec-compliant but leaves
  PSNR on the table when content correlates with the band ordering.
  A second open lever is high-bit-depth widening of the joint
  primitive (`encode_planar_inner_bd` already plumbs all four
  parameters through `q_slices`, `rp`, `bd`, `hsl`).
  (Round 206 lands the per-slice `Q[p]` override primitive
  (`encode_planar_hsl_qslice`, Annex C.2 Table C.1) — `q_slices`
  carries one `Q[p]` per slice in top-down `Yslh` order, the encoder
  emits each precinct's header with the slice-local `Q[p]`, and the
  decoder reads `Q[p]` per precinct so no decoder change is needed.
  All-equal `q_slices` is byte-identical to `encode_planar_hsl` at
  that `q`; single-entry + `hsl=0` is byte-identical to
  `encode_planar_lossy`.
  (Round 115 lands `R[p] > 0` precinct refinement per Annex C.2 Table C.1
  + Annex C.6.2 Table C.10 + Annex A.4.11 — WGT priorities `P[b] = b`,
  precinct-header `R[p]`, `T[p,b] = clamp(Q − G[b] − r, 0, 15)` via
  `encode_planar_rp`.
  Round 111 lands the `Qpih`-aware forward quantizer per Annex D.4
  Table D.3 (deadzone) + Annex D.5 Table D.4 (uniform round-to-nearest)
  via `forward_quant_index` — the `Qpih = 1` data sub-packet now carries
  uniform indices at `q > 0` instead of the deadzone-floored indices
  round 108 left in place; deadzone output stays byte-identical.
  Round 108 lands `Qpih = 1` uniform-inverse-quantizer signalling per
  Annex A.4.4 Table A.10 + Annex D.3.
  Round 103 lands `Hsl > 0` multi-slice emission per Annex B.10 — one
  SLH per slice, `Yslh = t`, precinct rows grouped `⌈Np,y / Hsl⌉`-ways.
  Round 95 lifts the round-9 (r91) blanket `Cpih = 0` restriction on
  `Sd > 0`: the operand window of each colour transform — `c < 3` for
  RCT, `c < 4` for Star-Tetrix — is now allowed to coexist with the
  CWD-suppressed tail as long as `Nc - Sd >= 3 / 4`. Round 8 lands
  `Cw > 0` multi-precinct-per-row.)
