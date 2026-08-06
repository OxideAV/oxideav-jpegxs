//! Packet body — Annex C.4 / C.5 / C.6.
//!
//! The packet body is a back-to-back concatenation of four
//! sub-packets in a fixed order:
//!
//! 1. **Significance** sub-packet (Table C.5). One bit per
//!    significance group of every (band, line) the packet covers, but
//!    only when `D[p,b] & 2 != 0` and `Dr[p,s] == 0`. Padded to a
//!    byte boundary at the end. The packet header does not signal its
//!    byte count; it is inferred from the geometry.
//! 2. **Bitplane-count** sub-packet (Table C.7). For every (band,
//!    line) the packet covers, dispatch on `Dr[p,s]` and
//!    `D[p,b] & 1`:
//!    * `Dr == 1` → raw mode (Table C.12, §C.6.4): `Br` bits per
//!      code group, no prediction.
//!    * `D & 1 == 0` → no-prediction VLC (Table C.14, §C.6.6).
//!      Predictor `mtop = T[p,b]`. Significance gating per
//!      `D & 2`.
//!    * `D & 1 == 1` → vertical prediction VLC (Table C.13, §C.6.5).
//!      Predictor uses `Mtop[p,λ,b,g]` from the previous line of the
//!      same band. When the current line is the first line of the band
//!      in the precinct (`λ − sy < L0[p,b]`), `Mtop` / `Ttop` come from
//!      the precinct directly above (`p − Np,x`) via the [`PrecinctTop`]
//!      predecessor the caller supplies (Annex C.6.3 Table C.11);
//!      otherwise they come from the line above within the same
//!      precinct. Significance gating per `D & 2`.
//!
//!    Padded to a byte boundary, then `Lcnt[p,s]` bytes total
//!    (filler).
//! 3. **Data** sub-packet (Table C.8). For every code group with
//!    `M[p,λ,b,g] > T[p,b]`, optionally read `Ng` sign bits
//!    (`Fs == 0`) followed by `(M − T) × Ng` magnitude bits MSB-first.
//!    Padded to byte + filler to `Ldat[p,s]`.
//! 4. **Sign** sub-packet (Table C.9). Only when `Fs == 1`. One bit
//!    per non-zero `v[p,λ,b,Ng×g+k]` — including the "meaningless"
//!    tail positions `Wpb ≤ Ng×g+k < Ncg×Ng` of a band whose width is
//!    not a multiple of `Ng`, whenever their transmitted magnitude is
//!    non-zero (Table C.9 NOTE 2). Padded + filler to `Lsgn[p,s]`.
//!
//! Vertical-prediction predecessor:
//!
//! * The caller passes an optional [`PrecinctTop`] holding the last
//!   decoded band line's bitplane counts and the truncation positions
//!   of the precinct directly above. The decoder uses it for the
//!   first-line-of-band cross-precinct predictor (Table C.11). When the
//!   predecessor is `None` and a vertical-prediction packet references a
//!   first-line band, the codestream is malformed — §C.6.1 / §C.6.3
//!   forbid vertical prediction at the topmost precinct of a slice.
//! * `entry.line` is supplied by the walker in band-grid units stepping
//!   by one (the spec image-grid line `λ` is divided by `sy[i]`), so the
//!   intra-precinct predecessor step `sy` (Table C.11) is one band-grid
//!   line and the last band line of the precinct above is its
//!   `L1[p,b] − 1 − L0[p,b]` stored line.

use crate::error::{JpegXsError as Error, Result};

use std::collections::HashMap;

use crate::entropy::bits::{vlc, BitReader};
use crate::entropy::packet_header::PacketHeader;
use crate::entropy::precinct_header::PrecinctHeader;
use crate::entropy::{precinct_truncation, PacketLayout, PrecinctGeometry};

/// Decoded coefficients for one band of one precinct. Magnitudes and
/// signs are kept separately in the form Annex D.2 / D.3 expects.
#[derive(Debug, Clone)]
pub struct BandCoefficients {
    /// `wpb` from the band geometry; redundant with the storage size
    /// but kept for callers that consume one band at a time.
    pub wpb: u32,
    /// Number of lines stored. Each line is `wpb` long. `lines == l1
    /// - l0` from the band geometry.
    pub num_lines: u32,
    /// `v[p,λ,b,x]` — quantization-index magnitudes. Row-major;
    /// `v[line_index * wpb + x]` for `line_index = λ - L0`.
    pub v: Vec<u32>,
    /// `s[p,λ,b,x]` — signs (0 = positive, 1 = negative).
    pub s: Vec<u8>,
    /// `M[p,λ,b,g]` — bitplane counts; `Ncg[p,b]` per line.
    pub m: Vec<u8>,
}

impl BandCoefficients {
    fn new(wpb: u32, ncg: u32, num_lines: u32) -> Self {
        let line = wpb as usize;
        Self {
            wpb,
            num_lines,
            v: vec![0u32; line * num_lines as usize],
            s: vec![0u8; line * num_lines as usize],
            m: vec![0u8; (ncg as usize) * (num_lines as usize)],
        }
    }
}

/// Output of [`decode_packet_body`]: one `BandCoefficients` per band
/// in the precinct geometry, plus the bytes consumed by the body.
#[derive(Debug)]
pub struct PacketDecode {
    /// Per-band coefficients indexed by band id.
    pub bands: Vec<BandCoefficients>,
    /// Bytes consumed by the four sub-packets, including filler.
    pub bytes_consumed: usize,
}

/// Decode one packet body. The caller has already parsed the
/// precinct header (which gives `T[p,b]` and `D[p,b]`) and the packet
/// header (which gives `Dr[p,s]`, `Ldat`, `Lcnt`, `Lsgn`).
///
/// `prev_state` is mutated as the bitplane-count decoder writes
/// into the per-(band, line) state map; later packets in the same
/// precinct can use those values for vertical prediction.
pub fn decode_packet_body(
    buf: &[u8],
    geom: &PrecinctGeometry,
    precinct: &PrecinctHeader,
    packet: &PacketHeader,
    layout: &PacketLayout,
    prev_state: &mut PrecinctState,
    top: Option<&PrecinctTop>,
) -> Result<PacketDecode> {
    let truncation = precinct_truncation(geom, precinct);

    // Lazy-initialise the per-band coefficient buffers if the caller
    // didn't pre-populate them. The walker (round 4) will keep these
    // alive across packets to accumulate per-line writes.
    if prev_state.coefficients.is_empty() {
        prev_state.coefficients = geom
            .bands
            .iter()
            .enumerate()
            .map(|(b, band)| {
                if !band.exists {
                    BandCoefficients::new(0, 0, 0)
                } else {
                    let lines = (band.l1 as u32).saturating_sub(band.l0 as u32);
                    BandCoefficients::new(band.wpb, geom.ncg(b), lines)
                }
            })
            .collect();
    }

    let mut total_consumed = 0usize;

    // === Significance sub-packet =========================================
    {
        let buf_sig = buf.get(total_consumed..).ok_or_else(|| {
            Error::invalid("jpegxs entropy: packet body truncated at sig sub-packet")
        })?;
        let mut reader = BitReader::new(buf_sig);
        for entry in &layout.entries {
            let bi = entry.band as usize;
            let band = &geom.bands[bi];
            if !band.exists {
                continue;
            }
            let line_index = (entry.line - band.l0) as usize;
            let dpb = precinct.d[bi];
            if packet.dr == 0 && (dpb & 2) != 0 {
                let ns = geom.ns(bi) as usize;
                for j in 0..ns {
                    let z = reader.read_bit()?;
                    prev_state
                        .sig_flags
                        .insert((entry.band, entry.line, j as u32), z);
                }
            } else {
                // No significance information for this band+line. Per
                // Table C.5 a Z bit of 1 flags the significance group
                // as all-insignificant, so the neutral default is
                // Z = 0 ("significant" — bitplane counts are VLC-coded).
                let ns = geom.ns(bi) as usize;
                for j in 0..ns {
                    prev_state
                        .sig_flags
                        .insert((entry.band, entry.line, j as u32), 0);
                }
            }
            // Drop unused: line_index isn't needed here; it's used by
            // the bitplane-count decoder below.
            let _ = line_index;
        }
        reader.align_to_byte();
        total_consumed += reader.bytes_consumed();
    }

    // === Bitplane-count sub-packet =======================================
    let lcnt = packet.lcnt as usize;
    {
        let buf_cnt = buf
            .get(total_consumed..total_consumed + lcnt)
            .ok_or_else(|| {
                Error::invalid("jpegxs entropy: packet body truncated at bitplane-count sub-packet")
            })?;
        let mut reader = BitReader::new(buf_cnt);
        for entry in &layout.entries {
            let bi = entry.band as usize;
            let band = &geom.bands[bi];
            if !band.exists {
                continue;
            }
            let line_index = (entry.line - band.l0) as usize;
            let ncg = geom.ncg(bi) as usize;
            let t = truncation[bi] as i32;
            let dpb = precinct.d[bi];
            let coef = &mut prev_state.coefficients[bi];

            // Bitplane-count upper bound 2^Br − 1. Tables C.12/C.13/C.14
            // all require 0 ≤ M[p,λ,b,g] ≤ (2^Br − 1) regardless of the
            // decode mode (raw / no-prediction / vertical). `Br` is a
            // u(4) field whose only conformant value is 4 (Table A.7),
            // so this caps M at 15, but the bound is computed from `Br`
            // to stay exact for any future widening. The cap is `i32`
            // so the VLC paths (which may produce a negative
            // mtop + Δm) reject under- and over-flow uniformly.
            //
            // The bound additionally intersects with the 32-bit
            // quantization-index representation: a magnitude with `M`
            // bitplanes reconstructs to `(v << T) + r < 2^M` (Annex
            // D.2), so `M ≤ 31` is the largest count the coefficient
            // pipeline represents exactly. A conforming encoder cannot
            // reach it — quantization indices derive from `Bw ≤ 20`-bit
            // nominal wavelet data (Table A.8) whose 5/3 dynamic-range
            // growth stays well below 31 bits — but with `Br = 8` the
            // Table C.12 syntactic range reaches 255, and a malformed
            // count of e.g. 200 would drive the per-plane magnitude
            // accumulation `d << plane` past the 32-bit coefficient
            // width (fuzz-surfaced, round 438).
            let m_max: i32 = {
                let br_cap: i32 = if geom.br >= 8 {
                    255
                } else {
                    (1i32 << geom.br) - 1
                };
                br_cap.min(31)
            };

            // Vertical predictor source line.
            // sy is taken to be 1 for round 3 (single-component
            // fixture). When λ - sy < L0, the predictor would come
            // from the previous precinct, which round 3 doesn't
            // support — that case is gated below.
            let sy: u16 = 1;

            if packet.dr == 1 {
                // Raw mode: Br bits per code group (Table C.12).
                for g in 0..ncg {
                    let m = reader.read_bits(geom.br)? as i32;
                    if m > m_max {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: raw bitplane count {m} exceeds the maximum {m_max} (Table C.12 min(2^Br - 1, 31))"
                        )));
                    }
                    coef.m[line_index * ncg + g] = m as u8;
                }
            } else if (dpb & 1) == 0 {
                // No-prediction VLC, Table C.14.
                let mtop = t;
                for g in 0..ncg {
                    let sig_group = g / geom.ss as usize;
                    let z = if (dpb & 2) != 0 {
                        // Significance coding enabled. Z = 1 flags the
                        // group as all-insignificant (Table C.5).
                        prev_state
                            .sig_flags
                            .get(&(entry.band, entry.line, sig_group as u32))
                            .copied()
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    let delta_m = if (dpb & 2) == 0 || z == 0 {
                        vlc(&mut reader, mtop, t)?
                    } else {
                        // Insignificant group (Z = 1) → Δm = 0
                        // (Table C.14 explicitly sets Δm = 0; the
                        // bitplane count = T because mtop = T).
                        0
                    };
                    let m = mtop + delta_m;
                    if !(0..=m_max).contains(&m) {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: decoded M[p,λ,b,g] = {m} outside 0..={m_max} (Table C.14 min(2^Br-1, 31))"
                        )));
                    }
                    coef.m[line_index * ncg + g] = m as u8;
                }
            } else {
                // Vertical prediction VLC, Table C.13.
                //
                // `entry.line` is in band-grid units stepping by one
                // (the walker converts the spec image-grid `λ` to
                // `λ / sy[i]`), so the spec predecessor step `sy`
                // (Table C.11) is one band-grid line here.
                let first_line_of_band = entry.line < band.l0 + sy;
                // Cross-precinct predictor (Table C.11, first branch):
                // when the current line is the top line of the band in
                // this precinct, `Mtop` / `Ttop` come from the precinct
                // directly above (`p − Np,x`) — its last band line and
                // its `T[p−Np,x,b]`. §C.6.1 / §C.6.3 forbid vertical
                // prediction at the topmost precinct of a slice, so a
                // missing predecessor here is a malformed codestream.
                let xpred = if first_line_of_band {
                    let pt = top.ok_or_else(|| {
                        Error::invalid(
                            "jpegxs entropy: vertical prediction selected at the top line of the topmost precinct of a slice (forbidden by ISO/IEC 21122-1 C.6.1/C.6.3)",
                        )
                    })?;
                    Some(pt)
                } else {
                    None
                };
                let prev_line_index = if first_line_of_band {
                    0usize
                } else {
                    ((entry.line - sy) - band.l0) as usize
                };
                // Ttop[p,b]: T[p,b] for predecessors inside the same
                // precinct, T[p−Np,x,b] when predicting across the
                // precinct boundary (Table C.11).
                let ttop = match xpred {
                    Some(pt) => *pt.t.get(bi).unwrap_or(&0) as i32,
                    None => t,
                };
                let teff = t.max(ttop);
                for g in 0..ncg {
                    let sig_group = g / geom.ss as usize;
                    let z = if (dpb & 2) != 0 {
                        // Z = 1 flags the group as all-insignificant
                        // (Table C.5).
                        prev_state
                            .sig_flags
                            .get(&(entry.band, entry.line, sig_group as u32))
                            .copied()
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    let m_above = match xpred {
                        Some(pt) => {
                            *pt.last_m.get(bi).and_then(|row| row.get(g)).unwrap_or(&0) as i32
                        }
                        None => coef.m[prev_line_index * ncg + g] as i32,
                    };
                    let mtop = m_above.max(teff);
                    let delta_m = if (dpb & 2) == 0 || z == 0 {
                        vlc(&mut reader, mtop, t)?
                    } else if geom.rm == 0 {
                        0
                    } else {
                        // Rm == 1 → Δm = T - mtop (so M = T).
                        t - mtop
                    };
                    let m = mtop + delta_m;
                    if !(0..=m_max).contains(&m) {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: decoded vertical M[p,λ,b,g] = {m} outside 0..={m_max} (Table C.13 min(2^Br-1, 31))"
                        )));
                    }
                    coef.m[line_index * ncg + g] = m as u8;
                }
            }
        }
        reader.align_to_byte();
        let body_consumed = reader.bytes_consumed();
        if body_consumed > lcnt {
            return Err(Error::invalid(format!(
                "jpegxs entropy: bitplane-count sub-packet read {body_consumed} > Lcnt = {lcnt}"
            )));
        }
        // Filler bytes follow up to Lcnt.
        total_consumed += lcnt;
    }

    // === Data sub-packet =================================================
    //
    // Table C.8 transmits coefficients in whole code groups of `Ng`, so
    // a band whose `Wpb` is not a multiple of `Ng` carries "meaningless"
    // tail coefficients past its right edge in the last code group. The
    // decoder discards their magnitudes, but Table C.9 NOTE 2 makes the
    // sign sub-packet include a sign bit for every such tail coefficient
    // whose transmitted magnitude is non-zero — so the tail magnitudes
    // must be tracked (per packet entry) for the sign pass to stay
    // bit-aligned. `tail_v[e]` holds the decoded magnitudes of entry
    // `e`'s positions `Wpb .. Ncg*Ng` (at most `Ng - 1` of them).
    let ldat = packet.ldat as usize;
    let mut tail_v: Vec<Vec<u32>> = vec![Vec::new(); layout.entries.len()];
    {
        let buf_dat = buf
            .get(total_consumed..total_consumed + ldat)
            .ok_or_else(|| {
                Error::invalid("jpegxs entropy: packet body truncated at data sub-packet")
            })?;
        let mut reader = BitReader::new(buf_dat);
        for (e, entry) in layout.entries.iter().enumerate() {
            let bi = entry.band as usize;
            let band = &geom.bands[bi];
            if !band.exists {
                continue;
            }
            let line_index = (entry.line - band.l0) as usize;
            let ncg = geom.ncg(bi) as usize;
            let t = truncation[bi] as u32;
            let coef = &mut prev_state.coefficients[bi];
            let line_offset = line_index * (band.wpb as usize);
            let tail_len = (ncg * geom.ng as usize).saturating_sub(band.wpb as usize);
            tail_v[e] = vec![0u32; tail_len];
            for g in 0..ncg {
                let m = coef.m[line_index * ncg + g] as u32;
                // Reset magnitudes for this group (per Table C.8:
                // "v[p,λ,b,Ng×g+k] = 0").
                for k in 0..geom.ng as usize {
                    let xpos = g * geom.ng as usize + k;
                    if xpos < band.wpb as usize {
                        coef.v[line_offset + xpos] = 0;
                    }
                }
                if m > t {
                    if geom.fs == 0 {
                        // Signs are interleaved into the data sub-packet.
                        for k in 0..geom.ng as usize {
                            let s = reader.read_bit()?;
                            let xpos = g * geom.ng as usize + k;
                            if xpos < band.wpb as usize {
                                coef.s[line_offset + xpos] = s;
                            }
                        }
                    }
                    // M - T bitplanes, MSB-first per spec ("for(i =
                    // M-T-1; i >= 0; i = i - 1)").
                    let nplanes = m - t;
                    for plane in (0..nplanes).rev() {
                        for k in 0..geom.ng as usize {
                            let d = reader.read_bit()? as u32;
                            let xpos = g * geom.ng as usize + k;
                            if xpos < band.wpb as usize {
                                coef.v[line_offset + xpos] |= d << plane;
                            } else {
                                // Meaningless tail coefficient: the
                                // magnitude is discarded, but whether it
                                // is non-zero decides a sign bit in the
                                // Fs = 1 sign sub-packet (Table C.9
                                // NOTE 2).
                                tail_v[e][xpos - band.wpb as usize] |= d << plane;
                            }
                        }
                    }
                }
            }
        }
        reader.align_to_byte();
        let body_consumed = reader.bytes_consumed();
        if body_consumed > ldat {
            return Err(Error::invalid(format!(
                "jpegxs entropy: data sub-packet read {body_consumed} > Ldat = {ldat}"
            )));
        }
        total_consumed += ldat;
    }

    // === Sign sub-packet =================================================
    if geom.fs == 1 {
        let lsgn = packet.lsgn as usize;
        let buf_sgn = buf
            .get(total_consumed..total_consumed + lsgn)
            .ok_or_else(|| {
                Error::invalid("jpegxs entropy: packet body truncated at sign sub-packet")
            })?;
        let mut reader = BitReader::new(buf_sgn);
        for (e, entry) in layout.entries.iter().enumerate() {
            let bi = entry.band as usize;
            let band = &geom.bands[bi];
            if !band.exists {
                continue;
            }
            let line_index = (entry.line - band.l0) as usize;
            let ncg = geom.ncg(bi) as usize;
            let coef = &mut prev_state.coefficients[bi];
            let line_offset = line_index * (band.wpb as usize);
            for g in 0..ncg {
                for k in 0..geom.ng as usize {
                    let xpos = g * geom.ng as usize + k;
                    if xpos >= band.wpb as usize {
                        // Meaningless tail coefficient of the last code
                        // group (Wpb not a multiple of Ng). Table C.9
                        // NOTE 2: the sign sub-packet carries a sign bit
                        // for it whenever its transmitted magnitude is
                        // non-zero. Consume and discard the bit so the
                        // following bands' signs stay aligned.
                        if tail_v[e][xpos - band.wpb as usize] != 0 {
                            let _ = reader.read_bit()?;
                        }
                        continue;
                    }
                    if coef.v[line_offset + xpos] != 0 {
                        coef.s[line_offset + xpos] = reader.read_bit()?;
                    }
                }
            }
        }
        reader.align_to_byte();
        let body_consumed = reader.bytes_consumed();
        if body_consumed > lsgn {
            return Err(Error::invalid(format!(
                "jpegxs entropy: sign sub-packet read {body_consumed} > Lsgn = {lsgn}"
            )));
        }
        total_consumed += lsgn;
    }

    // The PacketDecode `bands` is a copy of the running state for
    // callers that want it. The state mutation is the source of truth.
    Ok(PacketDecode {
        bands: prev_state.coefficients.clone(),
        bytes_consumed: total_consumed,
    })
}

/// Mutable per-precinct state carried across packet bodies. Round 3
/// uses this only to stash the decoded coefficients (so the test fixture
/// can read them out) and the significance flags (so the bitplane-
/// count decoder can gate against them within the same packet).
#[derive(Debug, Default)]
pub struct PrecinctState {
    /// Per-band coefficient buffers; one entry per band id.
    pub coefficients: Vec<BandCoefficients>,
    /// `Z[p,λ,b,j]` indexed by `(band, line, sig_group)`.
    pub sig_flags: HashMap<(u16, u16, u32), u8>,
}

/// Vertical-prediction predecessor carried from the precinct directly
/// above (same precinct column) into the precinct below.
///
/// Annex C.6.3 Table C.11: when a vertical-prediction packet is decoded
/// at the top line of a band in precinct `p`
/// (`λ − sy < L0[p,b]`), the bitplane-count predictor and the
/// truncation-position predictor come from precinct `p − Np,x`:
///
/// * `Mtop[p,λ,b,g] = M[p−Np,x, L1[p,b]−sy, b, g]` — the bitplane
///   counts of the **last line** of band `b` in the precinct above.
/// * `Ttop[p,b] = T[p−Np,x, b]` — the truncation position of band `b`
///   in the precinct above.
///
/// The codestream constraint in §C.6.1 / §C.6.3 forbids vertical
/// prediction at the topmost lines of the topmost precinct of a slice
/// or the image, so within a slice this predecessor always exists when
/// it is referenced. The decoder builds one `PrecinctTop` per decoded
/// precinct and caches it per precinct column for the row below.
#[derive(Debug, Clone, Default)]
pub struct PrecinctTop {
    /// Per-band bitplane counts of the last decoded line of the band
    /// (`M[p−Np,x, L1[p,b]−sy, b, g]`), indexed by band id then code
    /// group `g`. Empty for non-existent bands.
    pub last_m: Vec<Vec<u8>>,
    /// Per-band truncation position `T[p−Np,x, b]`, indexed by band id.
    pub t: Vec<u8>,
}

impl PrecinctTop {
    /// Capture the vertical-prediction predecessor of a precinct that
    /// has just been fully decoded into `state`, so the precinct
    /// directly below (same column) can predict from it per Table C.11.
    ///
    /// For each band the last decoded line is `L1[p,b] − 1 − L0[p,b]`
    /// in the band's local (band-grid) line indexing — the walker emits
    /// `entry.line` in band-grid units stepping by one, so the spec's
    /// `L1[p,b] − sy` image-grid predecessor reduces to the band's
    /// last stored line. `T[p,b]` is the precinct's per-band
    /// truncation position.
    pub fn capture(
        geom: &PrecinctGeometry,
        precinct: &PrecinctHeader,
        state: &PrecinctState,
    ) -> Self {
        let truncation = precinct_truncation(geom, precinct);
        let mut last_m: Vec<Vec<u8>> = Vec::with_capacity(geom.bands.len());
        for (bi, band) in geom.bands.iter().enumerate() {
            if !band.exists || band.l1 <= band.l0 {
                last_m.push(Vec::new());
                continue;
            }
            let ncg = geom.ncg(bi) as usize;
            let last_line = (band.l1 - band.l0 - 1) as usize;
            let coef = state.coefficients.get(bi);
            let row = match coef {
                Some(c) if c.m.len() >= (last_line + 1) * ncg => {
                    c.m[last_line * ncg..last_line * ncg + ncg].to_vec()
                }
                _ => vec![0u8; ncg],
            };
            last_m.push(row);
        }
        PrecinctTop {
            last_m,
            t: truncation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::precinct_header::PrecinctHeader;
    use crate::entropy::{BandGeometry, PacketEntry, PacketLayout, PrecinctGeometry};

    /// Hand-built fixture: 1 precinct, 1 band, 1 packet, single
    /// component. 16 coefficients per line → Ncg=4, Ns=1. T=0,
    /// D=0 (no prediction, no significance), Dr=0. Bitplane counts
    /// are encoded with VLC of (mtop=0, T=0) → unary alphabet → x.
    /// Code groups have bitplane counts [3, 0, 1, 2].
    /// Magnitudes per code group:
    ///   g=0, M=3 → 4 coefs with values
    ///       [5, 0, 7, 1] = bin [101, 000, 111, 001].
    ///       Signs (Fs=0): [+, +, -, -] = [0, 0, 1, 1].
    ///   g=1, M=0 → no data emitted, all zero.
    ///   g=2, M=1 → 4 coefs with values [1, 0, 1, 0]; signs [+,*,+,*]
    ///       (only non-zeros have signs; * = ignored).
    ///   g=3, M=2 → 4 coefs [3, 2, 1, 0]; signs [-, +, +, *].
    #[test]
    fn round_trip_handbuilt_single_packet() {
        // ---- Geometry ----
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 16,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1, // unused by the body decoder
            q: 0,
            r: 0,
            d: vec![0], // no prediction, no significance
            header_bytes: 0,
        };
        // ---- Build the body byte-by-byte ----
        // Significance sub-packet: D&2 == 0, so no bits — but the
        // sub-packet still pads to a byte boundary. Empty payload =>
        // 0 bytes consumed.
        //
        // Bitplane-count sub-packet:
        //   VLC of (0,0): θ=0; x>0 → unary → returns x. So encode each
        //   M with M unary one-bits + a 0 comma bit.
        //     M=3 → "1110"
        //     M=0 → "0"
        //     M=1 → "10"
        //     M=2 → "110"
        //   Total: "1110" "0" "10" "110" = "1110010110"  (10 bits)
        //   Padded to byte: "11100101 10000000" = 0xE5 0x80
        //   Lcnt = 2 bytes.
        //
        // Data sub-packet:
        //   g=0, M=3, T=0 → emit Ng=4 sign bits then (M-T)*Ng=12 bits
        //   of magnitudes, MSB-first. Signs [0,0,1,1] = "0011".
        //   Magnitudes (MSB plane first):
        //     plane 2 (bit 4 of "MSB" = highest): coef bits
        //       [bit2(5)=1, bit2(0)=0, bit2(7)=1, bit2(1)=0] = "1010"
        //     plane 1: [bit1(5)=0, bit1(0)=0, bit1(7)=1, bit1(1)=0] = "0010"
        //     plane 0: [bit0(5)=1, bit0(0)=0, bit0(7)=1, bit0(1)=1] = "1011"
        //   Group 0 contributes "0011 1010 0010 1011" = 16 bits.
        //
        //   g=1, M=0 → nothing emitted.
        //
        //   g=2, M=1, T=0 → 4 sign bits then 1 plane × 4 bits.
        //     Signs (only meaningful for non-zero coefs, but spec
        //     unconditionally emits Ng signs when Fs=0 — see Table C.8).
        //     Coefs [1, 0, 1, 0]. Signs we'll write [0, 0, 0, 0] = "0000".
        //     plane 0: [bit0(1)=1, bit0(0)=0, bit0(1)=1, bit0(0)=0] = "1010"
        //   Group 2 contributes "0000 1010" = 8 bits.
        //
        //   g=3, M=2, T=0 → 4 signs + 8 mag bits.
        //     Coefs [3, 2, 1, 0]. Signs [1, 0, 0, 0] = "1000".
        //     plane 1: [bit1(3)=1, bit1(2)=1, bit1(1)=0, bit1(0)=0] = "1100"
        //     plane 0: [bit0(3)=1, bit0(2)=0, bit0(1)=1, bit0(0)=0] = "1010"
        //   Group 3 contributes "1000 1100 1010" = 12 bits.
        //
        //   Total data bits: 16 + 0 + 8 + 12 = 36 bits → 5 bytes after
        //   padding.
        //
        //   Bit string: 0011 1010 0010 1011  0000 1010  1000 1100 1010
        //   Pack: 0011_1010 0010_1011 0000_1010 1000_1100 1010_0000
        //         = 0x3A 0x2B 0x0A 0x8C 0xA0
        //   Ldat = 5 bytes.
        //
        // Lsgn = 0 (Fs=0).

        // Significance sub-packet: 0 bytes (no bits, no padding needed).
        // Bitplane-count: 2 bytes (0xE5, 0x80).
        // Data: 5 bytes (0x3A, 0x2B, 0x0A, 0x8C, 0xA0).
        let body: Vec<u8> = vec![0xE5, 0x80, 0x3A, 0x2B, 0x0A, 0x8C, 0xA0];

        let packet = PacketHeader {
            dr: 0,
            ldat: 5,
            lcnt: 2,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect("packet body decode");
        // 0 (sig) + 2 (lcnt) + 5 (ldat) = 7 bytes consumed.
        assert_eq!(dec.bytes_consumed, 7);

        let band = &dec.bands[0];
        assert_eq!(band.m, vec![3, 0, 1, 2]);
        // v values across all 16 coefficients:
        let expected_v = vec![5u32, 0, 7, 1, 0, 0, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0];
        assert_eq!(band.v, expected_v);
        // Signs: 0 for positive / zero, 1 for negative.
        // Group 0: [0,0,1,1]. Group 1: [0,0,0,0] (M=0, never read).
        // Group 2: [0,0,0,0] (we wrote zeros). Group 3: [1,0,0,0].
        let expected_s = vec![0u8, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(band.s, expected_s);
    }

    /// Same hand-built fixture but with `Fs=1` (separate sign
    /// sub-packet). Verifies the sign sub-packet only contributes one
    /// bit per non-zero coefficient.
    #[test]
    fn round_trip_handbuilt_separate_signs() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 16,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 1,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0],
            header_bytes: 0,
        };

        // Bitplane counts identical to the Fs=0 test: M = [3, 0, 1, 2].
        // Bitplane-count sub-packet: same 2 bytes 0xE5 0x80.
        //
        // Data sub-packet (Fs=1, no signs in data):
        //   g=0, M=3 → 12 bits MSB-first:
        //     plane 2: [1,0,1,0]; plane 1: [0,0,1,0]; plane 0: [1,0,1,1]
        //     12 bits "1010 0010 1011".
        //   g=1, M=0 → nothing.
        //   g=2, M=1 → 4 bits:
        //     plane 0: [1,0,1,0] = "1010".
        //   g=3, M=2 → 8 bits:
        //     plane 1: [1,1,0,0]; plane 0: [1,0,1,0] = "1100 1010".
        //   Total = 12 + 4 + 8 = 24 bits = 3 bytes.
        //   Pack: 1010_0010 1011_1010 1100_1010 = 0xA2 0xBA 0xCA
        //   Ldat = 3.
        //
        // Sign sub-packet:
        //   Non-zero coefs across all groups:
        //     g=0: 5(+), 7(-), 1(-) → 3 non-zero → signs 0, 1, 1
        //     g=1: none
        //     g=2: 1(+), 1(+) → 2 non-zero → signs 0, 0
        //     g=3: 3(-), 2(+), 1(+) → 3 non-zero → signs 1, 0, 0
        //   Total bits: 3+0+2+3 = 8 → exactly one byte.
        //   Bit string "0 1 1 0 0 1 0 0" = 0x64.
        //   Lsgn = 1.

        // Bitplane-count: 0xE5 0x80 (same as Fs=0 test).
        // Data: 0xA2 0xBA 0xCA (3 bytes).
        // Sign: 0x64 (1 byte).
        let body: Vec<u8> = vec![0xE5, 0x80, 0xA2, 0xBA, 0xCA, 0x64];

        let packet = PacketHeader {
            dr: 0,
            ldat: 3,
            lcnt: 2,
            lsgn: 1,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect("packet body decode (Fs=1)");
        // 0 (sig) + 2 + 3 + 1 = 6 bytes.
        assert_eq!(dec.bytes_consumed, 6);

        let band = &dec.bands[0];
        assert_eq!(band.m, vec![3, 0, 1, 2]);
        let expected_v = vec![5u32, 0, 7, 1, 0, 0, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0];
        assert_eq!(band.v, expected_v);
        // Signs only set on non-zero coefficients.
        // g=0: idx 0=+, 2=-, 3=- → s = [0, ?, 1, 1].
        // g=1: all zero → s = [0,0,0,0].
        // g=2: idx 0,2 non-zero → s = [0, ?, 0, ?].
        // g=3: idx 0,1,2 non-zero → s = [1, 0, 0, ?].
        // The ? entries default to 0 since we never wrote them.
        let expected_s = vec![0u8, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(band.s, expected_s);
    }

    /// Significance sub-packet exercises the gating path: with D=2
    /// (significance enabled, no prediction), an insignificant group
    /// must not consume any bits in the bitplane-count sub-packet.
    #[test]
    fn significance_gating_skips_bitplane_count() {
        // Wpb=32 → Ncg=8, Ns=1 (one sig group covering all 8 code
        // groups). With Ns=1 we mark the whole line as insignificant
        // by writing one 0 bit, and no bitplane-count VLC follows.
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 32,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0b10], // significance enabled, no prediction
            header_bytes: 0,
        };
        // Significance sub-packet: 1 bit = 1 (Z = 1 flags the whole
        // significance group as all-insignificant per Table C.5).
        // Padded to a byte → 1 byte 0x80 (MSB-first).
        // Bitplane-count sub-packet: 0 bits used (Δm=0 inferred for
        // every group, which gives M = mtop = T = 0). When no bits
        // are written the padding round-up gives 0 bytes, so Lcnt=0
        // is legal.
        // Ldat = 0 → no data bytes (M=0 ≤ T=0 for every group).
        let body: Vec<u8> = vec![0x80];

        let packet = PacketHeader {
            dr: 0,
            ldat: 0,
            lcnt: 0,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect("decode insignificant packet");
        assert_eq!(dec.bytes_consumed, 1);
        let band = &dec.bands[0];
        assert!(band.m.iter().all(|&m| m == 0));
        assert!(band.v.iter().all(|&v| v == 0));
    }

    /// Raw-mode override: Dr=1 makes the bitplane-count decoder read
    /// `Br` bits per code group, regardless of D[p,b].
    #[test]
    fn raw_mode_override_reads_br_bits_per_group() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 16,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 1,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0b11], // would be vertical+sig if not overridden
            header_bytes: 0,
        };
        // Significance sub-packet skipped because Dr=1 (Table C.5).
        // Bitplane-count sub-packet (raw): Br=4 bits per group, 4
        // groups → 16 bits = 2 bytes. M = [3, 0, 1, 2]:
        //   "0011 0000 0001 0010" = 0x30 0x12.
        // Data sub-packet identical to the Fs=0 round-trip test: 5 bytes.
        let body: Vec<u8> = vec![0x30, 0x12, 0x3A, 0x2B, 0x0A, 0x8C, 0xA0];

        let packet = PacketHeader {
            dr: 1,
            ldat: 5,
            lcnt: 2,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect("decode raw-mode packet");
        // No significance sub-packet bits consumed since Dr=1.
        assert_eq!(dec.bytes_consumed, 7);
        let band = &dec.bands[0];
        assert_eq!(band.m, vec![3, 0, 1, 2]);
        let expected_v = vec![5u32, 0, 7, 1, 0, 0, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0];
        assert_eq!(band.v, expected_v);
    }

    /// A VLC-coded bitplane count that exceeds `2^Br − 1` must be
    /// rejected. Tables C.13 and C.14 both require
    /// `0 ≤ M[p,λ,b,g] ≤ (2^Br − 1)`; the raw path (Table C.12) already
    /// enforced it, but the no-prediction / vertical VLC paths used to
    /// only clamp to the byte range. With `Br = 4` the valid maximum is
    /// 15, so a unary VLC codeword that decodes to `M = 16` is malformed.
    ///
    /// Geometry: 1 band, `wpb = 16` → `Ncg = 4`, `T = 0`, `D = 0`
    /// (no prediction, no significance), `Dr = 0`. `vlc(mtop=0, T=0)`
    /// is the unary alphabet returning the count of leading 1-bits, so
    /// 16 ones followed by a 0 comma decodes the first code group to
    /// `M = 16 > 15`.
    #[test]
    fn nopred_vlc_bitplane_count_above_2pow_br_minus_1_rejected() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 16,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0], // no prediction, no significance
            header_bytes: 0,
        };
        // Bitplane-count sub-packet: first code group is 16 unary ones
        // then a 0 comma = "1111111111111111 0" (17 bits). Padded to
        // 3 bytes: 0xFF 0xFF 0x00. Lcnt = 3.
        let body: Vec<u8> = vec![0xFF, 0xFF, 0x00];
        let packet = PacketHeader {
            dr: 0,
            ldat: 0,
            lcnt: 3,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let err = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect_err("M = 16 must be rejected as out of 0..=2^Br-1");
        let msg = format!("{err}");
        assert!(
            msg.contains("2^Br-1") && msg.contains("16"),
            "unexpected error message: {msg}"
        );
    }

    /// A raw-mode bitplane count above the 32-bit representability cap
    /// must be rejected (fuzz-surfaced, round 438). With `Br = 8` the
    /// Table C.12 syntactic range reaches 255, but a magnitude with
    /// `M > 31` bitplanes cannot be represented by the 32-bit
    /// quantization-index pipeline (`(v << T) + r < 2^M`, Annex D.2) —
    /// and a conforming encoder cannot produce one from `Bw ≤ 20`-bit
    /// nominal wavelet data. Without the cap the per-plane magnitude
    /// accumulation `d << plane` overflows.
    ///
    /// Geometry: 1 band, `wpb = 4` → `Ncg = 1`, raw mode (`Dr = 1`),
    /// count byte `200`, and a data sub-packet long enough that the
    /// decoder would reach the overflowing shift if the count survived.
    #[test]
    fn raw_bitplane_count_above_32bit_representability_rejected() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 4,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 8,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0],
            header_bytes: 0,
        };
        // Bitplane-count sub-packet (raw): one Br=8 code = 200.
        // Followed by a generously sized data sub-packet of zeros.
        let mut body: Vec<u8> = vec![200];
        body.extend_from_slice(&[0u8; 128]);
        let packet = PacketHeader {
            dr: 1,
            ldat: 128,
            lcnt: 1,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let err = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect_err("M = 200 must be rejected above the 32-bit representability cap");
        let msg = format!("{err}");
        assert!(msg.contains("200"), "unexpected error message: {msg}");
    }

    /// Cross-precinct vertical prediction (Annex C.6.3 Table C.11):
    /// `D[p,b] & 1 == 1` selected at the FIRST line of the band in the
    /// precinct (`λ − sy < L0[p,b]`). The bitplane-count predictor
    /// `Mtop[p,λ,b,g]` and the truncation predictor `Ttop[p,b]` come
    /// from the precinct directly above (`p − Np,x`), supplied as a
    /// [`PrecinctTop`].
    ///
    /// One band, `wpb = 8` → `Ncg = 2`. `T[p,b] = 0`, `Ttop = 0` so the
    /// effective truncation `teff = 0`. The predecessor's last-line
    /// counts are `Mtop = [2, 0]`. For each code group:
    ///   g=0: `mtop = max(Mtop=2, teff=0) = 2`; θ = max(2−0)=2; VLC
    ///        codeword "0" (xi=0) → Δm=0 → `M = 2`.
    ///   g=1: `mtop = max(Mtop=0, teff=0) = 0`; θ=0; "0" → Δm=0 → M=0.
    /// So `M = [2, 0]`. The bitplane-count sub-packet is "00" → 0x00,
    /// `Lcnt = 1`.
    ///
    /// Data sub-packet: g=0 has `M=2 > T=0`, so `Ng=4` signs +
    /// `(M−T)·Ng = 8` magnitude bits. Coefficients `[1, 0, 0, 0]`,
    /// signs `[0,0,0,0]`:
    ///   signs "0000"; plane 1 "0000"; plane 0 "1000" → 12 bits →
    ///   0x00 0x80, `Ldat = 2`. g=1 (M=0) emits nothing.
    #[test]
    fn vertical_prediction_across_precincts() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 8,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        // D[0] = 1 → vertical prediction VLC.
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![1],
            header_bytes: 0,
        };
        // Predecessor from the precinct directly above: last-line counts
        // Mtop = [2, 0], Ttop = T[above,0] = 0.
        let top = PrecinctTop {
            last_m: vec![vec![2u8, 0]],
            t: vec![0u8],
        };

        // Bitplane-count: "00" → 0x00 (Lcnt=1).
        // Data: 0x00 0x80 (Ldat=2).
        let body: Vec<u8> = vec![0x00, 0x00, 0x80];
        let packet = PacketHeader {
            dr: 0,
            ldat: 2,
            lcnt: 1,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };

        let mut state = PrecinctState::default();
        let dec = decode_packet_body(
            &body,
            &geom,
            &precinct,
            &packet,
            &layout,
            &mut state,
            Some(&top),
        )
        .expect("cross-precinct vertical prediction decode");
        assert_eq!(dec.bytes_consumed, 3);
        let band = &dec.bands[0];
        // M predicted from the precinct above: [2, 0].
        assert_eq!(band.m, vec![2, 0]);
        // Magnitudes: g=0 coefs [1,0,0,0]; g=1 all zero.
        assert_eq!(band.v, vec![1u32, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// The effective truncation `teff = max(T[p,b], Ttop)` and the
    /// predictor `mtop = max(Mtop, teff)` must both honour the
    /// predecessor's `Ttop` (Table C.11 / C.13). Here `T[p,b] = 0` but
    /// the precinct above had `Ttop = 3`, and `Mtop = [1, 0]`.
    ///   g=0: teff = max(0, 3) = 3; mtop = max(1, 3) = 3; θ = max(3−0)=3;
    ///        VLC "0" → Δm=0 → M = 3.
    ///   g=1: teff = 3; mtop = max(0, 3) = 3; θ=3; "0" → Δm=0 → M = 3.
    /// So `M = [3, 3]`. Count sub-packet "00" → 0x00, Lcnt=1.
    /// Data: both groups have `M=3 > T=0` → each emits 4 signs +
    /// 12 magnitude bits. Coefs all zero, signs all zero → 16 bits per
    /// group, 32 bits total → 4 bytes 0x00 0x00 0x00 0x00, Ldat=4.
    #[test]
    fn vertical_prediction_across_precincts_with_ttop() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 8,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![1],
            header_bytes: 0,
        };
        let top = PrecinctTop {
            last_m: vec![vec![1u8, 0]],
            t: vec![3u8],
        };
        // Count "00" → 0x00 (Lcnt=1). Data: 32 zero bits → 0x00*4 (Ldat=4).
        let body: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00, 0x00];
        let packet = PacketHeader {
            dr: 0,
            ldat: 4,
            lcnt: 1,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(
            &body,
            &geom,
            &precinct,
            &packet,
            &layout,
            &mut state,
            Some(&top),
        )
        .expect("cross-precinct vertical prediction with Ttop");
        assert_eq!(band_m(&dec), vec![3, 3]);
    }

    /// Run mode `Rm = 1` (Table A.12 / C.6.5): an insignificant significance
    /// group in the vertical-prediction bitplane-count mode reconstructs to
    /// `M = T[p,b]` — `Δm = T − mtop` — **regardless of the predictor**,
    /// unlike `Rm = 0` which infers `M = mtop`. The data sub-packet then
    /// omits the group (`M = T` is not `> T`).
    ///
    /// Geometry: 1 band, `wpb = 8` → `Ncg = 2`, `Ss = 8` → `Ns = 1` (one
    /// significance group covering both code groups), `D = 3` (significance
    /// with vertical prediction), `Dr = 0`. The precinct above carries the
    /// predictor `Mtop = (5, 4)` (both `> T = 0`); under `Rm = 1` the whole
    /// group is signalled insignificant (`Z = 1`) yet still collapses to
    /// `M = 0`.
    #[test]
    fn rm1_vertical_insignificant_group_reconstructs_to_t() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 8,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 1,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        // D = 3 → significance + vertical prediction.
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0b11],
            header_bytes: 0,
        };
        // Predecessor last-line counts both exceed T → mtop = [5, 4].
        let top = PrecinctTop {
            last_m: vec![vec![5u8, 4]],
            t: vec![0u8],
        };
        // Significance sub-packet: 1 bit = 1 (Z = 1, insignificant group),
        // padded to 0x80. Count sub-packet: no VLC (Δm implied). Data: none.
        let body: Vec<u8> = vec![0x80];
        let packet = PacketHeader {
            dr: 0,
            ldat: 0,
            lcnt: 0,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(
            &body,
            &geom,
            &precinct,
            &packet,
            &layout,
            &mut state,
            Some(&top),
        )
        .expect("Rm=1 insignificant vertical-prediction decode");
        assert_eq!(dec.bytes_consumed, 1);
        // Rm=1: both code groups collapse to M = T = 0 (not the [5, 4]
        // predictor that Rm=0 would have inferred).
        assert_eq!(band_m(&dec), vec![0, 0]);
        assert!(dec.bands[0].v.iter().all(|&v| v == 0));
    }

    /// Companion contrast: the identical `Mtop = [5, 4]` predecessor and an
    /// insignificant group under `Rm = 0` instead reconstructs `M = mtop`,
    /// so the data sub-packet carries both groups' coefficients. This pins
    /// the `Rm` branch as behaviourally load-bearing (Table C.13).
    #[test]
    fn rm0_vertical_insignificant_group_reconstructs_to_mtop() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 8,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0b11],
            header_bytes: 0,
        };
        let top = PrecinctTop {
            last_m: vec![vec![5u8, 4]],
            t: vec![0u8],
        };
        // Significance byte 0x80 (Z = 1). Data sub-packet (Fs=0, Table C.8):
        // each group with M>T emits Ng signs + (M−T)·Ng magnitude bits =
        // Ng·(1 + M − T). g0: 4·(1+5) = 24 bits; g1: 4·(1+4) = 20 bits;
        // total 44 bits → 6 bytes (all-zero coefficients → all-zero bits).
        let body: Vec<u8> = vec![0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let packet = PacketHeader {
            dr: 0,
            ldat: 6,
            lcnt: 0,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(
            &body,
            &geom,
            &precinct,
            &packet,
            &layout,
            &mut state,
            Some(&top),
        )
        .expect("Rm=0 insignificant vertical-prediction decode");
        assert_eq!(band_m(&dec), vec![5, 4]);
    }

    /// Table C.9 NOTE 2: a band whose `Wpb` is not a multiple of `Ng`
    /// transmits "meaningless" tail coefficients in its last code group,
    /// and the `Fs = 1` sign sub-packet carries a sign bit for every such
    /// tail coefficient whose transmitted magnitude is non-zero. A decoder
    /// that skips those bits desynchronises every following sign in the
    /// same sub-packet (ISO/IEC 21122-4 stream 64 hits exactly this: the
    /// shifted bit only becomes visible at the next negative sign, in the
    /// 4th component).
    ///
    /// Fixture: two bands in one packet. Band 0 has `Wpb = 7` (`Ncg = 2`,
    /// `Ng = 4` → one tail position at xpos 7) with a **non-zero tail
    /// magnitude** on the wire; band 1 (`Wpb = 4`) follows with a negative
    /// coefficient. `T = 0`, `D = 0`, `Dr = 0`, `Fs = 1`.
    ///
    /// Bitplane-count sub-packet: `M = [1, 1]` (band 0), `[1]` (band 1);
    /// VLC(mtop=0, T=0) is unary → "10 10 10" → 0xA8, `Lcnt = 1`.
    ///
    /// Data sub-packet (one plane per group, 4 bits each):
    ///   band 0 g=0: coefs [1,0,1,0] → "1010";
    ///   band 0 g=1: coefs [1,1,1] + tail magnitude 1 → "1111";
    ///   band 1 g=0: coefs [0,1,0,0] → "0100".
    ///   → 0xAF 0x40, `Ldat = 2`.
    ///
    /// Sign sub-packet, one bit per non-zero magnitude **including the
    /// tail** (Table C.9 NOTE 2):
    ///   band 0: x0 (+) x2 (−) x4 (+) x5 (+) x6 (+), tail (+);
    ///   band 1: x1 (−).
    ///   → "0100001" → 0x42, `Lsgn = 1`.
    ///
    /// A decoder that drops the tail bit reads band 1's sign from the
    /// tail's position (0) and decodes +1 instead of −1.
    #[test]
    fn fs1_sign_bits_for_nonzero_tail_coefficients() {
        let geom = PrecinctGeometry {
            bands: vec![
                BandGeometry {
                    wpb: 7,
                    gain: 0,
                    priority: 0,
                    l0: 0,
                    l1: 1,
                    exists: true,
                },
                BandGeometry {
                    wpb: 4,
                    gain: 0,
                    priority: 0,
                    l0: 0,
                    l1: 1,
                    exists: true,
                },
            ],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 1,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![
                PacketEntry { band: 0, line: 0 },
                PacketEntry { band: 1, line: 0 },
            ],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![0, 0],
            header_bytes: 0,
        };
        let body: Vec<u8> = vec![0xA8, 0xAF, 0x40, 0x42];
        let packet = PacketHeader {
            dr: 0,
            ldat: 2,
            lcnt: 1,
            lsgn: 1,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect("Fs=1 non-zero-tail sign decode");
        assert_eq!(dec.bytes_consumed, 4);

        let band0 = &dec.bands[0];
        assert_eq!(band0.m, vec![1, 1]);
        assert_eq!(band0.v, vec![1u32, 0, 1, 0, 1, 1, 1]);
        assert_eq!(band0.s, vec![0u8, 0, 1, 0, 0, 0, 0]);

        // Band 1's sign must come AFTER the tail's sign bit: −1 at x=1.
        let band1 = &dec.bands[1];
        assert_eq!(band1.m, vec![1]);
        assert_eq!(band1.v, vec![0u32, 1, 0, 0]);
        assert_eq!(band1.s, vec![0u8, 1, 0, 0]);
    }

    /// Vertical prediction selected at a first-line band with NO
    /// predecessor (`top = None`) is a malformed codestream per §C.6.1 /
    /// §C.6.3 (vertical prediction is forbidden at the topmost precinct
    /// of a slice). The decoder must reject it rather than read garbage.
    #[test]
    fn vertical_prediction_first_line_without_predecessor_errors() {
        let geom = PrecinctGeometry {
            bands: vec![BandGeometry {
                wpb: 8,
                gain: 0,
                priority: 0,
                l0: 0,
                l1: 1,
                exists: true,
            }],
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        };
        let layout = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 0 }],
        };
        let precinct = PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d: vec![1],
            header_bytes: 0,
        };
        let body: Vec<u8> = vec![0x00, 0x00, 0x80];
        let packet = PacketHeader {
            dr: 0,
            ldat: 2,
            lcnt: 1,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let err = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state, None)
            .expect_err("first-line vertical prediction without predecessor must error");
        assert!(
            format!("{err}").contains("vertical prediction"),
            "expected a vertical-prediction error, got: {err}"
        );
    }

    /// Helper: read out the per-band bitplane counts of band 0's line 0.
    fn band_m(dec: &PacketDecode) -> Vec<u8> {
        dec.bands[0].m.clone()
    }
}
