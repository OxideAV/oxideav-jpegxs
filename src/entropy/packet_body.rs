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
//!      same band in the same precinct. Significance gating per
//!      `D & 2`.
//!
//!    Padded to a byte boundary, then `Lcnt[p,s]` bytes total
//!    (filler).
//! 3. **Data** sub-packet (Table C.8). For every code group with
//!    `M[p,λ,b,g] > T[p,b]`, optionally read `Ng` sign bits
//!    (`Fs == 0`) followed by `(M − T) × Ng` magnitude bits MSB-first.
//!    Padded to byte + filler to `Ldat[p,s]`.
//! 4. **Sign** sub-packet (Table C.9). Only when `Fs == 1`. One bit
//!    per non-zero `v[p,λ,b,Ng×g+k]`. Padded + filler to
//!    `Lsgn[p,s]`.
//!
//! Round-3 limitations:
//!
//! * **Vertical prediction across precincts is not yet supported.**
//!   The slice walker (round 4) will own the per-precinct
//!   `M[p_prev,L1[p,b]-sy,b,g]` cache. For now, vertical prediction
//!   only works when the predecessor line `λ-sy` lies inside the
//!   current precinct (i.e. `λ-sy >= L0[p,b]`). The spec already
//!   guarantees vertical prediction is never selected for the topmost
//!   line of the topmost precinct of a slice; the round-3 decoder
//!   defensively errors out if a packet references a line whose
//!   vertical predecessor escapes the precinct.
//! * The vertical subsampling factor `sy[i]` is taken to be 1 for
//!   round 3 (the single-component fixture has `sy = 1`). The walker
//!   will pass real values through.

use oxideav_core::{Error, Result};

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
                // No significance information for this band+line; treat
                // every group as "significant" for downstream gating.
                let ns = geom.ns(bi) as usize;
                for j in 0..ns {
                    prev_state
                        .sig_flags
                        .insert((entry.band, entry.line, j as u32), 1);
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

            // Vertical predictor source line.
            // sy is taken to be 1 for round 3 (single-component
            // fixture). When λ - sy < L0, the predictor would come
            // from the previous precinct, which round 3 doesn't
            // support — that case is gated below.
            let sy: u16 = 1;

            if packet.dr == 1 {
                // Raw mode: Br bits per code group.
                for g in 0..ncg {
                    let m = reader.read_bits(geom.br)? as u8;
                    let m_max = if geom.br >= 8 {
                        255
                    } else {
                        (1u32 << geom.br) as u8 - 1
                    };
                    if m > m_max {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: raw bitplane count {m} exceeds 2^Br - 1 = {m_max}"
                        )));
                    }
                    coef.m[line_index * ncg + g] = m;
                }
            } else if (dpb & 1) == 0 {
                // No-prediction VLC, Table C.14.
                let mtop = t;
                for g in 0..ncg {
                    let sig_group = g / geom.ss as usize;
                    let z = if (dpb & 2) != 0 {
                        // Significance coding enabled.
                        prev_state
                            .sig_flags
                            .get(&(entry.band, entry.line, sig_group as u32))
                            .copied()
                            .unwrap_or(1)
                    } else {
                        1
                    };
                    let delta_m = if (dpb & 2) == 0 || z != 0 {
                        vlc(&mut reader, mtop, t)?
                    } else {
                        // Insignificant group → Δm = 0 (Table C.14
                        // explicitly sets Δm = 0; the comment that
                        // bitplane count = T is satisfied because
                        // mtop = T).
                        0
                    };
                    let m = mtop + delta_m;
                    if !(0..=255).contains(&m) {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: decoded M[p,λ,b,g] = {m} out of byte range"
                        )));
                    }
                    coef.m[line_index * ncg + g] = m as u8;
                }
            } else {
                // Vertical prediction VLC, Table C.13.
                if entry.line < band.l0 + sy {
                    return Err(Error::invalid(
                        "jpegxs entropy: round-3 vertical prediction across precincts is not implemented (round 4 slice walker)",
                    ));
                }
                let prev_line_index = ((entry.line - sy) - band.l0) as usize;
                // Ttop[p,b] = T[p,b] for predecessors inside the same
                // precinct (Table C.11).
                let ttop = t;
                let teff = t.max(ttop);
                for g in 0..ncg {
                    let sig_group = g / geom.ss as usize;
                    let z = if (dpb & 2) != 0 {
                        prev_state
                            .sig_flags
                            .get(&(entry.band, entry.line, sig_group as u32))
                            .copied()
                            .unwrap_or(1)
                    } else {
                        1
                    };
                    let m_above = coef.m[prev_line_index * ncg + g] as i32;
                    let mtop = m_above.max(teff);
                    let delta_m = if (dpb & 2) == 0 || z != 0 {
                        vlc(&mut reader, mtop, t)?
                    } else if geom.rm == 0 {
                        0
                    } else {
                        // Rm == 1 → Δm = T - mtop (so M = T).
                        t - mtop
                    };
                    let m = mtop + delta_m;
                    if !(0..=255).contains(&m) {
                        return Err(Error::invalid(format!(
                            "jpegxs entropy: decoded vertical M[p,λ,b,g] = {m} out of byte range"
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
    let ldat = packet.ldat as usize;
    {
        let buf_dat = buf
            .get(total_consumed..total_consumed + ldat)
            .ok_or_else(|| {
                Error::invalid("jpegxs entropy: packet body truncated at data sub-packet")
            })?;
        let mut reader = BitReader::new(buf_dat);
        for entry in &layout.entries {
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
        for entry in &layout.entries {
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
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state)
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
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state)
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
        // Significance sub-packet: 1 bit = 0 (insignificant). Padded
        // to a byte → 1 byte 0x00.
        // Bitplane-count sub-packet: 0 bits used (Δm=0 inferred for
        // every group, which gives M = mtop = T = 0). Padded to 1
        // byte (Lcnt must be ≥ 1 in practice for the sub-packet's
        // padding byte — actually no, when no bits are written the
        // padding round-up gives 0 bytes, so Lcnt=0 is legal).
        // Significance sub-packet = 0x00 (one bit = 0, padded).
        // Lcnt = 0 → no bytes.
        // Ldat = 0 → no data bytes (M=0 ≤ T=0 for every group).
        let body: Vec<u8> = vec![0x00];

        let packet = PacketHeader {
            dr: 0,
            ldat: 0,
            lcnt: 0,
            lsgn: 0,
            short_form: true,
            header_bytes: 5,
        };
        let mut state = PrecinctState::default();
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state)
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
        let dec = decode_packet_body(&body, &geom, &precinct, &packet, &layout, &mut state)
            .expect("decode raw-mode packet");
        // No significance sub-packet bits consumed since Dr=1.
        assert_eq!(dec.bytes_consumed, 7);
        let band = &dec.bands[0];
        assert_eq!(band.m, vec![3, 0, 1, 2]);
        let expected_v = vec![5u32, 0, 7, 1, 0, 0, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0];
        assert_eq!(band.v, expected_v);
    }
}
