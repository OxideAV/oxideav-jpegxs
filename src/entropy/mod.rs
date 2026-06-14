//! Entropy decoding (ISO/IEC 21122-1:2022, Annex C).
//!
//! Round 3 scope. The entropy module decodes the four sub-packet types
//! that follow each slice header in a JPEG XS codestream:
//!
//! * [`precinct_header`] — precinct quantization `Q[p]`, refinement
//!   `R[p]`, and per-band bitplane-count coding mode `D[p,b]` (Annex
//!   C.2, Table C.1).
//! * [`packet_header`] — raw-mode override `Dr[p,s]` and the trio of
//!   sub-packet byte counts `Ldat`, `Lcnt`, `Lsgn` in either short or
//!   long form (Annex C.3, Table C.3).
//! * [`packet_body`] — the four sub-packets in order (significance,
//!   bitplane-count, data, sign), driven by the picture- and precinct-
//!   header flags (Annex C.4 / C.5 / C.6).
//!
//! The slice walker that synthesises the per-band sample geometry
//! (`Wpb[p,b]`, `Ncg[p,b]`, `Ns[p,b]`, `L0[p,b]`, `L1[p,b]`,
//! `I[p,b,λ,s]`) from the picture and component tables is *not* part
//! of round 3 — that's round 4. Round 3 instead exposes
//! [`PrecinctGeometry`] / [`BandGeometry`] / [`PacketLayout`] structs
//! that the caller (a hand-built fixture for now, the slice walker
//! later) populates explicitly.
//!
//! The primitive variable-length decoder (Annex C.7.1, Table C.15) and
//! the bit-stream cursor live in [`bits`].
//!
//! Allocation. Every output buffer (`v[p,λ,b,*]`, `s[p,λ,b,*]`,
//! `M[p,λ,b,*]`) is sized from the `BandGeometry` the caller supplies,
//! which in real use is itself derived from picture-header u16 fields
//! that the codestream parser already validated. The decoder never
//! mallocs proportionally to a length read from the wire.

pub mod bits;
pub mod packet_body;
pub mod packet_header;
pub mod precinct_header;

pub use bits::{vlc, BitReader};
pub use packet_body::{decode_packet_body, BandCoefficients, PacketDecode, PrecinctTop};
pub use packet_header::{parse_packet_header, PacketHeader};
pub use precinct_header::{parse_precinct_header, PrecinctHeader};

use crate::error::{JpegXsError as Error, Result};

/// Geometry of a single band inside a precinct, in the form the
/// entropy decoder needs. Mirrors the per-band quantities defined in
/// Annex B (B.5–B.9): `wpb` is `Wpb[p,b]`, `gain` is `G[b]`, `priority`
/// is `P[b]`, `l0` and `l1` are the band's first / one-past-last line
/// indices in the precinct, and `exists` corresponds to `b'x[b]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandGeometry {
    /// `Wpb[p,b]` — number of coefficients per line in this band.
    pub wpb: u32,
    /// `G[b]` — gain from the WGT segment, used in
    /// `T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)`.
    pub gain: u8,
    /// `P[b]` — priority from the WGT segment, used to test
    /// `P[b] < R[p]` for the refinement bit.
    pub priority: u8,
    /// `L0[p,b]` — first line index of band `b` in the precinct.
    pub l0: u16,
    /// `L1[p,b]` — one-past-last line index of band `b` in the precinct.
    pub l1: u16,
    /// `b'x[b]` — whether the band exists at all in this precinct (a
    /// false value here causes the band to be skipped in the precinct
    /// header `D[p,b]` loop, per Table C.1).
    pub exists: bool,
}

/// Geometry of a full precinct: the per-band metadata and the
/// per-precinct constants the entropy decoder needs from the picture
/// header (`Ng`, `Ss`, `Br`, `Fs`, `Rm`, `Rl`, `Lh`, and the
/// `Wf * Nc < 32752` short-header threshold computed by the caller).
#[derive(Debug, Clone)]
pub struct PrecinctGeometry {
    /// Per-band metadata indexed by band id.
    pub bands: Vec<BandGeometry>,
    /// `Ng` — coefficients per code group (PIH says 4).
    pub ng: u8,
    /// `Ss` — code groups per significance group (PIH says 8).
    pub ss: u8,
    /// `Br` — raw bitplane-count width in bits.
    pub br: u8,
    /// `Fs` — sign-coding strategy: 0 = signs in data sub-packet, 1 =
    /// separate sign sub-packet (Table A.11).
    pub fs: u8,
    /// `Rm` — run mode: 0 = runs indicate zero prediction residual,
    /// 1 = runs indicate zero coefficients (Table A.12).
    pub rm: u8,
    /// `Rl` — raw-mode selection per packet flag (Table A.7).
    pub rl: u8,
    /// `Lh` — long-header enforcement flag (Table A.7).
    pub lh: u8,
    /// True iff the codestream's picture header satisfies
    /// `Wf * Nc < 32752`. Combined with `lh == 0`, this selects the
    /// short packet header form (Table C.3).
    pub short_packet_header: bool,
}

impl PrecinctGeometry {
    /// `Ncg[p,b] = ceil(Wpb[p,b] / Ng)` — number of code groups in
    /// band `b` of this precinct (Annex B.8).
    pub fn ncg(&self, b: usize) -> u32 {
        let band = &self.bands[b];
        if self.ng == 0 {
            return 0;
        }
        band.wpb.div_ceil(self.ng as u32)
    }

    /// `Ns[p,b] = ceil(Wpb[p,b] / (Ng * Ss))` — number of significance
    /// groups in band `b` of this precinct (Annex B.9).
    pub fn ns(&self, b: usize) -> u32 {
        let band = &self.bands[b];
        let denom = (self.ng as u32) * (self.ss as u32);
        if denom == 0 {
            return 0;
        }
        band.wpb.div_ceil(denom)
    }

    /// True iff the short packet header form applies (Table C.3:
    /// `Wf*Nc < 32752 && Lh == 0`).
    pub fn use_short_packet_header(&self) -> bool {
        self.short_packet_header && self.lh == 0
    }
}

/// Inclusion record for one (band, line) pair inside a packet. The
/// codestream walker (round 4) builds these via the algorithm in Annex
/// B.7, Table B.4 (`I[p,b,λ,s]` flags). For round 3 the caller hands
/// them in directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketEntry {
    /// Band index `b`.
    pub band: u16,
    /// Line index `λ` inside the precinct.
    pub line: u16,
}

/// Layout of a single packet: the ordered list of (band, line) pairs
/// the packet covers. Order matters because the bitplane-count, data
/// and sign sub-packets all walk the bands+lines in this order.
#[derive(Debug, Clone)]
pub struct PacketLayout {
    pub entries: Vec<PacketEntry>,
}

/// Compute the truncation position `T[p,b]` per Annex C.6.2,
/// Table C.10:
///
/// * `r = (P[b] < R[p]) ? 1 : 0`
/// * `T[p,b] = clamp(Q[p] - G[b] - r, 0, 15)`
pub fn truncation_position(q: u8, r: u8, gain: u8, priority: u8) -> u8 {
    let refine = if priority < r { 1i32 } else { 0i32 };
    let t = (q as i32) - (gain as i32) - refine;
    t.clamp(0, 15) as u8
}

/// Convenience: compute every band's truncation position at once.
pub fn precinct_truncation(geom: &PrecinctGeometry, header: &PrecinctHeader) -> Vec<u8> {
    geom.bands
        .iter()
        .map(|band| truncation_position(header.q, header.r, band.gain, band.priority))
        .collect()
}

/// Compatibility check for a `PacketLayout`: every (band, line)
/// referenced must exist within its band's geometry.
pub(crate) fn validate_packet_layout(layout: &PacketLayout, geom: &PrecinctGeometry) -> Result<()> {
    for entry in &layout.entries {
        let bi = entry.band as usize;
        if bi >= geom.bands.len() {
            return Err(Error::invalid(format!(
                "jpegxs entropy: packet entry band {bi} out of range ({} bands)",
                geom.bands.len()
            )));
        }
        let band = &geom.bands[bi];
        if !band.exists {
            return Err(Error::invalid(format!(
                "jpegxs entropy: packet entry references non-existent band {bi}"
            )));
        }
        if entry.line < band.l0 || entry.line >= band.l1 {
            return Err(Error::invalid(format!(
                "jpegxs entropy: packet entry line {} outside band {bi} range [{}, {})",
                entry.line, band.l0, band.l1
            )));
        }
    }
    Ok(())
}

/// One packet's contribution to the precinct, as seen by the bitplane-
/// count buffer-bound validity check (Annex C.5.3.4, Table C.6). The
/// fields are exactly the per-packet quantities the algorithm reads:
/// the bitplane-count subpacket byte count `Lcnt[p,s]` from the packet
/// header, the raw-mode override flag `Dr[p,s]`, and the (band, line)
/// inclusion list `I[p,b,λ,s]` of the packet.
#[derive(Debug, Clone, Copy)]
pub struct PacketBufferInfo<'a> {
    /// `Lcnt[p,s]` — bitplane-count subpacket byte count (packet header).
    pub lcnt: u32,
    /// `Dr[p,s]` — raw-mode override flag (0 or 1).
    pub dr: u8,
    /// `I[p,b,λ,s]` — the (band, line) pairs this packet covers.
    pub entries: &'a [PacketEntry],
}

/// `Lsig[p,s]` — the significance-subpacket byte count of one packet,
/// inferred from the included bands (Annex C.5.3.2): one `Z` bit per
/// significance group of every present (band, line) whose precinct mode
/// has significance coding enabled (`D[p,b] & 2`) and that is not raw-
/// overridden (`Dr[p,s] == 0`), rounded up to whole bytes.
///
/// The decoder never signals `Lsig` on the wire; it is derived here the
/// same way the significance subpacket decoder in [`packet_body`]
/// consumes its bits, so the byte count is exact.
fn significance_subpacket_bytes(
    geom: &PrecinctGeometry,
    precinct: &PrecinctHeader,
    pkt: &PacketBufferInfo<'_>,
) -> u64 {
    if pkt.dr != 0 {
        return 0;
    }
    let mut bits: u64 = 0;
    for entry in pkt.entries {
        let bi = entry.band as usize;
        if bi >= geom.bands.len() || !geom.bands[bi].exists {
            continue;
        }
        if (precinct.d[bi] & 2) != 0 {
            bits += geom.ns(bi) as u64;
        }
    }
    bits.div_ceil(8)
}

/// Evaluate the bitplane-count buffer-size constraint of a precinct
/// (ISO/IEC 21122-1:2022 Annex C.5.3.4, Table C.6 `is_encoding_valid`).
///
/// The standard bounds the buffer an encoder may spend on entropy-coded
/// bitplane-count data by requiring that the coded size never exceed
/// the size the same data would occupy in raw mode (`Br` bits per code
/// group). The comparison is made per band when the picture-header
/// `Rl` flag is 0 (band-based raw-mode switch, §C.5.3.2) and per packet
/// when `Rl` is 1 (line/packet-based switch, §C.5.3.3).
///
/// This mirrors Table C.6's `valid` output: it returns `true` when the
/// precinct's mode selection satisfies the bound and `false` otherwise.
/// It is a **codestream-construction** constraint ("the codestream
/// shall be constructed in such a way that…"), so it is exposed as a
/// conformance predicate rather than a hard decode gate — a decoder
/// that reserves the raw-mode buffer can still reconstruct a precinct
/// that does not satisfy the bound (e.g. a degenerate all-zero picture
/// whose tiny single-line bands each occupy a whole byte). Callers that
/// want strict ISO conformance checking can reject `false`.
///
/// `packets` lists every non-empty packet of the precinct in wire
/// order with its `Lcnt[p,s]`, `Dr[p,s]`, and inclusion list. `Lsig`
/// is inferred via [`significance_subpacket_bytes`].
#[must_use]
pub fn bitplane_buffer_bound_satisfied(
    geom: &PrecinctGeometry,
    precinct: &PrecinctHeader,
    packets: &[PacketBufferInfo<'_>],
) -> bool {
    let br = geom.br as u64;
    let exists = |b: usize| b < geom.bands.len() && geom.bands[b].exists;
    // Per-packet inferred significance-subpacket sizes (Lsig[p,s]).
    let lsig: Vec<u64> = packets
        .iter()
        .map(|p| significance_subpacket_bytes(geom, precinct, p))
        .collect();
    // Br × Ncg[p,b'] summed over every band present in one packet.
    let packet_raw_bits = |pkt: &PacketBufferInfo<'_>| -> u64 {
        pkt.entries
            .iter()
            .filter(|e| exists(e.band as usize))
            .map(|e| br * geom.ncg(e.band as usize) as u64)
            .sum()
    };

    if geom.rl == 0 {
        // §C.5.3.2 — band-based: for every band b, the summed coded size
        // of the bitplane-count + significance subpackets covering b must
        // not exceed the raw-mode size of every band sharing each of
        // those packets+lines.
        for (bi, band) in geom.bands.iter().enumerate() {
            if !band.exists {
                continue;
            }
            let mut bytesize: u64 = 0;
            let mut rawsize_bits: u64 = 0;
            for (si, pkt) in packets.iter().enumerate() {
                // Lines of band b included in this packet.
                let lines_of_b =
                    pkt.entries.iter().filter(|e| e.band as usize == bi).count() as u64;
                if lines_of_b == 0 {
                    continue;
                }
                bytesize += lines_of_b * (pkt.lcnt as u64 + lsig[si]);
                // For each such line, every band b' present in the same
                // packet+line contributes Br × Ncg[p,b'] raw bits. The
                // walker emits at most one (band, line) entry per band per
                // packet, so summing over the packet's entries and scaling
                // by band b's line count matches the nested b'/λ loops.
                rawsize_bits += lines_of_b * packet_raw_bits(pkt);
            }
            if bytesize > rawsize_bits.div_ceil(8) {
                return false;
            }
        }
    } else {
        // §C.5.3.3 — packet/line-based: for every packet s, the coded
        // size (Lcnt + Lsig) must not exceed the raw-mode size of all
        // bitplane counts that packet carries.
        for (si, pkt) in packets.iter().enumerate() {
            let bytesize = pkt.lcnt as u64 + lsig[si];
            if bytesize > packet_raw_bits(pkt).div_ceil(8) {
                return false;
            }
        }
    }
    true
}

/// One packet's on-wire size as seen by the precinct-length consistency
/// check (Annex C.2 / Table C.1, with the packet syntax of Annex C.3 /
/// Table C.3 and the packet body of Table C.4). The fields are exactly
/// the per-packet quantities `Lprc[p]` accounts for: the byte count of
/// the packet header itself (`header_bytes`, 5 short / 7 long per
/// Table C.3), the bitplane-count subpacket byte count `Lcnt[p,s]`, the
/// data subpacket byte count `Ldat[p,s]`, the sign subpacket byte count
/// `Lsgn[p,s]` (present only when `Fs == 1`, Table C.4), the raw-mode
/// override flag `Dr[p,s]`, and the (band, line) inclusion list
/// `I[p,b,λ,s]` (from which the un-signalled `Lsig[p,s]` is inferred).
#[derive(Debug, Clone, Copy)]
pub struct PacketWireSize<'a> {
    /// Bytes consumed by the packet header (5 short / 7 long, Table C.3).
    pub header_bytes: u32,
    /// `Lcnt[p,s]` — bitplane-count subpacket byte count (packet header).
    pub lcnt: u32,
    /// `Ldat[p,s]` — data subpacket byte count (packet header).
    pub ldat: u32,
    /// `Lsgn[p,s]` — sign subpacket byte count (packet header). Only
    /// added to the precinct total when `Fs == 1` (Table C.4 includes
    /// the sign subpacket only when sign coding is enabled).
    pub lsgn: u32,
    /// `Dr[p,s]` — raw-mode override flag (0 or 1). Suppresses the
    /// inferred `Lsig[p,s]` contribution (a raw-mode packet has no
    /// significance subpacket, Annex C.3).
    pub dr: u8,
    /// `I[p,b,λ,s]` — the (band, line) pairs this packet covers, used to
    /// infer `Lsig[p,s]` via [`significance_subpacket_bytes`].
    pub entries: &'a [PacketEntry],
}

/// Verify that a precinct's `Lprc[p]` field is consistent with the
/// actual on-wire size of its packets (ISO/IEC 21122-1:2022 Annex C.2,
/// Table C.1).
///
/// Table C.1 defines `Lprc[p]` as the length of the entropy-coded data
/// of the precinct **including filler bytes**, counted "from the end of
/// the precinct header of this precinct up to, but not including the
/// first byte of the next precinct header, slice header or EOC". The
/// precinct header bytes themselves are therefore *not* counted; every
/// packet header and every subpacket *is*.
///
/// The total size occupied by the packets is, per the packet syntax
/// (Annex C.3, Table C.3) and the packet body (Table C.4), the sum over
/// packets `s` of:
///
/// * the packet header bytes (`header_bytes`, 5 short / 7 long),
/// * the inferred significance subpacket size `Lsig[p,s]`
///   ([`significance_subpacket_bytes`]) — un-signalled but reconstructed
///   bit-for-bit the same way the decoder consumes it,
/// * the bitplane-count subpacket size `Lcnt[p,s]`,
/// * the data subpacket size `Ldat[p,s]`,
/// * the sign subpacket size `Lsgn[p,s]`, but only when `Fs == 1`
///   (Table C.4 omits the sign subpacket entirely when sign coding is
///   disabled).
///
/// `Lprc[p]` must be **at least** that sum; the difference is the count
/// of optional filler bytes the precinct ends with (Annex C.2: "the
/// amount of filler bytes following the precinct can be inferred from
/// the `Lprc[p]` field"). A decoder skips them.
///
/// Returns `Ok(filler_bytes)` — the number of trailing filler bytes the
/// `Lprc[p]` field implies — when the packets fit, or `Err(_)` when the
/// summed packet size exceeds `Lprc[p]` (a malformed / inconsistent
/// codestream where `Lprc[p]` is too small to contain its own packets).
/// The packet sizes are summed in `u64` so an adversarial set of
/// per-packet counts cannot overflow the accumulator before the
/// comparison against the 24-bit `Lprc[p]`.
pub fn precinct_filler_bytes(
    geom: &PrecinctGeometry,
    precinct: &PrecinctHeader,
    packets: &[PacketWireSize<'_>],
) -> Result<u32> {
    let fs_on = geom.fs == 1;
    let mut total: u64 = 0;
    for pkt in packets {
        // Re-use the exact Lsig inference the bitplane-buffer-bound
        // predicate and the significance subpacket decoder both use.
        let buf_info = PacketBufferInfo {
            lcnt: pkt.lcnt,
            dr: pkt.dr,
            entries: pkt.entries,
        };
        let lsig = significance_subpacket_bytes(geom, precinct, &buf_info);
        total += pkt.header_bytes as u64;
        total += lsig;
        total += pkt.lcnt as u64;
        total += pkt.ldat as u64;
        if fs_on {
            total += pkt.lsgn as u64;
        }
    }
    let lprc = precinct.lprc as u64;
    if total > lprc {
        return Err(Error::invalid(format!(
            "jpegxs entropy: precinct packets occupy {total} bytes but Lprc[p] = {lprc} (Annex C.2 Table C.1)"
        )));
    }
    Ok((lprc - total) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(wpb: u32, gain: u8, priority: u8) -> BandGeometry {
        BandGeometry {
            wpb,
            gain,
            priority,
            l0: 0,
            l1: 1,
            exists: true,
        }
    }

    fn geom(bands: Vec<BandGeometry>) -> PrecinctGeometry {
        PrecinctGeometry {
            bands,
            ng: 4,
            ss: 8,
            br: 4,
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        }
    }

    #[test]
    fn ncg_and_ns_match_annex_b() {
        // Wpb = 32, Ng = 4 → Ncg = 8.
        // Ns = ceil(32 / (4*8)) = 1.
        let g = geom(vec![band(32, 0, 0)]);
        assert_eq!(g.ncg(0), 8);
        assert_eq!(g.ns(0), 1);
        // Wpb = 35, Ng = 4 → Ncg = 9.
        // Ns = ceil(35 / 32) = 2.
        let g = geom(vec![band(35, 0, 0)]);
        assert_eq!(g.ncg(0), 9);
        assert_eq!(g.ns(0), 2);
    }

    #[test]
    fn truncation_clamps_and_uses_priority() {
        // Q=10, R=5, P=3 → P<R so refine=1; G=2 → T = 10-2-1 = 7.
        assert_eq!(truncation_position(10, 5, 2, 3), 7);
        // Q=2, R=5, P=8 → P>=R so refine=0; G=10 → T = 2-10-0 = -8 → 0.
        assert_eq!(truncation_position(2, 5, 10, 8), 0);
        // Q=30, R=0, P=0 → P>=R so refine=0; G=2 → T = 28 → clamped 15.
        assert_eq!(truncation_position(30, 0, 2, 0), 15);
    }

    #[test]
    fn rejects_packet_entry_out_of_band() {
        let g = geom(vec![BandGeometry {
            wpb: 16,
            gain: 0,
            priority: 0,
            l0: 0,
            l1: 1,
            exists: true,
        }]);
        let bad = PacketLayout {
            entries: vec![PacketEntry { band: 0, line: 5 }],
        };
        assert!(validate_packet_layout(&bad, &g).is_err());
    }

    // ---- Annex C.5.3.4 / Table C.6 buffer-bound predicate ----

    fn hdr(d: Vec<u8>) -> PrecinctHeader {
        PrecinctHeader {
            lprc: 1,
            q: 0,
            r: 0,
            d,
            header_bytes: 6,
        }
    }

    /// `Lsig[p,s]` is inferred per Annex C.5.3.2: one bit per
    /// significance group of every present band whose mode enables
    /// significance coding (`D & 2`) and is not raw-overridden,
    /// rounded up to whole bytes.
    #[test]
    fn lsig_inference_counts_significance_groups() {
        // Two bands, Wpb=64 → Ns = ceil(64/(4*8)) = 2 each.
        let g = geom(vec![band(64, 0, 0), band(64, 0, 0)]);
        let h = hdr(vec![0b10, 0b00]); // band 0 sig-on, band 1 sig-off
        let pkt = PacketBufferInfo {
            lcnt: 0,
            dr: 0,
            entries: &[
                PacketEntry { band: 0, line: 0 },
                PacketEntry { band: 1, line: 0 },
            ],
        };
        // Only band 0 contributes Ns=2 bits → ceil(2/8) = 1 byte.
        assert_eq!(significance_subpacket_bytes(&g, &h, &pkt), 1);
        // Raw override (Dr=1) suppresses the significance subpacket.
        let pkt_raw = PacketBufferInfo { dr: 1, ..pkt };
        assert_eq!(significance_subpacket_bytes(&g, &h, &pkt_raw), 0);
    }

    /// Rl=0, single band with one line: coded Lcnt fitting the raw
    /// bound passes; one byte over fails. Wpb=64 → Ncg=16, Br=4 →
    /// raw = ceil(64/8) = 8 bytes.
    #[test]
    fn rl0_band_bound_pass_and_fail() {
        let g = geom(vec![band(64, 0, 0)]); // rl=0, br=4
        let h = hdr(vec![0b00]);
        let mk = |lcnt: u32| PacketBufferInfo {
            lcnt,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        // raw_bits = 4 * 16 = 64 → 8 bytes. Lcnt=8 fits, 9 overflows.
        assert!(bitplane_buffer_bound_satisfied(&g, &h, &[mk(8)]));
        assert!(!bitplane_buffer_bound_satisfied(&g, &h, &[mk(9)]));
    }

    /// Rl=0 degenerate small band across two single-line packets: each
    /// packet's Lcnt rounds up to a whole byte, so the coded size (2)
    /// can exceed the contiguous raw bound (1) — Table C.6 reports the
    /// precinct invalid, exactly the all-zero-tiny-picture case.
    #[test]
    fn rl0_tiny_band_two_lines_overshoots_raw_bound() {
        // Wpb=2 → Ncg=1, Br=4. Band spans 2 lines.
        let mut g = geom(vec![band(2, 0, 0)]);
        g.bands[0].l1 = 2;
        let h = hdr(vec![0b00]);
        let p0 = PacketBufferInfo {
            lcnt: 1,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        let p1 = PacketBufferInfo {
            lcnt: 1,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 1 }],
        };
        // bytesize = 2, rawsize_bits = 2 * (4*1) = 8 → 1 byte. 2 > 1.
        assert!(!bitplane_buffer_bound_satisfied(&g, &h, &[p0, p1]));
        // Grouping both lines into one packet (shared Lcnt=1) satisfies it.
        let grouped = PacketBufferInfo {
            lcnt: 1,
            dr: 0,
            entries: &[
                PacketEntry { band: 0, line: 0 },
                PacketEntry { band: 0, line: 1 },
            ],
        };
        assert!(bitplane_buffer_bound_satisfied(&g, &h, &[grouped]));
    }

    /// Rl=1 is the per-packet form: the bound is checked packet-by-
    /// packet without the cross-line summation, so a single-line tiny
    /// band passes (1 coded byte vs 1 raw byte) where Rl=0 summed two.
    #[test]
    fn rl1_per_packet_bound() {
        let mut g = geom(vec![band(2, 0, 0)]);
        g.rl = 1;
        let h = hdr(vec![0b00]);
        let ok = PacketBufferInfo {
            lcnt: 1,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        // packet raw = ceil(4*1/8) = 1 byte; Lcnt=1 fits.
        assert!(bitplane_buffer_bound_satisfied(&g, &h, &[ok]));
        let over = PacketBufferInfo { lcnt: 2, ..ok };
        assert!(!bitplane_buffer_bound_satisfied(&g, &h, &[over]));
    }

    // ---- Annex C.2 / Table C.1 precinct-length (Lprc[p]) consistency ----

    /// A precinct header carrying a chosen `Lprc[p]` with no `D[p,b]`
    /// significance coding (so the inferred `Lsig` is zero).
    fn hdr_lprc(lprc: u32, d: Vec<u8>) -> PrecinctHeader {
        PrecinctHeader {
            lprc,
            q: 0,
            r: 0,
            d,
            header_bytes: 6,
        }
    }

    /// One packet: 5-byte short header, Lcnt + Ldat fitting exactly, no
    /// significance coding, no signs (Fs=0). The summed packet size must
    /// equal header + Lcnt + Ldat and the predicate must report the
    /// remaining Lprc bytes as filler.
    #[test]
    fn precinct_filler_exact_and_with_padding() {
        let g = geom(vec![band(64, 0, 0)]); // fs = 0
        let h = hdr_lprc(0, vec![0b00]); // lprc set per-case below
        let pkt = PacketWireSize {
            header_bytes: 5,
            lcnt: 2,
            ldat: 3,
            lsgn: 9, // ignored: Fs = 0
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        // Packet occupies 5 + 2 + 3 = 10 bytes (Lsgn ignored, Fs=0).
        // Lprc = 10 → zero filler.
        let exact = PrecinctHeader {
            lprc: 10,
            ..h.clone()
        };
        assert_eq!(precinct_filler_bytes(&g, &exact, &[pkt]).unwrap(), 0);
        // Lprc = 13 → three filler bytes.
        let padded = PrecinctHeader {
            lprc: 13,
            ..h.clone()
        };
        assert_eq!(precinct_filler_bytes(&g, &padded, &[pkt]).unwrap(), 3);
        // Lprc = 9 → packets overflow the declared length → error.
        let toosmall = PrecinctHeader { lprc: 9, ..h };
        assert!(precinct_filler_bytes(&g, &toosmall, &[pkt]).is_err());
    }

    /// When `Fs == 1` the sign subpacket counts toward the precinct
    /// length; the same packet that fit at Fs=0 now needs Lsgn more
    /// bytes.
    #[test]
    fn precinct_filler_counts_signs_only_when_fs1() {
        let mut g = geom(vec![band(64, 0, 0)]);
        g.fs = 1;
        let pkt = PacketWireSize {
            header_bytes: 5,
            lcnt: 2,
            ldat: 3,
            lsgn: 4,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        // Fs=1: 5 + 2 + 3 + 4 = 14 bytes. Lprc = 14 → zero filler.
        let h = hdr_lprc(14, vec![0b00]);
        assert_eq!(precinct_filler_bytes(&g, &h, &[pkt]).unwrap(), 0);
        // Lprc = 13 (one short) → overflow.
        let h_small = hdr_lprc(13, vec![0b00]);
        assert!(precinct_filler_bytes(&g, &h_small, &[pkt]).is_err());
    }

    /// Significance coding (`D[p,b] & 2`) adds the inferred Lsig bytes
    /// to the precinct total; a raw-mode packet (`Dr = 1`) suppresses
    /// them, mirroring `significance_subpacket_bytes`.
    #[test]
    fn precinct_filler_includes_inferred_lsig() {
        let g = geom(vec![band(64, 0, 0)]); // Ns = ceil(64/32) = 2
        let h = hdr_lprc(0, vec![0b10]); // significance coding on
        let sig = PacketWireSize {
            header_bytes: 5,
            lcnt: 1,
            ldat: 1,
            lsgn: 0,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        // Lsig = ceil(Ns=2 / 8) = 1 byte → packet = 5 + 1 + 1 + 1 = 8.
        let h8 = PrecinctHeader {
            lprc: 8,
            ..h.clone()
        };
        assert_eq!(precinct_filler_bytes(&g, &h8, &[sig]).unwrap(), 0);
        // Raw override drops the Lsig byte → packet = 7, so Lprc=8 now
        // leaves one filler byte.
        let raw = PacketWireSize { dr: 1, ..sig };
        let h8b = PrecinctHeader { lprc: 8, ..h };
        assert_eq!(precinct_filler_bytes(&g, &h8b, &[raw]).unwrap(), 1);
    }

    /// Multiple packets sum; the filler is whatever Lprc has left over
    /// past the combined packet sizes.
    #[test]
    fn precinct_filler_sums_multiple_packets() {
        let mut g = geom(vec![band(64, 0, 0)]);
        g.bands[0].l1 = 2;
        let p0 = PacketWireSize {
            header_bytes: 5,
            lcnt: 1,
            ldat: 2,
            lsgn: 0,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 0 }],
        };
        let p1 = PacketWireSize {
            header_bytes: 5,
            lcnt: 1,
            ldat: 4,
            lsgn: 0,
            dr: 0,
            entries: &[PacketEntry { band: 0, line: 1 }],
        };
        // p0 = 8, p1 = 10 → 18 total. Lprc = 20 → 2 filler bytes.
        let h = hdr_lprc(20, vec![0b00]);
        assert_eq!(precinct_filler_bytes(&g, &h, &[p0, p1]).unwrap(), 2);
        // Lprc = 17 → overflow.
        let h_small = hdr_lprc(17, vec![0b00]);
        assert!(precinct_filler_bytes(&g, &h_small, &[p0, p1]).is_err());
    }
}
