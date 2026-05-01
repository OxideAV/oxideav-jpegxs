//! Packet header — Annex C.3, Table C.3.
//!
//! Each precinct carries one or more packets after its precinct
//! header. A packet header is only emitted if the packet is
//! non-empty (i.e. at least one `I[p,b,λ,s]` is set, see Annex B.7);
//! the round-3 entropy decoder takes that decision from the caller-
//! supplied [`PacketLayout`].
//!
//! Two header forms exist (Table C.3):
//!
//! * Short form — 1 + 15 + 13 + 11 = 40 bits (5 bytes), used when
//!   `Wf×Nc < 32752 && Lh == 0`.
//! * Long form  — 1 + 20 + 20 + 15 = 56 bits (7 bytes) otherwise.
//!
//! In both forms `Lsgn[p,s]` is signalled but ignored when `Fs == 0`
//! (Table C.3 explicitly requires the field's presence either way).
//!
//! The header is followed immediately by the packet body — no padding
//! bits are inserted between header and body (the field widths are
//! chosen so the header lands on a byte boundary).

use oxideav_core::{Error, Result};

use crate::entropy::bits::BitReader;
use crate::entropy::PrecinctGeometry;

/// Decoded packet header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    /// `Dr[p,s]` — raw-mode override (0 or 1).
    pub dr: u8,
    /// `Ldat[p,s]` — data sub-packet byte count.
    pub ldat: u32,
    /// `Lcnt[p,s]` — bitplane-count sub-packet byte count.
    pub lcnt: u32,
    /// `Lsgn[p,s]` — sign sub-packet byte count (ignored if `Fs==0`).
    pub lsgn: u32,
    /// True iff this header was decoded with the short-form layout.
    pub short_form: bool,
    /// Bytes consumed by the header itself (5 short, 7 long).
    pub header_bytes: usize,
}

const SHORT_LDAT_MAX: u32 = (1 << 15) - 1;
const SHORT_LCNT_MAX: u32 = (1 << 13) - 1;
const SHORT_LSGN_MAX: u32 = (1 << 11) - 1;
const LONG_LDAT_MAX: u32 = (1 << 20) - 1;
const LONG_LCNT_MAX: u32 = (1 << 20) - 1;
const LONG_LSGN_MAX: u32 = (1 << 15) - 1;

/// Parse a packet header from `buf`, dispatching on
/// `PrecinctGeometry::use_short_packet_header`.
pub fn parse_packet_header(buf: &[u8], geom: &PrecinctGeometry) -> Result<PacketHeader> {
    let mut reader = BitReader::new(buf);
    let dr = reader.read_bit()?;
    let short = geom.use_short_packet_header();
    let (ldat, lcnt, lsgn) = if short {
        let ldat = reader.read_bits(15)?;
        let lcnt = reader.read_bits(13)?;
        let lsgn = reader.read_bits(11)?;
        if ldat > SHORT_LDAT_MAX || lcnt > SHORT_LCNT_MAX || lsgn > SHORT_LSGN_MAX {
            return Err(Error::invalid(
                "jpegxs entropy: short packet header field overflow",
            ));
        }
        (ldat, lcnt, lsgn)
    } else {
        let ldat = reader.read_bits(20)?;
        let lcnt = reader.read_bits(20)?;
        let lsgn = reader.read_bits(15)?;
        if ldat > LONG_LDAT_MAX || lcnt > LONG_LCNT_MAX || lsgn > LONG_LSGN_MAX {
            return Err(Error::invalid(
                "jpegxs entropy: long packet header field overflow",
            ));
        }
        (ldat, lcnt, lsgn)
    };
    // The header field widths are chosen so that the header lands on
    // a byte boundary: 1+15+13+11=40 (short) and 1+20+20+15=56 (long).
    if !reader.at_byte_boundary() {
        return Err(Error::invalid(
            "jpegxs entropy: packet header did not land on a byte boundary",
        ));
    }
    Ok(PacketHeader {
        dr,
        ldat,
        lcnt,
        lsgn,
        short_form: short,
        header_bytes: reader.bytes_consumed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BandGeometry;

    fn geom_short() -> PrecinctGeometry {
        PrecinctGeometry {
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
        }
    }

    fn geom_long() -> PrecinctGeometry {
        let mut g = geom_short();
        g.short_packet_header = false;
        g
    }

    /// Build a 5-byte short packet header with the given fields.
    fn build_short(dr: u8, ldat: u32, lcnt: u32, lsgn: u32) -> Vec<u8> {
        // Lay the bits out into a 40-bit big-endian integer.
        let mut bits: u64 = 0;
        bits = (bits << 1) | (dr as u64 & 1);
        bits = (bits << 15) | (ldat as u64 & 0x7fff);
        bits = (bits << 13) | (lcnt as u64 & 0x1fff);
        bits = (bits << 11) | (lsgn as u64 & 0x07ff);
        let mut v = vec![0u8; 5];
        for (i, byte) in v.iter_mut().enumerate() {
            *byte = ((bits >> (8 * (4 - i))) & 0xff) as u8;
        }
        v
    }

    /// Build a 7-byte long packet header.
    fn build_long(dr: u8, ldat: u32, lcnt: u32, lsgn: u32) -> Vec<u8> {
        let mut bits: u64 = 0;
        bits = (bits << 1) | (dr as u64 & 1);
        bits = (bits << 20) | (ldat as u64 & 0xfffff);
        bits = (bits << 20) | (lcnt as u64 & 0xfffff);
        bits = (bits << 15) | (lsgn as u64 & 0x7fff);
        let mut v = vec![0u8; 7];
        for (i, byte) in v.iter_mut().enumerate() {
            *byte = ((bits >> (8 * (6 - i))) & 0xff) as u8;
        }
        v
    }

    #[test]
    fn parses_short_header() {
        let geom = geom_short();
        let buf = build_short(1, 0x1234, 0x0567, 0x0123);
        let h = parse_packet_header(&buf, &geom).unwrap();
        assert_eq!(h.dr, 1);
        assert_eq!(h.ldat, 0x1234);
        assert_eq!(h.lcnt, 0x0567);
        assert_eq!(h.lsgn, 0x0123);
        assert!(h.short_form);
        assert_eq!(h.header_bytes, 5);
    }

    #[test]
    fn parses_long_header() {
        let geom = geom_long();
        let buf = build_long(0, 0xabcde, 0x12345, 0x6789);
        let h = parse_packet_header(&buf, &geom).unwrap();
        assert_eq!(h.dr, 0);
        assert_eq!(h.ldat, 0xabcde);
        assert_eq!(h.lcnt, 0x12345);
        assert_eq!(h.lsgn, 0x6789);
        assert!(!h.short_form);
        assert_eq!(h.header_bytes, 7);
    }

    #[test]
    fn long_form_dispatch_when_lh_set() {
        let mut geom = geom_short();
        geom.lh = 1;
        // Even though short_packet_header is true, Lh=1 forces long.
        let buf = build_long(0, 1, 1, 1);
        let h = parse_packet_header(&buf, &geom).unwrap();
        assert!(!h.short_form);
        assert_eq!(h.header_bytes, 7);
    }

    #[test]
    fn rejects_truncated_buffer() {
        let geom = geom_short();
        let buf = vec![0u8; 4];
        assert!(parse_packet_header(&buf, &geom).is_err());
    }
}
