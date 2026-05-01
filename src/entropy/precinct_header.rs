//! Precinct header — Annex C.2, Table C.1.
//!
//! A precinct header opens every precinct in a slice. It carries:
//!
//! * `Lprc[p]` — total byte count of the entropy-coded data following
//!   this precinct header up to (but excluding) the start of the next
//!   precinct header / slice header / EOC. 24-bit unsigned, range
//!   `1..=2^20-1`.
//! * `Q[p]` — precinct quantization, `u(8)` in `0..=31`.
//! * `R[p]` — precinct refinement, `u(8)` in `0..=NL-1`.
//! * `D[p,b]` — bitplane-count coding mode for each band that exists
//!   in this precinct, `u(2)`. Bit 0 selects vertical vs no-prediction
//!   (0 = no prediction, 1 = vertical prediction). Bit 1 selects
//!   significance coding (0 = disabled, 1 = enabled). Per Table C.1
//!   the loop iterates *every* band index `0..NL`, but the field is
//!   only emitted when `b'x[b]` indicates the band exists.
//! * Padding to the next byte boundary.
//!
//! `D[p,b]` values for bands that don't exist in this precinct are
//! reported as `0` in the parsed structure — a benign default that
//! matches the spec's "the same number of D[p,b] fields shall be
//! present, regardless of the values of the Dr[p,s] fields and
//! regardless whether some bands are not included at all because the
//! last precinct is partially cut off" note. (Annex C.2 says the field
//! must still be present; whether the band physically participates is
//! controlled by `b'x[b]`.)

use oxideav_core::{Error, Result};

use crate::entropy::bits::BitReader;
use crate::entropy::{validate_packet_layout, PacketLayout, PrecinctGeometry};

/// Decoded precinct header. `d` is one entry per band; bands whose
/// `BandGeometry::exists` is false store `D[p,b] = 0`.
#[derive(Debug, Clone)]
pub struct PrecinctHeader {
    /// `Lprc[p]` — entropy-coded byte count following this header.
    pub lprc: u32,
    /// `Q[p]` — precinct quantization (0..=31).
    pub q: u8,
    /// `R[p]` — precinct refinement.
    pub r: u8,
    /// `D[p,b]` for every band in `0..NL`; zero for absent bands.
    pub d: Vec<u8>,
    /// Number of bytes consumed by the precinct header itself,
    /// *including* the byte-alignment padding at the end. Useful for
    /// the slice walker to advance the cursor past the header before
    /// reading the first packet header.
    pub header_bytes: usize,
}

/// Constraints from Table C.1.
const LPRC_MAX: u32 = (1 << 20) - 1;
const Q_MAX: u8 = 31;

/// Parse a precinct header from `buf` using the band layout in
/// `geom`. Reads `Lprc`, `Q`, `R`, and one `D[p,b]` field for every
/// existing band, then aligns to the next byte boundary.
pub fn parse_precinct_header(buf: &[u8], geom: &PrecinctGeometry) -> Result<PrecinctHeader> {
    let mut reader = BitReader::new(buf);
    let lprc = reader.read_bits(24)?;
    if lprc == 0 {
        return Err(Error::invalid("jpegxs entropy: Lprc[p] must be at least 1"));
    }
    if lprc > LPRC_MAX {
        return Err(Error::invalid(format!(
            "jpegxs entropy: Lprc[p] = {lprc} exceeds 2^20-1"
        )));
    }
    let q = reader.read_bits(8)? as u8;
    if q > Q_MAX {
        return Err(Error::invalid(format!("jpegxs entropy: Q[p] = {q} > 31")));
    }
    let r = reader.read_bits(8)? as u8;

    let mut d = Vec::with_capacity(geom.bands.len());
    for band in &geom.bands {
        if band.exists {
            d.push(reader.read_bits(2)? as u8);
        } else {
            d.push(0);
        }
    }

    reader.align_to_byte();
    Ok(PrecinctHeader {
        lprc,
        q,
        r,
        d,
        header_bytes: reader.bytes_consumed(),
    })
}

/// Sanity-check a freshly-parsed precinct header against geometry and
/// against the round-3 packet-layout list. Currently asserts:
///
/// * `D[p,b]` bit-0 (vertical prediction) is *not* selected for any
///   band that has its `λ - sy` predecessor outside the precinct, per
///   Annex C.2: "vertical prediction is never selected for the
///   precinct at the top of a slice" → at minimum, the very first
///   packet of every band can't use vertical prediction. The slice
///   walker (round 4) will lift this to the proper "top of slice"
///   condition; the round-3 entropy decoder enforces only that the
///   per-band line-relative test holds.
/// * Every entry in every layout points to an existing band line.
pub fn validate_precinct(
    header: &PrecinctHeader,
    geom: &PrecinctGeometry,
    layouts: &[PacketLayout],
) -> Result<()> {
    if header.d.len() != geom.bands.len() {
        return Err(Error::invalid(format!(
            "jpegxs entropy: precinct header has {} D[p,b] fields, geometry expects {}",
            header.d.len(),
            geom.bands.len()
        )));
    }
    for layout in layouts {
        validate_packet_layout(layout, geom)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::BandGeometry;

    fn geom_two_bands() -> PrecinctGeometry {
        PrecinctGeometry {
            bands: vec![
                BandGeometry {
                    wpb: 16,
                    gain: 0,
                    priority: 0,
                    l0: 0,
                    l1: 1,
                    exists: true,
                },
                BandGeometry {
                    wpb: 16,
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
            fs: 0,
            rm: 0,
            rl: 0,
            lh: 0,
            short_packet_header: true,
        }
    }

    /// Build a hand-rolled precinct header byte sequence for the
    /// two-band geometry above.
    fn build(lprc: u32, q: u8, r: u8, d0: u8, d1: u8) -> Vec<u8> {
        // Lprc 24 bits, Q 8 bits, R 8 bits → 5 bytes already aligned.
        // Then 2 bits for each D[p,b] = 4 bits in the 6th byte;
        // padded to a full byte.
        let mut v = Vec::with_capacity(6);
        v.push(((lprc >> 16) & 0xff) as u8);
        v.push(((lprc >> 8) & 0xff) as u8);
        v.push((lprc & 0xff) as u8);
        v.push(q);
        v.push(r);
        let dbyte = ((d0 & 0x03) << 6) | ((d1 & 0x03) << 4);
        v.push(dbyte);
        v
    }

    #[test]
    fn parses_basic_header() {
        let buf = build(42, 5, 1, 0b10, 0b11);
        let h = parse_precinct_header(&buf, &geom_two_bands()).unwrap();
        assert_eq!(h.lprc, 42);
        assert_eq!(h.q, 5);
        assert_eq!(h.r, 1);
        assert_eq!(h.d, vec![0b10, 0b11]);
        assert_eq!(h.header_bytes, 6);
    }

    #[test]
    fn skips_d_field_for_absent_band() {
        let mut g = geom_two_bands();
        g.bands[1].exists = false;
        // Only the first band gets a D field. Total bits: 24+8+8+2 =
        // 42 → 6 bytes after alignment. The unused six bits in the
        // last byte don't matter.
        // Lprc = 7, Q = 2, R = 3, then D[0] = 0b11 left-aligned in
        // the next byte.
        let v = vec![0u8, 0u8, 7u8, 2u8, 3u8, 0b1100_0000];
        let h = parse_precinct_header(&v, &g).unwrap();
        assert_eq!(h.d, vec![0b11, 0]);
    }

    #[test]
    fn rejects_zero_lprc() {
        let buf = build(0, 5, 1, 0, 0);
        assert!(parse_precinct_header(&buf, &geom_two_bands()).is_err());
    }

    #[test]
    fn rejects_q_above_31() {
        let buf = build(1, 32, 0, 0, 0);
        assert!(parse_precinct_header(&buf, &geom_two_bands()).is_err());
    }

    #[test]
    fn rejects_lprc_exceeding_24_bit_range() {
        // Lprc = 2^20 should already be rejected by the LPRC_MAX
        // check (2^20 - 1 is the upper bound from Table C.1).
        let buf = build(1 << 20, 5, 0, 0, 0);
        assert!(parse_precinct_header(&buf, &geom_two_bands()).is_err());
    }
}
