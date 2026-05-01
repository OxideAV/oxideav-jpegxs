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
pub use packet_body::{decode_packet_body, BandCoefficients, PacketDecode};
pub use packet_header::{parse_packet_header, PacketHeader};
pub use precinct_header::{parse_precinct_header, PrecinctHeader};

use oxideav_core::{Error, Result};

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
}
