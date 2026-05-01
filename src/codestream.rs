//! JPEG XS codestream marker-chain parser (ISO/IEC 21122-1:2022,
//! Annex A).
//!
//! Recovers the header chain — SOC, CAP, PIH, CDT, WGT, optional
//! COM/NLT/CWD/CTS/CRG — followed by one or more (SLH + entropy
//! data) slices, terminated by EOC. Compressed sample data is left in
//! place; the slice loop only records each slice's header position
//! and the byte range of the entropy-coded body up to the next SLH or
//! EOC.
//!
//! The parser is strict about ordering only where Part-1 §A.2 / §A.3
//! requires it: SOC must be first, CAP second, PIH third, and EOC
//! must terminate the stream. CDT and WGT must appear before the
//! first SLH.

use oxideav_core::{Error, Result};

use crate::component_table::{self, ComponentTable};
use crate::markers::Marker;
use crate::picture_header::{self, PictureHeader};
use crate::slice_header::{self, SliceHeader};

/// Records one (SLH, entropy-coded body) slice in the codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slice {
    pub header: SliceHeader,
    /// Byte offset of the first entropy-coded data byte (one past the
    /// SLH segment).
    pub data_offset: usize,
    /// Number of entropy-coded data bytes belonging to this slice
    /// (runs to the next SLH marker or to the EOC marker).
    pub data_length: usize,
}

/// Full parse result for one JPEG XS codestream.
#[derive(Debug, Clone)]
pub struct Codestream {
    /// Raw `cap[]` bit array from the CAP segment (after `Lcap`).
    pub cap: Vec<u8>,
    pub pih: PictureHeader,
    pub cdt: ComponentTable,
    /// Raw WGT segment payload (after `Lwgt`). Per §A.4.11 this is
    /// a sequence of (`G[b]`, `P[b]`) byte pairs over all bands.
    pub wgt: Vec<u8>,
    /// Optional NLT body if present, exactly as on the wire.
    pub nlt: Option<Vec<u8>>,
    /// Optional CWD body if present.
    pub cwd: Option<Vec<u8>>,
    /// Optional CTS body if present (mandatory iff `Cpih == 3`).
    pub cts: Option<Vec<u8>>,
    /// Optional CRG body if present.
    pub crg: Option<Vec<u8>>,
    /// All COM (extension) marker bodies, in order.
    pub com: Vec<Vec<u8>>,
    pub slices: Vec<Slice>,
    /// Byte offset of the EOC marker; `None` if the stream was
    /// truncated.
    pub eoc_offset: Option<usize>,
}

/// Parse a JPEG XS codestream byte buffer.
pub fn parse(buf: &[u8]) -> Result<Codestream> {
    let mut cur = Cursor::new(buf);

    // SOC must be first (§A.4.1).
    let m = cur.read_marker()?;
    if m != Marker::SOC {
        return Err(Error::invalid(format!(
            "jpegxs: expected SOC (FF10) at offset 0, got {:04X}",
            m.0
        )));
    }

    // CAP must be second (§A.4.3).
    let m = cur.read_marker()?;
    if m != Marker::CAP {
        return Err(Error::invalid(format!(
            "jpegxs: expected CAP (FF50) after SOC, got {:04X}",
            m.0
        )));
    }
    let cap_body = cur.read_len_segment()?;
    let cap = cap_body.to_vec();

    // PIH must be third (§A.4.4).
    let m = cur.read_marker()?;
    if m != Marker::PIH {
        return Err(Error::invalid(format!(
            "jpegxs: expected PIH (FF12) after CAP, got {:04X}",
            m.0
        )));
    }
    let pih_body = cur.read_len_segment()?;
    let pih = picture_header::parse(pih_body)?;

    // After PIH the spec allows CDT, WGT, and any optional segments
    // (NLT, CWD, CTS, CRG, COM) in any order before the first SLH —
    // see Table A.1. We require CDT and WGT to be present before
    // SLH (per §A.4.5, §A.4.11 they "shall precede the first slice
    // header").
    let mut cdt: Option<ComponentTable> = None;
    let mut wgt: Option<Vec<u8>> = None;
    let mut nlt: Option<Vec<u8>> = None;
    let mut cwd: Option<Vec<u8>> = None;
    let mut cts: Option<Vec<u8>> = None;
    let mut crg: Option<Vec<u8>> = None;
    let mut com: Vec<Vec<u8>> = Vec::new();

    // Loop until we see the first SLH (or EOC, which would be a
    // slice-less but technically possible edge case — we still treat
    // it as malformed because the spec requires at least one slice).
    let first_slh_marker = loop {
        let m = cur.read_marker()?;
        match m {
            Marker::CDT => {
                if cdt.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate CDT segment"));
                }
                let body = cur.read_len_segment()?;
                cdt = Some(component_table::parse(body, pih.nc)?);
            }
            Marker::WGT => {
                if wgt.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate WGT segment"));
                }
                let body = cur.read_len_segment()?;
                wgt = Some(body.to_vec());
            }
            Marker::NLT => {
                if nlt.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate NLT segment"));
                }
                let body = cur.read_len_segment()?;
                nlt = Some(body.to_vec());
            }
            Marker::CWD => {
                if cwd.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate CWD segment"));
                }
                let body = cur.read_len_segment()?;
                cwd = Some(body.to_vec());
            }
            Marker::CTS => {
                if cts.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate CTS segment"));
                }
                let body = cur.read_len_segment()?;
                cts = Some(body.to_vec());
            }
            Marker::CRG => {
                if crg.is_some() {
                    return Err(Error::invalid("jpegxs: duplicate CRG segment"));
                }
                let body = cur.read_len_segment()?;
                crg = Some(body.to_vec());
            }
            Marker::COM => {
                let body = cur.read_len_segment()?;
                com.push(body.to_vec());
            }
            Marker::SLH => break m,
            Marker::EOC => {
                return Err(Error::invalid(
                    "jpegxs: encountered EOC before any slice header",
                ));
            }
            other => {
                return Err(Error::invalid(format!(
                    "jpegxs: unexpected marker {:04X} ({}) in main header",
                    other.0,
                    other.name()
                )));
            }
        }
    };
    debug_assert_eq!(first_slh_marker, Marker::SLH);

    let cdt = cdt.ok_or_else(|| Error::invalid("jpegxs: missing mandatory CDT segment"))?;
    let wgt = wgt.ok_or_else(|| Error::invalid("jpegxs: missing mandatory WGT segment"))?;
    if pih.cpih == 3 && cts.is_none() {
        return Err(Error::invalid(
            "jpegxs: Cpih=3 requires CTS marker (Star-Tetrix)",
        ));
    }

    // We have just consumed the first SLH marker. Parse its body and
    // walk the rest of the codestream as alternating SLH segments and
    // entropy-coded blobs, terminated by EOC.
    let mut slices: Vec<Slice> = Vec::new();
    let body = cur.read_len_segment()?;
    let header = slice_header::parse(body)?;
    let data_offset = cur.pos();
    let mut current = (header, data_offset);

    let eoc_offset = loop {
        // Scan forward looking for the next SLH or EOC marker. JPEG XS
        // does NOT byte-stuff; the spec (NOTE 2 in §A.3) states that
        // bit patterns inside the entropy-coded data can collide with
        // marker bytes. So we cannot rely on a substring search. We
        // instead drive entropy-data length from the codestream
        // structure once a real bit-stream walker is available.
        //
        // For round 1 we use a forward scan that locates the next
        // 0xFF byte followed by 0x10/0x11/0x20 (EOC, SOC, or SLH).
        // This is fragile in the general case but works for hand-
        // built fixtures whose entropy bodies are crafted to avoid
        // those byte sequences. Round-2 work will replace this with a
        // length-driven slice walker once the bit-stream side is
        // implemented (the precinct/packet length fields described in
        // Annex C.2 give the exact byte count of every slice).
        let next = scan_next_slice_or_eoc(cur.buf, cur.pos());
        match next {
            Some((NextMarker::Slh, off)) => {
                let data_len = off - current.1;
                slices.push(Slice {
                    header: current.0,
                    data_offset: current.1,
                    data_length: data_len,
                });
                cur.set_pos(off);
                let m = cur.read_marker()?;
                debug_assert_eq!(m, Marker::SLH);
                let body = cur.read_len_segment()?;
                let header = slice_header::parse(body)?;
                current = (header, cur.pos());
            }
            Some((NextMarker::Eoc, off)) => {
                let data_len = off - current.1;
                slices.push(Slice {
                    header: current.0,
                    data_offset: current.1,
                    data_length: data_len,
                });
                break Some(off);
            }
            None => {
                // Stream is truncated. Capture whatever entropy data
                // remains and report `eoc_offset = None`.
                let data_len = cur.buf.len() - current.1;
                slices.push(Slice {
                    header: current.0,
                    data_offset: current.1,
                    data_length: data_len,
                });
                break None;
            }
        }
    };

    Ok(Codestream {
        cap,
        pih,
        cdt,
        wgt,
        nlt,
        cwd,
        cts,
        crg,
        com,
        slices,
        eoc_offset,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextMarker {
    Slh,
    Eoc,
}

/// Search forward from `start` for the next SLH (`FF 20`) or EOC
/// (`FF 11`) marker pair in `buf`. Returns the marker variant and its
/// byte offset (the offset of the leading `0xFF`) or `None` if the
/// buffer ends first.
///
/// Round-1 caveat: hand-built fixtures must avoid `FF 20` / `FF 11`
/// byte sequences inside entropy-coded slice data. Round-2 will replace
/// this with the spec-accurate length-driven walker once the precinct
/// header and packet header parsers (Annex C) land.
fn scan_next_slice_or_eoc(buf: &[u8], start: usize) -> Option<(NextMarker, usize)> {
    let mut i = start;
    while i + 1 < buf.len() {
        if buf[i] == 0xff {
            match buf[i + 1] {
                0x20 => return Some((NextMarker::Slh, i)),
                0x11 => return Some((NextMarker::Eoc, i)),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn read_marker(&mut self) -> Result<Marker> {
        if self.remaining() < 2 {
            return Err(Error::invalid(
                "jpegxs: truncated codestream while reading marker",
            ));
        }
        let m = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(Marker(m))
    }

    fn read_u16(&mut self) -> Result<u16> {
        if self.remaining() < 2 {
            return Err(Error::invalid(
                "jpegxs: truncated codestream while reading u16",
            ));
        }
        let v = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    /// Read a length-prefixed marker segment payload. The returned
    /// slice is the `Lxxx - 2` bytes that follow the length field.
    fn read_len_segment(&mut self) -> Result<&'a [u8]> {
        let lseg = self.read_u16()? as usize;
        if lseg < 2 {
            return Err(Error::invalid(format!(
                "jpegxs: marker segment length must be >= 2, got {lseg}"
            )));
        }
        let body_len = lseg - 2;
        if self.remaining() < body_len {
            return Err(Error::invalid(format!(
                "jpegxs: marker segment body {body_len} > remaining {}",
                self.remaining()
            )));
        }
        let slice = &self.buf[self.pos..self.pos + body_len];
        self.pos += body_len;
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the fixed PIH body for a tiny 4x3 single-component
    /// configuration suitable for round-trip tests.
    fn build_pih_body(nc: u8, wf: u16, hf: u16, cpih: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&0u32.to_be_bytes()); // Lcod = 0 (VBR)
        v.extend_from_slice(&0u16.to_be_bytes()); // Ppih
        v.extend_from_slice(&0u16.to_be_bytes()); // Plev
        v.extend_from_slice(&wf.to_be_bytes()); // Wf
        v.extend_from_slice(&hf.to_be_bytes()); // Hf
        v.extend_from_slice(&0u16.to_be_bytes()); // Cw = 0 (full image)
        v.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        v.push(nc); // Nc
        v.push(4); // Ng
        v.push(8); // Ss
        v.push(20); // Bw
        v.push(0x80); // Fq=8, Br=0
        v.push((cpih) & 0x0f); // Fslc=0, Ppoc=0, Cpih
        v.push(0x11); // NL,x=1, NL,y=1
        v.push(0x00); // Lh=0,Rl=0,Qpih=0,Fs=0,Rm=0
        v
    }

    /// Build a small valid JPEG XS codestream for one 4x3 grayscale
    /// component with one slice and no entropy-coded data. Use this as
    /// the canonical round-1 fixture.
    fn build_tiny_codestream() -> Vec<u8> {
        let mut v = Vec::new();
        // SOC
        v.extend_from_slice(&[0xff, 0x10]);
        // CAP — Lcap=2 (no capability bits)
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // PIH — Lpih=26, body=24 bytes
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        v.extend_from_slice(&build_pih_body(1, 4, 3, 0));
        // CDT — Lcdt = 2*Nc + 2 = 4, body = 2 bytes
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]); // B[0]=8, sx=1, sy=1
                                         // WGT — Lwgt=2, no bands
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // SLH — Lslh=4, Yslh=0
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        // (No entropy-coded data in this fixture.)
        // EOC
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn parses_tiny_codestream() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("parse tiny codestream");
        assert_eq!(cs.pih.width(), 4);
        assert_eq!(cs.pih.height(), 3);
        assert_eq!(cs.pih.num_components(), 1);
        assert_eq!(cs.cdt.components.len(), 1);
        assert_eq!(cs.cdt.components[0].bit_depth, 8);
        assert_eq!(cs.cdt.components[0].sx, 1);
        assert_eq!(cs.cdt.components[0].sy, 1);
        assert_eq!(cs.slices.len(), 1);
        assert_eq!(cs.slices[0].header.yslh, 0);
        // No entropy data between SLH and EOC.
        assert_eq!(cs.slices[0].data_length, 0);
        assert!(cs.eoc_offset.is_some());
    }

    #[test]
    fn parses_two_slices() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(1, 4, 6, 0));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&[8, 0x11]);
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        // Slice 0 — three bytes of fake entropy data (none of which
        // collide with marker prefixes).
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0x01, 0x02, 0x03]);
        // Slice 1
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&[0x04, 0x05]);
        buf.extend_from_slice(&[0xff, 0x11]);

        let cs = parse(&buf).expect("two-slice parse");
        assert_eq!(cs.slices.len(), 2);
        assert_eq!(cs.slices[0].header.yslh, 0);
        assert_eq!(cs.slices[0].data_length, 3);
        assert_eq!(cs.slices[1].header.yslh, 1);
        assert_eq!(cs.slices[1].data_length, 2);
    }

    #[test]
    fn rejects_missing_soc() {
        let buf = vec![0xff, 0x12, 0x00, 0x00];
        let err = parse(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("SOC"), "expected SOC error, got {msg}");
    }

    #[test]
    fn rejects_missing_cap() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(1, 4, 3, 0));
        let err = parse(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CAP"), "expected CAP error, got {msg}");
    }

    #[test]
    fn rejects_missing_cdt() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(1, 4, 3, 0));
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let err = parse(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CDT"), "expected CDT error, got {msg}");
    }

    #[test]
    fn rejects_cpih3_without_cts() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        // Cpih = 3 (Star-Tetrix). 4 components so CWD/CTS-related
        // arithmetic does not trip.
        buf.extend_from_slice(&build_pih_body(4, 4, 3, 3));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&((2 * 4 + 2) as u16).to_be_bytes());
        for _ in 0..4 {
            buf.extend_from_slice(&[8, 0x11]);
        }
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let err = parse(&buf).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Star-Tetrix"), "expected CTS error, got {msg}");
    }

    #[test]
    fn parses_optional_segments() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(1, 4, 3, 0));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&[8, 0x11]);
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        // COM segment: Lcom=6, Tcom=0x0000, two bytes of payload.
        buf.extend_from_slice(&[0xff, 0x15]);
        buf.extend_from_slice(&6u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(b"hi");
        // Slice + EOC
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);

        let cs = parse(&buf).expect("optional segments parse");
        assert_eq!(cs.com.len(), 1);
        assert_eq!(&cs.com[0][..2], &[0u8, 0u8]);
        assert_eq!(&cs.com[0][2..], b"hi");
    }

    #[test]
    fn cap_segment_captures_bits() {
        // Lcap=3, one byte of capabilities (0xC0 → bits 1,2 set).
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&3u16.to_be_bytes());
        buf.push(0xc0);
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(1, 4, 3, 0));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&[8, 0x11]);
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("cap parse");
        assert_eq!(cs.cap, vec![0xc0]);
    }
}
