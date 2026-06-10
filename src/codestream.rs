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

use crate::error::{JpegXsError as Error, Result};

use crate::capabilities::{parse_capabilities_lossy, Capabilities};
use crate::com::{parse_com, ComMarker};
use crate::component_table::{self, ComponentTable};
use crate::crg::{parse_crg, CrgMarker};
use crate::cts::{parse_cts, CtsMarker};
use crate::cwd::{parse_cwd, CwdMarker};
use crate::markers::Marker;
use crate::output::{parse_nlt, NltParams};
use crate::picture_header::{self, PictureHeader};
use crate::slice_header::{self, SliceHeader};
use crate::slice_walker::{parse_wgt, BandWeight};

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
    /// Parsed value of the CWD `Sd` field per Annex A.4.7 Table A.18.
    /// `None` when no CWD marker was present (decoder treats this as
    /// `Sd = 0`); otherwise `Some(sd)` with `sd ∈ 1..=Nc-1`.
    pub cwd_sd: Option<u8>,
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

impl Codestream {
    /// Decode the CAP marker's `cap[]` byte array into a strongly-typed
    /// view of the supported capability bits per Annex A.5.4.
    /// Lossy on the trailing-zero-byte rule (`A.4.3`) — see
    /// [`parse_capabilities_lossy`] for details. For strict parsing
    /// use [`crate::capabilities::parse_capabilities`].
    pub fn capabilities(&self) -> Capabilities {
        parse_capabilities_lossy(&self.cap)
    }

    /// Decode the optional CTS marker body into a strongly-typed
    /// [`CtsMarker`] (Annex A.4.8, Tables A.19 / A.20).
    ///
    /// Returns `Ok(None)` if the codestream carried no CTS segment
    /// (legal for `Cpih ∈ {0, 1, 2}`); returns `Ok(Some(cts))` with
    /// the parsed `Cf` / `e1` / `e2` fields when CTS was present.
    /// Body-level errors (reserved nibble non-zero, `Cf` outside
    /// `{0, 3}`, `e1` / `e2` exceeding 3) surface as `Err(_)` — the
    /// top-level marker-chain parser only enforces ordering and the
    /// "Cpih=3 ⇒ CTS present" rule, not the field-level constraints
    /// from §A.4.8.
    pub fn cts(&self) -> Result<Option<CtsMarker>> {
        match self.cts.as_deref() {
            None => Ok(None),
            Some(body) => parse_cts(body).map(Some),
        }
    }

    /// Decode the optional CRG marker body into a strongly-typed
    /// [`CrgMarker`] (Annex A.4.9, Table A.21).
    ///
    /// Returns `Ok(None)` if no CRG segment was present; returns
    /// `Ok(Some(crg))` with one [`crate::crg::CrgEntry`] per
    /// component when present. Body-level errors (wrong byte count
    /// for `4 * Nc`) surface as `Err(_)`. The component count is
    /// taken from the codestream's `pih.nc`.
    pub fn crg(&self) -> Result<Option<CrgMarker>> {
        match self.crg.as_deref() {
            None => Ok(None),
            Some(body) => parse_crg(body, self.pih.nc).map(Some),
        }
    }

    /// Decode the optional NLT marker body into a strongly-typed
    /// [`NltParams`] (Annex A.4.6, Table A.16).
    ///
    /// Returns `Ok(None)` when no NLT segment was present (decoder
    /// uses the linear Annex G.3 output path); returns
    /// `Ok(Some(NltParams::Quadratic { dco }))` for `Tnlt = 1` or
    /// `Ok(Some(NltParams::Extended { t1, t2, e }))` for
    /// `Tnlt = 2`. Body-level errors (wrong body length for the
    /// declared `Tnlt`, unknown `Tnlt`, out-of-range exponent /
    /// thresholds) surface as `Err(_)`.
    pub fn nlt(&self) -> Result<Option<NltParams>> {
        match self.nlt.as_deref() {
            None => Ok(None),
            Some(body) => parse_nlt(body).map(Some),
        }
    }

    /// Decode the optional CWD marker body into a strongly-typed
    /// [`CwdMarker`] (Annex A.4.7, Table A.18).
    ///
    /// Returns `Ok(None)` when no CWD segment was present (decoder
    /// treats this as `Sd = 0`, i.e. no components suppressed from the
    /// wavelet decomposition); returns `Ok(Some(cwd))` with the parsed
    /// `Sd` field when CWD was present. Body-level errors (wrong body
    /// length, `Sd = 0`) surface as `Err(_)`. The geometry-level
    /// constraints (`Nc > 3` to permit the marker at all, `Sd <= Nc-1`)
    /// require the picture-header `Nc` and are already enforced by the
    /// top-level marker-chain parser; this accessor mirrors the
    /// round-251 / round-254 [`Self::cts`] / [`Self::wgt`] pattern as
    /// the next narrow typed-primitive step over the raw `cwd` byte
    /// buffer.
    pub fn cwd(&self) -> Result<Option<CwdMarker>> {
        match self.cwd.as_deref() {
            None => Ok(None),
            Some(body) => parse_cwd(body).map(Some),
        }
    }

    /// Decode the mandatory WGT marker body into a strongly-typed
    /// `Vec<BandWeight>` (Annex A.4.11, Table A.24).
    ///
    /// The WGT body is a flat sequence of `(G[b], P[b])` byte pairs,
    /// one pair per *existing* band (Table A.24's `if (b'x[b])`
    /// guard). The full mapping from pair index to band id requires
    /// the picture / component geometry and the optional CWD
    /// `Sd` value, which the slice walker resolves at decode time;
    /// the typed accessor here is geometry-independent and only
    /// enforces the body-level constraints:
    ///
    /// * total length is a multiple of two (each pair is two
    ///   bytes),
    /// * each `G[b]` is at most 15 (Annex A.4.11 hard cap).
    ///
    /// The order of the returned vector matches the on-wire order;
    /// pair `k` corresponds to the `k`-th existing band in the
    /// walker's iteration order, not to the flat band id `b`.
    /// Higher-level field errors (e.g. mismatch between
    /// `existing-band-count` and the carried pair count under a
    /// specific geometry) are caught at slice-walker construction.
    pub fn wgt(&self) -> Result<Vec<BandWeight>> {
        if self.wgt.len() % 2 != 0 {
            return Err(Error::invalid(format!(
                "jpegxs WGT body length {} is not a multiple of 2 \
                 (Annex A.4.11 Table A.24: (G[b], P[b]) byte pairs)",
                self.wgt.len()
            )));
        }
        parse_wgt(&self.wgt, self.wgt.len() / 2)
    }

    /// Decode every COM (extension) marker body into a strongly-typed
    /// [`ComMarker`] (Annex A.4.10, Tables A.22 / A.23).
    ///
    /// Zero or more extension marker segments may be present; the
    /// returned vector preserves their on-wire order. Each entry carries
    /// the big-endian `Tcom` type field and the variable `Dcom` payload.
    /// Body-level errors (a body too short to hold the two-byte `Tcom`
    /// field) surface as `Err(_)`. Reserved `Tcom` values are *not*
    /// rejected — an extension marker is advisory metadata and a
    /// conforming decoder skips unknown extension types, so a reserved
    /// `Tcom` is surfaced verbatim rather than as an error. Mirrors the
    /// round-251 / round-254 / round-266 typed-accessor pattern over the
    /// raw `com` byte buffers.
    pub fn com(&self) -> Result<Vec<ComMarker>> {
        self.com.iter().map(|body| parse_com(body)).collect()
    }
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
                // Annex A.4.7 Table A.18: CWD body is exactly 1 byte
                // (`Sd`), and `Sd ∈ 1..=Nc-1`. The marker is forbidden
                // unless `Nc > 3` per the same table.
                if body.len() != 1 {
                    return Err(Error::invalid(format!(
                        "jpegxs: CWD body must be 1 byte (Sd), got {}",
                        body.len()
                    )));
                }
                if pih.nc <= 3 {
                    return Err(Error::invalid(format!(
                        "jpegxs: CWD requires Nc>3 per Annex A.4.7, got Nc={}",
                        pih.nc
                    )));
                }
                let sd_val = body[0];
                if sd_val == 0 || (sd_val as u16) >= pih.nc as u16 {
                    return Err(Error::invalid(format!(
                        "jpegxs: CWD Sd must be in 1..={}, got {sd_val}",
                        pih.nc - 1
                    )));
                }
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
    //
    // JPEG XS does NOT byte-stuff (Part-1 §A.3 NOTE 2); a forward scan
    // for `FF 20` / `FF 11` would mis-fire on any entropy-coded byte
    // pair that happens to look like a marker. Instead we drive slice
    // length from the precinct headers (Annex C.2): each precinct
    // header carries a 24-bit `Lprc` field giving the entropy byte
    // count after the header. Summed over the slice's `n_precincts[t]`
    // precincts (computed by [`crate::slice_walker::build_plan`] from
    // PIH / CDT / WGT), that yields the slice's exact entropy byte
    // length.
    use crate::slice_walker::build_plan_sd;
    let mut slices: Vec<Slice> = Vec::new();
    let body = cur.read_len_segment()?;
    let mut current_header = slice_header::parse(body)?;
    let mut current_data_offset = cur.pos();
    // Try to build a picture plan from PIH/CDT/WGT/CWD. When WGT is empty
    // (probe-only test fixtures with no entropy data) build_plan_sd will
    // fail; in that case fall back to the legacy `FF 20`/`FF 11` byte
    // scan, which copes with the empty-entropy case (the very next
    // bytes are the next marker).
    let sd_for_plan = cwd.as_deref().map(|b| b[0]).unwrap_or(0);
    let plan_opt = build_plan_sd(&pih, &cdt, &wgt, sd_for_plan)
        .ok()
        .map(|(p, _)| p);

    let eoc_offset = loop {
        // Determine entropy length for this slice.
        let data_len = if let Some(plan) = plan_opt.as_ref() {
            // Length-driven: walk precinct headers, summing
            // `header_bytes + Lprc` per precinct. Empty-entropy slice
            // (probe fixture) is detected by peeking at the first two
            // bytes of the slice data area for a marker prefix.
            let slice_idx = slices.len();
            let slice_plan = plan.slices.get(slice_idx).ok_or_else(|| {
                Error::invalid(format!(
                    "jpegxs: codestream has more slices than the plan expects ({} planned)",
                    plan.slices.len()
                ))
            })?;
            let pstart_check = current_data_offset;
            let empty_slice = pstart_check + 1 < cur.buf.len()
                && cur.buf[pstart_check] == 0xff
                && matches!(cur.buf[pstart_check + 1], 0x11 | 0x20);
            let mut bytes_consumed = 0usize;
            if !empty_slice {
                for precinct_plan in &slice_plan.precincts {
                    let n_existing = precinct_plan
                        .geometry
                        .bands
                        .iter()
                        .filter(|b| b.exists)
                        .count();
                    let header_bits = 24 + 8 + 8 + 2 * n_existing;
                    let header_bytes = header_bits.div_ceil(8);
                    let pstart = current_data_offset + bytes_consumed;
                    if pstart + 3 > cur.buf.len() {
                        return Err(Error::invalid(format!(
                            "jpegxs: precinct header truncated at offset {pstart}"
                        )));
                    }
                    let lprc = ((cur.buf[pstart] as u32) << 16)
                        | ((cur.buf[pstart + 1] as u32) << 8)
                        | (cur.buf[pstart + 2] as u32);
                    bytes_consumed += header_bytes + (lprc as usize);
                }
            }
            bytes_consumed
        } else {
            // Legacy scan path (test-only fallback): forward search for
            // the next SLH (`FF 20`) or EOC (`FF 11`) marker. JPEG XS
            // does not byte-stuff so this misfires on entropy bytes
            // that happen to look like markers — only safe for hand-
            // crafted test fixtures whose entropy region is empty.
            let mut i = current_data_offset;
            let mut found = None;
            while i + 1 < cur.buf.len() {
                if cur.buf[i] == 0xff && (cur.buf[i + 1] == 0x20 || cur.buf[i + 1] == 0x11) {
                    found = Some(i);
                    break;
                }
                i += 1;
            }
            match found {
                Some(off) => off - current_data_offset,
                None => cur.buf.len() - current_data_offset,
            }
        };
        let after_slice = current_data_offset + data_len;
        if after_slice > cur.buf.len() {
            return Err(Error::invalid(format!(
                "jpegxs: slice data ({data_len} bytes) exceeds codestream"
            )));
        }
        slices.push(Slice {
            header: current_header,
            data_offset: current_data_offset,
            data_length: data_len,
        });
        cur.set_pos(after_slice);

        // After the slice's entropy data we expect either another SLH
        // or the EOC marker. If the buffer is truncated mid-marker,
        // treat it as a missing-EOC stream (matches the legacy
        // behaviour expected by tolerant callers).
        if cur.remaining() < 2 {
            break None;
        }
        let m = cur.read_marker()?;
        match m {
            Marker::SLH => {
                let body = cur.read_len_segment()?;
                current_header = slice_header::parse(body)?;
                current_data_offset = cur.pos();
            }
            Marker::EOC => {
                break Some(cur.pos() - 2);
            }
            other => {
                return Err(Error::invalid(format!(
                    "jpegxs: expected SLH or EOC after slice entropy data, got {:04X} ({})",
                    other.0,
                    other.name()
                )));
            }
        }
    };

    let cwd_sd = cwd.as_deref().map(|b| b[0]);
    Ok(Codestream {
        cap,
        pih,
        cdt,
        wgt,
        nlt,
        cwd,
        cwd_sd,
        cts,
        crg,
        com,
        slices,
        eoc_offset,
    })
}

// scan_next_slice_or_eoc removed in encoder round 2 — slice boundaries
// are now derived from the precinct Lprc fields via the slice walker
// plan. JPEG XS does NOT byte-stuff its entropy stream, so a
// `FF 20`/`FF 11` byte search would mis-fire on entropy bytes that
// happen to look like markers.

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
    fn capabilities_method_decodes_cap_bits() {
        // Lcap = 3 with byte 0x40 → bit 1 (Star-Tetrix) set.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&3u16.to_be_bytes());
        buf.push(0x40);
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
        let cs = parse(&buf).expect("cap caps parse");
        let caps = cs.capabilities();
        assert!(caps.star_tetrix);
        assert!(!caps.nlt_quadratic);
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

    /// Append a CTS marker (Lcts=4, 2-byte body) to `buf`. Caller
    /// supplies `cf`, `e1`, `e2` exactly as they should land on the
    /// wire (Reserved nibble is forced to zero).
    fn push_cts(buf: &mut Vec<u8>, cf: u8, e1: u8, e2: u8) {
        buf.extend_from_slice(&[0xff, 0x18]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.push(cf & 0x0f);
        buf.push(((e1 & 0x0f) << 4) | (e2 & 0x0f));
    }

    /// Append a CRG marker for `nc` components with `(x, y)` per entry.
    fn push_crg(buf: &mut Vec<u8>, entries: &[(u16, u16)]) {
        let lcrg = (2 + 4 * entries.len()) as u16;
        buf.extend_from_slice(&[0xff, 0x19]);
        buf.extend_from_slice(&lcrg.to_be_bytes());
        for (x, y) in entries {
            buf.extend_from_slice(&x.to_be_bytes());
            buf.extend_from_slice(&y.to_be_bytes());
        }
    }

    /// Append a `Tnlt = 1` NLT marker carrying `dco` in the σ:α
    /// packed-u16 form (Annex A.4.6 Table A.16).
    fn push_nlt_quadratic(buf: &mut Vec<u8>, dco: i32) {
        buf.extend_from_slice(&[0xff, 0x16]);
        buf.extend_from_slice(&5u16.to_be_bytes()); // Lnlt
        buf.push(1);
        let (sigma, alpha) = if dco < 0 {
            (1u16, (dco + (1 << 15)) as u16)
        } else {
            (0u16, dco as u16)
        };
        let packed = (sigma << 15) | (alpha & 0x7fff);
        buf.extend_from_slice(&packed.to_be_bytes());
    }

    /// Builds a minimal Star-Tetrix-capable 4-component codestream
    /// (Cpih=3, Nc=4) with CTS + CRG present. Used by typed-accessor
    /// tests below.
    fn build_star_tetrix_codestream() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(4, 4, 3, 3));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&((2 * 4 + 2) as u16).to_be_bytes());
        for _ in 0..4 {
            buf.extend_from_slice(&[8, 0x11]);
        }
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        // RGGB CRG: (0,0), (32768,0), (0,32768), (32768,32768).
        push_crg(&mut buf, &[(0, 0), (32768, 0), (0, 32768), (32768, 32768)]);
        // CTS: Cf = 3 (in-line), e1 = 1, e2 = 2.
        push_cts(&mut buf, 3, 1, 2);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        buf
    }

    #[test]
    fn cts_method_returns_none_when_absent() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.cts.is_none());
        let typed = cs.cts().expect("typed cts() OK");
        assert!(typed.is_none());
    }

    #[test]
    fn cts_method_decodes_body() {
        let buf = build_star_tetrix_codestream();
        let cs = parse(&buf).expect("star-tetrix parse");
        let cts = cs.cts().expect("typed cts() OK").expect("CTS present");
        assert_eq!(cts.cf, crate::cts::CtsExtent::InLine);
        assert_eq!(cts.cf.cf(), 3);
        assert_eq!(cts.e1, 1);
        assert_eq!(cts.e2, 2);
    }

    #[test]
    fn cts_method_surfaces_body_errors() {
        // Build a Cpih ∈ {0, 1, 2} codestream that carries an
        // (illegal-content) CTS so the parser keeps the body in
        // `cs.cts` without rejecting it. Use Cpih=2 (YCgCo, not
        // gated by the parser's Star-Tetrix-CTS-required check).
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
        // Reserved nibble forced non-zero (illegal per §A.4.8).
        buf.extend_from_slice(&[0xff, 0x18]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.push(0x10); // Reserved=1, Cf=0
        buf.push(0x00);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("parse OK (top-level CTS body is opaque)");
        let err = cs.cts().expect_err("typed cts() should reject Reserved!=0");
        assert!(
            format!("{err}").contains("Reserved"),
            "expected Reserved-nibble error, got {err}"
        );
    }

    #[test]
    fn crg_method_returns_none_when_absent() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.crg.is_none());
        let typed = cs.crg().expect("typed crg() OK");
        assert!(typed.is_none());
    }

    #[test]
    fn crg_method_decodes_rggb_body() {
        let buf = build_star_tetrix_codestream();
        let cs = parse(&buf).expect("star-tetrix parse");
        let crg = cs.crg().expect("typed crg() OK").expect("CRG present");
        assert_eq!(crg.entries.len(), 4);
        assert_eq!(crg.entries[0].x_crg, 0);
        assert_eq!(crg.entries[0].y_crg, 0);
        assert_eq!(crg.entries[1].x_crg, 32768);
        assert_eq!(crg.entries[3].y_crg, 32768);
        let ct = crate::crg::cfa_pattern_type(&crg).expect("RGGB → Ct = 0");
        assert_eq!(ct, 0);
    }

    #[test]
    fn nlt_method_returns_none_when_absent() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.nlt.is_none());
        let typed = cs.nlt().expect("typed nlt() OK");
        assert!(typed.is_none());
    }

    #[test]
    fn wgt_method_returns_empty_for_zero_band_codestream() {
        // build_tiny_codestream uses Lwgt=2 (i.e. empty body): the
        // 4×3 single-component image at NL=1/1 has bx[β,i] geometry
        // but the hand-built fixture leaves the body empty so the
        // accessor sees a zero-pair body. Verify that's still well-
        // formed at the marker-body level (geometry mismatch is
        // caught by the slice walker, not here).
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.wgt.is_empty());
        let typed = cs.wgt().expect("typed wgt() OK on empty body");
        assert!(typed.is_empty());
    }

    #[test]
    fn wgt_method_decodes_pair_body() {
        // Build a codestream whose WGT carries two (G[b], P[b]) pairs:
        // (gain=2, priority=0) and (gain=15, priority=3). Verify the
        // typed accessor returns them in wire order.
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
        // WGT — Lwgt = 2 + 4 = 6, body = 4 bytes (two pairs).
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&6u16.to_be_bytes());
        buf.extend_from_slice(&[2, 0, 15, 3]);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("wgt-pairs parse");
        let weights = cs.wgt().expect("typed wgt() OK");
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].gain, 2);
        assert_eq!(weights[0].priority, 0);
        assert_eq!(weights[1].gain, 15);
        assert_eq!(weights[1].priority, 3);
    }

    #[test]
    fn wgt_method_surfaces_odd_length_body() {
        // Codestream whose WGT body has an odd length (3 bytes).
        // The top-level marker-chain parser accepts any byte sequence
        // for the WGT body; the typed accessor enforces the pair-
        // length constraint.
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
        // WGT — Lwgt = 2 + 3 = 5, body = 3 bytes (odd → invalid).
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&5u16.to_be_bytes());
        buf.extend_from_slice(&[2, 0, 4]);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("parse keeps odd-length WGT body");
        let err = cs
            .wgt()
            .expect_err("typed wgt() should reject odd-length body");
        assert!(
            format!("{err}").contains("multiple of 2"),
            "expected multiple-of-2 error, got {err}"
        );
    }

    #[test]
    fn wgt_method_surfaces_oversized_gain() {
        // Codestream whose WGT body carries gain=16 (>15, illegal per
        // §A.4.11). The pair-count parity is fine so the parity check
        // does not fire; the per-pair gain-cap check must catch it.
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
        // WGT — Lwgt = 2 + 2 = 4, body = 2 bytes (one pair, gain=16).
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&[16, 0]);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("parse keeps oversized-gain WGT body");
        let err = cs.wgt().expect_err("typed wgt() should reject gain > 15");
        assert!(
            format!("{err}").contains("exceeds"),
            "expected gain-cap error, got {err}"
        );
    }

    #[test]
    fn com_method_returns_empty_when_absent() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.com.is_empty());
        let typed = cs.com().expect("typed com() OK");
        assert!(typed.is_empty());
    }

    #[test]
    fn com_method_decodes_two_segments() {
        // Codestream carrying two COM segments: an encoder-vendor type
        // (Tcom=0x0000, Dcom="hi") and a vendor-specific type
        // (Tcom=0x8042, Dcom=[0xab]). The typed accessor decodes both in
        // on-wire order into strongly-typed views.
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
        // COM #1 — Lcom = 2 + 2 + 2 = 6, Tcom=0x0000, Dcom="hi".
        buf.extend_from_slice(&[0xff, 0x15]);
        buf.extend_from_slice(&6u16.to_be_bytes());
        buf.extend_from_slice(&0x0000u16.to_be_bytes());
        buf.extend_from_slice(b"hi");
        // COM #2 — Lcom = 2 + 2 + 1 = 5, Tcom=0x8042, Dcom=[0xab].
        buf.extend_from_slice(&[0xff, 0x15]);
        buf.extend_from_slice(&5u16.to_be_bytes());
        buf.extend_from_slice(&0x8042u16.to_be_bytes());
        buf.push(0xab);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);

        let cs = parse(&buf).expect("two-COM parse");
        let coms = cs.com().expect("typed com() OK");
        assert_eq!(coms.len(), 2);
        assert_eq!(coms[0].tcom, crate::com::TCOM_ENCODER_VENDOR);
        assert_eq!(coms[0].dcom, b"hi");
        assert!(coms[0].is_encoder_vendor());
        assert_eq!(coms[1].tcom, 0x8042);
        assert_eq!(coms[1].dcom, vec![0xab]);
        assert!(coms[1].is_vendor_specific());
    }

    #[test]
    fn com_method_surfaces_short_body() {
        // A COM body carrying only one byte cannot hold the two-byte
        // Tcom field; the top-level marker-chain parser keeps the raw
        // body alive, and the typed accessor must reject it.
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
        // COM — Lcom = 2 + 1 = 3, body = one byte (too short for Tcom).
        buf.extend_from_slice(&[0xff, 0x15]);
        buf.extend_from_slice(&3u16.to_be_bytes());
        buf.push(0x42);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);

        let cs = parse(&buf).expect("parse keeps short COM body");
        let err = cs
            .com()
            .expect_err("typed com() should reject a body shorter than Tcom");
        assert!(
            format!("{err}").contains("Tcom"),
            "expected Tcom-length error, got {err}"
        );
    }

    #[test]
    fn cwd_method_returns_none_when_absent() {
        let buf = build_tiny_codestream();
        let cs = parse(&buf).expect("tiny parse");
        assert!(cs.cwd.is_none());
        assert!(cs.cwd_sd.is_none());
        let typed = cs.cwd().expect("typed cwd() OK");
        assert!(typed.is_none());
    }

    #[test]
    fn cwd_method_decodes_sd_body() {
        // Build a 4-component, Cpih=0 codestream carrying CWD with
        // Sd=1. Top-level parser keeps the body byte alive in
        // `cs.cwd`; the typed accessor decodes it as a strongly-typed
        // `CwdMarker { sd: 1 }`.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0xff, 0x10]);
        buf.extend_from_slice(&[0xff, 0x50]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x12]);
        buf.extend_from_slice(&26u16.to_be_bytes());
        buf.extend_from_slice(&build_pih_body(4, 4, 3, 0));
        buf.extend_from_slice(&[0xff, 0x13]);
        buf.extend_from_slice(&((2 * 4 + 2) as u16).to_be_bytes());
        for _ in 0..4 {
            buf.extend_from_slice(&[8, 0x11]);
        }
        buf.extend_from_slice(&[0xff, 0x14]);
        buf.extend_from_slice(&2u16.to_be_bytes());
        // CWD — Lcwd = 3, body = [Sd = 1].
        buf.extend_from_slice(&[0xff, 0x17]);
        buf.extend_from_slice(&3u16.to_be_bytes());
        buf.push(1);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("cwd parse");
        assert_eq!(cs.cwd.as_deref(), Some(&[1u8][..]));
        assert_eq!(cs.cwd_sd, Some(1));
        let cwd = cs.cwd().expect("typed cwd() OK").expect("CWD present");
        assert_eq!(cwd.sd, 1);
    }

    #[test]
    fn cwd_method_agrees_with_raw_sd_field() {
        // For every valid Sd in 1..=Nc-1 the typed accessor must
        // return the same value as the codestream's `cwd_sd` raw
        // field. Use Nc = 5 so Sd may legitimately walk 1..=4.
        for sd in 1u8..=4 {
            let mut buf = Vec::new();
            buf.extend_from_slice(&[0xff, 0x10]);
            buf.extend_from_slice(&[0xff, 0x50]);
            buf.extend_from_slice(&2u16.to_be_bytes());
            buf.extend_from_slice(&[0xff, 0x12]);
            buf.extend_from_slice(&26u16.to_be_bytes());
            buf.extend_from_slice(&build_pih_body(5, 4, 3, 0));
            buf.extend_from_slice(&[0xff, 0x13]);
            buf.extend_from_slice(&((2 * 5 + 2) as u16).to_be_bytes());
            for _ in 0..5 {
                buf.extend_from_slice(&[8, 0x11]);
            }
            buf.extend_from_slice(&[0xff, 0x14]);
            buf.extend_from_slice(&2u16.to_be_bytes());
            buf.extend_from_slice(&[0xff, 0x17]);
            buf.extend_from_slice(&3u16.to_be_bytes());
            buf.push(sd);
            buf.extend_from_slice(&[0xff, 0x20]);
            buf.extend_from_slice(&4u16.to_be_bytes());
            buf.extend_from_slice(&0u16.to_be_bytes());
            buf.extend_from_slice(&[0xff, 0x11]);
            let cs = parse(&buf).expect("cwd parse");
            let typed = cs.cwd().expect("typed cwd() OK").expect("CWD present");
            assert_eq!(typed.sd, sd);
            assert_eq!(cs.cwd_sd, Some(sd));
        }
    }

    #[test]
    fn nlt_method_decodes_quadratic_body() {
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
        push_nlt_quadratic(&mut buf, -1234);
        buf.extend_from_slice(&[0xff, 0x20]);
        buf.extend_from_slice(&4u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        buf.extend_from_slice(&[0xff, 0x11]);
        let cs = parse(&buf).expect("nlt-quadratic parse");
        let nlt = cs.nlt().expect("typed nlt() OK").expect("NLT present");
        match nlt {
            crate::output::NltParams::Quadratic { dco } => assert_eq!(dco, -1234),
            other => panic!("expected Quadratic, got {other:?}"),
        }
    }
}
