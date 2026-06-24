//! JXS still-image file format — ISO/IEC 21122-3:2019 Annex A.
//!
//! The raw JPEG XS codestream (`SOC … EOC`, ISO/IEC 21122-1 Annex A) may
//! optionally be wrapped in the box-based **JXS file format** for still
//! images (file extension `.jxs`). This module parses that box structure
//! and extracts the embedded codestream so it can be handed to
//! [`crate::decode_jpeg_xs`].
//!
//! The box structure is the JPEG 2000 family box syntax (A.3, Table A.1):
//! every box is `LBox(4) | TBox(4) | [XLBox(8)] | DBox`, big-endian. The
//! conceptual file structure (Figure A.1) is a contiguous sequence of
//! boxes whose first box is the JPEG XS Signature box (A.5.1), followed by
//! the File Type box (A.5.2), the JPEG XS Header superbox (A.5.4, holding
//! at least an Image Header box and a Colour Specification box) somewhere
//! before the first Contiguous Codestream box (A.5.5), which carries the
//! actual ISO/IEC 21122-1 codestream.
//!
//! Per A.6, a conforming reader skips and ignores any box it does not
//! understand; this parser collects the boxes it recognises and surfaces
//! the rest as raw spans.

use crate::error::{JpegXsError, Result};

/// JPEG XS Signature box type `'JXS\040'` (A.5.1, 0x4A58_5320).
pub const TBOX_SIGNATURE: u32 = 0x4A58_5320;
/// File Type box type `'ftyp'` (A.5.2, 0x6674_7970).
pub const TBOX_FILETYPE: u32 = 0x6674_7970;
/// JPEG XS Header (super)box type `'jp2h'` (A.5.4, 0x6A78_7368).
pub const TBOX_HEADER: u32 = 0x6A78_7368;
/// Image Header box type `'ihdr'` (A.5.4.2, 0x6968_6472).
pub const TBOX_IMAGE_HEADER: u32 = 0x6968_6472;
/// Colour Specification box type `'colr'` (A.5.4.3, 0x636F_6C72).
pub const TBOX_COLOUR: u32 = 0x636F_6C72;
/// Channel Definition box type `'cdef'` (A.5.4.4, 0x6364_6566).
pub const TBOX_CHANNEL_DEF: u32 = 0x6364_6566;
/// Contiguous Codestream box type `'jp2c'` (A.5.5, 0x6A70_3263).
pub const TBOX_CODESTREAM: u32 = 0x6A70_3263;
/// JPEG XS Video Support (super)box type `'jpvs'` (A.5.3, 0x6A70_7673).
pub const TBOX_VIDEO_SUPPORT: u32 = 0x6A70_7673;
/// JPEG XS Profile and Level box type `'jxpl'` (A.5.3.3, 0x6A78_706C).
pub const TBOX_PROFILE_LEVEL: u32 = 0x6A78_706C;
/// Intellectual Property box type `'jp2i'` (A.5.6, 0x6A70_3269).
pub const TBOX_IPR: u32 = 0x6A70_3269;
/// XML box type `'xml\040'` (A.5.7, 0x786D_6C20).
pub const TBOX_XML: u32 = 0x786D_6C20;
/// UUID box type `'uuid'` (A.5.8, 0x7575_6964).
pub const TBOX_UUID: u32 = 0x7575_6964;

/// The fixed 12-byte JPEG XS Signature box (A.5.1):
/// `0x0000_000C 4A58_5320 0D0A_870A`.
pub const SIGNATURE_BOX: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x53, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
];

/// The four-byte `'jxs\040'` brand / compatibility code (A.5.2 Table A.3).
pub const BRAND_JXS: u32 = 0x6A78_7320;

/// Compression-type code for JPEG XS in the Image Header box `C` field
/// (A.5.4.2 Table A.16).
pub const COMPRESSION_JPEG_XS: u8 = 12;

/// CICP colour-specification method (A.5.4.3 Table A.18 — the sole
/// conformant `METH` value).
pub const COLR_METH_CICP: u8 = 5;

/// One parsed box: its type, byte offset of the start of its `DBox`
/// payload within the original buffer, and the payload length.
///
/// `payload_start + payload_len` never exceeds the buffer length; the
/// header (`LBox`/`TBox`/optional `XLBox`) is excluded from the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxSpan {
    /// Four-byte big-endian `TBox` type code.
    pub tbox: u32,
    /// Byte offset of the first `DBox` byte in the source buffer.
    pub payload_start: usize,
    /// Length of the `DBox` payload in bytes.
    pub payload_len: usize,
}

impl BoxSpan {
    /// The box payload (`DBox`) sliced out of the original buffer.
    pub fn payload<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        &buf[self.payload_start..self.payload_start + self.payload_len]
    }
}

/// Read a big-endian `u32` at `off`, erroring on a short buffer.
fn be_u32(buf: &[u8], off: usize) -> Result<u32> {
    buf.get(off..off + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| JpegXsError::invalid("jxs file: truncated 32-bit field"))
}

/// Read a big-endian `u64` at `off`, erroring on a short buffer.
fn be_u64(buf: &[u8], off: usize) -> Result<u64> {
    buf.get(off..off + 8)
        .map(|b| u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .ok_or_else(|| JpegXsError::invalid("jxs file: truncated 64-bit field"))
}

/// Parse the box header at `off` (A.3 Table A.1), returning the box span
/// and the offset of the next box.
///
/// Handles the three `LBox` length encodings: `LBox >= 8` is the literal
/// box length; `LBox == 1` uses the 8-byte `XLBox` extended length; and
/// `LBox == 0` means the box extends to the end of the buffer. `LBox`
/// values `2..=7` are reserved for ISO/IEC use and rejected.
fn parse_box_header(buf: &[u8], off: usize) -> Result<(BoxSpan, usize)> {
    let lbox = be_u32(buf, off)? as u64;
    let tbox = be_u32(buf, off + 4)?;
    let (payload_start, box_len) = match lbox {
        0 => {
            // Box runs to the end of the buffer.
            let header = 8;
            if off + header > buf.len() {
                return Err(JpegXsError::invalid("jxs file: box header overruns buffer"));
            }
            (off + header, buf.len() - off)
        }
        1 => {
            // Extended length in the 8-byte XLBox field.
            let xl = be_u64(buf, off + 8)?;
            let header = 16;
            if xl < header as u64 {
                return Err(JpegXsError::invalid(
                    "jxs file: XLBox length smaller than box header",
                ));
            }
            (off + header, xl as usize)
        }
        2..=7 => {
            return Err(JpegXsError::invalid(
                "jxs file: reserved LBox length value 2..=7",
            ));
        }
        _ => {
            // Literal length; LBox >= 8 here.
            (off + 8, lbox as usize)
        }
    };
    let box_end = off
        .checked_add(box_len)
        .ok_or_else(|| JpegXsError::invalid("jxs file: box length overflow"))?;
    if box_end > buf.len() || payload_start > box_end {
        return Err(JpegXsError::invalid("jxs file: box length overruns buffer"));
    }
    let span = BoxSpan {
        tbox,
        payload_start,
        payload_len: box_end - payload_start,
    };
    Ok((span, box_end))
}

/// Walk a contiguous run of boxes within `buf[start..end]`, returning each
/// box's span in file order. Used for both the top-level file and the
/// payload of a superbox.
fn parse_boxes(buf: &[u8], start: usize, end: usize) -> Result<Vec<BoxSpan>> {
    let mut spans = Vec::new();
    let mut off = start;
    while off < end {
        let (span, next) = parse_box_header(buf, off)?;
        if next <= off {
            // A zero-progress box would loop forever; reject.
            return Err(JpegXsError::invalid("jxs file: non-advancing box"));
        }
        if next > end {
            return Err(JpegXsError::invalid(
                "jxs file: box extends past its container",
            ));
        }
        spans.push(span);
        off = next;
    }
    Ok(spans)
}

/// Returns `true` when the buffer begins with the JPEG XS Signature box
/// (A.5.1). This is the recommended discriminator between a `.jxs` file
/// and a raw codestream (which begins with the `FF 10` SOC marker).
pub fn is_jxs_file(buf: &[u8]) -> bool {
    buf.len() >= SIGNATURE_BOX.len() && buf[..SIGNATURE_BOX.len()] == SIGNATURE_BOX
}

/// File Type box contents (A.5.2 Table A.4): the brand, minor version and
/// compatibility list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileType {
    /// `BR` — brand / major version code.
    pub brand: u32,
    /// `MinV` — minor version (shall be 0; readers tolerate non-zero).
    pub minor_version: u32,
    /// `CLi` — compatibility list (one or more 4-byte codes).
    pub compatibility: Vec<u32>,
}

impl FileType {
    /// Parse the `DBox` of a File Type box.
    pub fn parse(body: &[u8]) -> Result<FileType> {
        // BR(4) + MinV(4) + at least one CLi(4).
        if body.len() < 12 {
            return Err(JpegXsError::invalid(
                "jxs ftyp: body must hold BR, MinV and at least one CLi",
            ));
        }
        if body.len() % 4 != 0 {
            return Err(JpegXsError::invalid(
                "jxs ftyp: body length not a multiple of 4",
            ));
        }
        let brand = be_u32(body, 0)?;
        let minor_version = be_u32(body, 4)?;
        let mut compatibility = Vec::new();
        let mut off = 8;
        while off < body.len() {
            compatibility.push(be_u32(body, off)?);
            off += 4;
        }
        Ok(FileType {
            brand,
            minor_version,
            compatibility,
        })
    }

    /// `true` when `'jxs\040'` appears in the compatibility list — the
    /// A.5.2 conformance condition for a reader to interpret the file.
    pub fn is_jxs_compatible(&self) -> bool {
        self.compatibility.contains(&BRAND_JXS)
    }
}

/// Image Header box contents (A.5.4.2 Table A.16). The length of this box
/// is fixed at 22 bytes, so its `DBox` payload is exactly 14 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageHeader {
    /// `HEIGHT` — image-area height in sample-grid rows.
    pub height: u32,
    /// `WIDTH` — image-area width in sample-grid columns.
    pub width: u32,
    /// `NC` — number of components (matches the codestream `Nc`).
    pub num_components: u16,
    /// `BPC` raw byte; see [`ImageHeader::bit_depth`] /
    /// [`ImageHeader::is_signed`] / [`ImageHeader::is_varying_depth`].
    pub bpc: u8,
    /// `C` — compression type (12 for JPEG XS).
    pub compression: u8,
    /// `UnkC` — colourspace-unknown flag (0 known, 1 unknown).
    pub colourspace_unknown: u8,
    /// `IPR` — intellectual-property-rights presence flag (0/1).
    pub ipr: u8,
}

impl ImageHeader {
    /// Parse the 14-byte `DBox` of an Image Header box.
    pub fn parse(body: &[u8]) -> Result<ImageHeader> {
        // HEIGHT(4)+WIDTH(4)+NC(2)+BPC(1)+C(1)+UnkC(1)+IPR(1) = 14.
        if body.len() != 14 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: DBox must be exactly 14 bytes (22-byte box)",
            ));
        }
        let height = be_u32(body, 0)?;
        let width = be_u32(body, 4)?;
        let num_components = u16::from_be_bytes([body[8], body[9]]);
        let bpc = body[10];
        let compression = body[11];
        let colourspace_unknown = body[12];
        let ipr = body[13];
        if height == 0 || width == 0 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: HEIGHT and WIDTH shall be >= 1 (Table A.16)",
            ));
        }
        if num_components == 0 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: NC shall be >= 1 (Table A.16)",
            ));
        }
        if compression != COMPRESSION_JPEG_XS {
            return Err(JpegXsError::invalid(format!(
                "jxs ihdr: C={compression} is not JPEG XS (expected 12, Table A.16)"
            )));
        }
        if colourspace_unknown > 1 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: UnkC reserved value (>1, Table A.16)",
            ));
        }
        if ipr > 1 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: IPR reserved value (>1, Table A.16)",
            ));
        }
        // Table A.17 defines the BPC byte: bits 0..6 are (bit_depth − 1)
        // and the MSB is the sign flag (`0xxx_xxxx` unsigned, `1xxx_xxxx`
        // signed), with the all-ones `0xFF` byte meaning "components vary
        // in bit depth". The depth field tops out at the `x010_0101`
        // (37 → 38-bit) row, so a low-7-bit value above 0x25 — other than
        // the varying-depth `0xFF` byte — is a reserved code point.
        if bpc != 0xFF && (bpc & 0x7F) > 0x25 {
            return Err(JpegXsError::invalid(
                "jxs ihdr: BPC depth field above the 38-bit maximum (Table A.17)",
            ));
        }
        Ok(ImageHeader {
            height,
            width,
            num_components,
            bpc,
            compression,
            colourspace_unknown,
            ipr,
        })
    }

    /// `true` when `BPC == 0xFF`, i.e. components vary in bit depth
    /// (Table A.17). The per-component depths then come from the
    /// codestream, not this box.
    pub fn is_varying_depth(&self) -> bool {
        self.bpc == 0xFF
    }

    /// `true` when the components are signed (BPC MSB set, Table A.17).
    pub fn is_signed(&self) -> bool {
        self.bpc & 0x80 != 0 && !self.is_varying_depth()
    }

    /// Component bit depth = `(BPC & 0x7F) + 1` (Table A.17). Returns
    /// `None` for the varying-depth (`0xFF`) code, where no single depth
    /// applies.
    pub fn bit_depth(&self) -> Option<u8> {
        if self.is_varying_depth() {
            None
        } else {
            Some((self.bpc & 0x7F) + 1)
        }
    }
}

/// Colour Specification box contents (A.5.4.3) for the CICP method
/// (`METH == 5`). Other methods are tolerated by the parser but surface
/// only `meth`/`prec`/`approx` (their `METHDAT` is method-specific and
/// out of scope per A.5.4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourSpec {
    /// `METH` — specification method (5 = CICP).
    pub meth: u8,
    /// `PREC` — precedence (signed; readers ignore the value).
    pub prec: i8,
    /// `APPROX` — colourspace approximation (readers ignore the value).
    pub approx: u8,
    /// CICP code points (present only when `meth == 5`).
    pub cicp: Option<Cicp>,
}

/// Coding-Independent Code Points for a CICP colour specification
/// (A.5.4.3 Table A.20, Rec. ITU-T H.273).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cicp {
    /// `COLOUR_PRIMARIES`.
    pub colour_primaries: u16,
    /// `TRANSFER_CHARACTERISTICS`.
    pub transfer_characteristics: u16,
    /// `MATRIX_COEFFICIENTS`.
    pub matrix_coefficients: u16,
    /// `VIDEO_FULL_RANGE_FLAG` (the MSB of the final `V` byte).
    pub full_range: bool,
}

impl ColourSpec {
    /// Parse the `DBox` of a Colour Specification box.
    pub fn parse(body: &[u8]) -> Result<ColourSpec> {
        // METH(1)+PREC(1)+APPROX(1) = 3, then METHDAT (56 bits for CICP).
        if body.len() < 3 {
            return Err(JpegXsError::invalid(
                "jxs colr: body must hold METH, PREC and APPROX",
            ));
        }
        let meth = body[0];
        let prec = body[1] as i8;
        let approx = body[2];
        let cicp = if meth == COLR_METH_CICP {
            // METHDAT = CP(2)+TC(2)+MC(2)+V(1) = 7 bytes.
            if body.len() < 3 + 7 {
                return Err(JpegXsError::invalid(
                    "jxs colr: CICP METHDAT must be 7 bytes (Table A.20)",
                ));
            }
            let colour_primaries = u16::from_be_bytes([body[3], body[4]]);
            let transfer_characteristics = u16::from_be_bytes([body[5], body[6]]);
            let matrix_coefficients = u16::from_be_bytes([body[7], body[8]]);
            let v = body[9];
            Some(Cicp {
                colour_primaries,
                transfer_characteristics,
                matrix_coefficients,
                full_range: v & 0x80 != 0,
            })
        } else {
            // Other METH values: the remaining METHDAT layout is
            // method-specific; a conforming JXS reader may ignore the
            // whole box (A.5.4.3). We surface only the header fields.
            None
        };
        Ok(ColourSpec {
            meth,
            prec,
            approx,
            cicp,
        })
    }
}

/// One channel description in a Channel Definition box (A.5.4.4
/// Table A.25): channel index, type and association.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelDef {
    /// `Cni` — channel index.
    pub channel: u16,
    /// `Typi` — channel type (0 colour, 1 opacity, 2 premultiplied).
    pub typ: u16,
    /// `Asoci` — channel-to-colour association.
    pub assoc: u16,
}

/// Channel Definition box contents (A.5.4.4): an array of
/// [`ChannelDef`] descriptions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelDefinition {
    /// `N` channel descriptions in file order.
    pub channels: Vec<ChannelDef>,
}

impl ChannelDefinition {
    /// Parse the `DBox` of a Channel Definition box.
    pub fn parse(body: &[u8]) -> Result<ChannelDefinition> {
        if body.len() < 2 {
            return Err(JpegXsError::invalid("jxs cdef: body must hold the N count"));
        }
        let n = u16::from_be_bytes([body[0], body[1]]) as usize;
        // Each description is Cni(2)+Typi(2)+Asoci(2) = 6 bytes.
        let expected = 2 + n * 6;
        if body.len() != expected {
            return Err(JpegXsError::invalid(format!(
                "jxs cdef: body is {} bytes, expected {expected} for N={n} (Table A.25)",
                body.len()
            )));
        }
        if n == 0 {
            return Err(JpegXsError::invalid(
                "jxs cdef: N shall be >= 1 (Table A.25)",
            ));
        }
        let mut channels = Vec::with_capacity(n);
        let mut off = 2;
        for _ in 0..n {
            let channel = u16::from_be_bytes([body[off], body[off + 1]]);
            let typ = u16::from_be_bytes([body[off + 2], body[off + 3]]);
            let assoc = u16::from_be_bytes([body[off + 4], body[off + 5]]);
            channels.push(ChannelDef {
                channel,
                typ,
                assoc,
            });
            off += 6;
        }
        Ok(ChannelDefinition { channels })
    }
}

/// JPEG XS Profile and Level box contents (A.5.3.3 Table A.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileLevel {
    /// `Ppih` — profile of the codestream (ISO/IEC 21122-2 Annex A).
    pub ppih: u16,
    /// `Plev` — level + sublevel of the codestream.
    pub plev: u16,
}

impl ProfileLevel {
    /// Parse the 4-byte `DBox` of a JPEG XS Profile and Level box.
    pub fn parse(body: &[u8]) -> Result<ProfileLevel> {
        if body.len() != 4 {
            return Err(JpegXsError::invalid(
                "jxs jxpl: body must be exactly 4 bytes (Ppih, Plev)",
            ));
        }
        Ok(ProfileLevel {
            ppih: u16::from_be_bytes([body[0], body[1]]),
            plev: u16::from_be_bytes([body[2], body[3]]),
        })
    }
}

/// A parsed JPEG XS Header superbox (A.5.4): the mandatory Image Header,
/// one or more Colour Specification boxes, and an optional Channel
/// Definition box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderBox {
    /// Mandatory Image Header box (A.5.4.2). Shall be the first box.
    pub image_header: ImageHeader,
    /// One or more Colour Specification boxes (A.5.4.3).
    pub colour_specs: Vec<ColourSpec>,
    /// Optional Channel Definition box (A.5.4.4).
    pub channel_def: Option<ChannelDefinition>,
}

impl HeaderBox {
    /// Parse the JPEG XS Header superbox from its `DBox` payload.
    fn parse(buf: &[u8], span: BoxSpan) -> Result<HeaderBox> {
        let inner = parse_boxes(
            buf,
            span.payload_start,
            span.payload_start + span.payload_len,
        )?;
        // A.5.4.2: the Image Header box shall be the first box.
        let first = inner
            .first()
            .ok_or_else(|| JpegXsError::invalid("jxs jp2h: empty header superbox"))?;
        if first.tbox != TBOX_IMAGE_HEADER {
            return Err(JpegXsError::invalid(
                "jxs jp2h: first box shall be the Image Header box (A.5.4.2)",
            ));
        }
        let image_header = ImageHeader::parse(first.payload(buf))?;
        let mut colour_specs = Vec::new();
        let mut channel_def = None;
        for s in &inner[1..] {
            match s.tbox {
                TBOX_COLOUR => colour_specs.push(ColourSpec::parse(s.payload(buf))?),
                TBOX_CHANNEL_DEF => {
                    if channel_def.is_some() {
                        return Err(JpegXsError::invalid(
                            "jxs jp2h: at most one Channel Definition box (A.5.4.4)",
                        ));
                    }
                    channel_def = Some(ChannelDefinition::parse(s.payload(buf))?);
                }
                // A.6: skip and ignore boxes we do not recognise.
                _ => {}
            }
        }
        // A.5.4.1 / A.5.4.3: there shall be at least one Colour
        // Specification box in the JPEG XS Header box.
        if colour_specs.is_empty() {
            return Err(JpegXsError::invalid(
                "jxs jp2h: at least one Colour Specification box required (A.5.4.3)",
            ));
        }
        Ok(HeaderBox {
            image_header,
            colour_specs,
            channel_def,
        })
    }
}

/// A parsed JXS file (ISO/IEC 21122-3 Annex A): the recognised top-level
/// boxes plus the byte span of the first Contiguous Codestream box, from
/// which [`JxsFile::codestream`] extracts the raw ISO/IEC 21122-1
/// codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxsFile {
    /// File Type box (A.5.2).
    pub file_type: FileType,
    /// JPEG XS Header superbox (A.5.4).
    pub header: HeaderBox,
    /// Optional Profile and Level box (A.5.3.3), if a JPEG XS Video
    /// Support superbox carried one.
    pub profile_level: Option<ProfileLevel>,
    /// Byte span of the `DBox` of the first Contiguous Codestream box.
    codestream_span: BoxSpan,
}

impl JxsFile {
    /// The raw ISO/IEC 21122-1 codestream carried by the first
    /// Contiguous Codestream box (A.5.5).
    pub fn codestream<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        self.codestream_span.payload(buf)
    }
}

/// Parse the box chain of a JXS file (A.2.3 / A.5), validating the
/// mandatory box ordering and extracting the recognised structures.
///
/// Per A.2.3 / Figure A.1: the JPEG XS Signature box (A.5.1) shall be
/// first, the File Type box (A.5.2) shall immediately follow it, and the
/// JPEG XS Header box (A.5.4) shall fall before the first Contiguous
/// Codestream box (A.5.5). Unknown boxes are skipped (A.6).
pub fn parse_jxs_file(buf: &[u8]) -> Result<JxsFile> {
    let spans = parse_boxes(buf, 0, buf.len())?;
    let mut iter = spans.iter();

    // A.5.1: the Signature box shall be the first box.
    let sig = iter
        .next()
        .ok_or_else(|| JpegXsError::invalid("jxs file: empty (no boxes)"))?;
    if sig.tbox != TBOX_SIGNATURE {
        return Err(JpegXsError::invalid(
            "jxs file: first box shall be the JPEG XS Signature box (A.5.1)",
        ));
    }
    // The Signature box DBox shall be exactly the 4-byte CR/LF magic.
    if sig.payload(buf) != [0x0D, 0x0A, 0x87, 0x0A] {
        return Err(JpegXsError::invalid(
            "jxs file: Signature box contents are not 0x0D0A870A (A.5.1)",
        ));
    }

    // A.5.2: the File Type box shall immediately follow the Signature box.
    let ftyp_span = iter
        .next()
        .ok_or_else(|| JpegXsError::invalid("jxs file: missing File Type box (A.5.2)"))?;
    if ftyp_span.tbox != TBOX_FILETYPE {
        return Err(JpegXsError::invalid(
            "jxs file: File Type box shall immediately follow the Signature box (A.5.2)",
        ));
    }
    let file_type = FileType::parse(ftyp_span.payload(buf))?;
    if !file_type.is_jxs_compatible() {
        return Err(JpegXsError::invalid(
            "jxs file: File Type compatibility list lacks 'jxs\\040' (A.5.2)",
        ));
    }

    // Walk the remaining boxes: the JPEG XS Header box and the first
    // Contiguous Codestream box, in that required relative order, plus an
    // optional Profile and Level box from a Video Support superbox.
    let mut header: Option<HeaderBox> = None;
    let mut codestream_span: Option<BoxSpan> = None;
    let mut profile_level: Option<ProfileLevel> = None;
    for s in iter {
        match s.tbox {
            TBOX_HEADER => {
                if header.is_some() {
                    return Err(JpegXsError::invalid(
                        "jxs file: exactly one JPEG XS Header box (A.5.4.1)",
                    ));
                }
                header = Some(HeaderBox::parse(buf, *s)?);
            }
            // A.5.5: ignore all codestreams after the first.
            TBOX_CODESTREAM if codestream_span.is_none() => {
                // A.2.3 / A.5.5: the Header box shall fall before the
                // first Contiguous Codestream box.
                if header.is_none() {
                    return Err(JpegXsError::invalid(
                        "jxs file: Header box shall precede the first codestream (A.5.5)",
                    ));
                }
                codestream_span = Some(*s);
            }
            TBOX_VIDEO_SUPPORT => {
                // Optional superbox; pull out the jxpl Profile/Level box
                // if present (A.5.3.3). Other inner boxes are ignored.
                let inner = parse_boxes(buf, s.payload_start, s.payload_start + s.payload_len)?;
                for is in &inner {
                    if is.tbox == TBOX_PROFILE_LEVEL && profile_level.is_none() {
                        profile_level = Some(ProfileLevel::parse(is.payload(buf))?);
                    }
                }
            }
            // A.6: skip and ignore unknown / non-essential boxes.
            _ => {}
        }
    }

    let header = header
        .ok_or_else(|| JpegXsError::invalid("jxs file: missing JPEG XS Header box (A.5.4)"))?;
    let codestream_span = codestream_span.ok_or_else(|| {
        JpegXsError::invalid("jxs file: missing Contiguous Codestream box (A.5.5)")
    })?;

    Ok(JxsFile {
        file_type,
        header,
        profile_level,
        codestream_span,
    })
}

/// Decode a JXS file (ISO/IEC 21122-3 Annex A): parse the box wrapper,
/// extract the embedded codestream, and decode it through
/// [`crate::decode_jpeg_xs`].
///
/// The Image Header box `NC` field is cross-checked against the
/// codestream's `Nc` (A.5.4.2 declares them redundant and contradictory
/// files non-conforming); a mismatch is rejected.
pub fn decode_jxs_file(buf: &[u8]) -> Result<crate::image::JpegXsImage> {
    let file = parse_jxs_file(buf)?;
    let codestream = file.codestream(buf);
    // Cross-check the ihdr geometry against the codestream picture header
    // (A.5.4.2: contradictory files are non-conforming).
    let cs = crate::codestream::parse(codestream)?;
    let ihdr = &file.header.image_header;
    if u16::from(cs.pih.nc) != ihdr.num_components {
        return Err(JpegXsError::invalid(format!(
            "jxs file: ihdr NC={} disagrees with codestream Nc={} (A.5.4.2)",
            ihdr.num_components, cs.pih.nc
        )));
    }
    if cs.pih.width() != ihdr.width || cs.pih.height() != ihdr.height {
        return Err(JpegXsError::invalid(format!(
            "jxs file: ihdr {}x{} disagrees with codestream {}x{} (A.5.4.2)",
            ihdr.width,
            ihdr.height,
            cs.pih.width(),
            cs.pih.height()
        )));
    }
    crate::decode_jpeg_xs(codestream)
}

/// Serialize one box: `LBox(4) | TBox(4) | DBox`, big-endian (A.3
/// Table A.1). Uses the literal-length form (the optional `XLBox`
/// extended length is only needed past the 4 GiB box-length ceiling,
/// which a still image never reaches in practice).
fn serialize_box(out: &mut Vec<u8>, tbox: u32, body: &[u8]) {
    let len = (8 + body.len()) as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&tbox.to_be_bytes());
    out.extend_from_slice(body);
}

/// Builder for a conforming JXS still-image file (ISO/IEC 21122-3
/// Annex A). The mandatory Image Header and a CICP Colour Specification
/// box are derived from / supplied to the builder; the optional Channel
/// Definition and Profile/Level boxes can be added.
///
/// The Image Header `WIDTH` / `HEIGHT` / `NC` / `BPC` are taken from the
/// codestream picture header (A.5.4.2 declares them redundant), so the
/// emitted file is internally consistent by construction.
#[derive(Debug, Clone)]
pub struct JxsFileBuilder {
    cicp: Cicp,
    colourspace_unknown: u8,
    channels: Option<Vec<ChannelDef>>,
    profile_level: Option<ProfileLevel>,
}

impl JxsFileBuilder {
    /// New builder with a CICP colour specification. A common sRGB
    /// choice is `Cicp { colour_primaries: 1, transfer_characteristics:
    /// 13, matrix_coefficients: 0, full_range: false }`.
    pub fn new(cicp: Cicp) -> Self {
        Self {
            cicp,
            colourspace_unknown: 0,
            channels: None,
            profile_level: None,
        }
    }

    /// Set the `UnkC` colourspace-unknown flag (A.5.4.2; 0 or 1).
    pub fn colourspace_unknown(mut self, unknown: bool) -> Self {
        self.colourspace_unknown = u8::from(unknown);
        self
    }

    /// Add a Channel Definition box (A.5.4.4).
    pub fn channels(mut self, channels: Vec<ChannelDef>) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Add a JPEG XS Video Support superbox carrying a Profile/Level box
    /// (A.5.3.3). The `jpvi` video-information box that A.5.3.1 marks
    /// mandatory inside `jpvs` is out of scope here, so this is emitted
    /// only when the caller asks for the profile/level wrapper.
    pub fn profile_level(mut self, ppih: u16, plev: u16) -> Self {
        self.profile_level = Some(ProfileLevel { ppih, plev });
        self
    }

    /// Serialize the file around the supplied raw codestream. The
    /// codestream is parsed to derive the `ihdr` fields; an invalid
    /// codestream is rejected.
    pub fn build(&self, codestream: &[u8]) -> Result<Vec<u8>> {
        let cs = crate::codestream::parse(codestream)?;
        let width = cs.pih.width();
        let height = cs.pih.height();
        let nc = u16::from(cs.pih.nc);
        let bit_depth = cs.cdt.max_bit_depth();
        if bit_depth == 0 || bit_depth > 38 {
            return Err(JpegXsError::invalid(
                "jxs writer: component bit depth out of the Table A.17 range",
            ));
        }
        // BPC: unsigned components, depth − 1 in the low 7 bits.
        let bpc = bit_depth - 1;
        let ipr = 0u8;

        let mut file = Vec::new();
        // Signature box (A.5.1).
        file.extend_from_slice(&SIGNATURE_BOX);
        // File Type box (A.5.2): BR = jxs, MinV = 0, CLi = [jxs].
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        ftyp.extend_from_slice(&0u32.to_be_bytes());
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        serialize_box(&mut file, TBOX_FILETYPE, &ftyp);

        // JPEG XS Header superbox (A.5.4): ihdr, colr, optional cdef.
        let mut jp2h = Vec::new();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&nc.to_be_bytes());
        ihdr.push(bpc);
        ihdr.push(COMPRESSION_JPEG_XS);
        ihdr.push(self.colourspace_unknown);
        ihdr.push(ipr);
        serialize_box(&mut jp2h, TBOX_IMAGE_HEADER, &ihdr);

        let mut colr = vec![COLR_METH_CICP, 0, 0];
        colr.extend_from_slice(&self.cicp.colour_primaries.to_be_bytes());
        colr.extend_from_slice(&self.cicp.transfer_characteristics.to_be_bytes());
        colr.extend_from_slice(&self.cicp.matrix_coefficients.to_be_bytes());
        colr.push(if self.cicp.full_range { 0x80 } else { 0x00 });
        serialize_box(&mut jp2h, TBOX_COLOUR, &colr);

        if let Some(channels) = &self.channels {
            let mut cdef = Vec::new();
            cdef.extend_from_slice(&(channels.len() as u16).to_be_bytes());
            for c in channels {
                cdef.extend_from_slice(&c.channel.to_be_bytes());
                cdef.extend_from_slice(&c.typ.to_be_bytes());
                cdef.extend_from_slice(&c.assoc.to_be_bytes());
            }
            serialize_box(&mut jp2h, TBOX_CHANNEL_DEF, &cdef);
        }
        serialize_box(&mut file, TBOX_HEADER, &jp2h);

        // Optional JPEG XS Video Support superbox (jpvs) carrying jxpl.
        if let Some(pl) = &self.profile_level {
            let mut jpvs = Vec::new();
            let mut jxpl = Vec::new();
            jxpl.extend_from_slice(&pl.ppih.to_be_bytes());
            jxpl.extend_from_slice(&pl.plev.to_be_bytes());
            serialize_box(&mut jpvs, TBOX_PROFILE_LEVEL, &jxpl);
            serialize_box(&mut file, TBOX_VIDEO_SUPPORT, &jpvs);
        }

        // Contiguous Codestream box (A.5.5).
        serialize_box(&mut file, TBOX_CODESTREAM, codestream);
        Ok(file)
    }
}

/// Wrap a raw ISO/IEC 21122-1 codestream in a minimal conforming JXS
/// file (ISO/IEC 21122-3 Annex A) with a CICP sRGB colour specification
/// and no auxiliary boxes. For finer control use [`JxsFileBuilder`].
pub fn write_jxs_file(codestream: &[u8]) -> Result<Vec<u8>> {
    JxsFileBuilder::new(Cicp {
        colour_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 0,
        full_range: false,
    })
    .build(codestream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap a raw codestream in a minimal conforming JXS file (A.5):
    /// Signature box, File Type box, JPEG XS Header superbox (ihdr +
    /// colr), then a Contiguous Codestream box.
    fn wrap_codestream(
        cs: &[u8],
        width: u32,
        height: u32,
        nc: u16,
        bd: u8,
        signed: bool,
    ) -> Vec<u8> {
        fn boxed(tbox: u32, body: &[u8]) -> Vec<u8> {
            let len = 8 + body.len();
            let mut v = Vec::with_capacity(len);
            v.extend_from_slice(&(len as u32).to_be_bytes());
            v.extend_from_slice(&tbox.to_be_bytes());
            v.extend_from_slice(body);
            v
        }
        let mut file = Vec::new();
        // Signature box (fixed 12 bytes).
        file.extend_from_slice(&SIGNATURE_BOX);
        // File Type box: BR=jxs, MinV=0, CLi=[jxs].
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        ftyp.extend_from_slice(&0u32.to_be_bytes());
        ftyp.extend_from_slice(&BRAND_JXS.to_be_bytes());
        file.extend_from_slice(&boxed(TBOX_FILETYPE, &ftyp));
        // ihdr (14-byte DBox).
        let bpc = if signed { 0x80 | (bd - 1) } else { bd - 1 };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&nc.to_be_bytes());
        ihdr.push(bpc);
        ihdr.push(COMPRESSION_JPEG_XS);
        ihdr.push(0); // UnkC
        ihdr.push(0); // IPR
                      // colr CICP sRGB (A.5.4.3, primaries=1 tc=13 mc=0).
        let mut colr = vec![COLR_METH_CICP, 0, 0];
        colr.extend_from_slice(&1u16.to_be_bytes());
        colr.extend_from_slice(&13u16.to_be_bytes());
        colr.extend_from_slice(&0u16.to_be_bytes());
        colr.push(0);
        let mut jp2h = Vec::new();
        jp2h.extend_from_slice(&boxed(TBOX_IMAGE_HEADER, &ihdr));
        jp2h.extend_from_slice(&boxed(TBOX_COLOUR, &colr));
        file.extend_from_slice(&boxed(TBOX_HEADER, &jp2h));
        // jp2c codestream.
        file.extend_from_slice(&boxed(TBOX_CODESTREAM, cs));
        file
    }

    fn luma_codestream() -> (Vec<u8>, u32, u32) {
        // 8x4 ramp luma, lossless 4:4:4 NL=1/1.
        let w = 8u16;
        let h = 4u16;
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 7) as u8)
            .collect();
        let cs = crate::encoder::encode_planar(w, h, 1, 0, 1, 1, &[pixels]).expect("encode");
        (cs, w as u32, h as u32)
    }

    #[test]
    fn is_jxs_file_discriminates_signature() {
        let (cs, _, _) = luma_codestream();
        assert!(!is_jxs_file(&cs), "raw codestream is not a JXS file");
        let file = wrap_codestream(&cs, 8, 4, 1, 8, false);
        assert!(is_jxs_file(&file));
        assert!(!is_jxs_file(&[]));
        assert!(!is_jxs_file(&[0u8; 4]));
    }

    #[test]
    fn parse_jxs_file_extracts_boxes_and_codestream() {
        let (cs, w, h) = luma_codestream();
        let file = wrap_codestream(&cs, w, h, 1, 8, false);
        let parsed = parse_jxs_file(&file).expect("parse jxs");
        assert!(parsed.file_type.is_jxs_compatible());
        assert_eq!(parsed.header.image_header.width, w);
        assert_eq!(parsed.header.image_header.height, h);
        assert_eq!(parsed.header.image_header.num_components, 1);
        assert_eq!(parsed.header.image_header.bit_depth(), Some(8));
        assert!(!parsed.header.image_header.is_signed());
        assert_eq!(parsed.header.colour_specs.len(), 1);
        let cicp = parsed.header.colour_specs[0].cicp.expect("cicp");
        assert_eq!(cicp.colour_primaries, 1);
        assert_eq!(cicp.transfer_characteristics, 13);
        // Extracted codestream is byte-identical to the embedded one.
        assert_eq!(parsed.codestream(&file), &cs[..]);
    }

    #[test]
    fn decode_jxs_file_round_trips_luma() {
        let w = 8u16;
        let h = 4u16;
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 7) as u8)
            .collect();
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let file = wrap_codestream(&cs, w as u32, h as u32, 1, 8, false);
        let img = decode_jxs_file(&file).expect("decode jxs file");
        assert_eq!(img.width, w as u32);
        assert_eq!(img.height, h as u32);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn rejects_missing_signature() {
        let (cs, w, h) = luma_codestream();
        let mut file = wrap_codestream(&cs, w, h, 1, 8, false);
        // Corrupt the signature contents.
        file[8] = 0x00;
        assert!(parse_jxs_file(&file).is_err());
    }

    #[test]
    fn rejects_ftyp_not_following_signature() {
        // Signature box immediately followed by a header box, no ftyp.
        let (cs, w, h) = luma_codestream();
        let file = wrap_codestream(&cs, w, h, 1, 8, false);
        // Drop the ftyp box bytes (12-byte sig, then ftyp is next).
        let ftyp_len = u32::from_be_bytes([file[12], file[13], file[14], file[15]]) as usize;
        let mut broken = file[..12].to_vec();
        broken.extend_from_slice(&file[12 + ftyp_len..]);
        assert!(parse_jxs_file(&broken).is_err());
    }

    #[test]
    fn rejects_ihdr_codestream_nc_mismatch() {
        let (cs, w, h) = luma_codestream();
        // Claim 3 components in the ihdr while the codestream has 1.
        let file = wrap_codestream(&cs, w, h, 3, 8, false);
        let err = decode_jxs_file(&file).unwrap_err();
        assert!(matches!(err, JpegXsError::InvalidData(_)));
    }

    #[test]
    fn rejects_ihdr_dimension_mismatch() {
        let (cs, _w, h) = luma_codestream();
        let file = wrap_codestream(&cs, 99, h, 1, 8, false);
        assert!(decode_jxs_file(&file).is_err());
    }

    #[test]
    fn skips_unknown_and_xml_boxes() {
        // Insert an unknown box and an XML box between header and jp2c.
        fn boxed(tbox: u32, body: &[u8]) -> Vec<u8> {
            let len = 8 + body.len();
            let mut v = Vec::new();
            v.extend_from_slice(&(len as u32).to_be_bytes());
            v.extend_from_slice(&tbox.to_be_bytes());
            v.extend_from_slice(body);
            v
        }
        let (cs, w, h) = luma_codestream();
        let file = wrap_codestream(&cs, w, h, 1, 8, false);
        // Find the jp2c box and splice an unknown + xml box right before it.
        let parsed = parse_jxs_file(&file).unwrap();
        let cs_off = parsed.codestream_span.payload_start - 8;
        let mut spliced = file[..cs_off].to_vec();
        spliced.extend_from_slice(&boxed(0x6465_6164, b"\xde\xad")); // unknown 'dead'
        spliced.extend_from_slice(&boxed(TBOX_XML, b"<x/>"));
        spliced.extend_from_slice(&file[cs_off..]);
        let img = decode_jxs_file(&spliced).expect("decode with unknown boxes");
        assert_eq!(img.width, w);
    }

    #[test]
    fn parses_xlbox_extended_length() {
        // A box with LBox=1 carrying its length in XLBox.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_be_bytes()); // LBox = 1
        buf.extend_from_slice(&TBOX_XML.to_be_bytes());
        buf.extend_from_slice(&20u64.to_be_bytes()); // XLBox = 16 hdr + 4 body
        buf.extend_from_slice(b"abcd");
        let spans = parse_boxes(&buf, 0, buf.len()).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].tbox, TBOX_XML);
        assert_eq!(spans[0].payload(&buf), b"abcd");
    }

    #[test]
    fn parses_lbox_zero_to_end_of_buffer() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_be_bytes()); // LBox = 0 → to EOF
        buf.extend_from_slice(&TBOX_XML.to_be_bytes());
        buf.extend_from_slice(b"trailing");
        let spans = parse_boxes(&buf, 0, buf.len()).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].payload(&buf), b"trailing");
    }

    #[test]
    fn rejects_reserved_lbox_2_to_7() {
        for lbox in 2u32..=7 {
            let mut buf = Vec::new();
            buf.extend_from_slice(&lbox.to_be_bytes());
            buf.extend_from_slice(&TBOX_XML.to_be_bytes());
            assert!(
                parse_box_header(&buf, 0).is_err(),
                "LBox={lbox} must be rejected"
            );
        }
    }

    #[test]
    fn filetype_parse_validates_length() {
        assert!(FileType::parse(&[0u8; 11]).is_err()); // < 12
        assert!(FileType::parse(&[0u8; 13]).is_err()); // not multiple of 4
        let ft =
            FileType::parse(&[0x6A, 0x78, 0x73, 0x20, 0, 0, 0, 0, 0x6A, 0x78, 0x73, 0x20]).unwrap();
        assert!(ft.is_jxs_compatible());
        assert_eq!(ft.minor_version, 0);
    }

    #[test]
    fn imageheader_signed_and_varying_depth() {
        // Signed 12-bit: BPC = 0x80 | 11 = 0x8B.
        let mut body = vec![0, 0, 0, 4, 0, 0, 0, 4, 0, 1, 0x8B, 12, 0, 0];
        let ih = ImageHeader::parse(&body).unwrap();
        assert!(ih.is_signed());
        assert_eq!(ih.bit_depth(), Some(12));
        // Varying depth: BPC = 0xFF.
        body[10] = 0xFF;
        let ih = ImageHeader::parse(&body).unwrap();
        assert!(ih.is_varying_depth());
        assert_eq!(ih.bit_depth(), None);
        // Out-of-range depth field (low 7 bits = 0x40 > 0x25) rejected.
        body[10] = 0xC0;
        assert!(ImageHeader::parse(&body).is_err());
    }

    #[test]
    fn channel_definition_parse() {
        // N=2, two descriptions.
        let mut body = Vec::new();
        body.extend_from_slice(&2u16.to_be_bytes());
        for (cn, ty, asoc) in [(0u16, 0u16, 1u16), (1, 1, 0)] {
            body.extend_from_slice(&cn.to_be_bytes());
            body.extend_from_slice(&ty.to_be_bytes());
            body.extend_from_slice(&asoc.to_be_bytes());
        }
        let cdef = ChannelDefinition::parse(&body).unwrap();
        assert_eq!(cdef.channels.len(), 2);
        assert_eq!(
            cdef.channels[1],
            ChannelDef {
                channel: 1,
                typ: 1,
                assoc: 0
            }
        );
        // Wrong length rejected.
        body.push(0);
        assert!(ChannelDefinition::parse(&body).is_err());
    }

    #[test]
    fn profile_level_parse() {
        let pl = ProfileLevel::parse(&[0x12, 0x34, 0x56, 0x78]).unwrap();
        assert_eq!(pl.ppih, 0x1234);
        assert_eq!(pl.plev, 0x5678);
        assert!(ProfileLevel::parse(&[0, 0, 0]).is_err());
    }
}

#[cfg(test)]
mod writer_tests {
    use super::*;

    fn srgb() -> Cicp {
        Cicp {
            colour_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
            full_range: false,
        }
    }

    #[test]
    fn write_then_decode_luma_round_trips() {
        let (w, h) = (16u16, 8u16);
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 11) as u8)
            .collect();
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let file = write_jxs_file(&cs).unwrap();
        assert!(is_jxs_file(&file));
        // Parse re-derives the geometry from the codestream.
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.header.image_header.width, w as u32);
        assert_eq!(parsed.header.image_header.height, h as u32);
        assert_eq!(parsed.header.image_header.num_components, 1);
        assert_eq!(parsed.header.image_header.bit_depth(), Some(8));
        let img = decode_jxs_file(&file).unwrap();
        assert_eq!(img.planes[0].data, pixels);
    }

    #[test]
    fn write_rgb_with_channels_and_profile_level() {
        let (w, h) = (8u16, 8u16);
        let planes: Vec<Vec<u8>> = (0..3)
            .map(|c| {
                (0..(w as usize * h as usize))
                    .map(|i| (i + c * 17) as u8)
                    .collect()
            })
            .collect();
        // Cpih=0 (no colour transform) 3-component planar.
        let cs = crate::encoder::encode_planar(w, h, 3, 0, 1, 1, &planes).unwrap();
        let channels = vec![
            ChannelDef {
                channel: 0,
                typ: 0,
                assoc: 1,
            },
            ChannelDef {
                channel: 1,
                typ: 0,
                assoc: 2,
            },
            ChannelDef {
                channel: 2,
                typ: 0,
                assoc: 3,
            },
        ];
        let file = JxsFileBuilder::new(srgb())
            .channels(channels.clone())
            .profile_level(0x1234, 0x5678)
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.header.image_header.num_components, 3);
        let cdef = parsed.header.channel_def.as_ref().unwrap();
        assert_eq!(cdef.channels, channels);
        let pl = parsed.profile_level.unwrap();
        assert_eq!(pl.ppih, 0x1234);
        assert_eq!(pl.plev, 0x5678);
        let img = decode_jxs_file(&file).unwrap();
        assert_eq!(img.num_components, 3);
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(&img.planes[c].data, plane, "component {c} round-trips");
        }
    }

    #[test]
    fn write_high_bit_depth_sets_bpc() {
        // 12-bit luma: BPC low-7 == 11.
        let (w, h) = (8u16, 4u16);
        let samples: Vec<u16> = (0..(w as usize * h as usize))
            .map(|i| (i * 13) as u16)
            .collect();
        let cs = crate::encoder::encode_planar_highbd(
            w,
            h,
            1,
            0,
            1,
            1,
            12,
            std::slice::from_ref(&samples),
        )
        .unwrap();
        let file = write_jxs_file(&cs).unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.header.image_header.bit_depth(), Some(12));
        assert!(!parsed.header.image_header.is_signed());
        // Decodes back to the same 12-bit samples.
        let img = decode_jxs_file(&file).unwrap();
        assert_eq!(img.bit_depth, 12);
        assert_eq!(
            img.planes[0].data,
            samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<u8>>()
        );
    }

    #[test]
    fn write_rejects_invalid_codestream() {
        assert!(write_jxs_file(&[0xff, 0x10, 0x00]).is_err());
    }

    #[test]
    fn cicp_full_range_round_trips() {
        let (w, h) = (4u16, 4u16);
        let pixels = vec![100u8; (w * h) as usize];
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let mut cicp = srgb();
        cicp.full_range = true;
        cicp.colour_primaries = 9;
        let file = JxsFileBuilder::new(cicp).build(&cs).unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        let got = parsed.header.colour_specs[0].cicp.unwrap();
        assert!(got.full_range);
        assert_eq!(got.colour_primaries, 9);
    }
}
