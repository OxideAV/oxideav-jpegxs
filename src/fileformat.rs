//! JXS still-image file format — ISO/IEC 21122-3:2019 Annex A.
//!
//! The raw JPEG XS codestream (`SOC … EOC`, ISO/IEC 21122-1 Annex A) may
//! optionally be wrapped in the box-based **JXS file format** for still
//! images (file extension `.jxs`). This module parses that box structure
//! and extracts the embedded codestream so it can be handed to
//! [`crate::decode_components`].
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
/// JPEG XS Header (super)box type `'jxsh'` (A.5.4, 0x6A78_7368). The
/// 2019 text's prose and Table A.2 spell the name `'jp2h'` (the JP2
/// heritage name) but give this hex value, which reads `'jxsh'`; the hex
/// is what goes on the wire.
pub const TBOX_HEADER: u32 = 0x6A78_7368;
/// Image Header box type `'ihdr'` (A.5.4.2, 0x6968_6472).
pub const TBOX_IMAGE_HEADER: u32 = 0x6968_6472;
/// Colour Specification box type `'colr'` (A.5.4.3, 0x636F_6C72).
pub const TBOX_COLOUR: u32 = 0x636F_6C72;
/// Channel Definition box type `'cdef'` (A.5.4.4, 0x6364_6566).
pub const TBOX_CHANNEL_DEF: u32 = 0x6364_6566;
/// Exif box type `'exif'` (A.5.4 Table A.2, 0x6578_6966).
pub const TBOX_EXIF: u32 = 0x6578_6966;
/// Contiguous Codestream box type `'jp2c'` (A.5.5, 0x6A70_3263).
pub const TBOX_CODESTREAM: u32 = 0x6A70_3263;
/// JPEG XS Video Support (super)box type `'jpvs'` (A.5.3, 0x6A70_7673).
pub const TBOX_VIDEO_SUPPORT: u32 = 0x6A70_7673;
/// JPEG XS Video Information box type `'jpvi'` (A.5.3.2, 0x6A70_7669).
pub const TBOX_VIDEO_INFO: u32 = 0x6A70_7669;
/// JPEG XS Profile and Level box type `'jxpl'` (A.5.3.3, 0x6A78_706C).
pub const TBOX_PROFILE_LEVEL: u32 = 0x6A78_706C;
/// Buffer Model Description box type `'bmdm'` (A.5.3.4, 0x626D_646D).
pub const TBOX_BUFFER_MODEL: u32 = 0x626D_646D;
/// Mastering Display Metadata box type `'dmon'` (A.5.3.5, 0x646D_6F6E).
pub const TBOX_MASTERING_DISPLAY: u32 = 0x646D_6F6E;
/// JPEG XS Video Transport Parameter box type `'jptp'` (A.5.3.6, 0x6A70_7470).
pub const TBOX_TRANSPORT_PARAMS: u32 = 0x6A70_7470;
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

// ---------------------------------------------------------------------
// Media Type registrations — ISO/IEC 21122-3:2019 §A.7 / §C.5 / §C.6 /
// §D.2 (each following IETF RFC 6838).
// ---------------------------------------------------------------------

/// Media Type of the box-based JXS still-image file format (§A.7.2).
/// Registered magic: the 12-byte Signature box ([`SIGNATURE_BOX`]);
/// registered file extension: `jxs`.
pub const MEDIA_TYPE_JXS: &str = "image/jxs";

/// Media Type of a raw ISO/IEC 21122-1 codestream transported outside
/// any file format (§D.2.2 — e.g. an RTP payload). Registered magic:
/// the 4-byte prefix `FF10 FF50` (SOC immediately followed by CAP, the
/// mandatory 21122-1 Annex A marker order); registered file extension:
/// `jxsc`.
pub const MEDIA_TYPE_CODESTREAM: &str = "image/jxsc";

/// Media Type of an ISO/IEC 23008-12 (HEIF) file whose major brand
/// relates to a JPEG XS coded *image item* (§C.5.2). No registered
/// magic; registered file extension: `jxsi`. This crate does not parse
/// HEIF — the constant completes the Part-3 registration surface.
pub const MEDIA_TYPE_HEIF_IMAGE: &str = "image/jxsi";

/// Media Type of an ISO/IEC 23008-12 (HEIF) file whose major brand
/// relates to a JPEG XS coded *image sequence* (§C.6.2). No registered
/// magic; registered file extension: `jxss`. This crate does not parse
/// HEIF — the constant completes the Part-3 registration surface.
pub const MEDIA_TYPE_HEIF_SEQUENCE: &str = "image/jxss";

/// The §D.2.2 registered magic of a bare codestream: SOC (`FF10`)
/// immediately followed by CAP (`FF50`) — the mandatory ISO/IEC 21122-1
/// Annex A marker order, so every conforming codestream starts with it.
pub const CODESTREAM_MAGIC: [u8; 4] = [0xff, 0x10, 0xff, 0x50];

/// Classify a byte buffer by the two magic-number-bearing ISO/IEC
/// 21122-3 Media Type registrations: the §A.7.2 JXS box file (12-byte
/// Signature box) and the §D.2.2 bare codestream (`FF10 FF50`).
///
/// Returns [`MEDIA_TYPE_JXS`], [`MEDIA_TYPE_CODESTREAM`], or `None`
/// when neither registered magic matches (including the HEIF
/// registrations §C.5 / §C.6, which declare no magic — a HEIF reader
/// must resolve those from the FileTypeBox brand instead).
pub fn media_type(buf: &[u8]) -> Option<&'static str> {
    if is_jxs_file(buf) {
        Some(MEDIA_TYPE_JXS)
    } else if buf.len() >= CODESTREAM_MAGIC.len()
        && buf[..CODESTREAM_MAGIC.len()] == CODESTREAM_MAGIC
    {
        Some(MEDIA_TYPE_CODESTREAM)
    } else {
        None
    }
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
        if !body.len().is_multiple_of(4) {
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

/// Interlace mode of the coded frame (A.5.3.2 Table A.7 — the top two bits
/// of the `frat` field).
///
/// This is a property of the **frame**, not of the individual picture.
/// A.5.3.2 has the code specify "whether the original picture is
/// progressive or interlaced (as well as field order in the latter case)",
/// and the Annex B note has a two-picture (interlaced) sample indicated by
/// the one Video Support box of that sample. So values 1 and 2 give the
/// frame's field order — which video field the first coded picture of the
/// frame is — and a conforming interlaced stream carries the *same*
/// Interlace_Mode (and the same `jpvi` bytes) in both pictures of a frame.
/// Writers must not emit 1 for the first field and 2 for the second; the
/// Table A.7 row text ("picture is first video field") read on its own
/// invites that per-picture reading, and earlier releases of this module
/// documented the variants that way.
///
/// Value 3 is reserved (Table A.7); it parses as [`Self::Reserved`] and
/// round-trips verbatim instead of failing the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InterlaceMode {
    /// 0 — progressive frame (frame contains one full-height picture).
    Progressive,
    /// 1 — interlaced frame whose first coded picture is the first video
    /// field (field order "first field first"); signalled identically in
    /// both pictures of the frame.
    InterlacedFirstField,
    /// 2 — interlaced frame whose first coded picture is the second video
    /// field; signalled identically in both pictures of the frame.
    InterlacedSecondField,
    /// 3 — reserved by Table A.7; carried verbatim.
    Reserved,
}

impl InterlaceMode {
    /// Decode the 2-bit `Interlace_Mode` code (Table A.7). Only the low
    /// two bits of `code` are considered.
    pub fn from_code(code: u8) -> Self {
        match code & 0x03 {
            0 => InterlaceMode::Progressive,
            1 => InterlaceMode::InterlacedFirstField,
            2 => InterlaceMode::InterlacedSecondField,
            _ => InterlaceMode::Reserved,
        }
    }

    /// The 2-bit `Interlace_Mode` code (Table A.7).
    pub fn code(self) -> u8 {
        match self {
            InterlaceMode::Progressive => 0,
            InterlaceMode::InterlacedFirstField => 1,
            InterlaceMode::InterlacedSecondField => 2,
            InterlaceMode::Reserved => 3,
        }
    }

    /// `true` for the two interlaced codes (1 and 2): the frame is two
    /// coded pictures, one per video field, and the field rate is twice
    /// the `frat` frame rate (A.5.3.2).
    pub fn is_interlaced(self) -> bool {
        matches!(
            self,
            InterlaceMode::InterlacedFirstField | InterlaceMode::InterlacedSecondField
        )
    }
}

/// Frame-rate denominator selector (A.5.3.2 Table A.8): only the two
/// standardised values are legal; 0 and 3–63 are reserved and rejected on
/// parse (no frame rate can be computed from them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameRateDenominator {
    /// 1 — denominator value 1.000 (integer / PAL rates).
    Integer,
    /// 2 — denominator value 1.001 (NTSC pulldown rates).
    Ntsc,
}

/// Frame rate carried by the `frat` field (A.5.3.2 Table A.6). A `frat`
/// value of 0 means the frame rate is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRate {
    /// `frat == 0` — frame rate unknown.
    Unknown,
    /// Rational `numerator / denominator` frame rate with interlace
    /// signalling. The effective field rate is twice this when interlaced.
    Known {
        /// Progressive / interlaced field-order signalling.
        interlace: InterlaceMode,
        /// Denominator selector (1.000 or 1.001).
        denominator: FrameRateDenominator,
        /// Frame-rate numerator (Table A.6 `Framerate_Numerator`).
        numerator: u16,
    },
}

/// Sampling structure of the image (A.5.3.2 Table A.10 — the low nibble of
/// the `schar` field).
///
/// Table A.10 of ISO/IEC 21122-3:2019 assigns codes 0–2 and 4–6 and
/// reserves 3 and 7–15. Reserved codes parse as [`Self::Reserved`] and
/// round-trip verbatim instead of failing the box, so a stream from a
/// writer following a later edition of the table (4:2:0 sampling exists in
/// the codestream since ISO/IEC 21122-1:2022, and a code for it is expected
/// to be assigned where this edition says "Reserved") still opens; the
/// named variant for such a code is added once that edition's table is
/// available to this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplingStructure {
    /// 0 — 4:2:2 (YCbCr), `Nc = 3`.
    Yuv422,
    /// 1 — 4:4:4 (YCbCr), `Nc = 3`.
    Yuv444,
    /// 2 — 4:4:4 (RGB), `Nc = 3`.
    Rgb444,
    /// 4 — 4:2:2:4 (YCbCr + auxiliary), `Nc = 4`.
    YuvAux422,
    /// 5 — 4:4:4:4 (YCbCr + auxiliary), `Nc = 4`.
    YuvAux444,
    /// 6 — 4:4:4:4 (RGB + auxiliary), `Nc = 4`.
    RgbAux444,
    /// 3 or 7–15 — reserved by Table A.10 (2019 edition); the payload is
    /// the 4-bit code, carried verbatim.
    Reserved(u8),
}

impl SamplingStructure {
    /// Decode the 4-bit `Sampling_Structure` code (Table A.10). Only the
    /// low nibble of `code` is considered.
    pub fn from_code(code: u8) -> Self {
        match code & 0x0f {
            0 => SamplingStructure::Yuv422,
            1 => SamplingStructure::Yuv444,
            2 => SamplingStructure::Rgb444,
            4 => SamplingStructure::YuvAux422,
            5 => SamplingStructure::YuvAux444,
            6 => SamplingStructure::RgbAux444,
            other => SamplingStructure::Reserved(other),
        }
    }

    /// The 4-bit `Sampling_Structure` code (Table A.10).
    pub fn code(self) -> u8 {
        match self {
            SamplingStructure::Yuv422 => 0,
            SamplingStructure::Yuv444 => 1,
            SamplingStructure::Rgb444 => 2,
            SamplingStructure::YuvAux422 => 4,
            SamplingStructure::YuvAux444 => 5,
            SamplingStructure::RgbAux444 => 6,
            SamplingStructure::Reserved(code) => code & 0x0f,
        }
    }

    /// The Table A.10 `N_c` column: the number of components the sampling
    /// structure implies, or `None` for a reserved code.
    pub fn component_count(self) -> Option<u8> {
        match self {
            SamplingStructure::Yuv422 | SamplingStructure::Yuv444 | SamplingStructure::Rgb444 => {
                Some(3)
            }
            SamplingStructure::YuvAux422
            | SamplingStructure::YuvAux444
            | SamplingStructure::RgbAux444 => Some(4),
            SamplingStructure::Reserved(_) => None,
        }
    }
}

/// Sample characteristics carried by the `schar` field (A.5.3.2 Table A.9).
/// When `Valid_Flag` is 0 all sub-fields shall be 0 ([`Self::Invalid`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCharacteristics {
    /// `Valid_Flag == 0`: sample characteristics not signalled.
    Invalid,
    /// `Valid_Flag == 1`: bit depth and sampling structure are valid.
    Valid {
        /// `Sample_Bitdepth` — component bit depth minus 1 (4-bit field,
        /// so 0–15; mirrors the `ihdr` BPC low bits).
        bitdepth_minus1: u8,
        /// `Sampling_Structure` — chroma / auxiliary sampling.
        sampling: SamplingStructure,
    },
}

/// Time code of the coded frame (A.5.3.2 `tcod`, `HHMMSSFF`).
///
/// A.5.3.2 ranges the fields `HH` 0–23, `MM` 0–59, `SS` 0–59 and `FF`
/// 1–60, so no in-range value spells "no time code". The all-zero word is
/// accepted on both parse and write as [`Self::UNUSED`] — the time code
/// is not carried — because it cannot collide with any legal time code
/// (`FF >= 1`); it is what a sender that does not count frames in `tcod`
/// writes. Any other `FF == 0` (or out-of-range field) is still rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeCode {
    /// Hours (0–23).
    pub hours: u8,
    /// Minutes (0–59).
    pub minutes: u8,
    /// Seconds (0–59).
    pub seconds: u8,
    /// Frame count within the second (1–60; 0 only in [`Self::UNUSED`]).
    pub frames: u8,
}

impl TimeCode {
    /// The all-zero `tcod` word: time code not carried.
    pub const UNUSED: TimeCode = TimeCode {
        hours: 0,
        minutes: 0,
        seconds: 0,
        frames: 0,
    };

    /// A time code from its four fields (not range-checked here; the
    /// `jpvi` writer rejects out-of-range values).
    pub fn new(hours: u8, minutes: u8, seconds: u8, frames: u8) -> Self {
        TimeCode {
            hours,
            minutes,
            seconds,
            frames,
        }
    }

    /// `true` for the all-zero word ([`Self::UNUSED`]).
    pub fn is_unused(self) -> bool {
        self == Self::UNUSED
    }

    /// `true` when the word is one a conforming `jpvi` may carry: every
    /// field within its A.5.3.2 range, or the all-zero "unused" word.
    pub fn is_valid(self) -> bool {
        self.is_unused()
            || (self.hours <= 23
                && self.minutes <= 59
                && self.seconds <= 59
                && (1..=60).contains(&self.frames))
    }
}

/// JPEG XS Video Information box contents (`'jpvi'`, A.5.3.2 Table A.5). It
/// specifies frame rate, field coding, sample characteristics, time code
/// and maximum bit rate for a JPEG XS codestream used in a video stream.
/// This box is mandatory inside a JPEG XS Video Support superbox (A.5.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoInformation {
    /// `brat` — maximum instantaneous bit rate in Mbit/s (1 … 2³²−1).
    pub max_bit_rate: u32,
    /// `frat` — frame rate and interlace signalling.
    pub frame_rate: FrameRate,
    /// `schar` — sample characteristics (bit depth + sampling structure).
    pub sample: SampleCharacteristics,
    /// `tcod` — time code (`HH:MM:SS:FF`).
    pub timecode: TimeCode,
}

impl VideoInformation {
    /// A minimal conforming Video Information box carrying no useful
    /// signalling: the smallest legal bit rate (1 Mbit/s), an unknown frame
    /// rate, invalid sample characteristics, and time code `00:00:00:01`
    /// (the smallest word inside the Table A.5 ranges — `FF` is 1–60; a
    /// caller that wants the all-zero "not carried" word sets
    /// [`TimeCode::UNUSED`]). Used when a JPEG XS Video Support superbox
    /// must be emitted but the caller supplied no video metadata.
    pub fn unknown() -> Self {
        VideoInformation {
            max_bit_rate: 1,
            frame_rate: FrameRate::Unknown,
            sample: SampleCharacteristics::Invalid,
            timecode: TimeCode {
                hours: 0,
                minutes: 0,
                seconds: 0,
                frames: 1,
            },
        }
    }

    /// Parse the 14-byte `DBox` of a JPEG XS Video Information box
    /// (A.5.3.2 Table A.5: `brat` 4 B, `frat` 4 B, `schar` 2 B, `tcod` 4 B).
    pub fn parse(body: &[u8]) -> Result<VideoInformation> {
        if body.len() != 14 {
            return Err(JpegXsError::invalid(
                "jxs jpvi: body must be exactly 14 bytes (brat, frat, schar, tcod)",
            ));
        }
        let brat = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
        // brat is 1 … 2³²−1 (Table A.5): 0 is not a legal maximum bit rate.
        if brat == 0 {
            return Err(JpegXsError::invalid(
                "jxs jpvi: brat (max bit rate) must be non-zero (A.5.3.2)",
            ));
        }

        let frat = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
        let frame_rate = if frat == 0 {
            FrameRate::Unknown
        } else {
            // Interlace_Mode (Table A.7): a frame-level field-order code,
            // identical in both pictures of an interlaced frame. The
            // reserved value 3 is carried, not rejected.
            let interlace = InterlaceMode::from_code((frat >> 30) as u8);
            let denominator = match (frat >> 24) & 0x3f {
                1 => FrameRateDenominator::Integer,
                2 => FrameRateDenominator::Ntsc,
                _ => {
                    return Err(JpegXsError::invalid(
                        "jxs jpvi: frat Framerate_Denominator is reserved (Table A.8)",
                    ));
                }
            };
            FrameRate::Known {
                interlace,
                denominator,
                numerator: (frat & 0xffff) as u16,
            }
        };

        let schar = u16::from_be_bytes([body[8], body[9]]);
        let sample = if schar & 0x8000 == 0 {
            // Valid_Flag == 0: all schar sub-fields shall be 0.
            if schar != 0 {
                return Err(JpegXsError::invalid(
                    "jxs jpvi: schar Valid_Flag is 0 but sub-fields are non-zero (A.5.3.2)",
                ));
            }
            SampleCharacteristics::Invalid
        } else {
            let bitdepth_minus1 = ((schar >> 4) & 0x0f) as u8;
            // Sampling_Structure (Table A.10): reserved codes (3, 7–15 in
            // the 2019 table) are carried as `Reserved`, not rejected.
            let sampling = SamplingStructure::from_code((schar & 0x0f) as u8);
            SampleCharacteristics::Valid {
                bitdepth_minus1,
                sampling,
            }
        };

        let timecode = TimeCode {
            hours: body[8 + 2],
            minutes: body[8 + 3],
            seconds: body[8 + 4],
            frames: body[8 + 5],
        };
        // tcod ranges (A.5.3.2): HH 0-23, MM 0-59, SS 0-59, FF 1-60 — or
        // the all-zero "time code not carried" word.
        if !timecode.is_valid() {
            return Err(JpegXsError::invalid(
                "jxs jpvi: tcod field out of range (HH 0-23, MM 0-59, SS 0-59, FF 1-60, or all zero)",
            ));
        }

        Ok(VideoInformation {
            max_bit_rate: brat,
            frame_rate,
            sample,
            timecode,
        })
    }

    /// Serialize the 14-byte `DBox` of a JPEG XS Video Information box.
    /// Rejects out-of-range values a conforming box cannot carry.
    pub fn to_bytes(&self) -> Result<[u8; 14]> {
        if self.max_bit_rate == 0 {
            return Err(JpegXsError::invalid(
                "jxs jpvi: brat (max bit rate) must be non-zero (A.5.3.2)",
            ));
        }
        let frat: u32 = match self.frame_rate {
            FrameRate::Unknown => 0,
            FrameRate::Known {
                interlace,
                denominator,
                numerator,
            } => {
                let im = u32::from(interlace.code());
                let den = match denominator {
                    FrameRateDenominator::Integer => 1u32,
                    FrameRateDenominator::Ntsc => 2,
                };
                (im << 30) | (den << 24) | u32::from(numerator)
            }
        };
        let schar: u16 = match self.sample {
            SampleCharacteristics::Invalid => 0,
            SampleCharacteristics::Valid {
                bitdepth_minus1,
                sampling,
            } => {
                if bitdepth_minus1 > 0x0f {
                    return Err(JpegXsError::invalid(
                        "jxs jpvi: schar Sample_Bitdepth exceeds the 4-bit field (Table A.9)",
                    ));
                }
                let ss = u16::from(sampling.code());
                0x8000 | (u16::from(bitdepth_minus1) << 4) | ss
            }
        };
        let tc = self.timecode;
        if !tc.is_valid() {
            return Err(JpegXsError::invalid(
                "jxs jpvi: tcod field out of range (HH 0-23, MM 0-59, SS 0-59, FF 1-60, or all zero)",
            ));
        }

        let mut out = [0u8; 14];
        out[0..4].copy_from_slice(&self.max_bit_rate.to_be_bytes());
        out[4..8].copy_from_slice(&frat.to_be_bytes());
        out[8..10].copy_from_slice(&schar.to_be_bytes());
        // tcod (HHMMSSFF) — four contiguous bytes.
        out[10] = tc.hours;
        out[11] = tc.minutes;
        out[12] = tc.seconds;
        out[13] = tc.frames;
        Ok(out)
    }
}

/// Buffer Model Description box contents (`'bmdm'`, A.5.3.4 Table A.12).
/// Stores the buffer-model metadata (type + blanking-period coefficient
/// group counts) needed to test a codestream against the ISO/IEC 21122-2
/// buffer model. 6 bytes: `Tbmd` (8), `Rd` (8), `Ncg,hz` (16), `Ncg,vt`
/// (16), big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferModelDescription {
    /// `Tbmd` — buffer model type (0, 1 or 2; ISO/IEC 21122-2).
    pub model_type: u8,
    /// `Ncg,hz` — coefficient groups spanning a horizontal blanking period.
    pub ncg_horizontal: u16,
    /// `Ncg,vt` — coefficient groups spanning a vertical blanking period.
    pub ncg_vertical: u16,
}

impl BufferModelDescription {
    /// Parse the 6-byte `DBox` of a Buffer Model Description box. `Tbmd`
    /// shall be 0/1/2 and `Rd` shall be 0 (Table A.12); other values are
    /// reserved and rejected.
    pub fn parse(body: &[u8]) -> Result<BufferModelDescription> {
        if body.len() != 6 {
            return Err(JpegXsError::invalid(
                "jxs bmdm: body must be exactly 6 bytes (A.5.3.4)",
            ));
        }
        let model_type = body[0];
        if model_type > 2 {
            return Err(JpegXsError::invalid(
                "jxs bmdm: Tbmd must be 0, 1 or 2 (Table A.12)",
            ));
        }
        // Rd shall be 0; all other values reserved.
        if body[1] != 0 {
            return Err(JpegXsError::invalid("jxs bmdm: Rd must be 0 (Table A.12)"));
        }
        Ok(BufferModelDescription {
            model_type,
            ncg_horizontal: u16::from_be_bytes([body[2], body[3]]),
            ncg_vertical: u16::from_be_bytes([body[4], body[5]]),
        })
    }

    /// Serialize the 6-byte `DBox` (`Rd` = 0 by construction). Rejects a
    /// `Tbmd` outside 0/1/2.
    pub fn to_bytes(&self) -> Result<[u8; 6]> {
        if self.model_type > 2 {
            return Err(JpegXsError::invalid(
                "jxs bmdm: Tbmd must be 0, 1 or 2 (Table A.12)",
            ));
        }
        let mut out = [0u8; 6];
        out[0] = self.model_type;
        // out[1] = Rd = 0.
        out[2..4].copy_from_slice(&self.ncg_horizontal.to_be_bytes());
        out[4..6].copy_from_slice(&self.ncg_vertical.to_be_bytes());
        Ok(out)
    }
}

/// Mastering Display Metadata box contents (`'dmon'`, A.5.3.5 Figure A.11).
/// The chromaticity primaries and white point mirror the fields of SMPTE
/// ST 2086; `MCLL` / `MFALL` are the CTA-861-G content light levels. All
/// fields are big-endian unsigned integers, 28 bytes total.
///
/// Chromaticity coordinates are normalised in increments of 0.00002 (CIE
/// 1931), range 0…50000. Primary index `c0` is the green primary, `c1` the
/// blue, `c2` the red (A.5.3.5). Luminance values are in 0.0001 cd/m²
/// increments (`Lmin < Lmax`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplayMetadata {
    /// `Xc0`, `Yc0` — green primary chromaticity (0…50000).
    pub primary_green: (u16, u16),
    /// `Xc1`, `Yc1` — blue primary chromaticity (0…50000).
    pub primary_blue: (u16, u16),
    /// `Xc2`, `Yc2` — red primary chromaticity (0…50000).
    pub primary_red: (u16, u16),
    /// `Xwp`, `Ywp` — white point chromaticity (0…50000).
    pub white_point: (u16, u16),
    /// `Lmin` — minimum display luminance (×0.0001 cd/m²).
    pub min_luminance: u32,
    /// `Lmax` — maximum display luminance (×0.0001 cd/m²); `> Lmin`.
    pub max_luminance: u32,
    /// `MCLL` — Maximum Content Light Level (0x0000 if unknown).
    pub max_content_light_level: u16,
    /// `MFALL` — Maximum Frame Average Light Level (0x0000 if unknown).
    pub max_frame_avg_light_level: u16,
}

impl MasteringDisplayMetadata {
    fn coord_in_range((x, y): (u16, u16)) -> bool {
        x <= 50_000 && y <= 50_000
    }

    /// Parse the 28-byte `DBox` of a Mastering Display Metadata box.
    pub fn parse(body: &[u8]) -> Result<MasteringDisplayMetadata> {
        if body.len() != 28 {
            return Err(JpegXsError::invalid(
                "jxs dmon: body must be exactly 28 bytes (A.5.3.5)",
            ));
        }
        let u16at = |i: usize| u16::from_be_bytes([body[i], body[i + 1]]);
        let u32at = |i: usize| u32::from_be_bytes([body[i], body[i + 1], body[i + 2], body[i + 3]]);
        let out = MasteringDisplayMetadata {
            primary_green: (u16at(0), u16at(2)),
            primary_blue: (u16at(4), u16at(6)),
            primary_red: (u16at(8), u16at(10)),
            white_point: (u16at(12), u16at(14)),
            min_luminance: u32at(16),
            max_luminance: u32at(20),
            max_content_light_level: u16at(24),
            max_frame_avg_light_level: u16at(26),
        };
        // Chromaticity coordinates shall be in 0…50000 (A.5.3.5).
        if !Self::coord_in_range(out.primary_green)
            || !Self::coord_in_range(out.primary_blue)
            || !Self::coord_in_range(out.primary_red)
            || !Self::coord_in_range(out.white_point)
        {
            return Err(JpegXsError::invalid(
                "jxs dmon: chromaticity coordinate out of range 0…50000 (A.5.3.5)",
            ));
        }
        // Lmin shall be less than Lmax (A.5.3.5).
        if out.min_luminance >= out.max_luminance {
            return Err(JpegXsError::invalid(
                "jxs dmon: Lmin must be less than Lmax (A.5.3.5)",
            ));
        }
        Ok(out)
    }

    /// Serialize the 28-byte `DBox`. Rejects out-of-range values.
    pub fn to_bytes(&self) -> Result<[u8; 28]> {
        if !Self::coord_in_range(self.primary_green)
            || !Self::coord_in_range(self.primary_blue)
            || !Self::coord_in_range(self.primary_red)
            || !Self::coord_in_range(self.white_point)
        {
            return Err(JpegXsError::invalid(
                "jxs dmon: chromaticity coordinate out of range 0…50000 (A.5.3.5)",
            ));
        }
        if self.min_luminance >= self.max_luminance {
            return Err(JpegXsError::invalid(
                "jxs dmon: Lmin must be less than Lmax (A.5.3.5)",
            ));
        }
        let mut out = [0u8; 28];
        let mut put16 = |i: usize, v: u16| out[i..i + 2].copy_from_slice(&v.to_be_bytes());
        put16(0, self.primary_green.0);
        put16(2, self.primary_green.1);
        put16(4, self.primary_blue.0);
        put16(6, self.primary_blue.1);
        put16(8, self.primary_red.0);
        put16(10, self.primary_red.1);
        put16(12, self.white_point.0);
        put16(14, self.white_point.1);
        out[16..20].copy_from_slice(&self.min_luminance.to_be_bytes());
        out[20..24].copy_from_slice(&self.max_luminance.to_be_bytes());
        out[24..26].copy_from_slice(&self.max_content_light_level.to_be_bytes());
        out[26..28].copy_from_slice(&self.max_frame_avg_light_level.to_be_bytes());
        Ok(out)
    }
}

/// JPEG XS Video Transport Parameter box contents (`'jptp'`, A.5.3.6
/// Table A.15). Advises a packet-transport decoder on slice-to-core
/// assignment and reorder-buffer sizing. 6 bytes: `Slgs` (16), `Rsync`
/// (8), `Tseq` (8), `MTU` (16), big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoTransportParameters {
    /// `Slgs` — contiguous slices suggested per processing unit (0 = no
    /// suggestion).
    pub slice_group_size: u16,
    /// `Rsync` — suggested parallel decoding units (0 = no suggestion).
    pub parallel_units: u8,
}

impl VideoTransportParameters {
    /// Parse the 6-byte `DBox` of a Video Transport Parameter box. `Tseq`
    /// and `MTU` shall both be 0 (Table A.14 / A.15); non-zero is rejected.
    pub fn parse(body: &[u8]) -> Result<VideoTransportParameters> {
        if body.len() != 6 {
            return Err(JpegXsError::invalid(
                "jxs jptp: body must be exactly 6 bytes (A.5.3.6)",
            ));
        }
        let slice_group_size = u16::from_be_bytes([body[0], body[1]]);
        let parallel_units = body[2];
        let tseq = body[3];
        let mtu = u16::from_be_bytes([body[4], body[5]]);
        // Tseq: only 0 is defined; all other values reserved (Table A.14).
        if tseq != 0 {
            return Err(JpegXsError::invalid(
                "jxs jptp: Tseq must be 0 (Table A.14)",
            ));
        }
        // MTU shall be 0 (Table A.15); all other values reserved.
        if mtu != 0 {
            return Err(JpegXsError::invalid(
                "jxs jptp: MTU field must be 0 (Table A.15)",
            ));
        }
        Ok(VideoTransportParameters {
            slice_group_size,
            parallel_units,
        })
    }

    /// Serialize the 6-byte `DBox` (`Tseq` = 0, `MTU` = 0 by construction).
    pub fn to_bytes(&self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[0..2].copy_from_slice(&self.slice_group_size.to_be_bytes());
        out[2] = self.parallel_units;
        // out[3] = Tseq = 0, out[4..6] = MTU = 0.
        out
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
    /// Optional Exif box payload (A.5.4 Table A.2), if present. The EXIF
    /// contents themselves are opaque to this parser.
    pub exif: Option<Vec<u8>>,
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
        let mut exif = None;
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
                TBOX_EXIF if exif.is_none() => {
                    exif = Some(s.payload(buf).to_vec());
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
            exif,
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
    /// Optional Video Information box (A.5.3.2), if a JPEG XS Video
    /// Support superbox carried one.
    pub video_info: Option<VideoInformation>,
    /// Optional Buffer Model Description box (A.5.3.4).
    pub buffer_model: Option<BufferModelDescription>,
    /// Optional Mastering Display Metadata box (A.5.3.5).
    pub mastering_display: Option<MasteringDisplayMetadata>,
    /// Optional Video Transport Parameter box (A.5.3.6).
    pub transport_params: Option<VideoTransportParameters>,
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
    let mut video_info: Option<VideoInformation> = None;
    let mut buffer_model: Option<BufferModelDescription> = None;
    let mut mastering_display: Option<MasteringDisplayMetadata> = None;
    let mut transport_params: Option<VideoTransportParameters> = None;
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
                // Optional superbox; pull out the jpvi Video Information
                // (A.5.3.2) and jxpl Profile/Level (A.5.3.3) boxes if
                // present. Read is order-tolerant (A.6); other inner boxes
                // (bmdm / dmon / jptp) are ignored.
                let inner = parse_boxes(buf, s.payload_start, s.payload_start + s.payload_len)?;
                for is in &inner {
                    match is.tbox {
                        TBOX_VIDEO_INFO if video_info.is_none() => {
                            video_info = Some(VideoInformation::parse(is.payload(buf))?);
                        }
                        TBOX_PROFILE_LEVEL if profile_level.is_none() => {
                            profile_level = Some(ProfileLevel::parse(is.payload(buf))?);
                        }
                        TBOX_BUFFER_MODEL if buffer_model.is_none() => {
                            buffer_model = Some(BufferModelDescription::parse(is.payload(buf))?);
                        }
                        TBOX_MASTERING_DISPLAY if mastering_display.is_none() => {
                            mastering_display =
                                Some(MasteringDisplayMetadata::parse(is.payload(buf))?);
                        }
                        TBOX_TRANSPORT_PARAMS if transport_params.is_none() => {
                            transport_params =
                                Some(VideoTransportParameters::parse(is.payload(buf))?);
                        }
                        _ => {}
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
        video_info,
        buffer_model,
        mastering_display,
        transport_params,
        codestream_span,
    })
}

/// Decode a JXS file (ISO/IEC 21122-3 Annex A) to its contract image.
///
/// Historical entry point: [`crate::decode`] accepts both bare
/// codestreams and `.jxs` files; [`crate::decode_components`] gives the
/// codestream-order component planes.
#[deprecated(note = "use oxideav_jpegxs::decode (IMAGE_CRATE_API)")]
pub fn decode_jxs_file(buf: &[u8]) -> Result<crate::image::JpegXsImage> {
    crate::decode(buf)
}

/// Cross-check a parsed `.jxs` file against the codestream it wraps:
/// the Image Header box `NC` / `WIDTH` / `HEIGHT` against the picture
/// header (A.5.4.2 declares them redundant and contradictory files
/// non-conforming) and the `jxpl` Profile/Level box (when present)
/// against the codestream declarations it duplicates.
pub(crate) fn check_file_consistency(
    file: &JxsFile,
    cs: &crate::codestream::Codestream,
    codestream_len: usize,
) -> Result<()> {
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
    if let Some(pl) = &file.profile_level {
        check_jxpl_consistency(cs, codestream_len, pl)?;
    }
    Ok(())
}

/// Enforce that a `jxpl` Profile/Level box (A.5.3.3) does not
/// misdescribe the codestream it accompanies. The box duplicates the
/// PIH `Ppih` / `Plev` fields ("Profile of the codestream ... Level of
/// the codestream" — A.5.3.3 calls the information redundant with the
/// bitstream), so:
///
/// * When the codestream itself declares a non-zero `Ppih` / `Plev`,
///   a non-zero box field must repeat it exactly — a differing value
///   is a contradictory file, rejected like the A.5.4.2 geometry
///   mismatches. (A zero box field alongside a declared codestream is
///   tolerated as "no early-parse hint".)
/// * When the codestream declares nothing (`Ppih = 0` / `Plev = 0`)
///   and the box maps to a *known* 21122-2 profile / level, the claim
///   is verified against the stream through the same
///   [`crate::profile`] gates the decoder applies to PIH declarations
///   — a file that advertises a profile its stream does not satisfy is
///   rejected.
/// * A box value that maps to no known profile / level is tolerated:
///   the box is advisory metadata and a future profile must not make
///   old files undecodable (unlike the PIH `Ppih`, whose reserved
///   values a conforming *encoder* cannot emit).
fn check_jxpl_consistency(
    cs: &crate::codestream::Codestream,
    codestream_len: usize,
    pl: &ProfileLevel,
) -> Result<()> {
    if pl.ppih != 0 {
        if cs.pih.ppih != 0 {
            if pl.ppih != cs.pih.ppih {
                return Err(JpegXsError::invalid(format!(
                    "jxs file: jxpl Ppih=0x{:04X} disagrees with codestream Ppih=0x{:04X} \
                     (A.5.3.3 — the box duplicates the codestream field)",
                    pl.ppih, cs.pih.ppih
                )));
            }
        } else if let Some(profile) = crate::profile::Profile::from_ppih(pl.ppih) {
            crate::profile::check_codestream(cs, profile).map_err(|e| {
                JpegXsError::invalid(format!(
                    "jxs file: jxpl claims profile {} but the codestream does not satisfy \
                     it: {e}",
                    profile.name()
                ))
            })?;
        }
    }
    if pl.plev != 0 {
        if cs.pih.plev != 0 {
            if pl.plev != cs.pih.plev {
                return Err(JpegXsError::invalid(format!(
                    "jxs file: jxpl Plev=0x{:04X} disagrees with codestream Plev=0x{:04X} \
                     (A.5.3.3 — the box duplicates the codestream field)",
                    pl.plev, cs.pih.plev
                )));
            }
        } else if crate::profile::Level::from_plev_high(pl.plev).is_some() {
            // Verify the level / sublevel claim against the stream by
            // substituting the box's Plev into a probe header and
            // running the decoder's own gates.
            let mut probe = cs.clone();
            probe.pih.plev = pl.plev;
            probe.pih.ppih = if pl.ppih != 0 { pl.ppih } else { cs.pih.ppih };
            crate::profile::check_level(&probe).map_err(|e| {
                JpegXsError::invalid(format!(
                    "jxs file: jxpl claims a level the codestream does not satisfy: {e}"
                ))
            })?;
            crate::profile::check_codestream_size(&probe, codestream_len).map_err(|e| {
                JpegXsError::invalid(format!(
                    "jxs file: jxpl claims a sublevel the codestream does not satisfy: {e}"
                ))
            })?;
        }
    }
    Ok(())
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
    exif: Option<Vec<u8>>,
    profile_level: Option<ProfileLevel>,
    video_info: Option<VideoInformation>,
    buffer_model: Option<BufferModelDescription>,
    mastering_display: Option<MasteringDisplayMetadata>,
    transport_params: Option<VideoTransportParameters>,
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
            exif: None,
            profile_level: None,
            video_info: None,
            buffer_model: None,
            mastering_display: None,
            transport_params: None,
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

    /// Add an Exif box (A.5.4 Table A.2) inside the JPEG XS Header box,
    /// carrying the given opaque EXIF payload.
    pub fn exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }

    /// Add a Profile/Level box (`jxpl`, A.5.3.3) to the JPEG XS Video
    /// Support superbox. Setting this (or [`Self::video_information`])
    /// causes a conforming `jpvs` superbox to be emitted — A.5.3.1 makes
    /// both the `jpvi` Video Information box and the `jxpl` Profile/Level
    /// box mandatory inside `jpvs`, so a default `jpvi`
    /// ([`VideoInformation::unknown`]) and/or a default `jxpl`
    /// (`Unrestricted` / `0x0000`) is synthesised for whichever the caller
    /// did not supply.
    pub fn profile_level(mut self, ppih: u16, plev: u16) -> Self {
        self.profile_level = Some(ProfileLevel { ppih, plev });
        self
    }

    /// Add a Video Information box (`jpvi`, A.5.3.2) to the JPEG XS Video
    /// Support superbox. See [`Self::profile_level`] for the emission
    /// rules — the two boxes are co-mandatory inside `jpvs`.
    pub fn video_information(mut self, info: VideoInformation) -> Self {
        self.video_info = Some(info);
        self
    }

    /// Add a Buffer Model Description box (`bmdm`, A.5.3.4) to the JPEG XS
    /// Video Support superbox. Implies a `jpvs` (see
    /// [`Self::profile_level`]).
    pub fn buffer_model(mut self, desc: BufferModelDescription) -> Self {
        self.buffer_model = Some(desc);
        self
    }

    /// Add a Mastering Display Metadata box (`dmon`, A.5.3.5) to the JPEG
    /// XS Video Support superbox. Implies a `jpvs` (see
    /// [`Self::profile_level`]).
    pub fn mastering_display(mut self, meta: MasteringDisplayMetadata) -> Self {
        self.mastering_display = Some(meta);
        self
    }

    /// Add a Video Transport Parameter box (`jptp`, A.5.3.6) to the JPEG
    /// XS Video Support superbox. Implies a `jpvs` (see
    /// [`Self::profile_level`]).
    pub fn transport_parameters(mut self, params: VideoTransportParameters) -> Self {
        self.transport_params = Some(params);
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
        // BPC (Table A.17): if every component shares one bit depth, encode
        // `depth − 1` in the low 7 bits (unsigned components). If the
        // components vary, emit the `0xFF` "components vary in bit depth"
        // code, in which case the per-component depths come from the
        // codestream rather than the ihdr.
        let depths: Vec<u8> = cs.cdt.components.iter().map(|c| c.bit_depth).collect();
        let uniform = depths
            .first()
            .copied()
            .filter(|&d| depths.iter().all(|&x| x == d));
        let bpc = match uniform {
            Some(d) if (1..=38).contains(&d) => d - 1,
            Some(_) => {
                return Err(JpegXsError::invalid(
                    "jxs writer: component bit depth out of the Table A.17 range",
                ));
            }
            None => 0xFF, // components vary in bit depth (Table A.17)
        };
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
        if let Some(exif) = &self.exif {
            serialize_box(&mut jp2h, TBOX_EXIF, exif);
        }
        serialize_box(&mut file, TBOX_HEADER, &jp2h);

        // Optional JPEG XS Video Support superbox (jpvs). A.5.3.1 makes
        // both the jpvi Video Information box and the jxpl Profile/Level
        // box mandatory inside jpvs, and A.5.3 fixes their order (jpvi
        // first, jxpl second). If the caller supplied only one, synthesise
        // a conforming default for the other so the emitted superbox is
        // always valid.
        if self.profile_level.is_some()
            || self.video_info.is_some()
            || self.buffer_model.is_some()
            || self.mastering_display.is_some()
            || self.transport_params.is_some()
        {
            let mut jpvs = Vec::new();
            // jpvi — Video Information box (first).
            let info = self.video_info.unwrap_or_else(VideoInformation::unknown);
            serialize_box(&mut jpvs, TBOX_VIDEO_INFO, &info.to_bytes()?);
            // jxpl — Profile and Level box (second). A.5.3.3 defines the
            // box fields as "Profile of the codestream" / "Level of the
            // codestream" — a redundant early-parse copy of the PIH
            // fields — so the synthesised default mirrors whatever the
            // codestream itself declares, and a caller-supplied value is
            // held to the same consistency gate `decode_jxs_file`
            // enforces (an emitted file must not misdescribe its own
            // stream).
            let pl = self.profile_level.unwrap_or(ProfileLevel {
                ppih: cs.pih.ppih,
                plev: cs.pih.plev,
            });
            check_jxpl_consistency(&cs, codestream.len(), &pl)?;
            let mut jxpl = Vec::new();
            jxpl.extend_from_slice(&pl.ppih.to_be_bytes());
            jxpl.extend_from_slice(&pl.plev.to_be_bytes());
            serialize_box(&mut jpvs, TBOX_PROFILE_LEVEL, &jxpl);
            // Optional bmdm / dmon / jptp boxes follow jxpl (Figure A.7).
            if let Some(desc) = &self.buffer_model {
                serialize_box(&mut jpvs, TBOX_BUFFER_MODEL, &desc.to_bytes()?);
            }
            if let Some(meta) = &self.mastering_display {
                serialize_box(&mut jpvs, TBOX_MASTERING_DISPLAY, &meta.to_bytes()?);
            }
            if let Some(params) = &self.transport_params {
                serialize_box(&mut jpvs, TBOX_TRANSPORT_PARAMS, &params.to_bytes());
            }
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
#[allow(deprecated)]
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
        let img = crate::decode_components(&file).expect("decode jxs file");
        assert_eq!(img.width, w as u32);
        assert_eq!(img.height, h as u32);
        assert_eq!(img.len(), 1);
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
        let err = crate::decode_components(&file).unwrap_err();
        assert!(matches!(err, JpegXsError::InvalidData(_)));
    }

    #[test]
    fn rejects_ihdr_dimension_mismatch() {
        let (cs, _w, h) = luma_codestream();
        let file = wrap_codestream(&cs, 99, h, 1, 8, false);
        assert!(crate::decode_components(&file).is_err());
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
        let img = crate::decode_components(&spliced).expect("decode with unknown boxes");
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
#[allow(deprecated)]
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
        let img = crate::decode_components(&file).unwrap();
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
        let img = crate::decode_components(&file).unwrap();
        assert_eq!(img.len(), 3);
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(&img.planes[c].data, plane, "component {c} round-trips");
        }
    }

    /// A profile-shaped, fully-declared codestream: Main 444.12 (8-bit
    /// member of its bit-depth set), 16-image-row slices, CBR Lcod.
    fn declared_codestream() -> (Vec<u8>, Vec<Vec<u8>>) {
        let (w, h) = (32usize, 32usize);
        let planes: Vec<Vec<u8>> = (0..3)
            .map(|c| (0..w * h).map(|i| (i * 7 + c * 29) as u8).collect())
            .collect();
        let mut cs =
            crate::encoder::encode_planar_hsl(w as u16, h as u16, 3, 1, 2, 1, 0, 8, &planes)
                .unwrap();
        crate::signalling::declare_profile(&mut cs, crate::profile::Profile::Main444_12).unwrap();
        crate::signalling::declare_level_sublevel(
            &mut cs,
            crate::profile::Level::L2k1,
            crate::profile::Sublevel::Sublev3bpp,
        )
        .unwrap();
        (cs, planes)
    }

    #[test]
    fn jxpl_defaults_mirror_codestream_declarations() {
        // A.5.3.3 — the box duplicates the PIH fields, so the builder's
        // synthesised jxpl mirrors a declared codestream.
        let (cs, planes) = declared_codestream();
        let file = JxsFileBuilder::new(srgb())
            .video_information(VideoInformation::unknown())
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        let pl = parsed.profile_level.unwrap();
        assert_eq!(pl.ppih, crate::profile::Profile::Main444_12.ppih());
        assert_eq!(pl.plev, 0x1004); // 2k-1 | Sublev3bpp
        let img = crate::decode_components(&file).unwrap();
        for (c, plane) in planes.iter().enumerate() {
            assert_eq!(&img.planes[c].data, plane, "component {c}");
        }
    }

    #[test]
    fn jxpl_contradicting_codestream_rejected_on_build() {
        // The codestream declares Main 444.12; a caller-supplied jxpl
        // naming a different known profile misdescribes the stream.
        let (cs, _) = declared_codestream();
        let err = JxsFileBuilder::new(srgb())
            .profile_level(crate::profile::Profile::Light444_12.ppih(), 0)
            .build(&cs)
            .unwrap_err();
        assert!(
            format!("{err}").contains("disagrees with codestream Ppih"),
            "expected jxpl/PIH contradiction, got {err}"
        );
    }

    #[test]
    fn jxpl_contradicting_codestream_rejected_on_decode() {
        // Patch the emitted jxpl payload to a different known profile:
        // the reader must reject the contradictory file.
        let (cs, _) = declared_codestream();
        let mut file = JxsFileBuilder::new(srgb())
            .video_information(VideoInformation::unknown())
            .build(&cs)
            .unwrap();
        let tag = TBOX_PROFILE_LEVEL.to_be_bytes();
        let pos = file
            .windows(4)
            .position(|w| w == tag)
            .expect("jxpl box present");
        let want = crate::profile::Profile::Main444_12.ppih().to_be_bytes();
        assert_eq!(&file[pos + 4..pos + 6], &want, "payload follows the tag");
        let other = crate::profile::Profile::High444_12.ppih().to_be_bytes();
        file[pos + 4..pos + 6].copy_from_slice(&other);
        let err = crate::decode_components(&file).unwrap_err();
        assert!(
            format!("{err}").contains("disagrees with codestream Ppih"),
            "expected jxpl/PIH contradiction, got {err}"
        );
    }

    #[test]
    fn jxpl_claim_over_undeclared_stream_is_verified() {
        // The codestream declares nothing (Ppih = 0). A jxpl naming a
        // known profile is verified against the stream: a single
        // whole-picture slice violates every profile's 16-image-row
        // slice rule, so the claim is rejected...
        let (w, h) = (32u16, 32u16);
        let planes: Vec<Vec<u8>> = (0..3)
            .map(|c| {
                (0..(w as usize * h as usize))
                    .map(|i| (i * 3 + c * 11) as u8)
                    .collect()
            })
            .collect();
        let single_slice = crate::encoder::encode_planar(w, h, 3, 1, 2, 1, &planes).unwrap();
        let err = JxsFileBuilder::new(srgb())
            .profile_level(crate::profile::Profile::Main444_12.ppih(), 0)
            .build(&single_slice)
            .unwrap_err();
        assert!(
            format!("{err}").contains("does not satisfy"),
            "expected unsatisfied-claim rejection, got {err}"
        );
        // ... while the same claim over a stream that satisfies the
        // profile (16-image-row slices) is accepted and decodable, and
        // an unknown (future-profile) code point stays tolerated.
        let sliced = crate::encoder::encode_planar_hsl(w, h, 3, 1, 2, 1, 0, 8, &planes).unwrap();
        let file = JxsFileBuilder::new(srgb())
            .profile_level(crate::profile::Profile::Main444_12.ppih(), 0)
            .build(&sliced)
            .unwrap();
        crate::decode_components(&file).unwrap();
        let file = JxsFileBuilder::new(srgb())
            .profile_level(0x7777, 0x7700)
            .build(&single_slice)
            .unwrap();
        crate::decode_components(&file).unwrap();
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
        let img = crate::decode_components(&file).unwrap();
        assert_eq!(img.max_bit_depth(), 12);
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

    fn luma_cs() -> Vec<u8> {
        let (w, h) = (4u16, 4u16);
        let pixels = vec![64u8; (w * h) as usize];
        crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap()
    }

    /// A luma codestream that *satisfies* the Main-family profile
    /// constraints (mono 4:0:0, 8-bit, `NL = 2/1`, 16-image-row slices)
    /// so a known-profile `jxpl` claim passes the A.5.3.3 consistency
    /// gate.
    fn profile_shaped_luma_cs() -> Vec<u8> {
        let (w, h) = (32u16, 32u16);
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 5) as u8)
            .collect();
        crate::encoder::encode_planar_hsl(w, h, 1, 0, 2, 1, 0, 8, std::slice::from_ref(&pixels))
            .unwrap()
    }

    // ---- JPEG XS Video Information box (jpvi, A.5.3.2) ----

    #[test]
    fn jpvi_round_trips_known_frame_rate() {
        // 23.98 fps NTSC pulldown (num=24, denom=1.001), first field,
        // 8-bit 4:4:4 YCbCr, timecode 01:02:03:04, 500 Mbit/s.
        let info = VideoInformation {
            max_bit_rate: 500,
            frame_rate: FrameRate::Known {
                interlace: InterlaceMode::InterlacedFirstField,
                denominator: FrameRateDenominator::Ntsc,
                numerator: 24,
            },
            sample: SampleCharacteristics::Valid {
                bitdepth_minus1: 7,
                sampling: SamplingStructure::Yuv444,
            },
            timecode: TimeCode {
                hours: 1,
                minutes: 2,
                seconds: 3,
                frames: 4,
            },
        };
        let bytes = info.to_bytes().unwrap();
        assert_eq!(bytes.len(), 14);
        assert_eq!(VideoInformation::parse(&bytes).unwrap(), info);
    }

    #[test]
    fn jpvi_frat_subfield_layout() {
        // Interlace=2 (bits 31-30), denom=1 (bits 29-24), num=30 (low 16).
        let info = VideoInformation {
            max_bit_rate: 1,
            frame_rate: FrameRate::Known {
                interlace: InterlaceMode::InterlacedSecondField,
                denominator: FrameRateDenominator::Integer,
                numerator: 30,
            },
            sample: SampleCharacteristics::Invalid,
            timecode: TimeCode {
                hours: 0,
                minutes: 0,
                seconds: 0,
                frames: 1,
            },
        };
        let bytes = info.to_bytes().unwrap();
        let frat = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(frat >> 30, 2);
        assert_eq!((frat >> 24) & 0x3f, 1);
        assert_eq!(frat & 0xffff, 30);
        // Framerate_Reserved (bits 23-16) must be 0.
        assert_eq!((frat >> 16) & 0xff, 0);
    }

    #[test]
    fn jpvi_unknown_frame_rate_and_invalid_sample_are_zero() {
        let info = VideoInformation::unknown();
        let bytes = info.to_bytes().unwrap();
        // frat == 0 (unknown), schar == 0 (invalid).
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0]);
        assert_eq!(&bytes[8..10], &[0, 0]);
        assert_eq!(VideoInformation::parse(&bytes).unwrap(), info);
    }

    #[test]
    fn jpvi_rejects_reserved_and_out_of_range() {
        let base = VideoInformation::unknown().to_bytes().unwrap();
        // brat == 0 is illegal.
        let mut b = base;
        b[0..4].copy_from_slice(&0u32.to_be_bytes());
        assert!(VideoInformation::parse(&b).is_err());
        // Framerate_Denominator == 0 reserved (with non-zero frat).
        let mut b = base;
        b[4..8].copy_from_slice(&0x0000_0018u32.to_be_bytes()); // denom=0, num=24
        assert!(VideoInformation::parse(&b).is_err());
        // schar Valid_Flag=0 but sub-fields non-zero.
        let mut b = base;
        b[8..10].copy_from_slice(&0x0070u16.to_be_bytes());
        assert!(VideoInformation::parse(&b).is_err());
        // tcod FF == 0 with a non-zero HH is out of range (only the
        // all-zero word is the "unused" time code).
        let mut b = base;
        b[10] = 1;
        b[13] = 0;
        assert!(VideoInformation::parse(&b).is_err());
        // tcod FF == 61 out of range.
        let mut b = base;
        b[13] = 61;
        assert!(VideoInformation::parse(&b).is_err());
        // tcod HH == 24 out of range.
        let mut b = base;
        b[10] = 24;
        assert!(VideoInformation::parse(&b).is_err());
        // Wrong length.
        assert!(VideoInformation::parse(&base[..13]).is_err());
        // The writer applies the same tcod rule.
        let mut info = VideoInformation::unknown();
        info.timecode = TimeCode::new(0, 0, 1, 0);
        assert!(info.to_bytes().is_err());
        info.timecode = TimeCode::new(0, 0, 0, 61);
        assert!(info.to_bytes().is_err());
    }

    #[test]
    fn jpvi_all_zero_tcod_is_time_code_unused() {
        // Issue #10 (2): a sender that does not carry a time code writes
        // tcod = 0x0000_0000. That word is outside the Table A.5 ranges
        // (FF is 1–60) but collides with no legal time code, so it parses
        // as `TimeCode::UNUSED` rather than failing the box.
        let mut b = VideoInformation::unknown().to_bytes().unwrap();
        b[10..14].copy_from_slice(&[0, 0, 0, 0]);
        let info = VideoInformation::parse(&b).unwrap();
        assert!(info.timecode.is_unused());
        assert_eq!(info.timecode, TimeCode::UNUSED);
        assert_eq!(info.to_bytes().unwrap(), b);
        // And the writer emits it on request.
        let mut written = VideoInformation::unknown();
        written.timecode = TimeCode::UNUSED;
        assert_eq!(&written.to_bytes().unwrap()[10..14], &[0, 0, 0, 0]);
        // A real time code is not "unused".
        assert!(!TimeCode::new(0, 0, 0, 1).is_unused());
        assert!(TimeCode::new(23, 59, 59, 60).is_valid());
        assert!(!TimeCode::new(0, 60, 0, 1).is_valid());
    }

    #[test]
    fn jpvi_reserved_codes_round_trip_verbatim() {
        // Issue #10 (3): Table A.10 (2019) reserves Sampling_Structure 3
        // and 7–15, and Table A.7 reserves Interlace_Mode 3. A reader must
        // not lose the box over a code a later table may assign; the code
        // is surfaced as `Reserved` and written back bit-exactly.
        let base = VideoInformation::unknown().to_bytes().unwrap();
        for code in [3u8, 7, 15] {
            let mut b = base;
            b[8..10].copy_from_slice(&(0x8000u16 | 0x70 | u16::from(code)).to_be_bytes());
            let info = VideoInformation::parse(&b).unwrap();
            assert_eq!(
                info.sample,
                SampleCharacteristics::Valid {
                    bitdepth_minus1: 7,
                    sampling: SamplingStructure::Reserved(code),
                }
            );
            assert_eq!(info.to_bytes().unwrap(), b);
        }
        let mut b = base;
        b[4..8].copy_from_slice(&0xC100_0018u32.to_be_bytes()); // interlace=3, denom=1, num=24
        let info = VideoInformation::parse(&b).unwrap();
        assert_eq!(
            info.frame_rate,
            FrameRate::Known {
                interlace: InterlaceMode::Reserved,
                denominator: FrameRateDenominator::Integer,
                numerator: 24,
            }
        );
        assert_eq!(info.to_bytes().unwrap(), b);
        // Assigned codes still decode to their named variants, and the
        // code accessors invert each other over the whole nibble.
        for code in 0..16u8 {
            assert_eq!(SamplingStructure::from_code(code).code(), code);
        }
        for code in 0..4u8 {
            assert_eq!(InterlaceMode::from_code(code).code(), code);
        }
        assert_eq!(
            SamplingStructure::from_code(3),
            SamplingStructure::Reserved(3)
        );
        assert_eq!(
            SamplingStructure::from_code(4),
            SamplingStructure::YuvAux422
        );
        assert_eq!(SamplingStructure::Yuv422.component_count(), Some(3));
        assert_eq!(SamplingStructure::RgbAux444.component_count(), Some(4));
        assert_eq!(SamplingStructure::Reserved(3).component_count(), None);
    }

    #[test]
    fn jpvi_interlace_mode_is_a_frame_property_shared_by_both_fields() {
        // Issue #10 (1): Interlace_Mode gives the frame's field order and
        // is identical in both pictures of an interlaced frame. Wrapping
        // the two field pictures with the same VideoInformation yields
        // byte-identical Video Support superboxes.
        assert!(!InterlaceMode::Progressive.is_interlaced());
        assert!(InterlaceMode::InterlacedFirstField.is_interlaced());
        assert!(InterlaceMode::InterlacedSecondField.is_interlaced());
        assert!(!InterlaceMode::Reserved.is_interlaced());

        let frame_info = VideoInformation {
            max_bit_rate: 400,
            frame_rate: FrameRate::Known {
                interlace: InterlaceMode::InterlacedFirstField,
                denominator: FrameRateDenominator::Ntsc,
                numerator: 30,
            },
            sample: SampleCharacteristics::Valid {
                bitdepth_minus1: 9,
                sampling: SamplingStructure::Yuv422,
            },
            timecode: TimeCode::UNUSED,
        };
        let cs = luma_cs();
        let jpvs_bytes = |file: &[u8]| -> Vec<u8> {
            let span = parse_boxes(file, 0, file.len())
                .unwrap()
                .into_iter()
                .find(|s| s.tbox == TBOX_VIDEO_SUPPORT)
                .unwrap();
            file[span.payload_start..span.payload_start + span.payload_len].to_vec()
        };
        let first_field = JxsFileBuilder::new(srgb())
            .video_information(frame_info)
            .build(&cs)
            .unwrap();
        let second_field = JxsFileBuilder::new(srgb())
            .video_information(frame_info)
            .build(&cs)
            .unwrap();
        assert_eq!(jpvs_bytes(&first_field), jpvs_bytes(&second_field));
        for file in [&first_field, &second_field] {
            let parsed = parse_jxs_file(file).unwrap().video_info.unwrap();
            assert_eq!(parsed, frame_info);
            match parsed.frame_rate {
                FrameRate::Known { interlace, .. } => {
                    assert_eq!(interlace, InterlaceMode::InterlacedFirstField);
                }
                FrameRate::Unknown => panic!("frame rate lost"),
            }
        }
    }

    #[test]
    fn builder_emits_conforming_jpvs_with_both_boxes() {
        // The jxpl names Main 444.12 (a known profile), so the wrapped
        // stream must satisfy it — profile_shaped_luma_cs does.
        let cs = profile_shaped_luma_cs();
        let info = VideoInformation {
            max_bit_rate: 250,
            frame_rate: FrameRate::Known {
                interlace: InterlaceMode::Progressive,
                denominator: FrameRateDenominator::Integer,
                numerator: 60,
            },
            sample: SampleCharacteristics::Valid {
                bitdepth_minus1: 9,
                sampling: SamplingStructure::Rgb444,
            },
            timecode: TimeCode {
                hours: 10,
                minutes: 20,
                seconds: 30,
                frames: 40,
            },
        };
        let file = JxsFileBuilder::new(srgb())
            .video_information(info)
            .profile_level(0x3A40, 0x2400)
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.video_info.unwrap(), info);
        let pl = parsed.profile_level.unwrap();
        assert_eq!(pl.ppih, 0x3A40);
        assert_eq!(pl.plev, 0x2400);
        // A.5.3: jpvi shall be the first box in jpvs, jxpl the second.
        let jpvs = parse_boxes(&file, 0, file.len())
            .unwrap()
            .into_iter()
            .find(|s| s.tbox == TBOX_VIDEO_SUPPORT)
            .unwrap();
        let inner = parse_boxes(
            &file,
            jpvs.payload_start,
            jpvs.payload_start + jpvs.payload_len,
        )
        .unwrap();
        assert_eq!(inner[0].tbox, TBOX_VIDEO_INFO);
        assert_eq!(inner[1].tbox, TBOX_PROFILE_LEVEL);
    }

    #[test]
    fn builder_synthesises_mandatory_jpvi_when_only_profile_level_set() {
        // Setting only profile_level must still yield a conforming jpvs:
        // a default jpvi (VideoInformation::unknown) is synthesised.
        let cs = luma_cs();
        let file = JxsFileBuilder::new(srgb())
            .profile_level(0x1234, 0x5678)
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.video_info.unwrap(), VideoInformation::unknown());
        assert_eq!(parsed.profile_level.unwrap().ppih, 0x1234);
        // Round-trips through decode.
        assert!(crate::decode_components(&file).is_ok());
    }

    #[test]
    fn builder_synthesises_default_profile_level_when_only_video_info_set() {
        let cs = luma_cs();
        let file = JxsFileBuilder::new(srgb())
            .video_information(VideoInformation::unknown())
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        // Default jxpl is Unrestricted (0x0000 / 0x0000).
        let pl = parsed.profile_level.unwrap();
        assert_eq!(pl.ppih, 0);
        assert_eq!(pl.plev, 0);
    }

    // ---- Mastering Display Metadata box (dmon, A.5.3.5) ----

    fn bt709_dmon() -> MasteringDisplayMetadata {
        // Table A.13 default values (Rec. ITU-R BT.709-5).
        MasteringDisplayMetadata {
            primary_green: (0x3A98, 0x7530),
            primary_blue: (0x1D4C, 0x0BB8),
            primary_red: (0x7D00, 0x4074),
            white_point: (0x3D13, 0x4042),
            min_luminance: 0x0000_0064,
            max_luminance: 0x000F_4240,
            max_content_light_level: 0,
            max_frame_avg_light_level: 0,
        }
    }

    #[test]
    fn dmon_round_trips_bt709_defaults() {
        let meta = bt709_dmon();
        let bytes = meta.to_bytes().unwrap();
        assert_eq!(bytes.len(), 28);
        assert_eq!(MasteringDisplayMetadata::parse(&bytes).unwrap(), meta);
    }

    #[test]
    fn dmon_rejects_out_of_range_and_bad_luminance() {
        let base = bt709_dmon().to_bytes().unwrap();
        // Chromaticity coordinate > 50000 (Xc0 at bytes 0..2).
        let mut b = base;
        b[0..2].copy_from_slice(&50_001u16.to_be_bytes());
        assert!(MasteringDisplayMetadata::parse(&b).is_err());
        // Lmin >= Lmax.
        let mut b = base;
        b[16..20].copy_from_slice(&0x00FF_FFFFu32.to_be_bytes());
        assert!(MasteringDisplayMetadata::parse(&b).is_err());
        // Wrong length.
        assert!(MasteringDisplayMetadata::parse(&base[..27]).is_err());
        // Constructor-side validation.
        let mut bad = bt709_dmon();
        bad.white_point = (60_000, 0);
        assert!(bad.to_bytes().is_err());
    }

    // ---- Video Transport Parameter box (jptp, A.5.3.6) ----

    #[test]
    fn jptp_round_trips() {
        let params = VideoTransportParameters {
            slice_group_size: 4,
            parallel_units: 8,
        };
        let bytes = params.to_bytes();
        assert_eq!(bytes.len(), 6);
        // Tseq and MTU are 0 by construction.
        assert_eq!(&bytes[3..6], &[0, 0, 0]);
        assert_eq!(VideoTransportParameters::parse(&bytes).unwrap(), params);
    }

    #[test]
    fn jptp_rejects_reserved_tseq_and_mtu() {
        let base = VideoTransportParameters {
            slice_group_size: 1,
            parallel_units: 1,
        }
        .to_bytes();
        // Tseq (byte 3) non-zero is reserved.
        let mut b = base;
        b[3] = 1;
        assert!(VideoTransportParameters::parse(&b).is_err());
        // MTU (bytes 4..6) non-zero is reserved.
        let mut b = base;
        b[4..6].copy_from_slice(&1500u16.to_be_bytes());
        assert!(VideoTransportParameters::parse(&b).is_err());
        // Wrong length.
        assert!(VideoTransportParameters::parse(&base[..5]).is_err());
    }

    // ---- Buffer Model Description box (bmdm, A.5.3.4) ----

    #[test]
    fn bmdm_round_trips() {
        let desc = BufferModelDescription {
            model_type: 2,
            ncg_horizontal: 12345,
            ncg_vertical: 678,
        };
        let bytes = desc.to_bytes().unwrap();
        assert_eq!(bytes.len(), 6);
        // Rd (byte 1) is 0 by construction.
        assert_eq!(bytes[1], 0);
        assert_eq!(BufferModelDescription::parse(&bytes).unwrap(), desc);
    }

    #[test]
    fn bmdm_rejects_reserved_fields() {
        let base = BufferModelDescription {
            model_type: 0,
            ncg_horizontal: 1,
            ncg_vertical: 1,
        }
        .to_bytes()
        .unwrap();
        // Tbmd == 3 reserved.
        let mut b = base;
        b[0] = 3;
        assert!(BufferModelDescription::parse(&b).is_err());
        // Rd != 0 reserved.
        let mut b = base;
        b[1] = 1;
        assert!(BufferModelDescription::parse(&b).is_err());
        // Wrong length.
        assert!(BufferModelDescription::parse(&base[..5]).is_err());
        // Constructor-side Tbmd validation.
        assert!(
            BufferModelDescription {
                model_type: 4,
                ncg_horizontal: 0,
                ncg_vertical: 0,
            }
            .to_bytes()
            .is_err()
        );
    }

    #[test]
    fn builder_emits_all_jpvs_boxes_in_order() {
        // The jxpl names Main 422.10 (a known profile), so the wrapped
        // stream must satisfy it — profile_shaped_luma_cs does.
        let cs = profile_shaped_luma_cs();
        let meta = bt709_dmon();
        let bmdm = BufferModelDescription {
            model_type: 1,
            ncg_horizontal: 64,
            ncg_vertical: 8,
        };
        let params = VideoTransportParameters {
            slice_group_size: 2,
            parallel_units: 4,
        };
        let file = JxsFileBuilder::new(srgb())
            .video_information(VideoInformation::unknown())
            .profile_level(0x3540, 0x1000)
            .buffer_model(bmdm)
            .mastering_display(meta)
            .transport_parameters(params)
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.buffer_model.unwrap(), bmdm);
        assert_eq!(parsed.mastering_display.unwrap(), meta);
        assert_eq!(parsed.transport_params.unwrap(), params);
        // Figure A.7 order inside jpvs: jpvi, jxpl, bmdm, dmon, jptp.
        let jpvs = parse_boxes(&file, 0, file.len())
            .unwrap()
            .into_iter()
            .find(|s| s.tbox == TBOX_VIDEO_SUPPORT)
            .unwrap();
        let inner = parse_boxes(
            &file,
            jpvs.payload_start,
            jpvs.payload_start + jpvs.payload_len,
        )
        .unwrap();
        let types: Vec<u32> = inner.iter().map(|s| s.tbox).collect();
        assert_eq!(
            types,
            vec![
                TBOX_VIDEO_INFO,
                TBOX_PROFILE_LEVEL,
                TBOX_BUFFER_MODEL,
                TBOX_MASTERING_DISPLAY,
                TBOX_TRANSPORT_PARAMS
            ]
        );
    }

    // ---- Exif box (A.5.4 Table A.2) ----

    #[test]
    fn builder_round_trips_exif_box() {
        let cs = luma_cs();
        let exif = b"Exif\x00\x00II*\x00\x08\x00\x00\x00".to_vec();
        let file = JxsFileBuilder::new(srgb())
            .exif(exif.clone())
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert_eq!(parsed.header.exif.as_deref(), Some(exif.as_slice()));
        // The Exif box lives inside jp2h and does not disturb decode.
        assert!(crate::decode_components(&file).is_ok());
    }

    #[test]
    fn parse_without_exif_yields_none() {
        let cs = luma_cs();
        let file = write_jxs_file(&cs).unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert!(parsed.header.exif.is_none());
    }

    #[test]
    fn media_type_classifies_the_registered_magics() {
        // §D.2.2 — a bare codestream starts SOC + CAP.
        let cs = luma_cs();
        assert_eq!(&cs[..4], &CODESTREAM_MAGIC);
        assert_eq!(media_type(&cs), Some(MEDIA_TYPE_CODESTREAM));
        // §A.7.2 — the box file starts with the Signature box.
        let file = write_jxs_file(&cs).unwrap();
        assert_eq!(media_type(&file), Some(MEDIA_TYPE_JXS));
        // No registered magic: truncated prefixes, plain garbage, and a
        // SOC not followed by CAP (not the registered §D.2.2 magic).
        assert_eq!(media_type(&cs[..2]), None);
        assert_eq!(media_type(&file[..8]), None);
        assert_eq!(media_type(b"not a jxs"), None);
        assert_eq!(media_type(&[0xff, 0x10, 0xff, 0x12]), None);
        // The registration strings themselves.
        assert_eq!(MEDIA_TYPE_JXS, "image/jxs");
        assert_eq!(MEDIA_TYPE_CODESTREAM, "image/jxsc");
        assert_eq!(MEDIA_TYPE_HEIF_IMAGE, "image/jxsi");
        assert_eq!(MEDIA_TYPE_HEIF_SEQUENCE, "image/jxss");
    }

    #[test]
    fn builder_dmon_alone_synthesises_mandatory_boxes() {
        // A dmon box alone still yields a conforming jpvs (jpvi + jxpl
        // synthesised).
        let cs = luma_cs();
        let file = JxsFileBuilder::new(srgb())
            .mastering_display(bt709_dmon())
            .build(&cs)
            .unwrap();
        let parsed = parse_jxs_file(&file).unwrap();
        assert!(parsed.video_info.is_some());
        assert!(parsed.profile_level.is_some());
        assert_eq!(parsed.mastering_display.unwrap(), bt709_dmon());
        assert!(crate::decode_components(&file).is_ok());
    }
}
