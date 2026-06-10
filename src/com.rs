//! Extension marker (COM) — ISO/IEC 21122-1:2022 §A.4.10, Table A.22.
//!
//! Optional marker; zero or more extension marker segments may appear
//! in a codestream, and any present extension segment shall precede the
//! first slice header (§A.4.10 Usage). The COM body (the bytes after
//! the `Lcom` length field) is:
//!
//! * `Tcom` — `u(16)` type of the extension (Table A.23), big-endian on
//!   the wire like every other JPEG XS length / field word.
//! * `Dcom` — variable user-defined data (`Tcom`-dependent
//!   interpretation; e.g. a zero-terminated ISO/IEC 10646 string for the
//!   encoder-vendor (`0x0000`) and copyright (`0x0001`) types).
//!
//! `Tcom` encoding per Table A.23:
//!
//! | `Tcom`            | Meaning                                              |
//! | ----------------- | ---------------------------------------------------- |
//! | `0x0000`          | Encoder vendor (`Dcom` = zero-terminated 10646 string) |
//! | `0x0001`          | Copyright statement (`Dcom` = zero-terminated 10646 string) |
//! | `0x8000`–`0xffff` | Vendor-specific information                          |
//! | all other values  | Reserved for ISO/IEC use                             |
//!
//! This module decodes the body into a strongly-typed [`ComMarker`]
//! view, mirroring the round-251 / round-254 / round-266 typed-accessor
//! pattern (`cts` / `crg` / `nlt` / `cwd` / `wgt`). The body-level
//! parser only enforces what is observable from the COM body itself:
//! the body must carry the two-byte `Tcom` field. The "`Tcom` reserved
//! for ISO/IEC use" rows are *not* rejected — a decoder must tolerate
//! an unknown extension type and skip its `Dcom` (the marker is purely
//! advisory metadata), so the reserved range is surfaced verbatim
//! rather than as an error.

use crate::error::{JpegXsError as Error, Result};

/// `Tcom` value identifying the encoder-vendor extension — `Dcom` is a
/// zero-terminated ISO/IEC 10646 string identifying the encoder vendor
/// (Table A.23).
pub const TCOM_ENCODER_VENDOR: u16 = 0x0000;

/// `Tcom` value identifying a copyright statement — `Dcom` is a
/// zero-terminated ISO/IEC 10646 string carrying the statement
/// (Table A.23).
pub const TCOM_COPYRIGHT: u16 = 0x0001;

/// First `Tcom` value of the vendor-specific range (`0x8000`–`0xffff`,
/// Table A.23).
pub const TCOM_VENDOR_SPECIFIC_MIN: u16 = 0x8000;

/// Minimum COM body length in bytes: the `Tcom` field is two bytes and
/// `Dcom` may be empty.
pub const COM_MIN_BODY_LEN: usize = 2;

/// Decoded COM marker body per Table A.22.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComMarker {
    /// `Tcom` — type of the extension (Table A.23).
    pub tcom: u16,
    /// `Dcom` — user-defined data, exactly as on the wire. May be empty.
    pub dcom: Vec<u8>,
}

impl ComMarker {
    /// `true` when `Tcom` is the encoder-vendor type (`0x0000`).
    pub fn is_encoder_vendor(&self) -> bool {
        self.tcom == TCOM_ENCODER_VENDOR
    }

    /// `true` when `Tcom` is the copyright-statement type (`0x0001`).
    pub fn is_copyright(&self) -> bool {
        self.tcom == TCOM_COPYRIGHT
    }

    /// `true` when `Tcom` falls in the vendor-specific range
    /// (`0x8000`–`0xffff`).
    pub fn is_vendor_specific(&self) -> bool {
        self.tcom >= TCOM_VENDOR_SPECIFIC_MIN
    }
}

/// Parse a COM body (the bytes after `Lcom`). Per Table A.22 the body is
/// a big-endian `Tcom` `u(16)` followed by the variable `Dcom` payload.
/// Body-level errors surface here:
///
/// * `body.len() < 2` — the body is too short to carry the mandatory
///   two-byte `Tcom` field (Table A.22 / `Lcom >= 4`, of which two
///   bytes are the length field itself, leaving at least the two `Tcom`
///   bytes in the body).
///
/// Reserved `Tcom` values (the "all other values" row of Table A.23) are
/// returned verbatim rather than rejected: an extension marker is
/// advisory metadata and a conforming decoder skips unknown types, so a
/// reserved-range `Tcom` is a tolerate-and-pass-through case, not a
/// malformed body.
pub fn parse_com(body: &[u8]) -> Result<ComMarker> {
    if body.len() < COM_MIN_BODY_LEN {
        return Err(Error::invalid(format!(
            "jpegxs COM: body must be at least {COM_MIN_BODY_LEN} bytes \
             (Tcom u16, Annex A.4.10 Table A.22), got {}",
            body.len()
        )));
    }
    let tcom = u16::from_be_bytes([body[0], body[1]]);
    let dcom = body[2..].to_vec();
    Ok(ComMarker { tcom, dcom })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_encoder_vendor() {
        // Tcom = 0x0000, Dcom = zero-terminated "ox\0".
        let body = [0x00, 0x00, b'o', b'x', 0x00];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, TCOM_ENCODER_VENDOR);
        assert_eq!(com.dcom, b"ox\0");
        assert!(com.is_encoder_vendor());
        assert!(!com.is_copyright());
        assert!(!com.is_vendor_specific());
    }

    #[test]
    fn parses_copyright() {
        let body = [0x00, 0x01, b'c', 0x00];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, TCOM_COPYRIGHT);
        assert_eq!(com.dcom, b"c\0");
        assert!(com.is_copyright());
        assert!(!com.is_encoder_vendor());
        assert!(!com.is_vendor_specific());
    }

    #[test]
    fn parses_vendor_specific_min() {
        let body = [0x80, 0x00, 0xde, 0xad];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, TCOM_VENDOR_SPECIFIC_MIN);
        assert_eq!(com.dcom, vec![0xde, 0xad]);
        assert!(com.is_vendor_specific());
        assert!(!com.is_encoder_vendor());
        assert!(!com.is_copyright());
    }

    #[test]
    fn parses_vendor_specific_max() {
        let body = [0xff, 0xff];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, 0xffff);
        assert!(com.dcom.is_empty());
        assert!(com.is_vendor_specific());
    }

    #[test]
    fn tolerates_reserved_tcom() {
        // 0x0002 is in the "all other values — reserved for ISO/IEC use"
        // row of Table A.23. A decoder must skip unknown extensions, so
        // the body-level parser surfaces it verbatim instead of erroring.
        let body = [0x00, 0x02, 0x10, 0x20, 0x30];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, 0x0002);
        assert_eq!(com.dcom, vec![0x10, 0x20, 0x30]);
        assert!(!com.is_encoder_vendor());
        assert!(!com.is_copyright());
        assert!(!com.is_vendor_specific());
    }

    #[test]
    fn empty_dcom_is_valid() {
        // Tcom present, Dcom empty: the body is exactly the two Tcom
        // bytes.
        let body = [0x00, 0x01];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, TCOM_COPYRIGHT);
        assert!(com.dcom.is_empty());
    }

    #[test]
    fn rejects_short_body() {
        // Body too short to hold the two-byte Tcom field.
        assert!(parse_com(&[]).is_err());
        assert!(parse_com(&[0x00]).is_err());
        let err = parse_com(&[0x42]).unwrap_err();
        assert!(
            format!("{err}").contains("Tcom"),
            "expected Tcom-length error, got {err}"
        );
    }

    #[test]
    fn tcom_is_big_endian() {
        // 0x1234 on the wire is [0x12, 0x34] big-endian.
        let body = [0x12, 0x34, 0xaa];
        let com = parse_com(&body).unwrap();
        assert_eq!(com.tcom, 0x1234);
        assert_eq!(com.dcom, vec![0xaa]);
    }
}
