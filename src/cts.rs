//! Colour Transformation Specification marker (CTS) — ISO/IEC 21122-1:
//! 2022 §A.4.8, Tables A.19 / A.20.
//!
//! Mandatory iff `Cpih == 3` (Star-Tetrix). Body is a fixed 4-byte
//! payload after the `Lcts` length field:
//!
//! ```text
//! Reserved   u4   shall be 0
//! Cf         u4   transform extent — 0 = full, 3 = restricted in-line
//! e1         u4   exponent of first chroma component, 0..=3
//! e2         u4   exponent of second chroma component, 0..=3
//! ```
//!
//! `Cf` selects whether the inverse Star-Tetrix `access()` reflection
//! (Table F.12) is allowed to look at neighbouring lines (`Cf == 0`)
//! or must reflect any out-of-line access back into the current line
//! (`Cf == 3`). Other `Cf` values are reserved for ISO/IEC and rejected
//! by [`parse_cts`].

use oxideav_core::{Error, Result};

/// CTS marker `Cf` field values per Table A.20.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtsExtent {
    /// `Cf == 0`. Full transformation; access to the line above and
    /// below is permitted by the `access()` reflection rules.
    Full,
    /// `Cf == 3`. Restricted in-line transformation; the `access()`
    /// reflection rule reflects any vertical neighbour back into the
    /// current line.
    InLine,
}

impl CtsExtent {
    /// Numeric `Cf` value as it appears on the wire.
    pub fn cf(self) -> u8 {
        match self {
            CtsExtent::Full => 0,
            CtsExtent::InLine => 3,
        }
    }
}

/// Decoded CTS marker body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtsMarker {
    /// `Cf` — transform extent.
    pub cf: CtsExtent,
    /// `e1` — exponent of the first chroma component (0..=3).
    pub e1: u8,
    /// `e2` — exponent of the second chroma component (0..=3).
    pub e2: u8,
}

/// `Lcts` body length in bytes (the body is exactly 4 bytes after the
/// 2-byte length field).
pub const CTS_BODY_LEN: usize = 2;

/// Parse a CTS body (the bytes after `Lcts`). Per Table A.19 the body
/// is exactly 2 bytes: `Reserved|Cf` then `e1|e2`.
pub fn parse_cts(body: &[u8]) -> Result<CtsMarker> {
    if body.len() != CTS_BODY_LEN {
        return Err(Error::invalid(format!(
            "jpegxs CTS: body must be {CTS_BODY_LEN} bytes, got {}",
            body.len()
        )));
    }
    let reserved = (body[0] >> 4) & 0x0f;
    let cf_raw = body[0] & 0x0f;
    let e1 = (body[1] >> 4) & 0x0f;
    let e2 = body[1] & 0x0f;

    if reserved != 0 {
        return Err(Error::invalid(format!(
            "jpegxs CTS: Reserved nibble must be 0 per A.4.8, got {reserved}"
        )));
    }
    let cf = match cf_raw {
        0 => CtsExtent::Full,
        3 => CtsExtent::InLine,
        other => {
            return Err(Error::invalid(format!(
                "jpegxs CTS: Cf={other} reserved for ISO/IEC use (Table A.20 allows 0 or 3)"
            )));
        }
    };
    if e1 > 3 {
        return Err(Error::invalid(format!(
            "jpegxs CTS: e1={e1} exceeds Table A.19 cap of 3"
        )));
    }
    if e2 > 3 {
        return Err(Error::invalid(format!(
            "jpegxs CTS: e2={e2} exceeds Table A.19 cap of 3"
        )));
    }
    Ok(CtsMarker { cf, e1, e2 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_transform() {
        let body = [0x00, 0x12]; // Cf=0, e1=1, e2=2
        let cts = parse_cts(&body).unwrap();
        assert_eq!(cts.cf, CtsExtent::Full);
        assert_eq!(cts.cf.cf(), 0);
        assert_eq!(cts.e1, 1);
        assert_eq!(cts.e2, 2);
    }

    #[test]
    fn parses_in_line_transform() {
        let body = [0x03, 0x33]; // Cf=3, e1=3, e2=3
        let cts = parse_cts(&body).unwrap();
        assert_eq!(cts.cf, CtsExtent::InLine);
        assert_eq!(cts.cf.cf(), 3);
        assert_eq!(cts.e1, 3);
        assert_eq!(cts.e2, 3);
    }

    #[test]
    fn rejects_reserved_nibble_nonzero() {
        let body = [0x10, 0x00];
        let err = parse_cts(&body).unwrap_err();
        assert!(format!("{err}").contains("Reserved"));
    }

    #[test]
    fn rejects_reserved_cf_value() {
        for cf in [1u8, 2, 4, 5, 15] {
            let body = [cf, 0x00];
            let err = parse_cts(&body).unwrap_err();
            assert!(format!("{err}").contains("Cf="), "Cf={cf} should err");
        }
    }

    #[test]
    fn rejects_oversized_e1_or_e2() {
        // e1 > 3.
        let err = parse_cts(&[0x00, 0x40]).unwrap_err();
        assert!(format!("{err}").contains("e1"));
        // e2 > 3.
        let err = parse_cts(&[0x00, 0x04]).unwrap_err();
        assert!(format!("{err}").contains("e2"));
    }

    #[test]
    fn rejects_wrong_body_length() {
        assert!(parse_cts(&[]).is_err());
        assert!(parse_cts(&[0x00]).is_err());
        assert!(parse_cts(&[0x00, 0x00, 0x00]).is_err());
    }
}
