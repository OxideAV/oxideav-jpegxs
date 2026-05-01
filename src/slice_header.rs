//! Slice header (SLH) — ISO/IEC 21122-1:2022 §A.4.12, Table A.25.
//!
//! Body layout (after marker + `Lslh = 4`):
//!
//! ```text
//! Yslh   u16   slice index (top-down, 0..=65535)
//! ```
//!
//! `Lslh` is fixed at 4 (the two `Lslh` bytes plus the two `Yslh`
//! bytes), so the body parser sees exactly 2 bytes.

use oxideav_core::{Error, Result};

/// Decoded slice header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceHeader {
    /// Slice index counted from the top of the image (`Yslh`).
    pub yslh: u16,
}

/// Fixed `Lslh` value per Table A.25.
pub const SLH_LSLH: u16 = 4;

/// Parse a SLH segment body. `body` must be exactly 2 bytes.
pub fn parse(body: &[u8]) -> Result<SliceHeader> {
    if body.len() != (SLH_LSLH as usize) - 2 {
        return Err(Error::invalid(format!(
            "jpegxs: SLH body must be {} bytes, got {}",
            SLH_LSLH as usize - 2,
            body.len()
        )));
    }
    let yslh = u16::from_be_bytes([body[0], body[1]]);
    Ok(SliceHeader { yslh })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_yslh() {
        let h = parse(&[0x12, 0x34]).expect("ok");
        assert_eq!(h.yslh, 0x1234);
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(parse(&[]).is_err());
        assert!(parse(&[0x00]).is_err());
        assert!(parse(&[0x00, 0x00, 0x00]).is_err());
    }
}
