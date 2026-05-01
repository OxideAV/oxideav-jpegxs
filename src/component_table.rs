//! Component table (CDT) — ISO/IEC 21122-1:2022 §A.4.5, Table A.15.
//!
//! Body layout (after marker + `Lcdt`): for each component `c` in
//! `0..Nc`, two bytes:
//!
//! ```text
//! B[c]   u8   bit precision, 8..=16
//! sx:sy  u4|u4  horizontal sampling factor | vertical sampling factor
//! ```
//!
//! `Lcdt = 2 * Nc + 2`. `Nc` comes from the PIH (which must be parsed
//! first).

use oxideav_core::{Error, Result};

/// Per-component description from the CDT segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    /// Bit precision (`B[c]`), 8..=16.
    pub bit_depth: u8,
    /// Horizontal sampling factor (`sx[c]`).
    pub sx: u8,
    /// Vertical sampling factor (`sy[c]`).
    pub sy: u8,
}

/// Parsed component table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentTable {
    pub components: Vec<Component>,
}

/// Parse a CDT segment body (the bytes after `CDT` and `Lcdt`).
/// `nc` is the component count from the PIH and must already be in
/// `1..=8`. The body must therefore be exactly `2 * nc` bytes.
pub fn parse(body: &[u8], nc: u8) -> Result<ComponentTable> {
    let expected = (nc as usize) * 2;
    if body.len() != expected {
        return Err(Error::invalid(format!(
            "jpegxs: CDT body must be 2*Nc = {expected} bytes, got {}",
            body.len()
        )));
    }
    let mut components = Vec::with_capacity(nc as usize);
    for i in 0..nc as usize {
        let bd = body[i * 2];
        let pack = body[i * 2 + 1];
        let sx = pack >> 4;
        let sy = pack & 0x0f;
        if !(8..=16).contains(&bd) {
            return Err(Error::invalid(format!(
                "jpegxs: CDT component {i} bit depth {bd} out of range 8..=16"
            )));
        }
        if sx == 0 || sy == 0 {
            return Err(Error::invalid(format!(
                "jpegxs: CDT component {i} sampling factors {sx}/{sy} must be >= 1"
            )));
        }
        components.push(Component {
            bit_depth: bd,
            sx,
            sy,
        });
    }
    Ok(ComponentTable { components })
}

impl ComponentTable {
    /// Maximum bit depth across components (the value typically reported
    /// as the picture's bit depth at probe time).
    pub fn max_bit_depth(&self) -> u8 {
        self.components
            .iter()
            .map(|c| c.bit_depth)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_three_components() {
        // Y'CbCr 4:2:2 — Y=8 bit, Cb/Cr=8 bit, sx=1/2/2 sy=1.
        let body = [8, 0x11, 8, 0x21, 8, 0x21];
        let cdt = parse(&body, 3).expect("ok");
        assert_eq!(cdt.components.len(), 3);
        assert_eq!(cdt.components[0].bit_depth, 8);
        assert_eq!(cdt.components[0].sx, 1);
        assert_eq!(cdt.components[0].sy, 1);
        assert_eq!(cdt.components[1].sx, 2);
        assert_eq!(cdt.components[1].sy, 1);
        assert_eq!(cdt.max_bit_depth(), 8);
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(parse(&[8, 0x11], 2).is_err());
        assert!(parse(&[8, 0x11, 8, 0x11, 8, 0x11], 2).is_err());
    }

    #[test]
    fn rejects_out_of_range_bit_depth() {
        assert!(parse(&[7, 0x11], 1).is_err());
        assert!(parse(&[17, 0x11], 1).is_err());
    }

    #[test]
    fn rejects_zero_sampling_factor() {
        assert!(parse(&[8, 0x01], 1).is_err());
        assert!(parse(&[8, 0x10], 1).is_err());
    }
}
