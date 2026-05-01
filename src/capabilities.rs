//! Decoded view of the CAP marker's `cap[]` capability bit array
//! (ISO/IEC 21122-1:2022 Annex A.4.3, Tables A.5 / A.6).
//!
//! The wire format is a byte-aligned bit array. Bit `i` (counted MSB
//! first, big-endian within each byte) signals one optional decoder
//! capability per Table A.5:
//!
//! | bit | meaning                                                   |
//! |-----|-----------------------------------------------------------|
//! | 1   | Star-Tetrix transform + CTS marker support                |
//! | 2   | Quadratic non-linear transform support                    |
//! | 3   | Extended non-linear transform support                     |
//! | 4   | Vertical sub-sampling component(s) present (`sy[i] > 1`)  |
//! | 5   | Component-dependent wavelet decomposition (CWD) support   |
//! | 6   | Lossless decoding support                                 |
//! | 8   | Packet-based raw-mode switch support                      |
//!
//! Bit 0 is "intentionally unused" per Note 3 in §A.4.3; bit 7 and
//! bits >8 are reserved for ISO/IEC. The parser keeps the raw byte
//! array around for diagnostics and exposes the documented bits via
//! [`Capabilities`].
//!
//! The CAP segment also has the rule (§A.4.3) that for `Lcap > 2`, the
//! last cap byte must contain at least one set bit — i.e. the encoder
//! must shrink `Lcap` so trailing zero bytes are not transmitted. We
//! enforce that in [`parse_capabilities`].

use oxideav_core::{Error, Result};

/// Decoded CAP `cap[]` array.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// Bit 1 — Star-Tetrix transform + CTS marker support required.
    pub star_tetrix: bool,
    /// Bit 2 — Quadratic non-linear transform (NLT Tnlt=1) support.
    pub nlt_quadratic: bool,
    /// Bit 3 — Extended non-linear transform (NLT Tnlt=2) support.
    pub nlt_extended: bool,
    /// Bit 4 — At least one component with `sy[i] > 1` is present
    /// (vertical sub-sampling).
    pub vertical_subsampling: bool,
    /// Bit 5 — Component-dependent wavelet decomposition (CWD marker)
    /// support required.
    pub cwd: bool,
    /// Bit 6 — Lossless decoding support required (Fq=0 path).
    pub lossless: bool,
    /// Bit 8 — Packet-based raw-mode switch support required.
    pub raw_mode_switch: bool,
}

impl Capabilities {
    /// All-zero capability mask. Equivalent to `Default::default()`.
    pub const NONE: Capabilities = Capabilities {
        star_tetrix: false,
        nlt_quadratic: false,
        nlt_extended: false,
        vertical_subsampling: false,
        cwd: false,
        lossless: false,
        raw_mode_switch: false,
    };

    /// Whether any documented capability bit is set.
    pub fn any(&self) -> bool {
        self.star_tetrix
            || self.nlt_quadratic
            || self.nlt_extended
            || self.vertical_subsampling
            || self.cwd
            || self.lossless
            || self.raw_mode_switch
    }
}

/// Parse the `cap[]` byte array (the CAP marker body after `Lcap`)
/// into a [`Capabilities`] view.
///
/// Per §A.4.3:
///
/// * The empty array (zero-length body, i.e. `Lcap == 2`) is valid and
///   yields [`Capabilities::NONE`].
/// * For non-empty bodies, the spec requires the last byte to contain
///   at least one set bit (so trailing all-zero bytes are forbidden).
///   We enforce this with `Error::invalid` when violated.
///
/// Bits beyond bit 8 are accepted but ignored (they're reserved for
/// future extensions).
pub fn parse_capabilities(cap_body: &[u8]) -> Result<Capabilities> {
    if cap_body.is_empty() {
        return Ok(Capabilities::NONE);
    }
    // Spec rule: last byte cannot be all-zero for Lcap > 2.
    if *cap_body.last().expect("non-empty checked above") == 0 {
        return Err(Error::invalid(
            "jpegxs CAP: last byte of cap[] must be non-zero per A.4.3 (use a smaller Lcap)",
        ));
    }
    Ok(decode_known_bits(cap_body))
}

/// Lenient variant of [`parse_capabilities`] — accepts a possibly-
/// trailing-zero `cap[]` body (e.g. from a non-conformant encoder)
/// without erroring. Useful for probe-only paths that should not
/// reject a stream solely on a CAP-byte tidiness violation.
pub fn parse_capabilities_lossy(cap_body: &[u8]) -> Capabilities {
    if cap_body.is_empty() {
        return Capabilities::NONE;
    }
    decode_known_bits(cap_body)
}

fn decode_known_bits(cap_body: &[u8]) -> Capabilities {
    Capabilities {
        star_tetrix: bit(cap_body, 1),
        nlt_quadratic: bit(cap_body, 2),
        nlt_extended: bit(cap_body, 3),
        vertical_subsampling: bit(cap_body, 4),
        cwd: bit(cap_body, 5),
        lossless: bit(cap_body, 6),
        raw_mode_switch: bit(cap_body, 8),
    }
}

/// Read bit `i` from the cap[] array, MSB-first within each byte.
/// Returns `false` for any bit position past the array length.
fn bit(cap_body: &[u8], i: usize) -> bool {
    let byte_idx = i / 8;
    let in_byte = 7 - (i % 8);
    if byte_idx >= cap_body.len() {
        return false;
    }
    (cap_body[byte_idx] >> in_byte) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cap_decodes_to_none() {
        let caps = parse_capabilities(&[]).unwrap();
        assert_eq!(caps, Capabilities::NONE);
        assert!(!caps.any());
    }

    #[test]
    fn star_tetrix_bit_decodes() {
        // Bit 1 is the second-most-significant bit of byte 0 → 0x40.
        let caps = parse_capabilities(&[0x40]).unwrap();
        assert!(caps.star_tetrix);
        assert!(!caps.nlt_quadratic);
        assert!(!caps.nlt_extended);
        assert!(caps.any());
    }

    #[test]
    fn quadratic_nlt_bit_decodes() {
        // Bit 2 → 0x20.
        let caps = parse_capabilities(&[0x20]).unwrap();
        assert!(!caps.star_tetrix);
        assert!(caps.nlt_quadratic);
        assert!(!caps.nlt_extended);
    }

    #[test]
    fn extended_nlt_bit_decodes() {
        // Bit 3 → 0x10.
        let caps = parse_capabilities(&[0x10]).unwrap();
        assert!(caps.nlt_extended);
    }

    #[test]
    fn vertical_subsampling_bit_decodes() {
        // Bit 4 → 0x08.
        let caps = parse_capabilities(&[0x08]).unwrap();
        assert!(caps.vertical_subsampling);
    }

    #[test]
    fn cwd_bit_decodes() {
        // Bit 5 → 0x04.
        let caps = parse_capabilities(&[0x04]).unwrap();
        assert!(caps.cwd);
    }

    #[test]
    fn lossless_bit_decodes() {
        // Bit 6 → 0x02.
        let caps = parse_capabilities(&[0x02]).unwrap();
        assert!(caps.lossless);
    }

    #[test]
    fn raw_mode_switch_bit_in_second_byte() {
        // Bit 8 is the MSB of byte 1 → 0x80.
        // Need the last byte non-zero to satisfy the spec rule.
        let caps = parse_capabilities(&[0x40, 0x80]).unwrap();
        assert!(caps.star_tetrix);
        assert!(caps.raw_mode_switch);
    }

    #[test]
    fn all_byte_one_bits_decode() {
        // 0xFE = bits 0..6 all set in byte 0. Bit 0 is unused per
        // NOTE 3 so we still expect star_tetrix..lossless all true.
        let caps = parse_capabilities(&[0xFE]).unwrap();
        assert!(caps.star_tetrix);
        assert!(caps.nlt_quadratic);
        assert!(caps.nlt_extended);
        assert!(caps.vertical_subsampling);
        assert!(caps.cwd);
        assert!(caps.lossless);
        assert!(!caps.raw_mode_switch);
    }

    #[test]
    fn rejects_trailing_zero_byte() {
        // Lcap > 2, last byte all-zero — forbidden by §A.4.3.
        let err = parse_capabilities(&[0x40, 0x00]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("non-zero"), "expected diag, got {msg}");
    }

    #[test]
    fn lossy_accepts_trailing_zero() {
        // Same input as the rejection above — the lenient variant
        // returns the documented bits without erroring.
        let caps = parse_capabilities_lossy(&[0x40, 0x00]);
        assert!(caps.star_tetrix);
        assert!(!caps.raw_mode_switch);
    }

    #[test]
    fn ignores_bit_zero_and_reserved() {
        // 0x80 (bit 0 = unused) → no documented bit should fire.
        let caps = parse_capabilities(&[0x80]).unwrap();
        assert!(!caps.any());
    }
}
