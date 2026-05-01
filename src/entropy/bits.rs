//! Bit-stream reader and VLC primitive for JPEG XS Annex C.
//!
//! Bits in JPEG XS are packed MSB-first into bytes (the codestream
//! does not byte-stuff). The reader keeps a byte cursor and a bit
//! offset within the current byte, both monotonically advancing. Once
//! the byte cursor reaches the end of the slice or precinct buffer,
//! further reads return `Error::Decode`.
//!
//! The decoder holds the entire entropy buffer in memory (the
//! upstream codestream walker already constrains slice and precinct
//! lengths against `Lprc[p] ≤ 2^20 - 1` per Table C.1, so no
//! unbounded allocation is possible from this layer).

use oxideav_core::{Error, Result};

/// MSB-first bit reader over a byte slice.
#[derive(Debug)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Byte index of the current byte.
    byte_pos: usize,
    /// Bit offset inside `buf[byte_pos]`. 0 = MSB; 8 means we've
    /// consumed all bits of the current byte and need to advance.
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice for MSB-first bit consumption.
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Total bits remaining (counting from the current bit cursor to
    /// the end of the buffer).
    pub fn remaining_bits(&self) -> usize {
        if self.byte_pos >= self.buf.len() {
            return 0;
        }
        let in_byte = (8 - self.bit_pos as usize).min(8);
        let trailing = self.buf.len() - self.byte_pos - 1;
        in_byte + trailing * 8
    }

    /// Number of whole bytes consumed so far (rounding up if the bit
    /// cursor sits mid-byte).
    pub fn bytes_consumed(&self) -> usize {
        if self.bit_pos == 0 {
            self.byte_pos
        } else {
            self.byte_pos + 1
        }
    }

    /// True iff the bit cursor is currently at a byte boundary.
    pub fn at_byte_boundary(&self) -> bool {
        self.bit_pos == 0
    }

    /// Skip whatever bits remain in the current byte. No-op if already
    /// at a byte boundary. Annex C uses this for the `pad(8)` trailers
    /// at the end of significance / bitplane-count / data / sign
    /// sub-packets and the precinct header.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }

    /// Skip `n` whole bytes of filler (Table C.7 / C.8 / C.9). Must be
    /// at a byte boundary first.
    pub fn skip_bytes(&mut self, n: usize) -> Result<()> {
        if !self.at_byte_boundary() {
            return Err(Error::invalid(
                "jpegxs entropy: skip_bytes invoked off a byte boundary",
            ));
        }
        if self.byte_pos + n > self.buf.len() {
            return Err(Error::invalid(format!(
                "jpegxs entropy: filler skip {n} exceeds remaining {}",
                self.buf.len() - self.byte_pos
            )));
        }
        self.byte_pos += n;
        Ok(())
    }

    /// Read a single bit. Returns 0 or 1.
    pub fn read_bit(&mut self) -> Result<u8> {
        if self.byte_pos >= self.buf.len() {
            return Err(Error::invalid(
                "jpegxs entropy: read_bit past end of buffer",
            ));
        }
        let byte = self.buf[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 0x01;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    /// Read `n` bits (1..=32) MSB-first as an unsigned integer.
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::invalid(format!(
                "jpegxs entropy: read_bits({n}) > 32"
            )));
        }
        let mut out: u32 = 0;
        for _ in 0..n {
            out = (out << 1) | (self.read_bit()? as u32);
        }
        Ok(out)
    }
}

/// Annex C.7.1, Table C.15 — variable-length signed-quantity decoder.
///
/// Reads a unary count of 1-bits terminated by a 0 comma bit. The
/// decoded `x` (number of leading 1-bits) is then mapped to a signed
/// value through one of two sub-alphabets:
///
/// * If `x > 2θ` where `θ = max(r - t, 0)`, the symbol is in the
///   unary sub-alphabet → return `x - θ` (always positive).
/// * Else if `x > 0`, the symbol is in the signed binary sub-alphabet
///   → odd `x` → `-(x/2 + 1)` rounded down (`-ceil(x/2)`), even `x`
///   → `x/2` (positive).
/// * Else (`x == 0`) → 0.
///
/// The standard caps `x` at 32 — any 33rd consecutive 1-bit is a
/// hard error (the decoder has lost synchronisation).
///
/// Returns the decoded signed value as `i32`.
pub fn vlc(reader: &mut BitReader<'_>, r: i32, t: i32) -> Result<i32> {
    let theta = (r - t).max(0);
    let mut x: u32 = 0;
    loop {
        let b = reader.read_bit()?;
        if b == 0 {
            break;
        }
        x += 1;
        if x >= 32 {
            // Read one more bit to confirm the 32nd 1-bit was indeed
            // followed by a 0 (or another 1, which is the error).
            let next = reader.read_bit()?;
            if next == 1 {
                return Err(Error::invalid(
                    "jpegxs entropy: vlc decoder saw >32 consecutive 1-bits",
                ));
            }
            break;
        }
    }
    let xi = x as i32;
    if xi > 2 * theta {
        Ok(xi - theta)
    } else if xi > 0 {
        if xi & 1 == 1 {
            // Odd codeword → negative value. Spec: return -⎾x/2⏋
            Ok(-((xi + 1) / 2))
        } else {
            // Even codeword → positive value: return x/2.
            Ok(xi / 2)
        }
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_msb_first() {
        // 0b1010_0110, 0b1100_0011 → expect 1010, 0110, 1100, 0011.
        let buf = [0xa6u8, 0xc3];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(4).unwrap(), 0b1010);
        assert_eq!(r.read_bits(4).unwrap(), 0b0110);
        assert_eq!(r.read_bits(4).unwrap(), 0b1100);
        assert_eq!(r.read_bits(4).unwrap(), 0b0011);
        assert!(r.read_bit().is_err(), "should error past end");
    }

    #[test]
    fn align_skips_partial_byte() {
        // Read 3 bits, then align — next read should start at byte 1.
        let buf = [0xa6u8, 0x12];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits(3).unwrap();
        assert!(!r.at_byte_boundary());
        r.align_to_byte();
        assert!(r.at_byte_boundary());
        assert_eq!(r.read_bits(8).unwrap(), 0x12);
    }

    #[test]
    fn vlc_zero_codeword() {
        // Single 0 bit → 0.
        let buf = [0b0000_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 0, 0).unwrap(), 0);
    }

    #[test]
    fn vlc_unary_alphabet() {
        // r=0, t=0 → θ=0, so 2θ=0. Any x>=1 → unary sub-alphabet
        // returns x - 0 = x. Code "110" = two 1-bits + 0 → x=2 → 2.
        let buf = [0b1100_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 0, 0).unwrap(), 2);
    }

    #[test]
    fn vlc_signed_binary_alphabet() {
        // r=2, t=0 → θ=2, 2θ=4. x ≤ 4 selects the signed sub-alphabet.
        // x=1 (one 1-bit + 0) → odd → -1. Bit pattern 10000000.
        let buf = [0b1000_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 2, 0).unwrap(), -1);
        // x=2 (110) → even → +1.
        let buf = [0b1100_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 2, 0).unwrap(), 1);
        // x=3 (1110) → odd → -2 (ceil(3/2)=2).
        let buf = [0b1110_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 2, 0).unwrap(), -2);
        // x=4 (11110) → even → +2.
        let buf = [0b1111_0000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 2, 0).unwrap(), 2);
        // x=5 (111110) → x > 2θ=4, unary → x - θ = 5 - 2 = 3.
        let buf = [0b1111_1000u8];
        let mut r = BitReader::new(&buf);
        assert_eq!(vlc(&mut r, 2, 0).unwrap(), 3);
    }

    #[test]
    fn vlc_rejects_unsynchronised() {
        // 33 consecutive 1-bits in the input must trigger an error.
        // We feed five 0xFF bytes (40 ones) then a 0 bit.
        let buf = [0xff, 0xff, 0xff, 0xff, 0xff, 0x00];
        let mut r = BitReader::new(&buf);
        let res = vlc(&mut r, 0, 0);
        assert!(
            res.is_err(),
            "expected error from too many 1-bits, got {res:?}"
        );
    }

    #[test]
    fn skip_bytes_requires_byte_boundary() {
        let buf = [0xa6u8, 0x12, 0x34];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits(3).unwrap();
        assert!(r.skip_bytes(1).is_err());
        r.align_to_byte();
        assert!(r.skip_bytes(1).is_ok());
        assert_eq!(r.read_bits(8).unwrap(), 0x34);
    }

    #[test]
    fn bytes_consumed_rounds_up_when_mid_byte() {
        let buf = [0xa6u8, 0x12];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.bytes_consumed(), 0);
        let _ = r.read_bits(3).unwrap();
        assert_eq!(r.bytes_consumed(), 1);
        let _ = r.read_bits(5).unwrap();
        assert_eq!(r.bytes_consumed(), 1);
        let _ = r.read_bits(8).unwrap();
        assert_eq!(r.bytes_consumed(), 2);
    }
}
