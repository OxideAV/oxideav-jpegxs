//! Probe utility for JPEG XS codestreams.
//!
//! Given a raw byte buffer that begins with the SOC marker (`FF 10`),
//! parses just enough of the marker chain to return the geometric
//! summary commonly needed by routing layers / muxers.

use crate::codestream::{self, Codestream};

/// Summary information returned by [`probe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegXsFileInfo {
    /// Image width on the sample grid (`Wf` from the picture header).
    pub width: u32,
    /// Image height on the sample grid (`Hf` from the picture header).
    pub height: u32,
    /// Profile id from the picture header (`Ppih`); 0 means
    /// "no restrictions". See ISO/IEC 21122-2.
    pub profile: u16,
    /// Level + sublevel from the picture header (`Plev`).
    pub level: u16,
    /// Number of components from the picture header (`Nc`).
    pub num_components: u8,
    /// Maximum bit depth across all components (from the CDT).
    pub bit_depth: u8,
    /// Colour transformation id (`Cpih`). See Table A.9.
    pub cpih: u8,
    /// Lossless coding mode flag — true when `Fq == 0` (Table A.8).
    pub lossless: bool,
}

/// Try to parse just the JPEG XS header chain. Accepts either a bare
/// ISO/IEC 21122-1 codestream (begins with the SOC marker `FF 10`) or a
/// box-wrapped JXS still-image file (ISO/IEC 21122-3 Annex A, begins with
/// the JPEG XS Signature box); in the latter case the embedded codestream
/// is located and probed. Returns `None` if the buffer matches neither
/// shape or fails to parse cleanly. For richer error reporting use
/// [`codestream::parse`] directly.
pub fn probe(buf: &[u8]) -> Option<JpegXsFileInfo> {
    // A box-wrapped .jxs file: locate the embedded codestream first.
    let cs_buf: &[u8] = if crate::fileformat::is_jxs_file(buf) {
        let file = crate::fileformat::parse_jxs_file(buf).ok()?;
        // `codestream` borrows `buf`, not `file`, so the slice outlives
        // the parsed box structure dropped at the end of this branch.
        file.codestream(buf)
    } else if buf.len() >= 2 && buf[0] == 0xff && buf[1] == 0x10 {
        buf
    } else {
        return None;
    };
    let cs: Codestream = codestream::parse(cs_buf).ok()?;
    Some(JpegXsFileInfo {
        width: cs.pih.width(),
        height: cs.pih.height(),
        profile: cs.pih.ppih,
        level: cs.pih.plev,
        num_components: cs.pih.nc,
        bit_depth: cs.cdt.max_bit_depth(),
        cpih: cs.pih.cpih,
        lossless: cs.pih.is_lossless(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_unwraps_box_file() {
        // Encode a 12x8 luma codestream, wrap it, and confirm probe
        // reports the same geometry from the wrapped file as from the
        // bare codestream.
        let (w, h) = (12u16, 8u16);
        let pixels: Vec<u8> = (0..(w as usize * h as usize))
            .map(|i| (i * 9) as u8)
            .collect();
        let cs =
            crate::encoder::encode_planar(w, h, 1, 0, 1, 1, std::slice::from_ref(&pixels)).unwrap();
        let bare = probe(&cs).expect("probe bare codestream");
        let file = crate::fileformat::write_jxs_file(&cs).unwrap();
        let wrapped = probe(&file).expect("probe wrapped file");
        assert_eq!(bare, wrapped);
        assert_eq!(wrapped.width, w as u32);
        assert_eq!(wrapped.height, h as u32);
        assert_eq!(wrapped.num_components, 1);
    }

    #[test]
    fn probe_rejects_garbage_and_truncated_box() {
        assert!(probe(&[]).is_none());
        assert!(probe(&[0x00, 0x01, 0x02]).is_none());
        // Leading signature bytes but a truncated box chain.
        let mut sig = crate::fileformat::SIGNATURE_BOX.to_vec();
        sig.extend_from_slice(&[0x00, 0x00, 0x00, 0x10]); // dangling LBox
        assert!(probe(&sig).is_none());
    }
}
