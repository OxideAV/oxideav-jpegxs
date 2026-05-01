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

/// Try to parse just the JPEG XS header chain. Returns `None` if the
/// buffer does not begin with the SOC marker (`FF 10`) or fails to
/// parse cleanly. For richer error reporting use
/// [`codestream::parse`] directly.
pub fn probe(buf: &[u8]) -> Option<JpegXsFileInfo> {
    if buf.len() < 2 || buf[0] != 0xff || buf[1] != 0x10 {
        return None;
    }
    let cs: Codestream = codestream::parse(buf).ok()?;
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
