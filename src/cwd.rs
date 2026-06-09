//! Component Wavelet Decomposition marker (CWD) — ISO/IEC 21122-1:2022
//! §A.4.7, Table A.18.
//!
//! Optional marker. When present, the CWD body is exactly one byte
//! carrying the `Sd` field — the number of *suppressed* trailing
//! components (those skipped by the wavelet decomposition cascade so
//! that their samples are copied straight from the input rather than
//! going through the DWT). `Sd ∈ 1..=Nc-1` per Table A.18, and the
//! marker is forbidden unless `Nc > 3` per the same table.
//!
//! The geometry-level constraints (`Nc > 3`, `Sd ∈ 1..=Nc-1`) require
//! the picture header's `Nc` value and are enforced by the codestream
//! marker-chain parser at parse time. The body-level parser in this
//! module enforces only what is observable from the CWD body itself:
//! the body length is exactly one byte, and `Sd != 0`.

use crate::error::{JpegXsError as Error, Result};

/// Decoded CWD marker body — the single `Sd` field per Table A.18.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CwdMarker {
    /// `Sd` — number of trailing components suppressed from the wavelet
    /// decomposition. Range `1..=Nc-1` (the body-level parser only
    /// enforces `Sd != 0`; the upper bound depends on `Nc` and is
    /// checked at codestream-parser time).
    pub sd: u8,
}

/// `Lcwd` body length in bytes (the body is exactly 1 byte after the
/// 2-byte length field).
pub const CWD_BODY_LEN: usize = 1;

/// Parse a CWD body (the bytes after `Lcwd`). Per Table A.18 the body
/// is exactly 1 byte carrying `Sd`. Body-level errors surface here:
///
/// * `body.len() != 1` — wrong segment length.
/// * `Sd == 0` — `Sd = 0` is the no-suppression default and the
///   spec forbids emitting the marker in that case (Table A.18:
///   `Sd ∈ 1..=Nc-1`).
///
/// The geometry-dependent upper bound `Sd ≤ Nc-1` requires the
/// picture-header `Nc` and is enforced by the codestream marker-chain
/// parser, not by this body-level entry point.
pub fn parse_cwd(body: &[u8]) -> Result<CwdMarker> {
    if body.len() != CWD_BODY_LEN {
        return Err(Error::invalid(format!(
            "jpegxs CWD: body must be {CWD_BODY_LEN} byte (Sd), got {}",
            body.len()
        )));
    }
    let sd = body[0];
    if sd == 0 {
        return Err(Error::invalid(
            "jpegxs CWD: Sd must be >= 1 per Annex A.4.7 Table A.18",
        ));
    }
    Ok(CwdMarker { sd })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sd_one() {
        let body = [1u8];
        let cwd = parse_cwd(&body).unwrap();
        assert_eq!(cwd.sd, 1);
    }

    #[test]
    fn parses_sd_max_byte() {
        // Body-level parser does not know `Nc`; any non-zero `Sd` is
        // body-level valid.
        let body = [255u8];
        let cwd = parse_cwd(&body).unwrap();
        assert_eq!(cwd.sd, 255);
    }

    #[test]
    fn rejects_sd_zero() {
        let body = [0u8];
        let err = parse_cwd(&body).unwrap_err();
        assert!(
            format!("{err}").contains("Sd"),
            "expected Sd-range error, got {err}"
        );
    }

    #[test]
    fn rejects_wrong_body_length() {
        assert!(parse_cwd(&[]).is_err());
        assert!(parse_cwd(&[1, 2]).is_err());
        assert!(parse_cwd(&[1, 2, 3]).is_err());
    }
}
