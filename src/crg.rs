//! Component Registration marker (CRG) — ISO/IEC 21122-1:2022 §A.4.9,
//! Table A.21.
//!
//! Optional in general; mandatory iff `Cpih == 3` (Star-Tetrix). Body
//! after the `Lcrg` length field is `Nc` (`Xcrg[c]`, `Ycrg[c]`) pairs,
//! each a big-endian u16 in the range 0..=65535. The values express
//! the relative placement of component `c` relative to the sample-grid
//! point in units of `1 / 65536`. A value of `32768` corresponds to a
//! placement midway between two adjacent sample-grid points.
//!
//! For Star-Tetrix the only registrations the spec defines are the
//! four CFA configurations in Table F.9 (RGGB, BGGR, GRBG, GBRG); each
//! component carries an `Xcrg`, `Ycrg` value of either `0` or `32768`.
//! The `cfa_pattern_type` helper maps a parsed [`CrgMarker`] body back
//! to the CFA pattern type `Ct ∈ {0, 1}` per Table F.9.

use oxideav_core::{Error, Result};

/// One component's registration entry (per Table A.21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrgEntry {
    /// `Xcrg[c]` — relative horizontal placement, units of 1/65536.
    pub x_crg: u16,
    /// `Ycrg[c]` — relative vertical placement, units of 1/65536.
    pub y_crg: u16,
}

/// Decoded CRG marker body — one entry per component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrgMarker {
    pub entries: Vec<CrgEntry>,
}

/// Parse a CRG body (the bytes after `Lcrg`). `nc` is the number of
/// components from the picture header. The body must therefore be
/// exactly `4 * nc` bytes (two u16 fields per component).
pub fn parse_crg(body: &[u8], nc: u8) -> Result<CrgMarker> {
    let want = 4 * (nc as usize);
    if body.len() != want {
        return Err(Error::invalid(format!(
            "jpegxs CRG: body must be {want} bytes (4*Nc) for Nc={nc}, got {}",
            body.len()
        )));
    }
    let mut entries = Vec::with_capacity(nc as usize);
    for c in 0..nc as usize {
        let off = 4 * c;
        let x = u16::from_be_bytes([body[off], body[off + 1]]);
        let y = u16::from_be_bytes([body[off + 2], body[off + 3]]);
        entries.push(CrgEntry { x_crg: x, y_crg: y });
    }
    Ok(CrgMarker { entries })
}

/// CFA pattern type `Ct` per Table F.9, derived from the CRG entries
/// for the first four components. Returns `Some(0 | 1)` for one of the
/// four standard arrangements (RGGB / BGGR / GRBG / GBRG), or `None`
/// for any other combination (which Table F.9 explicitly marks as
/// "Reserved for ISO/IEC purposes").
///
/// Star-Tetrix mandates `Nc >= 4`; the helper returns `None` if fewer
/// entries are available.
pub fn cfa_pattern_type(crg: &CrgMarker) -> Option<u8> {
    if crg.entries.len() < 4 {
        return None;
    }
    let q: Vec<(u8, u8)> = crg
        .entries
        .iter()
        .take(4)
        .map(|e| (quantise(e.x_crg), quantise(e.y_crg)))
        .collect();

    // Table F.9 row patterns. Each row is a 4-tuple of (xq, yq) pairs
    // for components 0..3, where 0 means Xcrg/Ycrg == 0, 1 means
    // Xcrg/Ycrg == 32768. Rows are: RGGB → Ct=0; BGGR → Ct=0;
    // GRBG → Ct=1; GBRG → Ct=1.
    const RGGB: [(u8, u8); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
    const BGGR: [(u8, u8); 4] = [(1, 1), (1, 0), (0, 1), (0, 0)];
    const GRBG: [(u8, u8); 4] = [(1, 0), (0, 0), (1, 1), (0, 1)];
    const GBRG: [(u8, u8); 4] = [(0, 1), (0, 0), (1, 1), (1, 0)];

    let arr: [(u8, u8); 4] = [q[0], q[1], q[2], q[3]];
    if arr == RGGB || arr == BGGR {
        Some(0)
    } else if arr == GRBG || arr == GBRG {
        Some(1)
    } else {
        None
    }
}

/// Map an `Xcrg`/`Ycrg` value to the binary super-pixel coordinate
/// (`0` for `==0`, `1` for `==32768`). Any other value is mapped to
/// `2`, which never matches a Table F.9 row and so leads
/// `cfa_pattern_type` to return `None`.
fn quantise(v: u16) -> u8 {
    match v {
        0 => 0,
        32768 => 1,
        _ => 2,
    }
}

/// Component displacement vector `(δx, δy)` for component `c` under
/// CFA pattern type `Ct ∈ {0, 1}` per Table F.10. Returns `None` for
/// an out-of-range component or unknown `Ct`.
pub fn displacement(ct: u8, c: usize) -> Option<(u8, u8)> {
    // Table F.10 — `(δx, δy)` for c = 0..3.
    const CT0: [(u8, u8); 4] = [(0, 1), (1, 1), (0, 0), (1, 0)];
    const CT1: [(u8, u8); 4] = [(1, 1), (0, 1), (1, 0), (0, 0)];
    if c >= 4 {
        return None;
    }
    match ct {
        0 => Some(CT0[c]),
        1 => Some(CT1[c]),
        _ => None,
    }
}

/// Inverse `k[δx, δy]` mapping per Table F.11 — given a displacement
/// vector and a CFA pattern type, return the component index.
pub fn component_at(ct: u8, dx: u8, dy: u8) -> Option<u8> {
    // Table F.11.
    if ct == 0 {
        return match (dx, dy) {
            (0, 1) => Some(0),
            (1, 1) => Some(1),
            (0, 0) => Some(2),
            (1, 0) => Some(3),
            _ => None,
        };
    }
    if ct == 1 {
        return match (dx, dy) {
            (1, 1) => Some(0),
            (0, 1) => Some(1),
            (1, 0) => Some(2),
            (0, 0) => Some(3),
            _ => None,
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_for(entries: &[(u16, u16)]) -> Vec<u8> {
        let mut v = Vec::with_capacity(entries.len() * 4);
        for &(x, y) in entries {
            v.extend_from_slice(&x.to_be_bytes());
            v.extend_from_slice(&y.to_be_bytes());
        }
        v
    }

    #[test]
    fn parses_rggb_registration() {
        let body = body_for(&[(0, 0), (32768, 0), (0, 32768), (32768, 32768)]);
        let crg = parse_crg(&body, 4).unwrap();
        assert_eq!(crg.entries.len(), 4);
        assert_eq!(cfa_pattern_type(&crg), Some(0));
    }

    #[test]
    fn parses_bggr_registration() {
        let body = body_for(&[(32768, 32768), (32768, 0), (0, 32768), (0, 0)]);
        let crg = parse_crg(&body, 4).unwrap();
        assert_eq!(cfa_pattern_type(&crg), Some(0));
    }

    #[test]
    fn parses_grbg_and_gbrg() {
        let grbg = body_for(&[(32768, 0), (0, 0), (32768, 32768), (0, 32768)]);
        let crg = parse_crg(&grbg, 4).unwrap();
        assert_eq!(cfa_pattern_type(&crg), Some(1));

        let gbrg = body_for(&[(0, 32768), (0, 0), (32768, 32768), (32768, 0)]);
        let crg = parse_crg(&gbrg, 4).unwrap();
        assert_eq!(cfa_pattern_type(&crg), Some(1));
    }

    #[test]
    fn unknown_registration_is_none() {
        let body = body_for(&[(1, 2), (3, 4), (5, 6), (7, 8)]);
        let crg = parse_crg(&body, 4).unwrap();
        assert_eq!(cfa_pattern_type(&crg), None);
    }

    #[test]
    fn rejects_wrong_body_length() {
        assert!(parse_crg(&[], 1).is_err());
        assert!(parse_crg(&[0u8; 5], 1).is_err());
        assert!(parse_crg(&[0u8; 12], 4).is_err());
    }

    #[test]
    fn displacement_table_f10() {
        // Ct = 0 row: (0,1), (1,1), (0,0), (1,0).
        assert_eq!(displacement(0, 0), Some((0, 1)));
        assert_eq!(displacement(0, 1), Some((1, 1)));
        assert_eq!(displacement(0, 2), Some((0, 0)));
        assert_eq!(displacement(0, 3), Some((1, 0)));
        assert_eq!(displacement(0, 4), None);
        // Ct = 1 row: (1,1), (0,1), (1,0), (0,0).
        assert_eq!(displacement(1, 0), Some((1, 1)));
        assert_eq!(displacement(1, 1), Some((0, 1)));
        assert_eq!(displacement(1, 2), Some((1, 0)));
        assert_eq!(displacement(1, 3), Some((0, 0)));
        // Unknown Ct.
        assert_eq!(displacement(2, 0), None);
    }

    #[test]
    fn inverse_table_f11_round_trips_table_f10() {
        for ct in 0u8..=1 {
            for c in 0..4usize {
                let (dx, dy) = displacement(ct, c).unwrap();
                assert_eq!(component_at(ct, dx, dy), Some(c as u8));
            }
        }
    }
}
