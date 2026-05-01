//! Inverse multi-component (colour) transformations — ISO/IEC 21122-1:
//! 2022 Annex F.
//!
//! Two transforms are normative:
//!
//! * [`inverse_rct`] — Annex F.3, Table F.2. Cpih == 1. Three-component
//!   reversible colour transform that maps the encoder-side
//!   (Y, Cb, Cr) decorrelation back to (R, G, B). The forward direction
//!   (Annex F.4 / Table F.3) is `Y = (R + 2G + B) >> 2`,
//!   `Cb = B - G`, `Cr = R - G`; the inverse reconstructs
//!   `G = Y - ((Cb + Cr) >> 2)`, `R = G + Cr`, `B = G + Cb`.
//!
//!   Component layout per Table F.2: input component 0 is luma (Y),
//!   1 is Cb, 2 is Cr; output component 0 is red, 1 is green, 2 is
//!   blue. Components ≥ 3 are passed through unchanged (Annex F.2,
//!   Table F.1: "Set Ω[c,x,y] = O[c,x,y] for all components c≥3").
//!
//! * [`inverse_star_tetrix`] — Annex F.5, Tables F.4 / F.5 / F.6 / F.7
//!   / F.8. Cpih == 3. Four-component CFA-pixel decoder that combines
//!   inverse average, delta, Y, and CbCr lifting steps. The chroma
//!   weighting exponents `e1` and `e2` come from the CTS marker
//!   (Annex A.4.8), and the per-component sub-pixel displacement
//!   vector `(δx, δy)` from the CFA-pattern type `Ct` derived from CRG
//!   (Annex A.4.9, Tables F.9 / F.10).
//!
//! Both transforms operate **post-DWT, pre-output-mapping**: their
//! input is the int32 sample plane the inverse DWT produced for each
//! component (one element per (x, y) on the sample grid `Wf × Hf`),
//! and their output overwrites the same buffers in place. The decoder
//! pipeline runs Annex F → Annex G in that order (Table A.1 puts the
//! NLT step after the colour transform).
//!
//! Allocation. RCT and Star-Tetrix both run in-place on the caller-
//! owned per-component buffers — no heap traffic per transform call.

use oxideav_core::{Error, Result};

/// Apply the inverse reversible colour transform (Cpih == 1) to the
/// first three component planes in place.
///
/// `planes` is a slice of mutable per-component plane buffers, each
/// laid out row-major `wf × hf`. Components beyond index 2 are not
/// touched.
///
/// Per Annex F.2: Cpih == 1 requires `Nc >= 3` and `sx[i] == sy[i] == 1`
/// for `i < 3`. The caller must enforce that — `inverse_rct` only checks
/// the buffer count and length here.
pub fn inverse_rct(planes: &mut [&mut [i32]], wf: usize, hf: usize) -> Result<()> {
    if planes.len() < 3 {
        return Err(Error::invalid(format!(
            "jpegxs Cpih=1 (RCT) requires at least 3 component planes, got {}",
            planes.len()
        )));
    }
    let want = wf
        .checked_mul(hf)
        .ok_or_else(|| Error::invalid("jpegxs RCT: wf * hf overflow"))?;
    for (i, p) in planes.iter().enumerate().take(3) {
        if p.len() != want {
            return Err(Error::invalid(format!(
                "jpegxs RCT: component {i} has {} samples, expected {want}",
                p.len()
            )));
        }
    }

    // Annex F.3 Table F.2:
    //   o1 = i0 - ((i1 + i2) >> 2)   // green
    //   o0 = o1 + i2                  // red
    //   o2 = o1 + i1                  // blue
    //
    // i0 = O[0] (luma), i1 = O[1] (Cb), i2 = O[2] (Cr).
    // Output Ω[0] = red, Ω[1] = green, Ω[2] = blue.
    //
    // We split the slice borrows so we can write components 0/1/2 in
    // turn without aliasing (each loop iteration touches one (x, y)
    // across all three planes).
    let (p0_p1, p2_rest) = planes.split_at_mut(2);
    let (p0_slot, p1_slot) = p0_p1.split_at_mut(1);
    let p0 = &mut **p0_slot.get_mut(0).expect("checked above");
    let p1 = &mut **p1_slot.get_mut(0).expect("checked above");
    let p2 = &mut **p2_rest.get_mut(0).expect("checked above");

    for x in 0..want {
        let i0 = p0[x];
        let i1 = p1[x];
        let i2 = p2[x];
        let o1 = i0 - ((i1 + i2) >> 2);
        let o0 = o1 + i2;
        let o2 = o1 + i1;
        p0[x] = o0;
        p1[x] = o1;
        p2[x] = o2;
    }
    Ok(())
}

/// Inverse Star-Tetrix transform (Cpih == 3) — Annex F.5.
///
/// Round 5 wires the inverse Star-Tetrix into the decoder pipeline at
/// signature level only; the spec's four lifting steps each touch a
/// 4-neighbourhood of pixels indirected by the per-component CFA
/// displacement `(δx, δy)` derived from the CTS / CRG markers
/// (Tables F.9 / F.10) and the chroma exponents `e1`, `e2` from
/// the CTS body (Annex A.4.8). The full implementation requires both
/// the CRG marker parser and the CTS marker parser, plus the
/// `access(c, x, y, rx, ry)` reflection function in Table F.12. The
/// core CFA / live-camera decoders the framework targets do not use
/// Cpih == 3, so the Star-Tetrix path is left as a guarded stub the
/// decoder can call when the CTS / CRG support arrives.
pub fn inverse_star_tetrix(
    _planes: &mut [&mut [i32]],
    _wf: usize,
    _hf: usize,
    _e1: u8,
    _e2: u8,
    _ct: u8,
) -> Result<()> {
    Err(Error::Unsupported(
        "jpegxs Cpih=3 (Star-Tetrix) inverse colour transform: requires CTS+CRG marker parsing (round 6+)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward RCT companion (Annex F.4, Table F.3) used only by tests
    /// to round-trip the inverse. `o0 = (R + 2G + B) >> 2`,
    /// `o1 = B - G`, `o2 = R - G`.
    fn forward_rct(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
        let o0 = (r + 2 * g + b) >> 2;
        let o1 = b - g;
        let o2 = r - g;
        (o0, o1, o2)
    }

    #[test]
    fn rct_round_trips_constant_pixel() {
        // RGB (100, 200, 50). Forward → (Y, Cb, Cr).
        let (y, cb, cr) = forward_rct(100, 200, 50);
        let mut p0 = vec![y; 4];
        let mut p1 = vec![cb; 4];
        let mut p2 = vec![cr; 4];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        inverse_rct(&mut planes, 2, 2).unwrap();
        // Output order is R, G, B per Annex F.3 (the wires assignment in
        // Table F.2 puts red on Ω[0]).
        assert_eq!(p0, vec![100, 100, 100, 100]);
        assert_eq!(p1, vec![200, 200, 200, 200]);
        assert_eq!(p2, vec![50, 50, 50, 50]);
    }

    #[test]
    fn rct_round_trips_arbitrary_pixels() {
        let pix = [
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 64, 200),
        ];
        let mut p0: Vec<i32> = Vec::new();
        let mut p1: Vec<i32> = Vec::new();
        let mut p2: Vec<i32> = Vec::new();
        for &(r, g, b) in &pix {
            let (y, cb, cr) = forward_rct(r, g, b);
            p0.push(y);
            p1.push(cb);
            p2.push(cr);
        }
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        inverse_rct(&mut planes, pix.len(), 1).unwrap();
        for (i, &(r, g, b)) in pix.iter().enumerate() {
            assert_eq!((p0[i], p1[i], p2[i]), (r, g, b), "pixel {i}");
        }
    }

    #[test]
    fn rct_rejects_wrong_buffer_size() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 3];
        let mut planes: [&mut [i32]; 3] = [&mut p0, &mut p1, &mut p2];
        assert!(inverse_rct(&mut planes, 2, 2).is_err());
    }

    #[test]
    fn rct_rejects_too_few_components() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 2] = [&mut p0, &mut p1];
        assert!(inverse_rct(&mut planes, 2, 2).is_err());
    }

    #[test]
    fn star_tetrix_reports_unsupported() {
        let mut p0 = vec![0i32; 4];
        let mut p1 = vec![0i32; 4];
        let mut p2 = vec![0i32; 4];
        let mut p3 = vec![0i32; 4];
        let mut planes: [&mut [i32]; 4] = [&mut p0, &mut p1, &mut p2, &mut p3];
        let err = inverse_star_tetrix(&mut planes, 2, 2, 0, 0, 0).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Star-Tetrix"),
            "expected Star-Tetrix unsupported error, got {msg}"
        );
    }
}
