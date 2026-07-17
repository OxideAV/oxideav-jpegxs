//! Encoder-side conformance signalling — the `Lcod` / `Ppih` / `Plev`
//! fields of the picture header (ISO/IEC 21122-1:2022 Table 11 /
//! §A.4.4) and the profile / level / sublevel indicators they carry
//! (ISO/IEC 21122-2:2019 Annex A).
//!
//! Every encoder entry point in this crate emits a picture header with
//! `Lcod = 0` (variable-bitrate coding), `Ppih = 0` (unrestricted
//! profile) and `Plev = 0` (unrestricted level / sublevel). Those are
//! conforming values, but they claim nothing: §A.2.2 and §A.5 of
//! ISO/IEC 21122-2 state that the unrestricted profile / level "shall
//! not be considered as a conformance point". This module upgrades an
//! already-encoded codestream to a *self-describing* one:
//!
//! * [`declare_profile`] writes a non-zero `Ppih` (Table A.5),
//! * [`declare_level_sublevel`] writes a `Plev` (Tables A.12 / A.13),
//! * [`declare_cbr`] writes the SOC-to-EOC byte count into `Lcod`
//!   (constant-bitrate self-description, 21122-1 Table 11),
//!
//! and each of them **verifies the claim before keeping it**: the
//! patched stream is re-parsed and pushed through the same
//! [`crate::profile::check_codestream`] / [`crate::profile::check_level`]
//! / [`crate::profile::check_codestream_size`] gates the decoder runs on
//! every stream it accepts, so a claim the codestream does not actually
//! satisfy is rejected and the buffer is restored unchanged. An encoder
//! using this module therefore cannot emit a false profile / level /
//! sublevel / CBR declaration.
//!
//! [`pick_profile`], [`pick_level`] and [`pick_sublevel`] choose the
//! tightest claims a given stream satisfies, and [`declare_auto`]
//! applies all of them in one call.
//!
//! The fields are patched in place (they live at fixed offsets inside
//! the PIH marker body, located by walking the marker chain), so this
//! works with the output of *every* encoder entry point — including
//! streams that have been wrapped and unwrapped through the Part-3 box
//! file format — without threading extra parameters through the
//! encoding funnels.

use crate::codestream;
use crate::error::{JpegXsError as Error, Result};
use crate::profile::{
    check_codestream, check_codestream_size, check_level, max_codestream_size, Level, Profile,
    Sublevel,
};

/// Byte offset of the `Lcod` field relative to the start of the PIH
/// marker body (21122-1 §A.4.4: `Lcod` is the first field after
/// `Lpih`).
const LCOD_REL: usize = 0;
/// Byte offset of the `Ppih` field relative to the PIH body start.
const PPIH_REL: usize = 4;
/// Byte offset of the `Plev` field relative to the PIH body start.
const PLEV_REL: usize = 6;

/// Locate the PIH marker body inside `buf` by walking the marker chain
/// from SOC (21122-1 Annex A: every marker between SOC and the first
/// SLH is a marker segment with a two-byte big-endian length that
/// counts itself but not the two marker bytes).
///
/// Returns the byte offset of the first PIH body byte (the `Lcod`
/// field). Errors if `buf` does not start with SOC, the chain is
/// malformed, or SLH / EOC is reached before a PIH.
fn pih_body_offset(buf: &[u8]) -> Result<usize> {
    if buf.len() < 4 || buf[0] != 0xff || buf[1] != 0x10 {
        return Err(Error::invalid(
            "jpegxs signalling: buffer does not start with SOC (FF10)",
        ));
    }
    let mut pos = 2usize;
    loop {
        if pos + 2 > buf.len() {
            return Err(Error::invalid(
                "jpegxs signalling: marker chain ended before a PIH marker",
            ));
        }
        if buf[pos] != 0xff {
            return Err(Error::invalid(format!(
                "jpegxs signalling: expected a marker at offset {pos}, got 0x{:02X}",
                buf[pos]
            )));
        }
        match buf[pos + 1] {
            // PIH — body starts after the marker + Lpih length field.
            0x12 => {
                if pos + 4 > buf.len() {
                    return Err(Error::invalid(
                        "jpegxs signalling: PIH truncated before its length field",
                    ));
                }
                let lpih = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
                if lpih != crate::picture_header::PIH_LPIH as usize || pos + 2 + lpih > buf.len() {
                    return Err(Error::invalid(format!(
                        "jpegxs signalling: PIH Lpih={lpih} is not the Table A.7 fixed \
                         length of {} or overruns the buffer",
                        crate::picture_header::PIH_LPIH
                    )));
                }
                return Ok(pos + 4);
            }
            // SLH or EOC before any PIH — the stream carries no picture
            // header to patch.
            0x20 | 0x11 => {
                return Err(Error::invalid(
                    "jpegxs signalling: reached SLH/EOC before a PIH marker",
                ));
            }
            _ => {
                // Every other pre-slice marker (CAP, CDT, WGT, NLT, CWD,
                // CTS, CRG, COM) carries a two-byte length that counts
                // itself.
                if pos + 4 > buf.len() {
                    return Err(Error::invalid(
                        "jpegxs signalling: marker segment truncated before its length field",
                    ));
                }
                let len = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]) as usize;
                if len < 2 || pos + 2 + len > buf.len() {
                    return Err(Error::invalid(format!(
                        "jpegxs signalling: marker 0xFF{:02X} at offset {pos} has invalid \
                         segment length {len}",
                        buf[pos + 1]
                    )));
                }
                pos += 2 + len;
            }
        }
    }
}

/// Re-parse `buf` and run the exact conformance gates the decoder
/// applies to the declarations in its picture header:
///
/// * `Ppih` maps to a known profile and the stream satisfies every
///   observable constraint of that profile
///   ([`check_codestream`], 21122-2 Tables A.1–A.3);
/// * `Plev` decodes to a non-reserved level and the picture fits its
///   `Wmax` / `Hmax` / `Lmax` bounds ([`check_level`], Table A.6);
/// * the SOC-to-EOC byte count fits the declared sublevel's `Ssl,max`
///   bound ([`check_codestream_size`], §A.4.1), including the §A.4.2
///   "Full sublevel requires a non-unrestricted profile" rule;
/// * a non-zero `Lcod` equals the actual SOC-to-EOC byte count
///   (21122-1 Table 11).
///
/// A stream that passes this function decodes through
/// [`crate::decode_jpeg_xs`] without tripping any declaration gate —
/// they are the same checks.
pub fn verify_declarations(buf: &[u8]) -> Result<()> {
    let cs = codestream::parse(buf)?;
    match Profile::from_ppih(cs.pih.ppih) {
        Some(profile) => check_codestream(&cs, profile)?,
        None => {
            return Err(Error::invalid(format!(
                "jpegxs signalling: Ppih=0x{:04X} is reserved for ISO/IEC use (Table A.5)",
                cs.pih.ppih
            )));
        }
    }
    check_level(&cs)?;
    if Sublevel::from_plev_low(cs.pih.plev).is_none() {
        return Err(Error::invalid(format!(
            "jpegxs signalling: Plev low byte 0x{:02X} is reserved for ISO/IEC use (Table A.13)",
            cs.pih.plev & 0xff
        )));
    }
    check_codestream_size(&cs, buf.len())?;
    if cs.pih.lcod != 0 {
        let Some(eoc) = cs.eoc_offset else {
            return Err(Error::invalid(
                "jpegxs signalling: Lcod is non-zero but the stream carries no EOC",
            ));
        };
        let actual = eoc + 2;
        if cs.pih.lcod as usize != actual {
            return Err(Error::invalid(format!(
                "jpegxs signalling: Lcod={} does not match the actual SOC-to-EOC length \
                 of {actual} bytes (Table 11)",
                cs.pih.lcod
            )));
        }
    }
    Ok(())
}

/// Patch `bytes` into `buf[offset..offset + bytes.len()]`, run
/// [`verify_declarations`] on the result, and restore the previous
/// bytes (returning the verification error) if the new declaration does
/// not hold. On success the patch is kept.
fn patch_and_verify(buf: &mut [u8], offset: usize, bytes: &[u8]) -> Result<()> {
    let end = offset + bytes.len();
    let mut saved = [0u8; 4];
    let saved = &mut saved[..bytes.len()];
    saved.copy_from_slice(&buf[offset..end]);
    buf[offset..end].copy_from_slice(bytes);
    if let Err(e) = verify_declarations(buf) {
        buf[offset..end].copy_from_slice(saved);
        return Err(e);
    }
    Ok(())
}

/// Declare `profile` in the codestream's `Ppih` field (21122-2 Table
/// A.5), verifying first that the stream actually satisfies every
/// observable constraint of that profile (Tables A.1–A.3 via
/// [`check_codestream`]). On failure the buffer is unchanged and the
/// error names the violated constraint.
///
/// `Profile::Unrestricted` writes `Ppih = 0` (withdrawing any earlier
/// claim); note §A.2.2 — the unrestricted profile is not a conformance
/// point.
pub fn declare_profile(buf: &mut [u8], profile: Profile) -> Result<()> {
    let body = pih_body_offset(buf)?;
    patch_and_verify(buf, body + PPIH_REL, &profile.ppih().to_be_bytes())
}

/// Declare `level` and `sublevel` in the codestream's `Plev` field
/// (21122-2 Tables A.12 / A.13), verifying that the picture fits the
/// level's `Wmax` / `Hmax` / `Lmax` bounds and the SOC-to-EOC byte
/// count fits the sublevel's `Ssl,max` bound (§A.4.1). The §A.4.2 rule
/// (`Full` sublevel requires a non-unrestricted profile) is enforced —
/// declare the profile first. On failure the buffer is unchanged.
pub fn declare_level_sublevel(buf: &mut [u8], level: Level, sublevel: Sublevel) -> Result<()> {
    let body = pih_body_offset(buf)?;
    let plev = ((level.plev_high_byte() as u16) << 8) | sublevel.plev_low_byte() as u16;
    patch_and_verify(buf, body + PLEV_REL, &plev.to_be_bytes())
}

/// Declare constant-bitrate coding: write the actual SOC-to-EOC byte
/// count into `Lcod` (21122-1 Table 11 — "size of the entire codestream
/// in bytes ... if constant-bitrate coding is used"). Errors if the
/// buffer does not end exactly at its EOC marker (a trailing-garbage
/// buffer has no well-defined codestream length to declare) or exceeds
/// the 32-bit `Lcod` range. On failure the buffer is unchanged.
pub fn declare_cbr(buf: &mut [u8]) -> Result<()> {
    let body = pih_body_offset(buf)?;
    let n = buf.len();
    if n < 4 || buf[n - 2] != 0xff || buf[n - 1] != 0x11 {
        return Err(Error::invalid(
            "jpegxs signalling: declare_cbr requires the buffer to end at its EOC marker",
        ));
    }
    let lcod = u32::try_from(n).map_err(|_| {
        Error::invalid("jpegxs signalling: codestream length exceeds the 32-bit Lcod range")
    })?;
    patch_and_verify(buf, body + LCOD_REL, &lcod.to_be_bytes())
}

/// Withdraw a CBR declaration: write `Lcod = 0` (variable-bitrate
/// coding per 21122-1 Table 11).
pub fn declare_vbr(buf: &mut [u8]) -> Result<()> {
    let body = pih_body_offset(buf)?;
    patch_and_verify(buf, body + LCOD_REL, &0u32.to_be_bytes())
}

/// The eight non-unrestricted profiles in preference order for
/// [`pick_profile`]: the Light family (smallest decoder smoothing
/// buffer, Table A.2), then Main (Table A.1), then High (Table A.3),
/// each family ordered by ascending tool set / bit-depth reach.
const PROFILE_PREFERENCE: [Profile; 8] = [
    Profile::Light422_10,
    Profile::LightSubline422_10,
    Profile::Light444_12,
    Profile::Main422_10,
    Profile::Main444_12,
    Profile::Main4444_12,
    Profile::High444_12,
    Profile::High4444_12,
];

/// Pick the first profile (in [`PROFILE_PREFERENCE`] order — Light
/// families first, then Main, then High) whose full constraint set the
/// stream satisfies, falling back to [`Profile::Unrestricted`] when no
/// listed profile admits the stream (e.g. 4:2:0 chroma or a slice
/// height other than 16 image rows, which no 21122-2:2019 profile
/// permits).
pub fn pick_profile(buf: &[u8]) -> Result<Profile> {
    let cs = codestream::parse(buf)?;
    for profile in PROFILE_PREFERENCE {
        if check_codestream(&cs, profile).is_ok() {
            return Ok(profile);
        }
    }
    Ok(Profile::Unrestricted)
}

/// Pick the smallest level (Table A.6, ascending `Wmax` / `Hmax` /
/// `Lmax`) that admits a `width × height` picture. Levels that differ
/// only in maximum sample *rate* (4k-2 vs 4k-3, 8k-2 vs 8k-3) are not
/// distinguishable from a single picture, so the lower-rate member is
/// chosen. Returns [`Level::Unrestricted`] when the picture exceeds
/// even 10k-1.
pub fn pick_level(width: u32, height: u32) -> Level {
    let samples = width as u64 * height as u64;
    for level in [
        Level::L2k1,
        Level::L4k1,
        Level::L4k2,
        Level::L8k1,
        Level::L8k2,
        Level::L10k1,
    ] {
        // Every candidate level is bounded (only `Unrestricted` returns
        // `None`), so the MAX fallbacks never fire; they just keep the
        // comparison total.
        let w_ok = width <= level.max_width().unwrap_or(u32::MAX);
        let h_ok = height <= level.max_height().unwrap_or(u32::MAX);
        let l_ok = samples <= level.max_samples().unwrap_or(u64::MAX);
        if w_ok && h_ok && l_ok {
            return level;
        }
    }
    Level::Unrestricted
}

/// Pick the smallest sublevel (Table A.7, ascending nominal bpp — 3, 6,
/// 9, 12, then `Full`) whose `Ssl,max = ⌊Lmax × Nbpp / 8⌋` bound
/// (§A.4.1) admits a codestream of `codestream_len` bytes at `level`.
/// `Full` is only considered for a non-unrestricted `profile` (§A.4.2).
/// Returns [`Sublevel::Unrestricted`] when `level` is unrestricted
/// (no `Lmax`, so no sublevel bound is expressible) or the stream
/// exceeds every bounded sublevel.
pub fn pick_sublevel(codestream_len: usize, level: Level, profile: Profile) -> Sublevel {
    if matches!(level, Level::Unrestricted) {
        return Sublevel::Unrestricted;
    }
    let mut candidates = vec![
        Sublevel::Sublev3bpp,
        Sublevel::Sublev6bpp,
        Sublevel::Sublev9bpp,
        Sublevel::Sublev12bpp,
    ];
    if !matches!(profile, Profile::Unrestricted) {
        candidates.push(Sublevel::Full);
    }
    for sub in candidates {
        if let Some(max) = max_codestream_size(level, sub, profile) {
            if codestream_len as u64 <= max {
                return sub;
            }
        }
    }
    Sublevel::Unrestricted
}

/// Declare the tightest verified profile / level / sublevel the stream
/// satisfies, plus (when `cbr` is set) the CBR `Lcod` self-description.
/// Returns the chosen triple. The buffer is patched in place; on error
/// it is left with whichever declarations had already verified (each
/// individual declaration is atomic).
pub fn declare_auto(buf: &mut [u8], cbr: bool) -> Result<(Profile, Level, Sublevel)> {
    let profile = pick_profile(buf)?;
    declare_profile(buf, profile)?;
    let cs = codestream::parse(buf)?;
    let level = pick_level(cs.pih.width(), cs.pih.height());
    let sublevel = pick_sublevel(buf.len(), level, profile);
    declare_level_sublevel(buf, level, sublevel)?;
    if cbr {
        declare_cbr(buf)?;
    }
    Ok((profile, level, sublevel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder;
    use crate::encoder;

    /// Deterministic 8-bit test plane.
    fn plane(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h);
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(97);
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push(((x * 3 + y * 5) as u32 ^ (s >> 24)) as u8);
            }
        }
        v
    }

    /// A profile-shaped stream: 3-component 4:4:4, 8-bit, NL = 5/1,
    /// Cw = 0, Qpih = 0, slice height 16 image rows (Hsl = 8 precinct
    /// rows at NL,y = 1), lossless.
    fn encode_profile_shaped() -> Vec<u8> {
        let w = 64usize;
        let h = 64usize;
        let planes = vec![plane(w, h, 1), plane(w, h, 2), plane(w, h, 3)];
        encoder::encode_planar_hsl(w as u16, h as u16, 3, 1, 5, 1, 0, 8, &planes)
            .expect("profile-shaped encode")
    }

    fn decode_ok(buf: &[u8]) -> crate::image::JpegXsImage {
        decoder::decode_codestream(buf, None).expect("decode")
    }

    #[test]
    fn pih_body_offset_finds_lcod_ppih_plev() {
        let buf = encode_profile_shaped();
        let body = pih_body_offset(&buf).expect("pih body");
        // The encoder writes Lcod = 0, Ppih = 0, Plev = 0.
        assert_eq!(&buf[body..body + 8], &[0u8; 8]);
        // And the PIH marker + Lpih immediately precede the body.
        assert_eq!(&buf[body - 4..body], &[0xff, 0x12, 0x00, 26]);
    }

    #[test]
    fn pih_body_offset_rejects_non_soc() {
        assert!(pih_body_offset(&[0x00, 0x01, 0x02, 0x03]).is_err());
        // EOC-only stream: no PIH.
        assert!(pih_body_offset(&[0xff, 0x10, 0xff, 0x11]).is_err());
    }

    #[test]
    fn declare_profile_accepts_satisfied_claim_and_decodes() {
        let mut buf = encode_profile_shaped();
        let baseline = decode_ok(&buf);
        declare_profile(&mut buf, Profile::Main444_12).expect("Main 444.12 claim holds");
        let body = pih_body_offset(&buf).unwrap();
        assert_eq!(
            u16::from_be_bytes([buf[body + 4], buf[body + 5]]),
            Profile::Main444_12.ppih()
        );
        // The decoder's own profile gate accepts the claimed stream and
        // reconstruction is unchanged.
        let img = decode_ok(&buf);
        for (a, b) in img.planes.iter().zip(baseline.planes.iter()) {
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn declare_profile_rejects_false_claim_and_restores() {
        // The stream is 4:4:4 — Light 422.10 does not permit that
        // chroma format (Table A.2), so the claim must be refused and
        // the buffer left byte-identical.
        let mut buf = encode_profile_shaped();
        let before = buf.clone();
        let err = declare_profile(&mut buf, Profile::Light422_10).unwrap_err();
        assert!(
            format!("{err}").contains("chroma"),
            "expected chroma-format rejection, got {err}"
        );
        assert_eq!(buf, before, "failed claim must leave the stream unchanged");
    }

    #[test]
    fn declare_profile_rejects_wrong_slice_height() {
        // Same feature set but a single whole-picture slice: every
        // 21122-2 profile mandates 16-image-row slices, so the claim
        // must fail on Hsl.
        let w = 64usize;
        let h = 64usize;
        let planes = vec![plane(w, h, 1), plane(w, h, 2), plane(w, h, 3)];
        let mut buf =
            encoder::encode_planar(w as u16, h as u16, 3, 1, 5, 1, &planes).expect("encode");
        let err = declare_profile(&mut buf, Profile::Main444_12).unwrap_err();
        assert!(
            format!("{err}").contains("image rows"),
            "expected slice-height rejection, got {err}"
        );
    }

    #[test]
    fn declare_level_sublevel_and_cbr_roundtrip() {
        let mut buf = encode_profile_shaped();
        declare_profile(&mut buf, Profile::Main444_12).expect("profile");
        declare_level_sublevel(&mut buf, Level::L2k1, Sublevel::Full).expect("level");
        declare_cbr(&mut buf).expect("cbr");
        let cs = codestream::parse(&buf).unwrap();
        assert_eq!(cs.pih.plev, 0x1080);
        assert_eq!(cs.pih.lcod as usize, buf.len());
        // The fully-declared stream passes the decoder's gates.
        decode_ok(&buf);
        // And a later re-encode-free withdrawal back to VBR also holds.
        declare_vbr(&mut buf).expect("vbr");
        assert_eq!(codestream::parse(&buf).unwrap().pih.lcod, 0);
    }

    #[test]
    fn declare_level_rejects_too_small_level() {
        // A 64×64 picture fits 2k-1; pretend-clamp: patch a picture
        // larger than 2k-1's Wmax and check the level claim fails.
        let w = 4096usize;
        // One tall-enough row of precincts is unnecessary — use a
        // small height to keep the test fast.
        let h = 32usize;
        let planes = vec![plane(w, h, 1)];
        let mut buf = encoder::encode_planar_hsl(w as u16, h as u16, 1, 0, 5, 1, 0, 8, &planes)
            .expect("wide encode");
        let before = buf.clone();
        let err =
            declare_level_sublevel(&mut buf, Level::L2k1, Sublevel::Unrestricted).unwrap_err();
        assert!(
            format!("{err}").contains("Wmax"),
            "expected Wmax rejection, got {err}"
        );
        assert_eq!(buf, before);
        // 4k-1 admits it.
        declare_level_sublevel(&mut buf, Level::L4k1, Sublevel::Unrestricted).expect("4k-1");
    }

    #[test]
    fn full_sublevel_requires_profile() {
        // §A.4.2 — Full sublevel with Ppih = 0 is rejected.
        let mut buf = encode_profile_shaped();
        let err = declare_level_sublevel(&mut buf, Level::L2k1, Sublevel::Full).unwrap_err();
        assert!(
            format!("{err}").contains("Full sublevel"),
            "expected Full-sublevel rejection, got {err}"
        );
        declare_profile(&mut buf, Profile::Main444_12).expect("profile");
        declare_level_sublevel(&mut buf, Level::L2k1, Sublevel::Full).expect("Full now legal");
    }

    #[test]
    fn declare_cbr_rejects_trailing_garbage() {
        let mut buf = encode_profile_shaped();
        buf.push(0x00);
        assert!(declare_cbr(&mut buf).is_err());
    }

    #[test]
    fn pick_level_matches_table_a6_bounds() {
        assert_eq!(pick_level(2048, 2048), Level::L2k1);
        // 2048 wide but too many samples for 2k-1 (Lmax = 4194304).
        assert_eq!(pick_level(2048, 4096), Level::L4k1);
        assert_eq!(pick_level(4096, 2176), Level::L4k1);
        // 4096×4096 = 16777216 > 8912896 (4k-1) but fits 4k-2.
        assert_eq!(pick_level(4096, 4096), Level::L4k2);
        assert_eq!(pick_level(8192, 4352), Level::L8k1);
        assert_eq!(pick_level(8192, 8192), Level::L8k2);
        assert_eq!(pick_level(10240, 10240), Level::L10k1);
        assert_eq!(pick_level(10241, 16), Level::Unrestricted);
    }

    #[test]
    fn pick_sublevel_ascends_the_bpp_ladder() {
        // 2k-1 Lmax = 4194304 → Ssl,max(3bpp) = 1572864 bytes.
        assert_eq!(
            pick_sublevel(1_572_864, Level::L2k1, Profile::Unrestricted),
            Sublevel::Sublev3bpp
        );
        assert_eq!(
            pick_sublevel(1_572_865, Level::L2k1, Profile::Unrestricted),
            Sublevel::Sublev6bpp
        );
        // Beyond 12 bpp with no profile → no Full available.
        assert_eq!(
            pick_sublevel(6_291_457, Level::L2k1, Profile::Unrestricted),
            Sublevel::Unrestricted
        );
        // With a profile, Full (Main 422.10 → Nbpp = 20) picks up the
        // tail beyond Sublev12bpp.
        assert_eq!(
            pick_sublevel(6_291_457, Level::L2k1, Profile::Main422_10),
            Sublevel::Full
        );
        assert_eq!(
            pick_sublevel(usize::MAX, Level::L2k1, Profile::Main422_10),
            Sublevel::Unrestricted
        );
        assert_eq!(
            pick_sublevel(usize::MAX, Level::Unrestricted, Profile::Main422_10),
            Sublevel::Unrestricted
        );
    }

    #[test]
    fn declare_auto_signs_a_profile_shaped_stream() {
        let mut buf = encode_profile_shaped();
        let baseline = decode_ok(&buf);
        let (profile, level, sublevel) = declare_auto(&mut buf, true).expect("auto declare");
        // 4:4:4 8-bit NL,y=1 Qpih=0 Cw=0 slice=16 rows → Light 444.12
        // is the first preference-order fit.
        assert_eq!(profile, Profile::Light444_12);
        assert_eq!(level, Level::L2k1);
        assert_eq!(sublevel, Sublevel::Sublev3bpp);
        let cs = codestream::parse(&buf).unwrap();
        assert_eq!(cs.pih.ppih, Profile::Light444_12.ppih());
        assert_eq!(cs.pih.lcod as usize, buf.len());
        let img = decode_ok(&buf);
        for (a, b) in img.planes.iter().zip(baseline.planes.iter()) {
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn declare_auto_falls_back_to_unrestricted() {
        // 4:2:0 chroma is not a 21122-2:2019 profile chroma format —
        // declare_auto must claim nothing rather than something false.
        let w = 64usize;
        let h = 64usize;
        let planes = vec![
            plane(w, h, 1),
            plane(w / 2, h / 2, 2),
            plane(w / 2, h / 2, 3),
        ];
        let mut buf = encoder::encode_planar_subsampled(
            w as u16,
            h as u16,
            3,
            0,
            3,
            1,
            0,
            &[1, 2, 2],
            &[1, 2, 2],
            &planes,
        )
        .expect("4:2:0 encode");
        let (profile, _, _) = declare_auto(&mut buf, false).expect("auto declare");
        assert_eq!(profile, Profile::Unrestricted);
        decode_ok(&buf);
    }

    #[test]
    fn verify_declarations_accepts_every_legacy_stream() {
        // The all-zero (VBR / unrestricted) defaults every encoder
        // entry point emits are themselves a verifiable declaration.
        let buf = encode_profile_shaped();
        verify_declarations(&buf).expect("legacy defaults verify");
    }
}
