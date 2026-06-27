//! Profiles, levels and sublevels — ISO/IEC 21122-2:2019 Annex A.
//!
//! Part 2 of JPEG XS defines a limited set of decoder *profiles* that
//! constrain the syntax + parameter values an encoder may use, plus
//! *levels* (decoded-domain bounds: maximum width / height / total
//! sample count / sample rate) and *sublevels* (coded-domain bounds:
//! nominal bits-per-pixel). A codestream advertises its profile in the
//! `Ppih` field of the PIH (Table A.5) and its level + sublevel in the
//! `Plev` field (Tables A.12 + A.13).
//!
//! This module:
//!
//! * Maps `Ppih` u16 ↔ [`Profile`] enum per Table A.5
//!   ([`Profile::from_ppih`] / [`Profile::ppih`]).
//! * Maps `Plev` u16 → ([`Level`], [`Sublevel`]) per Tables A.12 + A.13
//!   ([`Level::from_plev_high`] / [`Sublevel::from_plev_low`]).
//! * Encodes the per-profile constraints from Tables A.1 / A.2 / A.3 as
//!   a [`ProfileLimits`] struct, exposed by [`Profile::limits`].
//! * Checks a parsed [`crate::codestream::Codestream`] against a profile
//!   ([`check_codestream`]), enforcing every constraint listed in the
//!   profile tables that is observable from the picture-header / CDT /
//!   capability bits. Buffer-model bounds (Annexes B / C / D) are out
//!   of scope for this module — they require a transmission-channel
//!   model that is computed by the host application, not the codec
//!   crate.
//!
//! Table A.5 maps the *coded-tool subset* / chroma / bit-depth / Nsbu
//! into one of eight profile values plus *Unrestricted* (`Ppih == 0`,
//! which "shall not be considered a conformance point" per §A.2.2).
//! Conformance checks against `Profile::Unrestricted` therefore always
//! succeed.

use crate::codestream::Codestream;
use crate::component_table::Component;
use crate::error::{JpegXsError as Error, Result};

/// The eight conformance profiles defined in ISO/IEC 21122-2:2019
/// Tables A.1, A.2, A.3 plus the *Unrestricted* (`Ppih == 0`) escape.
///
/// `Ppih` bit layout per NOTE 2 of §A.2.2 (informative): the high 4
/// bits identify the coding-tool set, the next 2 bits the chroma
/// format, the next 2 bits the max bit-depth, and the next 2 bits the
/// `Nsbu` smoothing-buffer-units encoding. The rest is reserved. The
/// hex values in [`Profile::ppih`] are the *normative* assignments
/// (Table A.5); the bit-field breakdown is informative only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// `Ppih == 0x0000`. No tool / parameter restrictions; §A.2.2 says
    /// "not a conformance point". Acts as a permissive default.
    Unrestricted,
    /// Table A.2 Light 422.10 — 4:0:0 / 4:2:2, bit-depth ∈ {8, 10}.
    Light422_10,
    /// Table A.2 Light 444.12 — 4:0:0 / 4:2:2 / 4:4:4, bit-depth ∈
    /// {8, 10, 12}.
    Light444_12,
    /// Table A.2 Light-Subline 422.10 — Light 422.10 with `NL,x ∈ [0..5]`,
    /// `NL,y = 0`, max column width `Cs ≤ 2048`.
    LightSubline422_10,
    /// Table A.1 Main 422.10 — 4:0:0 / 4:2:2, bit-depth ∈ {8, 10}.
    Main422_10,
    /// Table A.1 Main 444.12 — adds 4:4:4 and 12-bit support.
    Main444_12,
    /// Table A.1 Main 4444.12 — adds 4-component chroma (4:2:2:4, 4:4:4:4).
    Main4444_12,
    /// Table A.3 High 444.12 — Main 444.12 with `NL,y` up to 2.
    High444_12,
    /// Table A.3 High 4444.12 — Main 4444.12 with `NL,y` up to 2.
    High4444_12,
}

impl Profile {
    /// Returns the `Ppih` 16-bit value defined in Table A.5 for this
    /// profile, or `0x0000` for [`Profile::Unrestricted`].
    pub fn ppih(self) -> u16 {
        match self {
            Profile::Unrestricted => 0x0000,
            Profile::Light422_10 => 0x1500,
            Profile::Light444_12 => 0x1A00,
            Profile::LightSubline422_10 => 0x2500,
            Profile::Main422_10 => 0x3540,
            Profile::Main444_12 => 0x3A40,
            Profile::Main4444_12 => 0x3E40,
            Profile::High444_12 => 0x4A40,
            Profile::High4444_12 => 0x4E40,
        }
    }

    /// Parses a `Ppih` 16-bit value into a [`Profile`] per Table A.5.
    /// Values not listed in the table (Table A.5 footer: "all other
    /// values" → *Reserved for ISO/IEC purposes*) yield `None`.
    pub fn from_ppih(ppih: u16) -> Option<Profile> {
        Some(match ppih {
            0x0000 => Profile::Unrestricted,
            0x1500 => Profile::Light422_10,
            0x1A00 => Profile::Light444_12,
            0x2500 => Profile::LightSubline422_10,
            0x3540 => Profile::Main422_10,
            0x3A40 => Profile::Main444_12,
            0x3E40 => Profile::Main4444_12,
            0x4A40 => Profile::High444_12,
            0x4E40 => Profile::High4444_12,
            _ => return None,
        })
    }

    /// Short human-readable profile name (e.g. `"Main 422.10"`).
    pub fn name(self) -> &'static str {
        match self {
            Profile::Unrestricted => "Unrestricted",
            Profile::Light422_10 => "Light 422.10",
            Profile::Light444_12 => "Light 444.12",
            Profile::LightSubline422_10 => "Light-Subline 422.10",
            Profile::Main422_10 => "Main 422.10",
            Profile::Main444_12 => "Main 444.12",
            Profile::Main4444_12 => "Main 4444.12",
            Profile::High444_12 => "High 444.12",
            Profile::High4444_12 => "High 4444.12",
        }
    }

    /// Returns the table-encoded constraint set for this profile.
    /// [`Profile::Unrestricted`] returns [`ProfileLimits::unrestricted`].
    pub fn limits(self) -> ProfileLimits {
        match self {
            Profile::Unrestricted => ProfileLimits::unrestricted(),
            // Table A.2 row 1 (Light 422.10).
            Profile::Light422_10 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10],
                allowed_chroma: &[ChromaFormat::Mono, ChromaFormat::Yuv422],
                max_nly: 1,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOnly,
                column_mode: ColumnMode::FullWidthOnly,
                max_column_width: None,
                slice_height: 16,
                max_components: 3,
            },
            // Table A.2 row 2 (Light 444.12).
            Profile::Light444_12 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10, 12],
                allowed_chroma: &[
                    ChromaFormat::Mono,
                    ChromaFormat::Yuv422,
                    ChromaFormat::Yuv444,
                ],
                max_nly: 1,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOnly,
                column_mode: ColumnMode::FullWidthOnly,
                max_column_width: None,
                slice_height: 16,
                max_components: 3,
            },
            // Table A.2 row 3 (Light-Subline 422.10) — column width
            // capped at 2048 (formula A.2 special case for this
            // profile only). `NL,x` may be 0 (the subline path), and
            // `Cw == 0` (full-image columns) is also permitted up to
            // 2048 sampling-grid points.
            Profile::LightSubline422_10 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10],
                allowed_chroma: &[ChromaFormat::Mono, ChromaFormat::Yuv422],
                max_nly: 0,
                nlx_range: (0, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::ColumnCapped,
                max_column_width: Some(2048),
                slice_height: 16,
                max_components: 3,
            },
            // Table A.1 row 1 (Main 422.10).
            Profile::Main422_10 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10],
                allowed_chroma: &[ChromaFormat::Mono, ChromaFormat::Yuv422],
                max_nly: 1,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::SingleColumnUnlessNlyZero,
                max_column_width: None,
                slice_height: 16,
                max_components: 3,
            },
            // Table A.1 row 2 (Main 444.12).
            Profile::Main444_12 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10, 12],
                allowed_chroma: &[
                    ChromaFormat::Mono,
                    ChromaFormat::Yuv422,
                    ChromaFormat::Yuv444,
                ],
                max_nly: 1,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::SingleColumnUnlessNlyZero,
                max_column_width: None,
                slice_height: 16,
                max_components: 3,
            },
            // Table A.1 row 3 (Main 4444.12) — same as Main 444.12
            // plus 4-component planar (4:2:2:4 or 4:4:4:4).
            Profile::Main4444_12 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10, 12],
                allowed_chroma: &[
                    ChromaFormat::Mono,
                    ChromaFormat::Yuv422,
                    ChromaFormat::Yuv444,
                    ChromaFormat::Yuv4224,
                    ChromaFormat::Yuv4444,
                ],
                max_nly: 1,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::SingleColumnUnlessNlyZero,
                max_column_width: None,
                slice_height: 16,
                max_components: 4,
            },
            // Table A.3 row 1 (High 444.12) — same as Main 444.12 with
            // `NL,y` raised to 2.
            Profile::High444_12 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10, 12],
                allowed_chroma: &[
                    ChromaFormat::Mono,
                    ChromaFormat::Yuv422,
                    ChromaFormat::Yuv444,
                ],
                max_nly: 2,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::SingleColumnUnlessNlyZero,
                max_column_width: None,
                slice_height: 16,
                max_components: 3,
            },
            // Table A.3 row 2 (High 4444.12).
            Profile::High4444_12 => ProfileLimits {
                profile: self,
                allowed_bit_depths: &[8, 10, 12],
                allowed_chroma: &[
                    ChromaFormat::Mono,
                    ChromaFormat::Yuv422,
                    ChromaFormat::Yuv444,
                    ChromaFormat::Yuv4224,
                    ChromaFormat::Yuv4444,
                ],
                max_nly: 2,
                nlx_range: (1, 5),
                allowed_qpih: QpihAllowed::DeadzoneOrUniform,
                column_mode: ColumnMode::SingleColumnUnlessNlyZero,
                max_column_width: None,
                slice_height: 16,
                max_components: 4,
            },
        }
    }
}

/// Levels per Table A.6 — define an upper bound on the decoded-domain
/// width, height, total sampling-grid points per image, and sample
/// rate. Signalled in the high 8 bits of `Plev` (Table A.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    /// Unrestricted level (no decoded-domain bounds; §A.5 — "shall not
    /// be considered as a conformance point").
    Unrestricted,
    /// 2k-1: `Wmax = 2048`, `Hmax = 8192`, `Lmax = 4 194 304`.
    L2k1,
    /// 4k-1: `Wmax = 4096`, `Hmax = 16 384`, `Lmax = 8 912 896`.
    L4k1,
    /// 4k-2: `Wmax = 4096`, `Hmax = 16 384`, `Lmax = 16 777 216`.
    L4k2,
    /// 4k-3: `Wmax = 4096`, same `Lmax` as 4k-2, higher sample-rate.
    L4k3,
    /// 8k-1: `Wmax = 8192`, `Hmax = 32 768`, `Lmax = 35 651 584`.
    L8k1,
    /// 8k-2: `Wmax = 8192`, `Lmax = 67 108 864`.
    L8k2,
    /// 8k-3: 8k-2 limits with the highest sample-rate.
    L8k3,
    /// 10k-1: `Wmax = 10 240`, `Hmax = 40 960`, `Lmax = 104 857 600`.
    L10k1,
}

impl Level {
    /// Decode the high byte of `Plev` (Table A.12). Returns `None` for
    /// values marked *Reserved for ISO/IEC purposes*. Only the top 4
    /// bits of the high byte are checked per the `XXXX XXXX` mask in
    /// the table.
    pub fn from_plev_high(plev: u16) -> Option<Level> {
        // The high byte is `Plev >> 8`; only its high nibble carries
        // the level family, low nibble is the family index.
        Some(match plev >> 8 {
            0x00 => Level::Unrestricted,
            0x10 => Level::L2k1,
            0x20 => Level::L4k1,
            0x24 => Level::L4k2,
            0x28 => Level::L4k3,
            0x30 => Level::L8k1,
            0x34 => Level::L8k2,
            0x38 => Level::L8k3,
            0x40 => Level::L10k1,
            _ => return None,
        })
    }

    /// Maximum picture width `Wmax` in sampling-grid points
    /// (`None` for Unrestricted).
    pub fn max_width(self) -> Option<u32> {
        Some(match self {
            Level::Unrestricted => return None,
            Level::L2k1 => 2048,
            Level::L4k1 | Level::L4k2 | Level::L4k3 => 4096,
            Level::L8k1 | Level::L8k2 | Level::L8k3 => 8192,
            Level::L10k1 => 10240,
        })
    }

    /// Maximum picture height `Hmax` in sampling-grid points
    /// (`None` for Unrestricted).
    pub fn max_height(self) -> Option<u32> {
        Some(match self {
            Level::Unrestricted => return None,
            Level::L2k1 => 8192,
            Level::L4k1 | Level::L4k2 | Level::L4k3 => 16384,
            Level::L8k1 | Level::L8k2 | Level::L8k3 => 32768,
            Level::L10k1 => 40960,
        })
    }

    /// Maximum number of sampling grid points `Lmax` per image
    /// (`None` for Unrestricted).
    pub fn max_samples(self) -> Option<u64> {
        Some(match self {
            Level::Unrestricted => return None,
            Level::L2k1 => 4_194_304,
            Level::L4k1 => 8_912_896,
            Level::L4k2 | Level::L4k3 => 16_777_216,
            Level::L8k1 => 35_651_584,
            Level::L8k2 | Level::L8k3 => 67_108_864,
            Level::L10k1 => 104_857_600,
        })
    }
}

/// Sublevels per Table A.7 — define an upper bound on the coded-domain
/// nominal bits-per-pixel. Signalled in the low byte of `Plev` (Table
/// A.13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sublevel {
    /// Unrestricted sublevel (no coded-domain bound).
    Unrestricted,
    /// `Full` — uses each profile's max-decoded-bpp from Table A.4.
    Full,
    /// 12 bpp.
    Sublev12bpp,
    /// 9 bpp.
    Sublev9bpp,
    /// 6 bpp.
    Sublev6bpp,
    /// 3 bpp.
    Sublev3bpp,
}

impl Sublevel {
    /// Decode the low byte of `Plev` (Table A.13). The high 4 bits of
    /// `plev` are `XXXX XXXX` (level/don't-care); the low byte carries
    /// the sublevel id. Convenience wrapper around
    /// [`Sublevel::from_plev_low_byte`].
    pub fn from_plev_low(plev: u16) -> Option<Sublevel> {
        Sublevel::from_plev_low_byte((plev & 0x00ff) as u8)
    }

    /// Decode the `Plev` low byte directly per Table A.13. The five
    /// non-reserved rows are:
    ///
    /// * `0000 0000` — Unrestricted
    /// * `1000 0000` — Full
    /// * `0001 0000` — Sublev12bpp
    /// * `0000 1100` — Sublev9bpp
    /// * `0000 1000` — Sublev6bpp
    /// * `0000 0100` — Sublev3bpp
    ///
    /// Any other byte value is *Reserved for ISO/IEC purposes* and
    /// yields `None`.
    pub fn from_plev_low_byte(low: u8) -> Option<Sublevel> {
        // Sublev12bpp explicitly has the next-nibble bit `0001`.
        if low == 0x10 {
            return Some(Sublevel::Sublev12bpp);
        }
        if low == 0x80 {
            return Some(Sublevel::Full);
        }
        if low & 0x0f == 0x4 {
            return Some(Sublevel::Sublev3bpp);
        }
        if low & 0x0f == 0x8 {
            return Some(Sublevel::Sublev6bpp);
        }
        if low & 0x0f == 0xc {
            return Some(Sublevel::Sublev9bpp);
        }
        if low == 0x00 {
            return Some(Sublevel::Unrestricted);
        }
        None
    }

    /// Nominal bits-per-pixel `Nbpp` per Table A.7. `Full` returns
    /// `None` because its value is the profile's max-decoded-bpp from
    /// Table A.4 — see [`Profile::max_decoded_bpp`].
    pub fn nominal_bpp(self) -> Option<u32> {
        Some(match self {
            Sublevel::Unrestricted | Sublevel::Full => return None,
            Sublevel::Sublev12bpp => 12,
            Sublevel::Sublev9bpp => 9,
            Sublevel::Sublev6bpp => 6,
            Sublevel::Sublev3bpp => 3,
        })
    }

    /// Resolve the effective nominal bits-per-pixel `Nbpp` for this
    /// sublevel, consulting `profile` only for the `Full` sublevel
    /// (whose `Nbpp` is the profile's max-decoded-bpp from Table A.4).
    ///
    /// Returns `None` for the `Unrestricted` sublevel (no coded-domain
    /// bound), and `None` for `Full` with an `Unrestricted` profile
    /// (which Table A.7 forbids — `Full` "shall only be used if the
    /// profile value is not unrestricted").
    pub fn effective_nbpp(self, profile: Profile) -> Option<u32> {
        match self {
            Sublevel::Unrestricted => None,
            Sublevel::Full => profile.max_decoded_bpp(),
            other => other.nominal_bpp(),
        }
    }
}

/// Maximum admissible codestream size `Ssl,max` in bytes from SOC to EOC
/// (§A.4.1):
///
/// ```text
///   Ssl,max = floor(Lmax × Nbpp / 8)
/// ```
///
/// where `Lmax` is the level's maximum number of sampling-grid points
/// (`Level::max_samples`) and `Nbpp` is the sublevel's nominal
/// bits-per-pixel (`Sublevel::effective_nbpp`, profile-dependent for the
/// `Full` sublevel). Returns `None` when the level or sublevel is
/// unrestricted (no coded-domain bound, §A.5), in which case the
/// codestream size is unconstrained. The closed form is verified against
/// every numeric entry of Tables A.8–A.11 in the unit tests.
pub fn max_codestream_size(level: Level, sublevel: Sublevel, profile: Profile) -> Option<u64> {
    let lmax = level.max_samples()?;
    let nbpp = sublevel.effective_nbpp(profile)? as u64;
    Some(lmax.saturating_mul(nbpp) / 8)
}

/// Chroma format families implied by the per-component sampling
/// factors in the CDT. Matches the buckets in Tables A.1 / A.2 / A.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Single luma plane (`Nc == 1`). Listed as `4:0:0` in the tables.
    Mono,
    /// 3 components, 4:2:2 (`(sx, sy)` ∈ `{(1,1), (2,1), (2,1)}`).
    Yuv422,
    /// 3 components, 4:2:0 (`(sx, sy)` ∈ `{(1,1), (2,2), (2,2)}`).
    /// Although the tables list 4:2:0 only as a *decoded* chroma format
    /// — vertical sub-sampling is *not* a profile-listed format — we
    /// keep it as a separate variant so [`classify_chroma`] never
    /// silently mis-classifies a 4:2:0 layout.
    Yuv420,
    /// 3 components, 4:4:4 (all `(sx, sy) == (1, 1)`).
    Yuv444,
    /// 4 components, 4:2:2:4 — three 4:2:2 planes plus a fourth
    /// `(1, 1)` plane (typical alpha or Star-Tetrix R).
    Yuv4224,
    /// 4 components, 4:4:4:4 — all four planes `(1, 1)`.
    Yuv4444,
    /// Anything else — neither caller nor profile-table classification
    /// has a place for this layout.
    Other,
}

impl ChromaFormat {
    /// Short human-readable name (e.g. `"4:2:2"`).
    pub fn name(self) -> &'static str {
        match self {
            ChromaFormat::Mono => "4:0:0",
            ChromaFormat::Yuv422 => "4:2:2",
            ChromaFormat::Yuv420 => "4:2:0",
            ChromaFormat::Yuv444 => "4:4:4",
            ChromaFormat::Yuv4224 => "4:2:2:4",
            ChromaFormat::Yuv4444 => "4:4:4:4",
            ChromaFormat::Other => "other",
        }
    }
}

/// Classify the chroma format implied by a CDT component list per the
/// buckets in Tables A.1 / A.2 / A.3 / A.4.
pub fn classify_chroma(components: &[Component]) -> ChromaFormat {
    match components.len() {
        1 => ChromaFormat::Mono,
        3 => {
            let s: Vec<(u8, u8)> = components.iter().map(|c| (c.sx, c.sy)).collect();
            if s == [(1, 1), (1, 1), (1, 1)] {
                ChromaFormat::Yuv444
            } else if s == [(1, 1), (2, 1), (2, 1)] {
                ChromaFormat::Yuv422
            } else if s == [(1, 1), (2, 2), (2, 2)] {
                ChromaFormat::Yuv420
            } else {
                ChromaFormat::Other
            }
        }
        4 => {
            let s: Vec<(u8, u8)> = components.iter().map(|c| (c.sx, c.sy)).collect();
            if s == [(1, 1), (1, 1), (1, 1), (1, 1)] {
                ChromaFormat::Yuv4444
            } else if s == [(1, 1), (2, 1), (2, 1), (1, 1)] {
                ChromaFormat::Yuv4224
            } else {
                ChromaFormat::Other
            }
        }
        _ => ChromaFormat::Other,
    }
}

/// Which `Qpih` (Table A.10) values a profile permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QpihAllowed {
    /// `Qpih ∈ {0}` only (Light family).
    DeadzoneOnly,
    /// `Qpih ∈ {0, 1}` (Main, High, Light-Subline).
    DeadzoneOrUniform,
}

impl QpihAllowed {
    /// Whether the given 2-bit `Qpih` field is permitted by this
    /// profile.
    pub fn permits(self, qpih: u8) -> bool {
        match self {
            QpihAllowed::DeadzoneOnly => qpih == 0,
            QpihAllowed::DeadzoneOrUniform => qpih <= 1,
        }
    }
}

/// Column-mode rules (Tables A.1 / A.2 / A.3 footnotes `d` and `e`).
///
/// Tables A.1 / A.3 say "One column except when the number of vertical
/// decomposition levels is zero". Table A.2 row 1 / 2 (Light) says
/// "Only one column permitted (full width)". Table A.2 row 3
/// (Light-Subline) says "Max column width = 2 048".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnMode {
    /// `Cw == 0` (full-width column) is the only permitted layout
    /// regardless of `NL,y`.
    FullWidthOnly,
    /// `Cw == 0` is the default; `Cw > 0` is permitted only when
    /// `NL,y == 0` (footnote `e`) and the resulting `Cs` is bounded by
    /// Part-1's general constraint (no extra cap beyond §B.5).
    SingleColumnUnlessNlyZero,
    /// `Cw == 0` or `Cw > 0` permitted, but the resulting `Cs` is
    /// bounded by `max_column_width` (Light-Subline 422.10).
    ColumnCapped,
}

/// Per-profile constraint set encoded from Tables A.1 / A.2 / A.3.
///
/// `allowed_bit_depths` and `allowed_chroma` are static slices because
/// the profile table is itself static. `max_column_width` is `Some`
/// only for the Light-Subline 422.10 profile (Cs ≤ 2048).
#[derive(Debug, Clone, Copy)]
pub struct ProfileLimits {
    pub profile: Profile,
    pub allowed_bit_depths: &'static [u8],
    pub allowed_chroma: &'static [ChromaFormat],
    pub max_nly: u8,
    /// Inclusive (min, max) on `NL,x`. The lower bound is 0 only for
    /// Light-Subline.
    pub nlx_range: (u8, u8),
    pub allowed_qpih: QpihAllowed,
    pub column_mode: ColumnMode,
    pub max_column_width: Option<u32>,
    pub slice_height: u32,
    pub max_components: u8,
}

impl ProfileLimits {
    /// Unrestricted (`Ppih == 0`) — every constraint is permissive.
    pub fn unrestricted() -> ProfileLimits {
        ProfileLimits {
            profile: Profile::Unrestricted,
            // The CDT field permits 8..=16 per spec; expose all of them
            // so [`check_components`] never rejects an Unrestricted
            // stream solely on the bit-depth list.
            allowed_bit_depths: &[8, 9, 10, 11, 12, 13, 14, 15, 16],
            allowed_chroma: &[
                ChromaFormat::Mono,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv444,
                ChromaFormat::Yuv4224,
                ChromaFormat::Yuv4444,
                ChromaFormat::Other,
            ],
            max_nly: 8,
            nlx_range: (0, 8),
            allowed_qpih: QpihAllowed::DeadzoneOrUniform,
            column_mode: ColumnMode::SingleColumnUnlessNlyZero,
            max_column_width: None,
            slice_height: u32::MAX,
            max_components: 8,
        }
    }
}

impl Profile {
    /// Maximum decoded bits-per-pixel per Table A.4, for use with the
    /// `Full` sublevel.
    pub fn max_decoded_bpp(self) -> Option<u32> {
        Some(match self {
            Profile::Unrestricted => return None,
            Profile::Light422_10 | Profile::LightSubline422_10 | Profile::Main422_10 => 20,
            Profile::Light444_12 | Profile::Main444_12 | Profile::High444_12 => 36,
            Profile::Main4444_12 | Profile::High4444_12 => 48,
        })
    }
}

/// Compute the column width `Cs` in sampling-grid points per formula
/// (A.3) of §A.4.1:
///
/// ```text
///        ⎧ 8 × Cw × max_i(sx[i]) × 2^NL,x   if Cw > 0
/// Cs  =  ⎨
///        ⎩ Wf                              otherwise (Cw == 0)
/// ```
///
/// Returns the result as `u32` because the worst-case product
/// (`8 × 65535 × 8 × 2^5`) still fits comfortably in 32 bits.
pub fn column_width(cw: u16, max_sx: u8, nlx: u8, wf: u16) -> u32 {
    if cw == 0 {
        wf as u32
    } else {
        (cw as u32) * 8 * (max_sx as u32) * (1u32 << nlx as u32)
    }
}

/// Check that a parsed [`Codestream`] conforms to `profile`.
///
/// Returns `Ok(())` if every observable constraint from the profile
/// row in Tables A.1 / A.2 / A.3 is satisfied, or `Err(Error::invalid)`
/// with a per-failure diagnostic. [`Profile::Unrestricted`] always
/// succeeds (§A.2.2 — not a conformance point).
///
/// Constraints checked:
///
/// * Number of components `Nc` ≤ `max_components`.
/// * Per-component bit depth `B[i]` ∈ `allowed_bit_depths`.
/// * Chroma format (from CDT `(sx, sy)` factors) ∈ `allowed_chroma`.
/// * `NL,x ∈ nlx_range` and `NL,y ≤ max_nly`.
/// * `NL,x ≥ NL,y` (Table A.1 footnote `c`).
/// * `Qpih` ∈ allowed set.
/// * Slice height `Hsl == slice_height` (or `0` for full-image slice).
/// * Column-mode rules (`FullWidthOnly` / `SingleColumnUnlessNlyZero` /
///   `ColumnCapped`) including the `Cs ≤ 2048` cap for Light-Subline.
///
/// Buffer-model constraints (Annexes B/C/D) are out of scope; checking
/// them requires the transmission-channel rate, which is a host-side
/// quantity not carried in the codestream.
pub fn check_codestream(cs: &Codestream, profile: Profile) -> Result<()> {
    if matches!(profile, Profile::Unrestricted) {
        return Ok(());
    }
    let lim = profile.limits();
    let pih = &cs.pih;
    let cdt = &cs.cdt;

    // Component count.
    if pih.nc > lim.max_components {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: Nc={} exceeds profile max of {}",
            profile.name(),
            pih.nc,
            lim.max_components
        )));
    }
    if cdt.components.len() != pih.nc as usize {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: PIH Nc={} but CDT carries {} components",
            profile.name(),
            pih.nc,
            cdt.components.len()
        )));
    }

    // Bit-depth set.
    for (i, comp) in cdt.components.iter().enumerate() {
        if !lim.allowed_bit_depths.contains(&comp.bit_depth) {
            return Err(Error::invalid(format!(
                "jpegxs profile {}: component {} bit depth {} not in allowed set {:?}",
                profile.name(),
                i,
                comp.bit_depth,
                lim.allowed_bit_depths
            )));
        }
    }

    // Chroma format.
    let chroma = classify_chroma(&cdt.components);
    if !lim.allowed_chroma.contains(&chroma) {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: chroma format {} not allowed (permitted: {:?})",
            profile.name(),
            chroma.name(),
            lim.allowed_chroma
                .iter()
                .map(|c| c.name())
                .collect::<Vec<_>>(),
        )));
    }

    // Decomposition counts.
    let (nlx_min, nlx_max) = lim.nlx_range;
    if pih.nlx < nlx_min || pih.nlx > nlx_max {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: NL,x={} outside permitted range [{}, {}]",
            profile.name(),
            pih.nlx,
            nlx_min,
            nlx_max
        )));
    }
    if pih.nly > lim.max_nly {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: NL,y={} exceeds profile max of {}",
            profile.name(),
            pih.nly,
            lim.max_nly
        )));
    }
    // Table A.1 footnote c — NL,x must be at least NL,y.
    if pih.nlx < pih.nly {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: NL,x={} must be >= NL,y={} (Annex A footnote c)",
            profile.name(),
            pih.nlx,
            pih.nly,
        )));
    }

    // Qpih.
    if !lim.allowed_qpih.permits(pih.qpih) {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: Qpih={} not allowed (profile permits {:?})",
            profile.name(),
            pih.qpih,
            lim.allowed_qpih
        )));
    }

    // Slice height. Hsl is signalled in precincts; profiles fix it at
    // 16 image-row precincts (Table A.1/A.2/A.3 "Slice height in number
    // of image rows = 16"). A precinct row spans `2^NL,y` image rows,
    // so the precinct count per slice is `16 / 2^NL,y` (NL,y in
    // {0, 1, 2}). The check is equivalent to
    // `Hsl * 2^NL,y == slice_height`.
    let slice_image_rows = (pih.hsl as u32).saturating_mul(1u32 << pih.nly as u32);
    if slice_image_rows != lim.slice_height {
        return Err(Error::invalid(format!(
            "jpegxs profile {}: Hsl={} × 2^NL,y={} = {} image rows, profile requires {}",
            profile.name(),
            pih.hsl,
            pih.nly,
            slice_image_rows,
            lim.slice_height
        )));
    }

    // Column mode + Cs bound.
    let max_sx = cdt.components.iter().map(|c| c.sx).max().unwrap_or(1);
    let cs_val = column_width(pih.cw, max_sx, pih.nlx, pih.wf);
    match lim.column_mode {
        ColumnMode::FullWidthOnly => {
            if pih.cw != 0 {
                return Err(Error::invalid(format!(
                    "jpegxs profile {}: Cw={} not permitted (profile allows full-width columns only, Cw=0)",
                    profile.name(),
                    pih.cw,
                )));
            }
        }
        ColumnMode::SingleColumnUnlessNlyZero => {
            if pih.cw != 0 && pih.nly != 0 {
                return Err(Error::invalid(format!(
                    "jpegxs profile {}: Cw={} > 0 requires NL,y==0 (got NL,y={})",
                    profile.name(),
                    pih.cw,
                    pih.nly,
                )));
            }
        }
        ColumnMode::ColumnCapped => {
            // Light-Subline 422.10 caps Cs at the profile's
            // max_column_width regardless of how Cw + NL,x combine.
            if let Some(cap) = lim.max_column_width {
                if cs_val > cap {
                    return Err(Error::invalid(format!(
                        "jpegxs profile {}: Cs={} (Cw={}, max_sx={}, 2^NL,x={}, Wf={}) exceeds profile cap {}",
                        profile.name(),
                        cs_val,
                        pih.cw,
                        max_sx,
                        1u32 << pih.nlx as u32,
                        pih.wf,
                        cap,
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Check that the codestream's `Plev` field decodes a permitted level
/// and (if present) the picture's width / height / total sample count
/// fall within that level's `Wmax` / `Hmax` / `Lmax` bounds (Table
/// A.6).
///
/// Returns `Ok(None)` when the level is `Unrestricted` (decoded-domain
/// bounds are not checked); otherwise `Ok(Some(level))`. Returns
/// `Err(Error::invalid)` if `Plev` decodes to a reserved value or the
/// image exceeds a bound.
pub fn check_level(cs: &Codestream) -> Result<Option<Level>> {
    let Some(level) = Level::from_plev_high(cs.pih.plev) else {
        return Err(Error::invalid(format!(
            "jpegxs level: Plev high byte 0x{:02X} is reserved",
            cs.pih.plev >> 8
        )));
    };
    if matches!(level, Level::Unrestricted) {
        return Ok(None);
    }
    let w = cs.pih.width();
    let h = cs.pih.height();
    if let Some(wmax) = level.max_width() {
        if w > wmax {
            return Err(Error::invalid(format!(
                "jpegxs level {:?}: Wf={} exceeds Wmax={}",
                level, w, wmax
            )));
        }
    }
    if let Some(hmax) = level.max_height() {
        if h > hmax {
            return Err(Error::invalid(format!(
                "jpegxs level {:?}: Hf={} exceeds Hmax={}",
                level, h, hmax
            )));
        }
    }
    if let Some(lmax) = level.max_samples() {
        let samples = w as u64 * h as u64;
        if samples > lmax {
            return Err(Error::invalid(format!(
                "jpegxs level {:?}: Wf×Hf={} exceeds Lmax={}",
                level, samples, lmax
            )));
        }
    }
    Ok(Some(level))
}

/// Check that the on-wire codestream length `codestream_len` (the full
/// SOC-to-EOC byte count, including all markers) does not exceed the
/// `Ssl,max` coded-domain bound implied by the picture header's declared
/// level (`Plev` high byte) and sublevel (`Plev` low byte), per §A.4.1.
///
/// Returns `Ok(())` when no bound applies — an unrestricted level or
/// sublevel, or a `Plev` that decodes to no sublevel id (treated as
/// unconstrained here; [`check_level`] is responsible for rejecting a
/// reserved *level* high byte). Returns `Err(Error::invalid)` when the
/// codestream is larger than `Ssl,max`.
///
/// The profile (`Ppih`) is consulted only to resolve the `Full`
/// sublevel's `Nbpp` (Table A.4); an unmappable `Ppih` falls back to
/// `Unrestricted`, which leaves the `Full` sublevel without a bound
/// (the profile/level checks reject the reserved `Ppih` separately).
pub fn check_codestream_size(cs: &Codestream, codestream_len: usize) -> Result<()> {
    let Some(level) = Level::from_plev_high(cs.pih.plev) else {
        // Reserved level high byte — check_level reports this; nothing to
        // bound here.
        return Ok(());
    };
    let Some(sublevel) = Sublevel::from_plev_low(cs.pih.plev) else {
        return Ok(());
    };
    let profile = Profile::from_ppih(cs.pih.ppih).unwrap_or(Profile::Unrestricted);
    // §A.4.2: "The Full sublevel shall only be used if the profile value
    // is not unrestricted." A Full sublevel with Ppih=0 has no defined
    // Nbpp (Table A.7 defers it to the profile's max-decoded-bpp), so the
    // combination is non-conformant — reject it.
    if matches!(sublevel, Sublevel::Full) && matches!(profile, Profile::Unrestricted) {
        return Err(Error::invalid(
            "jpegxs: Full sublevel (Plev low 0x80) is only valid with a non-unrestricted profile \
             (Ppih ≠ 0) — §A.4.2"
                .to_string(),
        ));
    }
    if let Some(max) = max_codestream_size(level, sublevel, profile) {
        if codestream_len as u64 > max {
            return Err(Error::invalid(format!(
                "jpegxs sublevel {sublevel:?} @ level {level:?}: codestream is {codestream_len} \
                 bytes, exceeds Ssl,max = {max} bytes (floor(Lmax × Nbpp / 8), §A.4.1)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_table::{Component, ComponentTable};
    use crate::picture_header::PictureHeader;

    /// Build a minimal codestream that targets a given profile.
    #[allow(clippy::too_many_arguments)]
    fn make_cs(
        profile: Profile,
        nc: u8,
        bd: u8,
        sxsy: &[(u8, u8)],
        nlx: u8,
        nly: u8,
        cw: u16,
        wf: u16,
        hf: u16,
        qpih: u8,
        hsl: u16,
    ) -> Codestream {
        let pih = PictureHeader {
            lcod: 0,
            ppih: profile.ppih(),
            plev: 0,
            wf,
            hf,
            cw,
            hsl,
            nc,
            ng: 4,
            ss: 8,
            bw: bd,
            fq: 0,
            br: 0,
            fslc: 0,
            ppoc: 0,
            cpih: 0,
            nlx,
            nly,
            lh: 0,
            rl: 0,
            qpih,
            fs: 0,
            rm: 0,
        };
        let cdt = ComponentTable {
            components: sxsy
                .iter()
                .map(|&(sx, sy)| Component {
                    bit_depth: bd,
                    sx,
                    sy,
                })
                .collect(),
        };
        Codestream {
            cap: vec![],
            pih,
            cdt,
            wgt: vec![],
            nlt: None,
            cwd: None,
            cwd_sd: None,
            cts: None,
            crg: None,
            com: vec![],
            slices: vec![],
            eoc_offset: None,
        }
    }

    #[test]
    fn ppih_roundtrip_table_a5() {
        // Every value in Table A.5 round-trips through `from_ppih` →
        // `ppih`.
        for p in [
            Profile::Unrestricted,
            Profile::Light422_10,
            Profile::Light444_12,
            Profile::LightSubline422_10,
            Profile::Main422_10,
            Profile::Main444_12,
            Profile::Main4444_12,
            Profile::High444_12,
            Profile::High4444_12,
        ] {
            assert_eq!(
                Profile::from_ppih(p.ppih()),
                Some(p),
                "ppih {:04x}",
                p.ppih()
            );
        }
        // Spot-check an explicitly reserved value.
        assert_eq!(Profile::from_ppih(0xFFFF), None);
        assert_eq!(Profile::from_ppih(0x9999), None);
    }

    #[test]
    fn classify_chroma_buckets() {
        let mono = vec![Component {
            bit_depth: 8,
            sx: 1,
            sy: 1,
        }];
        assert_eq!(classify_chroma(&mono), ChromaFormat::Mono);

        let yuv422 = vec![(1, 1), (2, 1), (2, 1)]
            .into_iter()
            .map(|(sx, sy)| Component {
                bit_depth: 10,
                sx,
                sy,
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_chroma(&yuv422), ChromaFormat::Yuv422);

        let yuv420 = vec![(1, 1), (2, 2), (2, 2)]
            .into_iter()
            .map(|(sx, sy)| Component {
                bit_depth: 8,
                sx,
                sy,
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_chroma(&yuv420), ChromaFormat::Yuv420);

        let yuv4224 = vec![(1, 1), (2, 1), (2, 1), (1, 1)]
            .into_iter()
            .map(|(sx, sy)| Component {
                bit_depth: 12,
                sx,
                sy,
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_chroma(&yuv4224), ChromaFormat::Yuv4224);

        let yuv4444 = vec![(1, 1); 4]
            .into_iter()
            .map(|(sx, sy)| Component {
                bit_depth: 12,
                sx,
                sy,
            })
            .collect::<Vec<_>>();
        assert_eq!(classify_chroma(&yuv4444), ChromaFormat::Yuv4444);
    }

    #[test]
    fn check_passes_for_main_422_10() {
        let cs = make_cs(
            Profile::Main422_10,
            3,
            10,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1920,
            1080,
            1,
            8,
        );
        // Hsl=8 * 2^NL,y=1 → 16 image rows = profile slice_height.
        check_codestream(&cs, Profile::Main422_10).expect("conforms");
    }

    #[test]
    fn check_rejects_main_with_12bit() {
        let cs = make_cs(
            Profile::Main422_10,
            3,
            12,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1920,
            1080,
            0,
            8,
        );
        let err = check_codestream(&cs, Profile::Main422_10).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bit depth 12"), "got: {msg}");
    }

    #[test]
    fn check_rejects_light_with_uniform_qpih() {
        let cs = make_cs(
            Profile::Light422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1280,
            720,
            1, // Qpih=1 (uniform) — Light forbids.
            8,
        );
        let err = check_codestream(&cs, Profile::Light422_10).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Qpih=1"), "got: {msg}");
    }

    #[test]
    fn check_rejects_light_with_yuv444() {
        let cs = make_cs(
            Profile::Light422_10,
            3,
            8,
            &[(1, 1), (1, 1), (1, 1)],
            5,
            1,
            0,
            1280,
            720,
            0,
            8,
        );
        let err = check_codestream(&cs, Profile::Light422_10).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("4:4:4"), "got: {msg}");
    }

    #[test]
    fn check_rejects_main_422_with_cw_and_nly_nonzero() {
        // Main 422.10 footnote e — Cw > 0 only allowed when NL,y == 0.
        // Use cs=8*1*2*32=512 image-row precincts; matching Hsl so the
        // slice-height check stays clean is irrelevant — the Cw check
        // fires first.
        let cs = make_cs(
            Profile::Main422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1, // NL,y != 0
            1, // Cw > 0
            1920,
            1080,
            0,
            8,
        );
        let err = check_codestream(&cs, Profile::Main422_10).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Cw=1"), "got: {msg}");
    }

    #[test]
    fn check_subline_cap_2048() {
        // Light-Subline 422.10 caps Cs at 2048. With NL,x=5, Cw=1,
        // max_sx=2 → Cs = 8 * 1 * 2 * 32 = 512 — passes.
        let cs_ok = make_cs(
            Profile::LightSubline422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            0,
            1,
            1920,
            1080,
            0,
            16,
        );
        check_codestream(&cs_ok, Profile::LightSubline422_10).expect("Cs=512 ≤ 2048");

        // Bump Cw to 5 → Cs = 8 * 5 * 2 * 32 = 2560 — fails.
        let cs_bad = make_cs(
            Profile::LightSubline422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            0,
            5,
            1920,
            1080,
            0,
            16,
        );
        let err = check_codestream(&cs_bad, Profile::LightSubline422_10).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Cs=2560"), "got: {msg}");
        assert!(msg.contains("2048"), "got: {msg}");
    }

    #[test]
    fn check_high_nly_2_ok_for_high_profile() {
        let cs = make_cs(
            Profile::High444_12,
            3,
            12,
            &[(1, 1), (1, 1), (1, 1)],
            5,
            2,
            0,
            1920,
            1080,
            1,
            4,
        );
        // NL,y=2, Hsl=4 → 4 * 2^2 = 16 image rows.
        check_codestream(&cs, Profile::High444_12).expect("High permits NL,y=2");
        // The same stream against a Main profile must fail on NL,y.
        let err = check_codestream(&cs, Profile::Main444_12).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("NL,y=2"), "got: {msg}");
    }

    #[test]
    fn check_4444_profile_accepts_4_components() {
        let cs = make_cs(
            Profile::Main4444_12,
            4,
            12,
            &[(1, 1), (1, 1), (1, 1), (1, 1)],
            5,
            1,
            0,
            1024,
            1024,
            0,
            8,
        );
        check_codestream(&cs, Profile::Main4444_12).expect("Main4444 permits 4-comp");
        // Against Main444_12 (max_components=3) it must fail.
        let err = check_codestream(&cs, Profile::Main444_12).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Nc=4"), "got: {msg}");
    }

    #[test]
    fn unrestricted_accepts_anything() {
        let cs = make_cs(
            Profile::Unrestricted,
            8,
            16,
            &[(3, 5); 8],
            7,
            7,
            123,
            65000,
            65000,
            3,
            999,
        );
        check_codestream(&cs, Profile::Unrestricted).expect("unrestricted is permissive");
    }

    #[test]
    fn level_decode_table_a12() {
        // Spot-check each row by setting the high byte directly.
        assert_eq!(Level::from_plev_high(0x0000), Some(Level::Unrestricted));
        assert_eq!(Level::from_plev_high(0x1000), Some(Level::L2k1));
        assert_eq!(Level::from_plev_high(0x2000), Some(Level::L4k1));
        assert_eq!(Level::from_plev_high(0x2400), Some(Level::L4k2));
        assert_eq!(Level::from_plev_high(0x2800), Some(Level::L4k3));
        assert_eq!(Level::from_plev_high(0x3000), Some(Level::L8k1));
        assert_eq!(Level::from_plev_high(0x3400), Some(Level::L8k2));
        assert_eq!(Level::from_plev_high(0x3800), Some(Level::L8k3));
        assert_eq!(Level::from_plev_high(0x4000), Some(Level::L10k1));
        assert_eq!(Level::from_plev_high(0x9900), None);
    }

    #[test]
    fn sublevel_decode_table_a13() {
        assert_eq!(
            Sublevel::from_plev_low_byte(0x00),
            Some(Sublevel::Unrestricted)
        );
        assert_eq!(Sublevel::from_plev_low_byte(0x80), Some(Sublevel::Full));
        assert_eq!(
            Sublevel::from_plev_low_byte(0x10),
            Some(Sublevel::Sublev12bpp)
        );
        assert_eq!(
            Sublevel::from_plev_low_byte(0x0c),
            Some(Sublevel::Sublev9bpp)
        );
        assert_eq!(
            Sublevel::from_plev_low_byte(0x08),
            Some(Sublevel::Sublev6bpp)
        );
        assert_eq!(
            Sublevel::from_plev_low_byte(0x04),
            Some(Sublevel::Sublev3bpp)
        );
        // A value not in any row.
        assert_eq!(Sublevel::from_plev_low_byte(0x05), None);
    }

    #[test]
    fn level_check_enforces_wmax_hmax() {
        let mut cs = make_cs(
            Profile::Main422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1920,
            1080,
            0,
            8,
        );
        // Plev high byte 0x10 → 2k-1: Wmax=2048, Hmax=8192, Lmax≈4.2M.
        cs.pih.plev = 0x1000;
        let lvl = check_level(&cs).expect("conforms").unwrap();
        assert_eq!(lvl, Level::L2k1);

        // Bump Wf past 2048 → fails.
        cs.pih.wf = 3000;
        let err = check_level(&cs).unwrap_err();
        assert!(format!("{err}").contains("Wf=3000"));
    }

    #[test]
    fn level_check_rejects_lmax_overflow() {
        let mut cs = make_cs(
            Profile::Main422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            2048,
            8192,
            0,
            8,
        );
        // 2048 * 8192 = 16 777 216 — already at L4k2 Lmax, exceeds
        // L2k1's 4 194 304.
        cs.pih.plev = 0x1000;
        let err = check_level(&cs).unwrap_err();
        assert!(format!("{err}").contains("Lmax"));
    }

    #[test]
    fn ssl_max_matches_tables_a8_to_a11() {
        // Closed form Ssl,max = floor(Lmax × Nbpp / 8) reproduced against
        // every numeric entry of Tables A.8 (3 bpp), A.9 (6), A.10 (9),
        // A.11 (12). `profile` is irrelevant for non-Full sublevels.
        let levels = [
            (Level::L2k1, 0x10u16),
            (Level::L4k1, 0x20),
            (Level::L4k2, 0x24),
            (Level::L4k3, 0x28),
            (Level::L8k1, 0x30),
            (Level::L8k2, 0x34),
            (Level::L8k3, 0x38),
            (Level::L10k1, 0x40),
        ];
        // (sublevel, [Ssl,max per level above]).
        let cases: &[(Sublevel, [u64; 8])] = &[
            (
                Sublevel::Sublev3bpp,
                [
                    1_572_864, 3_342_336, 6_291_456, 6_291_456, 13_369_344, 25_165_824, 25_165_824,
                    39_321_600,
                ],
            ),
            (
                Sublevel::Sublev6bpp,
                [
                    3_145_728, 6_684_672, 12_582_912, 12_582_912, 26_738_688, 50_331_648,
                    50_331_648, 78_643_200,
                ],
            ),
            (
                Sublevel::Sublev9bpp,
                [
                    4_718_592,
                    10_027_008,
                    18_874_368,
                    18_874_368,
                    40_108_032,
                    75_497_472,
                    75_497_472,
                    117_964_800,
                ],
            ),
            (
                Sublevel::Sublev12bpp,
                [
                    6_291_456,
                    13_369_344,
                    25_165_824,
                    25_165_824,
                    53_477_376,
                    100_663_296,
                    100_663_296,
                    157_286_400,
                ],
            ),
        ];
        for (sub, expected) in cases {
            for ((level, _), &exp) in levels.iter().zip(expected.iter()) {
                let got = max_codestream_size(*level, *sub, Profile::Unrestricted)
                    .expect("bounded sublevel/level");
                assert_eq!(got, exp, "Ssl,max {sub:?} @ {level:?}");
            }
        }
    }

    #[test]
    fn unrestricted_level_or_sublevel_has_no_size_bound() {
        assert_eq!(
            max_codestream_size(
                Level::Unrestricted,
                Sublevel::Sublev3bpp,
                Profile::Main422_10
            ),
            None
        );
        assert_eq!(
            max_codestream_size(Level::L2k1, Sublevel::Unrestricted, Profile::Main422_10),
            None
        );
    }

    #[test]
    fn full_sublevel_nbpp_follows_profile() {
        // Full sublevel Nbpp is the profile's max-decoded-bpp (Table A.4):
        // Main 422.10 → 20, Main 4444.12 → 48. Ssl,max scales accordingly.
        let lmax = Level::L2k1.max_samples().unwrap();
        assert_eq!(
            max_codestream_size(Level::L2k1, Sublevel::Full, Profile::Main422_10),
            Some(lmax * 20 / 8)
        );
        assert_eq!(
            max_codestream_size(Level::L2k1, Sublevel::Full, Profile::Main4444_12),
            Some(lmax * 48 / 8)
        );
        // Full with an unrestricted profile has no Nbpp (Table A.7 forbids
        // the combination); we surface that as "no bound" rather than
        // panicking.
        assert_eq!(
            max_codestream_size(Level::L2k1, Sublevel::Full, Profile::Unrestricted),
            None
        );
    }

    #[test]
    fn check_codestream_size_rejects_oversized() {
        // L2k1 + Sublev3bpp → Ssl,max = 1 572 864 bytes.
        let mut cs = make_cs(
            Profile::Main422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1920,
            1080,
            0,
            8,
        );
        cs.pih.plev = 0x1004; // level 2k-1 (0x10), sublevel 3bpp (0x04)
        let max = 1_572_864usize;
        check_codestream_size(&cs, max).expect("exactly Ssl,max is allowed");
        let err = check_codestream_size(&cs, max + 1).unwrap_err();
        assert!(
            format!("{err}").contains("Ssl,max"),
            "expected Ssl,max rejection, got {err}"
        );
    }

    #[test]
    fn check_codestream_size_rejects_full_with_unrestricted_profile() {
        // §A.4.2: Full sublevel requires a non-unrestricted profile. A
        // Ppih=0 (Unrestricted) stream with Plev sublevel = Full (0x80) at
        // level 2k-1 (0x10) is non-conformant.
        let mut cs = make_cs(Profile::Unrestricted, 1, 8, &[(1, 1)], 1, 0, 0, 4, 1, 0, 1);
        cs.pih.ppih = 0x0000; // Unrestricted
        cs.pih.plev = 0x1080; // level 2k-1, sublevel Full
        let err = check_codestream_size(&cs, 100).unwrap_err();
        assert!(
            format!("{err}").contains("Full sublevel"),
            "expected Full+unrestricted rejection, got {err}"
        );
    }

    #[test]
    fn check_codestream_size_accepts_full_with_real_profile() {
        // Full sublevel with a real profile (Main 422.10 → Nbpp 20) is
        // valid and bounds the codestream at floor(Lmax × 20 / 8).
        let mut cs = make_cs(
            Profile::Main422_10,
            3,
            8,
            &[(1, 1), (2, 1), (2, 1)],
            5,
            1,
            0,
            1920,
            1080,
            0,
            8,
        );
        cs.pih.plev = 0x1080; // level 2k-1, sublevel Full
        let max = (Level::L2k1.max_samples().unwrap() * 20 / 8) as usize;
        check_codestream_size(&cs, max).expect("at-bound Full decodes");
        assert!(check_codestream_size(&cs, max + 1).is_err());
    }

    #[test]
    fn check_codestream_size_no_bound_when_unrestricted() {
        let mut cs = make_cs(Profile::Unrestricted, 1, 8, &[(1, 1)], 1, 0, 0, 4, 1, 0, 1);
        cs.pih.plev = 0x0000; // unrestricted level + sublevel
        check_codestream_size(&cs, usize::MAX).expect("unrestricted: no size bound");
    }
}
