//! JPEG XS codestream marker codes (ISO/IEC 21122-1:2022, Table A.2).
//!
//! All markers are two bytes, beginning with `0xff`. The second byte is
//! never `0x00` or `0xff`. Markers either stand alone (SOC, EOC) or
//! introduce a marker segment whose body is preceded by a big-endian
//! `Lxxx` 16-bit length field; that length includes the two `Lxxx`
//! bytes themselves but excludes the marker.

/// Two-byte JPEG XS marker code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Marker(pub u16);

impl Marker {
    /// Start of codestream (Table A.3). Stand-alone, mandatory, first.
    pub const SOC: Marker = Marker(0xff10);
    /// End of codestream (Table A.4). Stand-alone, mandatory, last.
    pub const EOC: Marker = Marker(0xff11);
    /// Picture header (Table A.7). Mandatory, exactly once.
    pub const PIH: Marker = Marker(0xff12);
    /// Component table (Table A.15). Mandatory, exactly once.
    pub const CDT: Marker = Marker(0xff13);
    /// Weights table (Table A.24). Mandatory, exactly once.
    pub const WGT: Marker = Marker(0xff14);
    /// Extension marker / comment (Table A.22). Optional, zero or more.
    pub const COM: Marker = Marker(0xff15);
    /// Nonlinearity marker (Table A.16). Optional, zero or one.
    pub const NLT: Marker = Marker(0xff16);
    /// Component-dependent wavelet decomposition (Table A.18). Optional.
    pub const CWD: Marker = Marker(0xff17);
    /// Colour transformation specification (Table A.19). Mandatory if
    /// `Cpih == 3` (Star-Tetrix), forbidden otherwise.
    pub const CTS: Marker = Marker(0xff18);
    /// Component registration (Table A.21). Optional, mandatory if
    /// `Cpih == 3`.
    pub const CRG: Marker = Marker(0xff19);
    /// Slice header (Table A.25). One per slice; entropy-coded data
    /// follows.
    pub const SLH: Marker = Marker(0xff20);
    /// Capabilities marker (Table A.6). Mandatory, second segment.
    pub const CAP: Marker = Marker(0xff50);

    /// Short symbolic name for diagnostics, or `"?"` for unrecognized.
    pub fn name(self) -> &'static str {
        match self {
            Marker::SOC => "SOC",
            Marker::EOC => "EOC",
            Marker::PIH => "PIH",
            Marker::CDT => "CDT",
            Marker::WGT => "WGT",
            Marker::COM => "COM",
            Marker::NLT => "NLT",
            Marker::CWD => "CWD",
            Marker::CTS => "CTS",
            Marker::CRG => "CRG",
            Marker::SLH => "SLH",
            Marker::CAP => "CAP",
            _ => "?",
        }
    }

    /// Is this marker a stand-alone marker (no length-prefixed payload)?
    /// Per Table A.2 only SOC and EOC stand alone in JPEG XS.
    pub fn is_standalone(self) -> bool {
        matches!(self, Marker::SOC | Marker::EOC)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_codes_match_table_a2() {
        // The exact two-byte codes from ISO/IEC 21122-1:2022 Table A.2.
        assert_eq!(Marker::SOC.0, 0xff10);
        assert_eq!(Marker::EOC.0, 0xff11);
        assert_eq!(Marker::PIH.0, 0xff12);
        assert_eq!(Marker::CDT.0, 0xff13);
        assert_eq!(Marker::WGT.0, 0xff14);
        assert_eq!(Marker::COM.0, 0xff15);
        assert_eq!(Marker::NLT.0, 0xff16);
        assert_eq!(Marker::CWD.0, 0xff17);
        assert_eq!(Marker::CTS.0, 0xff18);
        assert_eq!(Marker::CRG.0, 0xff19);
        assert_eq!(Marker::SLH.0, 0xff20);
        assert_eq!(Marker::CAP.0, 0xff50);
    }

    #[test]
    fn standalone_classification() {
        assert!(Marker::SOC.is_standalone());
        assert!(Marker::EOC.is_standalone());
        assert!(!Marker::PIH.is_standalone());
        assert!(!Marker::CDT.is_standalone());
        assert!(!Marker::WGT.is_standalone());
        assert!(!Marker::CAP.is_standalone());
        assert!(!Marker::SLH.is_standalone());
    }

    #[test]
    fn unknown_marker_name() {
        assert_eq!(Marker(0xff99).name(), "?");
    }
}
