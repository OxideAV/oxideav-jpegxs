//! JPEG XS — ISO/IEC 21122 low-latency image codec for production / IP
//! video (SMPTE ST 2110-22).
//!
//! Round 6 ships a working end-to-end decoder for the multi-component,
//! single-precinct-row subset of the standard with multi-level
//! wavelet cascade and Star-Tetrix CFA support:
//!
//! * Round 1 — Part-1 codestream marker-chain parser
//!   ([`codestream::parse`]); see [`codestream::Codestream`] for the
//!   captured geometry. [`probe`] returns [`probe::JpegXsFileInfo`]
//!   without instantiating a decoder.
//! * Round 2 — reversible 5/3 inverse DWT (Annex E), in [`dwt`].
//! * Round 3 — entropy decoder (Annex C) over hand-built precinct /
//!   packet geometry, in [`entropy`].
//! * Round 4 — slice / precinct / packet geometry walker (Annex
//!   B.5–B.10) in [`slice_walker`], inverse quantization (Annex D) in
//!   [`dequant`], and a wired-up [`Decoder`] in [`decoder`].
//! * Round 5 — multi-component dispatch in [`slice_walker`] /
//!   [`decoder`], inverse RCT colour transform (Annex F.3) in
//!   [`colour_transform`], NLT marker parser + linear / quadratic /
//!   extended output scaling (Annex G.3 / G.4 / G.5) in [`output`].
//! * Round 6 — multi-level inverse-DWT cascade
//!   ([`dwt::inverse_cascade_2d`], Annex E.2 Table E.1) wired into the
//!   decoder; inverse Star-Tetrix transform
//!   ([`colour_transform::inverse_star_tetrix`], Annex F.5) with the
//!   `access()` reflection from Table F.12; CTS marker parser
//!   ([`cts`]), CRG marker parser ([`crg`]) including the Table F.9 /
//!   F.10 / F.11 super-pixel look-up tables; CAP `cap[]` byte-array
//!   decoder ([`capabilities`]).
//!
//! Round-6 supported subset (`make_decoder` accepts):
//!
//! * `Nc ∈ {1, 2, 3, 4}`, `sx, sy ∈ {1, 2}` per component
//!   (4:4:4, 4:2:2, 4:2:0).
//! * `Cw == 0` (one precinct per row of the picture).
//! * `NL,x` ≥ `NL,y`, both bounded by typical spec values; tested at
//!   `NL = 3/3` end-to-end and `NL = 5/5` at the cascade level.
//! * `Cpih ∈ {0, 1, 3}` — no transform, RGB↔YCbCr reversible
//!   (Annex F.3), or Star-Tetrix (Annex F.5).
//! * `Qpih ∈ {0, 1}` (deadzone or uniform inverse quantizer).
//! * `Fq ∈ {0, 8}` (lossless / regular per Table A.8). 8-bit output
//!   samples (`B[i] == 8`); higher bit depths return `Unsupported`.
//! * NLT marker present → quadratic / extended output scaling
//!   dispatched per Annex G.4 / G.5.
//!
//! Out of round-6 scope (returns `Error::Unsupported`): `Cw > 0`
//! (custom precinct widths), CWD-driven `Sd > 0`, output bit depths
//! > 8, encoder side.

pub mod capabilities;
pub mod codestream;
pub mod colour_transform;
pub mod component_table;
pub mod crg;
pub mod cts;
pub mod decoder;
pub mod dequant;
pub mod dwt;
pub mod entropy;
pub mod markers;
pub mod output;
pub mod picture_header;
pub mod probe;
pub mod slice_header;
pub mod slice_walker;

pub use capabilities::{parse_capabilities, parse_capabilities_lossy, Capabilities};
pub use codestream::{Codestream, Slice};
pub use component_table::{Component, ComponentTable};
pub use crg::{cfa_pattern_type, parse_crg, CrgEntry, CrgMarker};
pub use cts::{parse_cts, CtsExtent, CtsMarker};
pub use markers::Marker;
pub use picture_header::PictureHeader;
pub use probe::{probe, JpegXsFileInfo};
pub use slice_header::SliceHeader;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder, Result,
};

/// Public codec id string. Matches the aggregator feature name `jpegxs`.
pub const CODEC_ID_STR: &str = "jpegxs";

/// Register the JPEG XS decoder factory.
///
/// Round 6 wires a working decoder for the multi-component
/// (Nc ∈ {1, 2, 3, 4}), single-precinct-row subset of the standard
/// with 4:4:4 / 4:2:2 / 4:2:0 sampling, multi-level inverse DWT
/// cascade (Annex E.2 Table E.1), Annex F.3 inverse RCT and Annex F.5
/// inverse Star-Tetrix colour transforms, and the full Annex G
/// output path including the NLT marker. `Cw > 0` (custom precinct
/// widths), CWD-driven `Sd > 0`, and output bit depths > 8 remain
/// out of scope.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxs_sw")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Decoder factory — see [`decoder::make_decoder`]. The `Decoder`
/// trait alias is re-exported from `oxideav_core`.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hand-built JPEG XS codestream: 4x3 single-component
    /// image with one slice and no entropy data. Enough for the marker
    /// parser + probe to round-trip.
    fn build_tiny_codestream() -> Vec<u8> {
        let mut v = Vec::new();
        // SOC
        v.extend_from_slice(&[0xff, 0x10]);
        // CAP — Lcap=2, no capability bits.
        v.extend_from_slice(&[0xff, 0x50]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // PIH — Lpih=26, body=24 bytes.
        v.extend_from_slice(&[0xff, 0x12]);
        v.extend_from_slice(&26u16.to_be_bytes());
        let mut pih = Vec::with_capacity(24);
        pih.extend_from_slice(&0u32.to_be_bytes()); // Lcod
        pih.extend_from_slice(&0u16.to_be_bytes()); // Ppih
        pih.extend_from_slice(&0u16.to_be_bytes()); // Plev
        pih.extend_from_slice(&4u16.to_be_bytes()); // Wf
        pih.extend_from_slice(&3u16.to_be_bytes()); // Hf
        pih.extend_from_slice(&0u16.to_be_bytes()); // Cw
        pih.extend_from_slice(&1u16.to_be_bytes()); // Hsl
        pih.extend_from_slice(&[1, 4, 8, 20]); // Nc, Ng, Ss, Bw
        pih.push(0x80); // Fq=8|Br=0
        pih.push(0x00); // Fslc=0|Ppoc=0|Cpih=0
        pih.push(0x11); // NL,x=1|NL,y=1
        pih.push(0x00); // Lh|Rl|Qpih|Fs|Rm
        v.extend_from_slice(&pih);
        // CDT — Lcdt=4, body=2.
        v.extend_from_slice(&[0xff, 0x13]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&[8, 0x11]);
        // WGT — Lwgt=2.
        v.extend_from_slice(&[0xff, 0x14]);
        v.extend_from_slice(&2u16.to_be_bytes());
        // SLH + EOC.
        v.extend_from_slice(&[0xff, 0x20]);
        v.extend_from_slice(&4u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&[0xff, 0x11]);
        v
    }

    #[test]
    fn probe_returns_geometry() {
        let buf = build_tiny_codestream();
        let info = probe(&buf).expect("probe tiny codestream");
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 3);
        assert_eq!(info.num_components, 1);
        assert_eq!(info.bit_depth, 8);
        assert_eq!(info.profile, 0);
        assert_eq!(info.level, 0);
        assert_eq!(info.cpih, 0);
        assert!(!info.lossless);
    }

    #[test]
    fn probe_rejects_non_jpegxs() {
        let buf = vec![0xff, 0xd8, 0x00, 0x00];
        assert!(probe(&buf).is_none());
        let buf = vec![];
        assert!(probe(&buf).is_none());
    }

    #[test]
    fn registration_yields_decoder() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = reg.make_decoder(&params).expect("round-4 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }
}
