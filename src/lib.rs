//! JPEG XS — ISO/IEC 21122 low-latency image codec for production / IP
//! video (SMPTE ST 2110-22).
//!
//! Round 4 ships a working end-to-end decoder for the single-component,
//! single-precinct subset of the standard:
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
//!   [`dequant`], and a wired-up [`Decoder`] in [`decoder`]. The
//!   decoder factory ([`make_decoder`]) returns a real decoder; on
//!   codestreams outside the round-4 subset it returns
//!   `Error::Unsupported` from `send_packet`.
//!
//! Round-4 supported subset (`make_decoder` accepts):
//!
//! * `Nc == 1` (single component), `sx == sy == 1`, `Cw == 0`.
//! * `NL,x ∈ {0, 1}`, `NL,y ∈ {0, 1}` (single-level inverse 2-D DWT
//!   only — multi-level cascade is round 5).
//! * `Cpih == 0` (no inverse colour transform — Annex F is round 5).
//! * `Qpih ∈ {0, 1}` (deadzone or uniform inverse quantizer).
//! * 8-bit output samples; the round-4 output mapping is the obvious
//!   linear path (`>> Fq` then add DC bias) — Annex G's full
//!   non-linearity / clip pipeline is round 5.
//!
//! Out of round-4 scope (returns `Error::Unsupported`): multi-component
//! configurations, 4:2:2 / 4:2:0 sampling, Annex F (RGB↔YCbCr,
//! Star-Tetrix), Annex G (NLT, DC-shift, clip), CAP-bit-driven feature
//! gating, multi-level wavelet decomposition (NL > 1), CWD-driven
//! component-dependent decomposition.

pub mod codestream;
pub mod component_table;
pub mod decoder;
pub mod dequant;
pub mod dwt;
pub mod entropy;
pub mod markers;
pub mod picture_header;
pub mod probe;
pub mod slice_header;
pub mod slice_walker;

pub use codestream::{Codestream, Slice};
pub use component_table::{Component, ComponentTable};
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
/// Round 4 wires a working decoder for the single-component, single-
/// precinct subset of the standard. Multi-component streams,
/// 4:2:2/4:2:0 sampling, multi-level wavelet decomposition (NL > 1),
/// inverse colour transforms (Annex F), and the full Annex G output
/// path arrive in round 5.
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

/// Decoder factory — see [`decoder::make_decoder`].
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
