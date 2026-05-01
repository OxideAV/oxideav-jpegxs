//! JPEG XS — ISO/IEC 21122 low-latency image codec for production / IP
//! video (SMPTE ST 2110-22).
//!
//! Round 1 ships:
//!
//! * A pure-Rust parser for the Part-1 codestream marker chain
//!   ([`codestream::parse`]) — SOC, CAP, PIH, CDT, WGT, plus optional
//!   COM / NLT / CWD / CTS / CRG, followed by one or more
//!   (SLH + entropy data) slices terminated by EOC. See
//!   [`codestream::Codestream`] for the captured geometry.
//! * A [`probe`] convenience that returns
//!   [`probe::JpegXsFileInfo`] (width × height × component count ×
//!   maximum bit depth × profile / level / colour transform id /
//!   lossless flag) without instantiating a decoder.
//! * Codec registration through
//!   [`oxideav_core::CodecRegistry::register`] so the framework can
//!   discover the codec under id `"jpegxs"`. The decoder factory
//!   currently returns `Error::Unsupported` — pixel decode lands in
//!   round 2 alongside the inverse DWT, the entropy coder, and the
//!   precinct walker (Annexes B–G).
//!
//! Round 1 caveats:
//!
//! * Slice boundaries are recovered by scanning forward for the next
//!   `FF 20` (SLH) or `FF 11` (EOC) marker pair. JPEG XS does not byte-
//!   stuff (Part-1 §A.3 NOTE 2), so the scan can over-shoot if the
//!   entropy-coded body happens to contain those byte sequences.
//!   Round-2 work replaces this with a length-driven walker once the
//!   precinct + packet header parsers (Annex C.2) land. Hand-built
//!   fixtures used by the test suite are crafted to avoid the
//!   collision.
//! * The CAP body is captured raw; the per-bit accessor that decodes
//!   Star-Tetrix / quadratic NLT / extended NLT / CWD / lossless /
//!   raw-mode-switch flags is deferred to round 2.

pub mod codestream;
pub mod component_table;
pub mod dwt;
pub mod markers;
pub mod picture_header;
pub mod probe;
pub mod slice_header;

pub use codestream::{Codestream, Slice};
pub use component_table::{Component, ComponentTable};
pub use markers::Marker;
pub use picture_header::PictureHeader;
pub use probe::{probe, JpegXsFileInfo};
pub use slice_header::SliceHeader;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder, Error, Result,
};

/// Public codec id string. Matches the aggregator feature name `jpegxs`.
pub const CODEC_ID_STR: &str = "jpegxs";

/// Register the JPEG XS decoder factory. Pixel decode is unimplemented
/// for round 1 — the factory returns `Error::Unsupported`. The encoder
/// slot is intentionally left unregistered.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("jpegxs_headers_only")
        .with_lossy(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

/// Decoder factory. Round 1 returns `Error::Unsupported`; the codestream
/// marker parser is exposed via [`codestream::parse`] / [`probe`] for
/// callers who only need geometry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Err(Error::Unsupported(
        "JPEG XS pixel decode not yet implemented".into(),
    ))
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
    fn registration_yields_unsupported_decoder() {
        let mut reg = CodecRegistry::new();
        register(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let res = reg.make_decoder(&params);
        let err = match res {
            Ok(_) => panic!("expected Err from round-1 decoder factory"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("not yet implemented"),
            "expected unimplemented error, got {msg}"
        );
    }
}
