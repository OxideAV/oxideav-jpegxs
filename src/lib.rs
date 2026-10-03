//! JPEG XS — ISO/IEC 21122 low-latency image codec for production / IP
//! video (SMPTE ST 2110-22): a complete Part-1 decoder (65/65 ISO/IEC
//! 21122-4 conformance vectors sample-exact), a Part-1 encoder across
//! the whole tool set, the Part-2 profile / level surface and the
//! Part-3 `.jxs` still-image file format.
//!
//! # Standalone use
//!
//! The crate follows the OxideAV image-crate API contract. With
//! `default-features = false` nothing but `std` is pulled in:
//!
//! ```no_run
//! let bytes = std::fs::read("in.jxs")?;
//! if oxideav_jpegxs::probe(&bytes) {
//!     let info = oxideav_jpegxs::info(&bytes)?;          // header only
//!     let img = oxideav_jpegxs::decode(&bytes)?;         // JpegXsImage, native layout
//!     let rgba: Vec<u8> = img.to_rgba8();
//!     let (w, h) = (img.width(), img.height());
//!     let opts = oxideav_jpegxs::EncodeOptions::default().with_quantization(2);
//!     let out = oxideav_jpegxs::encode_rgba8(w, h, &rgba, &opts)?;
//!     std::fs::write("out.jxs", out)?;
//!     let _ = info;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! * [`probe()`] / [`info`] / [`decode`] / [`decode_with`] /
//!   [`decode_rgb8`] / [`decode_rgba8`] / [`decode_from`] accept a bare
//!   ISO/IEC 21122-1 codestream or a `.jxs` file.
//! * [`encode`] / [`encode_rgb8`] / [`encode_rgba8`] / [`encode_to`]
//!   write a bare codestream, or a `.jxs` file with
//!   [`EncodeOptions::boxed`].
//! * [`JpegXsImage`] carries `width`, `height`, [`PixelFormat`],
//!   `planes`, [`ColorInfo`], [`Metadata`] and the significant
//!   `bit_depth`; [`JpegXsPixelFormat`] mirrors `oxideav_core::PixelFormat`
//!   by name (grey, planar GBR(A), planar YCbCr(A) at 8 / 10 / 12 / 14 /
//!   16 bits).
//! * [`decode_components`] / [`encode_components`] are the depth pair
//!   over [`Components`] — the planes exactly as the codestream carries
//!   them, for pictures without a contract layout (Star-Tetrix CFA,
//!   two- or five-plus-component sets) and for conformance comparison.
//! * [`inspect`] returns the raw header summary ([`JpegXsFileInfo`]) for
//!   any parseable stream.
//!
//! # Framework use
//!
//! With the default `registry` feature: [`register`] installs the codec
//! (decoder + encoder) and the `.jxs` extension into an
//! `oxideav_core::RuntimeContext`; [`make_decoder`] / [`make_encoder`]
//! are the factories; `From<JpegXsImage> for VideoFrame` and
//! [`JpegXsImage::from_video_frame`] bridge frames. The framework path
//! calls the standalone functions above — one implementation.
//!
//! The depth modules ([`codestream`], [`decoder`], [`encoder`], [`dwt`],
//! [`entropy`], [`profile`], [`signalling`], [`fileformat`], …) remain
//! public for callers who need the marker chain, the Annex-level
//! kernels, profile / level verification or the box builder.

pub mod api;
pub mod capabilities;
pub mod codestream;
pub mod colour_transform;
pub mod com;
pub mod component_table;
pub mod convert;
pub mod crg;
pub mod cts;
pub mod cwd;
pub mod decoder;
pub mod dequant;
pub mod dwt;
pub mod encoder;
pub mod entropy;
pub mod error;
pub mod fileformat;
pub mod image;
pub mod markers;
pub mod options;
pub mod output;
pub mod picture_header;
pub mod probe;
pub mod profile;
pub mod signalling;
pub mod slice_header;
pub mod slice_walker;

#[cfg(feature = "registry")]
pub mod registry;

#[cfg(feature = "registry")]
pub use registry::{
    __oxideav_entry, make_decoder, make_encoder, register, register_codecs, register_containers,
    register_registries,
};

// --- the contract surface -------------------------------------------------

pub use api::{
    decode, decode_components, decode_components_with, decode_from, decode_rgb8, decode_rgba8,
    decode_with, encode, encode_components, encode_rgb8, encode_rgba8, encode_to, info, probe,
};
pub use error::{Error, JpegXsError, Result};
pub use image::{
    ColorInfo, ColorModel, ColorRange, Components, DecodeOptions, ImageInfo, JpegXsImage,
    JpegXsPixelFormat, JpegXsPlane, Metadata, PixelFormat, Plane, RgbImage, RgbaImage,
};
pub use options::{EncodeOptions, Quantizer, RunMode, StarTetrixParams, Weights};
pub use probe::{inspect, JpegXsFileInfo};

// --- depth surface --------------------------------------------------------

pub use capabilities::{
    parse_capabilities, parse_capabilities_lossy, unsupported_cap_bits, Capabilities,
};
pub use codestream::{Codestream, Slice};
pub use com::{
    parse_com, ComMarker, TCOM_COPYRIGHT, TCOM_ENCODER_VENDOR, TCOM_VENDOR_SPECIFIC_MIN,
};
pub use component_table::{Component, ComponentTable};
pub use crg::{cfa_pattern_type, parse_crg, CrgEntry, CrgMarker};
pub use cts::{parse_cts, CtsExtent, CtsMarker};
pub use cwd::{parse_cwd, CwdMarker};
#[allow(deprecated)]
pub use encoder::{
    encode_image, encode_luma_8bit, encode_planar_cbr_target_bytes,
    encode_planar_cbr_target_bytes_highbd, encode_planar_cw, encode_planar_for_profile,
    encode_planar_for_profile_cbr_target_bytes, encode_planar_hsl_qslice,
    encode_planar_hsl_qslice_rp, encode_planar_hsl_qslice_rp_highbd,
    encode_planar_hsl_qslice_rp_target_bytes, encode_planar_hsl_qslice_rp_target_bytes_highbd,
    encode_planar_hsl_target_bytes, encode_planar_lossy_annex_h, encode_planar_qpr,
    encode_planar_qpr_rpr, encode_planar_qpr_rpr_target_bytes, encode_planar_rp_target_bytes,
    encode_planar_rpr, encode_planar_star_tetrix_annex_h, encode_planar_star_tetrix_highbd_lossy,
    encode_planar_subsampled_annex_h, encode_planar_subsampled_highbd,
    encode_planar_subsampled_highbd_lossy, encode_raw_luma, pick_q_slices_for_profile_target_bytes,
    pick_q_slices_for_target_bytes, pick_q_slices_rp_for_target_bytes,
    pick_q_slices_rp_for_target_bytes_highbd, pick_qpr_rpr_for_target_bytes,
    pick_rp_for_target_bytes,
};
#[allow(deprecated)]
pub use fileformat::decode_jxs_file;
pub use fileformat::{
    is_jxs_file, media_type, parse_jxs_file, write_jxs_file, BufferModelDescription, ChannelDef,
    ChannelDefinition, Cicp, ColourSpec, FileType, FrameRate, FrameRateDenominator, HeaderBox,
    ImageHeader, InterlaceMode, JxsFile, JxsFileBuilder, MasteringDisplayMetadata, ProfileLevel,
    SampleCharacteristics, SamplingStructure, TimeCode, VideoInformation, VideoTransportParameters,
    CODESTREAM_MAGIC, MEDIA_TYPE_CODESTREAM, MEDIA_TYPE_HEIF_IMAGE, MEDIA_TYPE_HEIF_SEQUENCE,
    MEDIA_TYPE_JXS,
};
pub use markers::Marker;
pub use output::{parse_nlt, NltParams};
pub use picture_header::PictureHeader;
pub use profile::{
    check_codestream as check_profile, check_codestream_size, check_level, classify_chroma,
    column_width, max_codestream_size, ChromaFormat, ColumnMode, Level, Profile, ProfileLimits,
    QpihAllowed, Sublevel,
};
pub use signalling::{
    declare_auto, declare_cbr, declare_cbr_padded, declare_level_sublevel, declare_profile,
    declare_vbr, insert_com, pad_to_size, pick_level, pick_profile, pick_sublevel,
    verify_declarations,
};
pub use slice_header::SliceHeader;
pub use slice_walker::{parse_wgt, BandWeight};

/// Public codec id string. Matches the aggregator feature name `jpegxs`.
pub const CODEC_ID_STR: &str = "jpegxs";

/// Historical standalone decode entry point: the codestream-order
/// component planes of one bare codestream.
///
/// [`decode`] is the contract form (native layout, colour, metadata,
/// `.jxs` files accepted); [`decode_components`] keeps the raw
/// component view this function returned.
#[deprecated(note = "use oxideav_jpegxs::decode or decode_components (IMAGE_CRATE_API)")]
pub fn decode_jpeg_xs(buf: &[u8]) -> Result<Components> {
    decode_components(buf)
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
    fn inspect_returns_geometry() {
        let buf = build_tiny_codestream();
        assert!(probe(&buf));
        let info = inspect(&buf).expect("inspect tiny codestream");
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
    fn probe_and_inspect_reject_non_jpegxs() {
        let buf = vec![0xff, 0xd8, 0x00, 0x00];
        assert!(!probe(&buf));
        assert!(inspect(&buf).is_none());
        let buf = vec![];
        assert!(!probe(&buf));
        assert!(inspect(&buf).is_none());
    }

    #[cfg(feature = "registry")]
    #[test]
    fn registration_yields_decoder() {
        use oxideav_core::{CodecId, CodecParameters, CodecRegistry};
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = reg.first_decoder(&params).expect("round-4 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }

    #[cfg(feature = "registry")]
    #[test]
    fn jxs_extension_resolves_to_jpegxs() {
        use oxideav_core::ContainerRegistry;
        let mut reg = ContainerRegistry::new();
        register_containers(&mut reg);
        // Canonical lower-case lookup.
        assert_eq!(reg.container_for_extension("jxs"), Some(CODEC_ID_STR));
        // Case-insensitive (the registry lower-cases both sides).
        assert_eq!(reg.container_for_extension("JXS"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("Jxs"), Some(CODEC_ID_STR));
        // Unrelated extensions do not collide.
        assert_eq!(reg.container_for_extension("jpg"), None);
    }

    #[cfg(feature = "registry")]
    #[test]
    fn register_via_runtime_context_installs_factories() {
        use oxideav_core::RuntimeContext;
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.codecs.decoder_ids().next().is_some(),
            "register(ctx) should install codec decoder factories"
        );
        assert_eq!(
            ctx.containers.container_for_extension("jxs"),
            Some(CODEC_ID_STR),
            "register(ctx) should install .jxs extension hint"
        );
    }
}
