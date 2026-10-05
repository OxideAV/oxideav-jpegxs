//! The crate-root contract functions: `probe`, `info`, `decode*`,
//! `encode*` plus the component-plane depth pair
//! [`decode_components`] / [`encode_components`].
//!
//! Everything here is `oxideav-core`-free. The `registry` adapter
//! ([`crate::registry`]) calls these same functions — one
//! implementation, two entry layers.
//!
//! Input is either a bare ISO/IEC 21122-1 codestream (`SOC` marker
//! first) or a `.jxs` still-image file (ISO/IEC 21122-3 Annex A, JPEG XS
//! Signature box first); every decode function accepts both.

use std::io::{Read, Write};

use crate::codestream::{self, Codestream};
use crate::error::{JpegXsError as Error, Result};
use crate::fileformat::{self, ChannelDef, Cicp, JxsFile, JxsFileBuilder};
use crate::image::{
    ColorInfo, ColorModel, ColorRange, Components, DecodeOptions, ImageInfo, JpegXsImage,
    JpegXsPixelFormat, Metadata, Plane, RgbImage, RgbaImage,
};
use crate::options::{EncodeOptions, Weights};

// ---------------------------------------------------------------------------
// probe / info
// ---------------------------------------------------------------------------

/// Signature sniff: `true` when `bytes` begins with a JPEG XS codestream
/// (`SOC` + `CAP` markers, `FF 10 FF 50`) or the JPEG XS Signature box
/// of a `.jxs` file. No allocation, no panic, `false` on short input.
pub fn probe(bytes: &[u8]) -> bool {
    bytes.starts_with(&fileformat::CODESTREAM_MAGIC) || fileformat::is_jxs_file(bytes)
}

/// Header-only summary: picture header, component table and (for a
/// `.jxs` file) the header boxes. No sample is decoded. `Unsupported`
/// when the picture has no contract layout (Star-Tetrix CFA, `Nc ∉ {1,
/// 3, 4}`, mixed bit depths, unusual sampling) — [`crate::inspect`] /
/// [`decode_components`] still cover such streams.
pub fn info(bytes: &[u8]) -> Result<ImageInfo> {
    let parsed = parse_input(bytes, &DecodeOptions::default().with_max_pixels(None))?;
    let (format, bit_depth) = layout_of(&parsed)?;
    let color = color_of(&parsed, format);
    let cs = &parsed.cs;
    Ok(ImageInfo {
        width: cs.pih.width(),
        height: cs.pih.height(),
        format,
        frames: 1,
        has_alpha: format.has_alpha(),
        color,
        has_icc: false,
        has_exif: parsed
            .file
            .as_ref()
            .is_some_and(|f| f.header.exif.is_some()),
        has_xmp: false,
        bit_depth,
        components: cs.pih.nc,
        cpih: cs.pih.cpih,
        profile: cs.pih.ppih,
        level: cs.pih.plev,
        lossless: cs.pih.is_lossless(),
        boxed: parsed.file.is_some(),
    })
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/// Decode the picture in its native contract layout with the default
/// [`DecodeOptions`].
pub fn decode(bytes: &[u8]) -> Result<JpegXsImage> {
    decode_with(bytes, &DecodeOptions::default())
}

/// Decode with explicit limits / strictness. Limits are enforced on the
/// picture header before any sample buffer is allocated.
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<JpegXsImage> {
    let parsed = parse_input(bytes, opts)?;
    let (format, bit_depth) = layout_of(&parsed)?;
    let color = color_of(&parsed, format);
    let metadata = metadata_of(&parsed);
    let comps = crate::decoder::decode_parsed(parsed.codestream, &parsed.cs)?;
    image_from_components(comps, format, bit_depth, color, metadata)
}

/// Decode to tightly packed 8-bit RGB.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage> {
    let img = decode(bytes)?;
    Ok(RgbImage::new(img.width, img.height, img.to_rgb8()))
}

/// Decode to tightly packed 8-bit RGBA (alpha from the fourth plane of
/// the `Gbrap*` / `Yuva*` layouts, opaque otherwise).
pub fn decode_rgba8(bytes: &[u8]) -> Result<RgbaImage> {
    let img = decode(bytes)?;
    Ok(RgbaImage::new(img.width, img.height, img.to_rgba8()))
}

/// Read `r` to end and [`decode`] it.
pub fn decode_from<R: Read>(mut r: R) -> Result<JpegXsImage> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    decode(&buf)
}

/// Decode the component planes exactly as the codestream carries them
/// (codestream component order, per-component bit depth and sampling,
/// no reordering). Works for every decodable stream, including the
/// layouts [`decode`] reports as `Unsupported`.
pub fn decode_components(bytes: &[u8]) -> Result<Components> {
    decode_components_with(bytes, &DecodeOptions::default())
}

/// [`decode_components`] with explicit limits / strictness.
pub fn decode_components_with(bytes: &[u8], opts: &DecodeOptions) -> Result<Components> {
    let parsed = parse_input(bytes, opts)?;
    crate::decoder::decode_parsed(parsed.codestream, &parsed.cs)
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// Encode an image in its own layout. Grey layouts become one
/// component; `Gbrp*` / `Gbrap*` become R, G, B(, A) components coded
/// through the reversible colour transform (`Cpih = 1`, unless
/// `opts.rct` is off); `Yuv*` / `Yuva*` become Y, Cb, Cr(, A) with the
/// layout's sampling factors and no transform. Nothing is converted
/// silently: every contract layout has a JPEG XS representation, so
/// `Unsupported` arises only from option combinations the encoder
/// cannot honour (see [`EncodeOptions`]).
///
/// With `opts.boxed` the codestream is wrapped in a `.jxs` file carrying
/// the image's [`ColorInfo`] as a CICP Colour Specification box, a
/// Channel Definition box for the alpha layouts and the Exif payload of
/// its [`Metadata`]; a bare codestream carries neither colour nor
/// metadata (ISO/IEC 21122-1 has no field for them).
pub fn encode(image: &JpegXsImage, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let comps = components_from_image(image, opts.rct)?;
    let cs = encode_components_bare(&comps, opts)?;
    if !opts.boxed {
        return Ok(cs);
    }
    wrap_in_jxs(
        &cs,
        &image.color,
        image.format.has_alpha(),
        image.metadata.exif.as_deref(),
    )
}

/// Wrap a bare codestream in a `.jxs` file (ISO/IEC 21122-3 Annex A):
/// `color` becomes the CICP Colour Specification box (an `Unspecified`
/// range becomes the limited-range flag — the CICP `V` byte has no
/// "unspecified"), `has_alpha` adds the Channel Definition box naming
/// channel 3 as whole-image opacity, `exif` the Exif box. Shared by
/// [`encode`] (`boxed = true`) and the framework `jxs` muxer.
pub(crate) fn wrap_in_jxs(
    cs: &[u8],
    color: &ColorInfo,
    has_alpha: bool,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let cicp = Cicp {
        colour_primaries: u16::from(color.primaries),
        transfer_characteristics: u16::from(color.transfer),
        matrix_coefficients: u16::from(color.matrix),
        full_range: matches!(color.range, ColorRange::Full),
    };
    let mut builder = JxsFileBuilder::new(cicp);
    if has_alpha {
        // A.5.4.4: colour channels associated with colour 1..=3, the
        // fourth channel is whole-image opacity.
        builder = builder.channels(vec![
            ChannelDef {
                channel: 0,
                typ: 0,
                assoc: 1,
            },
            ChannelDef {
                channel: 1,
                typ: 0,
                assoc: 2,
            },
            ChannelDef {
                channel: 2,
                typ: 0,
                assoc: 3,
            },
            ChannelDef {
                channel: 3,
                typ: 1,
                assoc: 0,
            },
        ]);
    }
    if let Some(exif) = exif {
        builder = builder.exif(exif.to_vec());
    }
    builder.build(cs)
}

/// Encode a component set exactly as given: `Nc = planes.len()`
/// components in codestream order with their bit depths, sampling
/// factors and `cpih`. This is the universal encoder funnel; every
/// historical `encode_planar_*` entry point is a special case of it.
/// With `opts.boxed` the file's CICP box is written as "unspecified"
/// (a component set carries no colour description).
pub fn encode_components(components: &Components, opts: &EncodeOptions) -> Result<Vec<u8>> {
    let cs = encode_components_bare(components, opts)?;
    if !opts.boxed {
        return Ok(cs);
    }
    JxsFileBuilder::new(Cicp {
        colour_primaries: u16::from(ColorInfo::UNSPECIFIED),
        transfer_characteristics: u16::from(ColorInfo::UNSPECIFIED),
        matrix_coefficients: u16::from(ColorInfo::UNSPECIFIED),
        full_range: false,
    })
    .colourspace_unknown(true)
    .build(&cs)
}

/// Encode tightly packed 8-bit RGB as planar RGB (`Gbrp8`, reversible
/// colour transform) with the sRGB colour description.
pub fn encode_rgb8(width: u32, height: u32, rgb: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    encode(&JpegXsImage::from_rgb8(width, height, rgb.to_vec())?, opts)
}

/// Encode tightly packed 8-bit RGBA as planar RGB + alpha (`Gbrap8`:
/// reversible colour transform on R, G, B, the alpha plane as the
/// pass-through fourth component).
pub fn encode_rgba8(width: u32, height: u32, rgba: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    encode(
        &JpegXsImage::from_rgba8(width, height, rgba.to_vec())?,
        opts,
    )
}

/// [`encode`] into a writer.
pub fn encode_to<W: Write>(image: &JpegXsImage, opts: &EncodeOptions, mut w: W) -> Result<()> {
    let bytes = encode(image, opts)?;
    w.write_all(&bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Input parsing + limits
// ---------------------------------------------------------------------------

/// A located and header-parsed input.
struct Parsed<'a> {
    /// The `.jxs` box structure, when the input is a file.
    file: Option<JxsFile>,
    /// The bare codestream bytes (the whole input, or the Contiguous
    /// Codestream box payload).
    codestream: &'a [u8],
    /// Parsed marker chain.
    cs: Codestream,
}

fn parse_input<'a>(bytes: &'a [u8], opts: &DecodeOptions) -> Result<Parsed<'a>> {
    if let Some(max) = opts.max_bytes {
        if bytes.len() as u64 > max {
            return Err(Error::limit(format!(
                "jpegxs: input of {} bytes exceeds max_bytes {max}",
                bytes.len()
            )));
        }
    }
    let (file, codestream): (Option<JxsFile>, &[u8]) = if fileformat::is_jxs_file(bytes) {
        let file = fileformat::parse_jxs_file(bytes)?;
        let cs = file.codestream(bytes);
        (Some(file), cs)
    } else if bytes.starts_with(&[0xff, 0x10]) {
        (None, bytes)
    } else {
        return Err(Error::invalid(
            "jpegxs: input is neither a JPEG XS codestream (SOC marker) nor a .jxs file \
             (JPEG XS Signature box)",
        ));
    };
    let cs = codestream::parse(codestream)?;
    let (w, h) = (cs.pih.width(), cs.pih.height());
    if let Some(max) = opts.max_width {
        if w > max {
            return Err(Error::limit(format!(
                "jpegxs: width {w} exceeds max_width {max}"
            )));
        }
    }
    if let Some(max) = opts.max_height {
        if h > max {
            return Err(Error::limit(format!(
                "jpegxs: height {h} exceeds max_height {max}"
            )));
        }
    }
    if let Some(max) = opts.max_pixels {
        let px = u64::from(w) * u64::from(h);
        if px > max {
            return Err(Error::limit(format!(
                "jpegxs: {w}x{h} = {px} pixels exceeds max_pixels {max}"
            )));
        }
    }
    if opts.strict {
        match cs.eoc_offset {
            None => return Err(Error::invalid("jpegxs: strict — EOC marker missing")),
            Some(eoc) if eoc + 2 != codestream.len() => {
                return Err(Error::invalid(format!(
                    "jpegxs: strict — {} trailing byte(s) after EOC",
                    codestream.len() - (eoc + 2)
                )));
            }
            Some(_) => {}
        }
    }
    if let Some(file) = &file {
        fileformat::check_file_consistency(file, &cs, codestream.len())?;
    }
    Ok(Parsed {
        file,
        codestream,
        cs,
    })
}

// ---------------------------------------------------------------------------
// Layout derivation
// ---------------------------------------------------------------------------

/// The CICP code points of the first CICP colour specification box, if
/// the input is a file that has one.
fn cicp_of(parsed: &Parsed<'_>) -> Option<Cicp> {
    parsed
        .file
        .as_ref()?
        .header
        .colour_specs
        .iter()
        .find_map(|c| c.cicp)
}

/// `true` when `bytes` is a `.jxs` file whose header carries a CICP
/// colour specification box — the one case where the FILE (not a
/// crate convention) defines the colour semantics. The registry
/// decoder stamps the frame's colour signal only then; a bare
/// codestream or a `.jxs` file with no CICP box leaves the frame
/// unstamped and the layout default on [`JpegXsImage::color`].
#[cfg(feature = "registry")]
pub(crate) fn carries_cicp(bytes: &[u8]) -> bool {
    fileformat::is_jxs_file(bytes)
        && fileformat::parse_jxs_file(bytes)
            .map(|f| f.header.colour_specs.iter().any(|c| c.cicp.is_some()))
            .unwrap_or(false)
}

/// Which contract layout the codestream decodes to, with its bit depth.
///
/// * `Nc = 1` → grey.
/// * `Nc ∈ {3, 4}`, `Cpih = 1` (RCT) → planar RGB(A).
/// * `Nc ∈ {3, 4}`, `Cpih = 0` → YCbCr(A) with the chroma sampling of
///   components 1 / 2 — unless the `.jxs` CICP box names the identity
///   matrix (`0`) at 4:4:4, which is RGB(A) without a transform.
/// * `Cpih = 3` (Star-Tetrix CFA), `Nc ∈ {2, 5..=8}`, mixed component
///   bit depths and sampling other than 4:4:4 / 4:2:2 / 4:2:0 →
///   `Unsupported` (the component view still decodes them).
fn layout_of(parsed: &Parsed<'_>) -> Result<(JpegXsPixelFormat, u8)> {
    let cs = &parsed.cs;
    let comps = &cs.cdt.components;
    let nc = comps.len();
    let bd = comps.first().map_or(8, |c| c.bit_depth);
    if comps.iter().any(|c| c.bit_depth != bd) {
        return Err(Error::unsupported(format!(
            "jpegxs: mixed component bit depths {:?} have no contract layout; use decode_components",
            comps.iter().map(|c| c.bit_depth).collect::<Vec<_>>()
        )));
    }
    let full_rate = |i: usize| comps.get(i).is_some_and(|c| c.sx == 1 && c.sy == 1);
    let no_view = |what: &str| {
        Error::unsupported(format!(
            "jpegxs: {what} has no contract layout; use decode_components"
        ))
    };
    let (model, alpha, divisors) = match (nc, cs.pih.cpih) {
        (1, _) => (ColorModel::Gray, false, (1, 1)),
        (3 | 4, 3) => return Err(no_view("a Star-Tetrix CFA picture (Cpih=3)")),
        (3 | 4, 1) => {
            if !(0..nc).all(full_rate) {
                return Err(no_view("a sub-sampled RCT picture"));
            }
            (ColorModel::Rgb, nc == 4, (1, 1))
        }
        (3 | 4, 0) => {
            if !full_rate(0) || (nc == 4 && !full_rate(3)) {
                return Err(no_view("a picture whose component 0 / 3 is sub-sampled"));
            }
            if comps[1].sx != comps[2].sx || comps[1].sy != comps[2].sy {
                return Err(no_view(
                    "a picture whose chroma components differ in sampling",
                ));
            }
            let divisors = (usize::from(comps[1].sx), usize::from(comps[1].sy));
            let identity = cicp_of(parsed).is_some_and(|c| c.matrix_coefficients == 0);
            if identity && divisors == (1, 1) {
                (ColorModel::Rgb, nc == 4, (1, 1))
            } else {
                (ColorModel::YCbCr, nc == 4, divisors)
            }
        }
        (n, _) => return Err(no_view(&format!("a {n}-component picture"))),
    };
    JpegXsPixelFormat::for_layout(model, alpha, divisors, bd)
        .map(|f| (f, bd))
        .ok_or_else(|| {
            no_view(&format!(
                "{} sampling {}:{}",
                match model {
                    ColorModel::Gray => "grey",
                    ColorModel::Rgb => "RGB",
                    ColorModel::YCbCr => "YCbCr",
                },
                divisors.0,
                divisors.1
            ))
        })
}

/// Colour description: the `.jxs` CICP box verbatim, else the layout's
/// documented default.
fn color_of(parsed: &Parsed<'_>, format: JpegXsPixelFormat) -> ColorInfo {
    match cicp_of(parsed) {
        Some(c) => cicp_color(&c),
        None => ColorInfo::default_for(format),
    }
}

/// A CICP box read verbatim into [`ColorInfo`].
fn cicp_color(c: &Cicp) -> ColorInfo {
    let cp = |v: u16| u8::try_from(v).unwrap_or(ColorInfo::UNSPECIFIED);
    ColorInfo::new(
        if c.full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        },
        cp(c.colour_primaries),
        cp(c.transfer_characteristics),
        cp(c.matrix_coefficients),
    )
}

/// What the framework demuxer publishes before any sample is decoded —
/// [`info`] with the layout made optional and the colour reduced to
/// what the FILE signals.
#[cfg(feature = "registry")]
pub(crate) struct Described {
    pub width: u32,
    pub height: u32,
    /// The contract layout, `None` when the picture has none
    /// (Star-Tetrix CFA, `Nc ∉ {1, 3, 4}`, mixed depths, odd sampling):
    /// the stream still opens, the decoder reports `Unsupported`.
    pub format: Option<JpegXsPixelFormat>,
    /// The `.jxs` CICP box, when the file has one — the only colour
    /// signal a frame may be stamped with (round-470 ruling).
    pub cicp: Option<ColorInfo>,
    /// The `.jxs` header carries an Exif box.
    pub has_exif: bool,
    /// Box-wrapped `.jxs` file (vs a bare codestream).
    pub boxed: bool,
}

/// Header-only walk for the framework demuxer. Only an `Unsupported`
/// layout verdict is absorbed — malformed input is an error here as in
/// [`info`].
#[cfg(feature = "registry")]
pub(crate) fn describe(bytes: &[u8]) -> Result<Described> {
    let parsed = parse_input(bytes, &DecodeOptions::default().with_max_pixels(None))?;
    let format = match layout_of(&parsed) {
        Ok((f, _)) => Some(f),
        Err(Error::Unsupported(_)) => None,
        Err(e) => return Err(e),
    };
    Ok(Described {
        width: parsed.cs.pih.width(),
        height: parsed.cs.pih.height(),
        format,
        cicp: cicp_of(&parsed).map(|c| cicp_color(&c)),
        has_exif: parsed
            .file
            .as_ref()
            .is_some_and(|f| f.header.exif.is_some()),
        boxed: parsed.file.is_some(),
    })
}

/// The bare codestream inside `bytes`: the `.jxs` Contiguous Codestream
/// box payload, or `bytes` itself.
#[cfg(feature = "registry")]
pub(crate) fn bare_codestream(bytes: &[u8]) -> Result<&[u8]> {
    if fileformat::is_jxs_file(bytes) {
        let file = fileformat::parse_jxs_file(bytes)?;
        Ok(file.codestream(bytes))
    } else if bytes.starts_with(&[0xff, 0x10]) {
        Ok(bytes)
    } else {
        Err(Error::invalid(
            "jpegxs: input is neither a JPEG XS codestream (SOC marker) nor a .jxs file \
             (JPEG XS Signature box)",
        ))
    }
}

fn metadata_of(parsed: &Parsed<'_>) -> Metadata {
    let mut m = Metadata::new();
    if let Some(exif) = parsed.file.as_ref().and_then(|f| f.header.exif.clone()) {
        m.exif = Some(exif);
    }
    m
}

/// Codestream-order components → contract image: RGB layouts reorder
/// R, G, B(, A) to G, B, R(, A); everything else keeps its order.
fn image_from_components(
    comps: Components,
    format: JpegXsPixelFormat,
    bit_depth: u8,
    color: ColorInfo,
    metadata: Metadata,
) -> Result<JpegXsImage> {
    let Components {
        width,
        height,
        mut planes,
        ..
    } = comps;
    if format.is_rgb() && planes.len() >= 3 {
        // R, G, B(, A) → G, B, R(, A).
        planes.swap(0, 1); // G, R, B
        planes.swap(1, 2); // G, B, R
    }
    let mut img = JpegXsImage::new(width, height, format, planes)?;
    img.color = color;
    img.metadata = metadata;
    img.bit_depth = bit_depth;
    Ok(img)
}

/// Contract image → codestream-order components (tight strides).
fn components_from_image(image: &JpegXsImage, rct: bool) -> Result<Components> {
    let format = image.format;
    if image.planes.len() != format.plane_count() {
        return Err(Error::invalid(format!(
            "jpegxs encoder: {} plane(s) do not fit {format} ({} expected)",
            image.planes.len(),
            format.plane_count()
        )));
    }
    let nominal = format.bytes_per_sample() * 8;
    if (nominal == 8) != (image.bit_depth == 8) || !(8..=16).contains(&image.bit_depth) {
        return Err(Error::invalid(format!(
            "jpegxs encoder: bit depth {} does not fit the {nominal}-bit storage of {format}",
            image.bit_depth
        )));
    }
    let bps = format.bytes_per_sample();
    let (dh, dv) = format.chroma_divisors();
    let mut planes = Vec::with_capacity(image.planes.len());
    let mut sampling = Vec::with_capacity(image.planes.len());
    for (i, p) in image.planes.iter().enumerate() {
        let (w, h) = format.plane_dimensions(image.width, image.height, i);
        let tight = w * bps;
        if p.stride < tight || p.data.len() < p.stride * h {
            return Err(Error::invalid(format!(
                "jpegxs encoder: plane {i} geometry does not fit {}x{} {format}",
                image.width, image.height
            )));
        }
        let data = if p.stride == tight {
            p.data.clone()
        } else {
            let mut out = Vec::with_capacity(tight * h);
            for row in p.data.chunks(p.stride).take(h) {
                out.extend_from_slice(&row[..tight]);
            }
            out
        };
        planes.push(Plane::new(tight, data));
        sampling.push(if i == 1 || i == 2 {
            (dh as u8, dv as u8)
        } else {
            (1, 1)
        });
    }
    if format.is_rgb() {
        // G, B, R(, A) → R, G, B(, A).
        planes.swap(0, 2); // R, B, G
        planes.swap(1, 2); // R, G, B
    }
    let cpih = if format.is_rgb() && rct { 1 } else { 0 };
    Components::new(
        image.width,
        image.height,
        cpih,
        vec![image.bit_depth; planes.len()],
        sampling,
        planes,
    )
}

// ---------------------------------------------------------------------------
// The encoder funnel
// ---------------------------------------------------------------------------

/// Pick `NL,x` / `NL,y` when the options leave them automatic: the
/// largest `NL,x ∈ 1..=5` the width admits (Table 11: `Wf ≥ max(sx) ×
/// 2^NL,x`) and `NL,y = 1` when the height admits it.
fn pick_levels(
    width: u32,
    height: u32,
    max_sx: u32,
    max_sy: u32,
    opts: &EncodeOptions,
) -> (u8, u8) {
    let nlx = opts.levels_x.unwrap_or_else(|| {
        let mut n = 1u8;
        while n < 5 && (max_sx << (n + 1)) <= width {
            n += 1;
        }
        n
    });
    // A vertically sub-sampled component needs NL,y >= 1 (Annex B.2:
    // N'L,y[i] = NL,y - log2(sy[i]) must stay >= 0), so pick 1 whenever
    // the height admits it or the sampling demands it; the encoder's
    // Table 11 gate reports the exact violation otherwise.
    let nly = opts.levels_y.unwrap_or({
        if nlx >= 1 && (max_sy > 1 || (max_sy << 1) <= height) {
            1
        } else {
            0
        }
    });
    (nlx, nly)
}

/// Planes in the wire byte format → `u16` samples (the profile /
/// high-bit-depth CBR entry points take `u16` planes).
fn planes_u16(components: &Components) -> Vec<Vec<u16>> {
    components
        .planes
        .iter()
        .zip(&components.bit_depths)
        .map(|(p, &bd)| {
            if bd > 8 {
                p.data
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                p.data.iter().map(|&b| u16::from(b)).collect()
            }
        })
        .collect()
}

#[allow(deprecated)] // the profile / CBR pickers are deprecated as public entry points only
fn encode_components_bare(c: &Components, opts: &EncodeOptions) -> Result<Vec<u8>> {
    // Re-validate: the fields are public, so a caller may have mutated
    // the set after `Components::new`.
    let c = Components::new(
        c.width,
        c.height,
        c.cpih,
        c.bit_depths.clone(),
        c.sampling.clone(),
        c.planes.clone(),
    )?;
    if opts.quantization > 15 {
        return Err(Error::invalid(format!(
            "jpegxs encoder: quantization {} outside 0..=15",
            opts.quantization
        )));
    }
    if !c.uniform_bit_depth() {
        return Err(Error::unsupported(format!(
            "jpegxs encoder: mixed component bit depths {:?} (one B[i] per codestream)",
            c.bit_depths
        )));
    }
    let bd = c.bit_depths[0];
    let nc = c.num_components();
    let width = c.width as u16;
    let height = c.height as u16;
    let sx: Vec<u8> = c.sampling.iter().map(|s| s.0).collect();
    let sy: Vec<u8> = c.sampling.iter().map(|s| s.1).collect();
    let max_sx = u32::from(sx.iter().copied().max().unwrap_or(1));
    let max_sy = u32::from(sy.iter().copied().max().unwrap_or(1));
    let (nlx, nly) = pick_levels(c.width, c.height, max_sx, max_sy, opts);
    let qpih = opts.quantizer.qpih();
    let q = opts.quantization;
    let st = opts.star_tetrix.unwrap_or_default();
    let planes: &[Vec<u8>] = &c.planes.iter().map(|p| p.data.clone()).collect::<Vec<_>>();

    let generic_only = opts.nlt.is_some()
        || opts.weights != Weights::Default
        || opts.column_width != 0
        || opts.suppressed_components != 0
        || opts.sign_packet
        || opts.run_mode.rm() != 0
        || opts.refinement != 0
        || opts.high_precision
        || !opts.q_slices.is_empty()
        || !opts.q_precincts.is_empty()
        || !opts.r_precincts.is_empty()
        || opts.star_tetrix.is_some();

    // Profile-shaped stream (optionally constant bitrate).
    if let Some(profile) = opts.profile {
        if generic_only {
            return Err(Error::unsupported(
                "jpegxs encoder: profile shaping composes with quantization / quantizer / \
                 levels / target_bytes only (NLT, weights, Cw, Sd, Fs, Rm, R[p], high \
                 precision, per-slice / per-precinct overrides and Star-Tetrix are not \
                 profile-shaped)",
            ));
        }
        let u16_planes = planes_u16(&c);
        return if let Some(target) = opts.target_bytes {
            crate::encoder::encode_planar_for_profile_cbr_target_bytes(
                profile,
                width,
                height,
                nc,
                c.cpih,
                nlx,
                nly,
                bd,
                qpih,
                &sx,
                &sy,
                target,
                &u16_planes,
            )
            .map(|(cs, _, _, _)| cs)
        } else {
            crate::encoder::encode_planar_for_profile(
                profile,
                width,
                height,
                nc,
                c.cpih,
                nlx,
                nly,
                bd,
                qpih,
                q,
                &sx,
                &sy,
                false,
                &u16_planes,
            )
            .map(|(cs, _, _)| cs)
        };
    }

    // Constant bitrate without a profile: the 4:4:4 rate pickers.
    if let Some(target) = opts.target_bytes {
        if generic_only || qpih != 0 {
            return Err(Error::unsupported(
                "jpegxs encoder: target_bytes without a profile composes with levels and \
                 slice_height only",
            ));
        }
        if sx.iter().any(|&v| v != 1) || sy.iter().any(|&v| v != 1) {
            return Err(Error::unsupported(
                "jpegxs encoder: target_bytes on a chroma-sub-sampled picture needs a \
                 profile (set EncodeOptions::profile)",
            ));
        }
        if c.cpih > 1 {
            return Err(Error::unsupported(
                "jpegxs encoder: target_bytes needs Cpih 0 or 1",
            ));
        }
        let hsl = opts.slice_height.unwrap_or(0);
        return if bd == 8 {
            crate::encoder::encode_planar_cbr_target_bytes(
                width, height, nc, c.cpih, nlx, nly, hsl, target, planes,
            )
            .map(|(cs, _)| cs)
        } else {
            let u16_planes = planes_u16(&c);
            crate::encoder::encode_planar_cbr_target_bytes_highbd(
                width,
                height,
                nc,
                c.cpih,
                nlx,
                nly,
                bd,
                hsl,
                target,
                &u16_planes,
            )
            .map(|(cs, _, _)| cs)
        };
    }

    // The generic funnel.
    let sd = opts.suppressed_components;
    let (gains, priorities) = match opts.weights {
        Weights::Default => (Vec::new(), Vec::new()),
        Weights::AnnexH => {
            crate::encoder::annex_h_weights(nc, c.cpih, sd, nlx, nly, st.cf, &sx, &sy)
                .unwrap_or_default()
        }
    };
    crate::encoder::encode_planar_inner_bd(
        width,
        height,
        nc,
        bd,
        c.cpih,
        nlx,
        nly,
        if opts.high_precision { 8 } else { 0 },
        q,
        &sx,
        &sy,
        st.e1,
        st.e2,
        st.cf,
        st.ct,
        opts.nlt,
        gains,
        priorities,
        opts.column_width,
        sd,
        u8::from(opts.sign_packet),
        opts.slice_height.unwrap_or(0),
        qpih,
        opts.refinement,
        opts.q_slices.clone(),
        opts.q_precincts.clone(),
        opts.r_precincts.clone(),
        planes,
        opts.run_mode.rm(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ColorRange;
    use crate::options::{Quantizer, RunMode};
    use crate::profile::Profile;

    fn gradient(w: u32, h: u32, seed: usize) -> Vec<u8> {
        (0..(w * h) as usize)
            .map(|i| ((i * 7 + seed * 13) % 251) as u8)
            .collect()
    }

    fn deep(w: u32, h: u32, bits: u8, seed: usize) -> Vec<u8> {
        let max = (1u32 << bits) - 1;
        (0..(w * h) as usize)
            .flat_map(|i| (((i * 97 + seed * 31) as u32) % (max + 1)).to_le_bytes()[..2].to_vec())
            .collect()
    }

    #[test]
    fn probe_is_a_pure_sniff() {
        assert!(!probe(&[]));
        assert!(!probe(&[0xff]));
        assert!(!probe(&[0xff, 0x10]));
        assert!(probe(&[0xff, 0x10, 0xff, 0x50]));
        assert!(!probe(&[0xff, 0xd8, 0xff, 0xe0]));
        assert!(probe(&fileformat::SIGNATURE_BOX));
        assert!(!probe(&fileformat::SIGNATURE_BOX[..11]));
    }

    #[test]
    fn rgb8_round_trip_is_lossless_bare_and_boxed() {
        let (w, h) = (17u32, 9u32);
        let rgb: Vec<u8> = (0..(w * h * 3) as usize)
            .map(|i| (i * 5 % 256) as u8)
            .collect();
        let bare = encode_rgb8(w, h, &rgb, &EncodeOptions::default()).unwrap();
        assert!(probe(&bare));
        let i = info(&bare).unwrap();
        assert_eq!((i.width, i.height), (w, h));
        assert_eq!(i.format, JpegXsPixelFormat::Gbrp8);
        assert_eq!(i.frames, 1);
        assert!(!i.has_alpha && !i.boxed && i.lossless);
        assert_eq!(i.cpih, 1);
        assert_eq!(i.color, ColorInfo::default_for(JpegXsPixelFormat::Gbrp8));
        let img = decode(&bare).unwrap();
        assert_eq!(img.format, JpegXsPixelFormat::Gbrp8);
        assert_eq!(img.to_rgb8(), rgb);
        assert_eq!(decode_rgb8(&bare).unwrap().data, rgb);
        let rgba = decode_rgba8(&bare).unwrap();
        assert_eq!(rgba.data.len(), (w * h * 4) as usize);
        assert!(rgba.data.chunks_exact(4).all(|p| p[3] == 255));
        // Boxed: colour + metadata survive; the image round-trips whole.
        let src = JpegXsImage::from_rgb8(w, h, rgb.clone())
            .unwrap()
            .with_metadata(Metadata::new().with_exif(vec![0x4d, 0x4d, 0, 0x2a]));
        let boxed = encode(&src, &EncodeOptions::default().with_boxed(true)).unwrap();
        assert!(probe(&boxed));
        let i = info(&boxed).unwrap();
        assert!(i.boxed && i.has_exif && !i.has_icc && !i.has_xmp);
        assert_eq!(i.color, ColorInfo::srgb());
        let back = decode(&boxed).unwrap();
        assert_eq!(back, src);
        // decode_from / encode_to.
        let mut out = Vec::new();
        encode_to(&src, &EncodeOptions::default(), &mut out).unwrap();
        assert_eq!(out, bare);
        let via_reader = decode_from(std::io::Cursor::new(&boxed)).unwrap();
        assert_eq!(via_reader, src);
    }

    #[test]
    fn rgba8_round_trip_carries_alpha() {
        let (w, h) = (8u32, 6u32);
        let rgba: Vec<u8> = (0..(w * h * 4) as usize)
            .map(|i| (i * 11 % 256) as u8)
            .collect();
        let bytes = encode_rgba8(w, h, &rgba, &EncodeOptions::default()).unwrap();
        let i = info(&bytes).unwrap();
        assert_eq!(i.format, JpegXsPixelFormat::Gbrap8);
        assert!(i.has_alpha);
        let img = decode(&bytes).unwrap();
        assert_eq!(img.to_rgba8(), rgba);
        let boxed = encode(
            &JpegXsImage::from_rgba8(w, h, rgba.clone()).unwrap(),
            &EncodeOptions::default().with_boxed(true),
        )
        .unwrap();
        let file = fileformat::parse_jxs_file(&boxed).unwrap();
        let cdef = file
            .header
            .channel_def
            .expect("alpha layouts write a cdef box");
        assert_eq!(cdef.channels[3].typ, 1);
        assert_eq!(decode(&boxed).unwrap().to_rgba8(), rgba);
    }

    #[test]
    fn gray_layouts_round_trip_at_every_depth() {
        let (w, h) = (12u32, 5u32);
        let g8 = JpegXsImage::new(
            w,
            h,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(w as usize, gradient(w, h, 1))],
        )
        .unwrap();
        let bytes = encode(&g8, &EncodeOptions::default()).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back, g8);
        assert_eq!(back.as_bytes(), g8.as_bytes());
        for (bits, fmt) in [
            (9, JpegXsPixelFormat::Gray16Le),
            (10, JpegXsPixelFormat::Gray10Le),
            (12, JpegXsPixelFormat::Gray12Le),
            (14, JpegXsPixelFormat::Gray16Le),
            (16, JpegXsPixelFormat::Gray16Le),
        ] {
            let img = JpegXsImage::new(
                w,
                h,
                fmt,
                vec![Plane::new(w as usize * 2, deep(w, h, bits, 2))],
            )
            .unwrap()
            .with_bit_depth(bits)
            .unwrap();
            let bytes = encode(&img, &EncodeOptions::default()).unwrap();
            let i = info(&bytes).unwrap();
            assert_eq!((i.format, i.bit_depth), (fmt, bits), "bits={bits}");
            let back = decode(&bytes).unwrap();
            assert_eq!(back, img, "bits={bits}");
            assert_eq!(back.to_rgb8().len(), (w * h * 3) as usize);
        }
    }

    #[test]
    fn ycbcr_layouts_round_trip() {
        let (w, h) = (16u32, 8u32);
        for fmt in [
            JpegXsPixelFormat::Yuv444P,
            JpegXsPixelFormat::Yuv422P,
            JpegXsPixelFormat::Yuv420P,
            JpegXsPixelFormat::Yuv422P10Le,
            JpegXsPixelFormat::Yuv444P12Le,
            JpegXsPixelFormat::Yuv420P16Le,
            JpegXsPixelFormat::Yuva444P,
            JpegXsPixelFormat::Yuva422P12Le,
        ] {
            let bits = fmt.nominal_bits();
            let planes: Vec<Plane> = (0..fmt.plane_count())
                .map(|i| {
                    let (pw, ph) = fmt.plane_dimensions(w, h, i);
                    let data = if bits > 8 {
                        deep(pw as u32, ph as u32, bits, i)
                    } else {
                        gradient(pw as u32, ph as u32, i)
                    };
                    Plane::new(pw * fmt.bytes_per_sample(), data)
                })
                .collect();
            let img = JpegXsImage::new(w, h, fmt, planes).unwrap();
            let bytes = encode(&img, &EncodeOptions::default()).unwrap();
            let i = info(&bytes).unwrap();
            assert_eq!(i.format, fmt, "{fmt}");
            assert_eq!(i.has_alpha, fmt.has_alpha());
            assert_eq!(i.cpih, 0);
            assert!(i.color.is_unspecified());
            let back = decode(&bytes).unwrap();
            assert_eq!(back, img, "{fmt}");
            assert_eq!(back.to_rgba8().len(), (w * h * 4) as usize);
            // Through the box, the CICP range flag comes back as Limited
            // (the V byte has no "unspecified"), everything else exact.
            let boxed = encode(&img, &EncodeOptions::default().with_boxed(true)).unwrap();
            let back = decode(&boxed).unwrap();
            assert_eq!(back.planes, img.planes);
            assert_eq!(back.color.range, ColorRange::Limited);
        }
    }

    #[test]
    fn rgb_without_rct_decodes_as_rgb_only_through_the_box() {
        let (w, h) = (6u32, 4u32);
        let src = JpegXsImage::from_rgb8(w, h, gradient(w * 3, h, 3)).unwrap();
        let bare = encode(&src, &EncodeOptions::default().with_rct(false)).unwrap();
        let i = info(&bare).unwrap();
        assert_eq!(i.cpih, 0);
        assert_eq!(
            i.format,
            JpegXsPixelFormat::Yuv444P,
            "bare: no transform → YCbCr 4:4:4"
        );
        let boxed = encode(
            &src,
            &EncodeOptions::default().with_rct(false).with_boxed(true),
        )
        .unwrap();
        let back = decode(&boxed).unwrap();
        assert_eq!(back.format, JpegXsPixelFormat::Gbrp8, "CICP matrix 0 → RGB");
        assert_eq!(back, src);
    }

    #[test]
    fn padded_strides_are_repacked_on_encode() {
        let (w, h) = (5u32, 3u32);
        let mut data = Vec::new();
        for y in 0..h as usize {
            data.extend((0..w as usize).map(|x| (x + y * 10) as u8));
            data.extend_from_slice(&[0xEE, 0xEE, 0xEE]); // padding
        }
        let img =
            JpegXsImage::new(w, h, JpegXsPixelFormat::Gray8, vec![Plane::new(8, data)]).unwrap();
        let back = decode(&encode(&img, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.planes[0].stride, 5);
        assert_eq!(back.to_rgb8(), img.to_rgb8());
    }

    #[test]
    fn lossy_options_decode_and_profile_shaping_signs() {
        let (w, h) = (64u32, 32u32);
        let img = JpegXsImage::from_rgb8(w, h, gradient(w * 3, h, 4)).unwrap();
        let lossy = encode(&img, &EncodeOptions::default().with_quantization(4)).unwrap();
        assert!(lossy.len() < encode(&img, &EncodeOptions::default()).unwrap().len());
        let i = info(&lossy).unwrap();
        assert!(!i.lossless || i.cpih == 1); // Fq stays 0 on the integer path
        assert_eq!(decode(&lossy).unwrap().format, JpegXsPixelFormat::Gbrp8);
        // Uniform quantiser, separate sign packet, run mode 1, refinement,
        // explicit levels, slice height, Annex H weights.
        let opts = EncodeOptions::default()
            .with_quantization(2)
            .with_quantizer(Quantizer::Uniform)
            .with_sign_packet(true)
            .with_run_mode(RunMode::ZeroCoefficients)
            .with_levels(Some(5), Some(1))
            .with_slice_height(Some(2))
            .with_weights(Weights::AnnexH);
        let bytes = encode(&img, &opts).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(
            (back.width, back.height, back.format),
            (w, h, JpegXsPixelFormat::Gbrp8)
        );
        // High precision + NLT quadratic.
        let hp = encode(
            &img,
            &EncodeOptions::default()
                .with_quantization(1)
                .with_high_precision(true),
        )
        .unwrap();
        assert!(!info(&hp).unwrap().lossless);
        decode(&hp).unwrap();
        let nlt = encode(
            &img,
            &EncodeOptions::default()
                .with_nlt(Some(crate::output::NltParams::Quadratic { dco: 0 })),
        )
        .unwrap();
        decode(&nlt).unwrap();
        // Profile shaping declares Ppih / Plev; CBR hits the exact size.
        let yuv = {
            let planes = (0..3)
                .map(|i| {
                    let (pw, ph) = JpegXsPixelFormat::Yuv422P10Le.plane_dimensions(w, h, i);
                    Plane::new(pw * 2, deep(pw as u32, ph as u32, 10, i))
                })
                .collect();
            JpegXsImage::new(w, h, JpegXsPixelFormat::Yuv422P10Le, planes).unwrap()
        };
        let signed = encode(
            &yuv,
            &EncodeOptions::default().with_profile(Some(Profile::Main422_10)),
        )
        .unwrap();
        let i = info(&signed).unwrap();
        assert_eq!(i.profile, Profile::Main422_10.ppih());
        assert_ne!(i.level, 0);
        assert_eq!(decode(&signed).unwrap(), yuv);
        let cbr = encode(
            &yuv,
            &EncodeOptions::default()
                .with_profile(Some(Profile::Main422_10))
                .with_target_bytes(Some(3000)),
        )
        .unwrap();
        assert_eq!(cbr.len(), 3000);
        decode(&cbr).unwrap();
        let cbr_plain = encode(
            &img,
            &EncodeOptions::default().with_target_bytes(Some(2500)),
        )
        .unwrap();
        assert_eq!(cbr_plain.len(), 2500);
        decode(&cbr_plain).unwrap();
        // Incompatible compositions are refused, not silently dropped.
        assert!(matches!(
            encode(
                &img,
                &EncodeOptions::default()
                    .with_profile(Some(Profile::Main444_12))
                    .with_sign_packet(true)
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            encode(
                &yuv,
                &EncodeOptions::default().with_target_bytes(Some(3000))
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            encode(&img, &EncodeOptions::default().with_quantization(16)),
            Err(Error::InvalidData(_))
        ));
    }

    #[test]
    fn decode_options_limits_and_strict() {
        let (w, h) = (20u32, 10u32);
        let bytes = encode_rgb8(w, h, &gradient(w * 3, h, 5), &EncodeOptions::default()).unwrap();
        let lim = |o: DecodeOptions| decode_with(&bytes, &o);
        assert!(matches!(
            lim(DecodeOptions::default().with_max_width(Some(19))),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::default().with_max_height(Some(9))),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::default().with_max_pixels(Some(199))),
            Err(Error::LimitExceeded(_))
        ));
        assert!(matches!(
            lim(DecodeOptions::default().with_max_bytes(Some(10))),
            Err(Error::LimitExceeded(_))
        ));
        assert!(lim(DecodeOptions::default()
            .with_max_width(None)
            .with_max_height(None)
            .with_max_pixels(None))
        .is_ok());
        // Trailing garbage: tolerated by default, rejected under strict.
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&[0, 1, 2, 3]);
        assert!(decode(&trailing).is_ok());
        assert!(matches!(
            decode_with(&trailing, &DecodeOptions::default().with_strict(true)),
            Err(Error::InvalidData(_))
        ));
        assert!(decode_with(&bytes, &DecodeOptions::default().with_strict(true)).is_ok());
        // Not JPEG XS at all.
        assert!(matches!(decode(b"PNG\r\n"), Err(Error::InvalidData(_))));
        assert!(matches!(info(&[]), Err(Error::InvalidData(_))));
        assert!(decode_components_with(&bytes, &DecodeOptions::default()).is_ok());
    }

    #[test]
    fn component_view_covers_layouts_without_a_contract_view() {
        // Two components: no contract layout, but the component view
        // decodes it and info() says why.
        let (w, h) = (8u32, 4u32);
        let comps = Components::new(
            w,
            h,
            0,
            vec![8, 8],
            vec![(1, 1), (1, 1)],
            vec![
                Plane::new(w as usize, gradient(w, h, 6)),
                Plane::new(w as usize, gradient(w, h, 7)),
            ],
        )
        .unwrap();
        let bytes = encode_components(&comps, &EncodeOptions::default()).unwrap();
        assert!(matches!(info(&bytes), Err(Error::Unsupported(_))));
        assert!(matches!(decode(&bytes), Err(Error::Unsupported(_))));
        let back = decode_components(&bytes).unwrap();
        assert_eq!(back, comps);
        // Star-Tetrix through the component funnel (Cpih = 3, four planes).
        let st = Components::new(
            16,
            16,
            3,
            vec![8; 4],
            vec![(1, 1); 4],
            (0..4)
                .map(|i| Plane::new(16, gradient(16, 16, i)))
                .collect(),
        )
        .unwrap();
        let bytes = encode_components(&st, &EncodeOptions::default().with_levels(Some(2), Some(1)))
            .unwrap();
        assert!(matches!(info(&bytes), Err(Error::Unsupported(_))));
        assert_eq!(decode_components(&bytes).unwrap(), st);
        // Boxed component set: unspecified CICP, decodes back.
        let boxed = encode_components(&comps, &EncodeOptions::default().with_boxed(true)).unwrap();
        assert!(fileformat::is_jxs_file(&boxed));
        assert_eq!(decode_components(&boxed).unwrap(), comps);
        // Mixed bit depths are refused by the encoder (one B[i] per stream).
        let mixed = Components::new(
            w,
            h,
            0,
            vec![8, 10],
            vec![(1, 1), (1, 1)],
            vec![
                Plane::new(w as usize, gradient(w, h, 6)),
                Plane::new(w as usize * 2, deep(w, h, 10, 7)),
            ],
        )
        .unwrap();
        assert!(matches!(
            encode_components(&mixed, &EncodeOptions::default()),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn auto_levels_fit_small_pictures() {
        for (w, h) in [(2u32, 2u32), (3, 2), (2, 7), (5, 5), (33, 1 + 1), (64, 64)] {
            let img = JpegXsImage::new(
                w,
                h,
                JpegXsPixelFormat::Gray8,
                vec![Plane::new(w as usize, gradient(w, h, 8))],
            )
            .unwrap();
            let bytes = encode(&img, &EncodeOptions::default()).unwrap_or_else(|e| {
                panic!("{w}x{h}: {e}");
            });
            assert_eq!(decode(&bytes).unwrap(), img, "{w}x{h}");
        }
        // 4:2:0 needs max(sx) × 2^NL,x ≤ Wf and max(sy) × 2^NL,y ≤ Hf
        // (Table 11): 4×4 is the smallest picture; 4×2 is refused.
        let img = JpegXsImage::new(
            4,
            4,
            JpegXsPixelFormat::Yuv420P,
            vec![
                Plane::new(4, gradient(4, 4, 1)),
                Plane::new(2, gradient(2, 2, 2)),
                Plane::new(2, gradient(2, 2, 3)),
            ],
        )
        .unwrap();
        let tiny = JpegXsImage::new(
            4,
            2,
            JpegXsPixelFormat::Yuv420P,
            vec![
                Plane::new(4, gradient(4, 2, 1)),
                Plane::new(2, gradient(2, 1, 2)),
                Plane::new(2, gradient(2, 1, 3)),
            ],
        )
        .unwrap();
        assert!(matches!(
            encode(&tiny, &EncodeOptions::default()),
            Err(Error::InvalidData(_))
        ));
        assert_eq!(
            decode(&encode(&img, &EncodeOptions::default()).unwrap()).unwrap(),
            img
        );
    }
}
