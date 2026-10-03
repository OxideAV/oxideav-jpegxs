//! `to_rgb8` / `to_rgba8` kernels for every [`JpegXsPixelFormat`].
//!
//! * Deep samples (`B > 8`) are rescaled to 8 bits by rounding
//!   `v × 255 / (2^B − 1)` to nearest (ties up) — an exact integer
//!   kernel; 8-bit samples pass through.
//! * Grey replicates the one channel into R = G = B.
//! * The planar RGB layouts (`Gbrp*` / `Gbrap*`) reorder G, B, R(, A)
//!   planes to packed R, G, B(, A).
//! * YCbCr is converted per pixel with the H.273 matrix the image's
//!   [`ColorInfo`] names (`1` BT.709, `5` / `6` BT.601, `9` BT.2020
//!   non-constant luminance; anything else — including "unspecified" —
//!   uses BT.709, the convention of the broadcast chains JPEG XS
//!   targets) and the signalled range (`Limited` or `Unspecified` →
//!   limited / studio range, the broadcast default; `Full` → full
//!   swing). Chroma of the 4:2:2 / 4:2:0 layouts is replicated
//!   (nearest-neighbour, no interpolation). Alpha planes are never
//!   touched by the matrix.
//! * Alpha: the fourth plane of the `Gbrap*` / `Yuva*` layouts,
//!   rescaled like any sample; `255` when the layout has none.

use crate::image::{ColorInfo, ColorModel, ColorRange, JpegXsImage, JpegXsPixelFormat};

/// Rescale a `bits`-deep sample to 8 bits, rounding to nearest.
#[inline]
fn to8(v: u32, bits: u8) -> u8 {
    if bits <= 8 {
        return v.min(255) as u8;
    }
    let max = (1u64 << bits) - 1;
    let v = (v as u64).min(max);
    ((v * 255 + max / 2) / max) as u8
}

/// `(Kr, Kb)` luma coefficients for an H.273 `MatrixCoefficients` code
/// point; BT.709 for everything this crate does not name.
fn luma_coefficients(matrix: u8) -> (f64, f64) {
    match matrix {
        5 | 6 => (0.299, 0.114),
        9 => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    }
}

/// One YCbCr sample triple (at `bits` depth) to 8-bit RGB.
fn ycbcr_to_rgb8(y: u32, cb: u32, cr: u32, bits: u8, limited: bool, matrix: u8) -> [u8; 3] {
    let (kr, kb) = luma_coefficients(matrix);
    let kg = 1.0 - kr - kb;
    let shift = u32::from(bits) - 8;
    let (y0, y_span, c0, c_span) = if limited {
        (
            f64::from(16u32 << shift),
            f64::from(219u32 << shift),
            f64::from(128u32 << shift),
            f64::from(224u32 << shift),
        )
    } else {
        let max = f64::from((1u32 << bits) - 1);
        (0.0, max, f64::from(1u32 << (bits - 1)), max)
    };
    let yn = (f64::from(y) - y0) / y_span;
    let cbn = (f64::from(cb) - c0) / c_span;
    let crn = (f64::from(cr) - c0) / c_span;
    let r = yn + 2.0 * (1.0 - kr) * crn;
    let b = yn + 2.0 * (1.0 - kb) * cbn;
    let g = yn - (2.0 * kb * (1.0 - kb) * cbn + 2.0 * kr * (1.0 - kr) * crn) / kg;
    let q = |v: f64| -> u8 { (v * 255.0).round().clamp(0.0, 255.0) as u8 };
    [q(r), q(g), q(b)]
}

/// Convert to tightly packed 8-bit RGB (`3 × width` bytes per row).
pub fn to_rgb8(img: &JpegXsImage) -> Vec<u8> {
    convert(img, false)
}

/// Convert to tightly packed 8-bit RGBA (`4 × width` bytes per row).
pub fn to_rgba8(img: &JpegXsImage) -> Vec<u8> {
    convert(img, true)
}

fn convert(img: &JpegXsImage, with_alpha: bool) -> Vec<u8> {
    let w = img.width as usize;
    let h = img.height as usize;
    let bpp = if with_alpha { 4 } else { 3 };
    let mut out = Vec::with_capacity(w * h * bpp);
    let bits = img.bit_depth;
    let format: JpegXsPixelFormat = img.format;
    let has_alpha = format.has_alpha();
    let (dh, dv) = format.chroma_divisors();
    let color: &ColorInfo = &img.color;
    let limited = !matches!(color.range, ColorRange::Full);
    for y in 0..h {
        for x in 0..w {
            let rgb: [u8; 3] = match format.color_model() {
                ColorModel::Gray => {
                    let v = to8(img.sample(0, x, y), bits);
                    [v, v, v]
                }
                ColorModel::Rgb => [
                    to8(img.sample(2, x, y), bits),
                    to8(img.sample(0, x, y), bits),
                    to8(img.sample(1, x, y), bits),
                ],
                ColorModel::YCbCr => {
                    let (cx, cy) = (x / dh, y / dv);
                    ycbcr_to_rgb8(
                        img.sample(0, x, y),
                        img.sample(1, cx, cy),
                        img.sample(2, cx, cy),
                        bits,
                        limited,
                        color.matrix,
                    )
                }
            };
            out.extend_from_slice(&rgb);
            if with_alpha {
                out.push(if has_alpha {
                    to8(img.sample(3, x, y), bits)
                } else {
                    255
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Plane;

    #[test]
    fn deep_samples_round_to_8_bits() {
        assert_eq!(to8(0, 10), 0);
        assert_eq!(to8(1023, 10), 255);
        assert_eq!(to8(512, 10), 128); // 512*255/1023 = 127.6 → 128
        assert_eq!(to8(0x0FFF, 12), 255);
        assert_eq!(to8(2048, 12), 128);
        assert_eq!(to8(65535, 16), 255);
        assert_eq!(to8(300, 8), 255); // clamp
        assert_eq!(to8(511, 9), 255);
    }

    #[test]
    fn gray_and_rgb_layouts() {
        let g = JpegXsImage::new(
            2,
            1,
            JpegXsPixelFormat::Gray8,
            vec![Plane::new(2, vec![7, 200])],
        )
        .unwrap();
        assert_eq!(g.to_rgb8(), vec![7, 7, 7, 200, 200, 200]);
        assert_eq!(g.to_rgba8(), vec![7, 7, 7, 255, 200, 200, 200, 255]);
        let rgb = JpegXsImage::from_rgb8(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(rgb.to_rgb8(), vec![1, 2, 3, 4, 5, 6]);
        let rgba = JpegXsImage::from_rgba8(1, 2, vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        assert_eq!(rgba.to_rgba8(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(rgba.to_rgb8(), vec![1, 2, 3, 5, 6, 7]);
        // 10-bit GBR → 8-bit by rounding.
        let le = |v: u16| v.to_le_bytes().to_vec();
        let deep = JpegXsImage::new(
            1,
            1,
            JpegXsPixelFormat::Gbrp10Le,
            vec![
                Plane::new(2, le(512)),
                Plane::new(2, le(0)),
                Plane::new(2, le(1023)),
            ],
        )
        .unwrap();
        assert_eq!(deep.to_rgb8(), vec![255, 128, 0]);
    }

    #[test]
    fn ycbcr_kernels() {
        // Limited-range BT.709 black / white / grey (chroma neutral).
        assert_eq!(ycbcr_to_rgb8(16, 128, 128, 8, true, 1), [0, 0, 0]);
        assert_eq!(ycbcr_to_rgb8(235, 128, 128, 8, true, 1), [255, 255, 255]);
        assert_eq!(ycbcr_to_rgb8(126, 128, 128, 8, true, 1), [128, 128, 128]);
        // Full range: Y maps 1:1 at neutral chroma.
        assert_eq!(ycbcr_to_rgb8(200, 128, 128, 8, false, 1), [200, 200, 200]);
        // 10-bit limited white.
        assert_eq!(ycbcr_to_rgb8(940, 512, 512, 10, true, 1), [255, 255, 255]);
        // Pure red in BT.601 full range: Y = 76, Cb = 85, Cr = 255.
        let [r, g, b] = ycbcr_to_rgb8(76, 85, 255, 8, false, 5);
        assert!(r >= 253 && g <= 2 && b <= 2, "{r} {g} {b}");
        // Pure blue in BT.709 limited: Y = 32, Cb = 240, Cr = 118.
        let [r, g, b] = ycbcr_to_rgb8(32, 240, 118, 8, true, 1);
        assert!(r <= 2 && g <= 2 && b >= 253, "{r} {g} {b}");
        // 4:2:0 chroma replication + alpha plane.
        let img = JpegXsImage::new(
            2,
            2,
            JpegXsPixelFormat::Yuva420P,
            vec![
                Plane::new(2, vec![16, 235, 126, 126]),
                Plane::new(1, vec![128]),
                Plane::new(1, vec![128]),
                Plane::new(2, vec![0, 64, 128, 255]),
            ],
        )
        .unwrap();
        assert_eq!(
            img.to_rgba8(),
            vec![0, 0, 0, 0, 255, 255, 255, 64, 128, 128, 128, 128, 128, 128, 128, 255]
        );
        assert_eq!(img.to_rgb8().len(), 12);
    }
}
