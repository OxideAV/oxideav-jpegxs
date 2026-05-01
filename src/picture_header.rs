//! Picture header (PIH) — ISO/IEC 21122-1:2022 §A.4.4, Table A.7.
//!
//! `Lpih` is fixed at 26 bytes for the body so the entire segment is
//! 28 bytes (2 marker + 26 length+payload). The body lays out as:
//!
//! ```text
//! Lcod   u32   total codestream byte count, or 0 for VBR
//! Ppih   u16   profile id (see ISO/IEC 21122-2)
//! Plev   u16   level + sublevel
//! Wf     u16   image width  (sample-grid units)
//! Hf     u16   image height (sample-grid units)
//! Cw     u16   precinct width unit, or 0 for full image
//! Hsl    u16   slice height (precincts)
//! Nc     u8    number of components, 1..=8
//! Ng     u8    coefficients per code group (4)
//! Ss     u8    code groups per significance group (8)
//! Bw     u8    nominal bit precision of wavelet coefs (20|18|B[0])
//! Fq:Br  u4|u4 fractional bits | bitplane-count raw width
//! Fslc:Ppoc:Cpih u1|u3|u4
//! NL,x:NL,y      u4|u4
//! Lh:Rl:Qpih:Fs:Rm  u1|u1|u2|u2|u2
//! ```
//!
//! Fields kept in the parsed [`PictureHeader`] struct in their decoded
//! form. The parser refuses obvious corruption (`Nc == 0` or `Nc > 8`,
//! `Lpih != 26`).

use oxideav_core::{Error, Result};

/// Decoded picture header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PictureHeader {
    /// Total codestream byte count (SOC..EOC inclusive) for CBR, or 0
    /// for variable-bitrate coding.
    pub lcod: u32,
    /// Profile identifier (see ISO/IEC 21122-2). 0 = unrestricted.
    pub ppih: u16,
    /// Level + sublevel encoding.
    pub plev: u16,
    /// Image width on the sample grid (`Wf`).
    pub wf: u16,
    /// Image height on the sample grid (`Hf`).
    pub hf: u16,
    /// Precinct width parameter (`Cw`). 0 means precincts span the
    /// entire image width.
    pub cw: u16,
    /// Slice height in precincts (`Hsl`).
    pub hsl: u16,
    /// Number of components (`Nc`), 1..=8.
    pub nc: u8,
    /// Coefficients per code group (`Ng`); spec says 4.
    pub ng: u8,
    /// Code groups per significance group (`Ss`); spec says 8.
    pub ss: u8,
    /// Nominal bit precision of wavelet coefficients (`Bw`).
    pub bw: u8,
    /// Fractional bits in the wavelet coefficients (`Fq`), 4 bits.
    pub fq: u8,
    /// Raw bitplane-count width (`Br`), 4 bits.
    pub br: u8,
    /// Slice coding mode flag (`Fslc`), 1 bit.
    pub fslc: u8,
    /// Progression order within precincts (`Ppoc`), 3 bits.
    pub ppoc: u8,
    /// Colour transformation id (`Cpih`), 4 bits. See Table A.9.
    pub cpih: u8,
    /// Number of horizontal wavelet decompositions (`NL,x`), 4 bits.
    pub nlx: u8,
    /// Number of vertical wavelet decompositions (`NL,y`), 4 bits.
    pub nly: u8,
    /// Long-header enforcement flag (`Lh`).
    pub lh: u8,
    /// Raw-mode selection per packet flag (`Rl`).
    pub rl: u8,
    /// Inverse-quantizer type (`Qpih`), 2 bits. See Table A.10.
    pub qpih: u8,
    /// Sign handling strategy (`Fs`), 2 bits. See Table A.11.
    pub fs: u8,
    /// Run mode (`Rm`), 2 bits. See Table A.12.
    pub rm: u8,
}

/// PIH segment body length per Table A.7. The marker (`PIH`) itself
/// adds another two bytes that are not counted in `Lpih`.
pub const PIH_LPIH: u16 = 26;

/// Parse a PIH segment body. `body` is the bytes after the marker AND
/// the `Lpih` length field — i.e. exactly `Lpih - 2 == 24` bytes.
pub fn parse(body: &[u8]) -> Result<PictureHeader> {
    if body.len() != (PIH_LPIH as usize) - 2 {
        return Err(Error::invalid(format!(
            "jpegxs: PIH body must be {} bytes, got {}",
            PIH_LPIH as usize - 2,
            body.len()
        )));
    }
    let lcod = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
    let ppih = u16::from_be_bytes([body[4], body[5]]);
    let plev = u16::from_be_bytes([body[6], body[7]]);
    let wf = u16::from_be_bytes([body[8], body[9]]);
    let hf = u16::from_be_bytes([body[10], body[11]]);
    let cw = u16::from_be_bytes([body[12], body[13]]);
    let hsl = u16::from_be_bytes([body[14], body[15]]);
    let nc = body[16];
    let ng = body[17];
    let ss = body[18];
    let bw = body[19];
    // body[20] = Fq (high nibble) | Br (low nibble)
    let fq = body[20] >> 4;
    let br = body[20] & 0x0f;
    // body[21] = Fslc (1 bit) | Ppoc (3 bits) | Cpih (4 bits)
    let fslc = (body[21] >> 7) & 0x01;
    let ppoc = (body[21] >> 4) & 0x07;
    let cpih = body[21] & 0x0f;
    // body[22] = NL,x (4 bits) | NL,y (4 bits)
    let nlx = body[22] >> 4;
    let nly = body[22] & 0x0f;
    // body[23] = Lh (1) | Rl (1) | Qpih (2) | Fs (2) | Rm (2)
    let lh = (body[23] >> 7) & 0x01;
    let rl = (body[23] >> 6) & 0x01;
    let qpih = (body[23] >> 4) & 0x03;
    let fs = (body[23] >> 2) & 0x03;
    let rm = body[23] & 0x03;

    if nc == 0 || nc > 8 {
        return Err(Error::invalid(format!(
            "jpegxs: PIH Nc must be 1..=8, got {nc}"
        )));
    }
    if wf == 0 || hf == 0 {
        return Err(Error::invalid(format!(
            "jpegxs: PIH dimensions must be non-zero, got {wf}x{hf}"
        )));
    }

    Ok(PictureHeader {
        lcod,
        ppih,
        plev,
        wf,
        hf,
        cw,
        hsl,
        nc,
        ng,
        ss,
        bw,
        fq,
        br,
        fslc,
        ppoc,
        cpih,
        nlx,
        nly,
        lh,
        rl,
        qpih,
        fs,
        rm,
    })
}

impl PictureHeader {
    /// Image width (`Wf`).
    pub fn width(&self) -> u32 {
        self.wf as u32
    }

    /// Image height (`Hf`).
    pub fn height(&self) -> u32 {
        self.hf as u32
    }

    /// Number of components.
    pub fn num_components(&self) -> usize {
        self.nc as usize
    }

    /// Whether this is the lossless coding mode (Table A.8: Bw=B[0],
    /// Fq=0).
    pub fn is_lossless(&self) -> bool {
        self.fq == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(nc: u8, wf: u16, hf: u16, cpih: u8, fq_br: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&wf.to_be_bytes());
        v.extend_from_slice(&hf.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.push(nc);
        v.push(4);
        v.push(8);
        v.push(20);
        v.push(fq_br);
        v.push(cpih & 0x0f);
        v.push(0x11);
        v.push(0x00);
        v
    }

    #[test]
    fn parses_normal_pih() {
        let pih = parse(&body(3, 1920, 1080, 1, 0x80)).expect("ok");
        assert_eq!(pih.width(), 1920);
        assert_eq!(pih.height(), 1080);
        assert_eq!(pih.num_components(), 3);
        assert_eq!(pih.cpih, 1);
        assert_eq!(pih.fq, 8);
        assert_eq!(pih.br, 0);
        assert!(!pih.is_lossless());
    }

    #[test]
    fn detects_lossless_mode() {
        // Fq = 0 (high nibble) → lossless per Table A.8.
        let pih = parse(&body(1, 4, 4, 0, 0x00)).expect("lossless");
        assert!(pih.is_lossless());
    }

    #[test]
    fn rejects_invalid_nc() {
        assert!(parse(&body(0, 4, 4, 0, 0x80)).is_err());
        assert!(parse(&body(9, 4, 4, 0, 0x80)).is_err());
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(parse(&body(1, 0, 4, 0, 0x80)).is_err());
        assert!(parse(&body(1, 4, 0, 0, 0x80)).is_err());
    }

    #[test]
    fn rejects_short_body() {
        assert!(parse(&[0u8; 10]).is_err());
    }

    #[test]
    fn unpacks_packed_bytes() {
        // Verify the packed nibble + bitfield decoding.
        let mut b = body(1, 4, 4, 0, 0x86);
        // body[20] = 0x86 → Fq=8, Br=6.
        // body[21] = 0xCD → Fslc=1, Ppoc=4, Cpih=0xD.
        b[21] = 0xCD;
        // body[22] = 0x73 → NL,x=7, NL,y=3.
        b[22] = 0x73;
        // body[23] = 0xE5 = 1110 0101
        //          → Lh=1, Rl=1, Qpih=2, Fs=1, Rm=1.
        b[23] = 0xE5;
        let p = parse(&b).expect("packed");
        assert_eq!(p.fq, 8);
        assert_eq!(p.br, 6);
        assert_eq!(p.fslc, 1);
        assert_eq!(p.ppoc, 4);
        assert_eq!(p.cpih, 0xD);
        assert_eq!(p.nlx, 7);
        assert_eq!(p.nly, 3);
        assert_eq!(p.lh, 1);
        assert_eq!(p.rl, 1);
        assert_eq!(p.qpih, 2);
        assert_eq!(p.fs, 1);
        assert_eq!(p.rm, 1);
    }
}
