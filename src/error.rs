//! Crate-local error type.
//!
//! Defined as a small std-only enum so the crate can be built with the
//! default `registry` feature off — i.e. without depending on
//! `oxideav-core` at all. When the `registry` feature is on (the default)
//! a `From<JpegXsError> for oxideav_core::Error` impl is enabled in
//! [`crate::registry`] so the `Decoder` / `Encoder` trait surface still
//! interoperates cleanly.
//!
//! The shape follows the workspace image-crate API contract: at least
//! `InvalidData`, `Unsupported`, `LimitExceeded` and `Io(std::io::Error)`
//! (with `From<std::io::Error>`), `Display` + `std::error::Error`, and no
//! `Clone` / `PartialEq` (an `io::Error` is neither).

use core::fmt;

/// Crate-local error type for the JPEG XS codec.
#[derive(Debug)]
#[non_exhaustive]
pub enum JpegXsError {
    /// Bitstream / marker / packet header / box structure was malformed,
    /// or a caller-supplied image has inconsistent geometry.
    InvalidData(String),
    /// Bitstream was syntactically valid but uses a feature this crate
    /// does not implement, or the input layout has no JPEG XS
    /// representation.
    Unsupported(String),
    /// A [`crate::DecodeOptions`] limit (dimensions, pixel count, input
    /// size) was exceeded before any sample buffer was allocated.
    LimitExceeded(String),
    /// An I/O error from [`crate::decode_from`] / [`crate::encode_to`].
    Io(std::io::Error),
}

/// Contract alias: `oxideav_jpegxs::Error` is [`JpegXsError`].
pub type Error = JpegXsError;

impl JpegXsError {
    /// Construct a [`JpegXsError::InvalidData`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct a [`JpegXsError::Unsupported`].
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Construct a [`JpegXsError::LimitExceeded`].
    pub fn limit(msg: impl Into<String>) -> Self {
        Self::LimitExceeded(msg.into())
    }
}

impl fmt::Display for JpegXsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
            Self::LimitExceeded(s) => write!(f, "limit exceeded: {}", s),
            Self::Io(e) => write!(f, "i/o error: {}", e),
        }
    }
}

impl std::error::Error for JpegXsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for JpegXsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Crate-local result alias used throughout the pipeline.
pub type Result<T> = core::result::Result<T, JpegXsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_source() {
        assert_eq!(JpegXsError::invalid("x").to_string(), "invalid data: x");
        assert_eq!(JpegXsError::unsupported("y").to_string(), "unsupported: y");
        assert_eq!(JpegXsError::limit("z").to_string(), "limit exceeded: z");
        let io: JpegXsError = std::io::Error::other("boom").into();
        assert!(matches!(io, JpegXsError::Io(_)));
        assert!(io.to_string().contains("boom"));
        assert!(std::error::Error::source(&io).is_some());
        assert!(std::error::Error::source(&JpegXsError::invalid("x")).is_none());
    }
}
