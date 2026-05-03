//! Crate-local error type.
//!
//! Defined as a small std-only enum so the crate can be built with the
//! default `registry` feature off — i.e. without depending on
//! `oxideav-core` at all. When the `registry` feature is on (the default)
//! a `From<JpegXsError> for oxideav_core::Error` impl is enabled in
//! [`crate::registry`] so the `Decoder` trait surface still
//! interoperates cleanly.
//!
//! The variants mirror the subset of `oxideav_core::Error` that the
//! JPEG XS decoder pipeline actually produces.

use core::fmt;

/// Crate-local error type for the JPEG XS decoder pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JpegXsError {
    /// Bitstream / marker / packet header was malformed.
    InvalidData(String),
    /// Bitstream was syntactically valid but uses a feature this crate
    /// does not implement yet.
    Unsupported(String),
}

impl JpegXsError {
    /// Construct a [`JpegXsError::InvalidData`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidData(msg.into())
    }

    /// Construct a [`JpegXsError::Unsupported`].
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}

impl fmt::Display for JpegXsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData(s) => write!(f, "invalid data: {}", s),
            Self::Unsupported(s) => write!(f, "unsupported: {}", s),
        }
    }
}

impl std::error::Error for JpegXsError {}

/// Crate-local result alias used throughout the pipeline.
pub type Result<T> = core::result::Result<T, JpegXsError>;
