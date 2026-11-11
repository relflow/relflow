//! Shared opaque artifact storage. Fields own preprocessing, codecs and encoder identities.
#[cfg(feature = "python")]
mod python;
pub mod store;
pub mod wire;
