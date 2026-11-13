//! Opaque artifact storage; consumers own codecs and transformation identities.
pub mod owner;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod store;
pub mod wire;

pub const PROTOCOL: u32 = 2;
