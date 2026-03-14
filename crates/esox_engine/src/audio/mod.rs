//! Audio system — kira integration (behind `audio` feature).

#[cfg(feature = "audio")]
pub mod spatial;

#[cfg(feature = "audio")]
pub use spatial::{AudioManager, SoundHandle, SpatialSoundHandle};
