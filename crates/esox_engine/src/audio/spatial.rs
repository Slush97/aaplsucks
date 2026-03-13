//! Spatial audio — listener sync from camera.

use glam::Vec3;

/// Handle to a loaded sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundHandle(pub(crate) usize);

/// Audio manager wrapping kira.
pub struct AudioManager {
    manager: kira::AudioManager,
    sounds: Vec<kira::sound::static_sound::StaticSoundData>,
    volume: f64,
}

impl AudioManager {
    pub fn new() -> Option<Self> {
        match kira::AudioManager::<kira::backend::cpal::CpalBackend>::new(
            kira::AudioManagerSettings::default(),
        ) {
            Ok(manager) => Some(Self {
                manager,
                sounds: Vec::new(),
                volume: 1.0,
            }),
            Err(e) => {
                tracing::warn!("audio init failed (no-op): {e}");
                None
            }
        }
    }

    /// Load a sound from a file path.
    pub fn load(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<SoundHandle, Box<dyn std::error::Error>> {
        let data = kira::sound::static_sound::StaticSoundData::from_file(path)?;
        let idx = self.sounds.len();
        self.sounds.push(data);
        Ok(SoundHandle(idx))
    }

    /// Play a loaded sound.
    pub fn play(&mut self, handle: SoundHandle) {
        if let Some(data) = self.sounds.get(handle.0) {
            if let Err(e) = self.manager.play(data.clone()) {
                tracing::warn!("audio play failed: {e}");
            }
        }
    }

    /// Set master volume (0.0 = mute, 1.0 = full).
    pub fn set_volume(&mut self, volume: f64) {
        self.volume = volume;
        // Convert amplitude to decibels: dB = 20 * log10(amplitude)
        let db = if volume <= 0.0 {
            kira::Decibels::SILENCE
        } else {
            kira::Decibels((20.0 * (volume as f32).log10()).max(-60.0))
        };
        let _ = self
            .manager
            .main_track()
            .set_volume(db, kira::Tween::default());
    }

    /// Update listener position (synced from active camera each frame).
    pub fn set_listener(&mut self, _position: Vec3, _forward: Vec3, _up: Vec3) {
        // Spatial audio positioning would be implemented here with kira's
        // spatial scene API. For now, this is a placeholder.
    }
}
