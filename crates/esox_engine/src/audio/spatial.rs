//! Spatial audio — listener sync from camera.

use glam::Vec3;

/// Handle to a loaded sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundHandle(pub(crate) usize);

/// Handle to a playing music track (looping sound with fade control).
pub struct MusicHandle {
    inner: kira::sound::static_sound::StaticSoundHandle,
}

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

    /// Play a loaded sound at the specified volume (0.0–1.0 amplitude).
    pub fn play_at_volume(&mut self, handle: SoundHandle, volume: f64) {
        if let Some(data) = self.sounds.get(handle.0) {
            let db = if volume <= 0.0 {
                kira::Decibels::SILENCE
            } else {
                kira::Decibels((20.0 * (volume as f32).log10()).max(-60.0))
            };
            let data = data.clone().with_settings(
                kira::sound::static_sound::StaticSoundSettings::default().volume(db),
            );
            if let Err(e) = self.manager.play(data) {
                tracing::warn!("audio play_at_volume failed: {e}");
            }
        }
    }

    /// Play a loaded sound as looping music with a fade-in.
    pub fn play_music(&mut self, handle: SoundHandle, fade_in_secs: f32) -> Option<MusicHandle> {
        if let Some(data) = self.sounds.get(handle.0) {
            let settings = kira::sound::static_sound::StaticSoundSettings::default()
                .loop_region(..)
                .fade_in_tween(Some(kira::Tween {
                    duration: std::time::Duration::from_secs_f32(fade_in_secs),
                    ..Default::default()
                }));
            let data = data.clone().with_settings(settings);
            match self.manager.play(data) {
                Ok(inner) => Some(MusicHandle { inner }),
                Err(e) => {
                    tracing::warn!("audio play_music failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    }

    /// Stop a playing music track with a fade-out.
    pub fn stop_music(&mut self, music: &mut MusicHandle, fade_out_secs: f32) {
        music.inner.stop(kira::Tween {
            duration: std::time::Duration::from_secs_f32(fade_out_secs),
            ..Default::default()
        });
    }

    /// Crossfade from one music track to another.
    pub fn crossfade_music(
        &mut self,
        from: &mut MusicHandle,
        to: SoundHandle,
        duration: f32,
    ) -> Option<MusicHandle> {
        self.stop_music(from, duration);
        self.play_music(to, duration)
    }
}

/// Compute distance-based volume attenuation (inverse-distance, clamped to [0, 1]).
pub fn distance_attenuation(listener: Vec3, source: Vec3, max_dist: f32) -> f64 {
    let dist = listener.distance(source);
    if dist >= max_dist {
        0.0
    } else {
        ((1.0 - dist / max_dist) as f64).clamp(0.0, 1.0)
    }
}
