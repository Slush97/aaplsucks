//! Spatial audio — listener sync from camera, spatial track support.

use glam::{Quat, Vec3};

/// Convert a glam Vec3 to a mint Vector3.
fn vec3_to_mint(v: Vec3) -> mint::Vector3<f32> {
    mint::Vector3 { x: v.x, y: v.y, z: v.z }
}

/// Convert a glam Quat to a mint Quaternion.
fn quat_to_mint(q: Quat) -> mint::Quaternion<f32> {
    mint::Quaternion {
        s: q.w,
        v: mint::Vector3 { x: q.x, y: q.y, z: q.z },
    }
}

/// Handle to a loaded sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundHandle(pub(crate) usize);

/// Handle to a playing music track (looping sound with fade control).
pub struct MusicHandle {
    inner: kira::sound::static_sound::StaticSoundHandle,
}

/// Handle to a spatial sound playing on a spatial track.
///
/// Use `set_position()` to move the emitter in world space.
pub struct SpatialSoundHandle {
    track: kira::track::SpatialTrackHandle,
}

impl SpatialSoundHandle {
    /// Update the emitter's world-space position.
    pub fn set_position(&mut self, position: Vec3) {
        self.track
            .set_position(vec3_to_mint(position), kira::Tween::default());
    }
}

/// Audio manager wrapping kira with spatial audio support.
pub struct AudioManager {
    manager: kira::AudioManager,
    sounds: Vec<kira::sound::static_sound::StaticSoundData>,
    volume: f64,
    listener: Option<kira::listener::ListenerHandle>,
}

impl AudioManager {
    pub fn new() -> Option<Self> {
        match kira::AudioManager::<kira::backend::cpal::CpalBackend>::new(
            kira::AudioManagerSettings::default(),
        ) {
            Ok(mut manager) => {
                // Create the listener at the origin.
                let zero_pos: mint::Vector3<f32> = mint::Vector3 { x: 0.0, y: 0.0, z: 0.0 };
                let identity_quat: mint::Quaternion<f32> = mint::Quaternion {
                    s: 1.0,
                    v: mint::Vector3 { x: 0.0, y: 0.0, z: 0.0 },
                };
                let listener = match manager.add_listener(zero_pos, identity_quat) {
                    Ok(l) => Some(l),
                    Err(e) => {
                        tracing::warn!("failed to create audio listener: {e}");
                        None
                    }
                };
                Some(Self {
                    manager,
                    sounds: Vec::new(),
                    volume: 1.0,
                    listener,
                })
            }
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

    /// Play a loaded sound (non-spatial, on the main track).
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

    /// Update listener position and orientation (synced from active camera each frame).
    pub fn set_listener(&mut self, position: Vec3, forward: Vec3, up: Vec3) {
        if let Some(ref mut listener) = self.listener {
            // Compute orientation quaternion: kira expects unrotated listener
            // to face -Z with +X right and +Y up.
            let right = forward.cross(up).normalize_or_zero();
            let corrected_up = right.cross(forward).normalize_or_zero();
            let orientation =
                Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, -forward));
            listener.set_position(vec3_to_mint(position), kira::Tween::default());
            listener.set_orientation(quat_to_mint(orientation), kira::Tween::default());
        }
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

    /// Play a sound at a 3D position with distance-based attenuation.
    ///
    /// Returns a handle that can be used to update the emitter position.
    /// The sound plays on a per-call spatial track linked to the listener.
    pub fn play_spatial(
        &mut self,
        handle: SoundHandle,
        position: Vec3,
        max_distance: f32,
    ) -> Option<SpatialSoundHandle> {
        let listener = self.listener.as_ref()?;
        let data = self.sounds.get(handle.0)?.clone();

        let builder = kira::track::SpatialTrackBuilder::new()
            .distances((1.0, max_distance))
            .persist_until_sounds_finish(true);

        let mut spatial_track = match self.manager.add_spatial_sub_track(listener, vec3_to_mint(position), builder) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("audio play_spatial: failed to create spatial track: {e}");
                return None;
            }
        };

        if let Err(e) = spatial_track.play(data) {
            tracing::warn!("audio play_spatial: play failed: {e}");
            return None;
        }

        Some(SpatialSoundHandle {
            track: spatial_track,
        })
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
