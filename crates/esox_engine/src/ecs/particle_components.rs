//! ECS components for the particle system.

use esox_gfx::mesh3d::{MaterialHandle, ParticlePoolHandle};
use glam::Vec3;

/// Particle emitter component. Attach to an entity with a `Transform3D` to emit
/// particles from that entity's world position.
pub struct ParticleEmitter {
    /// Handle to the GPU particle pool.
    pub pool: ParticlePoolHandle,
    /// Material used to render particles.
    pub material: MaterialHandle,
    /// Continuous spawn rate (particles per second). Set to 0 for burst-only.
    pub spawn_rate: f32,
    /// One-shot burst count. Reset to 0 after emitting.
    pub burst_count: u32,
    /// Minimum initial velocity (per-axis).
    pub velocity_min: Vec3,
    /// Maximum initial velocity (per-axis).
    pub velocity_max: Vec3,
    /// Gravity applied to particles.
    pub gravity: Vec3,
    /// Particle lifetime range [min, max] in seconds.
    pub lifetime: [f32; 2],
    /// Particle size at [birth, death].
    pub size: [f32; 2],
    /// Color at birth (RGBA, linear).
    pub color_start: [f32; 4],
    /// Color at death (RGBA, linear).
    pub color_end: [f32; 4],
    /// Whether this emitter is actively spawning.
    pub active: bool,
    /// Internal spawn accumulator (fractional particles).
    /// Internal spawn accumulator — leave at 0.0 when constructing.
    pub spawn_accumulator: f32,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            pool: ParticlePoolHandle(0),
            material: MaterialHandle(0),
            spawn_rate: 100.0,
            burst_count: 0,
            velocity_min: Vec3::new(-1.0, 1.0, -1.0),
            velocity_max: Vec3::new(1.0, 3.0, 1.0),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            lifetime: [0.5, 2.0],
            size: [0.1, 0.02],
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
            active: true,
            spawn_accumulator: 0.0,
        }
    }
}
