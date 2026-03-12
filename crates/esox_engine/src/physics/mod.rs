//! Physics backend trait and null implementation.

use glam::{Quat, Vec3};

/// Handle to a physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub u32);

/// Description for creating a physics body.
pub struct BodyDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub body_type: BodyType,
}

/// Type of physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Static,
    Dynamic,
    Kinematic,
}

/// Result of a raycast query.
pub struct RayHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub body: BodyHandle,
}

/// Trait for pluggable physics backends.
///
/// Implement this to wrap rapier, bullet, or any other physics engine.
/// The engine calls `step()` each fixed tick and syncs transforms back.
pub trait PhysicsBackend: 'static {
    fn step(&mut self, dt: f32);
    fn add_body(&mut self, desc: BodyDesc) -> BodyHandle;
    fn remove_body(&mut self, handle: BodyHandle);
    fn query_transform(&self, handle: BodyHandle) -> Option<(Vec3, Quat)>;
    fn set_transform(&mut self, handle: BodyHandle, pos: Vec3, rot: Quat);
    fn apply_force(&mut self, handle: BodyHandle, force: Vec3);
    fn apply_impulse(&mut self, handle: BodyHandle, impulse: Vec3);
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit>;
}

/// No-op physics backend (default).
pub struct NullPhysics;

impl PhysicsBackend for NullPhysics {
    fn step(&mut self, _dt: f32) {}
    fn add_body(&mut self, _desc: BodyDesc) -> BodyHandle {
        BodyHandle(0)
    }
    fn remove_body(&mut self, _handle: BodyHandle) {}
    fn query_transform(&self, _handle: BodyHandle) -> Option<(Vec3, Quat)> {
        None
    }
    fn set_transform(&mut self, _handle: BodyHandle, _pos: Vec3, _rot: Quat) {}
    fn apply_force(&mut self, _handle: BodyHandle, _force: Vec3) {}
    fn apply_impulse(&mut self, _handle: BodyHandle, _impulse: Vec3) {}
    fn raycast(&self, _origin: Vec3, _dir: Vec3, _max_dist: f32) -> Option<RayHit> {
        None
    }
}
