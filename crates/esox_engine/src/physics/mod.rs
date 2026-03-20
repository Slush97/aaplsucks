//! Physics backend trait and null implementation.

#[cfg(feature = "rapier")]
pub mod rapier;
pub mod entity_map;

use glam::{Quat, Vec3};

/// Handle to a physics body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub u32);

/// Shape of a physics collider.
#[derive(Debug, Clone)]
pub enum ColliderShape {
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
    ConvexHull { points: Vec<Vec3> },
    TriMesh { vertices: Vec<Vec3>, indices: Vec<[u32; 3]> },
}

/// Description for creating a collider.
#[derive(Debug, Clone)]
pub struct ColliderDesc {
    pub shape: ColliderShape,
    pub friction: f32,
    pub restitution: f32,
    pub is_sensor: bool,
    /// Collision group membership bits (default: all bits set).
    pub collision_group: u32,
    /// Collision filter mask — collides only with bodies whose group bits
    /// overlap this mask (default: all bits set).
    pub collision_mask: u32,
}

impl Default for ColliderDesc {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            friction: 0.5,
            restitution: 0.0,
            is_sensor: false,
            collision_group: 0xFFFF_FFFF,
            collision_mask: 0xFFFF_FFFF,
        }
    }
}

/// Description for creating a physics body.
pub struct BodyDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub body_type: BodyType,
    /// Optional inline collider — attached to the body on creation.
    pub collider: Option<ColliderDesc>,
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

/// Phase of a trigger volume overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerPhase {
    Enter,
    Stay,
    Exit,
}

/// An overlap event between two bodies where at least one is a sensor.
pub struct TriggerEvent {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub phase: TriggerPhase,
}

/// A contact event from the narrow phase.
pub struct ContactEvent {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    pub normal: Vec3,
    pub impulse: f32,
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

    /// Drain contact events accumulated during the last step.
    fn drain_contacts(&mut self) -> Vec<ContactEvent> {
        vec![]
    }

    /// Drain trigger (sensor overlap) events accumulated during the last step.
    fn drain_triggers(&mut self) -> Vec<TriggerEvent> {
        vec![]
    }

    /// Set the gravity vector.
    fn set_gravity(&mut self, _gravity: Vec3) {}

    /// Get the linear velocity of a body.
    fn get_linear_velocity(&self, _handle: BodyHandle) -> Option<Vec3> { None }

    /// Set the linear velocity of a body.
    fn set_linear_velocity(&mut self, _handle: BodyHandle, _vel: Vec3) {}

    /// Get the angular velocity of a body.
    fn get_angular_velocity(&self, _handle: BodyHandle) -> Option<Vec3> { None }

    /// Set the angular velocity of a body.
    fn set_angular_velocity(&mut self, _handle: BodyHandle, _vel: Vec3) {}

    /// Find all bodies whose colliders overlap a sphere.
    fn overlap_sphere(&self, _center: Vec3, _radius: f32) -> Vec<BodyHandle> { vec![] }

    /// Find all bodies whose colliders overlap an axis-aligned box.
    fn overlap_box(&self, _center: Vec3, _half_extents: Vec3) -> Vec<BodyHandle> { vec![] }

    /// Cast a shape along a direction and return the first hit.
    fn shape_cast(
        &self,
        _shape: &ColliderShape,
        _from: Vec3,
        _dir: Vec3,
        _max_dist: f32,
    ) -> Option<RayHit> {
        None
    }
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
