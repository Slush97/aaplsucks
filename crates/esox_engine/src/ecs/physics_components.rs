//! ECS components for physics bodies and colliders.

use crate::physics::{BodyHandle, BodyType, ColliderShape};

/// Marks an entity as having a physics rigid body.
pub struct RigidBodyComponent {
    pub handle: BodyHandle,
    pub body_type: BodyType,
}

/// Marks an entity as a trigger volume (sensor). Attach alongside a
/// `RigidBodyComponent` with a sensor collider to receive trigger events.
pub struct TriggerVolume {
    /// Optional tag for filtering trigger events by purpose (e.g. "checkpoint").
    pub tag: Option<&'static str>,
}

/// Collider description stored on an entity (informational — the actual
/// collider is created inside the physics backend via `BodyDesc::collider`).
pub struct ColliderComponent {
    pub shape: ColliderShape,
    pub friction: f32,
    pub restitution: f32,
    pub is_sensor: bool,
    pub collision_group: u32,
    pub collision_mask: u32,
}
