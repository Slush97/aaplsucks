//! System that syncs physics transforms back to the ECS world.

use hecs::World;

use crate::ecs::components::Transform3D;
use crate::physics::{BodyType, PhysicsBackend};

use super::physics_components::RigidBodyComponent;

/// After `physics.step()`, copy transforms of dynamic bodies back into the ECS.
pub fn physics_sync_system(world: &mut World, physics: &dyn PhysicsBackend) {
    for (_, (rb, transform)) in world
        .query_mut::<(&RigidBodyComponent, &mut Transform3D)>()
    {
        if rb.body_type != BodyType::Dynamic {
            continue;
        }
        if let Some((pos, rot)) = physics.query_transform(rb.handle) {
            transform.position = pos;
            transform.rotation = rot;
        }
    }
}
