//! Parent-child hierarchy and GlobalTransform propagation.

use glam::Mat4;
use hecs::World;

use super::components::{GlobalTransform, Transform3D};

/// Parent entity relationship.
pub struct Parent(pub hecs::Entity);

/// Children of an entity.
pub struct Children(pub Vec<hecs::Entity>);

/// Propagate transforms from root entities down the hierarchy.
///
/// Root entities are those with `Transform3D` but no `Parent`.
/// Sets `GlobalTransform = parent_global * local` recursively.
pub fn hierarchy_system(world: &mut World) {
    // Collect root entities (have Transform3D but no Parent).
    let roots: Vec<hecs::Entity> = world
        .query::<&Transform3D>()
        .without::<&Parent>()
        .iter()
        .map(|(e, _)| e)
        .collect();

    for root in roots {
        propagate_recursive(world, root, Mat4::IDENTITY);
    }
}

fn propagate_recursive(world: &mut World, entity: hecs::Entity, parent_global: Mat4) {
    let local = match world.get::<&Transform3D>(entity) {
        Ok(t) => t.matrix(),
        Err(_) => Mat4::IDENTITY,
    };
    let global = parent_global * local;

    // Update or insert GlobalTransform.
    if let Ok(mut gt) = world.get::<&mut GlobalTransform>(entity) {
        gt.0 = global;
    }

    // Recurse into children.
    let children: Vec<hecs::Entity> = match world.get::<&Children>(entity) {
        Ok(c) => c.0.clone(),
        Err(_) => return,
    };
    for child in children {
        propagate_recursive(world, child, global);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn root_entity_gets_local_as_global() {
        let mut world = World::new();
        let t = Transform3D {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        let e = world.spawn((t, GlobalTransform::default()));
        hierarchy_system(&mut world);
        let gt = world.get::<&GlobalTransform>(e).unwrap();
        let expected = t.matrix();
        assert!((gt.0 - expected).abs_diff_eq(Mat4::ZERO, 1e-6));
    }

    #[test]
    fn child_inherits_parent_transform() {
        let mut world = World::new();
        let parent_t = Transform3D {
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        let child_t = Transform3D {
            position: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        let parent = world.spawn((parent_t, GlobalTransform::default()));
        let child = world.spawn((
            child_t,
            GlobalTransform::default(),
            Parent(parent),
        ));
        world
            .insert_one(parent, Children(vec![child]))
            .unwrap();

        hierarchy_system(&mut world);

        let gt = world.get::<&GlobalTransform>(child).unwrap();
        // Child should be at (10, 5, 0) in world space.
        let pos = gt.0.col(3);
        assert!((pos.x - 10.0).abs() < 1e-6);
        assert!((pos.y - 5.0).abs() < 1e-6);
    }
}
