//! ECS systems — render extraction, light collection, camera sync.

use glam::Vec3;
use hecs::World;

use esox_gfx::mesh3d::{
    Camera, DirectionalLight, InstanceData, LightEnvironment, PointLight, Renderer3D, SpotLight,
};

use super::components::{
    Camera3D, DirectionalLightComponent, GlobalTransform, MeshRenderer, PointLightComponent,
    SpotLightComponent,
};

/// Extract renderable entities and issue draw calls to the renderer.
///
/// Queries all entities with `(GlobalTransform, MeshRenderer)`, groups by
/// (mesh, material), and batches draw calls.
pub fn render_extraction_system(world: &World, renderer: &mut Renderer3D) {
    // Collect draw data grouped by (mesh, material).
    // Simple approach: just issue one draw per entity for now,
    // relying on the renderer's internal batching/sorting.
    for (_entity, (gt, mr)) in world.query::<(&GlobalTransform, &MeshRenderer)>().iter() {
        if !mr.visible {
            continue;
        }
        let instance = InstanceData {
            model: gt.0.to_cols_array_2d(),
            color: mr.tint,
            params: [0.0; 4],
        };
        renderer.draw_with_material(mr.mesh, mr.material, &[instance]);
    }
}

/// Collect light components and build a LightEnvironment for the renderer.
pub fn light_collection_system(world: &World) -> LightEnvironment {
    let mut env = LightEnvironment {
        ambient_color: [0.1, 0.1, 0.1],
        ambient_intensity: 1.0,
        directional: DirectionalLight {
            direction: [0.0, -1.0, 0.0],
            color: [0.0; 3],
            intensity: 0.0,
        },
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
    };

    // Directional lights — use the first one found.
    for (_e, (gt, dl)) in world
        .query::<(&GlobalTransform, &DirectionalLightComponent)>()
        .iter()
    {
        // Direction from transform rotation: forward = -Z.
        let forward = gt.0.transform_vector3(-Vec3::Z).normalize();
        env.directional = DirectionalLight {
            direction: forward.into(),
            color: dl.color,
            intensity: dl.intensity,
        };
        break; // Only one directional light supported.
    }

    // Point lights.
    for (_e, (gt, pl)) in world
        .query::<(&GlobalTransform, &PointLightComponent)>()
        .iter()
    {
        let pos = gt.0.col(3).truncate();
        env.point_lights.push(PointLight {
            position: pos.into(),
            color: pl.color,
            intensity: pl.intensity,
            range: pl.range,
        });
    }

    // Spot lights.
    for (_e, (gt, sl)) in world
        .query::<(&GlobalTransform, &SpotLightComponent)>()
        .iter()
    {
        let pos = gt.0.col(3).truncate();
        let forward = gt.0.transform_vector3(-Vec3::Z).normalize();
        env.spot_lights.push(SpotLight {
            position: pos.into(),
            direction: forward.into(),
            color: sl.color,
            intensity: sl.intensity,
            range: sl.range,
            inner_cone_angle: sl.inner_cone_angle,
            outer_cone_angle: sl.outer_cone_angle,
        });
    }

    env
}

/// Find the active camera and produce a renderer Camera.
pub fn camera_sync_system(world: &World) -> Option<Camera> {
    for (_e, (gt, cam)) in world.query::<(&GlobalTransform, &Camera3D)>().iter() {
        if !cam.active {
            continue;
        }
        let position = gt.0.col(3).truncate();
        let forward = gt.0.transform_vector3(-Vec3::Z).normalize();
        let up = gt.0.transform_vector3(Vec3::Y).normalize();
        let target = position + forward;

        return Some(Camera {
            position,
            target,
            up,
            fov_y: cam.fov_y,
            near: cam.near,
            far: cam.far,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::Transform3D;
    use glam::{Mat4, Quat};

    #[test]
    fn camera_sync_finds_active() {
        let mut world = World::new();
        let t = Transform3D {
            position: Vec3::new(0.0, 5.0, 10.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        world.spawn((
            t,
            GlobalTransform(t.matrix()),
            Camera3D {
                fov_y: 1.0,
                near: 0.1,
                far: 100.0,
                active: true,
            },
        ));

        let cam = camera_sync_system(&world);
        assert!(cam.is_some());
        let cam = cam.unwrap();
        assert!((cam.position - Vec3::new(0.0, 5.0, 10.0)).length() < 1e-5);
    }

    #[test]
    fn light_collection_gathers_point_lights() {
        let mut world = World::new();
        world.spawn((
            GlobalTransform(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))),
            PointLightComponent {
                color: [1.0, 0.0, 0.0],
                intensity: 5.0,
                range: 10.0,
            },
        ));

        let env = light_collection_system(&world);
        assert_eq!(env.point_lights.len(), 1);
        assert!((env.point_lights[0].position[0] - 1.0).abs() < 1e-5);
    }
}
