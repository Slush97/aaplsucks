//! Scene serialization — save/load ECS worlds to `.scene.ron` files.

use std::collections::HashMap;

use glam::Vec3;
use hecs::World;
use serde::{Deserialize, Serialize};

use crate::assets::AssetManager;
use crate::ecs::components::{
    Camera3D, DirectionalLightComponent, MeshRenderer, PointLightComponent, SpotLightComponent,
    Tag, Transform3D,
};
use crate::ecs::hierarchy::{Children, Parent};
use crate::ecs::physics_components::{ColliderComponent, RigidBodyComponent, TriggerVolume};
use crate::ecs::particle_components::ParticleEmitter;
use crate::physics::{BodyDesc, BodyType, ColliderDesc, ColliderShape, PhysicsBackend};
use crate::physics::entity_map::PhysicsEntityMap;

fn default_version() -> u32 { 1 }

/// A serialized scene file.
#[derive(Debug, Serialize, Deserialize)]
pub struct SceneFile {
    #[serde(default = "default_version")]
    pub version: u32,
    pub entities: Vec<SceneEntity>,
}

/// A single entity in a serialized scene.
#[derive(Debug, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Transform3D>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_renderer: Option<SerializedMeshRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<Camera3D>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_light: Option<PointLightComponent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot_light: Option<SpotLightComponent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directional_light: Option<DirectionalLightComponent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid_body: Option<SerializedRigidBody>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collider: Option<SerializedCollider>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub particle_emitter: Option<SerializedParticleEmitter>,
}

/// String-based asset references for mesh renderer serialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedMeshRef {
    pub mesh: String,
    pub material: String,
    pub tint: [f32; 4],
    pub visible: bool,
}

/// Serializable rigid body descriptor (excludes runtime physics handle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedRigidBody {
    pub body_type: SerializedBodyType,
    #[serde(default = "default_mass")]
    pub mass: f32,
}

fn default_mass() -> f32 { 1.0 }

/// Serializable body type enum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SerializedBodyType {
    Static,
    Dynamic,
    Kinematic,
}

impl From<BodyType> for SerializedBodyType {
    fn from(bt: BodyType) -> Self {
        match bt {
            BodyType::Static => Self::Static,
            BodyType::Dynamic => Self::Dynamic,
            BodyType::Kinematic => Self::Kinematic,
        }
    }
}

impl From<SerializedBodyType> for BodyType {
    fn from(bt: SerializedBodyType) -> Self {
        match bt {
            SerializedBodyType::Static => Self::Static,
            SerializedBodyType::Dynamic => Self::Dynamic,
            SerializedBodyType::Kinematic => Self::Kinematic,
        }
    }
}

/// Serializable collider descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedCollider {
    pub shape: SerializedColliderShape,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
    #[serde(default)]
    pub is_sensor: bool,
    #[serde(default = "default_collision_bits")]
    pub collision_group: u32,
    #[serde(default = "default_collision_bits")]
    pub collision_mask: u32,
}

fn default_friction() -> f32 { 0.5 }
fn default_collision_bits() -> u32 { 0xFFFF_FFFF }

/// Serializable collider shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedColliderShape {
    Box { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
    ConvexHull { points: Vec<[f32; 3]> },
    TriMesh { vertices: Vec<[f32; 3]>, indices: Vec<[u32; 3]> },
}

impl From<ColliderShape> for SerializedColliderShape {
    fn from(shape: ColliderShape) -> Self {
        match shape {
            ColliderShape::Box { half_extents } => Self::Box {
                half_extents: half_extents.into(),
            },
            ColliderShape::Sphere { radius } => Self::Sphere { radius },
            ColliderShape::Capsule { half_height, radius } => Self::Capsule { half_height, radius },
            ColliderShape::ConvexHull { points } => Self::ConvexHull {
                points: points.iter().map(|p| [p.x, p.y, p.z]).collect(),
            },
            ColliderShape::TriMesh { vertices, indices } => Self::TriMesh {
                vertices: vertices.iter().map(|v| [v.x, v.y, v.z]).collect(),
                indices,
            },
        }
    }
}

impl From<SerializedColliderShape> for ColliderShape {
    fn from(shape: SerializedColliderShape) -> Self {
        match shape {
            SerializedColliderShape::Box { half_extents } => Self::Box {
                half_extents: Vec3::from(half_extents),
            },
            SerializedColliderShape::Sphere { radius } => Self::Sphere { radius },
            SerializedColliderShape::Capsule { half_height, radius } => {
                Self::Capsule { half_height, radius }
            }
            SerializedColliderShape::ConvexHull { points } => Self::ConvexHull {
                points: points.into_iter().map(Vec3::from).collect(),
            },
            SerializedColliderShape::TriMesh { vertices, indices } => Self::TriMesh {
                vertices: vertices.into_iter().map(Vec3::from).collect(),
                indices,
            },
        }
    }
}

/// Serializable particle emitter descriptor (excludes GPU handles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedParticleEmitter {
    pub spawn_rate: f32,
    pub velocity_min: [f32; 3],
    pub velocity_max: [f32; 3],
    pub gravity: [f32; 3],
    pub lifetime: [f32; 2],
    pub size: [f32; 2],
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    #[serde(default = "default_active")]
    pub active: bool,
}

fn default_active() -> bool { true }

/// Save the current ECS world to a `SceneFile`.
///
/// Entities without a `Transform3D` are skipped. GPU handles in `MeshRenderer`
/// are resolved to string names via the asset manager's reverse map.
pub fn save_scene(world: &World, assets: &AssetManager) -> SceneFile {
    // Collect all entities with Transform3D in a single pass.
    let transform_entities: Vec<_> = world.query::<&Transform3D>().iter()
        .map(|(entity, transform)| (entity, *transform))
        .collect();

    // Build entity → scene-local id mapping.
    let mut entity_to_id: HashMap<hecs::Entity, u32> = HashMap::new();
    for (i, (entity, _)) in transform_entities.iter().enumerate() {
        entity_to_id.insert(*entity, i as u32);
    }

    // Build scene entities.
    let mut entities = Vec::new();

    for &(entity, transform) in &transform_entities {
        let id = entity_to_id[&entity];

        let mesh_renderer = world
            .get::<&MeshRenderer>(entity)
            .ok()
            .and_then(|mr| {
                let mesh_name = assets.name_for_gpu_mesh(mr.mesh)?;
                let mat_name = assets.name_for_gpu_material(mr.material)?;
                Some(SerializedMeshRef {
                    mesh: mesh_name.to_owned(),
                    material: mat_name.to_owned(),
                    tint: mr.tint,
                    visible: mr.visible,
                })
            });

        let camera = world.get::<&Camera3D>(entity).ok().map(|c| *c);

        let point_light = world.get::<&PointLightComponent>(entity).ok().map(|l| *l);

        let spot_light = world.get::<&SpotLightComponent>(entity).ok().map(|l| *l);

        let directional_light = world.get::<&DirectionalLightComponent>(entity).ok().map(|l| *l);

        let parent = world
            .get::<&Parent>(entity)
            .ok()
            .and_then(|p| entity_to_id.get(&p.0).copied());

        let tag = world.get::<&Tag>(entity).ok().map(|t| t.0.clone());

        let rigid_body = world
            .get::<&RigidBodyComponent>(entity)
            .ok()
            .map(|rb| SerializedRigidBody {
                body_type: rb.body_type.into(),
                mass: 1.0, // mass not stored on component; use default
            });

        let collider = world
            .get::<&ColliderComponent>(entity)
            .ok()
            .map(|c| SerializedCollider {
                shape: c.shape.clone().into(),
                friction: c.friction,
                restitution: c.restitution,
                is_sensor: c.is_sensor,
                collision_group: c.collision_group,
                collision_mask: c.collision_mask,
            });

        // TriggerVolume implies a sensor collider; if we have one but no
        // explicit ColliderComponent, we still note it via the collider field's
        // is_sensor flag.  The collider field already captures this when both
        // components exist on the entity.

        let particle_emitter = world
            .get::<&ParticleEmitter>(entity)
            .ok()
            .map(|pe| SerializedParticleEmitter {
                spawn_rate: pe.spawn_rate,
                velocity_min: pe.velocity_min.into(),
                velocity_max: pe.velocity_max.into(),
                gravity: pe.gravity.into(),
                lifetime: pe.lifetime,
                size: pe.size,
                color_start: pe.color_start,
                color_end: pe.color_end,
                active: pe.active,
            });

        entities.push(SceneEntity {
            id,
            transform: Some(transform),
            mesh_renderer,
            camera,
            point_light,
            spot_light,
            directional_light,
            parent,
            tag,
            rigid_body,
            collider,
            particle_emitter,
        });
    }

    SceneFile { version: 1, entities }
}

/// Serialize a `SceneFile` to a RON string.
pub fn scene_to_ron(scene: &SceneFile) -> Result<String, ron::Error> {
    let config = ron::ser::PrettyConfig::default();
    ron::ser::to_string_pretty(scene, config)
}

/// Deserialize a `SceneFile` from a RON string.
pub fn scene_from_ron(ron_str: &str) -> Result<SceneFile, ron::error::SpannedError> {
    ron::from_str(ron_str)
}

/// Load a `SceneFile` into the ECS world, resolving asset references.
///
/// Returns a mapping from scene-local ID to the spawned `hecs::Entity`.
/// Entities with unresolvable mesh/material names will be spawned without
/// a `MeshRenderer` (a warning is logged).
///
/// If `physics` and `entity_map` are provided, rigid body and collider
/// components are reconstructed via the physics backend.
pub fn load_scene(
    scene: &SceneFile,
    world: &mut World,
    assets: &AssetManager,
    mut physics: Option<&mut dyn PhysicsBackend>,
    mut entity_map: Option<&mut PhysicsEntityMap>,
) -> HashMap<u32, hecs::Entity> {
    let mut id_to_entity: HashMap<u32, hecs::Entity> = HashMap::new();

    // First pass: spawn entities with components (except Parent/Children).
    for se in &scene.entities {
        let transform = se.transform.unwrap_or_default();

        let entity = world.spawn((transform, crate::ecs::components::GlobalTransform::default()));

        // MeshRenderer
        if let Some(ref mr) = se.mesh_renderer {
            let mesh = assets.find_mesh_by_name(&mr.mesh);
            let material = assets.find_material_by_name(&mr.material);
            match (mesh, material) {
                (Some(m), Some(mat)) => {
                    let _ = world.insert_one(
                        entity,
                        MeshRenderer {
                            mesh: m,
                            material: mat,
                            tint: mr.tint,
                            visible: mr.visible,
                        },
                    );
                }
                _ => {
                    tracing::warn!(
                        "scene entity {}: unresolvable mesh='{}' or material='{}'",
                        se.id, mr.mesh, mr.material,
                    );
                }
            }
        }

        if let Some(ref cam) = se.camera {
            let _ = world.insert_one(
                entity,
                Camera3D {
                    fov_y: cam.fov_y,
                    near: cam.near,
                    far: cam.far,
                    active: cam.active,
                },
            );
        }

        if let Some(ref pl) = se.point_light {
            let _ = world.insert_one(
                entity,
                PointLightComponent {
                    color: pl.color,
                    intensity: pl.intensity,
                    range: pl.range,
                    cast_shadows: pl.cast_shadows,
                },
            );
        }

        if let Some(ref sl) = se.spot_light {
            let _ = world.insert_one(
                entity,
                SpotLightComponent {
                    color: sl.color,
                    intensity: sl.intensity,
                    range: sl.range,
                    inner_cone_angle: sl.inner_cone_angle,
                    outer_cone_angle: sl.outer_cone_angle,
                    cast_shadows: sl.cast_shadows,
                },
            );
        }

        if let Some(ref dl) = se.directional_light {
            let _ = world.insert_one(
                entity,
                DirectionalLightComponent {
                    color: dl.color,
                    intensity: dl.intensity,
                },
            );
        }

        if let Some(ref tag) = se.tag {
            let _ = world.insert_one(entity, Tag(tag.clone()));
        }

        // Reconstruct physics body + collider via the backend.
        if let Some(ref rb) = se.rigid_body {
            if let (Some(physics), Some(entity_map)) = (physics.as_deref_mut(), entity_map.as_deref_mut()) {
                let collider_desc = se.collider.as_ref().map(|c| ColliderDesc {
                    shape: c.shape.clone().into(),
                    friction: c.friction,
                    restitution: c.restitution,
                    is_sensor: c.is_sensor,
                    collision_group: c.collision_group,
                    collision_mask: c.collision_mask,
                });
                let body_type: BodyType = rb.body_type.into();
                let handle = physics.add_body(BodyDesc {
                    position: transform.position,
                    rotation: transform.rotation,
                    body_type,
                    collider: collider_desc.clone(),
                });
                entity_map.insert(handle, entity);
                let _ = world.insert_one(entity, RigidBodyComponent { handle, body_type });
                if let Some(ref c) = se.collider {
                    let _ = world.insert_one(entity, ColliderComponent {
                        shape: c.shape.clone().into(),
                        friction: c.friction,
                        restitution: c.restitution,
                        is_sensor: c.is_sensor,
                        collision_group: c.collision_group,
                        collision_mask: c.collision_mask,
                    });
                    if c.is_sensor {
                        let _ = world.insert_one(entity, TriggerVolume { tag: None });
                    }
                }
            }
        } else if let Some(ref c) = se.collider {
            // Collider without a rigid body — store the component data for reference.
            let _ = world.insert_one(entity, ColliderComponent {
                shape: c.shape.clone().into(),
                friction: c.friction,
                restitution: c.restitution,
                is_sensor: c.is_sensor,
                collision_group: c.collision_group,
                collision_mask: c.collision_mask,
            });
        }

        // Note: ParticleEmitter requires GPU pool/material handles that must be
        // set up by game code after load.  We store the serialized descriptor as
        // a ParticleEmitter with default handles; game code is expected to fix
        // the pool/material before the emitter produces visible output.
        if let Some(ref pe) = se.particle_emitter {
            let _ = world.insert_one(entity, ParticleEmitter {
                spawn_rate: pe.spawn_rate,
                velocity_min: Vec3::from(pe.velocity_min),
                velocity_max: Vec3::from(pe.velocity_max),
                gravity: Vec3::from(pe.gravity),
                lifetime: pe.lifetime,
                size: pe.size,
                color_start: pe.color_start,
                color_end: pe.color_end,
                active: pe.active,
                ..Default::default()
            });
        }

        id_to_entity.insert(se.id, entity);
    }

    // Second pass: set up Parent and Children.
    let mut children_map: HashMap<hecs::Entity, Vec<hecs::Entity>> = HashMap::new();

    for se in &scene.entities {
        if let Some(parent_id) = se.parent {
            if let (Some(&child_entity), Some(&parent_entity)) =
                (id_to_entity.get(&se.id), id_to_entity.get(&parent_id))
            {
                let _ = world.insert_one(child_entity, Parent(parent_entity));
                children_map
                    .entry(parent_entity)
                    .or_default()
                    .push(child_entity);
            }
        }
    }

    for (parent_entity, children) in children_map {
        let _ = world.insert_one(parent_entity, Children(children));
    }

    id_to_entity
}

/// Despawn all entities except those in `keep` and clear physics state.
pub fn clear_scene(
    world: &mut World,
    physics: &mut dyn PhysicsBackend,
    entity_map: &mut PhysicsEntityMap,
    keep: &[hecs::Entity],
) {
    // Collect entities to despawn.
    let to_despawn: Vec<hecs::Entity> = world
        .iter()
        .map(|e| e.entity())
        .filter(|e| !keep.contains(e))
        .collect();

    for entity in &to_despawn {
        // Remove physics body if mapped.
        if let Some(handle) = entity_map.get_handle(*entity) {
            physics.remove_body(handle);
        }
        entity_map.remove_by_entity(*entity);
        let _ = world.despawn(*entity);
    }
}

// ── Prefab support ──
//
// A prefab is just a `SceneFile` treated as an entity template. These are
// thin convenience wrappers around the existing scene serialization.

/// Serialize a prefab (small scene) to a RON string.
pub fn prefab_to_ron(prefab: &SceneFile) -> Result<String, ron::Error> {
    scene_to_ron(prefab)
}

/// Deserialize a prefab from a RON string.
pub fn prefab_from_ron(ron_str: &str) -> Result<SceneFile, ron::error::SpannedError> {
    scene_from_ron(ron_str)
}

/// Instantiate a prefab into the world with a root transform offset.
///
/// Works like `load_scene` but applies `offset` to every root entity's
/// transform (entities without a parent in the prefab).
pub fn instantiate_prefab(
    prefab: &SceneFile,
    world: &mut World,
    assets: &AssetManager,
    offset: Transform3D,
) -> HashMap<u32, hecs::Entity> {
    // Determine which scene IDs are root entities (have no parent).
    let roots: std::collections::HashSet<u32> = prefab
        .entities
        .iter()
        .filter(|e| e.parent.is_none())
        .map(|e| e.id)
        .collect();

    // Build an offset-applied copy of the scene.
    let offset_mat = offset.matrix();
    let adjusted: Vec<SceneEntity> = prefab
        .entities
        .iter()
        .map(|se| {
            let mut se = SceneEntity {
                id: se.id,
                transform: se.transform,
                mesh_renderer: se.mesh_renderer.as_ref().map(|mr| SerializedMeshRef {
                    mesh: mr.mesh.clone(),
                    material: mr.material.clone(),
                    tint: mr.tint,
                    visible: mr.visible,
                }),
                camera: se.camera,
                point_light: se.point_light,
                spot_light: se.spot_light,
                directional_light: se.directional_light,
                parent: se.parent,
                tag: se.tag.clone(),
                rigid_body: se.rigid_body.clone(),
                collider: se.collider.clone(),
                particle_emitter: se.particle_emitter.clone(),
            };
            if roots.contains(&se.id) {
                if let Some(ref mut t) = se.transform {
                    let combined = offset_mat * t.matrix();
                    let (scale, rotation, position) = combined.to_scale_rotation_translation();
                    *t = Transform3D {
                        position,
                        rotation,
                        scale,
                    };
                } else {
                    se.transform = Some(offset);
                }
            }
            se
        })
        .collect();

    let adjusted_scene = SceneFile { version: 1, entities: adjusted };
    load_scene(&adjusted_scene, world, assets, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn roundtrip_empty_scene() {
        let scene = SceneFile {
            version: 1,
            entities: vec![],
        };
        let ron_str = scene_to_ron(&scene).unwrap();
        let loaded = scene_from_ron(&ron_str).unwrap();
        assert!(loaded.entities.is_empty());
    }

    #[test]
    fn roundtrip_transform_only() {
        let world = &mut World::new();
        let assets = AssetManager::new();

        let t = Transform3D {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        world.spawn((t, crate::ecs::components::GlobalTransform::default()));

        let scene = save_scene(world, &assets);
        assert_eq!(scene.entities.len(), 1);

        let ron_str = scene_to_ron(&scene).unwrap();
        let scene2 = scene_from_ron(&ron_str).unwrap();
        assert_eq!(scene2.entities.len(), 1);

        let se = &scene2.entities[0];
        let loaded_t = se.transform.unwrap();
        assert!((loaded_t.position - t.position).length() < 1e-6);
    }

    #[test]
    fn roundtrip_with_hierarchy() {
        let world = &mut World::new();
        let assets = AssetManager::new();

        let parent_t = Transform3D {
            position: Vec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };
        let child_t = Transform3D {
            position: Vec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        };

        let parent = world.spawn((
            parent_t,
            crate::ecs::components::GlobalTransform::default(),
        ));
        let child = world.spawn((
            child_t,
            crate::ecs::components::GlobalTransform::default(),
            Parent(parent),
        ));
        let _ = world.insert_one(parent, Children(vec![child]));

        let scene = save_scene(world, &assets);
        assert_eq!(scene.entities.len(), 2);

        // Roundtrip through RON.
        let ron_str = scene_to_ron(&scene).unwrap();
        let scene2 = scene_from_ron(&ron_str).unwrap();

        // Load into a new world.
        let world2 = &mut World::new();
        let id_map = load_scene(&scene2, world2, &assets, None, None);
        assert_eq!(id_map.len(), 2);

        // Verify parent-child relationship is reconstructed.
        let child_scene_id = scene2
            .entities
            .iter()
            .find(|e| e.parent.is_some())
            .unwrap()
            .id;
        let child_entity = id_map[&child_scene_id];
        let parent_ref = world2.get::<&Parent>(child_entity).unwrap();
        let parent_scene_id = scene2
            .entities
            .iter()
            .find(|e| e.parent.is_none())
            .unwrap()
            .id;
        assert_eq!(parent_ref.0, id_map[&parent_scene_id]);
    }

    #[test]
    fn roundtrip_with_camera_and_lights() {
        let world = &mut World::new();
        let assets = AssetManager::new();

        let t = Transform3D::default();
        let cam = Camera3D {
            fov_y: 1.2,
            near: 0.5,
            far: 500.0,
            active: true,
        };
        let pl = PointLightComponent {
            color: [1.0, 0.8, 0.6],
            intensity: 100.0,
            range: 50.0,
            cast_shadows: false,
        };
        world.spawn((
            t,
            crate::ecs::components::GlobalTransform::default(),
            cam,
            pl,
        ));

        let scene = save_scene(world, &assets);
        let ron_str = scene_to_ron(&scene).unwrap();
        let scene2 = scene_from_ron(&ron_str).unwrap();

        let se = &scene2.entities[0];
        let loaded_cam = se.camera.as_ref().unwrap();
        assert!((loaded_cam.fov_y - 1.2).abs() < 1e-6);
        assert!(loaded_cam.active);

        let loaded_pl = se.point_light.as_ref().unwrap();
        assert!((loaded_pl.intensity - 100.0).abs() < 1e-6);
    }

    #[test]
    fn prefab_instantiate_with_offset() {
        let prefab = SceneFile {
            version: 1,
            entities: vec![SceneEntity {
                id: 0,
                transform: Some(Transform3D {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                }),
                mesh_renderer: None,
                camera: None,
                point_light: None,
                spot_light: None,
                directional_light: None,
                parent: None,
                tag: None,
                rigid_body: None,
                collider: None,
                particle_emitter: None,
            }],
        };

        let offset = Transform3D {
            position: Vec3::new(10.0, 20.0, 30.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        let world = &mut World::new();
        let assets = AssetManager::new();
        let id_map = instantiate_prefab(&prefab, world, &assets, offset);

        assert_eq!(id_map.len(), 1);
        let entity = id_map[&0];
        let t = world.get::<&Transform3D>(entity).unwrap();
        // Combined position: offset (10,20,30) + entity local (1,0,0) = (11,20,30)
        assert!((t.position - Vec3::new(11.0, 20.0, 30.0)).length() < 1e-5);
    }

    #[test]
    fn prefab_roundtrip_through_ron() {
        let prefab = SceneFile {
            version: 1,
            entities: vec![SceneEntity {
                id: 0,
                transform: Some(Transform3D::default()),
                mesh_renderer: None,
                camera: None,
                point_light: None,
                spot_light: None,
                directional_light: None,
                parent: None,
                tag: None,
                rigid_body: None,
                collider: None,
                particle_emitter: None,
            }],
        };

        let ron_str = prefab_to_ron(&prefab).unwrap();
        let loaded = prefab_from_ron(&ron_str).unwrap();
        assert_eq!(loaded.entities.len(), 1);
    }
}
