//! Entity Component System — components and systems for 3D game entities.

pub mod components;
pub mod hierarchy;
pub mod lod;
pub mod particle_components;
pub mod physics_components;
pub mod physics_sync;
pub mod systems;

pub use components::{
    AnimGraphController, Animator, Camera3D, DirectionalLightComponent, GlobalTransform,
    MeshRenderer, PointLightComponent, SpotLightComponent, Tag, Transform3D,
};
pub use hierarchy::{Children, Parent, hierarchy_system};
pub use physics_components::{ColliderComponent, RigidBodyComponent, TriggerVolume};
pub use particle_components::ParticleEmitter;
pub use lod::{LodLevel, LodMesh};
pub use physics_sync::physics_sync_system;
pub use systems::{
    animation_system, camera_sync_system, chunked_render_extraction_system,
    light_collection_system, particle_system, render_extraction_system,
};
