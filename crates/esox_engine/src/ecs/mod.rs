//! Entity Component System — components and systems for 3D game entities.

pub mod components;
pub mod hierarchy;
pub mod systems;

pub use components::{
    Camera3D, DirectionalLightComponent, GlobalTransform, MeshRenderer, PointLightComponent,
    SpotLightComponent, Transform3D,
};
pub use hierarchy::{Children, Parent, hierarchy_system};
pub use systems::{camera_sync_system, light_collection_system, render_extraction_system};
