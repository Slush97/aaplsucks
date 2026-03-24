//! 3D mesh rendering — instanced meshes with materials, lighting, and depth testing.
//!
//! This module provides a complete 3D mesh rendering pipeline that shares the
//! [`GpuContext`](crate::GpuContext) with the existing 2D instanced-quad renderer.
//!
//! # Architecture
//!
//! - [`Vertex3D`] — 64-byte vertex format (position, normal, uv, color, tangent)
//! - [`MeshData`] — CPU-side geometry with procedural generators (cube, sphere, plane, etc.)
//! - [`MeshHandle`] — lightweight reference to GPU-resident mesh
//! - [`InstanceData`] — 96-byte per-instance data (model matrix, color, params)
//! - [`Transform`] — position/rotation/scale → model matrix
//! - [`Camera`] — perspective projection + view matrix
//! - [`MaterialDescriptor`] / [`MaterialHandle`] — surface appearance (Unlit, Lit, PBR)
//! - [`LightEnvironment`] — ambient + directional + up to 8 point lights
//! - [`Renderer3D`] — frame encoder with materials, lighting, and sorted/merged draw batching
//! - [`GltfScene`] / [`GltfSceneHandles`] — glTF/GLB asset loading
//! - [`Scene3D`] — scene graph with hierarchical transforms
//! - [`AnimationPlayer`] — skeletal animation playback

pub mod bounds;
pub mod bvh;
pub mod camera;
pub mod depth_resolve;
pub mod gltf_loader;
pub mod ibl;
pub mod instance;
pub mod light;
pub mod material;
pub mod mesh;
pub mod postprocess;
pub mod render_types;
pub mod renderer;
pub mod scene;
pub mod shader_library;
pub mod shaders_embedded;
pub mod shadow;
pub mod ssao;
pub mod skeleton;
pub mod particles;
pub mod skinning;
pub mod texture;
pub mod transform;
pub mod vertex;

pub use bounds::{Aabb, Containment, Frustum};
pub use camera::{Camera, CameraMode};
pub use gltf_loader::{AnimationClip, GltfScene, GltfSceneHandles};
pub use ibl::IblState;
pub use instance::InstanceData;
pub use light::{DirectionalLight, LightEnvironment, PointLight, SpotLight};
pub use material::{BlendMode3D, MaterialDescriptor, MaterialHandle, MaterialType};
pub use mesh::{MeshData, MeshHandle};
pub use postprocess::PostProcess3DConfig;
pub use render_types::BatchStats3D;
pub use renderer::Renderer3D;
pub use scene::Scene3D;
pub use shadow::{OmniShadowUniforms, PointShadowPass, ShadowConfig, ShadowPass, ShadowUniforms, SpotShadowPass};
pub use ssao::{SsaoConfig, SsaoParams, SsaoPass};
pub use particles::{EmitterParams, ParticlePoolHandle};
pub use skeleton::AnimationPlayer;
pub use texture::TextureHandle;
pub use transform::Transform;
pub use vertex::Vertex3D;
