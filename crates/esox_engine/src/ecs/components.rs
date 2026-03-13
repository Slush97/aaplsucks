//! ECS components for 3D game entities.

use glam::{Mat4, Quat, Vec3};

use esox_gfx::mesh3d::{AnimationClip, AnimationPlayer, MaterialHandle, MeshHandle};

use crate::animation_graph::AnimGraphRuntime;

/// Local-space transform (position, rotation, scale).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform3D {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform3D {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform3D {
    /// Compute the local 4x4 model matrix.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Convert to the renderer's `Transform` type.
    pub fn to_gfx_transform(&self) -> esox_gfx::mesh3d::Transform {
        esox_gfx::mesh3d::Transform {
            position: self.position,
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

/// World-space transform computed by the hierarchy system.
#[derive(Debug, Clone, Copy)]
pub struct GlobalTransform(pub Mat4);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(Mat4::IDENTITY)
    }
}

/// Mesh renderer component — entities with this + Transform3D are drawn.
pub struct MeshRenderer {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub tint: [f32; 4],
    pub visible: bool,
}

/// Camera component.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct Camera3D {
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 1000.0,
            active: false,
        }
    }
}

/// Point light component. Position derived from Transform3D.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct PointLightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    #[cfg_attr(feature = "serialization", serde(default))]
    pub cast_shadows: bool,
}

/// Spot light component. Position and direction derived from Transform3D.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct SpotLightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    #[cfg_attr(feature = "serialization", serde(default))]
    pub cast_shadows: bool,
}

/// Directional light component. Direction derived from Transform3D rotation (forward = -Z).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct DirectionalLightComponent {
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Generic string tag for identifying entities across scene save/load.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct Tag(pub String);

/// Skeletal animation component — drives skinned meshes via an animation player.
pub struct Animator {
    pub player: AnimationPlayer,
    pub clips: Vec<AnimationClip>,
    pub skinned_mesh_index: usize,
}

/// Animation graph controller — drives skinned meshes via a state machine with
/// crossfade blending. Replaces `Animator` for entities that need multiple
/// animation states (idle/walk/run/jump).
pub struct AnimGraphController {
    pub graph: AnimGraphRuntime,
    pub clips: Vec<AnimationClip>,
    pub skinned_mesh_index: usize,
}
