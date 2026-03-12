//! Spatial transform — position, rotation, scale to model matrix.

use glam::{Mat4, Quat, Vec3};

/// A spatial transform composed of position, rotation, and scale.
///
/// Call [`matrix()`](Transform::matrix) to produce the model matrix for GPU upload.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// World-space position.
    pub position: Vec3,
    /// Orientation as a unit quaternion.
    pub rotation: Quat,
    /// Non-uniform scale.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// Identity transform: origin, no rotation, unit scale.
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Create a transform with only a position.
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Self::IDENTITY
        }
    }

    /// Create a transform with position and uniform scale.
    pub fn from_position_scale(position: Vec3, scale: f32) -> Self {
        Self {
            position,
            scale: Vec3::splat(scale),
            ..Self::IDENTITY
        }
    }

    /// Create a transform with position and rotation.
    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Compute the 4x4 model matrix (scale -> rotate -> translate).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Compute the 4x4 model matrix as a column-major `[[f32; 4]; 4]` for GPU upload.
    pub fn matrix_cols(&self) -> [[f32; 4]; 4] {
        self.matrix().to_cols_array_2d()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity_matrix() {
        let m = Transform::IDENTITY.matrix();
        assert_eq!(m, Mat4::IDENTITY);
    }

    #[test]
    fn position_only() {
        let t = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
        let m = t.matrix();
        let (_, _, translation) = m.to_scale_rotation_translation();
        assert!((translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn uniform_scale() {
        let t = Transform::from_position_scale(Vec3::ZERO, 2.0);
        let m = t.matrix();
        let (scale, _, _) = m.to_scale_rotation_translation();
        assert!((scale - Vec3::splat(2.0)).length() < 1e-6);
    }

    #[test]
    fn matrix_cols_matches_matrix() {
        let t = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            scale: Vec3::new(1.0, 2.0, 0.5),
        };
        let m = t.matrix();
        let cols = t.matrix_cols();
        assert_eq!(cols, m.to_cols_array_2d());
    }
}
