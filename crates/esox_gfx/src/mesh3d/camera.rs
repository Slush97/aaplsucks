//! Camera — view and projection matrices for 3D rendering.

use glam::{Mat4, Vec3};

/// A 3D camera defined by position, look-at target, and projection parameters.
///
/// Uses a right-handed coordinate system with depth mapped to `[0, 1]` (wgpu convention).
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Point the camera looks at.
    pub target: Vec3,
    /// Up direction (usually `Vec3::Y`).
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    /// Compute the view matrix (world → camera space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Compute the perspective projection matrix for the given aspect ratio.
    ///
    /// Uses right-handed coordinates with depth range `[0, 1]`.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Compute the combined view-projection matrix.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    /// Forward direction (unit vector from position toward target).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_looks_at_origin() {
        let cam = Camera::default();
        assert_eq!(cam.target, Vec3::ZERO);
        assert!(cam.position.z > 0.0);
    }

    #[test]
    fn view_projection_is_product() {
        let cam = Camera::default();
        let aspect = 16.0 / 9.0;
        let vp = cam.view_projection(aspect);
        let expected = cam.projection_matrix(aspect) * cam.view_matrix();
        let diff = (vp - expected).abs().to_cols_array();
        assert!(diff.iter().all(|&d| d < 1e-6));
    }

    #[test]
    fn forward_direction() {
        let cam = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            ..Default::default()
        };
        let fwd = cam.forward();
        assert!((fwd - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
    }
}
