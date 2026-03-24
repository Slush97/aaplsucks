//! Camera — view and projection matrices for 3D rendering.

use glam::{Mat4, Vec3};

/// Projection mode for a 3D camera.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CameraMode {
    /// Standard perspective projection.
    Perspective,
    /// Orthographic projection. `ortho_size` is the half-height of the view
    /// volume; width is derived as `ortho_size * aspect`.
    Orthographic {
        ortho_size: f32,
    },
}

impl Default for CameraMode {
    fn default() -> Self {
        Self::Perspective
    }
}

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
    /// Projection mode (perspective or orthographic).
    pub mode: CameraMode,
    /// Vertical field of view in radians (used only in [`CameraMode::Perspective`]).
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
            mode: CameraMode::Perspective,
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

    /// Compute the projection matrix for the given aspect ratio.
    ///
    /// Uses right-handed coordinates with depth range `[0, 1]`.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        self.sub_projection(aspect, self.near, self.far)
    }

    /// Compute a projection matrix with custom near/far planes (same mode).
    ///
    /// Used by the shadow system to build per-cascade sub-frustum projections.
    pub fn sub_projection(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        match self.mode {
            CameraMode::Perspective => {
                Mat4::perspective_rh(self.fov_y, aspect, near, far)
            }
            CameraMode::Orthographic { ortho_size } => {
                let half_h = ortho_size;
                let half_w = ortho_size * aspect;
                Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
            }
        }
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

    #[test]
    fn orthographic_projection_is_symmetric() {
        let cam = Camera {
            mode: CameraMode::Orthographic { ortho_size: 10.0 },
            ..Default::default()
        };
        let aspect = 16.0 / 9.0;
        let proj = cam.projection_matrix(aspect);
        // Orthographic matrix should have zero perspective divide (row 3 = [0,0,_,_]).
        let cols = proj.to_cols_array_2d();
        assert!((cols[3][0]).abs() < 1e-6, "w.x should be 0");
        assert!((cols[3][1]).abs() < 1e-6, "w.y should be 0");
        assert!((cols[3][3] - 1.0).abs() < 1e-6, "w.w should be 1 for ortho");
    }

    #[test]
    fn sub_projection_matches_projection_at_default_planes() {
        let cam = Camera::default();
        let aspect = 16.0 / 9.0;
        let full = cam.projection_matrix(aspect);
        let sub = cam.sub_projection(aspect, cam.near, cam.far);
        let diff = (full - sub).abs().to_cols_array();
        assert!(diff.iter().all(|&d| d < 1e-6));
    }
}
