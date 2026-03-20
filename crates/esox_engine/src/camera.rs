//! Reusable camera controllers for common game camera patterns.
//!
//! Each controller stores previous + current state for interpolation.
//! Game code calls `update()` each fixed tick and `apply(alpha)` each render
//! frame, then writes the result to the camera entity's `Transform3D`.

use glam::{Mat3, Quat, Vec3};

use crate::input::InputManager;
use crate::time::TimeState;

// ── Helpers ──

fn look_at_quat(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize_or_zero();
    if forward.length_squared() < 1e-8 {
        return Quat::IDENTITY;
    }
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    if right.length_squared() < 1e-8 {
        // Looking straight up or down — pick arbitrary right.
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward);
        return Quat::from_mat3(&Mat3::from_cols(right, up, -forward));
    }
    let up = right.cross(forward);
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward))
}

// ── FPS Camera ──

/// First-person camera controller.
///
/// Reads input axes `"look_x"`, `"look_y"`, `"move_x"`, `"move_z"`.
pub struct FpsCameraController {
    pub sensitivity: f32,
    pub pitch_clamp: f32,
    pub move_speed: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub position: Vec3,
    pub prev_position: Vec3,
}

impl FpsCameraController {
    pub fn new(position: Vec3) -> Self {
        Self {
            sensitivity: 0.003,
            pitch_clamp: 1.4,
            move_speed: 5.0,
            yaw: 0.0,
            pitch: 0.0,
            position,
            prev_position: position,
        }
    }

    pub fn update(&mut self, input: &InputManager, time: &TimeState) {
        let dt = time.tick_dt;

        // Mouse look.
        let look_x = input.axis("look_x");
        let look_y = input.axis("look_y");
        self.yaw -= look_x * self.sensitivity;
        self.pitch = (self.pitch - look_y * self.sensitivity).clamp(-self.pitch_clamp, self.pitch_clamp);

        // Movement.
        let move_x = input.axis("move_x");
        let move_z = input.axis("move_z");
        let (sin_y, cos_y) = self.yaw.sin_cos();
        let forward = Vec3::new(-sin_y, 0.0, -cos_y);
        let right = Vec3::new(cos_y, 0.0, -sin_y);
        let move_dir = (forward * move_z + right * move_x).normalize_or_zero();

        self.prev_position = self.position;
        self.position += move_dir * self.move_speed * dt;
    }

    /// Returns `(position, rotation)` interpolated for the current render frame.
    pub fn apply(&self, alpha: f32) -> (Vec3, Quat) {
        let pos = self.prev_position.lerp(self.position, alpha);
        let rot = Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        (pos, rot)
    }
}

// ── Orbit Camera ──

/// Third-person orbit camera (extracted from the platformer pattern).
///
/// Orbits around a target position at a fixed distance and pitch.
/// Call `set_target()` each tick with the followed entity's position,
/// and `rotate()` to snap the orbit angle.
pub struct OrbitCameraController {
    pub distance: f32,
    pub pitch: f32,
    pub height_offset: f32,
    pub lerp_speed: f32,
    pub yaw: f32,
    /// Target yaw angle (snaps immediately on `rotate()`; `yaw` lerps toward it).
    /// Useful for camera-relative movement where controls should respond instantly.
    pub yaw_target: f32,
    target_pos: Vec3,
    prev_target_pos: Vec3,
}

impl OrbitCameraController {
    pub fn new(distance: f32, pitch: f32) -> Self {
        Self {
            distance,
            pitch,
            height_offset: 1.5,
            lerp_speed: 12.0,
            yaw: 0.0,
            yaw_target: 0.0,
            target_pos: Vec3::ZERO,
            prev_target_pos: Vec3::ZERO,
        }
    }

    /// Set the world-space position of the entity being followed.
    pub fn set_target(&mut self, pos: Vec3) {
        self.prev_target_pos = self.target_pos;
        self.target_pos = pos;
    }

    /// Add a delta to the orbit yaw target (e.g. `FRAC_PI_2` for a 90-degree snap).
    pub fn rotate(&mut self, delta_yaw: f32) {
        self.yaw_target += delta_yaw;
    }

    pub fn update(&mut self, _input: &InputManager, time: &TimeState) {
        let dt = time.tick_dt;
        let diff = self.yaw_target - self.yaw;
        self.yaw += diff * (self.lerp_speed * dt).min(1.0);
    }

    fn camera_position(&self, target: Vec3) -> Vec3 {
        let (sin_o, cos_o) = self.yaw.sin_cos();
        Vec3::new(
            target.x + self.distance * self.pitch.cos() * sin_o,
            target.y + self.height_offset + self.distance * self.pitch.sin(),
            target.z + self.distance * self.pitch.cos() * cos_o,
        )
    }

    /// Returns `(position, rotation)` interpolated for the current render frame.
    pub fn apply(&self, alpha: f32) -> (Vec3, Quat) {
        let visual_target = self.prev_target_pos.lerp(self.target_pos, alpha);
        let cam_pos = self.camera_position(visual_target);
        let look_target = visual_target + Vec3::Y * (self.height_offset * 0.33);
        (cam_pos, look_at_quat(cam_pos, look_target))
    }
}

// ── Follow Camera ──

/// Smooth third-person follow camera (extracted from the combat_demo pattern).
///
/// Follows the target with a fixed offset and configurable smoothing.
pub struct FollowCameraController {
    pub offset: Vec3,
    pub smoothing: f32,
    pub position: Vec3,
    pub prev_position: Vec3,
    target_pos: Vec3,
}

impl FollowCameraController {
    pub fn new(offset: Vec3) -> Self {
        Self {
            offset,
            smoothing: 4.0,
            position: Vec3::ZERO,
            prev_position: Vec3::ZERO,
            target_pos: Vec3::ZERO,
        }
    }

    /// Set the world-space position of the entity being followed.
    pub fn set_target(&mut self, pos: Vec3) {
        self.target_pos = pos;
    }

    pub fn update(&mut self, _input: &InputManager, time: &TimeState) {
        let dt = time.tick_dt;
        let desired = self.target_pos + self.offset;
        self.prev_position = self.position;
        self.position = self.position.lerp(desired, (self.smoothing * dt).min(1.0));
    }

    /// Returns `(position, rotation)` interpolated for the current render frame.
    pub fn apply(&self, alpha: f32) -> (Vec3, Quat) {
        let pos = self.prev_position.lerp(self.position, alpha);
        (pos, look_at_quat(pos, self.target_pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_camera_initial_state() {
        let cam = FpsCameraController::new(Vec3::new(0.0, 1.7, 0.0));
        let (pos, rot) = cam.apply(1.0);
        assert!((pos - Vec3::new(0.0, 1.7, 0.0)).length() < 1e-5);
        // At yaw=0, pitch=0, rotation should look along -Z.
        let forward = rot * -Vec3::Z;
        assert!((forward - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
    }

    #[test]
    fn fps_camera_interpolation() {
        let mut cam = FpsCameraController::new(Vec3::ZERO);
        cam.position = Vec3::new(10.0, 0.0, 0.0);
        let (pos, _) = cam.apply(0.5);
        assert!((pos.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn orbit_camera_position_at_zero_yaw() {
        let mut cam = OrbitCameraController::new(8.0, 0.4);
        cam.set_target(Vec3::ZERO);
        cam.set_target(Vec3::ZERO); // twice so prev == current
        let (pos, _) = cam.apply(1.0);
        // At yaw=0: x = 8*cos(0.4)*sin(0) = 0, z = 8*cos(0.4)*cos(0) = 8*cos(0.4)
        assert!(pos.x.abs() < 1e-4);
        assert!((pos.z - 8.0 * 0.4_f32.cos()).abs() < 0.1);
    }

    #[test]
    fn follow_camera_converges() {
        let time = TimeState {
            tick_dt: 1.0 / 60.0,
            frame_dt: 1.0 / 60.0,
            elapsed: 0.0,
            tick_count: 1,
            total_ticks: 0,
        };
        let input = InputManager::new();
        let mut cam = FollowCameraController::new(Vec3::new(0.0, 18.0, -14.0));
        cam.set_target(Vec3::new(5.0, 0.0, 0.0));
        // Run many ticks to converge.
        for _ in 0..600 {
            cam.update(&input, &time);
        }
        let desired = Vec3::new(5.0, 18.0, -14.0);
        let (pos, _) = cam.apply(1.0);
        assert!(
            (pos - desired).length() < 0.1,
            "camera should converge to target+offset, got {:?}",
            pos
        );
    }

    #[test]
    fn look_at_quat_basic() {
        let q = look_at_quat(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        // Should look along -Z, which is the default forward.
        let fwd = q * -Vec3::Z;
        assert!((fwd - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
    }
}
