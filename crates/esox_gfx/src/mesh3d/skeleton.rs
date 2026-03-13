//! Skeletal animation — joint hierarchy, animation clips, and CPU-side playback.

use glam::{Mat4, Quat, Vec3};

use super::gltf_loader::{AnimChannel, AnimProperty, AnimationClip, GltfSkin, Interpolation};
use super::transform::Transform;

/// Animation player — evaluates animation clips and produces skinning matrices.
pub struct AnimationPlayer {
    /// Joint names (for debugging).
    joint_names: Vec<Option<String>>,
    /// Parent index per joint (-1 = root).
    parent_indices: Vec<i32>,
    /// Inverse bind matrices (one per joint).
    inverse_bind_matrices: Vec<Mat4>,
    /// Bind-pose local transforms (rest pose, used for reset/clone).
    bind_pose_transforms: Vec<Transform>,
    /// Current local transforms per joint.
    local_transforms: Vec<Transform>,
    /// Computed world matrices per joint (root-to-leaf order).
    world_matrices: Vec<Mat4>,
    /// Final skinning matrices: world * inverse_bind.
    skinning_matrices: Vec<Mat4>,
    /// Number of joints.
    joint_count: usize,
    /// Currently playing clip index.
    current_clip: Option<usize>,
    /// Current playback time in seconds.
    time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether to loop the animation.
    pub looping: bool,
}

impl AnimationPlayer {
    /// Create a new animation player from a glTF skin.
    pub fn new(skin: &GltfSkin) -> Self {
        let joint_count = skin.joint_count;
        let mut player = Self {
            joint_names: skin.joint_names.clone(),
            parent_indices: skin.parent_indices.clone(),
            inverse_bind_matrices: skin.inverse_bind_matrices.clone(),
            bind_pose_transforms: skin.bind_pose_transforms.clone(),
            local_transforms: skin.bind_pose_transforms.clone(),
            world_matrices: vec![Mat4::IDENTITY; joint_count],
            skinning_matrices: vec![Mat4::IDENTITY; joint_count],
            joint_count,
            current_clip: None,
            time: 0.0,
            speed: 1.0,
            looping: true,
        };
        // Compute initial hierarchy so skinning matrices are valid before first advance().
        player.compute_hierarchy();
        player
    }

    /// Start playing an animation clip by index.
    pub fn play(&mut self, clip_index: usize, looping: bool) {
        self.current_clip = Some(clip_index);
        self.time = 0.0;
        self.looping = looping;
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.current_clip = None;
    }

    /// Get the current playback time.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Get the currently playing clip index.
    pub fn current_clip(&self) -> Option<usize> {
        self.current_clip
    }

    /// Number of joints.
    pub fn joint_count(&self) -> usize {
        self.joint_count
    }

    /// Advance animation by `dt` seconds and recompute skinning matrices.
    pub fn advance(&mut self, dt: f32, clips: &[AnimationClip]) {
        let clip_idx = match self.current_clip {
            Some(idx) if idx < clips.len() => idx,
            _ => {
                // No active clip — just recompute hierarchy from current local transforms.
                self.compute_hierarchy();
                return;
            }
        };

        let clip = &clips[clip_idx];
        self.time += dt * self.speed;

        if clip.duration > 0.0 {
            if self.looping {
                self.time = self.time.rem_euclid(clip.duration);
            } else {
                self.time = self.time.clamp(0.0, clip.duration);
            }
        }

        // Sample each channel and apply to local transforms.
        for channel in &clip.channels {
            if channel.joint_index >= self.joint_count {
                continue;
            }
            let value = sample_channel(channel, self.time);
            let transform = &mut self.local_transforms[channel.joint_index];
            match channel.property {
                AnimProperty::Translation => {
                    transform.position = Vec3::new(value[0], value[1], value[2]);
                }
                AnimProperty::Rotation => {
                    transform.rotation = Quat::from_xyzw(value[0], value[1], value[2], value[3])
                        .normalize();
                }
                AnimProperty::Scale => {
                    transform.scale = Vec3::new(value[0], value[1], value[2]);
                }
            }
        }

        self.compute_hierarchy();
    }

    /// Recompute world and skinning matrices from current local transforms.
    fn compute_hierarchy(&mut self) {
        // Root-to-leaf pass.
        for i in 0..self.joint_count {
            let local = self.local_transforms[i].matrix();
            let parent = self.parent_indices[i];
            self.world_matrices[i] = if parent >= 0 && (parent as usize) < self.joint_count {
                self.world_matrices[parent as usize] * local
            } else {
                local
            };
            self.skinning_matrices[i] =
                self.world_matrices[i] * self.inverse_bind_matrices[i];
        }
    }

    /// Get the computed skinning matrices (one per joint).
    ///
    /// These are `world_matrix * inverse_bind_matrix` for each joint,
    /// ready to be uploaded to the GPU.
    pub fn skinning_matrices(&self) -> &[Mat4] {
        &self.skinning_matrices
    }

    /// Create a fresh player sharing the same skeleton data but with reset playback state.
    pub fn clone_skeleton(&self) -> Self {
        let mut cloned = Self {
            joint_names: self.joint_names.clone(),
            parent_indices: self.parent_indices.clone(),
            inverse_bind_matrices: self.inverse_bind_matrices.clone(),
            bind_pose_transforms: self.bind_pose_transforms.clone(),
            local_transforms: self.bind_pose_transforms.clone(),
            world_matrices: vec![Mat4::IDENTITY; self.joint_count],
            skinning_matrices: vec![Mat4::IDENTITY; self.joint_count],
            joint_count: self.joint_count,
            current_clip: None,
            time: 0.0,
            speed: 1.0,
            looping: true,
        };
        cloned.compute_hierarchy();
        cloned
    }

    /// Get a joint name by index.
    pub fn joint_name(&self, index: usize) -> Option<&str> {
        self.joint_names.get(index).and_then(|n| n.as_deref())
    }
}

/// Sample a single animation channel at the given time.
fn sample_channel(channel: &AnimChannel, time: f32) -> [f32; 4] {
    let times = &channel.times;
    let values = &channel.values;

    if times.is_empty() || values.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }

    // Before first keyframe.
    if time <= times[0] {
        return values[0];
    }

    // After last keyframe.
    if time >= *times.last().unwrap() {
        return *values.last().unwrap();
    }

    // Binary search for the keyframe pair.
    let idx = match times.binary_search_by(|t| t.partial_cmp(&time).unwrap()) {
        Ok(i) => return values[i], // Exact match
        Err(i) => i,
    };

    let i0 = idx.saturating_sub(1);
    let i1 = idx.min(times.len() - 1);

    if i0 == i1 {
        return values[i0];
    }

    let t0 = times[i0];
    let t1 = times[i1];
    let t = if (t1 - t0).abs() > 1e-8 {
        (time - t0) / (t1 - t0)
    } else {
        0.0
    };

    match channel.interpolation {
        Interpolation::Step => values[i0],
        Interpolation::Linear => {
            match channel.property {
                AnimProperty::Rotation => {
                    // Slerp for quaternions (normalize keyframes for robustness).
                    let q0 = Quat::from_array(values[i0]).normalize();
                    let q1 = Quat::from_array(values[i1]).normalize();
                    q0.slerp(q1, t).to_array()
                }
                _ => {
                    // Lerp for translation/scale.
                    let a = values[i0];
                    let b = values[i1];
                    [
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t,
                        a[3] + (b[3] - a[3]) * t,
                    ]
                }
            }
        }
        Interpolation::CubicSpline => {
            // Cubic spline has 3 values per keyframe: in-tangent, value, out-tangent.
            if values.len() < times.len() * 3 {
                // Fallback to linear if data is malformed.
                let a = values[i0];
                let b = values[i1];
                return [
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ];
            }
            let dt = t1 - t0;
            // Indices into cubic spline triples: [in_tangent, value, out_tangent]
            let mut v0 = values[i0 * 3 + 1]; // value at i0
            let b0 = values[i0 * 3 + 2]; // out-tangent at i0
            let a1 = values[i1 * 3]; // in-tangent at i1
            let mut v1 = values[i1 * 3 + 1]; // value at i1

            // Normalize quaternion keyframe values before interpolation.
            if channel.property == AnimProperty::Rotation {
                v0 = Quat::from_array(v0).normalize().to_array();
                v1 = Quat::from_array(v1).normalize().to_array();
            }

            let t2 = t * t;
            let t3 = t2 * t;

            let mut result = [0.0f32; 4];
            for c in 0..4 {
                result[c] = (2.0 * t3 - 3.0 * t2 + 1.0) * v0[c]
                    + (t3 - 2.0 * t2 + t) * dt * b0[c]
                    + (-2.0 * t3 + 3.0 * t2) * v1[c]
                    + (t3 - t2) * dt * a1[c];
            }

            if channel.property == AnimProperty::Rotation {
                let q = Quat::from_array(result).normalize();
                q.to_array()
            } else {
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gltf_loader::{AnimChannel, AnimProperty, AnimationClip, GltfSkin, Interpolation};

    fn make_test_skin(joint_count: usize) -> GltfSkin {
        GltfSkin {
            joint_names: vec![None; joint_count],
            parent_indices: (0..joint_count as i32)
                .map(|i| if i == 0 { -1 } else { i - 1 })
                .collect(),
            inverse_bind_matrices: vec![Mat4::IDENTITY; joint_count],
            bind_pose_transforms: vec![Transform::IDENTITY; joint_count],
            joint_count,
        }
    }

    #[test]
    fn new_player_has_identity_skinning() {
        let skin = make_test_skin(3);
        let player = AnimationPlayer::new(&skin);
        for m in player.skinning_matrices() {
            assert_eq!(*m, Mat4::IDENTITY);
        }
    }

    #[test]
    fn advance_without_clip_keeps_identity() {
        let skin = make_test_skin(2);
        let mut player = AnimationPlayer::new(&skin);
        player.advance(1.0, &[]);
        for m in player.skinning_matrices() {
            assert_eq!(*m, Mat4::IDENTITY);
        }
    }

    #[test]
    fn linear_translation_interpolation() {
        let channel = AnimChannel {
            joint_index: 0,
            property: AnimProperty::Translation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: vec![[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        };

        let result = sample_channel(&channel, 0.5);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1]).abs() < 1e-5);
    }

    #[test]
    fn step_interpolation() {
        let channel = AnimChannel {
            joint_index: 0,
            property: AnimProperty::Translation,
            interpolation: Interpolation::Step,
            times: vec![0.0, 1.0],
            values: vec![[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        };

        let result = sample_channel(&channel, 0.5);
        assert_eq!(result, [0.0, 0.0, 0.0, 0.0]); // step: still at first value
    }

    #[test]
    fn rotation_slerp() {
        let q0 = Quat::IDENTITY;
        let q1 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);

        let channel = AnimChannel {
            joint_index: 0,
            property: AnimProperty::Rotation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: vec![q0.to_array(), q1.to_array()],
        };

        let result = sample_channel(&channel, 0.5);
        let q = Quat::from_array(result);
        let expected = q0.slerp(q1, 0.5);
        assert!(
            (q.dot(expected) - 1.0).abs() < 1e-4,
            "slerp result mismatch"
        );
    }

    #[test]
    fn looping_wraps_time() {
        let skin = make_test_skin(1);
        let mut player = AnimationPlayer::new(&skin);

        let clip = AnimationClip {
            name: None,
            duration: 2.0,
            channels: vec![AnimChannel {
                joint_index: 0,
                property: AnimProperty::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 2.0],
                values: vec![[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
            }],
        };

        player.play(0, true);
        player.advance(3.0, &[clip]); // 3.0 mod 2.0 = 1.0
        assert!((player.time() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn non_looping_clamps_time() {
        let skin = make_test_skin(1);
        let mut player = AnimationPlayer::new(&skin);

        let clip = AnimationClip {
            name: None,
            duration: 2.0,
            channels: vec![],
        };

        player.play(0, false);
        player.advance(5.0, &[clip]);
        assert!((player.time() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn clone_skeleton_resets_playback() {
        let skin = make_test_skin(3);
        let mut original = AnimationPlayer::new(&skin);

        let clip = AnimationClip {
            name: None,
            duration: 2.0,
            channels: vec![AnimChannel {
                joint_index: 0,
                property: AnimProperty::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 2.0],
                values: vec![[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
            }],
        };

        original.play(0, true);
        original.advance(1.0, &[clip]);
        assert!(original.time() > 0.0);
        assert!(original.current_clip().is_some());

        let cloned = original.clone_skeleton();
        assert_eq!(cloned.joint_count(), original.joint_count());
        assert!(cloned.current_clip().is_none());
        assert!((cloned.time() - 0.0).abs() < 1e-5);
        for m in cloned.skinning_matrices() {
            assert_eq!(*m, Mat4::IDENTITY);
        }
    }

    #[test]
    fn joint_hierarchy_propagation() {
        // 3 joints in a chain: root → child → grandchild
        // Each translates 1 unit along X.
        let skin = make_test_skin(3);
        let mut player = AnimationPlayer::new(&skin);

        let clip = AnimationClip {
            name: None,
            duration: 1.0,
            channels: vec![
                AnimChannel {
                    joint_index: 0,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[1.0, 0.0, 0.0, 0.0]],
                },
                AnimChannel {
                    joint_index: 1,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[1.0, 0.0, 0.0, 0.0]],
                },
                AnimChannel {
                    joint_index: 2,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[1.0, 0.0, 0.0, 0.0]],
                },
            ],
        };

        player.play(0, false);
        player.advance(0.0, &[clip]);

        // Root: world = translate(1,0,0)
        let root_pos = player.skinning_matrices()[0].col(3).truncate();
        assert!((root_pos.x - 1.0).abs() < 1e-5, "root x={}", root_pos.x);

        // Child: world = translate(2,0,0) (parent + own)
        let child_pos = player.skinning_matrices()[1].col(3).truncate();
        assert!(
            (child_pos.x - 2.0).abs() < 1e-5,
            "child x={}",
            child_pos.x
        );

        // Grandchild: world = translate(3,0,0)
        let gc_pos = player.skinning_matrices()[2].col(3).truncate();
        assert!((gc_pos.x - 3.0).abs() < 1e-5, "grandchild x={}", gc_pos.x);
    }
}
