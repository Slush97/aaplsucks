//! Animation state machine — graph-driven animation with crossfade blending.
//!
//! `AnimGraphRuntime` sits between game code and `AnimationPlayer`. Game code
//! sets named parameters (speed, grounded, jumping). The graph evaluates
//! transition conditions, manages state changes with crossfade blending, and
//! outputs blended skinning matrices.

use std::collections::HashMap;

use glam::Mat4;

use esox_gfx::mesh3d::{AnimationClip, AnimationPlayer};

// ── Parameters ──

/// A parameter value used in transition conditions and blend trees.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
}

/// Named parameter store for driving animation graph logic.
#[derive(Debug, Clone, Default)]
pub struct AnimParams {
    params: HashMap<String, ParamValue>,
}

impl AnimParams {
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.params.insert(name.to_string(), ParamValue::Float(value));
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.params.insert(name.to_string(), ParamValue::Bool(value));
    }

    pub fn get_float(&self, name: &str) -> f32 {
        match self.params.get(name) {
            Some(ParamValue::Float(v)) => *v,
            _ => 0.0,
        }
    }

    pub fn get_bool(&self, name: &str) -> bool {
        match self.params.get(name) {
            Some(ParamValue::Bool(v)) => *v,
            _ => false,
        }
    }
}

// ── Conditions ──

/// A single condition evaluated against the parameter store.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub enum Condition {
    FloatGt { param: String, threshold: f32 },
    FloatLt { param: String, threshold: f32 },
    FloatBetween { param: String, min: f32, max: f32 },
    BoolTrue { param: String },
    BoolFalse { param: String },
}

impl Condition {
    fn evaluate(&self, params: &AnimParams) -> bool {
        match self {
            Condition::FloatGt { param, threshold } => params.get_float(param) > *threshold,
            Condition::FloatLt { param, threshold } => params.get_float(param) < *threshold,
            Condition::FloatBetween { param, min, max } => {
                let v = params.get_float(param);
                v >= *min && v <= *max
            }
            Condition::BoolTrue { param } => params.get_bool(param),
            Condition::BoolFalse { param } => !params.get_bool(param),
        }
    }
}

// ── State sources ──

/// An entry in a 1D blend tree, mapping a parameter value to a clip.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendEntry {
    pub clip_index: usize,
    pub threshold: f32,
}

/// What an animation state plays.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub enum StateSource {
    /// Single animation clip.
    Clip { clip_index: usize },
    /// 1D blend tree driven by a parameter. Entries must be sorted by threshold.
    BlendTree1D { param: String, entries: Vec<BlendEntry> },
}

// ── Animation events ──

/// An event defined on an animation state, fired when playback crosses `time`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct AnimEvent {
    pub name: String,
    pub time: f32,
}

/// An event emitted by the runtime when an `AnimEvent` fires.
#[derive(Debug, Clone)]
pub struct FiredEvent {
    pub name: String,
    pub state_name: String,
}

// ── Graph definition (data-driven) ──

/// A transition between animation states.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    /// Index of the target state in `AnimGraphDef::states`.
    pub target_state: usize,
    /// All conditions must be true for this transition to fire (AND logic).
    pub conditions: Vec<Condition>,
    /// Crossfade duration in seconds.
    pub duration: f32,
    /// Lower values are checked first.
    pub priority: u32,
}

/// A single state in the animation graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct AnimState {
    pub name: String,
    pub source: StateSource,
    pub looping: bool,
    pub speed: f32,
    pub transitions: Vec<Transition>,
    pub events: Vec<AnimEvent>,
}

/// Data-driven animation graph definition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct AnimGraphDef {
    pub states: Vec<AnimState>,
    pub default_state: usize,
}

// ── Crossfade state ──

struct CrossfadeState {
    frozen_matrices: Vec<Mat4>,
    elapsed: f32,
    duration: f32,
}

// ── Runtime ──

/// Animation graph runtime — evaluates the graph each frame and produces
/// blended skinning matrices.
pub struct AnimGraphRuntime {
    def: AnimGraphDef,
    pub params: AnimParams,
    current_state: usize,
    /// Primary player for the current state (single-clip states).
    player_a: AnimationPlayer,
    /// One player per blend tree entry (empty for single-clip states).
    blend_players: Vec<AnimationPlayer>,
    /// Active crossfade from previous state.
    crossfade: Option<CrossfadeState>,
    /// Output skinning matrices (blended).
    output: Vec<Mat4>,
    /// Scratch buffer to avoid allocating during crossfade.
    scratch: Vec<Mat4>,
    /// Events fired this frame, drained by game code.
    fired_events: Vec<FiredEvent>,
    /// Previous playback time for event detection.
    prev_time: f32,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp_matrices(a: &[Mat4], b: &[Mat4], t: f32, out: &mut [Mat4]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        let (s_a, r_a, t_a) = a[i].to_scale_rotation_translation();
        let (s_b, r_b, t_b) = b[i].to_scale_rotation_translation();
        let s = s_a.lerp(s_b, t);
        let r = r_a.slerp(r_b, t);
        let p = t_a.lerp(t_b, t);
        out[i] = Mat4::from_scale_rotation_translation(s, r, p);
    }
}

impl AnimGraphRuntime {
    /// Create a new runtime from a graph definition and an initial `AnimationPlayer`.
    pub fn new(mut def: AnimGraphDef, player: AnimationPlayer) -> Self {
        // Pre-sort transitions by priority so evaluate_transitions can just iterate.
        for state in &mut def.states {
            state.transitions.sort_by_key(|t| t.priority);
        }

        let joint_count = player.joint_count();
        let output = vec![Mat4::IDENTITY; joint_count];
        let scratch = vec![Mat4::IDENTITY; joint_count];
        let current_state = def.default_state;

        let mut runtime = Self {
            def,
            params: AnimParams::default(),
            current_state,
            player_a: player,
            blend_players: Vec::new(),
            crossfade: None,
            output,
            scratch,
            fired_events: Vec::new(),
            prev_time: 0.0,
        };

        // Start playing the default state.
        runtime.enter_state(current_state);
        runtime
    }

    /// Get the index of the currently active state.
    pub fn current_state(&self) -> usize {
        self.current_state
    }

    /// Get the name of the currently active state.
    pub fn current_state_name(&self) -> &str {
        &self.def.states[self.current_state].name
    }

    /// Advance the graph by `dt` seconds.
    pub fn advance(&mut self, dt: f32, clips: &[AnimationClip]) {
        // 1. Check transitions (already sorted by priority).
        if let Some((target, duration)) = self.evaluate_transitions() {
            self.start_transition(target, duration, clips);
        }

        // 2. Advance current state players.
        let state = &self.def.states[self.current_state];

        match &state.source {
            StateSource::Clip { .. } => {
                self.player_a.speed = state.speed;
                self.player_a.advance(dt, clips);
            }
            StateSource::BlendTree1D { .. } => {
                for p in &mut self.blend_players {
                    p.speed = state.speed;
                    p.advance(dt, clips);
                }
            }
        }

        // 2.5. Detect animation events.
        {
            let state = &self.def.states[self.current_state];
            let current_time = match &state.source {
                StateSource::Clip { .. } => self.player_a.time(),
                StateSource::BlendTree1D { .. } => {
                    // All blend players advance in lockstep; use first.
                    self.blend_players.first().map_or(0.0, |p| p.time())
                }
            };

            // Get clip duration for wrap detection.
            let clip_duration = match &state.source {
                StateSource::Clip { clip_index } => {
                    clips.get(*clip_index).map_or(0.0, |c| c.duration)
                }
                StateSource::BlendTree1D { entries, .. } => {
                    entries.first()
                        .and_then(|e| clips.get(e.clip_index))
                        .map_or(0.0, |c| c.duration)
                }
            };

            let prev = self.prev_time;
            let curr = current_time;

            for event in &state.events {
                let fired = if curr >= prev {
                    // Normal forward playback: event fires if in (prev, curr].
                    event.time > prev && event.time <= curr
                } else if state.looping && clip_duration > 0.0 {
                    // Wrapped: fires if in (prev, duration] or [0, curr].
                    event.time > prev || event.time <= curr
                } else {
                    false
                };

                if fired {
                    self.fired_events.push(FiredEvent {
                        name: event.name.clone(),
                        state_name: state.name.clone(),
                    });
                }
            }

            self.prev_time = curr;
        }

        // 3. Compute blended output for current state.
        self.compute_state_output();

        // 4. Apply crossfade if active.
        if let Some(ref mut cf) = self.crossfade {
            cf.elapsed += dt;
            let t = smoothstep(cf.elapsed / cf.duration);
            self.scratch.copy_from_slice(&self.output);
            lerp_matrices(&cf.frozen_matrices, &self.scratch, t, &mut self.output);

            if cf.elapsed >= cf.duration {
                self.crossfade = None;
            }
        }
    }

    /// Get the final blended skinning matrices.
    pub fn skinning_matrices(&self) -> &[Mat4] {
        &self.output
    }

    /// Drain all events fired since the last call.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, FiredEvent> {
        self.fired_events.drain(..)
    }

    fn evaluate_transitions(&self) -> Option<(usize, f32)> {
        let state = &self.def.states[self.current_state];
        for transition in &state.transitions {
            if transition.target_state == self.current_state {
                continue;
            }
            if transition.conditions.iter().all(|c| c.evaluate(&self.params)) {
                return Some((transition.target_state, transition.duration));
            }
        }
        None
    }

    fn start_transition(&mut self, target: usize, duration: f32, _clips: &[AnimationClip]) {
        // Snapshot current output as frozen matrices.
        self.compute_state_output();
        let frozen = self.output.clone();

        self.crossfade = Some(CrossfadeState {
            frozen_matrices: frozen,
            elapsed: 0.0,
            duration: duration.max(0.001),
        });

        self.enter_state(target);
    }

    fn enter_state(&mut self, state_idx: usize) {
        self.current_state = state_idx;
        let state = &self.def.states[state_idx];

        match &state.source {
            StateSource::Clip { clip_index } => {
                self.player_a = self.player_a.clone_skeleton();
                self.player_a.play(*clip_index, state.looping);
                self.player_a.speed = state.speed;
                self.blend_players.clear();
            }
            StateSource::BlendTree1D { entries, .. } => {
                // One player per blend entry — all advance in lockstep so
                // crossing a threshold never resets playback time.
                self.blend_players.clear();
                for entry in entries {
                    let mut p = self.player_a.clone_skeleton();
                    p.play(entry.clip_index, state.looping);
                    p.speed = state.speed;
                    self.blend_players.push(p);
                }
            }
        }

        self.prev_time = 0.0;
    }

    fn compute_state_output(&mut self) {
        let state = &self.def.states[self.current_state];

        match &state.source {
            StateSource::Clip { .. } => {
                self.output.copy_from_slice(self.player_a.skinning_matrices());
            }
            StateSource::BlendTree1D { param, entries } => {
                if entries.is_empty() || self.blend_players.is_empty() {
                    return;
                }
                if entries.len() == 1 {
                    self.output
                        .copy_from_slice(self.blend_players[0].skinning_matrices());
                    return;
                }

                let value = self.params.get_float(param);
                let (idx_lo, idx_hi, t) = blend_tree_weights(entries, value);

                if idx_lo == idx_hi || t < 1e-6 {
                    self.output
                        .copy_from_slice(self.blend_players[idx_lo].skinning_matrices());
                } else {
                    let matrices_lo = self.blend_players[idx_lo].skinning_matrices();
                    let matrices_hi = self.blend_players[idx_hi].skinning_matrices();
                    lerp_matrices(matrices_lo, matrices_hi, t, &mut self.output);
                }
            }
        }
    }
}

/// Find the two bracketing blend entries and compute the interpolation weight.
/// Returns (lower_index, upper_index, t) where t=0 means 100% lower, t=1 means 100% upper.
fn blend_tree_weights(entries: &[BlendEntry], value: f32) -> (usize, usize, f32) {
    debug_assert!(!entries.is_empty());
    if entries.len() == 1 {
        return (0, 0, 0.0);
    }

    // Below first threshold.
    if value <= entries[0].threshold {
        return (0, 0, 0.0);
    }

    // Above last threshold.
    let last = entries.len() - 1;
    if value >= entries[last].threshold {
        return (last, last, 0.0);
    }

    // Find bracketing pair.
    for i in 0..last {
        if value >= entries[i].threshold && value <= entries[i + 1].threshold {
            let range = entries[i + 1].threshold - entries[i].threshold;
            let t = if range.abs() > 1e-8 {
                (value - entries[i].threshold) / range
            } else {
                0.0
            };
            return (i, i + 1, t);
        }
    }

    (last, last, 0.0)
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use esox_gfx::mesh3d::gltf_loader::{AnimChannel, AnimProperty, AnimationClip, GltfSkin, Interpolation};

    fn make_skin(joint_count: usize) -> GltfSkin {
        use esox_gfx::mesh3d::transform::Transform;
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

    fn make_clip(joint: usize, x_start: f32, x_end: f32, duration: f32) -> AnimationClip {
        AnimationClip {
            name: None,
            duration,
            channels: vec![AnimChannel {
                joint_index: joint,
                property: AnimProperty::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, duration],
                values: vec![[x_start, 0.0, 0.0, 0.0], [x_end, 0.0, 0.0, 0.0]],
            }],
        }
    }

    // ── Condition tests ──

    #[test]
    fn condition_float_gt() {
        let mut params = AnimParams::default();
        params.set_float("speed", 5.0);

        let cond = Condition::FloatGt { param: "speed".into(), threshold: 3.0 };
        assert!(cond.evaluate(&params));

        let cond = Condition::FloatGt { param: "speed".into(), threshold: 5.0 };
        assert!(!cond.evaluate(&params));
    }

    #[test]
    fn condition_float_lt() {
        let mut params = AnimParams::default();
        params.set_float("speed", 2.0);

        let cond = Condition::FloatLt { param: "speed".into(), threshold: 3.0 };
        assert!(cond.evaluate(&params));

        let cond = Condition::FloatLt { param: "speed".into(), threshold: 1.0 };
        assert!(!cond.evaluate(&params));
    }

    #[test]
    fn condition_float_between() {
        let mut params = AnimParams::default();
        params.set_float("speed", 5.0);

        let cond = Condition::FloatBetween { param: "speed".into(), min: 3.0, max: 7.0 };
        assert!(cond.evaluate(&params));

        let cond = Condition::FloatBetween { param: "speed".into(), min: 6.0, max: 10.0 };
        assert!(!cond.evaluate(&params));
    }

    #[test]
    fn condition_bool() {
        let mut params = AnimParams::default();
        params.set_bool("grounded", true);

        assert!(Condition::BoolTrue { param: "grounded".into() }.evaluate(&params));
        assert!(!Condition::BoolFalse { param: "grounded".into() }.evaluate(&params));

        params.set_bool("grounded", false);
        assert!(!Condition::BoolTrue { param: "grounded".into() }.evaluate(&params));
        assert!(Condition::BoolFalse { param: "grounded".into() }.evaluate(&params));
    }

    #[test]
    fn missing_param_defaults() {
        let params = AnimParams::default();
        assert_eq!(params.get_float("missing"), 0.0);
        assert!(!params.get_bool("missing"));
    }

    // ── Blend tree weight tests ──

    #[test]
    fn blend_tree_endpoints() {
        let entries = vec![
            BlendEntry { clip_index: 0, threshold: 0.0 },
            BlendEntry { clip_index: 1, threshold: 1.0 },
        ];

        let (lo, hi, t) = blend_tree_weights(&entries, 0.0);
        assert_eq!((lo, hi), (0, 0));
        assert!((t - 0.0).abs() < 1e-5);

        let (lo, hi, t) = blend_tree_weights(&entries, 1.0);
        assert_eq!((lo, hi), (1, 1));
        assert!((t - 0.0).abs() < 1e-5);
    }

    #[test]
    fn blend_tree_midpoint() {
        let entries = vec![
            BlendEntry { clip_index: 0, threshold: 0.0 },
            BlendEntry { clip_index: 1, threshold: 1.0 },
        ];

        let (lo, hi, t) = blend_tree_weights(&entries, 0.5);
        assert_eq!((lo, hi), (0, 1));
        assert!((t - 0.5).abs() < 1e-5);
    }

    #[test]
    fn blend_tree_below_range() {
        let entries = vec![
            BlendEntry { clip_index: 0, threshold: 1.0 },
            BlendEntry { clip_index: 1, threshold: 5.0 },
        ];

        let (lo, _hi, t) = blend_tree_weights(&entries, 0.0);
        assert_eq!(lo, 0);
        assert!((t - 0.0).abs() < 1e-5);
    }

    #[test]
    fn blend_tree_above_range() {
        let entries = vec![
            BlendEntry { clip_index: 0, threshold: 0.0 },
            BlendEntry { clip_index: 1, threshold: 1.0 },
        ];

        let (lo, _hi, t) = blend_tree_weights(&entries, 5.0);
        assert_eq!(lo, 1);
        assert!((t - 0.0).abs() < 1e-5);
    }

    #[test]
    fn blend_tree_three_entries() {
        let entries = vec![
            BlendEntry { clip_index: 0, threshold: 0.0 },
            BlendEntry { clip_index: 1, threshold: 5.0 },
            BlendEntry { clip_index: 2, threshold: 10.0 },
        ];

        // Between first and second.
        let (lo, hi, t) = blend_tree_weights(&entries, 2.5);
        assert_eq!((lo, hi), (0, 1));
        assert!((t - 0.5).abs() < 1e-5);

        // Between second and third.
        let (lo, hi, t) = blend_tree_weights(&entries, 7.5);
        assert_eq!((lo, hi), (1, 2));
        assert!((t - 0.5).abs() < 1e-5);
    }

    // ── Smoothstep tests ──

    #[test]
    fn smoothstep_boundaries() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-5);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-5);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn smoothstep_clamps() {
        assert!((smoothstep(-1.0) - 0.0).abs() < 1e-5);
        assert!((smoothstep(2.0) - 1.0).abs() < 1e-5);
    }

    // ── Runtime tests ──

    #[test]
    fn runtime_default_state_plays() {
        let skin = make_skin(2);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![make_clip(0, 0.0, 4.0, 1.0)];

        let def = AnimGraphDef {
            states: vec![AnimState {
                name: "idle".into(),
                source: StateSource::Clip { clip_index: 0 },
                looping: true,
                speed: 1.0,
                transitions: vec![],
                events: vec![],
            }],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);
        runtime.advance(0.5, &clips);

        assert_eq!(runtime.current_state(), 0);
        assert_eq!(runtime.current_state_name(), "idle");
        // Joint 0 should have moved halfway (x=2.0).
        let pos = runtime.skinning_matrices()[0].col(3).x;
        assert!((pos - 2.0).abs() < 1e-4, "got x={pos}");
    }

    #[test]
    fn transition_fires_on_condition() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![
            make_clip(0, 0.0, 1.0, 1.0), // idle
            make_clip(0, 0.0, 5.0, 1.0), // run
        ];

        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "idle".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::FloatGt {
                            param: "speed".into(),
                            threshold: 0.5,
                        }],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "run".into(),
                    source: StateSource::Clip { clip_index: 1 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);
        assert_eq!(runtime.current_state(), 0);

        // Set speed above threshold — transition should fire.
        runtime.params.set_float("speed", 1.0);
        runtime.advance(0.01, &clips);
        assert_eq!(runtime.current_state(), 1);
    }

    #[test]
    fn transition_priority_ordering() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![
            make_clip(0, 0.0, 1.0, 1.0),
            make_clip(0, 0.0, 2.0, 1.0),
            make_clip(0, 0.0, 3.0, 1.0),
        ];

        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "idle".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![
                        Transition {
                            target_state: 2,
                            conditions: vec![Condition::BoolTrue { param: "go".into() }],
                            duration: 0.1,
                            priority: 10, // lower priority
                        },
                        Transition {
                            target_state: 1,
                            conditions: vec![Condition::BoolTrue { param: "go".into() }],
                            duration: 0.1,
                            priority: 0, // higher priority (lower number)
                        },
                    ],
                    events: vec![],
                },
                AnimState { name: "a".into(), source: StateSource::Clip { clip_index: 1 }, looping: true, speed: 1.0, transitions: vec![], events: vec![] },
                AnimState { name: "b".into(), source: StateSource::Clip { clip_index: 2 }, looping: true, speed: 1.0, transitions: vec![], events: vec![] },
            ],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);
        runtime.params.set_bool("go", true);
        runtime.advance(0.01, &clips);
        // Should pick state 1 (priority 0) over state 2 (priority 10).
        assert_eq!(runtime.current_state(), 1);
    }

    #[test]
    fn crossfade_interpolation() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        // Clip 0: joint at x=0 -> x=0 (stationary at origin).
        // Clip 1: joint at x=10 -> x=10 (stationary at 10).
        let clips = vec![
            AnimationClip {
                name: None,
                duration: 1.0,
                channels: vec![AnimChannel {
                    joint_index: 0,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[0.0, 0.0, 0.0, 0.0]],
                }],
            },
            AnimationClip {
                name: None,
                duration: 1.0,
                channels: vec![AnimChannel {
                    joint_index: 0,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[10.0, 0.0, 0.0, 0.0]],
                }],
            },
        ];

        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "a".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::BoolTrue { param: "go".into() }],
                        duration: 1.0, // 1s crossfade for easy math
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "b".into(),
                    source: StateSource::Clip { clip_index: 1 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);
        // Pump one frame to establish state A output.
        runtime.advance(0.01, &clips);
        let x = runtime.skinning_matrices()[0].col(3).x;
        assert!(x.abs() < 1e-3, "should be near 0, got {x}");

        // Trigger transition.
        runtime.params.set_bool("go", true);
        runtime.advance(0.5, &clips); // halfway through 1s crossfade

        let x = runtime.skinning_matrices()[0].col(3).x;
        // smoothstep(0.5) = 0.5, so lerp(0, 10, 0.5) = 5.0
        assert!((x - 5.0).abs() < 0.5, "expected ~5.0, got {x}");
    }

    #[test]
    fn crossfade_completes() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![
            AnimationClip {
                name: None,
                duration: 1.0,
                channels: vec![AnimChannel {
                    joint_index: 0,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[0.0, 0.0, 0.0, 0.0]],
                }],
            },
            AnimationClip {
                name: None,
                duration: 1.0,
                channels: vec![AnimChannel {
                    joint_index: 0,
                    property: AnimProperty::Translation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![[10.0, 0.0, 0.0, 0.0]],
                }],
            },
        ];

        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "a".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::BoolTrue { param: "go".into() }],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "b".into(),
                    source: StateSource::Clip { clip_index: 1 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);
        runtime.advance(0.01, &clips);
        runtime.params.set_bool("go", true);

        // Advance well past the crossfade duration.
        runtime.advance(0.5, &clips);

        let x = runtime.skinning_matrices()[0].col(3).x;
        assert!((x - 10.0).abs() < 0.5, "expected ~10.0 after crossfade, got {x}");
    }

    // ── Serde round-trip test ──

    #[cfg(feature = "serialization")]
    #[test]
    fn serde_round_trip() {
        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "idle".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![
                            Condition::FloatGt { param: "speed".into(), threshold: 0.5 },
                            Condition::BoolTrue { param: "grounded".into() },
                        ],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "locomotion".into(),
                    source: StateSource::BlendTree1D {
                        param: "speed".into(),
                        entries: vec![
                            BlendEntry { clip_index: 1, threshold: 0.0 },
                            BlendEntry { clip_index: 2, threshold: 1.0 },
                        ],
                    },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let json = serde_json::to_string(&def).unwrap();
        let roundtrip: AnimGraphDef = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.states.len(), 2);
        assert_eq!(roundtrip.default_state, 0);
        assert_eq!(roundtrip.states[0].name, "idle");
        assert_eq!(roundtrip.states[1].name, "locomotion");
    }

    // ── Animation event tests ──

    #[test]
    fn event_fires_at_correct_time() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![make_clip(0, 0.0, 1.0, 1.0)];

        let def = AnimGraphDef {
            states: vec![AnimState {
                name: "walk".into(),
                source: StateSource::Clip { clip_index: 0 },
                looping: true,
                speed: 1.0,
                transitions: vec![],
                events: vec![AnimEvent { name: "footstep".into(), time: 0.5 }],
            }],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);

        // Advance to 0.3 — event at 0.5 should not fire.
        runtime.advance(0.3, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        assert!(events.is_empty(), "should not fire before event time");

        // Advance to 0.6 — event at 0.5 should fire.
        runtime.advance(0.3, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "footstep");
        assert_eq!(events[0].state_name, "walk");
    }

    #[test]
    fn event_wraps_on_loop() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        // Clip with 1.0s duration.
        let clips = vec![make_clip(0, 0.0, 1.0, 1.0)];

        let def = AnimGraphDef {
            states: vec![AnimState {
                name: "walk".into(),
                source: StateSource::Clip { clip_index: 0 },
                looping: true,
                speed: 1.0,
                transitions: vec![],
                events: vec![AnimEvent { name: "footstep".into(), time: 0.1 }],
            }],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);

        // Advance to 0.9.
        runtime.advance(0.9, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        assert_eq!(events.len(), 1, "should fire at 0.1 during 0..0.9");

        // Advance by 0.3 — should wrap (0.9 -> 1.0 -> 0.2) and fire event at 0.1.
        runtime.advance(0.3, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        assert_eq!(events.len(), 1, "should fire on loop wrap");
    }

    #[test]
    fn no_double_fire_during_crossfade() {
        let skin = make_skin(1);
        let player = AnimationPlayer::new(&skin);
        let clips = vec![
            make_clip(0, 0.0, 1.0, 1.0),
            make_clip(0, 0.0, 2.0, 1.0),
        ];

        let def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "a".into(),
                    source: StateSource::Clip { clip_index: 0 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::BoolTrue { param: "go".into() }],
                        duration: 0.5,
                        priority: 0,
                    }],
                    events: vec![AnimEvent { name: "event_a".into(), time: 0.2 }],
                },
                AnimState {
                    name: "b".into(),
                    source: StateSource::Clip { clip_index: 1 },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![AnimEvent { name: "event_b".into(), time: 0.3 }],
                },
            ],
            default_state: 0,
        };

        let mut runtime = AnimGraphRuntime::new(def, player);

        // Advance past event_a.
        runtime.advance(0.3, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "event_a");

        // Trigger transition to state b.
        runtime.params.set_bool("go", true);
        runtime.advance(0.4, &clips);
        let events: Vec<_> = runtime.drain_events().collect();
        // Should fire event_b at 0.3 in the new state, not event_a again.
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "event_b");
    }
}
