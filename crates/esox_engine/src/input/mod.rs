//! Input system — action mapping, axes, keyboard, and mouse state.

pub mod action;
pub mod axis;
#[cfg(feature = "gamepad")]
pub mod gamepad;

use std::collections::HashMap;

use esox_input::KeyCode;

pub use action::ActionBinding;
pub use axis::{AxisBinding, MouseAxis};

use action::ActionState;

/// Manages input state and action/axis bindings.
pub struct InputManager {
    /// Named actions and their bindings.
    actions: HashMap<String, Vec<ActionBinding>>,
    /// Current state per action name.
    action_states: HashMap<String, ActionState>,

    /// Named axes and their bindings.
    axes: HashMap<String, Vec<AxisBinding>>,

    /// Currently pressed keyboard keys.
    keys_down: HashMap<KeyCode, bool>,
    /// Keys that were pressed this frame (for JustPressed detection).
    keys_just_pressed: HashMap<KeyCode, bool>,
    /// Keys that were released this frame (for JustReleased detection).
    keys_just_released: HashMap<KeyCode, bool>,

    /// Currently pressed mouse buttons.
    mouse_buttons_down: [bool; 3],
    /// Mouse buttons just pressed this frame.
    mouse_buttons_just_pressed: [bool; 3],
    /// Mouse buttons just released this frame.
    mouse_buttons_just_released: [bool; 3],

    /// Current mouse position in pixels.
    pub(crate) mouse_pos: (f64, f64),
    /// Mouse delta for the current tick (distributed from frame delta).
    pub(crate) mouse_delta: (f64, f64),
    /// Accumulated mouse delta from OS events (reset each frame).
    mouse_delta_accum: (f64, f64),
    /// Mouse delta snapshot for the current frame.
    mouse_delta_frame: (f64, f64),
    /// Number of fixed ticks in the current frame (for delta distribution).
    frame_tick_count: u32,

    /// Scroll delta accumulated this frame.
    scroll_accum: f32,
    /// Scroll delta for the current tick (distributed from frame total).
    pub(crate) scroll_delta: f32,
    /// Snapshot for the current frame.
    scroll_frame: f32,

    /// Whether the cursor should be grabbed (confined + hidden) for mouse-look.
    cursor_grab: bool,
}

impl InputManager {
    pub(crate) fn new() -> Self {
        Self {
            actions: HashMap::new(),
            action_states: HashMap::new(),
            axes: HashMap::new(),
            keys_down: HashMap::new(),
            keys_just_pressed: HashMap::new(),
            keys_just_released: HashMap::new(),
            mouse_buttons_down: [false; 3],
            mouse_buttons_just_pressed: [false; 3],
            mouse_buttons_just_released: [false; 3],
            mouse_pos: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
            mouse_delta_accum: (0.0, 0.0),
            mouse_delta_frame: (0.0, 0.0),
            frame_tick_count: 1,
            scroll_accum: 0.0,
            scroll_delta: 0.0,
            scroll_frame: 0.0,
            cursor_grab: false,
        }
    }

    // ── Public API ──

    /// Bind a named action to an input.
    pub fn bind_action(&mut self, name: &str, binding: ActionBinding) {
        self.actions
            .entry(name.to_string())
            .or_default()
            .push(binding);
        self.action_states
            .entry(name.to_string())
            .or_insert(ActionState::Idle);
    }

    /// Bind a named axis to an input source.
    pub fn bind_axis(&mut self, name: &str, binding: AxisBinding) {
        self.axes
            .entry(name.to_string())
            .or_default()
            .push(binding);
    }

    /// Whether the action was just pressed this tick.
    pub fn just_pressed(&self, name: &str) -> bool {
        self.action_states
            .get(name)
            .is_some_and(|s| s.is_just_pressed())
    }

    /// Whether the action is currently held (pressed or just pressed).
    pub fn held(&self, name: &str) -> bool {
        self.action_states
            .get(name)
            .is_some_and(|s| s.is_held())
    }

    /// Whether the action was just released this tick.
    pub fn just_released(&self, name: &str) -> bool {
        self.action_states
            .get(name)
            .is_some_and(|s| s.is_just_released())
    }

    /// Read a named axis value.
    ///
    /// Key-based axes return values in [-1, +1]. Mouse-delta axes return raw
    /// pixel deltas (unclamped) so they can be used for smooth camera control.
    pub fn axis(&self, name: &str) -> f32 {
        let bindings = match self.axes.get(name) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut value = 0.0f32;
        let mut is_digital = true;
        for binding in bindings {
            match binding {
                AxisBinding::Keys { negative, positive } => {
                    let neg = self.is_key_down(*negative);
                    let pos = self.is_key_down(*positive);
                    if neg {
                        value -= 1.0;
                    }
                    if pos {
                        value += 1.0;
                    }
                }
                AxisBinding::MouseDelta(axis) => {
                    is_digital = false;
                    value += match axis {
                        MouseAxis::X => self.mouse_delta.0 as f32,
                        MouseAxis::Y => self.mouse_delta.1 as f32,
                    };
                }
            }
        }
        if is_digital {
            value.clamp(-1.0, 1.0)
        } else {
            value
        }
    }

    /// Current mouse position in pixels.
    pub fn mouse_pos(&self) -> (f64, f64) {
        self.mouse_pos
    }

    /// Mouse delta since last tick.
    pub fn mouse_delta(&self) -> (f64, f64) {
        self.mouse_delta
    }

    /// Whether a specific key is currently held down.
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.keys_down.get(&key).copied().unwrap_or(false)
    }

    /// Whether a specific key was just pressed this tick.
    pub fn just_pressed_key(&self, key: KeyCode) -> bool {
        self.keys_just_pressed.get(&key).copied().unwrap_or(false)
    }

    /// Scroll wheel delta for this tick (positive = scroll up).
    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    /// Set whether the cursor should be grabbed (confined + hidden) for mouse-look.
    pub fn set_cursor_grab(&mut self, grab: bool) {
        self.cursor_grab = grab;
    }

    /// Whether the cursor is currently grabbed.
    pub fn cursor_grabbed(&self) -> bool {
        self.cursor_grab
    }

    /// Whether a mouse button is currently held (0=left, 1=middle, 2=right).
    pub fn is_mouse_button_down(&self, button: u8) -> bool {
        self.mouse_buttons_down
            .get(button as usize)
            .copied()
            .unwrap_or(false)
    }

    // ── Internal event handlers (called by Engine) ──

    pub(crate) fn handle_key_event(&mut self, event: &esox_input::KeyEvent) {
        let code = event.physical_key;
        if event.pressed {
            // Only mark just_pressed on initial press, not repeats.
            if !self.keys_down.get(&code).copied().unwrap_or(false) {
                self.keys_just_pressed.insert(code, true);
            }
            self.keys_down.insert(code, true);
        } else {
            self.keys_down.insert(code, false);
            self.keys_just_released.insert(code, true);
        }
    }

    pub(crate) fn handle_mouse_move(&mut self, x: f64, y: f64) {
        let dx = x - self.mouse_pos.0;
        let dy = y - self.mouse_pos.1;
        self.mouse_pos = (x, y);
        self.mouse_delta_accum.0 += dx;
        self.mouse_delta_accum.1 += dy;
    }

    /// Accumulate raw mouse motion deltas (from `DeviceEvent::MouseMotion`).
    ///
    /// Used when the cursor is grabbed — `CursorMoved` window events stop
    /// firing, but raw device motion is still delivered.
    pub(crate) fn handle_raw_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_delta_accum.0 += dx;
        self.mouse_delta_accum.1 += dy;
    }

    pub(crate) fn handle_scroll(&mut self, delta_y: f32) {
        self.scroll_accum += delta_y;
    }

    pub(crate) fn handle_mouse_button(&mut self, button: u8, pressed: bool) {
        if let Some(slot) = self.mouse_buttons_down.get_mut(button as usize) {
            if pressed && !*slot {
                if let Some(jp) = self.mouse_buttons_just_pressed.get_mut(button as usize) {
                    *jp = true;
                }
            } else if !pressed && *slot {
                if let Some(jr) = self.mouse_buttons_just_released.get_mut(button as usize) {
                    *jr = true;
                }
            }
            *slot = pressed;
        }
    }

    /// Clear all key and mouse button state. Called on focus loss to prevent
    /// stuck keys when the compositor swallows release events.
    pub(crate) fn clear_all_state(&mut self) {
        self.keys_down.clear();
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
        self.mouse_buttons_down = [false; 3];
        self.mouse_buttons_just_pressed = [false; 3];
        self.mouse_buttons_just_released = [false; 3];
        self.action_states.values_mut().for_each(|s| *s = ActionState::Idle);
    }

    /// Called once per frame before the fixed tick loop.
    /// Snapshots mouse delta and stores tick count for even distribution.
    pub(crate) fn begin_frame(&mut self, tick_count: u32) {
        self.frame_tick_count = tick_count;
        // Snapshot mouse delta for this frame; keep accumulating if no ticks run.
        if tick_count > 0 {
            self.mouse_delta_frame = self.mouse_delta_accum;
            self.mouse_delta_accum = (0.0, 0.0);
            self.scroll_frame = self.scroll_accum;
            self.scroll_accum = 0.0;
        }
    }

    /// Called at the start of each fixed tick: distribute mouse delta, update action states.
    pub(crate) fn pre_update(&mut self) {
        // Distribute mouse delta evenly across ticks in this frame.
        let tc = self.frame_tick_count.max(1) as f64;
        self.mouse_delta = (
            self.mouse_delta_frame.0 / tc,
            self.mouse_delta_frame.1 / tc,
        );
        self.scroll_delta = self.scroll_frame / self.frame_tick_count.max(1) as f32;

        // Update action states from raw input.
        for (name, bindings) in &self.actions {
            let any_pressed = bindings.iter().any(|b| match b {
                ActionBinding::Key(k) => {
                    self.keys_just_pressed.get(k).copied().unwrap_or(false)
                }
                ActionBinding::MouseButton(b) => {
                    self.mouse_buttons_just_pressed
                        .get(*b as usize)
                        .copied()
                        .unwrap_or(false)
                }
            });
            let any_released = bindings.iter().any(|b| match b {
                ActionBinding::Key(k) => {
                    self.keys_just_released.get(k).copied().unwrap_or(false)
                }
                ActionBinding::MouseButton(b) => {
                    self.mouse_buttons_just_released
                        .get(*b as usize)
                        .copied()
                        .unwrap_or(false)
                }
            });
            let any_held = bindings.iter().any(|b| match b {
                ActionBinding::Key(k) => self.keys_down.get(k).copied().unwrap_or(false),
                ActionBinding::MouseButton(b) => {
                    self.mouse_buttons_down
                        .get(*b as usize)
                        .copied()
                        .unwrap_or(false)
                }
            });

            let state = self.action_states.entry(name.clone()).or_insert(ActionState::Idle);
            if any_pressed {
                *state = ActionState::JustPressed;
            } else if any_released {
                *state = ActionState::JustReleased;
            } else if any_held {
                *state = ActionState::Held;
            } else {
                *state = ActionState::Idle;
            }
        }
    }

    /// Called at the end of each fixed tick.
    /// Transient flags persist for all ticks in the frame (cleared in end_frame).
    pub(crate) fn post_update(&mut self) {
        // No-op: flags persist until end_frame() so that multi-tick frames
        // don't silently eat input events.
    }

    /// Called once per frame after the fixed tick loop: clear transient state.
    /// Only clears if at least one tick ran, so events aren't silently dropped
    /// during high-FPS frames where no fixed tick executes.
    pub(crate) fn end_frame(&mut self) {
        if self.frame_tick_count > 0 {
            self.keys_just_pressed.clear();
            self.keys_just_released.clear();
            self.mouse_buttons_just_pressed = [false; 3];
            self.mouse_buttons_just_released = [false; 3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_event(code: KeyCode, pressed: bool) -> esox_input::KeyEvent {
        esox_input::KeyEvent {
            key: esox_input::Key::Unidentified,
            physical_key: code,
            pressed,
            repeat: false,
            text: None,
        }
    }

    #[test]
    fn action_just_pressed() {
        let mut input = InputManager::new();
        input.bind_action("jump", ActionBinding::Key(KeyCode::Space));

        input.handle_key_event(&make_key_event(KeyCode::Space, true));
        input.pre_update();

        assert!(input.just_pressed("jump"));
        assert!(input.held("jump"));
        assert!(!input.just_released("jump"));
    }

    #[test]
    fn action_held_after_frame() {
        let mut input = InputManager::new();
        input.bind_action("jump", ActionBinding::Key(KeyCode::Space));

        input.handle_key_event(&make_key_event(KeyCode::Space, true));
        input.begin_frame(1);
        input.pre_update();
        input.post_update();
        input.end_frame();
        // Next frame, key is still down but not "just pressed".
        input.begin_frame(1);
        input.pre_update();

        assert!(!input.just_pressed("jump"));
        assert!(input.held("jump"));
    }

    #[test]
    fn just_pressed_survives_multi_tick_frame() {
        let mut input = InputManager::new();
        input.bind_action("jump", ActionBinding::Key(KeyCode::Space));

        input.handle_key_event(&make_key_event(KeyCode::Space, true));
        input.begin_frame(3);

        // Tick 0
        input.pre_update();
        assert!(input.just_pressed("jump"), "tick 0 should see just_pressed");
        input.post_update();

        // Tick 1 — event must survive within same frame
        input.pre_update();
        assert!(input.just_pressed("jump"), "tick 1 should still see just_pressed");
        input.post_update();

        input.end_frame();

        // Next frame — no longer just pressed
        input.begin_frame(1);
        input.pre_update();
        assert!(!input.just_pressed("jump"));
        assert!(input.held("jump"));
    }

    #[test]
    fn mouse_delta_distributed_across_ticks() {
        let mut input = InputManager::new();
        // Set initial position
        input.handle_mouse_move(100.0, 100.0);
        input.begin_frame(1);
        input.pre_update();
        input.post_update();
        input.end_frame();

        // Move mouse 60px in X
        input.handle_mouse_move(160.0, 100.0);
        input.begin_frame(3); // 3 ticks this frame

        input.pre_update();
        assert!(
            (input.mouse_delta().0 - 20.0).abs() < 1e-6,
            "each tick gets 1/3 of delta"
        );
        input.post_update();

        input.pre_update();
        assert!(
            (input.mouse_delta().0 - 20.0).abs() < 1e-6,
            "each tick gets 1/3 of delta"
        );
        input.post_update();

        input.pre_update();
        assert!(
            (input.mouse_delta().0 - 20.0).abs() < 1e-6,
            "each tick gets 1/3 of delta"
        );
        input.post_update();

        input.end_frame();
    }

    #[test]
    fn axis_keys() {
        let mut input = InputManager::new();
        input.bind_axis(
            "move_x",
            AxisBinding::Keys {
                negative: KeyCode::KeyA,
                positive: KeyCode::KeyD,
            },
        );

        input.handle_key_event(&make_key_event(KeyCode::KeyD, true));
        input.pre_update();
        assert!((input.axis("move_x") - 1.0).abs() < 1e-6);

        input.post_update();
        input.handle_key_event(&make_key_event(KeyCode::KeyA, true));
        input.pre_update();
        // Both keys: cancel out to 0.
        assert!(input.axis("move_x").abs() < 1e-6);
    }
}
