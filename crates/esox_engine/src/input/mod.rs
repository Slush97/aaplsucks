//! Input system — action mapping, axes, keyboard, and mouse state.

pub mod action;
pub mod axis;
#[cfg(feature = "gamepad")]
pub mod gamepad;

use std::collections::HashMap;

use winit::keyboard::KeyCode;

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
    /// Mouse delta since last frame.
    pub(crate) mouse_delta: (f64, f64),
    /// Accumulated mouse delta (reset each tick).
    mouse_delta_accum: (f64, f64),
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

    /// Whether a mouse button is currently held (0=left, 1=middle, 2=right).
    pub fn is_mouse_button_down(&self, button: u8) -> bool {
        self.mouse_buttons_down
            .get(button as usize)
            .copied()
            .unwrap_or(false)
    }

    // ── Internal event handlers (called by Engine) ──

    pub(crate) fn handle_key_event(&mut self, event: &winit::event::KeyEvent) {
        use winit::keyboard::PhysicalKey;
        if let PhysicalKey::Code(code) = event.physical_key {
            if event.state.is_pressed() {
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
    }

    pub(crate) fn handle_mouse_move(&mut self, x: f64, y: f64) {
        let dx = x - self.mouse_pos.0;
        let dy = y - self.mouse_pos.1;
        self.mouse_pos = (x, y);
        self.mouse_delta_accum.0 += dx;
        self.mouse_delta_accum.1 += dy;
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

    /// Called at the start of each fixed tick: transition JustPressed -> Held.
    pub(crate) fn pre_update(&mut self) {
        // Snapshot mouse delta for this tick.
        self.mouse_delta = self.mouse_delta_accum;
        self.mouse_delta_accum = (0.0, 0.0);

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

    /// Called at the end of each fixed tick: clear transient state.
    pub(crate) fn post_update(&mut self) {
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
        self.mouse_buttons_just_pressed = [false; 3];
        self.mouse_buttons_just_released = [false; 3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_event(code: KeyCode, pressed: bool) -> winit::event::KeyEvent {
        // Safety: KeyEvent requires platform-specific data that is not Default.
        // We construct via unsafe zeroed memory — only safe for testing input logic,
        // not for passing to winit internals.
        let mut event: winit::event::KeyEvent = unsafe { std::mem::zeroed() };
        event.physical_key = winit::keyboard::PhysicalKey::Code(code);
        event.logical_key =
            winit::keyboard::Key::Unidentified(winit::keyboard::NativeKey::Unidentified);
        event.state = if pressed {
            winit::event::ElementState::Pressed
        } else {
            winit::event::ElementState::Released
        };
        event.repeat = false;
        event
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
    fn action_held_after_update() {
        let mut input = InputManager::new();
        input.bind_action("jump", ActionBinding::Key(KeyCode::Space));

        input.handle_key_event(&make_key_event(KeyCode::Space, true));
        input.pre_update();
        input.post_update();
        // Next tick, key is still down but not "just pressed".
        input.pre_update();

        assert!(!input.just_pressed("jump"));
        assert!(input.held("jump"));
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
