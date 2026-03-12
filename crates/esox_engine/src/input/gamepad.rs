//! Gamepad support via gilrs (behind `gamepad` feature).

#[cfg(feature = "gamepad")]
use gilrs::{Axis, Button, Event, Gilrs};

/// Gamepad state manager.
#[cfg(feature = "gamepad")]
pub struct GamepadManager {
    gilrs: Gilrs,
    /// Deadzone for analog sticks.
    pub deadzone: f32,
}

#[cfg(feature = "gamepad")]
impl GamepadManager {
    pub fn new() -> Option<Self> {
        match Gilrs::new() {
            Ok(gilrs) => Some(Self {
                gilrs,
                deadzone: 0.15,
            }),
            Err(e) => {
                tracing::warn!("gamepad init failed: {e}");
                None
            }
        }
    }

    /// Poll gamepad events. Call once per frame.
    pub fn poll(&mut self) {
        while let Some(Event { .. }) = self.gilrs.next_event() {
            // Events consumed — state is queried from the active gamepad.
        }
    }

    /// Read an axis value from the first connected gamepad.
    pub fn axis(&self, axis: Axis) -> f32 {
        if let Some((_id, gamepad)) = self.gilrs.gamepads().next() {
            let val = gamepad
                .axis_data(axis)
                .map(|d| d.value())
                .unwrap_or(0.0);
            if val.abs() < self.deadzone {
                0.0
            } else {
                val
            }
        } else {
            0.0
        }
    }

    /// Check if a button is pressed on the first connected gamepad.
    pub fn is_pressed(&self, button: Button) -> bool {
        if let Some((_id, gamepad)) = self.gilrs.gamepads().next() {
            gamepad.is_pressed(button)
        } else {
            false
        }
    }
}
