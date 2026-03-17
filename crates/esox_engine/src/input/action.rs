//! Action bindings — named boolean actions triggered by input.

use esox_input::KeyCode;

/// A binding that triggers an action.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionBinding {
    /// Keyboard key.
    Key(KeyCode),
    /// Mouse button (0=left, 1=middle, 2=right).
    MouseButton(u8),
}

/// Current state of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActionState {
    /// Not pressed.
    Idle,
    /// Just pressed this tick.
    JustPressed,
    /// Held down (was JustPressed last tick).
    Held,
    /// Just released this tick.
    JustReleased,
}

impl ActionState {
    pub fn is_just_pressed(self) -> bool {
        self == Self::JustPressed
    }

    pub fn is_held(self) -> bool {
        matches!(self, Self::JustPressed | Self::Held)
    }

    pub fn is_just_released(self) -> bool {
        self == Self::JustReleased
    }
}
