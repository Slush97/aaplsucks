//! `esox_input` — Platform-independent input types.
//!
//! Pure data types with no windowing dependency. Used by `esox_ui`,
//! `esox_engine`, and `esox_platform` to decouple input handling from
//! the windowing backend.

pub use smol_str::SmolStr;

/// A logical key value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    /// A named key (Enter, Tab, arrows, etc.).
    Named(NamedKey),
    /// A character produced by the key.
    Character(SmolStr),
    /// An unidentified key.
    Unidentified,
}

/// Named (non-character) keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedKey {
    Enter,
    Tab,
    Space,
    Backspace,
    Delete,
    Escape,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Home,
    End,
    PageUp,
    PageDown,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
}

/// Physical key codes (scan codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Home,
    End,
    PageUp,
    PageDown,
    Space,
    Enter,
    Tab,
    Backspace,
    Delete,
    Escape,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Backquote,
    Comma,
    Period,
    Slash,
}

/// Modifier key state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers(u8);

impl Modifiers {
    pub const EMPTY: Modifiers = Modifiers(0);

    const SHIFT: u8 = 1;
    const CTRL: u8 = 2;
    const ALT: u8 = 4;
    const SUPER: u8 = 8;

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn shift(self) -> bool {
        self.0 & Self::SHIFT != 0
    }

    pub fn ctrl(self) -> bool {
        self.0 & Self::CTRL != 0
    }

    pub fn alt(self) -> bool {
        self.0 & Self::ALT != 0
    }

    pub fn super_key(self) -> bool {
        self.0 & Self::SUPER != 0
    }

    pub fn with_shift(mut self) -> Self {
        self.0 |= Self::SHIFT;
        self
    }

    pub fn with_ctrl(mut self) -> Self {
        self.0 |= Self::CTRL;
        self
    }

    pub fn with_alt(mut self) -> Self {
        self.0 |= Self::ALT;
        self
    }

    pub fn with_super(mut self) -> Self {
        self.0 |= Self::SUPER;
        self
    }

    /// Build from individual flags.
    pub fn from_flags(shift: bool, ctrl: bool, alt: bool, super_key: bool) -> Self {
        let mut bits = 0u8;
        if shift {
            bits |= Self::SHIFT;
        }
        if ctrl {
            bits |= Self::CTRL;
        }
        if alt {
            bits |= Self::ALT;
        }
        if super_key {
            bits |= Self::SUPER;
        }
        Self(bits)
    }
}

/// A keyboard event.
#[derive(Debug, Clone)]
pub struct KeyEvent {
    /// The logical key value.
    pub key: Key,
    /// The physical key code.
    pub physical_key: KeyCode,
    /// Whether the key is pressed (true) or released (false).
    pub pressed: bool,
    /// Whether this is a key repeat.
    pub repeat: bool,
    /// Text produced by this key event (after modifier processing).
    pub text: Option<SmolStr>,
}

/// Mouse cursor icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorIcon {
    Default,
    Text,
    Pointer,
    Grab,
    Grabbing,
    ColResize,
    RowResize,
    NotAllowed,
}
