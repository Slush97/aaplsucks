//! Axis bindings — named analog axes from digital or analog input.

use winit::keyboard::KeyCode;

/// A binding that produces an axis value in [-1, +1].
#[derive(Debug, Clone)]
pub enum AxisBinding {
    /// Two keys mapping to -1 and +1.
    Keys {
        negative: KeyCode,
        positive: KeyCode,
    },
    /// Mouse delta on X or Y axis.
    MouseDelta(MouseAxis),
}

/// Which mouse axis to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseAxis {
    X,
    Y,
}
