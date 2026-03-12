//! Widget response — returned from every widget call.

/// The result of drawing a widget. Check fields inline.
#[derive(Debug, Clone, Copy, Default)]
pub struct Response {
    /// The widget was clicked this frame.
    pub clicked: bool,
    /// The widget was right-clicked this frame.
    pub right_clicked: bool,
    /// The mouse is hovering over the widget.
    pub hovered: bool,
    /// The widget currently has keyboard focus.
    pub focused: bool,
    /// The widget's value changed this frame.
    pub changed: bool,
    /// The widget is disabled (no interaction).
    pub disabled: bool,
}
