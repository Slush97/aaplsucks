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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_default_all_false() {
        let r = Response::default();
        assert!(!r.clicked);
        assert!(!r.right_clicked);
        assert!(!r.hovered);
        assert!(!r.focused);
        assert!(!r.changed);
        assert!(!r.disabled);
    }
}
