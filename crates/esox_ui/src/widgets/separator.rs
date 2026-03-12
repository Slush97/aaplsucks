//! Separator widget — horizontal line.

use esox_gfx::ShapeBuilder;

use crate::state::{A11yNode, A11yRole};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a horizontal separator line.
    pub fn separator(&mut self) {
        let rect = self.allocate_rect(self.region.w, 1.0);
        self.push_a11y_node(A11yNode {
            id: 0, role: A11yRole::Separator, label: String::new(),
            value: None, rect, focused: false, disabled: false,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });
        self.frame.push(
            ShapeBuilder::rect(rect.x, rect.y, rect.w, 1.0)
                .color(self.theme.border)
                .build(),
        );
    }
}
