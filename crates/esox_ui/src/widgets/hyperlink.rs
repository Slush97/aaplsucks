//! Hyperlink widget — styled text link that returns click response.

use esox_gfx::ShapeBuilder;

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a hyperlink. Does NOT open URLs — returns `Response { clicked: true }`
    /// so the caller can decide what to do.
    pub fn hyperlink(&mut self, id: u64, label: &str, _url: &str) -> Response {
        let font_size = self.theme.font_size;
        let text_w = self.text.measure_text(label, font_size);
        let h = self.theme.item_height;

        let rect = self.allocate_rect_keyed(id, text_w, h);
        self.register_widget(id, rect, WidgetKind::Hyperlink);
        let response = self.widget_response(id, rect);

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::Link,
            label: label.to_string(),
            value: None,
            rect,
            focused: response.focused,
            disabled: response.disabled,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: Vec::new(),
        });

        // Focus ring.
        if response.focused && !response.disabled {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                2.0,
                self.theme.focus_ring_expand,
            );
        }

        // Color: accent, lerp to accent_hover on hover.
        let hover_t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
        let color = paint::lerp_color(self.theme.accent, self.theme.accent_hover, hover_t);

        // Draw text.
        let text_y = rect.y + (h - font_size) / 2.0;
        self.text.draw_ui_text(
            label,
            rect.x,
            text_y,
            color,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Underline on hover.
        if response.hovered || response.focused {
            let underline_y = text_y + font_size + 1.0;
            self.frame.push(
                ShapeBuilder::rect(rect.x, underline_y, text_w, 1.0)
                    .color(color)
                    .build(),
            );
        }

        response
    }
}
