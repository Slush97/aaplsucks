//! Chip widget — removable tag pill.
//!
//! # Examples
//!
//! ```ignore
//! if ui.chip(id!("tag"), "Rust").clicked {
//!     // The × was clicked — remove this chip
//!     tags.remove("Rust");
//! }
//!
//! // With a custom background color
//! ui.chip_colored(id!("priority"), "High", theme.red);
//! ```

use esox_gfx::Color;

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a removable chip (tag pill). Returns a Response where `clicked` means
    /// the × remove button was pressed.
    pub fn chip(&mut self, id: u64, label: &str) -> Response {
        self.chip_colored(id, label, self.theme.accent_dim)
    }

    /// Draw a removable chip with a custom background color.
    pub fn chip_colored(&mut self, id: u64, label: &str, color: Color) -> Response {
        let font_size = self.theme.font_size;
        let pad_x = self.theme.input_padding;
        let pad_y = self.theme.chip_pad_y;

        // Measure: label + gap + "×"
        let label_w = self.text.measure_text(label, font_size);
        let close_str = "\u{00d7}"; // ×
        let close_w = self.text.measure_text(close_str, font_size);
        let gap = pad_x * 0.5;
        let chip_w = pad_x + label_w + gap + close_w + pad_x;
        let chip_h = font_size + pad_y * 2.0;
        let radius = chip_h / 2.0; // capsule

        let rect = self.allocate_rect_keyed(id, chip_w, chip_h);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::Button,
            label: format!("Remove {label}"),
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

        // Background with hover animation.
        let bg = if response.disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            paint::lerp_color(color, self.theme.accent_hover, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, radius);

        // Label text.
        let text_color = if response.disabled {
            self.theme.disabled_fg
        } else {
            self.theme.fg
        };
        self.text.draw_ui_text(
            label,
            rect.x + pad_x,
            rect.y + pad_y,
            text_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        // × close indicator.
        let close_color = if response.disabled {
            self.theme.disabled_fg
        } else if response.hovered {
            self.theme.fg
        } else {
            self.theme.fg_muted
        };
        self.text.draw_ui_text(
            close_str,
            rect.x + pad_x + label_w + gap,
            rect.y + pad_y,
            close_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
