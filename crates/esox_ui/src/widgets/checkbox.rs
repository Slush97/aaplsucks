//! Checkbox widget — boolean toggle with a box + checkmark.

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, InputState, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a labeled checkbox. State stored in `input.text` as "true" or "false".
    pub fn checkbox(&mut self, id: u64, input: &mut InputState, label: &str) -> Response {
        let row_h = self.theme.button_height;
        let rect = self.allocate_rect_keyed(id, self.region.w, row_h);
        self.register_widget(id, rect, WidgetKind::Checkbox);

        let mut response = self.widget_response(id, rect);
        let checked = input.text == "true";
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Checkbox, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: Some(checked),
            value_range: None, children: Vec::new(),
        });

        if response.clicked {
            input.text = if checked { "false".into() } else { "true".into() };
            input.cursor = input.text.len();
            response.changed = true;
        }

        // Box position — vertically centered.
        let box_x = rect.x;
        let box_y = rect.y + (row_h - self.theme.checkbox_size) / 2.0;
        let box_rect = crate::layout::Rect::new(box_x, box_y, self.theme.checkbox_size, self.theme.checkbox_size);

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                box_rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Box background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            if checked {
                paint::lerp_color(self.theme.accent, self.theme.accent_hover, t)
            } else {
                paint::lerp_color(self.theme.bg_input, self.theme.bg_raised, t)
            }
        };
        paint::draw_rounded_rect(self.frame, box_rect, bg, self.theme.corner_radius);

        // Box border.
        if disabled {
            paint::draw_dashed_border(
                self.frame, box_rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
        } else {
            let border = if checked || response.focused {
                self.theme.accent
            } else {
                self.theme.border
            };
            paint::draw_border(self.frame, box_rect, border);
        }

        // Checkmark glyph.
        if checked {
            let check = "\u{2713}";
            let check_w = self.text.measure_text(check, 12.0);
            let check_color = if disabled { self.theme.disabled_fg } else { self.theme.fg };
            self.text.draw_ui_text(
                check,
                box_x + (self.theme.checkbox_size - check_w) / 2.0,
                box_y + (self.theme.checkbox_size - 12.0) / 2.0,
                check_color,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        // Label text.
        let label_color = if disabled {
            self.theme.disabled_fg
        } else if response.hovered {
            self.theme.fg
        } else {
            self.theme.fg_label
        };
        self.text.draw_ui_text(
            label,
            rect.x + self.theme.checkbox_size + self.theme.input_padding,
            rect.y + (row_h - self.theme.font_size) / 2.0,
            label_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
