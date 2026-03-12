//! Radio button widget — exclusive selection within a group.

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, InputState, WidgetKind};
use crate::Ui;

/// Outer circle size in logical pixels.
const BOX_SIZE: f32 = 16.0;
/// Inner dot size when selected.
const DOT_SIZE: f32 = 6.0;

impl<'f> Ui<'f> {
    /// Draw a labeled radio button. All options in a group share the same `InputState`,
    /// where `input.text` stores the selected option's index as a string (e.g., `"0"`, `"2"`).
    pub fn radio(
        &mut self,
        id: u64,
        input: &mut InputState,
        option_index: usize,
        label: &str,
    ) -> Response {
        let row_h = self.theme.button_height;
        let rect = self.allocate_rect(self.region.w, row_h);
        self.register_widget(id, rect, WidgetKind::Radio);

        let mut response = self.widget_response(id, rect);
        let selected = input.text.parse::<usize>() == Ok(option_index);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::RadioButton, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: Some(selected), checked: None,
            value_range: None, children: Vec::new(),
        });

        if response.clicked {
            input.text = format!("{}", option_index);
            input.cursor = input.text.len();
            response.changed = true;
        }

        // Circle position — vertically centered.
        let box_x = rect.x;
        let box_y = rect.y + (row_h - BOX_SIZE) / 2.0;
        let box_rect = crate::layout::Rect::new(box_x, box_y, BOX_SIZE, BOX_SIZE);

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                box_rect,
                self.theme.accent_dim,
                3.0,
                self.theme.focus_ring_expand,
            );
        }

        // Outer circle (border).
        if disabled {
            // Draw the outer circle with disabled bg, then dashed border.
            paint::draw_rounded_rect(self.frame, box_rect, self.theme.disabled_bg, BOX_SIZE / 2.0);
            paint::draw_dashed_border(
                self.frame, box_rect, self.theme.disabled_border,
                6.0, 4.0, 1.0,
            );
        } else {
            let border_color = if selected || response.focused {
                self.theme.accent
            } else {
                self.theme.border
            };
            paint::draw_rounded_rect(self.frame, box_rect, border_color, BOX_SIZE / 2.0);
        }

        // Inner circle (background).
        let inset = 1.0;
        let inner_size = BOX_SIZE - inset * 2.0;
        let inner_rect =
            crate::layout::Rect::new(box_x + inset, box_y + inset, inner_size, inner_size);

        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, 100.0);
            if selected {
                paint::lerp_color(self.theme.accent, self.theme.accent_hover, t)
            } else {
                paint::lerp_color(self.theme.bg_input, self.theme.bg_raised, t)
            }
        };
        paint::draw_rounded_rect(self.frame, inner_rect, bg, inner_size / 2.0);

        // Selected dot.
        if selected {
            let dot_x = box_x + (BOX_SIZE - DOT_SIZE) / 2.0;
            let dot_y = box_y + (BOX_SIZE - DOT_SIZE) / 2.0;
            let dot_rect = crate::layout::Rect::new(dot_x, dot_y, DOT_SIZE, DOT_SIZE);
            let dot_color = if disabled { self.theme.disabled_fg } else { self.theme.fg };
            paint::draw_rounded_rect(self.frame, dot_rect, dot_color, DOT_SIZE / 2.0);
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
            rect.x + BOX_SIZE + self.theme.input_padding,
            rect.y + (row_h - self.theme.font_size) / 2.0,
            label_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
