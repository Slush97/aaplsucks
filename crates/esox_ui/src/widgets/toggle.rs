//! Toggle / switch widget — boolean toggle with a sliding knob.

use esox_gfx::{BorderRadius, ShapeBuilder};

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, Easing, InputState, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a labeled toggle switch. State stored in `input.text` as "true" or "false".
    pub fn toggle(&mut self, id: u64, input: &mut InputState, label: &str) -> Response {
        let row_h = self.theme.button_height;
        let tw = self.theme.toggle_width;
        let th = self.theme.toggle_height;
        let inset = self.theme.toggle_knob_inset;

        let rect = self.allocate_rect_keyed(id, self.region.w, row_h);
        self.register_widget(id, rect, WidgetKind::Toggle);

        let mut response = self.widget_response(id, rect);
        let checked = input.text == "true";
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::ToggleButton,
            label: label.to_string(),
            value: None,
            rect,
            focused: response.focused,
            disabled,
            expanded: None,
            selected: None,
            checked: Some(checked),
            value_range: None,
            children: Vec::new(),
        });

        if response.clicked {
            input.text = if checked { "false".into() } else { "true".into() };
            input.cursor = input.text.len();
            response.changed = true;
        }

        // Track position — vertically centered.
        let track_x = rect.x;
        let track_y = rect.y + (row_h - th) / 2.0;
        let track_rect = crate::layout::Rect::new(track_x, track_y, tw, th);
        let track_radius = th / 2.0;

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                track_rect,
                self.theme.accent_dim,
                track_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Knob animation: 0.0 = off (left), 1.0 = on (right).
        let knob_anim_id = id ^ 0x70661E00;
        let t = self.animate_bool(
            knob_anim_id,
            checked,
            150.0,
            Easing::EaseOutCubic,
        );

        // Track color: lerp bg_input → accent.
        let track_color = if disabled {
            self.theme.disabled_bg
        } else {
            let hover_t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            let off = paint::lerp_color(self.theme.bg_input, self.theme.bg_raised, hover_t);
            let on = paint::lerp_color(self.theme.accent, self.theme.accent_hover, hover_t);
            paint::lerp_color(off, on, t)
        };

        // Draw track.
        self.frame.push(
            ShapeBuilder::rect(track_x, track_y, tw, th)
                .color(track_color)
                .border_radius(BorderRadius::uniform(track_radius))
                .build(),
        );

        // Draw border.
        if disabled {
            paint::draw_dashed_border(
                self.frame,
                track_rect,
                self.theme.disabled_border,
                self.theme.disabled_dash_len,
                self.theme.disabled_dash_gap,
                self.theme.disabled_dash_thickness,
            );
        } else if !checked {
            paint::draw_border(self.frame, track_rect, self.theme.border);
        }

        // Knob.
        let knob_d = th - inset * 2.0;
        let knob_x_off = track_x + inset;
        let knob_x_on = track_x + tw - inset - knob_d;
        let knob_x = knob_x_off + (knob_x_on - knob_x_off) * t;
        let knob_y = track_y + inset;
        let knob_color = if disabled {
            self.theme.disabled_fg
        } else {
            self.theme.fg_on_accent
        };
        self.frame.push(
            ShapeBuilder::rect(knob_x, knob_y, knob_d, knob_d)
                .color(knob_color)
                .border_radius(BorderRadius::uniform(knob_d / 2.0))
                .build(),
        );

        // Label.
        let label_color = if disabled {
            self.theme.disabled_fg
        } else if response.hovered {
            self.theme.fg
        } else {
            self.theme.fg_label
        };
        self.text.draw_ui_text(
            label,
            rect.x + tw + self.theme.input_padding,
            rect.y + (row_h - self.theme.font_size) / 2.0,
            label_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
