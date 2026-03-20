//! Slider widget — horizontal range control.

use esox_gfx::ShapeBuilder;

use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, InputState, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Typed f32 slider — takes `&mut f32` directly, no `InputState` boilerplate.
    pub fn slider_f32(&mut self, id: u64, value: &mut f32, min: f32, max: f32) -> Response {
        let mut input = InputState::new();
        input.text = if (max - min) >= 10.0 {
            format!("{}", (*value).round() as i32)
        } else {
            format!("{:.2}", *value)
        };
        let response = self.slider(id, &mut input, min, max);
        if response.changed {
            if let Ok(v) = input.text.parse::<f32>() {
                *value = v.clamp(min, max);
            }
        }
        response
    }

    /// Typed f64 slider — takes `&mut f64` directly, no `InputState` boilerplate.
    pub fn slider_f64(&mut self, id: u64, value: &mut f64, min: f64, max: f64) -> Response {
        let mut input = InputState::new();
        input.text = if (max - min) >= 10.0 {
            format!("{}", (*value).round() as i64)
        } else {
            format!("{:.2}", *value)
        };
        let response = self.slider(id, &mut input, min as f32, max as f32);
        if response.changed {
            if let Ok(v) = input.text.parse::<f64>() {
                *value = v.clamp(min, max);
            }
        }
        response
    }

    /// Draw a horizontal slider. Value is stored in `input.text` as a decimal string.
    /// Clicking anywhere on the track sets the value proportionally.
    pub fn slider(
        &mut self,
        id: u64,
        input: &mut InputState,
        min: f32,
        max: f32,
    ) -> Response {
        let rect = self.allocate_rect_keyed(id, self.region.w, self.theme.button_height);
        self.register_widget(id, rect, WidgetKind::Slider);

        let mut response = self.widget_response(id, rect);
        let disabled = response.disabled;

        // Parse current value, clamped to range.
        let current: f32 = input.text.parse().unwrap_or(min);
        let mut value = current.clamp(min, max);

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Slider, label: String::new(),
            value: Some(input.text.clone()), rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: None,
            value_range: Some((min, max, value)), children: Vec::new(),
        });

        // Handle click — map x to value.
        if response.clicked {
            let track_x = rect.x + self.theme.input_padding;
            let track_w = rect.w - self.theme.input_padding * 2.0;
            let rel = ((self.state.mouse.x - track_x) / track_w).clamp(0.0, 1.0);
            value = min + rel * (max - min);
            // Round to integer if range is large enough, otherwise 1 decimal.
            let formatted = if (max - min) >= 10.0 {
                format!("{}", value.round() as i32)
            } else {
                format!("{:.1}", value)
            };
            input.text = formatted;
            input.cursor = input.text.len();
            response.changed = true;
        }

        // Keyboard: arrow keys adjust value.
        if response.focused && !disabled {
            use esox_input::{Key, NamedKey};
            for (event, _) in &self.state.keys {
                if !event.pressed {
                    continue;
                }
                let step = if (max - min) >= 10.0 {
                    1.0
                } else {
                    (max - min) / 20.0
                };
                match &event.key {
                    Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowDown) => {
                        value = (value - step).clamp(min, max);
                        let formatted = if (max - min) >= 10.0 {
                            format!("{}", value.round() as i32)
                        } else {
                            format!("{:.1}", value)
                        };
                        input.text = formatted;
                        input.cursor = input.text.len();
                        response.changed = true;
                    }
                    Key::Named(NamedKey::ArrowRight | NamedKey::ArrowUp) => {
                        value = (value + step).clamp(min, max);
                        let formatted = if (max - min) >= 10.0 {
                            format!("{}", value.round() as i32)
                        } else {
                            format!("{:.1}", value)
                        };
                        input.text = formatted;
                        input.cursor = input.text.len();
                        response.changed = true;
                    }
                    _ => {}
                }
            }
        }

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Draw background.
        let bg = if disabled { self.theme.disabled_bg } else { self.theme.bg_input };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        // Border.
        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
        } else {
            let border_color = if response.focused {
                self.theme.accent
            } else {
                self.theme.border
            };
            paint::draw_border(self.frame, rect, border_color);
        }

        // Track area.
        let track_x = rect.x + self.theme.input_padding;
        let track_y = rect.y + rect.h / 2.0 - 2.0;
        let track_w = rect.w - self.theme.input_padding * 2.0;
        let track_h = self.theme.slider_track_height;

        // Track background.
        self.frame.push(
            ShapeBuilder::rect(track_x, track_y, track_w, track_h)
                .color(self.theme.bg_raised)
                .build(),
        );

        // Filled portion.
        let t = if (max - min).abs() < f32::EPSILON {
            0.0
        } else {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        };
        let filled_w = track_w * t;
        let fill_color = if disabled { self.theme.disabled_fg } else { self.theme.accent };
        if filled_w > 0.0 {
            self.frame.push(
                ShapeBuilder::rect(track_x, track_y, filled_w, track_h)
                    .color(fill_color)
                    .build(),
            );
        }

        // Thumb circle.
        let thumb_x = track_x + filled_w;
        let thumb_r = 6.0;
        let thumb_color = if disabled {
            self.theme.disabled_fg
        } else if response.hovered || response.focused {
            self.theme.accent_hover
        } else {
            self.theme.accent
        };
        self.frame.push(
            ShapeBuilder::circle(thumb_x, rect.y + rect.h / 2.0, thumb_r)
                .color(thumb_color)
                .build(),
        );

        // Value label on the right.
        let val_str = if input.text.is_empty() {
            format!("{}", min as i32)
        } else {
            input.text.clone()
        };
        let val_w = self.text.measure_text(&val_str, self.theme.font_size);
        self.text.draw_ui_text(
            &val_str,
            rect.x + rect.w - self.theme.input_padding - val_w,
            rect.y + (rect.h - self.theme.font_size) / 2.0,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
