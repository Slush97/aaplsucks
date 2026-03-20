//! Number input widget — numeric value with +/- buttons and scroll wheel support.
//!
//! # Examples
//!
//! ```ignore
//! let response = ui.number_input(id!("quantity"), &mut quantity, 1.0);
//! if response.changed {
//!     update_total(quantity);
//! }
//!
//! // With min/max bounds
//! let response = ui.number_input_clamped(id!("opacity"), &mut opacity, 0.1, 0.0, 1.0);
//! ```

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::id::HOVER_SALT;
use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, InputState, WidgetKind};
use crate::Ui;

/// Format a number with reasonable precision, trimming trailing zeros.
fn format_number(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        // Up to 6 decimal places, trim trailing zeros.
        let s = format!("{:.6}", v);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

impl<'f> Ui<'f> {
    /// Draw a number input with +/- buttons. Layout: `[-] value [+]`.
    ///
    /// Scroll wheel over the widget increments/decrements by `step`.
    /// Clicking the value text enters inline editing mode.
    pub fn number_input(
        &mut self,
        id: u64,
        value: &mut f64,
        step: f64,
    ) -> Response {
        self.number_input_inner(id, value, step, f64::NEG_INFINITY, f64::INFINITY)
    }

    /// Draw a number input clamped to `[min, max]`.
    pub fn number_input_clamped(
        &mut self,
        id: u64,
        value: &mut f64,
        step: f64,
        min: f64,
        max: f64,
    ) -> Response {
        self.number_input_inner(id, value, step, min, max)
    }

    fn number_input_inner(
        &mut self,
        id: u64,
        value: &mut f64,
        step: f64,
        min: f64,
        max: f64,
    ) -> Response {
        let total_w = self.region.w;
        let h = self.theme.button_height;
        let rect = self.allocate_rect_keyed(id, total_w, h);
        self.register_widget(id, rect, WidgetKind::TextInput);

        let mut response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::SpinButton,
            label: String::new(),
            value: Some(format_number(*value)),
            rect,
            focused: response.focused,
            disabled,
            expanded: None,
            selected: None,
            checked: None,
            value_range: if min.is_finite() && max.is_finite() {
                Some((min as f32, max as f32, *value as f32))
            } else {
                None
            },
            children: Vec::new(),
        });

        // Button width = height (square buttons).
        let btn_w = h;
        let minus_rect = Rect::new(rect.x, rect.y, btn_w, h);
        let plus_rect = Rect::new(rect.x + rect.w - btn_w, rect.y, btn_w, h);
        let value_rect = Rect::new(rect.x + btn_w, rect.y, rect.w - btn_w * 2.0, h);

        if disabled {
            // Disabled draw.
            paint::draw_rounded_rect(
                self.frame,
                rect,
                self.theme.disabled_bg,
                self.theme.corner_radius,
            );
            paint::draw_dashed_border(
                self.frame,
                rect,
                self.theme.disabled_border,
                self.theme.disabled_dash_len,
                self.theme.disabled_dash_gap,
                self.theme.disabled_dash_thickness,
            );

            let text = format_number(*value);
            let tw = self.text.measure_text(&text, self.theme.font_size);
            self.text.draw_ui_text(
                &text,
                value_rect.x + (value_rect.w - tw) / 2.0,
                rect.y + (h - self.theme.font_size) / 2.0,
                self.theme.disabled_fg,
                self.frame,
                self.gpu,
                self.resources,
            );

            // Minus label.
            let mw = self.text.measure_text("\u{2212}", self.theme.font_size);
            self.text.draw_ui_text(
                "\u{2212}",
                minus_rect.x + (minus_rect.w - mw) / 2.0,
                rect.y + (h - self.theme.font_size) / 2.0,
                self.theme.disabled_fg,
                self.frame,
                self.gpu,
                self.resources,
            );

            // Plus label.
            let pw = self.text.measure_text("+", self.theme.font_size);
            self.text.draw_ui_text(
                "+",
                plus_rect.x + (plus_rect.w - pw) / 2.0,
                rect.y + (h - self.theme.font_size) / 2.0,
                self.theme.disabled_fg,
                self.frame,
                self.gpu,
                self.resources,
            );

            return response;
        }

        // Detect sub-region clicks using mouse position.
        let mut minus_clicked = false;
        let mut plus_clicked = false;
        let mut value_clicked = false;

        if response.clicked {
            let mx = self.state.mouse.x;
            let my = self.state.mouse.y;
            if minus_rect.contains(mx, my) {
                minus_clicked = true;
            } else if plus_rect.contains(mx, my) {
                plus_clicked = true;
            } else if value_rect.contains(mx, my) {
                value_clicked = true;
            }
        }

        // Check if we are in inline editing mode.
        // We use a sub-id for the editing state stored in UiState's generic HashMap.
        let edit_id = id ^ 0xED17_FACE;
        let was_editing = self.state.number_edit_buffers.contains_key(&id);
        let mut editing = was_editing;

        // Enter editing mode on value click.
        if value_clicked && !was_editing {
            editing = true;
            let mut input = InputState::new();
            input.text = format_number(*value);
            input.cursor = input.text.len();
            input.select_all();
            self.state.number_edit_buffers.insert(id, input);
        }

        // Apply +/- button clicks.
        if minus_clicked && !editing {
            *value = (*value - step).max(min);
            response.changed = true;
        }
        if plus_clicked && !editing {
            *value = (*value + step).min(max);
            response.changed = true;
        }

        // Scroll wheel support.
        if !editing {
            if let Some((sx, sy, delta)) = self.state.pending_scroll {
                if rect.contains(sx, sy) {
                    if delta > 0.0 {
                        *value = (*value + step).min(max);
                    } else if delta < 0.0 {
                        *value = (*value - step).max(min);
                    }
                    self.state.pending_scroll = None;
                    response.changed = true;
                }
            }
        }

        // Handle inline editing.
        if editing {
            let input = self.state.number_edit_buffers.get_mut(&id).unwrap();

            // Process keyboard input when focused.
            if response.focused {
                let keys: Vec<_> = self.state.keys.clone();
                for (event, modifiers) in &keys {
                    if !event.pressed {
                        continue;
                    }
                    let ctrl = modifiers.ctrl();

                    match &event.key {
                        Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Tab) => {
                            // Commit the edit.
                            if let Ok(v) = input.text.parse::<f64>() {
                                *value = v.clamp(min, max);
                                response.changed = true;
                            }
                            self.state.number_edit_buffers.remove(&id);
                            editing = false;
                            break;
                        }
                        Key::Named(NamedKey::Escape) => {
                            // Cancel edit.
                            self.state.number_edit_buffers.remove(&id);
                            editing = false;
                            break;
                        }
                        Key::Named(NamedKey::Backspace) => {
                            input.save_undo();
                            input.delete_back();
                        }
                        Key::Named(NamedKey::Delete) => {
                            input.save_undo();
                            input.delete_forward();
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            if modifiers.shift() {
                                input.move_left_extend();
                            } else {
                                input.move_left();
                            }
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            if modifiers.shift() {
                                input.move_right_extend();
                            } else {
                                input.move_right();
                            }
                        }
                        Key::Named(NamedKey::Home) => {
                            input.home();
                        }
                        Key::Named(NamedKey::End) => {
                            input.end();
                        }
                        Key::Character(ch) if ctrl && ch.as_str() == "a" => {
                            input.select_all();
                        }
                        Key::Character(ch) if ctrl => {
                            // Let clipboard shortcuts pass through.
                        }
                        Key::Character(ch) => {
                            // Only allow numeric characters, minus, period.
                            for c in ch.chars() {
                                if c.is_ascii_digit()
                                    || c == '-'
                                    || c == '.'
                                    || c == 'e'
                                    || c == 'E'
                                    || c == '+'
                                {
                                    input.save_undo();
                                    input.insert_char(c);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Lose focus = commit.
            if was_editing && !response.focused {
                if let Some(input) = self.state.number_edit_buffers.remove(&id) {
                    if let Ok(v) = input.text.parse::<f64>() {
                        *value = v.clamp(min, max);
                        response.changed = true;
                    }
                }
                editing = false;
            }
        }

        // ── Draw ──

        // Focus ring.
        if response.focused {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Background.
        paint::draw_rounded_rect(
            self.frame,
            rect,
            self.theme.bg_input,
            self.theme.corner_radius,
        );

        // Border.
        let border_color = if response.focused {
            self.theme.accent
        } else {
            self.theme.border
        };
        paint::draw_border(self.frame, rect, border_color);

        // Minus button.
        let minus_hovered = minus_rect.contains(self.state.mouse.x, self.state.mouse.y);
        let minus_t = self.state.hover_t(id ^ HOVER_SALT, minus_hovered, self.theme.hover_duration_ms);
        let minus_bg = paint::lerp_color(
            self.theme.secondary_button_bg,
            self.theme.secondary_button_hover,
            minus_t,
        );
        paint::draw_rounded_rect(
            self.frame,
            minus_rect,
            minus_bg,
            self.theme.corner_radius,
        );

        let mw = self.text.measure_text("\u{2212}", self.theme.font_size);
        self.text.draw_ui_text(
            "\u{2212}",
            minus_rect.x + (minus_rect.w - mw) / 2.0,
            rect.y + (h - self.theme.font_size) / 2.0,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Plus button.
        let plus_hovered = plus_rect.contains(self.state.mouse.x, self.state.mouse.y);
        // Use a different salt for plus so hover anims don't collide.
        let plus_t = self.state.hover_t(edit_id ^ HOVER_SALT, plus_hovered, self.theme.hover_duration_ms);
        let plus_bg = paint::lerp_color(
            self.theme.secondary_button_bg,
            self.theme.secondary_button_hover,
            plus_t,
        );
        paint::draw_rounded_rect(
            self.frame,
            plus_rect,
            plus_bg,
            self.theme.corner_radius,
        );

        let pw = self.text.measure_text("+", self.theme.font_size);
        self.text.draw_ui_text(
            "+",
            plus_rect.x + (plus_rect.w - pw) / 2.0,
            rect.y + (h - self.theme.font_size) / 2.0,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Value text (or editing buffer).
        let text_y = rect.y + (h - self.theme.font_size) / 2.0;
        if editing {
            if let Some(input) = self.state.number_edit_buffers.get(&id) {
                let display = &input.text;
                let tw = self.text.measure_text(display, self.theme.font_size);
                let text_x = value_rect.x + (value_rect.w - tw) / 2.0;

                self.text.draw_ui_text(
                    display,
                    text_x,
                    text_y,
                    self.theme.fg,
                    self.frame,
                    self.gpu,
                    self.resources,
                );

                // Blinking cursor.
                if response.focused && self.state.cursor_blink {
                    let cursor_pos = input.cursor.min(display.len());
                    let cursor_x_offset =
                        self.text.measure_text(&display[..cursor_pos], self.theme.font_size);
                    let cx = text_x + cursor_x_offset;
                    if cx >= value_rect.x && cx <= value_rect.x + value_rect.w {
                        self.frame.push(
                            ShapeBuilder::rect(
                                cx,
                                rect.y + self.theme.label_pad_y + 2.0,
                                self.theme.cursor_width,
                                h - self.theme.label_pad_y * 2.0 - 4.0,
                            )
                            .color(self.theme.fg)
                            .build(),
                        );
                    }
                }
            }
        } else {
            let display = format_number(*value);
            let tw = self.text.measure_text(&display, self.theme.font_size);
            self.text.draw_ui_text(
                &display,
                value_rect.x + (value_rect.w - tw) / 2.0,
                text_y,
                self.theme.fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        // Separator lines between buttons and value.
        self.frame.push(
            ShapeBuilder::rect(minus_rect.x + minus_rect.w, rect.y, 1.0, h)
                .color(self.theme.border)
                .build(),
        );
        self.frame.push(
            ShapeBuilder::rect(plus_rect.x, rect.y, 1.0, h)
                .color(self.theme.border)
                .build(),
        );

        response
    }
}
