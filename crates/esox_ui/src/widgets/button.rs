//! Button widget.

use esox_gfx::Color;

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::WidgetKind;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw an accent-colored action button with an explicit max width.
    /// The button is left-aligned within the allocated region.
    pub fn button_max_width(&mut self, id: u64, label: &str, max_w: f32) -> Response {
        let btn_w = self.region.w.min(max_w);
        self.button_inner(id, label, btn_w)
    }

    /// Draw an accent-colored action button (full region width).
    pub fn button(&mut self, id: u64, label: &str) -> Response {
        let btn_w = self.region.w;
        self.button_inner(id, label, btn_w)
    }

    fn button_inner(&mut self, id: u64, label: &str, btn_w: f32) -> Response {
        let rect = self.allocate_rect(btn_w, self.theme.button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

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

        // Background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, 100.0);
            paint::lerp_color(self.theme.accent, self.theme.accent_hover, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        // Dashed border when disabled.
        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                6.0, 4.0, 1.0,
            );
        }

        // Centered label.
        let text_color = if disabled { self.theme.disabled_fg } else { self.theme.fg };
        let label_w = self.text.measure_text(label, self.theme.font_size);
        self.text.draw_ui_text(
            label,
            rect.x + (rect.w - label_w) / 2.0,
            rect.y + (rect.h - self.theme.font_size) / 2.0,
            text_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }

    /// Draw a ghost (outline) button — transparent bg with accent border. Good for secondary actions.
    pub fn ghost_button(&mut self, id: u64, label: &str) -> Response {
        let label_w = self.text.measure_text(label, self.theme.font_size);
        let btn_w = (label_w + self.theme.input_padding * 4.0).max(self.theme.small_button_min_w);
        let rect = self.allocate_rect(btn_w, self.theme.small_button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        // Hover fill — subtle accent tint.
        if !disabled {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, 120.0);
            if t > 0.0 {
                let fill = Color::new(
                    self.theme.accent.r,
                    self.theme.accent.g,
                    self.theme.accent.b,
                    0.10 * t,
                );
                paint::draw_rounded_rect(self.frame, rect, fill, self.theme.corner_radius);
            }
        }

        // Border.
        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                6.0, 4.0, 1.0,
            );
        } else {
            let border = if response.focused || response.hovered {
                self.theme.accent
            } else {
                self.theme.border
            };
            paint::draw_border(self.frame, rect, border);
        }

        // Label.
        let label_w = self.text.measure_text(label, self.theme.font_size);
        let text_color = if disabled {
            self.theme.disabled_fg
        } else if response.hovered {
            self.theme.accent
        } else {
            self.theme.fg_muted
        };
        self.text.draw_ui_text(
            label,
            rect.x + (rect.w - label_w) / 2.0,
            rect.y + (rect.h - self.theme.font_size) / 2.0,
            text_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }

    /// Draw a small button with configurable background color.
    pub fn small_button(&mut self, id: u64, label: &str, bg_color: Color) -> Response {
        let label_w = self.text.measure_text(label, self.theme.font_size);
        let btn_w = (label_w + self.theme.input_padding * 4.0).max(self.theme.small_button_min_w);
        let rect = self.allocate_rect(btn_w, self.theme.small_button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        // Background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, 100.0);
            Color::new(
                (bg_color.r + 0.08 * t).min(1.0),
                (bg_color.g + 0.08 * t).min(1.0),
                (bg_color.b + 0.08 * t).min(1.0),
                bg_color.a,
            )
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        let text_color = if disabled { self.theme.disabled_fg } else { self.theme.fg };
        let label_w = self.text.measure_text(label, self.theme.font_size);
        self.text.draw_ui_text(
            label,
            rect.x + (rect.w - label_w) / 2.0,
            rect.y + (rect.h - self.theme.font_size) / 2.0,
            text_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        response
    }
}
