//! Button widget.
//!
//! # Examples
//!
//! ```ignore
//! if ui.button(id!("save"), "Save").clicked {
//!     save_document();
//! }
//!
//! // Ghost (outline) button
//! if ui.ghost_button(id!("cancel"), "Cancel").clicked {
//!     cancel();
//! }
//!
//! // Small button with max width
//! ui.small_button(id!("ok"), "OK");
//! ```

use esox_gfx::Color;

use crate::id::HOVER_SALT;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, WidgetKind};
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
        let height = self.resolve_height();
        let corner_radius = self.resolve_corner_radius();
        let font_size = self.resolve_font_size();
        let bg_color = self.resolve_bg();
        let fg_color = self.resolve_fg();

        let rect = self.allocate_rect_keyed(id, btn_w, height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Button, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // Background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            paint::lerp_color(bg_color, self.theme.accent_hover, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, corner_radius);

        // Dashed border when disabled.
        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
        }

        // Centered label.
        let text_color = if disabled { self.theme.disabled_fg } else { fg_color };
        let label_w = self.text.measure_text(label, font_size);
        self.text.draw_ui_text(
            label,
            rect.x + (rect.w - label_w) / 2.0,
            rect.y + (rect.h - font_size) / 2.0,
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
        let rect = self.allocate_rect_keyed(id, btn_w, self.theme.small_button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Button, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        // Hover fill — subtle accent tint.
        if !disabled {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
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
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
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

    /// Draw a secondary button — `bg_raised` background, used for less prominent actions.
    pub fn secondary_button(&mut self, id: u64, label: &str) -> Response {
        let btn_w = self.region.w;
        let bg_normal = self.theme.secondary_button_bg;
        let bg_hover = self.theme.secondary_button_hover;
        self.button_variant(id, label, btn_w, bg_normal, bg_hover, self.theme.fg)
    }

    /// Draw a danger button — red background for destructive actions.
    pub fn danger_button(&mut self, id: u64, label: &str) -> Response {
        let btn_w = self.region.w;
        let bg_normal = self.theme.danger_button_bg;
        let bg_hover = self.theme.danger_button_hover;
        self.button_variant(id, label, btn_w, bg_normal, bg_hover, self.theme.fg)
    }

    fn button_variant(
        &mut self,
        id: u64,
        label: &str,
        btn_w: f32,
        bg_normal: esox_gfx::Color,
        bg_hover: esox_gfx::Color,
        text_color: esox_gfx::Color,
    ) -> Response {
        let rect = self.allocate_rect_keyed(id, btn_w, self.theme.button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Button, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame, rect, self.theme.accent_dim,
                self.theme.corner_radius, self.theme.focus_ring_expand,
            );
        }

        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            paint::lerp_color(bg_normal, bg_hover, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
        }

        let tc = if disabled { self.theme.disabled_fg } else { text_color };
        let label_w = self.text.measure_text(label, self.theme.font_size);
        self.text.draw_ui_text(
            label,
            rect.x + (rect.w - label_w) / 2.0,
            rect.y + (rect.h - self.theme.font_size) / 2.0,
            tc,
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
        let rect = self.allocate_rect_keyed(id, btn_w, self.theme.small_button_height);
        self.register_widget(id, rect, WidgetKind::Button);

        let response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Button, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        // Background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
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
