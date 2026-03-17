//! Badge widget — notification count circle/pill.
//!
//! # Examples
//!
//! ```ignore
//! ui.row(|ui| {
//!     ui.label("Inbox");
//!     ui.badge(5);
//! });
//!
//! // Just a presence dot (no number)
//! ui.badge_dot();
//! ```

use esox_gfx::{BorderRadius, ShapeBuilder};

use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a red notification badge with a count. Shows "99+" for values > 99.
    pub fn badge(&mut self, count: u32) {
        let text = if count > 99 {
            "99+".to_string()
        } else {
            count.to_string()
        };

        let font_size = self.theme.font_size * 0.75; // smaller text
        let pad_x = self.theme.badge_pad_x;
        let pad_y = self.theme.badge_pad_y;
        let text_w = self.text.measure_text(&text, font_size);
        let badge_w = (text_w + pad_x * 2.0).max(font_size + pad_y * 2.0); // at least circular
        let badge_h = font_size + pad_y * 2.0;
        let radius = badge_h / 2.0;

        let rect = self.allocate_rect(badge_w, badge_h);

        // Red background.
        let bg = self.theme.red;
        self.frame.push(
            ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
                .color(bg)
                .border_radius(BorderRadius::uniform(radius))
                .build(),
        );

        // White text, centered.
        let white = self.theme.fg_on_accent;
        self.text.draw_text(
            &text,
            rect.x + (rect.w - text_w) / 2.0,
            rect.y + pad_y,
            font_size,
            white,
            self.frame,
            self.gpu,
            self.resources,
        );
    }

    /// Draw a small colored dot badge (no number) — a "new" indicator.
    pub fn badge_dot(&mut self) {
        let dot_size = 8.0;
        let rect = self.allocate_rect(dot_size, dot_size);
        let radius = dot_size / 2.0;

        self.frame.push(
            ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
                .color(self.theme.red)
                .border_radius(BorderRadius::uniform(radius))
                .build(),
        );
    }
}
