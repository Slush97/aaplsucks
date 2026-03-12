//! Status bar widget — bottom bar with left/right text.
//!
//! # Examples
//!
//! ```ignore
//! ui.status_bar("Ready", "Line 42, Col 8");
//! ```

use crate::layout::Rect;
use crate::paint;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a full-width status bar with left-aligned and right-aligned text.
    pub fn status_bar(&mut self, left: &str, right: &str) {
        let h = self.theme.item_height;
        let pad_x = self.theme.input_padding;
        let font_size = self.theme.font_size;

        let rect = self.allocate_rect(self.region.w, h);

        // Background.
        paint::draw_rounded_rect(self.frame, rect, self.theme.bg_surface, 0.0);

        // Top border (1px line).
        let border_rect = Rect::new(rect.x, rect.y, rect.w, 1.0);
        paint::draw_rounded_rect(self.frame, border_rect, self.theme.border, 0.0);

        let text_y = rect.y + (h - font_size) / 2.0;
        let color = self.theme.fg_muted;

        // Left text.
        self.text.draw_ui_text(
            left,
            rect.x + pad_x,
            text_y,
            color,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Right text.
        let right_w = self.text.measure_text(right, font_size);
        self.text.draw_ui_text(
            right,
            rect.x + rect.w - right_w - pad_x,
            text_y,
            color,
            self.frame,
            self.gpu,
            self.resources,
        );
    }
}
