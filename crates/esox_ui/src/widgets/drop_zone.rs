//! Drop zone widget — file selection area with dashed border.

use std::path::PathBuf;

use crate::paint;
use crate::response::Response;
use crate::state::WidgetKind;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a file drop zone. Returns Response where `clicked` means open file dialog.
    pub fn drop_zone(&mut self, id: u64, files: &[PathBuf]) -> Response {
        let rect = self.allocate_rect_keyed(id, self.region.w, self.theme.drop_zone_height);
        self.register_widget(id, rect, WidgetKind::DropZone);

        let response = self.widget_response(id, rect);

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
        let bg = if response.hovered {
            self.theme.bg_raised
        } else {
            self.theme.bg_surface
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        // Dashed border.
        let border_color = if response.focused {
            self.theme.accent
        } else {
            self.theme.fg_dim
        };
        paint::draw_dashed_border(
            self.frame,
            rect,
            border_color,
            self.theme.drop_zone_dash,
            self.theme.drop_zone_dash_gap,
            self.theme.drop_zone_dash_thickness,
        );

        // Content — file names or placeholder.
        if files.is_empty() {
            let label = "Drop file here or click to browse";
            let label_w = self.text.measure_text(label, self.theme.font_size);
            self.text.draw_ui_text(
                label,
                rect.x + (rect.w - label_w) / 2.0,
                rect.y + (rect.h - self.theme.font_size) / 2.0,
                self.theme.fg_muted,
                self.frame,
                self.gpu,
                self.resources,
            );
        } else {
            let line_h = self.theme.font_size + 6.0;
            let max_lines = ((rect.h - self.theme.label_pad_y * 4.0) / line_h) as usize;
            let start_y =
                rect.y + (rect.h - (files.len().min(max_lines) as f32 * line_h)) / 2.0;

            for (i, path) in files.iter().take(max_lines).enumerate() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_else(|| path.to_string_lossy());
                let name_w = self.text.measure_text(&name, self.theme.font_size);
                self.text.draw_ui_text(
                    &name,
                    rect.x + (rect.w - name_w) / 2.0,
                    start_y + i as f32 * line_h,
                    self.theme.fg,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }

            if files.len() > max_lines {
                let extra = format!("+{} more", files.len() - max_lines);
                let extra_w = self.text.measure_text(&extra, self.theme.font_size);
                self.text.draw_ui_text(
                    &extra,
                    rect.x + (rect.w - extra_w) / 2.0,
                    start_y + max_lines as f32 * line_h,
                    self.theme.fg_muted,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }
        }

        response
    }
}
