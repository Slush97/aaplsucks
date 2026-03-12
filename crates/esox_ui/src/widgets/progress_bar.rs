//! Progress bar widget — horizontal fill indicator.

use esox_gfx::{BorderRadius, Color, ShapeBuilder};

use crate::paint;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a progress bar. `value` is clamped to `0.0..=1.0`. Uses accent color.
    pub fn progress_bar(&mut self, value: f32) {
        self.progress_bar_colored(value, self.theme.accent);
    }

    /// Draw a progress bar with a custom fill color.
    pub fn progress_bar_colored(&mut self, value: f32, color: Color) {
        let h = self.theme.progress_bar_height;
        let radius = h / 2.0;
        let rect = self.allocate_rect(self.region.w, h);
        let v = value.clamp(0.0, 1.0);

        self.push_a11y_node(crate::state::A11yNode {
            id: 0, role: crate::state::A11yRole::ProgressBar,
            label: format!("{:.0}%", v * 100.0),
            value: Some(v.to_string()), rect, focused: false, disabled: false,
            expanded: None, selected: None, checked: None,
            value_range: Some((0.0, 1.0, v)), children: Vec::new(),
        });

        // Track.
        paint::draw_rounded_rect(self.frame, rect, self.theme.bg_input, radius);

        // Fill.
        let fill_w = rect.w * v;
        if fill_w > 0.0 {
            self.frame.push(
                ShapeBuilder::rect(rect.x, rect.y, fill_w, h)
                    .color(color)
                    .border_radius(BorderRadius::uniform(radius))
                    .build(),
            );
        }
    }
}
