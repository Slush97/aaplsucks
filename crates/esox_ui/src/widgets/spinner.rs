//! Spinner / loading indicator widget — animated rotating dots.

use std::f32::consts::TAU;
use std::time::Instant;

use esox_gfx::{BorderRadius, Color, ShapeBuilder};

use crate::state::{A11yNode, A11yRole};
use crate::Ui;

/// Epoch for spinner phase calculation (process start).
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn epoch() -> Instant {
    *EPOCH.get_or_init(Instant::now)
}

impl<'f> Ui<'f> {
    /// Draw a spinner at the default size.
    pub fn spinner(&mut self) {
        let size = self.theme.spinner_size;
        self.spinner_sized(size);
    }

    /// Draw a spinner at a custom size.
    pub fn spinner_sized(&mut self, size: f32) {
        let rect = self.allocate_rect(size, size);
        self.state.spinner_active = true;

        self.push_a11y_node(A11yNode {
            id: 0,
            role: A11yRole::ProgressBar,
            label: "Loading".to_string(),
            value: None,
            rect,
            focused: false,
            disabled: false,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: Vec::new(),
        });

        let cx = rect.x + size / 2.0;
        let cy = rect.y + size / 2.0;
        let radius = size / 2.0 - 2.0; // ring radius
        let dot_r = size * 0.08; // dot radius
        let dot_count = 12;

        let speed = self.theme.spinner_speed;
        let elapsed = Instant::now().duration_since(epoch()).as_secs_f32();
        let phase = elapsed * speed * TAU;

        let base_color = self.theme.fg_muted;

        for i in 0..dot_count {
            let angle = (i as f32 / dot_count as f32) * TAU - std::f32::consts::FRAC_PI_2;
            let dx = cx + angle.cos() * radius - dot_r;
            let dy = cy + angle.sin() * radius - dot_r;

            // Alpha fades around the ring based on phase.
            let dot_phase = (i as f32 / dot_count as f32) * TAU;
            let diff = ((phase - dot_phase) % TAU + TAU) % TAU;
            let alpha = 0.15 + 0.85 * (1.0 - diff / TAU);

            let color = Color::new(base_color.r, base_color.g, base_color.b, alpha);

            self.frame.push(
                ShapeBuilder::rect(dx, dy, dot_r * 2.0, dot_r * 2.0)
                    .color(color)
                    .border_radius(BorderRadius::uniform(dot_r))
                    .build(),
            );
        }
    }
}
