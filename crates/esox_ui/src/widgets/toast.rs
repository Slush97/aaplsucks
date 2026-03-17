//! Toast notification widget — auto-dismissing notifications stacked in top-right corner.

use esox_gfx::{BorderRadius, Color, ShapeBuilder};

use crate::id::{fnv1a_mix, HOVER_SALT};
use crate::layout::Rect;
use crate::paint;
use crate::state::{Easing, ToastKind};
use crate::Ui;

const TOAST_ANIM_SALT: u64 = 0xd0a57fade1234567;

impl<'f> Ui<'f> {
    /// Draw all active toasts. Called from `finish()`.
    pub(crate) fn draw_toasts(&mut self) {
        if self.state.toasts.toasts.is_empty() {
            return;
        }

        let margin = self.theme.toast_margin;
        let toast_w = self.theme.toast_w;
        let toast_h = self.theme.toast_h;
        let corner = self.theme.corner_radius;
        let max_visible = self.theme.toast_max_visible;
        let fade_in_ms = self.theme.toast_fade_in_ms;
        let fade_out_ms = self.theme.toast_fade_out_ms;

        // Position: top-right corner of viewport.
        let start_x = self.region.x + self.region.w - toast_w - margin;
        let mut y = self.region.y + margin;

        // Collect toast info (avoid borrow issues).
        let toast_infos: Vec<_> = self.state.toasts.toasts.iter().take(max_visible).map(|t| {
            let elapsed = t.created.elapsed().as_millis() as u64;
            let phase_t = if elapsed < fade_in_ms as u64 {
                // Fade in.
                elapsed as f32 / fade_in_ms
            } else if elapsed < t.duration_ms {
                // Fully visible.
                1.0
            } else {
                // Fade out.
                let fade_elapsed = elapsed - t.duration_ms;
                1.0 - (fade_elapsed as f32 / fade_out_ms).min(1.0)
            };
            (t.id, t.kind, t.message.clone(), phase_t)
        }).collect();

        for (toast_id, kind, message, phase_t) in toast_infos {
            let opacity = Easing::EaseOutCubic.apply(phase_t);
            let slide_offset = (1.0 - opacity) * 30.0; // Slide in from right.

            let tx = start_x + slide_offset;
            let ty = y;

            // Background color by kind.
            let bg = match kind {
                ToastKind::Info => self.theme.toast_info_bg,
                ToastKind::Success => self.theme.toast_success_bg,
                ToastKind::Error => self.theme.toast_error_bg,
                ToastKind::Warning => self.theme.toast_warning_bg,
            };
            let bg_with_alpha = Color::new(bg.r, bg.g, bg.b, bg.a * opacity);

            // Shadow.
            self.frame.push(
                ShapeBuilder::rect(tx + 1.0, ty + 1.0, toast_w, toast_h)
                    .color(Color::new(0.0, 0.0, 0.0, 0.2 * opacity))
                    .border_radius(BorderRadius::uniform(corner))
                    .build(),
            );

            // Background.
            paint::draw_rounded_rect(
                self.frame,
                Rect::new(tx, ty, toast_w, toast_h),
                bg_with_alpha,
                corner,
            );

            // Message text.
            let text_color = Color::new(
                self.theme.fg.r,
                self.theme.fg.g,
                self.theme.fg.b,
                self.theme.fg.a * opacity,
            );
            self.text.draw_ui_text(
                &message,
                tx + self.theme.input_padding,
                ty + (toast_h - self.theme.font_size) / 2.0,
                text_color,
                self.frame,
                self.gpu,
                self.resources,
            );

            // Dismiss button (X).
            let x_size = 16.0;
            let x_rect = Rect::new(
                tx + toast_w - self.theme.input_padding - x_size,
                ty + (toast_h - x_size) / 2.0,
                x_size,
                x_size,
            );
            let dismiss_id = fnv1a_mix(toast_id, TOAST_ANIM_SALT);
            // Simple hit test — no full widget registration to avoid polluting focus chain.
            if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                if !*consumed && x_rect.contains(cx, cy) {
                    self.state.toasts.dismiss(toast_id);
                    *consumed = true;
                }
            }

            let x_hovered = x_rect.contains(self.state.mouse.x, self.state.mouse.y);
            let x_hover_t = self.state.hover_t(dismiss_id ^ HOVER_SALT, x_hovered, self.theme.hover_duration_ms);
            let x_color = paint::lerp_color(
                Color::new(self.theme.fg_muted.r, self.theme.fg_muted.g, self.theme.fg_muted.b, opacity),
                Color::new(self.theme.fg.r, self.theme.fg.g, self.theme.fg.b, opacity),
                x_hover_t,
            );
            let x_text = "\u{2715}";
            let x_w = self.text.measure_text(x_text, 12.0);
            self.text.draw_text(
                x_text,
                x_rect.x + (x_size - x_w) / 2.0,
                x_rect.y + (x_size - 12.0) / 2.0,
                12.0,
                x_color,
                self.frame,
                self.gpu,
                self.resources,
            );

            y += toast_h + margin * 0.5;
        }
    }
}
