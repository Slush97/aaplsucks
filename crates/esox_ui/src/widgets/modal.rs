//! Modal dialog widget — overlay with backdrop, focus trap, and animations.
//!
//! # Examples
//!
//! ```ignore
//! let resp = ui.modal(id!("confirm"), &mut open, "Confirm Delete", 400.0, |ui| {
//!     ui.label("Are you sure you want to delete this item?");
//! });
//! // resp.clicked when Confirm is pressed, resp.right_clicked reserved
//! ```

use esox_input::{Key, NamedKey};

use crate::id::{fnv1a_mix, HOVER_SALT};
use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{Easing, ModalAction, ModalState, WidgetKind};
use crate::Ui;

const MODAL_ANIM_SALT: u64 = 0xa7b3c5d9e1f20483;

impl<'f> Ui<'f> {
    /// Draw a modal dialog. When `*open` is true, draws backdrop + centered modal.
    /// The closure draws content inside the modal.
    pub fn modal(
        &mut self,
        id: u64,
        open: &mut bool,
        title: &str,
        width: f32,
        f: impl FnOnce(&mut Self),
    ) -> Response {
        if !*open {
            return Response::default();
        }

        // Push modal state if not present.
        if !self.state.modal_stack.iter().any(|m| m.id == id) {
            self.state.modal_stack.push(ModalState {
                id,
                open: true,
                saved_focus: self.state.focused,
            });
        }

        self.push_a11y_node(crate::state::A11yNode {
            id, role: crate::state::A11yRole::Dialog, label: title.to_string(),
            value: None, rect: self.region, focused: false, disabled: false,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        let anim_id = fnv1a_mix(id, MODAL_ANIM_SALT);
        let duration = self.theme.modal_fade_duration_ms;
        let opacity = self.state.anim_t(anim_id, 1.0, duration, Easing::EaseOutCubic);

        let vp = self.region;
        let modal_w = width.clamp(self.theme.modal_min_width, self.theme.modal_max_width).min(vp.w - self.theme.modal_margin * 2.0);
        let title_h = self.theme.modal_title_height;
        let pad = self.theme.modal_padding;
        let corner = self.theme.modal_corner_radius;

        // We'll draw the backdrop, then render modal content into a temp region,
        // then handle close.

        // Backdrop.
        let backdrop_color = esox_gfx::Color::new(
            self.theme.modal_backdrop.r,
            self.theme.modal_backdrop.g,
            self.theme.modal_backdrop.b,
            self.theme.modal_backdrop.a * opacity,
        );
        self.frame.push(
            esox_gfx::ShapeBuilder::rect(vp.x, vp.y, vp.w, vp.h)
                .color(backdrop_color)
                .build(),
        );

        // Backdrop click to close.
        let mut close = false;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            // We'll check below if click is outside modal rect.
            let _ = (cx, cy, consumed); // Will handle after we know modal rect.
        }

        // Escape to close.
        for (event, _) in &self.state.keys.clone() {
            if event.pressed && event.key == Key::Named(NamedKey::Escape) {
                close = true;
            }
        }

        // Save layout state and draw content into modal area.
        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_hit_clip = self.hit_clip;

        // Estimate modal height (we'll draw content, measure, then position).
        // For simplicity, center vertically in viewport with a reasonable max height.
        let max_h = vp.h * self.theme.modal_max_height_ratio;
        let modal_x = vp.x + (vp.w - modal_w) / 2.0;
        let modal_y_start = vp.y + vp.h * self.theme.modal_vertical_offset; // Start higher, will be clamped

        // Draw modal background.
        let _modal_bg_rect = Rect::new(modal_x, modal_y_start, modal_w, max_h);

        // Shadow.
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(modal_x + 2.0, modal_y_start + 2.0, modal_w, max_h),
            esox_gfx::Color::new(0.0, 0.0, 0.0, self.theme.modal_shadow_alpha * opacity),
            corner,
        );

        // Background.
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(modal_x, modal_y_start, modal_w, max_h),
            self.theme.bg_surface,
            corner,
        );

        // Title bar.
        let title_rect = Rect::new(modal_x, modal_y_start, modal_w, title_h);
        paint::draw_rounded_rect(self.frame, title_rect, self.theme.bg_raised, corner);

        // Title text.
        self.text.draw_ui_text(
            title,
            modal_x + pad,
            modal_y_start + (title_h - self.theme.font_size) / 2.0,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Close button (X).
        let close_btn_size = self.theme.modal_close_btn_size;
        let close_x = modal_x + modal_w - pad - close_btn_size;
        let close_y = modal_y_start + (title_h - close_btn_size) / 2.0;
        let close_rect = Rect::new(close_x, close_y, close_btn_size, close_btn_size);
        let close_id = fnv1a_mix(id, 0xC105E);
        self.register_widget(close_id, close_rect, WidgetKind::Button);
        let close_resp = self.widget_response(close_id, close_rect);

        let close_hover_t = self.state.hover_t(close_id ^ HOVER_SALT, close_resp.hovered, self.theme.hover_duration_ms);
        let close_color = paint::lerp_color(self.theme.fg_muted, self.theme.red, close_hover_t);
        let x_text = "\u{2715}";
        let x_w = self.text.measure_text(x_text, self.theme.font_size);
        self.text.draw_ui_text(
            x_text,
            close_x + (close_btn_size - x_w) / 2.0,
            close_y + (close_btn_size - self.theme.font_size) / 2.0,
            close_color,
            self.frame,
            self.gpu,
            self.resources,
        );
        if close_resp.clicked {
            close = true;
        }

        // Set hit_clip to modal rect for focus trap.
        let content_rect = Rect::new(modal_x + pad, modal_y_start + title_h + pad, modal_w - pad * 2.0, max_h - title_h - pad * 2.0);
        self.hit_clip = Some(Rect::new(modal_x, modal_y_start, modal_w, max_h));

        // Set cursor for content.
        self.cursor = crate::layout::Vec2 { x: content_rect.x, y: content_rect.y };
        self.region = content_rect;

        // Run content closure.
        f(self);

        // Restore.
        self.cursor = saved_cursor;
        self.region = saved_region;
        self.hit_clip = saved_hit_clip;

        // Check backdrop click (outside modal).
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if !*consumed {
                let modal_rect = Rect::new(modal_x, modal_y_start, modal_w, max_h);
                if !modal_rect.contains(cx, cy) {
                    close = true;
                    *consumed = true;
                }
            }
        }

        let mut response = Response::default();
        if close {
            *open = false;
            // Restore saved focus.
            if let Some(pos) = self.state.modal_stack.iter().position(|m| m.id == id) {
                let modal = self.state.modal_stack.remove(pos);
                self.state.focused = modal.saved_focus;
            }
            // Reset animation.
            self.state.anims.remove(&anim_id);
            response.changed = true;
        }

        response
    }

    /// Convenience: a confirmation modal with OK/Cancel buttons.
    pub fn modal_confirm(
        &mut self,
        id: u64,
        open: &mut bool,
        title: &str,
        message: &str,
    ) -> ModalAction {
        let mut action = ModalAction::None;
        let ok_id = fnv1a_mix(id, 0x0000_0001);
        let cancel_id = fnv1a_mix(id, 0x0000_0002);

        self.modal(id, open, title, 400.0, |ui| {
            ui.label(message);
            ui.add_space(16.0);
            ui.row(|ui| {
                if ui.button(ok_id, "OK").clicked {
                    action = ModalAction::Confirm;
                }
                if ui.ghost_button(cancel_id, "Cancel").clicked {
                    action = ModalAction::Cancel;
                }
            });
        });

        if action != ModalAction::None {
            *open = false;
            // Clean up modal stack.
            if let Some(pos) = self.state.modal_stack.iter().position(|m| m.id == id) {
                let modal = self.state.modal_stack.remove(pos);
                self.state.focused = modal.saved_focus;
            }
            let anim_id = fnv1a_mix(id, MODAL_ANIM_SALT);
            self.state.anims.remove(&anim_id);
        }

        action
    }

    /// Draw all active modals (called from finish). Currently no-op since modals are drawn inline.
    pub(crate) fn draw_modals(&mut self) {
        // Modals are drawn inline via the `modal()` method calls during the frame.
        // This method exists for future extensibility (e.g., deferred modal rendering).
    }
}
