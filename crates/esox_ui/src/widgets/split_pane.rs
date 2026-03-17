//! Split pane widget -- draggable divider between two resizable panels.
//!
//! # Examples
//!
//! ```ignore
//! ui.split_pane_h(id!("main_split"), 0.3, |ui| {
//!     ui.label("Left panel");
//! }, |ui| {
//!     ui.label("Right panel");
//! });
//!
//! ui.split_pane_v(id!("v_split"), 0.5, |ui| {
//!     ui.label("Top panel");
//! }, |ui| {
//!     ui.label("Bottom panel");
//! });
//! ```

use crate::layout::{Rect, Vec2};
use crate::state::WidgetKind;
use crate::Ui;

use esox_gfx::ShapeBuilder;


impl<'f> Ui<'f> {
    /// Horizontal split pane: left | right with a draggable vertical divider.
    ///
    /// `ratio` is the initial ratio (0.0-1.0) for the left panel, used only when
    /// no stored ratio exists for this `id`. The divider is draggable; the stored
    /// ratio is clamped to [0.05, 0.95].
    pub fn split_pane_h(
        &mut self,
        id: u64,
        ratio: f32,
        left: impl FnOnce(&mut Self),
        right: impl FnOnce(&mut Self),
    ) {
        self.split_pane_inner(id, ratio, true, left, right);
    }

    /// Vertical split pane: top / bottom with a draggable horizontal divider.
    ///
    /// `ratio` is the initial ratio (0.0-1.0) for the top panel, used only when
    /// no stored ratio exists for this `id`. The divider is draggable; the stored
    /// ratio is clamped to [0.05, 0.95].
    pub fn split_pane_v(
        &mut self,
        id: u64,
        ratio: f32,
        top: impl FnOnce(&mut Self),
        bottom: impl FnOnce(&mut Self),
    ) {
        self.split_pane_inner(id, ratio, false, top, bottom);
    }

    /// Shared implementation for horizontal and vertical split panes.
    fn split_pane_inner(
        &mut self,
        id: u64,
        initial_ratio: f32,
        horizontal: bool,
        first: impl FnOnce(&mut Self),
        second: impl FnOnce(&mut Self),
    ) {
        // Read or initialize stored ratio.
        let ratio = *self
            .state
            .split_ratios
            .entry(id)
            .or_insert_with(|| initial_ratio.clamp(0.05, 0.95));

        // Compute the total available rect. For horizontal splits we use the
        // full region width; for vertical we need an explicit height. We use
        // the remaining region height.
        let total_w = self.region.w;
        let total_h = self.region.h - (self.cursor.y - self.region.y);
        let origin_x = self.cursor.x;
        let origin_y = self.cursor.y;

        // Calculate panel rects and divider rect.
        let (first_rect, divider_rect, second_rect) = if horizontal {
            let available = total_w - self.theme.split_pane_divider;
            let left_w = available * ratio;
            let right_w = available - left_w;
            let lr = Rect::new(origin_x, origin_y, left_w, total_h);
            let dr = Rect::new(origin_x + left_w, origin_y, self.theme.split_pane_divider, total_h);
            let rr = Rect::new(origin_x + left_w + self.theme.split_pane_divider, origin_y, right_w, total_h);
            (lr, dr, rr)
        } else {
            let available = total_h - self.theme.split_pane_divider;
            let top_h = available * ratio;
            let bottom_h = available - top_h;
            let tr = Rect::new(origin_x, origin_y, total_w, top_h);
            let dr = Rect::new(origin_x, origin_y + top_h, total_w, self.theme.split_pane_divider);
            let br = Rect::new(origin_x, origin_y + top_h + self.theme.split_pane_divider, total_w, bottom_h);
            (tr, dr, br)
        };

        // --- Handle divider interaction ---

        // Register divider for hit testing / cursor icon.
        let divider_id = id.wrapping_add(1);
        let kind = if horizontal {
            WidgetKind::SplitDividerH
        } else {
            WidgetKind::SplitDividerV
        };
        self.state.hit_rects.push((divider_rect, divider_id, kind));

        // Start drag on click inside divider.
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if !*consumed && divider_rect.contains(cx, cy) {
                *consumed = true;
                self.state.split_drag = Some((id, horizontal));
            }
        }

        // Update ratio while dragging.
        if let Some((drag_id, _)) = self.state.split_drag {
            if drag_id == id && self.state.mouse_pressed {
                let new_ratio = if horizontal {
                    let available = total_w - self.theme.split_pane_divider;
                    if available > 0.0 {
                        (self.state.mouse.x - origin_x - self.theme.split_pane_divider / 2.0) / available
                    } else {
                        ratio
                    }
                } else {
                    let available = total_h - self.theme.split_pane_divider;
                    if available > 0.0 {
                        (self.state.mouse.y - origin_y - self.theme.split_pane_divider / 2.0) / available
                    } else {
                        ratio
                    }
                };
                let clamped = new_ratio.clamp(0.05, 0.95);
                self.state.split_ratios.insert(id, clamped);
                self.state.damage.invalidate_all();
            }
        }

        // --- Draw divider ---
        let divider_hovered = divider_rect.contains(self.state.mouse.x, self.state.mouse.y);
        let is_dragging = self
            .state
            .split_drag
            .map_or(false, |(drag_id, _)| drag_id == id);

        let divider_color = if is_dragging {
            self.theme.accent
        } else if divider_hovered {
            self.theme.border
        } else {
            self.theme.bg_surface
        };

        self.frame.push(
            ShapeBuilder::rect(divider_rect.x, divider_rect.y, divider_rect.w, divider_rect.h)
                .color(divider_color)
                .build(),
        );

        // --- Draw first panel (left / top) ---
        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;
        let saved_clip = self.frame.active_clip();
        let saved_hit_clip = self.hit_clip;

        // Set clip for first panel.
        let first_clip = match saved_clip {
            Some(prev) => {
                let prev_rect = Rect::new(prev[0], prev[1], prev[2], prev[3]);
                first_rect.intersect(&prev_rect).unwrap_or(first_rect)
            }
            None => first_rect,
        };
        self.frame.set_active_clip(Some(first_clip.to_clip_array()));
        self.hit_clip = Some(match saved_hit_clip {
            Some(prev) => first_rect.intersect(&prev).unwrap_or(first_rect),
            None => first_rect,
        });

        self.cursor = Vec2 {
            x: first_rect.x,
            y: first_rect.y,
        };
        self.region = first_rect;
        self.spacing = saved_spacing;

        first(self);

        // --- Draw second panel (right / bottom) ---
        let second_clip = match saved_clip {
            Some(prev) => {
                let prev_rect = Rect::new(prev[0], prev[1], prev[2], prev[3]);
                second_rect.intersect(&prev_rect).unwrap_or(second_rect)
            }
            None => second_rect,
        };
        self.frame
            .set_active_clip(Some(second_clip.to_clip_array()));
        self.hit_clip = Some(match saved_hit_clip {
            Some(prev) => second_rect.intersect(&prev).unwrap_or(second_rect),
            None => second_rect,
        });

        self.cursor = Vec2 {
            x: second_rect.x,
            y: second_rect.y,
        };
        self.region = second_rect;
        self.spacing = saved_spacing;

        second(self);

        // --- Restore state ---
        self.frame.set_active_clip(saved_clip);
        self.hit_clip = saved_hit_clip;
        self.cursor = saved_cursor;
        self.region = saved_region;
        self.spacing = saved_spacing;

        // Advance cursor past the entire split pane.
        if horizontal {
            self.cursor.y += total_h + self.spacing;
        } else {
            self.cursor.y += total_h + self.spacing;
        }
    }
}
