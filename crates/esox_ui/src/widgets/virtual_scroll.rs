//! Virtual scroll widget — only renders visible items for large lists.
//!
//! # Examples
//!
//! ```ignore
//! let mut vs = VirtualScrollState::new(items.len());
//! ui.virtual_scroll(id!("list"), &mut vs, 32.0, 400.0, |ui, i| {
//!     ui.label(&items[i]);
//! });
//! ```

use crate::layout::{Rect, Vec2};
use crate::paint;
use crate::response::Response;
use crate::state::{VirtualScrollState, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Uniform-height virtual scroll. Only calls `f` for visible items.
    pub fn virtual_scroll(
        &mut self,
        id: u64,
        state: &mut VirtualScrollState,
        item_height: f32,
        visible_height: f32,
        mut f: impl FnMut(&mut Self, usize),
    ) -> Response {
        let scrollbar_w = self.theme.scrollbar_width;
        let content_width = self.region.w - scrollbar_w;
        let container = self.allocate_rect_keyed(id, self.region.w, visible_height);

        let content_height = state.item_count as f32 * item_height;
        let max_scroll = (content_height - visible_height).max(0.0);
        let mut offset = match self.state.scroll_offsets.get_mut(&id) {
            Some((off, age)) => {
                *age = 0;
                // Pre-clamp using previous frame's max_scroll to avoid stranded offsets.
                if let Some((prev_max, _)) = self.state.prev_max_scroll.get(&id) {
                    off[0] = off[0].clamp(0.0, prev_max[0]);
                }
                off[0]
            }
            None => 0.0,
        };
        self.state.prev_max_scroll.insert(id, ([max_scroll, 0.0], 0));

        // Handle scroll_to.
        if let Some(target) = state.scroll_to.take() {
            let target_top = target as f32 * item_height;
            let target_bottom = target_top + item_height;
            if target_top < offset {
                offset = target_top;
            }
            if target_bottom > offset + visible_height {
                offset = target_bottom - visible_height;
            }
        }

        // Handle scrollbar drag.
        if let Some((drag_id, grab_offset)) = self.state.scrollbar_drag {
            if drag_id == id && self.state.mouse_pressed {
                let track_h = visible_height;
                let thumb_h = if content_height > 0.0 {
                    (visible_height / content_height * track_h)
                        .max(self.theme.scrollbar_min_thumb)
                        .min(track_h)
                } else {
                    track_h
                };
                let scrollable_range = track_h - thumb_h;
                if scrollable_range > 0.0 {
                    let thumb_top = self.state.mouse.y - grab_offset - container.y;
                    offset = (thumb_top / scrollable_range) * max_scroll;
                }
            }
        }

        // Mouse wheel — apply directly, no inertia.
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if container.contains(sx, sy) {
                offset += delta * self.theme.scroll_speed;
                self.state.pending_scroll = None;
            }
        }

        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, ([offset, 0.0], 0));

        // Compute visible range.
        let first_visible = (offset / item_height).floor() as usize;
        let last_visible = ((offset + visible_height) / item_height).ceil() as usize;
        let last_visible = last_visible.min(state.item_count);

        // Save layout state.
        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;
        let saved_active_clip = self.frame.active_clip();
        let saved_hit_clip = self.hit_clip;

        // Set clipping.
        let container_clip = Rect::new(container.x, container.y, container.w, container.h);
        let gpu_clip = match saved_active_clip {
            Some(prev) => {
                let prev_rect = Rect::new(prev[0], prev[1], prev[2], prev[3]);
                container_clip
                    .intersect(&prev_rect)
                    .unwrap_or(container_clip)
            }
            None => container_clip,
        };
        self.frame
            .set_active_clip(Some(gpu_clip.to_clip_array()));
        self.hit_clip = Some(match saved_hit_clip {
            Some(prev) => container_clip.intersect(&prev).unwrap_or(container_clip),
            None => container_clip,
        });

        // Render visible items.
        self.spacing = 0.0;
        for i in first_visible..last_visible {
            self.cursor = Vec2 {
                x: container.x,
                y: container.y + i as f32 * item_height - offset,
            };
            self.region = Rect::new(container.x, self.cursor.y, content_width, item_height);
            f(self, i);
        }

        // Restore layout state.
        self.cursor = saved_cursor;
        self.cursor.y = container.y + container.h + saved_spacing;
        self.region = saved_region;
        self.spacing = saved_spacing;
        self.frame.set_active_clip(saved_active_clip);
        self.hit_clip = saved_hit_clip;

        // Draw scrollbar.
        if content_height > visible_height {
            draw_scrollbar(
                self.frame,
                &mut self.state,
                self.theme,
                id,
                container,
                content_height,
                visible_height,
                &mut offset,
            );
            // Re-store the potentially updated offset.
            self.state.scroll_offsets.insert(id, ([offset.clamp(0.0, max_scroll), 0.0], 0));
        }

        let hovered = container.contains(self.state.mouse.x, self.state.mouse.y);
        Response {
            clicked: false,
            right_clicked: false,
            hovered,
            focused: false,
            changed: false,
            disabled: false,
        }
    }
}

/// Shared scrollbar drawing logic used by scrollable and virtual_scroll.
pub(crate) fn draw_scrollbar(
    frame: &mut esox_gfx::Frame,
    state: &mut crate::state::UiState,
    theme: &crate::theme::Theme,
    id: u64,
    container: Rect,
    content_height: f32,
    visible_height: f32,
    offset: &mut f32,
) {
    let scrollbar_w = theme.scrollbar_width;
    let max_scroll = (content_height - visible_height).max(0.0);

    let track_x = container.x + container.w - scrollbar_w;
    let track_y = container.y;
    let track_h = visible_height;

    // Track background.
    let track_rect = Rect::new(track_x, track_y, scrollbar_w, track_h);
    paint::draw_rounded_rect(frame, track_rect, theme.bg_raised, scrollbar_w / 2.0);

    // Thumb.
    let thumb_h = (visible_height / content_height * track_h)
        .max(theme.scrollbar_min_thumb)
        .min(track_h);
    let scrollable_range = track_h - thumb_h;
    let thumb_y = if max_scroll > 0.0 {
        track_y + (*offset / max_scroll) * scrollable_range
    } else {
        track_y
    };
    let thumb_rect = Rect::new(track_x, thumb_y, scrollbar_w, thumb_h);

    // Hover animation on thumb.
    let thumb_hovered = thumb_rect.contains(state.mouse.x, state.mouse.y);
    let thumb_hover_id = id.wrapping_mul(0x517cc1b727220a95);
    let t = state.hover_t(
        thumb_hover_id,
        thumb_hovered || state.scrollbar_drag.map_or(false, |(did, _)| did == id),
        theme.hover_duration_ms,
    );
    let thumb_color = paint::lerp_color(theme.fg_dim, theme.fg_muted, t);
    paint::draw_rounded_rect(frame, thumb_rect, thumb_color, scrollbar_w / 2.0);

    // Register for hit testing.
    let scrollbar_id = id.wrapping_add(1);
    state
        .hit_rects
        .push((thumb_rect, scrollbar_id, WidgetKind::Scrollbar));

    // Click on track or thumb to initiate drag.
    if let Some((cx, cy, ref mut consumed)) = state.mouse.pending_click {
        if !*consumed && track_rect.contains(cx, cy) {
            *consumed = true;
            if thumb_rect.contains(cx, cy) {
                // Grab thumb at click position.
                state.scrollbar_drag = Some((id, cy - thumb_y));
            } else {
                // Click on track: jump thumb center to click position, then drag.
                let half_thumb = thumb_h / 2.0;
                let new_thumb_top = (cy - track_y - half_thumb).clamp(0.0, scrollable_range);
                *offset = if scrollable_range > 0.0 {
                    (new_thumb_top / scrollable_range) * max_scroll
                } else {
                    0.0
                };
                state.scrollbar_drag = Some((id, half_thumb));
            }
        }
    }
}
