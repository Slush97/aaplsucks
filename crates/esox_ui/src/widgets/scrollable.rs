//! Scrollable container widget — GPU-clipped, mouse wheel + draggable scrollbar.

use crate::layout::{Rect, Vec2};
use crate::paint;
use crate::response::Response;
use crate::state::WidgetKind;
use crate::Ui;

impl<'f> Ui<'f> {
    /// A vertically scrollable container.
    ///
    /// `visible_height` is the on-screen height of the viewport. The closure
    /// `f` draws child widgets into an unbounded vertical region; any content
    /// exceeding `visible_height` is GPU-clipped and accessible via mouse
    /// wheel or scrollbar drag.
    pub fn scrollable(
        &mut self,
        id: u64,
        visible_height: f32,
        f: impl FnOnce(&mut Self),
    ) -> Response {
        let scrollbar_w = self.theme.scrollbar_width;

        // Always reserve scrollbar width for layout stability.
        let content_width = self.region.w - scrollbar_w;
        let container = self.allocate_rect(self.region.w, visible_height);

        // Read current scroll offset (and mark as accessed).
        let scroll_offset = match self.state.scroll_offsets.get_mut(&id) {
            Some((off, age)) => { *age = 0; *off }
            None => 0.0,
        };

        // --- Save layout state ---
        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;
        let saved_active_clip = self.frame.active_clip();
        let saved_hit_clip = self.hit_clip;

        // --- Set child layout ---
        self.cursor = Vec2 {
            x: container.x,
            y: container.y - scroll_offset,
        };
        self.region = Rect::new(
            container.x,
            container.y - scroll_offset,
            content_width,
            f32::MAX,
        );

        // --- Set clipping ---
        let container_clip = Rect::new(container.x, container.y, container.w, container.h);
        let gpu_clip = match saved_active_clip {
            Some(prev) => {
                let prev_rect = Rect::new(prev[0], prev[1], prev[2], prev[3]);
                container_clip.intersect(&prev_rect).unwrap_or(container_clip)
            }
            None => container_clip,
        };
        self.frame.set_active_clip(Some(gpu_clip.to_clip_array()));
        self.hit_clip = Some(match saved_hit_clip {
            Some(prev) => container_clip.intersect(&prev).unwrap_or(container_clip),
            None => container_clip,
        });

        // --- Run child content ---
        let content_start_y = self.cursor.y;
        f(self);
        let content_height = self.cursor.y - content_start_y - self.spacing; // subtract trailing spacing

        // --- Restore layout state ---
        self.cursor = saved_cursor;
        // Advance cursor past the container (allocate_rect already did this, but
        // we overwrote cursor — restore to just past the container).
        self.cursor.y = container.y + container.h + saved_spacing;
        self.region = saved_region;
        self.spacing = saved_spacing;
        self.frame.set_active_clip(saved_active_clip);
        self.hit_clip = saved_hit_clip;

        // --- Scroll logic ---
        let max_scroll = (content_height - visible_height).max(0.0);
        let mut offset = scroll_offset;

        // Handle scrollbar drag.
        if let Some((drag_id, grab_offset)) = self.state.scrollbar_drag {
            if drag_id == id && self.state.mouse_pressed {
                let track_y = container.y;
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
                    let thumb_top = self.state.mouse.y - grab_offset - track_y;
                    offset = (thumb_top / scrollable_range) * max_scroll;
                }
            }
        }

        // Consume scroll event if mouse is inside container.
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if container.contains(sx, sy) {
                offset -= delta * self.theme.scroll_speed;
                self.state.pending_scroll = None;
            }
        }

        // Clamp and store.
        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, (offset, 0));

        // --- Draw scrollbar ---
        let hovered_container = container.contains(self.state.mouse.x, self.state.mouse.y);
        if content_height > visible_height {
            let track_x = container.x + container.w - scrollbar_w;
            let track_y = container.y;
            let track_h = visible_height;

            // Track background (subtle).
            let track_rect = Rect::new(track_x, track_y, scrollbar_w, track_h);
            paint::draw_rounded_rect(
                self.frame,
                track_rect,
                self.theme.bg_raised,
                scrollbar_w / 2.0,
            );

            // Thumb.
            let thumb_h = (visible_height / content_height * track_h)
                .max(self.theme.scrollbar_min_thumb)
                .min(track_h);
            let scrollable_range = track_h - thumb_h;
            let thumb_y = if max_scroll > 0.0 {
                track_y + (offset / max_scroll) * scrollable_range
            } else {
                track_y
            };
            let thumb_rect = Rect::new(track_x, thumb_y, scrollbar_w, thumb_h);

            // Hover animation on thumb.
            let thumb_hovered = thumb_rect.contains(self.state.mouse.x, self.state.mouse.y);
            let thumb_hover_id = id.wrapping_mul(0x517cc1b727220a95);
            let t = self.state.hover_t(thumb_hover_id, thumb_hovered || self.state.scrollbar_drag.map_or(false, |(did, _)| did == id), 120.0);
            let thumb_color = paint::lerp_color(self.theme.fg_dim, self.theme.fg_muted, t);
            paint::draw_rounded_rect(
                self.frame,
                thumb_rect,
                thumb_color,
                scrollbar_w / 2.0,
            );

            // Register scrollbar for hit testing.
            let scrollbar_id = id.wrapping_add(1);
            self.state.hit_rects.push((thumb_rect, scrollbar_id, WidgetKind::Scrollbar));

            // Handle click on thumb to initiate drag.
            if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                if !*consumed && thumb_rect.contains(cx, cy) {
                    *consumed = true;
                    self.state.scrollbar_drag = Some((id, cy - thumb_y));
                }
            }
        }

        Response {
            clicked: false,
            right_clicked: false,
            hovered: hovered_container,
            focused: false,
            changed: false,
            disabled: false,
        }
    }
}
