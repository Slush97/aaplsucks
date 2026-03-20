//! Scrollable container widget — GPU-clipped, mouse wheel + draggable scrollbar.
//!
//! # Examples
//!
//! ```ignore
//! ui.scrollable(id!("log"), 300.0, |ui| {
//!     for line in &log_lines {
//!         ui.label(line);
//!     }
//! });
//! ```

use crate::layout::{Direction, Rect, Vec2};
use crate::layout_tree::{LayoutStyle, Overflow};
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
        let container = self.allocate_rect_keyed(id, self.region.w, visible_height);

        // Read current scroll offset (and mark as accessed).
        // Pre-clamp using previous frame's max_scroll to avoid stranded offsets on content shrink.
        let scroll_offset = match self.state.scroll_offsets.get_mut(&id) {
            Some((off, age)) => {
                *age = 0;
                if let Some((prev_max, _)) = self.state.prev_max_scroll.get(&id) {
                    off[0] = off[0].clamp(0.0, prev_max[0]);
                }
                off[0]
            }
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
        self.tree_build.open_container(Some(id), LayoutStyle {
            direction: Direction::Vertical,
            gap: self.spacing,
            overflow: Overflow::Scroll,
            ..Default::default()
        });
        let content_start_y = self.cursor.y;
        f(self);
        let content_height = self.cursor.y - content_start_y - self.spacing; // subtract trailing spacing
        self.tree_build.close_container();

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
        self.state.prev_max_scroll.insert(id, ([max_scroll, 0.0], 0));
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

        // Consume scroll event — only if this scrollable can actually scroll.
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if container.contains(sx, sy) {
                let new_offset = (offset + delta * self.theme.scroll_speed).clamp(0.0, max_scroll);
                // Only consume if we're not at a scroll limit (nested coordination).
                if (new_offset - offset).abs() > 0.01 {
                    offset = new_offset;
                    self.state.pending_scroll = None;
                }
                // If at limit, leave pending_scroll for parent to consume.
            }
        }

        // Clamp and store.
        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, ([offset, 0.0], 0));

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
            let t = self.state.hover_t(thumb_hover_id, thumb_hovered || self.state.scrollbar_drag.map_or(false, |(did, _)| did == id), self.theme.hover_duration_ms);
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

            // Handle click on track or thumb to initiate drag.
            if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
                if !*consumed && track_rect.contains(cx, cy) {
                    *consumed = true;
                    if thumb_rect.contains(cx, cy) {
                        // Grab thumb at click position.
                        self.state.scrollbar_drag = Some((id, cy - thumb_y));
                    } else {
                        // Click on track: jump thumb center to click position, then drag.
                        let half_thumb = thumb_h / 2.0;
                        let new_thumb_top = (cy - track_y - half_thumb).clamp(0.0, scrollable_range);
                        offset = if scrollable_range > 0.0 {
                            (new_thumb_top / scrollable_range) * max_scroll
                        } else {
                            0.0
                        };
                        self.state.scrollbar_drag = Some((id, half_thumb));
                        // Re-store the updated offset.
                        self.state.scroll_offsets.insert(id, ([offset.clamp(0.0, max_scroll), 0.0], 0));
                    }
                }
            }
        }

        self.push_a11y_node(crate::state::A11yNode {
            id, role: crate::state::A11yRole::ScrollView, label: String::new(),
            value: None, rect: container, focused: false, disabled: false,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        Response {
            clicked: false,
            right_clicked: false,
            hovered: hovered_container,
            focused: false,
            changed: false,
            disabled: false,
        }
    }

    /// A horizontally scrollable container.
    ///
    /// `visible_width` is the on-screen width of the viewport. Content exceeding
    /// this width is GPU-clipped with a horizontal scrollbar at the bottom.
    pub fn scrollable_horizontal(
        &mut self,
        id: u64,
        visible_width: f32,
        f: impl FnOnce(&mut Self),
    ) -> Response {
        let scrollbar_w = self.theme.scrollbar_width;
        let content_height = self.region.h - scrollbar_w;
        let container = self.allocate_rect_keyed(id, visible_width, content_height + scrollbar_w);

        let scroll_offset = match self.state.scroll_offsets.get_mut(&id) {
            Some((off, age)) => {
                *age = 0;
                if let Some((prev_max, _)) = self.state.prev_max_scroll.get(&id) {
                    off[1] = off[1].clamp(0.0, prev_max[1]);
                }
                off[1]
            }
            None => 0.0,
        };

        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;
        let saved_active_clip = self.frame.active_clip();
        let saved_hit_clip = self.hit_clip;

        self.cursor = Vec2 {
            x: container.x - scroll_offset,
            y: container.y,
        };
        self.region = Rect::new(
            container.x - scroll_offset,
            container.y,
            f32::MAX,
            content_height,
        );

        let container_clip = Rect::new(container.x, container.y, visible_width, content_height);
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

        self.tree_build.open_container(Some(id), LayoutStyle {
            direction: Direction::Horizontal,
            gap: self.spacing,
            overflow: Overflow::Scroll,
            ..Default::default()
        });
        let content_start_x = self.cursor.x;
        f(self);
        let content_width = self.cursor.x - content_start_x - self.spacing;
        self.tree_build.close_container();

        self.cursor = saved_cursor;
        self.cursor.y = container.y + container.h + saved_spacing;
        self.region = saved_region;
        self.spacing = saved_spacing;
        self.frame.set_active_clip(saved_active_clip);
        self.hit_clip = saved_hit_clip;

        let max_scroll = (content_width - visible_width).max(0.0);
        self.state.prev_max_scroll.insert(id, ([0.0, max_scroll], 0));
        let mut offset = scroll_offset;

        // Horizontal scroll from wheel (shift+scroll or trackpad).
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if container.contains(sx, sy) {
                let new_offset = (offset + delta * self.theme.scroll_speed).clamp(0.0, max_scroll);
                if (new_offset - offset).abs() > 0.01 {
                    offset = new_offset;
                    self.state.pending_scroll = None;
                }
            }
        }

        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, ([0.0, offset], 0));

        // Draw horizontal scrollbar at bottom.
        let hovered_container = container.contains(self.state.mouse.x, self.state.mouse.y);
        if content_width > visible_width {
            let track_x = container.x;
            let track_y = container.y + content_height;
            let track_w = visible_width;

            let track_rect = Rect::new(track_x, track_y, track_w, scrollbar_w);
            paint::draw_rounded_rect(
                self.frame, track_rect, self.theme.bg_raised, scrollbar_w / 2.0,
            );

            let thumb_w = (visible_width / content_width * track_w)
                .max(self.theme.scrollbar_min_thumb)
                .min(track_w);
            let scrollable_range = track_w - thumb_w;
            let thumb_x = if max_scroll > 0.0 {
                track_x + (offset / max_scroll) * scrollable_range
            } else {
                track_x
            };
            let thumb_rect = Rect::new(thumb_x, track_y, thumb_w, scrollbar_w);

            let thumb_hovered = thumb_rect.contains(self.state.mouse.x, self.state.mouse.y);
            let thumb_hover_id = id.wrapping_mul(0x517cc1b727220a95);
            let t = self.state.hover_t(thumb_hover_id, thumb_hovered, self.theme.hover_duration_ms);
            let thumb_color = paint::lerp_color(self.theme.fg_dim, self.theme.fg_muted, t);
            paint::draw_rounded_rect(
                self.frame, thumb_rect, thumb_color, scrollbar_w / 2.0,
            );

            let scrollbar_id = id.wrapping_add(1);
            self.state.hit_rects.push((thumb_rect, scrollbar_id, WidgetKind::Scrollbar));
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

    /// A bidirectionally scrollable container (both axes).
    ///
    /// Shows vertical scrollbar on right, horizontal at bottom, with a dead
    /// corner where they meet.
    pub fn scrollable_2d(
        &mut self,
        id: u64,
        visible_w: f32,
        visible_h: f32,
        f: impl FnOnce(&mut Self),
    ) -> Response {
        let scrollbar_w = self.theme.scrollbar_width;
        let content_area_w = visible_w - scrollbar_w;
        let content_area_h = visible_h - scrollbar_w;
        let container = self.allocate_rect_keyed(id, visible_w, visible_h);

        let (scroll_y, scroll_x) = match self.state.scroll_offsets.get_mut(&id) {
            Some((off, age)) => {
                *age = 0;
                if let Some((prev_max, _)) = self.state.prev_max_scroll.get(&id) {
                    off[0] = off[0].clamp(0.0, prev_max[0]);
                    off[1] = off[1].clamp(0.0, prev_max[1]);
                }
                (off[0], off[1])
            }
            None => (0.0, 0.0),
        };

        let saved_cursor = self.cursor;
        let saved_region = self.region;
        let saved_spacing = self.spacing;
        let saved_active_clip = self.frame.active_clip();
        let saved_hit_clip = self.hit_clip;

        self.cursor = Vec2 {
            x: container.x - scroll_x,
            y: container.y - scroll_y,
        };
        self.region = Rect::new(
            container.x - scroll_x,
            container.y - scroll_y,
            f32::MAX,
            f32::MAX,
        );

        let content_clip = Rect::new(container.x, container.y, content_area_w, content_area_h);
        let gpu_clip = match saved_active_clip {
            Some(prev) => {
                let prev_rect = Rect::new(prev[0], prev[1], prev[2], prev[3]);
                content_clip.intersect(&prev_rect).unwrap_or(content_clip)
            }
            None => content_clip,
        };
        self.frame.set_active_clip(Some(gpu_clip.to_clip_array()));
        self.hit_clip = Some(match saved_hit_clip {
            Some(prev) => content_clip.intersect(&prev).unwrap_or(content_clip),
            None => content_clip,
        });

        self.tree_build.open_container(Some(id), LayoutStyle {
            direction: Direction::Vertical,
            gap: self.spacing,
            overflow: Overflow::Scroll,
            ..Default::default()
        });
        let content_start = self.cursor;
        f(self);
        let content_width = self.cursor.x - content_start.x - self.spacing;
        let content_height = self.cursor.y - content_start.y - self.spacing;
        self.tree_build.close_container();

        self.cursor = saved_cursor;
        self.cursor.y = container.y + container.h + saved_spacing;
        self.region = saved_region;
        self.spacing = saved_spacing;
        self.frame.set_active_clip(saved_active_clip);
        self.hit_clip = saved_hit_clip;

        let max_scroll_y = (content_height - content_area_h).max(0.0);
        let max_scroll_x = (content_width - content_area_w).max(0.0);
        self.state.prev_max_scroll.insert(id, ([max_scroll_y, max_scroll_x], 0));
        let mut off_y = scroll_y;
        let mut off_x = scroll_x;

        // Vertical scroll from wheel.
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if container.contains(sx, sy) {
                let new_off = (off_y + delta * self.theme.scroll_speed).clamp(0.0, max_scroll_y);
                if (new_off - off_y).abs() > 0.01 {
                    off_y = new_off;
                    self.state.pending_scroll = None;
                }
            }
        }

        off_y = off_y.clamp(0.0, max_scroll_y);
        off_x = off_x.clamp(0.0, max_scroll_x);
        self.state.scroll_offsets.insert(id, ([off_y, off_x], 0));

        let hovered_container = container.contains(self.state.mouse.x, self.state.mouse.y);

        // Vertical scrollbar (right side).
        if content_height > content_area_h {
            let track_x = container.x + content_area_w;
            let track_rect = Rect::new(track_x, container.y, scrollbar_w, content_area_h);
            paint::draw_rounded_rect(
                self.frame, track_rect, self.theme.bg_raised, scrollbar_w / 2.0,
            );

            let thumb_h = (content_area_h / content_height * content_area_h)
                .max(self.theme.scrollbar_min_thumb)
                .min(content_area_h);
            let scrollable_range = content_area_h - thumb_h;
            let thumb_y = if max_scroll_y > 0.0 {
                container.y + (off_y / max_scroll_y) * scrollable_range
            } else {
                container.y
            };
            let thumb_rect = Rect::new(track_x, thumb_y, scrollbar_w, thumb_h);
            let thumb_hover_id = id.wrapping_mul(0x517cc1b727220a95);
            let t = self.state.hover_t(thumb_hover_id, thumb_rect.contains(self.state.mouse.x, self.state.mouse.y), self.theme.hover_duration_ms);
            paint::draw_rounded_rect(
                self.frame, thumb_rect,
                paint::lerp_color(self.theme.fg_dim, self.theme.fg_muted, t),
                scrollbar_w / 2.0,
            );
        }

        // Horizontal scrollbar (bottom).
        if content_width > content_area_w {
            let track_y = container.y + content_area_h;
            let track_rect = Rect::new(container.x, track_y, content_area_w, scrollbar_w);
            paint::draw_rounded_rect(
                self.frame, track_rect, self.theme.bg_raised, scrollbar_w / 2.0,
            );

            let thumb_w = (content_area_w / content_width * content_area_w)
                .max(self.theme.scrollbar_min_thumb)
                .min(content_area_w);
            let scrollable_range = content_area_w - thumb_w;
            let thumb_x = if max_scroll_x > 0.0 {
                container.x + (off_x / max_scroll_x) * scrollable_range
            } else {
                container.x
            };
            let thumb_rect = Rect::new(thumb_x, track_y, thumb_w, scrollbar_w);
            let thumb_hover_id = id.wrapping_mul(0x7a2b3c4d5e6f0a1b);
            let t = self.state.hover_t(thumb_hover_id, thumb_rect.contains(self.state.mouse.x, self.state.mouse.y), self.theme.hover_duration_ms);
            paint::draw_rounded_rect(
                self.frame, thumb_rect,
                paint::lerp_color(self.theme.fg_dim, self.theme.fg_muted, t),
                scrollbar_w / 2.0,
            );
        }

        // Dead corner (where scrollbars meet).
        if content_height > content_area_h && content_width > content_area_w {
            let corner = Rect::new(
                container.x + content_area_w,
                container.y + content_area_h,
                scrollbar_w,
                scrollbar_w,
            );
            paint::draw_rounded_rect(
                self.frame, corner, self.theme.bg_raised, 0.0,
            );
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

    /// Mark the next widget as a sticky header inside a scrollable.
    ///
    /// If the header's natural position has scrolled above the viewport, it is
    /// re-drawn clamped at the viewport top. Call this inside a `scrollable()`
    /// closure before drawing the header widget.
    pub fn sticky_header(&mut self, id: u64, f: impl FnOnce(&mut Self)) {
        // Record where the header would naturally be drawn.
        let natural_y = self.cursor.y;

        // Check if we're inside a scrollable (hit_clip set).
        let viewport_top = self.hit_clip.map(|r| r.y).unwrap_or(0.0);

        if natural_y < viewport_top {
            // Header has scrolled above viewport — clamp to top.
            let saved_y = self.cursor.y;
            self.cursor.y = viewport_top;

            // Save and track position for this sticky header.
            self.state.tree_children_heights.insert(id, natural_y);

            f(self);

            // Don't advance past the clamped position — the next widget
            // should still be at its natural position.
            let drawn_height = self.cursor.y - viewport_top;
            self.cursor.y = saved_y + drawn_height;
        } else {
            // Header is in view — draw normally.
            self.state.tree_children_heights.remove(&id);
            f(self);
        }
    }
}
