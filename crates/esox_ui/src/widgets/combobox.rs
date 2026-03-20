//! Combobox widget — filterable dropdown / autocomplete.
//!
//! Combines a text input with a dropdown list. Typing filters the options
//! by case-insensitive substring match. Arrow keys navigate, Enter selects,
//! Escape closes.
//!
//! # Examples
//!
//! ```ignore
//! let options = &["Apple", "Banana", "Cherry", "Date", "Elderberry"];
//! let response = ui.combobox(id!("fruit"), options, &mut selected);
//! if response.changed {
//!     println!("Selected: {:?}", selected.map(|i| options[i]));
//! }
//! ```

use esox_gfx::{BorderRadius, ShapeBuilder};
use esox_input::{Key, NamedKey};

use crate::id::HOVER_SALT;
use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, InputState, Overlay, WidgetKind};
use crate::Ui;

/// Maximum number of visible items in the dropdown before scrolling.
const MAX_VISIBLE_ITEMS: usize = 8;

impl<'f> Ui<'f> {
    /// Draw a combobox (filterable dropdown).
    ///
    /// `options` is the full list of choices. `selected` is the index of the
    /// currently selected option (mutated on selection). Returns a `Response`
    /// with `changed = true` when the selection changes.
    pub fn combobox(
        &mut self,
        id: u64,
        options: &[&str],
        selected: &mut Option<usize>,
    ) -> Response {
        let rect = self.allocate_rect_keyed(id, self.region.w, self.theme.button_height);
        self.register_widget(id, rect, WidgetKind::Combobox);

        let mut response = self.widget_response(id, rect);
        let disabled = response.disabled;

        let is_open = matches!(
            &self.state.overlay,
            Some(Overlay::ComboboxDropdown { id: oid, .. }) if *oid == id
        );

        let display_text = selected
            .and_then(|i| options.get(i))
            .copied()
            .unwrap_or("");

        self.push_a11y_node(A11yNode {
            id,
            role: A11yRole::Combobox,
            label: display_text.to_string(),
            value: selected.map(|i| i.to_string()),
            rect,
            focused: response.focused,
            disabled,
            expanded: Some(is_open),
            selected: None,
            checked: None,
            value_range: None,
            children: Vec::new(),
        });

        if disabled {
            self.draw_combobox_disabled(rect, display_text);
            return response;
        }

        // Ensure we have an InputState for this combobox's filter text.
        self.state.combobox_inputs.entry(id).or_default();

        // Handle click — toggle dropdown.
        if response.clicked {
            if is_open {
                // Close and clear filter.
                self.state.overlay = None;
                let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                input.text.clear();
                input.cursor = 0;
                input.scroll_offset = 0.0;
            } else {
                // Open dropdown with all options visible.
                let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                input.text.clear();
                input.cursor = 0;
                input.scroll_offset = 0.0;

                let filtered: Vec<usize> = (0..options.len()).collect();
                self.state.overlay = Some(Overlay::ComboboxDropdown {
                    id,
                    anchor: rect,
                    all_choices: options.iter().map(|s| s.to_string()).collect(),
                    filtered_indices: filtered,
                    highlighted: selected.map(|s| s.min(options.len().saturating_sub(1))),
                    scroll_offset: 0.0,
                });
            }
        }

        // Handle keyboard when focused.
        if response.focused && is_open {
            let keys: Vec<_> = self.state.keys.clone();
            let mut text_changed = false;
            let mut close_dropdown = false;
            let mut select_highlighted = false;

            // First pass: navigation keys (arrow, enter, escape).
            for (event, modifiers) in &keys {
                if !event.pressed {
                    continue;
                }
                let ctrl = modifiers.ctrl();

                match &event.key {
                    Key::Named(NamedKey::Escape) => {
                        close_dropdown = true;
                        break;
                    }
                    Key::Named(NamedKey::Enter) => {
                        select_highlighted = true;
                        break;
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        if let Some(Overlay::ComboboxDropdown {
                            ref filtered_indices,
                            ref mut highlighted,
                            ..
                        }) = self.state.overlay
                        {
                            if !filtered_indices.is_empty() {
                                let cur = highlighted.unwrap_or(0);
                                *highlighted = Some(if cur == 0 {
                                    filtered_indices.len() - 1
                                } else {
                                    cur - 1
                                });
                            }
                        }
                        continue;
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if let Some(Overlay::ComboboxDropdown {
                            ref filtered_indices,
                            ref mut highlighted,
                            ..
                        }) = self.state.overlay
                        {
                            if !filtered_indices.is_empty() {
                                let cur = highlighted
                                    .unwrap_or(filtered_indices.len().saturating_sub(1));
                                *highlighted =
                                    Some((cur + 1) % filtered_indices.len());
                            }
                        }
                        continue;
                    }
                    // Tab closes and moves focus.
                    Key::Named(NamedKey::Tab) => {
                        close_dropdown = true;
                        break;
                    }
                    _ => {
                        // Text editing keys — process through InputState.
                        let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                        let shift = modifiers.shift();

                        // Handle clipboard shortcuts.
                        if ctrl {
                            if let Key::Character(ch) = &event.key {
                                match ch.as_str() {
                                    "v" => {
                                        if let Some(clip) = &self.state.clipboard {
                                            if let Some(txt) = clip.read_text() {
                                                input.save_undo();
                                                input.insert_str(&txt);
                                                text_changed = true;
                                                self.state.reset_blink();
                                            }
                                        }
                                        continue;
                                    }
                                    "a" => {
                                        input.select_all();
                                        continue;
                                    }
                                    "z" => {
                                        if shift {
                                            input.redo();
                                        } else {
                                            input.undo();
                                        }
                                        text_changed = true;
                                        continue;
                                    }
                                    _ => continue,
                                }
                            }
                        }

                        let changed = process_combobox_text_key(input, &event.key, ctrl, shift);
                        if changed {
                            text_changed = true;
                            self.state.reset_blink();
                        }
                    }
                }
            }

            // If text changed, re-filter the options.
            if text_changed {
                let input = self.state.combobox_inputs.get(&id).unwrap();
                let filter = input.text.to_lowercase();

                if let Some(Overlay::ComboboxDropdown {
                    ref all_choices,
                    ref mut filtered_indices,
                    ref mut highlighted,
                    ref mut scroll_offset,
                    ..
                }) = self.state.overlay
                {
                    *filtered_indices = if filter.is_empty() {
                        (0..all_choices.len()).collect()
                    } else {
                        all_choices
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| s.to_lowercase().contains(&filter))
                            .map(|(i, _)| i)
                            .collect()
                    };
                    // Reset highlight to first match.
                    *highlighted = if filtered_indices.is_empty() {
                        None
                    } else {
                        Some(0)
                    };
                    *scroll_offset = 0.0;
                }

                // Update scroll for the input field.
                let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                let arrow_w = self.text.measure_text("\u{25BE}", self.theme.font_size)
                    + self.theme.input_padding;
                let inner_w =
                    rect.w - self.theme.input_padding * 2.0 - arrow_w;
                let cursor_x = self.text.measure_text(
                    &input.text[..input.cursor],
                    self.theme.font_size,
                );
                if cursor_x - input.scroll_offset > inner_w {
                    input.scroll_offset = cursor_x - inner_w;
                }
                if cursor_x < input.scroll_offset {
                    input.scroll_offset = cursor_x;
                }
                if input.scroll_offset < 0.0 {
                    input.scroll_offset = 0.0;
                }
            }

            if select_highlighted {
                if let Some(Overlay::ComboboxDropdown {
                    ref filtered_indices,
                    highlighted: Some(hi),
                    ..
                }) = self.state.overlay
                {
                    if let Some(&original_idx) = filtered_indices.get(hi) {
                        *selected = Some(original_idx);
                        response.changed = true;
                    }
                }
                self.state.overlay = None;
                let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                input.text.clear();
                input.cursor = 0;
                input.scroll_offset = 0.0;
            }

            if close_dropdown {
                self.state.overlay = None;
                let input = self.state.combobox_inputs.get_mut(&id).unwrap();
                input.text.clear();
                input.cursor = 0;
                input.scroll_offset = 0.0;
            }
        }

        // Handle scroll within the combobox dropdown.
        if is_open {
            if let Some((sx, sy, delta)) = self.state.pending_scroll {
                if let Some(Overlay::ComboboxDropdown {
                    ref anchor,
                    ref filtered_indices,
                    ref mut scroll_offset,
                    ..
                }) = self.state.overlay
                {
                    let dd_y = anchor.y + anchor.h + self.theme.dropdown_gap;
                    let visible_count = filtered_indices.len().min(MAX_VISIBLE_ITEMS);
                    let dd_h = visible_count as f32 * self.theme.item_height;

                    if sx >= anchor.x
                        && sx < anchor.x + anchor.w
                        && sy >= dd_y
                        && sy < dd_y + dd_h
                    {
                        let total_h = filtered_indices.len() as f32 * self.theme.item_height;
                        let max_scroll = (total_h - dd_h).max(0.0);
                        *scroll_offset =
                            (*scroll_offset - delta * self.theme.scroll_speed).clamp(0.0, max_scroll);
                        // Consume scroll so it doesn't propagate.
                        self.state.pending_scroll = None;
                    }
                }
            }
        }

        // ── Draw ──

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
        let bg = {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            paint::lerp_color(self.theme.bg_input, self.theme.bg_raised, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        // Border.
        let border_color = if response.focused || is_open {
            self.theme.accent
        } else {
            self.theme.border
        };
        paint::draw_border(self.frame, rect, border_color);

        let text_y = rect.y + (rect.h - self.theme.font_size) / 2.0;

        // Chevron (dropdown arrow).
        let chevron = "\u{25BE}";
        let chevron_color = if response.focused || is_open {
            self.theme.accent
        } else {
            self.theme.fg_dim
        };
        let chevron_w = self.text.measure_text(chevron, self.theme.font_size);
        self.text.draw_ui_text(
            chevron,
            rect.x + rect.w - self.theme.input_padding - chevron_w,
            text_y,
            chevron_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        let text_x = rect.x + self.theme.input_padding;
        let arrow_area = chevron_w + self.theme.input_padding;
        let inner_w = rect.w - self.theme.input_padding * 2.0 - arrow_area;

        if is_open {
            // Show filter text with cursor.
            let input = self.state.combobox_inputs.get(&id).unwrap();
            let scroll = input.scroll_offset;

            if input.text.is_empty() {
                // Placeholder.
                self.text.draw_ui_text(
                    "Type to filter\u{2026}",
                    text_x,
                    text_y,
                    self.theme.fg_dim,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            } else {
                // Filter text.
                self.text.draw_ui_text(
                    &input.text,
                    text_x - scroll,
                    text_y,
                    self.theme.fg,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }

            // Cursor.
            if response.focused && self.state.cursor_blink {
                let input = self.state.combobox_inputs.get(&id).unwrap();
                let cursor_x_in_text = self
                    .text
                    .measure_text(&input.text[..input.cursor], self.theme.font_size);
                let cx = text_x + cursor_x_in_text - input.scroll_offset;
                if cx >= text_x - 1.0 && cx <= text_x + inner_w + 1.0 {
                    self.frame.push(
                        ShapeBuilder::rect(
                            cx,
                            rect.y + self.theme.label_pad_y + 2.0,
                            self.theme.cursor_width,
                            rect.h - self.theme.label_pad_y * 2.0 - 4.0,
                        )
                        .color(self.theme.fg)
                        .build(),
                    );
                }
            }
        } else {
            // Show selected value or placeholder.
            if display_text.is_empty() {
                self.text.draw_ui_text(
                    "Select\u{2026}",
                    text_x,
                    text_y,
                    self.theme.fg_dim,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            } else {
                self.text.draw_ui_text(
                    display_text,
                    text_x,
                    text_y,
                    self.theme.fg,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }
        }

        response
    }

    /// Draw the combobox in disabled state.
    fn draw_combobox_disabled(&mut self, rect: Rect, display_text: &str) {
        paint::draw_rounded_rect(
            self.frame,
            rect,
            self.theme.disabled_bg,
            self.theme.corner_radius,
        );
        paint::draw_dashed_border(
            self.frame,
            rect,
            self.theme.disabled_border,
            self.theme.disabled_dash_len,
            self.theme.disabled_dash_gap,
            self.theme.disabled_dash_thickness,
        );

        let text_y = rect.y + (rect.h - self.theme.font_size) / 2.0;
        let text_x = rect.x + self.theme.input_padding;

        // Chevron.
        let chevron = "\u{25BE}";
        let chevron_w = self.text.measure_text(chevron, self.theme.font_size);
        self.text.draw_ui_text(
            chevron,
            rect.x + rect.w - self.theme.input_padding - chevron_w,
            text_y,
            self.theme.disabled_fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Text.
        if display_text.is_empty() {
            self.text.draw_ui_text(
                "Select\u{2026}",
                text_x,
                text_y,
                self.theme.disabled_fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        } else {
            self.text.draw_ui_text(
                display_text,
                text_x,
                text_y,
                self.theme.disabled_fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }
    }

    /// Draw the combobox dropdown overlay. Called from `draw_overlay()` in select.rs.
    pub(crate) fn draw_combobox_overlay(&mut self) -> Option<(u64, usize)> {
        let overlay = self.state.overlay.as_ref()?;
        let (id, anchor, all_choices, filtered_indices, highlighted, scroll_offset) = match overlay
        {
            Overlay::ComboboxDropdown {
                id,
                anchor,
                all_choices,
                filtered_indices,
                highlighted,
                scroll_offset,
            } => (
                *id,
                *anchor,
                all_choices.clone(),
                filtered_indices.clone(),
                *highlighted,
                *scroll_offset,
            ),
            _ => return None,
        };

        if filtered_indices.is_empty() {
            // Draw "no matches" indicator.
            let item_h = self.theme.item_height;
            let dd_x = anchor.x;
            let dd_y = anchor.y + anchor.h + self.theme.dropdown_gap;
            let dd_w = anchor.w;
            let dd_h = item_h;

            // Shadow.
            self.frame.push(
                ShapeBuilder::rect(dd_x - 1.0, dd_y - 1.0, dd_w + 2.0, dd_h + 2.0)
                    .color(self.theme.shadow)
                    .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                    .build(),
            );
            // Background.
            self.frame.push(
                ShapeBuilder::rect(dd_x, dd_y, dd_w, dd_h)
                    .color(self.theme.bg_surface)
                    .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                    .build(),
            );
            paint::draw_border(
                self.frame,
                Rect::new(dd_x, dd_y, dd_w, dd_h),
                self.theme.accent,
            );
            self.text.draw_ui_text(
                "No matches",
                dd_x + self.theme.input_padding,
                dd_y + (item_h - self.theme.font_size) / 2.0,
                self.theme.fg_dim,
                self.frame,
                self.gpu,
                self.resources,
            );
            return None;
        }

        let item_h = self.theme.item_height;
        let visible_count = filtered_indices.len().min(MAX_VISIBLE_ITEMS);
        let dd_x = anchor.x;
        let dd_y = anchor.y + anchor.h + self.theme.dropdown_gap;
        let dd_w = anchor.w;
        let dd_h = visible_count as f32 * item_h;

        // Handle click within dropdown.
        let mut selection_result = None;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if cx >= dd_x && cx < dd_x + dd_w && cy >= dd_y && cy < dd_y + dd_h {
                let idx = ((cy - dd_y + scroll_offset) / item_h) as usize;
                if idx < filtered_indices.len() {
                    let original_idx = filtered_indices[idx];
                    selection_result = Some((id, original_idx));
                }
                *consumed = true;
            }
            // Clicks outside are handled by the widget_response system
            // (it will defocus the combobox, and the next frame will close).
        }

        // Shadow.
        self.frame.push(
            ShapeBuilder::rect(dd_x - 1.0, dd_y - 1.0, dd_w + 2.0, dd_h + 2.0)
                .color(self.theme.shadow)
                .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                .build(),
        );

        // Background.
        self.frame.push(
            ShapeBuilder::rect(dd_x, dd_y, dd_w, dd_h)
                .color(self.theme.bg_surface)
                .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                .build(),
        );

        // Border.
        paint::draw_border(
            self.frame,
            Rect::new(dd_x, dd_y, dd_w, dd_h),
            self.theme.accent,
        );

        // Set clip rect for scrollable content.
        let saved_clip = self.frame.active_clip();
        self.frame
            .set_active_clip(Some([dd_x, dd_y, dd_w, dd_h]));

        // Get the filter text for highlight matching.
        let filter_text = self
            .state
            .combobox_inputs
            .get(&id)
            .map(|inp| inp.text.to_lowercase())
            .unwrap_or_default();

        // Ensure highlighted item is visible (scroll into view).
        if let Some(hi) = highlighted {
            if let Some(Overlay::ComboboxDropdown {
                ref mut scroll_offset,
                ..
            }) = self.state.overlay
            {
                let hi_top = hi as f32 * item_h;
                let hi_bottom = hi_top + item_h;
                if hi_top < *scroll_offset {
                    *scroll_offset = hi_top;
                } else if hi_bottom > *scroll_offset + dd_h {
                    *scroll_offset = hi_bottom - dd_h;
                }
            }
        }

        // Re-read scroll_offset after potential modification.
        let scroll_offset = match &self.state.overlay {
            Some(Overlay::ComboboxDropdown { scroll_offset, .. }) => *scroll_offset,
            _ => 0.0,
        };
        let first_visible = (scroll_offset / item_h) as usize;
        let last_visible = (first_visible + visible_count + 2).min(filtered_indices.len());

        // Draw visible items. We need `vi` as a positional index for layout
        // calculations and highlight checks, not just for indexing filtered_indices.
        #[allow(clippy::needless_range_loop)]
        for vi in first_visible..last_visible {
            let original_idx = filtered_indices[vi];
            let choice = &all_choices[original_idx];
            let iy = dd_y + vi as f32 * item_h - scroll_offset;

            // Skip items outside visible area.
            if iy + item_h < dd_y || iy > dd_y + dd_h {
                continue;
            }

            let is_highlighted = highlighted == Some(vi);

            if is_highlighted {
                self.frame.push(
                    ShapeBuilder::rect(dd_x + 1.0, iy, dd_w - 2.0, item_h)
                        .color(self.theme.bg_raised)
                        .build(),
                );
            }

            let text_item_y = iy + (item_h - self.theme.font_size) / 2.0;
            let text_item_x = dd_x + self.theme.input_padding;

            if !filter_text.is_empty() {
                // Draw with match highlighting: find the match substring and
                // draw it in accent color.
                let choice_lower = choice.to_lowercase();
                if let Some(match_start) = choice_lower.find(&filter_text) {
                    let match_end = match_start + filter_text.len();

                    // Before match.
                    let before = &choice[..match_start];
                    let before_w = self.text.measure_text(before, self.theme.font_size);
                    if !before.is_empty() {
                        self.text.draw_ui_text(
                            before,
                            text_item_x,
                            text_item_y,
                            self.theme.fg,
                            self.frame,
                            self.gpu,
                            self.resources,
                        );
                    }

                    // Match portion (accent color).
                    let matched = &choice[match_start..match_end];
                    self.text.draw_ui_text(
                        matched,
                        text_item_x + before_w,
                        text_item_y,
                        self.theme.accent,
                        self.frame,
                        self.gpu,
                        self.resources,
                    );

                    // After match.
                    let after = &choice[match_end..];
                    if !after.is_empty() {
                        let match_w =
                            self.text.measure_text(matched, self.theme.font_size);
                        self.text.draw_ui_text(
                            after,
                            text_item_x + before_w + match_w,
                            text_item_y,
                            self.theme.fg,
                            self.frame,
                            self.gpu,
                            self.resources,
                        );
                    }
                } else {
                    // Shouldn't happen (filtered), but fallback.
                    self.text.draw_ui_text(
                        choice,
                        text_item_x,
                        text_item_y,
                        self.theme.fg,
                        self.frame,
                        self.gpu,
                        self.resources,
                    );
                }
            } else {
                // No filter — plain text.
                self.text.draw_ui_text(
                    choice,
                    text_item_x,
                    text_item_y,
                    self.theme.fg,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }
        }

        // Restore clip.
        self.frame.set_active_clip(saved_clip);

        // Draw scrollbar if needed.
        if filtered_indices.len() > MAX_VISIBLE_ITEMS {
            let total_h = filtered_indices.len() as f32 * item_h;
            let thumb_ratio = dd_h / total_h;
            let thumb_h = (dd_h * thumb_ratio).max(self.theme.scrollbar_min_thumb);
            let scroll_range = total_h - dd_h;
            let thumb_y = if scroll_range > 0.0 {
                dd_y + (scroll_offset / scroll_range) * (dd_h - thumb_h)
            } else {
                dd_y
            };
            let sb_x = dd_x + dd_w - self.theme.scrollbar_width - 1.0;

            self.frame.push(
                ShapeBuilder::rect(sb_x, thumb_y, self.theme.scrollbar_width, thumb_h)
                    .color(self.theme.fg_dim)
                    .border_radius(BorderRadius::uniform(self.theme.scrollbar_width / 2.0))
                    .build(),
            );
        }

        // If a selection was made, close the overlay and clear the filter.
        if selection_result.is_some() {
            self.state.overlay = None;
            if let Some(input) = self.state.combobox_inputs.get_mut(&id) {
                input.text.clear();
                input.cursor = 0;
                input.scroll_offset = 0.0;
            }
        }

        selection_result
    }
}

/// Process a key event for the combobox filter input. Returns true if text changed.
fn process_combobox_text_key(input: &mut InputState, key: &Key, ctrl: bool, shift: bool) -> bool {
    match key {
        Key::Named(NamedKey::Backspace) => {
            input.save_undo();
            input.delete_back();
            true
        }
        Key::Named(NamedKey::Delete) => {
            input.save_undo();
            input.delete_forward();
            true
        }
        Key::Named(NamedKey::ArrowLeft) => {
            if shift {
                input.move_left_extend();
            } else {
                input.move_left();
            }
            false // cursor move, not text change
        }
        Key::Named(NamedKey::ArrowRight) => {
            if shift {
                input.move_right_extend();
            } else {
                input.move_right();
            }
            false
        }
        Key::Named(NamedKey::Home) => {
            if shift {
                input.home_extend();
            } else {
                input.home();
            }
            false
        }
        Key::Named(NamedKey::End) => {
            if shift {
                input.end_extend();
            } else {
                input.end();
            }
            false
        }
        Key::Named(NamedKey::Space) => {
            input.save_undo();
            input.insert_char(' ');
            true
        }
        Key::Character(_) if ctrl => false,
        Key::Character(ch) => {
            input.save_undo();
            for c in ch.chars() {
                if !c.is_control() {
                    input.insert_char(c);
                }
            }
            true
        }
        _ => false,
    }
}
