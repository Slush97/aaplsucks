//! Text area widget — multi-line text input with vertical scroll.

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{InputState, WidgetKind};
use crate::Ui;

// ── Line helpers ──

/// Count newlines before `offset` to get the line number (0-based).
fn line_of_offset(text: &str, offset: usize) -> usize {
    text[..offset].bytes().filter(|&b| b == b'\n').count()
}

/// Byte offset of the start of the line containing `offset`.
fn line_start(text: &str, offset: usize) -> usize {
    text[..offset]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or(0)
}

/// Byte offset of the end of the line containing `offset` (before the \n or text end).
fn line_end(text: &str, offset: usize) -> usize {
    text[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(text.len())
}

/// Byte offset where line `n` starts (0-based). Returns text.len() if n >= line_count.
fn line_start_of_nth(text: &str, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut count = 0;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            count += 1;
            if count == n {
                return i + 1;
            }
        }
    }
    text.len()
}

/// Number of lines (1 + count of \n).
fn line_count(text: &str) -> usize {
    1 + text.bytes().filter(|&b| b == b'\n').count()
}

impl<'f> Ui<'f> {
    /// Draw a multi-line text area. `rows` sets the visible height in lines.
    pub fn text_area(
        &mut self,
        id: u64,
        input: &mut InputState,
        rows: usize,
        placeholder: &str,
    ) -> Response {
        let font_size = self.theme.font_size;
        let lh = self.text.line_height(font_size);
        let pad = self.theme.input_padding;
        let visible_height = rows as f32 * lh + pad * 2.0;
        let rect = self.allocate_rect_keyed(id, self.region.w, visible_height);
        self.register_widget(id, rect, WidgetKind::TextInput);

        let mut response = self.widget_response(id, rect);
        let disabled = response.disabled;

        self.push_a11y_node(crate::state::A11yNode {
            id, role: crate::state::A11yRole::TextArea, label: placeholder.to_string(),
            value: Some(input.text.clone()), rect, focused: response.focused, disabled: response.disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        if disabled {
            // ── Disabled draw ──
            paint::draw_rounded_rect(self.frame, rect, self.theme.disabled_bg, self.theme.corner_radius);
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
            let text_x = rect.x + pad;
            let text_y = rect.y + pad;
            if input.text.is_empty() {
                self.text.draw_ui_text(placeholder, text_x, text_y, self.theme.disabled_fg, self.frame, self.gpu, self.resources);
            } else {
                // Draw visible lines.
                let total_lines = line_count(&input.text);
                let visible_lines = rows.min(total_lines);
                for i in 0..visible_lines {
                    let ls = line_start_of_nth(&input.text, i);
                    let le = line_end(&input.text, ls);
                    let line = &input.text[ls..le];
                    self.text.draw_ui_text(line, text_x, text_y + i as f32 * lh, self.theme.disabled_fg, self.frame, self.gpu, self.resources);
                }
            }
            return response;
        }

        // ── Click — place cursor ──
        if response.clicked {
            let scroll_y = match self.state.scroll_offsets.get_mut(&id) { Some((off, age)) => { *age = 0; off[0] } None => 0.0 };
            let click_x = self.state.mouse.x;
            let click_y = self.state.mouse.y;
            input.cursor = xy_to_cursor(input, &mut self.text, rect, click_x, click_y, scroll_y, font_size, pad, lh);
            input.selection = None;
        }

        // Consume IME committed text.
        if response.focused {
            if let Some(committed) = self.state.ime.committed.take() {
                input.save_undo();
                input.insert_str(&committed);
                response.changed = true;
                self.state.reset_blink();
            }
        }

        // ── Key processing ──
        if response.focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, modifiers) in &keys {
                if !event.pressed {
                    continue;
                }
                let ctrl = modifiers.ctrl();
                let shift = modifiers.shift();
                // Handle clipboard shortcuts.
                if ctrl {
                    if let Key::Character(ch) = &event.key {
                        match ch.as_str() {
                            "c" => {
                                if let (Some(sel_text), Some(clip)) = (input.selected_text(), &self.state.clipboard) {
                                    clip.write_text(sel_text);
                                }
                                continue;
                            }
                            "x" => {
                                if let Some(clip) = &self.state.clipboard {
                                    if let Some(sel_text) = input.selected_text() {
                                        clip.write_text(sel_text);
                                    }
                                    input.save_undo();
                                    input.delete_selection();
                                    response.changed = true;
                                    self.state.reset_blink();
                                }
                                continue;
                            }
                            "v" => {
                                if let Some(clip) = &self.state.clipboard {
                                    if let Some(text) = clip.read_text() {
                                        input.save_undo();
                                        input.insert_str(&text);
                                        response.changed = true;
                                        self.state.reset_blink();
                                    }
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
                let changed = process_text_area_key(input, &event.key, ctrl, shift, &mut self.text, font_size);
                if changed {
                    response.changed = true;
                    self.state.reset_blink();
                }
            }
        }

        // ── Scroll ──
        let scroll_y = match self.state.scroll_offsets.get_mut(&id) { Some((off, age)) => { *age = 0; off[0] } None => 0.0 };
        let content_height = line_count(&input.text) as f32 * lh;
        let inner_h = visible_height - pad * 2.0;
        let max_scroll = (content_height - inner_h).max(0.0);
        let mut offset = scroll_y;

        // Mouse wheel.
        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if rect.contains(sx, sy) {
                offset -= delta * self.theme.scroll_speed;
                self.state.pending_scroll = None;
            }
        }

        // Keep cursor visible.
        if response.focused {
            let cursor_line = line_of_offset(&input.text, input.cursor) as f32;
            let cursor_top = cursor_line * lh;
            let cursor_bottom = cursor_top + lh;
            if cursor_top < offset {
                offset = cursor_top;
            }
            if cursor_bottom > offset + inner_h {
                offset = cursor_bottom - inner_h;
            }
        }

        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, ([offset, 0.0], 0));

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
        paint::draw_rounded_rect(self.frame, rect, self.theme.bg_input, self.theme.corner_radius);

        // Border.
        let border_color = if response.focused { self.theme.accent } else { self.theme.border };
        paint::draw_border(self.frame, rect, border_color);

        // Set GPU clip.
        let saved_clip = self.frame.active_clip();
        self.frame.set_active_clip(Some(rect.to_clip_array()));

        let text_x = rect.x + pad;
        let text_y = rect.y + pad;

        if input.text.is_empty() && !response.focused {
            // Placeholder.
            self.text.draw_ui_text(placeholder, text_x, text_y, self.theme.fg_dim, self.frame, self.gpu, self.resources);
            self.frame.set_active_clip(saved_clip);
            return response;
        }

        // Draw visible lines.
        let first_visible_line = (offset / lh).floor() as usize;
        let last_visible_line = ((offset + inner_h) / lh).ceil() as usize;
        let total_lines = line_count(&input.text);

        for i in first_visible_line..last_visible_line.min(total_lines) {
            let ls = line_start_of_nth(&input.text, i);
            let le = line_end(&input.text, ls);
            let line = &input.text[ls..le];
            let ly = text_y + i as f32 * lh - offset;

            // Selection highlight for this line.
            if let Some((sel_start, sel_end)) = input.selection {
                let line_sel_start = sel_start.max(ls);
                let line_sel_end = sel_end.min(le);
                if line_sel_start < line_sel_end {
                    let sel_x0 = self.text.measure_text(&input.text[ls..line_sel_start], font_size);
                    let sel_x1 = self.text.measure_text(&input.text[ls..line_sel_end], font_size);
                    self.frame.push(
                        ShapeBuilder::rect(text_x + sel_x0, ly, sel_x1 - sel_x0, lh)
                            .color(self.theme.accent_dim)
                            .build(),
                    );
                }
            }

            // Text.
            self.text.draw_ui_text(line, text_x, ly, self.theme.fg, self.frame, self.gpu, self.resources);
        }

        // IME preedit + Cursor.
        if response.focused {
            let cursor_line = line_of_offset(&input.text, input.cursor);
            let cursor_ls = line_start(&input.text, input.cursor);
            let cursor_x_in_line = self.text.measure_text(&input.text[cursor_ls..input.cursor], font_size);
            let cy = text_y + cursor_line as f32 * lh - offset;
            let mut cx = text_x + cursor_x_in_line;

            // IME preedit rendering.
            if !self.state.ime.preedit.is_empty() {
                let preedit_w = self.text.measure_text(&self.state.ime.preedit, font_size);
                // Underline.
                self.frame.push(
                    ShapeBuilder::rect(cx, cy + lh - 1.0, preedit_w, 1.0)
                        .color(self.theme.fg_dim)
                        .build(),
                );
                // Preedit text.
                self.text.draw_ui_text(
                    &self.state.ime.preedit, cx, cy, self.theme.fg_dim,
                    self.frame, self.gpu, self.resources,
                );
                cx += preedit_w;
            }

            if self.state.cursor_blink {
                self.frame.push(
                    ShapeBuilder::rect(cx, cy, self.theme.cursor_width, lh)
                        .color(self.theme.fg)
                        .build(),
                );
            }
        }

        // Restore clip.
        self.frame.set_active_clip(saved_clip);

        response
    }
}

// ── Visual line for word wrap ──

struct VisualLine {
    text_start: usize,
    text_end: usize,
    #[allow(dead_code)]
    logical_line: usize,
}

/// Build visual lines from text with soft word wrap.
fn build_visual_lines(
    text: &str,
    text_renderer: &mut crate::text::TextRenderer,
    font_size: f32,
    content_width: f32,
) -> Vec<VisualLine> {
    let mut visual_lines = Vec::new();

    if text.is_empty() {
        visual_lines.push(VisualLine {
            text_start: 0,
            text_end: 0,
            logical_line: 0,
        });
        return visual_lines;
    }

    let mut logical_line = 0;
    let mut pos = 0;

    for line in text.split('\n') {
        let line_start = pos;
        let line_end = pos + line.len();

        if line.is_empty() {
            visual_lines.push(VisualLine {
                text_start: line_start,
                text_end: line_start,
                logical_line,
            });
        } else {
            let wraps = text_renderer.wrap_lines(line, font_size, content_width);
            for (ws, we) in wraps {
                visual_lines.push(VisualLine {
                    text_start: line_start + ws,
                    text_end: line_start + we,
                    logical_line,
                });
            }
        }

        pos = line_end + 1; // skip '\n'
        logical_line += 1;
    }

    visual_lines
}

/// Find which visual line contains a byte offset.
fn visual_line_of_offset(visual_lines: &[VisualLine], offset: usize) -> usize {
    for (i, vl) in visual_lines.iter().enumerate() {
        if offset <= vl.text_end {
            if offset >= vl.text_start {
                return i;
            }
        }
    }
    visual_lines.len().saturating_sub(1)
}

impl<'f> Ui<'f> {
    /// Multi-line text area with soft word wrap at widget width.
    pub fn text_area_wrapped(
        &mut self,
        id: u64,
        input: &mut InputState,
        rows: usize,
        placeholder: &str,
    ) -> Response {
        let font_size = self.theme.font_size;
        let lh = self.text.line_height(font_size);
        let pad = self.theme.input_padding;
        let visible_height = rows as f32 * lh + pad * 2.0;
        let content_width = self.region.w - pad * 2.0;
        let rect = self.allocate_rect_keyed(id, self.region.w, visible_height);
        self.register_widget(id, rect, WidgetKind::TextInput);

        let mut response = self.widget_response(id, rect);

        self.push_a11y_node(crate::state::A11yNode {
            id, role: crate::state::A11yRole::TextArea, label: placeholder.to_string(),
            value: Some(input.text.clone()), rect, focused: response.focused, disabled: response.disabled,
            expanded: None, selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        if response.disabled {
            paint::draw_rounded_rect(self.frame, rect, self.theme.disabled_bg, self.theme.corner_radius);
            paint::draw_dashed_border(self.frame, rect, self.theme.disabled_border, self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness);
            let text_x = rect.x + pad;
            let text_y = rect.y + pad;
            if input.text.is_empty() {
                self.text.draw_ui_text(placeholder, text_x, text_y, self.theme.disabled_fg, self.frame, self.gpu, self.resources);
            } else {
                let visual_lines = build_visual_lines(&input.text, &mut self.text, font_size, content_width);
                for (i, vl) in visual_lines.iter().take(rows).enumerate() {
                    let line = &input.text[vl.text_start..vl.text_end];
                    self.text.draw_ui_text(line, text_x, text_y + i as f32 * lh, self.theme.disabled_fg, self.frame, self.gpu, self.resources);
                }
            }
            return response;
        }

        // Build visual lines.
        let visual_lines = build_visual_lines(&input.text, &mut self.text, font_size, content_width);
        let total_visual = visual_lines.len();

        // Click — place cursor.
        if response.clicked {
            let scroll_y = match self.state.scroll_offsets.get_mut(&id) { Some((off, age)) => { *age = 0; off[0] } None => 0.0 };
            let click_x = self.state.mouse.x;
            let click_y = self.state.mouse.y;
            let text_y = rect.y + pad;
            let clicked_vl = ((click_y - text_y + scroll_y) / lh).floor().max(0.0) as usize;
            let target_vl = clicked_vl.min(total_visual.saturating_sub(1));
            let vl = &visual_lines[target_vl];
            let rel_x = click_x - (rect.x + pad);
            input.cursor = find_offset_for_x(
                &input.text, vl.text_start, vl.text_end, rel_x, &mut self.text, font_size,
            );
            input.selection = None;
        }

        // Consume IME committed text.
        if response.focused {
            if let Some(committed) = self.state.ime.committed.take() {
                input.save_undo();
                input.insert_str(&committed);
                response.changed = true;
                self.state.reset_blink();
            }
        }

        // Key processing.
        if response.focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, modifiers) in &keys {
                if !event.pressed {
                    continue;
                }
                let ctrl = modifiers.ctrl();
                let shift = modifiers.shift();
                // Handle clipboard shortcuts.
                if ctrl {
                    if let Key::Character(ch) = &event.key {
                        match ch.as_str() {
                            "c" => {
                                if let (Some(sel_text), Some(clip)) = (input.selected_text(), &self.state.clipboard) {
                                    clip.write_text(sel_text);
                                }
                                continue;
                            }
                            "x" => {
                                if let Some(clip) = &self.state.clipboard {
                                    if let Some(sel_text) = input.selected_text() {
                                        clip.write_text(sel_text);
                                    }
                                    input.save_undo();
                                    input.delete_selection();
                                    response.changed = true;
                                    self.state.reset_blink();
                                }
                                continue;
                            }
                            "v" => {
                                if let Some(clip) = &self.state.clipboard {
                                    if let Some(text) = clip.read_text() {
                                        input.save_undo();
                                        input.insert_str(&text);
                                        response.changed = true;
                                        self.state.reset_blink();
                                    }
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
                // Handle Up/Down/Home/End specially for visual lines.
                match &event.key {
                    Key::Named(NamedKey::ArrowUp) => {
                        let cur_vl = visual_line_of_offset(&visual_lines, input.cursor);
                        let new_pos = if cur_vl == 0 {
                            0
                        } else {
                            let vl = &visual_lines[cur_vl];
                            let visual_x = self.text.measure_text(&input.text[vl.text_start..input.cursor], font_size);
                            let target = &visual_lines[cur_vl - 1];
                            find_offset_for_x(&input.text, target.text_start, target.text_end, visual_x, &mut self.text, font_size)
                        };
                        if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
                        self.state.reset_blink();
                        continue;
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        let cur_vl = visual_line_of_offset(&visual_lines, input.cursor);
                        let new_pos = if cur_vl >= total_visual - 1 {
                            input.text.len()
                        } else {
                            let vl = &visual_lines[cur_vl];
                            let visual_x = self.text.measure_text(&input.text[vl.text_start..input.cursor], font_size);
                            let target = &visual_lines[cur_vl + 1];
                            find_offset_for_x(&input.text, target.text_start, target.text_end, visual_x, &mut self.text, font_size)
                        };
                        if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
                        self.state.reset_blink();
                        continue;
                    }
                    Key::Named(NamedKey::Home) => {
                        let cur_vl = visual_line_of_offset(&visual_lines, input.cursor);
                        let new_pos = visual_lines[cur_vl].text_start;
                        if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
                        self.state.reset_blink();
                        continue;
                    }
                    Key::Named(NamedKey::End) => {
                        let cur_vl = visual_line_of_offset(&visual_lines, input.cursor);
                        let new_pos = visual_lines[cur_vl].text_end;
                        if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
                        self.state.reset_blink();
                        continue;
                    }
                    _ => {}
                }
                let changed = process_text_area_key(input, &event.key, ctrl, shift, &mut self.text, font_size);
                if changed {
                    response.changed = true;
                    self.state.reset_blink();
                }
            }
        }

        // Scroll.
        let scroll_y = match self.state.scroll_offsets.get_mut(&id) { Some((off, age)) => { *age = 0; off[0] } None => 0.0 };
        let content_height = total_visual as f32 * lh;
        let inner_h = visible_height - pad * 2.0;
        let max_scroll = (content_height - inner_h).max(0.0);
        let mut offset = scroll_y;

        if let Some((sx, sy, delta)) = self.state.pending_scroll {
            if rect.contains(sx, sy) {
                offset -= delta * self.theme.scroll_speed;
                self.state.pending_scroll = None;
            }
        }

        // Keep cursor visible.
        if response.focused {
            let cursor_vl = visual_line_of_offset(&visual_lines, input.cursor) as f32;
            let cursor_top = cursor_vl * lh;
            let cursor_bottom = cursor_top + lh;
            if cursor_top < offset {
                offset = cursor_top;
            }
            if cursor_bottom > offset + inner_h {
                offset = cursor_bottom - inner_h;
            }
        }

        offset = offset.clamp(0.0, max_scroll);
        self.state.scroll_offsets.insert(id, ([offset, 0.0], 0));

        // Draw.
        if response.focused {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }
        paint::draw_rounded_rect(self.frame, rect, self.theme.bg_input, self.theme.corner_radius);
        let border_color = if response.focused { self.theme.accent } else { self.theme.border };
        paint::draw_border(self.frame, rect, border_color);

        let saved_clip = self.frame.active_clip();
        self.frame.set_active_clip(Some(rect.to_clip_array()));

        let text_x = rect.x + pad;
        let text_y = rect.y + pad;

        if input.text.is_empty() && !response.focused {
            self.text.draw_ui_text(placeholder, text_x, text_y, self.theme.fg_dim, self.frame, self.gpu, self.resources);
            self.frame.set_active_clip(saved_clip);
            return response;
        }

        // Rebuild visual lines (text may have changed from key processing).
        let visual_lines = build_visual_lines(&input.text, &mut self.text, font_size, content_width);
        let total_visual = visual_lines.len();

        let first_visible = (offset / lh).floor() as usize;
        let last_visible = ((offset + inner_h) / lh).ceil() as usize;

        for i in first_visible..last_visible.min(total_visual) {
            let vl = &visual_lines[i];
            let line = &input.text[vl.text_start..vl.text_end];
            let ly = text_y + i as f32 * lh - offset;

            // Selection highlight.
            if let Some((sel_start, sel_end)) = input.selection {
                let line_sel_start = sel_start.max(vl.text_start);
                let line_sel_end = sel_end.min(vl.text_end);
                if line_sel_start < line_sel_end {
                    let sel_x0 = self.text.measure_text(&input.text[vl.text_start..line_sel_start], font_size);
                    let sel_x1 = self.text.measure_text(&input.text[vl.text_start..line_sel_end], font_size);
                    self.frame.push(
                        ShapeBuilder::rect(text_x + sel_x0, ly, sel_x1 - sel_x0, lh)
                            .color(self.theme.accent_dim)
                            .build(),
                    );
                }
            }

            self.text.draw_ui_text(line, text_x, ly, self.theme.fg, self.frame, self.gpu, self.resources);
        }

        // IME preedit + Cursor.
        if response.focused {
            let cursor_vl_idx = visual_line_of_offset(&visual_lines, input.cursor);
            let vl = &visual_lines[cursor_vl_idx];
            let cursor_x_in_line = self.text.measure_text(&input.text[vl.text_start..input.cursor], font_size);
            let cy = text_y + cursor_vl_idx as f32 * lh - offset;
            let mut cx = text_x + cursor_x_in_line;

            // IME preedit rendering.
            if !self.state.ime.preedit.is_empty() {
                let preedit_w = self.text.measure_text(&self.state.ime.preedit, font_size);
                self.frame.push(
                    ShapeBuilder::rect(cx, cy + lh - 1.0, preedit_w, 1.0)
                        .color(self.theme.fg_dim)
                        .build(),
                );
                self.text.draw_ui_text(
                    &self.state.ime.preedit, cx, cy, self.theme.fg_dim,
                    self.frame, self.gpu, self.resources,
                );
                cx += preedit_w;
            }

            if self.state.cursor_blink {
                self.frame.push(
                    ShapeBuilder::rect(cx, cy, self.theme.cursor_width, lh)
                        .color(self.theme.fg)
                        .build(),
                );
            }
        }

        self.frame.set_active_clip(saved_clip);
        response
    }
}

/// Process a key event for the text area. Returns true if the input was modified.
fn process_text_area_key(
    input: &mut InputState,
    key: &Key,
    ctrl: bool,
    shift: bool,
    text_renderer: &mut crate::text::TextRenderer,
    font_size: f32,
) -> bool {
    match key {
        Key::Named(NamedKey::Enter) => {
            input.save_undo();
            input.insert_char('\n');
            true
        }
        Key::Named(NamedKey::Tab) => {
            input.save_undo();
            input.insert_str("    ");
            true
        }
        Key::Named(NamedKey::Backspace) => {
            input.save_undo();
            if ctrl { input.delete_word_back(); } else { input.delete_back(); }
            true
        }
        Key::Named(NamedKey::Delete) => {
            input.save_undo();
            if ctrl { input.delete_word_forward(); } else { input.delete_forward(); }
            true
        }
        Key::Named(NamedKey::ArrowLeft) => {
            if ctrl && shift {
                input.move_word_left_extend();
            } else if ctrl {
                input.move_word_left();
            } else if shift {
                input.move_left_extend();
            } else {
                input.move_left();
            }
            true
        }
        Key::Named(NamedKey::ArrowRight) => {
            if ctrl && shift {
                input.move_word_right_extend();
            } else if ctrl {
                input.move_word_right();
            } else if shift {
                input.move_right_extend();
            } else {
                input.move_right();
            }
            true
        }
        Key::Named(NamedKey::ArrowUp) => {
            let cur_line = line_of_offset(&input.text, input.cursor);
            let new_pos = if cur_line == 0 {
                0
            } else {
                let cur_ls = line_start(&input.text, input.cursor);
                let visual_x = text_renderer.measure_text(&input.text[cur_ls..input.cursor], font_size);
                let target_ls = line_start_of_nth(&input.text, cur_line - 1);
                let target_le = line_end(&input.text, target_ls);
                find_offset_for_x(&input.text, target_ls, target_le, visual_x, text_renderer, font_size)
            };
            if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
            true
        }
        Key::Named(NamedKey::ArrowDown) => {
            let cur_line = line_of_offset(&input.text, input.cursor);
            let total = line_count(&input.text);
            let new_pos = if cur_line >= total - 1 {
                input.text.len()
            } else {
                let cur_ls = line_start(&input.text, input.cursor);
                let visual_x = text_renderer.measure_text(&input.text[cur_ls..input.cursor], font_size);
                let target_ls = line_start_of_nth(&input.text, cur_line + 1);
                let target_le = line_end(&input.text, target_ls);
                find_offset_for_x(&input.text, target_ls, target_le, visual_x, text_renderer, font_size)
            };
            if shift { input.move_to_extend(new_pos); } else { input.move_to(new_pos); }
            true
        }
        Key::Named(NamedKey::Home) => {
            let ls = line_start(&input.text, input.cursor);
            if shift { input.move_to_extend(ls); } else { input.move_to(ls); }
            true
        }
        Key::Named(NamedKey::End) => {
            let le = line_end(&input.text, input.cursor);
            if shift { input.move_to_extend(le); } else { input.move_to(le); }
            true
        }
        Key::Named(NamedKey::Space) => {
            input.save_undo();
            input.insert_char(' ');
            true
        }
        Key::Character(ch) if ctrl && ch.as_str() == "a" => {
            input.select_all();
            true
        }
        Key::Character(ch) if ctrl && ch.as_str() == "z" => {
            if shift { input.redo(); } else { input.undo(); }
            true
        }
        Key::Character(ch) if ctrl && ch.as_str() == "Z" => {
            input.redo();
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

/// Find the byte offset within `text[line_start..line_end]` closest to visual x position.
fn find_offset_for_x(
    text: &str,
    ls: usize,
    le: usize,
    target_x: f32,
    text_renderer: &mut crate::text::TextRenderer,
    font_size: f32,
) -> usize {
    let line = &text[ls..le];
    let mut best = ls;
    let mut best_dist = target_x.abs();

    for (i, _) in line.char_indices() {
        let advance = text_renderer.measure_text(&line[..i], font_size);
        let dist = (advance - target_x).abs();
        if dist < best_dist {
            best = ls + i;
            best_dist = dist;
        }
    }
    // Check end of line.
    let end_advance = text_renderer.measure_text(line, font_size);
    if (end_advance - target_x).abs() < best_dist {
        best = le;
    }
    best
}

/// Map a click (x, y) to a cursor byte position in the text area.
fn xy_to_cursor(
    input: &InputState,
    text_renderer: &mut crate::text::TextRenderer,
    rect: Rect,
    click_x: f32,
    click_y: f32,
    scroll_y: f32,
    font_size: f32,
    pad: f32,
    line_height: f32,
) -> usize {
    let text_y = rect.y + pad;
    let clicked_line = ((click_y - text_y + scroll_y) / line_height).floor().max(0.0) as usize;
    let total = line_count(&input.text);
    let target_line = clicked_line.min(total.saturating_sub(1));

    let ls = line_start_of_nth(&input.text, target_line);
    let le = line_end(&input.text, ls);

    let text_x = rect.x + pad;
    let rel_x = click_x - text_x;

    find_offset_for_x(&input.text, ls, le, rel_x, text_renderer, font_size)
}
