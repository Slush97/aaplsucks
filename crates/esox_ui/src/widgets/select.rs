//! Select widget — dropdown trigger + overlay menu.

use esox_gfx::{BorderRadius, ShapeBuilder};
use esox_input::{Key, NamedKey};

use crate::id::HOVER_SALT;
use crate::layout::Rect;
use crate::paint;
use crate::response::Response;
use crate::state::{A11yNode, A11yRole, Overlay, SelectState, WidgetKind};
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a select field. Returns a Response where `changed` indicates a new selection.
    pub fn select(
        &mut self,
        id: u64,
        select: &mut SelectState,
        choices: &[&str],
    ) -> Response {
        let rect = self.allocate_rect_keyed(id, self.region.w, self.theme.button_height);
        self.register_widget(id, rect, WidgetKind::Select);

        let mut response = self.widget_response(id, rect);
        let disabled = response.disabled;

        let is_dropdown_open = matches!(
            &self.state.overlay,
            Some(Overlay::Dropdown { id: oid, .. }) if *oid == id
        );
        let selected_label = choices.get(select.selected_index).copied().unwrap_or("");
        self.push_a11y_node(A11yNode {
            id, role: A11yRole::Select, label: selected_label.to_string(),
            value: Some(select.selected_index.to_string()), rect,
            focused: response.focused, disabled,
            expanded: Some(is_dropdown_open), selected: None, checked: None,
            value_range: None, children: Vec::new(),
        });

        // Clamp selection.
        if !choices.is_empty() && select.selected_index >= choices.len() {
            select.selected_index = 0;
        }

        // Handle click — toggle dropdown (skip when disabled).
        if response.clicked && !disabled {
            self.toggle_overlay(id, rect, choices, select.selected_index);
        }

        // Handle Enter on focused select — toggle dropdown.
        if response.focused && !response.clicked && !disabled {
            let enter_pressed = self.state.keys.iter().any(|(event, _)| {
                event.pressed && event.key == Key::Named(NamedKey::Enter)
            });
            if enter_pressed {
                self.toggle_overlay(id, rect, choices, select.selected_index);
            }
        }

        // Handle arrow keys on focused select (when dropdown is closed).
        if response.focused {
            let is_open = matches!(
                &self.state.overlay,
                Some(Overlay::Dropdown { id: oid, .. }) if *oid == id
            );
            if !is_open && !choices.is_empty() {
                for (event, _) in &self.state.keys.clone() {
                    if !event.pressed {
                        continue;
                    }
                    match &event.key {
                        Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowUp) => {
                            if select.selected_index == 0 {
                                select.selected_index = choices.len() - 1;
                            } else {
                                select.selected_index -= 1;
                            }
                            response.changed = true;
                        }
                        Key::Named(NamedKey::ArrowRight | NamedKey::ArrowDown) => {
                            select.selected_index = (select.selected_index + 1) % choices.len();
                            response.changed = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Focus ring.
        if response.focused && !disabled {
            paint::draw_focus_ring(
                self.frame,
                rect,
                self.theme.accent_dim,
                self.theme.corner_radius,
                self.theme.focus_ring_expand,
            );
        }

        // ── Draw trigger ──

        let value = choices
            .get(select.selected_index)
            .copied()
            .unwrap_or("");

        // Background.
        let bg = if disabled {
            self.theme.disabled_bg
        } else {
            let t = self.state.hover_t(id ^ HOVER_SALT, response.hovered, self.theme.hover_duration_ms);
            paint::lerp_color(self.theme.bg_input, self.theme.bg_raised, t)
        };
        paint::draw_rounded_rect(self.frame, rect, bg, self.theme.corner_radius);

        // Border.
        if disabled {
            paint::draw_dashed_border(
                self.frame, rect, self.theme.disabled_border,
                self.theme.disabled_dash_len, self.theme.disabled_dash_gap, self.theme.disabled_dash_thickness,
            );
        } else {
            let border_color = if response.focused {
                self.theme.accent
            } else {
                self.theme.border
            };
            paint::draw_border(self.frame, rect, border_color);
        }

        let text_y = rect.y + (rect.h - self.theme.font_size) / 2.0;
        let text_color = if disabled { self.theme.disabled_fg } else { self.theme.fg };

        // Value text.
        self.text.draw_ui_text(
            value,
            rect.x + self.theme.input_padding,
            text_y,
            text_color,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Chevron.
        let chevron = "\u{25BE}";
        let chevron_color = if disabled {
            self.theme.disabled_fg
        } else if response.focused {
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

        response
    }

    fn toggle_overlay(
        &mut self,
        id: u64,
        anchor: Rect,
        choices: &[&str],
        selected: usize,
    ) {
        let is_open = matches!(
            &self.state.overlay,
            Some(Overlay::Dropdown { id: oid, .. }) if *oid == id
        );
        if is_open {
            self.state.overlay = None;
        } else {
            self.state.overlay = Some(Overlay::Dropdown {
                id,
                anchor,
                choices: choices.iter().map(|s| s.to_string()).collect(),
                hovered: Some(selected),
                selected,
            });
        }
    }

    /// Draw the dropdown/context menu overlay. Called from `finish()`.
    pub(crate) fn draw_overlay(&mut self) -> Option<(u64, usize)> {
        let overlay = self.state.overlay.as_ref()?;

        match overlay {
            Overlay::ContextMenu { .. } => return self.draw_context_menu_overlay(),
            Overlay::ComboboxDropdown { .. } => return self.draw_combobox_overlay(),
            _ => {}
        }

        let (id, anchor, choices) = match overlay {
            Overlay::Dropdown {
                id,
                anchor,
                choices,
                ..
            } => (*id, *anchor, choices.clone()),
            _ => unreachable!(),
        };

        let item_h = self.theme.item_height;
        let dd_x = anchor.x;
        let dd_y = anchor.y + anchor.h + self.theme.dropdown_gap;
        let dd_w = anchor.w;
        let dd_h = choices.len() as f32 * item_h;

        // Handle click within dropdown.
        let mut selection_result = None;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if cx >= dd_x && cx < dd_x + dd_w && cy >= dd_y && cy < dd_y + dd_h {
                let idx = ((cy - dd_y) / item_h) as usize;
                if idx < choices.len() {
                    selection_result = Some((id, idx));
                }
                *consumed = true;
            } else {
                // Clicked outside — close overlay.
                // Don't consume — let the click fall through.
            }
        }

        // Handle arrow keys + Enter within dropdown.
        {
            let overlay = self.state.overlay.as_mut()?;
            let (choices, hovered) = match overlay {
                Overlay::Dropdown { ref choices, ref mut hovered, .. } => (choices, hovered),
                _ => unreachable!(),
            };
            {
                for (event, _) in &self.state.keys {
                    if !event.pressed {
                        continue;
                    }
                    match &event.key {
                        Key::Named(NamedKey::ArrowUp) => {
                            let cur = hovered.unwrap_or(0);
                            *hovered = Some(if cur == 0 {
                                choices.len() - 1
                            } else {
                                cur - 1
                            });
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            let cur = hovered.unwrap_or(choices.len().saturating_sub(1));
                            *hovered = Some((cur + 1) % choices.len());
                        }
                        Key::Named(NamedKey::Enter) => {
                            if let Some(idx) = hovered {
                                selection_result = Some((id, *idx));
                            }
                        }
                        Key::Named(NamedKey::Escape) => {
                            // Will be closed below.
                            selection_result = None;
                            self.state.overlay = None;
                            return selection_result;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Re-read overlay for drawing (it may have been modified).
        let overlay = match self.state.overlay.as_ref() {
            Some(o) => o,
            None => return selection_result,
        };
        let (choices, hovered, selected) = match overlay {
            Overlay::Dropdown {
                choices,
                hovered,
                selected,
                ..
            } => (choices.clone(), *hovered, *selected),
            _ => unreachable!(),
        };
        let dd_h = choices.len() as f32 * item_h;

        // Shadow/backdrop.
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

        // Items.
        for (i, choice) in choices.iter().enumerate() {
            let iy = dd_y + i as f32 * item_h;
            let is_selected = i == selected;
            let is_hovered = hovered == Some(i);

            if is_hovered {
                self.frame.push(
                    ShapeBuilder::rect(dd_x + 1.0, iy, dd_w - 2.0, item_h)
                        .color(self.theme.bg_raised)
                        .build(),
                );
            }

            let text_color = if is_selected {
                self.theme.accent
            } else {
                self.theme.fg
            };

            self.text.draw_ui_text(
                choice,
                dd_x + self.theme.input_padding,
                iy + (item_h - self.theme.font_size) / 2.0,
                text_color,
                self.frame,
                self.gpu,
                self.resources,
            );

            if is_selected {
                let check_w = self.text.measure_text("\u{2713}", self.theme.font_size);
                self.text.draw_ui_text(
                    "\u{2713}",
                    dd_x + dd_w - self.theme.input_padding - check_w,
                    iy + (item_h - self.theme.font_size) / 2.0,
                    self.theme.accent,
                    self.frame,
                    self.gpu,
                    self.resources,
                );
            }
        }

        if selection_result.is_some() {
            self.state.overlay = None;
        }

        selection_result
    }

    /// Draw a context menu overlay. Called from `draw_overlay()`.
    fn draw_context_menu_overlay(&mut self) -> Option<(u64, usize)> {
        let overlay = self.state.overlay.as_ref()?;
        let (id, position, items) = match overlay {
            Overlay::ContextMenu {
                id,
                position,
                items,
                ..
            } => (*id, *position, items.clone()),
            _ => return None,
        };

        let item_h = self.theme.item_height;
        let menu_x = position.x;
        let menu_y = position.y;
        let menu_w = position.w;
        let menu_h = items.len() as f32 * item_h;

        // Handle click within/outside menu.
        let mut selection_result = None;
        if let Some((cx, cy, ref mut consumed)) = self.state.mouse.pending_click {
            if cx >= menu_x && cx < menu_x + menu_w && cy >= menu_y && cy < menu_y + menu_h {
                let idx = ((cy - menu_y) / item_h) as usize;
                if idx < items.len() {
                    selection_result = Some((id, idx));
                }
                *consumed = true;
            } else {
                // Clicked outside — close.
                *consumed = true;
                self.state.overlay = None;
                return None;
            }
        }

        // Handle arrow keys, Enter, Escape.
        {
            let overlay = self.state.overlay.as_mut()?;
            let (items, hovered) = match overlay {
                Overlay::ContextMenu {
                    ref items,
                    ref mut hovered,
                    ..
                } => (items, hovered),
                _ => return None,
            };
            for (event, _) in &self.state.keys {
                if !event.pressed {
                    continue;
                }
                match &event.key {
                    Key::Named(NamedKey::ArrowUp) => {
                        let cur = hovered.unwrap_or(0);
                        *hovered = Some(if cur == 0 {
                            items.len() - 1
                        } else {
                            cur - 1
                        });
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        let cur = hovered.unwrap_or(items.len().saturating_sub(1));
                        *hovered = Some((cur + 1) % items.len());
                    }
                    Key::Named(NamedKey::Enter) => {
                        if let Some(idx) = hovered {
                            selection_result = Some((id, *idx));
                        }
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.state.overlay = None;
                        return None;
                    }
                    _ => {}
                }
            }
        }

        // Re-read for drawing.
        let overlay = match self.state.overlay.as_ref() {
            Some(o) => o,
            None => return selection_result,
        };
        let (items, hovered) = match overlay {
            Overlay::ContextMenu {
                items, hovered, ..
            } => (items.clone(), *hovered),
            _ => return selection_result,
        };

        // Shadow.
        self.frame.push(
            ShapeBuilder::rect(menu_x - 1.0, menu_y - 1.0, menu_w + 2.0, menu_h + 2.0)
                .color(self.theme.shadow)
                .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                .build(),
        );

        // Background.
        self.frame.push(
            ShapeBuilder::rect(menu_x, menu_y, menu_w, menu_h)
                .color(self.theme.bg_surface)
                .border_radius(BorderRadius::uniform(self.theme.corner_radius))
                .build(),
        );

        // Border.
        paint::draw_border(
            self.frame,
            Rect::new(menu_x, menu_y, menu_w, menu_h),
            self.theme.accent,
        );

        // Items.
        for (i, item) in items.iter().enumerate() {
            let iy = menu_y + i as f32 * item_h;
            let is_hovered = hovered == Some(i);

            if is_hovered {
                self.frame.push(
                    ShapeBuilder::rect(menu_x + 1.0, iy, menu_w - 2.0, item_h)
                        .color(self.theme.bg_raised)
                        .build(),
                );
            }

            self.text.draw_ui_text(
                item,
                menu_x + self.theme.input_padding,
                iy + (item_h - self.theme.font_size) / 2.0,
                self.theme.fg,
                self.frame,
                self.gpu,
                self.resources,
            );
        }

        if selection_result.is_some() {
            self.state.overlay = None;
        }

        selection_result
    }
}
