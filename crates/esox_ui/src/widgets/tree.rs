//! Tree widget — expandable hierarchy with selection and multi-select.

use esox_gfx::ShapeBuilder;
use esox_input::{Key, NamedKey};

use crate::paint;
use crate::response::Response;
use crate::state::{TreeState, WidgetKind};
use crate::Ui;

/// Response from a tree node draw call.
pub struct TreeNodeResponse {
    pub response: Response,
    pub expanded: bool,
    pub selected: bool,
}

impl<'f> Ui<'f> {
    /// Draw a tree node. If expanded, caller draws children inside `tree_indent()`.
    pub fn tree_node(
        &mut self,
        id: u64,
        state: &mut TreeState,
        label: &str,
        has_children: bool,
    ) -> TreeNodeResponse {
        let item_h = self.theme.item_height;
        let font_size = self.theme.font_size;
        let pad = self.theme.input_padding;

        let rect = self.allocate_rect_keyed(id, self.region.w, item_h);
        self.register_widget(id, rect, WidgetKind::TreeNode);
        let mut response = self.widget_response(id, rect);

        let is_selected = state.selected_nodes.contains(&id)
            || (state.selected_nodes.is_empty() && state.selected == Some(id));
        let is_expanded = state.expanded.contains(&id);

        self.push_a11y_node(crate::state::A11yNode {
            id, role: crate::state::A11yRole::TreeItem, label: label.to_string(),
            value: None, rect, focused: response.focused, disabled: false,
            expanded: Some(is_expanded), selected: Some(is_selected), checked: None,
            value_range: None, children: Vec::new(),
        });

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
        if is_selected {
            self.frame.push(
                ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
                    .color(self.theme.accent_dim)
                    .build(),
            );
        } else if response.hovered {
            self.frame.push(
                ShapeBuilder::rect(rect.x, rect.y, rect.w, rect.h)
                    .color(self.theme.bg_raised)
                    .build(),
            );
        }

        // Click: toggle expand + select (with modifier support).
        let modifiers = self.state.modifiers;
        if response.clicked {
            let ctrl = modifiers.ctrl();
            let shift = modifiers.shift();

            if ctrl {
                // Ctrl+click: toggle in set.
                if state.selected_nodes.contains(&id) {
                    state.selected_nodes.remove(&id);
                } else {
                    state.selected_nodes.insert(id);
                }
                state.anchor_node = Some(id);
            } else if shift {
                // Shift+click: we'd need visible_order to do range select.
                // For now, just add to selection.
                state.selected_nodes.insert(id);
            } else {
                // Plain click: clear, select one.
                state.selected_nodes.clear();
                state.selected_nodes.insert(id);
                state.anchor_node = Some(id);
            }

            state.selected = Some(id);
            if has_children {
                if is_expanded {
                    state.expanded.remove(&id);
                } else {
                    state.expanded.insert(id);
                }
            }
            response.changed = true;
        }

        // Keyboard.
        if response.focused {
            let keys: Vec<_> = self.state.keys.clone();
            for (event, _mods) in &keys {
                if !event.pressed {
                    continue;
                }
                match &event.key {
                    Key::Named(NamedKey::Enter) | Key::Named(NamedKey::Space) => {
                        if has_children {
                            if is_expanded {
                                state.expanded.remove(&id);
                            } else {
                                state.expanded.insert(id);
                            }
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        if is_expanded && has_children {
                            state.expanded.remove(&id);
                            response.changed = true;
                        }
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        if !is_expanded && has_children {
                            state.expanded.insert(id);
                            response.changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Draw expand/collapse icon.
        let icon_x = rect.x + pad;
        let icon_y = rect.y + (item_h - font_size) / 2.0;
        let icon = if has_children {
            if is_expanded { "\u{25BC}" } else { "\u{25B6}" } // ▼ / ▶
        } else {
            "\u{2022}" // •
        };
        let icon_w = self.text.draw_text(
            icon,
            icon_x,
            icon_y,
            font_size * 0.7,
            self.theme.fg_muted,
            self.frame,
            self.gpu,
            self.resources,
        );

        // Draw label.
        let label_x = icon_x + icon_w + pad;
        self.text.draw_text(
            label,
            label_x,
            icon_y,
            font_size,
            self.theme.fg,
            self.frame,
            self.gpu,
            self.resources,
        );

        let expanded = if has_children {
            state.expanded.contains(&id)
        } else {
            false
        };

        TreeNodeResponse {
            response,
            expanded,
            selected: is_selected,
        }
    }
}
