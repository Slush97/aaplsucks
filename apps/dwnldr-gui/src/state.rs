//! App-level state types — form state and widget IDs.

use std::collections::HashMap;

pub use esox_ui::{DropZoneState, InputState, SelectState};

/// Form state keyed by (tool_id, field_key).
#[derive(Debug, Clone)]
pub struct FormState {
    entries: HashMap<(&'static str, &'static str), InputState>,
    selects: HashMap<(&'static str, &'static str), SelectState>,
}

impl FormState {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            selects: HashMap::new(),
        }
    }

    pub fn get_or_create(
        &mut self,
        tool_id: &'static str,
        key: &'static str,
    ) -> &mut InputState {
        self.entries
            .entry((tool_id, key))
            .or_insert_with(InputState::new)
    }

    pub fn get(&self, tool_id: &'static str, key: &'static str) -> Option<&InputState> {
        self.entries.get(&(tool_id, key))
    }

    pub fn get_or_create_select(
        &mut self,
        tool_id: &'static str,
        key: &'static str,
        num_choices: usize,
    ) -> &mut SelectState {
        let entry = self
            .selects
            .entry((tool_id, key))
            .or_insert(SelectState { selected_index: 0 });
        if num_choices > 0 && entry.selected_index >= num_choices {
            entry.selected_index = 0;
        }
        entry
    }

    pub fn select_value<'a>(
        &self,
        tool_id: &'static str,
        key: &'static str,
        choices: &'a [&'static str],
    ) -> &'a str {
        let idx = self
            .selects
            .get(&(tool_id, key))
            .map(|s| s.selected_index)
            .unwrap_or(0);
        choices.get(idx).copied().unwrap_or("")
    }
}
