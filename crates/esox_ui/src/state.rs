//! UI interaction state — focus, hit testing, input, keyboard/mouse routing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use esox_input::{KeyEvent, Modifiers};

use crate::layout::Rect;
use crate::widgets::menu_bar::MenuBarDeferred;

/// Trait for clipboard access. Implemented by platform layer.
pub trait ClipboardProvider {
    fn read_text(&self) -> Option<String>;
    fn write_text(&self, text: &str);
}

/// A snapshot of text state for undo/redo.
#[derive(Debug, Clone)]
struct UndoEntry {
    text: String,
    cursor: usize,
}

/// Undo/redo history for text input.
#[derive(Debug, Clone)]
pub struct UndoHistory {
    entries: VecDeque<UndoEntry>,
    index: usize,
}

impl UndoHistory {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            index: 0,
        }
    }

    fn push(&mut self, text: &str, cursor: usize) {
        // Truncate redo history.
        self.entries.truncate(self.index);
        self.entries.push_back(UndoEntry {
            text: text.to_string(),
            cursor,
        });
        // Cap at 100 entries.
        if self.entries.len() > 100 {
            self.entries.pop_front();
        }
        self.index = self.entries.len();
    }

    fn undo(&mut self) -> Option<&UndoEntry> {
        if self.index > 0 {
            self.index -= 1;
            Some(&self.entries[self.index])
        } else {
            None
        }
    }

    fn redo(&mut self) -> Option<&UndoEntry> {
        if self.index < self.entries.len() {
            let entry = &self.entries[self.index];
            self.index += 1;
            Some(entry)
        } else {
            None
        }
    }
}

/// Text input state: buffer, cursor, selection.
#[derive(Debug, Clone)]
pub struct InputState {
    /// The text content.
    pub text: String,
    /// Byte offset of the cursor within `text`.
    pub cursor: usize,
    /// Selection range as (start, end) byte offsets, where start <= end.
    pub selection: Option<(usize, usize)>,
    /// Horizontal scroll offset in pixels for long text.
    pub scroll_offset: f32,
    /// Anchor for shift+arrow selection expansion. When shift is held, cursor moves
    /// but anchor stays fixed; selection = (anchor.min(cursor), anchor.max(cursor)).
    pub selection_anchor: Option<usize>,
    /// Undo/redo history.
    undo_history: UndoHistory,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            selection: None,
            scroll_offset: 0.0,
            selection_anchor: None,
            undo_history: UndoHistory::new(),
        }
    }

    /// Delete selected text if any, returning whether something was deleted.
    pub fn delete_selection(&mut self) -> bool {
        if let Some((start, end)) = self.selection.take() {
            self.text.drain(start..end);
            self.cursor = start;
            true
        } else {
            false
        }
    }

    /// Insert a character at the cursor, replacing any selection.
    pub fn insert_char(&mut self, c: char) {
        self.delete_selection();
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Insert a string at the cursor, replacing any selection.
    pub fn insert_str(&mut self, s: &str) {
        self.delete_selection();
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Delete the character before the cursor (Backspace).
    pub fn delete_back(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor > 0 {
            let prev = self.text[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.text.drain(prev..self.cursor);
            self.cursor = prev;
        }
    }

    /// Delete the character after the cursor (Delete key).
    pub fn delete_forward(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor < self.text.len() {
            let next = self.cursor
                + self.text[self.cursor..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
            self.text.drain(self.cursor..next);
        }
    }

    /// Move cursor one character left.
    pub fn move_left(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        if self.cursor > 0 {
            self.cursor = self.text[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    /// Move cursor one character right.
    pub fn move_right(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        if self.cursor < self.text.len() {
            self.cursor += self.text[self.cursor..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
        }
    }

    /// Move cursor to the beginning of text.
    pub fn home(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        self.cursor = 0;
    }

    /// Move cursor to the end of text.
    pub fn end(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        self.cursor = self.text.len();
    }

    // ── Shift+extend selection methods ──

    /// Set anchor if not already set, then update selection from anchor to cursor.
    fn update_selection_from_anchor(&mut self) {
        let anchor = self.selection_anchor.unwrap_or(self.cursor);
        self.selection_anchor = Some(anchor);
        if anchor == self.cursor {
            self.selection = None;
        } else {
            self.selection = Some((anchor.min(self.cursor), anchor.max(self.cursor)));
        }
    }

    /// Move cursor left, extending selection (Shift+Left).
    pub fn move_left_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        if self.cursor > 0 {
            self.cursor = self.text[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        self.update_selection_from_anchor();
    }

    /// Move cursor right, extending selection (Shift+Right).
    pub fn move_right_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        if self.cursor < self.text.len() {
            self.cursor += self.text[self.cursor..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
        }
        self.update_selection_from_anchor();
    }

    /// Move cursor to start, extending selection (Shift+Home).
    pub fn home_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = 0;
        self.update_selection_from_anchor();
    }

    /// Move cursor to end, extending selection (Shift+End).
    pub fn end_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = self.text.len();
        self.update_selection_from_anchor();
    }

    /// Move cursor to an arbitrary position, extending selection.
    pub fn move_to_extend(&mut self, pos: usize) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = pos;
        self.update_selection_from_anchor();
    }

    /// Move cursor to an arbitrary position, clearing selection.
    pub fn move_to(&mut self, pos: usize) {
        self.selection = None;
        self.selection_anchor = None;
        self.cursor = pos;
    }

    // ── Word boundary helpers ──

    /// Find the byte offset of the word boundary to the left of `pos`.
    fn word_boundary_left(&self, pos: usize) -> usize {
        let bytes = self.text.as_bytes();
        let mut i = pos;
        // Skip whitespace backward.
        while i > 0 && bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        // Skip non-whitespace backward.
        while i > 0 && !bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        i
    }

    /// Find the byte offset of the word boundary to the right of `pos`.
    fn word_boundary_right(&self, pos: usize) -> usize {
        let bytes = self.text.as_bytes();
        let len = bytes.len();
        let mut i = pos;
        // Skip non-whitespace forward.
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        // Skip whitespace forward.
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    }

    /// Move cursor one word left (Ctrl+Left).
    pub fn move_word_left(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        self.cursor = self.word_boundary_left(self.cursor);
    }

    /// Move cursor one word right (Ctrl+Right).
    pub fn move_word_right(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
        self.cursor = self.word_boundary_right(self.cursor);
    }

    /// Move cursor one word left, extending selection (Ctrl+Shift+Left).
    pub fn move_word_left_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = self.word_boundary_left(self.cursor);
        self.update_selection_from_anchor();
    }

    /// Move cursor one word right, extending selection (Ctrl+Shift+Right).
    pub fn move_word_right_extend(&mut self) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = self.word_boundary_right(self.cursor);
        self.update_selection_from_anchor();
    }

    /// Delete the word before the cursor (Ctrl+Backspace).
    pub fn delete_word_back(&mut self) {
        if self.delete_selection() {
            return;
        }
        let target = self.word_boundary_left(self.cursor);
        if target < self.cursor {
            self.text.drain(target..self.cursor);
            self.cursor = target;
        }
    }

    /// Delete the word after the cursor (Ctrl+Delete).
    pub fn delete_word_forward(&mut self) {
        if self.delete_selection() {
            return;
        }
        let target = self.word_boundary_right(self.cursor);
        if target > self.cursor {
            self.text.drain(self.cursor..target);
        }
    }

    /// Select all text.
    pub fn select_all(&mut self) {
        if !self.text.is_empty() {
            self.selection = Some((0, self.text.len()));
            self.selection_anchor = Some(0);
            self.cursor = self.text.len();
        }
    }

    /// Save current state to undo history. Call before mutations.
    pub fn save_undo(&mut self) {
        // Only push if text differs from last entry.
        let dominated = self.undo_history.entries.back().is_some_and(|e| e.text == self.text);
        if !dominated {
            self.undo_history.push(&self.text, self.cursor);
        }
    }

    /// Undo the last text change.
    pub fn undo(&mut self) {
        if let Some(entry) = self.undo_history.undo() {
            self.text = entry.text.clone();
            self.cursor = entry.cursor.min(self.text.len());
            self.selection = None;
            self.selection_anchor = None;
        }
    }

    /// Redo the last undone change.
    pub fn redo(&mut self) {
        if let Some(entry) = self.undo_history.redo() {
            self.text = entry.text.clone();
            self.cursor = entry.cursor.min(self.text.len());
            self.selection = None;
            self.selection_anchor = None;
        }
    }

    /// Get the selected text, if any.
    pub fn selected_text(&self) -> Option<&str> {
        self.selection.map(|(s, e)| &self.text[s..e])
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for a select widget — just tracks the selected index.
#[derive(Debug, Clone)]
pub struct SelectState {
    pub selected_index: usize,
}

impl SelectState {
    pub fn new() -> Self {
        Self { selected_index: 0 }
    }
}

impl Default for SelectState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for the file drop zone.
#[derive(Debug, Clone)]
pub struct DropZoneState {
    /// Selected files.
    pub files: Vec<PathBuf>,
    /// Whether a file dialog is currently open.
    pub dialog_pending: bool,
}

impl DropZoneState {
    pub fn new() -> Self {
        Self {
            files: Vec::new(),
            dialog_pending: false,
        }
    }
}

impl Default for DropZoneState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for tab widget.
#[derive(Debug, Clone)]
pub struct TabState {
    pub selected: usize,
}

impl TabState {
    pub fn new() -> Self {
        Self { selected: 0 }
    }
}

impl Default for TabState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for virtual scroll widget.
#[derive(Debug, Clone)]
pub struct VirtualScrollState {
    pub item_count: usize,
    /// Set to Some(index) to auto-scroll that item into view.
    pub scroll_to: Option<usize>,
}

impl VirtualScrollState {
    pub fn new(item_count: usize) -> Self {
        Self {
            item_count,
            scroll_to: None,
        }
    }
}

/// Sort direction for table columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// State for table widget.
#[derive(Debug, Clone)]
pub struct TableState {
    pub selected_row: Option<usize>,
    /// User-overridden column widths (None = use default).
    pub column_widths: Vec<Option<f32>>,
    /// Active resize drag: (column_index, start_mouse_x, start_column_width).
    pub(crate) resize_drag: Option<(usize, f32, f32)>,
    /// Current sort state: (column_index, direction).
    pub sort: Option<(usize, SortDirection)>,
    /// Multi-select: set of selected row indices.
    pub selected_rows: HashSet<usize>,
    /// Anchor row for shift-click range selection.
    pub anchor_row: Option<usize>,
}

impl TableState {
    pub fn new() -> Self {
        Self {
            selected_row: None,
            column_widths: Vec::new(),
            resize_drag: None,
            sort: None,
            selected_rows: HashSet::new(),
            anchor_row: None,
        }
    }
}

impl Default for TableState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for tree widget.
#[derive(Debug, Clone)]
pub struct TreeState {
    pub expanded: HashSet<u64>,
    pub selected: Option<u64>,
    /// Multi-select: set of selected node IDs.
    pub selected_nodes: HashSet<u64>,
    /// Anchor node for shift-click range selection.
    pub anchor_node: Option<u64>,
}

impl TreeState {
    pub fn new() -> Self {
        Self {
            expanded: HashSet::new(),
            selected: None,
            selected_nodes: HashSet::new(),
            anchor_node: None,
        }
    }
}

impl Default for TreeState {
    fn default() -> Self {
        Self::new()
    }
}

/// State for widget-to-widget drag.
#[derive(Debug, Clone, Copy)]
pub struct DragPayload {
    pub source_id: u64,
    pub payload: u64,
    pub x: f32,
    pub y: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

/// Widget type hint for cursor icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetKind {
    TextInput,
    Button,
    DropZone,
    Select,
    Slider,
    Checkbox,
    Radio,
    Scrollbar,
    Tab,
    TableRow,
    TreeNode,
    Toggle,
    Hyperlink,
    ColumnResize,
    ResizeNS,
    ResizeEW,
    Grab,
    Grabbing,
    NotAllowed,
    SplitDividerH,
    SplitDividerV,
    Combobox,
    Custom(esox_input::CursorIcon),
}

/// Overlay state (dropdown menus drawn on top of everything).
pub enum Overlay {
    Dropdown {
        id: u64,
        anchor: Rect,
        choices: Vec<String>,
        hovered: Option<usize>,
        selected: usize,
    },
    ContextMenu {
        id: u64,
        position: Rect,
        items: Vec<String>,
        hovered: Option<usize>,
    },
    ComboboxDropdown {
        id: u64,
        anchor: Rect,
        /// Full list of options (unfiltered).
        all_choices: Vec<String>,
        /// Indices into `all_choices` that match the current filter.
        filtered_indices: Vec<usize>,
        /// Which item in the *filtered* list is highlighted (keyboard nav).
        highlighted: Option<usize>,
        /// Scroll offset for long filtered lists (in pixels).
        scroll_offset: f32,
    },
}

/// Tooltip state — hover delay + text.
pub struct TooltipState {
    pub widget_id: u64,
    pub hover_start: Instant,
    pub anchor: Rect,
    pub text: String,
    pub visible: bool,
}

/// Mouse tracking state.
#[derive(Debug, Default)]
pub struct MouseState {
    pub x: f32,
    pub y: f32,
    /// Pending click: position + consumed flag.
    pub pending_click: Option<(f32, f32, bool)>,
    /// Pending right-click: position + consumed flag.
    pub pending_right_click: Option<(f32, f32, bool)>,
}

/// Hover animation state — drives ease-out color transitions.
pub struct HoverAnim {
    pub from: f32,
    pub to: f32,
    pub start: Instant,
    pub duration_ms: f32,
}

impl HoverAnim {
    /// Current interpolation value in [0, 1] with ease-out cubic.
    pub fn t(&self) -> f32 {
        let p = (self.start.elapsed().as_millis() as f32 / self.duration_ms).clamp(0.0, 1.0);
        let eased = 1.0 - (1.0 - p).powi(3);
        self.from + (self.to - self.from) * eased
    }

    /// True when the animation has fully completed or from == to.
    pub fn is_settled(&self) -> bool {
        (self.from - self.to).abs() < 0.001
            || self.start.elapsed().as_millis() as f32 >= self.duration_ms
    }
}

/// Easing functions for animations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Easing {
    Linear,
    EaseOutCubic,
    EaseInOutCubic,
    EaseOutExpo,
}

impl Easing {
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::EaseOutCubic => 1.0 - (1.0 - t).powi(3),
            Easing::EaseInOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Easing::EaseOutExpo => {
                if (t - 1.0).abs() < f32::EPSILON {
                    1.0
                } else {
                    1.0 - 2.0f32.powf(-10.0 * t)
                }
            }
        }
    }
}

/// General-purpose animation state.
pub struct Anim {
    pub from: f32,
    pub to: f32,
    pub start: Instant,
    pub duration_ms: f32,
    pub easing: Easing,
    /// Whether this anim was queried this frame (for cleanup).
    pub(crate) queried: bool,
}

impl Anim {
    /// Current interpolation value.
    pub fn value(&self) -> f32 {
        let p = (self.start.elapsed().as_millis() as f32 / self.duration_ms).clamp(0.0, 1.0);
        let eased = self.easing.apply(p);
        self.from + (self.to - self.from) * eased
    }

    pub fn is_settled(&self) -> bool {
        (self.from - self.to).abs() < 0.001
            || self.start.elapsed().as_millis() as f32 >= self.duration_ms
    }
}

/// Result from a modal dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalAction {
    None,
    Confirm,
    Cancel,
}

/// Modal dialog state.
pub struct ModalState {
    pub id: u64,
    pub open: bool,
    pub saved_focus: Option<u64>,
}

/// Toast notification kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Success,
    Error,
    Warning,
}

/// A single toast notification.
pub struct Toast {
    pub id: u64,
    pub kind: ToastKind,
    pub message: String,
    pub created: Instant,
    pub duration_ms: u64,
    pub dismissed: bool,
}

/// Queue of active toast notifications.
pub struct ToastQueue {
    pub toasts: Vec<Toast>,
    pub next_id: u64,
}

impl ToastQueue {
    pub fn new() -> Self {
        Self {
            toasts: Vec::new(),
            next_id: 1,
        }
    }

    pub fn push(&mut self, kind: ToastKind, message: String, duration_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.toasts.push(Toast {
            id,
            kind,
            message,
            created: Instant::now(),
            duration_ms,
            dismissed: false,
        });
        id
    }

    pub fn dismiss(&mut self, id: u64) {
        if let Some(toast) = self.toasts.iter_mut().find(|t| t.id == id) {
            toast.dismissed = true;
        }
    }
}

impl Default for ToastQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Accessibility node role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum A11yRole {
    Button,
    Checkbox,
    RadioButton,
    TextInput,
    TextArea,
    Slider,
    Select,
    Tab,
    TabPanel,
    Table,
    TableRow,
    TableCell,
    Tree,
    TreeItem,
    ProgressBar,
    Dialog,
    Alert,
    Label,
    Separator,
    ScrollView,
    Group,
    ToggleButton,
    Link,
    SpinButton,
    Combobox,
}

/// A single accessibility node.
pub struct A11yNode {
    pub id: u64,
    pub role: A11yRole,
    pub label: String,
    pub value: Option<String>,
    pub rect: Rect,
    pub focused: bool,
    pub disabled: bool,
    pub expanded: Option<bool>,
    pub selected: Option<bool>,
    pub checked: Option<bool>,
    pub value_range: Option<(f32, f32, f32)>,
    pub children: Vec<u64>,
}

/// Accessibility tree built each frame.
pub struct A11yTree {
    pub nodes: Vec<A11yNode>,
    pub root_children: Vec<u64>,
}

impl A11yTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_children: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_children.clear();
    }

    pub fn push(&mut self, node: A11yNode) {
        self.root_children.push(node.id);
        self.nodes.push(node);
    }
}

impl Default for A11yTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Central UI interaction state. App owns this and passes `&mut` to `Ui`.
pub struct UiState {
    /// Currently focused widget ID.
    pub focused: Option<u64>,
    /// Focus chain — rebuilt each frame in widget call order.
    pub(crate) focus_chain: Vec<u64>,
    /// Hit-test rects — rebuilt each frame.
    pub(crate) hit_rects: Vec<(Rect, u64, WidgetKind)>,
    /// Mouse state.
    pub mouse: MouseState,
    /// Buffered key events — drained during the frame by widgets.
    pub(crate) keys: Vec<(KeyEvent, Modifiers)>,
    /// Current modifier keys state.
    pub modifiers: Modifiers,
    /// Cursor blink state.
    pub cursor_blink: bool,
    /// When the cursor blink last toggled.
    pub cursor_blink_time: Instant,
    /// Scroll offsets keyed by widget ID: (offset, frames_since_last_access).
    /// Scroll offsets: `[vertical, horizontal]` and age counter.
    pub scroll_offsets: HashMap<u64, ([f32; 2], u32)>,
    /// Timestamp of last frame start for dt calculation.
    pub(crate) last_frame_time: Instant,
    /// Damage tracker for frame-skip optimization.
    pub damage: esox_gfx::DamageTracker,
    /// Previous frame's hovered widget (for damage on hover change).
    prev_hovered: Option<u64>,
    /// Previous frame's focused widget (for damage on focus change).
    prev_focused: Option<u64>,
    /// Overlay (dropdown / context menu) state.
    pub overlay: Option<Overlay>,
    /// Tooltip state.
    pub tooltip: Option<TooltipState>,
    /// Hover animation states keyed by widget ID.
    pub hover_anims: HashMap<u64, HoverAnim>,
    /// General-purpose animations keyed by ID.
    pub anims: HashMap<u64, Anim>,
    /// Buffered scroll event: (mouse_x, mouse_y, delta_y).
    pub pending_scroll: Option<(f32, f32, f32)>,
    /// Active scrollbar drag: (scrollable_id, grab_offset_in_thumb).
    pub scrollbar_drag: Option<(u64, f32)>,
    /// Whether the left mouse button is currently held.
    pub mouse_pressed: bool,
    /// Active drag-and-drop payload.
    pub drag: Option<DragPayload>,
    /// Mouse position at last press (for dead zone).
    pub drag_start: Option<(f32, f32)>,
    /// Modal dialog stack.
    pub modal_stack: Vec<ModalState>,
    /// Toast notification queue.
    pub toasts: ToastQueue,
    /// Accessibility tree.
    pub a11y_tree: A11yTree,
    /// Whether accessibility is enabled.
    pub a11y_enabled: bool,
    /// Pending accessibility label for next widget.
    pub(crate) a11y_pending_label: Option<String>,
    /// Pending accessibility role for next widget.
    pub(crate) a11y_pending_role: Option<A11yRole>,
    /// Cached children heights for tree expand/collapse animations.
    pub tree_children_heights: HashMap<u64, f32>,
    /// Clipboard provider for Ctrl+C/X/V in text widgets.
    pub clipboard: Option<Box<dyn ClipboardProvider>>,
    /// HiDPI scale factor (default 1.0).
    pub scale_factor: f32,
    /// IME composition state.
    pub ime: ImeState,
    /// Whether a spinner was drawn this frame (needs continuous redraw).
    pub(crate) spinner_active: bool,
    /// Collapsing header open/closed state keyed by widget ID.
    pub collapsing_open: HashSet<u64>,
    /// Split pane ratios keyed by widget ID.
    pub split_ratios: HashMap<u64, f32>,
    /// Active split-pane divider drag: (split_id, is_horizontal).
    pub split_drag: Option<(u64, bool)>,
    /// Which menu bar dropdown is currently open (index into the menus slice).
    pub menu_bar_open: Option<usize>,
    /// Deferred menu bar dropdown painting data (drawn in finish() for z-order).
    pub(crate) menu_bar_deferred: Option<MenuBarDeferred>,
    /// Inline edit buffers for number_input widgets, keyed by widget ID.
    pub number_edit_buffers: HashMap<u64, InputState>,
    /// Per-combobox filter input state, keyed by widget ID.
    pub combobox_inputs: HashMap<u64, InputState>,
    /// Scroll velocity for momentum scrolling: `[vertical_vel, horizontal_vel]`.
    pub scroll_velocity: HashMap<u64, [f32; 2]>,
    /// Tile grid for partial redraw caching.
    pub tile_grid: Option<esox_gfx::TileGrid>,
    /// Cached solved layout tree from the previous frame.
    pub(crate) layout_cache: Option<crate::layout_tree::LayoutTree>,
    /// Whether the debug overlay is enabled (toggle with Alt+D or programmatically).
    pub debug_overlay: bool,
    /// Collected widget rects for debug overlay (populated when `debug_overlay` is true).
    pub(crate) debug_widget_rects: Vec<(Rect, u64, &'static str)>,
    /// The WidgetKind of the currently focused widget (set during register_widget).
    pub(crate) focused_kind: Option<WidgetKind>,
    /// Previous frame's max scroll values per scrollable, for pre-clamping on content shrink.
    /// Value: ([vertical_max, horizontal_max], age_counter).
    pub(crate) prev_max_scroll: HashMap<u64, ([f32; 2], u32)>,
    /// Whether the mouse moved since last frame (for targeted hover damage).
    pub(crate) mouse_moved: bool,
    /// Rects of widgets with active animations, for targeted animation damage.
    pub(crate) anim_rects: HashMap<u64, Rect>,
}

/// IME (Input Method Editor) composition state.
pub struct ImeState {
    /// Preedit text being composed (not yet committed).
    pub preedit: String,
    /// Cursor range within the preedit text.
    pub preedit_cursor: Option<(usize, usize)>,
    /// Whether IME is currently active.
    pub active: bool,
    /// Committed text to be consumed by the focused text widget.
    pub committed: Option<String>,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            focused: None,
            focus_chain: Vec::new(),
            hit_rects: Vec::new(),
            mouse: MouseState::default(),
            keys: Vec::new(),
            modifiers: Modifiers::empty(),
            cursor_blink: true,
            cursor_blink_time: Instant::now(),
            scroll_offsets: HashMap::new(),
            last_frame_time: Instant::now(),
            damage: esox_gfx::DamageTracker::new(),
            prev_hovered: None,
            prev_focused: None,
            overlay: None,
            tooltip: None,
            hover_anims: HashMap::new(),
            anims: HashMap::new(),
            pending_scroll: None,
            scrollbar_drag: None,
            mouse_pressed: false,
            drag: None,
            drag_start: None,
            modal_stack: Vec::new(),
            toasts: ToastQueue::new(),
            a11y_tree: A11yTree::new(),
            a11y_enabled: false,
            a11y_pending_label: None,
            a11y_pending_role: None,
            tree_children_heights: HashMap::new(),
            clipboard: None,
            scale_factor: 1.0,
            ime: ImeState {
                preedit: String::new(),
                preedit_cursor: None,
                active: false,
                committed: None,
            },
            spinner_active: false,
            collapsing_open: HashSet::new(),
            split_ratios: HashMap::new(),
            split_drag: None,
            menu_bar_open: None,
            menu_bar_deferred: None,
            number_edit_buffers: HashMap::new(),
            combobox_inputs: HashMap::new(),
            scroll_velocity: HashMap::new(),
            tile_grid: None,
            layout_cache: None,
            debug_overlay: false,
            debug_widget_rects: Vec::new(),
            focused_kind: None,
            prev_max_scroll: HashMap::new(),
            mouse_moved: false,
            anim_rects: HashMap::new(),
        }
    }

    /// Buffer a key event for processing during the frame.
    pub fn process_key(&mut self, event: KeyEvent, modifiers: Modifiers) {
        self.modifiers = modifiers;
        self.keys.push((event, modifiers));
        self.damage.invalidate_all();
    }

    /// Update modifier keys state.
    pub fn process_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }

    /// Process IME preedit event.
    pub fn on_ime_preedit(&mut self, text: String, cursor: Option<(usize, usize)>) {
        self.ime.preedit = text;
        self.ime.preedit_cursor = cursor;
        self.damage.invalidate_all();
    }

    /// Process IME commit event — buffer the text for the focused widget.
    pub fn on_ime_commit(&mut self, text: String) {
        self.ime.preedit.clear();
        self.ime.preedit_cursor = None;
        self.ime.committed = Some(text);
        self.damage.invalidate_all();
    }

    /// Process IME enabled/disabled event.
    pub fn on_ime_enabled(&mut self, enabled: bool) {
        self.ime.active = enabled;
        if !enabled {
            self.ime.preedit.clear();
            self.ime.preedit_cursor = None;
        }
    }

    /// Update mouse position. `item_height` and `dropdown_gap` are used for
    /// hover tracking within any open dropdown overlay.
    pub fn process_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        item_height: f32,
        dropdown_gap: f32,
    ) {
        // Flag mouse movement for targeted hover damage in begin_frame().
        if (self.mouse.x - x).abs() > 0.5 || (self.mouse.y - y).abs() > 0.5 {
            self.mouse_moved = true;
        }
        self.mouse.x = x;
        self.mouse.y = y;

        // Track hover within open overlay.
        match &mut self.overlay {
            Some(Overlay::Dropdown {
                ref anchor,
                ref choices,
                ref mut hovered,
                ..
            }) => {
                let dd_y = anchor.y + anchor.h + dropdown_gap;
                if x >= anchor.x
                    && x < anchor.x + anchor.w
                    && y >= dd_y
                {
                    let idx = ((y - dd_y) / item_height) as usize;
                    if idx < choices.len() {
                        *hovered = Some(idx);
                    } else {
                        *hovered = None;
                    }
                } else {
                    *hovered = None;
                }
            }
            Some(Overlay::ContextMenu {
                ref position,
                ref items,
                ref mut hovered,
                ..
            }) => {
                // position.x/y is the menu top-left; position.w is menu width.
                let menu_h = items.len() as f32 * item_height;
                if x >= position.x
                    && x < position.x + position.w
                    && y >= position.y
                    && y < position.y + menu_h
                {
                    let idx = ((y - position.y) / item_height) as usize;
                    if idx < items.len() {
                        *hovered = Some(idx);
                    } else {
                        *hovered = None;
                    }
                } else {
                    *hovered = None;
                }
            }
            Some(Overlay::ComboboxDropdown {
                ref anchor,
                ref filtered_indices,
                ref mut highlighted,
                ref scroll_offset,
                ..
            }) => {
                let max_visible = 8;
                let visible_count = filtered_indices.len().min(max_visible);
                let dd_y = anchor.y + anchor.h + dropdown_gap;
                let dd_h = visible_count as f32 * item_height;
                if x >= anchor.x
                    && x < anchor.x + anchor.w
                    && y >= dd_y
                    && y < dd_y + dd_h
                {
                    let idx = ((y - dd_y + *scroll_offset) / item_height) as usize;
                    if idx < filtered_indices.len() {
                        *highlighted = Some(idx);
                    } else {
                        *highlighted = None;
                    }
                } else {
                    *highlighted = None;
                }
            }
            None => {}
        }
    }

    /// Record a mouse click (left button press).
    pub fn process_mouse_click(&mut self, x: f32, y: f32) {
        self.mouse.pending_click = Some((x, y, false));
        self.mouse_pressed = true;
        self.damage.invalidate_all();
    }

    /// Record a right-click (button 2).
    pub fn process_right_click(&mut self, x: f32, y: f32) {
        self.mouse.pending_right_click = Some((x, y, false));
        self.damage.invalidate_all();
    }

    /// Record a mouse button release.
    pub fn process_mouse_release(&mut self) {
        self.mouse_pressed = false;
        self.scrollbar_drag = None;
        self.split_drag = None;
        // Drag ends on release — drag payload stays until end_frame so accept_drop can read it.
        self.drag_start = None;
    }

    /// Buffer a scroll (mouse wheel) event for processing during the frame.
    pub fn process_scroll(&mut self, x: f32, y: f32, delta_y: f32) {
        self.pending_scroll = Some((x, y, delta_y));
        self.damage.invalidate_all();
    }

    /// Update cursor blink. Call once per frame.
    pub fn update_blink(&mut self, blink_ms: u64) {
        let elapsed = self.cursor_blink_time.elapsed().as_millis() as u64;
        if elapsed >= blink_ms {
            self.cursor_blink = !self.cursor_blink;
            self.cursor_blink_time = Instant::now();
        }
    }

    /// Reset cursor blink to visible (call after text editing).
    pub fn reset_blink(&mut self) {
        self.cursor_blink = true;
        self.cursor_blink_time = Instant::now();
    }

    /// Get the cursor icon for the given position based on registered widgets.
    pub fn cursor_icon(&self, x: f32, y: f32) -> esox_input::CursorIcon {
        // Active drag overrides everything.
        if self.drag.is_some() {
            return esox_input::CursorIcon::Grabbing;
        }
        // Active split-pane divider drag overrides cursor.
        if let Some((_, is_horizontal)) = self.split_drag {
            return if is_horizontal {
                esox_input::CursorIcon::ColResize
            } else {
                esox_input::CursorIcon::RowResize
            };
        }
        // Iterate in reverse so the topmost (last-registered) widget wins.
        for (rect, _id, kind) in self.hit_rects.iter().rev() {
            if rect.contains(x, y) {
                return match kind {
                    WidgetKind::TextInput => esox_input::CursorIcon::Text,
                    WidgetKind::ColumnResize | WidgetKind::ResizeEW => esox_input::CursorIcon::ColResize,
                    WidgetKind::ResizeNS => esox_input::CursorIcon::RowResize,
                    WidgetKind::Grab => esox_input::CursorIcon::Grab,
                    WidgetKind::Grabbing => esox_input::CursorIcon::Grabbing,
                    WidgetKind::NotAllowed => esox_input::CursorIcon::NotAllowed,
                    WidgetKind::Button
                    | WidgetKind::DropZone
                    | WidgetKind::Select
                    | WidgetKind::Checkbox
                    | WidgetKind::Radio
                    | WidgetKind::Tab
                    | WidgetKind::TableRow
                    | WidgetKind::TreeNode
                    | WidgetKind::Toggle => esox_input::CursorIcon::Pointer,
                    WidgetKind::Hyperlink => esox_input::CursorIcon::Pointer,
                    WidgetKind::Slider | WidgetKind::Scrollbar => esox_input::CursorIcon::Default,
                    WidgetKind::SplitDividerH => esox_input::CursorIcon::ColResize,
                    WidgetKind::SplitDividerV => esox_input::CursorIcon::RowResize,
                    WidgetKind::Combobox => esox_input::CursorIcon::Text,
                    WidgetKind::Custom(icon) => *icon,
                };
            }
        }
        esox_input::CursorIcon::Default
    }

    /// Whether the UI needs continuous redraw (cursor blink, overlay, tooltip delay, active animations, etc.).
    pub fn needs_continuous_redraw(&self) -> bool {
        self.overlay.is_some()
            || self.focused.map_or(false, |id| self.is_text_widget(id))
            || self.hover_anims.values().any(|a| !a.is_settled())
            || self.anims.values().any(|a| !a.is_settled())
            || self.scrollbar_drag.is_some()
            || self.split_drag.is_some()
            || self.drag.is_some()
            || self.tooltip.as_ref().is_some_and(|t| !t.visible)
            || !self.modal_stack.is_empty()
            || !self.toasts.toasts.is_empty()
            || self.spinner_active
    }

    /// Whether the damage tracker indicates a redraw is needed.
    ///
    /// This is a frame-skip check: returns `false` when nothing changed
    /// since the last frame, allowing the platform to skip GPU submission.
    pub fn needs_redraw(&self) -> bool {
        self.damage.is_full_invalidation()
            || self.damage.regions().map_or(false, |r| !r.is_empty())
    }

    /// Get or update a hover animation, returning the current interpolation value.
    /// On first call the animation starts settled at the appropriate end value.
    pub fn hover_t(&mut self, id: u64, is_hovered: bool, duration_ms: f32) -> f32 {
        let target = if is_hovered { 1.0 } else { 0.0 };
        let anim = self.hover_anims.entry(id).or_insert_with(|| {
            // First-seen: start settled at the correct end value.
            HoverAnim {
                from: target,
                to: target,
                start: Instant::now(),
                duration_ms,
            }
        });
        if (anim.to - target).abs() > 0.001 {
            // Target changed — restart from current value.
            anim.from = anim.t();
            anim.to = target;
            anim.start = Instant::now();
        }
        anim.t()
    }

    /// Get or create a general-purpose animation. Returns current value.
    /// Restarts from current value when target changes.
    pub fn anim_t(&mut self, id: u64, target: f32, duration_ms: f32, easing: Easing) -> f32 {
        let anim = self.anims.entry(id).or_insert_with(|| {
            Anim {
                from: target,
                to: target,
                start: Instant::now(),
                duration_ms,
                easing,
                queried: true,
            }
        });
        anim.queried = true;
        if (anim.to - target).abs() > 0.001 {
            anim.from = anim.value();
            anim.to = target;
            anim.start = Instant::now();
            anim.easing = easing;
            anim.duration_ms = duration_ms;
        }
        anim.value()
    }

    /// Whether a given animation is currently active (not settled).
    pub fn anim_active(&self, id: u64) -> bool {
        self.anims.get(&id).map_or(false, |a| !a.is_settled())
    }

    /// Advance focus to the next widget in the focus chain.
    pub fn focus_next(&mut self) {
        if self.focus_chain.is_empty() {
            return;
        }
        let idx = self
            .focused
            .and_then(|f| self.focus_chain.iter().position(|w| *w == f));
        let next = match idx {
            Some(i) => (i + 1) % self.focus_chain.len(),
            None => 0,
        };
        self.focused = Some(self.focus_chain[next]);
        self.reset_blink();
    }

    /// Advance focus to the previous widget in the focus chain.
    pub fn focus_prev(&mut self) {
        if self.focus_chain.is_empty() {
            return;
        }
        let idx = self
            .focused
            .and_then(|f| self.focus_chain.iter().position(|w| *w == f));
        let prev = match idx {
            Some(0) => self.focus_chain.len() - 1,
            Some(i) => i - 1,
            None => self.focus_chain.len() - 1,
        };
        self.focused = Some(self.focus_chain[prev]);
        self.reset_blink();
    }

    /// Clear per-frame state. Called at the start of each frame.
    pub(crate) fn begin_frame(&mut self) {
        self.last_frame_time = Instant::now();

        // Damage detection: hover/focus changes, active animations, scroll velocity.
        if self.mouse_moved {
            let current_hovered = self.hit_rects.iter().rev()
                .find(|(r, _, _)| r.contains(self.mouse.x, self.mouse.y))
                .map(|(_, id, _)| *id);
            if current_hovered != self.prev_hovered {
                // Add damage for old and new hovered widget rects.
                if let Some(old_id) = self.prev_hovered {
                    if let Some((r, _, _)) = self.hit_rects.iter().find(|(_, id, _)| *id == old_id) {
                        self.damage.add(esox_gfx::DamageRect::new(r.x, r.y, r.w, r.h));
                    }
                }
                if let Some(new_id) = current_hovered {
                    if let Some((r, _, _)) = self.hit_rects.iter().find(|(_, id, _)| *id == new_id) {
                        self.damage.add(esox_gfx::DamageRect::new(r.x, r.y, r.w, r.h));
                    }
                }
            }
            self.mouse_moved = false;
        }
        if self.focused != self.prev_focused {
            self.damage.invalidate_all();
        }
        // Per-widget animation damage: only invalidate rects of unsettled animations.
        {
            let mut any_missing = false;
            for (id, anim) in &self.hover_anims {
                if !anim.is_settled() {
                    if let Some(r) = self.anim_rects.get(id) {
                        self.damage.add(esox_gfx::DamageRect::new(r.x, r.y, r.w, r.h));
                    } else {
                        any_missing = true;
                    }
                }
            }
            for (id, anim) in &self.anims {
                if !anim.is_settled() {
                    if let Some(r) = self.anim_rects.get(id) {
                        self.damage.add(esox_gfx::DamageRect::new(r.x, r.y, r.w, r.h));
                    } else {
                        any_missing = true;
                    }
                }
            }
            if any_missing {
                self.damage.invalidate_all();
            }
            self.anim_rects.clear();
        }
        if self.overlay.is_some() || !self.modal_stack.is_empty() || !self.toasts.toasts.is_empty()
            || self.menu_bar_open.is_some()
        {
            self.damage.invalidate_all();
        }

        self.focus_chain.clear();
        // Momentum scrolling: decay velocities and apply to offsets.
        self.scroll_velocity.retain(|id, vel| {
            let any_active = vel[0].abs() > 0.5 || vel[1].abs() > 0.5;
            if any_active {
                if let Some((off, _)) = self.scroll_offsets.get_mut(id) {
                    off[0] += vel[0];
                    off[1] += vel[1];
                }
                // Friction is applied per-frame; 0.92 is the default.
                vel[0] *= 0.92;
                vel[1] *= 0.92;
                if vel[0].abs() < 0.5 { vel[0] = 0.0; }
                if vel[1].abs() < 0.5 { vel[1] = 0.0; }
            }
            any_active
        });

        self.hit_rects.clear();

        // Automatic Tab/Shift+Tab focus cycling. Uses focused_kind from the
        // previous frame (set during register_widget calls). TextInput widgets
        // keep Tab for literal insertion.
        {
            use esox_input::{Key, NamedKey};
            let is_text_input = matches!(self.focused_kind, Some(WidgetKind::TextInput));
            let mut tab_action: Option<bool> = None; // Some(true) = shift+tab, Some(false) = tab
            if !is_text_input {
                for (event, mods) in &self.keys {
                    if event.pressed {
                        if let Key::Named(NamedKey::Tab) = &event.key {
                            tab_action = Some(mods.shift());
                            break;
                        }
                    }
                }
            }
            if let Some(shift) = tab_action {
                self.keys.retain(|(event, _)| {
                    !(event.pressed
                        && matches!(event.key, Key::Named(NamedKey::Tab)))
                });
                if shift {
                    self.focus_prev();
                } else {
                    self.focus_next();
                }
            }
            self.focused_kind = None; // reset for this frame
        }

        self.spinner_active = false;
        self.debug_widget_rects.clear();
        if self.a11y_enabled {
            self.a11y_tree.clear();
        }
        // Mark all anims as unqueried for cleanup.
        for anim in self.anims.values_mut() {
            anim.queried = false;
        }
    }

    /// End-of-frame cleanup. Clears consumed events.
    pub(crate) fn end_frame(&mut self) {
        self.keys.clear();
        self.pending_scroll = None;
        // Clear drag on mouse release.
        if !self.mouse_pressed {
            self.drag = None;
        }
        // Clear consumed click.
        if let Some((_, _, consumed)) = self.mouse.pending_click {
            if consumed {
                self.mouse.pending_click = None;
            }
        }
        // If click was not consumed by any widget, clear focus.
        if let Some((_, _, false)) = self.mouse.pending_click.take() {
            self.focused = None;
        }
        // Clear consumed right-click.
        if let Some((_, _, consumed)) = self.mouse.pending_right_click {
            if consumed {
                self.mouse.pending_right_click = None;
            }
        }
        // Unconsumed right-click — just clear it.
        if let Some((_, _, false)) = self.mouse.pending_right_click.take() {
            // No action needed.
        }
        // Prune settled anims that weren't queried this frame.
        self.anims.retain(|_, a| a.queried || !a.is_settled());
        // Remove expired and dismissed toasts.
        self.toasts.toasts.retain(|t| {
            if t.dismissed {
                return false;
            }
            let elapsed = t.created.elapsed().as_millis() as u64;
            // Keep for duration + fade_out time (300ms).
            elapsed < t.duration_ms + 300
        });
        // Prune stale scroll offsets (not accessed for >300 frames ≈ 5s at 60fps).
        self.scroll_offsets.retain(|_, (_, age)| {
            *age += 1;
            *age <= 300
        });
        // Prune settled hover animations when the map grows large.
        // Active widgets call hover_t() each frame, keeping their entries alive.
        // Settled anims for off-screen widgets accumulate — cap at 256.
        if self.hover_anims.len() > 256 {
            self.hover_anims.retain(|_, anim| !anim.is_settled());
        }
        // Prune stale prev_max_scroll (not accessed for >300 frames).
        self.prev_max_scroll.retain(|_, (_, age)| {
            *age += 1;
            *age <= 300
        });
        // Save hover/focus state for next frame's damage detection.
        self.prev_hovered = self.hit_rects.iter().rev()
            .find(|(r, _, _)| r.contains(self.mouse.x, self.mouse.y))
            .map(|(_, id, _)| *id);
        self.prev_focused = self.focused;
        // Reset damage tracker for next frame.
        self.damage.reset();
    }

    fn is_text_widget(&self, id: u64) -> bool {
        // Widgets registered as TextInput kind.
        self.hit_rects
            .iter()
            .any(|(_, wid, kind)| *wid == id && *kind == WidgetKind::TextInput)
    }
}

impl Default for UiState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ══════════════════════════════════════════════════════════════
    // InputState — cursor ops
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn insert_char_at_cursor() {
        let mut s = InputState::new();
        s.insert_char('A');
        assert_eq!(s.text, "A");
        assert_eq!(s.cursor, 1);
        s.insert_char('B');
        assert_eq!(s.text, "AB");
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn insert_char_mid_string() {
        let mut s = InputState::new();
        s.text = "ac".into();
        s.cursor = 1;
        s.insert_char('b');
        assert_eq!(s.text, "abc");
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn insert_str_at_cursor() {
        let mut s = InputState::new();
        s.insert_str("hello");
        assert_eq!(s.text, "hello");
        assert_eq!(s.cursor, 5);
        s.insert_str(" world");
        assert_eq!(s.text, "hello world");
        assert_eq!(s.cursor, 11);
    }

    #[test]
    fn delete_back_ascii() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 3;
        s.delete_back();
        assert_eq!(s.text, "ab");
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn delete_back_multibyte_utf8() {
        let mut s = InputState::new();
        s.text = "aé".into(); // 'é' is 2 bytes
        s.cursor = s.text.len();
        s.delete_back();
        assert_eq!(s.text, "a");
        assert_eq!(s.cursor, 1);
    }

    #[test]
    fn delete_back_at_start_is_noop() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 0;
        s.delete_back();
        assert_eq!(s.text, "abc");
        assert_eq!(s.cursor, 0);
    }

    #[test]
    fn delete_forward_ascii() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 1;
        s.delete_forward();
        assert_eq!(s.text, "ac");
        assert_eq!(s.cursor, 1);
    }

    #[test]
    fn delete_forward_at_end_is_noop() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 3;
        s.delete_forward();
        assert_eq!(s.text, "abc");
    }

    #[test]
    fn move_left_and_right() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 2;
        s.move_left();
        assert_eq!(s.cursor, 1);
        s.move_right();
        assert_eq!(s.cursor, 2);
    }

    #[test]
    fn move_left_at_start_stays() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 0;
        s.move_left();
        assert_eq!(s.cursor, 0);
    }

    #[test]
    fn move_right_at_end_stays() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 3;
        s.move_right();
        assert_eq!(s.cursor, 3);
    }

    #[test]
    fn home_and_end() {
        let mut s = InputState::new();
        s.text = "hello world".into();
        s.cursor = 5;
        s.home();
        assert_eq!(s.cursor, 0);
        s.end();
        assert_eq!(s.cursor, 11);
    }

    #[test]
    fn select_all_sets_selection() {
        let mut s = InputState::new();
        s.text = "hello".into();
        s.cursor = 2;
        s.select_all();
        assert_eq!(s.selection, Some((0, 5)));
        assert_eq!(s.cursor, 5);
    }

    #[test]
    fn select_all_empty_string_is_noop() {
        let mut s = InputState::new();
        s.select_all();
        assert!(s.selection.is_none());
    }

    #[test]
    fn delete_selection_removes_range() {
        let mut s = InputState::new();
        s.text = "hello world".into();
        s.selection = Some((5, 11));
        s.cursor = 11;
        assert!(s.delete_selection());
        assert_eq!(s.text, "hello");
        assert_eq!(s.cursor, 5);
        assert!(s.selection.is_none());
    }

    #[test]
    fn delete_selection_no_selection_returns_false() {
        let mut s = InputState::new();
        s.text = "hello".into();
        assert!(!s.delete_selection());
    }

    #[test]
    fn move_left_extend_creates_selection() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 2;
        s.move_left_extend();
        assert_eq!(s.cursor, 1);
        assert_eq!(s.selection, Some((1, 2)));
    }

    #[test]
    fn move_right_extend_creates_selection() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 1;
        s.move_right_extend();
        assert_eq!(s.cursor, 2);
        assert_eq!(s.selection, Some((1, 2)));
    }

    #[test]
    fn home_extend_selects_to_start() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 3;
        s.home_extend();
        assert_eq!(s.cursor, 0);
        assert_eq!(s.selection, Some((0, 3)));
    }

    #[test]
    fn end_extend_selects_to_end() {
        let mut s = InputState::new();
        s.text = "abc".into();
        s.cursor = 0;
        s.end_extend();
        assert_eq!(s.cursor, 3);
        assert_eq!(s.selection, Some((0, 3)));
    }

    #[test]
    fn selected_text_returns_correct_slice() {
        let mut s = InputState::new();
        s.text = "hello world".into();
        s.selection = Some((6, 11));
        assert_eq!(s.selected_text(), Some("world"));
    }

    #[test]
    fn selected_text_none_when_no_selection() {
        let s = InputState::new();
        assert_eq!(s.selected_text(), None);
    }

    // ══════════════════════════════════════════════════════════════
    // Easing::apply
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn easing_linear_boundaries() {
        assert!((Easing::Linear.apply(0.0)).abs() < 1e-6);
        assert!((Easing::Linear.apply(1.0) - 1.0).abs() < 1e-6);
        assert!((Easing::Linear.apply(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn easing_ease_out_cubic_boundaries() {
        assert!((Easing::EaseOutCubic.apply(0.0)).abs() < 1e-6);
        assert!((Easing::EaseOutCubic.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn easing_ease_in_out_cubic_boundaries_and_symmetry() {
        assert!((Easing::EaseInOutCubic.apply(0.0)).abs() < 1e-6);
        assert!((Easing::EaseInOutCubic.apply(1.0) - 1.0).abs() < 1e-6);
        let sum = Easing::EaseInOutCubic.apply(0.25) + Easing::EaseInOutCubic.apply(0.75);
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }

    #[test]
    fn easing_ease_out_expo_boundaries() {
        assert!((Easing::EaseOutExpo.apply(0.0)).abs() < 1e-3);
        assert!((Easing::EaseOutExpo.apply(1.0) - 1.0).abs() < 1e-6);
    }

    // ══════════════════════════════════════════════════════════════
    // UndoHistory
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn undo_redo_cycle() {
        let mut h = UndoHistory::new();
        h.push("a", 1);
        h.push("b", 2);
        let entry = h.undo().unwrap();
        assert_eq!(entry.text, "b");
        let entry = h.undo().unwrap();
        assert_eq!(entry.text, "a");
        let entry = h.redo().unwrap();
        assert_eq!(entry.text, "a");
        let entry = h.redo().unwrap();
        assert_eq!(entry.text, "b");
    }

    #[test]
    fn undo_past_beginning_returns_none() {
        let mut h = UndoHistory::new();
        h.push("a", 1);
        h.undo();
        assert!(h.undo().is_none());
    }

    #[test]
    fn redo_past_end_returns_none() {
        let mut h = UndoHistory::new();
        h.push("a", 1);
        assert!(h.redo().is_none());
    }

    #[test]
    fn push_after_undo_truncates_redo() {
        let mut h = UndoHistory::new();
        h.push("a", 1);
        h.push("b", 2);
        h.push("c", 3);
        h.undo(); // index=2
        h.undo(); // index=1
        h.push("d", 4);
        assert!(h.redo().is_none());
        let entry = h.undo().unwrap();
        assert_eq!(entry.text, "d");
        let entry = h.undo().unwrap();
        assert_eq!(entry.text, "a");
    }

    #[test]
    fn undo_history_caps_at_100() {
        let mut h = UndoHistory::new();
        for i in 0..110 {
            h.push(&format!("entry_{i}"), i);
        }
        assert_eq!(h.entries.len(), 100);
        assert_eq!(h.entries[0].text, "entry_10");
    }

    // ══════════════════════════════════════════════════════════════
    // ToastQueue
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn toast_push_returns_incrementing_ids() {
        let mut q = ToastQueue::new();
        let id1 = q.push(ToastKind::Info, "a".into(), 3000);
        let id2 = q.push(ToastKind::Error, "b".into(), 3000);
        let id3 = q.push(ToastKind::Success, "c".into(), 3000);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(q.toasts.len(), 3);
    }

    #[test]
    fn toast_dismiss_marks_dismissed() {
        let mut q = ToastQueue::new();
        let id = q.push(ToastKind::Warning, "warn".into(), 3000);
        assert!(!q.toasts[0].dismissed);
        q.dismiss(id);
        assert!(q.toasts[0].dismissed);
    }

    #[test]
    fn toast_dismiss_nonexistent_is_noop() {
        let mut q = ToastQueue::new();
        q.push(ToastKind::Info, "a".into(), 3000);
        q.dismiss(999);
        assert!(!q.toasts[0].dismissed);
    }

    // ══════════════════════════════════════════════════════════════
    // A11yTree
    // ══════════════════════════════════════════════════════════════

    #[test]
    fn a11y_tree_push_adds_node_and_root_child() {
        let mut tree = A11yTree::new();
        tree.push(A11yNode {
            id: 42,
            role: A11yRole::Button,
            label: "OK".into(),
            value: None,
            rect: Rect::new(0.0, 0.0, 80.0, 30.0),
            focused: false,
            disabled: false,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: vec![],
        });
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.root_children, vec![42]);
    }

    #[test]
    fn a11y_tree_clear_empties_both() {
        let mut tree = A11yTree::new();
        tree.push(A11yNode {
            id: 1,
            role: A11yRole::Label,
            label: "x".into(),
            value: None,
            rect: Rect::new(0.0, 0.0, 10.0, 10.0),
            focused: false,
            disabled: false,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: vec![],
        });
        tree.push(A11yNode {
            id: 2,
            role: A11yRole::Label,
            label: "y".into(),
            value: None,
            rect: Rect::new(0.0, 0.0, 10.0, 10.0),
            focused: false,
            disabled: false,
            expanded: None,
            selected: None,
            checked: None,
            value_range: None,
            children: vec![],
        });
        assert_eq!(tree.nodes.len(), 2);
        assert_eq!(tree.root_children.len(), 2);
        tree.clear();
        assert!(tree.nodes.is_empty());
        assert!(tree.root_children.is_empty());
    }
}
