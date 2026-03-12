//! UI interaction state — focus, hit testing, input, keyboard/mouse routing.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use winit::event::KeyEvent;
use winit::keyboard::ModifiersState;

use crate::layout::Rect;

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
}

impl InputState {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            selection: None,
            scroll_offset: 0.0,
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
        self.cursor = 0;
    }

    /// Move cursor to the end of text.
    pub fn end(&mut self) {
        self.selection = None;
        self.cursor = self.text.len();
    }

    /// Select all text.
    pub fn select_all(&mut self) {
        if !self.text.is_empty() {
            self.selection = Some((0, self.text.len()));
            self.cursor = self.text.len();
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
    ColumnResize,
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
    pub(crate) keys: Vec<(KeyEvent, ModifiersState)>,
    /// Current modifier keys state.
    pub modifiers: ModifiersState,
    /// Cursor blink state.
    pub cursor_blink: bool,
    /// When the cursor blink last toggled.
    pub cursor_blink_time: Instant,
    /// Scroll offsets keyed by widget ID.
    pub scroll_offsets: HashMap<u64, f32>,
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
}

impl UiState {
    pub fn new() -> Self {
        Self {
            focused: None,
            focus_chain: Vec::new(),
            hit_rects: Vec::new(),
            mouse: MouseState::default(),
            keys: Vec::new(),
            modifiers: ModifiersState::empty(),
            cursor_blink: true,
            cursor_blink_time: Instant::now(),
            scroll_offsets: HashMap::new(),
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
        }
    }

    /// Buffer a key event for processing during the frame.
    pub fn process_key(&mut self, event: KeyEvent, modifiers: ModifiersState) {
        self.modifiers = modifiers;
        self.keys.push((event, modifiers));
    }

    /// Update modifier keys state.
    pub fn process_modifiers(&mut self, modifiers: ModifiersState) {
        self.modifiers = modifiers;
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
            None => {}
        }
    }

    /// Record a mouse click (left button press).
    pub fn process_mouse_click(&mut self, x: f32, y: f32) {
        self.mouse.pending_click = Some((x, y, false));
        self.mouse_pressed = true;
    }

    /// Record a right-click (button 2).
    pub fn process_right_click(&mut self, x: f32, y: f32) {
        self.mouse.pending_right_click = Some((x, y, false));
    }

    /// Record a mouse button release.
    pub fn process_mouse_release(&mut self) {
        self.mouse_pressed = false;
        self.scrollbar_drag = None;
        // Drag ends on release — drag payload stays until end_frame so accept_drop can read it.
        self.drag_start = None;
    }

    /// Buffer a scroll (mouse wheel) event for processing during the frame.
    pub fn process_scroll(&mut self, x: f32, y: f32, delta_y: f32) {
        self.pending_scroll = Some((x, y, delta_y));
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
    pub fn cursor_icon(&self, x: f32, y: f32) -> winit::window::CursorIcon {
        // Iterate in reverse so the topmost (last-registered) widget wins.
        for (rect, _id, kind) in self.hit_rects.iter().rev() {
            if rect.contains(x, y) {
                return match kind {
                    WidgetKind::TextInput => winit::window::CursorIcon::Text,
                    WidgetKind::ColumnResize => winit::window::CursorIcon::ColResize,
                    WidgetKind::Button
                    | WidgetKind::DropZone
                    | WidgetKind::Select
                    | WidgetKind::Checkbox
                    | WidgetKind::Radio
                    | WidgetKind::Tab
                    | WidgetKind::TableRow
                    | WidgetKind::TreeNode => winit::window::CursorIcon::Pointer,
                    WidgetKind::Slider | WidgetKind::Scrollbar => winit::window::CursorIcon::Default,
                };
            }
        }
        winit::window::CursorIcon::Default
    }

    /// Whether the UI needs continuous redraw (cursor blink, overlay, tooltip delay, active animations, etc.).
    pub fn needs_continuous_redraw(&self) -> bool {
        self.overlay.is_some()
            || self.focused.map_or(false, |id| self.is_text_widget(id))
            || self.hover_anims.values().any(|a| !a.is_settled())
            || self.anims.values().any(|a| !a.is_settled())
            || self.scrollbar_drag.is_some()
            || self.drag.is_some()
            || self.tooltip.as_ref().is_some_and(|t| !t.visible)
            || !self.modal_stack.is_empty()
            || !self.toasts.toasts.is_empty()
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
        self.focus_chain.clear();
        self.hit_rects.clear();
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
        // Prune settled hover animations when the map grows large.
        // Active widgets call hover_t() each frame, keeping their entries alive.
        // Settled anims for off-screen widgets accumulate — cap at 256.
        if self.hover_anims.len() > 256 {
            self.hover_anims.retain(|_, anim| !anim.is_settled());
        }
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
