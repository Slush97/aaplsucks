//! Application state and `AppDelegate` implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender};

use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::{AppDelegate, MouseInputEvent};
use esox_ui::{fnv1a_runtime, id, TextRenderer, Theme, ThemeTransition, ToastKind, UiState};
use tokio::runtime::Handle;
use tokio::task::AbortHandle;
use winit::event::KeyEvent;
use winit::keyboard::ModifiersState;

use crate::layout::{LayoutTree, SIDEBAR_FOOTER_H, SIDEBAR_WIDTH};
use crate::render;
use crate::state::{DropZoneState, FormState};
use crate::render::ResultAction;
use crate::tools::{self, InputKind};

/// The result of a completed tool operation.
#[derive(Debug, Clone)]
pub enum JobResult {
    File(String),
    Files(Vec<String>),
    Text(String),
}

/// Events sent from background tasks to the main thread.
pub enum JobEvent {
    Progress {
        job_id: u64,
        message: String,
        percent: Option<f32>,
    },
    Done {
        job_id: u64,
        result: JobResult,
    },
    Error {
        job_id: u64,
        message: String,
    },
}

/// A running or completed tool job.
pub struct Job {
    pub id: u64,
    pub tool_id: &'static str,
    pub status: JobStatus,
    pub message: String,
    pub progress: Option<f32>,
    pub result: Option<JobResult>,
    pub abort_handle: Option<AbortHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Done,
    Error,
}

/// Maps widget u64 IDs → (tool_id, field_key) for clipboard/IME routing.
struct IdMap {
    table: HashMap<u64, (&'static str, &'static str)>,
}

impl IdMap {
    fn build(tool_id: &'static str) -> Self {
        let mut table = HashMap::new();
        table.insert(id!("__input__"), (tool_id, "__input__"));
        if let Some(tool_def) = tools::find_tool(tool_id) {
            for opt in tool_def.options {
                let widget_id = fnv1a_runtime(&format!("opt_{}", opt.key));
                table.insert(widget_id, (tool_id, opt.key));
            }
        }
        Self { table }
    }
}

/// Main application state.
pub struct App {
    pub active_tool: &'static str,
    pub jobs: Vec<Job>,
    pub vw: f32,
    pub vh: f32,
    pub layout: LayoutTree,
    pub text: Option<TextRenderer>,
    pub should_exit: bool,
    pub theme: Theme,

    // esox_ui interaction state.
    pub ui_state: UiState,
    id_map: IdMap,

    // Form data — app-owned, passed to esox_ui widgets.
    pub form_state: FormState,
    pub drop_state: HashMap<&'static str, DropZoneState>,

    // File dialog channel.
    pub file_rx: Receiver<(&'static str, Vec<PathBuf>)>,
    pub file_tx: SyncSender<(&'static str, Vec<PathBuf>)>,

    // Job system.
    pub rt_handle: Handle,
    pub job_tx: SyncSender<JobEvent>,
    pub job_rx: Receiver<JobEvent>,
    pub next_job_id: u64,

    // Theme transition.
    pub theme_transition: Option<ThemeTransition>,

    // HiDPI.
    pub scale_factor: f32,

    // Sidebar hover (not part of esox_ui — app-specific panel).
    pub hovered_item: Option<&'static str>,
    pub hovered_toggle: bool,
    pub theme_is_dark: bool,
    pub sidebar_focus_index: Option<usize>,

    // Deferred actions from the Ui pass (processed after Ui::finish).
    pub needs_file_dialog: bool,
    pub needs_execute: bool,
    pub result_action: Option<ResultAction>,
}

impl App {
    pub fn new(initial_w: f32, initial_h: f32) -> Self {
        let (file_tx, file_rx) = mpsc::sync_channel(4);
        let (job_tx, job_rx) = mpsc::sync_channel(64);

        let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        let rt_handle = rt.handle().clone();
        std::thread::spawn(move || {
            rt.block_on(std::future::pending::<()>());
        });

        Self {
            active_tool: "download",
            jobs: Vec::new(),
            vw: initial_w,
            vh: initial_h,
            layout: LayoutTree::new(initial_w, initial_h),
            text: None,
            should_exit: false,
            theme: Theme::dark(),
            ui_state: UiState::new(),
            id_map: IdMap::build("download"),
            form_state: FormState::new(),
            drop_state: HashMap::new(),
            file_rx,
            file_tx,
            rt_handle,
            job_tx,
            job_rx,
            next_job_id: 1,
            theme_transition: None,
            scale_factor: 1.0,
            hovered_item: None,
            hovered_toggle: false,
            theme_is_dark: true,
            sidebar_focus_index: None,
            needs_file_dialog: false,
            needs_execute: false,
            result_action: None,
        }
    }

    /// Toggle between light and dark themes with an animated transition.
    pub fn toggle_theme(&mut self) {
        self.theme_is_dark = !self.theme_is_dark;
        let target = if self.theme_is_dark {
            Theme::dark()
        } else {
            Theme::light()
        };
        let target = if self.scale_factor != 1.0 {
            target.scaled(self.scale_factor)
        } else {
            target
        };
        self.theme_transition = Some(ThemeTransition::new(self.theme.clone(), target, 300.0));
    }

    /// Returns true if (x, y) is over the sidebar theme toggle footer.
    pub fn hit_test_theme_toggle(&self, x: f64, y: f64) -> bool {
        let sb = &self.layout.sidebar;
        let footer_y = sb.h - SIDEBAR_FOOTER_H;
        x >= sb.x as f64
            && x < (sb.x + sb.w) as f64
            && y >= footer_y as f64
            && y < sb.h as f64
    }

    /// Find which sidebar tool item the mouse is over (excludes category headers and footer).
    pub fn hit_test_sidebar(&self, x: f64, y: f64) -> Option<&'static str> {
        let sidebar = &self.layout.sidebar;
        let footer_y = sidebar.h - SIDEBAR_FOOTER_H;
        if x < sidebar.x as f64 || x >= (sidebar.x + sidebar.w) as f64 {
            return None;
        }
        for item in &self.layout.sidebar_items {
            if item.is_header {
                continue;
            }
            if y >= item.y as f64
                && y < (item.y + item.h) as f64
                && (y as f32) < footer_y
            {
                return Some(item.tool_id);
            }
        }
        None
    }

    /// Open a file dialog for the current tool's drop zone.
    pub fn open_file_dialog(&mut self) {
        let tool = match tools::find_tool(self.active_tool) {
            Some(t) => t,
            None => return,
        };

        let tool_id = self.active_tool;
        let ds = self.drop_state.entry(tool_id).or_insert_with(DropZoneState::new);
        if ds.dialog_pending {
            return;
        }
        ds.dialog_pending = true;

        let tx = self.file_tx.clone();
        let multiple = matches!(&tool.input, InputKind::File { multiple: true, .. });
        let accept: Vec<String> = match &tool.input {
            InputKind::File { accept, .. } => {
                accept.iter().map(|s| s.trim_start_matches('.').to_string()).collect()
            }
            _ => vec![],
        };
        let is_folder = matches!(&tool.input, InputKind::Folder);

        std::thread::spawn(move || {
            let result = if is_folder {
                rfd::FileDialog::new()
                    .pick_folder()
                    .map(|p| vec![p])
                    .unwrap_or_default()
            } else if multiple {
                let mut dialog = rfd::FileDialog::new();
                if !accept.is_empty() && accept[0] != "*" {
                    let refs: Vec<&str> = accept.iter().map(|s| s.as_str()).collect();
                    dialog = dialog.add_filter("Files", &refs);
                }
                dialog.pick_files().unwrap_or_default()
            } else {
                let mut dialog = rfd::FileDialog::new();
                if !accept.is_empty() && accept[0] != "*" {
                    let refs: Vec<&str> = accept.iter().map(|s| s.as_str()).collect();
                    dialog = dialog.add_filter("Files", &refs);
                }
                dialog.pick_file().map(|p| vec![p]).unwrap_or_default()
            };
            let _ = tx.send((tool_id, result));
        });
    }

    pub fn poll_file_dialogs(&mut self) {
        while let Ok((tool_id, files)) = self.file_rx.try_recv() {
            let ds = self.drop_state.entry(tool_id).or_insert_with(DropZoneState::new);
            ds.dialog_pending = false;
            if !files.is_empty() {
                ds.files = files;
            }
        }
    }

    pub fn poll_jobs(&mut self) {
        while let Ok(event) = self.job_rx.try_recv() {
            match event {
                JobEvent::Progress { job_id, message, percent } => {
                    if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
                        job.message = message;
                        job.progress = percent;
                    }
                }
                JobEvent::Done { job_id, result } => {
                    if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
                        job.status = JobStatus::Done;
                        job.progress = Some(100.0);
                        let toast_msg = match &result {
                            JobResult::File(p) => {
                                let fname = std::path::Path::new(p)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(p);
                                format!("Done: {fname}")
                            }
                            JobResult::Files(ps) => format!("Done: {} files", ps.len()),
                            JobResult::Text(t) => {
                                let preview = if t.len() > 30 { &t[..30] } else { t };
                                format!("Result: {preview}")
                            }
                        };
                        job.message = toast_msg.clone();
                        job.result = Some(result);
                        self.ui_state.toasts.push(ToastKind::Success, toast_msg, 3500);
                    }
                }
                JobEvent::Error { job_id, message } => {
                    if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
                        job.status = JobStatus::Error;
                        job.message = message.clone();
                        self.ui_state.toasts.push(ToastKind::Error, message, 3500);
                    }
                }
            }
        }
    }

    pub fn execute_tool(&mut self) {
        let tool_id = self.active_tool;
        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let tool_label = tools::find_tool(tool_id)
            .map(|t| t.label)
            .unwrap_or(tool_id);

        match crate::dispatch::dispatch(
            tool_id,
            &self.form_state,
            &self.drop_state,
            job_id,
            self.job_tx.clone(),
            &self.rt_handle,
        ) {
            Ok(abort_handle) => {
                self.jobs.push(Job {
                    id: job_id,
                    tool_id,
                    status: JobStatus::Running,
                    message: format!("{tool_label}..."),
                    progress: None,
                    result: None,
                    abort_handle,
                });
            }
            Err(msg) => {
                self.jobs.push(Job {
                    id: job_id,
                    tool_id,
                    status: JobStatus::Error,
                    message: msg,
                    progress: None,
                    result: None,
                    abort_handle: None,
                });
            }
        }
    }

    pub fn latest_job_for_tool(&self, tool_id: &str) -> Option<&Job> {
        self.jobs.iter().rev().find(|j| {
            j.tool_id == tool_id
                && matches!(j.status, JobStatus::Done | JobStatus::Error)
        })
    }

    pub fn cancel_job(&mut self, job_id: u64) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            if job.status == JobStatus::Running {
                if let Some(h) = job.abort_handle.take() {
                    h.abort();
                }
                job.status = JobStatus::Error;
                job.message = "Cancelled".into();
            }
        }
    }

    fn open_result(&self, job_id: u64) {
        if let Some(job) = self.jobs.iter().find(|j| j.id == job_id) {
            match &job.result {
                Some(JobResult::File(path)) => {
                    let _ = open::that(path);
                }
                Some(JobResult::Files(paths)) => {
                    if let Some(first) = paths.first() {
                        if let Some(parent) = std::path::Path::new(first).parent() {
                            let _ = open::that(parent);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn copy_result(&self, job_id: u64) {
        if let Some(job) = self.jobs.iter().find(|j| j.id == job_id) {
            if let Some(JobResult::Text(text)) = &job.result {
                if let Ok(mut clip) = arboard::Clipboard::new() {
                    let _ = clip.set_text(text);
                }
            }
        }
    }
}

impl AppDelegate for App {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        self.text = Some(TextRenderer::new(gpu).expect("failed to create text renderer"));
        tracing::info!("dwnldr GUI initialized");
    }

    fn on_redraw(
        &mut self,
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        _perf: &esox_platform::perf::PerfMonitor,
    ) {
        self.poll_file_dialogs();
        self.poll_jobs();

        // Update theme transition.
        if let Some(ref transition) = self.theme_transition {
            self.theme = transition.current();
            if transition.is_done() {
                self.theme_transition = None;
            }
        }

        self.ui_state.update_blink(self.theme.cursor_blink_ms);

        // Reset deferred action flags.
        self.needs_file_dialog = false;
        self.needs_execute = false;
        self.result_action = None;

        render::draw_frame(self, gpu, resources, frame);

        // Process deferred actions from the Ui pass.
        if self.needs_file_dialog {
            self.open_file_dialog();
        }
        if self.needs_execute {
            self.execute_tool();
        }
        match self.result_action.take() {
            Some(ResultAction::Open(job_id)) => self.open_result(job_id),
            Some(ResultAction::Copy(job_id)) => self.copy_result(job_id),
            None => {}
        }
    }

    fn on_key(&mut self, event: &KeyEvent, modifiers: ModifiersState) {
        use winit::keyboard::{Key, NamedKey};

        if !event.state.is_pressed() {
            return;
        }

        let ctrl = modifiers.control_key();

        // App-level shortcuts first.
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                if self.ui_state.overlay.is_some() {
                    self.ui_state.overlay = None;
                } else if self.ui_state.focused.is_some() {
                    self.ui_state.focused = None;
                } else if let Some(job_id) = self
                    .jobs
                    .iter()
                    .rev()
                    .find(|j| j.status == JobStatus::Running)
                    .map(|j| j.id)
                {
                    self.cancel_job(job_id);
                } else {
                    self.should_exit = true;
                }
                return;
            }
            // Tab focus cycling is now handled automatically by esox_ui.
            Key::Named(NamedKey::Enter) if ctrl => {
                self.execute_tool();
                return;
            }
            // Sidebar keyboard navigation — when no widget is focused.
            Key::Named(NamedKey::ArrowUp) if self.ui_state.focused.is_none() => {
                self.sidebar_navigate(-1);
                return;
            }
            Key::Named(NamedKey::ArrowDown) if self.ui_state.focused.is_none() => {
                self.sidebar_navigate(1);
                return;
            }
            Key::Named(NamedKey::Enter) if self.ui_state.focused.is_none() => {
                if let Some(idx) = self.sidebar_focus_index {
                    let tool_items: Vec<_> = self.layout.sidebar_items.iter()
                        .filter(|i| !i.is_header)
                        .collect();
                    if let Some(item) = tool_items.get(idx) {
                        self.active_tool = item.tool_id;
                        self.id_map = IdMap::build(item.tool_id);
                        self.ui_state.scroll_offsets.remove(&id!("content_scroll"));
                        self.ui_state.overlay = None;
                    }
                }
                return;
            }
            // Clipboard operations — handled at app level for arboard access.
            Key::Character(ch) if ctrl && ch.as_str() == "c" => {
                if let Some(focused_id) = self.ui_state.focused {
                    // Find which form field this maps to and copy selection.
                    if let Some((tool, key)) = self.focused_form_key(focused_id) {
                        let state = self.form_state.get_or_create(tool, key);
                        if let Some(selected) = state.selected_text() {
                            let selected = selected.to_string();
                            if let Ok(mut clip) = arboard::Clipboard::new() {
                                let _ = clip.set_text(&selected);
                            }
                        }
                    }
                }
                return;
            }
            Key::Character(ch) if ctrl && ch.as_str() == "v" => {
                if let Some(focused_id) = self.ui_state.focused {
                    if let Some((tool, key)) = self.focused_form_key(focused_id) {
                        if let Ok(mut clip) = arboard::Clipboard::new() {
                            if let Ok(pasted) = clip.get_text() {
                                let clean: String =
                                    pasted.chars().filter(|c| *c != '\n' && *c != '\r').collect();
                                self.form_state.get_or_create(tool, key).insert_str(&clean);
                                self.ui_state.reset_blink();
                            }
                        }
                    }
                }
                return;
            }
            _ => {}
        }

        // Forward everything else to esox_ui for widget processing.
        self.ui_state.process_key(event.clone(), modifiers);
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.vw = width as f32;
        self.vh = height as f32;
        self.layout = LayoutTree::new(self.vw, self.vh);
    }

    fn on_mouse(&mut self, event: MouseInputEvent) {
        match event {
            MouseInputEvent::Moved { x, y } => {
                self.hovered_item = self.hit_test_sidebar(x, y);
                self.hovered_toggle = self.hit_test_theme_toggle(x, y);
                self.ui_state.process_mouse_move(
                    x as f32,
                    y as f32,
                    self.theme.item_height,
                    self.theme.dropdown_gap,
                );
            }
            MouseInputEvent::Press { x, y, button: 0 } => {
                // Theme toggle footer.
                if self.hit_test_theme_toggle(x, y) {
                    self.toggle_theme();
                    return;
                }

                // Sidebar tool click — handle at app level.
                if let Some(tool_id) = self.hit_test_sidebar(x, y) {
                    self.active_tool = tool_id;
                    self.id_map = IdMap::build(tool_id);
                    self.ui_state.focused = None;
                    self.ui_state.scroll_offsets.remove(&id!("content_scroll"));
                    self.ui_state.overlay = None;
                    // Update sidebar_focus_index.
                    self.sidebar_focus_index = self.layout.sidebar_items.iter()
                        .position(|item| !item.is_header && item.tool_id == tool_id);
                    return;
                }

                // Forward to esox_ui for content widgets.
                self.ui_state.process_mouse_click(x as f32, y as f32);
            }
            MouseInputEvent::Scroll { x, y, delta_y, .. } => {
                if x >= self.layout.content.x as f64 {
                    self.ui_state.process_scroll(x as f32, y as f32, -delta_y);
                }
            }
            _ => {}
        }
    }

    fn on_scale_changed(&mut self, scale_factor: f64, _gpu: &GpuContext) {
        self.scale_factor = scale_factor as f32;
        let base = if self.theme_is_dark { Theme::dark() } else { Theme::light() };
        self.theme = base.scaled(self.scale_factor);
        self.layout = LayoutTree::new_scaled(self.vw, self.vh, self.scale_factor);
    }

    fn on_paste(&mut self, text: &str) {
        if let Some(focused_id) = self.ui_state.focused {
            if let Some((tool, key)) = self.focused_form_key(focused_id) {
                let clean: String = text.chars().filter(|c| *c != '\n' && *c != '\r').collect();
                self.form_state.get_or_create(tool, key).insert_str(&clean);
                self.ui_state.reset_blink();
            }
        }
    }

    fn on_ime_commit(&mut self, text: &str) {
        if let Some(focused_id) = self.ui_state.focused {
            if let Some((tool, key)) = self.focused_form_key(focused_id) {
                for c in text.chars() {
                    if !c.is_control() {
                        self.form_state.get_or_create(tool, key).insert_char(c);
                    }
                }
                self.ui_state.reset_blink();
            }
        }
    }

    fn on_copy(&mut self) -> Option<String> {
        let focused_id = self.ui_state.focused?;
        let (tool, key) = self.focused_form_key(focused_id)?;
        let state = self.form_state.get_or_create(tool, key);
        state.selected_text().map(|s| s.to_string())
    }

    fn should_exit(&self) -> bool {
        self.should_exit
    }

    fn on_close(&mut self, window: &winit::window::Window) {
        let size = window.inner_size();
        let pos = window.outer_position().unwrap_or_default();
        let state = crate::WindowState {
            width: size.width,
            height: size.height,
            x: pos.x,
            y: pos.y,
        };
        let path = crate::state_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(toml_str) = toml::to_string_pretty(&state) {
            let _ = std::fs::write(&path, toml_str);
        }
    }

    fn needs_continuous_redraw(&self) -> bool {
        let has_running_job = self.jobs.iter().any(|j| j.status == JobStatus::Running);
        has_running_job
            || self.theme_transition.is_some()
            || self.ui_state.needs_continuous_redraw()
    }

    fn take_clear_color(&mut self) -> Option<[f32; 4]> {
        Some([
            self.theme.bg_base.r,
            self.theme.bg_base.g,
            self.theme.bg_base.b,
            self.theme.bg_base.a,
        ])
    }

    fn cursor_icon(&self, x: f64, y: f64) -> winit::window::CursorIcon {
        if x < SIDEBAR_WIDTH as f64 {
            if self.hit_test_theme_toggle(x, y) || self.hit_test_sidebar(x, y).is_some() {
                return winit::window::CursorIcon::Pointer;
            }
            return winit::window::CursorIcon::Default;
        }
        self.ui_state.cursor_icon(x as f32, y as f32)
    }

    fn on_file_dropped(&mut self, path: PathBuf) {
        let tool_id = self.active_tool;
        let ds = self.drop_state.entry(tool_id).or_insert_with(DropZoneState::new);
        ds.files.push(path);
    }

    fn on_file_hover(&mut self, _path: Option<PathBuf>, _x: f64, _y: f64) {}
}

// ── Helpers ──

impl App {
    /// Map an esox_ui widget u64 ID to a (tool_id, field_key) pair for form state access.
    fn focused_form_key(&self, ui_id: u64) -> Option<(&'static str, &'static str)> {
        self.id_map.table.get(&ui_id).copied()
    }

    /// Navigate sidebar by `delta` (+1 = down, -1 = up), skipping headers.
    fn sidebar_navigate(&mut self, delta: i32) {
        let tool_count = self.layout.sidebar_items.iter()
            .filter(|i| !i.is_header)
            .count();
        if tool_count == 0 {
            return;
        }
        let current = self.sidebar_focus_index.unwrap_or(0);
        let next = if delta > 0 {
            if current + 1 < tool_count { current + 1 } else { 0 }
        } else if current > 0 {
            current - 1
        } else {
            tool_count - 1
        };
        self.sidebar_focus_index = Some(next);
    }
}
