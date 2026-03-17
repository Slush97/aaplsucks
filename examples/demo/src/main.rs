use std::collections::HashMap;

use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, Clipboard, MouseInputEvent};
use esox_ui::{
    fnv1a_mix, id, ClipboardProvider, ColumnWidth, FieldStatus, ImageCache, ImageHandle,
    InputState, ModalAction, Rect, RichText, SelectState, TabState, TableColumn, TableState,
    TextRenderer, Theme, TreeState, UiState, VirtualScrollState,
};

/// Clipboard provider backed by the platform clipboard.
struct PlatformClipboard;

impl ClipboardProvider for PlatformClipboard {
    fn read_text(&self) -> Option<String> {
        Clipboard::read(0).ok()
    }

    fn write_text(&self, text: &str) {
        let _ = Clipboard::write(text);
    }
}

struct DemoApp {
    ui_state: UiState,
    text: Option<TextRenderer>,
    base_theme: Theme,
    theme: Theme,
    viewport: (u32, u32),
    checkbox_states: HashMap<u64, InputState>,
    slider_state: InputState,
    text_input_state: InputState,
    text_area_state: InputState,
    text_area_disabled_state: InputState,
    text_area_wrapped_state: InputState,
    select_state: SelectState,
    radio_state: InputState,
    progress: f32,
    // Tier 3 state.
    tab_state: TabState,
    table_state: TableState,
    tree_state: TreeState,
    virtual_scroll_state: VirtualScrollState,
    drag_drop_log: String,
    // Image widget.
    image_cache: Option<ImageCache>,
    test_image: Option<ImageHandle>,
    // Modal.
    modal_open: bool,
    confirm_modal_open: bool,
    confirm_result: String,
    // New widgets.
    toggle_state: InputState,
    number_value: f64,
    combobox_selected: Option<usize>,
}

impl DemoApp {
    fn new() -> Self {
        let mut slider = InputState::new();
        slider.text = "50".into();

        let mut ui_state = UiState::new();
        ui_state.clipboard = Some(Box::new(PlatformClipboard));
        Self {
            ui_state,
            text: None,
            base_theme: Theme::dark(),
            theme: Theme::dark(),
            viewport: (650, 1400),
            checkbox_states: HashMap::new(),
            slider_state: slider,
            text_input_state: InputState::new(),
            text_area_state: InputState::new(),
            text_area_disabled_state: {
                let mut s = InputState::new();
                s.text = "This text area is disabled.\nYou cannot edit it.".into();
                s
            },
            text_area_wrapped_state: {
                let mut s = InputState::new();
                s.text = "This text area uses soft word wrap. Long lines will break at word boundaries instead of extending beyond the widget. Try typing a long sentence to see it in action.".into();
                s
            },
            select_state: SelectState::new(),
            radio_state: {
                let mut s = InputState::new();
                s.text = "0".into();
                s
            },
            progress: 0.7,
            tab_state: TabState::new(),
            table_state: TableState::new(),
            tree_state: {
                let mut t = TreeState::new();
                t.expanded.insert(id!("tree_root"));
                t
            },
            virtual_scroll_state: VirtualScrollState::new(10_000),
            drag_drop_log: String::new(),
            image_cache: None,
            test_image: None,
            modal_open: false,
            confirm_modal_open: false,
            confirm_result: String::new(),
            toggle_state: InputState::new(),
            number_value: 42.0,
            combobox_selected: None,
        }
    }
}

impl AppDelegate for DemoApp {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        match TextRenderer::new(gpu) {
            Ok(tr) => self.text = Some(tr),
            Err(e) => eprintln!("failed to initialize text renderer: {e}"),
        }

        // Generate a small test image (8x8 gradient) for the image widget demo.
        let mut cache = ImageCache::new(gpu);
        let mut pixels = Vec::with_capacity(8 * 8 * 4);
        for y in 0..8u8 {
            for x in 0..8u8 {
                pixels.extend_from_slice(&[x * 32, y * 32, 128, 255]);
            }
        }
        // Encode as PNG in memory.
        let mut png_buf = std::io::Cursor::new(Vec::new());
        image::write_buffer_with_format(
            &mut png_buf,
            &pixels,
            8,
            8,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .ok();
        self.test_image = cache.load_from_bytes(png_buf.get_ref(), gpu);
        self.image_cache = Some(cache);
    }

    fn on_redraw(
        &mut self,
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        perf: &esox_platform::perf::PerfMonitor,
    ) {
        self.ui_state.update_blink(self.theme.cursor_blink_ms);

        // Animate progress bar.
        self.progress += 0.002;
        if self.progress > 1.0 {
            self.progress = 0.0;
        }

        let text = self.text.as_mut().unwrap();
        let vp = Rect::new(0.0, 0.0, self.viewport.0 as f32, self.viewport.1 as f32);
        let mut ui = esox_ui::Ui::begin(frame, gpu, resources, text, &mut self.ui_state, &self.theme, vp);

        let scroll_h = self.viewport.1 as f32;
        ui.scrollable(id!("page_scroll"), scroll_h, |ui| {
        ui.padding(24.0, |ui| {
            ui.max_width(600.0, |ui| {
                ui.heading("esox_ui Demo");
                ui.add_space(8.0);

                // ── Flex Columns ──
                ui.header_label("FLEX COLUMNS");
                ui.columns_spaced(12.0, &[2.0, 1.0], |ui, col| {
                    match col {
                        0 => {
                            ui.label("2/3 width column");
                            ui.button(id!("col_btn_left"), "Left");
                        }
                        1 => {
                            ui.label("1/3 width column");
                            ui.button(id!("col_btn_right"), "Right");
                        }
                        _ => {}
                    }
                });
                ui.add_space(8.0);
                ui.columns_spaced(8.0, &[1.0, 1.0, 1.0], |ui, col| {
                    let label = format!("Col {}", col + 1);
                    let btn_id = fnv1a_mix(id!("col3_btn"), col as u64);
                    ui.button(btn_id, &label);
                });
                ui.add_space(16.0);

                // ── New Widgets ──
                ui.header_label("TOGGLE");
                ui.toggle(id!("demo_toggle"), &mut self.toggle_state, "Dark mode");
                ui.add_space(16.0);

                ui.header_label("NUMBER INPUT");
                ui.number_input_clamped(id!("demo_num"), &mut self.number_value, 1.0, 0.0, 100.0);
                ui.add_space(16.0);

                ui.header_label("COMBOBOX");
                ui.combobox(id!("demo_combo"), &["Apple", "Banana", "Cherry", "Date", "Elderberry"], &mut self.combobox_selected);
                ui.add_space(16.0);

                ui.header_label("SPINNER");
                ui.row(|ui| {
                    ui.spinner();
                    ui.label(" Loading...");
                });
                ui.add_space(16.0);

                ui.header_label("CHIPS & BADGES");
                ui.row(|ui| {
                    ui.chip(id!("chip1"), "Tag A");
                    ui.chip(id!("chip2"), "Tag B");
                    ui.badge(3);
                    ui.badge_dot();
                });
                ui.add_space(16.0);

                ui.header_label("HYPERLINK");
                ui.hyperlink(id!("demo_link"), "Visit example.com", "https://example.com");
                ui.add_space(16.0);

                ui.header_label("COLLAPSING");
                ui.collapsing_header(id!("collapse1"), "Click to expand", false, |ui| {
                    ui.label("Hidden content revealed!");
                    ui.muted_label("This section is collapsible.");
                });
                ui.add_space(16.0);

                ui.header_label("CONTAINER / CARD");
                ui.card(|ui| {
                    ui.label("Content inside a card.");
                    ui.muted_label("Cards have elevated backgrounds.");
                });
                ui.add_space(16.0);

                ui.header_label("FORM FIELD");
                ui.form_field("Username", FieldStatus::None, "", |ui| {
                    ui.text_input(id!("form_input"), &mut self.text_input_state, "Enter username...")
                });
                ui.add_space(16.0);

                ui.header_label("EMPTY STATE");
                ui.empty_state("No items to display.");
                ui.add_space(16.0);

                ui.header_label("STATUS BAR");
                ui.status_bar("Ready", "Ln 42, Col 8");
                ui.add_space(16.0);

                // ── Tabs ──
                ui.header_label("TABS");
                ui.tabs(id!("demo_tabs"), &mut self.tab_state, &["Overview", "Settings", "About"], |ui, selected| {
                    ui.add_space(8.0);
                    match selected {
                        0 => {
                            ui.label("This is the Overview tab.");
                            ui.muted_label("Switch tabs with mouse or arrow keys.");
                        }
                        1 => {
                            ui.label("Settings would go here.");
                            ui.button(id!("settings_btn"), "Apply");
                        }
                        2 => {
                            ui.label("esox_ui Tier 3 demo application.");
                        }
                        _ => {}
                    }
                });
                ui.add_space(16.0);

                // ── Rich Text ──
                ui.header_label("RICH TEXT");
                let accent = ui.theme().accent;
                let green = ui.theme().green;
                let red = ui.theme().red;
                ui.rich_label(
                    &RichText::new()
                        .span("Normal ")
                        .bold("bold ")
                        .colored("accent ", accent)
                        .colored_bold("bold+green", green),
                );
                ui.add_space(4.0);
                ui.rich_label_wrapped(
                    &RichText::new()
                        .span("This is a ")
                        .bold("wrapped rich text ")
                        .span("paragraph. It supports ")
                        .colored("colored", accent)
                        .span(" and ")
                        .colored_bold("bold colored", red)
                        .span(" spans that wrap correctly across multiple lines when the text exceeds the available width."),
                );
                ui.add_space(16.0);

                // ── Text Input ──
                ui.header_label("TEXT INPUT");
                ui.text_input(id!("text_input"), &mut self.text_input_state, "Single-line input...");
                ui.add_space(16.0);

                // ── Select / Dropdown ──
                ui.header_label("SELECT");
                ui.select(id!("demo_select"), &mut self.select_state, &["Option A", "Option B", "Option C", "Option D"]);
                ui.add_space(16.0);

                // ── Slider ──
                ui.header_label("SLIDER");
                ui.slider(id!("demo_slider"), &mut self.slider_state, 0.0, 100.0);
                ui.add_space(16.0);

                // ── Image ──
                ui.header_label("IMAGE");
                if let (Some(cache), Some(handle)) = (&self.image_cache, self.test_image) {
                    ui.image(id!("test_image"), cache, handle, 64.0, 64.0);
                }
                ui.add_space(16.0);

                // ── Separator ──
                ui.header_label("SEPARATOR");
                ui.label("Above the separator");
                ui.separator();
                ui.label("Below the separator");
                ui.add_space(16.0);

                // ── Modals ──
                ui.header_label("MODALS");
                if ui.button(id!("open_modal"), "Open Modal").clicked {
                    self.modal_open = true;
                }
                ui.modal(id!("demo_modal"), &mut self.modal_open, "Demo Modal", 400.0, |ui| {
                    ui.label("This is a custom modal dialog.");
                    ui.label("Press Escape or click outside to close.");
                });
                if ui.button(id!("open_confirm"), "Confirm Dialog").clicked {
                    self.confirm_modal_open = true;
                }
                let action = ui.modal_confirm(id!("confirm_modal"), &mut self.confirm_modal_open, "Confirm", "Are you sure?");
                if action == ModalAction::Confirm {
                    self.confirm_result = "Confirmed!".into();
                } else if action == ModalAction::Cancel {
                    self.confirm_result = "Cancelled.".into();
                }
                if !self.confirm_result.is_empty() {
                    ui.muted_label(&self.confirm_result);
                }
                ui.add_space(16.0);

                // ── Text Area Word Wrap ──
                ui.header_label("TEXT AREA (WORD WRAP)");
                ui.text_area_wrapped(id!("text_area_wrap"), &mut self.text_area_wrapped_state, 5, "Type here with word wrap...");
                ui.add_space(16.0);

                // ── Virtual Scroll ──
                ui.header_label("VIRTUAL SCROLL (10,000 ITEMS)");
                self.virtual_scroll_state.item_count = 10_000;
                ui.virtual_scroll(id!("vscroll"), &mut self.virtual_scroll_state, 28.0, 200.0, |ui, i| {
                    let label = format!("Item #{}", i);
                    ui.label(&label);
                });
                ui.add_space(16.0);

                // ── Table ──
                ui.header_label("TABLE");
                let columns = [
                    TableColumn::new("#", ColumnWidth::Fixed(40.0)).not_sortable(),
                    TableColumn::new("Name", ColumnWidth::Weight(2.0)),
                    TableColumn::new("Status", ColumnWidth::Weight(1.0)),
                ];
                ui.table(id!("demo_table"), &mut self.table_state, &columns, 50, 8, |ui, row, col| {
                    match col {
                        0 => { let s = format!("{}", row); ui.label(&s); }
                        1 => { let s = format!("Item {}", row); ui.label(&s); }
                        2 => {
                            if row % 3 == 0 {
                                ui.label_colored("Active", ui.theme().green);
                            } else if row % 3 == 1 {
                                ui.label_colored("Pending", ui.theme().amber);
                            } else {
                                ui.label_colored("Error", ui.theme().red);
                            }
                        }
                        _ => {}
                    }
                });
                ui.add_space(16.0);

                // ── Tree ──
                ui.header_label("TREE");
                let r = ui.tree_node(id!("tree_root"), &mut self.tree_state, "Root", true);
                ui.animated_tree_indent(id!("tree_root_anim"), r.expanded, |ui| {
                    ui.tree_node(id!("tree_file1"), &mut self.tree_state, "file1.rs", false);
                    ui.tree_node(id!("tree_file2"), &mut self.tree_state, "file2.rs", false);
                    let r2 = ui.tree_node(id!("tree_src"), &mut self.tree_state, "src/", true);
                    ui.animated_tree_indent(id!("tree_src_anim"), r2.expanded, |ui| {
                        ui.tree_node(id!("tree_main"), &mut self.tree_state, "main.rs", false);
                        ui.tree_node(id!("tree_lib"), &mut self.tree_state, "lib.rs", false);
                    });
                    ui.tree_node(id!("tree_cargo"), &mut self.tree_state, "Cargo.toml", false);
                });
                ui.add_space(16.0);

                // ── Drag & Drop ──
                ui.header_label("DRAG & DROP");
                ui.columns_spaced(12.0, &[1.0, 1.0], |ui, col| {
                    match col {
                        0 => {
                            ui.label("Drag source:");
                            let btn_id = id!("drag_src");
                            ui.button(btn_id, "Drag me");
                            ui.drag_source(btn_id, 42);
                        }
                        1 => {
                            ui.label("Drop target:");
                            let hovering = ui.drop_target(Rect::new(
                                ui.cursor_x(), ui.cursor_y(), ui.region_width(), 36.0,
                            )).is_some();
                            if hovering {
                                ui.label_colored("[ Drop here! ]", ui.theme().accent);
                            } else {
                                ui.muted_label("[ Drop here ]");
                            }
                            let target_rect = Rect::new(
                                ui.cursor_x(), ui.cursor_y() - 36.0 - ui.theme().padding,
                                ui.region_width(), 36.0,
                            );
                            if let Some(payload) = ui.accept_drop(target_rect) {
                                self.drag_drop_log = format!("Dropped payload: {payload}");
                            }
                        }
                        _ => {}
                    }
                });
                if !self.drag_drop_log.is_empty() {
                    ui.muted_label(&self.drag_drop_log);
                }
                ui.add_space(16.0);

                // ── Existing Tier 1+2 sections ──
                ui.header_label("TEXT WRAPPING");
                ui.label_wrapped(
                    "This is a long paragraph that demonstrates word wrapping. The text will \
                     automatically break at word boundaries when it exceeds the available width."
                );
                ui.add_space(16.0);

                ui.header_label("ENABLED WIDGETS");
                ui.button(id!("enabled_btn"), "Enabled Button");
                ui.tooltip(id!("enabled_btn"), "Click to perform action");

                let cb_id2 = id!("enabled_cb");
                let cb_state2 = self.checkbox_states.entry(cb_id2).or_default();
                ui.checkbox(cb_id2, cb_state2, "Enabled Checkbox");
                ui.add_space(16.0);

                ui.header_label("RADIO BUTTONS");
                ui.radio(id!("radio_color_r"), &mut self.radio_state, 0, "Red");
                ui.radio(id!("radio_color_g"), &mut self.radio_state, 1, "Green");
                ui.radio(id!("radio_color_b"), &mut self.radio_state, 2, "Blue");
                ui.add_space(16.0);

                ui.header_label("PROGRESS BARS");
                ui.label("Accent:");
                ui.progress_bar(self.progress);
                ui.add_space(4.0);
                ui.label("Green:");
                ui.progress_bar_colored(0.85, ui.theme().green);
                ui.add_space(16.0);

                ui.header_label("TEXT AREA");
                ui.text_area(id!("text_area"), &mut self.text_area_state, 6, "Type multi-line text here...");
                ui.add_space(8.0);
                ui.muted_label("Disabled:");
                ui.text_area(id!("text_area_disabled"), &mut self.text_area_disabled_state, 3, "");
                ui.add_space(16.0);

                ui.header_label("SCROLLABLE");
                let btn_base = id!("btn");
                ui.scrollable(id!("main_scroll"), 200.0, |ui| {
                    for i in 0..15u64 {
                        let btn_id = fnv1a_mix(btn_base, i);
                        let label = format!("Button {i}");
                        ui.button(btn_id, &label);
                    }
                });
            });
        });
        }); // page_scroll

        ui.finish();

        // ── Performance overlay (top-right, drawn after UI so it's on top) ──
        let stats = perf.summary();
        let line_count = stats.lines().count();
        let overlay_h = line_count as f32 * 16.0 + 12.0;
        // Measure widest line.
        let mut max_w = 0.0f32;
        for line in stats.lines() {
            let w = text.measure_text(line, 12.0);
            if w > max_w { max_w = w; }
        }
        let overlay_w = max_w + 16.0;
        let overlay_x = self.viewport.0 as f32 - overlay_w - 4.0;
        // Background panel.
        frame.push(esox_gfx::QuadInstance {
            rect: [overlay_x, 4.0, overlay_w, overlay_h],
            uv: [0.0; 4],
            color: [0.0, 0.0, 0.0, 0.7],
            border_radius: [4.0, 4.0, 4.0, 4.0],
            sdf_params: [0.0; 4],
            flags: [0.0; 4],
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
        });
        // Text lines.
        for (i, line) in stats.lines().enumerate() {
            let y = 10.0 + i as f32 * 16.0;
            let w = text.measure_text(line, 12.0);
            text.draw_text(
                line,
                self.viewport.0 as f32 - w - 12.0,
                y,
                12.0,
                esox_gfx::Color::new(0.0, 1.0, 0.0, 0.9),
                frame,
                gpu,
                resources,
            );
        }
    }

    fn on_key(
        &mut self,
        event: &esox_platform::esox_input::KeyEvent,
        modifiers: esox_platform::esox_input::Modifiers,
    ) {
        self.ui_state.process_key(event.clone(), modifiers);
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.viewport = (width, height);
    }

    fn on_mouse(&mut self, event: MouseInputEvent) {
        match event {
            MouseInputEvent::Moved { x, y } => {
                self.ui_state.process_mouse_move(
                    x as f32,
                    y as f32,
                    self.theme.item_height,
                    self.theme.dropdown_gap,
                );
            }
            MouseInputEvent::Press { x, y, button: 0 } => {
                self.ui_state.process_mouse_click(x as f32, y as f32);
            }
            MouseInputEvent::Press { x, y, button: 2 } => {
                self.ui_state.process_right_click(x as f32, y as f32);
            }
            MouseInputEvent::Release { button: 0, .. } => {
                self.ui_state.process_mouse_release();
            }
            MouseInputEvent::Scroll { x, y, delta_y, .. } => {
                self.ui_state.process_scroll(x as f32, y as f32, delta_y);
            }
            _ => {}
        }
    }

    fn on_paste(&mut self, _text: &str) {}
    fn on_ime_commit(&mut self, text: &str) {
        self.ui_state.on_ime_commit(text.to_string());
    }
    fn on_ime_preedit(&mut self, text: String, cursor: Option<(usize, usize)>) {
        self.ui_state.on_ime_preedit(text, cursor);
    }
    fn on_ime_enabled(&mut self, enabled: bool) {
        self.ui_state.on_ime_enabled(enabled);
    }
    fn on_copy(&mut self) -> Option<String> { None }

    fn needs_redraw(&self) -> bool {
        self.ui_state.needs_redraw()
    }

    fn needs_continuous_redraw(&self) -> bool {
        self.ui_state.needs_continuous_redraw()
    }

    fn cursor_icon(&self, x: f64, y: f64) -> esox_platform::esox_input::CursorIcon {
        self.ui_state.cursor_icon(x as f32, y as f32)
    }

    fn on_scale_changed(&mut self, scale_factor: f64, _gpu: &GpuContext) {
        let factor = scale_factor as f32;
        self.ui_state.scale_factor = factor;
        self.theme = self.base_theme.scaled(factor);
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = PlatformConfig {
        window: WindowConfig {
            title: "esox_ui Demo".into(),
            width: Some(650),
            height: Some(1400),
            ..Default::default()
        },
        ..Default::default()
    };

    esox_platform::run(config, Box::new(DemoApp::new())).unwrap();
}
