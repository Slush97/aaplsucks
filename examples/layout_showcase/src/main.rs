//! Layout showcase — demonstrates flex wrap, responsive layout, nested scrolling,
//! styled components, paragraph widget, truncation modes, and debug overlay.

use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::{AppDelegate, Clipboard, MouseInputEvent};
use esox_ui::{
    id, ClipboardProvider, FlexItem, FlexWrap, Rect, StyleState, TextRenderer, Theme,
    TruncationMode, UiState, WidgetStyle, WidthClass,
};

struct PlatformClipboard;

impl ClipboardProvider for PlatformClipboard {
    fn read_text(&self) -> Option<String> {
        Clipboard::read(0).ok()
    }
    fn write_text(&self, text: &str) {
        let _ = Clipboard::write(text);
    }
}

struct App {
    ui_state: UiState,
    text: Option<TextRenderer>,
    theme: Theme,
    viewport: (u32, u32),
}

impl App {
    fn new() -> Self {
        let mut ui_state = UiState::new();
        ui_state.clipboard = Some(Box::new(PlatformClipboard));
        ui_state.debug_overlay = true;

        Self {
            ui_state,
            text: None,
            theme: Theme::dark(),
            viewport: (900, 700),
        }
    }
}

impl AppDelegate for App {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        self.text = Some(TextRenderer::new(gpu).expect("font"));
    }

    fn on_redraw(
        &mut self,
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        _perf: &esox_platform::perf::PerfMonitor,
    ) {
        let text = self.text.as_mut().unwrap();
        let viewport = Rect::new(0.0, 0.0, self.viewport.0 as f32, self.viewport.1 as f32);
        let mut ui = esox_ui::Ui::begin(frame, gpu, resources, text, &mut self.ui_state, &self.theme, viewport);

        let scroll_h = self.viewport.1 as f32;
        ui.scrollable(id!("page"), scroll_h, |ui| {
        ui.padding(16.0, |ui| {
            ui.heading("Layout Showcase");
            ui.add_space(8.0);

            // Responsive layout.
            let width_class = ui.width_class();
            let class_name = match width_class {
                WidthClass::Compact => "Compact",
                WidthClass::Medium => "Medium",
                WidthClass::Expanded => "Expanded",
            };
            ui.label(&format!("Width class: {class_name} ({:.0}px)", ui.region_width()));

            let debug_status = if ui.is_debug_overlay() { "ON" } else { "OFF" };
            ui.label(&format!("Debug overlay: {debug_status}"));

            ui.add_space(12.0);

            // --- Paragraph widget ---
            ui.header_label("PARAGRAPH WIDGET");
            ui.paragraph(
                id!("intro_paragraph"),
                "This paragraph widget automatically wraps text and responds to hover. \
                 It uses the theme's line_spacing for inter-line gaps and measures \
                 its height with measure_text_wrapped().",
            );

            ui.add_space(12.0);

            // --- Truncation modes ---
            ui.header_label("TRUNCATION MODES");
            let long_text = "This is a very long text that will be truncated in different modes to show the three available truncation options";
            ui.label("End:");
            ui.label_truncated_mode(long_text, TruncationMode::End);
            ui.label("Start:");
            ui.label_truncated_mode(long_text, TruncationMode::Start);
            ui.label("Middle:");
            ui.label_truncated_mode(long_text, TruncationMode::Middle);

            ui.add_space(12.0);

            // --- Style derivation ---
            ui.header_label("STYLE STATE DERIVATION");
            ui.row(|ui| {
                let base = ui.theme().accent;
                for (i, &state) in [StyleState::Normal, StyleState::Hovered, StyleState::Active, StyleState::Disabled].iter().enumerate() {
                    let bg = ui.theme().derive_bg(base, state);
                    let style = WidgetStyle {
                        bg: Some(bg),
                        ..Default::default()
                    };
                    ui.with_style(style, |ui| {
                        let name = match state {
                            StyleState::Normal => "Normal",
                            StyleState::Hovered => "Hovered",
                            StyleState::Focused => "Focused",
                            StyleState::Active => "Active",
                            StyleState::Disabled => "Disabled",
                        };
                        ui.button(id!("style_demo").wrapping_add(i as u64), name);
                    });
                }
            });

            ui.add_space(12.0);

            // --- Flex row with grow ---
            ui.header_label("FLEX ROW WITH GROW");
            ui.flex_row().gap(8.0).show_flex(id!("flex_grow"), |flex| {
                flex.item(FlexItem::default().grow(1.0), |ui| {
                    ui.button(id!("grow1"), "Grow 1");
                });
                flex.item_default(|ui| {
                    ui.button(id!("fixed"), "Fixed");
                });
                flex.item(FlexItem::default().grow(2.0), |ui| {
                    ui.button(id!("grow2"), "Grow 2");
                });
            });

            ui.add_space(12.0);

            // --- Flex wrap ---
            ui.header_label("FLEX WRAP");
            ui.flex_row().gap(8.0).wrap(FlexWrap::Wrap).show(|ui| {
                for i in 0..12 {
                    ui.button(id!("wrap_btn").wrapping_add(i), &format!("Item {i}"));
                }
            });

            ui.add_space(12.0);

            // --- Nested scrolling ---
            ui.header_label("NESTED SCROLLING");
            ui.scrollable(id!("outer_scroll"), 200.0, |ui| {
                for i in 0..5 {
                    ui.label(&format!("Outer item {i}"));
                }
                ui.muted_label("Inner scrollable (propagates at limit):");
                ui.scrollable(id!("inner_scroll"), 100.0, |ui| {
                    for j in 0..20 {
                        ui.label(&format!("  Inner item {j}"));
                    }
                });
                for i in 5..15 {
                    ui.label(&format!("Outer item {i}"));
                }
            });

            ui.add_space(12.0);

            // --- Text color inheritance ---
            ui.header_label("TEXT COLOR INHERITANCE");
            let green_style = WidgetStyle {
                text_color: Some(esox_gfx::Color::new(0.2, 0.9, 0.3, 1.0)),
                ..Default::default()
            };
            ui.with_style(green_style, |ui| {
                ui.label("This text inherits green via text_color");
                ui.label("So does this one");
            });
        });
        });

        ui.finish();
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
                    x as f32, y as f32,
                    self.theme.item_height, self.theme.dropdown_gap,
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

    fn on_scale_changed(&mut self, scale_factor: f64, _gpu: &GpuContext) {
        let factor = scale_factor as f32;
        self.ui_state.scale_factor = factor;
        self.theme = Theme::dark().scaled(factor);
    }

    fn needs_redraw(&self) -> bool {
        self.ui_state.needs_redraw()
    }

    fn needs_continuous_redraw(&self) -> bool {
        self.ui_state.needs_continuous_redraw()
    }

    fn cursor_icon(&self, x: f64, y: f64) -> esox_platform::esox_input::CursorIcon {
        self.ui_state.cursor_icon(x as f32, y as f32)
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = esox_platform::config::PlatformConfig {
        window: esox_platform::config::WindowConfig {
            title: "Layout Showcase".into(),
            width: Some(900),
            height: Some(700),
            ..Default::default()
        },
        ..Default::default()
    };

    esox_platform::run(config, Box::new(App::new())).unwrap();
}
