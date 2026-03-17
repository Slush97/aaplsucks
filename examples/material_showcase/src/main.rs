use std::collections::HashMap;

use esox_gfx::{Color, Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, Clipboard, MouseInputEvent};
use esox_ui::{
    id, ClipboardProvider, ColumnWidth, FieldStatus, InputState, ModalAction, Rect, RichText,
    SelectState, TabState, TableColumn, TableState, TextRenderer, Theme, ThemeTransition,
    TreeState, UiState, VirtualScrollState, WidgetStyle,
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

struct MaterialShowcase {
    ui_state: UiState,
    text: Option<TextRenderer>,
    viewport: (u32, u32),

    // Theme
    is_dark: bool,
    base_light: Theme,
    base_dark: Theme,
    theme: Theme,
    transition: Option<ThemeTransition>,

    // Navigation
    tab_state: TabState,

    // Overview state
    upload_progress: f32,

    // Forms state
    name_input: InputState,
    email_input: InputState,
    bio_input: InputState,
    role_select: SelectState,
    country_combo: Option<usize>,
    experience_value: f64,
    font_size_value: f64,
    newsletter_cb: HashMap<u64, InputState>,
    notifications_toggle: InputState,
    priority_radio: InputState,

    // Data state
    table_state: TableState,
    tree_state: TreeState,
    vscroll_state: VirtualScrollState,

    // Components state
    modal_open: bool,
    confirm_open: bool,
    confirm_result: String,
}

impl MaterialShowcase {
    fn new() -> Self {
        let base_light = Theme::light();
        let base_dark = Theme::dark();
        let theme = base_light.clone();

        let mut ui_state = UiState::new();
        ui_state.clipboard = Some(Box::new(PlatformClipboard));

        let mut email = InputState::new();
        email.text = "bad-email@@".into();

        let mut priority = InputState::new();
        priority.text = "1".into();

        Self {
            ui_state,
            text: None,
            viewport: (900, 700),
            is_dark: false,
            base_light,
            base_dark,
            theme,
            transition: None,
            tab_state: TabState::new(),
            upload_progress: 0.0,
            name_input: InputState::new(),
            email_input: email,
            bio_input: InputState::new(),
            role_select: SelectState::new(),
            country_combo: None,
            experience_value: 5.0,
            font_size_value: 14.0,
            newsletter_cb: HashMap::new(),
            notifications_toggle: InputState::new(),
            priority_radio: priority,
            table_state: TableState::new(),
            tree_state: {
                let mut t = TreeState::new();
                t.expanded.insert(id!("tree_workspace"));
                t
            },
            vscroll_state: VirtualScrollState::new(10_000),
            modal_open: false,
            confirm_open: false,
            confirm_result: String::new(),
        }
    }
}

impl AppDelegate for MaterialShowcase {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        match TextRenderer::new(gpu) {
            Ok(tr) => self.text = Some(tr),
            Err(e) => eprintln!("failed to initialize text renderer: {e}"),
        }
    }

    fn on_redraw(
        &mut self,
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        _perf: &esox_platform::perf::PerfMonitor,
    ) {
        // Update theme transition.
        if let Some(ref t) = self.transition {
            self.theme = t.current();
            if t.is_done() {
                self.theme = if self.is_dark {
                    self.base_dark.clone()
                } else {
                    self.base_light.clone()
                };
                self.transition = None;
            }
        }

        self.ui_state.update_blink(self.theme.cursor_blink_ms);

        // Animate upload progress.
        self.upload_progress += 1.0 / 60.0 * 0.3;
        if self.upload_progress > 1.0 {
            self.upload_progress = 0.0;
        }

        // Snapshot mutable state into locals so closures don't need &mut self.
        let upload_progress = self.upload_progress;
        let confirm_result = self.confirm_result.clone();

        let text = self.text.as_mut().unwrap();
        let vp = Rect::new(0.0, 0.0, self.viewport.0 as f32, self.viewport.1 as f32);
        let mut ui = esox_ui::Ui::begin(
            frame, gpu, resources, text, &mut self.ui_state, &self.theme, vp,
        );

        let scroll_h = self.viewport.1 as f32;

        ui.scrollable(id!("page_scroll"), scroll_h, |ui| {
            // ── App bar ──
            ui.surface(|ui| {
                ui.row(|ui| {
                    ui.heading("Material Showcase");
                    ui.fill_space(100.0);
                    let toggle_label = if self.is_dark { "Light" } else { "Dark" };
                    let btn_bg = ui.theme().secondary_button_bg;
                    if ui.small_button(id!("ms_theme_toggle"), toggle_label, btn_bg).clicked {
                        self.is_dark = !self.is_dark;
                        let target = if self.is_dark {
                            self.base_dark.clone()
                        } else {
                            self.base_light.clone()
                        };
                        self.transition =
                            Some(ThemeTransition::new(self.theme.clone(), target, 300.0));
                    }
                });
            });

            // ── Tabs ──
            let selected = self.tab_state.selected;
            ui.tabs(
                id!("ms_tabs"),
                &mut self.tab_state,
                &["Overview", "Forms", "Data", "Components"],
                |_ui, _selected| {},
            );

            ui.add_space(8.0);
            ui.padding(24.0, |ui| {
                ui.max_width(800.0, |ui| {
                    match selected {
                        // ════════════════════════════════════════════
                        // Tab 0 — Overview
                        // ════════════════════════════════════════════
                        0 => {
                            let wc = ui.width_class();
                            let wide = matches!(
                                wc,
                                esox_ui::WidthClass::Medium | esox_ui::WidthClass::Expanded
                            );

                            if wide {
                                ui.columns_spaced(16.0, &[1.0, 1.0], |ui, col| {
                                    match col {
                                        0 => {
                                            ui.card(|ui| {
                                                let accent = ui.theme().accent;
                                                let green = ui.theme().green;
                                                ui.rich_label(
                                                    &RichText::new()
                                                        .colored_bold("Dashboard", accent),
                                                );
                                                ui.add_space(8.0);
                                                ui.label_sized(
                                                    "1,247",
                                                    esox_ui::TextSize::Xxl,
                                                );
                                                ui.muted_label("Active users this week");
                                                ui.add_space(8.0);
                                                ui.progress_bar(0.72);
                                                ui.add_space(8.0);
                                                ui.row(|ui| {
                                                    ui.badge(12);
                                                    ui.chip(id!("chip_new"), "New");
                                                    ui.chip(id!("chip_active"), "Active");
                                                    ui.label_colored("72%", green);
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.card(|ui| {
                                                let accent = ui.theme().accent;
                                                let green = ui.theme().green;
                                                let red = ui.theme().red;
                                                ui.rich_label_wrapped(
                                                    &RichText::new()
                                                        .span("The toolkit renders ")
                                                        .colored_bold(
                                                            "every pixel on the GPU",
                                                            accent,
                                                        )
                                                        .span(". It supports ")
                                                        .colored("dark", green)
                                                        .span(" and ")
                                                        .colored("light", red)
                                                        .span(" themes with smooth transitions."),
                                                );
                                                ui.separator();
                                                ui.row(|ui| {
                                                    ui.chip(id!("chip_gpu"), "GPU");
                                                    ui.chip(id!("chip_imm"), "Immediate");
                                                    ui.chip(id!("chip_rust"), "Rust");
                                                });
                                            });
                                        }
                                        1 => {
                                            ui.card(|ui| {
                                                ui.header_label("UPLOAD PROGRESS");
                                                ui.add_space(4.0);
                                                let accent = ui.theme().accent;
                                                ui.progress_bar_colored(
                                                    upload_progress,
                                                    accent,
                                                );
                                                ui.add_space(8.0);
                                                ui.row(|ui| {
                                                    ui.spinner();
                                                    ui.label(" Processing...");
                                                });
                                            });
                                            ui.add_space(8.0);
                                            ui.card(|ui| {
                                                ui.header_label("SYSTEM STATUS");
                                                ui.add_space(4.0);
                                                let green = ui.theme().green;
                                                let amber = ui.theme().amber;
                                                let red = ui.theme().red;
                                                ui.label_colored("CPU: 42%", green);
                                                ui.progress_bar_colored(0.42, green);
                                                ui.add_space(4.0);
                                                ui.label_colored("Memory: 78%", amber);
                                                ui.progress_bar_colored(0.78, amber);
                                                ui.add_space(4.0);
                                                ui.label_colored("Disk: 91%", red);
                                                ui.progress_bar_colored(0.91, red);
                                            });
                                        }
                                        _ => {}
                                    }
                                });
                            } else {
                                ui.card(|ui| {
                                    let accent = ui.theme().accent;
                                    ui.rich_label(
                                        &RichText::new().colored_bold("Dashboard", accent),
                                    );
                                    ui.add_space(8.0);
                                    ui.label_sized("1,247", esox_ui::TextSize::Xxl);
                                    ui.muted_label("Active users this week");
                                    ui.add_space(8.0);
                                    ui.progress_bar(0.72);
                                    ui.add_space(4.0);
                                    ui.row(|ui| {
                                        ui.badge(12);
                                        ui.chip(id!("chip_new_s"), "New");
                                        ui.chip(id!("chip_active_s"), "Active");
                                    });
                                });
                                ui.add_space(8.0);
                                ui.card(|ui| {
                                    ui.header_label("UPLOAD PROGRESS");
                                    let accent = ui.theme().accent;
                                    ui.progress_bar_colored(upload_progress, accent);
                                    ui.add_space(4.0);
                                    ui.row(|ui| {
                                        ui.spinner();
                                        ui.label(" Processing...");
                                    });
                                });
                            }

                            ui.add_space(12.0);
                            ui.status_bar("Online", "v0.1.0");
                        }

                        // ════════════════════════════════════════════
                        // Tab 1 — Forms
                        // ════════════════════════════════════════════
                        1 => {
                            ui.card(|ui| {
                                ui.header_label("PROFILE");
                                ui.add_space(4.0);
                                ui.form_field("Name", FieldStatus::None, "", |ui| {
                                    ui.text_input(
                                        id!("ms_name"),
                                        &mut self.name_input,
                                        "Enter your name...",
                                    )
                                });
                                ui.form_field(
                                    "Email",
                                    FieldStatus::Error,
                                    "Invalid email format",
                                    |ui| {
                                        ui.text_input(
                                            id!("ms_email"),
                                            &mut self.email_input,
                                            "user@example.com",
                                        )
                                    },
                                );
                                ui.form_field("Bio", FieldStatus::None, "", |ui| {
                                    ui.text_area_wrapped(
                                        id!("ms_bio"),
                                        &mut self.bio_input,
                                        3,
                                        "Tell us about yourself...",
                                    )
                                });
                            });

                            ui.add_space(12.0);

                            ui.card(|ui| {
                                ui.header_label("PREFERENCES");
                                ui.add_space(4.0);
                                ui.form_field("Role", FieldStatus::None, "", |ui| {
                                    ui.select(
                                        id!("ms_role"),
                                        &mut self.role_select,
                                        &["Developer", "Designer", "Manager", "Other"],
                                    )
                                });
                                ui.form_field("Country", FieldStatus::None, "", |ui| {
                                    ui.combobox(
                                        id!("ms_country"),
                                        &[
                                            "United States",
                                            "Germany",
                                            "Japan",
                                            "Brazil",
                                            "Australia",
                                            "India",
                                            "Canada",
                                        ],
                                        &mut self.country_combo,
                                    )
                                });

                                ui.add_space(8.0);
                                ui.header_label("EXPERIENCE (YEARS)");
                                ui.slider_f64(
                                    id!("ms_exp"),
                                    &mut self.experience_value,
                                    0.0,
                                    20.0,
                                );
                                ui.add_space(8.0);

                                ui.header_label("FONT SIZE");
                                ui.number_input_clamped(
                                    id!("ms_fontsize"),
                                    &mut self.font_size_value,
                                    1.0,
                                    8.0,
                                    72.0,
                                );
                                ui.add_space(8.0);

                                ui.separator();
                                ui.add_space(4.0);

                                let cb_id = id!("ms_newsletter");
                                let cb_state = self.newsletter_cb.entry(cb_id).or_default();
                                ui.checkbox(cb_id, cb_state, "Subscribe to newsletter");

                                ui.toggle(
                                    id!("ms_notif"),
                                    &mut self.notifications_toggle,
                                    "Push notifications",
                                );
                                ui.add_space(8.0);

                                ui.header_label("PRIORITY");
                                ui.radio(
                                    id!("ms_pri_low"),
                                    &mut self.priority_radio,
                                    0,
                                    "Low",
                                );
                                ui.radio(
                                    id!("ms_pri_med"),
                                    &mut self.priority_radio,
                                    1,
                                    "Medium",
                                );
                                ui.radio(
                                    id!("ms_pri_hi"),
                                    &mut self.priority_radio,
                                    2,
                                    "High",
                                );
                            });

                            ui.add_space(12.0);

                            if ui.button(id!("ms_save"), "Save Profile").clicked {
                                ui.toast_success("Profile saved successfully!");
                            }
                            ui.add_space(4.0);
                            ui.ghost_button(id!("ms_cancel"), "Cancel");
                            ui.add_space(8.0);
                            ui.hyperlink(
                                id!("ms_terms"),
                                "Terms of Service",
                                "https://example.com/terms",
                            );
                        }

                        // ════════════════════════════════════════════
                        // Tab 2 — Data
                        // ════════════════════════════════════════════
                        2 => {
                            ui.card(|ui| {
                                ui.header_label("EMPLOYEE DIRECTORY");
                                ui.add_space(4.0);

                                let columns = [
                                    TableColumn::new("#", ColumnWidth::Fixed(40.0)).not_sortable(),
                                    TableColumn::new("Name", ColumnWidth::Weight(2.0)),
                                    TableColumn::new("Dept", ColumnWidth::Weight(1.5)),
                                    TableColumn::new("Status", ColumnWidth::Weight(1.0)),
                                ];

                                let names = [
                                    "Alice Chen",
                                    "Bob Smith",
                                    "Carol Wu",
                                    "David Kim",
                                    "Eva Brown",
                                    "Frank Liu",
                                    "Grace Lee",
                                    "Henry Park",
                                    "Iris Wang",
                                    "Jack Jones",
                                    "Kate Yang",
                                    "Leo Torres",
                                    "Mia Clark",
                                    "Noah Adams",
                                    "Olivia Hall",
                                    "Paul Scott",
                                    "Quinn Ross",
                                    "Ruby Green",
                                    "Sam White",
                                    "Tina Black",
                                ];
                                let depts = [
                                    "Engineering",
                                    "Design",
                                    "Marketing",
                                    "Engineering",
                                    "Sales",
                                    "Engineering",
                                    "Design",
                                    "Marketing",
                                    "Engineering",
                                    "Sales",
                                    "Engineering",
                                    "Design",
                                    "Marketing",
                                    "Engineering",
                                    "Sales",
                                    "Engineering",
                                    "Design",
                                    "Marketing",
                                    "Engineering",
                                    "Sales",
                                ];

                                ui.table(
                                    id!("ms_table"),
                                    &mut self.table_state,
                                    &columns,
                                    20,
                                    8,
                                    |ui, row, col| match col {
                                        0 => {
                                            let s = format!("{}", row + 1);
                                            ui.label(&s);
                                        }
                                        1 => ui.label(names[row % names.len()]),
                                        2 => ui.label(depts[row % depts.len()]),
                                        3 => {
                                            if row % 3 == 0 {
                                                ui.label_colored("Active", ui.theme().green);
                                            } else if row % 3 == 1 {
                                                ui.label_colored("Away", ui.theme().amber);
                                            } else {
                                                ui.label_colored("Offline", ui.theme().red);
                                            }
                                        }
                                        _ => {}
                                    },
                                );
                            });

                            ui.add_space(12.0);

                            ui.card(|ui| {
                                ui.header_label("PROJECT STRUCTURE");
                                ui.add_space(4.0);

                                let r = ui.tree_node(
                                    id!("tree_workspace"),
                                    &mut self.tree_state,
                                    "esox-workspace/",
                                    true,
                                );
                                ui.animated_tree_indent(
                                    id!("tree_ws_anim"),
                                    r.expanded,
                                    |ui| {
                                        let r2 = ui.tree_node(
                                            id!("tree_crates"),
                                            &mut self.tree_state,
                                            "crates/",
                                            true,
                                        );
                                        ui.animated_tree_indent(
                                            id!("tree_crates_anim"),
                                            r2.expanded,
                                            |ui| {
                                                ui.tree_node(
                                                    id!("tree_gfx"),
                                                    &mut self.tree_state,
                                                    "esox_gfx/",
                                                    false,
                                                );
                                                ui.tree_node(
                                                    id!("tree_ui"),
                                                    &mut self.tree_state,
                                                    "esox_ui/",
                                                    false,
                                                );
                                                ui.tree_node(
                                                    id!("tree_platform"),
                                                    &mut self.tree_state,
                                                    "esox_platform/",
                                                    false,
                                                );
                                            },
                                        );

                                        let r3 = ui.tree_node(
                                            id!("tree_examples"),
                                            &mut self.tree_state,
                                            "examples/",
                                            true,
                                        );
                                        ui.animated_tree_indent(
                                            id!("tree_ex_anim"),
                                            r3.expanded,
                                            |ui| {
                                                ui.tree_node(
                                                    id!("tree_demo"),
                                                    &mut self.tree_state,
                                                    "demo/",
                                                    false,
                                                );
                                                ui.tree_node(
                                                    id!("tree_showcase"),
                                                    &mut self.tree_state,
                                                    "material_showcase/",
                                                    false,
                                                );
                                            },
                                        );

                                        ui.tree_node(
                                            id!("tree_cargo"),
                                            &mut self.tree_state,
                                            "Cargo.toml",
                                            false,
                                        );
                                        ui.tree_node(
                                            id!("tree_readme"),
                                            &mut self.tree_state,
                                            "README.md",
                                            false,
                                        );
                                    },
                                );
                            });

                            ui.add_space(12.0);

                            ui.collapsing_header(
                                id!("ms_vscroll_collapse"),
                                "Virtual Scroll (10,000 items)",
                                false,
                                |ui| {
                                    self.vscroll_state.item_count = 10_000;
                                    ui.virtual_scroll(
                                        id!("ms_vscroll"),
                                        &mut self.vscroll_state,
                                        28.0,
                                        250.0,
                                        |ui, i| {
                                            let label = format!("Item #{}", i);
                                            ui.label(&label);
                                        },
                                    );
                                },
                            );
                        }

                        // ════════════════════════════════════════════
                        // Tab 3 — Components
                        // ════════════════════════════════════════════
                        3 => {
                            ui.card(|ui| {
                                ui.header_label("BUTTON GALLERY");
                                ui.add_space(4.0);
                                ui.button(id!("ms_btn_primary"), "Primary Button");
                                ui.add_space(4.0);
                                ui.secondary_button(id!("ms_btn_secondary"), "Secondary Button");
                                ui.add_space(4.0);
                                ui.danger_button(id!("ms_btn_danger"), "Danger Button");
                                ui.add_space(4.0);
                                ui.ghost_button(id!("ms_btn_ghost"), "Ghost Button");
                                ui.add_space(4.0);
                                let green = ui.theme().green;
                                ui.small_button(id!("ms_btn_small"), "Small", green);
                                ui.add_space(4.0);
                                ui.with_style(
                                    WidgetStyle {
                                        bg: Some(Color::new(0.5, 0.2, 0.8, 1.0)),
                                        fg: Some(Color::new(1.0, 1.0, 1.0, 1.0)),
                                        corner_radius: Some(16.0),
                                        ..Default::default()
                                    },
                                    |ui| {
                                        ui.button(id!("ms_btn_custom"), "Custom Styled");
                                    },
                                );
                            });

                            ui.add_space(12.0);

                            ui.card(|ui| {
                                ui.header_label("FEEDBACK");
                                ui.add_space(4.0);
                                if ui.button(id!("ms_toast_info"), "Show Info Toast").clicked {
                                    ui.toast_info("This is an informational message.");
                                }
                                ui.add_space(4.0);
                                if ui
                                    .button(id!("ms_toast_success"), "Show Success Toast")
                                    .clicked
                                {
                                    ui.toast_success("Operation completed successfully!");
                                }
                                ui.add_space(4.0);
                                if ui
                                    .button(id!("ms_toast_error"), "Show Error Toast")
                                    .clicked
                                {
                                    ui.toast_error("Something went wrong.");
                                }
                                ui.add_space(4.0);
                                if ui
                                    .button(id!("ms_toast_warn"), "Show Warning Toast")
                                    .clicked
                                {
                                    ui.toast_warning("Disk space running low.");
                                }
                                ui.add_space(8.0);

                                if ui.button(id!("ms_open_modal"), "Open Modal").clicked {
                                    self.modal_open = true;
                                }
                                ui.add_space(4.0);
                                if ui.button(id!("ms_open_confirm"), "Confirm Dialog").clicked {
                                    self.confirm_open = true;
                                }
                                if !confirm_result.is_empty() {
                                    ui.muted_label(&confirm_result);
                                }
                            });

                            ui.add_space(12.0);

                            ui.card(|ui| {
                                ui.header_label("INTERACTIVE");
                                ui.add_space(4.0);
                                ui.button(id!("ms_tooltip_btn"), "Hover for Tooltip");
                                ui.tooltip(
                                    id!("ms_tooltip_btn"),
                                    "This is a helpful tooltip!",
                                );
                                ui.add_space(4.0);
                                let resp = ui.button(
                                    id!("ms_ctx_btn"),
                                    "Right-click for Context Menu",
                                );
                                if resp.right_clicked {
                                    ui.context_menu(
                                        id!("ms_ctx_menu"),
                                        &["Cut", "Copy", "Paste", "Select All"],
                                    );
                                }
                            });

                            ui.add_space(12.0);

                            ui.card(|ui| {
                                ui.header_label("OTHER WIDGETS");
                                ui.add_space(4.0);
                                ui.empty_state("No notifications yet");
                                ui.add_space(8.0);
                                ui.separator();
                                ui.add_space(8.0);
                                ui.split_pane_h(
                                    id!("ms_split"),
                                    0.5,
                                    |ui| {
                                        ui.surface(|ui| {
                                            ui.label("Left Panel");
                                            ui.muted_label("Drag the divider");
                                        });
                                    },
                                    |ui| {
                                        ui.surface(|ui| {
                                            ui.label("Right Panel");
                                            ui.muted_label("to resize panels");
                                        });
                                    },
                                );
                                ui.add_space(8.0);
                                ui.collapsing_header(
                                    id!("ms_collapse"),
                                    "Expandable Section",
                                    false,
                                    |ui| {
                                        ui.label("Hidden content revealed!");
                                        ui.muted_label("This section is collapsible.");
                                    },
                                );
                            });
                        }

                        _ => {}
                    }
                });
            });
        }); // page_scroll

        // Modals (drawn after scrollable, on top of everything).
        ui.modal(
            id!("ms_modal"),
            &mut self.modal_open,
            "Example Modal",
            400.0,
            |ui| {
                ui.label("This is a modal dialog.");
                ui.label("Press Escape or click outside to close.");
                ui.add_space(8.0);
                ui.form_field("Modal Input", FieldStatus::None, "", |ui| {
                    ui.text_input(id!("ms_modal_input"), &mut self.name_input, "Type something...")
                });
            },
        );

        let action = ui.modal_confirm(
            id!("ms_confirm"),
            &mut self.confirm_open,
            "Confirm",
            "Are you sure you want to proceed?",
        );
        if action == ModalAction::Confirm {
            self.confirm_result = "Confirmed!".into();
        } else if action == ModalAction::Cancel {
            self.confirm_result = "Cancelled.".into();
        }

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
    fn on_copy(&mut self) -> Option<String> {
        None
    }

    fn needs_redraw(&self) -> bool {
        self.ui_state.needs_redraw()
    }

    fn needs_continuous_redraw(&self) -> bool {
        self.transition.is_some() || self.ui_state.needs_continuous_redraw()
    }

    fn cursor_icon(&self, x: f64, y: f64) -> esox_platform::esox_input::CursorIcon {
        self.ui_state.cursor_icon(x as f32, y as f32)
    }

    fn on_scale_changed(&mut self, scale_factor: f64, _gpu: &GpuContext) {
        let factor = scale_factor as f32;
        self.ui_state.scale_factor = factor;
        self.base_light = Theme::light().scaled(factor);
        self.base_dark = Theme::dark().scaled(factor);
        self.theme = if self.is_dark {
            self.base_dark.clone()
        } else {
            self.base_light.clone()
        };
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = PlatformConfig {
        window: WindowConfig {
            title: "Material Showcase".into(),
            width: Some(900),
            height: Some(700),
            ..Default::default()
        },
        ..Default::default()
    };

    esox_platform::run(config, Box::new(MaterialShowcase::new())).unwrap();
}
