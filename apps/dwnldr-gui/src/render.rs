//! Frame rendering — converts layout + state into esox_gfx quads.
//!
//! Sidebar + activity bar are drawn manually (app-specific layout).
//! Content area uses `esox_ui::Ui` for widget rendering.

use esox_gfx::{BorderRadius, Color, Frame, GpuContext, RenderResources, ShapeBuilder};
use esox_ui::{fnv1a_mix, fnv1a_runtime, id, lerp_color, Rect, Ui};

use crate::app::{App, JobResult, JobStatus};
use crate::layout::SIDEBAR_FOOTER_H;
use crate::tools::{self, InputKind, OptionKind};

/// Draw the complete UI frame.
pub fn draw_frame(
    app: &mut App,
    gpu: &GpuContext,
    resources: &mut RenderResources,
    frame: &mut Frame,
) {
    if let Some(ref mut text) = app.text {
        text.advance_generation();
    }
    draw_sidebar(app, gpu, resources, frame);
    draw_content(app, gpu, resources, frame);
    draw_activity_bar(app, gpu, resources, frame);
}

/// Draw the sidebar background and tool items.
fn draw_sidebar(
    app: &mut App,
    gpu: &GpuContext,
    resources: &mut RenderResources,
    frame: &mut Frame,
) {
    let sb = app.layout.sidebar;
    let theme = &app.theme;

    // Background + right border.
    frame.push(
        ShapeBuilder::rect(sb.x, sb.y, sb.w, sb.h)
            .color(theme.bg_surface)
            .build(),
    );
    frame.push(
        ShapeBuilder::rect(sb.x + sb.w - 1.0, sb.y, 1.0, sb.h)
            .color(theme.border)
            .build(),
    );

    let items: Vec<_> = app.layout.sidebar_items.clone();

    // Build tool-only index for sidebar_focus_index matching.
    let tool_items: Vec<usize> = items.iter().enumerate()
        .filter(|(_, item)| !item.is_header)
        .map(|(i, _)| i)
        .collect();

    for (item_idx, item) in items.iter().enumerate() {
        if item.is_header {
            if let Some(ref mut text) = app.text {
                text.draw_header_text(
                    item.header_label,
                    item.x + theme.padding,
                    item.y + (item.h - theme.header_font_size) / 2.0,
                    theme.fg_dim,
                    frame,
                    gpu,
                    resources,
                );
            }
        } else {
            let is_active = app.active_tool == item.tool_id;
            let is_hovered = app.hovered_item == Some(item.tool_id);

            // Check if this item has sidebar keyboard focus.
            let has_kb_focus = app.sidebar_focus_index
                .and_then(|fi| tool_items.get(fi).copied())
                .map_or(false, |idx| idx == item_idx);

            if is_active {
                frame.push(
                    ShapeBuilder::rect(item.x, item.y, item.w, item.h)
                        .color(theme.accent_dim)
                        .build(),
                );
                frame.push(
                    ShapeBuilder::rect(item.x, item.y + 4.0, 3.0, item.h - 8.0)
                        .color(theme.accent)
                        .build(),
                );
            } else {
                let hover_id = fnv1a_runtime(&format!("sidebar_{}", item.tool_id));
                let t = app.ui_state.hover_t(hover_id, is_hovered, 100.0);
                if t > 0.0 {
                    let hover_bg = lerp_color(
                        Color::new(0.0, 0.0, 0.0, 0.0),
                        theme.bg_raised,
                        t,
                    );
                    frame.push(
                        ShapeBuilder::rect(item.x, item.y, item.w, item.h)
                            .color(hover_bg)
                            .build(),
                    );
                }
            }

            // Focus ring for keyboard navigation.
            if has_kb_focus && !is_active {
                let expand = 1.0;
                frame.push(
                    ShapeBuilder::rect(
                        item.x + expand,
                        item.y + expand,
                        item.w - expand * 2.0,
                        item.h - expand * 2.0,
                    )
                    .color(theme.accent_dim)
                    .border_radius(BorderRadius::uniform(theme.corner_radius))
                    .build(),
                );
            }

            let label_color = if is_active { theme.accent } else { theme.fg };
            if let Some(tool) = tools::find_tool(item.tool_id) {
                if let Some(ref mut text) = app.text {
                    // Draw icon glyph.
                    text.draw_ui_text(
                        tool.icon,
                        item.x + theme.padding,
                        item.y + (item.h - theme.font_size) / 2.0,
                        if is_active { theme.accent } else { theme.fg_muted },
                        frame,
                        gpu,
                        resources,
                    );

                    // Draw label — offset right past the icon.
                    text.draw_ui_text(
                        tool.label,
                        item.x + theme.padding + 22.0,
                        item.y + (item.h - theme.font_size) / 2.0,
                        label_color,
                        frame,
                        gpu,
                        resources,
                    );
                }
            }
        }
    }

    // ── Theme toggle footer ──
    let footer_y = sb.y + sb.h - SIDEBAR_FOOTER_H;
    let is_toggle_hovered = app.hovered_toggle;

    // Top separator.
    frame.push(
        ShapeBuilder::rect(sb.x, footer_y, sb.w, 1.0)
            .color(theme.border)
            .build(),
    );

    // Hover background.
    let t = app.ui_state.hover_t(id!("sidebar_theme_toggle"), is_toggle_hovered, 100.0);
    if t > 0.0 {
        let hover_bg = lerp_color(Color::new(0.0, 0.0, 0.0, 0.0), theme.bg_raised, t);
        frame.push(
            ShapeBuilder::rect(sb.x, footer_y, sb.w, SIDEBAR_FOOTER_H)
                .color(hover_bg)
                .build(),
        );
    }

    // Indicator dot — amber for light, dim for dark.
    let dot_color = if app.theme_is_dark { theme.fg_dim } else { theme.amber };
    let dot_cx = sb.x + theme.padding + 5.0;
    let dot_cy = footer_y + SIDEBAR_FOOTER_H / 2.0;
    frame.push(ShapeBuilder::circle(dot_cx, dot_cy, 4.0).color(dot_color).build());

    let toggle_label = if app.theme_is_dark { "Dark Mode" } else { "Light Mode" };
    if let Some(ref mut text) = app.text {
        text.draw_ui_text(
            toggle_label,
            sb.x + theme.padding + 16.0,
            footer_y + (SIDEBAR_FOOTER_H - theme.font_size) / 2.0,
            theme.fg_muted,
            frame,
            gpu,
            resources,
        );
    }
}

/// Draw the content area using esox_ui widgets.
fn draw_content(
    app: &mut App,
    gpu: &GpuContext,
    resources: &mut RenderResources,
    frame: &mut Frame,
) {
    let ct = app.layout.content;
    let padding = app.theme.padding;

    // Content background.
    frame.push(
        ShapeBuilder::rect(ct.x, ct.y, ct.w, ct.h)
            .color(app.theme.bg_base)
            .build(),
    );

    let tool = match tools::find_tool(app.active_tool) {
        Some(t) => t,
        None => return,
    };
    let active_tool = app.active_tool;

    let viewport = Rect::new(ct.x, ct.y, ct.w, ct.h);

    // Pre-extract job data to avoid borrow conflicts with Ui.
    let latest_job = app.latest_job_for_tool(active_tool).map(|job| {
        (job.id, job.status, job.result.clone(), job.message.clone())
    });

    // Pre-extract running job data.
    let running_job = app.jobs.iter().rev()
        .find(|j| j.tool_id == active_tool && j.status == JobStatus::Running)
        .map(|j| (j.message.clone(), j.progress));

    let text = match app.text.as_mut() {
        Some(t) => t,
        None => return,
    };

    // Clone theme colors we need for result display (avoids borrow conflict).
    let red = app.theme.red;
    let fg_dim = app.theme.fg_dim;

    let mut ui = Ui::begin(
        frame,
        gpu,
        resources,
        text,
        &mut app.ui_state,
        &app.theme,
        viewport,
    );

    // Scrollable content area.
    ui.scrollable(id!("content_scroll"), ct.h, |ui| {
        // Center content column.
        ui.max_width(720.0, |ui| {
            ui.add_space(padding);

            // ── Tool title card ──
            ui.card(|ui| {
                ui.heading(tool.label);
                ui.add_space(padding / 2.0);

                // ── Primary input ──
                match &tool.input {
                    InputKind::Url | InputKind::Text { .. } => {
                        let placeholder = match &tool.input {
                            InputKind::Url => "Paste URL here...",
                            InputKind::Text { placeholder } => placeholder,
                            _ => "",
                        };
                        let input = app.form_state.get_or_create(active_tool, "__input__");
                        ui.text_input(id!("__input__"), input, placeholder);
                    }
                    InputKind::File { .. } | InputKind::Folder => {
                        let ds = app
                            .drop_state
                            .entry(active_tool)
                            .or_insert_with(esox_ui::DropZoneState::new);
                        let files = ds.files.clone();
                        let resp = ui.drop_zone(id!("dropzone"), &files);
                        if resp.clicked {
                            app.needs_file_dialog = true;
                        }
                    }
                }
            });

            // ── Options card ──
            if !tool.options.is_empty() {
                ui.card(|ui| {
                    for opt in tool.options {
                        match &opt.kind {
                            OptionKind::Toggle => {
                                let input = app.form_state.get_or_create(active_tool, opt.key);
                                let widget_id = fnv1a_runtime(&format!("opt_{}", opt.key));
                                ui.checkbox(widget_id, input, opt.label);
                            }
                            _ => {
                                ui.label_colored(opt.label, app.theme.fg_label);
                                ui.add_space(2.0);

                                match &opt.kind {
                                    OptionKind::Number { placeholder }
                                    | OptionKind::Text { placeholder } => {
                                        let ph = placeholder.unwrap_or("");
                                        let input = app.form_state.get_or_create(active_tool, opt.key);
                                        let widget_id = fnv1a_runtime(&format!("opt_{}", opt.key));
                                        ui.text_input(widget_id, input, ph);
                                    }
                                    OptionKind::Time => {
                                        let input = app.form_state.get_or_create(active_tool, opt.key);
                                        let widget_id = fnv1a_runtime(&format!("opt_{}", opt.key));
                                        ui.text_input(widget_id, input, "MM:SS");
                                    }
                                    OptionKind::Select { choices } => {
                                        let sel = app.form_state.get_or_create_select(
                                            active_tool,
                                            opt.key,
                                            choices.len(),
                                        );
                                        let widget_id = fnv1a_runtime(&format!("sel_{}", opt.key));
                                        ui.select(widget_id, sel, choices);
                                    }
                                    OptionKind::Slider { min, max, default } => {
                                        let input = app.form_state.get_or_create(active_tool, opt.key);
                                        if input.text.is_empty() {
                                            input.text = format!("{}", *default as i32);
                                            input.cursor = input.text.len();
                                        }
                                        let widget_id = fnv1a_runtime(&format!("opt_{}", opt.key));
                                        ui.slider(widget_id, input, *min, *max);
                                    }
                                    OptionKind::Toggle => unreachable!(),
                                }
                                ui.add_space(padding / 2.0);
                            }
                        }
                    }

                    // ── Action button inside options card. ──
                    ui.add_space(padding / 2.0);
                    let btn_resp = ui.button_max_width(id!("action"), tool.action_label, 200.0);
                    if btn_resp.clicked {
                        app.needs_execute = true;
                    }
                });
            } else {
                // No options — standalone action button.
                let btn_resp = ui.button_max_width(id!("action"), tool.action_label, 200.0);
                if btn_resp.clicked {
                    app.needs_execute = true;
                }
                ui.add_space(padding);
            }

            // ── Running job indicator ──
            if let Some((ref msg, progress)) = running_job {
                ui.card(|ui| {
                    ui.row(|ui| {
                        ui.spinner();
                        ui.label(msg);
                    });
                    if let Some(pct) = progress {
                        ui.add_space(4.0);
                        ui.progress_bar(pct / 100.0);
                    }
                });
            }

            // ── Result display ──
            if let Some((job_id, job_status, job_result, job_message)) = latest_job {
                match job_status {
                    JobStatus::Error => {
                        ui.card(|ui| {
                            ui.label_colored(&job_message, red);
                        });
                    }
                    JobStatus::Done => {
                        if let Some(result) = &job_result {
                            ui.card(|ui| {
                                match result {
                                    JobResult::File(path) => {
                                        let filename = std::path::Path::new(path)
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or(path);
                                        ui.muted_label(filename);
                                        ui.add_space(4.0);
                                        let open_id = fnv1a_mix(id!("open_"), job_id);
                                        let resp = ui.ghost_button(open_id, "Open File");
                                        if resp.clicked {
                                            app.result_action = Some(ResultAction::Open(job_id));
                                        }
                                    }
                                    JobResult::Files(paths) => {
                                        let label = format!("{} files produced", paths.len());
                                        ui.muted_label(&label);

                                        let max_show = 5;
                                        for (i, p) in paths.iter().enumerate() {
                                            if i >= max_show {
                                                let more = format!("+ {} more", paths.len() - max_show);
                                                ui.label_colored(&more, fg_dim);
                                                break;
                                            }
                                            let fname = std::path::Path::new(p)
                                                .file_name()
                                                .and_then(|n| n.to_str())
                                                .unwrap_or(p);
                                            ui.muted_label(fname);
                                        }
                                        ui.add_space(4.0);
                                        let open_id = fnv1a_mix(id!("open_"), job_id);
                                        let resp = ui.ghost_button(open_id, "Show in Folder");
                                        if resp.clicked {
                                            app.result_action = Some(ResultAction::Open(job_id));
                                        }
                                    }
                                    JobResult::Text(result_text) => {
                                        let display = if result_text.len() > 200 {
                                            &result_text[..200]
                                        } else {
                                            result_text
                                        };
                                        ui.label(display);
                                        ui.add_space(4.0);
                                        let copy_id = fnv1a_mix(id!("copy_"), job_id);
                                        let resp = ui.ghost_button(copy_id, "Copy");
                                        if resp.clicked {
                                            app.result_action = Some(ResultAction::Copy(job_id));
                                        }
                                    }
                                }
                            });
                        }
                    }
                    _ => {}
                }
            }

            // Bottom padding so last widget doesn't press against the activity bar.
            ui.add_space(padding * 3.0);
        });
    });

    // Finish Ui — draws overlays, toasts, cleans up per-frame state.
    // If a dropdown selection occurred, write it back to FormState.
    if let Some((sel_widget_id, idx)) = ui.finish() {
        for opt in tool.options {
            if fnv1a_runtime(&format!("sel_{}", opt.key)) == sel_widget_id {
                app.form_state
                    .get_or_create_select(active_tool, opt.key, 0)
                    .selected_index = idx;
                break;
            }
        }
    }
}

/// Deferred action from a result button click.
pub enum ResultAction {
    Open(u64),
    Copy(u64),
}

/// Draw the activity bar at the bottom.
fn draw_activity_bar(
    app: &mut App,
    gpu: &GpuContext,
    resources: &mut RenderResources,
    frame: &mut Frame,
) {
    let ab = &app.layout.activity_bar;
    let theme = &app.theme;

    frame.push(
        ShapeBuilder::rect(ab.x, ab.y, ab.w, ab.h)
            .color(theme.bg_surface)
            .build(),
    );
    frame.push(
        ShapeBuilder::rect(ab.x, ab.y, ab.w, 1.0)
            .color(theme.border)
            .build(),
    );

    if app.jobs.is_empty() {
        if let Some(ref mut text) = app.text {
            text.draw_header_text(
                "No active jobs",
                ab.x + theme.padding,
                ab.y + (ab.h - theme.header_font_size) / 2.0,
                theme.fg_dim,
                frame,
                gpu,
                resources,
            );
        }
    } else {
        let running = app.jobs.iter().filter(|j| j.status == JobStatus::Running).count();
        let has_error = app.jobs.iter().any(|j| j.status == JobStatus::Error);

        let dot_color = if running > 0 {
            theme.green
        } else if has_error {
            theme.red
        } else {
            theme.fg_muted
        };
        frame.push(
            ShapeBuilder::circle(
                ab.x + theme.padding + theme.status_dot_radius,
                ab.y + ab.h / 2.0,
                theme.status_dot_radius,
            )
            .color(dot_color)
            .build(),
        );

        let mut label = if running > 0 {
            format!("{running} job{} running", if running == 1 { "" } else { "s" })
        } else {
            format!(
                "{} job{} done",
                app.jobs.len(),
                if app.jobs.len() == 1 { "" } else { "s" }
            )
        };

        if let Some(latest) = app.jobs.iter().rev().find(|j| j.status == JobStatus::Running) {
            label.push_str(" \u{2014} ");
            label.push_str(&latest.message);
        } else if let Some(latest) = app.jobs.last() {
            label.push_str(" \u{2014} ");
            label.push_str(&latest.message);
        }

        if let Some(ref mut text) = app.text {
            text.draw_header_text(
                &label,
                ab.x + theme.padding + 16.0,
                ab.y + (ab.h - theme.header_font_size) / 2.0,
                theme.fg_muted,
                frame,
                gpu,
                resources,
            );
        }

        if let Some(latest) = app.jobs.iter().rev().find(|j| j.status == JobStatus::Running) {
            if let Some(pct) = latest.progress {
                let bar_w = ab.w * (pct / 100.0).clamp(0.0, 1.0);
                frame.push(
                    ShapeBuilder::rect(
                        ab.x,
                        ab.y + ab.h - theme.progress_bar_height,
                        bar_w,
                        theme.progress_bar_height,
                    )
                    .color(theme.accent)
                    .build(),
                );
            }
        }
    }
}
