//! Theme — all colors and sizes in one struct, plus builder and transition helpers.

use std::time::Instant;

use esox_gfx::Color;

use crate::paint::lerp_color;

/// Semantic text size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextSize {
    Xs,
    Sm,
    Base,
    Lg,
    Xl,
    Xxl,
    Custom(f32),
}

/// Pseudo-state for deriving colors from a base color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleState {
    Normal,
    Hovered,
    Focused,
    Active,
    Disabled,
}

/// Per-widget style overrides, pushed onto a stack via `Ui::with_style`.
#[derive(Debug, Clone, Default)]
pub struct WidgetStyle {
    pub bg: Option<Color>,
    pub fg: Option<Color>,
    pub border_color: Option<Color>,
    pub font_size: Option<f32>,
    pub corner_radius: Option<f32>,
    pub height: Option<f32>,
    /// Inherited text color for label-type widgets (labels, paragraphs).
    pub text_color: Option<Color>,
    /// Override spacing between children.
    pub spacing: Option<f32>,
}

/// Complete UI theme — all visual properties in one place.
#[derive(Debug, Clone)]
pub struct Theme {
    // Backgrounds (increasing brightness for depth).
    pub bg_base: Color,
    pub bg_surface: Color,
    pub bg_raised: Color,
    pub bg_input: Color,

    // Text colors.
    pub fg: Color,
    pub fg_muted: Color,
    pub fg_dim: Color,
    /// Form field labels — between fg and fg_muted.
    pub fg_label: Color,

    // Accent.
    pub accent: Color,
    pub accent_dim: Color,
    pub accent_hover: Color,

    // Status.
    pub green: Color,
    pub amber: Color,
    pub red: Color,

    // Border.
    pub border: Color,

    // Button backgrounds.
    pub green_button_bg: Color,
    pub secondary_button_bg: Color,
    pub secondary_button_hover: Color,
    pub danger_button_bg: Color,
    pub danger_button_hover: Color,

    // Overlay / toast colors.
    pub shadow: Color,
    pub toast_error_bg: Color,
    pub toast_success_bg: Color,

    // Layout constants.
    pub corner_radius: f32,
    pub padding: f32,
    pub input_padding: f32,
    pub item_height: f32,
    pub font_size: f32,
    pub header_font_size: f32,
    pub cursor_width: f32,
    pub cursor_blink_ms: u64,

    // Derived layout constants (moved from widget hardcodes).
    pub button_height: f32,
    pub small_button_height: f32,
    pub small_button_min_w: f32,
    pub focus_ring_expand: f32,
    pub dropdown_gap: f32,
    pub label_pad_y: f32,
    pub heading_font_size: f32,
    pub heading_height: f32,
    pub drop_zone_height: f32,
    pub drop_zone_dash: f32,
    pub drop_zone_dash_gap: f32,
    pub drop_zone_dash_thickness: f32,
    pub progress_bar_height: f32,
    pub status_dot_radius: f32,
    pub toast_w: f32,
    pub toast_h: f32,

    // Disabled.
    pub disabled_fg: Color,
    pub disabled_border: Color,
    pub disabled_bg: Color,

    // Tooltip.
    pub tooltip_delay_ms: u64,
    pub tooltip_font_size: f32,
    pub tooltip_padding: f32,
    pub tooltip_bg: Color,
    pub tooltip_fg: Color,

    // Context menu.
    pub context_menu_min_w: f32,

    // Scrollbar.
    pub scrollbar_width: f32,
    pub scrollbar_min_thumb: f32,
    pub scroll_speed: f32,
    pub scroll_friction: f32,

    // Tabs.
    pub tab_indicator_height: f32,
    pub tab_fade_duration_ms: f32,

    // Table.
    pub table_header_height: f32,
    pub table_zebra_bg: Color,
    pub column_resize_handle_width: f32,
    pub column_resize_min_width: f32,

    // Tree.
    pub tree_indent: f32,
    pub tree_expand_duration_ms: f32,

    // Toggle switch.
    pub toggle_width: f32,
    pub toggle_height: f32,
    pub toggle_knob_inset: f32,

    // Spinner.
    pub spinner_size: f32,
    pub spinner_speed: f32,

    // Animation durations.
    pub modal_fade_duration_ms: f32,
    pub toast_fade_in_ms: f32,
    pub toast_fade_out_ms: f32,

    // Modal.
    pub modal_backdrop: Color,
    pub modal_corner_radius: f32,
    pub modal_title_height: f32,
    pub modal_padding: f32,
    pub modal_min_width: f32,
    pub modal_max_width: f32,

    // Type scale — semantic text sizes.
    pub text_xs: f32,
    pub text_sm: f32,
    pub text_base: f32,
    pub text_lg: f32,
    pub text_xl: f32,
    pub text_2xl: f32,

    // Text layout.
    pub line_spacing: f32,

    // Responsive breakpoints.
    pub breakpoint_compact: f32,
    pub breakpoint_expanded: f32,

    // Form field helpers.
    pub form_label_gap: f32,
    pub form_helper_gap: f32,
    pub form_helper_font_size: f32,

    // Toast.
    pub toast_info_bg: Color,
    pub toast_warning_bg: Color,
    pub toast_duration_ms: u64,
    pub toast_max_visible: usize,
    pub toast_margin: f32,

    // Disabled border dashes.
    pub disabled_dash_len: f32,
    pub disabled_dash_gap: f32,
    pub disabled_dash_thickness: f32,

    // Uniform hover animation speed.
    pub hover_duration_ms: f32,

    // Widget sizes.
    pub checkbox_size: f32,
    pub radio_size: f32,
    pub radio_dot_size: f32,
    pub slider_track_height: f32,
    pub split_pane_divider: f32,
    pub badge_pad_x: f32,
    pub badge_pad_y: f32,
    pub chip_pad_y: f32,

    // Modal layout.
    pub modal_max_height_ratio: f32,
    pub modal_shadow_alpha: f32,
    pub modal_margin: f32,
    pub modal_close_btn_size: f32,
    pub modal_vertical_offset: f32,

    // Semantic colors.
    pub fg_on_accent: Color,
}

impl Theme {
    /// Default dark theme.
    pub fn dark() -> Self {
        Self {
            // Neutral charcoal steps — no color tint so the accent pops cleanly.
            bg_base:    Color::new(0.047, 0.047, 0.050, 1.0), // #0c0c0d
            bg_surface: Color::new(0.082, 0.082, 0.086, 1.0), // #151516 sidebar/panels
            bg_raised:  Color::new(0.122, 0.122, 0.127, 1.0), // #1f1f20 cards/hover
            bg_input:   Color::new(0.165, 0.165, 0.170, 1.0), // #2a2a2b input fields

            fg:       Color::new(0.900, 0.900, 0.900, 1.0), // #e6e6e6
            fg_muted: Color::new(0.530, 0.530, 0.530, 1.0), // #878787 secondary text
            fg_dim:   Color::new(0.360, 0.360, 0.360, 1.0), // #5c5c5c muted labels
            fg_label: Color::new(0.720, 0.720, 0.720, 1.0), // #b8b8b8 form labels

            // Blue accent — clean, not purple-shifted.
            accent:       Color::new(0.306, 0.533, 0.957, 1.0), // #4e88f4
            accent_dim:   Color::new(0.306, 0.533, 0.957, 0.18),
            accent_hover: Color::new(0.431, 0.627, 0.973, 1.0), // #6ea0f8

            green: Color::new(0.243, 0.812, 0.416, 1.0), // #3ecf6a
            amber: Color::new(0.961, 0.737, 0.133, 1.0), // #f5bc22
            red:   Color::new(0.941, 0.376, 0.376, 1.0), // #f06060

            border:          Color::new(0.200, 0.200, 0.205, 1.0), // #333334
            green_button_bg: Color::new(0.082, 0.380, 0.196, 1.0), // #156132
            secondary_button_bg:    Color::new(0.160, 0.160, 0.165, 1.0), // #292929
            secondary_button_hover: Color::new(0.200, 0.200, 0.205, 1.0), // #333334
            danger_button_bg:       Color::new(0.400, 0.100, 0.100, 1.0), // #661a1a
            danger_button_hover:    Color::new(0.500, 0.130, 0.130, 1.0), // #802121

            shadow: Color::new(0.0, 0.0, 0.0, 0.50),
            toast_error_bg:   Color::new(0.337, 0.078, 0.078, 1.0), // #561414
            toast_success_bg: Color::new(0.063, 0.255, 0.122, 1.0), // #10411f

            disabled_fg:     Color::new(0.30, 0.30, 0.30, 1.0),
            disabled_border: Color::new(0.16, 0.16, 0.165, 1.0),
            disabled_bg:     Color::new(0.10, 0.10, 0.105, 1.0),

            tooltip_bg: Color::new(0.85, 0.85, 0.85, 0.95),
            tooltip_fg: Color::new(0.10, 0.10, 0.10, 1.0),

            table_zebra_bg: Color::new(0.060, 0.060, 0.064, 1.0),

            modal_backdrop: Color::new(0.0, 0.0, 0.0, 0.5),

            toast_info_bg:    Color::new(0.106, 0.165, 0.310, 1.0), // dark blue
            toast_warning_bg: Color::new(0.310, 0.240, 0.078, 1.0), // dark amber

            ..Self::layout_defaults()
        }
    }

    /// Light theme.
    pub fn light() -> Self {
        Self {
            bg_base:    Color::new(0.980, 0.980, 0.980, 1.0), // #fafafa
            bg_surface: Color::new(0.941, 0.941, 0.945, 1.0), // #f0f0f1 sidebar/panels
            bg_raised:  Color::new(0.894, 0.894, 0.902, 1.0), // #e4e4e6 hover
            bg_input:   Color::new(1.000, 1.000, 1.000, 1.0), // #ffffff inputs

            fg:       Color::new(0.067, 0.067, 0.094, 1.0), // #111118
            fg_muted: Color::new(0.376, 0.376, 0.420, 1.0), // #60606b
            fg_dim:   Color::new(0.565, 0.565, 0.627, 1.0), // #9090a0
            fg_label: Color::new(0.220, 0.220, 0.271, 1.0), // #383845

            accent:       Color::new(0.200, 0.471, 0.941, 1.0), // #3378f0
            accent_dim:   Color::new(0.200, 0.471, 0.941, 0.15),
            accent_hover: Color::new(0.376, 0.557, 0.961, 1.0), // #608ef5

            green: Color::new(0.094, 0.631, 0.247, 1.0), // #18a13f
            amber: Color::new(0.737, 0.482, 0.020, 1.0), // #bc7b05
            red:   Color::new(0.780, 0.141, 0.141, 1.0), // #c72424

            border:          Color::new(0.800, 0.800, 0.820, 1.0), // #ccccd1
            green_button_bg: Color::new(0.820, 0.945, 0.839, 1.0), // #d1f1d6
            secondary_button_bg:    Color::new(0.900, 0.900, 0.910, 1.0), // #e6e6e8
            secondary_button_hover: Color::new(0.850, 0.850, 0.865, 1.0), // #d9d9dd
            danger_button_bg:       Color::new(0.880, 0.200, 0.200, 1.0), // #e13333
            danger_button_hover:    Color::new(0.800, 0.160, 0.160, 1.0), // #cc2929

            shadow: Color::new(0.0, 0.0, 0.0, 0.12),
            toast_error_bg:   Color::new(0.996, 0.886, 0.886, 1.0), // #fee2e2
            toast_success_bg: Color::new(0.863, 0.961, 0.882, 1.0), // #dcf5e1

            disabled_fg:     Color::new(0.70, 0.70, 0.72, 1.0),
            disabled_border: Color::new(0.82, 0.82, 0.84, 1.0),
            disabled_bg:     Color::new(0.92, 0.92, 0.93, 1.0),

            tooltip_bg: Color::new(0.15, 0.15, 0.18, 0.95),
            tooltip_fg: Color::new(0.92, 0.92, 0.92, 1.0),

            table_zebra_bg: Color::new(0.960, 0.960, 0.965, 1.0),

            modal_backdrop: Color::new(0.0, 0.0, 0.0, 0.3),

            toast_info_bg:    Color::new(0.886, 0.918, 0.996, 1.0), // light blue
            toast_warning_bg: Color::new(0.996, 0.957, 0.886, 1.0), // light amber

            ..Self::layout_defaults()
        }
    }

    /// High-contrast theme for accessibility — pure black/white, large borders, no transparency.
    pub fn high_contrast() -> Self {
        let black = Color::new(0.0, 0.0, 0.0, 1.0);
        let white = Color::new(1.0, 1.0, 1.0, 1.0);
        let yellow = Color::new(1.0, 1.0, 0.0, 1.0);
        let cyan = Color::new(0.0, 1.0, 1.0, 1.0);
        let green = Color::new(0.0, 1.0, 0.0, 1.0);
        let red = Color::new(1.0, 0.0, 0.0, 1.0);

        Self {
            bg_base:    black,
            bg_surface: black,
            bg_raised:  Color::new(0.15, 0.15, 0.15, 1.0),
            bg_input:   black,

            fg:       white,
            fg_muted: white,
            fg_dim:   Color::new(0.8, 0.8, 0.8, 1.0),
            fg_label: white,

            accent:       yellow,
            accent_dim:   Color::new(1.0, 1.0, 0.0, 0.3),
            accent_hover: cyan,

            green,
            amber: yellow,
            red,

            border:          white,
            green_button_bg: green,
            secondary_button_bg:    Color::new(0.25, 0.25, 0.25, 1.0),
            secondary_button_hover: Color::new(0.35, 0.35, 0.35, 1.0),
            danger_button_bg:       red,
            danger_button_hover:    Color::new(0.8, 0.0, 0.0, 1.0),

            shadow: Color::new(0.0, 0.0, 0.0, 1.0),
            toast_error_bg:   Color::new(0.4, 0.0, 0.0, 1.0),
            toast_success_bg: Color::new(0.0, 0.3, 0.0, 1.0),

            disabled_fg:     Color::new(0.5, 0.5, 0.5, 1.0),
            disabled_border: Color::new(0.5, 0.5, 0.5, 1.0),
            disabled_bg:     Color::new(0.1, 0.1, 0.1, 1.0),

            tooltip_bg: white,
            tooltip_fg: black,

            table_zebra_bg: Color::new(0.1, 0.1, 0.1, 1.0),

            modal_backdrop: Color::new(0.0, 0.0, 0.0, 0.85),

            toast_info_bg:    Color::new(0.0, 0.0, 0.4, 1.0),
            toast_warning_bg: Color::new(0.4, 0.3, 0.0, 1.0),

            // Larger sizes for readability.
            font_size: 16.0,
            heading_font_size: 24.0,
            header_font_size: 13.0,
            text_xs: 12.0,
            text_sm: 14.0,
            text_base: 16.0,
            text_lg: 18.0,
            text_xl: 22.0,
            text_2xl: 24.0,
            button_height: 40.0,
            item_height: 36.0,
            corner_radius: 2.0,
            focus_ring_expand: 3.0,
            cursor_width: 2.5,

            ..Self::layout_defaults()
        }
    }

    /// Return a copy with all dimensional fields multiplied by `factor`.
    /// Colors are unchanged. Useful for HiDPI scaling.
    pub fn scaled(&self, factor: f32) -> Self {
        let mut t = self.clone();
        t.corner_radius *= factor;
        t.padding *= factor;
        t.input_padding *= factor;
        t.item_height *= factor;
        t.font_size *= factor;
        t.header_font_size *= factor;
        t.cursor_width *= factor;
        t.button_height *= factor;
        t.small_button_height *= factor;
        t.small_button_min_w *= factor;
        t.focus_ring_expand *= factor;
        t.dropdown_gap *= factor;
        t.label_pad_y *= factor;
        t.heading_font_size *= factor;
        t.heading_height *= factor;
        t.drop_zone_height *= factor;
        t.drop_zone_dash *= factor;
        t.drop_zone_dash_gap *= factor;
        t.drop_zone_dash_thickness *= factor;
        t.progress_bar_height *= factor;
        t.status_dot_radius *= factor;
        t.toast_w *= factor;
        t.toast_h *= factor;
        t.tooltip_font_size *= factor;
        t.tooltip_padding *= factor;
        t.context_menu_min_w *= factor;
        t.scrollbar_width *= factor;
        t.scrollbar_min_thumb *= factor;
        t.scroll_speed *= factor;
        // scroll_friction is a ratio, not a size — don't scale it.
        t.tab_indicator_height *= factor;
        t.table_header_height *= factor;
        t.column_resize_handle_width *= factor;
        t.column_resize_min_width *= factor;
        t.tree_indent *= factor;
        t.toggle_width *= factor;
        t.toggle_height *= factor;
        t.toggle_knob_inset *= factor;
        t.spinner_size *= factor;
        t.modal_corner_radius *= factor;
        t.modal_title_height *= factor;
        t.modal_padding *= factor;
        t.modal_min_width *= factor;
        t.modal_max_width *= factor;
        t.toast_margin *= factor;
        t.text_xs *= factor;
        t.text_sm *= factor;
        t.text_base *= factor;
        t.text_lg *= factor;
        t.text_xl *= factor;
        t.text_2xl *= factor;
        t.line_spacing *= factor;
        t.breakpoint_compact *= factor;
        t.breakpoint_expanded *= factor;
        t.form_label_gap *= factor;
        t.form_helper_gap *= factor;
        t.form_helper_font_size *= factor;
        t.disabled_dash_len *= factor;
        t.disabled_dash_gap *= factor;
        t.disabled_dash_thickness *= factor;
        t.checkbox_size *= factor;
        t.radio_size *= factor;
        t.radio_dot_size *= factor;
        t.slider_track_height *= factor;
        t.split_pane_divider *= factor;
        t.badge_pad_x *= factor;
        t.badge_pad_y *= factor;
        t.chip_pad_y *= factor;
        t.modal_margin *= factor;
        t.modal_close_btn_size *= factor;
        // hover_duration_ms, modal_max_height_ratio, modal_shadow_alpha, modal_vertical_offset are ratios — don't scale.
        t
    }

    /// Start a builder from the dark theme.
    pub fn builder() -> ThemeBuilder {
        ThemeBuilder { base: Self::dark() }
    }

    /// Interpolate between two themes. Colors lerp; layout f32 snaps at t >= 0.5.
    pub fn lerp(a: &Theme, b: &Theme, t: f32) -> Theme {
        let t = t.clamp(0.0, 1.0);
        let snap = if t >= 0.5 { b } else { a };

        Theme {
            bg_base:      lerp_color(a.bg_base, b.bg_base, t),
            bg_surface:   lerp_color(a.bg_surface, b.bg_surface, t),
            bg_raised:    lerp_color(a.bg_raised, b.bg_raised, t),
            bg_input:     lerp_color(a.bg_input, b.bg_input, t),
            fg:           lerp_color(a.fg, b.fg, t),
            fg_muted:     lerp_color(a.fg_muted, b.fg_muted, t),
            fg_dim:       lerp_color(a.fg_dim, b.fg_dim, t),
            fg_label:     lerp_color(a.fg_label, b.fg_label, t),
            accent:       lerp_color(a.accent, b.accent, t),
            accent_dim:   lerp_color(a.accent_dim, b.accent_dim, t),
            accent_hover: lerp_color(a.accent_hover, b.accent_hover, t),
            green:        lerp_color(a.green, b.green, t),
            amber:        lerp_color(a.amber, b.amber, t),
            red:          lerp_color(a.red, b.red, t),
            border:       lerp_color(a.border, b.border, t),
            green_button_bg: lerp_color(a.green_button_bg, b.green_button_bg, t),
            secondary_button_bg: lerp_color(a.secondary_button_bg, b.secondary_button_bg, t),
            secondary_button_hover: lerp_color(a.secondary_button_hover, b.secondary_button_hover, t),
            danger_button_bg: lerp_color(a.danger_button_bg, b.danger_button_bg, t),
            danger_button_hover: lerp_color(a.danger_button_hover, b.danger_button_hover, t),
            shadow:       lerp_color(a.shadow, b.shadow, t),
            toast_error_bg:   lerp_color(a.toast_error_bg, b.toast_error_bg, t),
            toast_success_bg: lerp_color(a.toast_success_bg, b.toast_success_bg, t),
            disabled_fg:     lerp_color(a.disabled_fg, b.disabled_fg, t),
            disabled_border: lerp_color(a.disabled_border, b.disabled_border, t),
            disabled_bg:     lerp_color(a.disabled_bg, b.disabled_bg, t),
            tooltip_bg:      lerp_color(a.tooltip_bg, b.tooltip_bg, t),
            tooltip_fg:      lerp_color(a.tooltip_fg, b.tooltip_fg, t),
            table_zebra_bg:  lerp_color(a.table_zebra_bg, b.table_zebra_bg, t),
            modal_backdrop:  lerp_color(a.modal_backdrop, b.modal_backdrop, t),
            toast_info_bg:    lerp_color(a.toast_info_bg, b.toast_info_bg, t),
            toast_warning_bg: lerp_color(a.toast_warning_bg, b.toast_warning_bg, t),

            // Layout constants snap.
            corner_radius: snap.corner_radius,
            padding: snap.padding,
            input_padding: snap.input_padding,
            item_height: snap.item_height,
            font_size: snap.font_size,
            header_font_size: snap.header_font_size,
            cursor_width: snap.cursor_width,
            cursor_blink_ms: snap.cursor_blink_ms,
            button_height: snap.button_height,
            small_button_height: snap.small_button_height,
            small_button_min_w: snap.small_button_min_w,
            focus_ring_expand: snap.focus_ring_expand,
            dropdown_gap: snap.dropdown_gap,
            label_pad_y: snap.label_pad_y,
            heading_font_size: snap.heading_font_size,
            heading_height: snap.heading_height,
            drop_zone_height: snap.drop_zone_height,
            drop_zone_dash: snap.drop_zone_dash,
            drop_zone_dash_gap: snap.drop_zone_dash_gap,
            drop_zone_dash_thickness: snap.drop_zone_dash_thickness,
            progress_bar_height: snap.progress_bar_height,
            status_dot_radius: snap.status_dot_radius,
            toast_w: snap.toast_w,
            toast_h: snap.toast_h,
            tooltip_delay_ms: snap.tooltip_delay_ms,
            tooltip_font_size: snap.tooltip_font_size,
            tooltip_padding: snap.tooltip_padding,
            context_menu_min_w: snap.context_menu_min_w,
            scrollbar_width: snap.scrollbar_width,
            scrollbar_min_thumb: snap.scrollbar_min_thumb,
            scroll_speed: snap.scroll_speed,
            scroll_friction: snap.scroll_friction,
            tab_indicator_height: snap.tab_indicator_height,
            tab_fade_duration_ms: snap.tab_fade_duration_ms,
            table_header_height: snap.table_header_height,
            column_resize_handle_width: snap.column_resize_handle_width,
            column_resize_min_width: snap.column_resize_min_width,
            tree_indent: snap.tree_indent,
            tree_expand_duration_ms: snap.tree_expand_duration_ms,
            toggle_width: snap.toggle_width,
            toggle_height: snap.toggle_height,
            toggle_knob_inset: snap.toggle_knob_inset,
            spinner_size: snap.spinner_size,
            spinner_speed: snap.spinner_speed,
            modal_fade_duration_ms: snap.modal_fade_duration_ms,
            toast_fade_in_ms: snap.toast_fade_in_ms,
            toast_fade_out_ms: snap.toast_fade_out_ms,
            modal_corner_radius: snap.modal_corner_radius,
            modal_title_height: snap.modal_title_height,
            modal_padding: snap.modal_padding,
            modal_min_width: snap.modal_min_width,
            modal_max_width: snap.modal_max_width,
            text_xs: snap.text_xs,
            text_sm: snap.text_sm,
            text_base: snap.text_base,
            text_lg: snap.text_lg,
            text_xl: snap.text_xl,
            text_2xl: snap.text_2xl,
            line_spacing: snap.line_spacing,
            breakpoint_compact: snap.breakpoint_compact,
            breakpoint_expanded: snap.breakpoint_expanded,
            form_label_gap: snap.form_label_gap,
            form_helper_gap: snap.form_helper_gap,
            form_helper_font_size: snap.form_helper_font_size,
            toast_duration_ms: snap.toast_duration_ms,
            toast_max_visible: snap.toast_max_visible,
            toast_margin: snap.toast_margin,
            disabled_dash_len: snap.disabled_dash_len,
            disabled_dash_gap: snap.disabled_dash_gap,
            disabled_dash_thickness: snap.disabled_dash_thickness,
            hover_duration_ms: snap.hover_duration_ms,
            checkbox_size: snap.checkbox_size,
            radio_size: snap.radio_size,
            radio_dot_size: snap.radio_dot_size,
            slider_track_height: snap.slider_track_height,
            split_pane_divider: snap.split_pane_divider,
            badge_pad_x: snap.badge_pad_x,
            badge_pad_y: snap.badge_pad_y,
            chip_pad_y: snap.chip_pad_y,
            modal_max_height_ratio: snap.modal_max_height_ratio,
            modal_shadow_alpha: snap.modal_shadow_alpha,
            modal_margin: snap.modal_margin,
            modal_close_btn_size: snap.modal_close_btn_size,
            modal_vertical_offset: snap.modal_vertical_offset,
            fg_on_accent: lerp_color(a.fg_on_accent, b.fg_on_accent, t),
        }
    }

    /// Derive a background color for a given pseudo-state.
    ///
    /// `Hovered` lightens by 8%, `Active` darkens by 5%, `Disabled` desaturates.
    pub fn derive_bg(&self, base: Color, state: StyleState) -> Color {
        match state {
            StyleState::Normal => base,
            StyleState::Hovered => lighten(base, 0.08),
            StyleState::Focused => lerp_color(base, self.accent, 0.15),
            StyleState::Active => darken(base, 0.05),
            StyleState::Disabled => desaturate(base, 0.6),
        }
    }

    /// Derive a foreground color for a given pseudo-state.
    ///
    /// `Hovered` lightens slightly, `Disabled` desaturates.
    pub fn derive_fg(&self, base: Color, state: StyleState) -> Color {
        match state {
            StyleState::Normal => base,
            StyleState::Hovered => lighten(base, 0.05),
            StyleState::Focused => base,
            StyleState::Active => darken(base, 0.03),
            StyleState::Disabled => desaturate(base, 0.6),
        }
    }

    /// Resolve a `TextSize` to a concrete pixel value.
    pub fn resolve_text_size(&self, size: TextSize) -> f32 {
        match size {
            TextSize::Xs => self.text_xs,
            TextSize::Sm => self.text_sm,
            TextSize::Base => self.text_base,
            TextSize::Lg => self.text_lg,
            TextSize::Xl => self.text_xl,
            TextSize::Xxl => self.text_2xl,
            TextSize::Custom(v) => v,
        }
    }

    fn layout_defaults() -> Self {
        Self {
            bg_base: Color::new(0.0, 0.0, 0.0, 1.0),
            bg_surface: Color::new(0.0, 0.0, 0.0, 1.0),
            bg_raised: Color::new(0.0, 0.0, 0.0, 1.0),
            bg_input: Color::new(0.0, 0.0, 0.0, 1.0),
            fg: Color::new(0.0, 0.0, 0.0, 1.0),
            fg_muted: Color::new(0.0, 0.0, 0.0, 1.0),
            fg_dim: Color::new(0.0, 0.0, 0.0, 1.0),
            fg_label: Color::new(0.0, 0.0, 0.0, 1.0),
            accent: Color::new(0.0, 0.0, 0.0, 1.0),
            accent_dim: Color::new(0.0, 0.0, 0.0, 0.0),
            accent_hover: Color::new(0.0, 0.0, 0.0, 1.0),
            green: Color::new(0.0, 0.0, 0.0, 1.0),
            amber: Color::new(0.0, 0.0, 0.0, 1.0),
            red: Color::new(0.0, 0.0, 0.0, 1.0),
            border: Color::new(0.0, 0.0, 0.0, 1.0),
            green_button_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            secondary_button_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            secondary_button_hover: Color::new(0.0, 0.0, 0.0, 1.0),
            danger_button_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            danger_button_hover: Color::new(0.0, 0.0, 0.0, 1.0),
            shadow: Color::new(0.0, 0.0, 0.0, 0.0),
            toast_error_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            toast_success_bg: Color::new(0.0, 0.0, 0.0, 1.0),

            corner_radius: 6.0,
            padding: 12.0,
            input_padding: 8.0,
            item_height: 32.0,
            font_size: 14.0,
            header_font_size: 11.0,
            cursor_width: 1.5,
            cursor_blink_ms: 530,

            button_height: 36.0,
            small_button_height: 28.0,
            small_button_min_w: 80.0,
            focus_ring_expand: 2.0,
            dropdown_gap: 2.0,
            label_pad_y: 4.0,
            heading_font_size: 20.0,
            heading_height: 28.0,
            drop_zone_height: 120.0,
            drop_zone_dash: 8.0,
            drop_zone_dash_gap: 6.0,
            drop_zone_dash_thickness: 1.5,
            progress_bar_height: 3.0,
            status_dot_radius: 4.0,
            toast_w: 300.0,
            toast_h: 36.0,

            disabled_fg: Color::new(0.0, 0.0, 0.0, 1.0),
            disabled_border: Color::new(0.0, 0.0, 0.0, 1.0),
            disabled_bg: Color::new(0.0, 0.0, 0.0, 1.0),

            tooltip_delay_ms: 500,
            tooltip_font_size: 12.0,
            tooltip_padding: 6.0,
            tooltip_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            tooltip_fg: Color::new(0.0, 0.0, 0.0, 1.0),

            context_menu_min_w: 160.0,

            scrollbar_width: 8.0,
            scrollbar_min_thumb: 20.0,
            scroll_speed: 40.0,
            scroll_friction: 0.92,

            tab_indicator_height: 2.0,
            tab_fade_duration_ms: 150.0,

            table_header_height: 32.0,
            table_zebra_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            column_resize_handle_width: 6.0,
            column_resize_min_width: 40.0,

            tree_indent: 20.0,
            tree_expand_duration_ms: 200.0,

            toggle_width: 36.0,
            toggle_height: 20.0,
            toggle_knob_inset: 2.0,

            spinner_size: 24.0,
            spinner_speed: 0.8,

            modal_fade_duration_ms: 200.0,
            toast_fade_in_ms: 150.0,
            toast_fade_out_ms: 300.0,

            modal_backdrop: Color::new(0.0, 0.0, 0.0, 0.5),
            modal_corner_radius: 8.0,
            modal_title_height: 40.0,
            modal_padding: 16.0,
            modal_min_width: 300.0,
            modal_max_width: 600.0,

            text_xs: 10.0,
            text_sm: 12.0,
            text_base: 14.0,
            text_lg: 16.0,
            text_xl: 20.0,
            text_2xl: 28.0,

            line_spacing: 2.0,

            breakpoint_compact: 600.0,
            breakpoint_expanded: 1200.0,

            form_label_gap: 4.0,
            form_helper_gap: 2.0,
            form_helper_font_size: 12.0,

            toast_info_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            toast_warning_bg: Color::new(0.0, 0.0, 0.0, 1.0),
            toast_duration_ms: 3000,
            toast_max_visible: 5,
            toast_margin: 12.0,

            disabled_dash_len: 6.0,
            disabled_dash_gap: 4.0,
            disabled_dash_thickness: 1.0,

            hover_duration_ms: 100.0,

            checkbox_size: 16.0,
            radio_size: 16.0,
            radio_dot_size: 6.0,
            slider_track_height: 4.0,
            split_pane_divider: 5.0,
            badge_pad_x: 6.0,
            badge_pad_y: 2.0,
            chip_pad_y: 4.0,

            modal_max_height_ratio: 0.8,
            modal_shadow_alpha: 0.3,
            modal_margin: 32.0,
            modal_close_btn_size: 24.0,
            modal_vertical_offset: 0.15,

            fg_on_accent: Color::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}

/// Builder for constructing custom themes.
pub struct ThemeBuilder {
    base: Theme,
}

impl ThemeBuilder {
    pub fn from(base: Theme) -> Self {
        Self { base }
    }

    pub fn from_dark() -> Self {
        Self { base: Theme::dark() }
    }

    pub fn from_light() -> Self {
        Self { base: Theme::light() }
    }

    pub fn bg_base(mut self, c: Color) -> Self { self.base.bg_base = c; self }
    pub fn bg_surface(mut self, c: Color) -> Self { self.base.bg_surface = c; self }
    pub fn bg_raised(mut self, c: Color) -> Self { self.base.bg_raised = c; self }
    pub fn bg_input(mut self, c: Color) -> Self { self.base.bg_input = c; self }
    pub fn fg(mut self, c: Color) -> Self { self.base.fg = c; self }
    pub fn fg_muted(mut self, c: Color) -> Self { self.base.fg_muted = c; self }
    pub fn fg_dim(mut self, c: Color) -> Self { self.base.fg_dim = c; self }
    pub fn fg_label(mut self, c: Color) -> Self { self.base.fg_label = c; self }
    pub fn accent(mut self, c: Color) -> Self { self.base.accent = c; self }
    pub fn accent_dim(mut self, c: Color) -> Self { self.base.accent_dim = c; self }
    pub fn accent_hover(mut self, c: Color) -> Self { self.base.accent_hover = c; self }
    pub fn green(mut self, c: Color) -> Self { self.base.green = c; self }
    pub fn amber(mut self, c: Color) -> Self { self.base.amber = c; self }
    pub fn red(mut self, c: Color) -> Self { self.base.red = c; self }
    pub fn border(mut self, c: Color) -> Self { self.base.border = c; self }
    pub fn font_size(mut self, s: f32) -> Self { self.base.font_size = s; self }
    pub fn corner_radius(mut self, r: f32) -> Self { self.base.corner_radius = r; self }
    pub fn padding(mut self, p: f32) -> Self { self.base.padding = p; self }

    /// Derive accent, accent_dim, and accent_hover from an HSL hue (0-360 degrees).
    pub fn accent_from_hue(mut self, hue_deg: f32) -> Self {
        let (r, g, b) = hsl_to_rgb(hue_deg, 0.7, 0.63);
        self.base.accent = Color::new(r, g, b, 1.0);
        self.base.accent_dim = Color::new(r, g, b, 0.18);
        let (rh, gh, bh) = hsl_to_rgb(hue_deg, 0.75, 0.72);
        self.base.accent_hover = Color::new(rh, gh, bh, 1.0);
        self
    }

    pub fn build(self) -> Theme {
        self.base
    }
}

/// Theme transition helper for smooth switching between themes.
pub struct ThemeTransition {
    pub from: Theme,
    pub to: Theme,
    pub start: Instant,
    pub duration_ms: f32,
}

impl ThemeTransition {
    pub fn new(from: Theme, to: Theme, duration_ms: f32) -> Self {
        Self {
            from,
            to,
            start: Instant::now(),
            duration_ms,
        }
    }

    /// Get the interpolated theme at the current time.
    pub fn current(&self) -> Theme {
        let elapsed = self.start.elapsed().as_millis() as f32;
        let t = (elapsed / self.duration_ms).clamp(0.0, 1.0);
        Theme::lerp(&self.from, &self.to, t)
    }

    /// Whether the transition has completed.
    pub fn is_done(&self) -> bool {
        self.start.elapsed().as_millis() as f32 >= self.duration_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_and_light_have_distinct_bg_base() {
        let dark = Theme::dark();
        let light = Theme::light();
        assert_ne!(dark.bg_base, light.bg_base);
    }

    #[test]
    fn scaled_doubles_dimensional_fields() {
        let base = Theme::dark();
        let scaled = base.scaled(2.0);
        assert!((scaled.corner_radius - base.corner_radius * 2.0).abs() < 1e-6);
        assert!((scaled.padding - base.padding * 2.0).abs() < 1e-6);
        assert!((scaled.font_size - base.font_size * 2.0).abs() < 1e-6);
        assert!((scaled.button_height - base.button_height * 2.0).abs() < 1e-6);
        assert!((scaled.toast_w - base.toast_w * 2.0).abs() < 1e-6);
        assert!((scaled.scrollbar_width - base.scrollbar_width * 2.0).abs() < 1e-6);
    }

    #[test]
    fn scaled_preserves_colors() {
        let base = Theme::dark();
        let scaled = base.scaled(2.0);
        assert_eq!(scaled.bg_base, base.bg_base);
        assert_eq!(scaled.fg, base.fg);
        assert_eq!(scaled.accent, base.accent);
        assert_eq!(scaled.red, base.red);
        assert_eq!(scaled.border, base.border);
    }

    #[test]
    fn high_contrast_has_pure_black_bg_base() {
        let hc = Theme::high_contrast();
        assert_eq!(hc.bg_base, Color::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn high_contrast_exists_and_differs_from_dark() {
        let hc = Theme::high_contrast();
        let dark = Theme::dark();
        assert_ne!(hc.fg, dark.fg);
        assert_ne!(hc.accent, dark.accent);
    }
}

/// Lighten a color by a fraction (0.0–1.0).
fn lighten(c: Color, amount: f32) -> Color {
    Color::new(
        (c.r + (1.0 - c.r) * amount).min(1.0),
        (c.g + (1.0 - c.g) * amount).min(1.0),
        (c.b + (1.0 - c.b) * amount).min(1.0),
        c.a,
    )
}

/// Darken a color by a fraction (0.0–1.0).
fn darken(c: Color, amount: f32) -> Color {
    Color::new(
        (c.r * (1.0 - amount)).max(0.0),
        (c.g * (1.0 - amount)).max(0.0),
        (c.b * (1.0 - amount)).max(0.0),
        c.a,
    )
}

/// Desaturate a color by blending toward its luminance.
fn desaturate(c: Color, amount: f32) -> Color {
    let lum = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    Color::new(
        c.r + (lum - c.r) * amount,
        c.g + (lum - c.g) * amount,
        c.b + (lum - c.b) * amount,
        c.a,
    )
}

/// Convert HSL to linear RGB (h in degrees, s/l in 0-1).
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}
