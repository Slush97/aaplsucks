//! Card and surface container widgets — visual grouping with background + border.
//!
//! # Examples
//!
//! ```ignore
//! ui.card(|ui| {
//!     ui.heading("Settings");
//!     ui.text_input(id!("name"), &mut name, "Name…");
//! });
//!
//! ui.surface(|ui| {
//!     ui.label("Secondary content");
//! });
//! ```

use esox_gfx::Color;

use crate::layout::Rect;
use crate::paint;
use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a card container — `bg_raised` background with border and padding.
    ///
    /// Cards provide visual grouping for related content sections.
    pub fn card(&mut self, f: impl FnOnce(&mut Self)) {
        self.card_colored(self.theme.bg_raised, f);
    }

    /// Draw a card container with a custom background color.
    pub fn card_colored(&mut self, bg: Color, f: impl FnOnce(&mut Self)) {
        let pad = self.theme.padding;
        let radius = self.theme.corner_radius;
        let border_color = self.theme.border;

        // Save the insert point — push a transparent placeholder for the background.
        let placeholder_idx = self.frame.instance_len();
        self.frame.push(
            esox_gfx::ShapeBuilder::rect(0.0, 0.0, 0.0, 0.0)
                .color(Color::new(0.0, 0.0, 0.0, 0.0))
                .build(),
        );

        let start_y = self.cursor.y;
        self.padding(pad, f);
        let end_y = self.cursor.y;

        let card_rect = Rect::new(
            self.cursor.x,
            start_y,
            self.region.w,
            end_y - start_y,
        );

        // Replace placeholder with the correctly-sized background.
        self.frame.replace_instance(
            placeholder_idx,
            esox_gfx::ShapeBuilder::rect(card_rect.x, card_rect.y, card_rect.w, card_rect.h)
                .color(bg)
                .border_radius(esox_gfx::BorderRadius::uniform(radius))
                .build(),
        );

        // Draw border on top of content (fine for thin borders).
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(card_rect.x, card_rect.y, card_rect.w, 1.0),
            border_color,
            0.0,
        );
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(card_rect.x, card_rect.y + card_rect.h - 1.0, card_rect.w, 1.0),
            border_color,
            0.0,
        );
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(card_rect.x, card_rect.y, 1.0, card_rect.h),
            border_color,
            0.0,
        );
        paint::draw_rounded_rect(
            self.frame,
            Rect::new(card_rect.x + card_rect.w - 1.0, card_rect.y, 1.0, card_rect.h),
            border_color,
            0.0,
        );

        // Add spacing after the card.
        self.cursor.y += self.spacing;
    }

    /// Draw a surface container — `bg_surface` background with padding, no border.
    ///
    /// Surfaces are subtler than cards, good for secondary grouping.
    pub fn surface(&mut self, f: impl FnOnce(&mut Self)) {
        let pad = self.theme.padding;
        let radius = self.theme.corner_radius;
        let bg = self.theme.bg_surface;

        let placeholder_idx = self.frame.instance_len();
        self.frame.push(
            esox_gfx::ShapeBuilder::rect(0.0, 0.0, 0.0, 0.0)
                .color(Color::new(0.0, 0.0, 0.0, 0.0))
                .build(),
        );

        let start_y = self.cursor.y;
        self.padding(pad, f);
        let end_y = self.cursor.y;

        let surface_rect = Rect::new(
            self.cursor.x,
            start_y,
            self.region.w,
            end_y - start_y,
        );

        self.frame.replace_instance(
            placeholder_idx,
            esox_gfx::ShapeBuilder::rect(
                surface_rect.x,
                surface_rect.y,
                surface_rect.w,
                surface_rect.h,
            )
            .color(bg)
            .border_radius(esox_gfx::BorderRadius::uniform(radius))
            .build(),
        );

        self.cursor.y += self.spacing;
    }
}
