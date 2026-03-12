//! Separator widget — horizontal line.

use esox_gfx::ShapeBuilder;

use crate::Ui;

impl<'f> Ui<'f> {
    /// Draw a horizontal separator line.
    pub fn separator(&mut self) {
        let rect = self.allocate_rect(self.region.w, 1.0);
        self.frame.push(
            ShapeBuilder::rect(rect.x, rect.y, rect.w, 1.0)
                .color(self.theme.border)
                .build(),
        );
    }
}
