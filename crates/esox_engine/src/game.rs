//! The `Game` trait — user-facing game logic interface.

use crate::Ctx;

/// Trait implemented by the user to define game behavior.
///
/// The engine calls these methods at appropriate points in the frame loop:
/// - [`init`](Game::init) — once after GPU is ready
/// - [`update`](Game::update) — 0..N times per frame at fixed rate (default 60Hz)
/// - [`render`](Game::render) — once per frame after all updates
pub trait Game: 'static {
    /// One-time setup. Load assets, spawn entities, configure scene.
    fn init(&mut self, ctx: &mut Ctx);

    /// Fixed-rate update (default 60Hz). Game logic, input handling, physics.
    /// Called 0..N times per frame based on accumulated time.
    fn update(&mut self, ctx: &mut Ctx);

    /// Variable-rate render callback. Called once per frame after all updates.
    /// `alpha` is the interpolation factor [0, 1) between ticks for smooth rendering.
    fn render(&mut self, _ctx: &mut Ctx, _alpha: f32) {}

    /// 2D overlay (HUD, menus). Only available with the `ui` feature.
    #[cfg(feature = "ui")]
    fn ui(&mut self, _ui: &mut esox_ui::Ui, _ctx: &Ctx) {}

    /// Whether the game wants to exit.
    fn should_exit(&self) -> bool {
        false
    }
}
