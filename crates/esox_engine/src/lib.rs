//! `esox_engine` — Game engine crate on top of `esox_gfx` + `esox_platform`.
//!
//! Provides game-specific abstractions: fixed timestep, ECS, input mapping,
//! asset management, audio, and physics hooks. Implement the [`Game`] trait
//! instead of [`AppDelegate`](esox_platform::AppDelegate).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use esox_engine::*;
//!
//! struct MyGame;
//!
//! impl Game for MyGame {
//!     fn init(&mut self, ctx: &mut Ctx) { /* spawn entities */ }
//!     fn update(&mut self, ctx: &mut Ctx) { /* game logic at fixed rate */ }
//! }
//!
//! fn main() {
//!     esox_engine::run(EngineConfig::default(), MyGame).unwrap();
//! }
//! ```

pub mod assets;
pub mod audio;
pub mod ecs;
pub mod engine;
pub mod game;
pub mod input;
pub mod physics;
pub mod time;

// Re-export core types.
pub use assets::{AssetHandle, AssetId, AssetManager, MaterialAsset, MeshAsset, TextureAsset};
pub use ecs::{
    Camera3D, Children, DirectionalLightComponent, GlobalTransform, MeshRenderer, Parent,
    PointLightComponent, SpotLightComponent, Transform3D,
};
pub use engine::EngineConfig;
pub use game::Game;
pub use input::{ActionBinding, AxisBinding, InputManager, MouseAxis};
pub use physics::{BodyDesc, BodyHandle, BodyType, NullPhysics, PhysicsBackend, RayHit};
pub use time::TimeState;

// Re-export commonly used types from dependencies.
pub use esox_gfx;
pub use esox_platform;
pub use glam;
pub use hecs;
pub use winit;

/// Fat context providing access to all engine subsystems.
pub struct Ctx<'a> {
    pub world: &'a mut hecs::World,
    pub input: &'a mut InputManager,
    pub time: &'a TimeState,
    pub renderer: &'a mut esox_gfx::mesh3d::Renderer3D,
    pub gpu: &'a esox_gfx::GpuContext,
    pub assets: &'a mut AssetManager,
    pub viewport: (u32, u32),
}

/// Run the engine with the given config and game implementation.
pub fn run(config: EngineConfig, game: impl Game) -> Result<(), esox_platform::Error> {
    let platform_config = config.platform.clone();
    let engine = engine::Engine::new(config, Box::new(game));
    esox_platform::run(platform_config, Box::new(engine))
}
