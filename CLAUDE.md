# ESOX Game Engine

Rust game engine built on wgpu with hecs ECS, Rapier3D physics, and an immediate-mode UI toolkit.

## Project Structure

- `crates/esox_gfx` — GPU rendering (2D quads, 3D meshes, particles, shadows, post-processing)
- `crates/esox_engine` — Game engine layer (ECS, physics, audio, assets, input, camera controllers)
- `crates/esox_ui` — Immediate-mode UI widget toolkit
- `crates/esox_font` — Font loading, shaping, rasterization
- `crates/esox_input` — Platform-independent input types
- `crates/esox_platform` — Windowing and platform integration (winit)
- `examples/` — 13 example projects (editor, platformer, combat_demo, factory, etc.)

## Development Workflow

### Building and Testing
```bash
cargo build                              # full workspace build
cargo test -p esox_gfx -p esox_engine    # run engine + gfx tests
cargo test                               # run all tests
```

### Git Practices
- Commit messages: imperative mood, describe the "why" not just "what"
- Stage specific files (`git add <files>`) — never use `git add -A` or `git add .`
- Exclude unrelated changes from commits (check `git status` / `git diff` first)
- Always verify `cargo build` and `cargo test` pass before committing
- Don't push unless explicitly asked

### Code Conventions
- Feature-gate optional subsystems in Cargo.toml (e.g., `serialization`, `rapier`, `audio`, `ui`)
- Use `#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]` for optional serde
- New ECS components go in `crates/esox_engine/src/ecs/` — register in `ecs/mod.rs` and re-export from `lib.rs`
- New engine modules go in `crates/esox_engine/src/` — register in `lib.rs`
- The `Ctx` struct in `lib.rs` is the fat context passed to game callbacks — extend it for new subsystems
- `EngineState` in `engine.rs` holds all mutable state — add new subsystems there
- Use `..Default::default()` when constructing structs with new fields to keep examples forward-compatible

### Key APIs
- `Game` trait (`game.rs`): `init`, `update` (fixed tick), `render` (variable + alpha), `ui`
- `Ctx`: provides `&mut World`, `InputManager`, `Renderer3D`, `AssetManager`, `PhysicsBackend`, `ChunkManager`
- `render_extraction_system`: extracts ECS entities → draw calls. Supports LOD and chunk filtering.
- `Camera`: supports `CameraMode::Perspective` and `CameraMode::Orthographic { ortho_size }`

### Testing
- `MeshHandle` has a `pub(crate)` inner field — use `unsafe { std::mem::transmute(id) }` in tests to create test handles
- Tests for esox_engine are in-module `#[cfg(test)]` blocks
- Scene serialization tests need the `serialization` feature enabled
