# esox — GPU-native widget toolkit & 3D engine for Linux

## What this is

A standalone Rust widget toolkit extracted from the esocidae terminal emulator, now extended with a full 3D rendering engine. The goal is to solve the "building native apps in Rust on Linux sucks" problem — replace the WebKitGTK/Tauri approach with a GPU-rendered native toolkit that speaks Wayland directly via winit/wgpu.

Think "Flutter's rendering engine as a Rust library with no Dart, no VM, no Google."

Linux-only. We don't care about macOS or Windows.

## Workspace layout

```
crates/
  esox_gfx/       GPU rendering engine — arena scene graph, instanced quad pipeline, atlas, bloom
                   mesh3d feature: full 3D renderer (PBR, shadows, IBL, SSAO, motion blur, SDF)
                   Leaf crate, no internal deps. Uses wgpu 28, naga 28, bytemuck, winit, glam.

  esox_font/      Font shaping (rustybuzz), rasterization (swash), glyph caching with LRU eviction
                   Depends on esox_gfx for atlas allocation. R8 alpha mask format.

  esox_ui/        Immediate-mode widget library — production-ready
                   Depends on esox_gfx + esox_font. Edition 2021 (rest are 2024).

  esox_platform/  Windowing (winit), GPU surface init, clipboard (arboard), optional sandbox
                   Has its own config.rs (PlatformConfig) replacing the old eso_config dependency.
                   Sandbox (seccomp/landlock) is opt-in via "sandbox" feature.

apps/
  dwnldr-gui/    Media downloader GUI — real-world app using esox_ui

examples/
  demo/           2D widget showcase — tests all widgets
  demo3d/         Spinning 3D scene with 2D HUD overlay — tests mesh3d + platform integration
  bench3d/        10K object benchmark for 3D renderer performance
```

## Origin

Extracted from /home/esoc/code/esocidae/ (the esocidae terminal emulator workspace). The crates were copied, renamed from `eso_*` to `esox_*`, and the `eso_config` dependency in esox_platform was replaced with a local `PlatformConfig` struct. esox_ui was never actually used by the terminal — it was already a standalone library sitting in the workspace.

The 3D engine (`mesh3d`) was consolidated from phosphor_3d into esox_gfx behind a feature gate.

Related projects in /home/esoc/code/:
- **esocidae/** — terminal emulator (esox_gfx originated here)
- **phosphor/** — audio-reactive visualizer (evolved fork of the GPU stack, has compute shaders + 3D)
- **scry/** — vector graphics engine for terminal (tiny-skia based, different approach)
- **dwnldr/gui/** — media downloader that consumed the old eso_ui (validates non-terminal usage)

## Current state of esox_ui

### What works
- **Widgets**: button (full/max-width/ghost/small), label (regular/colored/heading/muted/header/wrapped/truncated/rich), text_input, text_area (with word wrap), slider, checkbox, radio, select (dropdown), separator, drop_zone, progress_bar, tabs, table (virtual-scrolled, sortable, resizable columns, multi-select), tree (expand/collapse, multi-select), modal, toast, scrollable, virtual_scroll, image, toggle, spinner, hyperlink, collapsing_header, card/surface (container), form_field, menu_bar, chip, badge, empty_state, status_bar, number_input, split_pane, combobox
- **Layout**: vertical (default), horizontal (`row()`), `max_width()`, `padding()`, `indent()`, `columns()`/`columns_spaced()` weighted flex, `constrained()` min/max/aspect, tree indent with animation
- **Interaction**: FNV1a widget IDs, hit testing, focus chain with Tab/Shift+Tab, hover animations (ease-out cubic), mouse click tracking, drag-and-drop, context menus, tooltips
- **Animation**: `ui.animate()`/`ui.animate_bool()`/`ui.is_animating()` public API wrapping internal `anim_t()` with easing + retargeting. Smooth inertial scrolling with exponential decay.
- **Theming**: `Theme` struct with dark/light/high-contrast presets, `ThemeBuilder`, `ThemeTransition` (animated switching), `scaled()` for HiDPI
- **Accessibility**: a11y tree built each frame, all widgets emit `A11yNode` metadata, AT-SPI2 bridge with role mapping and snapshot conversion
- **Damage tracking**: `DamageTracker` integration for frame-skip optimization when idle
- **Text**: text rendering with glyph caching, font fallback chains, word wrapping, IME composition, clipboard, undo/redo
- **Tests**: ~70 unit tests covering pure logic modules (id, layout, state, theme, paint, response, rich_text)

## Current state of 3D engine (esox_gfx mesh3d)

### What works
- **Mesh rendering**: vertex/index buffers, instanced draw, procedural generators (cube, sphere, plane, cylinder, cone, torus)
- **Materials**: Unlit, Lit (Lambertian), PBR (Cook-Torrance) with blend modes and pipeline key caching
- **Lighting**: directional light, up to 8 point lights, up to 4 spot lights with cone attenuation
- **Shadows**: cascaded shadow maps (up to 4 cascades), PCF soft shadows, configurable bias
- **IBL**: diffuse irradiance cubemap, specular prefiltered environment map (5 mips, GGX importance sampling), BRDF integration LUT, split-sum PBR
- **Post-processing**: bloom (dual-Kawase), SSAO (hemisphere kernel + bilateral blur), motion blur (velocity reconstruction), ACES tone mapping
- **SDF effects**: raymarching render pass with depth compositing, custom material support, alpha/additive blend
- **Asset loading**: glTF meshes, PBR materials, textures, skeletal animation, compute shader skinning
- **Scene graph**: hierarchical transforms, frustum culling, BVH spatial indexing, LOD, draw call merging
- **Platform integration**: 3D + 2D compositing (on_pre_render → on_redraw), shared surface acquisition
- **Tests**: 318 tests in esox_gfx (mesh3d feature)

## Architecture notes

- **Immediate mode**: widgets are method calls on `Ui` that return `Response { clicked, hovered, focused, changed }`. No retained widget tree. App owns all state.
- **GPU pipeline**: single draw call per frame via instanced quads. Arena scene graph (`Vec<Option<Node>>` with free list) gives O(1) alloc with zero per-frame allocations in steady state.
- **3D pipeline**: forward renderer, opaque sorted by pipeline/material/mesh, transparent sorted back-to-front. Renders to offscreen HDR target, composited with post-processing.
- **State structs**: `UiState` (central — focus, mouse, keys, overlays, hover anims), `InputState` (per text field), `SelectState` (per dropdown), `DropZoneState` (per drop zone).
- **Text pipeline**: esox_font::TextShaper (rustybuzz) → GlyphRasterizer (swash, R8 alpha mask) → GPU atlas via esox_gfx.
- **Vsync**: `PresentMode::Fifo` + CPU-side frame throttling to monitor refresh rate.

## Next steps

See ENGINE_ROADMAP.md for the 3D engine roadmap. Next is Phase 5 (game engine crate).

For the 2D toolkit:
1. **Partial redraw** — scissor-rect partial redraw to only re-render damaged regions.
2. **AT-SPI2 bridge phase 2** — register D-Bus objects per-widget, implement `org.a11y.atspi.Accessible` + `Component` interfaces, emit `StateChanged` signals.
3. **Rich text editing** — inline styled spans in text_input/text_area.
4. **Custom widgets** — pattern/API for user-defined widgets outside the crate.

## Build & check

```bash
cd /home/esoc/code/aaplsucks
cargo check                  # should be clean
cargo test --workspace       # 443 tests, all pass
cargo build --example demo   # 2D widget demo
cargo run -p demo3d          # 3D scene + 2D overlay (Escape to quit)
cargo run -p bench3d         # 10K object benchmark
```
