# esox — GPU-native widget toolkit for Linux

## What this is

A standalone Rust widget toolkit extracted from the esocidae terminal emulator. The goal is to solve the "building native apps in Rust on Linux sucks" problem — replace the WebKitGTK/Tauri approach with a GPU-rendered native toolkit that speaks Wayland directly via winit/wgpu.

Think "Flutter's rendering engine as a Rust library with no Dart, no VM, no Google."

Linux-only. We don't care about macOS or Windows.

## Workspace layout

```
crates/
  esox_gfx/       GPU rendering engine — arena scene graph, instanced quad pipeline, atlas, bloom
                   Leaf crate, no internal deps. Uses wgpu 28, naga 28, bytemuck, winit.

  esox_font/      Font shaping (rustybuzz), rasterization (swash), glyph caching with LRU eviction
                   Depends on esox_gfx for atlas allocation. Uses fontdb for system font discovery.

  esox_ui/        Immediate-mode widget library — the main crate to evolve
                   Depends on esox_gfx + esox_font. Edition 2021 (rest are 2024).

  esox_platform/  Windowing (winit), GPU surface init, clipboard (arboard), optional sandbox
                   Has its own config.rs (PlatformConfig) replacing the old eso_config dependency.
                   Sandbox (seccomp/landlock) is opt-in via "sandbox" feature.

examples/
  demo/           Placeholder binary that compiles against all crates — use for testing widgets
```

## Origin

Extracted from /home/esoc/code/esocidae/ (the esocidae terminal emulator workspace). The crates were copied, renamed from `eso_*` to `esox_*`, and the `eso_config` dependency in esox_platform was replaced with a local `PlatformConfig` struct. esox_ui was never actually used by the terminal — it was already a standalone library sitting in the workspace.

Related projects in /home/esoc/code/:
- **esocidae/** — terminal emulator (esox_gfx originated here)
- **phosphor/** — audio-reactive visualizer (evolved fork of the GPU stack, has compute shaders + 3D)
- **scry/** — vector graphics engine for terminal (tiny-skia based, different approach)
- **dwnldr/gui/** — media downloader that consumed the old eso_ui (validates non-terminal usage)

## Current state of esox_ui

### What works
- **Widgets**: button (full/max-width/ghost/small), label (regular/colored/heading/muted/header), text_input (single-line with cursor/selection/scroll), slider, checkbox, select (dropdown with overlay), separator, drop_zone (file picker)
- **Layout**: vertical (default), horizontal (`row()` closure), `max_width()` centering, `padding()`, `indent()`, `spacing()`, nesting via closure-based layout stack
- **Interaction**: FNV1a widget IDs, hit testing, focus chain with Tab/Shift+Tab, hover animations (ease-out cubic), mouse click tracking
- **Theming**: full Theme struct with dark/light presets, colors, layout constants, animation timings
- **Text**: single-size rendering per call, glyph caching, font fallback chains

### What's missing (priority order for building real apps)

1. **Scroll containers** — clip_rect exists in LayoutContext but is unused. Need scissor rect rendering in esox_gfx and a `scrollable()` container widget. This is the #1 blocker.

2. **Multi-line text layout** — currently single-line only. Need word wrapping, line breaking, inline spans. This is hard and important. Look at how cosmic-text or parley approach this.

3. **Clipping** — tied to scroll, but also needed for overflow handling generally. The GPU pipeline needs scissor rect support.

4. **Rich text** — bold/italic/colored spans within a single text block. Builds on multi-line.

5. **Multi-line text input** — textarea widget. Builds on multi-line layout.

6. **Progress bar** — theme has sizing for it but no widget exists.

7. **Tabs widget** — tab bar with content switching.

8. **Context menus** — right-click menus, similar overlay approach to select dropdown.

9. **Lists/tables** — virtualized scrolling list for large datasets.

10. **Tooltips** — hover-delayed overlay text.

## Architecture notes

- **Immediate mode**: widgets are method calls on `Ui` that return `Response { clicked, hovered, focused, changed }`. No retained widget tree. App owns all state.
- **GPU pipeline**: single draw call per frame via instanced quads. Arena scene graph (`Vec<Option<Node>>` with free list) gives O(1) alloc with zero per-frame allocations in steady state.
- **State structs**: `UiState` (central — focus, mouse, keys, overlays, hover anims), `InputState` (per text field), `SelectState` (per dropdown), `DropZoneState` (per drop zone).
- **Text pipeline**: esox_font::TextShaper (rustybuzz) → GlyphRasterizer (swash) → GPU atlas via esox_gfx.

## Recommended first task

**Add scissor rect support to esox_gfx and build a `scrollable()` container in esox_ui.** This unblocks everything else. The pattern:

1. In `esox_gfx`, add scissor rect to the render pass (wgpu supports `set_scissor_rect` natively)
2. In `esox_ui`, add a `scrollable(id, height, |ui| { ... })` method that:
   - Allocates a fixed-height region
   - Tracks scroll offset in UiState (already has `scroll_offsets: HashMap<u64, f32>`)
   - Sets clip_rect on the layout context
   - Offsets child positions by scroll amount
   - Renders a scrollbar
   - Handles mouse wheel events

After that, multi-line text layout is the next big piece.

## Build & check

```bash
cd /home/esoc/code/aaplsucks
cargo check   # should be clean, zero warnings
cargo build --example demo  # placeholder binary
```
