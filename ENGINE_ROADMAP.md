# esox — 3D Engine Roadmap

## Goal

Consolidate the GPU stacks from phosphor_3d and esocidae into `esox_gfx`, then build a game engine crate on top. Stop maintaining three forks of the same wgpu infrastructure.

Linux-only. Compute-efficient. Accessible hardware targets.

## Phase 0 — Consolidate GPU stack into esox_gfx ✓

Brought phosphor_3d's mesh rendering into `esox_gfx::mesh3d` behind `mesh3d` feature gate.

- `vertex.rs`, `mesh.rs`, `instance.rs`, `camera.rs`, `transform.rs`, `light.rs`, `material.rs`, `renderer.rs` — all ported
- Procedural generators: cube, sphere, plane, cylinder, cone, torus
- Materials: Unlit, Lit, PBR with blend modes, pipeline key caching
- Draw batching: opaque sorted by pipeline/material/mesh, transparent sorted back-to-front
- 214 tests pass (`cargo test -p esox_gfx --features mesh3d`)
- Particle system not yet ported (deferred to when needed)

## Phase 1 — 3D rendering foundation ✓

Wired 3D renderer into esox_platform's event loop with 2D UI composited on top.

- `GpuContext::acquire_surface()` — pre-acquire surface texture for sharing between passes
- `SurfaceFrame` / `ColorLoadOp` — surface wrapper + Clear/Load enum for 2D pass
- `FrameEncoder::encode_and_submit_with_surface()` — accepts pre-acquired surface, skips MSAA on Load to avoid clobbering 3D content
- `AppDelegate::on_pre_render()` — new trait method, returns `Vec<CommandBuffer>` for 3D pass
- Platform event loop: acquire surface → on_pre_render (3D) → submit → on_redraw (2D) → encode_and_submit_with_surface
- `examples/demo3d/` — spinning lit cube with 2D HUD overlay, runs at vsync on Intel Arc

**Known Phase 1 limitations:**
- 2D overlay not antialiased when composited over 3D (MSAA skipped to preserve 3D content)
- No post-process integration with 3D pass yet (post-process only sees the 2D pass's offscreen target)

## Phase 2 — Asset loading ✓

- glTF mesh loading via `gltf` crate (meshes, materials, scene hierarchy)
- Texture loading for materials (albedo, normal map, metallic-roughness, emissive)
- Skeletal animation data from glTF (joints, skins, animation clips)
- Compute shader skinning (offload joint transforms to GPU)
- Scene graph with hierarchical transform propagation
- `upload_gltf_scene()` — one-call pipeline from file to GPU-resident scene

**Done when:** can load and render an animated glTF character with PBR materials.

## Phase 3 — Efficiency ✓

These are the things that make it run well on modest hardware.

- **Frustum culling** — don't submit what the camera can't see
- **Draw call merging** — phosphor_3d already merges consecutive draws with same material+mesh; extend with multi-draw-indirect
- **LOD** — mesh detail swap by camera distance
- **Spatial indexing** — BVH or grid for broad-phase culling (extend DamageTracker concepts)
- **GPU-driven rendering** — compute shader that writes draw-indirect buffer after culling (stretch goal)

**Done when:** scene with 10K objects renders at 60fps on integrated graphics.

## Phase 4 — Visual quality

- **Shadow maps** — single cascaded shadow map for directional light
- **Spot lights** — extend light.rs
- **Environment mapping** — IBL for PBR (diffuse irradiance + specular prefilter)
- **Post-processing** — bloom already exists; add SSAO, motion blur as optional passes
- **SDF effects** — optional render pass using existing raymarching pipeline for particles, procedural terrain, volumetrics

**Done when:** scene looks good enough that you'd ship it.

## Phase 5 — Game engine crate (`esox_engine`)

Thin crate on top of esox_gfx + esox_platform that adds game-specific abstractions.

- **Fixed timestep** — decouple update from render (`accumulator += dt; while acc >= TICK { update() }`)
- **Entity/component storage** — `hecs` or hand-rolled sparse sets
- **Input action mapping** — bind physical keys to semantic actions, gamepad support
- **Audio** — `kira` integration (spatial audio, music, SFX)
- **Physics hooks** — trait-based integration point for rapier or custom
- **Asset manager** — async loading, handle-based references, hot-reload

**Done when:** can build a simple 3D game (e.g. a platformer) using only esox crates.

## Phase 6 — Tooling (stretch)

- Scene editor (using esox_ui for the editor UI, esox_gfx mesh3d for the viewport)
- Shader hot-reload (esocidae already has this — port the file watcher + Naga validation)
- GPU profiler overlay (extend PerfMonitor with GPU timestamp queries)
- Asset pipeline CLI (mesh optimization, texture compression)

## Dependency additions

```toml
# Phase 0
glam = "0.29"          # math (vec3, mat4, quat) — only with mesh3d feature

# Phase 2
gltf = "1"             # asset loading

# Phase 5
hecs = "0.10"          # ECS
kira = "3"             # audio
```

## Crate graph (target state)

```
esox_engine (Phase 5)
  ├── esox_gfx [mesh3d, particles, postprocess]
  ├── esox_platform
  ├── esox_ui (optional — for in-game UI)
  ├── hecs
  └── kira

esox_ui (unchanged)
  ├── esox_gfx [default]
  ├── esox_font
  └── esox_platform

phosphor (migrated consumer)
  ├── esox_gfx [mesh3d, particles]
  ├── phosphor_audio
  └── phosphor_av
```

## Non-goals

- macOS/Windows support
- Deferred rendering (forward is simpler, faster for moderate scenes, easier MSAA)
- Built-in physics engine (integrate, don't build)
- Scripting language (Rust is the scripting language)
