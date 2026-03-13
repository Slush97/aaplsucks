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

## Phase 4 — Visual quality ✓

- **Shadow maps** ✓ — cascaded shadow maps (CSM) for directional lights with PCF soft shadows
- **Spot lights** ✓ — spot light support in light.rs with inner/outer cone angles
- **Environment mapping** ✓ — IBL for PBR (diffuse irradiance + specular prefilter, procedural generation)
- **Post-processing** ✓ — bloom, SSAO, motion blur, SDF effects
- **Point light shadows** ✓ — cube map shadow maps
- **Spot light shadows** ✓ — shadow maps for spot lights

**Done when:** scene looks good enough that you'd ship it. ✓

## Phase 5 — Game engine crate (`esox_engine`) ✓

Thin crate on top of esox_gfx + esox_platform that adds game-specific abstractions.

- **Fixed timestep** ✓ — decouple update from render (`accumulator += dt; while acc >= TICK { update() }`)
- **Entity/component storage** ✓ — `hecs` sparse sets with `Transform3D`, `GlobalTransform`, hierarchy propagation
- **Input action mapping** ✓ — bind physical keys/mouse buttons to semantic actions, axis bindings, scroll wheel support
- **Audio** ✓ — `kira` integration (spatial audio, music, SFX)
- **Physics hooks** ✓ — trait-based `PhysicsBackend` with rapier implementation
- **Asset manager** ✓ — handle-based references, name↔handle resolution, hot-reload

**Done when:** can build a simple 3D game (e.g. a platformer) using only esox crates. ✓

## Phase 6 — Tooling ✓ (partial)

- Shader hot-reload (file watcher + Naga validation) ✓
- Scene editor — deferred to Phase 9 (needs serialization first)
- GPU profiler overlay — moved to Phase 7
- Asset pipeline CLI (mesh optimization, texture compression) — deferred

## Phase 7 — Content pipeline ✓

Foundation for creating and persisting game content without hardcoding Rust.

- **Scene serialization** ✓ — `ron` format for saving/loading entity worlds (Transform3D, MeshRenderer, lights, hierarchy). Derive `Serialize`/`Deserialize` on core components. `SceneFile` type that round-trips a hecs World to disk.
- **Rapier3d integration** ✓ — wire `rapier3d` into the existing `PhysicsBackend` trait. Collider components (box, sphere, capsule, mesh), rigid body sync with Transform3D, contact/trigger events.
- **Debug overlay** ✓ — FPS counter, draw call count, entity count, physics step time. Render as esox_ui overlay on top of 3D. Toggle with a key binding (F3 or similar).
- **Prefab system** ✓ — serialized entity templates that can be instantiated at runtime. `instantiate_prefab()` spawns entities from a `SceneFile` with transform offset.

**Done when:** can build a level in code, save it to a `.scene.ron` file, quit, relaunch, and load it back identically. Physics objects collide via rapier. Debug overlay shows stats.

## Phase 8 — Game feel ✓

The systems that make games feel like games.

- **Particle system** ✓ — GPU compute-driven particles with emitter components. Spawn rate, lifetime, velocity, gravity, color/size interpolation. Indirect draw with existing instanced mesh pipeline.
- **Animation state machine** ✓ — 1D blend trees, crossfade blending, transition graph with conditions and priorities. `AnimGraphController` component drives `AnimationPlayer`.
- **Animation events** ✓ — `AnimEvent` on states, fired when playback crosses event time (handles looping wrap). Game code drains via `AnimGraphRuntime::drain_events()`.
- **Trigger volumes** ✓ — sensor collider regions that fire Enter/Stay/Exit events. `TriggerVolume` marker component, `PhysicsEntityMap` for handle↔entity resolution.
- **Collision events** ✓ — contact callbacks from rapier exposed via `drain_contacts()` / `drain_triggers()`, resolvable to ECS entities via `PhysicsEntityMap`.
- **Audio improvements** ✓ — `play_at_volume`, music crossfade (`MusicHandle`), `distance_attenuation`, collision-triggered sounds.
- **2D blend trees** — deferred (no current demo needs them)

**Done when:** a character can run through a particle-emitting trigger zone, blend between walk/run/jump animations, and hear a spatial sound on collision. ✓

## Phase 9 — Scene editor (MVP ✓, in progress)

Built with esox_ui + esox_gfx as `examples/editor/`, implements the `Game` trait. 3D scene renders through the normal engine pipeline; UI panels drawn on top via `Game::ui()`.

- **Editor camera** ✓ — orbit mode (MMB rotate, scroll zoom, Shift+MMB pan) and fly mode (RMB + WASD/QE). Smooth transitions between modes.
- **Viewport picking** ✓ — left-click ray-AABB intersection against all `(GlobalTransform, MeshRenderer)` entities, selects closest hit. Uses `mesh_local_aabb()` for per-mesh bounds.
- **Scene hierarchy** ✓ — scrollable tree panel of all entities using `tree_node` + `tree_indent`. Root entities = no `Parent`, recurses `Children`. Click to select. Smart labels from `Tag`, component type, or entity ID fallback.
- **Entity inspector** ✓ — property panel showing editable components for selected entity:
  - Transform3D: position, euler rotation (degrees), scale via `number_input`
  - Camera3D: FOV, near, far (read-only for now)
  - MeshRenderer: visible, tint (read-only)
  - Point/Spot/Directional lights: color, intensity, range with live editing via pending edit queue
- **Entity spawning** ✓ — menu bar Entity menu: Add Empty, Cube, Sphere, Point Light, Spot Light. Delete key to remove.
- **Translate gizmo** ✓ — 3-axis arrows (red/green/blue) rendered at selected entity, scale-invariant sizing
- **Menu bar** ✓ — File (New/Open/Save stubs, Quit), Edit (Delete, Duplicate stub), View (Reset Camera, Focus Selected), Entity (spawn primitives)
- **3-panel layout** ✓ — hierarchy (left ~15%) | viewport (center) | inspector (right ~25%) via `split_pane_h`
- **Transform gizmo interaction** — drag-to-move on axis planes (not yet implemented)
- **Save/Load** — wire up File menu to `save_scene()` / `load_scene()` (not yet implemented)
- **Undo/redo** — command pattern with `UndoStack` (not yet implemented)
- **Play/stop** — snapshot world state, run game systems, restore on stop (not yet implemented)
- **Asset browser** — file picker for meshes, textures, prefabs (not yet implemented)
- **GPU profiler overlay** — timestamp queries per render pass, bar chart (not yet implemented)

**Done when:** can visually place objects, set up lights, assign materials, save the scene, and load it in a standalone game binary.

## Phase 10 — Networking (future, optional)

Architecture is already compatible (deterministic fixed timestep, input-as-data, state/render separation). If pursued:

- Client-server model with authoritative server
- Input prediction + rollback on client
- Entity replication (delta-compressed component snapshots)
- Consider `lightyear` crate or custom UDP protocol
- Scope: LAN co-op first, internet later

**Not a priority — listed to document that the architecture intentionally keeps the door open.**

## Dependency additions

```toml
# Phase 0
glam = "0.29"          # math (vec3, mat4, quat) — only with mesh3d feature

# Phase 2
gltf = "1"             # asset loading

# Phase 5
hecs = "0.10"          # ECS
kira = "0.12"          # audio

# Phase 7
ron = "0.8"            # scene serialization format
serde = "1"            # derive Serialize/Deserialize on components
rapier3d = "0.22"      # physics engine (behind `rapier` feature)
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
- Built-in physics engine (integrate rapier, don't build from scratch)
- Scripting language (Rust is the scripting language)
- Internet multiplayer (LAN co-op is Phase 10 stretch goal, MMO-scale is out of scope)
