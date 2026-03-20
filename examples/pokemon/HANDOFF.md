# Pokemon 3D Port — Phase 2 Complete

## Where We Are

Working Phase 2 prototype at `examples/pokemon/` with:

- **Coherent building geometry** — flood-fill groups adjacent building tiles into structures, spawns single box mesh per building with peaked roofs and door recesses (no more fragmented cubes)
- **Over-the-shoulder camera** — `FollowCameraController` positioned behind and above player (offset `0, 4.5, 5`), smooth following
- **Grid-snapped movement** — tile-based movement with collision checking, smooth interpolation between tiles
- **PBR materials + lighting** — all materials are PBR, directional sun light, procedural IBL, shadows, bloom, tone mapping
- **Procedural trees** — cylinder trunk + sphere canopy per tree tile
- **Water plane** — alpha-blended PBR surface at lower Y
- **Grass tufts** — small cone meshes scattered across grass tiles (2 per tile)
- **Fences** — thin wall meshes
- **Player capsule** — cylinder body + sphere head placeholder

### Technical Details
- Config: `postprocess: true`, `shadows: true`, `msaa: 4`
- Materials: all `MaterialType::PBR` with appropriate roughness/metallic
- Water uses `BlendMode3D::AlphaBlend`
- Map: 24x20 grid, collision grid for movement blocking
- Buildings found via flood-fill: ~3 groups (2 houses + Oak's Lab)
- Camera: `FollowCameraController` with smoothing=6.0, fov=55°

### Running
```bash
cd /home/esoc/code/aaplsucks && FIRERED_PATH=/home/esoc/pokefirered cargo run -p pokemon
```

## Architecture

The code is a single `main.rs` (~1050 lines) with these sections:

1. **Map parsing** (lines 1-140) — `parse_map_bin`, `classify_tile`, metatile behavior lookup
2. **Building flood-fill** (lines 142-210) — `find_buildings`, `BuildingFootprint`
3. **Mesh generators** (lines 212-310) — `make_building_box` (5-face no-bottom box), `make_peaked_roof` (triangular prism)
4. **Game state** (lines 340-400) — `PokemonViewer` with `FollowCameraController`, collision grid, grid-snapped movement
5. **Init** (lines 400-870) — loads map, creates materials, spawns buildings/tiles/trees/water/grass/player/camera/lights
6. **Update** (lines 870-960) — grid-snapped movement with collision, camera follow
7. **Render** (lines 960-970) — camera interpolation

## What's Next (Phase 3)

### Load a real player character
- Find/generate a low-poly humanoid glTF (Kenney, Quaternius, or AI-generated)
- Wire up idle/walk animation via `AnimGraphRuntime` (see `combat_demo` for pattern)
- Replace capsule placeholder

### Improve buildings
- Window indentations on walls (small recessed cubes)
- Chimney on one house
- Oak's Lab should be larger/taller than houses

### Polish environment
- Animated water (UV scroll or vertex displacement in update loop)
- Flower patches on specific tiles
- Sign posts at map edges
- Ledges / elevation changes

### Game mechanics
- NPC spawning from `map.json` warp/event data
- Door interaction (approach + press key → warp)
- Tall grass encounter zones (visual indicator)
- Pokemon encounter system

## Key Files

```
examples/pokemon/src/main.rs              — all game code
examples/pokemon/Cargo.toml               — depends on esox_engine with rapier feature

crates/esox_engine/src/camera.rs          — FPS, orbit, follow camera controllers
crates/esox_engine/src/game.rs            — Game trait (init, update, render, ui)
crates/esox_gfx/src/mesh3d/material.rs    — MaterialType (Unlit, Lit, PBR), BlendMode3D
crates/esox_gfx/src/mesh3d/mesh.rs        — MeshData::cube/sphere/cylinder/cone/plane/torus
examples/platformer/src/main.rs           — reference for camera + grass tufts + particles
examples/combat_demo/src/main.rs          — reference for glTF character loading + animation

/home/esoc/pokefirered/data/layouts/PalletTown/map.bin
/home/esoc/pokefirered/data/maps/PalletTown/map.json
/home/esoc/pokefirered/data/tilesets/primary/general/tiles.png
/home/esoc/pokefirered/data/tilesets/secondary/pallet_town/tiles.png
```
