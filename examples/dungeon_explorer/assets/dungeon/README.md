# Dungeon Assets

Place `.glb` files in this directory to replace the procedural fallback geometry.

## Recommended: Kenney Dungeon Kit

1. Download from https://kenney.nl/assets/dungeon-kit
2. Export pieces as `.glb` and place them here
3. Name files to match expected tags:
   - `wall_straight.glb`, `wall_corner.glb`
   - `floor.glb`, `ceiling.glb`
   - `pillar.glb`
   - `crate.glb`, `barrel.glb`
   - `torch.glb`

The demo will auto-detect any `.glb` files and derive piece names from the filename stem.
If no assets are found, procedural colored cubes/spheres are used instead.
