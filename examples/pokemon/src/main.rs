//! Pokemon 3D — Pallet Town over-the-shoulder explorer.
//!
//! Proper metatile-textured rendering with palette-aware tile composition,
//! coherent building geometry, PBR lighting, and follow camera.
//!
//! WASD to move, Escape to exit.

use esox_engine::*;
use esox_engine::glam::{Mat3, Quat, Vec3};
use esox_gfx::mesh3d::{
    BlendMode3D, MaterialDescriptor, MaterialHandle, MaterialType, MeshData,
    PostProcess3DConfig, ShadowConfig, Vertex3D,
};

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════
// Map parsing
// ═══════════════════════════════════════════════════════════════════════════

struct MapTile {
    metatile_id: u16,
    collision: u8,
    #[allow(dead_code)]
    elevation: u8,
}

struct PokemonMap {
    width: usize,
    #[allow(dead_code)]
    height: usize,
    tiles: Vec<MapTile>,
}

const NUM_METATILES_IN_PRIMARY: u16 = 640;

impl PokemonMap {
    fn tile(&self, x: usize, y: usize) -> &MapTile {
        &self.tiles[y * self.width + x]
    }
}

fn parse_map_bin(data: &[u8], width: usize, height: usize) -> PokemonMap {
    assert_eq!(data.len(), width * height * 2);
    let mut tiles = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let lo = data[i * 2] as u16;
        let hi = data[i * 2 + 1] as u16;
        let raw = lo | (hi << 8);
        tiles.push(MapTile {
            metatile_id: raw & 0x03FF,
            collision: ((raw >> 10) & 0x03) as u8,
            elevation: ((raw >> 12) & 0x0F) as u8,
        });
    }
    PokemonMap { width, height, tiles }
}

fn parse_metatile_behaviors(data: &[u8]) -> Vec<u16> {
    (0..data.len() / 4)
        .map(|i| {
            let attr = u32::from_le_bytes([
                data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3],
            ]);
            (attr & 0x1FF) as u16
        })
        .collect()
}

fn get_behavior(metatile_id: u16, primary: &[u16], secondary: &[u16]) -> u16 {
    if metatile_id < NUM_METATILES_IN_PRIMARY {
        primary.get(metatile_id as usize).copied().unwrap_or(0)
    } else {
        let idx = (metatile_id - NUM_METATILES_IN_PRIMARY) as usize;
        secondary.get(idx).copied().unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tile classification (geometry decisions only — textures come from metatiles)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TileKind {
    Grass,
    Path,
    Water,
    Tree,
    BuildingWall,
    BuildingRoof,
    Fence,
    Door,
}

fn classify_tile(tile: &MapTile, primary_beh: &[u16], secondary_beh: &[u16]) -> TileKind {
    let mid = tile.metatile_id;
    let beh = get_behavior(mid, primary_beh, secondary_beh);

    match beh {
        0x10..=0x1B => return TileKind::Water,
        0x02 => return TileKind::Grass,
        0x69 | 0x67 => return TileKind::Door,
        _ => {}
    }

    if matches!(mid, 291 | 298 | 299 | 300) {
        return TileKind::Water;
    }

    if mid < NUM_METATILES_IN_PRIMARY {
        if tile.collision > 0 {
            return match mid {
                20..=23 | 28..=31 | 36..=39 => TileKind::Tree,
                _ => TileKind::Fence,
            };
        }
        return match mid {
            1..=4 | 8 | 9 | 14 | 15 | 17 => TileKind::Path,
            _ => TileKind::Grass,
        };
    }

    let sid = mid - NUM_METATILES_IN_PRIMARY;

    if mid == 675 || mid == 684 {
        return TileKind::Door;
    }

    if tile.collision > 0 {
        return match sid {
            4 | 7 | 14 | 46 | 78 => TileKind::Fence,
            1..=3 | 5 | 6 | 8..=11 | 40..=41 | 48..=51 => TileKind::BuildingRoof,
            17..=20 | 24..=28 | 32..=36 | 44 | 45 => TileKind::BuildingWall,
            56..=60 | 64..=69 | 72..=77 | 80 | 88 => TileKind::BuildingWall,
            _ => TileKind::BuildingWall,
        };
    }

    match sid {
        14 => TileKind::Path,
        _ => TileKind::Grass,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Palette & tileset system — proper GBA metatile composition
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a JASC-PAL file into 16 RGB colors.
fn parse_jasc_pal(text: &str) -> [[u8; 3]; 16] {
    let mut colors = [[0u8; 3]; 16];
    for (i, line) in text.lines().skip(3).enumerate() {
        if i >= 16 { break; }
        let parts: Vec<u8> = line.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 3 {
            colors[i] = [parts[0], parts[1], parts[2]];
        }
    }
    colors
}

/// Load all 16 palettes from a tileset's palettes/ directory.
fn load_palettes(dir: &Path) -> [[[u8; 3]; 16]; 16] {
    let mut palettes = [[[0u8; 3]; 16]; 16];
    for p in 0..16 {
        let path = dir.join(format!("{:02}.pal", p));
        if let Ok(text) = std::fs::read_to_string(&path) {
            palettes[p] = parse_jasc_pal(&text);
        }
    }
    palettes
}

/// Build the combined BG palette as the GBA loads it.
/// Palettes 0-6 from primary, 7-12 from secondary, 13-15 from primary.
fn build_combined_palette(
    primary: &[[[u8; 3]; 16]; 16],
    secondary: &[[[u8; 3]; 16]; 16],
) -> [[[u8; 3]; 16]; 16] {
    let mut combined = [[[0u8; 3]; 16]; 16];
    for p in 0..7 { combined[p] = primary[p]; }
    for p in 7..13 { combined[p] = secondary[p]; }
    for p in 13..16 { combined[p] = primary[p]; }
    combined
}

/// Build reverse map: RGB → palette index for a palette-0 decoded PNG.
fn build_reverse_palette(pal0: &[[u8; 3]; 16]) -> HashMap<[u8; 3], u8> {
    pal0.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect()
}

/// Tile reference within a metatile definition.
#[derive(Clone, Copy)]
struct TileRef {
    tile_id: u16,
    xflip: bool,
    yflip: bool,
    palette_idx: u8,
}

/// Parse metatiles.bin → array of 8 tile refs per metatile.
fn parse_metatile_defs(data: &[u8]) -> Vec<[TileRef; 8]> {
    let count = data.len() / 16;
    (0..count)
        .map(|m| {
            let mut refs = [TileRef { tile_id: 0, xflip: false, yflip: false, palette_idx: 0 }; 8];
            for t in 0..8 {
                let off = m * 16 + t * 2;
                let raw = u16::from_le_bytes([data[off], data[off + 1]]);
                refs[t] = TileRef {
                    tile_id: raw & 0x3FF,
                    xflip: (raw >> 10) & 1 != 0,
                    yflip: (raw >> 11) & 1 != 0,
                    palette_idx: ((raw >> 12) & 0xF) as u8,
                };
            }
            refs
        })
        .collect()
}

const NUM_TILES_IN_PRIMARY: u16 = 640; // 8×8 pixel tiles in primary tileset

/// Composite a single metatile (16×16 px) into RGBA8 with proper palette switching.
fn composite_metatile(
    primary_img: &image::RgbaImage,
    secondary_img: &image::RgbaImage,
    primary_reverse: &HashMap<[u8; 3], u8>,
    secondary_reverse: &HashMap<[u8; 3], u8>,
    combined_pal: &[[[u8; 3]; 16]; 16],
    defs: &[TileRef; 8],
) -> Vec<u8> {
    let mut pixels = vec![0u8; 16 * 16 * 4];
    let quads: [(u32, u32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];

    for layer in 0..2u8 {
        for (q, &(qx, qy)) in quads.iter().enumerate() {
            let tr = defs[layer as usize * 4 + q];

            let (img, reverse) = if tr.tile_id < NUM_TILES_IN_PRIMARY {
                (primary_img as &image::RgbaImage, primary_reverse)
            } else {
                (secondary_img as &image::RgbaImage, secondary_reverse)
            };
            let local_id = if tr.tile_id < NUM_TILES_IN_PRIMARY {
                tr.tile_id as u32
            } else {
                (tr.tile_id - NUM_TILES_IN_PRIMARY) as u32
            };

            let tpr = img.width() / 8;
            let tc = local_id % tpr;
            let trow = local_id / tpr;

            for ty in 0..8u32 {
                for tx in 0..8u32 {
                    let sx = if tr.xflip { 7 - tx } else { tx };
                    let sy = if tr.yflip { 7 - ty } else { ty };
                    let ix = tc * 8 + sx;
                    let iy = trow * 8 + sy;

                    if ix >= img.width() || iy >= img.height() { continue; }

                    let p = img.get_pixel(ix, iy);
                    let rgb = [p[0], p[1], p[2]];
                    let idx = reverse.get(&rgb).copied().unwrap_or(0);

                    // Index 0 = transparent on GBA.
                    if idx == 0 {
                        if layer == 0 {
                            let px = (qy + ty) as usize * 16 + (qx + tx) as usize;
                            let o = px * 4;
                            pixels[o..o + 4].copy_from_slice(&[0, 0, 0, 0]);
                        }
                        continue;
                    }

                    let pal = &combined_pal[tr.palette_idx as usize];
                    let c = pal[idx as usize];
                    let px = (qy + ty) as usize * 16 + (qx + tx) as usize;
                    let o = px * 4;
                    pixels[o..o + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
                }
            }
        }
    }

    // Fill remaining transparent pixels with dark green.
    for i in (0..pixels.len()).step_by(4) {
        if pixels[i + 3] == 0 {
            pixels[i..i + 4].copy_from_slice(&[45, 85, 35, 255]);
        }
    }

    pixels
}

/// Scale a 16×16 RGBA image to 64×64 with nearest-neighbor (crisp pixel art).
fn scale_up_4x(pixels: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; 64 * 64 * 4];
    for y in 0..64u32 {
        for x in 0..64u32 {
            let sx = (x / 4) as usize;
            let sy = (y / 4) as usize;
            let si = (sy * 16 + sx) * 4;
            let di = (y as usize * 64 + x as usize) * 4;
            out[di..di + 4].copy_from_slice(&pixels[si..si + 4]);
        }
    }
    out
}

/// Compute the average non-transparent color of a metatile texture.
#[allow(dead_code)]
fn average_color(pixels: &[u8]) -> [f32; 4] {
    let mut r = 0u64;
    let mut g = 0u64;
    let mut b = 0u64;
    let mut count = 0u64;
    for i in (0..pixels.len()).step_by(4) {
        if pixels[i + 3] > 0 {
            r += pixels[i] as u64;
            g += pixels[i + 1] as u64;
            b += pixels[i + 2] as u64;
            count += 1;
        }
    }
    if count == 0 { return [0.5, 0.5, 0.5, 1.0]; }
    [
        r as f32 / count as f32 / 255.0,
        g as f32 / count as f32 / 255.0,
        b as f32 / count as f32 / 255.0,
        1.0,
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Building flood-fill
// ═══════════════════════════════════════════════════════════════════════════

struct BuildingFootprint {
    tiles: Vec<(usize, usize)>,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    has_door: bool,
}

fn is_building_tile(kind: TileKind) -> bool {
    matches!(kind, TileKind::BuildingWall | TileKind::BuildingRoof | TileKind::Door)
}

fn find_buildings(w: usize, h: usize, kinds: &[Vec<TileKind>]) -> Vec<BuildingFootprint> {
    let mut visited = vec![vec![false; w]; h];
    let mut buildings = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if visited[y][x] || !is_building_tile(kinds[y][x]) { continue; }
            let mut fp = BuildingFootprint {
                tiles: Vec::new(), min_x: x, min_y: y, max_x: x, max_y: y, has_door: false,
            };
            let mut queue = VecDeque::new();
            queue.push_back((x, y));
            visited[y][x] = true;
            while let Some((cx, cy)) = queue.pop_front() {
                fp.tiles.push((cx, cy));
                fp.min_x = fp.min_x.min(cx);
                fp.min_y = fp.min_y.min(cy);
                fp.max_x = fp.max_x.max(cx);
                fp.max_y = fp.max_y.max(cy);
                if kinds[cy][cx] == TileKind::Door { fp.has_door = true; }
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 { continue; }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if !visited[ny][nx] && is_building_tile(kinds[ny][nx]) {
                        visited[ny][nx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            buildings.push(fp);
        }
    }
    buildings
}

// ═══════════════════════════════════════════════════════════════════════════
// Mesh generators with tiling UVs
// ═══════════════════════════════════════════════════════════════════════════

/// Box mesh (no bottom face) with UVs that tile based on world dimensions.
fn make_building_box(width: f32, height: f32, depth: f32) -> MeshData {
    let hw = width * 0.5;
    let hd = depth * 0.5;
    let uw = width; // UV tiles = world units (1 texture per tile)
    let uh = height;
    let ud = depth;

    // (normal, positions[4], uvs[4])
    type Face = ([f32; 3], [[f32; 3]; 4], [[f32; 2]; 4]);
    let faces: [Face; 5] = [
        // Front (-Z)
        ([0.0, 0.0, -1.0],
         [[-hw, 0.0, -hd], [hw, 0.0, -hd], [hw, height, -hd], [-hw, height, -hd]],
         [[0.0, uh], [uw, uh], [uw, 0.0], [0.0, 0.0]]),
        // Back (+Z)
        ([0.0, 0.0, 1.0],
         [[hw, 0.0, hd], [-hw, 0.0, hd], [-hw, height, hd], [hw, height, hd]],
         [[0.0, uh], [uw, uh], [uw, 0.0], [0.0, 0.0]]),
        // Left (-X)
        ([-1.0, 0.0, 0.0],
         [[-hw, 0.0, hd], [-hw, 0.0, -hd], [-hw, height, -hd], [-hw, height, hd]],
         [[0.0, uh], [ud, uh], [ud, 0.0], [0.0, 0.0]]),
        // Right (+X)
        ([1.0, 0.0, 0.0],
         [[hw, 0.0, -hd], [hw, 0.0, hd], [hw, height, hd], [hw, height, -hd]],
         [[0.0, uh], [ud, uh], [ud, 0.0], [0.0, 0.0]]),
        // Top (+Y)
        ([0.0, 1.0, 0.0],
         [[-hw, height, -hd], [hw, height, -hd], [hw, height, hd], [-hw, height, hd]],
         [[0.0, 0.0], [uw, 0.0], [uw, ud], [0.0, ud]]),
    ];

    let mut vertices = Vec::with_capacity(20);
    let mut indices = Vec::with_capacity(30);
    for (normal, positions, uvs) in &faces {
        let base = vertices.len() as u32;
        for i in 0..4 {
            vertices.push(Vertex3D::new(positions[i], *normal, uvs[i]));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData::new(vertices, indices)
}

/// Peaked roof (triangular prism) with tiling UVs.
fn make_peaked_roof(width: f32, peak_height: f32, depth: f32) -> MeshData {
    let hw = width * 0.5;
    let hd = depth * 0.5;
    let slope_len = (peak_height * peak_height + hd * hd).sqrt();
    let ny = hd / slope_len;
    let nz = peak_height / slope_len;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Front slope
    {
        let n = [0.0, ny, -nz];
        let b = vertices.len() as u32;
        vertices.push(Vertex3D::new([-hw, 0.0, -hd], n, [0.0, slope_len]));
        vertices.push(Vertex3D::new([hw, 0.0, -hd], n, [width, slope_len]));
        vertices.push(Vertex3D::new([hw, peak_height, 0.0], n, [width, 0.0]));
        vertices.push(Vertex3D::new([-hw, peak_height, 0.0], n, [0.0, 0.0]));
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    // Back slope
    {
        let n = [0.0, ny, nz];
        let b = vertices.len() as u32;
        vertices.push(Vertex3D::new([hw, 0.0, hd], n, [0.0, slope_len]));
        vertices.push(Vertex3D::new([-hw, 0.0, hd], n, [width, slope_len]));
        vertices.push(Vertex3D::new([-hw, peak_height, 0.0], n, [width, 0.0]));
        vertices.push(Vertex3D::new([hw, peak_height, 0.0], n, [0.0, 0.0]));
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    // Left gable
    {
        let b = vertices.len() as u32;
        let n = [-1.0, 0.0, 0.0];
        vertices.push(Vertex3D::new([-hw, 0.0, -hd], n, [0.0, 1.0]));
        vertices.push(Vertex3D::new([-hw, peak_height, 0.0], n, [0.5, 0.0]));
        vertices.push(Vertex3D::new([-hw, 0.0, hd], n, [1.0, 1.0]));
        indices.extend_from_slice(&[b, b + 1, b + 2]);
    }
    // Right gable
    {
        let b = vertices.len() as u32;
        let n = [1.0, 0.0, 0.0];
        vertices.push(Vertex3D::new([hw, 0.0, hd], n, [0.0, 1.0]));
        vertices.push(Vertex3D::new([hw, peak_height, 0.0], n, [0.5, 0.0]));
        vertices.push(Vertex3D::new([hw, 0.0, -hd], n, [1.0, 1.0]));
        indices.extend_from_slice(&[b, b + 1, b + 2]);
    }
    MeshData::new(vertices, indices)
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants & collision
// ═══════════════════════════════════════════════════════════════════════════

const TILE_SIZE: f32 = 1.0;
const MOVE_SPEED: f32 = 4.0;
const WALL_HEIGHT: f32 = 2.2;
const ROOF_PEAK: f32 = 0.8;
const FENCE_HEIGHT: f32 = 0.6;

fn firered_path() -> String {
    std::env::var("FIRERED_PATH").expect("FIRERED_PATH environment variable must be set")
}

struct CollisionGrid {
    width: usize,
    height: usize,
    blocked: Vec<bool>,
}

impl CollisionGrid {
    fn is_blocked(&self, x: f32, z: f32) -> bool {
        let gx = x as i32;
        let gz = z as i32;
        if gx < 0 || gz < 0 || gx >= self.width as i32 || gz >= self.height as i32 {
            return true;
        }
        self.blocked[gz as usize * self.width + gx as usize]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Game state
// ═══════════════════════════════════════════════════════════════════════════

struct PokemonViewer {
    player_pos: Vec3,
    player_target: Vec3,
    player_facing: Vec3,
    moving: bool,
    camera: FollowCameraController,
    camera_entity: Option<hecs::Entity>,
    collision: CollisionGrid,
    map_width: usize,
    map_height: usize,
    exit: bool,
}

impl PokemonViewer {
    fn new() -> Self {
        Self {
            player_pos: Vec3::ZERO,
            player_target: Vec3::ZERO,
            player_facing: Vec3::new(0.0, 0.0, -1.0),
            moving: false,
            camera: FollowCameraController::new(Vec3::new(0.5, 2.8, 3.5)),
            camera_entity: None,
            collision: CollisionGrid { width: 0, height: 0, blocked: Vec::new() },
            map_width: 0,
            map_height: 0,
            exit: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Game trait implementation
// ═══════════════════════════════════════════════════════════════════════════

impl Game for PokemonViewer {
    fn init(&mut self, ctx: &mut Ctx) {
        use esox_engine::esox_input::KeyCode;

        ctx.input.bind_axis("move_x", AxisBinding::Keys {
            negative: KeyCode::KeyA, positive: KeyCode::KeyD,
        });
        ctx.input.bind_axis("move_z", AxisBinding::Keys {
            negative: KeyCode::KeyS, positive: KeyCode::KeyW,
        });
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));

        let fr_path = firered_path();
        let base = Path::new(&fr_path);

        // ── Load map data ──
        let map_bin = std::fs::read(base.join("data/layouts/PalletTown/map.bin"))
            .expect("failed to read map.bin");
        let primary_attrs = std::fs::read(
            base.join("data/tilesets/primary/general/metatile_attributes.bin"),
        ).expect("failed to read primary metatile_attributes.bin");
        let secondary_attrs = std::fs::read(
            base.join("data/tilesets/secondary/pallet_town/metatile_attributes.bin"),
        ).expect("failed to read secondary metatile_attributes.bin");

        let map_width: usize = 24;
        let map_height: usize = 20;
        self.map_width = map_width;
        self.map_height = map_height;

        let map = parse_map_bin(&map_bin, map_width, map_height);
        let primary_beh = parse_metatile_behaviors(&primary_attrs);
        let secondary_beh = parse_metatile_behaviors(&secondary_attrs);

        // ── Load tileset data ──
        let primary_img = image::open(base.join("data/tilesets/primary/general/tiles.png"))
            .expect("failed to load primary tiles.png").to_rgba8();
        let secondary_img = image::open(base.join("data/tilesets/secondary/pallet_town/tiles.png"))
            .expect("failed to load secondary tiles.png").to_rgba8();

        let primary_pals = load_palettes(&base.join("data/tilesets/primary/general/palettes"));
        let secondary_pals = load_palettes(&base.join("data/tilesets/secondary/pallet_town/palettes"));
        let combined_pal = build_combined_palette(&primary_pals, &secondary_pals);

        let primary_reverse = build_reverse_palette(&primary_pals[0]);
        let secondary_reverse = build_reverse_palette(&secondary_pals[0]);

        let primary_meta_data = std::fs::read(
            base.join("data/tilesets/primary/general/metatiles.bin"),
        ).expect("failed to read primary metatiles.bin");
        let secondary_meta_data = std::fs::read(
            base.join("data/tilesets/secondary/pallet_town/metatiles.bin"),
        ).expect("failed to read secondary metatiles.bin");

        let primary_meta_defs = parse_metatile_defs(&primary_meta_data);
        let secondary_meta_defs = parse_metatile_defs(&secondary_meta_data);

        // ── Classify tiles and collect unique metatile IDs ──
        let mut kinds = vec![vec![TileKind::Grass; map_width]; map_height];
        let mut collision_blocked = vec![false; map_width * map_height];
        let mut unique_metatile_ids = HashSet::new();

        for y in 0..map_height {
            for x in 0..map_width {
                let tile = map.tile(x, y);
                kinds[y][x] = classify_tile(tile, &primary_beh, &secondary_beh);
                collision_blocked[y * map_width + x] = tile.collision > 0;
                unique_metatile_ids.insert(tile.metatile_id);
            }
        }

        self.collision = CollisionGrid {
            width: map_width, height: map_height, blocked: collision_blocked,
        };

        // ── Composite metatile textures and create materials ──
        let mut metatile_textures: HashMap<u16, Vec<u8>> = HashMap::new();
        let mut metatile_materials: HashMap<u16, MaterialHandle> = HashMap::new();

        for &mid in &unique_metatile_ids {
            let defs = if mid < NUM_METATILES_IN_PRIMARY {
                &primary_meta_defs[mid as usize]
            } else {
                let idx = (mid - NUM_METATILES_IN_PRIMARY) as usize;
                if idx >= secondary_meta_defs.len() { continue; }
                &secondary_meta_defs[idx]
            };

            let pixels_16 = composite_metatile(
                &primary_img, &secondary_img,
                &primary_reverse, &secondary_reverse,
                &combined_pal, defs,
            );
            let pixels_64 = scale_up_4x(&pixels_16);

            let tex = ctx.renderer.upload_texture(ctx.gpu, 64, 64, &pixels_64);
            let tex_handle = match tex {
                Some(t) => t,
                None => continue,
            };

            let mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [1.0, 1.0, 1.0, 1.0],
                texture: Some(tex_handle),
                roughness: 0.85,
                metallic: 0.0,
                ..MaterialDescriptor::default()
            });

            metatile_textures.insert(mid, pixels_16);
            metatile_materials.insert(mid, mat);
        }

        eprintln!("[pokemon] composited {} unique metatile textures", metatile_materials.len());

        // ── Shared meshes ──
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(1.0));
        let plane_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::plane(1.0, 1.0, 1));
        let sphere_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::sphere(1.0, 12, 8));
        let cylinder_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cylinder(1.0, 1.0, 8));
        let grass_tuft_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cone(0.12, 0.4, 5));

        // ── Fallback materials (for tiles without textures) ──
        let mat_fallback = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.3, 0.5, 0.2, 1.0],
            roughness: 0.9,
            ..MaterialDescriptor::default()
        });
        let mat_water = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.20, 0.40, 0.70, 0.7],
            roughness: 0.15,
            metallic: 0.1,
            blend_mode: BlendMode3D::AlphaBlend,
            ..MaterialDescriptor::default()
        });
        let mat_tree_trunk = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.40, 0.28, 0.15, 1.0],
            roughness: 0.9,
            ..MaterialDescriptor::default()
        });
        let mat_tree_canopy = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.15, 0.38, 0.12, 1.0],
            roughness: 0.85,
            ..MaterialDescriptor::default()
        });
        let mat_grass_tuft = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.22, 0.48, 0.14, 1.0],
            roughness: 0.85,
            double_sided: true,
            ..MaterialDescriptor::default()
        });
        let mat_player = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.9, 0.3, 0.3, 1.0],
            roughness: 0.4,
            metallic: 0.2,
            ..MaterialDescriptor::default()
        });

        // ── Base ground plane ──
        let base_ground = MeshData::plane(
            map_width as f32 * TILE_SIZE,
            map_height as f32 * TILE_SIZE, 1,
        );
        let base_ground_mesh = ctx.renderer.upload_mesh(ctx.gpu, &base_ground);
        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(
                    map_width as f32 * 0.5 * TILE_SIZE,
                    -0.01,
                    map_height as f32 * 0.5 * TILE_SIZE,
                ),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: base_ground_mesh,
                material: mat_fallback,
                tint: [1.0; 4],
                visible: true,
            },
        ));

        // ── Find buildings ──
        let mut tile_consumed = vec![vec![false; map_width]; map_height];
        let buildings = find_buildings(map_width, map_height, &kinds);
        eprintln!("[pokemon] found {} building groups", buildings.len());

        for building in &buildings {
            if !building.has_door || building.tiles.len() < 8 { continue; }

            for &(tx, ty) in &building.tiles {
                tile_consumed[ty][tx] = true;
            }

            let bw = (building.max_x - building.min_x + 1) as f32 * TILE_SIZE;
            let bd = (building.max_y - building.min_y + 1) as f32 * TILE_SIZE;
            let cx = (building.min_x as f32 + building.max_x as f32 + 1.0) * 0.5 * TILE_SIZE;
            let cz = (building.min_y as f32 + building.max_y as f32 + 1.0) * 0.5 * TILE_SIZE;

            // Pick representative wall and roof metatile textures.
            let mut wall_ids: HashMap<u16, usize> = HashMap::new();
            let mut roof_ids: HashMap<u16, usize> = HashMap::new();
            for &(tx, ty) in &building.tiles {
                let mid = map.tile(tx, ty).metatile_id;
                match kinds[ty][tx] {
                    TileKind::BuildingWall => *wall_ids.entry(mid).or_insert(0) += 1,
                    TileKind::BuildingRoof => *roof_ids.entry(mid).or_insert(0) += 1,
                    _ => {}
                }
            }
            let best_wall_mid = wall_ids.iter().max_by_key(|e| e.1).map(|e| *e.0);
            let best_roof_mid = roof_ids.iter().max_by_key(|e| e.1).map(|e| *e.0);

            // Wall material: use metatile texture or compute average color.
            let wall_mat = best_wall_mid
                .and_then(|mid| metatile_materials.get(&mid).copied())
                .unwrap_or(mat_fallback);

            let roof_mat = best_roof_mid
                .and_then(|mid| metatile_materials.get(&mid).copied())
                .unwrap_or(mat_fallback);

            // Spawn wall box.
            let wall_data = make_building_box(bw, WALL_HEIGHT, bd);
            let wall_mesh = ctx.renderer.upload_mesh(ctx.gpu, &wall_data);
            ctx.world.spawn((
                Transform3D { position: Vec3::new(cx, 0.0, cz), ..Transform3D::default() },
                GlobalTransform::default(),
                MeshRenderer { mesh: wall_mesh, material: wall_mat, tint: [1.0; 4], visible: true },
            ));

            // Spawn peaked roof.
            let roof_data = make_peaked_roof(bw + 0.3, ROOF_PEAK, bd + 0.3);
            let roof_mesh = ctx.renderer.upload_mesh(ctx.gpu, &roof_data);
            ctx.world.spawn((
                Transform3D { position: Vec3::new(cx, WALL_HEIGHT, cz), ..Transform3D::default() },
                GlobalTransform::default(),
                MeshRenderer { mesh: roof_mesh, material: roof_mat, tint: [1.0; 4], visible: true },
            ));

            // Door panels.
            for &(tx, ty) in &building.tiles {
                if kinds[ty][tx] != TileKind::Door { continue; }
                let dx = (tx as f32 + 0.5) * TILE_SIZE;
                let is_south = ty == building.max_y;
                let dz_off = if is_south { bd * 0.5 + 0.08 } else { -(bd * 0.5 + 0.08) };
                let door_mid = map.tile(tx, ty).metatile_id;
                let door_mat = metatile_materials.get(&door_mid).copied().unwrap_or(mat_fallback);
                ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(dx, 0.9, cz + dz_off),
                        scale: Vec3::new(TILE_SIZE * 0.8, 1.8, 0.08),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer { mesh: cube_mesh, material: door_mat, tint: [1.0; 4], visible: true },
                ));
            }
        }

        // ── Spawn individual tiles ──
        let mut grass_positions: Vec<(f32, f32)> = Vec::new();

        for y in 0..map_height {
            for x in 0..map_width {
                let kind = kinds[y][x];
                let mid = map.tile(x, y).metatile_id;
                let wx = (x as f32 + 0.5) * TILE_SIZE;
                let wz = (y as f32 + 0.5) * TILE_SIZE;
                let mat = metatile_materials.get(&mid).copied().unwrap_or(mat_fallback);

                match kind {
                    TileKind::Grass | TileKind::Path => {
                        ctx.world.spawn((
                            Transform3D {
                                position: Vec3::new(wx, 0.0, wz),
                                scale: Vec3::new(TILE_SIZE, 1.0, TILE_SIZE),
                                ..Transform3D::default()
                            },
                            GlobalTransform::default(),
                            MeshRenderer { mesh: plane_mesh, material: mat, tint: [1.0; 4], visible: true },
                        ));
                        if kind == TileKind::Grass { grass_positions.push((wx, wz)); }
                    }

                    TileKind::Water => {
                        ctx.world.spawn((
                            Transform3D {
                                position: Vec3::new(wx, -0.1, wz),
                                scale: Vec3::new(TILE_SIZE, 1.0, TILE_SIZE),
                                ..Transform3D::default()
                            },
                            GlobalTransform::default(),
                            MeshRenderer { mesh: plane_mesh, material: mat_water, tint: [1.0; 4], visible: true },
                        ));
                    }

                    TileKind::Tree => {
                        // Trunk.
                        ctx.world.spawn((
                            Transform3D {
                                position: Vec3::new(wx, 0.7, wz),
                                scale: Vec3::new(0.12, 1.4, 0.12),
                                ..Transform3D::default()
                            },
                            GlobalTransform::default(),
                            MeshRenderer { mesh: cylinder_mesh, material: mat_tree_trunk, tint: [1.0; 4], visible: true },
                        ));
                        // Canopy.
                        ctx.world.spawn((
                            Transform3D {
                                position: Vec3::new(wx, 2.2, wz),
                                scale: Vec3::new(0.7, 0.8, 0.7),
                                ..Transform3D::default()
                            },
                            GlobalTransform::default(),
                            MeshRenderer { mesh: sphere_mesh, material: mat_tree_canopy, tint: [1.0; 4], visible: true },
                        ));
                    }

                    TileKind::Fence => {
                        ctx.world.spawn((
                            Transform3D {
                                position: Vec3::new(wx, FENCE_HEIGHT * 0.5, wz),
                                scale: Vec3::new(TILE_SIZE, FENCE_HEIGHT, 0.12),
                                ..Transform3D::default()
                            },
                            GlobalTransform::default(),
                            MeshRenderer { mesh: cube_mesh, material: mat, tint: [1.0; 4], visible: true },
                        ));
                    }

                    TileKind::BuildingWall | TileKind::BuildingRoof => {
                        if !tile_consumed[y][x] {
                            let h = if kind == TileKind::BuildingRoof { 1.5 } else { 1.2 };
                            ctx.world.spawn((
                                Transform3D {
                                    position: Vec3::new(wx, h * 0.5, wz),
                                    scale: Vec3::new(TILE_SIZE, h, TILE_SIZE),
                                    ..Transform3D::default()
                                },
                                GlobalTransform::default(),
                                MeshRenderer { mesh: cube_mesh, material: mat, tint: [1.0; 4], visible: true },
                            ));
                        }
                    }

                    TileKind::Door => {
                        if !tile_consumed[y][x] {
                            ctx.world.spawn((
                                Transform3D {
                                    position: Vec3::new(wx, 0.0, wz),
                                    scale: Vec3::new(TILE_SIZE, 1.0, TILE_SIZE),
                                    ..Transform3D::default()
                                },
                                GlobalTransform::default(),
                                MeshRenderer { mesh: plane_mesh, material: mat, tint: [1.0; 4], visible: true },
                            ));
                        }
                    }
                }
            }
        }

        // ── Grass tufts ──
        for (i, &(gx, gz)) in grass_positions.iter().enumerate() {
            for j in 0..2usize {
                let seed = (i * 2 + j) as u32;
                let h1 = seed.wrapping_mul(2654435761) >> 16;
                let h2 = seed.wrapping_mul(1013904223).wrapping_add(1664525) >> 16;
                let ox = ((h1 % 100) as f32 / 100.0 - 0.5) * TILE_SIZE * 0.8;
                let oz = ((h2 % 100) as f32 / 100.0 - 0.5) * TILE_SIZE * 0.8;
                let s = 0.7 + ((h1 % 60) as f32 / 100.0);
                ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(gx + ox, 0.2 * s, gz + oz),
                        scale: Vec3::splat(s),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer { mesh: grass_tuft_mesh, material: mat_grass_tuft, tint: [1.0; 4], visible: true },
                ));
            }
        }

        // ── Player ──
        let sx = 12.5 * TILE_SIZE;
        let sz = 12.5 * TILE_SIZE;
        self.player_pos = Vec3::new(sx, 0.0, sz);
        self.player_target = self.player_pos;

        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(sx, 0.5, sz),
                scale: Vec3::new(0.25, 0.6, 0.25),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder_mesh, material: mat_player, tint: [1.0; 4], visible: true },
            Tag("player_body".into()),
        ));
        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(sx, 1.1, sz),
                scale: Vec3::splat(0.22),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere_mesh, material: mat_player, tint: [1.0; 4], visible: true },
            Tag("player_head".into()),
        ));

        // ── Camera ──
        self.camera.smoothing = 6.0;
        self.camera.set_target(self.player_pos + Vec3::Y);
        self.camera.position = self.player_pos + Vec3::Y + self.camera.offset;
        self.camera.prev_position = self.camera.position;

        let (cp, cr) = self.camera.apply(1.0);
        self.camera_entity = Some(ctx.world.spawn((
            Transform3D { position: cp, rotation: cr, ..Transform3D::default() },
            GlobalTransform::default(),
            Camera3D { active: true, fov_y: 55.0_f32.to_radians(), near: 0.1, far: 200.0 },
        )));

        // ── Sun ──
        ctx.world.spawn((
            Transform3D {
                rotation: Quat::from_rotation_x(-1.2) * Quat::from_rotation_y(0.5),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent { color: [1.0, 0.96, 0.88], intensity: 3.5 },
        ));

        // ── Rendering ──
        ctx.renderer.generate_procedural_ibl(ctx.gpu);
        ctx.renderer.set_shadow_config(ShadowConfig {
            shadow_distance: 40.0,
            depth_bias: 0.003,
            normal_bias: 0.03,
            ..ShadowConfig::default()
        });
        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.02,
            bloom_threshold: 4.0,
            bloom_soft_knee: 0.1,
            tone_map_enabled: true,
            ssao_enabled: false,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        });

        eprintln!("[pokemon] Pallet Town loaded — WASD to move, Esc to quit");
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") { self.exit = true; return; }
        let dt = ctx.time.tick_dt;

        let mx = ctx.input.axis("move_x");
        let mz = ctx.input.axis("move_z");

        if !self.moving {
            let mut dir = Vec3::ZERO;
            if mz.abs() > 0.1 {
                dir = Vec3::new(0.0, 0.0, -mz.signum());
            } else if mx.abs() > 0.1 {
                dir = Vec3::new(mx.signum(), 0.0, 0.0);
            }
            if dir != Vec3::ZERO {
                self.player_facing = dir;
                let target = self.player_pos + dir * TILE_SIZE;
                if !self.collision.is_blocked(target.x, target.z) {
                    self.player_target = target;
                    self.moving = true;
                }
            }
        }

        if self.moving {
            let diff = self.player_target - self.player_pos;
            if diff.length() < 0.05 {
                self.player_pos = self.player_target;
                self.moving = false;
            } else {
                self.player_pos += diff.normalize() * MOVE_SPEED * dt;
                let new_diff = self.player_target - self.player_pos;
                if new_diff.dot(diff) < 0.0 {
                    self.player_pos = self.player_target;
                    self.moving = false;
                }
            }
        }

        self.player_pos.x = self.player_pos.x.clamp(0.5, (self.map_width as f32 - 0.5) * TILE_SIZE);
        self.player_pos.z = self.player_pos.z.clamp(0.5, (self.map_height as f32 - 0.5) * TILE_SIZE);

        let rot = look_at_quat_dir(self.player_facing);
        for (_id, (t, tag)) in ctx.world.query_mut::<(&mut Transform3D, &Tag)>() {
            if tag.0 == "player_body" {
                t.position = Vec3::new(self.player_pos.x, 0.5, self.player_pos.z);
                t.rotation = rot;
            } else if tag.0 == "player_head" {
                t.position = Vec3::new(self.player_pos.x, 1.1, self.player_pos.z);
            }
        }

        self.camera.set_target(self.player_pos + Vec3::Y);
        self.camera.update(&ctx.input, &ctx.time);
    }

    fn render(&mut self, ctx: &mut Ctx, alpha: f32) {
        if let Some(cam_e) = self.camera_entity {
            let (p, r) = self.camera.apply(alpha);
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(cam_e) {
                t.position = p;
                t.rotation = r;
            }
        }
    }

    fn should_exit(&self) -> bool { self.exit }
}

fn look_at_quat_dir(dir: Vec3) -> Quat {
    let fwd = Vec3::new(dir.x, 0.0, dir.z).normalize();
    if fwd.length_squared() < 1e-6 { return Quat::IDENTITY; }
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    Quat::from_mat3(&Mat3::from_cols(right, up, -fwd))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("pokemon=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "Pokemon — Pallet Town".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        postprocess: true,
        shadows: true,
        ..EngineConfig::default()
    };

    if let Err(e) = esox_engine::run(config, PokemonViewer::new()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
