# vpk_viewer

View a Deadlock model straight from a VPK (or a plain `.glb`), with hot reload.
It wraps [`vpkmerge`](#building)'s `.vmdl_c` → GLB exporter and the esox GLB
renderer into one window: edit the source on disk and the view re-exports and
reloads automatically.

## Building

> **Note:** this example depends on `vpkmerge-core` via a **machine-local path**
> dependency (`../../../grimoire-workspace/vpkmerge/vpkmerge-core`). It is a
> personal dev tool, not a portable artifact. A fresh clone of this repo will
> **not** build `vpk_viewer` (and a full `cargo build` will fail on it) unless
> that sibling workspace is present at the expected path. If you just want to
> run it, grab the prebuilt binary from the GitHub Releases page instead.

```bash
cargo run --release -p vpk_viewer -- <args>
```

## Usage

```bash
# View a GLB directly
vpk_viewer model.glb

# Auto-discover a hero by codename inside a skin VPK
vpk_viewer --vpk hero_skin.vpk --hero hornet --base pak01_dir.vpk

# Point at an explicit .vmdl_c entry inside a VPK
vpk_viewer --vpk pak01_dir.vpk --entry models/heroes/hornet/hornet.vmdl_c
```

| Flag | Description |
| --- | --- |
| `<glb>` | A `.glb` to view directly (alternative to `--vpk`). |
| `--vpk <PATH>` | Skin or base VPK containing the model. |
| `--hero <NAME>` | Hero codename to auto-discover (e.g. `hornet`). |
| `--entry <PATH>` | Explicit `.vmdl_c` entry path inside the VPK. |
| `--base <PATH>` | Base `pak01_dir.vpk` for materials/textures the skin doesn't ship. |
| `--clip <NAME>` | Keep only these clips (repeatable). Trimming makes re-exports much faster. |
| `--no-anim` | Strip all animation clips (static mesh + skeleton). |
| `--pose [CLIP[@FRAME]]` | Bake a static single-frame pose: bare for menu/idle, or `CLIP@FRAME`. |
| `--no-watch` | Disable the file watcher (load once, like `glb_viewer`). |
