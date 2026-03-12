use esox_font::face::FontFace;
use esox_font::rasterizer::GlyphRasterizer;
use esox_font::shaper::TextShaper;
use esox_font::{FontId, GlyphCache, GlyphKey};
use esox_gfx::{AtlasAllocator, AtlasId, ShelfAllocator};

fn main() {
    let data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-data/JetBrainsMono-Regular.ttf"
    ))
    .unwrap();
    let face = FontFace::from_bytes(FontId(0), data).unwrap();

    // Test at both sizes:
    // 1. visual_demo default (24px)
    // 2. actual terminal default (14pt → ~18.67px)
    for size in [24.0_f32, 14.0 * (96.0 / 72.0)] {
        let metrics = face.metrics(size);
        println!("Font: JetBrains Mono Regular @ {size:.2}px");
        println!(
            "Metrics: cell={}x{}, ascent={:.1}, descent={:.1}\n",
            metrics.cell_width, metrics.cell_height, metrics.ascent, metrics.descent
        );

        let mut shaper = TextShaper::new();
        let mut rasterizer = GlyphRasterizer::new();
        let mut cache = GlyphCache::new();
        let mut alloc = ShelfAllocator::new(AtlasId(0), 2048, 2048);

        let text = "esocidae";
        let run = shaper.shape(&face, text, size);

        println!("Shaped \"{text}\" → {} glyphs\n", run.glyphs.len());

        let shades = [' ', '░', '▒', '▓', '█'];

        for (i, glyph) in run.glyphs.iter().enumerate() {
            let key = GlyphKey {
                font_id: FontId(0),
                glyph_id: glyph.glyph_id,
                size_tenths: (size * 10.0) as u32,
                style: 0,
            };
            let cached = cache
                .get_or_insert(key, &face, &mut rasterizer, &mut alloc, size, false)
                .unwrap();

            let rast = rasterizer
                .rasterize(&face, glyph.glyph_id, size, 0)
                .unwrap();

            let ch = text.chars().nth(i).unwrap_or('?');
            println!(
                "┌─ '{}' glyph_id={:<4} {}x{:<3} bearing=({:+.0},{:+.0})  atlas@({},{} {}x{})",
                ch,
                glyph.glyph_id,
                rast.width,
                rast.height,
                cached.bearing_x,
                cached.bearing_y,
                cached.region.x,
                cached.region.y,
                cached.region.w,
                cached.region.h
            );

            if rast.width > 0 && rast.height > 0 {
                for y in 0..rast.height {
                    print!("│ ");
                    for x in 0..rast.width {
                        let idx = ((y * rast.width + x) * 4 + 3) as usize;
                        let alpha = rast.data[idx];
                        let shade = (alpha as usize * (shades.len() - 1)) / 255;
                        print!("{}", shades[shade]);
                    }
                    println!();
                }
            }
            println!();
        }

        // Check clipping potential
        println!("--- Cell clipping analysis ---");
        for ch in text.chars() {
            let font_ref = face.as_swash_ref();
            let glyph_id = u32::from(font_ref.charmap().map(ch));
            let rast = rasterizer.rasterize(&face, glyph_id, size, 0).unwrap();
            let glyph_top = metrics.ascent - rast.bearing_y;
            let glyph_bottom = glyph_top + rast.height as f32;
            let clip_top = if glyph_top < 0.0 {
                format!(" *** clips TOP by {:.1}px", -glyph_top)
            } else {
                String::new()
            };
            let clip_bot = if glyph_bottom > metrics.cell_height {
                format!(
                    " *** clips BOTTOM by {:.1}px",
                    glyph_bottom - metrics.cell_height
                )
            } else {
                String::new()
            };
            println!(
                "  '{}': glyph_top={:.1} glyph_bottom={:.1} cell=[0, {:.1}]{}{}",
                ch, glyph_top, glyph_bottom, metrics.cell_height, clip_top, clip_bot
            );
        }

        let uploads = cache.drain_uploads();
        println!(
            "\nCache: {} entries | {} uploads drained | atlas {:.2}% full\n",
            cache.len(),
            uploads.len(),
            alloc.utilization() * 100.0
        );
        println!("{}", "=".repeat(60));
    }
}
