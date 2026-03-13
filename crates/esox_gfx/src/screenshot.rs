//! Screenshot capture via GPU texture readback.
//!
//! Copies the rendered surface texture to a staging buffer, then reads
//! the pixels back to CPU memory and saves them as a PNG file.

use std::path::PathBuf;

/// Bytes-per-row alignment required by wgpu for buffer-to-texture copies.
const COPY_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Holds the staging buffer and metadata needed for a screenshot capture.
pub struct ScreenshotCapture {
    /// GPU staging buffer with MAP_READ | COPY_DST usage.
    pub buffer: wgpu::Buffer,
    /// Surface width in pixels.
    pub width: u32,
    /// Surface height in pixels.
    pub height: u32,
    /// Bytes per pixel for the surface format.
    pub bytes_per_pixel: u32,
    /// Padded bytes per row (aligned to COPY_BYTES_PER_ROW_ALIGNMENT).
    pub padded_bytes_per_row: u32,
    /// The surface texture format (needed for pixel conversion).
    pub format: wgpu::TextureFormat,
}

impl ScreenshotCapture {
    /// Create a new screenshot capture buffer sized for the given surface.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let bytes_per_pixel = bytes_per_pixel(format);
        let unpadded = width * bytes_per_pixel;
        let padded_bytes_per_row = (unpadded + COPY_ALIGN - 1) / COPY_ALIGN * COPY_ALIGN;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            width,
            height,
            bytes_per_pixel,
            padded_bytes_per_row,
            format,
        }
    }

    /// Encode a copy from the surface texture into the staging buffer.
    ///
    /// Call this on the command encoder *after* all render passes but
    /// *before* `encoder.finish()`.
    pub fn encode_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: surface_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Map the staging buffer, read the pixels, and save as PNG.
    ///
    /// This blocks the current thread until the GPU finishes and the
    /// buffer is mapped.  Call from a background thread to avoid stalling
    /// the render loop.
    pub fn save_blocking(self, device: &wgpu::Device, path: PathBuf) {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::error!("screenshot buffer map failed: {e}");
                return;
            }
            Err(_) => {
                tracing::error!("screenshot map channel closed");
                return;
            }
        }

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);

        for row in 0..self.height {
            let offset = (row * self.padded_bytes_per_row) as usize;
            let row_bytes = &data[offset..offset + (self.width * self.bytes_per_pixel) as usize];
            match self.format {
                // BGRA → RGBA swap
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                    for chunk in row_bytes.chunks_exact(4) {
                        pixels.push(chunk[2]); // R
                        pixels.push(chunk[1]); // G
                        pixels.push(chunk[0]); // B
                        pixels.push(chunk[3]); // A
                    }
                }
                // RGBA formats — copy directly
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                    pixels.extend_from_slice(row_bytes);
                }
                // HDR Rgba16Float — tonemap to 8-bit
                wgpu::TextureFormat::Rgba16Float => {
                    for chunk in row_bytes.chunks_exact(8) {
                        let r = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                        let g = half_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
                        let b = half_to_f32(u16::from_le_bytes([chunk[4], chunk[5]]));
                        let a = half_to_f32(u16::from_le_bytes([chunk[6], chunk[7]]));
                        pixels.push((r.clamp(0.0, 1.0) * 255.0) as u8);
                        pixels.push((g.clamp(0.0, 1.0) * 255.0) as u8);
                        pixels.push((b.clamp(0.0, 1.0) * 255.0) as u8);
                        pixels.push((a.clamp(0.0, 1.0) * 255.0) as u8);
                    }
                }
                other => {
                    tracing::error!(?other, "unsupported surface format for screenshot");
                    return;
                }
            }
        }

        drop(data);
        self.buffer.unmap();

        let img = image::RgbaImage::from_raw(self.width, self.height, pixels);
        match img {
            Some(img) => match img.save(&path) {
                Ok(()) => tracing::info!(?path, "screenshot saved"),
                Err(e) => tracing::error!(?path, "screenshot save failed: {e}"),
            },
            None => tracing::error!("failed to create image from pixel data"),
        }
    }
}

/// Bytes per pixel for common surface formats.
fn bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::Bgra8Unorm
        | wgpu::TextureFormat::Bgra8UnormSrgb
        | wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        _ => 4, // fallback
    }
}

/// Convert an IEEE 754 half-precision float to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 1i32;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f_exp = (127 - 15 + e) as u32;
        return f32::from_bits((sign << 31) | (f_exp << 23) | (m << 13));
    }
    if exp == 31 {
        let f_mant = if mant != 0 { 1u32 << 22 } else { 0 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | f_mant);
    }

    let f_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f_exp << 23) | (mant << 13))
}
