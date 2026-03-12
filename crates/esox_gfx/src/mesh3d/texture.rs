//! GPU-resident textures for 3D materials.

/// Handle to a GPU-resident texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub(crate) u32);

/// Maximum texture dimension (width or height).
const MAX_DIMENSION: u32 = 8192;

/// GPU-resident texture (RGBA8).
pub(crate) struct Texture3D {
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}

impl Texture3D {
    /// Upload RGBA8 data to a new GPU texture (sRGB format — for color textures).
    ///
    /// Returns `None` if `data.len() != width * height * 4`.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Option<Self> {
        Self::upload_inner(device, queue, width, height, data, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    /// Upload RGBA8 data to a new GPU texture (linear format — for data textures like
    /// normal maps, metallic-roughness maps).
    ///
    /// Returns `None` if `data.len() != width * height * 4`.
    pub fn upload_linear(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Option<Self> {
        Self::upload_inner(device, queue, width, height, data, wgpu::TextureFormat::Rgba8Unorm)
    }

    fn upload_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        data: &[u8],
        format: wgpu::TextureFormat,
    ) -> Option<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return None;
        }

        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            tracing::warn!(
                "texture dimensions {}x{} exceed max {MAX_DIMENSION}, rejecting",
                width,
                height,
            );
            return None;
        }
        let w = width;
        let h = height;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Some(Self {
            texture,
            view,
            width: w,
            height: h,
        })
    }

    /// Create a 1x1 white fallback texture (sRGB).
    pub fn fallback_white(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::upload(device, queue, 1, 1, &[255, 255, 255, 255])
            .expect("fallback texture upload should not fail")
    }

    /// Create a 1x1 flat normal fallback texture (linear).
    /// Encodes normal (0, 0, 1) as (128, 128, 255, 255).
    pub fn fallback_normal(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::upload_linear(device, queue, 1, 1, &[128, 128, 255, 255])
            .expect("fallback normal texture upload should not fail")
    }

    /// Create a 1x1 default metallic-roughness fallback texture (linear).
    /// glTF channel packing: G=roughness, B=metallic.
    /// Default: metallic=0 (B=0), roughness=0.5 (G=128).
    pub fn fallback_metallic_roughness(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::upload_linear(device, queue, 1, 1, &[0, 128, 0, 255])
            .expect("fallback MR texture upload should not fail")
    }

    /// Upload RGBA8 data decoded from an image file (PNG/JPEG).
    ///
    /// If `srgb` is true, uses sRGB format (for albedo/emissive).
    /// If false, uses linear format (for normal/metallic-roughness maps).
    #[cfg(feature = "mesh3d")]
    pub fn upload_from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        srgb: bool,
    ) -> Option<Self> {
        let img = image::load_from_memory(data).ok()?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        if srgb {
            Self::upload(device, queue, w, h, &rgba)
        } else {
            Self::upload_linear(device, queue, w, h, &rgba)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_handle_equality() {
        let a = TextureHandle(0);
        let b = TextureHandle(0);
        let c = TextureHandle(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
