/// Linear RGBA color (premultiplied alpha, GPU-ready).
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Color {
    /// Red channel (linear, 0.0–1.0).
    pub r: f32,
    /// Green channel (linear, 0.0–1.0).
    pub g: f32,
    /// Blue channel (linear, 0.0–1.0).
    pub b: f32,
    /// Alpha channel (0.0–1.0).
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Opaque black.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Opaque white.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Create a new color from linear RGBA components.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Convert an sRGB byte triplet (0–255) plus alpha to linear RGBA.
    pub fn from_srgb(r: u8, g: u8, b: u8, a: f32) -> Self {
        Self {
            r: srgb_to_linear(r),
            g: srgb_to_linear(g),
            b: srgb_to_linear(b),
            a,
        }
    }

    /// Parse a hex color string (`"#RRGGBB"` or `"#RGB"`) into linear RGBA.
    ///
    /// Returns `None` if the string is not a valid hex color.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix('#')?;
        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(Self::from_srgb(r, g, b, 1.0))
            }
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
                Some(Self::from_srgb(r * 17, g * 17, b * 17, 1.0))
            }
            _ => None,
        }
    }

    /// Return this color with a different alpha value.
    pub const fn with_alpha(self, a: f32) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a,
        }
    }

    /// Return this color with RGB channels premultiplied by alpha.
    ///
    /// Required for correct output when using `CompositeAlphaMode::PreMultiplied`,
    /// where the compositor expects `(R×A, G×A, B×A, A)` in the framebuffer.
    pub const fn premultiplied(self) -> Self {
        Self {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    /// Convert this linear color back to sRGB bytes (ignoring alpha).
    pub fn to_srgb(self) -> [u8; 3] {
        [
            linear_to_srgb(self.r),
            linear_to_srgb(self.g),
            linear_to_srgb(self.b),
        ]
    }
}

/// Convert a single sRGB byte to a linear float.
pub fn srgb_to_linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a single linear float to an sRGB byte.
fn linear_to_srgb(value: f32) -> u8 {
    let s = if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_correct() {
        assert_eq!(Color::BLACK, Color::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(Color::WHITE, Color::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(Color::TRANSPARENT.a, 0.0);
    }

    #[test]
    fn srgb_roundtrip() {
        // Every sRGB byte should survive a roundtrip.
        for i in 0..=255u8 {
            let linear = srgb_to_linear(i);
            let back = linear_to_srgb(linear);
            assert_eq!(i, back, "roundtrip failed for sRGB {i}");
        }
    }

    #[test]
    fn srgb_boundary_values() {
        assert_eq!(srgb_to_linear(0), 0.0);
        assert_eq!(linear_to_srgb(0.0), 0);
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-5);
        assert_eq!(linear_to_srgb(1.0), 255);
    }

    #[test]
    fn from_srgb_produces_linear() {
        let c = Color::from_srgb(255, 0, 0, 0.5);
        assert!((c.r - 1.0).abs() < 1e-5);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 0.5);
    }

    #[test]
    fn to_srgb_inverse_of_from_srgb() {
        let c = Color::from_srgb(128, 64, 200, 1.0);
        let [r, g, b] = c.to_srgb();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 200);
    }

    #[test]
    fn from_hex_rrggbb() {
        let c = Color::from_hex("#ff0000").unwrap();
        assert!((c.r - 1.0).abs() < 1e-5);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 1.0);
    }

    #[test]
    fn from_hex_short() {
        let c = Color::from_hex("#f00").unwrap();
        // #f00 expands to #ff0000
        assert!((c.r - 1.0).abs() < 1e-5);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
    }

    #[test]
    fn from_hex_black() {
        let c = Color::from_hex("#000000").unwrap();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
    }

    #[test]
    fn from_hex_white() {
        let c = Color::from_hex("#ffffff").unwrap();
        let [r, g, b] = c.to_srgb();
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn from_hex_mixed() {
        let c = Color::from_hex("#80ff00").unwrap();
        let [r, g, b] = c.to_srgb();
        assert_eq!((r, g, b), (128, 255, 0));
    }

    #[test]
    fn from_hex_invalid() {
        assert!(Color::from_hex("").is_none());
        assert!(Color::from_hex("#").is_none());
        assert!(Color::from_hex("#gg0000").is_none());
        assert!(Color::from_hex("#12345").is_none());
        assert!(Color::from_hex("#1234567").is_none());
        assert!(Color::from_hex("ff0000").is_none());
    }

    #[test]
    fn from_hex_case_insensitive() {
        let upper = Color::from_hex("#FF8800").unwrap();
        let lower = Color::from_hex("#ff8800").unwrap();
        assert_eq!(upper, lower);
    }

    #[test]
    fn premultiplied_scales_rgb_by_alpha() {
        let c = Color::new(0.8, 0.6, 0.4, 0.5).premultiplied();
        assert!((c.r - 0.4).abs() < 1e-6);
        assert!((c.g - 0.3).abs() < 1e-6);
        assert!((c.b - 0.2).abs() < 1e-6);
        assert!((c.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn premultiplied_opaque_is_identity() {
        let c = Color::new(0.5, 0.7, 0.9, 1.0);
        let p = c.premultiplied();
        assert_eq!(c, p);
    }

    #[test]
    fn pod_layout() {
        // Color must be 16 bytes (4 × f32) for GPU upload.
        assert_eq!(size_of::<Color>(), 16);
    }
}
