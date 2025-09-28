//! LAB color space support for zpix
//! Provides conversion between RGB, XYZ, and LAB color spaces
//! LAB is a perceptually uniform color space ideal for color correction

const std = @import("std");
const math = std.math;

/// LAB color values
pub const LabColor = struct {
    l: f32, // Lightness: 0-100
    a: f32, // Green-Red axis: -128 to +127
    b: f32, // Blue-Yellow axis: -128 to +127

    pub fn init(l: f32, a: f32, b: f32) LabColor {
        return LabColor{ .l = l, .a = a, .b = b };
    }
};

/// XYZ color values (intermediate for RGB<->LAB conversion)
pub const XyzColor = struct {
    x: f32, // X component: 0-95.047
    y: f32, // Y component: 0-100.000
    z: f32, // Z component: 0-108.883

    pub fn init(x: f32, y: f32, z: f32) XyzColor {
        return XyzColor{ .x = x, .y = y, .z = z };
    }
};

/// Standard illuminants for color space conversions
pub const Illuminant = enum {
    d65, // Daylight 6500K (sRGB standard)
    d50, // Daylight 5000K
    a,   // Incandescent 2856K
    c,   // Average daylight 6774K

    pub fn getWhitePoint(self: Illuminant) XyzColor {
        return switch (self) {
            .d65 => XyzColor.init(95.047, 100.000, 108.883),
            .d50 => XyzColor.init(96.422, 100.000, 82.521),
            .a => XyzColor.init(109.850, 100.000, 35.585),
            .c => XyzColor.init(98.074, 100.000, 118.232),
        };
    }
};

/// RGB working spaces
pub const RgbWorkingSpace = enum {
    srgb,
    adobe_rgb,
    pro_photo_rgb,
    rec2020,

    pub fn getGamma(self: RgbWorkingSpace) f32 {
        return switch (self) {
            .srgb => 2.4,
            .adobe_rgb => 2.2,
            .pro_photo_rgb => 1.8,
            .rec2020 => 2.4,
        };
    }

    pub fn getRgbToXyzMatrix(self: RgbWorkingSpace) [9]f32 {
        return switch (self) {
            .srgb => [9]f32{
                0.4124564, 0.3575761, 0.1804375,
                0.2126729, 0.7151522, 0.0721750,
                0.0193339, 0.1191920, 0.9503041,
            },
            .adobe_rgb => [9]f32{
                0.5767309, 0.1855540, 0.1881852,
                0.2973769, 0.6273491, 0.0752741,
                0.0270343, 0.0706872, 0.9911085,
            },
            .pro_photo_rgb => [9]f32{
                0.7976749, 0.1351917, 0.0313534,
                0.2880402, 0.7118741, 0.0000857,
                0.0000000, 0.0000000, 0.8252100,
            },
            .rec2020 => [9]f32{
                0.6369580, 0.1446169, 0.1688809,
                0.2627045, 0.6780980, 0.0593017,
                0.0000000, 0.0280727, 1.0609851,
            },
        };
    }

    pub fn getXyzToRgbMatrix(self: RgbWorkingSpace) [9]f32 {
        return switch (self) {
            .srgb => [9]f32{
                3.2404542, -1.5371385, -0.4985314,
                -0.9692660, 1.8760108, 0.0415560,
                0.0556434, -0.2040259, 1.0572252,
            },
            .adobe_rgb => [9]f32{
                2.0413690, -0.5649464, -0.3446944,
                -0.9692660, 1.8760108, 0.0415560,
                0.0134474, -0.1183897, 1.0154096,
            },
            .pro_photo_rgb => [9]f32{
                1.3459433, -0.2556075, -0.0511118,
                -0.5445989, 1.5081673, 0.0205351,
                0.0000000, 0.0000000, 1.2118128,
            },
            .rec2020 => [9]f32{
                1.7166511, -0.3556708, -0.2533663,
                -0.6666844, 1.6164812, 0.0157685,
                0.0176399, -0.0427706, 0.9421031,
            },
        };
    }
};

/// Convert RGB to LAB color space
pub fn rgbToLab(r: f32, g: f32, b: f32, working_space: RgbWorkingSpace, illuminant: Illuminant) LabColor {
    const xyz = rgbToXyz(r, g, b, working_space);
    return xyzToLab(xyz, illuminant);
}

/// Convert LAB to RGB color space
pub fn labToRgb(lab: LabColor, working_space: RgbWorkingSpace, illuminant: Illuminant) [3]f32 {
    const xyz = labToXyz(lab, illuminant);
    return xyzToRgb(xyz, working_space);
}

/// Convert RGB to XYZ color space
pub fn rgbToXyz(r: f32, g: f32, b: f32, working_space: RgbWorkingSpace) XyzColor {
    // Apply gamma correction (linearize)
    const linear_r = gammaExpand(r, working_space);
    const linear_g = gammaExpand(g, working_space);
    const linear_b = gammaExpand(b, working_space);

    // Apply RGB to XYZ transformation matrix
    const matrix = working_space.getRgbToXyzMatrix();

    const x = matrix[0] * linear_r + matrix[1] * linear_g + matrix[2] * linear_b;
    const y = matrix[3] * linear_r + matrix[4] * linear_g + matrix[5] * linear_b;
    const z = matrix[6] * linear_r + matrix[7] * linear_g + matrix[8] * linear_b;

    return XyzColor.init(x * 100.0, y * 100.0, z * 100.0);
}

/// Convert XYZ to RGB color space
pub fn xyzToRgb(xyz: XyzColor, working_space: RgbWorkingSpace) [3]f32 {
    // Normalize XYZ values
    const x = xyz.x / 100.0;
    const y = xyz.y / 100.0;
    const z = xyz.z / 100.0;

    // Apply XYZ to RGB transformation matrix
    const matrix = working_space.getXyzToRgbMatrix();

    const linear_r = matrix[0] * x + matrix[1] * y + matrix[2] * z;
    const linear_g = matrix[3] * x + matrix[4] * y + matrix[5] * z;
    const linear_b = matrix[6] * x + matrix[7] * y + matrix[8] * z;

    // Apply gamma compression
    const r = gammaCompress(linear_r, working_space);
    const g = gammaCompress(linear_g, working_space);
    const b = gammaCompress(linear_b, working_space);

    return [3]f32{
        @min(1.0, @max(0.0, r)),
        @min(1.0, @max(0.0, g)),
        @min(1.0, @max(0.0, b))
    };
}

/// Convert XYZ to LAB color space
pub fn xyzToLab(xyz: XyzColor, illuminant: Illuminant) LabColor {
    const white_point = illuminant.getWhitePoint();

    // Normalize by white point
    const xn = xyz.x / white_point.x;
    const yn = xyz.y / white_point.y;
    const zn = xyz.z / white_point.z;

    // Apply LAB transformation
    const fx = labF(xn);
    const fy = labF(yn);
    const fz = labF(zn);

    const l = 116.0 * fy - 16.0;
    const a = 500.0 * (fx - fy);
    const b = 200.0 * (fy - fz);

    return LabColor.init(l, a, b);
}

/// Convert LAB to XYZ color space
pub fn labToXyz(lab: LabColor, illuminant: Illuminant) XyzColor {
    const white_point = illuminant.getWhitePoint();

    const fy = (lab.l + 16.0) / 116.0;
    const fx = lab.a / 500.0 + fy;
    const fz = fy - lab.b / 200.0;

    const xn = labFInverse(fx);
    const yn = labFInverse(fy);
    const zn = labFInverse(fz);

    const x = xn * white_point.x;
    const y = yn * white_point.y;
    const z = zn * white_point.z;

    return XyzColor.init(x, y, z);
}

/// LAB transformation function
fn labF(t: f32) f32 {
    const delta = 6.0 / 29.0;
    const delta_cubed = delta * delta * delta;

    if (t > delta_cubed) {
        return math.pow(f32, t, 1.0 / 3.0);
    } else {
        return (t / (3.0 * delta * delta)) + (4.0 / 29.0);
    }
}

/// Inverse LAB transformation function
fn labFInverse(t: f32) f32 {
    const delta = 6.0 / 29.0;

    if (t > delta) {
        return t * t * t;
    } else {
        return 3.0 * delta * delta * (t - 4.0 / 29.0);
    }
}

/// Gamma expansion (linearization) for different working spaces
fn gammaExpand(value: f32, working_space: RgbWorkingSpace) f32 {
    const clamped = @min(1.0, @max(0.0, value));

    return switch (working_space) {
        .srgb => {
            if (clamped <= 0.04045) {
                return clamped / 12.92;
            } else {
                return math.pow(f32, (clamped + 0.055) / 1.055, 2.4);
            }
        },
        .adobe_rgb, .pro_photo_rgb, .rec2020 => {
            const gamma = working_space.getGamma();
            return math.pow(f32, clamped, gamma);
        },
    };
}

/// Gamma compression for different working spaces
fn gammaCompress(value: f32, working_space: RgbWorkingSpace) f32 {
    const clamped = @min(1.0, @max(0.0, value));

    return switch (working_space) {
        .srgb => {
            if (clamped <= 0.0031308) {
                return clamped * 12.92;
            } else {
                return 1.055 * math.pow(f32, clamped, 1.0 / 2.4) - 0.055;
            }
        },
        .adobe_rgb, .pro_photo_rgb, .rec2020 => {
            const gamma = working_space.getGamma();
            return math.pow(f32, clamped, 1.0 / gamma);
        },
    };
}

/// Calculate color difference using Delta E CIE76
pub fn deltaE76(lab1: LabColor, lab2: LabColor) f32 {
    const dl = lab1.l - lab2.l;
    const da = lab1.a - lab2.a;
    const db = lab1.b - lab2.b;

    return math.sqrt(dl * dl + da * da + db * db);
}

/// Calculate color difference using Delta E CIE94
pub fn deltaE94(lab1: LabColor, lab2: LabColor) f32 {
    const dl = lab1.l - lab2.l;
    const da = lab1.a - lab2.a;
    const db = lab1.b - lab2.b;

    const c1 = math.sqrt(lab1.a * lab1.a + lab1.b * lab1.b);
    const c2 = math.sqrt(lab2.a * lab2.a + lab2.b * lab2.b);
    const dc = c1 - c2;

    const dh_squared = da * da + db * db - dc * dc;
    const dh = if (dh_squared > 0) math.sqrt(dh_squared) else 0.0;

    const kl = 1.0;
    const kc = 1.0;
    const kh = 1.0;

    const sl = 1.0;
    const sc = 1.0 + 0.045 * c1;
    const sh = 1.0 + 0.015 * c1;

    const dl_term = dl / (kl * sl);
    const dc_term = dc / (kc * sc);
    const dh_term = dh / (kh * sh);

    return math.sqrt(dl_term * dl_term + dc_term * dc_term + dh_term * dh_term);
}

/// Convert image from RGB to LAB color space
pub fn convertImageToLab(allocator: std.mem.Allocator, rgb_data: []const u8, width: u32, height: u32,
                        working_space: RgbWorkingSpace, illuminant: Illuminant) ![]f32 {
    const lab_data = try allocator.alloc(f32, width * height * 3);

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = (y * width + x) * 3;

            const r = @as(f32, @floatFromInt(rgb_data[idx])) / 255.0;
            const g = @as(f32, @floatFromInt(rgb_data[idx + 1])) / 255.0;
            const b = @as(f32, @floatFromInt(rgb_data[idx + 2])) / 255.0;

            const lab = rgbToLab(r, g, b, working_space, illuminant);

            lab_data[idx] = lab.l;
            lab_data[idx + 1] = lab.a;
            lab_data[idx + 2] = lab.b;
        }
    }

    return lab_data;
}

/// Convert image from LAB to RGB color space
pub fn convertImageToRgb(allocator: std.mem.Allocator, lab_data: []const f32, width: u32, height: u32,
                        working_space: RgbWorkingSpace, illuminant: Illuminant) ![]u8 {
    const rgb_data = try allocator.alloc(u8, width * height * 3);

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = (y * width + x) * 3;

            const lab = LabColor.init(lab_data[idx], lab_data[idx + 1], lab_data[idx + 2]);
            const rgb = labToRgb(lab, working_space, illuminant);

            rgb_data[idx] = @intFromFloat(@min(255.0, @max(0.0, rgb[0] * 255.0)));
            rgb_data[idx + 1] = @intFromFloat(@min(255.0, @max(0.0, rgb[1] * 255.0)));
            rgb_data[idx + 2] = @intFromFloat(@min(255.0, @max(0.0, rgb[2] * 255.0)));
        }
    }

    return rgb_data;
}

/// Adjust color saturation in LAB space
pub fn adjustSaturation(lab: LabColor, saturation: f32) LabColor {
    return LabColor.init(lab.l, lab.a * saturation, lab.b * saturation);
}

/// Adjust lightness in LAB space
pub fn adjustLightness(lab: LabColor, lightness_offset: f32) LabColor {
    return LabColor.init(
        @min(100.0, @max(0.0, lab.l + lightness_offset)),
        lab.a,
        lab.b
    );
}