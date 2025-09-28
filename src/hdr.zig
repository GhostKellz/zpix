//! HDR tone mapping and high dynamic range image processing
//! Provides various tone mapping algorithms for converting HDR to LDR

const std = @import("std");
const math = std.math;

/// HDR tone mapping algorithms
pub const ToneMappingAlgorithm = enum {
    reinhard,           // Basic Reinhard tone mapping
    reinhard_extended,  // Extended Reinhard with white point
    aces,              // ACES filmic tone mapping
    uncharted2,        // Uncharted 2 tone mapping
    exposure,          // Simple exposure adjustment
    linear,            // Linear clamp
};

/// HDR tone mapping parameters
pub const ToneMappingParams = struct {
    algorithm: ToneMappingAlgorithm = .reinhard,
    exposure: f32 = 1.0,
    white_point: f32 = 1.0,
    gamma: f32 = 2.2,

    // Algorithm-specific parameters
    reinhard_key: f32 = 0.18,      // Key value for Reinhard
    aces_a: f32 = 2.51,            // ACES shoulder strength
    aces_b: f32 = 0.03,            // ACES linear strength
    aces_c: f32 = 2.43,            // ACES linear angle
    aces_d: f32 = 0.59,            // ACES toe strength
    aces_e: f32 = 0.14,            // ACES toe numerator
};

/// HDR image data (32-bit float per channel)
pub const HdrImage = struct {
    width: u32,
    height: u32,
    data: []f32, // RGB float data
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !HdrImage {
        const data = try allocator.alloc(f32, width * height * 3);
        return HdrImage{
            .width = width,
            .height = height,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HdrImage) void {
        self.allocator.free(self.data);
    }

    pub fn getPixel(self: *const HdrImage, x: u32, y: u32) [3]f32 {
        const idx = (y * self.width + x) * 3;
        return [3]f32{ self.data[idx], self.data[idx + 1], self.data[idx + 2] };
    }

    pub fn setPixel(self: *HdrImage, x: u32, y: u32, color: [3]f32) void {
        const idx = (y * self.width + x) * 3;
        self.data[idx] = color[0];
        self.data[idx + 1] = color[1];
        self.data[idx + 2] = color[2];
    }
};

/// Apply tone mapping to HDR image, producing LDR result
pub fn toneMap(allocator: std.mem.Allocator, hdr: *const HdrImage, params: ToneMappingParams) ![]u8 {
    const ldr_data = try allocator.alloc(u8, hdr.width * hdr.height * 3);

    // Pre-calculate luminance statistics if needed
    var avg_luminance: f32 = 0.0;
    var max_luminance: f32 = 0.0;

    if (params.algorithm == .reinhard or params.algorithm == .reinhard_extended) {
        var log_sum: f32 = 0.0;
        var pixel_count: u32 = 0;

        for (0..hdr.height) |y| {
            for (0..hdr.width) |x| {
                const pixel = hdr.getPixel(@intCast(x), @intCast(y));
                const luminance = rgbToLuminance(pixel);
                if (luminance > 0.0) {
                    log_sum += @log(luminance + 0.001);
                    max_luminance = @max(max_luminance, luminance);
                    pixel_count += 1;
                }
            }
        }

        if (pixel_count > 0) {
            avg_luminance = @exp(log_sum / @as(f32, @floatFromInt(pixel_count)));
        }
    }

    // Apply tone mapping to each pixel
    for (0..hdr.height) |y| {
        for (0..hdr.width) |x| {
            const idx = (y * hdr.width + x) * 3;
            const hdr_pixel = hdr.getPixel(@intCast(x), @intCast(y));

            // Apply exposure
            const exposed_pixel = [3]f32{
                hdr_pixel[0] * params.exposure,
                hdr_pixel[1] * params.exposure,
                hdr_pixel[2] * params.exposure,
            };

            // Apply tone mapping algorithm
            const tone_mapped = switch (params.algorithm) {
                .linear => linearToneMap(exposed_pixel),
                .exposure => exposureToneMap(exposed_pixel, 1.0),
                .reinhard => reinhardToneMap(exposed_pixel, avg_luminance, params.reinhard_key),
                .reinhard_extended => reinhardExtendedToneMap(exposed_pixel, avg_luminance, params.reinhard_key, params.white_point),
                .aces => acesToneMap(exposed_pixel, params),
                .uncharted2 => uncharted2ToneMap(exposed_pixel),
            };

            // Apply gamma correction and convert to 8-bit
            ldr_data[idx] = floatToU8(gammaCorrect(tone_mapped[0], params.gamma));
            ldr_data[idx + 1] = floatToU8(gammaCorrect(tone_mapped[1], params.gamma));
            ldr_data[idx + 2] = floatToU8(gammaCorrect(tone_mapped[2], params.gamma));
        }
    }

    return ldr_data;
}

/// Convert HDR data from u16 format to float HDR image
pub fn u16ToHdr(allocator: std.mem.Allocator, data: []const u16, width: u32, height: u32, max_value: u16) !HdrImage {
    var hdr = try HdrImage.init(allocator, width, height);
    const scale = 1.0 / @as(f32, @floatFromInt(max_value));

    for (0..data.len) |i| {
        hdr.data[i] = @as(f32, @floatFromInt(data[i])) * scale;
    }

    return hdr;
}

/// Calculate luminance from RGB values
fn rgbToLuminance(rgb: [3]f32) f32 {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/// Linear tone mapping (simple clamp)
fn linearToneMap(color: [3]f32) [3]f32 {
    return [3]f32{
        @min(1.0, @max(0.0, color[0])),
        @min(1.0, @max(0.0, color[1])),
        @min(1.0, @max(0.0, color[2])),
    };
}

/// Simple exposure tone mapping
fn exposureToneMap(color: [3]f32, exposure_bias: f32) [3]f32 {
    const scale = math.pow(f32, 2.0, exposure_bias);
    return [3]f32{
        1.0 - @exp(-color[0] * scale),
        1.0 - @exp(-color[1] * scale),
        1.0 - @exp(-color[2] * scale),
    };
}

/// Reinhard tone mapping
fn reinhardToneMap(color: [3]f32, avg_luminance: f32, key: f32) [3]f32 {
    const luminance = rgbToLuminance(color);
    if (luminance <= 0.0) return [3]f32{ 0.0, 0.0, 0.0 };

    const scaled_luminance = key * luminance / (avg_luminance + 0.001);
    const tone_mapped_luminance = scaled_luminance / (1.0 + scaled_luminance);

    const scale = tone_mapped_luminance / luminance;
    return [3]f32{
        color[0] * scale,
        color[1] * scale,
        color[2] * scale,
    };
}

/// Extended Reinhard tone mapping with white point
fn reinhardExtendedToneMap(color: [3]f32, avg_luminance: f32, key: f32, white_point: f32) [3]f32 {
    const luminance = rgbToLuminance(color);
    if (luminance <= 0.0) return [3]f32{ 0.0, 0.0, 0.0 };

    const scaled_luminance = key * luminance / (avg_luminance + 0.001);
    const white_scale = 1.0 + scaled_luminance / (white_point * white_point);
    const tone_mapped_luminance = (scaled_luminance * white_scale) / (1.0 + scaled_luminance);

    const scale = tone_mapped_luminance / luminance;
    return [3]f32{
        color[0] * scale,
        color[1] * scale,
        color[2] * scale,
    };
}

/// ACES filmic tone mapping
fn acesToneMap(color: [3]f32, params: ToneMappingParams) [3]f32 {
    const a = params.aces_a;
    const b = params.aces_b;
    const c = params.aces_c;
    const d = params.aces_d;
    const e = params.aces_e;

    return [3]f32{
        acesChannel(color[0], a, b, c, d, e),
        acesChannel(color[1], a, b, c, d, e),
        acesChannel(color[2], a, b, c, d, e),
    };
}

fn acesChannel(x: f32, a: f32, b: f32, c: f32, d: f32, e: f32) f32 {
    return @min(1.0, @max(0.0, (x * (a * x + b)) / (x * (c * x + d) + e)));
}

/// Uncharted 2 tone mapping
fn uncharted2ToneMap(color: [3]f32) [3]f32 {
    const exposure_bias = 2.0;
    const shoulder_strength = 0.15;
    const linear_strength = 0.50;
    const linear_angle = 0.10;
    const toe_strength = 0.20;
    const toe_numerator = 0.02;
    const toe_denominator = 0.30;
    const linear_white_point = 11.2;

    const curr_r = uncharted2Function(color[0] * exposure_bias, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);
    const curr_g = uncharted2Function(color[1] * exposure_bias, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);
    const curr_b = uncharted2Function(color[2] * exposure_bias, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);
    const white_scale = 1.0 / uncharted2Function(linear_white_point, shoulder_strength, linear_strength, linear_angle, toe_strength, toe_numerator, toe_denominator);

    return [3]f32{
        curr_r * white_scale,
        curr_g * white_scale,
        curr_b * white_scale,
    };
}

fn uncharted2Function(x: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) f32 {
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

/// Apply gamma correction
fn gammaCorrect(value: f32, gamma: f32) f32 {
    if (value <= 0.0) return 0.0;
    return math.pow(f32, value, 1.0 / gamma);
}

/// Convert float [0,1] to u8 [0,255]
fn floatToU8(value: f32) u8 {
    return @intFromFloat(@min(255.0, @max(0.0, value * 255.0)));
}

/// Create HDR image from standard LDR image (for testing)
pub fn ldrToHdr(allocator: std.mem.Allocator, ldr_data: []const u8, width: u32, height: u32) !HdrImage {
    var hdr = try HdrImage.init(allocator, width, height);

    for (0..ldr_data.len) |i| {
        // Simple conversion: normalize to [0,1] and apply inverse gamma
        const normalized = @as(f32, @floatFromInt(ldr_data[i])) / 255.0;
        hdr.data[i] = math.pow(f32, normalized, 2.2);
    }

    return hdr;
}

/// Auto-exposure calculation for HDR images
pub fn calculateAutoExposure(hdr: *const HdrImage) f32 {
    var log_sum: f32 = 0.0;
    var pixel_count: u32 = 0;

    for (0..hdr.height) |y| {
        for (0..hdr.width) |x| {
            const pixel = hdr.getPixel(@intCast(x), @intCast(y));
            const luminance = rgbToLuminance(pixel);
            if (luminance > 0.0) {
                log_sum += @log(luminance + 0.001);
                pixel_count += 1;
            }
        }
    }

    if (pixel_count == 0) return 1.0;

    const avg_luminance = @exp(log_sum / @as(f32, @floatFromInt(pixel_count)));
    const target_luminance = 0.18; // 18% gray

    return target_luminance / (avg_luminance + 0.001);
}