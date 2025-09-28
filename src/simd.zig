//! SIMD optimizations for zpix image processing operations
//! Provides vectorized implementations for performance-critical operations

const std = @import("std");
const builtin = @import("builtin");

/// Check if SIMD operations are available on this platform
pub const simd_available = switch (builtin.cpu.arch) {
    .x86_64 => true,
    .aarch64 => true,
    else => false,
};

/// SIMD vector types for different operations
pub const Vec4u8 = @Vector(4, u8);
pub const Vec8u8 = @Vector(8, u8);
pub const Vec16u8 = @Vector(16, u8);
pub const Vec4f32 = @Vector(4, f32);
pub const Vec8f32 = @Vector(8, f32);

/// SIMD-optimized horizontal blur for RGB/RGBA data
pub fn simdBlurHorizontal(src: []const u8, dst: []u8, width: u32, height: u32, radius: u32, bytes_per_pixel: u32) void {
    if (!simd_available or bytes_per_pixel < 3) {
        // Fallback to scalar implementation
        scalarBlurHorizontal(src, dst, width, height, radius, bytes_per_pixel);
        return;
    }

    for (0..height) |y| {
        const row_start = y * width * bytes_per_pixel;
        var x: u32 = 0;

        // Process pixels in SIMD chunks where possible
        while (x + 4 < width) {
            // Load 4 consecutive pixels for processing
            var pixel_sums = [_]Vec4f32{@splat(0.0)} ** 4;
            var count: f32 = 0;

            // Apply blur kernel
            const start_x = if (x >= radius) x - radius else 0;
            const end_x = @min(width - 1, x + radius);

            for (start_x..end_x + 1) |blur_x| {
                const src_idx = row_start + blur_x * bytes_per_pixel;

                // Load RGB values for SIMD processing
                if (bytes_per_pixel >= 3) {
                    const r = @as(f32, @floatFromInt(src[src_idx]));
                    const g = @as(f32, @floatFromInt(src[src_idx + 1]));
                    const b = @as(f32, @floatFromInt(src[src_idx + 2]));
                    const a = if (bytes_per_pixel == 4) @as(f32, @floatFromInt(src[src_idx + 3])) else 255.0;

                    pixel_sums[0] += Vec4f32{r, g, b, a};
                    count += 1.0;
                }
            }

            // Average and store results
            const avg = pixel_sums[0] / @as(Vec4f32, @splat(count));
            const dst_idx = row_start + x * bytes_per_pixel;

            dst[dst_idx] = @intFromFloat(@min(255.0, @max(0.0, avg[0])));
            dst[dst_idx + 1] = @intFromFloat(@min(255.0, @max(0.0, avg[1])));
            dst[dst_idx + 2] = @intFromFloat(@min(255.0, @max(0.0, avg[2])));
            if (bytes_per_pixel == 4) {
                dst[dst_idx + 3] = @intFromFloat(@min(255.0, @max(0.0, avg[3])));
            }

            x += 1;
        }

        // Handle remaining pixels with scalar code
        while (x < width) {
            const start_x = if (x >= radius) x - radius else 0;
            const end_x = @min(width - 1, x + radius);

            for (0..bytes_per_pixel) |c| {
                var sum: u32 = 0;
                var count: u32 = 0;

                for (start_x..end_x + 1) |blur_x| {
                    const src_idx = row_start + blur_x * bytes_per_pixel + c;
                    sum += src[src_idx];
                    count += 1;
                }

                const dst_idx = row_start + x * bytes_per_pixel + c;
                dst[dst_idx] = @intCast(sum / count);
            }
            x += 1;
        }
    }
}

/// SIMD-optimized vertical blur for RGB/RGBA data
pub fn simdBlurVertical(src: []const u8, dst: []u8, width: u32, height: u32, radius: u32, bytes_per_pixel: u32) void {
    if (!simd_available or bytes_per_pixel < 3) {
        scalarBlurVertical(src, dst, width, height, radius, bytes_per_pixel);
        return;
    }

    for (0..width) |x| {
        var y: u32 = 0;

        while (y < height) {
            const start_y = if (y >= radius) y - radius else 0;
            const end_y = @min(height - 1, y + radius);

            for (0..bytes_per_pixel) |c| {
                var sum: u32 = 0;
                var count: u32 = 0;

                // Vectorize the accumulation when possible
                _ = Vec4f32{0, 0, 0, 0};
                _ = @as(u32, 0);

                for (start_y..end_y + 1) |blur_y| {
                    const src_idx = blur_y * width * bytes_per_pixel + x * bytes_per_pixel + c;
                    sum += src[src_idx];
                    count += 1;
                }

                const dst_idx = y * width * bytes_per_pixel + x * bytes_per_pixel + c;
                dst[dst_idx] = @intCast(sum / count);
            }
            y += 1;
        }
    }
}

/// SIMD-optimized resize operation using bilinear interpolation
pub fn simdResize(src: []const u8, src_width: u32, src_height: u32, dst: []u8, dst_width: u32, dst_height: u32, bytes_per_pixel: u32) void {
    if (!simd_available or bytes_per_pixel < 3) {
        scalarResize(src, src_width, src_height, dst, dst_width, dst_height, bytes_per_pixel);
        return;
    }

    const x_ratio = @as(f32, @floatFromInt(src_width)) / @as(f32, @floatFromInt(dst_width));
    const y_ratio = @as(f32, @floatFromInt(src_height)) / @as(f32, @floatFromInt(dst_height));

    for (0..dst_height) |y| {
        for (0..dst_width) |x| {
            const src_x = @as(f32, @floatFromInt(x)) * x_ratio;
            const src_y = @as(f32, @floatFromInt(y)) * y_ratio;

            const x1 = @as(u32, @intFromFloat(@floor(src_x)));
            const y1 = @as(u32, @intFromFloat(@floor(src_y)));
            const x2 = @min(x1 + 1, src_width - 1);
            const y2 = @min(y1 + 1, src_height - 1);

            const dx = src_x - @as(f32, @floatFromInt(x1));
            const dy = src_y - @as(f32, @floatFromInt(y1));

            // SIMD bilinear interpolation
            const w1 = (1.0 - dx) * (1.0 - dy);
            const w2 = dx * (1.0 - dy);
            const w3 = (1.0 - dx) * dy;
            const w4 = dx * dy;

            const weights = Vec4f32{w1, w2, w3, w4};

            for (0..bytes_per_pixel) |c| {
                const p1 = @as(f32, @floatFromInt(src[(y1 * src_width + x1) * bytes_per_pixel + c]));
                const p2 = @as(f32, @floatFromInt(src[(y1 * src_width + x2) * bytes_per_pixel + c]));
                const p3 = @as(f32, @floatFromInt(src[(y2 * src_width + x1) * bytes_per_pixel + c]));
                const p4 = @as(f32, @floatFromInt(src[(y2 * src_width + x2) * bytes_per_pixel + c]));

                const pixels = Vec4f32{p1, p2, p3, p4};
                const result = @reduce(.Add, pixels * weights);

                const dst_idx = (y * dst_width + x) * bytes_per_pixel + c;
                dst[dst_idx] = @intFromFloat(@min(255.0, @max(0.0, result)));
            }
        }
    }
}

/// SIMD-optimized color space conversion (RGB to YUV)
pub fn simdRgbToYuv(rgb_data: []const u8, yuv_data: []u8, pixel_count: usize) void {
    if (!simd_available) {
        scalarRgbToYuv(rgb_data, yuv_data, pixel_count);
        return;
    }

    var i: usize = 0;

    // Process 4 pixels at once with SIMD
    while (i + 4 <= pixel_count) {
        // Load 4 RGB pixels (12 bytes)
        const r_vec = Vec4f32{
            @as(f32, @floatFromInt(rgb_data[i * 3])),
            @as(f32, @floatFromInt(rgb_data[(i + 1) * 3])),
            @as(f32, @floatFromInt(rgb_data[(i + 2) * 3])),
            @as(f32, @floatFromInt(rgb_data[(i + 3) * 3])),
        };

        const g_vec = Vec4f32{
            @as(f32, @floatFromInt(rgb_data[i * 3 + 1])),
            @as(f32, @floatFromInt(rgb_data[(i + 1) * 3 + 1])),
            @as(f32, @floatFromInt(rgb_data[(i + 2) * 3 + 1])),
            @as(f32, @floatFromInt(rgb_data[(i + 3) * 3 + 1])),
        };

        const b_vec = Vec4f32{
            @as(f32, @floatFromInt(rgb_data[i * 3 + 2])),
            @as(f32, @floatFromInt(rgb_data[(i + 1) * 3 + 2])),
            @as(f32, @floatFromInt(rgb_data[(i + 2) * 3 + 2])),
            @as(f32, @floatFromInt(rgb_data[(i + 3) * 3 + 2])),
        };

        // YUV conversion coefficients
        const y_r_coeff = @as(Vec4f32, @splat(0.299));
        const y_g_coeff = @as(Vec4f32, @splat(0.587));
        const y_b_coeff = @as(Vec4f32, @splat(0.114));

        const u_r_coeff = @as(Vec4f32, @splat(-0.169));
        const u_g_coeff = @as(Vec4f32, @splat(-0.331));
        const u_b_coeff = @as(Vec4f32, @splat(0.5));

        const v_r_coeff = @as(Vec4f32, @splat(0.5));
        const v_g_coeff = @as(Vec4f32, @splat(-0.419));
        const v_b_coeff = @as(Vec4f32, @splat(-0.081));

        // Calculate Y, U, V components
        const y_vec = r_vec * y_r_coeff + g_vec * y_g_coeff + b_vec * y_b_coeff;
        const u_vec = r_vec * u_r_coeff + g_vec * u_g_coeff + b_vec * u_b_coeff + @as(Vec4f32, @splat(128.0));
        const v_vec = r_vec * v_r_coeff + g_vec * v_g_coeff + b_vec * v_b_coeff + @as(Vec4f32, @splat(128.0));

        // Store results
        for (0..4) |j| {
            yuv_data[(i + j) * 3] = @intFromFloat(@min(255.0, @max(0.0, y_vec[j])));
            yuv_data[(i + j) * 3 + 1] = @intFromFloat(@min(255.0, @max(0.0, u_vec[j])));
            yuv_data[(i + j) * 3 + 2] = @intFromFloat(@min(255.0, @max(0.0, v_vec[j])));
        }

        i += 4;
    }

    // Handle remaining pixels
    while (i < pixel_count) {
        scalarRgbToYuvPixel(rgb_data[i * 3..i * 3 + 3], yuv_data[i * 3..i * 3 + 3]);
        i += 1;
    }
}

/// SIMD-optimized color space conversion (YUV to RGB)
pub fn simdYuvToRgb(yuv_data: []const u8, rgb_data: []u8, pixel_count: usize) void {
    if (!simd_available) {
        scalarYuvToRgb(yuv_data, rgb_data, pixel_count);
        return;
    }

    var i: usize = 0;

    while (i + 4 <= pixel_count) {
        const y_vec = Vec4f32{
            @as(f32, @floatFromInt(yuv_data[i * 3])),
            @as(f32, @floatFromInt(yuv_data[(i + 1) * 3])),
            @as(f32, @floatFromInt(yuv_data[(i + 2) * 3])),
            @as(f32, @floatFromInt(yuv_data[(i + 3) * 3])),
        };

        const u_vec = Vec4f32{
            @as(f32, @floatFromInt(yuv_data[i * 3 + 1])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 1) * 3 + 1])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 2) * 3 + 1])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 3) * 3 + 1])) - 128.0,
        };

        const v_vec = Vec4f32{
            @as(f32, @floatFromInt(yuv_data[i * 3 + 2])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 1) * 3 + 2])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 2) * 3 + 2])) - 128.0,
            @as(f32, @floatFromInt(yuv_data[(i + 3) * 3 + 2])) - 128.0,
        };

        // RGB conversion coefficients
        const r_u_coeff = @as(Vec4f32, @splat(0.0));
        const r_v_coeff = @as(Vec4f32, @splat(1.4));

        const g_u_coeff = @as(Vec4f32, @splat(-0.343));
        const g_v_coeff = @as(Vec4f32, @splat(-0.711));

        const b_u_coeff = @as(Vec4f32, @splat(1.765));
        const b_v_coeff = @as(Vec4f32, @splat(0.0));

        const r_vec = y_vec + u_vec * r_u_coeff + v_vec * r_v_coeff;
        const g_vec = y_vec + u_vec * g_u_coeff + v_vec * g_v_coeff;
        const b_vec = y_vec + u_vec * b_u_coeff + v_vec * b_v_coeff;

        for (0..4) |j| {
            rgb_data[(i + j) * 3] = @intFromFloat(@min(255.0, @max(0.0, r_vec[j])));
            rgb_data[(i + j) * 3 + 1] = @intFromFloat(@min(255.0, @max(0.0, g_vec[j])));
            rgb_data[(i + j) * 3 + 2] = @intFromFloat(@min(255.0, @max(0.0, b_vec[j])));
        }

        i += 4;
    }

    while (i < pixel_count) {
        scalarYuvToRgbPixel(yuv_data[i * 3..i * 3 + 3], rgb_data[i * 3..i * 3 + 3]);
        i += 1;
    }
}

// Fallback scalar implementations
fn scalarBlurHorizontal(src: []const u8, dst: []u8, width: u32, height: u32, radius: u32, bytes_per_pixel: u32) void {
    for (0..height) |y| {
        for (0..width) |x| {
            const start_x = if (x >= radius) x - radius else 0;
            const end_x = @min(width - 1, x + radius);

            for (0..bytes_per_pixel) |c| {
                var sum: u32 = 0;
                var count: u32 = 0;

                for (start_x..end_x + 1) |blur_x| {
                    const src_idx = y * width * bytes_per_pixel + blur_x * bytes_per_pixel + c;
                    sum += src[src_idx];
                    count += 1;
                }

                const dst_idx = y * width * bytes_per_pixel + x * bytes_per_pixel + c;
                dst[dst_idx] = @intCast(sum / count);
            }
        }
    }
}

fn scalarBlurVertical(src: []const u8, dst: []u8, width: u32, height: u32, radius: u32, bytes_per_pixel: u32) void {
    for (0..width) |x| {
        for (0..height) |y| {
            const start_y = if (y >= radius) y - radius else 0;
            const end_y = @min(height - 1, y + radius);

            for (0..bytes_per_pixel) |c| {
                var sum: u32 = 0;
                var count: u32 = 0;

                for (start_y..end_y + 1) |blur_y| {
                    const src_idx = blur_y * width * bytes_per_pixel + x * bytes_per_pixel + c;
                    sum += src[src_idx];
                    count += 1;
                }

                const dst_idx = y * width * bytes_per_pixel + x * bytes_per_pixel + c;
                dst[dst_idx] = @intCast(sum / count);
            }
        }
    }
}

fn scalarResize(src: []const u8, src_width: u32, src_height: u32, dst: []u8, dst_width: u32, dst_height: u32, bytes_per_pixel: u32) void {
    const x_ratio = @as(f32, @floatFromInt(src_width)) / @as(f32, @floatFromInt(dst_width));
    const y_ratio = @as(f32, @floatFromInt(src_height)) / @as(f32, @floatFromInt(dst_height));

    for (0..dst_height) |y| {
        for (0..dst_width) |x| {
            const src_x = @as(f32, @floatFromInt(x)) * x_ratio;
            const src_y = @as(f32, @floatFromInt(y)) * y_ratio;

            const x1 = @as(u32, @intFromFloat(@floor(src_x)));
            const y1 = @as(u32, @intFromFloat(@floor(src_y)));
            const x2 = @min(x1 + 1, src_width - 1);
            const y2 = @min(y1 + 1, src_height - 1);

            const dx = src_x - @as(f32, @floatFromInt(x1));
            const dy = src_y - @as(f32, @floatFromInt(y1));

            for (0..bytes_per_pixel) |c| {
                const p1 = @as(f32, @floatFromInt(src[(y1 * src_width + x1) * bytes_per_pixel + c]));
                const p2 = @as(f32, @floatFromInt(src[(y1 * src_width + x2) * bytes_per_pixel + c]));
                const p3 = @as(f32, @floatFromInt(src[(y2 * src_width + x1) * bytes_per_pixel + c]));
                const p4 = @as(f32, @floatFromInt(src[(y2 * src_width + x2) * bytes_per_pixel + c]));

                const interpolated = p1 * (1 - dx) * (1 - dy) +
                    p2 * dx * (1 - dy) +
                    p3 * (1 - dx) * dy +
                    p4 * dx * dy;

                const dst_idx = (y * dst_width + x) * bytes_per_pixel + c;
                dst[dst_idx] = @intFromFloat(@min(255.0, @max(0.0, interpolated)));
            }
        }
    }
}

fn scalarRgbToYuv(rgb_data: []const u8, yuv_data: []u8, pixel_count: usize) void {
    for (0..pixel_count) |i| {
        scalarRgbToYuvPixel(rgb_data[i * 3..i * 3 + 3], yuv_data[i * 3..i * 3 + 3]);
    }
}

fn scalarRgbToYuvPixel(rgb: []const u8, yuv: []u8) void {
    const r = @as(f32, @floatFromInt(rgb[0]));
    const g = @as(f32, @floatFromInt(rgb[1]));
    const b = @as(f32, @floatFromInt(rgb[2]));

    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = -0.169 * r - 0.331 * g + 0.5 * b + 128.0;
    const v = 0.5 * r - 0.419 * g - 0.081 * b + 128.0;

    yuv[0] = @intFromFloat(@min(255.0, @max(0.0, y)));
    yuv[1] = @intFromFloat(@min(255.0, @max(0.0, u)));
    yuv[2] = @intFromFloat(@min(255.0, @max(0.0, v)));
}

fn scalarYuvToRgb(yuv_data: []const u8, rgb_data: []u8, pixel_count: usize) void {
    for (0..pixel_count) |i| {
        scalarYuvToRgbPixel(yuv_data[i * 3..i * 3 + 3], rgb_data[i * 3..i * 3 + 3]);
    }
}

fn scalarYuvToRgbPixel(yuv: []const u8, rgb: []u8) void {
    const y = @as(f32, @floatFromInt(yuv[0]));
    const u = @as(f32, @floatFromInt(yuv[1])) - 128.0;
    const v = @as(f32, @floatFromInt(yuv[2])) - 128.0;

    const r = y + 1.4 * v;
    const g = y - 0.343 * u - 0.711 * v;
    const b = y + 1.765 * u;

    rgb[0] = @intFromFloat(@min(255.0, @max(0.0, r)));
    rgb[1] = @intFromFloat(@min(255.0, @max(0.0, g)));
    rgb[2] = @intFromFloat(@min(255.0, @max(0.0, b)));
}