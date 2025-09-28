//! Format conversion example for zpix
//! This example demonstrates converting between different image formats

const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("🔄 zpix Format Conversion Example\n");
    std.debug.print("=================================\n\n");

    // Create a test image with different regions
    std.debug.print("🎨 Creating test image with different color regions...\n");
    var test_image = try zpix.Image.init(allocator, 200, 200, .rgb);
    defer test_image.deinit();

    // Fill with different colored quadrants
    for (0..200) |y| {
        for (0..200) |x| {
            const idx = (y * 200 + x) * 3;

            // Determine quadrant
            const is_left = x < 100;
            const is_top = y < 100;

            if (is_top and is_left) {
                // Top-left: Red
                test_image.data[idx] = 255;
                test_image.data[idx + 1] = 0;
                test_image.data[idx + 2] = 0;
            } else if (is_top and !is_left) {
                // Top-right: Green
                test_image.data[idx] = 0;
                test_image.data[idx + 1] = 255;
                test_image.data[idx + 2] = 0;
            } else if (!is_top and is_left) {
                // Bottom-left: Blue
                test_image.data[idx] = 0;
                test_image.data[idx + 1] = 0;
                test_image.data[idx + 2] = 255;
            } else {
                // Bottom-right: Yellow
                test_image.data[idx] = 255;
                test_image.data[idx + 1] = 255;
                test_image.data[idx + 2] = 0;
            }
        }
    }

    std.debug.print("✅ Created 200x200 test image with colored quadrants\n\n");

    // Save in different formats
    const formats = [_]struct { ext: []const u8, format: zpix.ImageFormat }{
        .{ .ext = "bmp", .format = .bmp },
        .{ .ext = "png", .format = .png },
        .{ .ext = "webp", .format = .webp },
    };

    for (formats) |fmt| {
        const filename = try std.fmt.allocPrint(allocator, "/tmp/test_image.{s}", .{fmt.ext});
        defer allocator.free(filename);

        const start_time = std.time.nanoTimestamp();

        test_image.save(filename, fmt.format) catch |err| {
            std.debug.print("❌ Failed to save as {s}: {}\n", .{ fmt.ext, err });
            continue;
        };

        const end_time = std.time.nanoTimestamp();
        const save_time = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

        std.debug.print("✅ Saved as {s} ({d:.2}ms): {s}\n", .{ fmt.ext, save_time, filename });
    }

    std.debug.print("\n🔍 Format Comparison:\n");
    std.debug.print("---------------------\n");
    std.debug.print("BMP:  Uncompressed, large file size, excellent compatibility\n");
    std.debug.print("PNG:  Lossless compression, supports alpha, web-friendly\n");
    std.debug.print("WebP: Modern format, smaller files, good for web use\n\n");

    // Demonstrate loading and re-saving (round-trip test)
    std.debug.print("🔄 Testing round-trip conversion...\n");

    // Try to load the PNG we just saved
    if (zpix.Image.load(allocator, "/tmp/test_image.png")) |loaded_image| {
        defer loaded_image.deinit();

        std.debug.print("✅ Successfully loaded PNG: {}x{} {s}\n", .{
            loaded_image.width,
            loaded_image.height,
            @tagName(loaded_image.format),
        });

        // Convert to BMP
        try loaded_image.save("/tmp/png_to_bmp.bmp", .bmp);
        std.debug.print("✅ Converted PNG → BMP: /tmp/png_to_bmp.bmp\n");

        // Verify pixel data integrity
        var differences: u32 = 0;
        for (0..@min(test_image.data.len, loaded_image.data.len)) |i| {
            if (test_image.data[i] != loaded_image.data[i]) {
                differences += 1;
            }
        }

        if (differences == 0) {
            std.debug.print("✅ Round-trip conversion: Perfect match!\n");
        } else {
            std.debug.print("⚠️  Round-trip conversion: {} pixel differences\n", .{differences});
        }
    } else |err| {
        std.debug.print("❌ Failed to load PNG for round-trip test: {}\n", .{err});
    }

    std.debug.print("\n🎉 Format conversion example completed!\n");
    std.debug.print("📁 Check /tmp/ directory for converted images.\n");
}