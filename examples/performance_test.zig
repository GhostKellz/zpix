//! Performance testing example for zpix
//! This example demonstrates performance characteristics and optimization techniques

const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("⚡ zpix Performance Testing Example\n");
    std.debug.print("==================================\n\n");

    // Test different image sizes to show scaling behavior
    const test_sizes = [_][2]u32{
        .{ 64, 64 },
        .{ 256, 256 },
        .{ 512, 512 },
        .{ 1024, 1024 },
    };

    std.debug.print("📊 Performance Scaling Test\n");
    std.debug.print("Size      | Create   | Resize   | Blur     | Memory (KB)\n");
    std.debug.print("----------|----------|----------|----------|------------\n");

    for (test_sizes) |size| {
        const width = size[0];
        const height = size[1];

        // Test image creation
        const create_start = std.time.nanoTimestamp();
        var test_image = try zpix.Image.init(allocator, width, height, .rgb);
        defer test_image.deinit();
        const create_end = std.time.nanoTimestamp();

        // Fill with test pattern
        for (0..height) |y| {
            for (0..width) |x| {
                const idx = (y * width + x) * 3;
                test_image.data[idx] = @intCast((x * 255) / width);
                test_image.data[idx + 1] = @intCast((y * 255) / height);
                test_image.data[idx + 2] = @intCast(((x + y) * 255) / (width + height));
            }
        }

        // Test resize operation
        const resize_start = std.time.nanoTimestamp();
        try test_image.resize(width / 2, height / 2);
        const resize_end = std.time.nanoTimestamp();

        // Test blur operation
        const blur_start = std.time.nanoTimestamp();
        try test_image.blur(2);
        const blur_end = std.time.nanoTimestamp();

        // Calculate times and memory usage
        const create_time = @as(f64, @floatFromInt(create_end - create_start)) / 1_000_000.0;
        const resize_time = @as(f64, @floatFromInt(resize_end - resize_start)) / 1_000_000.0;
        const blur_time = @as(f64, @floatFromInt(blur_end - blur_start)) / 1_000_000.0;
        const memory_kb = (test_image.data.len) / 1024;

        std.debug.print("{:4}x{:4} | {:6.2}ms | {:6.2}ms | {:6.2}ms | {:7} KB\n", .{
            width, height, create_time, resize_time, blur_time, memory_kb,
        });
    }

    std.debug.print("\n🧮 Memory Efficiency Test\n");
    std.debug.print("--------------------------\n");

    // Test memory efficiency for different pixel formats
    const formats = [_]zpix.PixelFormat{ .grayscale, .rgb, .rgba };
    const test_width = 512;
    const test_height = 512;

    for (formats) |format| {
        var format_image = try zpix.Image.init(allocator, test_width, test_height, format);
        defer format_image.deinit();

        const bytes_per_pixel = zpix.bytesPerPixel(format);
        const expected_size = test_width * test_height * bytes_per_pixel;
        const actual_size = format_image.data.len;
        const efficiency = (@as(f64, @floatFromInt(expected_size)) / @as(f64, @floatFromInt(actual_size))) * 100.0;

        std.debug.print("{s:9}: {:6} bytes ({:4} BPP) - {d:.1}% efficient\n", .{
            @tagName(format),
            actual_size,
            bytes_per_pixel,
            efficiency,
        });
    }

    std.debug.print("\n⏱️  Operation Timing Comparison\n");
    std.debug.print("-------------------------------\n");

    // Create a standard test image for operation comparisons
    var ops_image = try zpix.Image.init(allocator, 256, 256, .rgb);
    defer ops_image.deinit();

    // Fill with noise pattern
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();
    for (ops_image.data) |*pixel| {
        pixel.* = random.int(u8);
    }

    // Test different operations
    const operations = [_]struct {
        name: []const u8,
        operation: fn (image: *zpix.Image) anyerror!void,
    }{
        .{ .name = "Brightness +50", .operation = struct {
            fn op(image: *zpix.Image) !void {
                try image.adjustBrightness(50);
            }
        }.op },
        .{ .name = "Brightness -30", .operation = struct {
            fn op(image: *zpix.Image) !void {
                try image.adjustBrightness(-30);
            }
        }.op },
        .{ .name = "Contrast 1.5x", .operation = struct {
            fn op(image: *zpix.Image) !void {
                try image.adjustContrast(1.5);
            }
        }.op },
        .{ .name = "Blur radius 1", .operation = struct {
            fn op(image: *zpix.Image) !void {
                try image.blur(1);
            }
        }.op },
        .{ .name = "Blur radius 3", .operation = struct {
            fn op(image: *zpix.Image) !void {
                try image.blur(3);
            }
        }.op },
    };

    for (operations) |op| {
        // Create a copy for each operation
        var op_image = try zpix.Image.init(allocator, 256, 256, .rgb);
        defer op_image.deinit();
        @memcpy(op_image.data, ops_image.data);

        // Time the operation
        const op_start = std.time.nanoTimestamp();
        try op.operation(&op_image);
        const op_end = std.time.nanoTimestamp();

        const op_time = @as(f64, @floatFromInt(op_end - op_start)) / 1_000_000.0;
        std.debug.print("{s:15}: {d:6.2}ms\n", .{ op.name, op_time });
    }

    std.debug.print("\n🚀 File I/O Performance\n");
    std.debug.print("-----------------------\n");

    // Test file save performance for different formats
    var io_image = try zpix.Image.init(allocator, 400, 400, .rgb);
    defer io_image.deinit();

    // Fill with gradient
    for (0..400) |y| {
        for (0..400) |x| {
            const idx = (y * 400 + x) * 3;
            io_image.data[idx] = @intCast(x * 255 / 399);
            io_image.data[idx + 1] = @intCast(y * 255 / 399);
            io_image.data[idx + 2] = @intCast(((x + y) / 2) * 255 / 399);
        }
    }

    const io_formats = [_]struct { name: []const u8, format: zpix.ImageFormat }{
        .{ .name = "BMP", .format = .bmp },
        .{ .name = "PNG", .format = .png },
    };

    for (io_formats) |fmt| {
        const filename = try std.fmt.allocPrint(allocator, "/tmp/perf_test.{s}", .{fmt.name});
        defer allocator.free(filename);

        const save_start = std.time.nanoTimestamp();
        io_image.save(filename, fmt.format) catch |err| {
            std.debug.print("{s:3} save: Failed ({})\n", .{ fmt.name, err });
            continue;
        };
        const save_end = std.time.nanoTimestamp();

        const save_time = @as(f64, @floatFromInt(save_end - save_start)) / 1_000_000.0;

        // Check file size
        const file = std.fs.openFileAbsolute(filename, .{}) catch continue;
        defer file.close();
        const file_size = file.getEndPos() catch 0;

        std.debug.print("{s:3} save: {d:6.2}ms ({d:7} bytes)\n", .{ fmt.name, save_time, file_size });
    }

    std.debug.print("\n📈 Performance Tips:\n");
    std.debug.print("1. Use appropriate pixel formats (grayscale < RGB < RGBA)\n");
    std.debug.print("2. Process images in-place when possible to save memory\n");
    std.debug.print("3. PNG saves are often faster than BMP due to compression\n");
    std.debug.print("4. Blur operations scale quadratically with radius\n");
    std.debug.print("5. Use resize before other operations to improve performance\n");

    std.debug.print("\n🎉 Performance testing completed!\n");
}