const std = @import("std");
const zpix = @import("zpix");

const Command = enum {
    convert,
    @"test",
    benchmark,
    help,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printHelp();
        return;
    }

    const command_str = args[1];
    const command = std.meta.stringToEnum(Command, command_str) orelse {
        std.debug.print("Unknown command: {s}\n", .{command_str});
        try printHelp();
        return;
    };

    switch (command) {
        .convert => try handleConvert(allocator, args[2..]),
        .@"test" => try handleTest(allocator),
        .benchmark => try handleBenchmark(allocator),
        .help => try printHelp(),
    }
}

fn handleConvert(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    if (args.len < 2) {
        std.debug.print("Usage: zpix convert <input> <output>\n", .{});
        return;
    }

    const input_path = args[0];
    const output_path = args[1];

    // Determine output format from extension
    const output_format = getFormatFromPath(output_path) orelse {
        std.debug.print("Unsupported output format\n", .{});
        return;
    };

    std.debug.print("Converting {s} to {s}\n", .{ input_path, output_path });

    // Load image
    var image = zpix.Image.load(allocator, input_path) catch |err| {
        std.debug.print("Failed to load image: {}\n", .{err});
        return;
    };
    defer image.deinit();

    // Save image
    image.save(output_path, output_format) catch |err| {
        std.debug.print("Failed to save image: {}\n", .{err});
        return;
    };

    std.debug.print("Conversion completed successfully!\n", .{});
}

fn handleTest(allocator: std.mem.Allocator) !void {
    std.debug.print("Running zpix test suite...\n", .{});

    // Create a simple 2x2 RGB image
    var image = try zpix.Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    // Fill with red pixels: R=255, G=0, B=0
    for (0..4) |i| {
        image.data[i * 3] = 255; // R
        image.data[i * 3 + 1] = 0; // G
        image.data[i * 3 + 2] = 0; // B
    }

    // Save as BMP
    try image.save("/tmp/test.bmp", .bmp);
    std.debug.print("✓ Image saved to /tmp/test.bmp\n", .{});

    // Save as PNG
    try image.save("/tmp/test.png", .png);
    std.debug.print("✓ Image saved to /tmp/test.png\n", .{});

    // Test JPEG loading (if file exists)
    const jpeg_path = "/data/projects/zpix/file_example_JPG_100kB.jpg";
    if (std.fs.openFileAbsolute(jpeg_path, .{})) |jf| {
        jf.close();
        var jpeg_image = try zpix.Image.load(allocator, jpeg_path);
        defer jpeg_image.deinit();
        try jpeg_image.save("/tmp/test_loaded.jpg.bmp", .bmp);
        std.debug.print("✓ JPEG loaded and saved as BMP\n", .{});
    } else |_| {
        std.debug.print("- No JPEG file found at {s}\n", .{jpeg_path});
    }

    std.debug.print("All tests completed!\n", .{});
}

fn handleBenchmark(allocator: std.mem.Allocator) !void {
    std.debug.print("Running zpix performance benchmarks...\n", .{});

    // Benchmark 1: Image creation and basic operations
    const start_time = std.time.nanoTimestamp();

    // Create test images of various sizes
    const sizes = [_][2]u32{ .{ 100, 100 }, .{ 500, 500 }, .{ 1000, 1000 } };

    for (sizes) |size| {
        const width = size[0];
        const height = size[1];

        std.debug.print("Benchmarking {}x{} image operations...\n", .{ width, height });

        // Test image creation
        const create_start = std.time.nanoTimestamp();
        var image = try zpix.Image.init(allocator, width, height, .rgb);
        defer image.deinit();
        const create_end = std.time.nanoTimestamp();

        // Fill with test pattern
        for (0..image.data.len) |i| {
            image.data[i] = @intCast(i % 256);
        }

        // Test resize
        const resize_start = std.time.nanoTimestamp();
        try image.resize(width / 2, height / 2);
        const resize_end = std.time.nanoTimestamp();

        // Test blur
        const blur_start = std.time.nanoTimestamp();
        try image.blur(3);
        const blur_end = std.time.nanoTimestamp();

        // Test brightness adjustment
        const brightness_start = std.time.nanoTimestamp();
        try image.adjustBrightness(30);
        const brightness_end = std.time.nanoTimestamp();

        // Print results
        const create_time = @as(f64, @floatFromInt(create_end - create_start)) / 1_000_000.0;
        const resize_time = @as(f64, @floatFromInt(resize_end - resize_start)) / 1_000_000.0;
        const blur_time = @as(f64, @floatFromInt(blur_end - blur_start)) / 1_000_000.0;
        const brightness_time = @as(f64, @floatFromInt(brightness_end - brightness_start)) / 1_000_000.0;

        std.debug.print("  Create: {d:.2}ms\n", .{create_time});
        std.debug.print("  Resize: {d:.2}ms\n", .{resize_time});
        std.debug.print("  Blur:   {d:.2}ms\n", .{blur_time});
        std.debug.print("  Brightness: {d:.2}ms\n", .{brightness_time});
    }

    // Benchmark 2: File I/O operations
    std.debug.print("\nBenchmarking file I/O operations...\n", .{});

    // Create a test image for I/O benchmarking
    var test_image = try zpix.Image.init(allocator, 512, 512, .rgb);
    defer test_image.deinit();

    // Fill with gradient pattern
    for (0..test_image.height) |y| {
        for (0..test_image.width) |x| {
            const idx = (y * test_image.width + x) * 3;
            test_image.data[idx] = @intCast(x % 256);     // R
            test_image.data[idx + 1] = @intCast(y % 256); // G
            test_image.data[idx + 2] = @intCast((x + y) % 256); // B
        }
    }

    // Test BMP save
    const bmp_save_start = std.time.nanoTimestamp();
    try test_image.save("/tmp/benchmark.bmp", .bmp);
    const bmp_save_end = std.time.nanoTimestamp();

    // Test PNG save
    const png_save_start = std.time.nanoTimestamp();
    try test_image.save("/tmp/benchmark.png", .png);
    const png_save_end = std.time.nanoTimestamp();

    // Test JPEG save (if implemented)
    const jpeg_save_start = std.time.nanoTimestamp();
    test_image.save("/tmp/benchmark.jpg", .jpeg) catch |err| {
        std.debug.print("  JPEG Save: Not implemented ({})\n", .{err});
    };
    const jpeg_save_end = std.time.nanoTimestamp();

    // Test WebP save (if implemented)
    const webp_save_start = std.time.nanoTimestamp();
    test_image.save("/tmp/benchmark.webp", .webp) catch |err| {
        std.debug.print("  WebP Save: Not implemented ({})\n", .{err});
    };
    const webp_save_end = std.time.nanoTimestamp();

    const bmp_save_time = @as(f64, @floatFromInt(bmp_save_end - bmp_save_start)) / 1_000_000.0;
    const png_save_time = @as(f64, @floatFromInt(png_save_end - png_save_start)) / 1_000_000.0;
    const jpeg_save_time = @as(f64, @floatFromInt(jpeg_save_end - jpeg_save_start)) / 1_000_000.0;
    const webp_save_time = @as(f64, @floatFromInt(webp_save_end - webp_save_start)) / 1_000_000.0;

    std.debug.print("  BMP Save:  {d:.2}ms\n", .{bmp_save_time});
    std.debug.print("  PNG Save:  {d:.2}ms\n", .{png_save_time});
    if (jpeg_save_time > 0.01) std.debug.print("  JPEG Save: {d:.2}ms\n", .{jpeg_save_time});
    if (webp_save_time > 0.01) std.debug.print("  WebP Save: {d:.2}ms\n", .{webp_save_time});

    const end_time = std.time.nanoTimestamp();
    const total_time = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    // Benchmark 3: Memory usage testing
    std.debug.print("\nBenchmarking memory usage...\n", .{});

    // Test memory efficiency for different image sizes
    const mem_test_sizes = [_][2]u32{ .{ 256, 256 }, .{ 512, 512 }, .{ 1024, 1024 } };

    for (mem_test_sizes) |size| {
        const width = size[0];
        const height = size[1];
        const mem_start = std.time.nanoTimestamp();

        // Create image and measure memory footprint
        var mem_image = try zpix.Image.init(allocator, width, height, .rgba); // RGBA for max memory usage
        defer mem_image.deinit();

        const bytes_per_pixel = zpix.bytesPerPixel(.rgba);
        const expected_bytes = width * height * bytes_per_pixel;
        const actual_bytes = mem_image.data.len;

        std.debug.print("  {}x{} RGBA - Expected: {} bytes, Actual: {} bytes\n",
                       .{ width, height, expected_bytes, actual_bytes });

        // Test multiple allocations
        var images = try std.ArrayList(zpix.Image).initCapacity(allocator, 10);
        defer {
            for (images.items) |*img| {
                img.deinit();
            }
            images.deinit(allocator);
        }

        const multi_alloc_start = std.time.nanoTimestamp();

        // Create 10 smaller images to test allocation patterns
        for (0..10) |_| {
            const small_image = try zpix.Image.init(allocator, width / 4, height / 4, .rgb);
            try images.append(allocator, small_image);
        }

        const multi_alloc_end = std.time.nanoTimestamp();
        const multi_alloc_time = @as(f64, @floatFromInt(multi_alloc_end - multi_alloc_start)) / 1_000_000.0;

        std.debug.print("  Multi-allocation (10x {}x{} RGB): {d:.2}ms\n",
                       .{ width / 4, height / 4, multi_alloc_time });

        const mem_end = std.time.nanoTimestamp();
        const mem_test_time = @as(f64, @floatFromInt(mem_end - mem_start)) / 1_000_000.0;
        std.debug.print("  Total memory test time: {d:.2}ms\n", .{mem_test_time});
    }

    std.debug.print("\nTotal benchmark time: {d:.2}ms\n", .{total_time});
    std.debug.print("Benchmark completed successfully!\n", .{});
}

fn printHelp() !void {
    std.debug.print(
        \\zpix - Image Processing Library v0.1.0
        \\
        \\Usage:
        \\  zpix convert <input> <output>  Convert image between formats
        \\  zpix test                      Run test suite
        \\  zpix benchmark                 Run performance benchmarks
        \\  zpix help                      Show this help
        \\
        \\Supported formats:
        \\  BMP (load/save), JPEG (load), PNG (save)
        \\  WebP (detect), TIFF (detect), GIF (detect)
        \\  AVIF (detect), SVG (detect)
        \\
        \\Examples:
        \\  zpix convert image.jpg output.bmp
        \\  zpix convert image.bmp output.png
        \\  zpix test
        \\
    , .{});
}

fn getFormatFromPath(path: []const u8) ?zpix.ImageFormat {
    if (std.mem.endsWith(u8, path, ".bmp") or std.mem.endsWith(u8, path, ".BMP")) {
        return .bmp;
    } else if (std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".PNG")) {
        return .png;
    } else if (std.mem.endsWith(u8, path, ".jpg") or std.mem.endsWith(u8, path, ".jpeg") or
               std.mem.endsWith(u8, path, ".JPG") or std.mem.endsWith(u8, path, ".JPEG")) {
        return .jpeg;
    } else if (std.mem.endsWith(u8, path, ".webp") or std.mem.endsWith(u8, path, ".WEBP")) {
        return .webp;
    } else if (std.mem.endsWith(u8, path, ".tiff") or std.mem.endsWith(u8, path, ".tif") or
               std.mem.endsWith(u8, path, ".TIFF") or std.mem.endsWith(u8, path, ".TIF")) {
        return .tiff;
    } else if (std.mem.endsWith(u8, path, ".gif") or std.mem.endsWith(u8, path, ".GIF")) {
        return .gif;
    } else if (std.mem.endsWith(u8, path, ".avif") or std.mem.endsWith(u8, path, ".AVIF")) {
        return .avif;
    } else if (std.mem.endsWith(u8, path, ".svg") or std.mem.endsWith(u8, path, ".SVG")) {
        return .svg;
    }
    return null;
}
