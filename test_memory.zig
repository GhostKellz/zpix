//! Memory safety test for zpix v0.1.0
//! Tests for memory leaks, double-frees, and other memory issues

const std = @import("std");
const zpix = @import("src/root.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{
    .safety = true,
    .retain_metadata = true,
    .verbose_log = true,
}){};

const allocator = gpa.allocator();

pub fn main() !void {
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.log.err("Memory leak detected!", .{});
            std.process.exit(1);
        } else {
            std.log.info("✅ No memory leaks detected!", .{});
        }
    }

    std.log.info("🧪 Testing zpix v0.1.0 for memory safety...", .{});

    try testBasicImageOperations();
    try testAdvancedColorScience();
    try testZeroCopyOperations();
    try testBatchProcessing();
    try testMetadataHandling();
    try testGPUAcceleration();

    std.log.info("🎉 All memory safety tests passed!", .{});
}

fn testBasicImageOperations() !void {
    std.log.info("📸 Testing basic image operations...", .{});

    // Test image creation and destruction
    {
        var image = try zpix.Image.init(allocator, 100, 100, .rgb);
        defer image.deinit();

        // Test resize
        try image.resize(200, 150);

        // Test blur
        try image.blur(5);

        // Test brightness/contrast
        try image.adjustBrightness(10);
        try image.adjustContrast(1.5);

        // Test rotation
        try image.rotate90();

        // Test crop
        try image.crop(10, 10, 50, 50);
    }

    // Test color conversions
    {
        var image = try zpix.Image.init(allocator, 50, 50, .rgb);
        defer image.deinit();

        try image.convertToGrayscale();
    }

    std.log.info("✅ Basic image operations memory test passed", .{});
}

fn testAdvancedColorScience() !void {
    std.log.info("🎨 Testing advanced color science...", .{});

    // Test HDR operations
    {
        var hdr_image = try zpix.HdrImage.init(allocator, 100, 100);
        defer hdr_image.deinit();

        const params = zpix.ToneMappingParams{
            .algorithm = .reinhard,
            .exposure = 1.5,
        };

        const ldr_data = try zpix.hdr_functions.toneMap(allocator, &hdr_image, params);
        defer allocator.free(ldr_data);
    }

    // Test LAB color space
    {
        const test_data = try allocator.alloc(u8, 300); // 100 pixels RGB
        defer allocator.free(test_data);
        @memset(test_data, 128);

        const lab_data = try zpix.lab_functions.convertImageToLab(allocator, test_data, 10, 10, .srgb, .d65);
        defer allocator.free(lab_data);

        const rgb_data = try zpix.lab_functions.convertImageToRgb(allocator, lab_data, 10, 10, .srgb, .d65);
        defer allocator.free(rgb_data);
    }

    // Test color profiles
    {
        const test_data = try allocator.alloc(u8, 300);
        defer allocator.free(test_data);
        @memset(test_data, 100);

        const converted = try zpix.color_profile_functions.convertImageProfile(allocator, test_data, 10, 10, .srgb, .adobe_rgb_1998);
        defer allocator.free(converted);
    }

    std.log.info("✅ Advanced color science memory test passed", .{});
}

fn testZeroCopyOperations() !void {
    std.log.info("⚡ Testing zero-copy operations...", .{});

    const test_data = try allocator.alloc(u8, 1200); // 20x20 RGB
    defer allocator.free(test_data);
    @memset(test_data, 64);

    // Test image views
    {
        var view = zpix.ImageView.init(test_data, 20, 20, 3);
        const const_view = zpix.ConstImageView.init(test_data, 20, 20, 3);

        // Test sub-views
        _ = const_view.subView(5, 5, 10, 10);
        var mutable_sub = view.subView(5, 5, 10, 10);

        // Test in-place operations
        zpix.InPlaceOps.adjustBrightness(&mutable_sub, 20);
        zpix.InPlaceOps.flipHorizontal(&mutable_sub);
        zpix.InPlaceOps.flipVertical(&mutable_sub);
    }

    // Test buffer pool
    {
        var pool = zpix.BufferPool.init(allocator);
        defer pool.deinit();

        const buffer1 = try pool.acquire(1000);
        const buffer2 = try pool.acquire(2000);

        try pool.release(buffer1);
        try pool.release(buffer2);

        // Reuse buffers
        const buffer3 = try pool.acquire(500);
        try pool.release(buffer3);
    }

    std.log.info("✅ Zero-copy operations memory test passed", .{});
}

fn testBatchProcessing() !void {
    std.log.info("🔄 Testing batch processing...", .{});

    // Test batch job builder
    {
        var builder = zpix.BatchJobBuilder.init(allocator);
        defer builder.deinit();

        // Add some mock files
        _ = try builder.addFile("test1.jpg");
        _ = try builder.addFile("test2.png");
        _ = try builder.setOutputDirectory("output/");

        const operation = zpix.BatchParams{
            .resize = .{ .width = 800, .height = 600 },
        };
        _ = builder.setOperation(operation);
    }

    // Test cancellation token
    {
        var token = zpix.CancellationToken.init();
        token.cancel();
        std.debug.assert(token.isCancelled());
    }

    std.log.info("✅ Batch processing memory test passed", .{});
}

fn testMetadataHandling() !void {
    std.log.info("📋 Testing metadata handling...", .{});

    // Test EXIF data
    {
        var exif = zpix.ExifData.init(allocator);
        defer exif.deinit();

        // Test adding entries (simplified)
        const rational = zpix.metadata.Rational{ .numerator = 1, .denominator = 125 };
        const rationals = try allocator.alloc(zpix.metadata.Rational, 1);
        defer allocator.free(rationals);
        rationals[0] = rational;

        const entry = zpix.ExifEntry.init(
            .exposure_time,
            .rational,
            1,
            zpix.ExifValue{ .rational = rationals }
        );

        try exif.addEntry(entry);
    }

    // Test XMP data
    {
        var xmp = try zpix.XmpData.init(allocator, "<x:xmpmeta><rdf:RDF><rdf:Description/></rdf:RDF></x:xmpmeta>");
        defer xmp.deinit();
    }

    // Test IPTC data
    {
        var iptc = zpix.IptcData.init(allocator);
        defer iptc.deinit();

        try iptc.setValue(.headline, "Test Headline");
        try iptc.setValue(.keywords, "test,memory,safety");
    }

    std.log.info("✅ Metadata handling memory test passed", .{});
}

fn testGPUAcceleration() !void {
    std.log.info("🚀 Testing GPU acceleration...", .{});

    // Test Vulkan device
    {
        var vulkan_device = zpix.VulkanComputeDevice.init(allocator) catch |err| switch (err) {
            error.VulkanNotAvailable => {
                std.log.warn("Vulkan not available, skipping test", .{});
                return;
            },
            else => return err,
        };
        defer vulkan_device.deinit();

        if (vulkan_device.instance.isAvailable()) {
            try vulkan_device.createPipeline(.resize_bilinear);
            try vulkan_device.createPipeline(.blur_gaussian);
        }
    }

    std.log.info("✅ GPU acceleration memory test passed", .{});
}

// Helper function to create test data
fn createTestData(allocator_arg: std.mem.Allocator, size: usize) ![]u8 {
    const data = try allocator_arg.alloc(u8, size);
    for (data, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    return data;
}