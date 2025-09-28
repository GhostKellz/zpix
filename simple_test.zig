//! Simple functionality test for zpix v0.1.0
//! Basic operations to verify the library works correctly

const std = @import("std");
const zpix = @import("src/root.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{
    .safety = true,
    .retain_metadata = true,
}){};

const allocator = gpa.allocator();

pub fn main() !void {
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.log.err("❌ Memory leak detected!", .{});
            std.process.exit(1);
        } else {
            std.log.info("✅ No memory leaks detected!", .{});
        }
    }

    std.log.info("🧪 Testing zpix v0.1.0 basic functionality...", .{});

    try testImageCreation();
    try testImageOperations();
    try testMetadataStructures();

    std.log.info("🎉 All basic functionality tests passed!", .{});
    std.log.info("🚀 zpix v0.1.0 is working correctly!", .{});
}

fn testImageCreation() !void {
    std.log.info("📸 Testing image creation and memory management...", .{});

    // Test basic image creation
    {
        var image = try zpix.Image.init(allocator, 100, 100, .rgb);
        defer image.deinit();

        std.log.info("  ✓ Created 100x100 RGB image ({} bytes)", .{image.getMemoryUsage()});

        // Test memory calculations
        const kb = image.getMemoryUsageKB();
        const mb = image.getMemoryUsageMB();
        std.log.info("  ✓ Memory usage: {d:.2} KB, {d:.4} MB", .{kb, mb});
    }

    // Test different formats
    {
        var rgba_image = try zpix.Image.init(allocator, 50, 50, .rgba);
        defer rgba_image.deinit();
        std.log.info("  ✓ Created 50x50 RGBA image ({} bytes)", .{rgba_image.getMemoryUsage()});

        var gray_image = try zpix.Image.init(allocator, 200, 200, .grayscale);
        defer gray_image.deinit();
        std.log.info("  ✓ Created 200x200 grayscale image ({} bytes)", .{gray_image.getMemoryUsage()});
    }

    std.log.info("✅ Image creation test passed", .{});
}

fn testImageOperations() !void {
    std.log.info("🎨 Testing basic image operations...", .{});

    var image = try zpix.Image.init(allocator, 64, 64, .rgb);
    defer image.deinit();

    // Test resize
    const original_size = image.getMemoryUsage();
    try image.resize(128, 128);
    const new_size = image.getMemoryUsage();
    std.log.info("  ✓ Resized from {} to {} bytes", .{original_size, new_size});

    // Test basic filters
    try image.blur(3);
    std.log.info("  ✓ Applied blur filter", .{});

    try image.adjustBrightness(10);
    std.log.info("  ✓ Adjusted brightness", .{});

    try image.adjustContrast(1.2);
    std.log.info("  ✓ Adjusted contrast", .{});

    // Test rotation
    try image.rotate90();
    std.log.info("  ✓ Rotated 90 degrees", .{});

    // Test color conversion
    try image.convertToGrayscale();
    std.log.info("  ✓ Converted to grayscale", .{});

    std.log.info("✅ Image operations test passed", .{});
}

fn testMetadataStructures() !void {
    std.log.info("📋 Testing metadata structures...", .{});

    // Test EXIF data structure
    {
        var exif = zpix.ExifData.init(allocator);
        defer exif.deinit();
        std.log.info("  ✓ Created EXIF data structure", .{});
    }

    // Test XMP data structure
    {
        var xmp = try zpix.XmpData.init(allocator, "<test>data</test>");
        defer xmp.deinit();
        std.log.info("  ✓ Created XMP data structure", .{});
    }

    // Test IPTC data structure
    {
        var iptc = zpix.IptcData.init(allocator);
        defer iptc.deinit();
        try iptc.setValue(.headline, "Test headline");
        std.log.info("  ✓ Created IPTC data structure", .{});
    }

    // Test advanced structures
    {
        var hdr = try zpix.HdrImage.init(allocator, 32, 32);
        defer hdr.deinit();
        std.log.info("  ✓ Created HDR image structure", .{});
    }

    std.log.info("✅ Metadata structures test passed", .{});
}