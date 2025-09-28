//! Basic usage example for zpix
//! This example demonstrates loading, processing, and saving images

const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("🖼️  zpix Basic Usage Example\n");
    std.debug.print("============================\n\n");

    // Example 1: Create a simple image programmatically
    std.debug.print("📦 Creating a 100x100 gradient image...\n");
    var gradient_image = try zpix.Image.init(allocator, 100, 100, .rgb);
    defer gradient_image.deinit();

    // Fill with a red-blue gradient
    for (0..100) |y| {
        for (0..100) |x| {
            const idx = (y * 100 + x) * 3;
            gradient_image.data[idx] = @intCast(x * 255 / 99);     // Red gradient
            gradient_image.data[idx + 1] = 0;                      // No green
            gradient_image.data[idx + 2] = @intCast(y * 255 / 99); // Blue gradient
        }
    }

    // Save as BMP
    try gradient_image.save("/tmp/gradient.bmp", .bmp);
    std.debug.print("✅ Saved gradient image to /tmp/gradient.bmp\n\n");

    // Example 2: Create a test pattern with RGBA
    std.debug.print("🎨 Creating a test pattern with alpha channel...\n");
    var alpha_image = try zpix.Image.init(allocator, 64, 64, .rgba);
    defer alpha_image.deinit();

    // Create a checkerboard pattern with varying alpha
    for (0..64) |y| {
        for (0..64) |x| {
            const idx = (y * 64 + x) * 4;
            const is_white = ((x / 8) + (y / 8)) % 2 == 0;

            if (is_white) {
                alpha_image.data[idx] = 255;     // R
                alpha_image.data[idx + 1] = 255; // G
                alpha_image.data[idx + 2] = 255; // B
                alpha_image.data[idx + 3] = 255; // A - fully opaque
            } else {
                alpha_image.data[idx] = 0;       // R
                alpha_image.data[idx + 1] = 0;   // G
                alpha_image.data[idx + 2] = 0;   // B
                alpha_image.data[idx + 3] = @intCast((x + y) * 2); // A - varying transparency
            }
        }
    }

    // Save as PNG (supports alpha)
    try alpha_image.save("/tmp/checkerboard.png", .png);
    std.debug.print("✅ Saved checkerboard with alpha to /tmp/checkerboard.png\n\n");

    // Example 3: Image processing operations
    std.debug.print("🔧 Demonstrating image processing operations...\n");
    var processing_image = try zpix.Image.init(allocator, 50, 50, .rgb);
    defer processing_image.deinit();

    // Fill with a solid color
    for (processing_image.data) |*pixel| {
        pixel.* = 128; // Medium gray
    }

    std.debug.print("   📏 Original size: {}x{}\n", .{ processing_image.width, processing_image.height });

    // Resize the image
    try processing_image.resize(100, 100);
    std.debug.print("   📐 After resize: {}x{}\n", .{ processing_image.width, processing_image.height });

    // Adjust brightness
    try processing_image.adjustBrightness(50);
    std.debug.print("   ☀️  Applied brightness adjustment (+50)\n");

    // Apply blur
    try processing_image.blur(3);
    std.debug.print("   🌫️  Applied blur (radius: 3)\n");

    // Save processed image
    try processing_image.save("/tmp/processed.bmp", .bmp);
    std.debug.print("✅ Saved processed image to /tmp/processed.bmp\n\n");

    std.debug.print("🎉 Basic usage example completed successfully!\n");
    std.debug.print("📁 Check /tmp/ directory for generated images.\n");
}