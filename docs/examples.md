# zpix Examples

This document provides practical examples of using zpix for common image processing tasks.

## 📸 Basic Usage

### Loading and Saving Images

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load an image
    var image = try zpix.Image.load(allocator, "input.png");
    defer image.deinit();

    // Save as BMP
    try image.save("output.bmp", .bmp);
}
```

### Creating Images Programmatically

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn createGradient(allocator: std.mem.Allocator) !zpix.Image {
    var image = try zpix.Image.init(allocator, 256, 256, .rgb);
    errdefer image.deinit();

    // Create a red-to-blue gradient
    var y: u32 = 0;
    while (y < image.height) : (y += 1) {
        var x: u32 = 0;
        while (x < image.width) : (x += 1) {
            const r: u8 = @intCast(x * 255 / image.width);
            const b: u8 = @intCast(y * 255 / image.height);
            const g: u8 = 128;

            const index = (y * image.width + x) * 3;
            image.data[index] = r;     // Red
            image.data[index + 1] = g; // Green
            image.data[index + 2] = b; // Blue
        }
    }

    return image;
}
```

## 🎨 Image Processing Pipeline

### Photo Enhancement

```zig
pub fn enhancePhoto(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    var image = try zpix.Image.load(allocator, input_path);
    defer image.deinit();

    // Apply enhancement pipeline
    try image.brightness(10);    // Slightly brighter
    try image.contrast(1.2);     // Increase contrast
    try image.blur(1);           // Subtle sharpening effect

    try image.save(output_path, .bmp);
}
```

### Thumbnail Generation

```zig
pub fn createThumbnail(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    var image = try zpix.Image.load(allocator, input_path);
    defer image.deinit();

    // Resize to thumbnail size
    try image.resize(150, 150);

    // Optional: Convert to grayscale for smaller file size
    try image.convert(.grayscale);

    try image.save(output_path, .bmp);
}
```

### Image Composition

```zig
pub fn overlayImages(allocator: std.mem.Allocator, base_path: []const u8, overlay_path: []const u8, output_path: []const u8) !void {
    var base = try zpix.Image.load(allocator, base_path);
    defer base.deinit();

    var overlay = try zpix.Image.load(allocator, overlay_path);
    defer overlay.deinit();

    // Ensure overlay fits within base image
    if (overlay.width > base.width or overlay.height > base.height) {
        try overlay.resize(
            @min(overlay.width, base.width),
            @min(overlay.height, base.height)
        );
    }

    // Simple alpha blending (assuming RGBA overlay)
    if (overlay.format == .rgba) {
        var y: u32 = 0;
        while (y < overlay.height) : (y += 1) {
            var x: u32 = 0;
            while (x < overlay.width) : (x += 1) {
                const base_idx = ((y + 10) * base.width + (x + 10)) * 3; // Offset by 10px
                const overlay_idx = (y * overlay.width + x) * 4;

                const alpha = @as(f32, @floatFromInt(overlay.data[overlay_idx + 3])) / 255.0;

                // Blend RGB channels
                for (0..3) |c| {
                    const base_val = @as(f32, @floatFromInt(base.data[base_idx + c]));
                    const overlay_val = @as(f32, @floatFromInt(overlay.data[overlay_idx + c]));
                    base.data[base_idx + c] = @intFromFloat(
                        base_val * (1.0 - alpha) + overlay_val * alpha
                    );
                }
            }
        }
    }

    try base.save(output_path, .bmp);
}
```

## 🔄 Format Conversion

### Batch Conversion

```zig
pub fn convertImages(allocator: std.mem.Allocator, input_dir: []const u8, output_dir: []const u8) !void {
    var dir = try std.fs.cwd().openDir(input_dir, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;

        // Check if it's an image file
        const ext = std.fs.path.extension(entry.basename);
        if (!std.mem.eql(u8, ext, ".png") and !std.mem.eql(u8, ext, ".jpg")) continue;

        // Build paths
        const input_path = try std.fs.path.join(allocator, &[_][]const u8{ input_dir, entry.path });
        defer allocator.free(input_path);

        const basename = std.fs.path.stem(entry.basename);
        const output_filename = try std.fmt.allocPrint(allocator, "{s}.bmp", .{basename});
        defer allocator.free(output_filename);

        const output_path = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, output_filename });
        defer allocator.free(output_path);

        // Convert image
        var image = try zpix.Image.load(allocator, input_path);
        defer image.deinit();

        try image.save(output_path, .bmp);
    }
}
```

## 🎯 Advanced Techniques

### Custom Pixel Processing

```zig
pub fn applySepiaTone(image: *zpix.Image) !void {
    if (image.format != .rgb) return error.UnsupportedFormat;

    var i: usize = 0;
    while (i < image.data.len) : (i += 3) {
        const r = @as(f32, @floatFromInt(image.data[i]));
        const g = @as(f32, @floatFromInt(image.data[i + 1]));
        const b = @as(f32, @floatFromInt(image.data[i + 2]));

        // Sepia tone formula
        const new_r = r * 0.393 + g * 0.769 + b * 0.189;
        const new_g = r * 0.349 + g * 0.686 + b * 0.168;
        const new_b = r * 0.272 + g * 0.534 + b * 0.131;

        image.data[i] = @intFromFloat(std.math.clamp(new_r, 0, 255));
        image.data[i + 1] = @intFromFloat(std.math.clamp(new_g, 0, 255));
        image.data[i + 2] = @intFromFloat(std.math.clamp(new_b, 0, 255));
    }
}
```

### Memory-Efficient Processing

```zig
pub fn processLargeImage(allocator: std.mem.Allocator, path: []const u8) !void {
    // Use arena allocator for temporary operations
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temp_allocator = arena.allocator();

    var image = try zpix.Image.load(allocator, path);
    defer image.deinit();

    // Perform operations using temp allocator where possible
    // This reduces peak memory usage

    // Resize using temp allocator for intermediate calculations
    try image.resize(1024, 768);

    // Apply filters
    try image.contrast(1.1);
    try image.brightness(5);

    try image.save("processed.bmp", .bmp);
}
```

## 🧪 Testing Examples

```zig
const std = @import("std");
const zpix = @import("zpix");

test "image processing pipeline" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create test image
    var image = try zpix.Image.init(allocator, 100, 100, .rgb);
    defer image.deinit();

    // Fill with test pattern
    @memset(image.data, 128);

    // Apply processing
    try image.brightness(20);
    try image.resize(50, 50);

    // Verify results
    try std.testing.expectEqual(@as(u32, 50), image.width);
    try std.testing.expectEqual(@as(u32, 50), image.height);
}
```

## 📝 Command Line Usage

```bash
# Build the CLI tool
zig build

# Process an image
./zig-out/bin/zpix process input.png output.bmp --brightness 10 --contrast 1.2

# Create thumbnail
./zig-out/bin/zpix thumbnail input.jpg thumbnail.bmp 150x150

# Convert format
./zig-out/bin/zpix convert input.png output.bmp
```

## 🔗 Integration Examples

### Using with HTTP Server

```zig
const std = @import("std");
const zpix = @import("zpix");
// Assuming you have an HTTP server library

pub fn handleImageUpload(allocator: std.mem.Allocator, image_data: []const u8) ![]const u8 {
    // Parse uploaded image
    var image = try zpix.Image.loadFromMemory(allocator, image_data);
    defer image.deinit();

    // Process image (resize for web)
    try image.resize(800, 600);

    // Convert to JPEG for web delivery
    // Note: JPEG saving not yet implemented in MVP
    // For now, save as BMP
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    // This would be the API for saving to memory
    // try image.saveToMemory(&output, .jpeg);

    return output.toOwnedSlice();
}
```

These examples demonstrate the versatility and ease of use of the zpix library for various image processing tasks.