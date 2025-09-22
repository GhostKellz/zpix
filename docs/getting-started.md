# Getting Started with zpix

This guide will help you get up and running with zpix quickly.

## 📦 Installation

### Using Zig's Package Manager

```bash
zig fetch --save https://github.com/ghostkellz/zpix/archive/refs/heads/main.tar.gz
```

This will automatically add zpix to your `build.zig.zon`.

### Manual Installation

Add zpix to your `build.zig.zon`:

```zig
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .zpix = .{
            .url = "https://github.com/ghostkellz/zpix/archive/refs/heads/main.tar.gz",
            .hash = "12208a1b2c3d4e5f6789...", // Get this from zig fetch
        },
    },
}
```

Then in your `build.zig`:

```zig
const zpix = b.dependency("zpix", .{});
exe.root_module.addImport("zpix", zpix.module("zpix"));
```

## 🚀 Your First zpix Program

Create a new Zig file (e.g., `main.zig`):

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load an image
    var image = try zpix.Image.load(allocator, "input.png");
    defer image.deinit();

    // Print some info
    std.debug.print("Loaded image: {}x{} ({})\n",
        .{image.width, image.height, image.format});

    // Save as BMP
    try image.save("output.bmp", .bmp);
    std.debug.print("Image saved as BMP!\n", .{});
}
```

## 🏗️ Project Structure

A typical zpix project looks like this:

```
my-project/
├── build.zig
├── build.zig.zon
├── src/
│   └── main.zig
└── assets/
    └── images/
        ├── input.png
        └── ...
```

## 📋 Supported Formats

### Loading Images

```zig
// BMP files
var bmp_image = try zpix.Image.load(allocator, "image.bmp");

// PNG files
var png_image = try zpix.Image.load(allocator, "image.png");

// JPEG files (partial support)
var jpg_image = try zpix.Image.load(allocator, "image.jpg");
```

### Saving Images

```zig
// Save as BMP (currently the only supported output format)
try image.save("output.bmp", .bmp);
```

## 🎨 Basic Image Operations

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var image = try zpix.Image.load(allocator, "photo.png");
    defer image.deinit();

    // Resize to 800x600
    try image.resize(800, 600);

    // Crop a 200x200 region starting at (100, 100)
    try image.crop(100, 100, 200, 200);

    // Adjust brightness (+20)
    try image.brightness(20);

    // Increase contrast by 20%
    try image.contrast(1.2);

    // Apply blur with radius 2
    try image.blur(2);

    // Convert to grayscale
    try image.convert(.grayscale);

    // Save the result
    try image.save("processed.bmp", .bmp);
}
```

## 🔧 Creating Images Programmatically

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a 256x256 RGB image
    var image = try zpix.Image.init(allocator, 256, 256, .rgb);
    defer image.deinit();

    // Fill with a red gradient
    for (0..256) |y| {
        for (0..256) |x| {
            const idx = (y * 256 + x) * 3;
            image.data[idx] = @intCast(x);     // Red increases left to right
            image.data[idx + 1] = 0;           // No green
            image.data[idx + 2] = @intCast(y); // Blue increases top to bottom
        }
    }

    try image.save("gradient.bmp", .bmp);
}
```

## 🐛 Error Handling

zpix uses Zig's error handling system. Common errors include:

```zig
var image = zpix.Image.load(allocator, "nonexistent.png") catch |err| {
    switch (err) {
        error.FileNotFound => std.debug.print("File not found!\n", .{}),
        error.UnknownFormat => std.debug.print("Unsupported format!\n", .{}),
        error.OutOfMemory => std.debug.print("Not enough memory!\n", .{}),
        else => std.debug.print("Unknown error: {}\n", .{err}),
    }
    return err;
};
defer image.deinit();
```

## 📚 Next Steps

- Check out the [API Reference](api-reference.md) for detailed function documentation
- Look at [Examples](examples.md) for more advanced usage patterns
- Read about the [Architecture](architecture.md) to understand how zpix works internally

## 🆘 Getting Help

- [GitHub Issues](https://github.com/ghostkellz/zpix/issues) - Report bugs or request features
- [GitHub Discussions](https://github.com/ghostkellz/zpix/discussions) - Ask questions and get help
- Check the [README](../README.md) for additional information