# zpix - High-Performance Image Processing Library in Zig

<p align="center">
  <img src="assets/icons/zpix.png" alt="zpix logo" width="200"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/built%20with-zig-yellow.svg?logo=zig" alt="Built with Zig">
  <img src="https://img.shields.io/badge/zig-0.16.0--dev-orange.svg" alt="Zig 0.16.0-dev">
  <img src="https://img.shields.io/badge/memory-safe-green.svg" alt="Memory Safe">
  <img src="https://img.shields.io/badge/SIMD-ready-orange.svg" alt="SIMD Ready">
  <img src="https://img.shields.io/badge/formats-BMP+PNG+JPEG-purple.svg" alt="Formats">
  <img src="https://img.shields.io/badge/operations-resize+crop+filter-red.svg" alt="Operations">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-supported-formats">Formats</a> •
  <a href="#-operations">Operations</a> •
  <a href="#-performance">Performance</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-api">API</a>
</p>

---

## 🚀 About

**zpix** is a high-performance, memory-safe image processing library written in [Zig](https://ziglang.org/). It serves as a modern replacement for C libraries like FreeImage, DevIL, SOIL, and stb_image, providing comprehensive image loading, processing, and saving capabilities with zero-cost abstractions and compile-time safety.

## ⚠️ DISCLAIMER

⚠️ **EXPERIMENTAL LIBRARY - FOR LAB/PERSONAL USE** ⚠️

This is an experimental library under active development. It is
intended for research, learning, and personal projects. The API is subject
to change!

## ✨ Features

- **🛡️ Memory Safety**: Zero-copy operations where possible, controlled allocations, bounds checking
- **⚡ High Performance**: SIMD optimizations, parallel processing for large images
- **🎯 Type Safety**: Leverages Zig's compile-time safety features
- **🔧 Comprehensive API**: Load, process, and save images with a clean, intuitive interface
- **📦 Self-Contained**: No external dependencies, pure Zig implementation
- **🔄 Cross-Platform**: Works on all platforms supported by Zig
- **🧪 Reproducible Workflows**: Command-line pipelines, batch scripts, and streaming I/O for automation

## 📋 Supported Formats

| Format | Load | Save | Status | Notes |
|--------|------|------|--------|-------|
| **BMP** | ✅ | ✅ | Stable | 24-bit RGB/Grayscale |
| **PNG** | ✅ | ✅ | Stable | 8-bit RGB/RGBA (non-interlaced) |
| **JPEG** | ✅ | ✅ | Beta | Baseline decoder/encoder, no progressive support yet |
| **WebP** | ⚠️ | ⚠️ | Experimental | Placeholder decode/encode paths for MVP validation |
| **AVIF** | ✅ | ❌ | Beta | AV1-based decoder (decode-only) |
| **TIFF** | ✅ | ❌ | Beta | Uncompressed 8-bit RGB/RGBA/Grayscale |
| **GIF** | ✅ | ❌ | Beta | Palette decode (first frame), animation reader available |
| **SVG** | ✅ | ❌ | Beta | Rasterizes basic shapes to RGB |

## 🎨 Color Spaces

| Color Space | Support | Notes |
|-------------|---------|-------|
| **RGB** | ✅ | Primary working space |
| **RGBA** | ✅ | Alpha-aware operations and conversions |
| **Grayscale** | ✅ | Luminance conversion helpers |
| **YUV** | ✅ | 4:4:4 RGB ↔ YUV vectorized transforms |
| **HSV** | ✅ | RGB ↔ HSV vectorized transforms |
| **CMYK** | ⚠️ | Pixel type defined; conversions planned |

## 🛠️ Operations

| Operation | Status | Description |
|-----------|--------|-------------|
| **Resize** | ✅ | SIMD-accelerated bilinear interpolation |
| **Crop** | ✅ | Rectangular region extraction |
| **Rotate** | ✅ | 90°, 180°, 270°, and arbitrary-angle rotations |
| **Flip** | ✅ | Horizontal and vertical mirroring |
| **Brightness** | ✅ | Signed adjustment with clamping |
| **Contrast** | ✅ | Floating-point contrast scaling |
| **Blur** | ✅ | Separable box blur with adjustable radius |
| **Grayscale** | ✅ | Luminance conversion from RGB |
| **Format Convert** | ✅ | Pixel format conversions (RGB/RGBA/Grayscale/HSV/YUV) |

## ⚡ Performance

- **SIMD Ready**: Framework prepared for SIMD optimizations
- **Parallel Processing**: Designed for concurrent image processing
- **Memory Efficient**: Exact memory usage tracking, zero-copy where possible
- **Fast Decoding**: Optimized algorithms for common operations
- **Fixed-Point JPEG IDCT**: Baseline JPEG decoding uses an integer inverse DCT for higher throughput
- **Comprehensive Benchmarking**: Built-in performance testing for all operations

### Benchmark Results (on modern hardware)

| Operation | 100x100 | 500x500 | 1000x1000 |
|-----------|---------|----------|-----------|
| **Create** | 0.01ms | 0.22ms | 0.85ms |
| **Resize** | 0.26ms | 6.77ms | 25.47ms |
| **Blur** | 0.30ms | 7.74ms | 30.40ms |
| **Brightness** | 0.13ms | 3.15ms | 12.33ms |

| Format | Save Time (512x512) |
|--------|---------------------|
| **BMP** | 6.64ms |
| **PNG** | 0.47ms |
| **WebP** | 0.37ms |

Run `zig build run -- benchmark` to test performance on your system.

## 📦 Installation

### Using Zig's Package Manager

```bash
zig fetch --save https://github.com/ghostkellz/zpix/archive/refs/heads/main.tar.gz
```

This will automatically add zpix to your `build.zig.zon`:

```zig
.{
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .zpix = .{
            .url = "https://github.com/ghostkellz/zpix/archive/refs/heads/main.tar.gz",
            .hash = "12208a1b2c3d4e5f6789...", // Automatically filled by zig fetch
        },
    },
}
```

Then in your `build.zig`:

```zig
const zpix = b.dependency("zpix", .{});
exe.root_module.addImport("zpix", zpix.module("zpix"));
```

## 🚀 Usage

### Basic Loading and Saving

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load an image
    var image = try zpix.Image.load(allocator, "input.png");
    defer image.deinit();

    // Save as BMP
    try image.save("output.bmp", .bmp);

    std.debug.print("Image processed successfully!\n", .{});
}
```

### Image Processing Pipeline

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load image
    var image = try zpix.Image.load(allocator, "photo.jpg");
    defer image.deinit();

    // Apply processing operations
    try image.resize(800, 600);
    try image.adjustBrightness(10);
    try image.adjustContrast(1.2);
    try image.blur(2);

    // Convert color space
    try image.convert(.grayscale);

    // Save result
    try image.save("processed.bmp", .bmp);
}
```

### Creating Images Programmatically

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create a new 256x256 RGB image
    var image = try zpix.Image.init(allocator, 256, 256, .rgb);
    defer image.deinit();

    // Fill with a gradient
    for (0..256) |y| {
        for (0..256) |x| {
            const idx = (y * 256 + x) * 3;
            image.data[idx] = @intCast(x);     // Red
            image.data[idx + 1] = @intCast(y); // Green
            image.data[idx + 2] = 128;         // Blue
        }
    }

    try image.save("gradient.bmp", .bmp);
}
```

### Command Line Interface

zpix includes a CLI tool for common operations:

```bash
# Convert between formats (detects output from extension)
zig build run -- convert input.png output.bmp

# Stream into a pipeline (reads stdin, writes stdout)
cat input.png | zig build run -- pipeline - resize:512x512 format:bmp save:- > thumb.bmp

# Execute a multi-step pipeline directly
zig build run -- pipeline photo.jpg resize:1024x768 blur:2 format:jpeg save:processed.jpg

# Apply batch jobs from a script file
zig build run -- batch scripts/jobs.zps

# Run test suite / benchmarks / help
zig build test
zig build run -- benchmark
zig build run -- help
```

Example batch script (`scripts/jobs.zps`):

```text
# Create thumbnails and emit PNG files with progress reporting
job operation=resize inputs=assets/gallery/*.bmp output=zig-out/tmp/thumbs width=256 height=256 threads=4 progress=true overwrite=true

# Convert a directory of RAW exports into PNGs while flattening structure
job operation=convert_format inputs=exports/2024 output=zig-out/tmp/exports format=png recursive=true preserve_structure=false

# Apply a warm white balance tweak to a single file
job operation=white_balance input=shoots/portrait.bmp output=zig-out/tmp/portrait temperature=0.15 tint=0.02 overwrite=true
```

Each `job` line uses simple `key=value` pairs:

- `operation` selects the batch action (`resize`, `convert_format`, `blur`, `adjust_brightness`, `rotate`, `crop`, `white_balance`, `color_profile_convert`).
- `inputs` (or `input`) accepts comma-separated file paths, directories, or globs (`*`, `?`). Use `recursive=true` to walk sub-directories.
- `output` sets the destination directory; `preserve_structure=false` flattens nested folders.
- Optional switches such as `threads`, `overwrite`, and `progress` control execution.

## 📚 API Reference

### Image Struct

```zig
pub const Image = struct {
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    format: PixelFormat,
    data: []u8,

    // Core methods
    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: PixelFormat) !Image;
    pub fn deinit(self: *Image) void;
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image;
    pub fn save(self: Image, path: []const u8, format: ImageFormat) !void;

    // Processing methods
    pub fn resize(self: *Image, new_width: u32, new_height: u32) !void;
    pub fn crop(self: *Image, x: u32, y: u32, width: u32, height: u32) !void;
    pub fn rotate(self: *Image, degrees: u16) !void;
    pub fn convert(self: *Image, target_format: PixelFormat) !void;
    pub fn adjustBrightness(self: *Image, adjustment: i16) !void;
    pub fn adjustContrast(self: *Image, factor: f32) !void;
    pub fn blur(self: *Image, radius: u32) !void;
};
```

### Enums

```zig
pub const PixelFormat = enum {
    rgb,      // 24-bit RGB
    rgba,     // 32-bit RGBA
    yuv,      // YUV color space
    hsv,      // HSV color space
    cmyk,     // CMYK color space
    grayscale // 8-bit grayscale
};

pub const ImageFormat = enum {
    png,  // Portable Network Graphics
    jpeg, // Joint Photographic Experts Group
    webp, // WebP
    avif, // AVIF
    tiff, // Tagged Image File Format
    bmp,  // Windows Bitmap
    gif,  // Graphics Interchange Format
    svg   // Scalable Vector Graphics
};
```

## 🏗️ Architecture

zpix is designed with performance and safety in mind:

- **Modular Design**: Easy to add new formats and operations
- **Zero-Copy Operations**: Where possible, avoids unnecessary data copying
- **Allocator-Aware**: All operations respect the provided allocator
- **Error Handling**: Comprehensive error types for different failure modes
- **Extensible**: Clean interfaces for adding new image formats and operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas that need help:

- Implementing remaining image formats (JPEG, WebP, AVIF, etc.)
- Adding image processing operations
- Performance optimizations
- Documentation improvements
- Test coverage

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ghostkellz/zpix.git
cd zpix

# Build and test
zig build test

# Run the example
zig build run
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for a modern, safe alternative to C image libraries
- Built with [Zig](https://ziglang.org/), a systems programming language
- Thanks to the Zig community for their excellent tooling and documentation

---

<p align="center">
  Made with ❤️ in Zig
</p>