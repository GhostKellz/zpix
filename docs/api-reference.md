# zpix API Reference

This document provides a complete reference for the zpix API.

## 📦 Core Types

### Image

The main image structure containing pixel data and metadata.

```zig
pub const Image = struct {
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    format: PixelFormat,
    data: []u8,
    // ... methods
};
```

### PixelFormat

Enumeration of supported pixel formats.

```zig
pub const PixelFormat = enum {
    rgb,      // 24-bit RGB (3 bytes per pixel)
    rgba,     // 32-bit RGBA (4 bytes per pixel)
    yuv,      // YUV color space (3 bytes per pixel)
    hsv,      // HSV color space (3 bytes per pixel)
    cmyk,     // CMYK color space (4 bytes per pixel)
    grayscale // 8-bit grayscale (1 byte per pixel)
};
```

### ImageFormat

Enumeration of supported image file formats.

```zig
pub const ImageFormat = enum {
    png,  // Portable Network Graphics
    jpeg, // Joint Photographic Experts Group
    webp, // WebP format
    avif, // AVIF format
    tiff, // Tagged Image File Format
    bmp,  // Windows Bitmap
    gif,  // Graphics Interchange Format
    svg   // Scalable Vector Graphics
};
```

## 🏗️ Constructor Functions

### Image.init

Creates a new image with the specified dimensions and pixel format.

```zig
pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: PixelFormat) !Image
```

**Parameters:**
- `allocator`: Memory allocator to use
- `width`: Image width in pixels
- `height`: Image height in pixels
- `format`: Pixel format for the image

**Returns:** New `Image` instance

**Example:**
```zig
var image = try zpix.Image.init(allocator, 800, 600, .rgb);
defer image.deinit();
```

### Image.load

Loads an image from a file.

```zig
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image
```

**Parameters:**
- `allocator`: Memory allocator to use
- `path`: Path to the image file

**Returns:** Loaded `Image` instance

**Supported Formats:** BMP, PNG, JPEG (baseline), WebP (experimental), AVIF (decode-only), TIFF (uncompressed 8-bit), GIF (palette), SVG (rasterized)

**Example:**
```zig
var image = try zpix.Image.load(allocator, "photo.png");
defer image.deinit();
```

## 💾 Saving Functions

### Image.save

Saves an image to a file.

```zig
pub fn save(self: Image, path: []const u8, format: ImageFormat) !void
```

**Parameters:**
- `self`: The image to save
- `path`: Output file path
- `format`: Output format

**Supported Formats:** BMP, PNG, JPEG (baseline encoder), WebP (experimental MVP)

**Example:**
```zig
try image.save("output.bmp", .bmp);
```

## 🧹 Memory Management

### Image.deinit

Frees all memory associated with the image.

```zig
pub fn deinit(self: *Image) void
```

**Example:**
```zig
var image = try zpix.Image.load(allocator, "input.png");
defer image.deinit(); // Always call this!
```

## 🎨 Image Processing Functions

### Image.resize

Resizes the image to new dimensions using bilinear interpolation.

```zig
pub fn resize(self: *Image, new_width: u32, new_height: u32) !void
```

**Parameters:**
- `new_width`: New width in pixels
- `new_height`: New height in pixels

**Note:** This function modifies the image in-place.

### Image.crop

Crops a rectangular region from the image.

```zig
pub fn crop(self: *Image, x: u32, y: u32, width: u32, height: u32) !void
```

**Parameters:**
- `x`: X-coordinate of crop region
- `y`: Y-coordinate of crop region
- `width`: Width of crop region
- `height`: Height of crop region

### Image.rotateArbitrary

Rotates the image by any angle in degrees (specializations exist for 90°, 180°, 270°).

```zig
pub fn rotateArbitrary(self: *Image, degrees: f32) !void
```

**Parameters:**
- `degrees`: Rotation angle in degrees (positive = clockwise)

### Image.convertColorSpaceVectorized

Converts the image to a different pixel format using SIMD-backed transforms.

```zig
pub fn convertColorSpaceVectorized(self: *Image, target_format: PixelFormat) !void
```

**Parameters:**
- `target_format`: Target pixel format (e.g. `.hsv`, `.yuv`, `.rgb`)

**Supported Conversions:**
- RGB ↔ HSV
- RGB ↔ YUV
- RGBA ↔ RGB (alpha preserved)
- RGB ↔ Grayscale (via `convertToGrayscale` convenience helper)

### Image.convertToGrayscale

Converts the image to grayscale using luminance weighting.

```zig
pub fn convertToGrayscale(self: *Image) !void
```

### Image.adjustBrightness

Adjusts the brightness of the image.

```zig
pub fn adjustBrightness(self: *Image, adjustment: i16) !void
```

**Parameters:**
- `adjustment`: Brightness adjustment (-255 to +255)

### Image.adjustContrast

Adjusts the contrast of the image.

```zig
pub fn adjustContrast(self: *Image, factor: f32) !void
```

**Parameters:**
- `factor`: Contrast factor (0.0 = no contrast, 1.0 = normal, >1.0 = more contrast)

### Image.blur

Applies a box blur to the image.

```zig
pub fn blur(self: *Image, radius: u8) !void
```

**Parameters:**
- `radius`: Blur radius (1-255)

### Image.flipHorizontal / Image.flipVertical

Mirrors the image across the X or Y axis.

```zig
pub fn flipHorizontal(self: *Image) !void
pub fn flipVertical(self: *Image) !void
```

## 🔧 Utility Functions

### bytesPerPixel

Returns the number of bytes per pixel for a given format.

```zig
pub fn bytesPerPixel(format: PixelFormat) u32
```

**Returns:** Bytes per pixel (1-4)

## 📊 Error Types

zpix uses Zig's error union system. Common errors include:

- `error.FileNotFound` - Input file doesn't exist
- `error.UnknownFormat` - Unsupported image format
- `error.InvalidData` - Corrupted or invalid image data
- `error.OutOfMemory` - Memory allocation failed
- `error.UnsupportedFormat` - Format not supported for operation
- `error.InvalidDimensions` - Invalid image dimensions

## 🔍 Example Usage

```zig
const std = @import("std");
const zpix = @import("zpix");

pub fn processImage(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    // Load image
    var image = try zpix.Image.load(allocator, input_path);
    defer image.deinit();

    // Apply processing
    try image.resize(800, 600);
    try image.adjustBrightness(20);
    try image.convertToGrayscale();

    // Save result
    try image.save(output_path, .png);
}
```

## 📝 Notes

- All image processing functions modify the image in-place
- Always call `deinit()` when done with an image to free memory
- Image data is stored in row-major order (top to bottom, left to right)
- RGB data is stored as R, G, B byte sequence per pixel
- All coordinates use (0,0) as top-left origin