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

**Supported Formats:** BMP, PNG, JPEG (partial)

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

**Supported Formats:** BMP

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

### Image.rotate

Rotates the image by 90, 180, or 270 degrees.

```zig
pub fn rotate(self: *Image, degrees: u16) !void
```

**Parameters:**
- `degrees`: Rotation angle (90, 180, or 270)

### Image.convert

Converts the image to a different pixel format.

```zig
pub fn convert(self: *Image, target_format: PixelFormat) !void
```

**Parameters:**
- `target_format`: Target pixel format

**Supported Conversions:**
- RGB ↔ Grayscale
- RGBA → RGB
- RGB → YUV

### Image.brightness

Adjusts the brightness of the image.

```zig
pub fn brightness(self: *Image, adjustment: i16) !void
```

**Parameters:**
- `adjustment`: Brightness adjustment (-255 to +255)

### Image.contrast

Adjusts the contrast of the image.

```zig
pub fn contrast(self: *Image, factor: f32) !void
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
    try image.brightness(20);
    try image.convert(.grayscale);

    // Save result
    try image.save(output_path, .bmp);
}
```

## 📝 Notes

- All image processing functions modify the image in-place
- Always call `deinit()` when done with an image to free memory
- Image data is stored in row-major order (top to bottom, left to right)
- RGB data is stored as R, G, B byte sequence per pixel
- All coordinates use (0,0) as top-left origin