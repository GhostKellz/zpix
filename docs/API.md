# zpix API Reference

This document provides comprehensive API documentation for the zpix image processing library.

## Table of Contents

- [Core Types](#core-types)
- [Image Struct](#image-struct)
- [File I/O Operations](#file-io-operations)
- [Image Processing Operations](#image-processing-operations)
- [Utility Functions](#utility-functions)
- [Error Handling](#error-handling)
- [Memory Management](#memory-management)

## Core Types

### PixelFormat

Defines the pixel format and color space of an image.

```zig
pub const PixelFormat = enum {
    rgb,      // 24-bit RGB (3 bytes per pixel)
    rgba,     // 32-bit RGBA with alpha channel (4 bytes per pixel)
    yuv,      // YUV color space (3 bytes per pixel)
    hsv,      // HSV color space (3 bytes per pixel)
    cmyk,     // CMYK color space (4 bytes per pixel)
    grayscale // 8-bit grayscale (1 byte per pixel)
};
```

### ImageFormat

Specifies the file format for saving and loading images.

```zig
pub const ImageFormat = enum {
    png,  // Portable Network Graphics
    jpeg, // Joint Photographic Experts Group
    webp, // WebP
    avif, // AV1 Image File Format
    tiff, // Tagged Image File Format
    bmp,  // Windows Bitmap
    gif,  // Graphics Interchange Format
    svg   // Scalable Vector Graphics
};
```

## Image Struct

The `Image` struct is the core data structure representing an image in memory.

### Fields

```zig
pub const Image = struct {
    allocator: std.mem.Allocator,  // Memory allocator used for this image
    width: u32,                    // Image width in pixels
    height: u32,                   // Image height in pixels
    format: PixelFormat,           // Pixel format
    data: []u8,                    // Raw pixel data

    // ... methods
};
```

### Constructor and Destructor

#### `init`

Creates a new image with specified dimensions and pixel format.

```zig
pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: PixelFormat) !Image
```

**Parameters:**
- `allocator`: Memory allocator to use for pixel data
- `width`: Image width in pixels (1-65535)
- `height`: Image height in pixels (1-65535)
- `format`: Pixel format for the image

**Returns:** New `Image` instance

**Errors:**
- `error.InvalidDimensions`: Width or height is 0
- `error.DimensionsTooLarge`: Width or height exceeds 65535
- `error.OutOfMemory`: Insufficient memory for pixel data

**Example:**
```zig
const allocator = std.heap.page_allocator;
var image = try zpix.Image.init(allocator, 640, 480, .rgb);
defer image.deinit();
```

#### `deinit`

Releases memory allocated for the image.

```zig
pub fn deinit(self: *Image) void
```

**Note:** Always call `deinit()` when done with an image to prevent memory leaks.

## File I/O Operations

### `load`

Loads an image from a file, automatically detecting the format.

```zig
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image
```

**Parameters:**
- `allocator`: Memory allocator to use
- `path`: Absolute path to the image file

**Returns:** Loaded `Image` instance

**Supported Formats:**
- BMP (24-bit RGB)
- PNG (8-bit RGB/RGBA)
- JPEG (partial support)
- WebP (basic support)

**Errors:**
- `error.FileNotFound`: File doesn't exist
- `error.EmptyPath`: Path is empty
- `error.PathTooLong`: Path exceeds 4096 characters
- `error.UnknownFormat`: File format not recognized
- `error.InvalidImageData`: Corrupted image data

**Example:**
```zig
var image = try zpix.Image.load(allocator, "/path/to/image.png");
defer image.deinit();
```

### `save`

Saves an image to a file in the specified format.

```zig
pub fn save(self: Image, path: []const u8, format: ImageFormat) !void
```

**Parameters:**
- `path`: Output file path
- `format`: Target image format

**Supported Formats:**
- BMP (RGB only)
- PNG (RGB/RGBA with alpha support)
- WebP (basic lossless support)

**Errors:**
- `error.UnsupportedFormat`: Format not supported for saving
- `error.AccessDenied`: Cannot write to file location
- `error.OutOfMemory`: Insufficient memory for encoding

**Example:**
```zig
try image.save("/path/to/output.png", .png);
```

## Image Processing Operations

### `resize`

Resizes the image using bilinear interpolation.

```zig
pub fn resize(self: *Image, new_width: u32, new_height: u32) !void
```

**Parameters:**
- `new_width`: Target width in pixels
- `new_height`: Target height in pixels

**Note:** Modifies the image in-place. Original data is replaced.

**Example:**
```zig
try image.resize(800, 600);
```

### `crop`

Extracts a rectangular region from the image.

```zig
pub fn crop(self: *Image, x: u32, y: u32, width: u32, height: u32) !void
```

**Parameters:**
- `x`: Left edge of crop region
- `y`: Top edge of crop region
- `width`: Width of crop region
- `height`: Height of crop region

**Errors:**
- `error.InvalidCropRegion`: Crop region extends outside image bounds

### `rotate`

Rotates the image by the specified angle.

```zig
pub fn rotate(self: *Image, degrees: u16) !void
```

**Parameters:**
- `degrees`: Rotation angle (90, 180, or 270)

**Errors:**
- `error.UnsupportedRotation`: Angle not supported

### `convert`

Converts the image to a different pixel format.

```zig
pub fn convert(self: *Image, target_format: PixelFormat) !void
```

**Parameters:**
- `target_format`: Target pixel format

**Supported Conversions:**
- RGB ↔ RGBA
- RGB/RGBA → Grayscale
- RGB ↔ YUV (basic support)

### `adjustBrightness`

Adjusts the brightness of the image.

```zig
pub fn adjustBrightness(self: *Image, adjustment: i16) !void
```

**Parameters:**
- `adjustment`: Brightness change (-255 to +255)

**Note:** Values are clamped to valid range (0-255).

### `adjustContrast`

Adjusts the contrast of the image.

```zig
pub fn adjustContrast(self: *Image, factor: f32) !void
```

**Parameters:**
- `factor`: Contrast multiplier (0.0 = no contrast, 1.0 = no change, >1.0 = increased contrast)

### `blur`

Applies a blur effect to the image.

```zig
pub fn blur(self: *Image, radius: u32) !void
```

**Parameters:**
- `radius`: Blur radius in pixels

**Note:** Uses optimized separable blur algorithm. Performance scales with radius.

## Utility Functions

### `bytesPerPixel`

Returns the number of bytes per pixel for a given format.

```zig
pub fn bytesPerPixel(format: PixelFormat) u32
```

**Returns:**
- `grayscale`: 1 byte
- `rgb`, `yuv`, `hsv`: 3 bytes
- `rgba`, `cmyk`: 4 bytes

**Example:**
```zig
const bpp = zpix.bytesPerPixel(.rgba); // Returns 4
```

## Error Handling

zpix uses Zig's error handling system. Common error types include:

```zig
// Image creation errors
error.InvalidDimensions       // Width or height is 0
error.DimensionsTooLarge      // Dimensions exceed limits
error.OutOfMemory            // Insufficient memory

// File I/O errors
error.FileNotFound           // File doesn't exist
error.AccessDenied           // Permission denied
error.UnknownFormat          // Unsupported file format
error.InvalidImageData       // Corrupted image data

// Processing errors
error.UnsupportedFormat      // Operation not supported for format
error.InvalidCropRegion      // Crop region out of bounds
error.UnsupportedRotation    // Invalid rotation angle
```

Always handle errors appropriately:

```zig
const image = zpix.Image.load(allocator, "image.png") catch |err| switch (err) {
    error.FileNotFound => {
        std.debug.print("Image file not found\n", .{});
        return;
    },
    error.UnknownFormat => {
        std.debug.print("Unsupported image format\n", .{});
        return;
    },
    else => return err,
};
```

## Memory Management

zpix uses explicit memory management with allocators:

1. **Always call `deinit()`** on images when done
2. **Use appropriate allocators** for your use case
3. **Monitor memory usage** for large images

### Memory Usage Calculation

```zig
const memory_usage = width * height * bytesPerPixel(format);
```

### Best Practices

1. Use `defer image.deinit()` immediately after creation
2. Process images in-place when possible to save memory
3. Consider using arena allocators for batch processing
4. Use streaming operations for very large images

## Performance Considerations

1. **Format Choice**: Grayscale < RGB < RGBA in memory usage
2. **Operation Order**: Resize before other operations when possible
3. **Blur Radius**: Performance scales quadratically with radius
4. **Memory Layout**: Images are stored in row-major order
5. **SIMD**: Framework ready for SIMD optimizations

## Thread Safety

zpix is **not thread-safe**. Images should not be accessed from multiple threads without external synchronization. For concurrent processing, create separate image instances or use appropriate locking mechanisms.