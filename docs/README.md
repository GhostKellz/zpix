# zpix Documentation

Welcome to the zpix documentation! This directory contains comprehensive documentation for the zpix image processing library.

## Documentation Index

### Core Documentation
- **[API Reference](API.md)** - Complete API documentation with examples
- **[Main README](../README.md)** - Project overview and quick start guide

### Examples
- **[Basic Usage](../examples/basic_usage.zig)** - Fundamental operations and image creation
- **[Format Conversion](../examples/format_conversion.zig)** - Converting between different image formats
- **[Performance Testing](../examples/performance_test.zig)** - Performance benchmarking and optimization

## Getting Started

1. **Installation**: Follow the [installation guide](../README.md#installation) in the main README
2. **First Steps**: Try the [basic usage example](../examples/basic_usage.zig)
3. **API Reference**: Check the [API documentation](API.md) for detailed method information
4. **Performance**: Run the [performance tests](../examples/performance_test.zig) to understand characteristics

## Quick Reference

### Core Types
```zig
const zpix = @import("zpix");

// Pixel formats
.rgb, .rgba, .grayscale, .yuv, .hsv, .cmyk

// Image formats
.bmp, .png, .jpeg, .webp, .avif, .tiff, .gif, .svg
```

### Basic Operations
```zig
// Create an image
var image = try zpix.Image.init(allocator, 640, 480, .rgb);
defer image.deinit();

// Load from file
var loaded = try zpix.Image.load(allocator, "input.png");
defer loaded.deinit();

// Process the image
try image.resize(800, 600);
try image.adjustBrightness(20);
try image.blur(2);

// Save to file
try image.save("output.bmp", .bmp);
```

### CLI Usage
```bash
# Convert formats
zig build run -- convert input.png output.jpeg

# Run a multi-step pipeline
zig build run -- pipeline photo.png resize:1024x768 blur:2 format:png save:processed.png

# Execute scripted pipelines
zig build run -- batch scripts/jobs.zps

# Tests & benchmarks
zig build test
zig build run -- benchmark
```

## Current Status (v0.1.0 Beta)

### ✅ Completed Features
- **Core Image Operations**: Create, load, save, resize, rotate (arbitrary), crop, flip
- **Format Support**: BMP & PNG (load/save), JPEG baseline encoder/decoder, TIFF (uncompressed), GIF (palette), AVIF decoder, SVG rasterizer, experimental WebP codec
- **Processing Pipeline**: Brightness, contrast, blur, grayscale, color-space conversions
- **CLI Tooling**: `convert`, `pipeline`, `batch`, `test`, and `benchmark` commands with streaming I/O support
- **Performance & Memory**: SIMD-enabled resize/blur, controlled allocations, benchmark suite
- **Documentation & Tests**: Comprehensive docs, examples, and automated tests via `zig build test`

### 🔄 In Development
- Production-ready WebP codec (full lossy/lossless)
- Multi-threaded batch processing with directory walkers & progress reporting
- Metadata ingestion (EXIF, XMP, ICC profiles)
- Vulkan compute backend and extended GPU acceleration

### 📋 Planned Features
- Advanced filters and tone-mapping workflows
- Plugin architecture for custom formats/operations
- Language bindings (C API, Python, WASM, Rust)
- AI-powered enhancements (super-resolution, content-aware scaling)

## Contributing

We welcome contributions! Areas where help is needed:

1. **Format Implementation**: Adding support for new image formats
2. **Performance**: SIMD optimizations and performance improvements
3. **Documentation**: More examples and tutorials
4. **Testing**: Test cases and edge case handling
5. **Features**: New image processing operations

See the main [README](../README.md#contributing) for development setup instructions.

## Support

- **Issues**: Report bugs at the project repository
- **Examples**: Check the `examples/` directory for usage patterns
- **API Questions**: Refer to the [API documentation](API.md)
- **Performance**: Use the built-in benchmark tools to test on your system

## License

zpix is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

---

*Documentation last updated: December 2024*
*Library version: 0.1.0-beta*