# zpix Documentation

Welcome to the zpix documentation! This is the official documentation for zpix, a high-performance image processing library written in Zig.

## 📚 Documentation Index

- [Getting Started](getting-started.md) - Quick start guide
- [API Reference](api-reference.md) - Complete API documentation
- [Contributing](contributing.md) - How to contribute to zpix
- [Changelog](changelog.md) - Version history and changes
- [Examples](examples.md) - Code examples and tutorials
- [Architecture](architecture.md) - Internal design and architecture

## 🚀 Quick Links

- [GitHub Repository](https://github.com/ghostkellz/zpix)
- [README](../README.md) - Main project documentation
- [Issues](https://github.com/ghostkellz/zpix/issues) - Report bugs or request features

## 📖 About zpix

zpix is a modern, memory-safe image processing library that serves as a replacement for C libraries like FreeImage, DevIL, and stb_image. It provides comprehensive image loading, processing, and saving capabilities with zero-cost abstractions and compile-time safety.

### Key Features

- **Memory Safety**: Zero-copy operations where possible, controlled allocations
- **High Performance**: SIMD-ready framework, optimized algorithms
- **Type Safety**: Leverages Zig's compile-time safety features
- **Comprehensive API**: Load, process, and save images with a clean interface
- **Self-Contained**: No external dependencies, pure Zig implementation

### Supported Formats (Current)

- **BMP**: Full load/save for 24-bit RGB & grayscale
- **PNG**: Load/save for 8-bit RGB/RGBA (non-interlaced)
- **JPEG**: Baseline decoder & encoder (no progressive yet)
- **WebP**: Experimental placeholder codec for MVP validation
- **AVIF**: AV1-based decoder (decode only)
- **TIFF**: Uncompressed 8-bit RGB/RGBA/Grayscale
- **GIF**: Palette decoding (first frame) plus animation reader
- **SVG**: Rasterizes basic SVG shapes into RGB images

### Available Operations

- **Resize**: SIMD-accelerated bilinear interpolation
- **Crop**: Rectangular region extraction
- **Rotate**: 90°, 180°, 270°, and arbitrary-angle rotations
- **Flip**: Horizontal and vertical mirroring
- **Brightness**: Adjustable intensity control
- **Contrast**: Factor-based adjustment
- **Blur**: Box blur with configurable radius
- **Grayscale**: Luminance conversion from RGB
- **Color Transforms**: RGB ↔ HSV, RGB ↔ YUV conversions

### CLI & Automation

- `zpix convert` for one-off format conversions (stdin/stdout aware)
- `zpix pipeline` to chain operations (e.g. `resize`, `blur`, `format`, `save`)
- `zpix batch` to execute multiple pipelines from `.zps` scripts
- Streaming helpers for piping data through stdin/stdout without temp files
- `zig build test` runs both library and CLI regression suites

## 🤝 Contributing

We welcome contributions! Please see the [Contributing Guide](../README.md#contributing) in the main README for details on how to get involved.

## 📄 License

zpix is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.