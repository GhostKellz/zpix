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

### Supported Formats (MVP)

- **BMP**: Full load/save support for 24-bit RGB
- **PNG**: Load support for 8-bit RGB images
- **JPEG**: Detection and partial decoding framework

### Available Operations

- **Resize**: High-quality bilinear interpolation
- **Crop**: Rectangular region extraction
- **Rotate**: 90°, 180°, 270° rotations
- **Brightness**: Adjustable intensity control
- **Contrast**: Factor-based adjustment
- **Blur**: Box blur with configurable radius

## 🤝 Contributing

We welcome contributions! Please see the [Contributing Guide](../README.md#contributing) in the main README for details on how to get involved.

## 📄 License

zpix is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.