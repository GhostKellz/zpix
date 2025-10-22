# Changelog

All notable changes to zpix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CLI `pipeline` command for multi-step image processing
- CLI `batch` runner for scripted pipelines (`.zps` files)
- Streaming stdin/stdout helpers for convert/pipeline workflows
- Integration tests covering CLI pipelines and format validation
- Initial project structure and build system
- BMP image format support (load/save)
- PNG image format support (load/save)
- JPEG image format support (baseline encoder/decoder)
- Basic image processing operations:
  - Resize with bilinear interpolation
  - Crop
  - Rotate (90°, 180°, 270°)
  - Brightness adjustment
  - Contrast adjustment
  - Box blur
- Multiple pixel formats: RGB, RGBA, Grayscale, YUV, HSV, CMYK
- Command-line interface for basic operations
- Comprehensive documentation
- Unit test framework

### In Progress
- Enhanced WebP codec (full decode/encode)
- TIFF multi-page and compression support
- Metadata (EXIF/XMP/ICC) ingestion
- Multi-threaded batch processing and progress reporting
- Vulkan compute backend

### Planned
- GPU acceleration support
- Advanced batch processing (multi-threading, directory walkers)
- Plugin system for custom operations
- GUI interface
- WASM compilation target

## [0.1.0] - 2024-01-XX

### Added
- Initial release
- BMP format support
- Basic image operations
- CLI interface
- Documentation

---

## Version History

### Development Roadmap

#### Phase 1: Core Foundation (Current)
- ✅ BMP format support
- ✅ PNG format support
- ✅ Basic image operations
- ✅ Documentation
- 🔄 JPEG format support

#### Phase 2: Extended Formats
- WebP support
- AVIF support
- TIFF support
- GIF support (with animation)
- SVG support

#### Phase 3: Advanced Features
- GPU acceleration
- SIMD optimizations
- Plugin architecture
- GUI application
- Batch processing

#### Phase 4: Ecosystem
- WASM support
- Language bindings (Python, Rust, etc.)
- Cloud integration
- Performance monitoring

## Contributing to Changes

When making changes, please:
1. Update this changelog in the same PR
2. Add entries under `[Unreleased]` section
3. Categorize changes as:
   - `Added` for new features
   - `Changed` for changes in existing functionality
   - `Deprecated` for soon-to-be removed features
   - `Removed` for now removed features
   - `Fixed` for any bug fixes
   - `Security` for vulnerability fixes

## Release Process

1. Update version in `build.zig.zon`
2. Move unreleased changes to new version section
3. Create git tag
4. Publish release on GitHub
5. Update package registries if applicable

---

*For the latest updates, see the [TODO.md](TODO.md) file.*