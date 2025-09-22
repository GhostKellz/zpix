# zpix Architecture

This document describes the internal architecture and design decisions of the zpix image processing library.

## 🏗️ Overall Architecture

zpix follows a modular, layered architecture designed for performance, safety, and extensibility:

```
┌─────────────────────────────────────┐
│             CLI Layer               │
│         (main.zig)                  │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│          Core Library               │
│         (root.zig)                  │
│                                     │
│  ┌─────────┬─────────┬─────────┐    │
│  │  Image  │ Format  │ Process │    │
│  │ Struct  │ Parsers │ Engine  │    │
│  └─────────┴─────────┴─────────┘    │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│         Memory Management           │
│      (Zig Standard Library)         │
└─────────────────────────────────────┘
```

## 📦 Core Components

### Image Structure

The `Image` struct is the central data structure:

```zig
pub const Image = struct {
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    format: PixelFormat,
    data: []u8,

    // Methods for loading, saving, processing
};
```

**Design Decisions:**
- **Owned data**: The `Image` struct owns its pixel data
- **Explicit allocator**: Memory management is explicit and controlled
- **Format metadata**: Pixel format is stored to enable format-aware operations
- **Contiguous storage**: Pixel data is stored in a single contiguous buffer

### Pixel Formats

zpix supports multiple pixel formats with different memory layouts:

```zig
pub const PixelFormat = enum {
    rgb,      // 24-bit RGB (R,G,B bytes)
    rgba,     // 32-bit RGBA (R,G,B,A bytes)
    yuv,      // YUV color space
    hsv,      // HSV color space
    cmyk,     // CMYK color space
    grayscale // 8-bit grayscale
};
```

**Memory Layout:**
- **RGB**: `[R,G,B,R,G,B,...]` (row-major, top-to-bottom)
- **RGBA**: `[R,G,B,A,R,G,B,A,...]`
- **Grayscale**: `[Y,Y,Y,...]`

### Format Parsers

Each image format has a dedicated parser module:

```
Format Parsers
├── BMP Parser
│   ├── Header parsing
│   ├── Pixel data extraction
│   └── Color palette handling
├── PNG Parser
│   ├── Chunk parsing
│   ├── ZLIB decompression
│   └── Filter application
└── JPEG Parser (Partial)
    ├── Marker parsing
    ├── Huffman table building
    └── Entropy decoding
```

**Parser Design:**
- **Streaming**: Parsers process data incrementally
- **Error handling**: Comprehensive error reporting
- **Memory efficient**: Minimal allocations during parsing

## ⚙️ Processing Engine

The image processing operations are implemented as methods on the `Image` struct:

### Operation Categories

1. **Geometric Transformations**
   - Resize (bilinear interpolation)
   - Crop (region extraction)
   - Rotate (90°, 180°, 270°)

2. **Color Adjustments**
   - Brightness (additive)
   - Contrast (multiplicative)
   - Color space conversion

3. **Filters**
   - Blur (box filter)
   - Future: Sharpen, edge detection, etc.

### Processing Pipeline

```zig
// Example pipeline
try image.resize(800, 600);
try image.brightness(10);
try image.convert(.grayscale);
```

**Pipeline Characteristics:**
- **In-place operations**: Most operations modify the image directly
- **Memory efficient**: Reuses existing buffers where possible
- **Composable**: Operations can be chained arbitrarily

## 🔧 Memory Management

### Allocation Strategy

- **Explicit ownership**: Each `Image` owns its data buffer
- **Allocator parameter**: Memory management is controlled by the caller
- **Arena allocation**: Recommended for temporary operations
- **No hidden allocations**: All allocations are explicit

### Memory Layout

```
Image Memory Layout
┌─────────────────┐
│ Image struct    │
│ - allocator     │
│ - width         │
│ - height        │
│ - format        │
│ - data ptr ─────┼──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │
│ Pixel Data      │◄─┘
│ (contiguous)    │
│ R G B R G B ... │
└─────────────────┘
```

## 🚀 Performance Considerations

### SIMD Readiness

- **Data layout**: Contiguous pixel data enables SIMD operations
- **Algorithm design**: Operations are designed for vectorization
- **Future optimization**: Framework ready for SIMD implementation

### Zero-Copy Operations

- **Direct access**: Pixel data is accessible without copying
- **In-place processing**: Operations modify data in-place where possible
- **Memory mapping**: Future support for memory-mapped files

### Algorithm Selection

- **Resize**: Bilinear interpolation (balance of quality/speed)
- **Blur**: Box filter (simple, fast, good for many use cases)
- **Color conversion**: Direct formulas (no lookup tables)

## 🛡️ Safety Features

### Compile-Time Safety

- **Type safety**: Zig's type system prevents many errors
- **Bounds checking**: Array access is bounds-checked
- **Error handling**: All operations return errors or success

### Runtime Safety

- **Memory safety**: No buffer overflows or use-after-free
- **Resource management**: RAII-style cleanup with `deinit()`
- **Validation**: Input validation for all operations

## 🔌 Extensibility

### Format Support

Adding new formats follows this pattern:

```zig
// 1. Add to ImageFormat enum
pub const ImageFormat = enum {
    // ... existing formats
    new_format,
};

// 2. Implement load function
pub fn loadNewFormat(allocator: std.mem.Allocator, reader: anytype) !Image {
    // Parse format-specific headers
    // Decode pixel data
    // Return Image struct
}

// 3. Add to main load function
pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image {
    // Detect format
    // Call appropriate loader
}
```

### Operation Extensions

Adding new operations:

```zig
// 1. Add method to Image struct
pub fn newOperation(self: *Image, param: ParamType) !void {
    // Implement operation
    // Modify self.data in-place
}

// 2. Handle different pixel formats
switch (self.format) {
    .rgb => self.applyToRgb(param),
    .grayscale => self.applyToGrayscale(param),
    // ...
}
```

## 🧪 Testing Strategy

### Unit Tests

- **Per operation**: Each operation has dedicated tests
- **Format support**: Each format has load/save tests
- **Edge cases**: Invalid inputs, boundary conditions
- **Memory leaks**: Tests use leak detection

### Integration Tests

- **CLI testing**: End-to-end command line testing
- **Pipeline testing**: Multi-operation pipelines
- **Format conversion**: Cross-format compatibility

## 📊 Performance Metrics

### Benchmarks (Planned)

- **Load/save speed**: Time to process various image sizes
- **Memory usage**: Peak memory during operations
- **Throughput**: Images processed per second
- **SIMD acceleration**: Performance improvement with SIMD

### Optimization Opportunities

- **SIMD implementation**: Vectorized pixel operations
- **Parallel processing**: Multi-threaded operations
- **GPU acceleration**: OpenCL/CUDA backends
- **Memory pooling**: Reuse allocation patterns

## 🔮 Future Architecture

### Plugin System

```
Plugin Architecture
┌─────────────────────────────────────┐
│            Core zpix                │
└─────────────────────────────────────┘
               │
       ┌───────┼───────┐
       │       │       │
┌──────▼──┐ ┌──▼──┐ ┌──▼────┐
│Format   │ │Filter│ │Output │
│Plugins  │ │Plugins│ │Plugins│
└─────────┘ └──────┘ └───────┘
```

### Distributed Processing

- **Cluster support**: Distributed image processing
- **Streaming**: Process large images without full loading
- **Caching**: Intelligent memory/disk caching

### Language Bindings

- **C API**: Stable C interface for other languages
- **Python bindings**: Native Python integration
- **WASM**: Browser-based image processing

This architecture provides a solid foundation for a high-performance, safe, and extensible image processing library.