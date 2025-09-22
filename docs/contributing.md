# Contributing to zpix

Thank you for your interest in contributing to zpix! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/zpix.git
   cd zpix
   ```

2. **Install Zig:**
   - Download from [ziglang.org](https://ziglang.org/download/)
   - Use version 0.16.0-dev or later

3. **Build the project:**
   ```bash
   zig build
   ```

4. **Run tests:**
   ```bash
   zig build test
   ```

## 📋 Development Workflow

### 1. Choose an Issue
- Check the [TODO.md](TODO.md) for planned features
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add JPEG support for baseline DCT"
```

Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or fixing tests
- `chore:` - Maintenance tasks

### 5. Push and Create PR
```bash
git push origin your-branch-name
```
Then create a pull request on GitHub.

## 🏗️ Project Structure

```
zpix/
├── src/
│   ├── main.zig          # CLI interface
│   └── root.zig          # Core library
├── docs/                 # Documentation
├── assets/               # Icons and assets
├── build.zig             # Build configuration
├── build.zig.zon         # Dependencies
└── README.md             # Project documentation
```

### Key Files
- `src/root.zig` - Main library implementation
- `src/main.zig` - Command-line interface
- `build.zig` - Build system configuration

## 💻 Coding Guidelines

### Zig Style
- Use 4 spaces for indentation
- Follow [Zig naming conventions](https://ziglang.org/documentation/master/#Style-Guide)
- Use `snake_case` for functions and variables
- Use `PascalCase` for types and structs
- Use `SCREAMING_SNAKE_CASE` for constants

### Code Quality
- Write clear, readable code with comments
- Use meaningful variable and function names
- Handle errors properly with `try` or error unions
- Avoid unnecessary allocations
- Prefer compile-time operations when possible

### Memory Management
- Use the provided allocator parameter
- Always free allocated memory
- Consider using arena allocators for temporary allocations
- Document ownership semantics

## 🧪 Testing

### Unit Tests
Add tests in the same file as the code being tested:

```zig
test "image resize" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var image = try Image.init(allocator, 100, 100, .rgb);
    defer image.deinit();

    try image.resize(50, 50);
    try std.testing.expectEqual(@as(u32, 50), image.width);
    try std.testing.expectEqual(@as(u32, 50), image.height);
}
```

### Running Tests
```bash
# Run all tests
zig build test

# Run specific test
zig test src/root.zig --test-filter "image resize"
```

## 📚 Documentation

### Code Documentation
- Use `///` for function documentation
- Document parameters and return values
- Explain complex algorithms
- Include usage examples

```zig
/// Resizes the image to new dimensions using bilinear interpolation.
/// This function modifies the image in-place.
/// - new_width: New width in pixels
/// - new_height: New height in pixels
pub fn resize(self: *Image, new_width: u32, new_height: u32) !void {
    // Implementation...
}
```

### API Documentation
- Update `docs/api-reference.md` for new public APIs
- Add examples to `docs/getting-started.md`
- Update README.md for major features

## 🔧 Adding New Features

### Image Formats
1. Add the format to `ImageFormat` enum
2. Implement load/save functions in `root.zig`
3. Add format detection logic
4. Update tests and documentation

### Image Operations
1. Add the function to `Image` struct
2. Implement the algorithm
3. Handle different pixel formats
4. Add comprehensive tests

## 🐛 Debugging

### Common Issues
- **Memory leaks:** Use `std.testing.allocator` in tests to detect leaks
- **Build errors:** Check Zig version compatibility
- **Runtime crashes:** Use `zig build --debug` for debug builds

### Tools
- `zig fmt` - Format code
- `zig build --debug` - Build with debug info
- `valgrind` - Memory debugging (on Linux)

## 📞 Getting Help

- **Issues:** Use GitHub issues for bugs and feature requests
- **Discussions:** Use GitHub discussions for questions
- **Discord:** Join the Zig community on Discord

## 📜 License

By contributing to zpix, you agree that your contributions will be licensed under the same license as the project (MIT).

## 🙏 Recognition

Contributors are recognized in the README.md file. Thank you for helping make zpix better!