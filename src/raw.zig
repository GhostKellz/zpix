//! RAW image format support for zpix
//! Provides detection, basic parsing, and simple demosaicing for common RAW formats

const std = @import("std");

/// Common RAW file formats
pub const RawFormat = enum {
    cr2,   // Canon CR2
    nef,   // Nikon NEF
    arw,   // Sony ARW
    dng,   // Adobe DNG
    raf,   // Fujifilm RAF
    orf,   // Olympus ORF
    rw2,   // Panasonic RW2
    unknown,
};

/// RAW file metadata
pub const RawMetadata = struct {
    format: RawFormat,
    width: u32,
    height: u32,
    bits_per_sample: u16,
    iso: u32,
    exposure_time: f32,
    f_number: f32,
    focal_length: f32,
    white_balance: [3]f32,
    color_matrix: [9]f32,
    camera_make: [32]u8,
    camera_model: [32]u8,
    lens_model: [64]u8,
    timestamp: u64,
    bayer_pattern: BayerPattern,

    pub fn init() RawMetadata {
        return RawMetadata{
            .format = .unknown,
            .width = 0,
            .height = 0,
            .bits_per_sample = 12,
            .iso = 100,
            .exposure_time = 1.0 / 125.0,
            .f_number = 2.8,
            .focal_length = 50.0,
            .white_balance = [3]f32{ 1.0, 1.0, 1.0 },
            .color_matrix = [9]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 },
            .camera_make = std.mem.zeroes([32]u8),
            .camera_model = std.mem.zeroes([32]u8),
            .lens_model = std.mem.zeroes([64]u8),
            .timestamp = 0,
            .bayer_pattern = .rggb,
        };
    }
};

/// Bayer color filter array patterns
pub const BayerPattern = enum {
    rggb,  // Red-Green-Green-Blue
    bggr,  // Blue-Green-Green-Red
    grbg,  // Green-Red-Blue-Green
    gbrg,  // Green-Blue-Red-Green
};

/// Detect RAW file format from file signature
pub fn detectRawFormat(file: std.fs.File) !RawFormat {
    try file.seekTo(0);

    var header: [16]u8 = undefined;
    const bytes_read = try file.read(&header);
    if (bytes_read < 16) return .unknown;

    // Check for various RAW format signatures

    // Canon CR2 (TIFF-based with CR2 marker)
    if (std.mem.eql(u8, header[0..2], "II") or std.mem.eql(u8, header[0..2], "MM")) {
        const magic = if (header[0] == 'I')
            std.mem.readInt(u16, header[2..4], .little)
        else
            std.mem.readInt(u16, header[2..4], .big);

        if (magic == 42) {
            // Look for CR2 signature at offset 8
            if (std.mem.eql(u8, header[8..11], "CR\x02")) {
                return .cr2;
            }
        }
    }

    // Nikon NEF (TIFF-based)
    if (std.mem.startsWith(u8, &header, "Nikon") or
        (header[0] == 'I' and header[1] == 'I' and header[2] == 42)) {
        // Additional NEF-specific checks
        try file.seekTo(0x1c);
        var nef_check: [4]u8 = undefined;
        _ = file.read(&nef_check) catch return .nef;
        if (std.mem.eql(u8, &nef_check, "NEF\x00")) {
            return .nef;
        }
        return .nef; // Assume NEF if TIFF-like and not CR2
    }

    // Sony ARW (TIFF-based)
    if (std.mem.startsWith(u8, &header, "SONY")) {
        return .arw;
    }

    // Adobe DNG (TIFF-based with DNG marker)
    if ((std.mem.eql(u8, header[0..2], "II") or std.mem.eql(u8, header[0..2], "MM")) and
        std.mem.indexOf(u8, &header, "DNG") != null) {
        return .dng;
    }

    // Fujifilm RAF
    if (std.mem.startsWith(u8, &header, "FUJIFILM")) {
        return .raf;
    }

    // Olympus ORF
    if (std.mem.startsWith(u8, &header, "OLYMP")) {
        return .orf;
    }

    // Panasonic RW2 (TIFF-based)
    if (std.mem.startsWith(u8, &header, "PANA")) {
        return .rw2;
    }

    return .unknown;
}

/// Extract basic metadata from RAW file
pub fn extractRawMetadata(_: std.mem.Allocator, file: std.fs.File) !RawMetadata {
    const format = try detectRawFormat(file);
    var metadata = RawMetadata.init();
    metadata.format = format;

    switch (format) {
        .cr2, .nef, .dng => try parseTiffBasedRaw(file, &metadata),
        .arw => try parseSonyArw(file, &metadata),
        .raf => try parseFujiRaf(file, &metadata),
        .orf => try parseOlympusOrf(file, &metadata),
        .rw2 => try parsePanasonicRw2(file, &metadata),
        .unknown => return error.UnsupportedRawFormat,
    }

    return metadata;
}

/// Simple bilinear demosaicing for Bayer pattern data
pub fn demosaicBilinear(allocator: std.mem.Allocator, bayer_data: []const u16, width: u32, height: u32, pattern: BayerPattern) ![]u8 {
    const rgb_data = try allocator.alloc(u8, width * height * 3);

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            const rgb_idx = idx * 3;

            // Get current pixel value (normalize from 16-bit to 8-bit)
            const current = @as(u8, @intCast(@min(255, bayer_data[idx] >> 8)));

            // Determine color channel based on position and Bayer pattern
            const color = getBayerColor(x, y, pattern);

            // Initialize RGB values
            var r: u8 = 0;
            var g: u8 = 0;
            var b: u8 = 0;

            switch (color) {
                .red => {
                    r = current;
                    g = interpolateGreen(bayer_data, x, y, width, height, pattern);
                    b = interpolateBlue(bayer_data, x, y, width, height, pattern);
                },
                .green => {
                    r = interpolateRed(bayer_data, x, y, width, height, pattern);
                    g = current;
                    b = interpolateBlue(bayer_data, x, y, width, height, pattern);
                },
                .blue => {
                    r = interpolateRed(bayer_data, x, y, width, height, pattern);
                    g = interpolateGreen(bayer_data, x, y, width, height, pattern);
                    b = current;
                },
            }

            rgb_data[rgb_idx] = r;
            rgb_data[rgb_idx + 1] = g;
            rgb_data[rgb_idx + 2] = b;
        }
    }

    return rgb_data;
}

const BayerColor = enum { red, green, blue };

fn getBayerColor(x: usize, y: usize, pattern: BayerPattern) BayerColor {
    const even_x = (x % 2) == 0;
    const even_y = (y % 2) == 0;

    return switch (pattern) {
        .rggb => if (even_y) (if (even_x) .red else .green) else (if (even_x) .green else .blue),
        .bggr => if (even_y) (if (even_x) .blue else .green) else (if (even_x) .green else .red),
        .grbg => if (even_y) (if (even_x) .green else .red) else (if (even_x) .blue else .green),
        .gbrg => if (even_y) (if (even_x) .green else .blue) else (if (even_x) .red else .green),
    };
}

fn interpolateRed(bayer_data: []const u16, x: usize, y: usize, width: u32, height: u32, pattern: BayerPattern) u8 {
    var sum: u32 = 0;
    var count: u32 = 0;

    // Sample neighboring red pixels
    const directions = [_][2]i32{ .{-1, -1}, .{-1, 1}, .{1, -1}, .{1, 1}, .{-2, 0}, .{2, 0}, .{0, -2}, .{0, 2} };

    for (directions) |dir| {
        const nx = @as(i32, @intCast(x)) + dir[0];
        const ny = @as(i32, @intCast(y)) + dir[1];

        if (nx >= 0 and nx < width and ny >= 0 and ny < height) {
            const idx = @as(usize, @intCast(ny)) * width + @as(usize, @intCast(nx));
            if (getBayerColor(@intCast(nx), @intCast(ny), pattern) == .red) {
                sum += @as(u32, bayer_data[idx] >> 8);
                count += 1;
            }
        }
    }

    if (count > 0) {
        return @intCast(@min(255, sum / count));
    }
    return 0;
}

fn interpolateGreen(bayer_data: []const u16, x: usize, y: usize, width: u32, height: u32, pattern: BayerPattern) u8 {
    var sum: u32 = 0;
    var count: u32 = 0;

    // Sample neighboring green pixels
    const directions = [_][2]i32{ .{-1, 0}, .{1, 0}, .{0, -1}, .{0, 1} };

    for (directions) |dir| {
        const nx = @as(i32, @intCast(x)) + dir[0];
        const ny = @as(i32, @intCast(y)) + dir[1];

        if (nx >= 0 and nx < width and ny >= 0 and ny < height) {
            const idx = @as(usize, @intCast(ny)) * width + @as(usize, @intCast(nx));
            if (getBayerColor(@intCast(nx), @intCast(ny), pattern) == .green) {
                sum += @as(u32, bayer_data[idx] >> 8);
                count += 1;
            }
        }
    }

    if (count > 0) {
        return @intCast(@min(255, sum / count));
    }
    return 0;
}

fn interpolateBlue(bayer_data: []const u16, x: usize, y: usize, width: u32, height: u32, pattern: BayerPattern) u8 {
    var sum: u32 = 0;
    var count: u32 = 0;

    // Sample neighboring blue pixels
    const directions = [_][2]i32{ .{-1, -1}, .{-1, 1}, .{1, -1}, .{1, 1}, .{-2, 0}, .{2, 0}, .{0, -2}, .{0, 2} };

    for (directions) |dir| {
        const nx = @as(i32, @intCast(x)) + dir[0];
        const ny = @as(i32, @intCast(y)) + dir[1];

        if (nx >= 0 and nx < width and ny >= 0 and ny < height) {
            const idx = @as(usize, @intCast(ny)) * width + @as(usize, @intCast(nx));
            if (getBayerColor(@intCast(nx), @intCast(ny), pattern) == .blue) {
                sum += @as(u32, bayer_data[idx] >> 8);
                count += 1;
            }
        }
    }

    if (count > 0) {
        return @intCast(@min(255, sum / count));
    }
    return 0;
}

// Format-specific parsing functions
fn parseTiffBasedRaw(file: std.fs.File, metadata: *RawMetadata) !void {
    try file.seekTo(0);

    var header: [8]u8 = undefined;
    _ = try file.read(&header);

    const is_little_endian = std.mem.eql(u8, header[0..2], "II");
    const endian: std.builtin.Endian = if (is_little_endian) .little else .big;

    const ifd_offset = if (is_little_endian)
        std.mem.readInt(u32, header[4..8], .little)
    else
        std.mem.readInt(u32, header[4..8], .big);

    try file.seekTo(ifd_offset);

    var count_buf: [2]u8 = undefined;
    _ = try file.read(&count_buf);
    const num_entries = std.mem.readInt(u16, &count_buf, endian);

    // Parse TIFF tags for metadata
    for (0..num_entries) |_| {
        var entry: [12]u8 = undefined;
        _ = try file.read(&entry);

        const tag = std.mem.readInt(u16, entry[0..2], endian);
        const value = std.mem.readInt(u32, entry[8..12], endian);

        switch (tag) {
            256 => metadata.width = value,    // ImageWidth
            257 => metadata.height = value,   // ImageLength
            258 => metadata.bits_per_sample = @intCast(value & 0xFFFF), // BitsPerSample
            else => {},
        }
    }
}

fn parseSonyArw(file: std.fs.File, metadata: *RawMetadata) !void {
    _ = file;
    // Sony ARW specific parsing
    metadata.camera_make = "Sony".*;
    metadata.bayer_pattern = .rggb;
}

fn parseFujiRaf(file: std.fs.File, metadata: *RawMetadata) !void {
    _ = file;
    // Fujifilm RAF specific parsing
    metadata.camera_make = "Fujifilm".*;
    metadata.bayer_pattern = .gbrg; // Fuji X-Trans is more complex, but simplified here
}

fn parseOlympusOrf(file: std.fs.File, metadata: *RawMetadata) !void {
    _ = file;
    // Olympus ORF specific parsing
    metadata.camera_make = "Olympus".*;
    metadata.bayer_pattern = .grbg;
}

fn parsePanasonicRw2(file: std.fs.File, metadata: *RawMetadata) !void {
    _ = file;
    // Panasonic RW2 specific parsing
    metadata.camera_make = "Panasonic".*;
    metadata.bayer_pattern = .bggr;
}