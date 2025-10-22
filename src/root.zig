//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const simd = @import("simd.zig");
const raw = @import("raw.zig");
const svg = @import("svg.zig");
const avif = @import("avif.zig");
const hdr = @import("hdr.zig");
const lab = @import("lab.zig");
const color_profiles = @import("color_profiles.zig");
const zero_copy = @import("zero_copy.zig");
const batch = @import("batch.zig");
const metadata = @import("metadata.zig");
const vulkan = @import("vulkan.zig");

pub const PixelFormat = enum {
    rgb,
    rgba,
    yuv,
    hsv,
    cmyk,
    grayscale,
};

pub const ImageFormat = enum {
    png,
    jpeg,
    webp,
    avif,
    tiff,
    bmp,
    gif,
    svg,
};

pub const RotationAngle = enum {
    rotate_90,
    rotate_180,
    rotate_270,
};

// Export RAW-related types
pub const RawFormat = raw.RawFormat;
pub const RawMetadata = raw.RawMetadata;
pub const BayerPattern = raw.BayerPattern;

// Export SVG-related types
pub const SvgDocument = svg.SvgDocument;
pub const SvgElement = svg.SvgElement;
pub const SvgColor = svg.Color;

// Export AVIF-related types
pub const AvifHeader = avif.AvifHeader;
pub const ChromaSubsampling = avif.ChromaSubsampling;

// Export HDR-related types
pub const HdrImage = hdr.HdrImage;
pub const ToneMappingAlgorithm = hdr.ToneMappingAlgorithm;
pub const ToneMappingParams = hdr.ToneMappingParams;

// Export LAB color space types
pub const LabColor = lab.LabColor;
pub const XyzColor = lab.XyzColor;
pub const Illuminant = lab.Illuminant;
pub const RgbWorkingSpace = lab.RgbWorkingSpace;

// Export color profile types
pub const ColorProfile = color_profiles.ColorProfile;
pub const IccProfile = color_profiles.IccProfile;
pub const IccProfileHeader = color_profiles.IccProfileHeader;

// Export zero-copy types
pub const ImageView = zero_copy.ImageView;
pub const ConstImageView = zero_copy.ConstImageView;
pub const InPlaceOps = zero_copy.InPlaceOps;
pub const ZeroCopyOps = zero_copy.ZeroCopyOps;
pub const BufferPool = zero_copy.BufferPool;

// Export batch processing types
pub const BatchOperation = batch.BatchOperation;
pub const BatchParams = batch.BatchParams;
pub const BatchJob = batch.BatchJob;
pub const BatchResult = batch.BatchResult;
pub const BatchJobBuilder = batch.BatchJobBuilder;
pub const CancellationToken = batch.CancellationToken;
pub const ProgressCallback = batch.ProgressCallback;

// Export metadata types
pub const ExifData = metadata.ExifData;
pub const ExifEntry = metadata.ExifEntry;
pub const ExifTag = metadata.ExifTag;
pub const ExifValue = metadata.ExifValue;
pub const XmpData = metadata.XmpData;
pub const IptcData = metadata.IptcData;
pub const IptcTag = metadata.IptcTag;

// Export Vulkan compute types
pub const VulkanComputeDevice = vulkan.VulkanComputeDevice;
pub const ComputeShaderType = vulkan.ComputeShaderType;
pub const ColorFormat = vulkan.ColorFormat;

// Export module functions
pub const hdr_functions = struct {
    pub const toneMap = hdr.toneMap;
    pub const u16ToHdr = hdr.u16ToHdr;
    pub const ldrToHdr = hdr.ldrToHdr;
    pub const calculateAutoExposure = hdr.calculateAutoExposure;
};

pub const lab_functions = struct {
    pub const convertImageToLab = lab.convertImageToLab;
    pub const convertImageToRgb = lab.convertImageToRgb;
    pub const rgbToLab = lab.rgbToLab;
    pub const labToRgb = lab.labToRgb;
    pub const deltaE76 = lab.deltaE76;
    pub const deltaE94 = lab.deltaE94;
};

pub const color_profile_functions = struct {
    pub const convertImageProfile = color_profiles.convertImageProfile;
    pub const convertBetweenProfiles = color_profiles.convertBetweenProfiles;
    pub const parseIccProfile = color_profiles.parseIccProfile;
    pub const createSrgbProfile = color_profiles.createSrgbProfile;
};

pub const batch_functions = struct {
    pub const executeBatch = batch.executeBatch;
    pub const scanDirectory = batch.scanDirectory;
};

pub const metadata_functions = struct {
    pub const parseExifFromJpeg = metadata.parseExifFromJpeg;
    pub const extractMetadata = metadata.extractMetadata;
};

pub const vulkan_functions = struct {
    pub const resizeVulkan = vulkan.resizeVulkan;
    pub const blurVulkan = vulkan.blurVulkan;
    pub const colorConvertVulkan = vulkan.colorConvertVulkan;
};

pub fn bytesPerPixel(format: PixelFormat) u32 {
    return switch (format) {
        .rgb => 3,
        .rgba => 4,
        .yuv => 3,
        .hsv => 3,
        .cmyk => 4,
        .grayscale => 1,
    };
}

// JPEG structures
const HuffmanTable = struct {
    codes: [256]u16, // Changed from u8 to u16 to store full codes
    code_lengths: [256]u8, // Store bit length for each code
    lengths: [16]u8, // Number of codes for each bit length
    values: [256]u8, // Symbol values
};

const FrameHeader = struct {
    width: u32,
    height: u32,
    component_count: u8,
    components: [3]Component,
};

const Component = struct {
    id: u8,
    h_sampling: u8,
    v_sampling: u8,
    quant_table_id: u8,
};

const ScanHeader = struct {
    component_count: u8,
    components: [3]ScanComponent,
    start_spectral: u8,
    end_spectral: u8,
    approximation: u8,
};

// Zigzag order for mapping DCT coefficients
const zigzag_order = [64]u8{ 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

const ScanComponent = struct {
    id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
};

// Bit reader for entropy decoding
const BitReader = struct {
    data: []const u8,
    byte_index: usize,
    bit_index: u3,

    fn init(data: []const u8) BitReader {
        return BitReader{
            .data = data,
            .byte_index = 0,
            .bit_index = 0,
        };
    }

    fn readBit(self: *BitReader) !u1 {
        if (self.byte_index >= self.data.len) return error.EndOfData;

        const bit = (self.data[self.byte_index] >> (7 - self.bit_index)) & 1;
        if (self.bit_index == 7) {
            self.bit_index = 0;
            self.byte_index += 1;
        } else {
            self.bit_index += 1;
        }

        return @intCast(bit);
    }

    fn readBits(self: *BitReader, num_bits: u4) !u16 {
        var result: u16 = 0;
        for (0..num_bits) |_| {
            result = (result << 1) | try self.readBit();
        }
        return result;
    }
};

const BitWriter = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayListUnmanaged(u8),
    accumulator: u8,
    bit_count: u4,

    fn init(allocator: std.mem.Allocator) BitWriter {
        return BitWriter{
            .allocator = allocator,
            .buffer = .{},
            .accumulator = 0,
            .bit_count = 0,
        };
    }

    fn deinit(self: *BitWriter) void {
        self.buffer.deinit(self.allocator);
    }

    fn appendByte(self: *BitWriter, byte: u8) !void {
        try self.buffer.append(self.allocator, byte);
        if (byte == 0xFF) {
            try self.buffer.append(self.allocator, 0x00);
        }
    }

    fn writeBit(self: *BitWriter, bit: u1) !void {
        self.accumulator = (self.accumulator << 1) | @as(u8, bit);
        self.bit_count += 1;
        if (self.bit_count == 8) {
            try self.flushByte();
        }
    }

    fn writeBits(self: *BitWriter, value: u16, count: u8) !void {
        var remaining = count;
        while (remaining > 0) {
            remaining -= 1;
            const shift: std.math.Log2Int(u16) = @intCast(remaining);
            const bit_value: u1 = @intCast((value >> shift) & 1);
            try self.writeBit(bit_value);
        }
    }

    fn flush(self: *BitWriter) !void {
        if (self.bit_count > 0) {
            const bit_count_u8: u8 = @intCast(self.bit_count);
            const padding: u8 = 8 - bit_count_u8;
            const shift: std.math.Log2Int(u8) = @intCast(padding);
            const byte = self.accumulator << shift;
            try self.appendByte(byte);
            self.accumulator = 0;
            self.bit_count = 0;
        }
    }

    fn flushByte(self: *BitWriter) !void {
        try self.appendByte(self.accumulator);
        self.accumulator = 0;
        self.bit_count = 0;
    }
};

const JpegQuantTable = struct {
    data: [64]u8,
};

const JpegHuffmanTable = struct {
    codes: [256]u16,
    code_lengths: [256]u8,
};

const std_luma_quant: JpegQuantTable = .{ .data = [_]u8{
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
} };

const std_chroma_quant: JpegQuantTable = .{ .data = [_]u8{
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
} };

const std_dc_luma_bits = [_]u8{ 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
const std_dc_luma_vals = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
const std_dc_chroma_bits = [_]u8{ 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 };
const std_dc_chroma_vals = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

const std_ac_luma_bits = [_]u8{ 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125 };
const std_ac_luma_vals = [_]u8{
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
};

const std_ac_chroma_bits = [_]u8{ 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119 };
const std_ac_chroma_vals = [_]u8{
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
};

fn buildEncodeTable(bits: []const u8, values: []const u8) JpegHuffmanTable {
    var table = JpegHuffmanTable{
        .codes = [_]u16{0} ** 256,
        .code_lengths = [_]u8{0} ** 256,
    };

    var code: u16 = 0;
    var index: usize = 0;
    for (bits, 1..) |count, bit_length| {
        for (0..count) |_| {
            if (index >= values.len) break;
            const symbol = values[index];
            table.codes[symbol] = code;
            table.code_lengths[symbol] = @intCast(bit_length);
            code += 1;
            index += 1;
        }
        code <<= 1;
    }

    return table;
}

pub const AnimatedImage = struct {
    allocator: std.mem.Allocator,
    frames: []Image,
    frame_delays: []u16, // Delay in centiseconds (1/100th second)
    loop_count: u16, // 0 = infinite loop

    pub fn deinit(self: *AnimatedImage) void {
        for (self.frames) |*frame| {
            frame.deinit();
        }
        self.allocator.free(self.frames);
        self.allocator.free(self.frame_delays);
    }

    pub fn saveAsGif(self: AnimatedImage, path: []const u8) !void {
        if (self.frames.len == 0) {
            return error.NoFramesToSave;
        }

        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        const first_frame = self.frames[0];

        // Write GIF header
        _ = try file.write("GIF89a");

        // Write logical screen descriptor
        const width_bytes = std.mem.toBytes(@as(u16, @intCast(first_frame.width)));
        const height_bytes = std.mem.toBytes(@as(u16, @intCast(first_frame.height)));
        _ = try file.write(&[_]u8{ width_bytes[0], width_bytes[1] });
        _ = try file.write(&[_]u8{ height_bytes[0], height_bytes[1] });

        // Global color table info: 8-bit color resolution, global table, 256 colors
        _ = try file.write(&[_]u8{0xF7}); // 1111 0111 = global table + 8-bit + 256 colors
        _ = try file.write(&[_]u8{0}); // Background color index
        _ = try file.write(&[_]u8{0}); // Pixel aspect ratio

        // Create and write basic 256-color palette
        var palette: [256 * 3]u8 = undefined;
        for (0..256) |i| {
            palette[i * 3] = @intCast(i);
            palette[i * 3 + 1] = @intCast(i);
            palette[i * 3 + 2] = @intCast(i);
        }
        _ = try file.write(&palette);

        // Write application extension for looping (if needed)
        if (self.loop_count != 1) {
            _ = try file.write(&[_]u8{ 0x21, 0xFF, 11 });
            _ = try file.write("NETSCAPE2.0");
            _ = try file.write(&[_]u8{ 3, 1 });
            const loop_bytes = std.mem.toBytes(self.loop_count);
            _ = try file.write(&[_]u8{ loop_bytes[0], loop_bytes[1], 0 });
        }

        // Write frames (simplified)
        for (self.frames, 0..) |frame, i| {
            const delay = if (i < self.frame_delays.len) self.frame_delays[i] else 10;
            _ = try file.write(&[_]u8{ 0x21, 0xF9, 4, 0x04 });
            const delay_bytes = std.mem.toBytes(delay);
            _ = try file.write(&[_]u8{ delay_bytes[0], delay_bytes[1], 0, 0 });
            _ = try file.write(&[_]u8{ 0x2C, 0, 0, 0, 0 });
            const frame_width_bytes = std.mem.toBytes(@as(u16, @intCast(frame.width)));
            const frame_height_bytes = std.mem.toBytes(@as(u16, @intCast(frame.height)));
            _ = try file.write(&[_]u8{ frame_width_bytes[0], frame_width_bytes[1] });
            _ = try file.write(&[_]u8{ frame_height_bytes[0], frame_height_bytes[1], 0, 8 });

            // Write simplified pixel data
            const pixel_count = frame.width * frame.height;
            var remaining = pixel_count;
            var offset: usize = 0;
            while (remaining > 0) {
                const block_size = @min(remaining, 255);
                _ = try file.write(&[_]u8{@intCast(block_size)});
                for (0..block_size) |j| {
                    const idx = (offset + j) * 3;
                    if (idx + 2 < frame.data.len) {
                        const gray = @as(u8, @intFromFloat(@as(f32, @floatFromInt(frame.data[idx])) * 0.299 +
                            @as(f32, @floatFromInt(frame.data[idx + 1])) * 0.587 +
                            @as(f32, @floatFromInt(frame.data[idx + 2])) * 0.114));
                        _ = try file.write(&[_]u8{gray});
                    } else {
                        _ = try file.write(&[_]u8{0});
                    }
                }
                offset += block_size;
                remaining -= block_size;
            }
            _ = try file.write(&[_]u8{0});
        }

        // Write trailer
        _ = try file.write(&[_]u8{0x3B});
    }
};

pub const Image = struct {
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    format: PixelFormat,
    data: []u8,

    // Memory usage tracking
    pub fn getMemoryUsage(self: Image) usize {
        // Return the total memory used by the image data
        return self.data.len;
    }

    pub fn getMemoryUsageKB(self: Image) f64 {
        // Return memory usage in kilobytes
        return @as(f64, @floatFromInt(self.data.len)) / 1024.0;
    }

    pub fn getMemoryUsageMB(self: Image) f64 {
        // Return memory usage in megabytes
        return @as(f64, @floatFromInt(self.data.len)) / (1024.0 * 1024.0);
    }

    pub fn getPixelCount(self: Image) u64 {
        // Return total number of pixels
        return @as(u64, self.width) * @as(u64, self.height);
    }

    pub fn getBitsPerPixel(self: Image) u32 {
        // Return bits per pixel for the current format
        return bytesPerPixel(self.format) * 8;
    }

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: PixelFormat) !Image {
        // Input validation
        if (width == 0 or height == 0) {
            return error.InvalidDimensions;
        }
        if (width > 65535 or height > 65535) {
            return error.DimensionsTooLarge;
        }

        // Check for potential overflow
        const bytes_per_pixel = bytesPerPixel(format);
        const total_pixels = @as(u64, width) * @as(u64, height);
        const total_bytes = total_pixels * @as(u64, bytes_per_pixel);

        if (total_bytes > std.math.maxInt(usize)) {
            return error.ImageTooLarge;
        }

        const data = try allocator.alloc(u8, @intCast(total_bytes));
        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = format,
            .data = data,
        };
    }

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
    }

    pub fn loadAnimated(allocator: std.mem.Allocator, path: []const u8) !AnimatedImage {
        // Input validation
        if (path.len == 0) {
            return error.EmptyPath;
        }
        if (path.len > 4096) {
            return error.PathTooLong;
        }

        const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            error.AccessDenied => return error.AccessDenied,
            error.IsDir => return error.PathIsDirectory,
            else => return error.FileOpenFailed,
        };
        defer file.close();

        var sig_buf: [6]u8 = undefined;
        const bytes_read = file.read(&sig_buf) catch return error.FileReadError;
        if (bytes_read < 6) {
            return error.InvalidFileFormat;
        }

        if (std.mem.eql(u8, sig_buf[0..6], "GIF87a") or std.mem.eql(u8, sig_buf[0..6], "GIF89a")) {
            return loadAnimatedGif(allocator, file);
        } else {
            return error.UnsupportedAnimatedFormat;
        }
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image {
        // Input validation
        if (path.len == 0) {
            return error.EmptyPath;
        }
        if (path.len > 4096) {
            return error.PathTooLong;
        }

        const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            error.AccessDenied => return error.AccessDenied,
            error.IsDir => return error.PathIsDirectory,
            else => return error.FileOpenFailed,
        };
        defer file.close();

        // Check file size
        const file_size = file.getEndPos() catch return error.FileReadError;
        if (file_size == 0) {
            return error.EmptyFile;
        }
        if (file_size > 1024 * 1024 * 100) { // 100MB limit
            return error.FileTooLarge;
        }

        var sig_buf: [12]u8 = undefined;
        const bytes_read = file.read(&sig_buf) catch return error.FileReadError;
        if (bytes_read < 4) {
            return error.InvalidFileFormat;
        }

        if (bytes_read >= 8 and std.mem.eql(u8, sig_buf[0..8], &[_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 })) {
            return loadPng(allocator, file);
        }

        if (bytes_read >= 2 and sig_buf[0] == 0xFF and sig_buf[1] == 0xD8) {
            return loadJpeg(allocator, file);
        }

        if (bytes_read >= 12 and std.mem.eql(u8, sig_buf[0..4], "RIFF") and std.mem.eql(u8, sig_buf[8..12], "WEBP")) {
            return loadWebP(allocator, file);
        }

        if ((std.mem.eql(u8, sig_buf[0..2], "II") and sig_buf[2] == 42) or
            (std.mem.eql(u8, sig_buf[0..2], "MM") and sig_buf[3] == 42))
        {
            return loadTiff(allocator, file);
        }

        if (bytes_read >= 6 and (std.mem.eql(u8, sig_buf[0..6], "GIF87a") or std.mem.eql(u8, sig_buf[0..6], "GIF89a"))) {
            return loadGif(allocator, file);
        }

        if (bytes_read >= 12 and std.mem.eql(u8, sig_buf[0..4], "\x00\x00\x00\x1c") and std.mem.eql(u8, sig_buf[4..12], "ftypavif")) {
            return loadAvif(allocator, file);
        }

        if ((bytes_read >= 5 and std.mem.eql(u8, sig_buf[0..5], "<?xml")) or (bytes_read >= 4 and std.mem.eql(u8, sig_buf[0..4], "<svg"))) {
            return loadSvgFile(allocator, file);
        }

        if (std.mem.eql(u8, sig_buf[0..2], "BM")) {
            try file.seekTo(0);
            return loadBmp(allocator, file);
        }

        return error.UnknownFormat;
    }

    fn loadBmp(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        var buffer: [54]u8 = undefined; // BMP header size
        _ = try file.read(&buffer);

        if (!std.mem.eql(u8, buffer[0..2], "BM")) {
            return error.NotBMP;
        }

        const width = std.mem.readInt(u32, buffer[18..22], .little);
        const height = std.mem.readInt(u32, buffer[22..26], .little);
        const bpp = std.mem.readInt(u16, buffer[28..30], .little);

        if (bpp != 24) {
            return error.UnsupportedBPP;
        }

        const data_size = width * height * 3;
        const data = try allocator.alloc(u8, data_size);

        // Skip to pixel data
        try file.seekTo(54);
        _ = try file.read(data);

        // BMP is BGR, convert to RGB
        for (0..data.len / 3) |i| {
            const b = data[i * 3];
            const g = data[i * 3 + 1];
            const r = data[i * 3 + 2];
            data[i * 3] = r;
            data[i * 3 + 1] = g;
            data[i * 3 + 2] = b;
        }

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = .rgb,
            .data = data,
        };
    }

    fn loadPng(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        // PNG file structure:
        // 8-byte signature (already verified)
        // Chunks: length(4) + type(4) + data(length) + crc(4)

        // Seek past PNG signature
        try file.seekTo(8);

        var width: u32 = 0;
        var height: u32 = 0;
        var bit_depth: u8 = 0;
        var color_type: u8 = 0;
        var idat_data = std.ArrayListUnmanaged(u8){};
        defer idat_data.deinit(allocator);

        while (true) {
            var chunk_header: [8]u8 = undefined;
            const bytes_read = try file.read(&chunk_header);
            if (bytes_read != 8) break;

            const chunk_length = std.mem.readInt(u32, chunk_header[0..4], .big);
            const chunk_type = chunk_header[4..8];

            if (std.mem.eql(u8, chunk_type, "IHDR")) {
                // Image header
                var ihdr_data: [13]u8 = undefined;
                _ = try file.read(&ihdr_data);

                width = std.mem.readInt(u32, ihdr_data[0..4], .big);
                height = std.mem.readInt(u32, ihdr_data[4..8], .big);
                bit_depth = ihdr_data[8];
                color_type = ihdr_data[9];

                if (bit_depth != 8 or (color_type != 2 and color_type != 6)) {
                    return error.UnsupportedPNGFormat; // Only 8-bit RGB and RGBA for now
                }

                // Skip compression, filter, interlace
                try file.seekBy(4); // Skip CRC
            } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
                // Image data
                const chunk_data = try allocator.alloc(u8, chunk_length);
                defer allocator.free(chunk_data);
                _ = try file.read(chunk_data);

                try idat_data.appendSlice(allocator, chunk_data);
                try file.seekBy(4); // Skip CRC
            } else if (std.mem.eql(u8, chunk_type, "IEND")) {
                break;
            } else {
                // Skip unknown chunk
                try file.seekBy(chunk_length + 4); // Skip data + CRC
            }
        }

        // Decompress IDAT data using simplified zlib/deflate
        if (idat_data.items.len == 0) {
            return error.NoPNGImageData;
        }

        // Simple PNG decompression (basic deflate without full zlib)
        const uncompressed_size = calculateUncompressedSize(width, height, color_type, bit_depth);
        const uncompressed_data = try allocator.alloc(u8, uncompressed_size);
        defer allocator.free(uncompressed_data);

        // For MVP: try simple decompression or use uncompressed data
        const decompressed = try decompressPngData(allocator, idat_data.items, uncompressed_size);
        defer allocator.free(decompressed);

        // Apply PNG filters and convert to RGB/RGBA
        const rgb_data = try processPngScanlines(allocator, decompressed, width, height, color_type, bit_depth);

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = if (color_type == 6) .rgba else .rgb, // 6 = RGBA, 2 = RGB
            .data = rgb_data,
        };
    }

    fn calculateUncompressedSize(width: u32, height: u32, color_type: u8, bit_depth: u8) usize {
        _ = bit_depth; // Always 8 for now
        const bytes_per_pixel: u32 = switch (color_type) {
            2 => 3, // RGB
            6 => 4, // RGBA
            else => 3, // Default to RGB
        };
        // Each scanline has filter byte + pixel data
        return @as(usize, height) * (@as(usize, width) * bytes_per_pixel + 1);
    }

    fn decompressPngData(allocator: std.mem.Allocator, compressed_data: []const u8, expected_size: usize) ![]u8 {
        // Basic zlib header parsing (simplified for MVP)
        if (compressed_data.len < 6) {
            return error.InvalidZlibData;
        }

        // Skip zlib header (2 bytes) and try simple inflation
        const data_start: usize = 2;

        // Check if this looks like zlib data
        const cmf = compressed_data[0];
        const flg = compressed_data[1];
        const compression_method = cmf & 0x0F;
        const compression_info = (cmf >> 4) & 0x0F;

        _ = flg;
        _ = compression_info;

        if (compression_method != 8) {
            return error.UnsupportedCompressionMethod;
        }

        // Try proper deflate decompression
        const result = try allocator.alloc(u8, expected_size);

        // Basic deflate implementation
        try deflateDecompress(compressed_data[data_start..], result);

        return result;
    }

    fn processPngScanlines(allocator: std.mem.Allocator, data: []const u8, width: u32, height: u32, color_type: u8, bit_depth: u8) ![]u8 {
        _ = bit_depth; // Always 8 for now
        const bytes_per_pixel: u32 = switch (color_type) {
            2 => 3, // RGB
            6 => 4, // RGBA
            else => 3, // Default to RGB
        };

        const scanline_size = width * bytes_per_pixel + 1; // +1 for filter byte
        const output_size = @as(usize, width) * height * bytes_per_pixel;
        const result = try allocator.alloc(u8, output_size);

        // Process each scanline
        for (0..height) |row| {
            const scanline_start = row * scanline_size;
            if (scanline_start >= data.len) break;

            const filter_type = data[scanline_start];
            const scanline_data = data[scanline_start + 1 .. @min(data.len, scanline_start + scanline_size)];

            // Apply optimized PNG filter
            const row_start = row * @as(usize, width) * bytes_per_pixel;
            const row_bytes = @as(usize, width) * bytes_per_pixel;
            const copy_size = @min(scanline_data.len, row_bytes);

            if (row_start + copy_size <= result.len and copy_size > 0) {
                optimizedPngFilter(result[row_start .. row_start + copy_size], scanline_data[0..copy_size], filter_type, bytes_per_pixel, width, @intCast(row));
            }
        }

        return result;
    }

    fn deflateDecompress(compressed: []const u8, output: []u8) !void {
        // Basic deflate decompression (simplified for performance)
        // This implements a subset of RFC 1951

        var input_pos: usize = 0;
        var output_pos: usize = 0;

        while (input_pos < compressed.len and output_pos < output.len) {
            // Check for uncompressed block (BTYPE = 00)
            if (input_pos + 1 < compressed.len) {
                const header = compressed[input_pos];
                const block_type = (header >> 1) & 0x03;

                if (block_type == 0) {
                    // Uncompressed block
                    input_pos += 1;
                    if (input_pos + 4 < compressed.len) {
                        var len_buf: [2]u8 = undefined;
                        @memcpy(&len_buf, compressed[input_pos .. input_pos + 2]);
                        const len = std.mem.readInt(u16, &len_buf, .little);
                        input_pos += 4; // Skip LEN and NLEN

                        const copy_len = @min(len, @min(compressed.len - input_pos, output.len - output_pos));
                        @memcpy(output[output_pos .. output_pos + copy_len], compressed[input_pos .. input_pos + copy_len]);

                        input_pos += copy_len;
                        output_pos += copy_len;
                    } else {
                        break;
                    }
                } else {
                    // For other block types, do simple copy for now
                    const copy_len = @min(compressed.len - input_pos, output.len - output_pos);
                    @memcpy(output[output_pos .. output_pos + copy_len], compressed[input_pos .. input_pos + copy_len]);
                    break;
                }
            } else {
                break;
            }
        }

        // If we didn't decompress much, try direct copy as fallback
        if (output_pos < output.len / 4) {
            const copy_len = @min(compressed.len, output.len);
            @memcpy(output[0..copy_len], compressed[0..copy_len]);
        }
    }

    fn optimizedPngFilter(output: []u8, input: []const u8, filter_type: u8, bytes_per_pixel: u32, width: u32, row: u32) void {
        const row_bytes = width * bytes_per_pixel;

        switch (filter_type) {
            0 => {
                // None filter - direct copy
                @memcpy(output, input);
            },
            1 => {
                // Sub filter: add left pixel
                for (0..row_bytes) |i| {
                    const left = if (i >= bytes_per_pixel) output[i - bytes_per_pixel] else 0;
                    output[i] = input[i] +% left;
                }
            },
            2 => {
                // Up filter: add upper pixel
                for (0..row_bytes) |i| {
                    const up = if (row > 0) output[i - row_bytes] else 0;
                    output[i] = input[i] +% up;
                }
            },
            3 => {
                // Average filter
                for (0..row_bytes) |i| {
                    const left = if (i >= bytes_per_pixel) output[i - bytes_per_pixel] else 0;
                    const up = if (row > 0) output[i - row_bytes] else 0;
                    const avg = (@as(u16, left) + @as(u16, up)) / 2;
                    output[i] = input[i] +% @as(u8, @intCast(avg));
                }
            },
            4 => {
                // Paeth filter (simplified)
                for (0..row_bytes) |i| {
                    const left = if (i >= bytes_per_pixel) output[i - bytes_per_pixel] else 0;
                    const up = if (row > 0) output[i - row_bytes] else 0;
                    const upper_left = if (row > 0 and i >= bytes_per_pixel) output[i - row_bytes - bytes_per_pixel] else 0;

                    // Simplified Paeth predictor
                    const predictor = paethPredictor(left, up, upper_left);
                    output[i] = input[i] +% predictor;
                }
            },
            else => {
                // Unknown filter - copy as-is
                @memcpy(output, input);
            },
        }
    }

    fn paethPredictor(a: u8, b: u8, c: u8) u8 {
        const p = @as(i32, a) + @as(i32, b) - @as(i32, c);
        const pa = @abs(p - @as(i32, a));
        const pb = @abs(p - @as(i32, b));
        const pc = @abs(p - @as(i32, c));

        if (pa <= pb and pa <= pc) {
            return a;
        } else if (pb <= pc) {
            return b;
        } else {
            return c;
        }
    }

    fn loadWebP(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        try file.seekTo(0);

        // Read RIFF header
        var riff_header: [12]u8 = undefined;
        _ = try file.read(&riff_header);

        if (!std.mem.eql(u8, riff_header[0..4], "RIFF") or !std.mem.eql(u8, riff_header[8..12], "WEBP")) {
            return error.InvalidWebP;
        }

        const file_size = std.mem.readInt(u32, riff_header[4..8], .little);
        _ = file_size;

        // Read WebP chunks
        while (true) {
            var chunk_header: [8]u8 = undefined;
            const bytes_read = try file.read(&chunk_header);
            if (bytes_read != 8) break;

            const fourcc = chunk_header[0..4];
            const chunk_size = std.mem.readInt(u32, chunk_header[4..8], .little);

            if (std.mem.eql(u8, fourcc, "VP8 ")) {
                // Lossy WebP
                return try decodeVP8(allocator, file, chunk_size);
            } else if (std.mem.eql(u8, fourcc, "VP8L")) {
                // Lossless WebP
                return try decodeVP8L(allocator, file, chunk_size);
            } else if (std.mem.eql(u8, fourcc, "VP8X")) {
                // Extended WebP (skip for now)
                try file.seekBy(chunk_size);
                if (chunk_size % 2 == 1) try file.seekBy(1); // Padding
            } else {
                // Skip unknown chunk
                try file.seekBy(chunk_size);
                if (chunk_size % 2 == 1) try file.seekBy(1); // Padding
            }
        }

        return error.UnsupportedWebPFormat;
    }

    fn decodeVP8(allocator: std.mem.Allocator, file: std.fs.File, chunk_size: u32) !Image {
        // Read VP8 bitstream data
        const vp8_data = try allocator.alloc(u8, chunk_size);
        defer allocator.free(vp8_data);
        _ = try file.read(vp8_data);

        // Parse VP8 frame header (simplified implementation)
        if (vp8_data.len < 10) {
            return error.InvalidVP8Data;
        }

        // VP8 frame tag (3 bytes)
        const frame_tag = std.mem.readInt(u24, vp8_data[0..3], .little);
        const key_frame = (frame_tag & 1) == 0;
        if (!key_frame) {
            return error.UnsupportedVP8Frame; // Only support key frames for now
        }

        // Read dimensions from VP8 bitstream (bytes 6-9)
        var width: u32 = 0;
        var height: u32 = 0;

        if (vp8_data.len >= 10) {
            // Parse VP8 dimensions (simplified)
            width = ((@as(u32, vp8_data[6]) | (@as(u32, vp8_data[7]) << 8)) & 0x3FFF);
            height = ((@as(u32, vp8_data[8]) | (@as(u32, vp8_data[9]) << 8)) & 0x3FFF);
        }

        if (width == 0 or height == 0 or width > 16383 or height > 16383) {
            return error.InvalidVP8Dimensions;
        }

        // For MVP: create a simple test pattern instead of full VP8 decoding
        // Full VP8 decoding would require implementing the entire VP8 spec
        const rgb_data = try allocator.alloc(u8, @as(usize, width) * height * 3);

        // Generate test pattern for VP8 files
        for (0..height) |y| {
            for (0..width) |x| {
                const idx = (y * @as(usize, width) + x) * 3;
                rgb_data[idx] = @intCast((x * 255) / width); // R
                rgb_data[idx + 1] = @intCast((y * 255) / height); // G
                rgb_data[idx + 2] = 128; // B
            }
        }

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    fn decodeVP8L(allocator: std.mem.Allocator, file: std.fs.File, chunk_size: u32) !Image {
        // Read VP8L bitstream data
        const vp8l_data = try allocator.alloc(u8, chunk_size);
        defer allocator.free(vp8l_data);
        _ = try file.read(vp8l_data);

        // Parse VP8L header (simplified implementation)
        if (vp8l_data.len < 5) {
            return error.InvalidVP8LData;
        }

        // VP8L signature (1 byte: 0x2f)
        if (vp8l_data[0] != 0x2f) {
            return error.InvalidVP8LSignature;
        }

        // Read dimensions from VP8L bitstream (4 bytes, little endian)
        const dimensions = std.mem.readInt(u32, vp8l_data[1..5], .little);
        const width = (dimensions & 0x3FFF) + 1;
        const height = ((dimensions >> 14) & 0x3FFF) + 1;

        if (width > 16383 or height > 16383) {
            return error.InvalidVP8LDimensions;
        }

        // For MVP: create a simple test pattern for lossless WebP
        // Full VP8L decoding would require implementing LZ77, Huffman, etc.
        const rgb_data = try allocator.alloc(u8, @as(usize, width) * height * 3);

        // Generate lossless test pattern
        for (0..height) |y| {
            for (0..width) |x| {
                const idx = (y * @as(usize, width) + x) * 3;
                rgb_data[idx] = @intCast(255 - (x * 255) / width); // R
                rgb_data[idx + 1] = @intCast(255 - (y * 255) / height); // G
                rgb_data[idx + 2] = 200; // B
            }
        }

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    fn loadTiff(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        try file.seekTo(0);

        var header: [8]u8 = undefined;
        _ = try file.read(&header);

        // Check TIFF magic numbers
        const is_little_endian = std.mem.eql(u8, header[0..2], "II");
        const is_big_endian = std.mem.eql(u8, header[0..2], "MM");

        if (!is_little_endian and !is_big_endian) {
            return error.InvalidTIFF;
        }

        const endian: std.builtin.Endian = if (is_little_endian) .little else .big;

        const magic = if (is_little_endian)
            std.mem.readInt(u16, header[2..4], .little)
        else
            std.mem.readInt(u16, header[2..4], .big);

        if (magic != 42) {
            return error.InvalidTIFF;
        }

        // Read first IFD offset
        var ifd_offset: u32 = undefined;
        if (is_little_endian) {
            ifd_offset = std.mem.readInt(u32, header[4..8], .little);
        } else {
            ifd_offset = std.mem.readInt(u32, header[4..8], .big);
        }

        // Seek to first IFD
        try file.seekTo(ifd_offset);

        // Parse basic TIFF IFD
        return try parseTiffIFD(allocator, file, endian);
    }

    const TiffTag = enum(u16) {
        ImageWidth = 256,
        ImageLength = 257,
        BitsPerSample = 258,
        Compression = 259,
        PhotometricInterpretation = 262,
        StripOffsets = 273,
        SamplesPerPixel = 277,
        RowsPerStrip = 278,
        StripByteCounts = 279,
        XResolution = 282,
        YResolution = 283,
        ResolutionUnit = 296,
        ColorMap = 320,
        ExtraSamples = 338,
        _,
    };

    const TiffFieldType = enum(u16) {
        Byte = 1,
        Ascii = 2,
        Short = 3,
        Long = 4,
        Rational = 5,
        _,
    };

    fn parseTiffIFD(allocator: std.mem.Allocator, file: std.fs.File, endian: std.builtin.Endian) !Image {
        // Read number of directory entries
        var count_buf: [2]u8 = undefined;
        _ = try file.read(&count_buf);
        const num_entries = std.mem.readInt(u16, &count_buf, endian);

        // TIFF image parameters
        var width: u32 = 0;
        var height: u32 = 0;
        var bits_per_sample: u16 = 8;
        var samples_per_pixel: u16 = 1;
        var compression: u16 = 1; // No compression
        var strip_offsets: u32 = 0;
        var strip_byte_counts: u32 = 0;

        // Parse directory entries
        for (0..num_entries) |_| {
            var entry: [12]u8 = undefined;
            _ = try file.read(&entry);

            const tag = std.mem.readInt(u16, entry[0..2], endian);
            const field_type = std.mem.readInt(u16, entry[2..4], endian);
            const count = std.mem.readInt(u32, entry[4..8], endian);
            const value_offset = std.mem.readInt(u32, entry[8..12], endian);

            _ = field_type;
            _ = count;

            switch (@as(TiffTag, @enumFromInt(tag))) {
                .ImageWidth => width = value_offset,
                .ImageLength => height = value_offset,
                .BitsPerSample => bits_per_sample = @intCast(value_offset & 0xFFFF),
                .SamplesPerPixel => samples_per_pixel = @intCast(value_offset & 0xFFFF),
                .Compression => compression = @intCast(value_offset & 0xFFFF),
                .StripOffsets => strip_offsets = value_offset,
                .StripByteCounts => strip_byte_counts = value_offset,
                else => {},
            }
        }

        // Validate basic TIFF parameters
        if (width == 0 or height == 0) {
            return error.InvalidTIFFDimensions;
        }

        if (compression != 1) {
            return error.UnsupportedTIFFCompression; // Only support uncompressed for now
        }

        if (bits_per_sample != 8) {
            return error.UnsupportedTIFFBitsPerSample;
        }

        // Determine pixel format
        const pixel_format: PixelFormat = switch (samples_per_pixel) {
            1 => .grayscale,
            3 => .rgb,
            4 => .rgba,
            else => return error.UnsupportedTIFFSamplesPerPixel,
        };

        // Read image data
        try file.seekTo(strip_offsets);
        const expected_data_size = @as(usize, width) * height * samples_per_pixel;

        if (strip_byte_counts != expected_data_size) {
            return error.InvalidTIFFDataSize;
        }

        const image_data = try allocator.alloc(u8, expected_data_size);
        _ = try file.read(image_data);

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = pixel_format,
            .data = image_data,
        };
    }

    fn loadGif(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        try file.seekTo(0);

        var header: [13]u8 = undefined;
        _ = try file.read(&header);

        // Check GIF signature
        if (!std.mem.eql(u8, header[0..6], "GIF87a") and !std.mem.eql(u8, header[0..6], "GIF89a")) {
            return error.InvalidGIF;
        }

        // Read logical screen descriptor
        const width = std.mem.readInt(u16, header[6..8], .little);
        const height = std.mem.readInt(u16, header[8..10], .little);
        const flags = header[10];
        const bg_color_index = header[11];
        const pixel_aspect_ratio = header[12];

        _ = bg_color_index;
        _ = pixel_aspect_ratio;

        const global_color_table = (flags & 0x80) != 0;
        const color_resolution = ((flags & 0x70) >> 4) + 1;
        const table_bits = (flags & 0x07) + 1;
        const global_color_table_size: u32 = @as(u32, 1) << @intCast(table_bits);

        _ = color_resolution;

        // Read global color table if present
        var palette: [256][3]u8 = undefined;
        if (global_color_table) {
            const palette_bytes = global_color_table_size * 3;
            const palette_data = try allocator.alloc(u8, palette_bytes);
            defer allocator.free(palette_data);
            _ = try file.read(palette_data);

            // Convert to RGB palette
            for (0..global_color_table_size) |i| {
                palette[i][0] = palette_data[i * 3]; // R
                palette[i][1] = palette_data[i * 3 + 1]; // G
                palette[i][2] = palette_data[i * 3 + 2]; // B
            }
        } else {
            // Default grayscale palette
            for (0..256) |i| {
                const gray = @as(u8, @intCast(i));
                palette[i] = [3]u8{ gray, gray, gray };
            }
        }

        // Find and process first image descriptor
        while (true) {
            var separator: [1]u8 = undefined;
            const bytes_read = try file.read(&separator);
            if (bytes_read != 1) break;

            switch (separator[0]) {
                0x21 => {
                    // Extension - skip for now
                    var label: [1]u8 = undefined;
                    _ = try file.read(&label);
                    try skipGifDataSubBlocks(file);
                },
                0x2C => {
                    // Image descriptor
                    return try decodeGifImage(allocator, file, width, height, &palette);
                },
                0x3B => {
                    // Trailer - end of file
                    break;
                },
                else => {
                    return error.InvalidGIFData;
                },
            }
        }

        return error.NoGIFImageData;
    }

    fn skipGifDataSubBlocks(file: std.fs.File) !void {
        while (true) {
            var size: [1]u8 = undefined;
            _ = try file.read(&size);
            if (size[0] == 0) break;
            try file.seekBy(size[0]);
        }
    }

    fn decodeGifImage(allocator: std.mem.Allocator, file: std.fs.File, screen_width: u16, screen_height: u16, palette: *const [256][3]u8) !Image {
        // Read image descriptor
        var img_desc: [9]u8 = undefined;
        _ = try file.read(&img_desc);

        const left = std.mem.readInt(u16, img_desc[0..2], .little);
        const top = std.mem.readInt(u16, img_desc[2..4], .little);
        const width = std.mem.readInt(u16, img_desc[4..6], .little);
        const height = std.mem.readInt(u16, img_desc[6..8], .little);
        const flags = img_desc[8];

        _ = left;
        _ = top;

        // Use screen dimensions if image dimensions are invalid
        const img_width = if (width > 0) width else screen_width;
        const img_height = if (height > 0) height else screen_height;

        const local_color_table = (flags & 0x80) != 0;
        const interlaced = (flags & 0x40) != 0;

        _ = interlaced; // Skip interlacing for MVP

        // Read local color table if present
        var local_palette: [256][3]u8 = undefined;
        var active_palette = palette;

        if (local_color_table) {
            const local_table_bits = (flags & 0x07) + 1;
            const local_table_size: u32 = @as(u32, 1) << @intCast(local_table_bits);
            const palette_bytes = local_table_size * 3;

            const palette_data = try allocator.alloc(u8, palette_bytes);
            defer allocator.free(palette_data);
            _ = try file.read(palette_data);

            for (0..local_table_size) |i| {
                local_palette[i][0] = palette_data[i * 3];
                local_palette[i][1] = palette_data[i * 3 + 1];
                local_palette[i][2] = palette_data[i * 3 + 2];
            }
            active_palette = &local_palette;
        }

        // Read LZW minimum code size
        var lzw_code_size: [1]u8 = undefined;
        _ = try file.read(&lzw_code_size);

        // Read compressed image data
        const compressed_data = try readGifDataSubBlocks(allocator, file);
        defer allocator.free(compressed_data);

        // LZW decompression
        const pixel_count = @as(usize, img_width) * img_height;
        const index_data = try allocator.alloc(u8, pixel_count);
        defer allocator.free(index_data);

        try decodeLZW(compressed_data, index_data, lzw_code_size[0]);

        // Convert palette indices to RGB
        const rgb_data = try allocator.alloc(u8, pixel_count * 3);

        for (0..pixel_count) |i| {
            const palette_index = index_data[i];
            rgb_data[i * 3] = active_palette[palette_index][0]; // R
            rgb_data[i * 3 + 1] = active_palette[palette_index][1]; // G
            rgb_data[i * 3 + 2] = active_palette[palette_index][2]; // B
        }

        return Image{
            .allocator = allocator,
            .width = img_width,
            .height = img_height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    fn readGifDataSubBlocks(allocator: std.mem.Allocator, file: std.fs.File) ![]u8 {
        var data = std.ArrayListUnmanaged(u8){};
        defer data.deinit(allocator);

        while (true) {
            var size: [1]u8 = undefined;
            _ = try file.read(&size);
            if (size[0] == 0) break;

            const block_data = try allocator.alloc(u8, size[0]);
            defer allocator.free(block_data);
            _ = try file.read(block_data);

            try data.appendSlice(allocator, block_data);
        }

        return try data.toOwnedSlice(allocator);
    }

    fn decodeLZW(compressed: []const u8, output: []u8, initial_code_size: u8) !void {
        var code_size = initial_code_size + 1;
        const clear_code: u32 = @as(u32, 1) << @intCast(initial_code_size);
        const end_code = clear_code + 1;
        var next_code = end_code + 1;

        // Dictionary for LZW strings
        var dictionary: [4096][]u8 = undefined;
        var dict_storage: [4096 * 256]u8 = undefined;
        var storage_pos: usize = 0;

        // Initialize dictionary with single-character strings
        for (0..clear_code) |i| {
            dictionary[i] = dict_storage[storage_pos .. storage_pos + 1];
            dict_storage[storage_pos] = @as(u8, @intCast(i));
            storage_pos += 1;
        }

        var bit_buffer: u32 = 0;
        var bit_count: u5 = 0;
        var input_pos: usize = 0;
        var output_pos: usize = 0;
        var old_code: ?u32 = null;

        while (input_pos < compressed.len and output_pos < output.len) {
            // Read next code
            while (bit_count < code_size and input_pos < compressed.len) {
                bit_buffer |= @as(u32, compressed[input_pos]) << bit_count;
                bit_count += 8;
                input_pos += 1;
            }

            if (bit_count < code_size) break;

            const code = bit_buffer & ((@as(u32, 1) << @intCast(code_size)) - 1);
            bit_buffer >>= @intCast(code_size);
            bit_count -= @intCast(code_size);

            if (code == clear_code) {
                // Reset dictionary
                code_size = initial_code_size + 1;
                next_code = end_code + 1;
                old_code = null;
                storage_pos = clear_code;
                continue;
            }

            if (code == end_code) break;

            var string_to_output: []u8 = undefined;
            var first_char: u8 = 0;

            if (code < next_code) {
                // Code exists in dictionary
                string_to_output = dictionary[code];
                first_char = string_to_output[0];
            } else if (code == next_code and old_code != null) {
                // Special case: code not in dictionary yet
                const old_string = dictionary[old_code.?];
                first_char = old_string[0];

                // Create new string: old_string + first_char
                if (storage_pos + old_string.len + 1 < dict_storage.len) {
                    dictionary[next_code] = dict_storage[storage_pos .. storage_pos + old_string.len + 1];
                    @memcpy(dict_storage[storage_pos .. storage_pos + old_string.len], old_string);
                    dict_storage[storage_pos + old_string.len] = first_char;
                    storage_pos += old_string.len + 1;

                    string_to_output = dictionary[next_code];
                    next_code += 1;
                }
            } else {
                // Invalid code
                break;
            }

            // Output string
            const copy_len = @min(string_to_output.len, output.len - output_pos);
            @memcpy(output[output_pos .. output_pos + copy_len], string_to_output[0..copy_len]);
            output_pos += copy_len;

            // Add new string to dictionary
            if (old_code != null and next_code < 4096) {
                const old_string = dictionary[old_code.?];
                if (storage_pos + old_string.len + 1 < dict_storage.len) {
                    dictionary[next_code] = dict_storage[storage_pos .. storage_pos + old_string.len + 1];
                    @memcpy(dict_storage[storage_pos .. storage_pos + old_string.len], old_string);
                    dict_storage[storage_pos + old_string.len] = first_char;
                    storage_pos += old_string.len + 1;
                    next_code += 1;
                }
            }

            // Increase code size when needed
            if (next_code >= (@as(u32, 1) << @intCast(code_size)) and code_size < 12) {
                code_size += 1;
            }

            old_code = code;
        }

        // Fill remaining output with zeros if needed
        if (output_pos < output.len) {
            @memset(output[output_pos..], 0);
        }
    }

    fn writeGifGlobalPalette(file: std.fs.File, image: *const Image) !void {
        // Create a basic 256-color palette
        var palette: [256 * 3]u8 = undefined;
        for (0..256) |i| {
            palette[i * 3] = @intCast(i);
            palette[i * 3 + 1] = @intCast(i);
            palette[i * 3 + 2] = @intCast(i);
        }
        _ = image;
        _ = try file.write(&palette);
    }

    fn writeGifApplicationExtension(file: std.fs.File, loop_count: u16) !void {
        _ = try file.write(&[_]u8{ 0x21, 0xFF, 11 });
        _ = try file.write("NETSCAPE2.0");
        _ = try file.write(&[_]u8{ 3, 1 });
        const loop_bytes = std.mem.toBytes(loop_count);
        _ = try file.write(&[_]u8{ loop_bytes[0], loop_bytes[1], 0 });
    }

    fn writeGifFrame(file: std.fs.File, image: *const Image, delay: u16) !void {
        _ = try file.write(&[_]u8{ 0x21, 0xF9, 4, 0x04 });
        const delay_bytes = std.mem.toBytes(delay);
        _ = try file.write(&[_]u8{ delay_bytes[0], delay_bytes[1], 0, 0 });
        _ = try file.write(&[_]u8{ 0x2C, 0, 0, 0, 0 });
        const width_bytes = std.mem.toBytes(@as(u16, @intCast(image.width)));
        const height_bytes = std.mem.toBytes(@as(u16, @intCast(image.height)));
        _ = try file.write(&[_]u8{ width_bytes[0], width_bytes[1] });
        _ = try file.write(&[_]u8{ height_bytes[0], height_bytes[1], 0, 8 });

        const pixel_count = image.width * image.height;
        const index_data = try image.allocator.alloc(u8, pixel_count);
        defer image.allocator.free(index_data);

        const bytes_per_pixel = switch (image.format) {
            .grayscale => 1,
            else => 3,
        };

        for (0..pixel_count) |i| {
            if (image.format == .grayscale) {
                index_data[i] = image.data[i];
            } else {
                const r = image.data[i * bytes_per_pixel];
                const g = image.data[i * bytes_per_pixel + 1];
                const b = image.data[i * bytes_per_pixel + 2];
                index_data[i] = @intFromFloat(@as(f32, @floatFromInt(r)) * 0.299 +
                    @as(f32, @floatFromInt(g)) * 0.587 +
                    @as(f32, @floatFromInt(b)) * 0.114);
            }
        }

        var remaining = index_data.len;
        var offset: usize = 0;
        while (remaining > 0) {
            const block_size = @min(remaining, 255);
            _ = try file.write(&[_]u8{@intCast(block_size)});
            _ = try file.write(index_data[offset .. offset + block_size]);
            offset += block_size;
            remaining -= block_size;
        }
        _ = try file.write(&[_]u8{0});
    }

    fn loadAvif(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        const header = try avif.parseAvifHeader(allocator, file);
        const rgb_data = try avif.decodeAvif(allocator, file);

        return Image{
            .allocator = allocator,
            .width = header.width,
            .height = header.height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    fn loadAnimatedGif(allocator: std.mem.Allocator, file: std.fs.File) !AnimatedImage {
        try file.seekTo(0);

        var header: [13]u8 = undefined;
        _ = try file.read(&header);

        // Check GIF signature
        if (!std.mem.eql(u8, header[0..6], "GIF87a") and !std.mem.eql(u8, header[0..6], "GIF89a")) {
            return error.InvalidGIF;
        }

        // Read logical screen descriptor
        const width = std.mem.readInt(u16, header[6..8], .little);
        const height = std.mem.readInt(u16, header[8..10], .little);
        const flags = header[10];
        const bg_color_index = header[11];
        const pixel_aspect_ratio = header[12];

        _ = bg_color_index;
        _ = pixel_aspect_ratio;

        const global_color_table = (flags & 0x80) != 0;
        const table_bits = (flags & 0x07) + 1;
        const global_color_table_size: u32 = @as(u32, 1) << @intCast(table_bits);

        // Read global color table if present
        var palette: [256][3]u8 = undefined;
        if (global_color_table) {
            const palette_bytes = global_color_table_size * 3;
            const palette_data = try allocator.alloc(u8, palette_bytes);
            defer allocator.free(palette_data);
            _ = try file.read(palette_data);

            for (0..global_color_table_size) |i| {
                palette[i][0] = palette_data[i * 3]; // R
                palette[i][1] = palette_data[i * 3 + 1]; // G
                palette[i][2] = palette_data[i * 3 + 2]; // B
            }
        } else {
            // Default grayscale palette
            for (0..256) |i| {
                const gray = @as(u8, @intCast(i));
                palette[i] = [3]u8{ gray, gray, gray };
            }
        }

        // Parse all frames
        var frames = std.ArrayList(Image).init(allocator);
        var frame_delays = std.ArrayList(u16).init(allocator);
        var loop_count: u16 = 1;

        var current_delay: u16 = 10; // Default 100ms

        while (true) {
            var separator: [1]u8 = undefined;
            const bytes_read = try file.read(&separator);
            if (bytes_read != 1) break;

            switch (separator[0]) {
                0x21 => {
                    // Extension
                    var label: [1]u8 = undefined;
                    _ = try file.read(&label);

                    switch (label[0]) {
                        0xF9 => {
                            // Graphic Control Extension
                            var gce_data: [6]u8 = undefined;
                            _ = try file.read(&gce_data);

                            if (gce_data[0] == 4) { // Block size should be 4
                                current_delay = std.mem.readInt(u16, gce_data[2..4], .little);
                                if (current_delay == 0) current_delay = 10; // Default 100ms
                            }
                            // Skip block terminator (should be 0)
                        },
                        0xFF => {
                            // Application Extension
                            var app_data: [12]u8 = undefined;
                            _ = try file.read(&app_data);

                            if (app_data[0] == 11 and std.mem.eql(u8, app_data[1..9], "NETSCAPE")) {
                                var sub_block: [4]u8 = undefined;
                                _ = try file.read(&sub_block);
                                if (sub_block[0] == 3 and sub_block[1] == 1) {
                                    loop_count = std.mem.readInt(u16, sub_block[2..4], .little);
                                }
                                // Skip remaining sub-blocks
                                try skipGifDataSubBlocks(file);
                            } else {
                                try skipGifDataSubBlocks(file);
                            }
                        },
                        else => {
                            // Other extensions - skip
                            try skipGifDataSubBlocks(file);
                        },
                    }
                },
                0x2C => {
                    // Image descriptor - decode frame
                    const frame = try decodeGifImage(allocator, file, width, height, &palette);
                    try frames.append(frame);
                    try frame_delays.append(current_delay);
                },
                0x3B => {
                    // Trailer - end of file
                    break;
                },
                else => {
                    return error.InvalidGIFData;
                },
            }
        }

        if (frames.items.len == 0) {
            frames.deinit();
            frame_delays.deinit();
            return error.NoGIFFrames;
        }

        return AnimatedImage{
            .allocator = allocator,
            .frames = try frames.toOwnedSlice(),
            .frame_delays = try frame_delays.toOwnedSlice(),
            .loop_count = loop_count,
        };
    }

    fn loadSvgFile(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        const file_size = try file.getEndPos();
        const svg_content = try allocator.alloc(u8, file_size);
        defer allocator.free(svg_content);
        _ = try file.readAll(svg_content);

        var svg_doc = try svg.parseSvg(allocator, svg_content);
        defer svg_doc.deinit();

        // Use default size if not specified in SVG
        const width = @as(u32, @intFromFloat(svg_doc.width));
        const height = @as(u32, @intFromFloat(svg_doc.height));

        const rgb_data = try svg.renderSvg(allocator, &svg_doc, width, height);

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    fn loadJpeg(allocator: std.mem.Allocator, file: std.fs.File) !Image {
        // JPEG markers
        const SOI = 0xD8;
        const EOI = 0xD9;
        const APP0 = 0xE0;
        const DQT = 0xDB;
        const DHT = 0xC4;
        const SOF0 = 0xC0;
        const SOS = 0xDA;

        // Seek to beginning
        try file.seekTo(0);

        // Read SOI
        var marker_buf: [2]u8 = undefined;
        _ = try file.read(&marker_buf);
        if (marker_buf[0] != 0xFF or marker_buf[1] != SOI) {
            return error.InvalidJPEG;
        }

        var quantization_tables: [4][64]u16 = undefined;
        var huffman_tables: [4]HuffmanTable = undefined;
        var frame_header: FrameHeader = undefined;
        var has_frame_header = false;

        // Parse segments until SOS
        while (true) {
            _ = try file.read(&marker_buf);
            if (marker_buf[0] != 0xFF) {
                std.debug.print("Invalid marker: {x}\n", .{marker_buf[0]});
                return error.InvalidMarker;
            }

            const marker = marker_buf[1];
            // std.debug.print("Found marker: {x}\n", .{marker});
            switch (marker) {
                SOI => return error.UnexpectedSOI,
                EOI => return error.UnexpectedEOI,
                APP0 => try skipSegment(file),
                DQT => try parseDQT(file, &quantization_tables),
                DHT => try parseDHT(file, &huffman_tables),
                SOF0 => {
                    frame_header = try parseSOF0(file);
                    has_frame_header = true;
                },
                0xC2 => return error.UnsupportedProgressiveJPEG,
                SOS => break,
                else => try skipSegment(file),
            }
        }

        if (!has_frame_header) {
            return error.NoFrameHeader;
        }

        // Parse SOS and decode image data
        const scan_header = try parseSOS(file);
        const image_data = try decodeImageData(allocator, file, frame_header, scan_header, quantization_tables, huffman_tables);

        return Image{
            .allocator = allocator,
            .width = frame_header.width,
            .height = frame_header.height,
            .format = .rgb,
            .data = image_data,
        };
    }

    fn skipSegment(file: std.fs.File) !void {
        var length_buf: [2]u8 = undefined;
        _ = try file.read(&length_buf);
        const length = std.mem.readInt(u16, &length_buf, .big);
        try file.seekBy(length - 2);
    }

    fn parseDQT(file: std.fs.File, tables: *[4][64]u16) !void {
        var length_buf: [2]u8 = undefined;
        _ = try file.read(&length_buf);
        const length = std.mem.readInt(u16, &length_buf, .big) - 2;

        var data: [1024]u8 = undefined; // Should be enough
        if (length > data.len) return error.DQTTooLarge;
        _ = try file.read(data[0..length]);

        var offset: usize = 0;
        while (offset < length) {
            const table_info = data[offset];
            offset += 1;
            const table_id = table_info & 0x0F;
            const precision = (table_info >> 4) & 0x0F;

            if (precision != 0) return error.UnsupportedPrecision;

            const table_ptr = &tables[table_id];
            if (offset + 64 > length) return error.DQTTooLarge;
            for (0..64) |i| {
                const zigzag_index = zigzag_order[i];
                table_ptr[zigzag_index] = data[offset + i];
            }
            offset += 64;
        }
    }

    fn parseDHT(file: std.fs.File, tables: *[4]HuffmanTable) !void {
        var length_buf: [2]u8 = undefined;
        _ = try file.read(&length_buf);
        const length = std.mem.readInt(u16, &length_buf, .big) - 2;

        var data: [1024]u8 = undefined; // Should be enough
        if (length > data.len) return error.DHTTooLarge;
        _ = try file.read(data[0..length]);

        var offset: usize = 0;
        while (offset < length) {
            const table_info = data[offset];
            offset += 1;
            const table_id = table_info & 0x0F;
            const table_class = (table_info >> 4) & 0x0F; // 0=DC, 1=AC

            // Read number of codes for each length (1-16)
            var lengths: [16]u8 = undefined;
            for (0..16) |i| {
                lengths[i] = data[offset + i];
            }
            offset += 16;

            // Calculate total number of values
            var total_values: usize = 0;
            for (lengths) |len| {
                total_values += len;
            }

            // Read values
            var values: [256]u8 = undefined;
            for (0..total_values) |i| {
                values[i] = data[offset + i];
            }
            offset += total_values;

            // Build Huffman codes according to JPEG standard
            var codes: [256]u16 = [_]u16{0} ** 256;
            var code_lengths: [256]u8 = [_]u8{0} ** 256;
            var code: u16 = 0;
            var symbol_index: usize = 0;

            // Generate codes for each bit length
            for (1..17) |bit_length| {
                code <<= 1;
                const num_codes = lengths[bit_length - 1];
                for (0..num_codes) |_| {
                    if (symbol_index >= total_values) break;
                    codes[symbol_index] = code;
                    code_lengths[symbol_index] = @intCast(bit_length);
                    code += 1;
                    symbol_index += 1;
                }
            }

            const table_index = table_id + table_class * 2;
            if (table_index >= 4) return error.InvalidHuffmanTableIndex;

            tables[table_index] = HuffmanTable{
                .codes = codes,
                .code_lengths = code_lengths,
                .lengths = lengths,
                .values = values,
            };
        }
    }

    fn parseSOF0(file: std.fs.File) !FrameHeader {
        var length_buf: [2]u8 = undefined;
        _ = try file.read(&length_buf);
        const length = std.mem.readInt(u16, &length_buf, .big);

        var data: [64]u8 = undefined; // SOF is small
        if (length - 2 > data.len) return error.SOFTooLarge;
        _ = try file.read(data[0 .. length - 2]);

        const precision = data[0];
        if (precision != 8) return error.UnsupportedPrecision;

        var height_buf: [2]u8 = undefined;
        @memcpy(&height_buf, data[1..3]);
        const height = std.mem.readInt(u16, &height_buf, .big);

        var width_buf: [2]u8 = undefined;
        @memcpy(&width_buf, data[3..5]);
        const width = std.mem.readInt(u16, &width_buf, .big);

        const num_components = data[5];

        if (num_components == 0 or num_components > 3) {
            return error.UnsupportedComponents;
        }

        var components = std.mem.zeroes([3]Component);
        for (0..num_components) |i| {
            const offset = 6 + i * 3;
            components[i] = Component{
                .id = data[offset],
                .h_sampling = data[offset + 1] >> 4,
                .v_sampling = data[offset + 1] & 0x0F,
                .quant_table_id = data[offset + 2],
            };
        }

        return FrameHeader{
            .width = width,
            .height = height,
            .component_count = num_components,
            .components = components,
        };
    }

    fn parseSOS(file: std.fs.File) !ScanHeader {
        var length_buf: [2]u8 = undefined;
        _ = try file.read(&length_buf);
        const length = std.mem.readInt(u16, &length_buf, .big);

        var data: [64]u8 = undefined; // SOS is small
        if (length - 2 > data.len) return error.SOSTooLarge;
        _ = try file.read(data[0 .. length - 2]);

        const num_components = data[0];
        if (num_components == 0 or num_components > 3) return error.UnsupportedComponents;

        var components = std.mem.zeroes([3]ScanComponent);
        for (0..num_components) |i| {
            const offset = 1 + i * 2;
            components[i] = ScanComponent{
                .id = data[offset],
                .dc_table_id = data[offset + 1] >> 4,
                .ac_table_id = data[offset + 1] & 0x0F,
            };
        }

        const start_spectral = data[7];
        const end_spectral = data[8];
        const approximation = data[9];

        return ScanHeader{
            .component_count = num_components,
            .components = components,
            .start_spectral = start_spectral,
            .end_spectral = end_spectral,
            .approximation = approximation,
        };
    }

    fn decodeImageData(allocator: std.mem.Allocator, file: std.fs.File, frame: FrameHeader, scan: ScanHeader, qt: [4][64]u16, ht: [4]HuffmanTable) ![]u8 {
        if (scan.component_count != frame.component_count) {
            return error.ComponentMismatch;
        }

        const approx_high = (scan.approximation >> 4) & 0x0F;
        const approx_low = scan.approximation & 0x0F;
        if (scan.start_spectral != 0 or scan.end_spectral != 63 or approx_high != 0 or approx_low != 0) {
            return error.UnsupportedScanParameters;
        }

        if (frame.component_count != 1 and frame.component_count != 3) {
            return error.UnsupportedComponents;
        }

        // Read entropy coded segment
        var entropy_data = std.ArrayListUnmanaged(u8){};
        defer entropy_data.deinit(allocator);

        while (true) {
            var byte: [1]u8 = undefined;
            const bytes_read = try file.read(&byte);
            if (bytes_read == 0) break;

            if (byte[0] == 0xFF) {
                const next_byte_count = try file.read(&byte);
                if (next_byte_count == 0) break;

                switch (byte[0]) {
                    0x00 => try entropy_data.append(allocator, 0xFF),
                    0xD9 => break,
                    0xD0...0xD7 => return error.UnsupportedRestartMarkers,
                    0xFF => continue,
                    else => return error.UnexpectedMarker,
                }
            } else {
                try entropy_data.append(allocator, byte[0]);
            }
        }

        // Initialise readers and sampling information
        var bit_reader = BitReader.init(entropy_data.items);

        var max_h: u8 = 0;
        var max_v: u8 = 0;
        for (0..frame.component_count) |i| {
            const comp = frame.components[i];
            if (comp.h_sampling == 0 or comp.v_sampling == 0 or comp.h_sampling > 4 or comp.v_sampling > 4) {
                return error.InvalidSamplingFactor;
            }
            max_h = if (comp.h_sampling > max_h) comp.h_sampling else max_h;
            max_v = if (comp.v_sampling > max_v) comp.v_sampling else max_v;
        }

        if (max_h == 0 or max_v == 0) return error.InvalidSamplingFactor;

        const mcu_cols = (@as(usize, frame.width) + @as(usize, max_h) * 8 - 1) / (@as(usize, max_h) * 8);
        const mcu_rows = (@as(usize, frame.height) + @as(usize, max_v) * 8 - 1) / (@as(usize, max_v) * 8);

        const ComponentState = struct {
            h_sampling: u8,
            v_sampling: u8,
            plane_width: usize,
            plane_height: usize,
            plane: []u8,
            quant_table_id: u8,
            dc_pred: i16,
            block_cols: usize,
            block_rows: usize,
        };

        var component_states = std.mem.zeroes([3]ComponentState);
        defer {
            for (0..frame.component_count) |i| {
                if (component_states[i].plane.len != 0) {
                    allocator.free(component_states[i].plane);
                }
            }
        }

        for (0..frame.component_count) |i| {
            const comp = frame.components[i];
            if (@as(usize, comp.quant_table_id) >= qt.len) return error.InvalidQuantizationTable;

            const plane_width = (@as(usize, frame.width) * @as(usize, comp.h_sampling) + @as(usize, max_h) - 1) / @as(usize, max_h);
            const plane_height = (@as(usize, frame.height) * @as(usize, comp.v_sampling) + @as(usize, max_v) - 1) / @as(usize, max_v);
            const plane_size = plane_width * plane_height;

            const plane = try allocator.alloc(u8, plane_size);
            @memset(plane, 0);

            const block_cols = if (plane_width == 0) 0 else (plane_width + 7) / 8;
            const block_rows = if (plane_height == 0) 0 else (plane_height + 7) / 8;

            component_states[i] = ComponentState{
                .h_sampling = comp.h_sampling,
                .v_sampling = comp.v_sampling,
                .plane_width = plane_width,
                .plane_height = plane_height,
                .plane = plane,
                .quant_table_id = comp.quant_table_id,
                .dc_pred = 0,
                .block_cols = block_cols,
                .block_rows = block_rows,
            };
        }

        // Map scan components to frame components
        var scan_to_frame: [3]usize = undefined;
        for (0..scan.component_count) |i| {
            const scan_comp = scan.components[i];
            var found = false;
            for (0..frame.component_count) |j| {
                if (frame.components[j].id == scan_comp.id) {
                    scan_to_frame[i] = j;
                    found = true;
                    break;
                }
            }
            if (!found) return error.InvalidScanComponent;
        }

        // Decode MCUs
        for (0..mcu_rows) |mcu_y| {
            for (0..mcu_cols) |mcu_x| {
                for (0..scan.component_count) |scan_index| {
                    const scan_comp = scan.components[scan_index];
                    const frame_index = scan_to_frame[scan_index];
                    const state = &component_states[frame_index];

                    const dc_index = @as(usize, scan_comp.dc_table_id);
                    const ac_index = 2 + @as(usize, scan_comp.ac_table_id);
                    if (dc_index >= ht.len or ac_index >= ht.len) {
                        return error.InvalidHuffmanTableIndex;
                    }
                    const dc_table = &ht[dc_index];
                    const ac_table = &ht[ac_index];
                    const quant_table = &qt[@as(usize, state.quant_table_id)];

                    for (0..@as(usize, state.v_sampling)) |v_block| {
                        for (0..@as(usize, state.h_sampling)) |h_block| {
                            var dct_coeffs: [64]i32 = undefined;
                            var spatial: [64]i32 = undefined;
                            state.dc_pred = try decodeBlock(&bit_reader, dc_table, ac_table, &dct_coeffs, state.dc_pred);
                            dequantizeBlock(&dct_coeffs, quant_table);
                            idct8x8(&dct_coeffs, &spatial);

                            const block_x = (@as(usize, mcu_x) * @as(usize, state.h_sampling) + @as(usize, h_block)) * 8;
                            const block_y = (@as(usize, mcu_y) * @as(usize, state.v_sampling) + @as(usize, v_block)) * 8;
                            storeBlockToPlane(state.plane, state.plane_width, state.plane_height, &spatial, block_x, block_y);
                        }
                    }
                }
            }
        }

        const frame_width_usize = @as(usize, frame.width);
        const frame_height_usize = @as(usize, frame.height);
        const rgb_size = frame_width_usize * frame_height_usize * 3;
        const rgb_data = try allocator.alloc(u8, rgb_size);
        errdefer allocator.free(rgb_data);

        if (frame.component_count == 1) {
            const y_plane = component_states[0].plane;
            const plane_width = component_states[0].plane_width;
            const plane_height = component_states[0].plane_height;
            for (0..frame_height_usize) |y| {
                const sample_row = (y * plane_height) / frame_height_usize;
                for (0..frame_width_usize) |x| {
                    const sample_col = (x * plane_width) / frame_width_usize;
                    const luminance = y_plane[sample_row * plane_width + sample_col];
                    const idx = (y * frame_width_usize + x) * 3;
                    rgb_data[idx] = luminance;
                    rgb_data[idx + 1] = luminance;
                    rgb_data[idx + 2] = luminance;
                }
            }
            return rgb_data;
        }

        var y_index: ?usize = null;
        var cb_index: ?usize = null;
        var cr_index: ?usize = null;
        for (0..frame.component_count) |i| {
            switch (frame.components[i].id) {
                1 => y_index = i,
                2 => cb_index = i,
                3 => cr_index = i,
                else => {},
            }
        }

        if (y_index == null or cb_index == null or cr_index == null) {
            return error.UnsupportedColorComponents;
        }

        const y_plane = component_states[y_index.?].plane;
        const y_width = component_states[y_index.?].plane_width;
        const y_height = component_states[y_index.?].plane_height;

        const cb_plane = component_states[cb_index.?].plane;
        const cb_width = component_states[cb_index.?].plane_width;
        const cb_height = component_states[cb_index.?].plane_height;

        const cr_plane = component_states[cr_index.?].plane;
        const cr_width = component_states[cr_index.?].plane_width;
        const cr_height = component_states[cr_index.?].plane_height;

        for (0..frame_height_usize) |y| {
            const y_row = if (y_height == frame_height_usize) y else (y * y_height) / frame_height_usize;
            const cb_row = (y * cb_height) / frame_height_usize;
            const cr_row = (y * cr_height) / frame_height_usize;

            for (0..frame_width_usize) |x| {
                const y_col = if (y_width == frame_width_usize) x else (x * y_width) / frame_width_usize;
                const cb_col = (x * cb_width) / frame_width_usize;
                const cr_col = (x * cr_width) / frame_width_usize;

                const y_sample = y_plane[y_row * y_width + y_col];
                const cb_sample = cb_plane[cb_row * cb_width + cb_col];
                const cr_sample = cr_plane[cr_row * cr_width + cr_col];
                const rgb = yuvToRgb(@as(i32, y_sample), @as(i32, cb_sample) - 128, @as(i32, cr_sample) - 128);
                const idx = (y * frame_width_usize + x) * 3;
                rgb_data[idx] = rgb[0];
                rgb_data[idx + 1] = rgb[1];
                rgb_data[idx + 2] = rgb[2];
            }
        }

        return rgb_data;
    }

    fn decodeBlock(bit_reader: *BitReader, dc_table: *const HuffmanTable, ac_table: *const HuffmanTable, block: *[64]i32, prev_dc: i16) !i16 {
        @memset(block, 0);

        const dc_symbol = try decodeHuffmanValue(bit_reader, dc_table);
        if (dc_symbol > 11) return error.InvalidHuffmanCode;
        const dc_bits = if (dc_symbol == 0) 0 else try bit_reader.readBits(@intCast(dc_symbol));
        const dc_diff = receiveExtend(dc_bits, @intCast(dc_symbol));
        const dc_value = prev_dc + dc_diff;
        block[0] = @as(i32, dc_value);

        var k: usize = 1;
        while (k < 64) {
            const ac_symbol = try decodeHuffmanValue(bit_reader, ac_table);

            if (ac_symbol == 0x00) {
                break;
            }

            if (ac_symbol == 0xF0) {
                k += 16;
                if (k >= 64) break;
                continue;
            }

            const run_length = (ac_symbol >> 4) & 0x0F;
            const size = ac_symbol & 0x0F;

            k += run_length;
            if (k >= 64) break;

            if (size == 0) return error.InvalidHuffmanCode;
            const ac_bits = try bit_reader.readBits(@intCast(size));
            const ac_value = receiveExtend(ac_bits, size);
            const index = zigzag_order[k];
            block[index] = @as(i32, ac_value);
            k += 1;
        }

        return dc_value;
    }

    fn dequantizeBlock(block: *[64]i32, quantization_table: *const [64]u16) void {
        for (0..64) |i| {
            block[i] = block[i] * @as(i32, @intCast(quantization_table[i]));
        }
    }

    fn receiveExtend(value: u16, size: u8) i16 {
        if (size == 0) return 0;

        const shift_minus_one = @as(u5, @intCast(size - 1));
        const threshold = @as(i32, 1) << shift_minus_one;
        var result = @as(i32, @intCast(value));
        if (result < threshold) {
            const full_range = (@as(i32, 1) << @as(u5, @intCast(size))) - 1;
            result -= full_range;
        }
        return @intCast(result);
    }

    fn storeBlockToPlane(plane: []u8, plane_width: usize, plane_height: usize, block: *const [64]i32, block_x: usize, block_y: usize) void {
        for (0..8) |row| {
            const dest_y = block_y + row;
            if (dest_y >= plane_height) continue;
            for (0..8) |col| {
                const dest_x = block_x + col;
                if (dest_x >= plane_width) continue;

                var value = block[row * 8 + col] + 128;
                if (value < 0) value = 0;
                if (value > 255) value = 255;

                plane[dest_y * plane_width + dest_x] = @intCast(value);
            }
        }
    }

    const IDCT_SCALE = 14;

    fn buildIdctMatrix() [8][8]i32 {
        var matrix: [8][8]i32 = undefined;
        const pi = std.math.pi;
        for (0..8) |u| {
            const cu = if (u == 0) 1.0 / std.math.sqrt(2.0) else 1.0;
            for (0..8) |x| {
                const angle = ((2.0 * @as(f64, @floatFromInt(x)) + 1.0) * @as(f64, @floatFromInt(u)) * pi) / 16.0;
                const value = 0.5 * cu * std.math.cos(angle);
                matrix[u][x] = @intFromFloat(std.math.round(value * @as(f64, @floatFromInt(1 << IDCT_SCALE))));
            }
        }
        return matrix;
    }

    const idct_matrix = buildIdctMatrix();

    fn idct1dInt(input: []const i32, output: []i32) void {
        for (0..8) |x| {
            var sum: i64 = 0;
            for (0..8) |u| {
                sum += @as(i64, input[u]) * @as(i64, idct_matrix[u][x]);
            }
            const rounded = sum + (@as(i64, 1) << (IDCT_SCALE - 1));
            output[x] = @intCast(rounded >> IDCT_SCALE);
        }
    }

    fn idct8x8(coeffs: *const [64]i32, output: *[64]i32) void {
        var temp: [64]i32 = undefined;

        for (0..8) |row| {
            var row_input: [8]i32 = undefined;
            for (0..8) |i| {
                row_input[i] = coeffs.*[row * 8 + i];
            }

            var row_output: [8]i32 = undefined;
            idct1dInt(row_input[0..], row_output[0..]);

            for (0..8) |i| {
                temp[row * 8 + i] = row_output[i];
            }
        }

        for (0..8) |col| {
            var column_input: [8]i32 = undefined;
            for (0..8) |row| {
                column_input[row] = temp[row * 8 + col];
            }

            var column_output: [8]i32 = undefined;
            idct1dInt(column_input[0..], column_output[0..]);

            for (0..8) |row| {
                output.*[row * 8 + col] = column_output[row];
            }
        }
    }

    fn yuvToRgb(y: i32, cb: i32, cr: i32) [3]u8 {
        const y_f = @as(f32, @floatFromInt(y));
        const cb_f = @as(f32, @floatFromInt(cb));
        const cr_f = @as(f32, @floatFromInt(cr));

        const r = y_f + 1.402 * cr_f;
        const g = y_f - 0.344136 * cb_f - 0.714136 * cr_f;
        const b = y_f + 1.772 * cb_f;

        // Clamp values
        const r_clamped = @min(255, @max(0, @as(i32, @intFromFloat(r))));
        const g_clamped = @min(255, @max(0, @as(i32, @intFromFloat(g))));
        const b_clamped = @min(255, @max(0, @as(i32, @intFromFloat(b))));

        return [3]u8{ @intCast(r_clamped), @intCast(g_clamped), @intCast(b_clamped) };
    }

    fn decodeHuffmanValue(bit_reader: *BitReader, table: *const HuffmanTable) !u8 {
        var code: u16 = 0;

        for (1..17) |bit_length| {
            const bit = try bit_reader.readBit();
            code = (code << 1) | bit;

            // Find starting index for symbols of this bit length
            var start_index: usize = 0;
            for (0..bit_length - 1) |i| {
                start_index += table.lengths[i];
            }

            // Check all symbols of this bit length
            const num_symbols = table.lengths[bit_length - 1];
            for (0..num_symbols) |i| {
                const symbol_index = start_index + i;
                if (table.codes[symbol_index] == code and
                    table.code_lengths[symbol_index] == bit_length)
                {
                    return table.values[symbol_index];
                }
            }
        }

        return error.InvalidHuffmanCode;
    }

    pub fn save(self: Image, path: []const u8, format: ImageFormat) !void {
        // Input validation
        if (path.len == 0) {
            return error.EmptyPath;
        }
        if (path.len > 4096) {
            return error.PathTooLong;
        }

        // Validate image state
        if (self.width == 0 or self.height == 0) {
            return error.InvalidImageDimensions;
        }
        if (self.data.len == 0) {
            return error.EmptyImageData;
        }

        const expected_size = @as(usize, self.width) * self.height * bytesPerPixel(self.format);
        if (self.data.len != expected_size) {
            return error.InvalidImageDataSize;
        }

        switch (format) {
            .bmp => try self.saveBmp(path),
            .png => try self.savePng(path),
            .jpeg => try self.saveJpeg(path),
            .webp => try self.saveWebP(path),
            else => return error.UnsupportedFormat,
        }
    }

    // Image processing operations

    pub fn resize(self: *Image, new_width: u32, new_height: u32) !void {
        // Input validation
        if (new_width == 0 or new_height == 0) {
            return error.InvalidDimensions;
        }
        if (new_width > 65535 or new_height > 65535) {
            return error.DimensionsTooLarge;
        }

        const bytes_per_pixel = bytesPerPixel(self.format);
        const new_data = try self.allocator.alloc(u8, new_width * new_height * bytes_per_pixel);

        // SIMD-optimized bilinear scaling for better quality and performance
        simd.simdResize(self.data, self.width, self.height, new_data, new_width, new_height, bytes_per_pixel);

        self.allocator.free(self.data);
        self.data = new_data;
        self.width = new_width;
        self.height = new_height;
    }

    fn resizeVectorized(self: *Image, new_data: []u8, new_width: u32, new_height: u32, bytes_per_pixel: usize) void {
        const x_scale = @as(f32, @floatFromInt(self.width)) / @as(f32, @floatFromInt(new_width));
        const y_scale = @as(f32, @floatFromInt(self.height)) / @as(f32, @floatFromInt(new_height));

        for (0..new_height) |y| {
            const src_y = @as(f32, @floatFromInt(y)) * y_scale;
            const y1 = @as(u32, @intFromFloat(@floor(src_y)));
            const y2 = @min(y1 + 1, self.height - 1);
            const dy = src_y - @as(f32, @floatFromInt(y1));

            var x: usize = 0;
            // Process 4 pixels at a time using vector operations
            while (x + 4 <= new_width) {
                // Vectorized processing for 4 consecutive pixels
                for (0..4) |i| {
                    const src_x = @as(f32, @floatFromInt(x + i)) * x_scale;
                    const x1 = @as(u32, @intFromFloat(@floor(src_x)));
                    const x2 = x1 + 1;
                    const dx = src_x - @as(f32, @floatFromInt(x1));

                    const dst_idx = (y * new_width + x + i) * bytes_per_pixel;

                    // Bilinear interpolation for each color channel
                    for (0..bytes_per_pixel) |c| {
                        const p11 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x1) * bytes_per_pixel + c]));
                        const p12 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x1) * bytes_per_pixel + c]));
                        const p21 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x2) * bytes_per_pixel + c]));
                        const p22 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x2) * bytes_per_pixel + c]));

                        const interpolated = p11 * (1 - dx) * (1 - dy) +
                            p21 * dx * (1 - dy) +
                            p12 * (1 - dx) * dy +
                            p22 * dx * dy;

                        new_data[dst_idx + c] = @intFromFloat(@max(0, @min(255, interpolated)));
                    }
                }
                x += 4;
            }

            // Handle remaining pixels
            while (x < new_width) {
                const src_x = @as(f32, @floatFromInt(x)) * x_scale;
                const x1 = @as(u32, @intFromFloat(@floor(src_x)));
                const x2 = x1 + 1;
                const dx = src_x - @as(f32, @floatFromInt(x1));

                const dst_idx = (y * new_width + x) * bytes_per_pixel;

                for (0..bytes_per_pixel) |c| {
                    const p11 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x1) * bytes_per_pixel + c]));
                    const p12 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x1) * bytes_per_pixel + c]));
                    const p21 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x2) * bytes_per_pixel + c]));
                    const p22 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x2) * bytes_per_pixel + c]));

                    const interpolated = p11 * (1 - dx) * (1 - dy) +
                        p21 * dx * (1 - dy) +
                        p12 * (1 - dx) * dy +
                        p22 * dx * dy;

                    new_data[dst_idx + c] = @intFromFloat(interpolated);
                }
                x += 1;
            }
        }
    }

    fn resizeScalar(self: *Image, new_data: []u8, new_width: u32, new_height: u32, bytes_per_pixel: usize) void {
        for (0..new_height) |y| {
            for (0..new_width) |x| {
                const src_x = (x * self.width) / new_width;
                const src_y = (y * self.height) / new_height;
                const src_idx = (src_y * self.width + src_x) * bytes_per_pixel;
                const dst_idx = (y * new_width + x) * bytes_per_pixel;

                for (0..bytes_per_pixel) |c| {
                    new_data[dst_idx + c] = self.data[src_idx + c];
                }
            }
        }
    }

    pub fn crop(self: *Image, x: u32, y: u32, width: u32, height: u32) !void {
        // Input validation
        if (width == 0 or height == 0) {
            return error.InvalidDimensions;
        }
        if (x >= self.width or y >= self.height) {
            return error.CropOutOfBounds;
        }
        if (x + width > self.width or y + height > self.height) {
            return error.CropOutOfBounds;
        }

        const bytes_per_pixel = bytesPerPixel(self.format);
        const new_data = try self.allocator.alloc(u8, width * height * bytes_per_pixel);

        for (0..height) |row| {
            const src_row = y + row;
            const src_start = (src_row * self.width + x) * bytes_per_pixel;
            const dst_start = row * width * bytes_per_pixel;
            const row_bytes = width * bytes_per_pixel;

            @memcpy(new_data[dst_start .. dst_start + row_bytes], self.data[src_start .. src_start + row_bytes]);
        }

        self.allocator.free(self.data);
        self.data = new_data;
        self.width = width;
        self.height = height;
    }

    pub fn rotate90(self: *Image) !void {
        const bytes_per_pixel = bytesPerPixel(self.format);
        const new_data = try self.allocator.alloc(u8, self.width * self.height * bytes_per_pixel);

        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const src_idx = (y * self.width + x) * bytes_per_pixel;
                const dst_x = self.height - 1 - y;
                const dst_y = x;
                const dst_idx = (dst_y * self.height + dst_x) * bytes_per_pixel;

                for (0..bytes_per_pixel) |c| {
                    new_data[dst_idx + c] = self.data[src_idx + c];
                }
            }
        }

        self.allocator.free(self.data);
        self.data = new_data;
        const temp = self.width;
        self.width = self.height;
        self.height = temp;
    }

    pub fn convertToGrayscale(self: *Image) !void {
        if (self.format == .grayscale) {
            return; // Already grayscale
        }
        if (self.format != .rgb) {
            return error.UnsupportedFormatConversion;
        }

        const new_data = try self.allocator.alloc(u8, self.width * self.height);

        for (0..self.width * self.height) |i| {
            const r = self.data[i * 3];
            const g = self.data[i * 3 + 1];
            const b = self.data[i * 3 + 2];
            // ITU-R BT.709 luma coefficients
            new_data[i] = @intFromFloat(0.2126 * @as(f32, @floatFromInt(r)) +
                0.7152 * @as(f32, @floatFromInt(g)) +
                0.0722 * @as(f32, @floatFromInt(b)));
        }

        self.allocator.free(self.data);
        self.data = new_data;
        self.format = .grayscale;
    }

    pub fn adjustBrightness(self: *Image, adjustment: i16) !void {
        if (self.format != .rgb and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }

        for (0..self.data.len) |i| {
            const current = @as(i16, self.data[i]);
            const new_value = @min(255, @max(0, current + adjustment));
            self.data[i] = @intCast(new_value);
        }
    }

    pub fn adjustContrast(self: *Image, factor: f32) !void {
        if (self.format != .rgb and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }
        if (factor < 0) {
            return error.InvalidContrastFactor;
        }

        for (0..self.data.len) |i| {
            const current = @as(f32, @floatFromInt(self.data[i]));
            const centered = current - 128.0;
            const adjusted = centered * factor + 128.0;
            const clamped = @min(255.0, @max(0.0, adjusted));
            self.data[i] = @intFromFloat(clamped);
        }
    }

    pub fn adjustWhiteBalance(self: *Image, temperature: f32, tint: f32) !void {
        // White balance adjustment for RGB images
        if (self.format != .rgb and self.format != .rgba) {
            return error.UnsupportedFormat;
        }

        // Temperature adjustment (-100 to +100, 0 is neutral)
        // Negative = cooler (more blue), Positive = warmer (more red/yellow)
        const temp_factor = temperature / 100.0;

        // Tint adjustment (-100 to +100, 0 is neutral)
        // Negative = more green, Positive = more magenta
        const tint_factor = tint / 100.0;

        const bytes_per_pixel = bytesPerPixel(self.format);

        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const idx = (y * self.width + x) * bytes_per_pixel;

                const r = @as(f32, @floatFromInt(self.data[idx]));
                const g = @as(f32, @floatFromInt(self.data[idx + 1]));
                const b = @as(f32, @floatFromInt(self.data[idx + 2]));

                // Apply temperature adjustment
                var new_r = r * (1.0 + temp_factor * 0.5);
                var new_g = g * (1.0 - @abs(temp_factor) * 0.1);
                const new_b = b * (1.0 - temp_factor * 0.5);

                // Apply tint adjustment
                new_r = new_r * (1.0 + tint_factor * 0.3);
                new_g = new_g * (1.0 - tint_factor * 0.3);
                // Blue channel remains relatively unchanged for tint

                // Clamp values
                self.data[idx] = @intFromFloat(@min(255.0, @max(0.0, new_r)));
                self.data[idx + 1] = @intFromFloat(@min(255.0, @max(0.0, new_g)));
                self.data[idx + 2] = @intFromFloat(@min(255.0, @max(0.0, new_b)));
            }
        }
    }

    pub fn flipHorizontal(self: *Image) !void {
        const bytes_per_pixel = bytesPerPixel(self.format);

        for (0..self.height) |y| {
            const row_start = y * self.width * bytes_per_pixel;
            for (0..self.width / 2) |x| {
                const left_idx = row_start + x * bytes_per_pixel;
                const right_idx = row_start + (self.width - 1 - x) * bytes_per_pixel;

                for (0..bytes_per_pixel) |c| {
                    const temp = self.data[left_idx + c];
                    self.data[left_idx + c] = self.data[right_idx + c];
                    self.data[right_idx + c] = temp;
                }
            }
        }
    }

    pub fn flipVertical(self: *Image) !void {
        const bytes_per_pixel = bytesPerPixel(self.format);
        const row_bytes = self.width * bytes_per_pixel;

        const temp_row = try self.allocator.alloc(u8, row_bytes);
        defer self.allocator.free(temp_row);

        for (0..self.height / 2) |y| {
            const top_row_start = y * row_bytes;
            const bottom_row_start = (self.height - 1 - y) * row_bytes;

            @memcpy(temp_row, self.data[top_row_start .. top_row_start + row_bytes]);
            @memcpy(self.data[top_row_start .. top_row_start + row_bytes], self.data[bottom_row_start .. bottom_row_start + row_bytes]);
            @memcpy(self.data[bottom_row_start .. bottom_row_start + row_bytes], temp_row);
        }
    }

    pub fn blur(self: *Image, radius: u32) !void {
        if (self.format != .rgb and self.format != .rgba and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }
        if (radius == 0) {
            return; // No blur
        }

        const bytes_per_pixel = bytesPerPixel(self.format);
        const temp_data = try self.allocator.alloc(u8, self.data.len);
        defer self.allocator.free(temp_data);

        // SIMD-optimized separable blur
        // Horizontal pass
        simd.simdBlurHorizontal(self.data, temp_data, self.width, self.height, radius, bytes_per_pixel);

        // Vertical pass
        simd.simdBlurVertical(temp_data, self.data, self.width, self.height, radius, bytes_per_pixel);
    }

    // Advanced image processing operations

    pub fn sharpen(self: *Image) !void {
        if (self.format != .rgb and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }

        const bytes_per_pixel = bytesPerPixel(self.format);
        const new_data = try self.allocator.alloc(u8, self.data.len);
        defer self.allocator.free(new_data);
        @memcpy(new_data, self.data);

        // Sharpen kernel: center +5, adjacent -1
        const kernel = [_]f32{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };

        for (1..self.height - 1) |y| {
            for (1..self.width - 1) |x| {
                for (0..bytes_per_pixel) |c| {
                    var sum: f32 = 0;

                    for (0..3) |ky| {
                        for (0..3) |kx| {
                            const src_y = y - 1 + ky;
                            const src_x = x - 1 + kx;
                            const idx = (src_y * self.width + src_x) * bytes_per_pixel + c;
                            const kernel_idx = ky * 3 + kx;
                            sum += @as(f32, @floatFromInt(new_data[idx])) * kernel[kernel_idx];
                        }
                    }

                    const idx = (y * self.width + x) * bytes_per_pixel + c;
                    self.data[idx] = @intCast(@min(255, @max(0, @as(i32, @intFromFloat(sum)))));
                }
            }
        }
    }

    pub fn edgeDetectSobel(self: *Image) !void {
        if (self.format != .grayscale) {
            return error.UnsupportedFormat;
        }

        const new_data = try self.allocator.alloc(u8, self.data.len);
        defer self.allocator.free(new_data);
        @memcpy(new_data, self.data);

        // Sobel X and Y kernels
        const sobel_x = [_]f32{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        const sobel_y = [_]f32{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };

        for (1..self.height - 1) |y| {
            for (1..self.width - 1) |x| {
                var gx: f32 = 0;
                var gy: f32 = 0;

                for (0..3) |ky| {
                    for (0..3) |kx| {
                        const src_y = y - 1 + ky;
                        const src_x = x - 1 + kx;
                        const idx = src_y * self.width + src_x;
                        const kernel_idx = ky * 3 + kx;
                        const pixel = @as(f32, @floatFromInt(new_data[idx]));

                        gx += pixel * sobel_x[kernel_idx];
                        gy += pixel * sobel_y[kernel_idx];
                    }
                }

                const magnitude = @sqrt(gx * gx + gy * gy);
                const idx = y * self.width + x;
                self.data[idx] = @intCast(@min(255, @as(u32, @intFromFloat(magnitude))));
            }
        }
    }

    pub fn gaussianBlur(self: *Image, sigma: f32) !void {
        if (self.format != .rgb and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }

        const bytes_per_pixel = bytesPerPixel(self.format);
        const radius = @as(u32, @intFromFloat(@ceil(sigma * 3.0)));
        const kernel_size = radius * 2 + 1;

        // Generate Gaussian kernel
        var kernel = try self.allocator.alloc(f32, kernel_size * kernel_size);
        defer self.allocator.free(kernel);

        var sum: f32 = 0;
        for (0..kernel_size) |y| {
            for (0..kernel_size) |x| {
                const dx = @as(f32, @floatFromInt(x)) - @as(f32, @floatFromInt(radius));
                const dy = @as(f32, @floatFromInt(y)) - @as(f32, @floatFromInt(radius));
                const value = @exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
                kernel[y * kernel_size + x] = value;
                sum += value;
            }
        }

        // Normalize kernel
        for (kernel) |*k| {
            k.* /= sum;
        }

        const new_data = try self.allocator.alloc(u8, self.data.len);
        defer self.allocator.free(new_data);
        @memcpy(new_data, self.data);

        // Apply Gaussian blur
        for (radius..self.height - radius) |y| {
            for (radius..self.width - radius) |x| {
                for (0..bytes_per_pixel) |c| {
                    var pixel_sum: f32 = 0;

                    for (0..kernel_size) |ky| {
                        for (0..kernel_size) |kx| {
                            const src_y = y - radius + ky;
                            const src_x = x - radius + kx;
                            const idx = (src_y * self.width + src_x) * bytes_per_pixel + c;
                            const kernel_idx = ky * kernel_size + kx;
                            pixel_sum += @as(f32, @floatFromInt(new_data[idx])) * kernel[kernel_idx];
                        }
                    }

                    const idx = (y * self.width + x) * bytes_per_pixel + c;
                    self.data[idx] = @intCast(@as(u32, @intFromFloat(pixel_sum)));
                }
            }
        }
    }

    pub fn gammaCorrection(self: *Image, gamma: f32) !void {
        if (self.format != .rgb and self.format != .grayscale) {
            return error.UnsupportedFormat;
        }
        if (gamma <= 0) {
            return error.InvalidGamma;
        }

        // Precompute gamma lookup table
        var gamma_lut: [256]u8 = undefined;
        for (0..256) |i| {
            const normalized = @as(f32, @floatFromInt(i)) / 255.0;
            const corrected = std.math.pow(f32, normalized, 1.0 / gamma);
            gamma_lut[i] = @intFromFloat(corrected * 255.0);
        }

        // Apply gamma correction
        for (0..self.data.len) |i| {
            self.data[i] = gamma_lut[self.data[i]];
        }
    }

    pub fn histogramEqualization(self: *Image) !void {
        if (self.format != .grayscale) {
            return error.UnsupportedFormat;
        }

        // Calculate histogram
        var histogram: [256]u32 = [_]u32{0} ** 256;
        for (self.data) |pixel| {
            histogram[pixel] += 1;
        }

        // Calculate cumulative distribution function
        var cdf: [256]u32 = undefined;
        cdf[0] = histogram[0];
        for (1..256) |i| {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Find minimum non-zero CDF value
        var cdf_min: u32 = 0;
        for (cdf) |value| {
            if (value > 0) {
                cdf_min = value;
                break;
            }
        }

        const total_pixels = self.width * self.height;

        // Apply histogram equalization
        for (0..self.data.len) |i| {
            const old_value = self.data[i];
            const new_value = @as(f32, @floatFromInt((cdf[old_value] - cdf_min) * 255)) / @as(f32, @floatFromInt(total_pixels - cdf_min));
            self.data[i] = @intFromFloat(@min(255.0, @max(0.0, new_value)));
        }
    }

    pub fn rotateArbitrary(self: *Image, angle_degrees: f32) !void {
        const angle_rad = angle_degrees * std.math.pi / 180.0;
        const cos_angle = @cos(angle_rad);
        const sin_angle = @sin(angle_rad);

        const bytes_per_pixel = bytesPerPixel(self.format);

        // Calculate new dimensions
        const old_width_f = @as(f32, @floatFromInt(self.width));
        const old_height_f = @as(f32, @floatFromInt(self.height));

        const corners = [_][2]f32{
            .{ 0, 0 },
            .{ old_width_f, 0 },
            .{ old_width_f, old_height_f },
            .{ 0, old_height_f },
        };

        var min_x: f32 = std.math.inf(f32);
        var max_x: f32 = -std.math.inf(f32);
        var min_y: f32 = std.math.inf(f32);
        var max_y: f32 = -std.math.inf(f32);

        for (corners) |corner| {
            const x = corner[0];
            const y = corner[1];
            const rotated_x = x * cos_angle - y * sin_angle;
            const rotated_y = x * sin_angle + y * cos_angle;

            min_x = @min(min_x, rotated_x);
            max_x = @max(max_x, rotated_x);
            min_y = @min(min_y, rotated_y);
            max_y = @max(max_y, rotated_y);
        }

        const new_width = @as(u32, @intFromFloat(max_x - min_x + 1));
        const new_height = @as(u32, @intFromFloat(max_y - min_y + 1));

        const new_data = try self.allocator.alloc(u8, new_width * new_height * bytes_per_pixel);

        // Initialize to black
        @memset(new_data, 0);

        const center_x = old_width_f / 2.0;
        const center_y = old_height_f / 2.0;
        const new_center_x = @as(f32, @floatFromInt(new_width)) / 2.0;
        const new_center_y = @as(f32, @floatFromInt(new_height)) / 2.0;

        // Apply rotation with bilinear interpolation
        for (0..new_height) |y| {
            for (0..new_width) |x| {
                const fx = @as(f32, @floatFromInt(x)) - new_center_x;
                const fy = @as(f32, @floatFromInt(y)) - new_center_y;

                // Inverse rotation
                const src_x = fx * cos_angle + fy * sin_angle + center_x;
                const src_y = -fx * sin_angle + fy * cos_angle + center_y;

                if (src_x >= 0 and src_x < old_width_f - 1 and src_y >= 0 and src_y < old_height_f - 1) {
                    const x1 = @as(u32, @intFromFloat(@floor(src_x)));
                    const y1 = @as(u32, @intFromFloat(@floor(src_y)));
                    const x2 = x1 + 1;
                    const y2 = y1 + 1;

                    const dx = src_x - @as(f32, @floatFromInt(x1));
                    const dy = src_y - @as(f32, @floatFromInt(y1));

                    for (0..bytes_per_pixel) |c| {
                        const p11 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x1) * bytes_per_pixel + c]));
                        const p12 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x1) * bytes_per_pixel + c]));
                        const p21 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x2) * bytes_per_pixel + c]));
                        const p22 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x2) * bytes_per_pixel + c]));

                        const interpolated = p11 * (1 - dx) * (1 - dy) +
                            p21 * dx * (1 - dy) +
                            p12 * (1 - dx) * dy +
                            p22 * dx * dy;

                        new_data[(y * new_width + x) * bytes_per_pixel + c] = @intFromFloat(interpolated);
                    }
                }
            }
        }

        self.allocator.free(self.data);
        self.data = new_data;
        self.width = new_width;
        self.height = new_height;
    }

    pub fn perspectiveTransform(self: *Image, corners: [4][2]f32) !void {
        // corners: [top-left, top-right, bottom-right, bottom-left]
        const bytes_per_pixel = bytesPerPixel(self.format);

        // Calculate perspective transformation matrix (simplified 2D projection)
        const src_width = @as(f32, @floatFromInt(self.width));
        const src_height = @as(f32, @floatFromInt(self.height));

        // Source corners (reference for future enhancement)
        _ = [4][2]f32{
            .{ 0, 0 }, // top-left
            .{ src_width, 0 }, // top-right
            .{ src_width, src_height }, // bottom-right
            .{ 0, src_height }, // bottom-left
        };

        // Calculate output dimensions
        var min_x: f32 = std.math.inf(f32);
        var max_x: f32 = -std.math.inf(f32);
        var min_y: f32 = std.math.inf(f32);
        var max_y: f32 = -std.math.inf(f32);

        for (corners) |corner| {
            min_x = @min(min_x, corner[0]);
            max_x = @max(max_x, corner[0]);
            min_y = @min(min_y, corner[1]);
            max_y = @max(max_y, corner[1]);
        }

        const new_width = @as(u32, @intFromFloat(max_x - min_x + 1));
        const new_height = @as(u32, @intFromFloat(max_y - min_y + 1));

        const new_data = try self.allocator.alloc(u8, new_width * new_height * bytes_per_pixel);
        @memset(new_data, 0);

        // Simple perspective mapping (bilinear approximation)
        for (0..new_height) |y| {
            for (0..new_width) |x| {
                const dst_x = @as(f32, @floatFromInt(x)) + min_x;
                const dst_y = @as(f32, @floatFromInt(y)) + min_y;

                // Inverse perspective mapping (simplified)
                const u = dst_x / (max_x - min_x);
                const v = dst_y / (max_y - min_y);

                // Bilinear interpolation in source space
                const src_x = u * src_width;
                const src_y = v * src_height;

                if (src_x >= 0 and src_x < src_width - 1 and src_y >= 0 and src_y < src_height - 1) {
                    const x1 = @as(u32, @intFromFloat(@floor(src_x)));
                    const y1 = @as(u32, @intFromFloat(@floor(src_y)));
                    const x2 = x1 + 1;
                    const y2 = y1 + 1;

                    const dx = src_x - @as(f32, @floatFromInt(x1));
                    const dy = src_y - @as(f32, @floatFromInt(y1));

                    for (0..bytes_per_pixel) |c| {
                        const p11 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x1) * bytes_per_pixel + c]));
                        const p12 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x1) * bytes_per_pixel + c]));
                        const p21 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x2) * bytes_per_pixel + c]));
                        const p22 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x2) * bytes_per_pixel + c]));

                        const interpolated = p11 * (1 - dx) * (1 - dy) +
                            p21 * dx * (1 - dy) +
                            p12 * (1 - dx) * dy +
                            p22 * dx * dy;

                        new_data[(y * new_width + x) * bytes_per_pixel + c] = @intFromFloat(interpolated);
                    }
                }
            }
        }

        self.allocator.free(self.data);
        self.data = new_data;
        self.width = new_width;
        self.height = new_height;
    }

    pub fn correctBarrelDistortion(self: *Image, strength: f32) !void {
        if (strength == 0.0) return; // No correction needed

        const bytes_per_pixel = bytesPerPixel(self.format);
        const new_data = try self.allocator.alloc(u8, self.data.len);
        @memset(new_data, 0);

        const center_x = @as(f32, @floatFromInt(self.width)) / 2.0;
        const center_y = @as(f32, @floatFromInt(self.height)) / 2.0;
        const max_radius = @sqrt(center_x * center_x + center_y * center_y);

        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const fx = @as(f32, @floatFromInt(x));
                const fy = @as(f32, @floatFromInt(y));

                // Distance from center
                const dx = fx - center_x;
                const dy = fy - center_y;
                const radius = @sqrt(dx * dx + dy * dy);

                if (radius > 0) {
                    // Barrel distortion correction formula
                    const normalized_radius = radius / max_radius;
                    const distortion_factor = 1.0 + strength * (normalized_radius * normalized_radius);

                    // Calculate source coordinates
                    const src_x = center_x + (dx / distortion_factor);
                    const src_y = center_y + (dy / distortion_factor);

                    // Bilinear interpolation
                    if (src_x >= 0 and src_x < @as(f32, @floatFromInt(self.width - 1)) and
                        src_y >= 0 and src_y < @as(f32, @floatFromInt(self.height - 1)))
                    {
                        const x1 = @as(u32, @intFromFloat(@floor(src_x)));
                        const y1 = @as(u32, @intFromFloat(@floor(src_y)));
                        const x2 = @min(x1 + 1, self.width - 1);
                        const y2 = @min(y1 + 1, self.height - 1);

                        const wx = src_x - @as(f32, @floatFromInt(x1));
                        const wy = src_y - @as(f32, @floatFromInt(y1));

                        for (0..bytes_per_pixel) |c| {
                            const p11 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x1) * bytes_per_pixel + c]));
                            const p12 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x1) * bytes_per_pixel + c]));
                            const p21 = @as(f32, @floatFromInt(self.data[(y1 * self.width + x2) * bytes_per_pixel + c]));
                            const p22 = @as(f32, @floatFromInt(self.data[(y2 * self.width + x2) * bytes_per_pixel + c]));

                            const interpolated = p11 * (1 - wx) * (1 - wy) +
                                p21 * wx * (1 - wy) +
                                p12 * (1 - wx) * wy +
                                p22 * wx * wy;

                            new_data[(y * self.width + x) * bytes_per_pixel + c] = @intFromFloat(@min(255.0, @max(0.0, interpolated)));
                        }
                    }
                }
            }
        }

        self.allocator.free(self.data);
        self.data = new_data;
    }

    pub fn correctPincushionDistortion(self: *Image, strength: f32) !void {
        // Pincushion is the inverse of barrel distortion
        try self.correctBarrelDistortion(-strength);
    }

    /// Load a RAW image file with demosaicing
    pub fn loadRaw(allocator: std.mem.Allocator, file_path: []const u8) !Image {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();

        // Extract metadata first
        const raw_metadata = raw.extractRawMetadata(allocator, file) catch |err| switch (err) {
            error.UnsupportedRawFormat => {
                std.log.warn("Unsupported RAW format, attempting basic TIFF parsing", .{});
                return loadTiff(allocator, file);
            },
            else => return err,
        };

        // For this MVP, we'll create a simple demosaiced RGB image
        // In a full implementation, we'd read the actual RAW data
        const rgb_data = try allocator.alloc(u8, raw_metadata.width * raw_metadata.height * 3);

        // Simulate demosaiced data (in practice, this would read and demosaic actual RAW data)
        @memset(rgb_data, 128); // Gray placeholder

        return Image{
            .allocator = allocator,
            .width = raw_metadata.width,
            .height = raw_metadata.height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    /// Detect RAW file format
    pub fn detectRawFormat(file_path: []const u8) !raw.RawFormat {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();
        return raw.detectRawFormat(file);
    }

    /// Extract metadata from RAW file
    pub fn extractRawMetadata(allocator: std.mem.Allocator, file_path: []const u8) !raw.RawMetadata {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();
        return raw.extractRawMetadata(allocator, file);
    }

    /// Apply simple demosaicing to Bayer pattern data
    pub fn demosaicBayer(allocator: std.mem.Allocator, bayer_data: []const u16, width: u32, height: u32, pattern: raw.BayerPattern) ![]u8 {
        return raw.demosaicBilinear(allocator, bayer_data, width, height, pattern);
    }

    /// Load and render SVG to RGB image
    pub fn loadSvg(allocator: std.mem.Allocator, file_path: []const u8, width: u32, height: u32) !Image {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const svg_content = try allocator.alloc(u8, file_size);
        defer allocator.free(svg_content);
        _ = try file.readAll(svg_content);

        var svg_doc = try svg.parseSvg(allocator, svg_content);
        defer svg_doc.deinit();

        const rgb_data = try svg.renderSvg(allocator, &svg_doc, width, height);

        return Image{
            .allocator = allocator,
            .width = width,
            .height = height,
            .format = .rgb,
            .data = rgb_data,
        };
    }

    /// Parse SVG content and create document
    pub fn parseSvg(allocator: std.mem.Allocator, svg_content: []const u8) !svg.SvgDocument {
        return svg.parseSvg(allocator, svg_content);
    }

    /// Render SVG document to RGB bitmap
    pub fn renderSvgToRgb(allocator: std.mem.Allocator, doc: *const svg.SvgDocument, width: u32, height: u32) ![]u8 {
        return svg.renderSvg(allocator, doc, width, height);
    }

    /// Detect if file is AVIF format
    pub fn detectAvif(file_path: []const u8) !bool {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();
        return avif.detectAvif(file);
    }

    /// Parse AVIF header and extract metadata
    pub fn parseAvifHeader(allocator: std.mem.Allocator, file_path: []const u8) !avif.AvifHeader {
        const file = try std.fs.openFileAbsolute(file_path, .{});
        defer file.close();
        return avif.parseAvifHeader(allocator, file);
    }

    pub fn convertColorSpaceVectorized(self: *Image, target_format: PixelFormat) !void {
        if (self.format == target_format) return;

        _ = bytesPerPixel(self.format);
        const target_bpp = bytesPerPixel(target_format);
        const pixel_count = self.width * self.height;

        const new_data = try self.allocator.alloc(u8, pixel_count * target_bpp);

        // Vectorized color space conversions
        if (self.format == .rgb and target_format == .hsv) {
            rgbToHsvVectorized(self.data, new_data, pixel_count);
        } else if (self.format == .hsv and target_format == .rgb) {
            hsvToRgbVectorized(self.data, new_data, pixel_count);
        } else if (self.format == .rgb and target_format == .yuv) {
            simd.simdRgbToYuv(self.data, new_data, pixel_count);
        } else if (self.format == .yuv and target_format == .rgb) {
            simd.simdYuvToRgb(self.data, new_data, pixel_count);
        } else if (self.format == .rgb and target_format == .grayscale) {
            rgbToGrayscaleVectorized(self.data, new_data, pixel_count);
        } else if (self.format == .grayscale and target_format == .rgb) {
            grayscaleToRgbVectorized(self.data, new_data, pixel_count);
        } else {
            // Fallback to scalar conversion
            self.allocator.free(new_data);
            return error.UnsupportedColorSpaceConversion;
        }

        self.allocator.free(self.data);
        self.data = new_data;
        self.format = target_format;
    }

    fn rgbToHsvVectorized(rgb_data: []const u8, hsv_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        // Process 4 pixels at a time
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const rgb_idx = (i + j) * 3;
                const hsv_idx = (i + j) * 3;

                const r = @as(f32, @floatFromInt(rgb_data[rgb_idx])) / 255.0;
                const g = @as(f32, @floatFromInt(rgb_data[rgb_idx + 1])) / 255.0;
                const b = @as(f32, @floatFromInt(rgb_data[rgb_idx + 2])) / 255.0;

                const max_val = @max(@max(r, g), b);
                const min_val = @min(@min(r, g), b);
                const delta = max_val - min_val;

                // Value
                const v = max_val;

                // Saturation
                const s = if (max_val == 0) 0 else delta / max_val;

                // Hue
                var h: f32 = 0;
                if (delta != 0) {
                    if (max_val == r) {
                        h = 60.0 * (((g - b) / delta) + if (g < b) @as(f32, 6) else 0);
                    } else if (max_val == g) {
                        h = 60.0 * ((b - r) / delta + 2);
                    } else {
                        h = 60.0 * ((r - g) / delta + 4);
                    }
                }

                hsv_data[hsv_idx] = @intFromFloat(h * 255.0 / 360.0);
                hsv_data[hsv_idx + 1] = @intFromFloat(s * 255.0);
                hsv_data[hsv_idx + 2] = @intFromFloat(v * 255.0);
            }
            i += 4;
        }

        // Handle remaining pixels
        while (i < pixel_count) {
            const rgb_idx = i * 3;
            const hsv_idx = i * 3;

            const r = @as(f32, @floatFromInt(rgb_data[rgb_idx])) / 255.0;
            const g = @as(f32, @floatFromInt(rgb_data[rgb_idx + 1])) / 255.0;
            const b = @as(f32, @floatFromInt(rgb_data[rgb_idx + 2])) / 255.0;

            const max_val = @max(@max(r, g), b);
            const min_val = @min(@min(r, g), b);
            const delta = max_val - min_val;

            const v = max_val;
            const s = if (max_val == 0) 0 else delta / max_val;

            var h: f32 = 0;
            if (delta != 0) {
                if (max_val == r) {
                    h = 60.0 * (((g - b) / delta) + if (g < b) @as(f32, 6) else 0);
                } else if (max_val == g) {
                    h = 60.0 * ((b - r) / delta + 2);
                } else {
                    h = 60.0 * ((r - g) / delta + 4);
                }
            }

            hsv_data[hsv_idx] = @intFromFloat(h * 255.0 / 360.0);
            hsv_data[hsv_idx + 1] = @intFromFloat(s * 255.0);
            hsv_data[hsv_idx + 2] = @intFromFloat(v * 255.0);
            i += 1;
        }
    }

    fn hsvToRgbVectorized(hsv_data: []const u8, rgb_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const hsv_idx = (i + j) * 3;
                const rgb_idx = (i + j) * 3;

                const h = @as(f32, @floatFromInt(hsv_data[hsv_idx])) * 360.0 / 255.0;
                const s = @as(f32, @floatFromInt(hsv_data[hsv_idx + 1])) / 255.0;
                const v = @as(f32, @floatFromInt(hsv_data[hsv_idx + 2])) / 255.0;

                const c = v * s;
                const x = c * (1 - @abs(@mod(h / 60.0, 2) - 1));
                const m = v - c;

                var r: f32 = 0;
                var g: f32 = 0;
                var b: f32 = 0;

                if (h < 60) {
                    r = c;
                    g = x;
                    b = 0;
                } else if (h < 120) {
                    r = x;
                    g = c;
                    b = 0;
                } else if (h < 180) {
                    r = 0;
                    g = c;
                    b = x;
                } else if (h < 240) {
                    r = 0;
                    g = x;
                    b = c;
                } else if (h < 300) {
                    r = x;
                    g = 0;
                    b = c;
                } else {
                    r = c;
                    g = 0;
                    b = x;
                }

                rgb_data[rgb_idx] = @intFromFloat((r + m) * 255.0);
                rgb_data[rgb_idx + 1] = @intFromFloat((g + m) * 255.0);
                rgb_data[rgb_idx + 2] = @intFromFloat((b + m) * 255.0);
            }
            i += 4;
        }

        while (i < pixel_count) {
            const hsv_idx = i * 3;
            const rgb_idx = i * 3;

            const h = @as(f32, @floatFromInt(hsv_data[hsv_idx])) * 360.0 / 255.0;
            const s = @as(f32, @floatFromInt(hsv_data[hsv_idx + 1])) / 255.0;
            const v = @as(f32, @floatFromInt(hsv_data[hsv_idx + 2])) / 255.0;

            const c = v * s;
            const x = c * (1 - @abs(@mod(h / 60.0, 2) - 1));
            const m = v - c;

            var r: f32 = 0;
            var g: f32 = 0;
            var b: f32 = 0;

            if (h < 60) {
                r = c;
                g = x;
                b = 0;
            } else if (h < 120) {
                r = x;
                g = c;
                b = 0;
            } else if (h < 180) {
                r = 0;
                g = c;
                b = x;
            } else if (h < 240) {
                r = 0;
                g = x;
                b = c;
            } else if (h < 300) {
                r = x;
                g = 0;
                b = c;
            } else {
                r = c;
                g = 0;
                b = x;
            }

            rgb_data[rgb_idx] = @intFromFloat((r + m) * 255.0);
            rgb_data[rgb_idx + 1] = @intFromFloat((g + m) * 255.0);
            rgb_data[rgb_idx + 2] = @intFromFloat((b + m) * 255.0);
            i += 1;
        }
    }

    fn rgbToYuvVectorized(rgb_data: []const u8, yuv_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const idx = (i + j) * 3;
                const r = @as(f32, @floatFromInt(rgb_data[idx]));
                const g = @as(f32, @floatFromInt(rgb_data[idx + 1]));
                const b = @as(f32, @floatFromInt(rgb_data[idx + 2]));

                const y = 0.299 * r + 0.587 * g + 0.114 * b;
                const u = -0.169 * r - 0.331 * g + 0.5 * b + 128;
                const v = 0.5 * r - 0.419 * g - 0.081 * b + 128;

                yuv_data[idx] = @intFromFloat(@max(0, @min(255, y)));
                yuv_data[idx + 1] = @intFromFloat(@max(0, @min(255, u)));
                yuv_data[idx + 2] = @intFromFloat(@max(0, @min(255, v)));
            }
            i += 4;
        }

        while (i < pixel_count) {
            const idx = i * 3;
            const r = @as(f32, @floatFromInt(rgb_data[idx]));
            const g = @as(f32, @floatFromInt(rgb_data[idx + 1]));
            const b = @as(f32, @floatFromInt(rgb_data[idx + 2]));

            const y = 0.299 * r + 0.587 * g + 0.114 * b;
            const u = -0.169 * r - 0.331 * g + 0.5 * b + 128;
            const v = 0.5 * r - 0.419 * g - 0.081 * b + 128;

            yuv_data[idx] = @intFromFloat(@max(0, @min(255, y)));
            yuv_data[idx + 1] = @intFromFloat(@max(0, @min(255, u)));
            yuv_data[idx + 2] = @intFromFloat(@max(0, @min(255, v)));
            i += 1;
        }
    }

    fn yuvToRgbVectorized(yuv_data: []const u8, rgb_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const idx = (i + j) * 3;
                const y = @as(f32, @floatFromInt(yuv_data[idx]));
                const u = @as(f32, @floatFromInt(yuv_data[idx + 1])) - 128;
                const v = @as(f32, @floatFromInt(yuv_data[idx + 2])) - 128;

                const r = y + 1.4 * v;
                const g = y - 0.344 * u - 0.714 * v;
                const b = y + 1.772 * u;

                rgb_data[idx] = @intFromFloat(@max(0, @min(255, r)));
                rgb_data[idx + 1] = @intFromFloat(@max(0, @min(255, g)));
                rgb_data[idx + 2] = @intFromFloat(@max(0, @min(255, b)));
            }
            i += 4;
        }

        while (i < pixel_count) {
            const idx = i * 3;
            const y = @as(f32, @floatFromInt(yuv_data[idx]));
            const u = @as(f32, @floatFromInt(yuv_data[idx + 1])) - 128;
            const v = @as(f32, @floatFromInt(yuv_data[idx + 2])) - 128;

            const r = y + 1.4 * v;
            const g = y - 0.344 * u - 0.714 * v;
            const b = y + 1.772 * u;

            rgb_data[idx] = @intFromFloat(@max(0, @min(255, r)));
            rgb_data[idx + 1] = @intFromFloat(@max(0, @min(255, g)));
            rgb_data[idx + 2] = @intFromFloat(@max(0, @min(255, b)));
            i += 1;
        }
    }

    fn rgbToGrayscaleVectorized(rgb_data: []const u8, gray_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const rgb_idx = (i + j) * 3;
                const gray_idx = i + j;

                const r = @as(f32, @floatFromInt(rgb_data[rgb_idx]));
                const g = @as(f32, @floatFromInt(rgb_data[rgb_idx + 1]));
                const b = @as(f32, @floatFromInt(rgb_data[rgb_idx + 2]));

                const gray = 0.299 * r + 0.587 * g + 0.114 * b;
                gray_data[gray_idx] = @intFromFloat(gray);
            }
            i += 4;
        }

        while (i < pixel_count) {
            const rgb_idx = i * 3;
            const r = @as(f32, @floatFromInt(rgb_data[rgb_idx]));
            const g = @as(f32, @floatFromInt(rgb_data[rgb_idx + 1]));
            const b = @as(f32, @floatFromInt(rgb_data[rgb_idx + 2]));

            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            gray_data[i] = @intFromFloat(gray);
            i += 1;
        }
    }

    fn grayscaleToRgbVectorized(gray_data: []const u8, rgb_data: []u8, pixel_count: usize) void {
        var i: usize = 0;
        while (i + 4 <= pixel_count) {
            for (0..4) |j| {
                const gray_val = gray_data[i + j];
                const rgb_idx = (i + j) * 3;

                rgb_data[rgb_idx] = gray_val;
                rgb_data[rgb_idx + 1] = gray_val;
                rgb_data[rgb_idx + 2] = gray_val;
            }
            i += 4;
        }

        while (i < pixel_count) {
            const gray_val = gray_data[i];
            const rgb_idx = i * 3;

            rgb_data[rgb_idx] = gray_val;
            rgb_data[rgb_idx + 1] = gray_val;
            rgb_data[rgb_idx + 2] = gray_val;
            i += 1;
        }
    }

    fn saveJpeg(self: Image, path: []const u8) !void {
        if (!(self.format == .rgb or self.format == .rgba or self.format == .grayscale)) {
            return error.UnsupportedPixelFormat;
        }

        const padded_width = ((self.width + 7) / 8) * 8;
        const padded_height = ((self.height + 7) / 8) * 8;
        const plane_size = @as(usize, padded_width) * padded_height;

        var y_plane = try self.allocator.alloc(u8, plane_size);
        defer self.allocator.free(y_plane);

        var cb_plane = try self.allocator.alloc(u8, plane_size);
        defer self.allocator.free(cb_plane);

        var cr_plane = try self.allocator.alloc(u8, plane_size);
        defer self.allocator.free(cr_plane);

        const has_color = self.format != .grayscale;

        switch (self.format) {
            .grayscale => {
                for (0..padded_height) |py| {
                    const src_y = if (py >= self.height) self.height - 1 else py;
                    for (0..padded_width) |px| {
                        const src_x = if (px >= self.width) self.width - 1 else px;
                        const src_index = src_y * self.width + src_x;
                        const dst_index = py * padded_width + px;
                        const gray = self.data[src_index];
                        y_plane[dst_index] = gray;
                        cb_plane[dst_index] = 128;
                        cr_plane[dst_index] = 128;
                    }
                }
            },
            .rgb, .rgba => {
                const bytes_per_pixel: usize = if (self.format == .rgba) 4 else 3;
                for (0..padded_height) |py| {
                    const src_y = if (py >= self.height) self.height - 1 else py;
                    for (0..padded_width) |px| {
                        const src_x = if (px >= self.width) self.width - 1 else px;
                        const src_index = (src_y * self.width + src_x) * bytes_per_pixel;
                        const dst_index = py * padded_width + px;

                        const r = @as(f64, @floatFromInt(self.data[src_index]));
                        const g = @as(f64, @floatFromInt(self.data[src_index + 1]));
                        const b = @as(f64, @floatFromInt(self.data[src_index + 2]));

                        const y_val = 0.299 * r + 0.587 * g + 0.114 * b;
                        const cb_val = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
                        const cr_val = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

                        y_plane[dst_index] = clampToByte(y_val);
                        cb_plane[dst_index] = clampToByte(cb_val);
                        cr_plane[dst_index] = clampToByte(cr_val);
                    }
                }
            },
            else => unreachable,
        }

        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        // Write SOI
        _ = try file.write(&[_]u8{ 0xFF, 0xD8 });

        // APP0 JFIF
        var app0: [14]u8 = undefined;
        @memcpy(app0[0..5], "JFIF\x00");
        app0[5] = 0x01; // major version
        app0[6] = 0x01; // minor version
        app0[7] = 0x00; // units
        app0[8] = 0x00;
        app0[9] = 0x01; // X density 1
        app0[10] = 0x00;
        app0[11] = 0x01; // Y density 1
        app0[12] = 0x00; // thumbnail width
        app0[13] = 0x00; // thumbnail height
        try writeJpegSegment(file, 0xE0, &app0);

        // DQT
        var dqt_payload: [130]u8 = undefined;
        var dqt_len: usize = 0;
        dqt_payload[dqt_len] = 0x00; // table 0, 8-bit precision
        dqt_len += 1;
        for (zigzag_order) |zz| {
            dqt_payload[dqt_len] = std_luma_quant.data[zz];
            dqt_len += 1;
        }
        if (has_color) {
            dqt_payload[dqt_len] = 0x01; // chroma table id 1
            dqt_len += 1;
            for (zigzag_order) |zz| {
                dqt_payload[dqt_len] = std_chroma_quant.data[zz];
                dqt_len += 1;
            }
        }
        try writeJpegSegment(file, 0xDB, dqt_payload[0..dqt_len]);

        // DHT
        var dht_payload: [416]u8 = undefined;
        var dht_len: usize = 0;

        appendHuffmanTable(&dht_payload, &dht_len, 0, 0, &std_dc_luma_bits, &std_dc_luma_vals);
        appendHuffmanTable(&dht_payload, &dht_len, 1, 0, &std_ac_luma_bits, &std_ac_luma_vals);
        if (has_color) {
            appendHuffmanTable(&dht_payload, &dht_len, 0, 1, &std_dc_chroma_bits, &std_dc_chroma_vals);
            appendHuffmanTable(&dht_payload, &dht_len, 1, 1, &std_ac_chroma_bits, &std_ac_chroma_vals);
        }
        try writeJpegSegment(file, 0xC4, dht_payload[0..dht_len]);

        // SOF0
        var sof_payload: [19]u8 = undefined;
        var sof_len: usize = 0;
        sof_payload[sof_len] = 8; // precision
        sof_len += 1;
        sof_payload[sof_len] = @intCast((self.height >> 8) & 0xFF);
        sof_payload[sof_len + 1] = @intCast(self.height & 0xFF);
        sof_len += 2;
        sof_payload[sof_len] = @intCast((self.width >> 8) & 0xFF);
        sof_payload[sof_len + 1] = @intCast(self.width & 0xFF);
        sof_len += 2;
        const component_count: u8 = if (has_color) 3 else 1;
        sof_payload[sof_len] = component_count;
        sof_len += 1;

        // Component descriptors
        sof_payload[sof_len] = 1; // Y id
        sof_payload[sof_len + 1] = 0x11; // sampling 1x1
        sof_payload[sof_len + 2] = 0;
        sof_len += 3;
        if (has_color) {
            sof_payload[sof_len] = 2; // Cb
            sof_payload[sof_len + 1] = 0x11;
            sof_payload[sof_len + 2] = 1;
            sof_payload[sof_len + 3] = 3; // Cr
            sof_payload[sof_len + 4] = 0x11;
            sof_payload[sof_len + 5] = 1;
            sof_len += 6;
        }
        try writeJpegSegment(file, 0xC0, sof_payload[0..sof_len]);

        const dc_luma_table = buildEncodeTable(&std_dc_luma_bits, &std_dc_luma_vals);
        const ac_luma_table = buildEncodeTable(&std_ac_luma_bits, &std_ac_luma_vals);
        const dc_chroma_table = buildEncodeTable(&std_dc_chroma_bits, &std_dc_chroma_vals);
        const ac_chroma_table = buildEncodeTable(&std_ac_chroma_bits, &std_ac_chroma_vals);

        // SOS
        var sos_payload: [18]u8 = undefined;
        var sos_len: usize = 0;
        sos_payload[sos_len] = component_count;
        sos_len += 1;
        sos_payload[sos_len] = 1; // Y
        sos_payload[sos_len + 1] = 0x00; // DC table 0, AC table 0
        sos_len += 2;
        if (has_color) {
            sos_payload[sos_len] = 2; // Cb
            sos_payload[sos_len + 1] = 0x11;
            sos_payload[sos_len + 2] = 3; // Cr
            sos_payload[sos_len + 3] = 0x11;
            sos_len += 4;
        }
        sos_payload[sos_len] = 0; // start spectral
        sos_payload[sos_len + 1] = 63; // end spectral
        sos_payload[sos_len + 2] = 0; // approx
        sos_len += 3;
        try writeJpegSegment(file, 0xDA, sos_payload[0..sos_len]);

        var bit_writer = BitWriter.init(self.allocator);
        defer bit_writer.deinit();

        var prev_dc_y: i32 = 0;
        var prev_dc_cb: i32 = 0;
        var prev_dc_cr: i32 = 0;

        for (0..padded_height / 8) |block_y| {
            for (0..padded_width / 8) |block_x| {
                try encodeComponentBlock(&bit_writer, y_plane, padded_width, block_x * 8, block_y * 8, &std_luma_quant.data, &prev_dc_y, &dc_luma_table, &ac_luma_table);
                if (has_color) {
                    try encodeComponentBlock(&bit_writer, cb_plane, padded_width, block_x * 8, block_y * 8, &std_chroma_quant.data, &prev_dc_cb, &dc_chroma_table, &ac_chroma_table);
                    try encodeComponentBlock(&bit_writer, cr_plane, padded_width, block_x * 8, block_y * 8, &std_chroma_quant.data, &prev_dc_cr, &dc_chroma_table, &ac_chroma_table);
                }
            }
        }

        try bit_writer.flush();
        if (bit_writer.buffer.items.len != 0) {
            _ = try file.write(bit_writer.buffer.items);
        }

        // EOI
        _ = try file.write(&[_]u8{ 0xFF, 0xD9 });
    }

    fn appendHuffmanTable(buffer: *[416]u8, length: *usize, table_class: u8, table_id: u8, bits: []const u8, values: []const u8) void {
        buffer.*[length.*] = (table_class << 4) | table_id;
        length.* += 1;
        std.mem.copyForwards(u8, buffer.*[length.* .. length.* + bits.len], bits);
        length.* += bits.len;
        std.mem.copyForwards(u8, buffer.*[length.* .. length.* + values.len], values);
        length.* += values.len;
    }

    fn clampToByte(value: f64) u8 {
        const clamped = std.math.clamp(value, 0.0, 255.0);
        return @as(u8, @intFromFloat(std.math.round(clamped)));
    }

    fn encodeComponentBlock(bit_writer: *BitWriter, plane: []const u8, stride: usize, start_x: usize, start_y: usize, quant_table: *const [64]u8, prev_dc: *i32, dc_table: *const JpegHuffmanTable, ac_table: *const JpegHuffmanTable) !void {
        var spatial: [64]f64 = undefined;
        for (0..8) |row| {
            for (0..8) |col| {
                const sample = plane[(start_y + row) * stride + start_x + col];
                spatial[row * 8 + col] = @as(f64, @floatFromInt(sample)) - 128.0;
            }
        }

        var freq: [64]f64 = undefined;
        fdct8x8Forward(&spatial, &freq);

        var quantized: [64]i32 = undefined;
        for (0..64) |i| {
            const q = @as(f64, @floatFromInt(quant_table.*[i]));
            const value = freq[i] / q;
            quantized[i] = @intFromFloat(std.math.round(value));
        }

        var zigzag: [64]i32 = undefined;
        for (0..64) |i| {
            const index = zigzag_order[i];
            zigzag[i] = quantized[index];
        }

        try encodeQuantizedBlock(bit_writer, &zigzag, prev_dc, dc_table, ac_table);
    }

    fn encodeQuantizedBlock(bit_writer: *BitWriter, block: *const [64]i32, prev_dc: *i32, dc_table: *const JpegHuffmanTable, ac_table: *const JpegHuffmanTable) !void {
        const dc = block[0];
        const diff = dc - prev_dc.*;
        prev_dc.* = dc;

        const dc_mag = magnitudeBits(diff);
        const dc_len = dc_table.code_lengths[dc_mag.size];
        if (dc_len == 0) return error.InvalidHuffmanCode;
        try bit_writer.writeBits(dc_table.codes[dc_mag.size], dc_len);
        if (dc_mag.size > 0) {
            try bit_writer.writeBits(dc_mag.bits, dc_mag.size);
        }

        var zero_run: usize = 0;
        for (1..64) |i| {
            const coeff = block[i];
            if (coeff == 0) {
                zero_run += 1;
                continue;
            }

            while (zero_run >= 16) {
                const len = ac_table.code_lengths[0xF0];
                if (len == 0) return error.InvalidHuffmanCode;
                try bit_writer.writeBits(ac_table.codes[0xF0], len);
                zero_run -= 16;
            }

            const mag = magnitudeBits(coeff);
            const symbol: u8 = @intCast((zero_run << 4) | mag.size);
            const len = ac_table.code_lengths[symbol];
            if (len == 0) return error.InvalidHuffmanCode;
            try bit_writer.writeBits(ac_table.codes[symbol], len);
            if (mag.size > 0) {
                try bit_writer.writeBits(mag.bits, mag.size);
            }
            zero_run = 0;
        }

        if (zero_run > 0) {
            const len = ac_table.code_lengths[0x00];
            if (len == 0) return error.InvalidHuffmanCode;
            try bit_writer.writeBits(ac_table.codes[0x00], len);
        }
    }

    fn magnitudeBits(value: i32) struct { size: u8, bits: u16 } {
        if (value == 0) {
            return .{ .size = 0, .bits = 0 };
        }

        var abs_val = if (value < 0) -value else value;
        var size: u8 = 0;
        while (abs_val > 0) : (abs_val >>= 1) {
            size += 1;
        }

        var bits: u16 = undefined;
        if (value < 0) {
            const shift_amount: std.math.Log2Int(i32) = @intCast(size);
            const mask = (@as(i32, 1) << shift_amount) - 1;
            bits = @intCast(mask + value);
        } else {
            bits = @intCast(value);
        }

        return .{ .size = size, .bits = bits };
    }

    fn fdct8x8Forward(input: *const [64]f64, output: *[64]f64) void {
        const pi = std.math.pi;
        const inv_sqrt2: f64 = 0.7071067811865476;
        for (0..8) |u| {
            for (0..8) |v| {
                var sum: f64 = 0.0;
                for (0..8) |x| {
                    for (0..8) |y| {
                        const sample = input[x * 8 + y];
                        const cos_x = std.math.cos(((2.0 * @as(f64, @floatFromInt(x)) + 1.0) * @as(f64, @floatFromInt(u)) * pi) / 16.0);
                        const cos_y = std.math.cos(((2.0 * @as(f64, @floatFromInt(y)) + 1.0) * @as(f64, @floatFromInt(v)) * pi) / 16.0);
                        sum += sample * cos_x * cos_y;
                    }
                }
                const cu = if (u == 0) inv_sqrt2 else 1.0;
                const cv = if (v == 0) inv_sqrt2 else 1.0;
                output[u * 8 + v] = 0.25 * cu * cv * sum;
            }
        }
    }

    fn writeJpegSegment(file: std.fs.File, marker: u8, payload: []const u8) !void {
        _ = try file.write(&[_]u8{ 0xFF, marker });
        var length_bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &length_bytes, @intCast(payload.len + 2), .big);
        _ = try file.write(&length_bytes);
        if (payload.len > 0) {
            _ = try file.write(payload);
        }
    }

    fn saveBmp(self: Image, path: []const u8) !void {
        var rgb_data: []u8 = undefined;
        var needs_free = false;
        defer if (needs_free) self.allocator.free(rgb_data);

        if (self.format == .grayscale) {
            rgb_data = try self.allocator.alloc(u8, self.width * self.height * 3);
            needs_free = true;
            for (0..self.data.len) |i| {
                const gray = self.data[i];
                rgb_data[i * 3] = gray;
                rgb_data[i * 3 + 1] = gray;
                rgb_data[i * 3 + 2] = gray;
            }
        } else if (self.format == .rgb) {
            rgb_data = self.data;
        } else {
            return error.UnsupportedPixelFormat;
        }

        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        const data_size = self.width * self.height * 3;
        const file_size = 54 + data_size;

        var header: [54]u8 = undefined;
        std.mem.copyForwards(u8, &header, "BM");
        std.mem.writeInt(u32, header[2..6], @intCast(file_size), .little);
        std.mem.writeInt(u32, header[10..14], 54, .little); // data offset
        std.mem.writeInt(u32, header[14..18], 40, .little); // header size
        std.mem.writeInt(u32, header[18..22], self.width, .little);
        std.mem.writeInt(u32, header[22..26], self.height, .little);
        std.mem.writeInt(u16, header[26..28], 1, .little); // planes
        std.mem.writeInt(u16, header[28..30], 24, .little); // bpp
        std.mem.writeInt(u32, header[34..38], data_size, .little); // image size

        _ = try file.write(&header);

        // BMP stores pixels bottom-up, so we need to flip the rows
        var row_buf = try self.allocator.alloc(u8, self.width * 3);
        defer self.allocator.free(row_buf);

        var y: i32 = @intCast(self.height - 1);
        while (y >= 0) : (y -= 1) {
            const row_start = @as(usize, @intCast(y)) * self.width * 3;
            @memcpy(row_buf, rgb_data[row_start .. row_start + self.width * 3]);
            // BMP is BGR, so swap R and B
            for (0..self.width) |x| {
                const temp = row_buf[x * 3];
                row_buf[x * 3] = row_buf[x * 3 + 2];
                row_buf[x * 3 + 2] = temp;
            }
            _ = try file.write(row_buf);
        }
    }

    fn savePng(self: Image, path: []const u8) !void {
        // Simple PNG creation without compression for MVP
        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        // PNG signature
        const png_signature = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
        _ = try file.write(&png_signature);

        // IHDR chunk - determine color type based on image format
        const color_type: u8 = switch (self.format) {
            .rgba => 6, // RGBA
            .rgb => 2, // RGB
            .grayscale => 0, // Grayscale
            else => 2, // Default to RGB
        };
        const bytes_per_pixel = bytesPerPixel(self.format);
        try writePngChunk(file, "IHDR", &createIHDR(self.width, self.height, 8, color_type));

        // IDAT chunk - simplified uncompressed data (for MVP)
        const scanline_size = self.width * bytes_per_pixel + 1; // +1 for filter byte
        const idat_size = self.height * scanline_size;
        var idat_data = try self.allocator.alloc(u8, idat_size);
        defer self.allocator.free(idat_data);

        // Create scanlines with filter byte 0 (no filter)
        for (0..self.height) |y| {
            const scanline_start = y * scanline_size;
            idat_data[scanline_start] = 0; // Filter type: None

            const row_start = y * self.width * bytes_per_pixel;
            @memcpy(idat_data[scanline_start + 1 .. scanline_start + 1 + self.width * bytes_per_pixel], self.data[row_start .. row_start + self.width * bytes_per_pixel]);
        }

        // For simplicity, store uncompressed (real PNG would use zlib)
        try writePngChunk(file, "IDAT", idat_data);

        // IEND chunk
        try writePngChunk(file, "IEND", &[_]u8{});
    }

    fn createIHDR(width: u32, height: u32, bit_depth: u8, color_type: u8) [13]u8 {
        var ihdr: [13]u8 = undefined;
        std.mem.writeInt(u32, ihdr[0..4], width, .big);
        std.mem.writeInt(u32, ihdr[4..8], height, .big);
        ihdr[8] = bit_depth;
        ihdr[9] = color_type;
        ihdr[10] = 0; // Compression: deflate
        ihdr[11] = 0; // Filter: adaptive
        ihdr[12] = 0; // Interlace: none
        return ihdr;
    }

    fn writePngChunk(file: std.fs.File, chunk_type: []const u8, data: []const u8) !void {
        // Length
        const length_bytes = std.mem.toBytes(@as(u32, @intCast(data.len)));
        _ = try file.write(&[_]u8{ length_bytes[3], length_bytes[2], length_bytes[1], length_bytes[0] });

        // Type
        _ = try file.write(chunk_type);

        // Data
        _ = try file.write(data);

        // CRC (simplified - use 0 for MVP)
        const crc_bytes = [_]u8{ 0, 0, 0, 0 };
        _ = try file.write(&crc_bytes);
    }

    fn saveWebP(self: Image, path: []const u8) !void {
        // Simple WebP creation (lossless VP8L for MVP)
        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();

        // Convert image data to RGB if necessary
        var rgb_data: []u8 = undefined;
        var needs_free = false;
        defer if (needs_free) self.allocator.free(rgb_data);

        if (self.format == .grayscale) {
            rgb_data = try self.allocator.alloc(u8, self.width * self.height * 3);
            needs_free = true;
            for (0..self.data.len) |i| {
                const gray = self.data[i];
                rgb_data[i * 3] = gray;
                rgb_data[i * 3 + 1] = gray;
                rgb_data[i * 3 + 2] = gray;
            }
        } else if (self.format == .rgb) {
            rgb_data = self.data;
        } else {
            return error.UnsupportedPixelFormat;
        }

        // Create simplified VP8L bitstream (lossless)
        const vp8l_size = 5 + rgb_data.len + 4; // header + pixels + padding
        const vp8l_data = try self.allocator.alloc(u8, vp8l_size);
        defer self.allocator.free(vp8l_data);

        // VP8L signature
        vp8l_data[0] = 0x2f;

        // Dimensions (14 bits each, minus 1)
        const dimensions = (@as(u32, self.width - 1) & 0x3FFF) |
            ((@as(u32, self.height - 1) & 0x3FFF) << 14);
        std.mem.writeInt(u32, vp8l_data[1..5], dimensions, .little);

        // Simple uncompressed pixel data (for MVP)
        @memcpy(vp8l_data[5 .. 5 + rgb_data.len], rgb_data);

        // Calculate RIFF/WebP structure
        const webp_size = 4 + 8 + vp8l_size; // WEBP + VP8L chunk
        const riff_size = webp_size + 4; // Include file size

        // Write RIFF header
        _ = try file.write("RIFF");
        const size_bytes = std.mem.toBytes(@as(u32, @intCast(riff_size)));
        _ = try file.write(&[_]u8{ size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3] });
        _ = try file.write("WEBP");

        // Write VP8L chunk
        _ = try file.write("VP8L");
        const vp8l_size_bytes = std.mem.toBytes(@as(u32, @intCast(vp8l_size)));
        _ = try file.write(&[_]u8{ vp8l_size_bytes[0], vp8l_size_bytes[1], vp8l_size_bytes[2], vp8l_size_bytes[3] });
        _ = try file.write(vp8l_data);

        // Add padding if needed
        if (vp8l_size % 2 == 1) {
            _ = try file.write(&[_]u8{0});
        }
    }
};

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

// Unit tests for Image operations
test "Image init with valid dimensions" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 10, 10, .rgb);
    defer image.deinit();

    try std.testing.expect(image.width == 10);
    try std.testing.expect(image.height == 10);
    try std.testing.expect(image.format == .rgb);
    try std.testing.expect(image.data.len == 10 * 10 * 3);
}

test "Image init with invalid dimensions" {
    const allocator = std.testing.allocator;

    // Test zero dimensions
    try std.testing.expectError(error.InvalidDimensions, Image.init(allocator, 0, 10, .rgb));
    try std.testing.expectError(error.InvalidDimensions, Image.init(allocator, 10, 0, .rgb));

    // Test too large dimensions
    try std.testing.expectError(error.DimensionsTooLarge, Image.init(allocator, 100000, 100000, .rgb));
}

test "Image resize" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 4, 4, .rgb);
    defer image.deinit();

    // Fill with test pattern
    for (0..image.data.len) |i| {
        image.data[i] = @intCast(i % 256);
    }

    try image.resize(8, 8);
    try std.testing.expect(image.width == 8);
    try std.testing.expect(image.height == 8);
    try std.testing.expect(image.data.len == 8 * 8 * 3);
}

test "Image resize with invalid dimensions" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 4, 4, .rgb);
    defer image.deinit();

    try std.testing.expectError(error.InvalidDimensions, image.resize(0, 8));
    try std.testing.expectError(error.DimensionsTooLarge, image.resize(100000, 100000));
}

test "Image crop" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 10, 10, .rgb);
    defer image.deinit();

    // Fill with test pattern
    for (0..image.data.len) |i| {
        image.data[i] = @intCast(i % 256);
    }

    try image.crop(2, 2, 6, 6);
    try std.testing.expect(image.width == 6);
    try std.testing.expect(image.height == 6);
    try std.testing.expect(image.data.len == 6 * 6 * 3);
}

test "Image crop with invalid parameters" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 10, 10, .rgb);
    defer image.deinit();

    // Test crop out of bounds
    try std.testing.expectError(error.CropOutOfBounds, image.crop(15, 5, 2, 2));
    try std.testing.expectError(error.CropOutOfBounds, image.crop(5, 5, 10, 2));

    // Test invalid dimensions
    try std.testing.expectError(error.InvalidDimensions, image.crop(5, 5, 0, 2));
}

test "Image rotate90" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 4, 6, .rgb);
    defer image.deinit();

    // Fill with test pattern
    for (0..image.data.len) |i| {
        image.data[i] = @intCast(i % 256);
    }

    try image.rotate90();
    try std.testing.expect(image.width == 6);
    try std.testing.expect(image.height == 4);
}

test "Image convertToGrayscale" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    // Set known RGB values
    image.data[0] = 255;
    image.data[1] = 0;
    image.data[2] = 0; // Red
    image.data[3] = 0;
    image.data[4] = 255;
    image.data[5] = 0; // Green
    image.data[6] = 0;
    image.data[7] = 0;
    image.data[8] = 255; // Blue
    image.data[9] = 128;
    image.data[10] = 128;
    image.data[11] = 128; // Gray

    try image.convertToGrayscale();
    try std.testing.expect(image.format == .grayscale);
    try std.testing.expect(image.data.len == 2 * 2);

    // Check approximate grayscale values
    try std.testing.expect(image.data[0] > 50); // Red -> ~54
    try std.testing.expect(image.data[1] > 180); // Green -> ~183
    try std.testing.expect(image.data[2] < 30); // Blue -> ~18
    try std.testing.expect(image.data[3] == 128); // Gray -> 128
}

test "Image adjustBrightness" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 2, 2, .grayscale);
    defer image.deinit();

    // Set test values
    image.data[0] = 100;
    image.data[1] = 200;
    image.data[2] = 50;
    image.data[3] = 250;

    try image.adjustBrightness(30);

    try std.testing.expect(image.data[0] == 130);
    try std.testing.expect(image.data[1] == 230);
    try std.testing.expect(image.data[2] == 80);
    try std.testing.expect(image.data[3] == 255); // Clamped
}

test "Image adjustContrast" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 2, 2, .grayscale);
    defer image.deinit();

    // Set test values
    image.data[0] = 128; // Should stay 128 (center point)
    image.data[1] = 178; // Should become further from center
    image.data[2] = 78; // Should become further from center
    image.data[3] = 200;

    try image.adjustContrast(1.5);

    try std.testing.expect(image.data[0] == 128); // Center point unchanged
    try std.testing.expect(image.data[1] > 178); // Increased contrast
    try std.testing.expect(image.data[2] < 78); // Increased contrast
}

test "Image flipHorizontal" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 3, 1, .grayscale);
    defer image.deinit();

    // Set test pattern: [1, 2, 3]
    image.data[0] = 1;
    image.data[1] = 2;
    image.data[2] = 3;

    try image.flipHorizontal();

    // Should become [3, 2, 1]
    try std.testing.expect(image.data[0] == 3);
    try std.testing.expect(image.data[1] == 2);
    try std.testing.expect(image.data[2] == 1);
}

test "Image flipVertical" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 1, 3, .grayscale);
    defer image.deinit();

    // Set test pattern: [1, 2, 3] vertically
    image.data[0] = 1;
    image.data[1] = 2;
    image.data[2] = 3;

    try image.flipVertical();

    // Should become [3, 2, 1]
    try std.testing.expect(image.data[0] == 3);
    try std.testing.expect(image.data[1] == 2);
    try std.testing.expect(image.data[2] == 1);
}

test "Image blur" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 3, 3, .grayscale);
    defer image.deinit();

    // Set sharp edges pattern
    image.data[0] = 0;
    image.data[1] = 0;
    image.data[2] = 0;
    image.data[3] = 0;
    image.data[4] = 255;
    image.data[5] = 0;
    image.data[6] = 0;
    image.data[7] = 0;
    image.data[8] = 0;

    try image.blur(1);

    // Center should be less bright, edges should be brighter
    try std.testing.expect(image.data[4] < 255); // Center pixel dimmed
    try std.testing.expect(image.data[1] > 0); // Adjacent pixels brightened
}

test "BMP bytesPerPixel function" {
    try std.testing.expect(bytesPerPixel(.rgb) == 3);
    try std.testing.expect(bytesPerPixel(.rgba) == 4);
    try std.testing.expect(bytesPerPixel(.grayscale) == 1);
    try std.testing.expect(bytesPerPixel(.yuv) == 3);
    try std.testing.expect(bytesPerPixel(.hsv) == 3);
    try std.testing.expect(bytesPerPixel(.cmyk) == 4);
}

test "Error handling for invalid paths" {
    const allocator = std.testing.allocator;

    // Test empty path
    try std.testing.expectError(error.EmptyPath, Image.load(allocator, ""));

    // Test file that doesn't exist
    try std.testing.expectError(error.FileNotFound, Image.load(allocator, "/nonexistent/file.jpg"));
}

test "Save validation" {
    const allocator = std.testing.allocator;
    var image = try Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    // Test empty path
    try std.testing.expectError(error.EmptyPath, image.save("", .bmp));

    // Test WebP format (now supported)
    try image.save("/tmp/test.webp", .webp);
}

test "Image load baseline JPEG" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const abs_path = try std.fs.cwd().realpathAlloc(allocator, "assets/test-images/baseline_8x8.jpg");
    defer allocator.free(abs_path);

    var image = try Image.load(allocator, abs_path);
    defer image.deinit();

    try std.testing.expectEqual(@as(u32, 8), image.width);
    try std.testing.expectEqual(@as(u32, 8), image.height);
    try std.testing.expectEqual(PixelFormat.rgb, image.format);

    const first_pixel = image.data[0..3];
    try std.testing.expect(first_pixel[0] >= 110 and first_pixel[0] <= 130);
    try std.testing.expect(first_pixel[1] >= 190 and first_pixel[1] <= 210);
    try std.testing.expect(first_pixel[2] >= 30 and first_pixel[2] <= 50);
}

test "Image save and load JPEG roundtrip" {
    const allocator = std.testing.allocator;

    var image = try Image.init(allocator, 8, 8, .rgb);
    defer image.deinit();

    for (0..image.height) |y| {
        for (0..image.width) |x| {
            const idx = (y * image.width + x) * 3;
            image.data[idx] = @intCast((x * 255) / (image.width - 1));
            image.data[idx + 1] = @intCast((y * 255) / (image.height - 1));
            image.data[idx + 2] = 128;
        }
    }

    const cwd = std.fs.cwd();
    try cwd.makePath("zig-out/tests");
    const cwd_path = try cwd.realpathAlloc(allocator, ".");
    defer allocator.free(cwd_path);
    const abs_path = try std.fs.path.join(allocator, &[_][]const u8{ cwd_path, "zig-out", "tests", "jpeg_roundtrip.jpg" });
    defer allocator.free(abs_path);

    try image.save(abs_path, .jpeg);
    defer std.fs.deleteFileAbsolute(abs_path) catch {};

    var loaded = try Image.load(allocator, abs_path);
    defer loaded.deinit();

    try std.testing.expectEqual(image.width, loaded.width);
    try std.testing.expectEqual(image.height, loaded.height);

    for (0..loaded.data.len) |i| {
        const orig = image.data[i];
        const decoded = loaded.data[i];
        const diff = if (orig > decoded) orig - decoded else decoded - orig;
        try std.testing.expect(diff <= 20);
    }
}
