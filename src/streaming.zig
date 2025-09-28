//! Streaming image processing for large files
//! Processes images in chunks to minimize memory usage

const std = @import("std");
const zpix = @import("root.zig");

pub const ChunkSize = struct {
    width: u32,
    height: u32,
};

/// Streaming image processor that processes images in chunks
pub const StreamingProcessor = struct {
    allocator: std.mem.Allocator,
    chunk_width: u32,
    chunk_height: u32,

    pub fn init(allocator: std.mem.Allocator, chunk_width: u32, chunk_height: u32) StreamingProcessor {
        return .{
            .allocator = allocator,
            .chunk_width = chunk_width,
            .chunk_height = chunk_height,
        };
    }

    /// Process an image file in chunks, applying the given operation to each chunk
    pub fn processFile(
        self: *StreamingProcessor,
        input_path: []const u8,
        output_path: []const u8,
        operation: fn (chunk: *zpix.Image) anyerror!void,
    ) !void {
        // Open input file
        const input_file = try std.fs.openFileAbsolute(input_path, .{});
        defer input_file.close();

        // Detect format
        var sig_buf: [12]u8 = undefined;
        _ = try input_file.read(&sig_buf);
        try input_file.seekTo(0);

        // For this example, we'll focus on BMP which is easier to stream
        if (!std.mem.eql(u8, sig_buf[0..2], "BM")) {
            return error.OnlyBMPSupportedForStreaming;
        }

        // Read BMP header
        var header: [54]u8 = undefined;
        _ = try input_file.read(&header);

        const width = std.mem.readInt(u32, header[18..22], .little);
        const height = std.mem.readInt(u32, header[22..26], .little);
        const bpp = std.mem.readInt(u16, header[28..30], .little);

        if (bpp != 24) {
            return error.Only24BitBMPSupported;
        }

        // Create output file
        const output_file = try std.fs.createFileAbsolute(output_path, .{});
        defer output_file.close();

        // Write header
        _ = try output_file.write(&header);

        // Process in chunks
        const bytes_per_pixel = 3;
        const row_size = width * bytes_per_pixel;

        // Calculate chunk dimensions
        const num_chunks_x = (width + self.chunk_width - 1) / self.chunk_width;
        const num_chunks_y = (height + self.chunk_height - 1) / self.chunk_height;

        // Process each chunk
        var y: u32 = 0;
        while (y < height) {
            const chunk_h = @min(self.chunk_height, height - y);

            // Allocate buffer for chunk rows
            var chunk_buffer = try self.allocator.alloc(u8, row_size * chunk_h);
            defer self.allocator.free(chunk_buffer);

            // Read chunk rows
            try input_file.seekTo(54 + y * row_size);
            _ = try input_file.read(chunk_buffer);

            var x: u32 = 0;
            while (x < width) {
                const chunk_w = @min(self.chunk_width, width - x);

                // Create a chunk image
                var chunk = try zpix.Image.init(self.allocator, chunk_w, chunk_h, .rgb);
                defer chunk.deinit();

                // Copy data to chunk
                for (0..chunk_h) |row| {
                    const src_start = row * row_size + x * bytes_per_pixel;
                    const dst_start = row * chunk_w * bytes_per_pixel;
                    @memcpy(
                        chunk.data[dst_start..dst_start + chunk_w * bytes_per_pixel],
                        chunk_buffer[src_start..src_start + chunk_w * bytes_per_pixel],
                    );
                }

                // Process the chunk
                try operation(&chunk);

                // Copy processed data back
                for (0..chunk_h) |row| {
                    const src_start = row * chunk_w * bytes_per_pixel;
                    const dst_start = row * row_size + x * bytes_per_pixel;
                    @memcpy(
                        chunk_buffer[dst_start..dst_start + chunk_w * bytes_per_pixel],
                        chunk.data[src_start..src_start + chunk_w * bytes_per_pixel],
                    );
                }

                x += self.chunk_width;
            }

            // Write processed chunk rows
            try output_file.seekTo(54 + y * row_size);
            _ = try output_file.write(chunk_buffer);

            y += self.chunk_height;
        }
    }

    /// Stream process with progress callback
    pub fn processFileWithProgress(
        self: *StreamingProcessor,
        input_path: []const u8,
        output_path: []const u8,
        operation: fn (chunk: *zpix.Image) anyerror!void,
        progress_callback: fn (current: u32, total: u32) void,
    ) !void {
        // Similar to processFile but with progress tracking
        const input_file = try std.fs.openFileAbsolute(input_path, .{});
        defer input_file.close();

        var sig_buf: [12]u8 = undefined;
        _ = try input_file.read(&sig_buf);
        try input_file.seekTo(0);

        if (!std.mem.eql(u8, sig_buf[0..2], "BM")) {
            return error.OnlyBMPSupportedForStreaming;
        }

        var header: [54]u8 = undefined;
        _ = try input_file.read(&header);

        const width = std.mem.readInt(u32, header[18..22], .little);
        const height = std.mem.readInt(u32, header[22..26], .little);

        const total_chunks = ((width + self.chunk_width - 1) / self.chunk_width) *
                           ((height + self.chunk_height - 1) / self.chunk_height);
        var current_chunk: u32 = 0;

        const output_file = try std.fs.createFileAbsolute(output_path, .{});
        defer output_file.close();
        _ = try output_file.write(&header);

        const bytes_per_pixel = 3;
        const row_size = width * bytes_per_pixel;

        var y: u32 = 0;
        while (y < height) {
            const chunk_h = @min(self.chunk_height, height - y);
            var chunk_buffer = try self.allocator.alloc(u8, row_size * chunk_h);
            defer self.allocator.free(chunk_buffer);

            try input_file.seekTo(54 + y * row_size);
            _ = try input_file.read(chunk_buffer);

            var x: u32 = 0;
            while (x < width) {
                const chunk_w = @min(self.chunk_width, width - x);
                var chunk = try zpix.Image.init(self.allocator, chunk_w, chunk_h, .rgb);
                defer chunk.deinit();

                for (0..chunk_h) |row| {
                    const src_start = row * row_size + x * bytes_per_pixel;
                    const dst_start = row * chunk_w * bytes_per_pixel;
                    @memcpy(
                        chunk.data[dst_start..dst_start + chunk_w * bytes_per_pixel],
                        chunk_buffer[src_start..src_start + chunk_w * bytes_per_pixel],
                    );
                }

                try operation(&chunk);

                for (0..chunk_h) |row| {
                    const src_start = row * chunk_w * bytes_per_pixel;
                    const dst_start = row * row_size + x * bytes_per_pixel;
                    @memcpy(
                        chunk_buffer[dst_start..dst_start + chunk_w * bytes_per_pixel],
                        chunk.data[src_start..src_start + chunk_w * bytes_per_pixel],
                    );
                }

                current_chunk += 1;
                progress_callback(current_chunk, total_chunks);

                x += self.chunk_width;
            }

            try output_file.seekTo(54 + y * row_size);
            _ = try output_file.write(chunk_buffer);

            y += self.chunk_height;
        }
    }
};

/// Line-by-line streaming reader for very large images
pub const StreamingReader = struct {
    file: std.fs.File,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    current_row: u32,
    row_buffer: []u8,
    allocator: std.mem.Allocator,

    pub fn initBMP(allocator: std.mem.Allocator, path: []const u8) !StreamingReader {
        const file = try std.fs.openFileAbsolute(path, .{});

        var header: [54]u8 = undefined;
        _ = try file.read(&header);

        if (!std.mem.eql(u8, header[0..2], "BM")) {
            return error.NotBMP;
        }

        const width = std.mem.readInt(u32, header[18..22], .little);
        const height = std.mem.readInt(u32, header[22..26], .little);
        const bpp = std.mem.readInt(u16, header[28..30], .little);

        if (bpp != 24) {
            return error.UnsupportedBPP;
        }

        const bytes_per_pixel = 3;
        const row_buffer = try allocator.alloc(u8, width * bytes_per_pixel);

        return .{
            .file = file,
            .width = width,
            .height = height,
            .bytes_per_pixel = bytes_per_pixel,
            .current_row = 0,
            .row_buffer = row_buffer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StreamingReader) void {
        self.allocator.free(self.row_buffer);
        self.file.close();
    }

    pub fn readRow(self: *StreamingReader) !?[]u8 {
        if (self.current_row >= self.height) {
            return null;
        }

        const row_size = self.width * self.bytes_per_pixel;
        const offset = 54 + self.current_row * row_size;
        try self.file.seekTo(offset);
        _ = try self.file.read(self.row_buffer);

        self.current_row += 1;
        return self.row_buffer;
    }

    pub fn reset(self: *StreamingReader) !void {
        self.current_row = 0;
        try self.file.seekTo(54);
    }
};