//! Zero-copy operations for zpix
//! Provides memory-efficient image processing without unnecessary allocations
//! Uses views, slices, and in-place operations where possible

const std = @import("std");

/// Image view that references existing data without copying
pub const ImageView = struct {
    data: []u8,
    width: u32,
    height: u32,
    channels: u32,
    stride: u32, // Bytes per row (may include padding)

    pub fn init(data: []u8, width: u32, height: u32, channels: u32) ImageView {
        return ImageView{
            .data = data,
            .width = width,
            .height = height,
            .channels = channels,
            .stride = width * channels,
        };
    }

    pub fn initWithStride(data: []u8, width: u32, height: u32, channels: u32, stride: u32) ImageView {
        return ImageView{
            .data = data,
            .width = width,
            .height = height,
            .channels = channels,
            .stride = stride,
        };
    }

    pub fn getPixel(self: *const ImageView, x: u32, y: u32) []u8 {
        const idx = y * self.stride + x * self.channels;
        return self.data[idx..idx + self.channels];
    }

    pub fn setPixel(self: *ImageView, x: u32, y: u32, pixel: []const u8) void {
        const idx = y * self.stride + x * self.channels;
        @memcpy(self.data[idx..idx + self.channels], pixel);
    }

    pub fn getRow(self: *const ImageView, y: u32) []u8 {
        const start = y * self.stride;
        const end = start + (self.width * self.channels);
        return self.data[start..end];
    }

    pub fn subView(self: *const ImageView, x: u32, y: u32, w: u32, h: u32) ImageView {
        std.debug.assert(x + w <= self.width);
        std.debug.assert(y + h <= self.height);

        const start_idx = y * self.stride + x * self.channels;
        return ImageView{
            .data = self.data[start_idx..],
            .width = w,
            .height = h,
            .channels = self.channels,
            .stride = self.stride,
        };
    }
};

/// Const image view for read-only operations
pub const ConstImageView = struct {
    data: []const u8,
    width: u32,
    height: u32,
    channels: u32,
    stride: u32,

    pub fn init(data: []const u8, width: u32, height: u32, channels: u32) ConstImageView {
        return ConstImageView{
            .data = data,
            .width = width,
            .height = height,
            .channels = channels,
            .stride = width * channels,
        };
    }

    pub fn initWithStride(data: []const u8, width: u32, height: u32, channels: u32, stride: u32) ConstImageView {
        return ConstImageView{
            .data = data,
            .width = width,
            .height = height,
            .channels = channels,
            .stride = stride,
        };
    }

    pub fn getPixel(self: *const ConstImageView, x: u32, y: u32) []const u8 {
        const idx = y * self.stride + x * self.channels;
        return self.data[idx..idx + self.channels];
    }

    pub fn getRow(self: *const ConstImageView, y: u32) []const u8 {
        const start = y * self.stride;
        const end = start + (self.width * self.channels);
        return self.data[start..end];
    }

    pub fn subView(self: *const ConstImageView, x: u32, y: u32, w: u32, h: u32) ConstImageView {
        std.debug.assert(x + w <= self.width);
        std.debug.assert(y + h <= self.height);

        const start_idx = y * self.stride + x * self.channels;
        return ConstImageView{
            .data = self.data[start_idx..],
            .width = w,
            .height = h,
            .channels = self.channels,
            .stride = self.stride,
        };
    }
};

/// In-place operations that modify existing image data
pub const InPlaceOps = struct {
    /// Apply brightness adjustment in-place
    pub fn adjustBrightness(view: *ImageView, brightness: i32) void {
        for (0..view.height) |y| {
            const row = view.getRow(@intCast(y));
            for (0..row.len) |i| {
                const new_value = @as(i32, row[i]) + brightness;
                row[i] = @intCast(@min(255, @max(0, new_value)));
            }
        }
    }

    /// Apply contrast adjustment in-place
    pub fn adjustContrast(view: *ImageView, contrast: f32) void {
        for (0..view.height) |y| {
            const row = view.getRow(@intCast(y));
            for (0..row.len) |i| {
                const normalized = @as(f32, @floatFromInt(row[i])) / 255.0;
                const adjusted = (normalized - 0.5) * contrast + 0.5;
                row[i] = @intFromFloat(@min(255.0, @max(0.0, adjusted * 255.0)));
            }
        }
    }

    /// Convert to grayscale in-place (requires RGB data, modifies to single channel)
    pub fn toGrayscale(view: *ImageView) void {
        if (view.channels != 3) return;

        for (0..view.height) |y| {
            for (0..view.width) |x| {
                const pixel = view.getPixel(@intCast(x), @intCast(y));
                const gray = @as(u8, @intFromFloat(
                    0.299 * @as(f32, @floatFromInt(pixel[0])) +
                    0.587 * @as(f32, @floatFromInt(pixel[1])) +
                    0.114 * @as(f32, @floatFromInt(pixel[2]))
                ));
                pixel[0] = gray;
                // Leave other channels as-is for now (could be compacted)
            }
        }
    }

    /// Flip image horizontally in-place
    pub fn flipHorizontal(view: *ImageView) void {
        for (0..view.height) |y| {
            for (0..view.width / 2) |x| {
                const left_pixel = view.getPixel(@intCast(x), @intCast(y));
                const right_pixel = view.getPixel(@intCast(view.width - 1 - x), @intCast(y));

                // Swap pixels
                var temp: [4]u8 = undefined;
                @memcpy(temp[0..view.channels], left_pixel);
                @memcpy(left_pixel, right_pixel);
                @memcpy(right_pixel, temp[0..view.channels]);
            }
        }
    }

    /// Flip image vertically in-place
    pub fn flipVertical(view: *ImageView) void {
        for (0..view.height / 2) |y| {
            const top_row = view.getRow(@intCast(y));
            const bottom_row = view.getRow(@intCast(view.height - 1 - y));

            // Swap entire rows
            const row_size = view.width * view.channels;
            for (0..row_size) |i| {
                const temp = top_row[i];
                top_row[i] = bottom_row[i];
                bottom_row[i] = temp;
            }
        }
    }

    /// Rotate image 180 degrees in-place
    pub fn rotate180(view: *ImageView) void {
        const total_pixels = view.width * view.height;
        for (0..total_pixels / 2) |i| {
            const j = total_pixels - 1 - i;

            const y1 = i / view.width;
            const x1 = i % view.width;
            const y2 = j / view.width;
            const x2 = j % view.width;

            const pixel1 = view.getPixel(@intCast(x1), @intCast(y1));
            const pixel2 = view.getPixel(@intCast(x2), @intCast(y2));

            // Swap pixels
            var temp: [4]u8 = undefined;
            @memcpy(temp[0..view.channels], pixel1);
            @memcpy(pixel1, pixel2);
            @memcpy(pixel2, temp[0..view.channels]);
        }
    }
};

/// Zero-copy operations that work with views
pub const ZeroCopyOps = struct {
    /// Copy rectangle from source to destination view
    pub fn copyRect(src: *const ConstImageView, dst: *ImageView,
                   src_x: u32, src_y: u32, dst_x: u32, dst_y: u32, w: u32, h: u32) void {
        std.debug.assert(src.channels == dst.channels);
        std.debug.assert(src_x + w <= src.width);
        std.debug.assert(src_y + h <= src.height);
        std.debug.assert(dst_x + w <= dst.width);
        std.debug.assert(dst_y + h <= dst.height);

        for (0..h) |y| {
            const src_row_start = (src_y + y) * src.stride + src_x * src.channels;
            const dst_row_start = (dst_y + y) * dst.stride + dst_x * dst.channels;
            const copy_size = w * src.channels;

            @memcpy(
                dst.data[dst_row_start..dst_row_start + copy_size],
                src.data[src_row_start..src_row_start + copy_size]
            );
        }
    }

    /// Blend two views with alpha blending (dst = src * alpha + dst * (1 - alpha))
    pub fn blendViews(src: *const ConstImageView, dst: *ImageView, alpha: f32) void {
        std.debug.assert(src.width == dst.width);
        std.debug.assert(src.height == dst.height);
        std.debug.assert(src.channels == dst.channels);

        const inv_alpha = 1.0 - alpha;

        for (0..dst.height) |y| {
            for (0..dst.width) |x| {
                const src_pixel = src.getPixel(@intCast(x), @intCast(y));
                const dst_pixel = dst.getPixel(@intCast(x), @intCast(y));

                for (0..dst.channels) |c| {
                    const blended = @as(f32, @floatFromInt(src_pixel[c])) * alpha +
                                   @as(f32, @floatFromInt(dst_pixel[c])) * inv_alpha;
                    dst_pixel[c] = @intFromFloat(@min(255.0, @max(0.0, blended)));
                }
            }
        }
    }

    /// Compare two views and return difference statistics
    pub fn compareViews(view1: *const ConstImageView, view2: *const ConstImageView) ViewDifference {
        std.debug.assert(view1.width == view2.width);
        std.debug.assert(view1.height == view2.height);
        std.debug.assert(view1.channels == view2.channels);

        var diff = ViewDifference{};
        var total_diff: u64 = 0;
        var max_diff: u32 = 0;
        var diff_pixels: u32 = 0;

        for (0..view1.height) |y| {
            for (0..view1.width) |x| {
                const pixel1 = view1.getPixel(@intCast(x), @intCast(y));
                const pixel2 = view2.getPixel(@intCast(x), @intCast(y));

                var pixel_diff: u32 = 0;
                for (0..view1.channels) |c| {
                    const channel_diff = @abs(@as(i32, pixel1[c]) - @as(i32, pixel2[c]));
                    pixel_diff += @intCast(channel_diff);
                }

                total_diff += pixel_diff;
                max_diff = @max(max_diff, pixel_diff);
                if (pixel_diff > 0) diff_pixels += 1;
            }
        }

        const total_pixels = view1.width * view1.height;
        diff.average_difference = @as(f32, @floatFromInt(total_diff)) / @as(f32, @floatFromInt(total_pixels * view1.channels));
        diff.max_difference = max_diff;
        diff.different_pixels = diff_pixels;
        diff.similarity_percentage = 100.0 * @as(f32, @floatFromInt(total_pixels - diff_pixels)) / @as(f32, @floatFromInt(total_pixels));

        return diff;
    }
};

/// Statistics from comparing two image views
pub const ViewDifference = struct {
    average_difference: f32 = 0.0,
    max_difference: u32 = 0,
    different_pixels: u32 = 0,
    similarity_percentage: f32 = 100.0,
};

/// Memory-mapped file view for large image processing
pub const MappedImageView = struct {
    file: std.fs.File,
    mapping: []align(std.mem.page_size) u8,
    view: ImageView,

    pub fn init(file: std.fs.File, width: u32, height: u32, channels: u32) !MappedImageView {
        const file_size = try file.getEndPos();
        const mapping = try std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            file.handle,
            0
        );

        const view = ImageView.init(mapping, width, height, channels);

        return MappedImageView{
            .file = file,
            .mapping = mapping,
            .view = view,
        };
    }

    pub fn deinit(self: *MappedImageView) void {
        std.posix.munmap(self.mapping);
    }

    pub fn sync(self: *MappedImageView) !void {
        try std.posix.msync(self.mapping, .async);
    }
};

/// Streaming operations for processing large images in chunks
pub const StreamingOps = struct {
    /// Process image in horizontal strips to reduce memory usage
    pub fn processInStrips(src_view: *const ConstImageView, dst_view: *ImageView,
                          strip_height: u32, processor: fn(*const ConstImageView, *ImageView) void) void {
        var y: u32 = 0;
        while (y < src_view.height) {
            const current_height = @min(strip_height, src_view.height - y);

            const src_strip = src_view.subView(0, y, src_view.width, current_height);
            var dst_strip = dst_view.subView(0, y, dst_view.width, current_height);

            processor(&src_strip, &dst_strip);

            y += current_height;
        }
    }

    /// Process image in tiles for cache-friendly operations
    pub fn processInTiles(src_view: *const ConstImageView, dst_view: *ImageView,
                         tile_width: u32, tile_height: u32, processor: fn(*const ConstImageView, *ImageView) void) void {
        var y: u32 = 0;
        while (y < src_view.height) {
            const current_height = @min(tile_height, src_view.height - y);

            var x: u32 = 0;
            while (x < src_view.width) {
                const current_width = @min(tile_width, src_view.width - x);

                const src_tile = src_view.subView(x, y, current_width, current_height);
                var dst_tile = dst_view.subView(x, y, current_width, current_height);

                processor(&src_tile, &dst_tile);

                x += current_width;
            }

            y += current_height;
        }
    }
};

/// Buffer pool for reusing memory allocations
pub const BufferPool = struct {
    buffers: std.ArrayList([]u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BufferPool {
        return BufferPool{
            .buffers = std.ArrayList([]u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferPool) void {
        for (self.buffers.items) |buffer| {
            self.allocator.free(buffer);
        }
        self.buffers.deinit(self.allocator);
    }

    pub fn acquire(self: *BufferPool, size: usize) ![]u8 {
        // Look for a suitable existing buffer
        for (self.buffers.items, 0..) |buffer, i| {
            if (buffer.len >= size) {
                // Remove from pool and return
                const result = self.buffers.swapRemove(i);
                return result[0..size];
            }
        }

        // No suitable buffer found, allocate new one
        return try self.allocator.alloc(u8, size);
    }

    pub fn release(self: *BufferPool, buffer: []u8) !void {
        // Expand the slice back to its original allocation size if needed
        try self.buffers.append(self.allocator, buffer);
    }
};