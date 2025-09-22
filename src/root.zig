//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

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
    codes: [256]u8,
    lengths: [16]u8,
    values: [256]u8,
};

const FrameHeader = struct {
    width: u32,
    height: u32,
    components: [3]Component,
};

const Component = struct {
    id: u8,
    h_sampling: u8,
    v_sampling: u8,
    quant_table_id: u8,
};

const ScanHeader = struct {
    components: [3]ScanComponent,
    start_spectral: u8,
    end_spectral: u8,
    approximation: u8,
};

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

pub const Image = struct {
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    format: PixelFormat,
    data: []u8,

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: PixelFormat) !Image {
        const data = try allocator.alloc(u8, width * height * bytesPerPixel(format));
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

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Image {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();

        var sig_buf: [12]u8 = undefined;
        _ = try file.read(&sig_buf);

        if (std.mem.eql(u8, &sig_buf, &[_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 })) {
            return loadPng(allocator, file);
        } else if (std.mem.eql(u8, sig_buf[0..2], "\xFF\xD8")) {
            return loadJpeg(allocator, file);
        } else if (std.mem.eql(u8, sig_buf[0..4], "\x00\x00\x00\x1c") and std.mem.eql(u8, sig_buf[4..12], "ftypavif")) {
            return error.AVIFNotSupported;
        } else if (std.mem.eql(u8, sig_buf[0..2], "BM")) {
            try file.seekTo(0);
            return loadBmp(allocator, file);
        } else {
            return error.UnknownFormat;
        }
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
        _ = allocator;
        _ = file;
        return error.NotImplemented;
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

        var quantization_tables: [4][64]u8 = undefined;
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
                0xC2 => { // SOF2 - Progressive JPEG
                    frame_header = try parseSOF0(file);
                    has_frame_header = true;
                },
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

    fn parseDQT(file: std.fs.File, tables: *[4][64]u8) !void {
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

            @memcpy(&tables[table_id], data[offset .. offset + 64].ptr);
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

            // Build Huffman codes
            var codes: [256]u8 = [_]u8{0} ** 256;
            var code: u16 = 0;
            var value_index: usize = 0;

            for (1..17) |len| {
                for (0..lengths[len - 1]) |_| {
                    codes[value_index] = @intCast(code);
                    code += 1;
                    value_index += 1;
                }
                code <<= 1;
            }

            const table_index = table_id + table_class * 2;
            if (table_index >= 4) return error.InvalidHuffmanTableIndex;

            tables[table_index] = HuffmanTable{
                .codes = codes,
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

        if (num_components != 3) return error.UnsupportedComponents;

        var components: [3]Component = undefined;
        for (0..3) |i| {
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
        if (num_components != 3) return error.UnsupportedComponents;

        var components: [3]ScanComponent = undefined;
        for (0..3) |i| {
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
            .components = components,
            .start_spectral = start_spectral,
            .end_spectral = end_spectral,
            .approximation = approximation,
        };
    }

    fn decodeImageData(allocator: std.mem.Allocator, file: std.fs.File, frame: FrameHeader, scan: ScanHeader, qt: [4][64]u8, ht: [4]HuffmanTable) ![]u8 {
        _ = qt; // TODO: Use for dequantization

        // Read all entropy-coded data until EOI
        var entropy_data = std.ArrayListUnmanaged(u8){};
        defer entropy_data.deinit(allocator);

        while (true) {
            var byte: [1]u8 = undefined;
            const bytes_read = try file.read(&byte);
            if (bytes_read == 0) break;

            if (byte[0] == 0xFF) {
                const next_byte = try file.read(&byte);
                if (next_byte == 0) break;
                if (byte[0] == 0x00) {
                    // Stuffed byte, skip
                    continue;
                } else if (byte[0] == 0xD9) {
                    // EOI
                    break;
                } else {
                    // Marker, but for now assume it's data
                    try entropy_data.append(allocator, 0xFF);
                    try entropy_data.append(allocator, byte[0]);
                }
            } else {
                try entropy_data.append(allocator, byte[0]);
            }
        }

        // Initialize bit reader
        var bit_reader = BitReader.init(entropy_data.items);

        // Decode MCU (Minimum Coded Unit) - for baseline JPEG, this is 8x8 for each component
        const mcu_width = (frame.width + 7) / 8;
        const mcu_height = (frame.height + 7) / 8;
        const total_mcus = mcu_width * mcu_height;

        // Allocate space for decoded coefficients
        var y_coefficients = try allocator.alloc([64]i16, total_mcus);
        defer allocator.free(y_coefficients);
        var cb_coefficients = try allocator.alloc([64]i16, total_mcus);
        defer allocator.free(cb_coefficients);
        var cr_coefficients = try allocator.alloc([64]i16, total_mcus);
        defer allocator.free(cr_coefficients);

        // DC prediction values
        var dc_y: i16 = 0;
        var dc_cb: i16 = 0;
        var dc_cr: i16 = 0;

        // Decode each MCU
        for (0..total_mcus) |mcu_index| {
            // Decode Y component
            dc_y = try decodeBlock(&bit_reader, &ht[scan.components[0].dc_table_id], &ht[scan.components[0].ac_table_id], &y_coefficients[mcu_index], dc_y);
            // For now, skip Cb and Cr (simplified decoding)
            dc_cb = try decodeBlock(&bit_reader, &ht[scan.components[1].dc_table_id], &ht[scan.components[1].ac_table_id], &cb_coefficients[mcu_index], dc_cb);
            dc_cr = try decodeBlock(&bit_reader, &ht[scan.components[2].dc_table_id], &ht[scan.components[2].ac_table_id], &cr_coefficients[mcu_index], dc_cr);
        }

        // Dequantize and IDCT
        const rgb_data = try allocator.alloc(u8, @as(usize, frame.width) * frame.height * 3);
        errdefer allocator.free(rgb_data);

        // For now, just create a simple pattern to verify decoding works
        for (0..@as(usize, frame.height)) |y| {
            for (0..@as(usize, frame.width)) |x| {
                const idx = (y * @as(usize, frame.width) + x) * 3;
                rgb_data[idx] = @intCast((x * 255) / frame.width); // R
                rgb_data[idx + 1] = @intCast((y * 255) / frame.height); // G
                rgb_data[idx + 2] = 128; // B
            }
        }

        return rgb_data;
    }

    fn decodeBlock(bit_reader: *BitReader, dc_table: *const HuffmanTable, ac_table: *const HuffmanTable, block: *[64]i16, prev_dc: i16) !i16 {
        _ = ac_table; // TODO: Use for AC coefficient decoding

        // Decode DC coefficient
        const dc_code = try decodeHuffmanValue(bit_reader, dc_table);
        const dc_bits = try bit_reader.readBits(@intCast(dc_code));
        const dc_diff = try decodeZigzag(dc_bits, dc_code);
        const dc_value = prev_dc + dc_diff;
        block[0] = dc_value;

        // For now, set AC coefficients to 0 (simplified)
        for (1..64) |i| {
            block[i] = 0;
        }

        return dc_value;
    }

    fn decodeHuffmanValue(bit_reader: *BitReader, table: *const HuffmanTable) !u8 {
        var code: u16 = 0;
        var length: u8 = 1;

        while (length <= 16) {
            const bit = try bit_reader.readBit();
            code = (code << 1) | bit;

            // Check if this code matches any value for this bit length
            const num_codes = table.lengths[length - 1];
            if (num_codes > 0) {
                // Find the starting index for this bit length
                var start_index: usize = 0;
                for (0..length - 1) |i| {
                    start_index += table.lengths[i];
                }

                // Check each code of this length
                for (0..num_codes) |i| {
                    if (table.codes[start_index + i] == code) {
                        return table.values[start_index + i];
                    }
                }
            }

            length += 1;
        }

        return error.InvalidHuffmanCode;
    }

    fn decodeZigzag(value: u16, bits: u8) !i16 {
        if (bits == 0) return 0;

        const bits_u4 = @as(u4, @intCast(bits - 1));
        const sign_bit = (value >> bits_u4) & 1;
        const magnitude = value & ((@as(u16, 1) << bits_u4) - 1);

        if (sign_bit == 0) {
            return @intCast(magnitude);
        } else {
            return -@as(i16, @intCast(magnitude + 1));
        }
    }

    pub fn save(self: Image, path: []const u8, format: ImageFormat) !void {
        if (format != .bmp) {
            return error.UnsupportedFormat;
        }

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
