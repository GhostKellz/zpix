//! Comprehensive metadata support for zpix
//! Handles EXIF, XMP, and IPTC metadata reading and writing

const std = @import("std");

/// EXIF data types
pub const ExifDataType = enum(u16) {
    byte = 1,       // BYTE
    ascii = 2,      // ASCII
    short = 3,      // SHORT
    long = 4,       // LONG
    rational = 5,   // RATIONAL
    sbyte = 6,      // SBYTE
    undefined = 7,  // UNDEFINED
    sshort = 8,     // SSHORT
    slong = 9,      // SLONG
    srational = 10, // SRATIONAL
    float = 11,     // FLOAT
    double = 12,    // DOUBLE
};

/// EXIF tag definitions
pub const ExifTag = enum(u16) {
    // IFD0 tags
    image_width = 0x0100,
    image_length = 0x0101,
    bits_per_sample = 0x0102,
    compression = 0x0103,
    photometric_interpretation = 0x0106,
    image_description = 0x010E,
    make = 0x010F,
    model = 0x0110,
    strip_offsets = 0x0111,
    orientation = 0x0112,
    samples_per_pixel = 0x0115,
    rows_per_strip = 0x0116,
    strip_byte_counts = 0x0117,
    x_resolution = 0x011A,
    y_resolution = 0x011B,
    planar_configuration = 0x011C,
    resolution_unit = 0x0128,
    transfer_function = 0x012D,
    software = 0x0131,
    date_time = 0x0132,
    artist = 0x013B,
    white_point = 0x013E,
    primary_chromaticities = 0x013F,
    color_map = 0x0140,
    tile_width = 0x0142,
    tile_length = 0x0143,
    tile_offsets = 0x0144,
    tile_byte_counts = 0x0145,
    ycbcr_coefficients = 0x0211,
    ycbcr_sub_sampling = 0x0212,
    ycbcr_positioning = 0x0213,
    reference_black_white = 0x0214,
    copyright = 0x8298,
    exif_ifd_pointer = 0x8769,
    gps_info_ifd_pointer = 0x8825,

    // EXIF IFD tags
    exposure_time = 0x829A,
    f_number = 0x829D,
    exposure_program = 0x8822,
    spectral_sensitivity = 0x8824,
    iso_speed_ratings = 0x8827,
    oecf = 0x8828,
    exif_version = 0x9000,
    date_time_original = 0x9003,
    date_time_digitized = 0x9004,
    components_configuration = 0x9101,
    compressed_bits_per_pixel = 0x9102,
    shutter_speed_value = 0x9201,
    aperture_value = 0x9202,
    brightness_value = 0x9203,
    exposure_bias_value = 0x9204,
    max_aperture_value = 0x9205,
    subject_distance = 0x9206,
    metering_mode = 0x9207,
    light_source = 0x9208,
    flash = 0x9209,
    focal_length = 0x920A,
    subject_area = 0x9214,
    maker_note = 0x927C,
    user_comment = 0x9286,
    subsec_time = 0x9290,
    subsec_time_original = 0x9291,
    subsec_time_digitized = 0x9292,
    flashpix_version = 0xA000,
    color_space = 0xA001,
    pixel_x_dimension = 0xA002,
    pixel_y_dimension = 0xA003,
    related_sound_file = 0xA004,
    flash_energy = 0xA20B,
    spatial_frequency_response = 0xA20C,
    focal_plane_x_resolution = 0xA20E,
    focal_plane_y_resolution = 0xA20F,
    focal_plane_resolution_unit = 0xA210,
    subject_location = 0xA214,
    exposure_index = 0xA215,
    sensing_method = 0xA217,
    file_source = 0xA300,
    scene_type = 0xA301,
    cfa_pattern = 0xA302,
    custom_rendered = 0xA401,
    exposure_mode = 0xA402,
    white_balance = 0xA403,
    digital_zoom_ratio = 0xA404,
    focal_length_in_35mm_film = 0xA405,
    scene_capture_type = 0xA406,
    gain_control = 0xA407,
    contrast = 0xA408,
    saturation = 0xA409,
    sharpness = 0xA40A,
    device_setting_description = 0xA40B,
    subject_distance_range = 0xA40C,
    image_unique_id = 0xA420,
    lens_specification = 0xA432,
    lens_make = 0xA433,
    lens_model = 0xA434,

    // GPS IFD tags
    gps_version_id = 0x0000,
    gps_latitude_ref = 0x0001,
    gps_latitude = 0x0002,
    gps_longitude_ref = 0x0003,
    gps_longitude = 0x0004,
    gps_altitude_ref = 0x0005,
    gps_altitude = 0x0006,
    gps_time_stamp = 0x0007,
    gps_satellites = 0x0008,
    gps_status = 0x0009,
    gps_measure_mode = 0x000A,
    gps_dop = 0x000B,
    gps_speed_ref = 0x000C,
    gps_speed = 0x000D,
    gps_track_ref = 0x000E,
    gps_track = 0x000F,
    gps_img_direction_ref = 0x0010,
    gps_img_direction = 0x0011,
    gps_map_datum = 0x0012,
    gps_dest_latitude_ref = 0x0013,
    gps_dest_latitude = 0x0014,
    gps_dest_longitude_ref = 0x0015,
    gps_dest_longitude = 0x0016,
    gps_dest_bearing_ref = 0x0017,
    gps_dest_bearing = 0x0018,
    gps_dest_distance_ref = 0x0019,
    gps_dest_distance = 0x001A,
    gps_processing_method = 0x001B,
    gps_area_information = 0x001C,
    gps_date_stamp = 0x001D,
    gps_differential = 0x001E,
};

/// EXIF entry
pub const ExifEntry = struct {
    tag: ExifTag,
    data_type: ExifDataType,
    count: u32,
    value: ExifValue,

    pub fn init(tag: ExifTag, data_type: ExifDataType, count: u32, value: ExifValue) ExifEntry {
        return ExifEntry{
            .tag = tag,
            .data_type = data_type,
            .count = count,
            .value = value,
        };
    }
};

/// EXIF value union
pub const ExifValue = union(ExifDataType) {
    byte: []u8,
    ascii: []u8,
    short: []u16,
    long: []u32,
    rational: []Rational,
    sbyte: []i8,
    undefined: []u8,
    sshort: []i16,
    slong: []i32,
    srational: []SignedRational,
    float: []f32,
    double: []f64,
};

/// EXIF rational number
pub const Rational = struct {
    numerator: u32,
    denominator: u32,

    pub fn toFloat(self: Rational) f64 {
        if (self.denominator == 0) return 0.0;
        return @as(f64, @floatFromInt(self.numerator)) / @as(f64, @floatFromInt(self.denominator));
    }

    pub fn fromFloat(value: f64) Rational {
        // Simple conversion - could be improved with continued fractions
        const denominator: u32 = 10000;
        const numerator: u32 = @intFromFloat(value * @as(f64, @floatFromInt(denominator)));
        return Rational{ .numerator = numerator, .denominator = denominator };
    }
};

/// EXIF signed rational number
pub const SignedRational = struct {
    numerator: i32,
    denominator: i32,

    pub fn toFloat(self: SignedRational) f64 {
        if (self.denominator == 0) return 0.0;
        return @as(f64, @floatFromInt(self.numerator)) / @as(f64, @floatFromInt(self.denominator));
    }
};

/// EXIF data container
pub const ExifData = struct {
    entries: std.HashMap(ExifTag, ExifEntry, ExifTagContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    const ExifTagContext = struct {
        pub fn hash(self: @This(), key: ExifTag) u64 {
            _ = self;
            return @intFromEnum(key);
        }

        pub fn eql(self: @This(), a: ExifTag, b: ExifTag) bool {
            _ = self;
            return a == b;
        }
    };

    pub fn init(allocator: std.mem.Allocator) ExifData {
        return ExifData{
            .entries = std.HashMap(ExifTag, ExifEntry, ExifTagContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExifData) void {
        var iterator = self.entries.iterator();
        while (iterator.next()) |entry| {
            freeExifValue(self.allocator, entry.value_ptr.value);
        }
        self.entries.deinit();
    }

    pub fn addEntry(self: *ExifData, entry: ExifEntry) !void {
        try self.entries.put(entry.tag, entry);
    }

    pub fn getEntry(self: *const ExifData, tag: ExifTag) ?ExifEntry {
        return self.entries.get(tag);
    }

    pub fn getString(self: *const ExifData, tag: ExifTag) ?[]const u8 {
        if (self.getEntry(tag)) |entry| {
            return switch (entry.value) {
                .ascii => |data| data,
                else => null,
            };
        }
        return null;
    }

    pub fn getNumber(self: *const ExifData, tag: ExifTag) ?f64 {
        if (self.getEntry(tag)) |entry| {
            return switch (entry.value) {
                .short => |data| if (data.len > 0) @floatFromInt(data[0]) else null,
                .long => |data| if (data.len > 0) @floatFromInt(data[0]) else null,
                .rational => |data| if (data.len > 0) data[0].toFloat() else null,
                .sshort => |data| if (data.len > 0) @floatFromInt(data[0]) else null,
                .slong => |data| if (data.len > 0) @floatFromInt(data[0]) else null,
                .srational => |data| if (data.len > 0) data[0].toFloat() else null,
                .float => |data| if (data.len > 0) data[0] else null,
                .double => |data| if (data.len > 0) data[0] else null,
                else => null,
            };
        }
        return null;
    }
};

/// Parse EXIF data from JPEG file
pub fn parseExifFromJpeg(allocator: std.mem.Allocator, file: std.fs.File) !ExifData {
    try file.seekTo(0);

    var exif_data = ExifData.init(allocator);
    errdefer exif_data.deinit();

    // Look for EXIF marker in JPEG
    var buffer: [65536]u8 = undefined;
    var pos: u64 = 2; // Skip JPEG SOI marker

    while (true) {
        try file.seekTo(pos);
        const bytes_read = try file.read(buffer[0..4]);
        if (bytes_read < 4) break;

        if (buffer[0] != 0xFF) break;

        const marker = buffer[1];
        const length = std.mem.readInt(u16, buffer[2..4], .big);

        if (marker == 0xE1) { // APP1 marker (EXIF)
            if (length < 6) continue;

            try file.seekTo(pos + 4);
            const header_bytes = try file.read(buffer[0..@min(length - 2, buffer.len)]);

            if (header_bytes >= 6 and std.mem.eql(u8, buffer[0..4], "Exif")) {
                // Found EXIF data
                const exif_start = pos + 10; // Skip marker, length, and "Exif\0\0"
                try parseExifData(allocator, &exif_data, file, exif_start, length - 8);
                break;
            }
        }

        pos += 2 + length;
    }

    return exif_data;
}

/// Parse TIFF-format EXIF data
fn parseExifData(allocator: std.mem.Allocator, exif_data: *ExifData, file: std.fs.File, offset: u64, size: u32) !void {
    try file.seekTo(offset);

    var header: [8]u8 = undefined;
    _ = try file.read(&header);

    // Determine endianness
    const is_little_endian = std.mem.eql(u8, header[0..2], "II");
    const endian: std.builtin.Endian = if (is_little_endian) .little else .big;

    // Verify TIFF magic number
    const magic = std.mem.readInt(u16, header[2..4], endian);
    if (magic != 42) return error.InvalidExifData;

    // Get first IFD offset
    const first_ifd_offset = std.mem.readInt(u32, header[4..8], endian);

    // Parse IFD
    try parseIfd(allocator, exif_data, file, offset + first_ifd_offset, endian, offset, size);
}

/// Parse Image File Directory
fn parseIfd(allocator: std.mem.Allocator, exif_data: *ExifData, file: std.fs.File, ifd_offset: u64, endian: std.builtin.Endian, base_offset: u64, max_size: u32) !void {
    try file.seekTo(ifd_offset);

    var count_buf: [2]u8 = undefined;
    _ = try file.read(&count_buf);
    const entry_count = std.mem.readInt(u16, &count_buf, endian);

    for (0..entry_count) |_| {
        var entry_buf: [12]u8 = undefined;
        _ = try file.read(&entry_buf);

        const tag_value = std.mem.readInt(u16, entry_buf[0..2], endian);
        const data_type_value = std.mem.readInt(u16, entry_buf[2..4], endian);
        const count = std.mem.readInt(u32, entry_buf[4..8], endian);
        const value_offset = std.mem.readInt(u32, entry_buf[8..12], endian);

        // Convert to enums
        const tag: ExifTag = @enumFromInt(tag_value);
        const data_type: ExifDataType = @enumFromInt(data_type_value);

        // Read value data
        const value = try readExifValue(allocator, file, data_type, count, value_offset, endian, base_offset, max_size);
        const entry = ExifEntry.init(tag, data_type, count, value);

        try exif_data.addEntry(entry);

        // Handle sub-IFDs
        if (tag == .exif_ifd_pointer and data_type == .long and count > 0) {
            switch (value) {
                .long => |data| {
                    if (data.len > 0) {
                        try parseIfd(allocator, exif_data, file, base_offset + data[0], endian, base_offset, max_size);
                    }
                },
                else => {},
            }
        }
    }
}

/// Read EXIF value based on type and count
fn readExifValue(allocator: std.mem.Allocator, file: std.fs.File, data_type: ExifDataType, count: u32, value_offset: u32, endian: std.builtin.Endian, base_offset: u64, max_size: u32) !ExifValue {
    const type_size = getDataTypeSize(data_type);
    const total_size = type_size * count;

    var data: []u8 = undefined;

    if (total_size <= 4) {
        // Value is stored directly in the offset field
        data = try allocator.alloc(u8, 4);
        std.mem.writeInt(u32, data[0..4], value_offset, endian);
        data = data[0..total_size];
    } else {
        // Value is stored at the offset location
        if (value_offset + total_size > max_size) return error.InvalidExifData;

        try file.seekTo(base_offset + value_offset);
        data = try allocator.alloc(u8, total_size);
        _ = try file.read(data);
    }

    return switch (data_type) {
        .byte => ExifValue{ .byte = data },
        .ascii => ExifValue{ .ascii = data },
        .short => blk: {
            const shorts = try allocator.alloc(u16, count);
            for (shorts, 0..) |*short, i| {
                const offset_in_data = i * 2;
                short.* = std.mem.readInt(u16, data[offset_in_data..offset_in_data + 2], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .short = shorts };
        },
        .long => blk: {
            const longs = try allocator.alloc(u32, count);
            for (longs, 0..) |*long, i| {
                const offset_in_data = i * 4;
                long.* = std.mem.readInt(u32, data[offset_in_data..offset_in_data + 4], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .long = longs };
        },
        .rational => blk: {
            const rationals = try allocator.alloc(Rational, count);
            for (rationals, 0..) |*rational, i| {
                const offset_in_data = i * 8;
                rational.numerator = std.mem.readInt(u32, data[offset_in_data..offset_in_data + 4], endian);
                rational.denominator = std.mem.readInt(u32, data[offset_in_data + 4..offset_in_data + 8], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .rational = rationals };
        },
        .sbyte => ExifValue{ .sbyte = @as([]i8, @ptrCast(data)) },
        .undefined => ExifValue{ .undefined = data },
        .sshort => blk: {
            const sshorts = try allocator.alloc(i16, count);
            for (sshorts, 0..) |*sshort, i| {
                const offset_in_data = i * 2;
                sshort.* = std.mem.readInt(i16, data[offset_in_data..offset_in_data + 2], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .sshort = sshorts };
        },
        .slong => blk: {
            const slongs = try allocator.alloc(i32, count);
            for (slongs, 0..) |*slong, i| {
                const offset_in_data = i * 4;
                slong.* = std.mem.readInt(i32, data[offset_in_data..offset_in_data + 4], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .slong = slongs };
        },
        .srational => blk: {
            const srationals = try allocator.alloc(SignedRational, count);
            for (srationals, 0..) |*srational, i| {
                const offset_in_data = i * 8;
                srational.numerator = std.mem.readInt(i32, data[offset_in_data..offset_in_data + 4], endian);
                srational.denominator = std.mem.readInt(i32, data[offset_in_data + 4..offset_in_data + 8], endian);
            }
            allocator.free(data);
            break :blk ExifValue{ .srational = srationals };
        },
        .float => blk: {
            const floats = try allocator.alloc(f32, count);
            for (floats, 0..) |*float, i| {
                const offset_in_data = i * 4;
                const int_val = std.mem.readInt(u32, data[offset_in_data..offset_in_data + 4], endian);
                float.* = @bitCast(int_val);
            }
            allocator.free(data);
            break :blk ExifValue{ .float = floats };
        },
        .double => blk: {
            const doubles = try allocator.alloc(f64, count);
            for (doubles, 0..) |*double, i| {
                const offset_in_data = i * 8;
                const int_val = std.mem.readInt(u64, data[offset_in_data..offset_in_data + 8], endian);
                double.* = @bitCast(int_val);
            }
            allocator.free(data);
            break :blk ExifValue{ .double = doubles };
        },
    };
}

/// Get size in bytes for EXIF data type
fn getDataTypeSize(data_type: ExifDataType) u32 {
    return switch (data_type) {
        .byte, .ascii, .sbyte, .undefined => 1,
        .short, .sshort => 2,
        .long, .slong, .float => 4,
        .rational, .srational, .double => 8,
    };
}

/// Free EXIF value memory
fn freeExifValue(allocator: std.mem.Allocator, value: ExifValue) void {
    switch (value) {
        .byte => |data| allocator.free(data),
        .ascii => |data| allocator.free(data),
        .short => |data| allocator.free(data),
        .long => |data| allocator.free(data),
        .rational => |data| allocator.free(data),
        .sbyte => |data| allocator.free(@as([]u8, @ptrCast(data))),
        .undefined => |data| allocator.free(data),
        .sshort => |data| allocator.free(data),
        .slong => |data| allocator.free(data),
        .srational => |data| allocator.free(data),
        .float => |data| allocator.free(data),
        .double => |data| allocator.free(data),
    }
}

/// XMP metadata support
pub const XmpData = struct {
    xml_content: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, xml_content: []const u8) !XmpData {
        return XmpData{
            .xml_content = try allocator.dupe(u8, xml_content),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *XmpData) void {
        self.allocator.free(self.xml_content);
    }

    pub fn getValue(self: *const XmpData, namespace: []const u8, property: []const u8) ?[]const u8 {
        // Simple XMP property extraction (could be enhanced with proper XML parsing)
        const search_pattern = try std.fmt.allocPrint(self.allocator, "{s}:{s}=\"", .{ namespace, property }) catch return null;
        defer self.allocator.free(search_pattern);

        if (std.mem.indexOf(u8, self.xml_content, search_pattern)) |start| {
            const value_start = start + search_pattern.len;
            if (std.mem.indexOf(u8, self.xml_content[value_start..], "\"")) |end| {
                return self.xml_content[value_start..value_start + end];
            }
        }
        return null;
    }
};

/// IPTC metadata support
pub const IptcData = struct {
    records: std.HashMap(IptcTag, []u8, IptcTagContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    const IptcTagContext = struct {
        pub fn hash(self: @This(), key: IptcTag) u64 {
            _ = self;
            return @intFromEnum(key);
        }

        pub fn eql(self: @This(), a: IptcTag, b: IptcTag) bool {
            _ = self;
            return a == b;
        }
    };

    pub fn init(allocator: std.mem.Allocator) IptcData {
        return IptcData{
            .records = std.HashMap(IptcTag, []u8, IptcTagContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IptcData) void {
        var iterator = self.records.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.records.deinit();
    }

    pub fn getValue(self: *const IptcData, tag: IptcTag) ?[]const u8 {
        return self.records.get(tag);
    }

    pub fn setValue(self: *IptcData, tag: IptcTag, value: []const u8) !void {
        const owned_value = try self.allocator.dupe(u8, value);
        try self.records.put(tag, owned_value);
    }
};

/// IPTC tag definitions
pub const IptcTag = enum(u8) {
    object_type_reference = 3,
    object_attribute_reference = 4,
    object_name = 5,
    edit_status = 7,
    editorial_update = 8,
    urgency = 10,
    subject_reference = 12,
    category = 15,
    supplemental_category = 20,
    fixture_identifier = 22,
    keywords = 25,
    content_location_code = 26,
    content_location_name = 27,
    release_date = 30,
    release_time = 35,
    expiration_date = 37,
    expiration_time = 38,
    special_instructions = 40,
    action_advised = 42,
    reference_service = 45,
    reference_date = 47,
    reference_number = 50,
    date_created = 55,
    time_created = 60,
    digital_creation_date = 62,
    digital_creation_time = 63,
    originating_program = 65,
    program_version = 70,
    object_cycle = 75,
    byline = 80,
    byline_title = 85,
    city = 90,
    sub_location = 92,
    province_state = 95,
    country_primary_location_code = 100,
    country_primary_location_name = 101,
    original_transmission_reference = 103,
    headline = 105,
    credit = 110,
    source = 115,
    copyright_notice = 116,
    contact = 118,
    caption_abstract = 120,
    writer_editor = 122,
    rasterized_caption = 125,
    image_type = 130,
    image_orientation = 131,
    language_identifier = 135,
    audio_type = 150,
    audio_sampling_rate = 151,
    audio_sampling_resolution = 152,
    audio_duration = 153,
    audio_outcue = 154,
};

/// Extract metadata from various image formats
pub fn extractMetadata(allocator: std.mem.Allocator, file: std.fs.File, file_path: []const u8) !struct {
    exif: ?ExifData = null,
    xmp: ?XmpData = null,
    iptc: ?IptcData = null,
} {
    const extension = std.fs.path.extension(file_path);

    var result = .{
        .exif = null,
        .xmp = null,
        .iptc = null,
    };

    if (std.ascii.eqlIgnoreCase(extension, ".jpg") or std.ascii.eqlIgnoreCase(extension, ".jpeg")) {
        // JPEG metadata extraction
        result.exif = parseExifFromJpeg(allocator, file) catch null;
        // TODO: Extract XMP and IPTC from JPEG
    } else if (std.ascii.eqlIgnoreCase(extension, ".tiff") or std.ascii.eqlIgnoreCase(extension, ".tif")) {
        // TIFF metadata extraction
        // TODO: Implement TIFF metadata extraction
    }
    // TODO: Add support for other formats

    return result;
}