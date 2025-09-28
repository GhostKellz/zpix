//! Color profile support for zpix
//! Provides sRGB, Adobe RGB, and other standard color space definitions
//! Includes ICC profile support for professional color management

const std = @import("std");
const math = std.math;

/// Standard color profiles
pub const ColorProfile = enum {
    srgb,
    adobe_rgb_1998,
    pro_photo_rgb,
    rec2020,
    display_p3,
    rec709,
    linear_rgb,

    pub fn getName(self: ColorProfile) []const u8 {
        return switch (self) {
            .srgb => "sRGB IEC61966-2.1",
            .adobe_rgb_1998 => "Adobe RGB (1998)",
            .pro_photo_rgb => "ProPhoto RGB",
            .rec2020 => "ITU-R BT.2020",
            .display_p3 => "Display P3",
            .rec709 => "ITU-R BT.709",
            .linear_rgb => "Linear RGB",
        };
    }

    pub fn getGamma(self: ColorProfile) f32 {
        return switch (self) {
            .srgb => 2.4, // sRGB uses mixed gamma
            .adobe_rgb_1998 => 2.2,
            .pro_photo_rgb => 1.8,
            .rec2020 => 2.4,
            .display_p3 => 2.4,
            .rec709 => 2.4,
            .linear_rgb => 1.0, // Linear
        };
    }

    pub fn getWhitePoint(self: ColorProfile) [2]f32 {
        return switch (self) {
            .srgb, .rec709, .rec2020 => [2]f32{ 0.3127, 0.3290 }, // D65
            .adobe_rgb_1998, .pro_photo_rgb => [2]f32{ 0.3127, 0.3290 }, // D65
            .display_p3 => [2]f32{ 0.3127, 0.3290 }, // D65
            .linear_rgb => [2]f32{ 0.3127, 0.3290 }, // D65
        };
    }

    pub fn getPrimaries(self: ColorProfile) [6]f32 {
        return switch (self) {
            .srgb, .rec709 => [6]f32{
                0.6400, 0.3300, // Red
                0.3000, 0.6000, // Green
                0.1500, 0.0600, // Blue
            },
            .adobe_rgb_1998 => [6]f32{
                0.6400, 0.3300, // Red
                0.2100, 0.7100, // Green
                0.1500, 0.0600, // Blue
            },
            .pro_photo_rgb => [6]f32{
                0.7347, 0.2653, // Red
                0.1596, 0.8404, // Green
                0.0366, 0.0001, // Blue
            },
            .rec2020 => [6]f32{
                0.7080, 0.2920, // Red
                0.1700, 0.7970, // Green
                0.1310, 0.0460, // Blue
            },
            .display_p3 => [6]f32{
                0.6800, 0.3200, // Red
                0.2650, 0.6900, // Green
                0.1500, 0.0600, // Blue
            },
            .linear_rgb => [6]f32{
                0.6400, 0.3300, // Red (same as sRGB)
                0.3000, 0.6000, // Green
                0.1500, 0.0600, // Blue
            },
        };
    }

    pub fn getRgbToXyzMatrix(self: ColorProfile) [9]f32 {
        return switch (self) {
            .srgb, .rec709, .linear_rgb => [9]f32{
                0.4124564, 0.3575761, 0.1804375,
                0.2126729, 0.7151522, 0.0721750,
                0.0193339, 0.1191920, 0.9503041,
            },
            .adobe_rgb_1998 => [9]f32{
                0.5767309, 0.1855540, 0.1881852,
                0.2973769, 0.6273491, 0.0752741,
                0.0270343, 0.0706872, 0.9911085,
            },
            .pro_photo_rgb => [9]f32{
                0.7976749, 0.1351917, 0.0313534,
                0.2880402, 0.7118741, 0.0000857,
                0.0000000, 0.0000000, 0.8252100,
            },
            .rec2020 => [9]f32{
                0.6369580, 0.1446169, 0.1688809,
                0.2627045, 0.6780980, 0.0593017,
                0.0000000, 0.0280727, 1.0609851,
            },
            .display_p3 => [9]f32{
                0.4865709, 0.2656677, 0.1982173,
                0.2289746, 0.6917385, 0.0792869,
                0.0000000, 0.0451134, 1.0439444,
            },
        };
    }

    pub fn getXyzToRgbMatrix(self: ColorProfile) [9]f32 {
        return switch (self) {
            .srgb, .rec709, .linear_rgb => [9]f32{
                3.2404542, -1.5371385, -0.4985314,
                -0.9692660, 1.8760108, 0.0415560,
                0.0556434, -0.2040259, 1.0572252,
            },
            .adobe_rgb_1998 => [9]f32{
                2.0413690, -0.5649464, -0.3446944,
                -0.9692660, 1.8760108, 0.0415560,
                0.0134474, -0.1183897, 1.0154096,
            },
            .pro_photo_rgb => [9]f32{
                1.3459433, -0.2556075, -0.0511118,
                -0.5445989, 1.5081673, 0.0205351,
                0.0000000, 0.0000000, 1.2118128,
            },
            .rec2020 => [9]f32{
                1.7166511, -0.3556708, -0.2533663,
                -0.6666844, 1.6164812, 0.0157685,
                0.0176399, -0.0427706, 0.9421031,
            },
            .display_p3 => [9]f32{
                2.4934969, -0.9313836, -0.4027108,
                -0.8294890, 1.7626641, 0.0236247,
                0.0358458, -0.0761723, 0.9568845,
            },
        };
    }
};

/// ICC profile header structure
pub const IccProfileHeader = struct {
    profile_size: u32,
    preferred_cmm_type: [4]u8,
    profile_version: u32,
    device_class: [4]u8,
    color_space: [4]u8,
    pcs: [4]u8,
    creation_date: [12]u8,
    platform_signature: [4]u8,
    profile_flags: u32,
    device_manufacturer: [4]u8,
    device_model: [4]u8,
    device_attributes: u64,
    rendering_intent: u32,
    illuminant: [12]u8,
    profile_creator: [4]u8,
    reserved: [44]u8,

    pub fn init() IccProfileHeader {
        return std.mem.zeroes(IccProfileHeader);
    }
};

/// ICC profile tag entry
pub const IccTagEntry = struct {
    signature: [4]u8,
    offset: u32,
    size: u32,
};

/// ICC profile data
pub const IccProfile = struct {
    header: IccProfileHeader,
    tag_table: []IccTagEntry,
    tag_data: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) IccProfile {
        return IccProfile{
            .header = IccProfileHeader.init(),
            .tag_table = &[_]IccTagEntry{},
            .tag_data = &[_]u8{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IccProfile) void {
        if (self.tag_table.len > 0) {
            self.allocator.free(self.tag_table);
        }
        if (self.tag_data.len > 0) {
            self.allocator.free(self.tag_data);
        }
    }
};

/// Convert RGB values between color profiles
pub fn convertBetweenProfiles(r: f32, g: f32, b: f32, src_profile: ColorProfile, dst_profile: ColorProfile) [3]f32 {
    if (src_profile == dst_profile) {
        return [3]f32{ r, g, b };
    }

    // Convert source RGB to XYZ
    const linear_src = linearizeRgb(r, g, b, src_profile);
    const xyz = rgbToXyz(linear_src[0], linear_src[1], linear_src[2], src_profile);

    // Convert XYZ to destination RGB
    const linear_dst = xyzToRgb(xyz[0], xyz[1], xyz[2], dst_profile);
    return compandRgb(linear_dst[0], linear_dst[1], linear_dst[2], dst_profile);
}

/// Linearize RGB values (remove gamma encoding)
pub fn linearizeRgb(r: f32, g: f32, b: f32, profile: ColorProfile) [3]f32 {
    return switch (profile) {
        .srgb => [3]f32{
            srgbToLinear(r),
            srgbToLinear(g),
            srgbToLinear(b),
        },
        .linear_rgb => [3]f32{ r, g, b },
        else => {
            const gamma = profile.getGamma();
            return [3]f32{
                math.pow(f32, @max(0.0, r), gamma),
                math.pow(f32, @max(0.0, g), gamma),
                math.pow(f32, @max(0.0, b), gamma),
            };
        },
    };
}

/// Apply gamma encoding to linear RGB values
pub fn compandRgb(r: f32, g: f32, b: f32, profile: ColorProfile) [3]f32 {
    return switch (profile) {
        .srgb => [3]f32{
            linearToSrgb(r),
            linearToSrgb(g),
            linearToSrgb(b),
        },
        .linear_rgb => [3]f32{ r, g, b },
        else => {
            const gamma = profile.getGamma();
            return [3]f32{
                math.pow(f32, @max(0.0, r), 1.0 / gamma),
                math.pow(f32, @max(0.0, g), 1.0 / gamma),
                math.pow(f32, @max(0.0, b), 1.0 / gamma),
            };
        },
    };
}

/// Convert linear RGB to XYZ
pub fn rgbToXyz(r: f32, g: f32, b: f32, profile: ColorProfile) [3]f32 {
    const matrix = profile.getRgbToXyzMatrix();

    const x = matrix[0] * r + matrix[1] * g + matrix[2] * b;
    const y = matrix[3] * r + matrix[4] * g + matrix[5] * b;
    const z = matrix[6] * r + matrix[7] * g + matrix[8] * b;

    return [3]f32{ x, y, z };
}

/// Convert XYZ to linear RGB
pub fn xyzToRgb(x: f32, y: f32, z: f32, profile: ColorProfile) [3]f32 {
    const matrix = profile.getXyzToRgbMatrix();

    const r = matrix[0] * x + matrix[1] * y + matrix[2] * z;
    const g = matrix[3] * x + matrix[4] * y + matrix[5] * z;
    const b = matrix[6] * x + matrix[7] * y + matrix[8] * z;

    return [3]f32{
        @min(1.0, @max(0.0, r)),
        @min(1.0, @max(0.0, g)),
        @min(1.0, @max(0.0, b)),
    };
}

/// sRGB to linear conversion (proper sRGB transfer function)
pub fn srgbToLinear(value: f32) f32 {
    const clamped = @min(1.0, @max(0.0, value));
    if (clamped <= 0.04045) {
        return clamped / 12.92;
    } else {
        return math.pow(f32, (clamped + 0.055) / 1.055, 2.4);
    }
}

/// Linear to sRGB conversion (proper sRGB transfer function)
pub fn linearToSrgb(value: f32) f32 {
    const clamped = @min(1.0, @max(0.0, value));
    if (clamped <= 0.0031308) {
        return clamped * 12.92;
    } else {
        return 1.055 * math.pow(f32, clamped, 1.0 / 2.4) - 0.055;
    }
}

/// Parse ICC profile from binary data
pub fn parseIccProfile(allocator: std.mem.Allocator, data: []const u8) !IccProfile {
    if (data.len < 128) return error.InvalidIccProfile;

    var profile = IccProfile.init(allocator);

    // Parse header (first 128 bytes)
    profile.header.profile_size = std.mem.readInt(u32, data[0..4], .big);
    @memcpy(&profile.header.preferred_cmm_type, data[4..8]);
    profile.header.profile_version = std.mem.readInt(u32, data[8..12], .big);
    @memcpy(&profile.header.device_class, data[12..16]);
    @memcpy(&profile.header.color_space, data[16..20]);
    @memcpy(&profile.header.pcs, data[20..24]);
    @memcpy(&profile.header.creation_date, data[24..36]);
    @memcpy(&profile.header.platform_signature, data[40..44]);
    profile.header.profile_flags = std.mem.readInt(u32, data[44..48], .big);
    @memcpy(&profile.header.device_manufacturer, data[48..52]);
    @memcpy(&profile.header.device_model, data[52..56]);
    profile.header.device_attributes = std.mem.readInt(u64, data[56..64], .big);
    profile.header.rendering_intent = std.mem.readInt(u32, data[64..68], .big);
    @memcpy(&profile.header.illuminant, data[68..80]);
    @memcpy(&profile.header.profile_creator, data[80..84]);

    // Parse tag table
    if (data.len < 132) return error.InvalidIccProfile;
    const tag_count = std.mem.readInt(u32, data[128..132], .big);

    if (data.len < 132 + tag_count * 12) return error.InvalidIccProfile;

    profile.tag_table = try allocator.alloc(IccTagEntry, tag_count);
    for (0..tag_count) |i| {
        const offset = 132 + i * 12;
        @memcpy(&profile.tag_table[i].signature, data[offset..offset + 4]);
        profile.tag_table[i].offset = std.mem.readInt(u32, data[offset + 4..offset + 8], .big);
        profile.tag_table[i].size = std.mem.readInt(u32, data[offset + 8..offset + 12], .big);
    }

    // Copy remaining tag data
    const tag_data_start = 132 + tag_count * 12;
    if (data.len > tag_data_start) {
        profile.tag_data = try allocator.alloc(u8, data.len - tag_data_start);
        @memcpy(profile.tag_data, data[tag_data_start..]);
    }

    return profile;
}

/// Create embedded sRGB profile
pub fn createSrgbProfile(allocator: std.mem.Allocator) !IccProfile {
    var profile = IccProfile.init(allocator);

    // Set basic header information
    profile.header.profile_size = 3144; // Standard sRGB profile size
    @memcpy(&profile.header.preferred_cmm_type, "ADBE");
    profile.header.profile_version = 0x02200000;
    @memcpy(&profile.header.device_class, "mntr");
    @memcpy(&profile.header.color_space, "RGB ");
    @memcpy(&profile.header.pcs, "XYZ ");
    @memcpy(&profile.header.platform_signature, "APPL");
    @memcpy(&profile.header.profile_creator, "zpix");

    // Add standard sRGB tags (simplified)
    const tag_entries = [_]IccTagEntry{
        IccTagEntry{ .signature = "desc".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "rXYZ".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "gXYZ".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "bXYZ".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "rTRC".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "gTRC".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "bTRC".*, .offset = 0, .size = 0 },
        IccTagEntry{ .signature = "wtpt".*, .offset = 0, .size = 0 },
    };

    profile.tag_table = try allocator.alloc(IccTagEntry, tag_entries.len);
    @memcpy(profile.tag_table, &tag_entries);

    return profile;
}

/// Convert image between color profiles
pub fn convertImageProfile(allocator: std.mem.Allocator, src_data: []const u8, width: u32, height: u32,
                          src_profile: ColorProfile, dst_profile: ColorProfile) ![]u8 {
    const dst_data = try allocator.alloc(u8, src_data.len);

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = (y * width + x) * 3;

            const r = @as(f32, @floatFromInt(src_data[idx])) / 255.0;
            const g = @as(f32, @floatFromInt(src_data[idx + 1])) / 255.0;
            const b = @as(f32, @floatFromInt(src_data[idx + 2])) / 255.0;

            const converted = convertBetweenProfiles(r, g, b, src_profile, dst_profile);

            dst_data[idx] = @intFromFloat(@min(255.0, @max(0.0, converted[0] * 255.0)));
            dst_data[idx + 1] = @intFromFloat(@min(255.0, @max(0.0, converted[1] * 255.0)));
            dst_data[idx + 2] = @intFromFloat(@min(255.0, @max(0.0, converted[2] * 255.0)));
        }
    }

    return dst_data;
}

/// Calculate color gamut coverage between two profiles
pub fn calculateGamutCoverage(src_profile: ColorProfile, dst_profile: ColorProfile) f32 {
    const src_primaries = src_profile.getPrimaries();
    const dst_primaries = dst_profile.getPrimaries();

    // Calculate triangle areas (simplified gamut comparison)
    const src_area = triangleArea(
        src_primaries[0], src_primaries[1], // Red
        src_primaries[2], src_primaries[3], // Green
        src_primaries[4], src_primaries[5]  // Blue
    );

    const dst_area = triangleArea(
        dst_primaries[0], dst_primaries[1], // Red
        dst_primaries[2], dst_primaries[3], // Green
        dst_primaries[4], dst_primaries[5]  // Blue
    );

    return @min(1.0, dst_area / src_area);
}

fn triangleArea(x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) f32 {
    return @abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
}