//! Basic SVG parsing and rendering for zpix
//! Supports basic shapes, paths, and simple styling

const std = @import("std");

/// SVG element types
pub const SvgElement = union(enum) {
    rect: Rect,
    circle: Circle,
    line: Line,
    path: Path,
    text: Text,
};

/// Rectangle element
pub const Rect = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    fill: Color,
    stroke: Color,
    stroke_width: f32,
};

/// Circle element
pub const Circle = struct {
    cx: f32,
    cy: f32,
    r: f32,
    fill: Color,
    stroke: Color,
    stroke_width: f32,
};

/// Line element
pub const Line = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    stroke: Color,
    stroke_width: f32,
};

/// Path element (simplified)
pub const Path = struct {
    commands: []PathCommand,
    fill: Color,
    stroke: Color,
    stroke_width: f32,
};

/// Path commands
pub const PathCommand = union(enum) {
    move_to: Point,
    line_to: Point,
    curve_to: CurveControl,
    close_path: void,
};

pub const Point = struct {
    x: f32,
    y: f32,
};

pub const CurveControl = struct {
    cp1: Point,
    cp2: Point,
    end: Point,
};

/// Text element
pub const Text = struct {
    x: f32,
    y: f32,
    content: []const u8,
    font_size: f32,
    fill: Color,
};

/// Color representation
pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    pub const black = Color{ .r = 0, .g = 0, .b = 0, .a = 255 };
    pub const white = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
    pub const red = Color{ .r = 255, .g = 0, .b = 0, .a = 255 };
    pub const green = Color{ .r = 0, .g = 255, .b = 0, .a = 255 };
    pub const blue = Color{ .r = 0, .g = 0, .b = 255, .a = 255 };
    pub const transparent = Color{ .r = 0, .g = 0, .b = 0, .a = 0 };
};

/// SVG document
pub const SvgDocument = struct {
    width: f32,
    height: f32,
    viewbox: struct {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    elements: std.ArrayList(SvgElement),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SvgDocument {
        return SvgDocument{
            .width = 100,
            .height = 100,
            .viewbox = .{ .x = 0, .y = 0, .width = 100, .height = 100 },
            .elements = std.ArrayList(SvgElement).initCapacity(allocator, 10) catch unreachable,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SvgDocument) void {
        self.elements.deinit(self.allocator);
    }
};

/// Parse SVG from XML content
pub fn parseSvg(allocator: std.mem.Allocator, svg_content: []const u8) !SvgDocument {
    var doc = SvgDocument.init(allocator);

    // Simple XML parsing (very basic implementation)
    var lines = std.mem.splitSequence(u8, svg_content, "\n");
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        if (std.mem.startsWith(u8, trimmed, "<svg")) {
            try parseSvgTag(trimmed, &doc);
        } else if (std.mem.startsWith(u8, trimmed, "<rect")) {
            const rect = try parseRect(trimmed);
            try doc.elements.append(allocator, .{ .rect = rect });
        } else if (std.mem.startsWith(u8, trimmed, "<circle")) {
            const circle = try parseCircle(trimmed);
            try doc.elements.append(allocator, .{ .circle = circle });
        } else if (std.mem.startsWith(u8, trimmed, "<line")) {
            const line_elem = try parseLine(trimmed);
            try doc.elements.append(allocator, .{ .line = line_elem });
        } else if (std.mem.startsWith(u8, trimmed, "<path")) {
            const path = try parsePath(allocator, trimmed);
            try doc.elements.append(allocator, .{ .path = path });
        }
    }

    return doc;
}

/// Render SVG to RGB bitmap
pub fn renderSvg(allocator: std.mem.Allocator, doc: *const SvgDocument, width: u32, height: u32) ![]u8 {
    const rgb_data = try allocator.alloc(u8, width * height * 3);
    @memset(rgb_data, 255); // White background

    const scale_x = @as(f32, @floatFromInt(width)) / doc.viewbox.width;
    const scale_y = @as(f32, @floatFromInt(height)) / doc.viewbox.height;

    for (doc.elements.items) |element| {
        switch (element) {
            .rect => |rect| try renderRect(rgb_data, width, height, rect, scale_x, scale_y),
            .circle => |circle| try renderCircle(rgb_data, width, height, circle, scale_x, scale_y),
            .line => |line| try renderLine(rgb_data, width, height, line, scale_x, scale_y),
            .path => |path| try renderPath(rgb_data, width, height, path, scale_x, scale_y),
            .text => |text| try renderText(rgb_data, width, height, text, scale_x, scale_y),
        }
    }

    return rgb_data;
}

// Parsing functions
fn parseSvgTag(tag: []const u8, doc: *SvgDocument) !void {
    // Extract width and height from SVG tag
    if (extractAttribute(tag, "width")) |width_str| {
        doc.width = std.fmt.parseFloat(f32, width_str) catch 100.0;
    }
    if (extractAttribute(tag, "height")) |height_str| {
        doc.height = std.fmt.parseFloat(f32, height_str) catch 100.0;
    }
    if (extractAttribute(tag, "viewBox")) |viewbox_str| {
        var values = std.mem.splitSequence(u8, viewbox_str, " ");
        doc.viewbox.x = std.fmt.parseFloat(f32, values.next() orelse "0") catch 0.0;
        doc.viewbox.y = std.fmt.parseFloat(f32, values.next() orelse "0") catch 0.0;
        doc.viewbox.width = std.fmt.parseFloat(f32, values.next() orelse "100") catch 100.0;
        doc.viewbox.height = std.fmt.parseFloat(f32, values.next() orelse "100") catch 100.0;
    }
}

fn parseRect(tag: []const u8) !Rect {
    return Rect{
        .x = parseFloatAttribute(tag, "x") orelse 0.0,
        .y = parseFloatAttribute(tag, "y") orelse 0.0,
        .width = parseFloatAttribute(tag, "width") orelse 10.0,
        .height = parseFloatAttribute(tag, "height") orelse 10.0,
        .fill = parseColorAttribute(tag, "fill") orelse Color.black,
        .stroke = parseColorAttribute(tag, "stroke") orelse Color.transparent,
        .stroke_width = parseFloatAttribute(tag, "stroke-width") orelse 1.0,
    };
}

fn parseCircle(tag: []const u8) !Circle {
    return Circle{
        .cx = parseFloatAttribute(tag, "cx") orelse 0.0,
        .cy = parseFloatAttribute(tag, "cy") orelse 0.0,
        .r = parseFloatAttribute(tag, "r") orelse 5.0,
        .fill = parseColorAttribute(tag, "fill") orelse Color.black,
        .stroke = parseColorAttribute(tag, "stroke") orelse Color.transparent,
        .stroke_width = parseFloatAttribute(tag, "stroke-width") orelse 1.0,
    };
}

fn parseLine(tag: []const u8) !Line {
    return Line{
        .x1 = parseFloatAttribute(tag, "x1") orelse 0.0,
        .y1 = parseFloatAttribute(tag, "y1") orelse 0.0,
        .x2 = parseFloatAttribute(tag, "x2") orelse 10.0,
        .y2 = parseFloatAttribute(tag, "y2") orelse 10.0,
        .stroke = parseColorAttribute(tag, "stroke") orelse Color.black,
        .stroke_width = parseFloatAttribute(tag, "stroke-width") orelse 1.0,
    };
}

fn parsePath(allocator: std.mem.Allocator, tag: []const u8) !Path {
    _ = allocator;
    // Simplified path parsing - in practice this would be much more complex
    return Path{
        .commands = &[_]PathCommand{},
        .fill = parseColorAttribute(tag, "fill") orelse Color.black,
        .stroke = parseColorAttribute(tag, "stroke") orelse Color.transparent,
        .stroke_width = parseFloatAttribute(tag, "stroke-width") orelse 1.0,
    };
}

// Rendering functions
fn renderRect(rgb_data: []u8, width: u32, height: u32, rect: Rect, scale_x: f32, scale_y: f32) !void {
    const x1 = @as(u32, @intFromFloat(rect.x * scale_x));
    const y1 = @as(u32, @intFromFloat(rect.y * scale_y));
    const x2 = @min(width - 1, @as(u32, @intFromFloat((rect.x + rect.width) * scale_x)));
    const y2 = @min(height - 1, @as(u32, @intFromFloat((rect.y + rect.height) * scale_y)));

    // Fill rectangle
    if (rect.fill.a > 0) {
        for (y1..y2 + 1) |y| {
            for (x1..x2 + 1) |x| {
                const idx = (y * width + x) * 3;
                rgb_data[idx] = rect.fill.r;
                rgb_data[idx + 1] = rect.fill.g;
                rgb_data[idx + 2] = rect.fill.b;
            }
        }
    }

    // Draw stroke
    if (rect.stroke.a > 0 and rect.stroke_width > 0) {
        const stroke_w = @as(u32, @intFromFloat(rect.stroke_width * scale_x));

        // Top and bottom borders
        for (0..stroke_w) |sw| {
            if (y1 + sw < height) {
                for (x1..x2 + 1) |x| {
                    const idx = ((y1 + sw) * width + x) * 3;
                    rgb_data[idx] = rect.stroke.r;
                    rgb_data[idx + 1] = rect.stroke.g;
                    rgb_data[idx + 2] = rect.stroke.b;
                }
            }
            if (y2 >= sw) {
                for (x1..x2 + 1) |x| {
                    const idx = ((y2 - sw) * width + x) * 3;
                    rgb_data[idx] = rect.stroke.r;
                    rgb_data[idx + 1] = rect.stroke.g;
                    rgb_data[idx + 2] = rect.stroke.b;
                }
            }
        }

        // Left and right borders
        for (0..stroke_w) |sw| {
            if (x1 + sw < width) {
                for (y1..y2 + 1) |y| {
                    const idx = (y * width + (x1 + sw)) * 3;
                    rgb_data[idx] = rect.stroke.r;
                    rgb_data[idx + 1] = rect.stroke.g;
                    rgb_data[idx + 2] = rect.stroke.b;
                }
            }
            if (x2 >= sw) {
                for (y1..y2 + 1) |y| {
                    const idx = (y * width + (x2 - sw)) * 3;
                    rgb_data[idx] = rect.stroke.r;
                    rgb_data[idx + 1] = rect.stroke.g;
                    rgb_data[idx + 2] = rect.stroke.b;
                }
            }
        }
    }
}

fn renderCircle(rgb_data: []u8, width: u32, height: u32, circle: Circle, scale_x: f32, scale_y: f32) !void {
    const cx = circle.cx * scale_x;
    const cy = circle.cy * scale_y;
    const r = circle.r * @min(scale_x, scale_y);

    const x1 = @as(u32, @intFromFloat(@max(0, cx - r)));
    const y1 = @as(u32, @intFromFloat(@max(0, cy - r)));
    const x2 = @as(u32, @intFromFloat(@min(@as(f32, @floatFromInt(width - 1)), cx + r)));
    const y2 = @as(u32, @intFromFloat(@min(@as(f32, @floatFromInt(height - 1)), cy + r)));

    for (y1..y2 + 1) |y| {
        for (x1..x2 + 1) |x| {
            const dx = @as(f32, @floatFromInt(x)) - cx;
            const dy = @as(f32, @floatFromInt(y)) - cy;
            const dist = @sqrt(dx * dx + dy * dy);

            if (dist <= r and circle.fill.a > 0) {
                const idx = (y * width + x) * 3;
                rgb_data[idx] = circle.fill.r;
                rgb_data[idx + 1] = circle.fill.g;
                rgb_data[idx + 2] = circle.fill.b;
            } else if (circle.stroke.a > 0 and
                      dist <= r and dist >= r - circle.stroke_width * @min(scale_x, scale_y)) {
                const idx = (y * width + x) * 3;
                rgb_data[idx] = circle.stroke.r;
                rgb_data[idx + 1] = circle.stroke.g;
                rgb_data[idx + 2] = circle.stroke.b;
            }
        }
    }
}

fn renderLine(rgb_data: []u8, width: u32, height: u32, line: Line, scale_x: f32, scale_y: f32) !void {
    if (line.stroke.a == 0) return;

    const x1 = @as(i32, @intFromFloat(line.x1 * scale_x));
    const y1 = @as(i32, @intFromFloat(line.y1 * scale_y));
    const x2 = @as(i32, @intFromFloat(line.x2 * scale_x));
    const y2 = @as(i32, @intFromFloat(line.y2 * scale_y));

    // Bresenham's line algorithm
    var x = x1;
    var y = y1;
    const dx = if (x2 > x1) x2 - x1 else x1 - x2;
    const dy = if (y2 > y1) y2 - y1 else y1 - y2;
    const sx: i32 = if (x1 < x2) 1 else -1;
    const sy: i32 = if (y1 < y2) 1 else -1;
    var err = dx - dy;

    while (true) {
        if (x >= 0 and x < width and y >= 0 and y < height) {
            const idx = (@as(usize, @intCast(y)) * width + @as(usize, @intCast(x))) * 3;
            rgb_data[idx] = line.stroke.r;
            rgb_data[idx + 1] = line.stroke.g;
            rgb_data[idx + 2] = line.stroke.b;
        }

        if (x == x2 and y == y2) break;

        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

fn renderPath(rgb_data: []u8, width: u32, height: u32, path: Path, scale_x: f32, scale_y: f32) !void {
    _ = rgb_data;
    _ = width;
    _ = height;
    _ = path;
    _ = scale_x;
    _ = scale_y;
    // Path rendering would be implemented here - complex curve rendering
}

fn renderText(rgb_data: []u8, width: u32, height: u32, text: Text, scale_x: f32, scale_y: f32) !void {
    _ = rgb_data;
    _ = width;
    _ = height;
    _ = text;
    _ = scale_x;
    _ = scale_y;
    // Basic text rendering would be implemented here - requires font rasterization
}

// Utility functions
fn extractAttribute(tag: []const u8, attr_name: []const u8) ?[]const u8 {
    const attr_pattern = std.fmt.allocPrint(std.heap.page_allocator, "{s}=\"", .{attr_name}) catch return null;
    defer std.heap.page_allocator.free(attr_pattern);

    if (std.mem.indexOf(u8, tag, attr_pattern)) |start| {
        const value_start = start + attr_pattern.len;
        if (std.mem.indexOf(u8, tag[value_start..], "\"")) |end| {
            return tag[value_start..value_start + end];
        }
    }
    return null;
}

fn parseFloatAttribute(tag: []const u8, attr_name: []const u8) ?f32 {
    if (extractAttribute(tag, attr_name)) |value| {
        return std.fmt.parseFloat(f32, value) catch null;
    }
    return null;
}

fn parseColorAttribute(tag: []const u8, attr_name: []const u8) ?Color {
    if (extractAttribute(tag, attr_name)) |value| {
        if (std.mem.eql(u8, value, "none")) {
            return Color.transparent;
        } else if (std.mem.eql(u8, value, "black")) {
            return Color.black;
        } else if (std.mem.eql(u8, value, "white")) {
            return Color.white;
        } else if (std.mem.eql(u8, value, "red")) {
            return Color.red;
        } else if (std.mem.eql(u8, value, "green")) {
            return Color.green;
        } else if (std.mem.eql(u8, value, "blue")) {
            return Color.blue;
        } else if (std.mem.startsWith(u8, value, "#")) {
            // Parse hex color
            if (value.len >= 7) {
                const r = std.fmt.parseInt(u8, value[1..3], 16) catch 0;
                const g = std.fmt.parseInt(u8, value[3..5], 16) catch 0;
                const b = std.fmt.parseInt(u8, value[5..7], 16) catch 0;
                return Color{ .r = r, .g = g, .b = b, .a = 255 };
            }
        }
    }
    return null;
}