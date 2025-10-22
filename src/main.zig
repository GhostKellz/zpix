const std = @import("std");
const zpix = @import("zpix");
const posix = std.posix;

fn copyFdToFile(fd: posix.fd_t, file: *std.fs.File) !void {
    var buffer: [16 * 1024]u8 = undefined;
    while (true) {
        const amount = try posix.read(fd, buffer[0..]);
        if (amount == 0) break;

        var written: usize = 0;
        while (written < amount) {
            const slice = buffer[written..amount];
            const count = try file.write(slice);
            written += count;
        }
    }
}

fn copyFileToFd(file: *std.fs.File, fd: posix.fd_t) !void {
    var buffer: [16 * 1024]u8 = undefined;
    while (true) {
        const amount = try file.read(buffer[0..]);
        if (amount == 0) break;

        var written: usize = 0;
        while (written < amount) {
            const slice = buffer[written..amount];
            const count = try posix.write(fd, slice);
            written += count;
        }
    }
}

const Command = enum {
    convert,
    @"test",
    benchmark,
    pipeline,
    batch,
    help,
};

const BatchOption = struct {
    key: []const u8,
    value: []const u8,
};

fn findOption(options: []const BatchOption, name: []const u8) ?[]const u8 {
    for (options) |opt| {
        if (std.ascii.eqlIgnoreCase(opt.key, name)) return opt.value;
    }
    return null;
}

fn parseBool(value: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(value, "true") or std.ascii.eqlIgnoreCase(value, "yes") or std.ascii.eqlIgnoreCase(value, "on") or std.mem.eql(u8, value, "1")) {
        return true;
    }
    if (std.ascii.eqlIgnoreCase(value, "false") or std.ascii.eqlIgnoreCase(value, "no") or std.ascii.eqlIgnoreCase(value, "off") or std.mem.eql(u8, value, "0")) {
        return false;
    }
    return null;
}

fn parseUInt(comptime T: type, value: []const u8) ?T {
    return std.fmt.parseInt(T, value, 10) catch null;
}

fn parseInt(comptime T: type, value: []const u8) ?T {
    return std.fmt.parseInt(T, value, 10) catch null;
}

fn parseFloat(value: []const u8) ?f32 {
    return std.fmt.parseFloat(f32, value) catch null;
}

fn argSlice(arg: [:0]u8) []const u8 {
    return std.mem.sliceTo(arg, 0);
}

fn makeAbsolutePath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        return allocator.dupe(u8, path);
    }

    const cwd_path = try std.fs.cwd().realpathAlloc(allocator, ".");
    defer allocator.free(cwd_path);

    return std.fs.path.join(allocator, &[_][]const u8{ cwd_path, path });
}

fn ensureParentDirs(path: []const u8) !void {
    if (std.fs.path.dirname(path)) |dir| {
        var cwd = std.fs.cwd();
        try cwd.makePath(dir);
    }
}

fn createTempPath(allocator: std.mem.Allocator, prefix: []const u8, suffix: []const u8) ![]u8 {
    var cwd = std.fs.cwd();
    try cwd.makePath("zig-out/tmp");

    const timestamp = std.time.microTimestamp();
    const magnitude = if (timestamp < 0) -timestamp else timestamp;
    const unique: u64 = @intCast(magnitude);
    const name = try std.fmt.allocPrint(allocator, "{s}-{d}{s}", .{ prefix, unique, suffix });
    defer allocator.free(name);

    const rel_path = try std.fs.path.join(allocator, &[_][]const u8{ "zig-out", "tmp", name });
    defer allocator.free(rel_path);

    return makeAbsolutePath(allocator, rel_path);
}

fn loadImageFromSource(allocator: std.mem.Allocator, source: []const u8) !zpix.Image {
    if (std.mem.eql(u8, source, "-")) {
        const temp_path = try createTempPath(allocator, "stdin", ".bin");
        defer {
            std.fs.deleteFileAbsolute(temp_path) catch {};
            allocator.free(temp_path);
        }

        var file = try std.fs.createFileAbsolute(temp_path, .{});
        defer file.close();
        try copyFdToFile(posix.STDIN_FILENO, &file);
        try file.seekTo(0);

        return zpix.Image.load(allocator, temp_path);
    }

    const abs_path = try makeAbsolutePath(allocator, source);
    defer allocator.free(abs_path);
    return zpix.Image.load(allocator, abs_path);
}

fn parseImageFormat(name: []const u8) ?zpix.ImageFormat {
    if (std.ascii.eqlIgnoreCase(name, "png")) return .png;
    if (std.ascii.eqlIgnoreCase(name, "jpeg") or std.ascii.eqlIgnoreCase(name, "jpg")) return .jpeg;
    if (std.ascii.eqlIgnoreCase(name, "webp")) return .webp;
    if (std.ascii.eqlIgnoreCase(name, "avif")) return .avif;
    if (std.ascii.eqlIgnoreCase(name, "tiff") or std.ascii.eqlIgnoreCase(name, "tif")) return .tiff;
    if (std.ascii.eqlIgnoreCase(name, "bmp")) return .bmp;
    if (std.ascii.eqlIgnoreCase(name, "gif")) return .gif;
    if (std.ascii.eqlIgnoreCase(name, "svg")) return .svg;
    return null;
}

fn saveImageToDestination(allocator: std.mem.Allocator, image: *zpix.Image, dest: []const u8, format_hint: ?zpix.ImageFormat) !void {
    if (std.mem.eql(u8, dest, "-")) {
        const fmt = format_hint orelse return error.MissingFormatForStdout;
        const temp_path = try createTempPath(allocator, "stdout", ".bin");
        defer {
            std.fs.deleteFileAbsolute(temp_path) catch {};
            allocator.free(temp_path);
        }

        try image.*.save(temp_path, fmt);

        var tmp_file = try std.fs.openFileAbsolute(temp_path, .{});
        defer tmp_file.close();

        try copyFileToFd(&tmp_file, posix.STDOUT_FILENO);
        return;
    }

    const abs_path = try makeAbsolutePath(allocator, dest);
    defer allocator.free(abs_path);
    try ensureParentDirs(abs_path);
    const format = getFormatFromPath(abs_path) orelse format_hint orelse return error.UnsupportedOutputFormat;
    try image.*.save(abs_path, format);
}

fn runPipeline(allocator: std.mem.Allocator, input_raw: []const u8, raw_steps: []const []const u8) !void {
    const input = std.mem.trim(u8, input_raw, " \t\r\n");
    if (input.len == 0) {
        std.debug.print("Pipeline input cannot be empty.\n", .{});
        return error.InvalidPipelineStep;
    }

    var image = loadImageFromSource(allocator, input) catch |err| {
        std.debug.print("Failed to load pipeline input '{s}': {}\n", .{ input, err });
        return err;
    };
    defer image.deinit();

    var output_format: ?zpix.ImageFormat = null;
    var saved_any = false;

    for (raw_steps) |raw_step| {
        const step = std.mem.trim(u8, raw_step, " \t\r\n");
        if (step.len == 0) continue;

        if (std.mem.startsWith(u8, step, "resize:")) {
            if (step.len <= 7) {
                std.debug.print("resize step requires WIDTHxHEIGHT\n", .{});
                return error.InvalidPipelineStep;
            }
            const dims = step[7..];
            const sep = std.mem.indexOfScalar(u8, dims, 'x') orelse std.mem.indexOfScalar(u8, dims, 'X') orelse {
                std.debug.print("resize step must use WIDTHxHEIGHT\n", .{});
                return error.InvalidPipelineStep;
            };
            const width_str = dims[0..sep];
            const height_str = dims[sep + 1 ..];
            const width = std.fmt.parseInt(u32, width_str, 10) catch {
                std.debug.print("Invalid width in resize step: {s}\n", .{width_str});
                return error.InvalidPipelineStep;
            };
            const height = std.fmt.parseInt(u32, height_str, 10) catch {
                std.debug.print("Invalid height in resize step: {s}\n", .{height_str});
                return error.InvalidPipelineStep;
            };
            try image.resize(width, height);
        } else if (std.mem.startsWith(u8, step, "blur:")) {
            if (step.len <= 5) {
                std.debug.print("blur step requires radius\n", .{});
                return error.InvalidPipelineStep;
            }
            const radius = std.fmt.parseInt(u32, step[5..], 10) catch {
                std.debug.print("Invalid blur radius: {s}\n", .{step[5..]});
                return error.InvalidPipelineStep;
            };
            try image.blur(radius);
        } else if (std.mem.startsWith(u8, step, "brightness:")) {
            if (step.len <= 11) {
                std.debug.print("brightness step requires value\n", .{});
                return error.InvalidPipelineStep;
            }
            const value_i32 = std.fmt.parseInt(i32, step[11..], 10) catch {
                std.debug.print("Invalid brightness adjustment: {s}\n", .{step[11..]});
                return error.InvalidPipelineStep;
            };
            if (value_i32 < std.math.minInt(i16) or value_i32 > std.math.maxInt(i16)) {
                std.debug.print("brightness adjustment out of range: {d}\n", .{value_i32});
                return error.InvalidPipelineStep;
            }
            try image.adjustBrightness(@intCast(value_i32));
        } else if (std.mem.startsWith(u8, step, "contrast:")) {
            if (step.len <= 9) {
                std.debug.print("contrast step requires factor\n", .{});
                return error.InvalidPipelineStep;
            }
            const factor = std.fmt.parseFloat(f32, step[9..]) catch {
                std.debug.print("Invalid contrast factor: {s}\n", .{step[9..]});
                return error.InvalidPipelineStep;
            };
            try image.adjustContrast(factor);
        } else if (std.mem.startsWith(u8, step, "crop:")) {
            if (step.len <= 5) {
                std.debug.print("crop step requires x,y,width,height\n", .{});
                return error.InvalidPipelineStep;
            }
            const parts = step[5..];
            var values = [_]u32{ 0, 0, 0, 0 };
            var count: usize = 0;
            var it = std.mem.tokenizeScalar(u8, parts, ',');
            while (it.next()) |token| {
                if (count >= values.len) break;
                values[count] = std.fmt.parseInt(u32, token, 10) catch {
                    std.debug.print("Invalid crop value: {s}\n", .{token});
                    return error.InvalidPipelineStep;
                };
                count += 1;
            }
            if (count != 4) {
                std.debug.print("crop step must provide x,y,width,height\n", .{});
                return error.InvalidPipelineStep;
            }
            try image.crop(values[0], values[1], values[2], values[3]);
        } else if (std.ascii.eqlIgnoreCase(step, "grayscale") or std.ascii.eqlIgnoreCase(step, "convert:grayscale")) {
            try image.convertToGrayscale();
        } else if (std.mem.startsWith(u8, step, "rotate:")) {
            if (step.len <= 7) {
                std.debug.print("rotate step requires angle\n", .{});
                return error.InvalidPipelineStep;
            }
            const angle = std.fmt.parseFloat(f32, step[7..]) catch {
                std.debug.print("Invalid rotate angle: {s}\n", .{step[7..]});
                return error.InvalidPipelineStep;
            };
            if (std.math.approxEqAbs(f32, angle, 90.0, 0.0001)) {
                try image.rotate90();
            } else if (std.math.approxEqAbs(f32, angle, 180.0, 0.0001)) {
                try image.rotateArbitrary(180.0);
            } else if (std.math.approxEqAbs(f32, angle, 270.0, 0.0001)) {
                try image.rotateArbitrary(270.0);
            } else {
                try image.rotateArbitrary(angle);
            }
        } else if (std.ascii.eqlIgnoreCase(step, "flip:h") or std.ascii.eqlIgnoreCase(step, "flip:horizontal")) {
            try image.flipHorizontal();
        } else if (std.ascii.eqlIgnoreCase(step, "flip:v") or std.ascii.eqlIgnoreCase(step, "flip:vertical")) {
            try image.flipVertical();
        } else if (std.mem.startsWith(u8, step, "format") or std.mem.startsWith(u8, step, "output-format") or std.mem.startsWith(u8, step, "--format")) {
            const delim = std.mem.indexOfScalar(u8, step, ':') orelse std.mem.indexOfScalar(u8, step, '=') orelse {
                std.debug.print("format step requires value\n", .{});
                return error.InvalidPipelineStep;
            };
            const value = std.mem.trim(u8, step[delim + 1 ..], " \t");
            if (value.len == 0) {
                std.debug.print("format step missing value\n", .{});
                return error.InvalidPipelineStep;
            }
            output_format = parseImageFormat(value) orelse {
                std.debug.print("Unknown format: {s}\n", .{value});
                return error.InvalidPipelineStep;
            };
        } else if (std.mem.startsWith(u8, step, "save") or std.mem.startsWith(u8, step, "write")) {
            const delim = std.mem.indexOfScalar(u8, step, ':') orelse std.mem.indexOfScalar(u8, step, '=') orelse {
                std.debug.print("save step requires destination\n", .{});
                return error.InvalidPipelineStep;
            };
            const destination = std.mem.trim(u8, step[delim + 1 ..], " \t");
            if (destination.len == 0) {
                std.debug.print("save step missing destination\n", .{});
                return error.InvalidPipelineStep;
            }
            try saveImageToDestination(allocator, &image, destination, output_format);
            saved_any = true;
        } else {
            std.debug.print("Unknown pipeline step: {s}\n", .{step});
            return error.InvalidPipelineStep;
        }
    }

    if (!saved_any) {
        std.debug.print("Pipeline finished with no save step executed. Add save:<path> to persist output.\n", .{});
    }
}

fn handlePipeline(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    if (args.len < 2) {
        std.debug.print(
            "Usage: zpix pipeline <input> <step> [<step> ...]\n" ++
                "Example: zpix pipeline photo.png resize:800x600 blur:2 format:jpeg save:out.jpg\n",
            .{},
        );
        return;
    }

    const input = argSlice(args[0]);

    var steps = std.ArrayListUnmanaged([]const u8){};
    defer steps.deinit(allocator);

    for (args[1..]) |arg| {
        const trimmed = std.mem.trim(u8, argSlice(arg), " \t\r\n");
        if (trimmed.len == 0) continue;
        try steps.append(allocator, trimmed);
    }

    if (steps.items.len == 0) {
        std.debug.print("No pipeline steps provided.\n", .{});
        return;
    }

    try runPipeline(allocator, input, steps.items);
}

fn addInputEntry(
    allocator: std.mem.Allocator,
    inputs: *std.ArrayList([]const u8),
    temp_allocations: *std.ArrayList([]const u8),
    entry: []const u8,
    recursive: bool,
) !void {
    const trimmed = std.mem.trim(u8, entry, " \t\r\n");
    if (trimmed.len == 0) return;

    const has_wildcard = std.mem.indexOfScalar(u8, trimmed, '*') != null or
        std.mem.indexOfScalar(u8, trimmed, '?') != null;

    if (has_wildcard) {
        const dir_path = std.fs.path.dirname(trimmed) orelse ".";
        const pattern = std.fs.path.basename(trimmed);
        const patterns = [_][]const u8{pattern};

        const files = zpix.batch_functions.scanDirectory(allocator, dir_path, &patterns, recursive) catch |err| {
            std.debug.print("  failed to expand pattern '{s}': {}\n", .{ trimmed, err });
            return;
        };
        defer allocator.free(files);

        if (files.len == 0) {
            std.debug.print("  warning: pattern '{s}' matched no files\n", .{trimmed});
        }

        for (files) |file| {
            try inputs.append(file);
            try temp_allocations.append(file);
        }
        return;
    }

    if (std.fs.cwd().openDir(trimmed, .{})) |dir| {
        defer dir.close();
        const patterns = [_][]const u8{"*"};
        const files = zpix.batch_functions.scanDirectory(allocator, trimmed, &patterns, recursive) catch |err| {
            std.debug.print("  failed to read directory '{s}': {}\n", .{ trimmed, err });
            return;
        };
        defer allocator.free(files);

        if (files.len == 0) {
            std.debug.print("  warning: directory '{s}' contains no matching files\n", .{trimmed});
        }

        for (files) |file| {
            try inputs.append(file);
            try temp_allocations.append(file);
        }
        return;
    } else |_| {}

    // Treat as explicit file path; warn if inaccessible but still attempt.
    if (std.fs.cwd().access(trimmed, .{})) |_| {} else |_| {
        std.debug.print("  warning: file '{s}' not found at parse time (will attempt anyway)\n", .{trimmed});
    }

    try inputs.append(trimmed);
}

fn batchProgressCallback(completed: u32, total: u32, current_file: []const u8) void {
    std.debug.print("  [{}/{}] {s}\n", .{ completed, total, current_file });
}

const BatchParseError = error{InvalidBatchSpec};

fn parseBatchOperation(options: []const BatchOption, op_name: []const u8, line_no: usize) BatchParseError!zpix.BatchParams {
    if (std.ascii.eqlIgnoreCase(op_name, "resize")) {
        const width_value = findOption(options, "width") orelse {
            std.debug.print("Line {}: resize requires width=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };
        const height_value = findOption(options, "height") orelse {
            std.debug.print("Line {}: resize requires height=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const width = parseUInt(u32, width_value) orelse {
            std.debug.print("Line {}: invalid resize width '{s}'\n", .{ line_no, width_value });
            return BatchParseError.InvalidBatchSpec;
        };
        const height = parseUInt(u32, height_value) orelse {
            std.debug.print("Line {}: invalid resize height '{s}'\n", .{ line_no, height_value });
            return BatchParseError.InvalidBatchSpec;
        };

        const maintain = blk: {
            if (findOption(options, "maintain_aspect")) |raw| {
                const parsed = parseBool(raw) orelse {
                    std.debug.print("Line {}: maintain_aspect expects true/false\n", .{line_no});
                    break :blk true;
                };
                break :blk parsed;
            }
            break :blk true;
        };

        return zpix.BatchParams{ .resize = .{ .width = width, .height = height, .maintain_aspect = maintain } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "convert_format") or std.ascii.eqlIgnoreCase(op_name, "convert")) {
        const format_value = findOption(options, "format") orelse {
            std.debug.print("Line {}: convert_format requires format=<png|jpeg|bmp|webp>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const format = parseImageFormat(format_value) orelse {
            std.debug.print("Line {}: unknown image format '{s}'\n", .{ line_no, format_value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .convert_format = .{ .target_format = format } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "adjust_brightness") or std.ascii.eqlIgnoreCase(op_name, "brightness")) {
        const value = findOption(options, "value") orelse findOption(options, "amount") orelse {
            std.debug.print("Line {}: adjust_brightness requires value=<integer>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const parsed = parseInt(i32, value) orelse {
            std.debug.print("Line {}: invalid brightness value '{s}'\n", .{ line_no, value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .adjust_brightness = .{ .brightness = parsed } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "adjust_contrast") or std.ascii.eqlIgnoreCase(op_name, "contrast")) {
        const value = findOption(options, "factor") orelse {
            std.debug.print("Line {}: adjust_contrast requires factor=<float>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const parsed = parseFloat(value) orelse {
            std.debug.print("Line {}: invalid contrast factor '{s}'\n", .{ line_no, value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .adjust_contrast = .{ .contrast = parsed } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "blur")) {
        const radius_value = findOption(options, "radius") orelse {
            std.debug.print("Line {}: blur requires radius=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const radius = parseUInt(u32, radius_value) orelse {
            std.debug.print("Line {}: invalid blur radius '{s}'\n", .{ line_no, radius_value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .blur = .{ .radius = radius } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "rotate")) {
        const angle_value = findOption(options, "angle") orelse {
            std.debug.print("Line {}: rotate requires angle=<90|180|270>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const angle_parsed = parseFloat(angle_value) orelse blk: {
            const int_angle = parseInt(i32, angle_value) orelse break :blk null;
            break :blk @as(f32, @floatFromInt(int_angle));
        } orelse {
            std.debug.print("Line {}: invalid rotate angle '{s}'\n", .{ line_no, angle_value });
            return BatchParseError.InvalidBatchSpec;
        };

        const rotation = if (std.math.approxEqAbs(f32, angle_parsed, 90.0, 0.01))
            zpix.RotationAngle.rotate_90
        else if (std.math.approxEqAbs(f32, angle_parsed, 180.0, 0.01))
            zpix.RotationAngle.rotate_180
        else if (std.math.approxEqAbs(f32, angle_parsed, 270.0, 0.01))
            zpix.RotationAngle.rotate_270
        else {
            std.debug.print("Line {}: rotate angle must be 90, 180, or 270\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .rotate = .{ .angle = rotation } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "crop")) {
        const x_value = findOption(options, "x") orelse {
            std.debug.print("Line {}: crop requires x=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };
        const y_value = findOption(options, "y") orelse {
            std.debug.print("Line {}: crop requires y=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };
        const width_value = findOption(options, "width") orelse {
            std.debug.print("Line {}: crop requires width=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };
        const height_value = findOption(options, "height") orelse {
            std.debug.print("Line {}: crop requires height=<pixels>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const x = parseUInt(u32, x_value) orelse {
            std.debug.print("Line {}: invalid crop x '{s}'\n", .{ line_no, x_value });
            return BatchParseError.InvalidBatchSpec;
        };
        const y = parseUInt(u32, y_value) orelse {
            std.debug.print("Line {}: invalid crop y '{s}'\n", .{ line_no, y_value });
            return BatchParseError.InvalidBatchSpec;
        };
        const width = parseUInt(u32, width_value) orelse {
            std.debug.print("Line {}: invalid crop width '{s}'\n", .{ line_no, width_value });
            return BatchParseError.InvalidBatchSpec;
        };
        const height = parseUInt(u32, height_value) orelse {
            std.debug.print("Line {}: invalid crop height '{s}'\n", .{ line_no, height_value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .crop = .{ .x = x, .y = y, .width = width, .height = height } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "white_balance")) {
        const temp_value = findOption(options, "temperature") orelse {
            std.debug.print("Line {}: white_balance requires temperature=<float>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };
        const tint_value = findOption(options, "tint") orelse {
            std.debug.print("Line {}: white_balance requires tint=<float>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        const temperature = parseFloat(temp_value) orelse {
            std.debug.print("Line {}: invalid temperature '{s}'\n", .{ line_no, temp_value });
            return BatchParseError.InvalidBatchSpec;
        };
        const tint = parseFloat(tint_value) orelse {
            std.debug.print("Line {}: invalid tint '{s}'\n", .{ line_no, tint_value });
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .white_balance = .{ .temperature = temperature, .tint = tint } };
    }

    if (std.ascii.eqlIgnoreCase(op_name, "color_profile_convert") or std.ascii.eqlIgnoreCase(op_name, "profile")) {
        const profile_value = findOption(options, "profile") orelse {
            std.debug.print("Line {}: color_profile_convert requires profile=<name>\n", .{line_no});
            return BatchParseError.InvalidBatchSpec;
        };

        return zpix.BatchParams{ .color_profile_convert = .{ .target_profile = profile_value } };
    }

    std.debug.print("Line {}: unknown batch operation '{s}'\n", .{ line_no, op_name });
    return BatchParseError.InvalidBatchSpec;
}

fn handleBatch(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    if (args.len < 1) {
        std.debug.print("Usage: zpix batch <script.zps>\n", .{});
        return;
    }

    const script_path = argSlice(args[0]);
    const abs_path = try makeAbsolutePath(allocator, script_path);
    defer allocator.free(abs_path);

    var file = std.fs.openFileAbsolute(abs_path, .{}) catch |err| {
        std.debug.print("Failed to open batch script '{s}': {}\n", .{ abs_path, err });
        return;
    };
    defer file.close();

    var script_bytes = std.ArrayListUnmanaged(u8){};
    defer script_bytes.deinit(allocator);

    var buffer: [4096]u8 = undefined;
    while (true) {
        const amount = try file.read(buffer[0..]);
        if (amount == 0) break;
        try script_bytes.appendSlice(allocator, buffer[0..amount]);
    }

    const script_data = script_bytes.items;

    var line_no: usize = 1;
    var line_iter = std.mem.splitScalar(u8, script_data, '\n');
    var executed_jobs: usize = 0;
    while (line_iter.next()) |line_mem| {
        const trimmed_line = std.mem.trim(u8, line_mem, " \t\r\n");
        if (trimmed_line.len == 0 or trimmed_line[0] == '#') {
            line_no += 1;
            continue;
        }

        var tokenizer = std.mem.tokenizeAny(u8, trimmed_line, " \t");
        const first = tokenizer.next() orelse {
            line_no += 1;
            continue;
        };

        if (!std.ascii.eqlIgnoreCase(first, "job")) {
            std.debug.print("Line {}: expected 'job' prefix\n", .{line_no});
            line_no += 1;
            continue;
        }

        var options = std.ArrayList(BatchOption).init(allocator);
        defer options.deinit();

        while (tokenizer.next()) |token| {
            if (token.len == 0) continue;
            if (token[0] == '#') break;
            const eq = std.mem.indexOfScalar(u8, token, '=') orelse {
                std.debug.print("Line {}: expected key=value, found '{s}'\n", .{ line_no, token });
                continue;
            };
            const key = std.mem.trim(u8, token[0..eq], " \t");
            const value = std.mem.trim(u8, token[eq + 1 ..], " \t");
            if (key.len == 0) {
                std.debug.print("Line {}: empty option name\n", .{line_no});
                continue;
            }
            try options.append(.{ .key = key, .value = value });
        }

        const op_name = findOption(options.items, "operation") orelse {
            std.debug.print("Line {}: missing operation=<name>\n", .{line_no});
            line_no += 1;
            continue;
        };

        const output_dir_value = findOption(options.items, "output") orelse {
            std.debug.print("Line {}: missing output=<dir>\n", .{line_no});
            line_no += 1;
            continue;
        };

        const inputs_value = findOption(options.items, "inputs") orelse findOption(options.items, "input") orelse {
            std.debug.print("Line {}: missing inputs=<path[,path...]>)\n", .{line_no});
            line_no += 1;
            continue;
        };

        var recursive_flag = false;
        if (findOption(options.items, "recursive")) |raw| {
            if (parseBool(raw)) |parsed| {
                recursive_flag = parsed;
            } else {
                std.debug.print("Line {}: recursive expects true/false (got '{s}')\n", .{ line_no, raw });
            }
        }

        var overwrite_flag = false;
        if (findOption(options.items, "overwrite")) |raw| {
            if (parseBool(raw)) |parsed| {
                overwrite_flag = parsed;
            } else {
                std.debug.print("Line {}: overwrite expects true/false (got '{s}')\n", .{ line_no, raw });
            }
        }

        var preserve_flag = true;
        if (findOption(options.items, "preserve_structure")) |raw| {
            if (parseBool(raw)) |parsed| {
                preserve_flag = parsed;
            } else {
                std.debug.print("Line {}: preserve_structure expects true/false (got '{s}')\n", .{ line_no, raw });
            }
        }

        var progress_flag = false;
        if (findOption(options.items, "progress")) |raw| {
            if (parseBool(raw)) |parsed| {
                progress_flag = parsed;
            } else {
                std.debug.print("Line {}: progress expects true/false (got '{s}')\n", .{ line_no, raw });
            }
        }

        var threads_count: u32 = 0;
        if (findOption(options.items, "threads")) |raw| {
            if (parseUInt(u32, raw)) |parsed| {
                threads_count = parsed;
            } else {
                std.debug.print("Line {}: threads expects positive integer (got '{s}')\n", .{ line_no, raw });
            }
        }

        var inputs = std.ArrayList([]const u8).init(allocator);
        defer inputs.deinit();
        var temp_allocations = std.ArrayList([]const u8).init(allocator);
        defer {
            for (temp_allocations.items) |path| {
                allocator.free(path);
            }
            temp_allocations.deinit();
        }

        var input_split = std.mem.splitScalar(u8, inputs_value, ',');
        while (input_split.next()) |entry| {
            if (std.mem.trim(u8, entry, " \t").len == 0) continue;
            addInputEntry(allocator, &inputs, &temp_allocations, entry, recursive_flag) catch |err| {
                std.debug.print("Line {}: failed to add input '{s}': {}\n", .{ line_no, entry, err });
            };
        }

        if (inputs.items.len == 0) {
            std.debug.print("Line {}: no input files matched\n", .{line_no});
            line_no += 1;
            continue;
        }

        const operation = parseBatchOperation(options.items, op_name, line_no) catch {
            line_no += 1;
            continue;
        };

        var job = zpix.BatchJob.init(allocator, inputs.items, output_dir_value, operation) catch |err| {
            std.debug.print("Line {}: failed to create job: {}\n", .{ line_no, err });
            line_no += 1;
            continue;
        };
        defer job.deinit(allocator);

        job.thread_count = threads_count;
        job.overwrite_existing = overwrite_flag;
        job.preserve_structure = preserve_flag;
        if (progress_flag) job.progress_callback = batchProgressCallback;

        var result = zpix.batch_functions.executeBatch(allocator, &job) catch |err| {
            std.debug.print("Line {}: batch execution failed: {}\n", .{ line_no, err });
            line_no += 1;
            continue;
        };
        defer result.deinit();

        std.debug.print(
            "Line {}: processed {} files ({} skipped, {} failed) in {} ms\n",
            .{ line_no, result.processed_files, result.skipped_files, result.failed_files, result.processing_time_ms },
        );

        if (result.errors.items.len > 0) {
            for (result.errors.items) |err_item| {
                std.debug.print("    error: {s} - {s}\n", .{ err_item.file_path, err_item.error_message });
            }
        }

        executed_jobs += 1;
        line_no += 1;
    }

    if (executed_jobs == 0) {
        std.debug.print("No batch jobs executed (script may be empty).\n", .{});
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printHelp();
        return;
    }

    const command_str = args[1];
    const command = std.meta.stringToEnum(Command, command_str) orelse {
        std.debug.print("Unknown command: {s}\n", .{command_str});
        try printHelp();
        return;
    };

    switch (command) {
        .convert => try handleConvert(allocator, args[2..]),
        .@"test" => try handleTest(allocator),
        .benchmark => try handleBenchmark(allocator),
        .pipeline => try handlePipeline(allocator, args[2..]),
        .batch => try handleBatch(allocator, args[2..]),
        .help => try printHelp(),
    }
}

fn handleConvert(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    if (args.len < 2) {
        std.debug.print("Usage: zpix convert <input> <output> [format]\n", .{});
        return;
    }

    const input_path = argSlice(args[0]);
    const output_path = argSlice(args[1]);

    var explicit_format: ?zpix.ImageFormat = null;
    if (args.len >= 3) {
        const format_token = argSlice(args[2]);
        explicit_format = parseImageFormat(format_token);
        if (explicit_format == null) {
            std.debug.print("Unknown output format: {s}\n", .{format_token});
            return;
        }
    }

    // Determine output format from extension
    const output_format = getFormatFromPath(output_path) orelse explicit_format;
    if (std.mem.eql(u8, output_path, "-") and output_format == null) {
        std.debug.print("Output '-' requires explicit format (e.g. zpix convert in.png - png)\n", .{});
        return;
    }

    std.debug.print("Converting {s} to {s}\n", .{ input_path, output_path });

    // Load image
    var image = loadImageFromSource(allocator, input_path) catch |err| {
        std.debug.print("Failed to load image: {}\n", .{err});
        return;
    };
    defer image.deinit();

    // Save image
    saveImageToDestination(allocator, &image, output_path, output_format) catch |err| {
        std.debug.print("Failed to save image: {}\n", .{err});
        return;
    };

    std.debug.print("Conversion completed successfully!\n", .{});
}

fn handleTest(allocator: std.mem.Allocator) !void {
    std.debug.print("Running zpix test suite...\n", .{});

    // Create a simple 2x2 RGB image
    var image = try zpix.Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    // Fill with red pixels: R=255, G=0, B=0
    for (0..4) |i| {
        image.data[i * 3] = 255; // R
        image.data[i * 3 + 1] = 0; // G
        image.data[i * 3 + 2] = 0; // B
    }

    // Save as BMP
    try image.save("/tmp/test.bmp", .bmp);
    std.debug.print("✓ Image saved to /tmp/test.bmp\n", .{});

    // Save as PNG
    try image.save("/tmp/test.png", .png);
    std.debug.print("✓ Image saved to /tmp/test.png\n", .{});

    // Test JPEG loading (if file exists)
    const jpeg_path = "/data/projects/zpix/file_example_JPG_100kB.jpg";
    if (std.fs.openFileAbsolute(jpeg_path, .{})) |jf| {
        jf.close();
        var jpeg_image = try zpix.Image.load(allocator, jpeg_path);
        defer jpeg_image.deinit();
        try jpeg_image.save("/tmp/test_loaded.jpg.bmp", .bmp);
        std.debug.print("✓ JPEG loaded and saved as BMP\n", .{});
    } else |_| {
        std.debug.print("- No JPEG file found at {s}\n", .{jpeg_path});
    }

    std.debug.print("All tests completed!\n", .{});
}

fn handleBenchmark(allocator: std.mem.Allocator) !void {
    std.debug.print("Running zpix performance benchmarks...\n", .{});

    // Benchmark 1: Image creation and basic operations
    const start_time = std.time.nanoTimestamp();

    // Create test images of various sizes
    const sizes = [_][2]u32{ .{ 100, 100 }, .{ 500, 500 }, .{ 1000, 1000 } };

    for (sizes) |size| {
        const width = size[0];
        const height = size[1];

        std.debug.print("Benchmarking {}x{} image operations...\n", .{ width, height });

        // Test image creation
        const create_start = std.time.nanoTimestamp();
        var image = try zpix.Image.init(allocator, width, height, .rgb);
        defer image.deinit();
        const create_end = std.time.nanoTimestamp();

        // Fill with test pattern
        for (0..image.data.len) |i| {
            image.data[i] = @intCast(i % 256);
        }

        // Test resize
        const resize_start = std.time.nanoTimestamp();
        try image.resize(width / 2, height / 2);
        const resize_end = std.time.nanoTimestamp();

        // Test blur
        const blur_start = std.time.nanoTimestamp();
        try image.blur(3);
        const blur_end = std.time.nanoTimestamp();

        // Test brightness adjustment
        const brightness_start = std.time.nanoTimestamp();
        try image.adjustBrightness(30);
        const brightness_end = std.time.nanoTimestamp();

        // Print results
        const create_time = @as(f64, @floatFromInt(create_end - create_start)) / 1_000_000.0;
        const resize_time = @as(f64, @floatFromInt(resize_end - resize_start)) / 1_000_000.0;
        const blur_time = @as(f64, @floatFromInt(blur_end - blur_start)) / 1_000_000.0;
        const brightness_time = @as(f64, @floatFromInt(brightness_end - brightness_start)) / 1_000_000.0;

        std.debug.print("  Create: {d:.2}ms\n", .{create_time});
        std.debug.print("  Resize: {d:.2}ms\n", .{resize_time});
        std.debug.print("  Blur:   {d:.2}ms\n", .{blur_time});
        std.debug.print("  Brightness: {d:.2}ms\n", .{brightness_time});
    }

    // Benchmark 2: File I/O operations
    std.debug.print("\nBenchmarking file I/O operations...\n", .{});

    // Create a test image for I/O benchmarking
    var test_image = try zpix.Image.init(allocator, 512, 512, .rgb);
    defer test_image.deinit();

    // Fill with gradient pattern
    for (0..test_image.height) |y| {
        for (0..test_image.width) |x| {
            const idx = (y * test_image.width + x) * 3;
            test_image.data[idx] = @intCast(x % 256); // R
            test_image.data[idx + 1] = @intCast(y % 256); // G
            test_image.data[idx + 2] = @intCast((x + y) % 256); // B
        }
    }

    // Test BMP save
    const bmp_save_start = std.time.nanoTimestamp();
    try test_image.save("/tmp/benchmark.bmp", .bmp);
    const bmp_save_end = std.time.nanoTimestamp();

    // Test PNG save
    const png_save_start = std.time.nanoTimestamp();
    try test_image.save("/tmp/benchmark.png", .png);
    const png_save_end = std.time.nanoTimestamp();

    // Test JPEG save (if implemented)
    const jpeg_save_start = std.time.nanoTimestamp();
    test_image.save("/tmp/benchmark.jpg", .jpeg) catch |err| {
        std.debug.print("  JPEG Save: Not implemented ({})\n", .{err});
    };
    const jpeg_save_end = std.time.nanoTimestamp();

    // Test WebP save (if implemented)
    const webp_save_start = std.time.nanoTimestamp();
    test_image.save("/tmp/benchmark.webp", .webp) catch |err| {
        std.debug.print("  WebP Save: Not implemented ({})\n", .{err});
    };
    const webp_save_end = std.time.nanoTimestamp();

    const bmp_save_time = @as(f64, @floatFromInt(bmp_save_end - bmp_save_start)) / 1_000_000.0;
    const png_save_time = @as(f64, @floatFromInt(png_save_end - png_save_start)) / 1_000_000.0;
    const jpeg_save_time = @as(f64, @floatFromInt(jpeg_save_end - jpeg_save_start)) / 1_000_000.0;
    const webp_save_time = @as(f64, @floatFromInt(webp_save_end - webp_save_start)) / 1_000_000.0;

    std.debug.print("  BMP Save:  {d:.2}ms\n", .{bmp_save_time});
    std.debug.print("  PNG Save:  {d:.2}ms\n", .{png_save_time});
    if (jpeg_save_time > 0.01) std.debug.print("  JPEG Save: {d:.2}ms\n", .{jpeg_save_time});
    if (webp_save_time > 0.01) std.debug.print("  WebP Save: {d:.2}ms\n", .{webp_save_time});

    const end_time = std.time.nanoTimestamp();
    const total_time = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    // Benchmark 3: Memory usage testing
    std.debug.print("\nBenchmarking memory usage...\n", .{});

    // Test memory efficiency for different image sizes
    const mem_test_sizes = [_][2]u32{ .{ 256, 256 }, .{ 512, 512 }, .{ 1024, 1024 } };

    for (mem_test_sizes) |size| {
        const width = size[0];
        const height = size[1];
        const mem_start = std.time.nanoTimestamp();

        // Create image and measure memory footprint
        var mem_image = try zpix.Image.init(allocator, width, height, .rgba); // RGBA for max memory usage
        defer mem_image.deinit();

        const bytes_per_pixel = zpix.bytesPerPixel(.rgba);
        const expected_bytes = width * height * bytes_per_pixel;
        const actual_bytes = mem_image.data.len;

        std.debug.print("  {}x{} RGBA - Expected: {} bytes, Actual: {} bytes\n", .{ width, height, expected_bytes, actual_bytes });

        // Test multiple allocations
        var images = std.ArrayListUnmanaged(zpix.Image){};
        defer {
            for (images.items) |*img| {
                img.deinit();
            }
            images.deinit(allocator);
        }

        const multi_alloc_start = std.time.nanoTimestamp();

        // Create 10 smaller images to test allocation patterns
        for (0..10) |_| {
            const small_image = try zpix.Image.init(allocator, width / 4, height / 4, .rgb);
            try images.append(allocator, small_image);
        }

        const multi_alloc_end = std.time.nanoTimestamp();
        const multi_alloc_time = @as(f64, @floatFromInt(multi_alloc_end - multi_alloc_start)) / 1_000_000.0;

        std.debug.print("  Multi-allocation (10x {}x{} RGB): {d:.2}ms\n", .{ width / 4, height / 4, multi_alloc_time });

        const mem_end = std.time.nanoTimestamp();
        const mem_test_time = @as(f64, @floatFromInt(mem_end - mem_start)) / 1_000_000.0;
        std.debug.print("  Total memory test time: {d:.2}ms\n", .{mem_test_time});
    }

    std.debug.print("\nTotal benchmark time: {d:.2}ms\n", .{total_time});
    std.debug.print("Benchmark completed successfully!\n", .{});
}

fn printHelp() !void {
    const help_text =
        "zpix - Image Processing Library v0.1.0\n\n" ++
        "Usage:\n" ++
        "  zpix convert <input> <output> [format]   Convert image between formats (use '-' for streams)\n" ++
        "  zpix pipeline <input> <steps...>         Apply operations in sequence, e.g. resize:800x600 save:out.png\n" ++
        "  zpix batch <file>                        Execute multiple pipeline jobs from a script\n" ++
        "  zpix test                                Run test suite\n" ++
        "  zpix benchmark                           Run performance benchmarks\n" ++
        "  zpix help                                Show this help\n\n" ++
        "Supported formats:\n" ++
        "  BMP (load/save), JPEG (load/save), PNG (load/save)\n" ++
        "  WebP (detect/save), TIFF (detect), GIF (detect)\n" ++
        "  AVIF (detect), SVG (detect)\n\n" ++
        "Examples:\n" ++
        "  zpix convert photo.png - jpeg            # stream PNG from disk to stdout as JPEG\n" ++
        "  zpix pipeline photo.jpg resize:1024x768 blur:2 format:png save:processed.png\n" ++
        "  zpix pipeline - resize:256x256 format:png save:thumbnail.png\n" ++
        "  zpix batch scripts/jobs.zps\n\n" ++
        "Pipeline steps: resize:WxH, blur:R, brightness:V, contrast:F, crop:X,Y,W,H, rotate:deg,\n" ++
        "  flip:h|v, grayscale, format:ext, save:path\n" ++
        "Use '-' as input/output to stream via stdin/stdout. When streaming to stdout specify a format hint.\n\n";

    std.debug.print("{s}", .{help_text});
}

fn getFormatFromPath(path: []const u8) ?zpix.ImageFormat {
    if (std.mem.endsWith(u8, path, ".bmp") or std.mem.endsWith(u8, path, ".BMP")) {
        return .bmp;
    } else if (std.mem.endsWith(u8, path, ".png") or std.mem.endsWith(u8, path, ".PNG")) {
        return .png;
    } else if (std.mem.endsWith(u8, path, ".jpg") or std.mem.endsWith(u8, path, ".jpeg") or
        std.mem.endsWith(u8, path, ".JPG") or std.mem.endsWith(u8, path, ".JPEG"))
    {
        return .jpeg;
    } else if (std.mem.endsWith(u8, path, ".webp") or std.mem.endsWith(u8, path, ".WEBP")) {
        return .webp;
    } else if (std.mem.endsWith(u8, path, ".tiff") or std.mem.endsWith(u8, path, ".tif") or
        std.mem.endsWith(u8, path, ".TIFF") or std.mem.endsWith(u8, path, ".TIF"))
    {
        return .tiff;
    } else if (std.mem.endsWith(u8, path, ".gif") or std.mem.endsWith(u8, path, ".GIF")) {
        return .gif;
    } else if (std.mem.endsWith(u8, path, ".avif") or std.mem.endsWith(u8, path, ".AVIF")) {
        return .avif;
    } else if (std.mem.endsWith(u8, path, ".svg") or std.mem.endsWith(u8, path, ".SVG")) {
        return .svg;
    }
    return null;
}

test "pipeline resize and save step reduces image" {
    const allocator = std.testing.allocator;

    const input_path = try createTempPath(allocator, "pipeline-input", ".bmp");
    defer {
        std.fs.deleteFileAbsolute(input_path) catch {};
        allocator.free(input_path);
    }

    var image = try zpix.Image.init(allocator, 4, 4, .rgb);
    defer image.deinit();

    for (0..image.height) |y| {
        for (0..image.width) |x| {
            const idx = (y * image.width + x) * 3;
            image.data[idx] = @intCast((x * 255) / image.width);
            image.data[idx + 1] = @intCast((y * 255) / image.height);
            image.data[idx + 2] = 200;
        }
    }

    try image.save(input_path, .bmp);

    const output_path = try createTempPath(allocator, "pipeline-output", ".bmp");
    defer {
        std.fs.deleteFileAbsolute(output_path) catch {};
        allocator.free(output_path);
    }

    const save_step = try std.fmt.allocPrint(allocator, "save:{s}", .{output_path});
    defer allocator.free(save_step);

    var steps = std.ArrayListUnmanaged([]const u8){};
    defer steps.deinit(allocator);

    try steps.append(allocator, "resize:2x2");
    try steps.append(allocator, "format:bmp");
    try steps.append(allocator, save_step);

    try runPipeline(allocator, input_path, steps.items);

    var processed = try zpix.Image.load(allocator, output_path);
    defer processed.deinit();

    try std.testing.expectEqual(@as(u32, 2), processed.width);
    try std.testing.expectEqual(@as(u32, 2), processed.height);
}

test "stdout destination requires explicit format" {
    const allocator = std.testing.allocator;
    var image = try zpix.Image.init(allocator, 2, 2, .rgb);
    defer image.deinit();

    try std.testing.expectError(error.MissingFormatForStdout, saveImageToDestination(allocator, &image, "-", null));
}
