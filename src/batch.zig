//! Multi-threaded batch processing for zpix
//! Provides efficient parallel processing of multiple images with progress tracking

const std = @import("std");
const root = @import("root.zig");

/// Batch operation types
pub const BatchOperation = enum {
    resize,
    convert_format,
    adjust_brightness,
    adjust_contrast,
    blur,
    rotate,
    crop,
    white_balance,
    color_profile_convert,
    custom,
};

/// Parameters for batch operations
pub const BatchParams = union(BatchOperation) {
    resize: struct {
        width: u32,
        height: u32,
        maintain_aspect: bool = true,
    },
    convert_format: struct {
        target_format: root.ImageFormat,
    },
    adjust_brightness: struct {
        brightness: i32,
    },
    adjust_contrast: struct {
        contrast: f32,
    },
    blur: struct {
        radius: u32,
    },
    rotate: struct {
        angle: root.RotationAngle,
    },
    crop: struct {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    white_balance: struct {
        temperature: f32,
        tint: f32,
    },
    color_profile_convert: struct {
        target_profile: []const u8,
    },
    custom: struct {
        processor: *const fn(*root.Image) anyerror!void,
    },
};

/// Progress callback function type
pub const ProgressCallback = *const fn(completed: u32, total: u32, current_file: []const u8) void;

/// Cancellation token for stopping batch operations
pub const CancellationToken = struct {
    cancelled: std.atomic.Value(bool),

    pub fn init() CancellationToken {
        return CancellationToken{
            .cancelled = std.atomic.Value(bool).init(false),
        };
    }

    pub fn cancel(self: *CancellationToken) void {
        self.cancelled.store(true, .release);
    }

    pub fn isCancelled(self: *const CancellationToken) bool {
        return self.cancelled.load(.acquire);
    }
};

/// Batch processing job
pub const BatchJob = struct {
    input_files: [][]const u8,
    output_dir: []const u8,
    operation: BatchParams,
    progress_callback: ?ProgressCallback = null,
    cancellation_token: ?*CancellationToken = null,
    thread_count: u32 = 0, // 0 = auto-detect
    overwrite_existing: bool = false,
    preserve_structure: bool = true,

    pub fn init(allocator: std.mem.Allocator, input_files: [][]const u8, output_dir: []const u8, operation: BatchParams) !BatchJob {
        const files_copy = try allocator.alloc([]const u8, input_files.len);
        for (input_files, 0..) |file, i| {
            files_copy[i] = try allocator.dupe(u8, file);
        }

        return BatchJob{
            .input_files = files_copy,
            .output_dir = try allocator.dupe(u8, output_dir),
            .operation = operation,
        };
    }

    pub fn deinit(self: *BatchJob, allocator: std.mem.Allocator) void {
        for (self.input_files) |file| {
            allocator.free(file);
        }
        allocator.free(self.input_files);
        allocator.free(self.output_dir);
    }
};

/// Batch processing results
pub const BatchResult = struct {
    total_files: u32,
    processed_files: u32,
    failed_files: u32,
    skipped_files: u32,
    processing_time_ms: u64,
    errors: std.ArrayList(BatchError),

    pub fn init(allocator: std.mem.Allocator) BatchResult {
        return BatchResult{
            .total_files = 0,
            .processed_files = 0,
            .failed_files = 0,
            .skipped_files = 0,
            .processing_time_ms = 0,
            .errors = std.ArrayList(BatchError).init(allocator),
        };
    }

    pub fn deinit(self: *BatchResult) void {
        for (self.errors.items) |*error_item| {
            error_item.deinit();
        }
        self.errors.deinit();
    }
};

/// Batch processing error
pub const BatchError = struct {
    file_path: []u8,
    error_message: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, file_path: []const u8, error_message: []const u8) !BatchError {
        return BatchError{
            .file_path = try allocator.dupe(u8, file_path),
            .error_message = try allocator.dupe(u8, error_message),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchError) void {
        self.allocator.free(self.file_path);
        self.allocator.free(self.error_message);
    }
};

/// Worker thread data
const WorkerData = struct {
    allocator: std.mem.Allocator,
    job: *const BatchJob,
    result: *BatchResult,
    file_index: *std.atomic.Value(u32),
    completed_count: *std.atomic.Value(u32),
    mutex: *std.Thread.Mutex,
};

/// Execute batch processing job
pub fn executeBatch(allocator: std.mem.Allocator, job: *const BatchJob) !BatchResult {
    const start_time = std.time.milliTimestamp();

    var result = BatchResult.init(allocator);
    result.total_files = @intCast(job.input_files.len);

    // Determine thread count
    const thread_count = if (job.thread_count > 0)
        job.thread_count
    else
        @max(1, std.Thread.getCpuCount() catch 4);

    // Create output directory if it doesn't exist
    std.fs.cwd().makePath(job.output_dir) catch {};

    // Shared data for worker threads
    var file_index = std.atomic.Value(u32).init(0);
    var completed_count = std.atomic.Value(u32).init(0);
    var mutex = std.Thread.Mutex{};

    var worker_data = WorkerData{
        .allocator = allocator,
        .job = job,
        .result = &result,
        .file_index = &file_index,
        .completed_count = &completed_count,
        .mutex = &mutex,
    };

    // Create worker threads
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, workerThread, .{&worker_data});
    }

    // Wait for all threads to complete
    for (threads) |*thread| {
        thread.join();
    }

    result.processing_time_ms = @intCast(std.time.milliTimestamp() - start_time);
    return result;
}

/// Worker thread function
fn workerThread(data: *WorkerData) void {
    while (true) {
        // Check for cancellation
        if (data.job.cancellation_token) |token| {
            if (token.isCancelled()) break;
        }

        // Get next file to process
        const index = data.file_index.fetchAdd(1, .acq_rel);
        if (index >= data.job.input_files.len) break;

        const input_file = data.job.input_files[index];

        // Process the file
        processFile(data, input_file, index) catch |err| {
            // Handle error
            data.mutex.lock();
            defer data.mutex.unlock();

            data.result.failed_files += 1;

            const error_msg = switch (err) {
                error.FileNotFound => "File not found",
                error.AccessDenied => "Access denied",
                error.OutOfMemory => "Out of memory",
                error.InvalidFormat => "Invalid image format",
                else => "Unknown error",
            };

            const batch_error = BatchError.init(data.allocator, input_file, error_msg) catch return;
            data.result.errors.append(batch_error) catch return;
        };

        // Update progress
        const completed = data.completed_count.fetchAdd(1, .acq_rel) + 1;
        if (data.job.progress_callback) |callback| {
            callback(completed, @intCast(data.job.input_files.len), input_file);
        }
    }
}

/// Process a single file
fn processFile(data: *WorkerData, input_file: []const u8, index: u32) !void {
    _ = index;

    // Load image
    var image = try root.Image.load(input_file, data.allocator);
    defer image.deinit();

    // Apply operation
    try applyOperation(&image, data.job.operation);

    // Generate output filename
    const output_path = try generateOutputPath(data.allocator, input_file, data.job.output_dir, data.job.preserve_structure);
    defer data.allocator.free(output_path);

    // Check if output file exists
    if (!data.job.overwrite_existing) {
        if (std.fs.cwd().access(output_path, .{})) {
            data.mutex.lock();
            defer data.mutex.unlock();
            data.result.skipped_files += 1;
            return;
        } else |_| {}
    }

    // Create output directory if needed
    if (std.fs.path.dirname(output_path)) |dir| {
        std.fs.cwd().makePath(dir) catch {};
    }

    // Save image
    try image.save(output_path);

    data.mutex.lock();
    defer data.mutex.unlock();
    data.result.processed_files += 1;
}

/// Apply batch operation to image
fn applyOperation(image: *root.Image, operation: BatchParams) !void {
    switch (operation) {
        .resize => |params| {
            if (params.maintain_aspect) {
                const aspect_ratio = @as(f32, @floatFromInt(image.width)) / @as(f32, @floatFromInt(image.height));
                const new_height = @as(u32, @intFromFloat(@as(f32, @floatFromInt(params.width)) / aspect_ratio));
                try image.resize(params.width, new_height);
            } else {
                try image.resize(params.width, params.height);
            }
        },
        .convert_format => |params| {
            image.format = params.target_format;
        },
        .adjust_brightness => |params| {
            try image.adjustBrightness(params.brightness);
        },
        .adjust_contrast => |params| {
            try image.adjustContrast(params.contrast);
        },
        .blur => |params| {
            try image.blur(params.radius);
        },
        .rotate => |params| {
            try image.rotate(params.angle);
        },
        .crop => |params| {
            try image.crop(params.x, params.y, params.width, params.height);
        },
        .white_balance => |params| {
            try image.adjustWhiteBalance(params.temperature, params.tint);
        },
        .color_profile_convert => |_| {
            // TODO: Implement color profile conversion
        },
        .custom => |params| {
            try params.processor(image);
        },
    }
}

/// Generate output path for processed image
fn generateOutputPath(allocator: std.mem.Allocator, input_path: []const u8, output_dir: []const u8, preserve_structure: bool) ![]u8 {
    const basename = std.fs.path.basename(input_path);

    if (preserve_structure) {
        // Preserve directory structure relative to a common root
        const relative_path = if (std.fs.path.dirname(input_path)) |dir|
            try std.fs.path.relative(allocator, ".", dir)
        else
            try allocator.dupe(u8, ".");
        defer allocator.free(relative_path);

        const output_subdir = try std.fs.path.join(allocator, &[_][]const u8{ output_dir, relative_path });
        defer allocator.free(output_subdir);

        return try std.fs.path.join(allocator, &[_][]const u8{ output_subdir, basename });
    } else {
        return try std.fs.path.join(allocator, &[_][]const u8{ output_dir, basename });
    }
}

/// Scan directory for image files matching patterns
pub fn scanDirectory(allocator: std.mem.Allocator, dir_path: []const u8, patterns: []const []const u8, recursive: bool) ![][]const u8 {
    var files = std.ArrayList([]const u8).init(allocator);
    defer {
        // Clean up on error
        for (files.items) |file| {
            allocator.free(file);
        }
        files.deinit();
    }

    try scanDirectoryRecursive(allocator, &files, dir_path, patterns, recursive);

    return try files.toOwnedSlice();
}

/// Recursive directory scanning implementation
fn scanDirectoryRecursive(allocator: std.mem.Allocator, files: *std.ArrayList([]const u8), dir_path: []const u8, patterns: []const []const u8, recursive: bool) !void {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch return;
    defer dir.close();

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        const full_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, entry.name });

        switch (entry.kind) {
            .file => {
                // Check if file matches any pattern
                for (patterns) |pattern| {
                    if (matchesPattern(entry.name, pattern)) {
                        try files.append(full_path);
                        break;
                    }
                } else {
                    allocator.free(full_path);
                }
            },
            .directory => {
                if (recursive) {
                    try scanDirectoryRecursive(allocator, files, full_path, patterns, recursive);
                }
                allocator.free(full_path);
            },
            else => {
                allocator.free(full_path);
            },
        }
    }
}

/// Simple pattern matching (supports * wildcards)
fn matchesPattern(filename: []const u8, pattern: []const u8) bool {
    if (std.mem.indexOf(u8, pattern, "*")) |star_pos| {
        const prefix = pattern[0..star_pos];
        const suffix = pattern[star_pos + 1..];

        return std.mem.startsWith(u8, filename, prefix) and std.mem.endsWith(u8, filename, suffix);
    } else {
        return std.mem.eql(u8, filename, pattern);
    }
}

/// Create batch job for common operations
pub const BatchJobBuilder = struct {
    allocator: std.mem.Allocator,
    input_files: std.ArrayList([]const u8),
    output_dir: ?[]const u8 = null,
    operation: ?BatchParams = null,
    thread_count: u32 = 0,
    progress_callback: ?ProgressCallback = null,
    cancellation_token: ?*CancellationToken = null,

    pub fn init(allocator: std.mem.Allocator) BatchJobBuilder {
        return BatchJobBuilder{
            .allocator = allocator,
            .input_files = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *BatchJobBuilder) void {
        for (self.input_files.items) |file| {
            self.allocator.free(file);
        }
        self.input_files.deinit();
        if (self.output_dir) |dir| {
            self.allocator.free(dir);
        }
    }

    pub fn addFile(self: *BatchJobBuilder, file_path: []const u8) !*BatchJobBuilder {
        try self.input_files.append(try self.allocator.dupe(u8, file_path));
        return self;
    }

    pub fn addDirectory(self: *BatchJobBuilder, dir_path: []const u8, patterns: []const []const u8, recursive: bool) !*BatchJobBuilder {
        const files = try scanDirectory(self.allocator, dir_path, patterns, recursive);
        defer self.allocator.free(files);

        for (files) |file| {
            try self.input_files.append(file);
        }
        return self;
    }

    pub fn setOutputDirectory(self: *BatchJobBuilder, output_dir: []const u8) !*BatchJobBuilder {
        if (self.output_dir) |old_dir| {
            self.allocator.free(old_dir);
        }
        self.output_dir = try self.allocator.dupe(u8, output_dir);
        return self;
    }

    pub fn setOperation(self: *BatchJobBuilder, operation: BatchParams) *BatchJobBuilder {
        self.operation = operation;
        return self;
    }

    pub fn setThreadCount(self: *BatchJobBuilder, count: u32) *BatchJobBuilder {
        self.thread_count = count;
        return self;
    }

    pub fn setProgressCallback(self: *BatchJobBuilder, callback: ProgressCallback) *BatchJobBuilder {
        self.progress_callback = callback;
        return self;
    }

    pub fn setCancellationToken(self: *BatchJobBuilder, token: *CancellationToken) *BatchJobBuilder {
        self.cancellation_token = token;
        return self;
    }

    pub fn build(self: *BatchJobBuilder) !BatchJob {
        if (self.output_dir == null) return error.NoOutputDirectory;
        if (self.operation == null) return error.NoOperation;
        if (self.input_files.items.len == 0) return error.NoInputFiles;

        const files = try self.input_files.toOwnedSlice();
        return BatchJob{
            .input_files = files,
            .output_dir = self.output_dir.?,
            .operation = self.operation.?,
            .thread_count = self.thread_count,
            .progress_callback = self.progress_callback,
            .cancellation_token = self.cancellation_token,
            .overwrite_existing = false,
            .preserve_structure = true,
        };
    }
};