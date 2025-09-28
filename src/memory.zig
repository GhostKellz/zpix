//! Memory monitoring and management utilities for zpix
//! Provides tracking of memory allocations and usage patterns

const std = @import("std");

/// Global memory statistics tracker
pub const MemoryStats = struct {
    total_allocated: std.atomic.Value(u64),
    current_allocated: std.atomic.Value(u64),
    peak_allocated: std.atomic.Value(u64),
    allocation_count: std.atomic.Value(u64),
    deallocation_count: std.atomic.Value(u64),

    pub fn init() MemoryStats {
        return .{
            .total_allocated = std.atomic.Value(u64).init(0),
            .current_allocated = std.atomic.Value(u64).init(0),
            .peak_allocated = std.atomic.Value(u64).init(0),
            .allocation_count = std.atomic.Value(u64).init(0),
            .deallocation_count = std.atomic.Value(u64).init(0),
        };
    }

    pub fn recordAllocation(self: *MemoryStats, size: usize) void {
        _ = self.total_allocated.fetchAdd(@intCast(size), .monotonic);
        const current = self.current_allocated.fetchAdd(@intCast(size), .monotonic) + @as(u64, @intCast(size));

        // Update peak if necessary
        var peak = self.peak_allocated.load(.monotonic);
        while (current > peak) {
            if (self.peak_allocated.cmpxchgWeak(peak, current, .monotonic, .monotonic)) |actual_peak| {
                peak = actual_peak;
            } else {
                break;
            }
        }

        _ = self.allocation_count.fetchAdd(1, .monotonic);
    }

    pub fn recordDeallocation(self: *MemoryStats, size: usize) void {
        _ = self.current_allocated.fetchSub(@intCast(size), .monotonic);
        _ = self.deallocation_count.fetchAdd(1, .monotonic);
    }

    pub fn reset(self: *MemoryStats) void {
        self.total_allocated.store(0, .monotonic);
        self.current_allocated.store(0, .monotonic);
        self.peak_allocated.store(0, .monotonic);
        self.allocation_count.store(0, .monotonic);
        self.deallocation_count.store(0, .monotonic);
    }

    pub fn report(self: *const MemoryStats, writer: anytype) !void {
        const total = self.total_allocated.load(.monotonic);
        const current = self.current_allocated.load(.monotonic);
        const peak = self.peak_allocated.load(.monotonic);
        const allocs = self.allocation_count.load(.monotonic);
        const deallocs = self.deallocation_count.load(.monotonic);

        try writer.print("Memory Statistics:\n", .{});
        try writer.print("  Total Allocated:   {} bytes ({d:.2} MB)\n", .{ total, @as(f64, @floatFromInt(total)) / (1024.0 * 1024.0) });
        try writer.print("  Current Allocated: {} bytes ({d:.2} MB)\n", .{ current, @as(f64, @floatFromInt(current)) / (1024.0 * 1024.0) });
        try writer.print("  Peak Allocated:    {} bytes ({d:.2} MB)\n", .{ peak, @as(f64, @floatFromInt(peak)) / (1024.0 * 1024.0) });
        try writer.print("  Allocations:       {}\n", .{allocs});
        try writer.print("  Deallocations:     {}\n", .{deallocs});
        try writer.print("  Active Allocations: {}\n", .{allocs - deallocs});
    }
};

/// Tracking allocator that monitors all allocations
pub const TrackingAllocator = struct {
    backing_allocator: std.mem.Allocator,
    stats: *MemoryStats,

    pub fn init(backing: std.mem.Allocator, stats: *MemoryStats) std.mem.Allocator {
        const self = backing.create(TrackingAllocator) catch @panic("Failed to create tracking allocator");
        self.* = .{
            .backing_allocator = backing,
            .stats = stats,
        };

        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result) |_| {
            self.stats.recordAllocation(len);
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;
        const result = self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);
        if (result) {
            if (new_len > old_len) {
                self.stats.recordAllocation(new_len - old_len);
            } else {
                self.stats.recordDeallocation(old_len - new_len);
            }
        }
        return result;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.stats.recordDeallocation(buf.len);
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }
};

/// Memory pool for efficient repeated allocations of same-sized objects
pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    block_size: usize,
    blocks: std.ArrayList([]u8),
    free_list: std.ArrayList([]u8),

    pub fn init(allocator: std.mem.Allocator, block_size: usize) MemoryPool {
        return .{
            .allocator = allocator,
            .block_size = block_size,
            .blocks = std.ArrayList([]u8).init(allocator),
            .free_list = std.ArrayList([]u8).init(allocator),
        };
    }

    pub fn deinit(self: *MemoryPool) void {
        for (self.blocks.items) |block| {
            self.allocator.free(block);
        }
        self.blocks.deinit();
        self.free_list.deinit();
    }

    pub fn alloc(self: *MemoryPool) ![]u8 {
        if (self.free_list.items.len > 0) {
            return self.free_list.pop();
        }

        const block = try self.allocator.alloc(u8, self.block_size);
        try self.blocks.append(self.allocator, block);
        return block;
    }

    pub fn free(self: *MemoryPool, block: []u8) !void {
        if (block.len != self.block_size) {
            return error.InvalidBlockSize;
        }
        try self.free_list.append(self.allocator, block);
    }

    pub fn reset(self: *MemoryPool) void {
        self.free_list.clearRetainingCapacity();
        for (self.blocks.items) |block| {
            self.free_list.append(self.allocator, block) catch {};
        }
    }
};