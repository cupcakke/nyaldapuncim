const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    InvalidHyperparameter,
    InvalidClipRange,
    InvalidToken,
    AllocationFailed,
    PartialRowCleanup,
};

pub const WeightKind = enum {
    weights_s,
    weights_t,
};

pub const StackArrayKind = enum {
    weights_s,
    weights_t,
    master_weights_s,
    master_weights_t,
    momentum_s,
    momentum_t,
    fisher_s,
    fisher_t,
};


pub const RSFOptimizerState = struct {
    master_weights_s: []f32,
    master_weights_t: []f32,
    momentum_s: []f32,
    momentum_t: []f32,
    fisher_s: []f32,
    fisher_t: []f32,
    step: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RSFOptimizerState) void {
        self.allocator.free(self.master_weights_s);
        self.allocator.free(self.master_weights_t);
        self.allocator.free(self.momentum_s);
        self.allocator.free(self.momentum_t);
        self.allocator.free(self.fisher_s);
        self.allocator.free(self.fisher_t);
        self.* = undefined;
    }
};

pub const EmbeddingOptimizerState = struct {
    master_weights: []f32,
    momentum: []f32,
    fisher: []f32,
    step: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *EmbeddingOptimizerState) void {
        self.allocator.free(self.master_weights);
        self.allocator.free(self.momentum);
        self.allocator.free(self.fisher);
        self.* = undefined;
    }
};

pub const DeviceBufferF16 = struct {
    ptr: *anyopaque,
    count: usize,
};

pub const DeviceBufferF32 = struct {
    ptr: *anyopaque,
    count: usize,
};

pub const DeviceBufferU64 = struct {
    ptr: *anyopaque,
    count: usize,
};

fn freeFutharkError(message: ?[*:0]const u8) void {
    if (message) |ptr| std.c.free(@ptrCast(@constCast(ptr)));
}

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,
    mutex: std.Thread.Mutex = .{},

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        if (comptime gpu_enabled) {
            const cache_file: ?[*:0]const u8 = if (std.posix.getenv("JAIDE_FUTHARK_CACHE")) |cache_path| blk: {
                std.debug.print("[FutharkContext] GPU kernel cache: {s}\n", .{cache_path});
                break :blk @as([*:0]const u8, @ptrCast(cache_path.ptr));
            } else blk: {
                std.debug.print("[FutharkContext] WARN: JAIDE_FUTHARK_CACHE not set — NVRTC will recompile on every container start\n", .{});
                break :blk null;
            };
            futhark.configureGpuContext(cfg, cache_file) catch return AccelError.FutharkConfigFailed;
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkContextFailed;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx, .cfg = cfg, .mutex = .{} };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            _ = futhark.futhark_context_clear_caches(ctx);
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn syncLocked(self: *Self) AccelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.sync();
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }

    pub fn getDataPointer1D(self: *Self, array: *FutharkArray1DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_1d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }

    pub fn getDataPointer3D(self: *Self, array: *FutharkArray3DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_3d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointerF32_1D(self: *Self, array: *FutharkArray1DF32) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_f32_1d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointerF32_2D(self: *Self, array: *FutharkArray2DF32) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_f32_2d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointerF32_3D(self: *Self, array: *FutharkArray3DF32) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_f32_3d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointer1DI64(self: *Self, array: *FutharkArray1DI64) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_i64_1d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointer1DU64(self: *Self, array: *FutharkArray1DU64) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_u64_1d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointer2DU64(self: *Self, array: *FutharkArray2DU64) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_u64_2d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }
};

fn get1DDevicePtr(ctx: *FutharkContext, array: *FutharkArray1DF16) AccelError!*anyopaque {
    return ctx.getDataPointer1D(array);
}

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    fallback_slice: ?[]align(64) u8,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0, .fallback_slice = null };
        }

        if (comptime gpu_enabled) {
            var ptr: ?*anyopaque = null;
            const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
            if (err != cuda.cudaSuccess) {
                return AccelError.CudaHostAllocFailed;
            }
            return Self{
                .ptr = ptr,
                .size = size,
                .fallback_slice = null,
            };
        }

        const slice = std.heap.page_allocator.alignedAlloc(u8, 64, size) catch return AccelError.CudaHostAllocFailed;
        return Self{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .fallback_slice = slice,
        };
    }

    pub fn free(self: *Self) void {
        if (self.fallback_slice) |slice| {
            std.heap.page_allocator.free(slice);
            self.fallback_slice = null;
            self.ptr = null;
            self.size = 0;
            return;
        }
        if (self.ptr) |p| {
            if (comptime gpu_enabled) {
                _ = cuda.cudaFreeHost(p);
            }
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        const count = self.size / @sizeOf(T);
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != length) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn newZeros(ctx: *FutharkContext, length: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;

        const zeros = allocator.alloc(f16, length) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn values1D(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;

        const buf = allocator.alloc(f16, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);

        const result = futhark.futhark_values_f16_1d(ctx.ctx, self.arr, @ptrCast(buf.ptr));
        if (result != 0) return AccelError.FutharkValuesFailed;

        const sync_result = futhark.futhark_context_sync(ctx.ctx);
        if (sync_result != 0) return AccelError.FutharkSyncFailed;

        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        const rows = self.rows;
        const cols = self.cols;

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        if (flat.len != d0 * d1 * d2) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_3d(
            ctx.ctx,
            @ptrCast(flat.ptr),
            @intCast(d0),
            @intCast(d1),
            @intCast(d2),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF16 {
        if (self.dim0 == 0 or self.dim1 == 0 or self.dim2 == 0) return AccelError.InvalidDimensions;
        const first = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const count = std.math.mul(usize, first, self.dim2) catch return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointer3D(self), .count = count };
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_3d(ctx.ctx, self.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) {
            tensor.deinit();
            return AccelError.FutharkSyncFailed;
        }
        return tensor;
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null or self.arr == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const values = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(values);
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, values.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return values;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF32 {
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointerF32_2D(self), .count = total };
    }
};

pub const FutharkArray3DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, d0, d1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, d2) catch return AccelError.InvalidDimensions;
        if (data.len != total) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_3d(ctx.ctx, data.ptr, @intCast(d0), @intCast(d1), @intCast(d2));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn newZeros(ctx: *FutharkContext, d0: usize, d1: usize, d2: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, d0, d1) catch return AccelError.AllocationFailed;
        const total = std.math.mul(usize, d01, d2) catch return AccelError.AllocationFailed;
        const zeros = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_3d(ctx.ctx, zeros.ptr, @intCast(d0), @intCast(d1), @intCast(d2));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_3d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF32 {
        if (self.dim0 == 0 or self.dim1 == 0 or self.dim2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointerF32_3D(self), .count = total };
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) {
            tensor.deinit();
            return AccelError.FutharkSyncFailed;
        }
        return tensor;
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const f32) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DF32, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f32, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF32 {
        if (self.len == 0) return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointerF32_1D(self), .count = self.len };
    }
};

pub const FutharkArray1DU64 = struct {
    arr: ?*futhark.struct_futhark_u64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_u64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DU64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]u64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(u64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_u64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_u64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DU64 = struct {
    arr: ?*futhark.struct_futhark_u64_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, data: []const u64, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, rows, cols) catch return AccelError.InvalidDimensions;
        if (data.len != total) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_u64_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]u64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(u64, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_u64_2d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferU64 {
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointer2DU64(self), .count = total };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_u64_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }
};

pub const RSFLayer = struct {
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,

    pub fn free(self: *RSFLayer, ctx: *FutharkContext) void {
        self.weights_t.free(ctx);
        self.weights_s.free(ctx);
    }
};

pub const FusedStepScalars = struct {
    loss: f32,
    reconstruction_loss: f32,
    logdet_mean: f32,
};

pub const FusedStepResult = struct {
    input_delta: FutharkArray3DF16,
    pending: ?*futhark.struct_futhark_opaque_tup10_fused_stack_step,
    finalized: bool,
    scalars: FusedStepScalars,

    pub fn finalize(self: *FusedStepResult, ctx: *FutharkContext) AccelError!FusedStepScalars {
        if (self.finalized) return self.scalars;
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        const tup = self.pending orelse {
            self.finalized = true;
            return self.scalars;
        };
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        var loss_out: f32 = 0.0;
        var recon_out: f32 = 0.0;
        var logdet_out: f32 = 0.0;
        const p7 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_7(ctx.ctx, &loss_out, tup);
        const p8 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_8(ctx.ctx, &recon_out, tup);
        const p9 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_9(ctx.ctx, &logdet_out, tup);
        _ = futhark.futhark_free_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(ctx.ctx, tup);
        self.pending = null;
        if (p7 != 0 or p8 != 0 or p9 != 0) return AccelError.FutharkTrainingStepFailed;
        if (!std.math.isFinite(loss_out) or !std.math.isFinite(recon_out) or !std.math.isFinite(logdet_out)) return AccelError.FutharkTrainingStepFailed;
        self.scalars = .{ .loss = loss_out, .reconstruction_loss = recon_out, .logdet_mean = logdet_out };
        self.finalized = true;
        return self.scalars;
    }

    pub fn deinit(self: *FusedStepResult, ctx: *FutharkContext) void {
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        self.input_delta.free(ctx);
        if (self.pending) |tup| {
            _ = futhark.futhark_free_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(ctx.ctx, tup);
            self.pending = null;
        }
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    layers: []RSFLayer,
    layers_owner: std.mem.Allocator,
    allocator: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,
    scratch_lengths_buf: []i64 = &[_]i64{},
    scratch_lengths_cap: usize = 0,
    stack_weights_s: ?FutharkArray3DF16 = null,
    stack_weights_t: ?FutharkArray3DF16 = null,
    stack_master_weights_s: ?FutharkArray3DF32 = null,
    stack_master_weights_t: ?FutharkArray3DF32 = null,
    stack_momentum_s: ?FutharkArray3DF32 = null,
    stack_momentum_t: ?FutharkArray3DF32 = null,
    stack_fisher_s: ?FutharkArray3DF32 = null,
    stack_fisher_t: ?FutharkArray3DF32 = null,
    stack_valid: bool = false,
    layers_mirror_valid: bool = true,
    optimizer_step: u64 = 0,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        return initMultiLayerWithDepthScale(model_dim, num_layers, allocator, true);
    }

    pub fn initMultiLayerWithDepthScale(
        model_dim: usize,
        num_layers: usize,
        allocator: std.mem.Allocator,
        depth_compensation: bool,
    ) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        const half: usize = model_dim / 2;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const base_seed: u64 = 0x4A41494445204E4F;
        const depth_scale: f32 = if (depth_compensation)
            1.0 / @sqrt(@as(f32, @floatFromInt(num_layers)))
        else
            1.0;
        const init_stddev: f32 = depth_scale * 0.25 / @sqrt(@as(f32, @floatFromInt(half)));

        var layers = allocator.alloc(RSFLayer, num_layers) catch return AccelError.AllocationFailed;
        errdefer allocator.free(layers);

        const total: usize = half * (half + 1);
        const ws_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(ws_buf);
        const wt_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(wt_buf);

        var layers_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < layers_built) : (idx += 1) {
                layers[idx].free(&ctx);
            }
        }

        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            const layer_seed: u64 = base_seed +% (@as(u64, @intCast(layer_idx)) *% 0x9E3779B97F4A7C15);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const rnd = rng.random();
            for (ws_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            for (wt_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            {
                var d: usize = 0;
                while (d < half) : (d += 1) {
                    ws_buf[d * (half + 1) + half] = @as(f16, 0.0);
                    wt_buf[d * (half + 1) + half] = @as(f16, 0.0);
                }
            }
            const target_spectral_norm: f32 = 0.9;
            const sigma_s = try matrixSpectralNorm(ws_buf, half, half + 1, 30, allocator);
            if (sigma_s > target_spectral_norm) {
                const factor = target_spectral_norm / sigma_s;
                for (ws_buf) |*value| value.* = @floatCast(@as(f32, @floatCast(value.*)) * factor);
            }
            const sigma_t = try matrixSpectralNorm(wt_buf, half, half + 1, 30, allocator);
            if (sigma_t > target_spectral_norm) {
                const factor = target_spectral_norm / sigma_t;
                for (wt_buf) |*value| value.* = @floatCast(@as(f32, @floatCast(value.*)) * factor);
            }

            const weights_s = try FutharkArray2DF16.newFromFlat(&ctx, ws_buf, half, half + 1);
            const weights_t = try FutharkArray2DF16.newFromFlat(&ctx, wt_buf, half, half + 1);

            layers[layer_idx] = .{
                .weights_s = weights_s,
                .weights_t = weights_t,
            };
            layers_built += 1;
        }

        const max_batch: usize = 2048;
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);

        return Self{
            .ctx = ctx,
            .layers = layers,
            .layers_owner = allocator,
            .allocator = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = @as(f16, -5.0),
            .clip_max = @as(f16, 5.0),
            .initialized = true,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        if (self.scratch_lengths_buf.len > 0) {
            self.allocator.free(self.scratch_lengths_buf);
            self.scratch_lengths_buf = &[_]i64{};
        }
        self.freeStackArrays();

        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            self.layers[i].free(&self.ctx);
        }
        self.layers_owner.free(self.layers);
        self.ctx.deinit();
        self.initialized = false;
    }

    fn freeStackArrays(self: *Self) void {
        if (self.stack_fisher_t) |*a| a.free(&self.ctx);
        if (self.stack_fisher_s) |*a| a.free(&self.ctx);
        if (self.stack_momentum_t) |*a| a.free(&self.ctx);
        if (self.stack_momentum_s) |*a| a.free(&self.ctx);
        if (self.stack_weights_t) |*a| a.free(&self.ctx);
        if (self.stack_weights_s) |*a| a.free(&self.ctx);
        if (self.stack_master_weights_t) |*a| a.free(&self.ctx);
        if (self.stack_master_weights_s) |*a| a.free(&self.ctx);
        self.stack_fisher_t = null;
        self.stack_fisher_s = null;
        self.stack_momentum_t = null;
        self.stack_momentum_s = null;
        self.stack_weights_t = null;
        self.stack_weights_s = null;
        self.stack_master_weights_t = null;
        self.stack_master_weights_s = null;
        self.stack_valid = false;
    }

    fn markStackDirty(self: *Self) void {
        self.stack_valid = false;
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn layerPtr(self: *Self, layer_idx: usize) AccelError!*RSFLayer {
        if (!self.initialized) return AccelError.NullPointer;
        if (layer_idx >= self.layers.len) return AccelError.InvalidDimensions;
        return &self.layers[layer_idx];
    }

    fn stackStateDimensionsMatch(state: FutharkArray3DF32, layers: usize, half: usize, cols: usize) bool {
        return state.dim0 == layers and state.dim1 == half and state.dim2 == cols and state.arr != null;
    }

    /// Repack only the f16 forward weights.  Persistent f32 optimizer state is
    /// created once and survives checkpoint loads, scaling, and host-side
    /// weight replacement.
    pub fn ensureStackPacked(self: *Self) AccelError!void {
        if (self.stack_valid and self.stack_weights_s != null and self.stack_weights_t != null and
            self.stack_master_weights_s != null and self.stack_master_weights_t != null and
            self.stack_momentum_s != null and self.stack_momentum_t != null and
            self.stack_fisher_s != null and self.stack_fisher_t != null) return;

        const l_count = self.layers.len;
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, l_count, per_layer) catch return AccelError.InvalidDimensions;
        const ws_flat = self.allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(ws_flat);
        const wt_flat = self.allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(wt_flat);

        for (self.layers, 0..) |*layer, layer_index| {
            const ws = layer.weights_s.valuesFlat(&self.ctx, self.allocator) catch return AccelError.FutharkValuesFailed;
            defer self.allocator.free(ws);
            const wt = layer.weights_t.valuesFlat(&self.ctx, self.allocator) catch return AccelError.FutharkValuesFailed;
            defer self.allocator.free(wt);
            if (ws.len != per_layer or wt.len != per_layer) return AccelError.InvalidDimensions;
            @memcpy(ws_flat[layer_index * per_layer .. (layer_index + 1) * per_layer], ws);
            @memcpy(wt_flat[layer_index * per_layer .. (layer_index + 1) * per_layer], wt);
        }

        var replacement_s = try FutharkArray3DF16.newFromFlat(&self.ctx, ws_flat, l_count, half, cols);
        errdefer replacement_s.free(&self.ctx);
        var replacement_t = try FutharkArray3DF16.newFromFlat(&self.ctx, wt_flat, l_count, half, cols);
        errdefer replacement_t.free(&self.ctx);
        const ws_master_flat = self.allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(ws_master_flat);
        const wt_master_flat = self.allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(wt_master_flat);
        for (ws_flat, ws_master_flat) |value, *master| master.* = @floatCast(value);
        for (wt_flat, wt_master_flat) |value, *master| master.* = @floatCast(value);
        var replacement_master_s = try FutharkArray3DF32.newFromFlat(&self.ctx, ws_master_flat, l_count, half, cols);
        errdefer replacement_master_s.free(&self.ctx);
        var replacement_master_t = try FutharkArray3DF32.newFromFlat(&self.ctx, wt_master_flat, l_count, half, cols);
        errdefer replacement_master_t.free(&self.ctx);

        const zeros = self.allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        if (self.stack_momentum_s == null) self.stack_momentum_s = try FutharkArray3DF32.newFromFlat(&self.ctx, zeros, l_count, half, cols);
        if (self.stack_momentum_t == null) self.stack_momentum_t = try FutharkArray3DF32.newFromFlat(&self.ctx, zeros, l_count, half, cols);
        if (self.stack_fisher_s == null) self.stack_fisher_s = try FutharkArray3DF32.newFromFlat(&self.ctx, zeros, l_count, half, cols);
        if (self.stack_fisher_t == null) self.stack_fisher_t = try FutharkArray3DF32.newFromFlat(&self.ctx, zeros, l_count, half, cols);
        if (!stackStateDimensionsMatch(self.stack_momentum_s.?, l_count, half, cols) or
            !stackStateDimensionsMatch(self.stack_momentum_t.?, l_count, half, cols) or
            !stackStateDimensionsMatch(self.stack_fisher_s.?, l_count, half, cols) or
            !stackStateDimensionsMatch(self.stack_fisher_t.?, l_count, half, cols)) return AccelError.InvalidDimensions;

        if (self.stack_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_weights_t) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_t) |*old| old.free(&self.ctx);
        self.stack_weights_s = replacement_s;
        self.stack_weights_t = replacement_t;
        self.stack_master_weights_s = replacement_master_s;
        self.stack_master_weights_t = replacement_master_t;
        self.stack_valid = true;
        self.layers_mirror_valid = true;
    }

    fn assignLayerWeightsDirect(self: *Self, layer_idx: usize, is_s: bool, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (is_s) {
            layer.weights_s.free(&self.ctx);
            layer.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        } else {
            layer.weights_t.free(&self.ctx);
            layer.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        }
    }

    pub fn syncLayersFromStack(self: *Self) AccelError!void {
        if (!self.stack_valid) return;
        if (self.layers_mirror_valid) return;
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const l_count = self.layers.len;
        const total = std.math.mul(usize, l_count, per_layer) catch return AccelError.InvalidDimensions;

        if (self.stack_weights_s) |*sws| {
            const flat = try sws.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(flat);
            if (flat.len != total) return AccelError.InvalidDimensions;
            var li: usize = 0;
            while (li < l_count) : (li += 1) {
                try self.assignLayerWeightsDirect(li, true, flat[li * per_layer .. (li + 1) * per_layer], half, cols);
            }
        }
        if (self.stack_weights_t) |*swt| {
            const flat = try swt.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(flat);
            if (flat.len != total) return AccelError.InvalidDimensions;
            var li: usize = 0;
            while (li < l_count) : (li += 1) {
                try self.assignLayerWeightsDirect(li, false, flat[li * per_layer .. (li + 1) * per_layer], half, cols);
            }
        }
        self.layers_mirror_valid = true;
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!RSFOptimizerState {
        try self.ensureStackPacked();
        const master_s = try self.stack_master_weights_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(master_s);
        const master_t = try self.stack_master_weights_t.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(master_t);
        const ms = try self.stack_momentum_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(ms);
        const mt = try self.stack_momentum_t.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(mt);
        const fs = try self.stack_fisher_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(fs);
        const ft = try self.stack_fisher_t.?.valuesFlat(&self.ctx, allocator);
        return .{ .master_weights_s = master_s, .master_weights_t = master_t, .momentum_s = ms, .momentum_t = mt, .fisher_s = fs, .fisher_t = ft, .step = self.optimizer_step, .allocator = allocator };
    }

    pub fn setOptimizerState(
        self: *Self,
        master_weights_s: []const f32,
        master_weights_t: []const f32,
        momentum_s: []const f32,
        momentum_t: []const f32,
        fisher_s: []const f32,
        fisher_t: []const f32,
        step: u64,
    ) AccelError!void {
        try self.ensureStackPacked();
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, self.num_layers, per_layer) catch return AccelError.InvalidDimensions;
        if (master_weights_s.len != total or master_weights_t.len != total or momentum_s.len != total or momentum_t.len != total or fisher_s.len != total or fisher_t.len != total) return AccelError.InvalidDimensions;
        for (master_weights_s) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (master_weights_t) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum_s) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum_t) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (fisher_s) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;
        for (fisher_t) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;

        var new_master_s = try FutharkArray3DF32.newFromFlat(&self.ctx, master_weights_s, self.num_layers, half, cols);
        errdefer new_master_s.free(&self.ctx);
        var new_master_t = try FutharkArray3DF32.newFromFlat(&self.ctx, master_weights_t, self.num_layers, half, cols);
        errdefer new_master_t.free(&self.ctx);
        var new_ms = try FutharkArray3DF32.newFromFlat(&self.ctx, momentum_s, self.num_layers, half, cols);
        errdefer new_ms.free(&self.ctx);
        var new_mt = try FutharkArray3DF32.newFromFlat(&self.ctx, momentum_t, self.num_layers, half, cols);
        errdefer new_mt.free(&self.ctx);
        var new_fs = try FutharkArray3DF32.newFromFlat(&self.ctx, fisher_s, self.num_layers, half, cols);
        errdefer new_fs.free(&self.ctx);
        var new_ft = try FutharkArray3DF32.newFromFlat(&self.ctx, fisher_t, self.num_layers, half, cols);
        errdefer new_ft.free(&self.ctx);
        if (self.stack_master_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_t) |*old| old.free(&self.ctx);
        if (self.stack_momentum_s) |*old| old.free(&self.ctx);
        if (self.stack_momentum_t) |*old| old.free(&self.ctx);
        if (self.stack_fisher_s) |*old| old.free(&self.ctx);
        if (self.stack_fisher_t) |*old| old.free(&self.ctx);
        self.stack_master_weights_s = new_master_s;
        self.stack_master_weights_t = new_master_t;
        self.stack_momentum_s = new_ms;
        self.stack_momentum_t = new_mt;
        self.stack_fisher_s = new_fs;
        self.stack_fisher_t = new_ft;
        self.optimizer_step = step;
    }

    pub fn getStackDevicePtr(self: *Self, kind: StackArrayKind) AccelError!DeviceBufferF16 {
        if (!self.initialized) return AccelError.NullPointer;
        try self.ensureStackPacked();
        switch (kind) {
            .weights_s => {
                if (self.stack_weights_s) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .weights_t => {
                if (self.stack_weights_t) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            else => return AccelError.InvalidDimensions,
        }
    }

    pub fn getStackDevicePtrF32(self: *Self, kind: StackArrayKind) AccelError!DeviceBufferF32 {
        if (!self.initialized) return AccelError.NullPointer;
        try self.ensureStackPacked();
        switch (kind) {
            .master_weights_s => {
                if (self.stack_master_weights_s) |*state| return state.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .master_weights_t => {
                if (self.stack_master_weights_t) |*state| return state.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .momentum_s => {
                if (self.stack_momentum_s) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .momentum_t => {
                if (self.stack_momentum_t) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .fisher_s => {
                if (self.stack_fisher_s) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            .fisher_t => {
                if (self.stack_fisher_t) |*s| return s.deviceBuffer(&self.ctx);
                return AccelError.NullPointer;
            },
            else => return AccelError.InvalidDimensions,
        }
    }

    pub fn refreshForwardWeightsFromMaster(self: *Self) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        const master_s = self.stack_master_weights_s orelse return AccelError.NullPointer;
        const master_t = self.stack_master_weights_t orelse return AccelError.NullPointer;
        var forward_s: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &forward_s, master_s.arr) != 0 or forward_s == null) return AccelError.FutharkScaleWeightsFailed;
        var forward_t: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &forward_t, master_t.arr) != 0 or forward_t == null) {
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, forward_s);
            return AccelError.FutharkScaleWeightsFailed;
        }
        if (self.stack_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_weights_t) |*old| old.free(&self.ctx);
        const half = self.model_dim / 2;
        const cols = half + 1;
        self.stack_weights_s = .{ .arr = forward_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_weights_t = .{ .arr = forward_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_valid = true;
        self.layers_mirror_valid = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();

        try self.syncLayersFromStack();

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var current_arr: ?*futhark.struct_futhark_f16_2d = input.arr;
        const rows = input.rows;
        const cols = input.cols;

        var li: usize = 0;
        while (li < self.layers.len) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var next_arr: ?*futhark.struct_futhark_f16_2d = null;
            const result = futhark.futhark_entry_rsf_forward(
                self.ctx.ctx,
                &next_arr,
                current_arr,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (result != 0) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.FutharkForwardFailed;
            }
            if (next_arr == null) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.NullPointer;
            }

            if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
            current_arr = next_arr;
        }

        return FutharkArray2DF16{ .arr = current_arr, .rows = rows, .cols = cols };
    }

    pub fn stackForward(self: *Self, inputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null) return AccelError.NullPointer;
        if (inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_forward(
            self.ctx.ctx,
            &out,
            inputs.arr,
            sws.arr,
            swt.arr,
            clip_min_bits,
            clip_max_bits,
        );
        if (rc != 0 or out == null) {
            if (out) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
            return AccelError.FutharkForwardFailed;
        }
        return FutharkArray3DF16{ .arr = out, .dim0 = inputs.dim0, .dim1 = inputs.dim1, .dim2 = inputs.dim2 };
    }

    pub fn stackInverse(self: *Self, outputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (outputs.arr == null) return AccelError.NullPointer;
        if (outputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_inverse(
            self.ctx.ctx,
            &out,
            outputs.arr,
            sws.arr,
            swt.arr,
            clip_min_bits,
            clip_max_bits,
        );
        if (rc != 0 or out == null) {
            if (out) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
            return AccelError.FutharkBackwardFailed;
        }
        return FutharkArray3DF16{ .arr = out, .dim0 = outputs.dim0, .dim1 = outputs.dim1, .dim2 = outputs.dim2 };
    }

    pub fn fusedTrainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        sequence_lengths: []const usize,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        reconstruction_alpha: f32,
        forward_scale: f32,
        logdet_weight: f32,
    ) AccelError!FusedStepResult {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim or sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;
        if (!std.math.isFinite(learning_rate) or learning_rate < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(momentum_beta) or momentum_beta < 0.0 or momentum_beta >= 1.0 or !std.math.isFinite(fisher_gamma) or fisher_gamma < 0.0 or fisher_gamma >= 1.0 or !std.math.isFinite(epsilon) or epsilon <= 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(reconstruction_alpha) or !std.math.isFinite(forward_scale) or !std.math.isFinite(logdet_weight)) return AccelError.InvalidHyperparameter;

        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > inputs.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }
        var lengths_array = try FutharkArray1DI64.newFromSlice(&self.ctx, lengths_i64);
        defer lengths_array.free(&self.ctx);

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const smws = self.stack_master_weights_s orelse return AccelError.NullPointer;
        const smwt = self.stack_master_weights_t orelse return AccelError.NullPointer;
        const sms = self.stack_momentum_s orelse return AccelError.NullPointer;
        const smt = self.stack_momentum_t orelse return AccelError.NullPointer;
        const sfs = self.stack_fisher_s orelse return AccelError.NullPointer;
        const sft = self.stack_fisher_t orelse return AccelError.NullPointer;

        const clip_min_f32: f32 = @floatCast(self.clip_min);
        const clip_max_f32: f32 = @floatCast(self.clip_max);

        var final_outputs: ?*futhark.struct_futhark_f16_3d = null;
        var out_tup: ?*futhark.struct_futhark_opaque_tup10_fused_stack_step = null;

        const fwd_rc = futhark.futhark_entry_rsf_stack_forward(
            self.ctx.ctx,
            &final_outputs,
            inputs.arr,
            sws.arr,
            swt.arr,
            @bitCast(self.clip_min),
            @bitCast(self.clip_max),
        );
        if (fwd_rc != 0 or final_outputs == null) {
                if (final_outputs) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
            return AccelError.FutharkForwardFailed;
        }

        const bwd_rc = futhark.futhark_entry_rsf_stack_backward_sfd_fused(
            self.ctx.ctx,
            &out_tup,
            final_outputs,
            targets.arr,
            inputs.arr,
            lengths_array.arr,
            sws.arr,
            swt.arr,
            smws.arr,
            smwt.arr,
            sms.arr,
            smt.arr,
            sfs.arr,
            sft.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            @intCast(@min(self.optimizer_step +| 1, @as(u64, std.math.maxInt(i64)))),
            epsilon,
            clip_min_f32,
            clip_max_f32,
            reconstruction_alpha,
            forward_scale,
            logdet_weight,
        );
        if (final_outputs) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
        if (bwd_rc != 0 or out_tup == null) {
            if (out_tup) |t| _ = futhark.futhark_free_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(self.ctx.ctx, t);
            const es = futhark.futhark_context_get_error(self.ctx.ctx);
            defer freeFutharkError(es);
            if (es) |m| std.debug.print("[Futhark rsf_stack_backward_sfd_fused error] {s}\n", .{std.mem.span(m)});
            return AccelError.FutharkTrainingStepFailed;
        }

        var new_master_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_master_t: ?*futhark.struct_futhark_f32_3d = null;
        var new_ms: ?*futhark.struct_futhark_f32_3d = null;
        var new_mt: ?*futhark.struct_futhark_f32_3d = null;
        var new_fs: ?*futhark.struct_futhark_f32_3d = null;
        var new_ft: ?*futhark.struct_futhark_f32_3d = null;
        var delta: ?*futhark.struct_futhark_f16_3d = null;

        const tup = out_tup.?;
        const p0 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(self.ctx.ctx, &new_master_s, tup);
        const p1 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(self.ctx.ctx, &new_master_t, tup);
        const p2 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(self.ctx.ctx, &new_ms, tup);
        const p3 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_3(self.ctx.ctx, &new_mt, tup);
        const p4 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_4(self.ctx.ctx, &new_fs, tup);
        const p5 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_5(self.ctx.ctx, &new_ft, tup);
        const p6 = futhark.futhark_project_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_6(self.ctx.ctx, &delta, tup);

        if (p0 != 0 or p1 != 0 or p2 != 0 or p3 != 0 or p4 != 0 or p5 != 0 or p6 != 0 or
            new_master_s == null or new_master_t == null or new_ms == null or new_mt == null or
            new_fs == null or new_ft == null or delta == null)
        {
            if (new_master_s) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (new_master_t) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (new_ms) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (new_mt) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (new_fs) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (new_ft) |a| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, a);
            if (delta) |a| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, a);
            _ = futhark.futhark_free_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(self.ctx.ctx, tup);
            return AccelError.FutharkTrainingStepFailed;
        }
        var new_ws: ?*futhark.struct_futhark_f16_3d = null;
        var new_wt: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &new_ws, new_master_s) != 0 or new_ws == null or
            futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &new_wt, new_master_t) != 0 or new_wt == null)
        {
            if (new_ws) |a| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, a);
            if (new_wt) |a| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, a);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_master_s);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_master_t);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_ms);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_mt);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_fs);
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, new_ft);
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, delta);
            _ = futhark.futhark_free_opaque_tup10_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(self.ctx.ctx, tup);
            return AccelError.FutharkTrainingStepFailed;
        }

        self.freeStackArraysForReplacement();
        const half = self.model_dim / 2;
        const cols = half + 1;
        self.stack_weights_s = .{ .arr = new_ws, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_weights_t = .{ .arr = new_wt, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_master_weights_s = .{ .arr = new_master_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_master_weights_t = .{ .arr = new_master_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_momentum_s = .{ .arr = new_ms, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_momentum_t = .{ .arr = new_mt, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_fisher_s = .{ .arr = new_fs, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_fisher_t = .{ .arr = new_ft, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_valid = true;
        self.layers_mirror_valid = false;
        self.optimizer_step +|= 1;

        return FusedStepResult{
            .input_delta = .{ .arr = delta, .dim0 = inputs.dim0, .dim1 = inputs.dim1, .dim2 = inputs.dim2 },
            .pending = out_tup,
            .finalized = false,
            .scalars = .{ .loss = 0.0, .reconstruction_loss = 0.0, .logdet_mean = 0.0 },
        };
    }

    fn freeStackArraysForReplacement(self: *Self) void {
        if (self.stack_fisher_t) |*a| a.free(&self.ctx);
        self.stack_fisher_t = null;
        if (self.stack_fisher_s) |*a| a.free(&self.ctx);
        self.stack_fisher_s = null;
        if (self.stack_momentum_t) |*a| a.free(&self.ctx);
        self.stack_momentum_t = null;
        if (self.stack_momentum_s) |*a| a.free(&self.ctx);
        self.stack_momentum_s = null;
        if (self.stack_master_weights_t) |*a| a.free(&self.ctx);
        self.stack_master_weights_t = null;
        if (self.stack_master_weights_s) |*a| a.free(&self.ctx);
        self.stack_master_weights_s = null;
        if (self.stack_weights_t) |*a| a.free(&self.ctx);
        self.stack_weights_t = null;
        if (self.stack_weights_s) |*a| a.free(&self.ctx);
        self.stack_weights_s = null;
        self.stack_valid = false;
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (scale_factor == @as(f16, 0.0)) return AccelError.InvalidDimensions;
        try self.syncLayersFromStack();

        const scale_bits: u16 = @bitCast(scale_factor);
        for (self.layers) |*layer| {
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var new_ws: ?*futhark.struct_futhark_f16_2d = null;
            const result_s = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_ws,
                layer.weights_s.arr,
                scale_bits,
            );
            if (result_s != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_ws != null) {
                const old = layer.weights_s.arr;
                layer.weights_s.arr = new_ws;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }

            var new_wt: ?*futhark.struct_futhark_f16_2d = null;
            const result_t = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_wt,
                layer.weights_t.arr,
                scale_bits,
            );
            if (result_t != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_wt != null) {
                const old = layer.weights_t.arr;
                layer.weights_t.arr = new_wt;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }
        }
        self.markStackDirty();
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_s.free(&self.ctx);
        layer.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        self.markStackDirty();
        self.layers_mirror_valid = true;
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_t.free(&self.ctx);
        layer.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        self.markStackDirty();
        self.layers_mirror_valid = true;
    }

    pub fn readLayerWeightsFlat(self: *Self, layer_idx: usize, kind: WeightKind, allocator: std.mem.Allocator) AccelError![]f16 {
        try self.syncLayersFromStack();
        const layer = try self.layerPtr(layer_idx);
        return switch (kind) {
            .weights_s => readMatFlat(self, &layer.weights_s, allocator),
            .weights_t => readMatFlat(self, &layer.weights_t, allocator),
        };
    }

    pub fn getLayerDevicePtr(
        self: *Self,
        layer_idx: usize,
        kind: WeightKind,
    ) AccelError!DeviceBufferF16 {
        if (!self.initialized) return AccelError.NullPointer;
        try self.syncLayersFromStack();
        const layer = try self.layerPtr(layer_idx);
        const half = self.model_dim / 2;
        return switch (kind) {
            .weights_s => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_s), .count = half * (half + 1) },
            .weights_t => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_t), .count = half * (half + 1) },
        };
    }

    pub fn scaleLayerMatrix(self: *Self, layer_idx: usize, kind: WeightKind, scale_factor: f16) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        try self.syncLayersFromStack();
        const scale_f32: f32 = @floatCast(scale_factor);
        if (!std.math.isFinite(scale_f32) or scale_f32 < 0.0) return AccelError.InvalidDimensions;
        const layer = try self.layerPtr(layer_idx);
        const matrix: *FutharkArray2DF16 = switch (kind) {
            .weights_s => &layer.weights_s,
            .weights_t => &layer.weights_t,
        };
        if (matrix.arr == null) return AccelError.NullPointer;
        var scaled: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_scale_matrix_f16(
            self.ctx.ctx,
            &scaled,
            matrix.arr,
            @bitCast(scale_factor),
        );
        if (result != 0 or scaled == null) {
            if (scaled) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkScaleWeightsFailed;
        }
        const old = matrix.arr;
        matrix.arr = scaled;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
        self.markStackDirty();
    }

    fn readMatFlat(self: *Self, mat: *FutharkArray2DF16, allocator: std.mem.Allocator) AccelError![]f16 {
        const half = self.model_dim / 2;
        const cols = half + 1;
        if (mat.rows != half or mat.cols != cols) return AccelError.InvalidDimensions;
        return mat.valuesFlat(&self.ctx, allocator);
    }

    fn matrixSpectralNorm(data: []const f16, rows: usize, cols: usize, iterations: usize, allocator: std.mem.Allocator) AccelError!f32 {
        const left = allocator.alloc(f32, rows) catch return AccelError.AllocationFailed;
        defer allocator.free(left);
        const right = allocator.alloc(f32, cols) catch return AccelError.AllocationFailed;
        defer allocator.free(right);
        const initial = 1.0 / @sqrt(@as(f32, @floatFromInt(cols)));
        @memset(right, initial);
        const count = @max(@as(usize, 1), iterations);
        var iteration: usize = 0;
        while (iteration < count) : (iteration += 1) {
            @memset(left, 0.0);
            for (0..rows) |row| {
                for (0..cols) |col| {
                    left[row] += @as(f32, @floatCast(data[row * cols + col])) * right[col];
                }
            }
            var norm: f64 = 0.0;
            for (left) |value| norm += @as(f64, value) * @as(f64, value);
            const left_norm: f32 = @floatCast(@sqrt(norm));
            if (!std.math.isFinite(left_norm) or left_norm <= 1e-12) return 0.0;
            for (left) |*value| value.* /= left_norm;
            @memset(right, 0.0);
            for (0..rows) |row| {
                for (0..cols) |col| {
                    right[col] += @as(f32, @floatCast(data[row * cols + col])) * left[row];
                }
            }
            norm = 0.0;
            for (right) |value| norm += @as(f64, value) * @as(f64, value);
            const right_norm: f32 = @floatCast(@sqrt(norm));
            if (!std.math.isFinite(right_norm) or right_norm <= 1e-12) return 0.0;
            for (right) |*value| value.* /= right_norm;
        }
        var sigma: f64 = 0.0;
        for (0..rows) |row| {
            for (0..cols) |col| {
                sigma += @as(f64, left[row]) * @as(f64, @floatCast(data[row * cols + col])) * @as(f64, right[col]);
            }
        }
        return @floatCast(@abs(sigma));
    }

    /// Enforces the same coupling-layer spectral constraint as the CPU model.
    /// Optimizer state is intentionally untouched when the f16 forward mirror
    /// is rescaled.
    pub fn spectralNormalizeLayers(self: *Self, target: f32, iterations: usize) AccelError!void {
        if (!std.math.isFinite(target) or target <= 0.0 or iterations == 0) return AccelError.InvalidHyperparameter;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.syncLayersFromStack();
        const half = self.model_dim / 2;
        const cols = half + 1;
        for (self.layers, 0..) |*layer, layer_index| {
            const ws = try layer.weights_s.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(ws);
            const sigma_s = try matrixSpectralNorm(ws, half, cols, iterations, self.allocator);
            if (sigma_s > target) try self.scaleLayerMatrix(layer_index, .weights_s, @floatCast(target / sigma_s));
            const wt = try layer.weights_t.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(wt);
            const sigma_t = try matrixSpectralNorm(wt, half, cols, iterations, self.allocator);
            if (sigma_t > target) try self.scaleLayerMatrix(layer_index, .weights_t, @floatCast(target / sigma_t));
        }
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const minimum: f32 = @floatCast(clip_min_val);
        const maximum: f32 = @floatCast(clip_max_val);
        if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return AccelError.InvalidClipRange;
        if (minimum >= maximum or minimum < -20.0 or maximum > 20.0) return AccelError.InvalidClipRange;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, output.rows * output.cols) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) {
            result.deinit();
            return AccelError.FutharkSyncFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DI64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]i64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(i64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_i64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const EmbeddingAccelerator = struct {
    ctx: *FutharkContext,
    weight: FutharkArray2DF16,
    master_weight: FutharkArray2DF32,
    grad_weight: FutharkArray2DF16,
    vocab_size: usize,
    dim: usize,
    initialized: bool,
    allocator: std.mem.Allocator,
    optimizer_step: u64 = 0,
    scratch_token_buf: []i64 = &[_]i64{},
    scratch_token_cap: usize = 0,
    scratch_lengths_buf: []i64 = &[_]i64{},
    scratch_lengths_cap: usize = 0,
    scratch_positions_buf: []i64 = &[_]i64{},
    scratch_positions_cap: usize = 0,
    momentum_state: ?FutharkArray2DF32 = null,
    fisher_state: ?FutharkArray2DF32 = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;

        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const total = vocab_size * dim;
        const weight_data = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(weight_data);
        for (weight_data) |*v| {
            v.* = @floatCast((rnd.float(f32) - 0.5) * 0.02);
        }

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_data, vocab_size, dim);
        errdefer weight.free(ctx);
        const master_data = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer allocator.free(master_data);
        for (weight_data, master_data) |value, *master| master.* = @floatCast(value);
        var master_weight = try FutharkArray2DF32.newFromFlat(ctx, master_data, vocab_size, dim);
        errdefer master_weight.free(ctx);
        var grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, allocator);
        errdefer grad_weight.free(ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const scratch_token_buf = allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_token_buf);
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);
        const scratch_positions_buf = allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = ctx,
            .weight = weight,
            .master_weight = master_weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .allocator = allocator,
            .scratch_token_buf = scratch_token_buf,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = scratch_positions_buf,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        if (self.fisher_state) |*s| s.free(self.ctx);
        if (self.momentum_state) |*s| s.free(self.ctx);
        self.fisher_state = null;
        self.momentum_state = null;
        self.grad_weight.free(self.ctx);
        self.master_weight.free(self.ctx);
        self.weight.free(self.ctx);
        if (self.scratch_positions_buf.len > 0) self.allocator.free(self.scratch_positions_buf);
        if (self.scratch_lengths_buf.len > 0) self.allocator.free(self.scratch_lengths_buf);
        if (self.scratch_token_buf.len > 0) self.allocator.free(self.scratch_token_buf);
        self.initialized = false;
    }

    pub fn cloneDevice(self: *Self) AccelError!Self {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        const total_elements = self.vocab_size * self.dim;
        const flat = self.allocator.alloc(f16, total_elements) catch return AccelError.AllocationFailed;
        defer self.allocator.free(flat);
        const rc_values = futhark.futhark_values_f16_2d(self.ctx.ctx, self.weight.arr, @ptrCast(flat.ptr));
        if (rc_values != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        var weight_copy = FutharkArray2DF16.newFromFlat(self.ctx, flat, self.vocab_size, self.dim) catch return AccelError.FutharkArrayNewFailed;
        errdefer weight_copy.free(self.ctx);
        const master_flat = try self.master_weight.valuesFlat(self.ctx, self.allocator);
        defer self.allocator.free(master_flat);
        var master_copy = try FutharkArray2DF32.newFromFlat(self.ctx, master_flat, self.vocab_size, self.dim);
        errdefer master_copy.free(self.ctx);
        var grad_copy = FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator) catch return AccelError.FutharkArrayNewFailed;
        errdefer grad_copy.free(self.ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const st = self.allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer self.allocator.free(st);
        const sl = self.allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer self.allocator.free(sl);
        const sp = self.allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = self.ctx,
            .weight = weight_copy,
            .master_weight = master_copy,
            .grad_weight = grad_copy,
            .vocab_size = self.vocab_size,
            .dim = self.dim,
            .initialized = true,
            .allocator = self.allocator,
            .scratch_token_buf = st,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = sl,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = sp,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn initWithWeights(ctx: *FutharkContext, allocator: std.mem.Allocator, vocab_size: usize, dim: usize, weight_f16: []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        if (weight_f16.len != vocab_size * dim) return AccelError.InvalidDimensions;

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_f16, vocab_size, dim);
        errdefer weight.free(ctx);
        const master_data = allocator.alloc(f32, weight_f16.len) catch return AccelError.AllocationFailed;
        defer allocator.free(master_data);
        for (weight_f16, master_data) |value, *master| master.* = @floatCast(value);
        var master_weight = try FutharkArray2DF32.newFromFlat(ctx, master_data, vocab_size, dim);
        errdefer master_weight.free(ctx);
        var grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, allocator);
        errdefer grad_weight.free(ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const scratch_token_buf = allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_token_buf);
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);
        const scratch_positions_buf = allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = ctx,
            .weight = weight,
            .master_weight = master_weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .allocator = allocator,
            .scratch_token_buf = scratch_token_buf,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = scratch_positions_buf,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn forwardPadded(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        sequence_length: usize,
    ) AccelError!FutharkArray3DF16 {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (sequence_lengths.len == 0 or sequence_length == 0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, sequence_lengths.len, sequence_length) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > sequence_length) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        const positions_i64 = if (sequence_length <= self.scratch_positions_cap) self.scratch_positions_buf[0..sequence_length] else (self.allocator.alloc(i64, sequence_length) catch return AccelError.AllocationFailed);
        defer if (sequence_length > self.scratch_positions_cap) self.allocator.free(positions_i64);
        for (positions_i64, 0..) |*position, index| {
            position.* = @intCast(index);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);
        var position_array = try FutharkArray1DI64.newFromSlice(self.ctx, positions_i64);
        defer position_array.free(self.ctx);

        var output: ?*futhark.struct_futhark_f16_3d = null;
        const result = futhark.futhark_entry_embedding_forward_padded(
            self.ctx.ctx,
            &output,
            token_array.arr,
            length_array.arr,
            position_array.arr,
            self.weight.arr,
        );
        if (result != 0 or output == null) {
            if (output) |value| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, value);
            return AccelError.FutharkForwardFailed;
        }
        return FutharkArray3DF16{
            .arr = output,
            .dim0 = sequence_lengths.len,
            .dim1 = sequence_length,
            .dim2 = self.dim,
        };
    }

    pub fn backwardPaddedAccumulate(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        gradient_output: *FutharkArray3DF16,
        clip_norm: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (!std.math.isFinite(clip_norm) or clip_norm < 0.0) return AccelError.InvalidHyperparameter;
        if (gradient_output.arr == null or gradient_output.dim2 != self.dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != gradient_output.dim0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, gradient_output.dim0, gradient_output.dim1) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > gradient_output.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);

        var new_gradient: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_embedding_backward_padded(
            self.ctx.ctx,
            &new_gradient,
            token_array.arr,
            length_array.arr,
            gradient_output.arr,
            self.grad_weight.arr,
            clip_norm,
        );
        if (result != 0 or new_gradient == null) {
            if (new_gradient) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkBackwardFailed;
        }
        const old_gradient = self.grad_weight.arr;
        self.grad_weight.arr = new_gradient;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_gradient);
    }

    pub fn getGradientDevicePtr(self: *Self) AccelError!DeviceBufferF16 {
        if (!self.initialized or self.grad_weight.arr == null) return AccelError.NullPointer;
        const count = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        return .{
            .ptr = try self.ctx.getDataPointer(&self.grad_weight),
            .count = count,
        };
    }

    pub fn getWeightDevicePtr(self: *Self) AccelError!DeviceBufferF16 {
        if (!self.initialized or self.weight.arr == null) return AccelError.NullPointer;
        const count = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        return .{
            .ptr = try self.ctx.getDataPointer(&self.weight),
            .count = count,
        };
    }

    pub fn scaleGradient(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null or self.grad_weight.arr == null) return AccelError.NullPointer;
        const scale_f32: f32 = @floatCast(scale_factor);
        if (!std.math.isFinite(scale_f32) or scale_f32 < 0.0) return AccelError.InvalidDimensions;
        var scaled: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_scale_matrix_f16(
            self.ctx.ctx,
            &scaled,
            self.grad_weight.arr,
            @bitCast(scale_factor),
        );
        if (result != 0 or scaled == null) {
            if (scaled) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkScaleWeightsFailed;
        }
        const old = self.grad_weight.arr;
        self.grad_weight.arr = scaled;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }

    pub fn forward(self: *Self, tokens: []const u32) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            if (@as(usize, tokens[i]) >= self.vocab_size) return AccelError.InvalidToken;
            t.* = @intCast(tokens[i]);
        }

        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var out: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_forward(
            self.ctx.ctx,
            &out,
            tok_arr.arr,
            self.weight.arr,
        );

        if (rc != 0 or out == null) return AccelError.FutharkForwardFailed;
        return FutharkArray2DF16{ .arr = out, .rows = tokens.len, .cols = self.dim };
    }

    pub fn backwardAccumulate(self: *Self, tokens: []const u32, grad_output: *FutharkArray2DF16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            if (@as(usize, tokens[i]) >= self.vocab_size) return AccelError.InvalidToken;
            t.* = @intCast(tokens[i]);
        }
        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);
        var new_grad: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_backward(
            self.ctx.ctx,
            &new_grad,
            tok_arr.arr,
            grad_output.arr,
            self.grad_weight.arr,
        );
        if (rc != 0 or new_grad == null) return AccelError.FutharkBackwardFailed;
        const old_grad = self.grad_weight.arr;
        self.grad_weight.arr = new_grad;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_grad);
    }

    pub fn ensureFisherState(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.momentum_state != null and self.fisher_state != null) return;
        if (self.momentum_state == null) {
            self.momentum_state = try FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator);
        }
        if (self.fisher_state == null) {
            self.fisher_state = try FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator);
        }
    }

    pub fn applyUpdateFusedSFD(
        self: *Self,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (!std.math.isFinite(learning_rate) or learning_rate < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(momentum_beta) or momentum_beta < 0.0 or momentum_beta >= 1.0 or
            !std.math.isFinite(fisher_gamma) or fisher_gamma < 0.0 or fisher_gamma >= 1.0 or
            !std.math.isFinite(epsilon) or epsilon <= 0.0) return AccelError.InvalidHyperparameter;
        try self.ensureFisherState();
        const ms = &(self.momentum_state orelse return AccelError.NullPointer);
        const fs = &(self.fisher_state orelse return AccelError.NullPointer);

        var tuple: ?*futhark.struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 = null;
        const rc = futhark.futhark_entry_embedding_update_sfd_master(
            self.ctx.ctx,
            &tuple,
            self.master_weight.arr,
            self.grad_weight.arr,
            ms.arr,
            fs.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            @intCast(@min(self.optimizer_step +| 1, @as(u64, std.math.maxInt(i64)))),
            epsilon,
        );
        if (rc != 0 or tuple == null) return AccelError.FutharkSFDUpdateFailed;
        var new_master: ?*futhark.struct_futhark_f32_2d = null;
        var new_momentum: ?*futhark.struct_futhark_f32_2d = null;
        var new_fisher: ?*futhark.struct_futhark_f32_2d = null;
        const p0 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_0(self.ctx.ctx, &new_master, tuple);
        const p1 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_1(self.ctx.ctx, &new_momentum, tuple);
        const p2 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_2(self.ctx.ctx, &new_fisher, tuple);
        _ = futhark.futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(self.ctx.ctx, tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or new_master == null or new_momentum == null or new_fisher == null) {
            if (new_master) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            if (new_momentum) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            if (new_fisher) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_forward: ?*futhark.struct_futhark_f16_2d = null;
        if (futhark.futhark_entry_master_weights_to_f16_2d(self.ctx.ctx, &new_forward, new_master) != 0 or new_forward == null) {
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_master);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_momentum);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_fisher);
            if (new_forward) |array| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, array);
            return AccelError.FutharkSFDUpdateFailed;
        }

        const zeroed_grad = FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator) catch |err| {
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, new_forward);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_master);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_momentum);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_fisher);
            return err;
        };
        self.weight.free(self.ctx);
        self.master_weight.free(self.ctx);
        ms.free(self.ctx);
        fs.free(self.ctx);
        self.weight.arr = new_forward;
        self.weight.rows = self.vocab_size;
        self.weight.cols = self.dim;
        self.master_weight.arr = new_master;
        self.master_weight.rows = self.vocab_size;
        self.master_weight.cols = self.dim;
        ms.* = .{ .arr = new_momentum, .rows = self.vocab_size, .cols = self.dim };
        fs.* = .{ .arr = new_fisher, .rows = self.vocab_size, .cols = self.dim };
        self.optimizer_step +|= 1;

        self.grad_weight.free(self.ctx);
        self.grad_weight = zeroed_grad;
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!EmbeddingOptimizerState {
        try self.ensureFisherState();
        const master = try self.master_weight.valuesFlat(self.ctx, allocator);
        errdefer allocator.free(master);
        const momentum = try self.momentum_state.?.valuesFlat(self.ctx, allocator);
        errdefer allocator.free(momentum);
        const fisher = try self.fisher_state.?.valuesFlat(self.ctx, allocator);
        return .{ .master_weights = master, .momentum = momentum, .fisher = fisher, .step = self.optimizer_step, .allocator = allocator };
    }

    pub fn setOptimizerState(self: *Self, master_weights: []const f32, momentum: []const f32, fisher: []const f32, step: u64) AccelError!void {
        const total = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        if (master_weights.len != total or momentum.len != total or fisher.len != total) return AccelError.InvalidDimensions;
        for (master_weights) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (fisher) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;
        var new_master = try FutharkArray2DF32.newFromFlat(self.ctx, master_weights, self.vocab_size, self.dim);
        errdefer new_master.free(self.ctx);
        var new_m = try FutharkArray2DF32.newFromFlat(self.ctx, momentum, self.vocab_size, self.dim);
        errdefer new_m.free(self.ctx);
        var new_f = try FutharkArray2DF32.newFromFlat(self.ctx, fisher, self.vocab_size, self.dim);
        errdefer new_f.free(self.ctx);
        self.master_weight.free(self.ctx);
        if (self.momentum_state) |*old| old.free(self.ctx);
        if (self.fisher_state) |*old| old.free(self.ctx);
        self.master_weight = new_master;
        self.momentum_state = new_m;
        self.fisher_state = new_f;
        self.optimizer_step = step;
    }

    pub fn sourceSumSquares(self: *Self) AccelError!f32 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weight.arr == null) return AccelError.NullPointer;
        var total: f32 = 0.0;
        const rc = futhark.futhark_entry_embedding_sum_squares(
            self.ctx.ctx,
            &total,
            self.weight.arr,
        );
        if (rc != 0) return AccelError.FutharkComputeLossFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        if (!std.math.isFinite(total)) return 0.0;
        return total;
    }

    pub fn sourceRootMeanSquare(self: *Self) AccelError!f32 {
        const total = try self.sourceSumSquares();
        const count = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        if (count == 0) return 0.0;
        const mean = total / @as(f32, @floatFromInt(count));
        if (!std.math.isFinite(mean) or mean <= 0.0) return 0.0;
        return @sqrt(mean);
    }

    pub fn readGradFlat(self: *Self, allocator: std.mem.Allocator) AccelError![]f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.grad_weight.arr == null) return AccelError.NullPointer;
        const total = self.vocab_size * self.dim;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, self.grad_weight.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn setGradFromFlat(self: *Self, data: []const f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (data.len != self.vocab_size * self.dim) return AccelError.InvalidDimensions;
        const new_grad = try FutharkArray2DF16.newFromFlat(self.ctx, data, self.vocab_size, self.dim);
        const old = self.grad_weight.arr;
        self.grad_weight.arr = new_grad.arr;
        self.grad_weight.rows = new_grad.rows;
        self.grad_weight.cols = new_grad.cols;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }

    pub fn spectralNormalize(
        self: *Self,
        u: *FutharkArray1DF32,
        v: *FutharkArray1DF32,
        power_iters: usize,
    ) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weight.arr == null or u.arr == null or v.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup3_spectral = null;
        const rc = futhark.futhark_entry_embedding_spectral_normalize(
            self.ctx.ctx,
            &out_tup,
            self.weight.arr,
            u.arr,
            v.arr,
            @intCast(power_iters),
        );
        if (rc != 0 or out_tup == null) return AccelError.FutharkForwardFailed;
        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        var new_u: ?*futhark.struct_futhark_f32_1d = null;
        var new_v: ?*futhark.struct_futhark_f32_1d = null;
        const p0 = futhark.futhark_project_opaque_tup3_spectral_0(self.ctx.ctx, &new_weight, out_tup);
        const p1 = futhark.futhark_project_opaque_tup3_spectral_1(self.ctx.ctx, &new_u, out_tup);
        const p2 = futhark.futhark_project_opaque_tup3_spectral_2(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup3_spectral(self.ctx.ctx, out_tup);
        if (p0 != 0 or new_weight == null or p1 != 0 or new_u == null or p2 != 0 or new_v == null) {
            if (new_weight) |w| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, w);
            if (new_u) |nu| _ = futhark.futhark_free_f32_1d(self.ctx.ctx, nu);
            if (new_v) |nv| _ = futhark.futhark_free_f32_1d(self.ctx.ctx, nv);
            return AccelError.FutharkForwardFailed;
        }
        var normalized_master: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_forward_weights_to_f32_2d(self.ctx.ctx, &normalized_master, new_weight) != 0 or normalized_master == null) {
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, new_weight);
            _ = futhark.futhark_free_f32_1d(self.ctx.ctx, new_u);
            _ = futhark.futhark_free_f32_1d(self.ctx.ctx, new_v);
            if (normalized_master) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            return AccelError.FutharkForwardFailed;
        }
        const old_w = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_w);
        self.master_weight.free(self.ctx);
        self.master_weight = .{ .arr = normalized_master, .rows = self.vocab_size, .cols = self.dim };
        const old_u = u.arr;
        u.arr = new_u;
        _ = futhark.futhark_free_f32_1d(self.ctx.ctx, old_u);
        const old_v = v.arr;
        v.arr = new_v;
        _ = futhark.futhark_free_f32_1d(self.ctx.ctx, old_v);
    }

    pub fn readWeightFlat(self: *Self, allocator: std.mem.Allocator) AccelError![]f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.weight.arr == null) return AccelError.NullPointer;
        const total = self.vocab_size * self.dim;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, self.weight.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn resetGrad(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const zeroed = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator);
        const old = self.grad_weight.arr;
        self.grad_weight.arr = zeroed.arr;
        self.grad_weight.rows = zeroed.rows;
        self.grad_weight.cols = zeroed.cols;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }
};

pub const GraphBatchEncodeResult = struct {
    hashes: []u64,
    re_a: []f32,
    im_a: []f32,
    re_b: []f32,
    im_b: []f32,
    edge_srcs: []i64,
    edge_tgts: []i64,
    node_count: usize,
    edge_count: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GraphBatchEncodeResult) void {
        self.allocator.free(self.hashes);
        self.allocator.free(self.re_a);
        self.allocator.free(self.im_a);
        self.allocator.free(self.re_b);
        self.allocator.free(self.im_b);
        self.allocator.free(self.edge_srcs);
        self.allocator.free(self.edge_tgts);
    }
};

pub const PackedBitmask = struct {
    words: []u64,
    num_nodes: usize,
    words_per_row: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PackedBitmask) void {
        self.allocator.free(self.words);
        self.words = &[_]u64{};
        self.num_nodes = 0;
        self.words_per_row = 0;
    }

    pub fn bit(self: *const PackedBitmask, row: usize, col: usize) bool {
        if (row >= self.num_nodes or col >= self.num_nodes) return false;
        const idx = row * self.words_per_row + col / 64;
        if (idx >= self.words.len) return false;
        return (self.words[idx] & (@as(u64, 1) << @intCast(col % 64))) != 0;
    }
};

pub const GraphBatchEncodeBitmaskResult = struct {
    hashes: []u64,
    re_a: []f32,
    im_a: []f32,
    re_b: []f32,
    im_b: []f32,
    bitmask: PackedBitmask,
    edge_srcs: []i64,
    edge_tgts: []i64,
    node_count: usize,
    edge_count: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GraphBatchEncodeBitmaskResult) void {
        self.allocator.free(self.hashes);
        self.allocator.free(self.re_a);
        self.allocator.free(self.im_a);
        self.allocator.free(self.re_b);
        self.allocator.free(self.im_b);
        self.bitmask.deinit();
        self.allocator.free(self.edge_srcs);
        self.allocator.free(self.edge_tgts);
    }
};

fn setGlobalBitmaskBit(words: []u64, words_per_row: usize, row: usize, col: usize) void {
    const idx = row * words_per_row + col / 64;
    if (idx < words.len) words[idx] |= @as(u64, 1) << @intCast(col % 64);
}

pub fn batchEncodeGraph(
    ctx: *FutharkContext,
    hashes: []const u64,
    seed: u64,
    allocator: std.mem.Allocator,
) AccelError!GraphBatchEncodeResult {
    if (ctx.ctx == null) return AccelError.NullPointer;
    ctx.mutex.lock();
    defer ctx.mutex.unlock();
    if (hashes.len == 0) return AccelError.InvalidDimensions;

    var acc_hashes = std.ArrayList(u64).init(allocator);
    errdefer acc_hashes.deinit();
    var acc_re_a = std.ArrayList(f32).init(allocator);
    errdefer acc_re_a.deinit();
    var acc_im_a = std.ArrayList(f32).init(allocator);
    errdefer acc_im_a.deinit();
    var acc_re_b = std.ArrayList(f32).init(allocator);
    errdefer acc_re_b.deinit();
    var acc_im_b = std.ArrayList(f32).init(allocator);
    errdefer acc_im_b.deinit();
    var acc_edge_srcs = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_srcs.deinit();
    var acc_edge_tgts = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_tgts.deinit();

    acc_hashes.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    const edge_capacity = std.math.mul(usize, hashes.len, 3) catch return AccelError.InvalidDimensions;
    acc_edge_srcs.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;
    acc_edge_tgts.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;

    var offset: usize = 0;
    while (offset < hashes.len) {
        const chunk_end = hashes.len;
        const chunk = hashes[offset..chunk_end];
        const chunk_n = chunk.len;
        const chunk_ne = std.math.mul(usize, chunk_n, 3) catch return AccelError.InvalidDimensions;

        var in_chunk = FutharkArray1DU64.newFromSlice(ctx, chunk) catch |err| {
            std.debug.print("[batchEncodeGraph] chunk offset={d} upload failed: {}\n", .{ offset, err });
            return err;
        };
        defer in_chunk.free(ctx);

        var out_tup: ?*futhark.struct_futhark_opaque_tup7_graph_encode = null;
        defer {
            if (out_tup) |p| {
                _ = futhark.futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(ctx.ctx, p);
            }
        }

        var out_ids: ?*futhark.struct_futhark_u64_1d = null;
        var out_re_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_re_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_edge_srcs: ?*futhark.struct_futhark_i64_1d = null;
        var out_edge_tgts: ?*futhark.struct_futhark_i64_1d = null;

        defer {
            if (out_ids) |p| _ = futhark.futhark_free_u64_1d(ctx.ctx, p);
            if (out_re_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_re_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_edge_srcs) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
            if (out_edge_tgts) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
        }

        const rc = futhark.futhark_entry_graph_batch_encode(
            ctx.ctx,
            &out_tup,
            in_chunk.arr,
            seed,
        );

        if (rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            defer freeFutharkError(ctx_err);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark entry error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] Futhark entry failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, rc });
            }
            return AccelError.FutharkForwardFailed;
        }

        const sync_rc = futhark.futhark_context_sync(ctx.ctx);
        if (sync_rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            defer freeFutharkError(ctx_err);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark sync error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] futhark_context_sync failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, sync_rc });
            }
            return AccelError.FutharkSyncFailed;
        }

        const tup = out_tup orelse {
            std.debug.print("[batchEncodeGraph] out_tup null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        };
        const proj0 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(ctx.ctx, &out_ids, tup);
        const proj1 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(ctx.ctx, &out_re_a, tup);
        const proj2 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(ctx.ctx, &out_im_a, tup);
        const proj3 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(ctx.ctx, &out_re_b, tup);
        const proj4 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(ctx.ctx, &out_im_b, tup);
        const proj5 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(ctx.ctx, &out_edge_srcs, tup);
        const proj6 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(ctx.ctx, &out_edge_tgts, tup);
        if (proj0 != 0 or proj1 != 0 or proj2 != 0 or proj3 != 0 or proj4 != 0 or proj5 != 0 or proj6 != 0) {
            std.debug.print("[batchEncodeGraph] projection failed at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.FutharkForwardFailed;
        }

        if (out_ids == null) {
            std.debug.print("[batchEncodeGraph] out_ids null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_a == null) {
            std.debug.print("[batchEncodeGraph] out_re_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_a == null) {
            std.debug.print("[batchEncodeGraph] out_im_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_b == null) {
            std.debug.print("[batchEncodeGraph] out_re_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_b == null) {
            std.debug.print("[batchEncodeGraph] out_im_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_edge_srcs == null) {
            std.debug.print("[batchEncodeGraph] out_edge_srcs null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }
        if (out_edge_tgts == null) {
            std.debug.print("[batchEncodeGraph] out_edge_tgts null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }

        const ids_buf = allocator.alloc(u64, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(ids_buf);
        const re_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_a_buf);
        const im_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_a_buf);
        const re_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_b_buf);
        const im_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_b_buf);
        const edge_src_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_src_buf);
        const edge_tgt_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_tgt_buf);

        if (futhark.futhark_values_u64_1d(ctx.ctx, out_ids, ids_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_a, re_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_a, im_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_b, re_b_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_b, im_b_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_srcs, edge_src_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_tgts, edge_tgt_buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        // All asynchronous D2H transfers complete at one synchronization point.
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        acc_hashes.appendSlice(ids_buf) catch return AccelError.AllocationFailed;
        acc_re_a.appendSlice(re_a_buf) catch return AccelError.AllocationFailed;
        acc_im_a.appendSlice(im_a_buf) catch return AccelError.AllocationFailed;
        acc_re_b.appendSlice(re_b_buf) catch return AccelError.AllocationFailed;
        acc_im_b.appendSlice(im_b_buf) catch return AccelError.AllocationFailed;
        for (edge_src_buf) |value| acc_edge_srcs.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;
        for (edge_tgt_buf) |value| acc_edge_tgts.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;

        offset = chunk_end;
    }

    const total_n = acc_hashes.items.len;
    const total_ne = acc_edge_srcs.items.len;

    const owned_hashes = acc_hashes.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_hashes);
    const owned_re_a = acc_re_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_a);
    const owned_im_a = acc_im_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_a);
    const owned_re_b = acc_re_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_b);
    const owned_im_b = acc_im_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_b);
    const owned_edge_srcs = acc_edge_srcs.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_edge_srcs);
    const owned_edge_tgts = acc_edge_tgts.toOwnedSlice() catch return AccelError.AllocationFailed;

    return GraphBatchEncodeResult{
        .hashes = owned_hashes,
        .re_a = owned_re_a,
        .im_a = owned_im_a,
        .re_b = owned_re_b,
        .im_b = owned_im_b,
        .edge_srcs = owned_edge_srcs,
        .edge_tgts = owned_edge_tgts,
        .node_count = total_n,
        .edge_count = total_ne,
        .allocator = allocator,
    };
}

pub fn batchEncodeGraphBitmask(
    ctx: *FutharkContext,
    hashes: []const u64,
    seed: u64,
    allocator: std.mem.Allocator,
) AccelError!GraphBatchEncodeBitmaskResult {
    if (ctx.ctx == null) return AccelError.NullPointer;
    ctx.mutex.lock();
    defer ctx.mutex.unlock();
    if (hashes.len == 0) return AccelError.InvalidDimensions;

    const total_nodes = hashes.len;
    const words_per_row = (total_nodes - 1) / 64 + 1;
    const total_words = std.math.mul(usize, total_nodes, words_per_row) catch return AccelError.AllocationFailed;
    const global_words = allocator.alloc(u64, total_words) catch return AccelError.AllocationFailed;
    errdefer allocator.free(global_words);
    @memset(global_words, 0);

    var acc_hashes = std.ArrayList(u64).init(allocator);
    errdefer acc_hashes.deinit();
    var acc_re_a = std.ArrayList(f32).init(allocator);
    errdefer acc_re_a.deinit();
    var acc_im_a = std.ArrayList(f32).init(allocator);
    errdefer acc_im_a.deinit();
    var acc_re_b = std.ArrayList(f32).init(allocator);
    errdefer acc_re_b.deinit();
    var acc_im_b = std.ArrayList(f32).init(allocator);
    errdefer acc_im_b.deinit();
    var acc_edge_srcs = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_srcs.deinit();
    var acc_edge_tgts = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_tgts.deinit();

    acc_hashes.ensureTotalCapacity(total_nodes) catch return AccelError.AllocationFailed;
    acc_re_a.ensureTotalCapacity(total_nodes) catch return AccelError.AllocationFailed;
    acc_im_a.ensureTotalCapacity(total_nodes) catch return AccelError.AllocationFailed;
    acc_re_b.ensureTotalCapacity(total_nodes) catch return AccelError.AllocationFailed;
    acc_im_b.ensureTotalCapacity(total_nodes) catch return AccelError.AllocationFailed;
    const edge_capacity = std.math.mul(usize, total_nodes, 3) catch return AccelError.InvalidDimensions;
    acc_edge_srcs.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;
    acc_edge_tgts.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;

    var offset: usize = 0;
    while (offset < total_nodes) {
        const chunk_end = total_nodes;
        const chunk = hashes[offset..chunk_end];
        const chunk_n = chunk.len;
        const chunk_ne = std.math.mul(usize, chunk_n, 3) catch return AccelError.InvalidDimensions;
        const chunk_words_per_row = (chunk_n - 1) / 64 + 1;

        var in_chunk = FutharkArray1DU64.newFromSlice(ctx, chunk) catch |err| {
            std.debug.print("[batchEncodeGraphBitmask] chunk offset={d} upload failed: {}\n", .{ offset, err });
            return err;
        };
        defer in_chunk.free(ctx);

        var out_tup: ?*futhark.struct_futhark_opaque_tup8_graph_encode_bitmask = null;
        defer {
            if (out_tup) |p| {
                _ = futhark.futhark_free_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64(ctx.ctx, p);
            }
        }

        var out_ids: ?*futhark.struct_futhark_u64_1d = null;
        var out_re_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_re_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_bitmask: ?*futhark.struct_futhark_u64_2d = null;
        var out_edge_srcs: ?*futhark.struct_futhark_i64_1d = null;
        var out_edge_tgts: ?*futhark.struct_futhark_i64_1d = null;

        defer {
            if (out_ids) |p| _ = futhark.futhark_free_u64_1d(ctx.ctx, p);
            if (out_re_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_re_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_bitmask) |p| _ = futhark.futhark_free_u64_2d(ctx.ctx, p);
            if (out_edge_srcs) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
            if (out_edge_tgts) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
        }

        const rc = futhark.futhark_entry_graph_batch_encode_bitmask(
            ctx.ctx,
            &out_tup,
            in_chunk.arr,
            seed,
        );
        if (rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            defer freeFutharkError(ctx_err);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraphBitmask] Futhark entry error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraphBitmask] Futhark entry failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, rc });
            }
            return AccelError.FutharkForwardFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;

        const tup = out_tup orelse return AccelError.NullPointer;
        const q0 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_0(ctx.ctx, &out_ids, tup);
        const q1 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_1(ctx.ctx, &out_re_a, tup);
        const q2 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_2(ctx.ctx, &out_im_a, tup);
        const q3 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_3(ctx.ctx, &out_re_b, tup);
        const q4 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_4(ctx.ctx, &out_im_b, tup);
        const q5 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_5(ctx.ctx, &out_bitmask, tup);
        const q6 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_6(ctx.ctx, &out_edge_srcs, tup);
        const q7 = futhark.futhark_project_opaque_tup8_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr2d_u64_arr1d_i64_arr1d_i64_7(ctx.ctx, &out_edge_tgts, tup);
        if (q0 != 0 or q1 != 0 or q2 != 0 or q3 != 0 or q4 != 0 or q5 != 0 or q6 != 0 or q7 != 0) return AccelError.FutharkForwardFailed;
        if (out_ids == null or out_re_a == null or out_im_a == null or out_re_b == null or out_im_b == null or
            out_bitmask == null or out_edge_srcs == null or out_edge_tgts == null) return AccelError.NullPointer;

        const ids_buf = allocator.alloc(u64, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(ids_buf);
        const re_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_a_buf);
        const im_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_a_buf);
        const re_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_b_buf);
        const im_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_b_buf);
        const chunk_word_count = std.math.mul(usize, chunk_n, chunk_words_per_row) catch return AccelError.InvalidDimensions;
        const bitmask_buf = allocator.alloc(u64, chunk_word_count) catch return AccelError.AllocationFailed;
        defer allocator.free(bitmask_buf);
        const edge_src_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_src_buf);
        const edge_tgt_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_tgt_buf);

        if (futhark.futhark_values_u64_1d(ctx.ctx, out_ids, ids_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_a, re_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_a, im_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_b, re_b_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_b, im_b_buf.ptr) != 0 or
            futhark.futhark_values_u64_2d(ctx.ctx, out_bitmask, bitmask_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_srcs, edge_src_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_tgts, edge_tgt_buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        acc_hashes.appendSlice(ids_buf) catch return AccelError.AllocationFailed;
        acc_re_a.appendSlice(re_a_buf) catch return AccelError.AllocationFailed;
        acc_im_a.appendSlice(im_a_buf) catch return AccelError.AllocationFailed;
        acc_re_b.appendSlice(re_b_buf) catch return AccelError.AllocationFailed;
        acc_im_b.appendSlice(im_b_buf) catch return AccelError.AllocationFailed;
        var bitmask_row: usize = 0;
        while (bitmask_row < chunk_n) : (bitmask_row += 1) {
            const row_words = bitmask_buf[bitmask_row * chunk_words_per_row .. (bitmask_row + 1) * chunk_words_per_row];
            for (row_words, 0..) |word, word_index| {
                var remaining = word;
                while (remaining != 0) {
                    const bit_index: usize = @intCast(@ctz(remaining));
                    setGlobalBitmaskBit(global_words, words_per_row, offset + bitmask_row, offset + word_index * 64 + bit_index);
                    remaining &= remaining - 1;
                }
            }
        }
        for (edge_src_buf) |value| acc_edge_srcs.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;
        for (edge_tgt_buf) |value| acc_edge_tgts.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;

        offset = chunk_end;
    }

    const total_ne = acc_edge_srcs.items.len;

    const owned_hashes = acc_hashes.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_hashes);
    const owned_re_a = acc_re_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_a);
    const owned_im_a = acc_im_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_a);
    const owned_re_b = acc_re_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_b);
    const owned_im_b = acc_im_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_b);
    const owned_edge_srcs = acc_edge_srcs.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_edge_srcs);
    const owned_edge_tgts = acc_edge_tgts.toOwnedSlice() catch return AccelError.AllocationFailed;

    return GraphBatchEncodeBitmaskResult{
        .hashes = owned_hashes,
        .re_a = owned_re_a,
        .im_a = owned_im_a,
        .re_b = owned_re_b,
        .im_b = owned_im_b,
        .bitmask = .{
            .words = global_words,
            .num_nodes = total_nodes,
            .words_per_row = words_per_row,
            .allocator = allocator,
        },
        .edge_srcs = owned_edge_srcs,
        .edge_tgts = owned_edge_tgts,
        .node_count = total_nodes,
        .edge_count = total_ne,
        .allocator = allocator,
    };
}

pub fn packBitmaskEdges(
    ctx: *FutharkContext,
    srcs: []const i64,
    tgts: []const i64,
    num_nodes: usize,
    symmetric: bool,
    allocator: std.mem.Allocator,
) AccelError!PackedBitmask {
    if (ctx.ctx == null) return AccelError.NullPointer;
    if (num_nodes == 0) return AccelError.InvalidDimensions;
    if (srcs.len != tgts.len) return AccelError.InvalidDimensions;
    const words_per_row = (num_nodes + 63) / 64;
    const words = allocator.alloc(u64, num_nodes * words_per_row) catch return AccelError.AllocationFailed;
    errdefer allocator.free(words);
    if (srcs.len == 0) {
        @memset(words, 0);
        return PackedBitmask{ .words = words, .num_nodes = num_nodes, .words_per_row = words_per_row, .allocator = allocator };
    }

    var src_arr = try FutharkArray1DI64.newFromSlice(ctx, srcs);
    defer src_arr.free(ctx);
    var tgt_arr = try FutharkArray1DI64.newFromSlice(ctx, tgts);
    defer tgt_arr.free(ctx);

    var out: ?*futhark.struct_futhark_u64_2d = null;
    const rc = futhark.futhark_entry_bitmask_pack_edges(
        ctx.ctx,
        &out,
        src_arr.arr,
        tgt_arr.arr,
        @intCast(num_nodes),
        if (symmetric) @as(i64, 1) else @as(i64, 0),
    );
    if (rc != 0 or out == null) {
        if (out) |o| _ = futhark.futhark_free_u64_2d(ctx.ctx, o);
        return AccelError.FutharkForwardFailed;
    }
    defer _ = futhark.futhark_free_u64_2d(ctx.ctx, out);
    if (futhark.futhark_values_u64_2d(ctx.ctx, out, words.ptr) != 0) return AccelError.FutharkValuesFailed;
    if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
    return PackedBitmask{ .words = words, .num_nodes = num_nodes, .words_per_row = words_per_row, .allocator = allocator };
}

pub fn bitmaskPropagateSignalHost(
    ctx: *FutharkContext,
    bitmask: *const PackedBitmask,
    signal: []const f32,
    steps: usize,
    decay: f32,
    allocator: std.mem.Allocator,
) AccelError![]f32 {
    if (ctx.ctx == null) return AccelError.NullPointer;
    if (bitmask.num_nodes == 0 or bitmask.words_per_row == 0) return AccelError.InvalidDimensions;
    if (signal.len != bitmask.num_nodes) return AccelError.InvalidDimensions;
    if (!std.math.isFinite(decay) or decay < 0.0 or decay > 1.0) return AccelError.InvalidDimensions;

    var adjacency = try FutharkArray2DU64.newFromFlat(ctx, bitmask.words, bitmask.num_nodes, bitmask.words_per_row);
    defer adjacency.free(ctx);
    var signal_arr = try FutharkArray1DF32.newFromSlice(ctx, signal);
    defer signal_arr.free(ctx);

    var out: ?*futhark.struct_futhark_f32_1d = null;
    const steps_i64: i64 = @intCast(steps);
    const rc = if (steps_i64 <= 1)
        futhark.futhark_entry_bitmask_propagate_signal(ctx.ctx, &out, adjacency.arr, signal_arr.arr)
    else
        futhark.futhark_entry_bitmask_propagate_steps(ctx.ctx, &out, adjacency.arr, signal_arr.arr, steps_i64, decay);
    if (rc != 0 or out == null) {
        if (out) |o| _ = futhark.futhark_free_f32_1d(ctx.ctx, o);
        return AccelError.FutharkForwardFailed;
    }
    defer _ = futhark.futhark_free_f32_1d(ctx.ctx, out);

    const result = allocator.alloc(f32, bitmask.num_nodes) catch return AccelError.AllocationFailed;
    errdefer allocator.free(result);
    if (futhark.futhark_values_f32_1d(ctx.ctx, out, result.ptr) != 0) return AccelError.FutharkValuesFailed;
    if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
    return result;
}
