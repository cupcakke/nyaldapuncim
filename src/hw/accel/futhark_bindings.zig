const _build_gpu_enabled: bool = blk: {
    const opts = @import("build_options");
    if (@hasDecl(opts, "gpu_acceleration")) break :blk opts.gpu_acceleration;
    break :blk false;
};

pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f16_1d = opaque {};
pub const struct_futhark_f16_2d = opaque {};
pub const struct_futhark_f16_3d = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};
pub const struct_futhark_i64_2d = opaque {};
pub const struct_futhark_opaque_tup3_grad_full = opaque {};
pub const struct_futhark_opaque_tup5_embedding_spectral = opaque {};
pub const struct_futhark_opaque_tup3_stack_spectral = opaque {};
pub const struct_futhark_opaque_tup7_graph_encode = opaque {};
pub const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 = opaque {};
pub const struct_futhark_opaque_tup6_fused_stack_gradients = opaque {};
pub const struct_futhark_opaque_tup3_stack_sfd = opaque {};

pub const GpuConfigurationError = error{GpuAccelerationDisabled};

pub const gpu_default_group_size: c_int = 256;
pub const gpu_default_num_groups: c_int = 4096;
pub const gpu_default_tile_size: c_int = 128;
pub const gpu_arch_sm: c_int = 100;

pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_cache_file(cfg: ?*struct_futhark_context_config, path: [*:0]const u8) void;

pub fn configureGpuContext(
    cfg: ?*struct_futhark_context_config,
    cache_file: ?[*:0]const u8,
) GpuConfigurationError!void {
    if (comptime _build_gpu_enabled) {
        futhark_context_config_set_device(cfg, "");
        futhark_context_config_set_default_group_size(cfg, gpu_default_group_size);
        futhark_context_config_set_default_num_groups(cfg, gpu_default_num_groups);
        futhark_context_config_set_default_tile_size(cfg, gpu_default_tile_size);
        if (cache_file) |path| futhark_context_config_set_cache_file(cfg, path);
        return;
    }
    return error.GpuAccelerationDisabled;
}

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;
pub extern "c" fn futhark_context_get_error(ctx: ?*struct_futhark_context) ?[*:0]const u8;
pub extern "c" fn futhark_context_clear_caches(ctx: ?*struct_futhark_context) c_int;

pub extern "c" fn futhark_new_f16_1d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64) ?*struct_futhark_f16_1d;
pub extern "c" fn futhark_new_f16_2d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d(ctx: ?*struct_futhark_context, data: ?[*]const u16, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;
pub extern "c" fn futhark_new_f16_2d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_f16_3d_from_f32(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;

pub extern "c" fn futhark_free_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d) c_int;
pub extern "c" fn futhark_free_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_free_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d) c_int;

pub extern "c" fn futhark_values_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]u16) c_int;
pub extern "c" fn futhark_values_f16_2d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f16_3d_to_f32(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, data: ?[*]f32) c_int;

pub extern "c" fn futhark_values_raw_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d) ?*anyopaque;
pub extern "c" fn futhark_shape_f16_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_1d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_f16_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_2d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_f16_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f16_3d, dims: ?[*]i64) c_int;

pub extern "c" fn futhark_new_raw_f16_2d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_new_raw_f16_3d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f16_3d;
pub extern "c" fn futhark_new_raw_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_raw_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_raw_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_raw_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64) ?*struct_futhark_i64_1d;
pub extern "c" fn futhark_new_raw_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]u8, dim0: i64) ?*struct_futhark_u64_1d;

pub extern "c" fn futhark_new_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]const u64, dim0: i64) ?*struct_futhark_u64_1d;
pub extern "c" fn futhark_new_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64) ?*struct_futhark_i64_1d;
pub extern "c" fn futhark_new_i64_2d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64, dim1: i64) ?*struct_futhark_i64_2d;

pub extern "c" fn futhark_free_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_free_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_free_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_free_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) c_int;
pub extern "c" fn futhark_free_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) c_int;
pub extern "c" fn futhark_free_i64_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_2d) c_int;

pub extern "c" fn futhark_values_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, data: ?[*]u64) c_int;
pub extern "c" fn futhark_values_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, data: ?[*]i64) c_int;
pub extern "c" fn futhark_values_i64_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_2d, data: ?[*]i64) c_int;

pub extern "c" fn futhark_values_raw_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_i64_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_2d) ?*anyopaque;
pub extern "c" fn futhark_values_raw_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) ?*anyopaque;

pub extern "c" fn futhark_shape_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, dims: ?[*]i64) c_int;
pub extern "c" fn futhark_shape_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, dims: ?[*]i64) c_int;

pub extern "c" fn futhark_entry_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_2d, a: ?*struct_futhark_f32_2d, b: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_batch_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_3d, a: ?*struct_futhark_f32_3d, b: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_entry_dot(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*struct_futhark_f32_1d, b: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_clip_fisher(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, fisher: ?*struct_futhark_f32_1d, clip_val: f32) c_int;
pub extern "c" fn futhark_entry_reduce_gradients(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, gradients: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_rank_segments(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, query_hash: u64, segment_hashes: ?*struct_futhark_u64_1d, base_scores: ?*struct_futhark_f32_1d) c_int;

pub extern "c" fn futhark_entry_rsf_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    input: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_rsf_backward(
    ctx: ?*struct_futhark_context,
    out_grad_ws: ?*?*struct_futhark_f16_2d,
    out_grad_wt: ?*?*struct_futhark_f16_2d,
    input: ?*struct_futhark_f16_2d,
    grad_output: ?*struct_futhark_f16_2d,
    weights_s: ?*struct_futhark_f16_2d,
    weights_t: ?*struct_futhark_f16_2d,
    clip_min: u16,
    clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_scale_weights_inplace(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f16_2d, weights: ?*struct_futhark_f16_2d, scale: u16) c_int;

pub extern "c" fn futhark_entry_scale_matrix_f16(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    values: ?*struct_futhark_f16_2d,
    scale_factor: u16,
) c_int;

pub extern "c" fn futhark_entry_scale_matrix_f32(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    values: ?*struct_futhark_f32_2d,
    scale_factor: f32,
) c_int;

pub extern "c" fn futhark_entry_clip_matrix_global_norm_f32(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    values: ?*struct_futhark_f32_2d,
    clip_norm: f32,
) c_int;

pub extern "c" fn futhark_entry_batch_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_2d,
    in2_weights_t: ?*const struct_futhark_f16_2d,
    in3_clip_min: u16,
    in4_clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_batch_compute_loss(
    ctx: ?*struct_futhark_context,
    out: ?*u16,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_compute_initial_grad_l2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_compute_loss_masked(
    ctx: ?*struct_futhark_context,
    out: ?*u16,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
    in2_lengths: ?*const struct_futhark_i64_1d,
) c_int;

pub extern "c" fn futhark_entry_compute_initial_grad_l2_masked(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
    in2_lengths: ?*const struct_futhark_i64_1d,
) c_int;

pub extern "c" fn futhark_entry_batch_add_reconstruction_delta_masked(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_forward_delta: ?*const struct_futhark_f16_3d,
    in1_reconstructed: ?*const struct_futhark_f16_3d,
    in2_original: ?*const struct_futhark_f16_3d,
    in3_lengths: ?*const struct_futhark_i64_1d,
    in4_alpha: u16,
    in5_forward_scale: u16,
) c_int;

pub extern "c" fn futhark_entry_batch_compute_reconstruction_loss_masked(
    ctx: ?*struct_futhark_context,
    out: ?*u16,
    in0_reconstructed: ?*const struct_futhark_f16_3d,
    in1_original: ?*const struct_futhark_f16_3d,
    in2_lengths: ?*const struct_futhark_i64_1d,
) c_int;

pub extern "c" fn futhark_entry_embedding_sum_squares(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    in0_source: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_batch_gradients_full(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_grad_full,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_grad_outputs: ?*const struct_futhark_f16_3d,
    in2_weights_s: ?*const struct_futhark_f16_2d,
    in3_weights_t: ?*const struct_futhark_f16_2d,
    in4_clip_min: u16,
    in5_clip_max: u16,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup3_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    obj: ?*const struct_futhark_opaque_tup3_grad_full,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    obj: ?*const struct_futhark_opaque_tup3_grad_full,
) c_int;

pub extern "c" fn futhark_entry_master_weights_to_f16_3d(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    weights: ?*const struct_futhark_f32_3d,
) c_int;

pub extern "c" fn futhark_entry_master_weights_to_f16_2d(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    weights: ?*const struct_futhark_f32_2d,
) c_int;

pub extern "c" fn futhark_entry_forward_weights_to_f32_2d(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    weights: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_update_sfd_master(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
    master_weight: ?*const struct_futhark_f32_2d,
    grad_weight: ?*const struct_futhark_f32_2d,
    momentum_state: ?*const struct_futhark_f32_2d,
    fisher_state: ?*const struct_futhark_f32_2d,
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    optimizer_step: i64,
    epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;
pub extern "c" fn futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_inputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_3d,
    in2_weights_t: ?*const struct_futhark_f16_3d,
    in3_clip_min: u16,
    in4_clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_inverse(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_3d,
    in2_weights_t: ?*const struct_futhark_f16_3d,
    in3_clip_min: u16,
    in4_clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_rsf_stack_backward_gradients_fused(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup6_fused_stack_gradients,
    in0_final_outputs: ?*const struct_futhark_f16_3d,
    in1_targets: ?*const struct_futhark_f16_3d,
    in2_originals: ?*const struct_futhark_f16_3d,
    in3_lengths: ?*const struct_futhark_i64_1d,
    in4_weights_s: ?*const struct_futhark_f16_3d,
    in5_weights_t: ?*const struct_futhark_f16_3d,
    in6_gradient_scale: f32,
    in7_clip_min: f32,
    in8_clip_max: f32,
    in9_reconstruction_alpha: f32,
    in10_forward_scale: f32,
    in11_logdet_weight: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_3(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_4(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_5(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup6_fused_stack_gradients,
) c_int;

pub extern "c" fn futhark_entry_stack_update_sfd_master(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_stack_sfd,
    master_weights: ?*const struct_futhark_f32_3d,
    gradients: ?*const struct_futhark_f32_3d,
    momentum_state: ?*const struct_futhark_f32_3d,
    fisher_state: ?*const struct_futhark_f32_3d,
    learning_rate: f32,
    momentum_beta: f32,
    fisher_gamma: f32,
    optimizer_step: i64,
    epsilon: f32,
    trust_ratio: f32,
    weight_floor: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_sfd,
) c_int;

pub extern "c" fn futhark_entry_oftb_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    inputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_oftb_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    grad_outputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_oftb_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    inputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_oftb_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    grad_outputs: ?*const struct_futhark_f16_3d,
) c_int;

pub extern "c" fn futhark_entry_batch_rsf_inverse(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    in0_outputs: ?*const struct_futhark_f16_3d,
    in1_weights_s: ?*const struct_futhark_f16_2d,
    in2_weights_t: ?*const struct_futhark_f16_2d,
    in3_clip_min: u16,
    in4_clip_max: u16,
) c_int;

pub extern "c" fn futhark_entry_embedding_forward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_2d,
    tokens: ?*const struct_futhark_i64_1d,
    weight: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_backward(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    tokens: ?*const struct_futhark_i64_1d,
    grad_output: ?*const struct_futhark_f16_2d,
    grad_weight: ?*const struct_futhark_f32_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_forward_padded(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f16_3d,
    tokens: ?*const struct_futhark_i64_1d,
    lengths: ?*const struct_futhark_i64_1d,
    positions: ?*const struct_futhark_i64_1d,
    weight: ?*const struct_futhark_f16_2d,
) c_int;

pub extern "c" fn futhark_entry_embedding_backward_padded(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    tokens: ?*const struct_futhark_i64_1d,
    lengths: ?*const struct_futhark_i64_1d,
    grad_output: ?*const struct_futhark_f16_3d,
    grad_weight: ?*const struct_futhark_f32_2d,
) c_int;

pub extern "c" fn futhark_entry_stack_spectral_normalize(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup3_stack_spectral,
    weights: ?*const struct_futhark_f32_3d,
    target: f32,
    power_iters: i64,
) c_int;

pub extern "c" fn futhark_free_opaque_tup3_arr3d_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_3d,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup3_arr3d_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup3_stack_spectral,
) c_int;

pub extern "c" fn futhark_entry_embedding_spectral_normalize(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup5_embedding_spectral,
    weight: ?*const struct_futhark_f32_2d,
    u: ?*const struct_futhark_f32_1d,
    v: ?*const struct_futhark_f32_1d,
    power_iters: i64,
    target: f32,
) c_int;

pub extern "c" fn futhark_free_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_2d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_3(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_4(
    ctx: ?*struct_futhark_context,
    out: ?*f32,
    obj: ?*const struct_futhark_opaque_tup5_embedding_spectral,
) c_int;

pub extern "c" fn futhark_entry_graph_batch_encode(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_opaque_tup7_graph_encode,
    in0: ?*const struct_futhark_u64_1d,
    in1: u64,
) c_int;

pub extern "c" fn futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(
    ctx: ?*struct_futhark_context,
    obj: ?*struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_u64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_f32_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_i64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;

pub extern "c" fn futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(
    ctx: ?*struct_futhark_context,
    out: ?*?*struct_futhark_i64_1d,
    obj: ?*const struct_futhark_opaque_tup7_graph_encode,
) c_int;
