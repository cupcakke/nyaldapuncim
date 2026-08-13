let matmul_f16 [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : *[m][n]f16 =
  let bt = transpose b
  in map (\a_row ->
    map (\b_col ->
      f16.sum (map2 (f16.*) a_row b_col)) bt) a


entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): *[m][n]f32 =
  let bt = transpose b
  in map (\a_row -> map (\b_col -> f32.sum (map2 (*) a_row b_col)) bt) a

let oftb_scale_f32 : f32 = 0.7071067811865476

let clamp_f16_value (v: f32) : f32 =
  let safe = if f32.isnan v || f32.isinf v then 0f32 else v
  in f32.max (-60000f32) (f32.min 60000f32 safe)

let clamp_f16_weight (v: f32) : f32 =
  let safe = if f32.isnan v || f32.isinf v then 0f32 else v
  in f32.max (-65504f32) (f32.min 65504f32 safe)

let mask_embedding_gradient [dim] (gradient: [dim]f32) (valid: bool) : [dim]f32 =
  if valid then gradient else replicate dim 0f32

entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[n][half*2]f16 =
  let d = half * 2
  in map (\row ->
    let x1 = map f32.f16 (row[0:half] :> [half]f16)
    let x2 = map f32.f16 (row[half:d] :> [half]f16)
    let scale = map (\j ->
      let sum = f32.f16 weights_s[j][half] + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_s[j][0:half] :> [half]f16) x2)
      let clipped = f32.max (f32.f16 clip_min) (f32.min (f32.f16 clip_max) sum)
      in f32.exp clipped) (iota half)
    let y1 = map2 (*) x1 scale
    let trans = map (\j ->
      let value = f32.f16 weights_t[j][half] + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_t[j][0:half] :> [half]f16) y1)
      in if f32.isnan value || f32.isinf value then 0f32 else value) (iota half)
    let y2 = map2 (+) x2 trans
    let output = map (\value -> f16.f32 (clamp_f16_value value)) (y1 ++ y2)
    in output :> [half*2]f16) input

entry rsf_backward [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let weights_t_body = map (\row -> row[0:half] :> [half]f16) weights_t
  let weights_t_t = transpose weights_t_body
  let per_tok = map2 (\row g_row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:half*2] :> [half]f16
    let pre_scale = map (\j ->
      weights_s[j][half] f16.+ f16.sum (map2 (f16.*) (weights_s[j][0:half] :> [half]f16) x2)) (iota half)
    let scale = map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps))) pre_scale
    let y1 = map2 (f16.*) x1 scale
    let dy1 = g_row[0:half] :> [half]f16
    let dy2 = g_row[half:half*2] :> [half]f16
    let dy1_total = map2 (\dy1_j wt_t_row ->
      dy1_j f16.+ f16.sum (map2 (f16.*) wt_t_row dy2)) dy1 weights_t_t
    let ds = map2 (\ps j ->
      if ps f16.>= clip_min && ps f16.<= clip_max
      then dy1_total[j] f16.* y1[j]
      else f16.i32 0) pre_scale (iota half)
    in (ds, x2, dy2, y1)) input grad_output
  let ds_all  = map (\(a,_,_,_) -> a) per_tok
  let x2_all  = map (\(_,b,_,_) -> b) per_tok
  let dy2_all = map (\(_,_,c,_) -> c) per_tok
  let y1_all  = map (\(_,_,_,d) -> d) per_tok
  let ds_t  = transpose ds_all
  let dy2_t = transpose dy2_all
  let acc_ws = map2 (\ds_row bias ->
    let inner = map (\x2_row -> f16.sum (map2 (f16.*) ds_row x2_row)) (transpose x2_all)
    in inner ++ [bias] :> [half+1]f16) ds_t (map f16.sum ds_t)
  let acc_wt = map2 (\dy2_row bias ->
    let inner = map (\y1_row -> f16.sum (map2 (f16.*) dy2_row y1_row)) (transpose y1_all)
    in inner ++ [bias] :> [half+1]f16) dy2_t (map f16.sum dy2_t)
  in (copy acc_ws, copy acc_wt)

let sfd_fisher_update_core [d][e]
  (weights: [d][e]f32) (gradients: [d][e]f32)
  (momentum_state: [d][e]f32) (fisher_state: [d][e]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32)
  (optimizer_step: i64) (epsilon: f32) (trust_ratio: f32) (weight_floor: f32)
  : ([d][e]f32, [d][e]f32, [d][e]f32) =
  let safe_beta = f32.max 0f32 (f32.min 0.99999994f32 momentum_beta)
  let safe_gamma = f32.max 0f32 (f32.min 0.99999994f32 fisher_gamma)
  let safe_eps = f32.max epsilon 1e-12f32
  let step_f = f32.i64 (i64.max 1 optimizer_step)
  let momentum_correction = f32.max safe_eps (1f32 - safe_beta f32.** step_f)
  let fisher_correction = f32.max safe_eps (1f32 - safe_gamma f32.** step_f)
  let momentum_next = map2 (map2 (\m g ->
    let safe_g = if f32.isnan g || f32.isinf g then 0f32 else g
    let candidate = safe_beta * m + (1f32 - safe_beta) * safe_g
    in if f32.isnan candidate || f32.isinf candidate then m else candidate)) momentum_state gradients
  let fisher_next = map2 (map2 (\f g ->
    let safe_g = if f32.isnan g || f32.isinf g then 0f32 else g
    let candidate = safe_gamma * f + (1f32 - safe_gamma) * safe_g * safe_g
    in if f32.isnan candidate || f32.isinf candidate then f else f32.min 1e6f32 candidate)) fisher_state gradients
  let weights_next = map4 (map4 (\w g m f ->
    let safe_w = if f32.isnan w || f32.isinf w then 0f32 else w
    let m_hat = m / momentum_correction
    let f_hat = f / fisher_correction
    let raw_step = learning_rate * m_hat / (f32.sqrt (f32.max f_hat 0f32) + safe_eps)
    let safe_trust_ratio = f32.max 0f32 (f32.min 1f32 trust_ratio)
    let max_step = safe_trust_ratio * f32.max weight_floor (f32.abs safe_w)
    let clipped_step = f32.max (-max_step) (f32.min max_step raw_step)
    let updated = safe_w - clipped_step
    in if f32.isnan g || f32.isinf g || f32.isnan w || f32.isinf w || f32.isnan updated || f32.isinf updated
       then w
       else updated)) weights gradients momentum_next fisher_next
  in (copy weights_next, copy momentum_next, copy fisher_next)

entry master_weights_to_f16_3d [layers][rows][columns] (weights: [layers][rows][columns]f32): *[layers][rows][columns]f16 =
  map (map (map (\value -> f16.f32 (clamp_f16_weight value)))) weights

entry stack_update_sfd_master [layers][rows][columns]
  (master_weights: *[layers][rows][columns]f32)
  (gradients: [layers][rows][columns]f32)
  (momentum_state: *[layers][rows][columns]f32)
  (fisher_state: *[layers][rows][columns]f32)
  (learning_rate: f32)
  (momentum_beta: f32)
  (fisher_gamma: f32)
  (optimizer_step: i64)
  (epsilon: f32)
  (trust_ratio: f32)
  (weight_floor: f32)
  : (*[layers][rows][columns]f32, *[layers][rows][columns]f32, *[layers][rows][columns]f32) =
  let updates = map4 (\w g m f ->
    sfd_fisher_update_core w g m f learning_rate momentum_beta fisher_gamma optimizer_step epsilon trust_ratio weight_floor
    ) master_weights gradients momentum_state fisher_state
  in (map (\(w,_,_) -> w) updates,
      map (\(_,m,_) -> m) updates,
      map (\(_,_,f) -> f) updates)

entry master_weights_to_f16_2d [rows][columns] (weights: [rows][columns]f32): *[rows][columns]f16 =
  map (map (\value -> f16.f32 (clamp_f16_weight value))) weights

entry forward_weights_to_f32_2d [rows][columns] (weights: [rows][columns]f16): *[rows][columns]f32 =
  map (map f32.f16) weights

entry embedding_update_sfd_master [vocab_size][dim]
  (master_weight: *[vocab_size][dim]f32) (grad_weight: [vocab_size][dim]f32)
  (momentum_state: *[vocab_size][dim]f32) (fisher_state: *[vocab_size][dim]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32) (optimizer_step: i64) (epsilon: f32)
  (trust_ratio: f32) (weight_floor: f32)
  : ([vocab_size][dim]f32, [vocab_size][dim]f32, [vocab_size][dim]f32) =
  in sfd_fisher_update_core master_weight grad_weight momentum_state fisher_state learning_rate momentum_beta fisher_gamma optimizer_step epsilon trust_ratio weight_floor

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry batch_forward [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][half*2]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t clip_min clip_max) inputs

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff_f32 = map2 (map2 (map2 (\o t ->
    let diff = (f32.f16 o) - (f32.f16 t)
    in diff * diff))) outputs targets
  let total_f32 = f32.sum (flatten (flatten squared_diff_f32))
  let count_f32 = f32.i64 (batch_size * seq_len * d)
  let mean_f32 = total_f32 / count_f32
  in f16.f32 mean_f32

entry batch_gradients [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let results = map2 (\inp g_out ->
    rsf_backward inp g_out weights_s weights_t clip_min clip_max) inputs grad_outputs
  let gs_list = map (\(gs, _) -> gs) results
  let gt_list = map (\(_, gt) -> gt) results
  let gs_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gs_list
  let gt_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gt_list
  in (copy gs_total, copy gt_total)

let rsf_backward_full [n][half]
  (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, *[n][half*2]f16) =
  let ws_body = map (\row -> row[0:half] :> [half]f16) weights_s
  let ws_bias = map (\row -> row[half]) weights_s
  let wt_body = map (\row -> row[0:half] :> [half]f16) weights_t
  let ws_body_t = transpose ws_body
  let x1_all  = map (\row -> row[0:half]      :> [half]f16) input
  let x2_all  = map (\row -> row[half:half*2] :> [half]f16) input
  let dy1_all = map (\row -> row[0:half]      :> [half]f16) grad_output
  let dy2_all = map (\row -> row[half:half*2] :> [half]f16) grad_output
  let x2_ws_t       = matmul_f16 x2_all ws_body_t
  let pre_scale_all = map (\row -> map2 (f16.+) row ws_bias) x2_ws_t
  let scale_all     = map (map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps)))) pre_scale_all
  let y1_all        = map2 (map2 (f16.*)) x1_all scale_all
  let dy2_wt        = matmul_f16 dy2_all wt_body
  let dy1_total_all = map2 (map2 (f16.+)) dy1_all dy2_wt
  let ds_all = map2 (map2 (\ps_val prod ->
    if ps_val f16.>= clip_min && ps_val f16.<= clip_max then prod else f16.i32 0)) pre_scale_all (map2 (map2 (f16.*)) dy1_total_all y1_all)
  let dx1_all  = map2 (map2 (f16.*)) dy1_total_all scale_all
  let ds_ws    = matmul_f16 ds_all ws_body
  let dx2_all  = map2 (map2 (f16.+)) dy2_all ds_ws
  let g_in     = map2 (\r1 r2 -> r1 ++ r2 :> [half*2]f16) dx1_all dx2_all
  let ds_t     = transpose ds_all
  let dy2_t    = transpose dy2_all
  let acc_ws_body = matmul_f16 ds_t x2_all
  let acc_ws_bias = map f16.sum ds_t
  let acc_ws      = map2 (\row bias -> row ++ [bias] :> [half+1]f16) acc_ws_body acc_ws_bias
  let acc_wt_body = matmul_f16 dy2_t y1_all
  let acc_wt_bias = map f16.sum dy2_t
  let acc_wt      = map2 (\row bias -> row ++ [bias] :> [half+1]f16) acc_wt_body acc_wt_bias
  in (copy acc_ws, copy acc_wt, copy g_in)

entry batch_rsf_inverse [batch_size][seq_len][half]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let flat_outputs = flatten outputs
  let ws_body   = map (\row -> row[0:half] :> [half]f16) weights_s
  let ws_bias   = map (\row -> row[half]) weights_s
  let wt_body   = map (\row -> row[0:half] :> [half]f16) weights_t
  let wt_bias   = map (\row -> row[half]) weights_t
  let wt_body_t = transpose wt_body
  let ws_body_t = transpose ws_body
  let y1_all       = map (\row -> row[0:half]      :> [half]f16) flat_outputs
  let y2_all       = map (\row -> row[half:half*2] :> [half]f16) flat_outputs
  let y1_wt_t      = matmul_f16 y1_all wt_body_t
  let trans_all    = map (\row -> map2 (f16.+) row wt_bias) y1_wt_t
  let x2_all       = map2 (map2 (f16.-)) y2_all trans_all
  let x2_ws_t      = matmul_f16 x2_all ws_body_t
  let pre_scale_all = map (\row -> map2 (f16.+) row ws_bias) x2_ws_t
  let scale_all    = map (map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps)))) pre_scale_all
  let x1_all       = map2 (map2 (f16./)) y1_all scale_all
  let flat_inputs  = map2 (\r1 r2 -> r1 ++ r2 :> [half*2]f16) x1_all x2_all
  in copy (unflatten flat_inputs :> [batch_size][seq_len][half*2]f16)

entry batch_gradients_full [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, *[batch_size][seq_len][half*2]f16) =
  let flat_inputs = flatten inputs
  let flat_grads  = flatten grad_outputs
  let (gs, gt, flat_g_in) = rsf_backward_full flat_inputs flat_grads weights_s weights_t clip_min clip_max
  let g_in = unflatten flat_g_in :> [batch_size][seq_len][half*2]f16
  in (copy gs, copy gt, copy g_in)

entry compute_initial_grad_l2 [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16)
  : *[batch_size][seq_len][d]f16 =
  map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets

entry batch_compute_loss_masked [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16)
  (targets: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64) : f16 =
  let squared_diff_f32 = map2 (\length bi ->
    let sample = outputs[bi]
    let target = targets[bi]
    in map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in map2 (\o t ->
        if active then
          let diff = f32.f16 o - f32.f16 t
          in if f32.isnan diff || f32.isinf diff then 0f32 else diff * diff
        else 0f32) sample[j] target[j]) (iota seq_len)) lengths (iota batch_size)
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = valid_tokens * d
  let total = f32.sum (flatten (flatten squared_diff_f32))
  let safe_total = if f32.isnan total || f32.isinf total then 0f32 else total
  in if count <= 0
     then f16.i32 0
     else f16.f32 (safe_total / f32.i64 count)

entry compute_initial_grad_l2_masked [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64)
  : *[batch_size][seq_len][d]f16 =
  map2 (\length bi ->
    let sample = outputs[bi]
    let target = targets[bi]
    in map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in if active
          then map2 (\o t ->
            let diff = f32.f16 o - f32.f16 t
            let safe_diff = if f32.isnan diff || f32.isinf diff then 0f32 else f32.max (-100f32) (f32.min 100f32 diff)
            in f16.f32 (2f32 * safe_diff)) sample[j] target[j]
         else replicate d (f16.i32 0)) (iota seq_len)) lengths (iota batch_size)

entry xavier_fill_inplace [d] (_weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale) (iota d)) (iota d)

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry scale_matrix_f16 [rows][columns] (values: *[rows][columns]f16) (scale_factor: f16) : *[rows][columns]f16 =
  map (map (\value -> value f16.* scale_factor)) values

entry scale_matrix_f32 [rows][columns] (values: *[rows][columns]f32) (scale_factor: f32) : *[rows][columns]f32 =
  map (map (\value -> value * scale_factor)) values

entry clip_matrix_global_norm_f32 [rows][columns]
  (values: *[rows][columns]f32) (clip_norm: f32) : *[rows][columns]f32 =
  let flat_values = flatten values
  let maximum_absolute_value = reduce f32.max 0f32 (map f32.abs flat_values)
  let scaled_norm_squared =
    if maximum_absolute_value > 0f32
    then f32.sum (map (\value ->
      let scaled = value / maximum_absolute_value
      in scaled * scaled) flat_values)
    else 0f32
  let norm = maximum_absolute_value * f32.sqrt scaled_norm_squared
  let scale =
    if clip_norm > 0f32 && norm > clip_norm && norm > 1e-12f32
    then clip_norm / norm
    else 1f32
  in map (map (* scale)) values

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

let oftb_scale : f16 = f16.f32 0.7071067811865476

entry oftb_forward_single [seq_len][dim] (input: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let x1 = map f32.f16 (row[0:half] :> [half]f16)
    let x2 = map f32.f16 (row[half:dim] :> [half]f16)
    let new_x1 = map2 (\a b -> (a - b) * f32.f16 oftb_scale) x1 x2
    let new_x2 = map2 (\a b -> (a + b) * f32.f16 oftb_scale) x1 x2
    let output = map (\value -> f16.f32 (clamp_f16_value value)) (new_x1 ++ new_x2)
    in output :> [dim]f16) input

entry oftb_backward_single [seq_len][dim] (grad_output: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let g1 = map f32.f16 (row[0:half] :> [half]f16)
    let g2 = map f32.f16 (row[half:dim] :> [half]f16)
    let new_g1 = map2 (\a b -> (a + b) * f32.f16 oftb_scale) g1 g2
    let new_g2 = map2 (\a b -> (b - a) * f32.f16 oftb_scale) g1 g2
    let output = map (\value ->
      let safe_value = if f32.isnan value || f32.isinf value then 0f32 else f32.max (-100f32) (f32.min 100f32 value)
      in f16.f32 safe_value) (new_g1 ++ new_g2)
    in output :> [dim]f16) grad_output

entry oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_forward_single sample) inputs

entry oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_backward_single sample) grad_outputs

entry batch_oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_forward inputs

entry batch_oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_backward grad_outputs

entry embedding_forward [n][vocab_size][dim] (tokens: [n]i64) (weight: [vocab_size][dim]f16) : *[n][dim]f16 =
  map (\tok ->
    let t = if tok >= 0 && tok < vocab_size then tok else 0
    in weight[t]) tokens

entry embedding_forward_padded [n][batch_size][seq_len][vocab_size][dim]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (positions: [seq_len]i64)
  (weight: [vocab_size][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map2 (\batch_index length ->
    map (\sequence_index ->
      let flat_index = batch_index * seq_len + sequence_index
      in if sequence_index < i64.max 0 (i64.min seq_len length) && flat_index < n
         then let token = tokens[flat_index]
              let safe_token = if token >= 0 && token < vocab_size then token else 0
              in weight[safe_token]
         else replicate dim (f16.i32 0)) positions) (iota batch_size) lengths

entry embedding_backward [n][vocab_size][dim] (tokens: [n]i64) (grad_output: [n][dim]f16) (grad_weight: [vocab_size][dim]f32) : *[vocab_size][dim]f32 =
  let valid = map (\token -> token >= 0 && token < vocab_size) tokens
  let safe_tokens = map2 (\token is_valid -> if is_valid then token else 0) tokens valid
  let gradients = map2 (\row is_valid -> if is_valid then map f32.f16 row else replicate dim 0f32) grad_output valid
  let updates = hist (map2 (+)) (replicate dim 0f32) vocab_size safe_tokens gradients
  in map2 (map2 (+)) grad_weight updates

entry embedding_backward_padded [n][batch_size][seq_len][dim][vocab_size]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (grad_output: [batch_size][seq_len][dim]f16)
  (grad_weight: [vocab_size][dim]f32) : *[vocab_size][dim]f32 =
  let masked_gradient_f32 = map2 (\length bi ->
    map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in if active
         then map f32.f16 grad_output[bi][j]
         else replicate dim 0f32) (iota seq_len)) lengths (iota batch_size)
  let flat_grad_flat = flatten masked_gradient_f32
  let flat_tokens_all = flatten (map2 (\length bi ->
    map (\j ->
      let flat_index = bi * seq_len + j
      let valid = j < i64.max 0 (i64.min seq_len length) && flat_index < n
      in if valid then tokens[flat_index] else -1) (iota seq_len)) lengths (iota batch_size))
  let validity = map2 (\t j ->
    let bi = j / seq_len
    let jj = j % seq_len
    let length = lengths[bi]
    let active = jj < i64.max 0 (i64.min seq_len length) && j < n
    in t >= 0 && t < vocab_size && active) flat_tokens_all (iota (batch_size * seq_len))
  let valid_tokens_unclamped = map2 (\t v -> if v then t else 0) flat_tokens_all validity
  let valid_grads : [batch_size*seq_len][dim]f32 = map2 mask_embedding_gradient flat_grad_flat validity
  let safe_tokens = map (\t -> if t >= 0 && t < vocab_size then t else 0) valid_tokens_unclamped
  let updates = hist (map2 (+)) (replicate dim 0f32) vocab_size safe_tokens valid_grads
  in map2 (map2 (+)) grad_weight updates

let spectral_normalize_matrix [rows][columns]
  (weight: [rows][columns]f32)
  (target: f32)
  (power_iters: i64)
  : ([rows][columns]f32, f32, f32) =
  let initial_value = 1f32 / f32.sqrt (f32.i64 columns)
  let initial_v = replicate columns initial_value
  let weight_t = transpose weight
  let (final_u, final_v) =
    loop (_u, v) = (replicate rows 0f32, initial_v) for iteration < i64.max 1 power_iters do
      let _ = iteration
      let raw_u = map (\row -> f32.sum (map2 (*) row v)) weight
      let u_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_u))
      let safe_u_norm = f32.max u_norm 1e-12f32
      let next_u = map (/ safe_u_norm) raw_u
      let raw_v = map (\column -> f32.sum (map2 (*) column next_u)) weight_t
      let v_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_v))
      let safe_v_norm = f32.max v_norm 1e-12f32
      let next_v = map (/ safe_v_norm) raw_v
      in (next_u, next_v)
  let projected = map (\row -> f32.sum (map2 (*) row final_v)) weight
  let sigma = f32.abs (f32.sum (map2 (*) final_u projected))
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) weight
  in (normalized, sigma, sigma * scale)

entry stack_spectral_normalize [layers][rows][columns]
  (weights: *[layers][rows][columns]f32)
  (target: f32)
  (power_iters: i64)
  : (*[layers][rows][columns]f32, f32, f32) =
  let results = map (\weight -> spectral_normalize_matrix weight target power_iters) weights
  let normalized = map (\(weight, _, _) -> weight) results
  let before = reduce f32.max 0f32 (map (\(_, sigma, _) -> sigma) results)
  let after = reduce f32.max 0f32 (map (\(_, _, sigma) -> sigma) results)
  in (normalized, before, after)

entry embedding_spectral_normalize [vocab_size][dim]
  (weight: *[vocab_size][dim]f32)
  (u: *[vocab_size]f32)
  (v: *[dim]f32)
  (power_iters: i64)
  (target: f32) : (*[vocab_size][dim]f32, *[vocab_size]f32, *[dim]f32, f32, f32) =
  let weight_t = transpose weight
  let (final_u, final_v) =
    loop (ua, _va) = (u, v) for loop_k < i64.max 1 power_iters do
      let _ = loop_k
      let raw_v = map (\column -> f32.sum (map2 (*) column ua)) weight_t
      let v_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_v))
      let next_v = map (/ f32.max v_norm 1e-12f32) raw_v
      let raw_u = map (\row -> f32.sum (map2 (*) row next_v)) weight
      let u_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_u))
      let next_u = map (/ f32.max u_norm 1e-12f32) raw_u
      in (next_u, copy next_v)
  let projected = map (\row -> f32.sum (map2 (*) row final_v)) weight
  let sigma = f32.abs (f32.sum (map2 (*) final_u projected))
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) weight
  in (normalized, copy final_u, copy final_v, sigma, sigma * scale)

let graph_derive_qubit_states [n] (hashes: [n]u64) : ([n]f32, [n]f32, [n]f32, [n]f32) =
  let pi = 3.14159265358979323846f32
  let inv_m = 1f32 / 1000000f32
  let raw_re_a = map (\h -> f32.cos (pi * f32.u64 (h % 1000000u64) * inv_m)) hashes
  let raw_im_a = map (\h -> f32.sin (pi * f32.u64 ((h >> 20u64) % 1000000u64) * inv_m)) hashes
  let raw_re_b = map (\h -> f32.cos (pi * 2f32 * f32.u64 ((h >> 40u64) % 1000000u64) * inv_m)) hashes
  let raw_im_b = map (\h -> f32.sin (pi * 2f32 * f32.u64 ((h >> 32u64) % 1000000u64) * inv_m)) hashes
  let norms = map4 (\ra ia rb ib ->
    let s = ra * ra + ia * ia + rb * rb + ib * ib
    in if s > 1e-30f32 then f32.sqrt s else 1f32) raw_re_a raw_im_a raw_re_b raw_im_b
  in (map2 (/) raw_re_a norms, map2 (/) raw_im_a norms, map2 (/) raw_re_b norms, map2 (/) raw_im_b norms)

entry graph_batch_encode [n] (data_hashes: [n]u64) (_seed: u64) : ([]u64, []f32, []f32, []f32, []f32, []i64, []i64) =
  let (re_a, im_a, re_b, im_b) = graph_derive_qubit_states data_hashes
  let ne = n * 3
  let edge_srcs = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i else -1i64)
  let edge_tgts = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i - pred_k - 1 else -1i64)
  in (copy data_hashes, re_a, im_a, re_b, im_b, edge_srcs, edge_tgts)

entry batch_add_reconstruction_delta_masked [batch_size][seq_len][d]
  (forward_delta: [batch_size][seq_len][d]f16)
  (reconstructed: [batch_size][seq_len][d]f16)
  (original: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64)
  (alpha: f16)
  (forward_scale: f16)
  : *[batch_size][seq_len][d]f16 =
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = if valid_tokens > 0 then valid_tokens * d else 1
  let count_f32 = f32.i64 count
  let alpha_f32 = f32.f16 alpha
  let forward_scale_f32 = f32.f16 forward_scale
  in map2 (\length bi ->
    let fd = forward_delta[bi]
    let rc = reconstructed[bi]
    let og = original[bi]
    let limit = i64.max 0 (i64.min seq_len length)
    in map (\j ->
      let active = j < limit
      in map3 (\f r o ->
        let f_f32 = f32.f16 f
        let safe_f = if f32.isnan f_f32 || f32.isinf f_f32 then 0f32 else f_f32
        in if active
           then
             let diff = f32.f16 r - f32.f16 o
             let safe_diff = if f32.isnan diff || f32.isinf diff
                             then 0f32
                             else f32.max (-100f32) (f32.min 100f32 diff)
             let combined = forward_scale_f32 * safe_f
                            + alpha_f32 * 2f32 * safe_diff / count_f32
             let bounded = f32.max (-65504f32) (f32.min 65504f32 combined)
             in f16.f32 bounded
           else
             let scaled = forward_scale_f32 * safe_f
             let bounded = f32.max (-65504f32) (f32.min 65504f32 scaled)
             in f16.f32 bounded) fd[j] rc[j] og[j]) (iota seq_len)) lengths (iota batch_size)

entry batch_compute_reconstruction_loss_masked [batch_size][seq_len][d]
  (reconstructed: [batch_size][seq_len][d]f16)
  (original: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64)
  : f16 =
  let squared_diff_f32 = map2 (\length bi ->
    let rc = reconstructed[bi]
    let og = original[bi]
    let limit = i64.max 0 (i64.min seq_len length)
    in map (\j ->
      let active = j < limit
      in map2 (\r o ->
        if active
        then
          let diff = f32.f16 r - f32.f16 o
          in if f32.isnan diff || f32.isinf diff then 0f32 else diff * diff
        else 0f32) rc[j] og[j]) (iota seq_len)) lengths (iota batch_size)
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = valid_tokens * d
  let total = f32.sum (flatten (flatten squared_diff_f32))
  let safe_total = if f32.isnan total || f32.isinf total then 0f32 else total
  in if count <= 0
     then f16.i32 0
     else f16.f32 (safe_total / f32.i64 count)

entry embedding_sum_squares [vocab_size][dim] (source: [vocab_size][dim]f16) : f32 =
  let squared = map (\row ->
    map (\v ->
      let x = f32.f16 v
      in if f32.isnan x || f32.isinf x then 0f32 else x * x) row) source
  let total = f32.sum (flatten squared)
  in if f32.isnan total || f32.isinf total then 0f32 else total

let rsf_stack_coupling_row [half]
  (row: [half*2]f32)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min_f32: f32) (clip_max_f32: f32) : [half*2]f32 =
  let x1 = row[0:half] :> [half]f32
  let x2 = row[half:half*2] :> [half]f32
  let scale = map (\d ->
    let sum = f32.f16 weights_s[d][half]
              + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_s[d][0:half] :> [half]f16) x2)
    let clipped = f32.max clip_min_f32 (f32.min clip_max_f32 sum)
    in f32.exp clipped) (iota half)
  let y1 = map2 (*) x1 scale
  let y2 = map2 (\x2_j j ->
    let trans = f32.f16 weights_t[j][half]
                + f32.sum (map2 (\w u -> f32.f16 w * u) (weights_t[j][0:half] :> [half]f16) y1)
    in x2_j + (if f32.isnan trans || f32.isinf trans then 0f32 else trans)) x2 (iota half)
  let o1 = map2 (\a b -> (a - b) * oftb_scale_f32) y1 y2
  let o2 = map2 (\a b -> (a + b) * oftb_scale_f32) y1 y2
  in (map clamp_f16_value (o1 ++ o2)) :> [half*2]f32

let rsf_stack_invert_row [half]
  (row: [half*2]f32)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min_f32: f32) (clip_max_f32: f32) : [half*2]f32 =
  let y1p = row[0:half] :> [half]f32
  let y2p = row[half:half*2] :> [half]f32
  let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
  let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
  let x2 = map (\d ->
    let trans = f32.f16 weights_t[d][half]
                + f32.sum (map2 (\w u -> f32.f16 w * u) (weights_t[d][0:half] :> [half]f16) u1)
    let safe_trans = if f32.isnan trans || f32.isinf trans then 0f32 else trans
    in u2[d] - safe_trans) (iota half)
  let x1 = map (\d ->
    let pre = f32.f16 weights_s[d][half]
              + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_s[d][0:half] :> [half]f16) x2)
    let clipped = f32.max clip_min_f32 (f32.min clip_max_f32 pre)
    in u1[d] / f32.exp clipped) (iota half)
  in (x1 ++ x2) :> [half*2]f32

entry rsf_stack_forward [batch_size][seq_len][half][num_layers]
  (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][half+1]f16)
  (weights_t: [num_layers][half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let clip_min_f32 = f32.f16 clip_min
  let clip_max_f32 = f32.f16 clip_max
  let flat = flatten inputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for l < num_layers do
      rsf_stack_coupling_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

entry rsf_stack_inverse [batch_size][seq_len][half][num_layers]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][half+1]f16)
  (weights_t: [num_layers][half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let clip_min_f32 = f32.f16 clip_min
  let clip_max_f32 = f32.f16 clip_max
  let flat = flatten outputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for i < num_layers do
      let l = num_layers - 1 - i
      in rsf_stack_invert_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

entry rsf_stack_backward_gradients_fused [batch_size][seq_len][half][num_layers]
  (final_outputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (originals: [batch_size][seq_len][half*2]f16)
  (lengths: [batch_size]i64)
  (weights_s: *[num_layers][half][half+1]f16)
  (weights_t: *[num_layers][half][half+1]f16)
  (gradient_scale: f32)
  (clip_min: f32)
  (clip_max: f32)
  (reconstruction_alpha: f32)
  (forward_scale: f32)
  (logdet_weight: f32)
  : (*[num_layers][half][half+1]f32, *[num_layers][half][half+1]f32,
     *[batch_size][seq_len][half*2]f16,
     f32, f32, f32) =
  let d2 = half * 2
  let flat_final = flatten final_outputs
  let flat_targets = flatten targets
  let flat_orig = flatten originals
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let valid_tokens = i64.sum limits
  let count_elements = if valid_tokens > 0 then valid_tokens * d2 else 1
  let count_elements_f32 = f32.i64 count_elements
  let count_tokens_f32 = f32.max 1f32 (f32.i64 valid_tokens)
  let active_indices = filter (\t ->
    let b = t / seq_len
    let j = t % seq_len
    in j < limits[b]) (iota (batch_size * seq_len))
  let active_final = map (\t -> flat_final[t]) active_indices
  let active_targets = map (\t -> flat_targets[t]) active_indices
  let active_orig = map (\t -> flat_orig[t]) active_indices
  let initial_grads = map2 (\y t ->
    map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe_diff = if f32.isnan diff || f32.isinf diff then 0f32 else f32.max (-100f32) (f32.min 100f32 diff)
      in 2f32 * safe_diff / count_elements_f32) y t) active_final active_targets
  let y_start = map (map f32.f16) active_final
  let gs_zero = replicate half (replicate (half + 1) 0f32)
  let (gs_stack, gt_stack, x_stack, g_stack, ld_stack) =
    loop (gs_acc, gt_acc, y_all, g_all, ld_all) =
      (replicate num_layers (copy gs_zero),
       replicate num_layers (copy gs_zero),
       y_start,
       initial_grads,
       replicate valid_tokens 0f32)
    for i < num_layers do
      let l = num_layers - 1 - i
      let ws = copy weights_s[l]
      let wt = copy weights_t[l]
      let per_tok = map2 (\y_row g_row ->
        let y1p = y_row[0:half] :> [half]f32
        let y2p = y_row[half:d2] :> [half]f32
        let g1p = g_row[0:half] :> [half]f32
        let g2p = g_row[half:d2] :> [half]f32
        let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
        let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
        let h1 = map2 (\a b -> (a + b) * oftb_scale_f32) g1p g2p
        let h2 = map2 (\a b -> (b - a) * oftb_scale_f32) g1p g2p
        let dy1_total = map (\j ->
          h1[j] + f32.sum (map (\d -> h2[d] * f32.f16 wt[d][j]) (iota half))) (iota half)
        let x2 = map (\d ->
          let trans = f32.f16 wt[d][half]
                      + f32.sum (map2 (\w u -> f32.f16 w * u) (wt[d][0:half] :> [half]f16) u1)
          let safe_trans = if f32.isnan trans || f32.isinf trans then 0f32 else trans
          in u2[d] - safe_trans) (iota half)
        let pre_scale = map (\d ->
          f32.f16 ws[d][half]
          + f32.sum (map2 (\w x -> f32.f16 w * x) (ws[d][0:half] :> [half]f16) x2)) (iota half)
        let clipped = map (\p -> f32.max clip_min (f32.min clip_max p)) pre_scale
        let scale = map f32.exp clipped
        let x1 = map2 (/) u1 scale
        let dx1 = map2 (*) dy1_total scale
        let ld_shift = logdet_weight / count_tokens_f32
        let ds = map3 (\p dt_j u_j ->
          if p >= clip_min && p <= clip_max then dt_j * u_j + ld_shift else 0f32) pre_scale dy1_total u1
        let dx2 = map (\j ->
          h2[j] + f32.sum (map (\d -> ds[d] * f32.f16 ws[d][j]) (iota half))) (iota half)
        let y_next = (x1 ++ x2) :> [half*2]f32
        let g_next = (dx1 ++ dx2) :> [half*2]f32
        let ld_tok = f32.sum clipped
        in (ds, h2, x2, u1, y_next, g_next, ld_tok)) y_all g_all
      let ds_columns = transpose (map (\(ds,_,_,_,_,_,_) -> ds) per_tok)
      let h2_columns = transpose (map (\(_,h2,_,_,_,_,_) -> h2) per_tok)
      let x2_columns = transpose (map (\(_,_,x2,_,_,_,_) -> x2) per_tok)
      let u1_columns = transpose (map (\(_,_,_,u1,_,_,_) -> u1) per_tok)
      let gs_l_total = map (\ds_column ->
        let row_body = map (\x2_column -> f32.sum (map2 (*) ds_column x2_column)) x2_columns
        in (row_body ++ [f32.sum ds_column]) :> [half+1]f32) ds_columns
      let gt_l_total = map (\h2_column ->
        let row_body = map (\u1_column -> f32.sum (map2 (*) h2_column u1_column)) u1_columns
        in (row_body ++ [f32.sum h2_column]) :> [half+1]f32) h2_columns
      let y_next_all = map (\(_,_,_,_,y_next,_,_) -> y_next) per_tok
      let g_next_all = map (\(_,_,_,_,_,g_next,_) -> g_next) per_tok
      let ld_next = map2 (+) ld_all (map (\(_,_,_,_,_,_,ld_tok) -> ld_tok) per_tok)
      in (gs_acc with [l] = gs_l_total,
          gt_acc with [l] = gt_l_total,
          y_next_all,
          g_next_all,
          ld_next)
  let gs_normalized = map (map (map (* gradient_scale))) gs_stack
  let gt_normalized = map (map (map (* gradient_scale))) gt_stack
  let loss_total = reduce (+) 0f32 (map2 (\y t ->
    f32.sum (map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe = if f32.isnan diff || f32.isinf diff then 0f32 else diff
      in safe * safe) y t)) active_final active_targets)
  let loss = loss_total / count_elements_f32
  let recon_total = reduce (+) 0f32 (map2 (\x_row o_row ->
    f32.sum (map2 (\xv ov ->
      let diff = xv - f32.f16 ov
      let safe = if f32.isnan diff || f32.isinf diff then 0f32 else diff
      in safe * safe) x_row o_row)) x_stack active_orig)
  let recon_loss = recon_total / count_elements_f32
  let logdet_total = reduce (+) 0f32 ld_stack
  let logdet_mean = logdet_total / count_tokens_f32
  let active_input_delta = map3 (\g_row x_row o_row ->
    map3 (\gv xv ov ->
      let base = forward_scale * gv
      let diff = xv - f32.f16 ov
      let safe_diff = if f32.isnan diff || f32.isinf diff then 0f32 else f32.max (-100f32) (f32.min 100f32 diff)
      let combined = base + reconstruction_alpha * 2f32 * safe_diff / count_elements_f32
      in f16.f32 (f32.max (-65504f32) (f32.min 65504f32 combined))) g_row x_row o_row
    ) g_stack x_stack active_orig
  let zero_delta = replicate (batch_size * seq_len) (replicate d2 0f16)
  let input_delta = scatter zero_delta active_indices active_input_delta
  let input_delta_3d = unflatten input_delta :> [batch_size][seq_len][half*2]f16
  in (copy gs_normalized, copy gt_normalized, input_delta_3d,
      f32.max 0f32 loss, f32.max 0f32 recon_loss, logdet_mean)
