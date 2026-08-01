import triton
import triton.language as tl

from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table


_CONCAT_CONFIGS = [
    triton.Config({"NUM_SM": num_sm, "BLOCK": block}, num_warps=4, num_stages=3)
    for num_sm in (84, 128, 92)
    for block in (256, 512, 1024)
]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": block}, num_warps=4, num_stages=3)
        for block in (128, 256, 512, 1024)
    ],
    key=[],
)
@triton.jit
def position_offset_kernel(pos_ptr, offset_ptr, output_ptr, batch_size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    addr = tl.arange(0, BLOCK) + pid * BLOCK
    mask = addr < batch_size
    posx = tl.load(offset_ptr + addr, mask=mask)
    posy = tl.load(offset_ptr + addr + 1, mask=mask)
    empty = posx == posy
    result = tl.load(pos_ptr + posy - 1, mask=mask & (~empty), other=-1)
    tl.store(output_ptr + addr, result + 1, mask=mask)


@triton.autotune(configs=_CONCAT_CONFIGS, key=[])
@triton.jit
def nested_concat_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    x_offset_ptr,
    y_offset_ptr,
    batchsize,
    DIM: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SM: tl.constexpr,
):
    tid = tl.cast(tl.program_id(0), tl.int64)
    for i in range(batchsize):
        begin = tl.load(x_offset_ptr + i) * DIM
        end = tl.load(x_offset_ptr + i + 1) * DIM
        rbegin = tl.load(y_offset_ptr + i) * DIM
        src = begin // ALIGN * ALIGN
        block = tl.cdiv(end - src, BLOCK)
        while tid < block:
            block_addr = tl.arange(0, BLOCK) + src + tid * BLOCK
            mask = (block_addr >= begin) & (block_addr < end)
            value = tl.load(x_ptr + block_addr, mask=mask)
            tl.store(o_ptr + block_addr + rbegin, value, mask=mask)
            tid += NUM_SM
        tid -= block
    for i in range(batchsize):
        begin = tl.load(y_offset_ptr + i) * DIM
        end = tl.load(y_offset_ptr + i + 1) * DIM
        rbegin = tl.load(x_offset_ptr + i + 1) * DIM
        src = begin // ALIGN * ALIGN
        block = tl.cdiv(end - src, BLOCK)
        while tid < block:
            block_addr = tl.arange(0, BLOCK) + src + tid * BLOCK
            mask = (block_addr >= begin) & (block_addr < end)
            value = tl.load(y_ptr + block_addr, mask=mask)
            tl.store(o_ptr + block_addr + rbegin, value, mask=mask)
            tid += NUM_SM
        tid -= block


@triton.autotune(configs=_CONCAT_CONFIGS, key=[])
@triton.jit
def position_concat_kernel(
    x_ptr,
    y_ptr,
    o_ptr,
    x_offset_ptr,
    y_offset_ptr,
    position_offset_ptr,
    batchsize,
    DIM: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SM: tl.constexpr,
):
    tid = tl.cast(tl.program_id(0), tl.int64)
    for i in range(batchsize):
        begin = tl.load(x_offset_ptr + i) * DIM
        end = tl.load(x_offset_ptr + i + 1) * DIM
        rbegin = tl.load(y_offset_ptr + i) * DIM
        src = begin // ALIGN * ALIGN
        block = tl.cdiv(end - src, BLOCK)
        while tid < block:
            block_addr = tl.arange(0, BLOCK) + src + tid * BLOCK
            mask = (block_addr >= begin) & (block_addr < end)
            value = tl.load(x_ptr + block_addr, mask=mask)
            tl.store(o_ptr + block_addr + rbegin, value, mask=mask)
            tid += NUM_SM
        tid -= block
    for i in range(batchsize):
        begin = tl.load(y_offset_ptr + i) * DIM
        end = tl.load(y_offset_ptr + i + 1) * DIM
        rbegin = tl.load(x_offset_ptr + i + 1) * DIM
        position_offset = tl.load(position_offset_ptr + i)
        src = begin // ALIGN * ALIGN
        block = tl.cdiv(end - src, BLOCK)
        while tid < block:
            block_addr = tl.arange(0, BLOCK) + src + tid * BLOCK
            mask = (block_addr >= begin) & (block_addr < end)
            value = tl.load(y_ptr + block_addr, mask=mask) + position_offset
            tl.store(o_ptr + block_addr + rbegin, value, mask=mask)
            tid += NUM_SM
        tid -= block


@triton.jit
def _load_block(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def _attention_mask(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]
    if TYPE == 1:
        return (
            triu_causal
            & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))
        ) | (q_offset[:, None] == k_offset[None, :])
    if TYPE == 2:
        return (
            tril_causal
            & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))
        ) | (q_offset[:, None] == k_offset[None, :])


@triton.jit
def _store_block(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
    elif EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0,))
    elif EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1,))
    else:
        tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2)],
    key=[],
)
@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    q_attn_arg_ptr,
    k_attn_arg_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    q_head,
    kv_head,
    scale,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    MASK_FN: tl.constexpr,
    SPARSE_OPT: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start
    if SPARSE_OPT:
        begin = 0
        if k_len == 0:
            return
        end = k_len
    else:
        if MASK_FN & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape=(q_len, QK_DIM),
        strides=(q_head * QK_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape=(QK_DIM, k_len),
        strides=(1, kv_head * QK_DIM),
        offsets=(0, begin),
        block_shape=(QK_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape=(k_len, V_DIM),
        strides=(kv_head * V_DIM, 1),
        offsets=(begin, 0),
        block_shape=(BLOCK_N, V_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape=(q_len, V_DIM),
        strides=(q_head * V_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_DIM),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + q_start * q_head + start_qh,
        shape=(q_len,),
        strides=(q_head,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    q_arg_block_ptr = tl.make_block_ptr(
        base=q_attn_arg_ptr + q_start,
        shape=(q_len,),
        strides=(1,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    k_arg_block_ptr = tl.make_block_ptr(
        base=k_attn_arg_ptr + k_start,
        shape=(k_len,),
        strides=(1,),
        offsets=(begin,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
    maximum = tl.full((BLOCK_M,), value=-(2**30), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    q = _load_block(q_block_ptr, False, True)
    q_attn_arg = _load_block(q_arg_block_ptr, False, True)

    for start_n in range(begin, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_attn_arg = _load_block(k_arg_block_ptr, False, True)
        offset_n = start_n + tl.arange(0, BLOCK_N)
        mask = _attention_mask(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            k = _load_block(k_block_ptr, True, False)
            score = tl.dot(q, k)
            score = tl.where(mask & ((offset_n < k_len)[None, :]), score, -(2**30))
            new_maximum = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.math.exp2((maximum - new_maximum) * qk_scale)
            probability = tl.math.exp2((score - new_maximum[:, None]) * qk_scale)
            probability_sum = tl.sum(probability, 1)
            acc *= alpha[:, None]
            v = _load_block(v_block_ptr, False, True)
            acc += tl.dot(probability.to(dtype), v)
            denominator = denominator * alpha + probability_sum
            maximum = new_maximum
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        k_arg_block_ptr = tl.advance(k_arg_block_ptr, (BLOCK_N,))

    acc /= denominator[:, None]
    log_sum_exp = maximum * scale + tl.log(denominator)
    _store_block(o_block_ptr, acc.to(dtype), False, True)
    _store_block(l_block_ptr, log_sum_exp, False, True)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2)],
    key=[],
)
@triton.jit
def softcap_kernel(x_ptr, y_ptr, n_elements, softcap, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    value = x.to(tl.float32) / softcap
    tanh = 2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0
    y = softcap * tanh.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return (x / (1.0 + tl.exp(-x))).to(dtype)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        )
    ],
    key=[],
)
@triton.jit
def fused_swiglu_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    g_ptr,
    fc_ptr,
    M,
    N,
    K,
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = y_ptr.type.element_ty
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + offset_xm[:, None] * K + offset_k[None, :]
    w_g_ptrs = w_g_ptr + offset_k[:, None] * N + offset_wn[None, :]
    w_fc_ptrs = w_fc_ptr + offset_k[:, None] * N + offset_wn[None, :]
    b_g = tl.load(b_g_ptr + offset_wn, mask=offset_wn < N, other=0.0)
    b_fc = tl.load(b_fc_ptr + offset_wn, mask=offset_wn < N, other=0.0)

    accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offset_k < K - k * BLOCK_SIZE_K
        x = tl.load(x_ptrs, mask=mask[None, :], other=0.0)
        w_g = tl.load(w_g_ptrs, mask=mask[:, None], other=0.0)
        w_fc = tl.load(w_fc_ptrs, mask=mask[:, None], other=0.0)
        accumulator_g = tl.dot(x, w_g, accumulator_g)
        accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
        x_ptrs += BLOCK_SIZE_K
        w_g_ptrs += BLOCK_SIZE_K * N
        w_fc_ptrs += BLOCK_SIZE_K * N

    accumulator_g = (accumulator_g + b_g[None, :]).to(dtype)
    accumulator_fc = (accumulator_fc + b_fc[None, :]).to(dtype)
    y = (_fast_silu(accumulator_g).to(tl.float32) * accumulator_fc.to(tl.float32)).to(dtype)
    offset_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + N * offset_ym[:, None] + offset_yn[None, :]
    y_mask = (offset_ym[:, None] < M) & (offset_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)
    if IS_TRAINING:
        tl.store(g_ptr + N * offset_ym[:, None] + offset_yn[None, :], accumulator_g, mask=y_mask)
        tl.store(fc_ptr + N * offset_ym[:, None] + offset_yn[None, :], accumulator_fc, mask=y_mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 16}, num_warps=16, num_stages=3)],
    key=[],
)
@triton.jit
def rope_kernel(
    in_ptr,
    pos_ptr,
    cu_seqlens,
    out_ptr,
    head,
    base,
    DIM: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    start_b = tl.program_id(1)
    start_h = tl.program_id(2)
    begin = tl.load(cu_seqlens + start_b)
    length = tl.load(cu_seqlens + start_b + 1) - begin
    if start_m * BLOCK_M >= length:
        return

    begin = begin.to(tl.int64)
    input_base = in_ptr + begin * head * DIM + start_h * DIM
    output_base = out_ptr + begin * head * DIM + start_h * DIM
    x0_block_ptr = tl.make_block_ptr(
        base=input_base,
        shape=(length, DIM),
        strides=(head * DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM // 2),
        order=(1, 0),
    )
    y0_block_ptr = tl.make_block_ptr(
        base=input_base,
        shape=(length, DIM),
        strides=(head * DIM, 1),
        offsets=(start_m * BLOCK_M, DIM // 2),
        block_shape=(BLOCK_M, DIM // 2),
        order=(1, 0),
    )
    x1_block_ptr = tl.make_block_ptr(
        base=output_base,
        shape=(length, DIM),
        strides=(head * DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM // 2),
        order=(1, 0),
    )
    y1_block_ptr = tl.make_block_ptr(
        base=output_base,
        shape=(length, DIM),
        strides=(head * DIM, 1),
        offsets=(start_m * BLOCK_M, DIM // 2),
        block_shape=(BLOCK_M, DIM // 2),
        order=(1, 0),
    )
    pos_block_ptr = tl.make_block_ptr(
        base=pos_ptr + begin,
        shape=(length,),
        strides=(1,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )
    x0 = tl.load(x0_block_ptr, boundary_check=(0,))
    y0 = tl.load(y0_block_ptr, boundary_check=(0,))
    pos = tl.load(pos_block_ptr, boundary_check=(0,))
    offset_n = tl.arange(0, DIM // 2)
    inv_freq = tl.exp(tl.log(base) * (-2.0 / DIM * offset_n))
    freqs = pos[:, None] * inv_freq[None, :]
    sin = tl.sin(freqs)
    cos = tl.cos(freqs)
    if REVERSE:
        sin = -sin
    dtype = in_ptr.type.element_ty
    tl.store(x1_block_ptr, (x0 * cos - y0 * sin).to(dtype), boundary_check=(0,))
    tl.store(y1_block_ptr, (x0 * sin + y0 * cos).to(dtype), boundary_check=(0,))


def _build_constant_args():
    constants = {}

    def assign(indices, value):
        for index in indices:
            constants[index] = value.copy()

    assign((1, 30, 43, 56, 69, 82), {"DIM": 512, "ALIGN": 16})
    assign((3, 4, 32, 33, 45, 46, 58, 59, 71, 72, 83, 84, 85), {"DIM": 1, "ALIGN": 16})
    constants[81] = {}
    assign(
        (5, 9, 13, 17, 21, 38, 88, 94, 100, 106, 112),
        {
            "q_head": 8,
            "kv_head": 8,
            "scale": 0.125,
            "QK_DIM": 64,
            "V_DIM": 64,
            "MASK_FN": 1,
            "SPARSE_OPT": False,
            "DTYPE": 19,
        },
    )
    assign(
        (25, 118),
        {
            "q_head": 16,
            "kv_head": 16,
            "scale": 0.17677669529663687,
            "QK_DIM": 32,
            "V_DIM": 32,
            "MASK_FN": 1,
            "SPARSE_OPT": False,
            "DTYPE": 19,
        },
    )
    assign(
        (34, 47, 51, 60, 64, 73, 77),
        {
            "q_head": 4,
            "kv_head": 4,
            "scale": 0.08838834764831843,
            "QK_DIM": 128,
            "V_DIM": 128,
            "MASK_FN": 1,
            "SPARSE_OPT": False,
            "DTYPE": 19,
        },
    )
    assign(
        (
            6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
            35, 37, 39, 41, 48, 50, 52, 54, 61, 63, 65, 67,
            74, 76, 78, 80, 89, 91, 95, 97, 101, 103, 107, 109,
            113, 115, 119, 121,
        ),
        {"softcap": 50},
    )
    assign(
        (7, 11, 15, 19, 23, 27, 90, 96, 102, 108, 114, 120),
        {"N": 1024, "K": 512, "IS_TRAINING": False},
    )
    assign(
        (36, 40, 49, 53, 62, 66, 75, 79),
        {"N": 512, "K": 512, "IS_TRAINING": False},
    )
    assign(
        (86, 87, 92, 93, 98, 99, 104, 105, 110, 111),
        {"head": 8, "base": 10000.0, "DIM": 64, "REVERSE": False},
    )
    assign(
        (116, 117),
        {"head": 16, "base": 10000.0, "DIM": 32, "REVERSE": False},
    )
    return constants


_MODEL_CONSTANT_ARGS = _build_constant_args()


def launch_model_kernel(
    *,
    kernel_idx,
    constant_args_idx,
    grid,
    tma_descriptor_metadata,
    kwargs,
    tensors_to_clone,
):
    if tma_descriptor_metadata:
        raise RuntimeError("TMA descriptors are not supported by this runnable")

    if kernel_idx == 0:
        kernel = position_offset_kernel

        def launch_grid(meta):
            if meta["BLOCK"] == 128:
                return grid[0]
            if meta["BLOCK"] == 256:
                return grid[1]
            if meta["BLOCK"] == 512:
                return grid[2]
            return grid[3]

    elif kernel_idx == 1:
        kernel = nested_concat_kernel

        def launch_grid(meta):
            return (meta["NUM_SM"], 1, 1)

    elif kernel_idx == 2:
        kernel = position_concat_kernel

        def launch_grid(meta):
            return (meta["NUM_SM"], 1, 1)

    elif kernel_idx == 3:
        kernel = flash_attention_kernel
        launch_grid = grid[0]
    elif kernel_idx == 4:
        kernel = softcap_kernel
        launch_grid = grid[0]
    elif kernel_idx == 5:
        kernel = fused_swiglu_kernel
        launch_grid = grid[0]
    elif kernel_idx == 6:
        kernel = rope_kernel
        launch_grid = grid[0]
    else:
        raise RuntimeError(f"unknown model kernel index: {kernel_idx}")

    kernel[launch_grid](**kwargs, **_MODEL_CONSTANT_ARGS[constant_args_idx])
    return {name: kwargs[name] for name in tensors_to_clone}


def register_model_kernels():
    kernels = (
        position_offset_kernel,
        nested_concat_kernel,
        position_concat_kernel,
        flash_attention_kernel,
        softcap_kernel,
        fused_swiglu_kernel,
        rope_kernel,
    )
    kernel_side_table.reset_table()
    for expected_index, kernel in enumerate(kernels):
        actual_index = kernel_side_table.add_kernel(kernel)
        if actual_index != expected_index:
            raise RuntimeError(f"kernel index mismatch: expected {expected_index}, got {actual_index}")
