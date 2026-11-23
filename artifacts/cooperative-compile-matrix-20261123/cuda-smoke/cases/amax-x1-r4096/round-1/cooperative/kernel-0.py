def triton_unk_fused_add_amax_0(in_ptr0, in_ptr1, out_ptr0, sem_ptr, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT : tl.constexpr):
    xnumel = 1
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = triton_helpers.constexpr_next_power_of_2((4096 + RSPLIT - 1) // RSPLIT)
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
    RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
    HAS_RSPLIT: tl.constexpr = RSPLIT > 1
    rsplit_id = tl.program_id(0)
    num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
    rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
    rsplit_start = rsplit_chunk * rsplit_id
    rsplit_end = rsplit_chunk * (rsplit_id + 1)
    rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :] + rsplit_start
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    rsplit_arange = tl.arange(0, RSPLIT_NEXT_POWER_OF_2)[None, :]
    if RSPLIT_IS_POWER_OF_2:
        rsplit_mask: tl.constexpr = None
    else:
        rsplit_mask = rsplit_arange < RSPLIT
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    if HAS_RSPLIT:
        tmp6_ws = (ws_ptr + 0).to(tl.pointer_type(tl.float32))
        tl.store(tmp6_ws + (xindex * RSPLIT + rsplit_id), tmp6, None)
    if HAS_RSPLIT:
        triton_helpers.x_grid_barrier(sem_ptr + tl.program_id(1))
    if HAS_RSPLIT:
        tmp6_peers = tl.load(tmp6_ws + (xindex * RSPLIT + rsplit_arange), rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, float("-inf")))
        tmp6 = triton_helpers.max2(tmp6_peers, 1)[:, None]
    if rsplit_id == (0 % RSPLIT):
        tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp6, None)
