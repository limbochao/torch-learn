def triton_poi_fused_avg_pool2d_15(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = ((xindex // ks0) % ks0)
    x2 = xindex // ks1
    tmp0 = tl.load(in_ptr0 + (x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + (triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + x2 + 2*x0 + 2*x1 + x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x1*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4)) + (triton_helpers.div_floor_integer((-1) + ks2,  4))), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 4.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tl.store(out_ptr0 + (x0 + x1 + x2 + x1*(triton_helpers.div_floor_integer((-1) + ks2,  8)) + x2*(triton_helpers.div_floor_integer((-1) + ks2,  8))*(triton_helpers.div_floor_integer((-1) + ks2,  8)) + 2*x2*(triton_helpers.div_floor_integer((-1) + ks2,  8))), tmp8, xmask)
