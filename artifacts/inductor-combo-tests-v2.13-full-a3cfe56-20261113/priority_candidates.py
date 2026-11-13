"""Source-based repair candidates; execution status is attached after the full run."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATES = {}


def add(name, priority, order, purpose, missing, entry, acceptance, evidence, kind='功能适配'):
    CANDIDATES[name] = {
        'priority': priority, 'order': order, 'purpose': purpose,
        'source_assessment': missing, 'fix_entry': entry, 'acceptance': acceptance,
        'evidence': evidence, 'kind': kind,
    }


add('test_foreach_basic', 'P1', 10,
    '图输入上的 foreach_add 节点输入距离为 0。',
    '旧的无条件 NPU skip 挡住输入距离检查；这不是 FOREACH 分组融合验收。',
    'test_snode_runtime.py 的 decorator 和 _get_snodes 观测。',
    '取消旧 skip，确认 snodes 非空，再检查每个真实节点的 min/max_input_distance 为 0。',
    ['E10'], '测试入口')
add('test_foreach_after_extern', 'P1', 11,
    'mm extern 深度 0，后续 foreach_add 的真实依赖深度至少为 1。',
    '旧的无条件 NPU skip 挡住依赖距离检查。',
    'test_snode_runtime.py 的 decorator 和 _get_snodes 观测。',
    '保留原矩阵乘输入、非空节点、最小距离 0 和最大距离至少 1 的检查。',
    ['E10'], '测试入口')
add('test_dynamic_shapes_persistent_reduction_mixed_x_dim_xpu', 'P1', 12,
    '不同 X 长度的 sum/mean/max 动态归约及编译期 autotune。',
    'requires_xpu 门控；主体没有 XPU 硬件 intrinsic，不能认作 NPU 功能不支持。',
    '测试设备门控及真实 NPU autotune/launch 观测。',
    '用 NPU 通用入口执行原动态形状约束及数值比较；保留编译期 autotune 目标。',
    ['E9', 'E3'], '测试入口')
add('test_persistent_reduction_size_hint', 'P1', 20,
    '两个 max/min persistent 成员的候选尺寸覆盖真实归约长度。',
    '原 FileCheck 固定 CUDA x/r0_ 字典；NPU 采用多轴 metadata 和 native heuristic。',
    '测试生成代码检查；NPUComboKernel.select_heuristics。',
    '核对 NPU persistent 模式、两个成员 16/32 的完整 R 及共享候选包络，保留数值比较。',
    ['E1', 'E2'], '测试观测适配')
add('test_sort_in_combo_kernel_forces_persistent_reduction', 'P1', 21,
    '即使一般 persistent 策略被关闭，Combo 的 sort 成员仍满足整行排序需求。',
    '不能用 CUDA heuristic 文本代替 NPU 实际 sort 模式；需按当前首个失败点定界。',
    '测试 force_non_persistent_for_sort、NPU sort lowering 和成员 persistent 分类。',
    '原整数排序与 tanh 数值通过，确认 sort 采用合法完整归约范围而非只更换 decorator 字符串。',
    ['E1', 'E2'], '测试观测／成员模式')
add('test_single_combo_kernels', 'P1', 22,
    '尺寸差异过大的两个 pointwise 保持 standalone，覆盖 benchmark 开关。',
    'Grid1D 是 CUDA 生成 ABI；实际定义数、benchmark probe 和执行次数必须分别核对。',
    'test_single_combo_kernels 的原生 NPU grid 和生成／launch 计数检查。',
    '两种开关都保留数值、两个独立计算和无 Combo 的目标，按真实 wrapper 与 launcher 验收。',
    ['E1', 'E2'], '测试观测适配')
add('test_combo_kernel_yz_overflow', 'P1', 23,
    '大尺寸 permute/clone 的 launch grid 不溢出。',
    '测试使用 CUDA traceEvents 字典和固定三维 grid；NPU 采用一维 dispatch。',
    '测试真实 NPU launcher/grid 观测。',
    '保留大尺寸与数值检查，确认真实 NPU grid 的 X 合法、Y/Z 为 1，排除 autotune launch。',
    ['E1', 'E9'], '测试观测适配')
add('test_combo_kernel_per_config_subkernel_r0_r1', 'P1', 24,
    '具有两条真实归约轴的成员按后端可兼容的分块 ABI 分区。',
    '原断言要求 CUDA 二维归约被排除而产生两个 kernel；NPU 的轴接口需独立核对。',
    '测试 native 成员轴、分区结果及 wrapper；NPUComboKernel.horizontal_partition。',
    '保留 permute 布局和 sum(1,2)，检查实际轴兼容边界与数值，不能仅关闭数量断言。',
    ['E1'], '测试观测／分区契约')
add('test_combo_kernel_per_config_subkernel_poi_3d', 'P1', 25,
    '具有三维非连续切片的 pointwise 按可兼容的轴 ABI 分区。',
    '原数量对应 CUDA 三维 tiling 排除规则；NPU 的实际轴和分区需核对。',
    '测试 native 成员轴、分区结果及 wrapper；NPUComboKernel.horizontal_partition。',
    '保留非连续切片输入和数值，验证原生轴合法性与分区边界，不用裸 kernel 数替代目标。',
    ['E1'], '测试观测／分区契约')
add('test_combo_inductor_meta_has_optimize_mem', 'P1', 30,
    '推理图显式传 optimize_mem=True，声明 autotune 输入备份／恢复的内存优化资格。',
    'NPU Combo 未显式导出 optimize_mem；native benchmark 直接 clone，helper 尚非其消费路径。',
    'NPUComboKernel.jit_line、NPUCachingAutotuner 的输入备份／恢复调用链。',
    '显式推理字段正确；如果接通内存优化消费者，同时验证 mutation、alias 和恢复。',
    ['E2', 'E5'], '图阶段 metadata')
add('test_combo_inductor_meta_optimize_mem_false_in_training_forward', 'P1', 31,
    '训练前向显式传 optimize_mem=False，防止输入保存／恢复破坏训练状态。',
    '训练前向字段未导出，runtime 缺省为 True；当前 native clone 路径不能据此认定输入已污染。',
    'NPUComboKernel.jit_line 的图阶段值及 NPUCachingAutotuner 输入保护。',
    '训练前向 False，推理／反向另行核对；保留数值、mutation 和梯度保护。',
    ['E2', 'E5'], '图阶段 metadata')
add('test_combo_kernel_per_subkernel_reduction_hint', 'P1', 32,
    '分别记录行归约 INNER 与列归约 OUTER 的成员访问分类。',
    'jit_line 仅传首成员 hint，combo_grid_meta 未输出 reduction_hint_i；native 已有成员轴约束。',
    'NPUComboKernel.combo_grid_meta／autotune_subkernel_meta 及候选评价入口。',
    '按实际 stride/归约轴输出两个真实分类，保留数值；分类影响优化时还需验证真实候选与 winner。',
    ['E1', 'E2', 'E3'], '成员 metadata')
add('test_combo_pointwise_combo_grid_meta_has_per_subkernel_fields', 'P1', 40,
    '三个 pointwise 成员分别带有效 load/store/reduction 计数、autotune_hints 和 atomic_add_found。',
    '缺 inductor_meta_i；NPU CSEProxy 的有效 store 路径也未递增 num_store。',
    'NPU CSEProxy.store／store_reduction、统一 standalone 与 Combo 的统计出口。',
    '逐成员核对真实 load/store、reduction=0、优化提示及 atomic 标志，不能填 0 或空字典。',
    ['E2', 'E4', 'E11'], '成员统计')
add('test_combo_reduction_combo_grid_meta_has_per_subkernel_fields', 'P1', 41,
    'sum 和 mean 成员分别携带真实 load、最终 store 与 reduction 数量。',
    '缺 inductor_meta_i；归约计数入口已有，但最终归约写回统计未统一。',
    'NPU CSEProxy.store_reduction 与成员统计出口。',
    '与 standalone 的真实 load/store/reduction 对照，覆盖最终写回、广播及被消除的写回。',
    ['E2', 'E4', 'E11'], '成员统计')
add('test_combo_reduction_deterministic_has_contiguous_rdim_per_subkernel', 'P1', 42,
    '确定性归约携带逐成员 has_loadstore_with_contiguous_rdim。',
    '缺成员字典；native load 未同步社区连续归约标志，不能将默认 False 当真实统计。',
    'NPU load 地址／stride 分析、成员 metadata 出口与 native 确定性候选选择。',
    '按真实连续归约维判断标志，保留数值；不能把缺字段解释成没有确定性归约。',
    ['E2', 'E4'], '成员布局统计')
add('test_combo_per_kernel_inductor_meta_matches_standalone', 'P1', 43,
    'Combo 成员与 standalone 的 load/store/reduction、atomic、no_x_dim、提示及 tiling_scores 一致。',
    '成员统计出口缺失；standalone 也需补完整必需字段，空正则匹配不能充当基准。',
    '统一 native per-kernel metadata，并修测试字段解析与非空校验。',
    '两侧必需字段均非空，真实值按 NPU split 轴和布局对应，修计数后再比较。',
    ['E2', 'E4', 'E11'], '统计一致性')
add('test_combo_kernel_num_gb_and_flop_match_standalone_sum', 'P1', 44,
    'Combo 总访存字节／FLOP 与其 standalone 成员之和相等。',
    'Combo 未聚合 kernel_num_gb，普通 NPU 也未统一导出 kernel_flop。',
    'NPU 字节／FLOP 估算、standalone 出口与 Combo 聚合。',
    '先得到非空有效基准再聚合；本输入常规读写为 16384B，不能以空匹配求和 0 通过。',
    ['E2', 'E12'], '成本统计')
add('test_combo_inductor_meta_has_kernel_num_gb_under_profile_bandwidth', 'P1', 45,
    'profile_bandwidth 只携带真实 kernel_num_gb，避免额外 FLOP 估算。',
    '普通 NPU 已按 profile 条件估算字节，Combo 未聚合并传出。',
    'NPUComboKernel.jit_line 的 profile 条件和成员字节数聚合。',
    'profile 模式有真实总字节、无 kernel_flop，并联验未开 profile 时不输出该字段。',
    ['E2', 'E12'], 'profile 成本统计')
add('test_benchmark_combo_kernel_emits_real_num_gb', 'P1', 46,
    '独立 benchmark 模板得到真实 GB 数，带宽报告才能使用有效分子。',
    'NPU Combo 调用 codegen_kernel_benchmark(num_gb=0)，当前模板收到硬编码零。',
    'NPUComboKernel.codegen_kernel 传真实成员字节估算之和。',
    '正的真实浮点 num_gb、正确的 NPU device 参数及模板实际消费，不能仅把 0 改成 0.0。',
    ['E2', 'E12'], 'benchmark 成本统计')
add('test_triton_kernel_to_post_grad_tracing_combo_kernel', 'P1', 50,
    '每个真实 Combo kernel 能回溯到其 post-grad FX 节点和成员 origins。',
    'define_kernel／codegen_comment 接收 Combo 外壳，而其 node=None，origins 未按真实分区展开。',
    'NPUTritonScheduling.codegen_combo_kernel 按 node_group 展开 get_nodes。',
    '核对实际分区、实际命名和缓存命中；Inductor 与 AOT 两入口都需映射到真实 FX origins。',
    ['E6'], 'provenance')
add('test_combo_kernel_store_mask', 'P2', 60,
    '动态两个 mean 在一个 Combo 中执行，store 使用合法尾部 mask。',
    '旧 NPU skip 阻断验收；解除后须先证明原图形成 Combo，再检查 store mask。',
    'test_torchinductor 的旧 skip、NPU 动态归约轴兼容性及 store mask 生成。',
    '原输入与数值保留，真实一个 Combo，最后写回 mask 覆盖有效输出；不能跳过数量检查直接称 mask 已实现。',
    ['E1', 'E4'], '动态归约融合与 mask')
add('test_fuse_mix_order_reductions_combo_kernels', 'P2', 61,
    '共享输入的行／列归约联合执行，消费者单独执行，其他独立归约形成 Combo。',
    'NPU mix_order_reduction=False；这是共享输入 mixed-order 联合归约能力目标。',
    'NPU mixed-order 调度、lowering／成员资源分析，避免与独立成员 hint 混淆。',
    '原共享 x 图数值正确，证明行列联合归约、消费者隔离及独立 y/z Combo，保留原 3-kernel 目标。',
    ['E1', 'E15'], 'mixed-order 联合归约')
add('test_combo_kernel_scalar_store_broadcast', 'P2', 62,
    '标量加法、向量归约、读取该标量的归约在一次合法计算中组合。',
    '需要协调标量 producer/consumer 依赖和不同成员轴 ABI；不能笼统写成融合策略差异。',
    'NPU scalar write／broadcast、垂直融合边界和 Combo 成员轴兼容性。',
    '原随机输入数值正确，真实一次 launch，标量只作合法写回且消费者读取正确值。',
    ['E1', 'E15'], '标量广播与成员组合')
add('test_combo_kernel_per_config_subkernel_block_size', 'P1', 26,
    '八种 channels_last 转换的逐成员独立分块与实际 grid。',
    '原 eager contiguous(memory_format=channels_last) 是前置阶段；独立成员 tile 和 CUDA profiler ABI 另有目标。',
    '先处理 NPU eager memory-format 路径，再接成员 block ABI 与原生 grid 观测。',
    '原 NCHW 转 channels_last 的 eager／compiled 都正确，再验独立分块与 launch；替代 stride 不代表布局已支持。',
    ['E1', 'E9'], 'eager 布局与独立分块')
add('test_combo_kernel_no_bench_stitched_config', 'P2', 70,
    '无 autotune 时三个成员各自的默认 X 分块与 body/grid 一致。',
    'NPU default_config 使用共享 split block，未实现三个独立 XBLOCK_i 拼接。',
    '成员 block ABI、默认候选生成、dispatch 边界与 launcher 参数映射。',
    '数值通过且无 benchmark，三个真实独立默认配置与 body/grid 同源，不以共享块或空匹配通过。',
    ['E1', 'E3'], '独立默认分块')
add('test_combo_kernel_no_bench_persistent_reduction', 'P2', 71,
    '无 autotune 的三个 persistent 成员保留完整 R，X 默认分块独立。',
    '静态完整 R 生成已有；成员独立 X 默认配置和 dispatch 尚未接通。',
    '成员 X block ABI／默认配置及 persistent R 常量生成。',
    '完整 R 在 body 固定，三个独立 X 默认值与实际 grid/launcher 一致，保留数值和无 benchmark。',
    ['E1', 'E3'], 'persistent 独立默认分块')
add('test_combo_kernel_no_bench_carve_out', 'P2', 72,
    '无 autotune 默认配置不适合 bucketize 时将其分离，另两个 pointwise 形成 Combo。',
    '需核对 native 默认配置接受／拒绝边界，不能直接照搬 CUDA 的候选数量门槛。',
    'NPU 默认候选／资源检查和 Combo partition 的 carve-out 决策。',
    '保留 bucketize 与其他输入，真实一个 standalone 加一个 Combo 且数值正确，说明拒绝理由。',
    ['E1', 'E3'], '默认配置 carve-out')

for name, order, purpose in (
    ('test_combo_kernel_max_autotune', 80, '三个激活成员各自生成候选并在完整 Combo 中顺序评价。'),
    ('test_combo_kernel_max_autotune_with_reduction', 81, 'sum/mean 成员各自的归约候选及顺序评价。'),
    ('test_combo_autotune_many_subkernels', 82, '六个不同工作量／算子的独立成员候选和顺序评价。'),
    ('test_combo_autotune_grouping', 83, '相同 heuristic 输入的成员共享一次调优，不同尺寸分组。'),
):
    add(name, 'P2', order, purpose,
        'native 当前共享 tile 搜索；原目标是逐成员候选／顺序调优及分组，缺社区 combo_tuning_groups 契约。',
        '先接成员 block ABI 和真实 metadata，再实现独立候选、完整 Combo 评价和可观测分组。',
        '保留数值和实际单 Combo，验证真实候选、组成员及 winner；不能只替换日志字符串通过。',
        ['E1', 'E2', 'E3', 'E11'], '逐成员 autotune')
add('test_combo_autotune_grouping_by_metadata_fingerprint', 'P2', 84,
    '相同 metadata 合并一组，尺寸／load 计数／tiling_scores 变化分为两组。',
    '该方法直接调用社区 helper 的模拟 metadata，PASS 也只证明 helper，不能代表 NPU 真实生成路径。',
    '社区 fingerprint helper 的 fixture 及 NPU 成员 metadata／候选连接。',
    '保留四种合并／分离子场景的原断言，并另核对真实 NPU metadata 到分组的连接。',
    ['E2', 'E11'], '分组 fingerprint')
add('test_combo_max_persistent_rblock_gated_on_per_subkernel', 'P2', 85,
    '共享 X 分块时应用联合 R 资源约束，真正独立 X 分块时取消该共享约束。',
    '原 max_persistent_rblock 是社区共享-X/HIP 约束字段；NPU 需验证其真实 UB／成员资源规则。',
    'NPU 共享／独立 block ABI 与每成员资源约束，不移植裸 4096 阈值。',
    '区分实际共享与独立模式，验证合法完整 R、资源受限候选及字段语义；开关 True 不等于独立 X 已实现。',
    ['E1', 'E3', 'E11'], '共享／独立资源门控')
add('test_combo_kernel_coordesc_tunes_largest_subkernel_first', 'P2', 86,
    '逐成员坐标下降先修改工作量最大的第 1 个成员 X 分块。',
    '共享 block 无独立成员坐标，不能验证仅先改变 XBLOCK_1 的目标。',
    '成员独立 ABI、coordesc 字段与工作量排序、完整配置评价。',
    '真实尝试多个不同成员配置，首步只改最大成员，并保持数值；日志缺失不代表数值错误。',
    ['E1', 'E3', 'E11'], '成员 coordinate descent')
add('test_combo_rms_norm_forwards_add_persistent_rblock', 'P2', 87,
    '两个 RMSNorm 成员的候选保留 standalone 的完整 R／大 R 优化资格。',
    '缺逐成员 add_persistent_rblock 出口；NPU 原图可已是完整-R persistent，不能据字段认定缺 persistent。',
    'native standalone／Combo 的实际 R 候选和资源约束，再定义该优化标志的等价消费。',
    '原 (2048,4096) 图数值正确，真实候选与 winner 对照；不能补无消费者标志冒充优化。',
    ['E1', 'E2', 'E3', 'E11'], '成员 R 候选资格')
add('test_combo_triton_meta_has_disable_ftz', 'P2', 90,
    'disable_ftz 开关从 Inductor 传到编译器和实际数学路径，控制非正规数保留。',
    'NPU triton_meta 未导出该字段，当前安装 Triton 的 NPUOptions 未声明它；通用编译选项及数学路径消费仍需定界。',
    'NPU jit_line、Triton Ascend compiler NPUOptions、指数数学 lowering／下层消费。',
    'False/True 字段正确并有真实消费者；验证 exp/exp2 边界，不用只打印开关替代数值能力。',
    ['E2', 'E8'], 'FTZ 配置与数值能力')
add('test_combo_kernel_peak_memory_wide_resnet', 'P3', 100,
    '峰值门控拒绝增加内存的合法 Combo 候选并降低运行峰值。',
    '源码已有 gate；原首次执行同时计入编译／autotune 内存，峰值不降不能直接证明 gate 缺失。',
    '用例候选有效性、正 memory delta 与 warm 运行峰值观测；再核对实际拒绝决策。',
    '证明存在会被门控拒绝的真实 NPU Combo，并用相同原生内存统计隔离运行峰值；不只放宽阈值。',
    ['E7', 'E13'], '门控验收有效性')
add('test_combo_kernel_fusion_within_same_stream', 'P1', 27,
    '同一用户 stream 上的独立 pointwise 可组合，显式同步后消费者结果正确。',
    '需按实际失败阶段核对 NPU stream 归属、同步与 Combo eligibility。',
    'NPU 流的 lowering／scheduler 归属与生成 wrapper。',
    '保留同流、显式同步、数值及至多 2 kernel 目标，不能绕开流操作仅验证计算。',
    ['E1', 'E15'], '流与组合')
add('test_combo_kernels_pointwise_only', 'P2', 63,
    'pointwise_only 开关只允许 pointwise 组合，归约单独执行。',
    '需核对实际 pointwise／reduction 成员是否按开关分区，不以总 kernel 数独立推断开关无效。',
    'scheduler.create_combo_kernel_nodes 和 NPU partition／实际 wrapper。',
    '两种开关数值正确，记录真实成员和调用，保留 pointwise-only 的成员边界。',
    ['E1', 'E15'], '按成员类型门控')
add('test_combo_fp_self_sub_rewrite', 'P1', 28,
    '浮点 c-c 简化在 Combo 内保留正确语义和有效生成代码。',
    '必须按实际数值／结构失败点定界，不能仅增加 tl.where 字符串假通过。',
    'NPU 浮点表达式重写、生成 DSL 和实际 Combo。',
    '原输入逐元素精确相等，并核对 c-c 的数值语义及真实一个 Combo。',
    ['E1', 'E4'], '浮点重写')

HARDWARE = {
    'test_combo_kernel_tma_descriptor_block_name': 'NVIDIA SM90+ TMA descriptor 搬运协议。',
    'test_pdl_codegen_in_combo_kernel': 'NVIDIA SM90+ CUDA grid dependency control 和 launch_pdl。',
    'test_pdl_combo_kernel_pointwise': 'NVIDIA PDL 每成员 wait／launch_dependents 顺序。',
    'test_pdl_combo_kernel_reduction': '类级开启的 NVIDIA PDL 归约 launch 协议。',
    'test_combo_kernel_amd_special_config_args': 'ROCm/HIP 专属 compile options 与 constexpr 分离。',
}

if __name__ == '__main__':
    rows = json.loads((ROOT / 'manifest.json').read_text())
    overlap = {(r['file'], r['class'], r['name'])
               for r in json.loads((ROOT / 'overlap60-jobs.json').read_text())}
    outside = [r for r in rows if (r['file'], r['class'], r['name']) not in overlap]
    assert {r['name'] for r in outside} == set(CANDIDATES) | set(HARDWARE)
    assert not set(CANDIDATES) & set(HARDWARE)
    result = {'draft': True, 'result_source': 'awaiting_current_full_run',
              'candidates': CANDIDATES, 'hardware_exclusions': HARDWARE}
    (ROOT / 'priority-candidates.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(CANDIDATES)} source-based candidates, {len(HARDWARE)} hardware exclusions')
