s0 = 8
s27 = 2
s53 = 4


import os
os.environ['TRITON_CACHE_DIR'] = '/home/l30023782/PTA_213/combo-case-9a86e68-20261114/cache-triton'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/home/l30023782/PTA_213/combo-case-9a86e68-20261114/debug'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/home/l30023782/PTA_213/combo-case-9a86e68-20261114/cache-inductor'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['PYTORCH_TESTING_DEVICE_FOR_CUSTOM'] = 'privateuse1'
os.environ['PYTORCH_TESTING_DEVICE_EXCEPT_FOR'] = 'cuda,cpu'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.suppress_errors = False
torch._dynamo.config.raise_on_ctx_manager_usage = True
torch._dynamo.config.nested_graph_breaks = True
torch._dynamo.config.log_compilation_metrics = False
torch._inductor.config.debug = True
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.post_grad_fusion_options = {'fav3_partition': {}}
torch._inductor.config.implicit_fallbacks = False
torch._inductor.config.loop_reindexing_after_fusion = False
torch._inductor.config.combo_kernels = True
torch._inductor.config.debug_index_asserts = True
torch._inductor.config.aten_distributed_optimizations.collective_estimator = 'benchmark'
torch._inductor.config.aten_distributed_optimizations.compute_estimator = 'benchmark'
torch._inductor.config.aten_distributed_optimizations.enable_fusion_regions = False
torch._inductor.config.aten_distributed_optimizations.pre_bucketing_fsdp_collectives = False
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.generate_intermediate_hooks = True
torch._inductor.config.cpp.min_chunk_size = 1
torch._inductor.config.cpp.march = ''
torch._inductor.config.triton.coalesce_tiling_analysis = False
torch._inductor.config.triton.autotune_pointwise = False
torch._inductor.config.triton.mix_order_reduction = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = True
torch._inductor.config.test_configs.runtime_triton_shape_assert = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.enable_autograd_cache = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.13.0+cpu
# torch cuda version: None
# torch git version: cf30153c4c131c8164ee7798e5022d810682e2cb


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# Ascend950PR_9579 : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        mean = torch.ops.aten.mean.default(arg3_1)
        mean_1 = torch.ops.aten.mean.dim(arg3_1, [-2], True);  arg3_1 = None
        return (mean, mean_1)

def load_args(reader):
    reader.symint(2, expr='s27')  # arg0_1
    reader.symint(4, expr='s53')  # arg1_1
    reader.symint(8, expr='s0')  # arg2_1
    buf0 = reader.storage(None, 4*s0*s53*(s27 - 1) + 4*s0*(s53 - 1) + 4*s0, device=device(type='npu', index=0))
    reader.tensor(buf0, (1, s27, s53, s0), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)