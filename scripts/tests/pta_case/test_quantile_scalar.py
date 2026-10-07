import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"
import torch
import torch_npu
import re
import subprocess
from packaging import version
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutilsops import TestUtilsOps
from torch._inductor.utils import run_and_get_code
import torch.nn.functional as F
from torch._dynamo.exc import TorchRuntimeError
import torch_npu._inductor

current_version = version.parse(torch.__version__.split('+')[0])
pt212__version = version.parse('2.12')

os.system("export TORCH_COMPILE_DEBUG=1")
os.system("rm -rf /tmp/torchinductor_root")
os.system("rm -rf ./torch_compile_debug")

# PyTorch 2.12 introduced this FX baseline in be7dbd8a340 (#176804).
if current_version >= pt212__version:
    fx_content = """class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 10]"):
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.5, dtype = torch.float32, layout =
    torch.strided, device = device(type='npu', index=0))
    sort = torch.ops.aten.sort.default(arg0_1);  arg0_1 = None
    getitem: "f32[3, 10]" = sort[0];  sort = None
    mul: "f32[]" = torch.ops.aten.mul.Scalar(scalar_tensor, 9);  scalar_tensor = None
    isnan: "b8[3, 10]" = torch.ops.aten.isnan.default(getitem)
    any_1: "b8[3, 1]" = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
    expand: "f32[3, 1]" = torch.ops.aten.expand.default(mul, [3, 1]);  mul = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 9.0, dtype = torch.float32, layout = torch.strided,
    device = device(type='npu', index=0), pin_memory = False)
    where: "f32[3, 1]" = torch.ops.aten.where.self(any_1, full_default, expand);  any_1 = full_default = expand =
    None
    convert_element_type: "i64[3, 1]" = torch.ops.prims.convert_element_type.default(where, torch.int64)
    gather: "f32[3, 1]" = torch.ops.aten.gather.default(getitem, -1, convert_element_type)
    sub: "f32[3, 1]" = torch.ops.aten.sub.Tensor(where, convert_element_type);  convert_element_type = None
    ceil: "f32[3, 1]" = torch.ops.aten.ceil.default(where);  where = None
    convert_element_type_1: "i64[3, 1]" = torch.ops.prims.convert_element_type.default(ceil, torch.int64);  ceil =
    None
    gather_1: "f32[3, 1]" = torch.ops.aten.gather.default(getitem, -1, convert_element_type_1);  getitem =
    convert_element_type_1 = None
    sub_1: "f32[3, 1]" = torch.ops.aten.sub.Tensor(gather_1, gather)
    abs_1: "f32[3, 1]" = torch.ops.aten.abs.default(sub)
    ge: "b8[3, 1]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
    sub_2: "f32[3, 1]" = torch.ops.aten.sub.Tensor(1.0, sub)
    neg: "f32[3, 1]" = torch.ops.aten.neg.default(sub_2);  sub_2 = None
    where_1: "f32[3, 1]" = torch.ops.aten.where.self(ge, neg, sub);  neg = sub = None
    where_2: "f32[3, 1]" = torch.ops.aten.where.self(ge, gather_1, gather);  ge = gather_1 = gather = None
    addcmul: "f32[3, 1]" = torch.ops.aten.addcmul.default(where_2, where_1, sub_1);  where_2 = where_1 = sub_1 =
    None
    squeeze: "f32[3]" = torch.ops.aten.squeeze.dim(addcmul, -1);  addcmul = None
    return (squeeze,)"""
else:
    fx_content = """class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 10]"):
             # File: /data/home/z60040679/graphnew/test_api.py:154 in fn, code: return torch.ops.aten.quantile.scalar(
            scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.5, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0))
            sort = torch.ops.aten.sort.default(arg0_1);  arg0_1 = None
            getitem: "f32[3, 10]" = sort[0];  sort = None
            mul: "f32[]" = torch.ops.aten.mul.Scalar(scalar_tensor, 9);  scalar_tensor = None
            isnan: "b8[3, 10]" = torch.ops.aten.isnan.default(getitem)
            any_1: "b8[3, 1]" = torch.ops.aten.any.dim(isnan, -1, True);  isnan = None
            expand: "f32[3, 1]" = torch.ops.aten.expand.default(mul, [3, 1]);  mul = None
            full_default: "f32[]" = torch.ops.aten.full.default([], 9.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
            where: "f32[3, 1]" = torch.ops.aten.where.self(any_1, full_default, expand);  any_1 = full_default = expand = None
            convert_element_type: "i64[3, 1]" = torch.ops.prims.convert_element_type.default(where, torch.int64)
            gather: "f32[3, 1]" = torch.ops.aten.gather.default(getitem, -1, convert_element_type)
            sub: "f32[3, 1]" = torch.ops.aten.sub.Tensor(where, convert_element_type);  convert_element_type = None
            ceil: "f32[3, 1]" = torch.ops.aten.ceil.default(where);  where = None
            convert_element_type_1: "i64[3, 1]" = torch.ops.prims.convert_element_type.default(ceil, torch.int64);  ceil = None
            gather_1: "f32[3, 1]" = torch.ops.aten.gather.default(getitem, -1, convert_element_type_1);  getitem = convert_element_type_1 = None
            abs_1: "f32[3, 1]" = torch.ops.aten.abs.default(sub)
            ge: "b8[3, 1]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
            sub_1: "f32[3, 1]" = torch.ops.aten.sub.Tensor(sub, 1)
            where_1: "f32[3, 1]" = torch.ops.aten.where.self(ge, sub_1, sub);  sub_1 = sub = None
            where_2: "f32[3, 1]" = torch.ops.aten.where.self(ge, gather_1, gather);  ge = None
            sub_2: "f32[3, 1]" = torch.ops.aten.sub.Tensor(gather_1, gather);  gather_1 = gather = None
            mul_1: "f32[3, 1]" = torch.ops.aten.mul.Tensor(where_1, sub_2);  where_1 = sub_2 = None
            add: "f32[3, 1]" = torch.ops.aten.add.Tensor(mul_1, where_2);  mul_1 = where_2 = None
            squeeze: "f32[3]" = torch.ops.aten.squeeze.dim(add, -1);  add = None
            return (squeeze,)"""


@instantiate_parametrized_tests
class TestQuantileScalar(TestUtilsOps):
    def extract_code_without_comments(self, code: str) -> str:
        # 去除注释行（以 # 开头）
        lines = [line.strip() for line in code.strip().split('\n') if not line.strip().startswith('#')]
        # 去除空行
        lines = [line for line in lines if line]
        # 合并为单行，去除多余空格
        cleaned = re.sub(r'\s+', ' ', ' '.join(lines))

        def clean(s: str) -> str:
            # 标准化：统一换行、空格、制表符
            s = s.replace('\r', '').replace('\n', ' ').replace('\t', ' ')
            code = str(' '.join(s.split()))
            return code

        return clean(cleaned)

    @parametrize("dynamic", [False])
    @parametrize("shape", [(3, 10)])
    @parametrize("dtype", [torch.float32])
    def test_quantile_scalar(self, shape, dtype, dynamic):
        x = torch.randn(shape, dtype=dtype, device="npu")

        def fn(x):
            return torch.ops.aten.quantile.scalar(
                x, q=0.5, dim=-1, keepdim=False, interpolation="linear"
            )

        out_eager = fn(x)
        compiled_fn = torch.compile(fn, backend="inductor", dynamic=dynamic)
        out_compiled, codes = run_and_get_code(compiled_fn, x)

        self.assertEqual(out_eager, out_compiled, atol=1e-3, rtol=1e-3)
        self.assertTrue('quantile' in codes[0])

        forward_result = subprocess.run(
            "cat ./torch_compile_debug/run*/torchinductor/model_*.0/fx_graph_readable.py",
            shell=True,
            capture_output=True,
            text=True
        )
        if forward_result.returncode == 0:
            forward_code = forward_result.stdout
        else:
            forward_code = "No forward code found or error occurred."
        fx_contentnew = self.extract_code_without_comments(fx_content)
        resnew = self.extract_code_without_comments(forward_code)
        print("fx_contentnew", fx_contentnew)
        print("resnew", resnew)
        self.assertEqual(fx_contentnew, resnew)


if __name__ == "__main__":
    run_tests()
