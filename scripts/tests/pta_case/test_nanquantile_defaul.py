import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"
import torch
import torch_npu
import re
import subprocess
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
from packaging import version

current_version = version.parse(torch.__version__.split('+')[0])
pt29__version = version.parse('2.9')
pt212__version = version.parse('2.12')

os.system("export TORCH_COMPILE_DEBUG=1")
os.system("rm -rf /tmp/torchinductor_root")
os.system("rm -rf ./torch_compile_debug")
# PyTorch 2.12 introduced this FX baseline in be7dbd8a340 (#176804).
if current_version >= pt212__version:
    fx_content = """class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3]"):
    unsqueeze: "f32[3, 4, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
    permute: "f32[1, 4, 3]" = torch.ops.aten.permute.default(unsqueeze, [2, 1, 0]);  unsqueeze = None
    sort = torch.ops.aten.sort.default(permute);  permute = None
    getitem: "f32[1, 4, 3]" = sort[0];  sort = None
    view: "f32[4, 3]" = torch.ops.aten.view.default(getitem, [4, 3]);  getitem = None
    isnan: "b8[4, 3]" = torch.ops.aten.isnan.default(view)
    logical_not: "b8[4, 3]" = torch.ops.aten.logical_not.default(isnan);  isnan = None
    sum_1: "i64[4, 1]" = torch.ops.aten.sum.dim_IntList(logical_not, [-1], True);  logical_not = None
    sub: "i64[4, 1]" = torch.ops.aten.sub.Scalar(sum_1, 1);  sum_1 = None
    mul: "f32[4, 3]" = torch.ops.aten.mul.Tensor(arg1_1, sub);  arg1_1 = sub = None
    lt: "b8[4, 3]" = torch.ops.aten.lt.Scalar(mul, 0)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided,
    device = device(type='npu', index=0), pin_memory = False)
    where: "f32[4, 3]" = torch.ops.aten.where.self(lt, full_default, mul);  lt = full_default = mul = None
    convert_element_type: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(where, torch.int64)
    gather: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type)
    sub_1: "f32[4, 3]" = torch.ops.aten.sub.Tensor(where, convert_element_type);  convert_element_type = None
    ceil: "f32[4, 3]" = torch.ops.aten.ceil.default(where);  where = None
    convert_element_type_1: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(ceil, torch.int64);  ceil =
    None
    gather_1: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type_1);  view =
    convert_element_type_1 = None
    sub_2: "f32[4, 3]" = torch.ops.aten.sub.Tensor(gather_1, gather)
    abs_1: "f32[4, 3]" = torch.ops.aten.abs.default(sub_1)
    ge: "b8[4, 3]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
    sub_3: "f32[4, 3]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    neg: "f32[4, 3]" = torch.ops.aten.neg.default(sub_3);  sub_3 = None
    where_1: "f32[4, 3]" = torch.ops.aten.where.self(ge, neg, sub_1);  neg = sub_1 = None
    where_2: "f32[4, 3]" = torch.ops.aten.where.self(ge, gather_1, gather);  ge = gather_1 = gather = None
    addcmul: "f32[4, 3]" = torch.ops.aten.addcmul.default(where_2, where_1, sub_2);  where_2 = where_1 = sub_2 =
    None
    unsqueeze_1: "f32[1, 4, 3]" = torch.ops.aten.unsqueeze.default(addcmul, 0);  addcmul = None
    permute_1: "f32[3, 4, 1]" = torch.ops.aten.permute.default(unsqueeze_1, [2, 1, 0]);  unsqueeze_1 = None
    squeeze: "f32[3, 4]" = torch.ops.aten.squeeze.dim(permute_1, -1);  permute_1 = None
    return (squeeze,)"""
elif current_version >= pt29__version:
    fx_content ="""class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3]"):
        # File: /data/home/z60040679/graphnew/test_api.py:195 in fn, code: return torch.ops.aten.nanquantile.default(a, q, dim=0, keepdim=False)
        unsqueeze: "f32[3, 4, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, -1);  arg0_1 = None
        permute: "f32[1, 4, 3]" = torch.ops.aten.permute.default(unsqueeze, [2, 1, 0]);  unsqueeze = None
        sort = torch.ops.aten.sort.default(permute);  permute = None
        getitem: "f32[1, 4, 3]" = sort[0];  sort = None
        view: "f32[4, 3]" = torch.ops.aten.view.default(getitem, [4, 3]);  getitem = None
        isnan: "b8[4, 3]" = torch.ops.aten.isnan.default(view)
        logical_not: "b8[4, 3]" = torch.ops.aten.logical_not.default(isnan);  isnan = None
        sum_1: "i64[4, 1]" = torch.ops.aten.sum.dim_IntList(logical_not, [-1], True);  logical_not = None
        sub: "i64[4, 1]" = torch.ops.aten.sub.Scalar(sum_1, 1);  sum_1 = None
        mul: "f32[4, 3]" = torch.ops.aten.mul.Tensor(arg1_1, sub);  arg1_1 = sub = None
        lt: "b8[4, 3]" = torch.ops.aten.lt.Scalar(mul, 0)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[4, 3]" = torch.ops.aten.where.self(lt, full_default, mul);  lt = full_default = mul = None
        convert_element_type: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(where, torch.int64)
        gather: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type)
        sub_1: "f32[4, 3]" = torch.ops.aten.sub.Tensor(where, convert_element_type);  convert_element_type = None
        ceil: "f32[4, 3]" = torch.ops.aten.ceil.default(where);  where = None
        convert_element_type_1: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(ceil, torch.int64);  ceil = None
        gather_1: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type_1);  view = convert_element_type_1 = None
        abs_1: "f32[4, 3]" = torch.ops.aten.abs.default(sub_1)
        ge: "b8[4, 3]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
        sub_2: "f32[4, 3]" = torch.ops.aten.sub.Tensor(sub_1, 1)
        where_1: "f32[4, 3]" = torch.ops.aten.where.self(ge, sub_2, sub_1);  sub_2 = sub_1 = None
        where_2: "f32[4, 3]" = torch.ops.aten.where.self(ge, gather_1, gather);  ge = None
        sub_3: "f32[4, 3]" = torch.ops.aten.sub.Tensor(gather_1, gather);  gather_1 = gather = None
        mul_1: "f32[4, 3]" = torch.ops.aten.mul.Tensor(where_1, sub_3);  where_1 = sub_3 = None
        add: "f32[4, 3]" = torch.ops.aten.add.Tensor(mul_1, where_2);  mul_1 = where_2 = None
        unsqueeze_1: "f32[1, 4, 3]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        permute_1: "f32[3, 4, 1]" = torch.ops.aten.permute.default(unsqueeze_1, [2, 1, 0]);  unsqueeze_1 = None
        squeeze: "f32[3, 4]" = torch.ops.aten.squeeze.dim(permute_1, -1);  permute_1 = None
        return (squeeze,)"""
else:
    fx_content = """class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "f32[3]", arg1_1: "f32[3, 4]"):
             # File: /data/z60040679/test_api.py:195 in fn, code: return torch.ops.aten.nanquantile.default(a, q, dim=0, keepdim=False)
            unsqueeze: "f32[3, 4, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, -1);  arg1_1 = None
            permute: "f32[1, 4, 3]" = torch.ops.aten.permute.default(unsqueeze, [2, 1, 0]);  unsqueeze = None
            sort = torch.ops.aten.sort.default(permute);  permute = None
            getitem: "f32[1, 4, 3]" = sort[0];  sort = None
            view: "f32[4, 3]" = torch.ops.aten.view.default(getitem, [4, 3]);  getitem = None
            isnan: "b8[4, 3]" = torch.ops.aten.isnan.default(view)
            logical_not: "b8[4, 3]" = torch.ops.aten.logical_not.default(isnan);  isnan = None
            sum_1: "i64[4, 1]" = torch.ops.aten.sum.dim_IntList(logical_not, [-1], True);  logical_not = None
            sub: "i64[4, 1]" = torch.ops.aten.sub.Scalar(sum_1, 1);  sum_1 = None
            mul: "f32[4, 3]" = torch.ops.aten.mul.Tensor(arg0_1, sub);  arg0_1 = sub = None
            lt: "b8[4, 3]" = torch.ops.aten.lt.Scalar(mul, 0)
            full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
            where: "f32[4, 3]" = torch.ops.aten.where.self(lt, full_default, mul);  lt = full_default = mul = None
            convert_element_type: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(where, torch.int64)
            gather: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type)
            sub_1: "f32[4, 3]" = torch.ops.aten.sub.Tensor(where, convert_element_type);  convert_element_type = None
            ceil: "f32[4, 3]" = torch.ops.aten.ceil.default(where);  where = None
            convert_element_type_1: "i64[4, 3]" = torch.ops.prims.convert_element_type.default(ceil, torch.int64);  ceil = None
            gather_1: "f32[4, 3]" = torch.ops.aten.gather.default(view, -1, convert_element_type_1);  view = convert_element_type_1 = None
            abs_1: "f32[4, 3]" = torch.ops.aten.abs.default(sub_1)
            ge: "b8[4, 3]" = torch.ops.aten.ge.Scalar(abs_1, 0.5);  abs_1 = None
            sub_2: "f32[4, 3]" = torch.ops.aten.sub.Tensor(sub_1, 1)
            where_1: "f32[4, 3]" = torch.ops.aten.where.self(ge, sub_2, sub_1);  sub_2 = sub_1 = None
            where_2: "f32[4, 3]" = torch.ops.aten.where.self(ge, gather_1, gather);  ge = None
            sub_3: "f32[4, 3]" = torch.ops.aten.sub.Tensor(gather_1, gather);  gather_1 = gather = None
            mul_1: "f32[4, 3]" = torch.ops.aten.mul.Tensor(where_1, sub_3);  where_1 = sub_3 = None
            add: "f32[4, 3]" = torch.ops.aten.add.Tensor(mul_1, where_2);  mul_1 = where_2 = None
            unsqueeze_1: "f32[1, 4, 3]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
            permute_1: "f32[3, 4, 1]" = torch.ops.aten.permute.default(unsqueeze_1, [2, 1, 0]);  unsqueeze_1 = None
            squeeze: "f32[3, 4]" = torch.ops.aten.squeeze.dim(permute_1, -1);  permute_1 = None
            return (squeeze,)
            """


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
    @parametrize('shape', [(3, 4)])
    @parametrize('dtype', ['float32'])
    def test_nanquantile_defaul(self, shape, dtype, dynamic):
        a = torch.randn(shape, dtype=torch.float32, device="npu")
        a[0, 0] = float('nan')
        q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device="npu")

        def fn(a, q):
            return torch.ops.aten.nanquantile.default(a, q, dim=0, keepdim=False)

        r1 = fn(a, q)
        func = torch.compile(fn, backend="inductor", dynamic=dynamic)
        r, codes = run_and_get_code(func, a, q)
        self.assertEqual(r, r1, atol=1e-3, rtol=1e-3)
        self.assertTrue('nanquantile' in codes[0])

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
