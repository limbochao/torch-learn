import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"
import torch
import torch_npu
import re
import subprocess
from torch.testing._internal.common_utils import (
    run_tests, parametrize, instantiate_parametrized_tests,
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

# PyTorch 2.12 introduced this FX baseline in 2f93063b00b (#175582).
if current_version >= pt212__version:
    fx_content = """class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
    mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = None
    iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device =
    device(type='npu', index=0), requires_grad = False)
    unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device =
    device(type='npu', index=0), requires_grad = False)
    unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    ge: "b8[2, 2]" = torch.ops.aten.ge.Scalar(sub, 0);  sub = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided,
    device = device(type='npu', index=0), pin_memory = False)
    where: "f32[2, 2]" = torch.ops.aten.where.self(ge, mul, full_default);  ge = mul = full_default = None
    add: "f32[2, 2]" = torch.ops.aten.add.Tensor(where, arg1_1);  where = arg1_1 = None
    return (add,)
    class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77, s77]", arg2_1: "f32[s77, s77]"):
    mul: "f32[s77, s77]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = None
    iota: "i64[s77]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device =
    device(type='npu', index=0), requires_grad = False)
    unsqueeze: "i64[1, s77]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
    iota_1: "i64[s77]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device =
    device(type='npu', index=0), requires_grad = False);  arg0_1 = None
    unsqueeze_1: "i64[s77, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    sub_12: "i64[s77, s77]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    ge_26: "b8[s77, s77]" = torch.ops.aten.ge.Scalar(sub_12, 0);  sub_12 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided,
    device = device(type='npu', index=0), pin_memory = False)
    where: "f32[s77, s77]" = torch.ops.aten.where.self(ge_26, mul, full_default);  ge_26 = mul = full_default = None
    add_18: "f32[s77, s77]" = torch.ops.aten.add.Tensor(where, arg2_1);  where = arg2_1 = None
    return (add_18,)"""
elif current_version >= pt29__version:
    fx_content = """class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # File: /data/home/z60040679/graphnew/test_api.py:132 in op_calc, code: x = x * y
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = None

        # File: /data/home/z60040679/graphnew/test_api.py:133 in op_calc, code: x.triu_()
        iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        ge: "b8[2, 2]" = torch.ops.aten.ge.Scalar(sub, 0);  sub = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[2, 2]" = torch.ops.aten.where.self(ge, mul, full_default);  ge = mul = full_default = None

        # File: /data/home/z60040679/graphnew/test_api.py:134 in op_calc, code: out = x + y
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(where, arg1_1);  where = arg1_1 = None
        return (add,)
        ######dynamic graph#####
        class <lambda>(torch.nn.Module):
            def forward(self, arg0_1: "Sym(s77)", arg1_1: "f32[s77, s77]", arg2_1: "f32[s77, s77]"):
                # File: /data/home/z60040679/graphnew/test_api.py:132 in op_calc, code: x = x * y
                mul: "f32[s77, s77]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = None

                # File: /data/home/z60040679/graphnew/test_api.py:133 in op_calc, code: x.triu_()
                iota: "i64[s77]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
                unsqueeze: "i64[1, s77]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
                iota_1: "i64[s77]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False);  arg0_1 = None
                unsqueeze_1: "i64[s77, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
                sub_12: "i64[s77, s77]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
                ge_22: "b8[s77, s77]" = torch.ops.aten.ge.Scalar(sub_12, 0);  sub_12 = None
                full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
                where: "f32[s77, s77]" = torch.ops.aten.where.self(ge_22, mul, full_default);  ge_22 = mul = full_default = None

                # File: /data/home/z60040679/graphnew/test_api.py:134 in op_calc, code: out = x + y
                add_18: "f32[s77, s77]" = torch.ops.aten.add.Tensor(where, arg2_1);  where = arg2_1 = None
                return (add_18,)"""
else:
    fx_content = """class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
             # File: /data/home/z60040679/graphnew/test_api.py:132 in op_calc, code: x = x * y
            mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = None

             # File: /data/home/z60040679/graphnew/test_api.py:133 in op_calc, code: x.triu_()
            iota: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
            unsqueeze: "i64[1, 2]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
            iota_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
            unsqueeze_1: "i64[2, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
            sub: "i64[2, 2]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
            ge: "b8[2, 2]" = torch.ops.aten.ge.Scalar(sub, 0);  sub = None
            full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
            where: "f32[2, 2]" = torch.ops.aten.where.self(ge, mul, full_default);  ge = mul = full_default = None

             # File: /data/home/z60040679/graphnew/test_api.py:134 in op_calc, code: out = x + y
            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(where, arg1_1);  where = arg1_1 = None
            return (add,)
            ######dynamic graph#####
            class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0, s0]", arg2_1: "f32[s0, s0]"):
             # File: /data/home/z60040679/graphnew/test_api.py:132 in op_calc, code: x = x * y
            mul: "f32[s0, s0]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = None

             # File: /data/home/z60040679/graphnew/test_api.py:133 in op_calc, code: x.triu_()
            iota: "i64[s0]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
            unsqueeze: "i64[1, s0]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
            iota_1: "i64[s0]" = torch.ops.prims.iota.default(arg0_1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False);  arg0_1 = None
            unsqueeze_1: "i64[s0, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
            sub_12: "i64[s0, s0]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
            ge_18: "b8[s0, s0]" = torch.ops.aten.ge.Scalar(sub_12, 0);  sub_12 = None
            full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
            where: "f32[s0, s0]" = torch.ops.aten.where.self(ge_18, mul, full_default);  ge_18 = mul = full_default = None

             # File: /data/home/z60040679/graphnew/test_api.py:134 in op_calc, code: out = x + y
            add_18: "f32[s0, s0]" = torch.ops.aten.add.Tensor(where, arg2_1);  where = arg2_1 = None
            return (add_18,)"""


@instantiate_parametrized_tests
class TestQuantileScalar(TestUtilsOps):
    test_count = 0

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

    @parametrize("dynamic", [False, True])
    @parametrize("shape", [(2, 2)])
    @parametrize("dtype", ["float32"])
    def test_aten_triu_inplace_in_graph(self, shape, dtype, dynamic):
        def op_calc(x, y):
            x = x * y
            x.triu_()
            out = x + y
            return out

        x = self._generate_tensor(shape, dtype).npu()
        y = self._generate_tensor(shape, dtype).npu()

        eager_out = op_calc(x, y)

        compiled_op_calc = torch.compile(op_calc, backend="inductor", dynamic=dynamic)
        inductor_out, codes = run_and_get_code(compiled_op_calc, x, y)

        self.assertEqual(eager_out, inductor_out)
        self.assertTrue('triu' in codes[0])
        TestQuantileScalar.test_count += 1
        if TestQuantileScalar.test_count == 2:
            self.forward_code()

    def forward_code(self):

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
        forward_resultnew = subprocess.run(
            "cat ./torch_compile_debug/run*/torchinductor/model_*.1/fx_graph_readable.py",
            shell=True,
            capture_output=True,
            text=True
        )
        if forward_resultnew.returncode == 0:
            forward_codenew = forward_resultnew.stdout
        else:
            forward_codenew = "No forward code found or error occurred."
        res = (forward_code + "\n\n" +
               "### dynamic Graph ###\n" +
               forward_codenew)
        fx_contentnew = self.extract_code_without_comments(fx_content)
        resnew = self.extract_code_without_comments(res)
        print("fx_contentnew", fx_contentnew)
        print("resnew", resnew)
        self.assertEqual(fx_contentnew, resnew)


if __name__ == "__main__":
    run_tests()
