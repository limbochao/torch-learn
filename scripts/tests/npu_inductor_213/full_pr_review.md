# PR #47384 完整同步与驱动问题核对

2026-10-09 通过 GitCode API 和 Git 提交图交叉核对 [PR #47384](https://gitcode.com/Ascend/pytorch/pull/47384)。
PR 包含 6 个提交，当前 HEAD 为 `16437be0ffc9d67c8b27855a234243e35385c58b`，
与目标分支的 merge-base 为 `38115969528873d4d98fa26e9e55193e04e1d0bc`。
HEAD 表示全部前序提交累积后的文件树；同步时检出完整 HEAD，并未只应用最后一次提交的 diff。

本地及验证容器均建立了独立、干净的完整源码 worktree。6 个提交均为该 HEAD 的祖先，
Git diff 文件集合与 API 返回结果一致，3 个最终变更文件的本地/远程 SHA-256 一致。
PR 净差异中唯一的生产文件与已安装文件字节一致，其余两个文件从完整源码 worktree 执行。
验证继续使用原 torch/torch_npu/Triton 构建；PR 没有 C++ 或二进制变更，不重新构建整个 wheel。

- [完整提交与文件清单](../../../artifacts/npu-inductor-pr47384-full-20261009/manifest.json)
- [完整 PR 净差异](../../../artifacts/npu-inductor-pr47384-full-20261009/full-pr.patch)
- [完整状态下的驱动探针](../../../artifacts/npu-inductor-pr47384-full-20261009/driver-probe.json)
- [原始 arange1 回归报告](../../../artifacts/npu-inductor-pr47384-full-20261009/arange1-results.json)
- [两个 contrib 文件的执行统计](../../../artifacts/npu-inductor-pr47384-full-20261009/contrib-results.json)

## 全部提交及最终修改

| 提交 | 修改 |
|---|---|
| `7ce9ba7e` | 迁移模式的 Triton 可用性、GPU 类型缓存和驱动适配 |
| `8cc0410c` | 撤销前一提交中的固定 CUDA capability 包装 |
| `5d54b6e5` | 调整 ONNX 测试同步清单 |
| `8008a42e` | 为两个 contrib 文件中的测试类增加 skip |
| `3c87a4ec` | 撤销 ONNX 同步清单修改，与 `5d54b6e5` 抵消 |
| `16437be0` | 默认 benchmark 设备改为 NPU |

最终净差异为：

| 文件 | 影响 |
|---|---|
| `torch_npu/contrib/transfer_to_npu.py` | 全部迁移运行时适配，已包含早先提交的累积结果 |
| `test/contrib/test_host_empty_cache_api.py` | 给测试类增加无条件 skip |
| `test/contrib/test_transfer_to_npu_env.py` | 给两个测试类增加无条件 skip |

此前只同步并重测了第一个文件，没有完整核对全部提交和另外两个测试文件。
这次补齐了完整 PR 的同步与校验；此前的迁移文件 SHA-256 本身与完整 HEAD 相同。

## 完整状态下的复现

使用原验证环境：torch `2.13.0+cpu`、torch_npu `2.13.0+git5fd5ddf`、Triton `3.6.0`。
原始 `test_torchinductor.py::test_arange1` 使用 4 个编译线程、4 个 NPU 预编译线程和新缓存，
仍因 `libcuda.so cannot found!` 失败，setup/call/teardown 报告完整。
两个 contrib 文件使用完整 HEAD 的代码执行，24 项全部按类上的标记跳过，不能作为功能通过的证据。
沿用 `TORCH_TRANSFER_TO_NPU=1`、`TORCH_NPU_DEVICE_CAPABILITY=8.0` 的既有验证配置；
后者仅兼容社区测试的 capability 比较，不代表 NPU 硬件能力。

驱动探针记录：

```text
driver_before                  = NPUDriver
CudaDriver.is_active()         = False
torch.cuda.is_available()      = True
torch.version.hip              = None
_is_backend_active("nvidia")   = True
set_driver_to_gpu()            = AssertionError: libcuda.so cannot found!
```

当前 PyTorch helper 在 `CudaDriver.is_active()` 返回 False 后仍继续检查
`torch.cuda.is_available() and torch.version.hip is None`。
迁移模式把 CUDA 接口映射为 NPU 接口，使这个 NVIDIA 兜底条件成立。
随后 helper 在遍历中先尝试构造 `CudaDriver`；加载 CUDA 库时失败，尚未实际替换已有的 NPUDriver。
另外两处测试 skip 不参与这个运行时判断，因此完整 PR 的结论仍是驱动误选。
探针中的 helper 文件 SHA-256 与此前保留的源码证据一致，
见 [具体函数源码](../../../artifacts/npu-inductor-pr47384-20261008/driver-probe.json)。

## PR 内的候选修复

修改仍可集中在 `transfer_to_npu.py::_patch_triton_nvidia_driver`：

1. 保留 Triton 自身的 `CudaDriver.is_active=False`，用于其原生 driver 探测。
2. 同时包装当前 PyTorch 的 `_is_backend_active`：迁移模式下 NVIDIA 返回 False，其余后端沿用原判断。
3. 保留 NPU 可用性与 Triton 安装检查；按实际注册表处理 NVIDIA，避免用 `except Exception: pass` 隐藏适配错误。
4. 对没有 `_is_backend_active` 的旧版 PyTorch，通过属性存在性检查保持兼容。

[候选差异](../../../artifacts/npu-inductor-pr47384-full-20261009/candidate-driver-fix.patch)
基于完整 PR HEAD 生成，尚未写入 PR 或替换安装文件。
这保留了原 `set_driver_to_gpu()` 对活跃 NPU driver 的复用逻辑。
还需要在主进程及编译 worker 中验证该适配都已执行；已有
`TORCH_TRANSFER_TO_NPU=1` 启动入口会在新进程导入 torch_npu 时启用迁移模块。
仅在父进程临时 monkey patch，不能证明子进程也已获得修复。

[进程内机制验证](../../../artifacts/npu-inductor-pr47384-full-20261009/predicate-fix-probe.json)
已确认候选适配使 NVIDIA 判定从 True 变为 False，Ascend 判定仍为 True；
连续两次调用原 `set_driver_to_gpu()` 均成功，保持同一 NPUDriver 实例。
安装文件未修改；该探针没有执行 kernel 或编译 worker，不能计为修改版完整回归通过。

本次只补测了驱动机制、一个原始回归用例和两个 PR 测试文件；没有把这些结果计为重新执行全部 247 项。
