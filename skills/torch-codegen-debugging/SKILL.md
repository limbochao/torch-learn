---
name: torch-codegen-debugging
description: Use this skill for PyTorch/torch_npu/Inductor codegen failures, wrong generated kernels, DSL/Triton lowering bugs, scheduler or tiling issues, compile-time crashes, or accuracy failures that appear only after torch.compile. It guides the investigation from minimal repro, generated code capture, axis/stride/reduction analysis, version comparison, focused instrumentation, fix design, and verification.
---

# Torch Codegen Debugging

Use this workflow when a problem points to codegen rather than eager operator semantics. Typical signals include:

- `torch.compile` fails before runtime execution.
- The eager result is correct but compiled output is wrong.
- A generated DSL/Triton kernel has suspicious indexing, reshape, reduction, mask, or store placement.
- A change in scheduler, tiling, lowering, or kernel type changes the failure.
- Debug trace is missing because the failure happens during codegen.

## Ground Rules

Start with code evidence. Avoid explaining a failure from logs alone when generated code, FX graph, scheduler node metadata, or source code can be inspected.

Keep the repro narrow. A single compiled function, a single generated kernel, or a single scheduler node is easier to reason about than a model-level failure.

Separate facts from hypotheses. Mark inferred behavior as inference until it is confirmed by DSL, source code, or a reproducer.

Preserve user changes. If the worktree is dirty, inspect relevant diffs and avoid reverting unrelated files.

## Investigation Flow

### 1. Classify the Failure

Record:

- Failure stage: Dynamo, AOTAutograd, lowering, scheduling, codegen, kernel compile, runtime launch, or accuracy check.
- Device path: NPU target, kernel type, backend path, fallback behavior.
- Symptom: exception, wrong output, illegal memory access, compile error, missing trace, or precision mismatch.
- Inputs: shapes, strides, dtypes, dynamic/static mode, env flags that affect codegen.

If the failure is an accuracy issue, always compare eager and compiled outputs using the same inputs.

### 2. Build or Reduce a Repro

Prefer a standalone script that:

- Constructs deterministic inputs with `torch.manual_seed`.
- Prints shape, stride, dtype, dynamic flag, and check flag.
- Compiles exactly the target function with `torch.compile`.
- Optionally runs eager and `torch.testing.assert_close`.
- Keeps environment switches explicit through `CHECK`, `DYNAMIC`, or shape variables.

For reduction kernels, preserve non-contiguous stride patterns. A simplified contiguous input can hide the bug.

### 3. Capture the Generated Artifact

Use the least invasive method that works:

- Enable debug trace when the kernel is emitted.
- If trace is not produced because failure happens earlier, add temporary logging around the codegen point.
- Print the full generated kernel, not only the failing line.
- Also capture relevant metadata: `axis_names`, `tiling_axis`, `split_axis`, `low_dims`, `numof_reduction_axis`, `npu_kernel_type`, runtime block args, and static axis values.

Keep debug logging local and easy to remove. Do not leave noisy unconditional logs in final code.

### 4. Read the DSL Mechanically

Map generated symbols back to tensor semantics:

- Identify preserved axes such as `x0`, and reduction axes such as `r0`, `r1`, `r2`.
- Match `tl.load` address formulas to tensor shape and stride.
- Inspect how `base_*`, `loop_*`, and masks define actual tensor layout.
- Track tensor value layout before `permute`, `reshape`, `sum`, and `store`.
- Check whether stores happen after the full reduction or inside a partial loop.

For each suspected bug, name the exact generated lines or operations that are wrong and state the expected behavior.

### 5. Compare Versions or Paths

When behavior differs across versions, devices, or fallback paths:

- Compare generated kernels first.
- Then compare the source code that decides axis order, tiling, kernel type, and reduction dim.
- Avoid treating a working fallback path as proof that the main path is correct.
- If one device can fallback to another kernel type and another cannot, focus on the shared failing path.

### 6. Add Focused Instrumentation

Instrumentation should answer one question at a time:

- Which axis order was selected?
- Which reduction dim was chosen?
- Which tiling config was selected?
- Which node or store index owns the generated line?
- Which branch emitted `prefix`, `post_loop_store`, or `stores`?

Use clear prefixes such as `[DEBUG]` only while debugging. Remove or gate them before finalizing unless the user explicitly wants persistent logs.

### 7. Design the Fix

Tie the fix to the DSL defect:

- If a buffer is initialized in the wrong loop, move the codegen emission site, not just the generated text.
- If a reduction dim is wrong, fix the analysis that computes the dim.
- If a value layout is wrong before flattening, derive the `permute` order from axis metadata instead of hardcoding a case.
- If a tiling config exposes invalid tail behavior, filter or rank configs only under the narrow condition that needs it.

Limit the affected branch. A fix for contiguous multi-reduction should not silently rewrite single-reduction or non-contiguous behavior.

### 8. Verify

Run checks at three levels when possible:

- Local syntax and formatting: `py_compile`, `git diff --check`.
- Repro correctness: `CHECK=1` with the minimal script on the target device.
- Artifact validation: inspect the regenerated DSL and confirm the specific bad behavior is gone.

The final report should include the command, result, generated-kernel evidence, and any tests that could not be run.

## Final Report Shape

Use this order for findings:

1. Repro and symptom.
2. Generated DSL evidence.
3. Source-code cause.
4. Fix summary.
5. Verification result.
6. Remaining risk.

Keep code references concrete. Prefer file paths and function names, and include line numbers when available.
