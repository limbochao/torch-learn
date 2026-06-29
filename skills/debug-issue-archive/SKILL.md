---
name: debug-issue-archive
description: Use this skill when turning a debugging session, bug investigation, codegen analysis, traceback diagnosis, or repro workflow into a durable issue-log document. It helps create sanitized, readable, evidence-backed docs with repro scripts, logs, generated artifacts, root cause, fix strategy, validation, and risk notes. Use it whenever the user asks to archive, document, write up, summarize, or preserve a debug process.
---

# Debug Issue Archive

Use this workflow to turn a debugging session into a document that another engineer can read without the full chat history.

The output should be useful as an issue log, not a transcript. Keep the narrative technical, concrete, and sanitized.

## Before Writing

Collect the minimum evidence needed for a stable record:

- Problem symptom and failure stage.
- Environment class, not personal host details.
- Repro script or exact command.
- Relevant logs, traceback, generated code, or intermediate artifact.
- Source-code references that support the root cause.
- Fix summary and verification result.

If the repro lives outside the target documentation repository, copy or recreate a minimal version inside the repository before linking to it.

## Sanitization

Remove personal or machine-specific data before writing:

- Usernames, home directories, private mount paths, container names, trace run IDs, process IDs.
- Full backend hashes unless they are required to identify the build.
- Internal URLs, tokens, keys, customer data, or private model names.
- Raw logs that contain unrelated environment dumps.

Use generic labels when needed:

- `<workspace>`
- `<repo>`
- `<debug-trace>`
- `<container>`
- `<target-device>`

Do not include a path unless the file exists in the same repository or the path is intentionally generic.

## Document Structure

Use practical section titles. The exact title names can vary, but the document should cover:

- Background.
- Minimal repro.
- Failure artifact or key log.
- Artifact behavior analysis.
- Expected behavior.
- Root cause or fix strategy.
- Impact and risk.
- Verified result.

Avoid over-fragmenting the document. Related points can share one section when it reads better.

## Repro Section

A good repro section includes:

- The script path in the same repository.
- The target function or operation.
- Input shapes, strides, dtypes, and flags.
- The command to run.
- The expected pass/fail behavior.
- Any environment variables used by the script.

Prefer a small script that prints enough context for future debugging. For accuracy issues, include an optional eager comparison controlled by a flag such as `CHECK=1`.

## Artifact Section

Include complete artifacts when completeness matters:

- Full generated kernel body for codegen issues.
- Relevant traceback block for exceptions.
- Full minimal FX/readable graph when the graph structure is the evidence.

Trim only unrelated metadata, and explain what was removed. For generated DSL, keep formatting close to the real output unless readability requires small changes.

## Analysis Style

Write for a technical reader who was not present during the investigation:

- Define local terms once, such as DSL, reduction lane, or tile.
- Explain how to read the artifact before pointing out the bug.
- Connect each conclusion to a code line, generated operation, log message, or repro behavior.
- State what is known and what is inferred.

Avoid chat-like phrasing, filler, and broad claims. Prefer direct statements:

- Good: "`tl.store` is inside `loop_r2`, so each `r2` tile writes a partial result."
- Weak: "It seems like the store might be wrong."

## Fix and Risk

The fix section should answer:

- Which source component changes?
- Which generated behavior changes?
- Which cases are intended to be affected?
- Which cases should remain unchanged?
- What risk remains?

Mention the scope precisely, such as `numof_reduction_axis() > 1 and is_contiguous_reduction()`, rather than saying “multi-reduction” if the implementation is narrower.

## Verification

Record both functional and artifact validation:

- Command used.
- Output marker such as `check=passed`.
- Any local checks such as `py_compile` or `git diff --check`.
- Generated artifact evidence after the fix.

If validation was not possible, say what was not run and why.

## Repository Hygiene

When archiving into a documentation repository:

- Add the repro script under a script or repro directory.
- Link the issue log from the relevant index.
- Keep filenames descriptive and stable.
- Do not overwrite unrelated user changes.
- Run a text scan for personal paths and obvious machine-specific data.

Suggested scan patterns:

```bash
grep -RInE "/home/[^ ]+|/data/[^ ]+|PID|run_[0-9]|root@|localhost|token|secret" docs scripts || true
```

Run Markdown and script checks where available:

```bash
git diff --check
python -m py_compile <repro-script>
```
