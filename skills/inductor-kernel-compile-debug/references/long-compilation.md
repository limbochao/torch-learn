# Long compilation, OOM, and candidate deadlines

Use this reference when a model keeps compiling, a compiler process aborts, or
the task requires surviving an SSH interruption. These are supervision rules,
not additional flags implemented by `trace_compile.py`.

## Distinguish the outcomes

- A candidate compiler exception can be expected rejection during autotune.
  Preserve the candidate and inspect the final model result before declaring
  the scenario blocked.
- Host OOM is different from device OOM and compiler UB/resource overflow.
  Identify the affected process, exit signal/traceback and memory evidence.
  UB overflow alone is not evidence of host OOM or a compiler defect.
- No new model log lines does not prove a hang. Check active compiler child
  identity, creation time, candidate ID/config and newly written artifacts.
  Compilation may be progressing through many short candidates.
- Model elapsed time and one candidate's compiler elapsed time are different
  measurements. A long total is not a per-candidate timeout.

## Reduce memory pressure without changing the case

Keep shape, dtype, operators and selected failure config intact. If OOM or a
user-requested memory limit warrants it, reduce compilation/precompilation
concurrency for the task and document the change. Inspect current source for
the supported settings; examples in some NPU stacks include
`TORCHINDUCTOR_COMPILE_THREADS` and `TORCHNPU_PRECOMPILE_THREADS`.
Do not assume these exist or take effect in every version. Do not silently
make the reduced setting a machine-wide default.

For an ordinary exception, replay the captured candidate directly before
rerunning the entire model. Shrinking inputs can stop reproducing the original
failure and must be reported as a separate case. Bound retries by the task's
time/resource budget; repeated OOM with the same settings is not progress.

## Durable monitoring

Use a task-owned directory with command, stack/loaded-source identity, logs,
candidate journal, source archive and model status. Persist PID plus process
creation time so a reused PID cannot be mistaken for the original test.
Distinguish running, completed, failed and monitoring-disconnected states;
only assign a terminal result after observing process status and output.

Keep the runner alive independently of the SSH connection when remote work
is expected to be long. Reconnect and inspect the existing run before retrying.
Do not restart an unobserved job just because an SSH handshake failed.

## Per-candidate deadlines

Use an explicit configurable task policy. A previously agreed 3600-second
deadline is an example, not a universal default. A model-wide deadline can
coexist with it, but must be recorded and reported separately.

If a watchdog is needed, ensure the implementation:

1. Archives source/config before compiler entry, then associates each timer
   with that candidate. An outer queue/batch timer is not a candidate timer.
2. Measures elapsed time with a monotonic clock and disarms the timer on
   success or ordinary exception. Concurrent candidates have separate identity.
3. Persists candidate ID, target, options, signature/constants, elapsed time,
   compiler command and available temporary IR before terminating owned work.
4. Terminates only processes/groups created for this task, checks their
   ownership and creation time, and waits for termination. Never kill compiler
   processes by a machine-wide name match.
5. Reports a timed-out candidate separately from a blocked/aborted model run.
   If continuing autotune after cancellation is unsupported, say the model run
   was aborted rather than inventing an accuracy result.

The current tracer's unmatched `BEGIN` records identify incomplete attempts,
but cannot alone distinguish timeout, OOM, external termination or disconnect.
Record the actual supervising evidence. Do not claim a candidate timeout was
enforced merely because the wrapper journaled its start.
