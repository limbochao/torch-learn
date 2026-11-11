# d12230b556 Combo failure-only validation

2026-11-11, NPU Ascend950PR, conda `pta_213_case`.

- Actual Inductor Python source: `sync/combo-kernel-master:d12230b556f11b27c9ff752a0dd41635791efc83`.
- Base wheel/C++: `torch_npu 2.13.0+git64dc85c`; this is a 191-file Python overlay, not a rebuilt wheel.
- Test source: `c1ce0688898b9ca1edebc2a4ff4dbffd8b287f28`, 189 Python files verified.
- Native pytest, fresh caches, physical devices 3 and 6. No custom plugin or new test edits.
- This run: 33 previously failing methods / 35 failing instances; 1 method/instance PASS, 32 methods/34 instances FAIL.
- Both phases completed; installed source hashes unchanged. No environment ERROR or retry.

`priority/` and `remaining/` are the two actual execution phases. Their `jobs.json`, logs, XML and status describe only this run.
Top-level `jobs.json`/`status.json` merge these reports into the prior 105-method/314-selected-instance snapshot:
69 PASS / 32 FAIL / 4 SKIP methods, 275 PASS / 34 FAIL / 5 SKIP instances.
Historical PASS and SKIP were not rerun. Each report keeps its actual version under `validation`;
`rerun_history` records earlier execution provenance. Do not treat the merged status as a full latest-code run.

`failed-manifest.json` contains the exact 35 selected node IDs; the two phase manifests partition it.
`failure-delta.json` compares old/new failing locations: cumsum passes, bandwidth reaches a later assertion,
and the other 33 failing instances stop at the same test locations.
`baseline-pass-comparison.json` shows all 60 overlapping 2.10 baseline PASS methods now PASS,
59 from earlier validation plus the cumsum PASS in this run.

`overlay.json` records old/new source hashes; `expected-overlay-sha256.json` is the production source manifest.
`test-source-sha256.json` records the verified test snapshot. `cumsum-generated-check.json` checks the raw generated wrapper:
one helper definition, associative scan present, no ATen cumsum call.
Its raw SHA refers to the pre-sanitization source; `archive-integrity.json` maps raw and archived SHA values.
Personal home/installed paths, host/process identities, profiler trace identifiers and backend hashes are redacted.
XML is parsed and reserialized to keep reports valid after redaction; numerical results and failure locations are unchanged.

[Current assessment](../../docs/notes/compiler/codegen/inductor-combo-tests-v2.13-latest-failures.md)
