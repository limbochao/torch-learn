# Validation evidence

`adapted-failures-20261111/` contains the final native pytest run: 16 source methods,
17 originally failing instances. `jobs.json` merges those reports into the prior
105-method selection; unexecuted reports keep their original validation provenance.
`current-analysis.json` contains only current FAIL/SKIP analyses.
`generated-code/` contains the passive output-code capture for the three count checks;
these sources precede the assertion adaptation and establish the observed NPU paths.
`executed-test-sha256.json` records the final test snapshot checked against the local commit.
Paths/backend identifiers are sanitized; archive receipts retain original and archived hashes.
Current results and analysis are maintained only in the versioned report and CSV.
