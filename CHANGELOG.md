## v0.2.0 (Apple‑Silicon release)

- Add Apple‑Silicon–tuned backends: Stim (stabilizer), Qiskit Aer SV (fp32/fp64, mem caps), Tensor‑network via quimb+cotengra with slicing, and MQT DDSIM (DD) integration.
- Router with YAML policy, structured logs (JSONL), and justification for backend choice.
- Error‑mitigation passes with Mitiq: ZNE and CDR; YAML per‑subcircuit policy; PEC/VD stubs with TODOs.
- QCEC artifacts: JSON records for equivalence proofs around semantics‑preserving passes.
- FT/RE: Qualtran bridge demo stub and Azure Resource Estimator wrapper with "unavailable" structured results when not configured.
- Mac environment & docs: conda arm64 environment, arm64 Dockerfile, Apple‑Silicon notes.
- CLI expansions: `ariadne-mac` analyze/route/mitigate/resources with threads/mem/precision flags.
- Examples & notebooks updated for Apple‑Silicon; reports written under `reports/` with runlogs.
- CI: GitHub Actions for macOS and Ubuntu; lint, typecheck, unit tests; heavy tests skipped via markers.

## v0.4.0

## v0.5.0

- Segmentation: Added `passes/segment.py` to partition circuits into `clifford`, `low_treewidth`, and `dense_sv` segments with a JSON manifest (hashes, qubit spans). `route.execute_segmented` executes segments per-engine and logs per-segment metrics and backends.
- Deferred measurement: `passes/defer_measure.py` rewrites measure+classical control into controlled Cliffords when safe; integrated into Stim conversion.
- Stim conversion: Expanded to support full Clifford set (I/X/Y/Z/H/S/Sdg, CX/CY/CZ, SWAP via CX, RX/RY(±π/2), RESET/MEASURE, limited conditionals). Strict rejects non‑Clifford ops.
- TN concurrency: ProcessPoolExecutor (spawn) contracts cotengra slices concurrently; per-slice timing and planned peak bytes logged; respects `--threads` via `OMP_NUM_THREADS=1` in workers.
- Router tuner v2: `tune-router` microbenchmarks SV (28–31q) and TN (QAOA 80–120q) plus DD redundancy toy; writes before/after reports and updates YAML on significant improvements (Mann‑Whitney U, p<0.05).
- Azure RE: If AZURE_QUANTUM_* is configured, submits minimal jobs and writes tables under `reports/resources/`; otherwise returns structured `unavailable`.
- Observability: JSONL schema v2 adds `schema_version`, `segment_id`, `segment_backend`. Fixed duplicates and ensured unique `run_id` per run. `summarize` aggregates segmented runs.
- Backends: Expanded Qiskit→Stim converter (full Clifford set incl. RX/RY(±π/2), SWAP via CX, RESET/MEASURE, limited conditionals). Observable estimation for Z, ZZ, X via basis changes. Aer SV strict mem‑cap guard with fp32/fp64 and friendly TN/DD fallback note. TN real contraction under cap with cotengra slicing; logs tree, nslices, and planned peak; plans saved in `reports/trees/`. DDSIM supports path/probability modes.
- Router: Explainable, policy‑driven decisions with not‑chosen justifications; JSONL schema includes run_id/ts/cmd/seed/metrics/mem‑cap/nslices. Added `tune-router` CLI to auto‑calibrate thresholds by microbenchmarks and update YAML.
- Mitigation: YAML‑driven ZNE (Mitiq) and CDR wrapper with on‑disk cache by circuit hash; provenance marked non‑equivalence‑preserving.
- Verification: Added ZX peephole optimizations (H•H, CX•CX) with QCEC artifacts. Positive and negative test cases.
- QEC: Added `qec-demo` CLI using Stim sampling and optional PyMatching decode; prints throughput and a proxy logical‑error rate.
- FT/RE: Qualtran demo builder and Azure RE wrapper that submits when `AZURE_QUANTUM_*` is configured or returns a structured `unavailable` record.
- Observability: `summarize` CLI writes a Markdown dashboard aggregating recent runs.
- Docs/CI: Apple‑Silicon notes expanded; CI matrix `{macos, ubuntu} × {3.11, 3.12}` with pip cache and notebook smoke run.
