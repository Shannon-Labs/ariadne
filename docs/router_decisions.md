**How We Decide Which Simulator To Use**

- Inputs: metrics (n_qubits, depth, two-qubit depth/count/density, treewidth estimate, redundancy score, Clifford ratio), and a YAML policy.
- Policy (configs/router_policy.yaml):
  - Prefer `stim` if circuit is Clifford-only.
  - Prefer `sv` if state-vector fits under cap (try fp32 then fp64) and n≤limits.
  - Prefer `dd` if redundancy score ≥ threshold and n≤limit.
  - Else choose `tn` (quimb+cotengra) and plan with slicing to respect memory cap.

Logging
- Writes JSONL under `reports/runlogs/` with run_id, ts, cmd, seed, metrics, chosen backend, mem cap, TN slices, wall time, and not‑chosen justifications.

Segmentation
- The pipeline can segment a circuit into `clifford`, `low_treewidth`, and `dense_sv` regions. Each segment is routed independently with lightweight boundary adapters (basis changes), producing a manifest for reproducibility.
