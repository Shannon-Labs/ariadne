# Ariadne Routing Calibration Plan

This document captures the requirements for adding a lightweight calibration workflow so our routing heuristics are based on real measurements instead of hand-tuned numbers.

## Motivation

`QuantumRouter.channel_capacity_match` uses static `BackendCapacity` metadata to decide which simulator to run. Those numbers were originally guesses (e.g. an Apple Silicon boost of 5×). Now that the hybrid Metal backend is operational and we can run benchmarks, we want a repeatable way to align the routing scores with observed performance on each developer machine.

## Desired Calibration Flow

1. **Command:** Provide a small CLI entry point (`python -m ariadne.calibration` or `ariadne calibrate`) that executes a handful of circuits and times each available backend (Stim, Qiskit, Metal, tensor network, DDSIM when installed).
2. **Circuits:** Reuse the canonical cases from `benchmarks/metal_vs_cpu.py` and `benchmarks/router_comparison.py` – small/medium Clifford, small/medium general, and a QAOA/VQE circuit.
3. **Measurement:** Use a low shot count (e.g. 128) and 2–3 repetitions to keep runtime short (<60 s). Skip unavailable backends gracefully.
4. **Normalization:** Use the direct Qiskit Basic timings as the baseline. Compute relative factors such as `speedup = t_qiskit / t_backend`.
5. **Persistence:** Write the derived multipliers to `~/.ariadne/calibration.json`. Include metadata (timestamp, circuit set, version, Python/OS info) for troubleshooting.
6. **Runtime Override:** On router initialization, attempt to load the calibration file and override the default `BackendCapacity` entries. Expose a helper `router.update_capacity(backend, **kwargs)` for manual tweaks.
7. **Fallbacks:** If the calibration file is missing or malformed, stick with the baked-in defaults so Ariadne still works out of the box.

## Data Model

An example calibration JSON might look like:

```json
{
  "version": 1,
  "generated_at": "2025-09-21T10:24:00Z",
  "system": {
    "platform": "Darwin",
    "machine": "arm64",
    "python": "3.13.2"
  },
  "relative_capacities": {
    "stim": { "clifford_capacity": 10.0 },
    "jax_metal": { "general_capacity": 11.4, "apple_silicon_boost": 1.6 },
    "tensor_network": { "general_capacity": 6.2 },
    "qiskit": { "general_capacity": 8.0 }
  }
}
```

## Implementation Notes

- Create a new module (`ariadne/calibration.py`) that exposes `run_calibration()` and returns the updated capacity dictionary.
- Add a minimal CLI wrapper (e.g. `python -m ariadne.calibration`) that writes the JSON file and prints a short summary table.
- Update `QuantumRouter.__init__` to call a helper that reads the calibration file and patches `self.backend_capacities`.
- Add unit tests for the loader (ensure invalid files are ignored).
- Update README with a short “Calibrating routing heuristics” subsection.

## Outstanding Questions

- Should calibration be opt-in or run automatically on first import? (Recommendation: opt-in to avoid surprise benchmarking.)
- Do we track separate measurements for Clifford vs. non-Clifford circuits? (Initial approach: store per-category multipliers if desired.)
- How do we invalidate stale calibration files after major releases? (Embed a version number in the JSON and only load when compatible.)

Once Claude (or another contributor) delivers the calibration patch, this document will guide review and integration.
