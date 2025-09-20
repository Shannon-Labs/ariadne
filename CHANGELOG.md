# Changelog

All notable changes to Ariadne will be documented here.

## Unreleased

- Replace the placeholder CUDA backend with a lightweight statevector
  implementation that falls back to the CPU when CUDA is unavailable.
- Remove marketing copy that claimed unverified performance figures and clean up
  optional documentation.
- Simplify the router heuristics and guard against selecting the CUDA backend
  when no GPU support is present.
- Refresh the CUDA tests so that they exercise the fallback path and only access
  the GPU when it is explicitly available.