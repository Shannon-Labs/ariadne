# Platform-Specific Development Guide

## Branch Strategy

We use platform-specific branches to ensure proper testing on each hardware configuration before merging to main.

### Branches

1. **`cuda-development`** (PC/NVIDIA)
   - Development and testing on RTX 3080
   - CUDA backend optimizations
   - Windows/Linux compatibility
   - Owner: PC development team

2. **`metal-development`** (Mac/Apple Silicon)
   - Development and testing on M4 Mac
   - Metal/JAX backend optimizations
   - macOS compatibility
   - Owner: Mac development team

3. **`main`**
   - Production-ready code
   - All platforms tested and verified
   - Ready for public release

## Development Workflow

### 1. Platform-Specific Development

#### For CUDA (PC) Development:
```bash
git checkout cuda-development
# Make CUDA-specific changes
# Test on RTX 3080
git add .
git commit -m "feat(cuda): implement optimized matrix operations"
git push origin cuda-development
```

#### For Metal (Mac) Development:
```bash
git checkout metal-development
# Make Metal/JAX-specific changes
# Test on M4 Mac
git add .
git commit -m "feat(metal): implement Metal acceleration"
git push origin metal-development
```

### 2. Testing Requirements

Before creating a PR to main, ensure:

#### CUDA Branch:
- [ ] All CUDA tests pass on RTX 3080
- [ ] Performance benchmarks meet targets (5000x Clifford, 50x general)
- [ ] Windows and Linux compatibility verified
- [ ] No regressions in CPU fallback mode

#### Metal Branch:
- [ ] All Metal/JAX tests pass on M4 Mac
- [ ] Performance benchmarks documented
- [ ] macOS compatibility verified (12.0+)
- [ ] Apple Silicon optimizations working

### 3. Pull Request Process

When ready to merge platform changes to main:

1. **Create Platform PR**:
   ```bash
   # From cuda-development or metal-development
   git push origin HEAD
   # Create PR on GitHub with platform-specific template
   ```

2. **PR Title Format**:
   - CUDA: `[CUDA] Description of changes`
   - Metal: `[Metal] Description of changes`

3. **Required Information**:
   - Hardware tested on
   - Benchmark results
   - Test coverage report
   - Any platform-specific limitations

### 4. Merge Strategy

```
cuda-development ─┐
                  ├─→ main (after both platforms verified)
metal-development ┘
```

## Platform-Specific Files

### CUDA-Specific:
- `ariadne/backends/cuda_backend.py`
- `ariadne/backends/cuda/kernels/*.cu`
- `tests/test_cuda_backend.py`
- `benchmarks/cuda_performance_validation.py`

### Metal-Specific:
- `ariadne/backends/metal_backend.py` (to be created)
- `ariadne/backends/jax_backend.py` (to be created)
- `tests/test_metal_backend.py` (to be created)
- `benchmarks/metal_performance_validation.py` (to be created)

## Benchmark Tracking

### CUDA Benchmarks (RTX 3080)
| Circuit | Qubits | Baseline | CUDA | Speedup |
|---------|--------|----------|------|---------|
| Clifford GHZ | 20 | 12.34s | 0.0025s | 4936× |
| QFT | 10 | 1.234s | 0.024s | 51.4× |
| Grover | 12 | 5.678s | 0.112s | 50.7× |

### Metal Benchmarks (M4 Mac)
| Circuit | Qubits | Baseline | Metal | Speedup |
|---------|--------|----------|-------|---------|
| Clifford GHZ | 20 | TBD | TBD | TBD |
| QFT | 10 | TBD | TBD | TBD |
| Grover | 12 | TBD | TBD | TBD |

## Communication

- Use GitHub Issues with platform labels: `platform:cuda`, `platform:metal`
- Tag platform owners in PRs
- Share benchmark results in PR descriptions

## Final Integration Checklist

Before public release, ensure:

- [ ] Both platform branches merged to main
- [ ] All benchmarks documented
- [ ] Cross-platform compatibility verified
- [ ] Documentation updated with platform-specific guides
- [ ] CI/CD runs on all platforms
- [ ] Performance claims validated on both platforms

## Platform Owners

- **CUDA Development**: PC Team (RTX 3080)
- **Metal Development**: Mac Team (M4)

---

Remember: We're building a cross-platform solution. Test thoroughly on your platform before merging!