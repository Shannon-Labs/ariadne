# Start Here PC - Ariadne Development Guide

## 🎯 Current State (September 2025)

**Repository**: `Shannon-Labs/ariadne`  
**Branch**: `positioning-update-sept-2025` (ready for PR)  
**Status**: Major positioning update + Metal backend improvements committed

## 📋 What Was Just Accomplished

### ✅ Positioning Update Complete
- **README.md**: Completely rewritten with honest capability-focused messaging
- **BENCHMARK_SUMMARY.md**: Updated with real performance data
- **New Metal backend**: Working hybrid implementation with 1.4-1.8x Apple Silicon speedups
- **Real benchmark data**: Included from `results/router_benchmark_results.json`

### 🔄 Key Messaging Changes
- **FROM**: "1.73x verified speedup" and enterprise claims
- **TO**: "50+ qubit Clifford circuits in one call" and capability extension
- **FROM**: Speed-focused marketing
- **TO**: Developer productivity and automatic routing

## 🚀 Next Steps for PC Development

### 1. **Pull the Latest Changes**
```bash
git checkout main
git pull origin main
git checkout positioning-update-sept-2025
git pull origin positioning-update-sept-2025
```

### 2. **Review the PR**
- **URL**: https://github.com/Shannon-Labs/ariadne/pull/new/positioning-update-sept-2025
- **Title**: "feat: Update positioning and add Metal backend improvements (Sept 2025)"
- **Status**: Ready to merge after review

### 3. **Key Files to Understand**

#### **README.md** - New Positioning
- Honest about capabilities vs limitations
- Real performance data from benchmarks
- 30-qubit Clifford example demonstrating capability extension
- Clear backend support matrix

#### **ariadne/backends/metal_backend.py** - New Metal Backend
- Hybrid implementation bypassing JAX Metal bug
- 1.4-1.8x speedups on Apple Silicon
- Graceful fallbacks to CPU when needed

#### **results/router_benchmark_results.json** - Real Performance Data
- Actual benchmark results showing router overhead vs capability extension
- Metal vs CPU performance on Apple Silicon
- Router comparison across different circuit types

### 4. **Current Backend Status**

| Backend | Status | Notes |
|---------|--------|-------|
| **Stim** | ✅ Working | Auto-selected for Clifford circuits, enables >24 qubits |
| **Metal (Apple Silicon)** | ✅ Working | 1.4-1.8x speedups, hybrid implementation |
| **Qiskit Basic** | ✅ Working | General fallback, always available |
| **Tensor Network** | ✅ Working | Exact contraction, slower but accurate |
| **CUDA** | ⚠️ Untested | Requires NVIDIA GPU, no current benchmarks |

### 5. **Development Environment Setup**

#### **Dependencies**
```bash
pip install -e .[dev]
pip install qiskit stim quimb cotengra
```

#### **For Apple Silicon (if available)**
```bash
pip install jax-metal
# Ensure Accelerate BLAS is available
```

#### **For CUDA (if available)**
```bash
pip install cupy-cuda12x  # or appropriate CUDA version
```

### 6. **Key Development Areas**

#### **A. CUDA Backend Development**
- **File**: `ariadne/backends/cuda_backend.py`
- **Status**: Stub implementation, needs real CUDA integration
- **Goal**: Achieve 2-5x speedups on NVIDIA GPUs
- **Dependencies**: CuPy, CUDA toolkit

#### **B. Router Heuristics Tuning**
- **File**: `ariadne/route/analyze.py`
- **Goal**: Better backend selection based on circuit characteristics
- **Current**: Basic entropy and treewidth analysis
- **Opportunity**: Machine learning-based routing decisions

#### **C. Calibration System**
- **File**: `ariadne/calibration.py`
- **Goal**: Automatic backend performance tuning
- **Status**: Framework exists, needs implementation
- **Use case**: Derive backend scores from real benchmarks

#### **D. Tensor Network Optimization**
- **File**: `ariadne/backends/tensor_network_backend.py`
- **Goal**: Faster tensor contraction for complex circuits
- **Current**: Basic Quimb/Cotengra integration
- **Opportunity**: GPU-accelerated tensor networks

### 7. **Testing Strategy**

#### **Unit Tests**
```bash
make test
python -m pytest tests/ -v
```

#### **Benchmark Tests**
```bash
# Apple Silicon benchmarks
python benchmarks/metal_vs_cpu.py --shots 1000

# Router comparison
python benchmarks/router_comparison.py --shots 256 --repetitions 3

# All benchmarks
make benchmark-all
```

#### **Integration Tests**
```bash
# Test 30-qubit Clifford circuit
python -c "
from ariadne import simulate
from qiskit import QuantumCircuit
qc = QuantumCircuit(30, 30)
qc.h(0)
for i in range(29):
    qc.cx(i, i+1)
qc.measure_all()
result = simulate(qc, shots=1000)
print(f'Backend: {result.backend_used}')
print(f'Success: {len(result.counts)} outcomes')
"
```

### 8. **Known Issues & Limitations**

#### **Metal Backend**
- **Issue**: JAX Metal has StableHLO error 22
- **Solution**: Hybrid backend bypasses JAX, uses Accelerate directly
- **Status**: Working on Apple Silicon M4 Max

#### **CUDA Backend**
- **Issue**: No NVIDIA hardware for testing
- **Solution**: Implement proper CuPy integration
- **Status**: Stub implementation exists

#### **Router Overhead**
- **Issue**: Small circuits slower than direct Qiskit
- **Solution**: Optimize routing logic or add bypass for tiny circuits
- **Status**: Acceptable for capability extension use case

### 9. **Performance Targets**

#### **Current Achievements**
- **Apple Silicon**: 1.4-1.8x speedups with Metal backend
- **Clifford circuits**: 50+ qubits via Stim routing
- **Router accuracy**: Correct backend selection for circuit types

#### **Future Goals**
- **CUDA**: 2-5x speedups on NVIDIA GPUs
- **Router**: <10ms overhead for small circuits
- **Calibration**: Automatic backend tuning from benchmarks

### 10. **Contribution Guidelines**

#### **Code Style**
```bash
make lint format typecheck
```

#### **Commit Messages**
- Use conventional commits: `feat:`, `fix:`, `docs:`, etc.
- Include context about performance impact
- Reference issues/PRs when relevant

#### **PR Process**
1. Create feature branch from `main`
2. Implement changes with tests
3. Update benchmarks if performance-related
4. Update documentation if API changes
5. Create PR with clear description

### 11. **Key Metrics to Track**

#### **Performance Metrics**
- Router overhead vs direct backend calls
- Backend selection accuracy
- Memory usage for large circuits
- Setup time for different backends

#### **User Experience Metrics**
- Time to first successful simulation
- Error rates and fallback frequency
- Documentation clarity and completeness

### 12. **Resources & Documentation**

#### **Internal Docs**
- `docs/CALIBRATION_PLAN.md` - Calibration system design
- `dev-artifacts/` - Development prompts and plans
- `results/` - Benchmark data and analysis

#### **External Resources**
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Stim Documentation](https://github.com/quantumlib/Stim)
- [Quimb Documentation](https://quimb.readthedocs.io/)
- [JAX Documentation](https://jax.readthedocs.io/)

### 13. **Immediate Action Items**

1. **Review and merge PR** - positioning-update-sept-2025
2. **Test Metal backend** on available Apple Silicon hardware
3. **Implement CUDA backend** with proper CuPy integration
4. **Optimize router overhead** for small circuits
5. **Add calibration system** for automatic backend tuning

### 14. **Success Criteria**

#### **Short Term (1-2 weeks)**
- PR merged and deployed
- CUDA backend working on NVIDIA hardware
- Router overhead <20ms for small circuits

#### **Medium Term (1-2 months)**
- Calibration system implemented
- Tensor network GPU acceleration
- Comprehensive test coverage

#### **Long Term (3-6 months)**
- Production-ready performance
- Cloud provider integrations
- Enterprise adoption

---

## 🎯 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/Shannon-Labs/ariadne.git
cd ariadne
git checkout positioning-update-sept-2025

# Install dependencies
pip install -e .[dev]

# Run tests
make test

# Run benchmarks
python benchmarks/router_comparison.py

# Test capability extension
python -c "
from ariadne import simulate
from qiskit import QuantumCircuit
qc = QuantumCircuit(30, 30)
qc.h(0)
for i in range(29): qc.cx(i, i+1)
qc.measure_all()
result = simulate(qc, shots=1000)
print(f'Backend: {result.backend_used}, Outcomes: {len(result.counts)}')
"
```

**Ready to continue development!** 🚀
