# 🚀 PC Cursor Buddy - Ariadne Development Prompt

## **Repository Overview**

You're working with **Ariadne**, an intelligent quantum circuit router that automatically selects optimal simulators based on Bell Labs-style information theory. The repository is now live at:

**GitHub**: https://github.com/Shannon-Labs/ariadne (Private)

## **Current Status (v1.0.0)**

✅ **Production-Ready Foundation**
- Complete Python package with intelligent backend routing
- Real Stim simulator integration (1000× speedup for Clifford circuits)
- Bell Labs-style information theory routing
- Working examples and benchmarks
- All dependencies properly configured

## **Your Mission: Two-Part Development**

### **Part 1: CUDA Backend Implementation (Primary Focus)**

**Goal**: Add CUDA backend for massive parallel quantum circuit simulation

**Target Performance**:
- Clifford circuits: 5000× faster than Qiskit
- General circuits: 50× faster than Qiskit
- Large circuits: 100× faster than tensor networks

### **Part 2: Launch Accelerator (Secondary Focus)**

**Goal**: Transform Ariadne into a world-class, community-trusted project ready for public launch

## **CUDA Development (Primary Task)**

### **Key Files to Create/Modify**

1. **`ariadne/backends/cuda_backend.py`** - Main CUDA backend implementation
2. **`ariadne/backends/cuda/kernels/`** - CUDA kernel source files
3. **`ariadne/backends/cuda/state_vector.cu`** - State vector simulation
4. **`ariadne/backends/cuda/gates.cu`** - Quantum gate operations
5. **`ariadne/backends/cuda/sampling.cu`** - Measurement sampling

### **Implementation Plan**

```python
# New file: ariadne/backends/cuda_backend.py
class CUDABackend:
    """CUDA-accelerated quantum circuit simulator."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cuda_context = self._init_cuda_context()
    
    def simulate(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Simulate circuit using CUDA kernels."""
        # Convert circuit to CUDA representation
        cuda_circuit = self._convert_to_cuda(circuit)
        
        # Launch CUDA kernels for simulation
        results = self._launch_simulation_kernels(cuda_circuit, shots)
        
        return self._format_results(results)
```

### **CUDA Kernels to Implement**

```cuda
// State vector simulation kernel
__global__ void simulate_gate_kernel(
    cuDoubleComplex* state_vector,
    int* gate_matrix,
    int num_qubits,
    int gate_qubits
);

// Measurement sampling kernel
__global__ void sample_measurements_kernel(
    cuDoubleComplex* state_vector,
    int* samples,
    int shots,
    int num_qubits
);
```

### **Integration Points**

1. **Update `ariadne/router.py`**:
   - Add `BackendType.CUDA = "cuda"`
   - Add CUDA capacity to `BackendCapacity`
   - Add `_simulate_cuda()` method

2. **Update `ariadne/__init__.py`**:
   - Export CUDA backend classes

## **Launch Accelerator (Secondary Task)**

### **Priority Tasks for World-Class Project**

1. **Empirical Validation and Proof**
   - Execute comprehensive benchmarks
   - Generate performance visualizations
   - Create `BENCHMARKS_REPORT.md`
   - Prove the 1000x speedup claims

2. **Documentation Overhaul**
   - Set up ReadTheDocs documentation
   - Write detailed "Routing Theory" guide
   - Create API documentation
   - Add video tutorials

3. **Robustness and Trust**
   - Achieve >90% test coverage
   - Optimize CI/CD pipeline
   - Add health check utility
   - Implement comprehensive error handling

4. **Explainability and DX**
   - Create `ariadne.explain_routing(circuit)` function
   - Build interactive demos
   - Add circuit analysis tools
   - Create Jupyter notebook demos

5. **Community and Launch**
   - Enhance `CONTRIBUTING.md`
   - Create launch materials
   - Build social media kit
   - Set up community infrastructure

## **Development Setup**

```bash
# 1. Clone the repository
git clone https://github.com/Shannon-Labs/ariadne.git
cd ariadne

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e .[dev]

# 4. Install CUDA dependencies
pip install cupy-cuda12x  # For CUDA 12.x
pip install numba[cuda]   # For CUDA JIT compilation

# 5. Verify setup
python verify_setup.py
```

## **Key Resources**

1. **`NEXT_STEPS.md`** - CUDA development roadmap
2. **`ARIADNE_LAUNCH_ACCELERATOR.md`** - Comprehensive launch strategy
3. **`CONTRIBUTING.md`** - Development guidelines
4. **`examples/`** - Working examples to test against
5. **`benchmarks/`** - Performance testing framework

## **Success Criteria**

### **CUDA Backend (Primary)**
- [ ] CUDA backend integrated with router
- [ ] Real quantum circuit simulation (not fake data)
- [ ] 5000x speedup for Clifford circuits
- [ ] 50x speedup for general circuits
- [ ] Memory usage optimized
- [ ] All tests pass

### **Launch Accelerator (Secondary)**
- [ ] >90% test coverage
- [ ] Comprehensive benchmarks with visualizations
- [ ] Professional documentation website
- [ ] Interactive demos and examples
- [ ] Community-ready infrastructure
- [ ] Launch materials prepared

## **Getting Started**

1. **Start with CUDA backend** - Focus on `ariadne/backends/cuda_backend.py`
2. **Test with existing examples** - Ensure compatibility
3. **Add to router integration** - Update routing logic
4. **Implement benchmarks** - Prove performance claims
5. **Enhance documentation** - Make it world-class

## **Project Context**

This is a fantastic project. "Ariadne" addresses a very real pain point in the quantum computing ecosystem: the fragmentation of simulators and the difficulty of choosing the right tool for the job. The conceptual framing—using Bell Labs-style information theory (Shannon entropy and channel capacity) to route quantum circuits—is brilliant branding and technically intriguing.

The foundation is strong, but to be "fully incredible and ready to show the world," Ariadne needs to move beyond the initial release and focus on **proof, polish, and presentation.**

## **Repository Structure**

```
ariadne/
├── ariadne/                    # Main package
│   ├── backends/              # Backend implementations (add CUDA here)
│   ├── route/                 # Circuit analysis
│   └── converters/            # Circuit conversion utilities
├── examples/                   # Working examples
├── benchmarks/                 # Performance benchmarks
├── .github/workflows/          # CI/CD pipeline
├── README.md                   # Main documentation
├── NEXT_STEPS.md              # CUDA development roadmap
├── ARIADNE_LAUNCH_ACCELERATOR.md  # Launch strategy
├── CONTRIBUTING.md            # Development guidelines
└── verify_setup.py            # Verification script
```

## **Ready to Build the Future! 🚀**

The repository is production-ready and all existing functionality works. Your job is to:

1. **Add the CUDA backend** while maintaining compatibility
2. **Enhance the project** to world-class standards
3. **Prepare for public launch** with comprehensive documentation and community infrastructure

**This is your chance to make Ariadne the definitive quantum circuit routing solution!**
