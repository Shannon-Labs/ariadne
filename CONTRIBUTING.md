# Contributing to Ariadne 🔮

Thank you for your interest in contributing to Ariadne! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of quantum computing
- Familiarity with Python development

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ariadne.git
cd ariadne

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e .[dev]

# 4. Run tests to ensure everything works
pytest tests/
```

## 🎯 Areas for Contribution

### 1. CUDA Backend Implementation
- Implement CUDA kernels for quantum gate operations
- Add GPU memory management
- Optimize for different GPU architectures

### 2. Performance Optimizations
- Memory usage optimization
- Parallel processing improvements
- Caching system implementation

### 3. New Backends
- Add support for new quantum simulators
- Implement specialized backends for specific use cases
- Add cloud-based simulation backends

### 4. Circuit Analysis
- Improve circuit analysis algorithms
- Add new circuit metrics
- Implement circuit optimization passes

### 5. Documentation
- Improve API documentation
- Add more examples
- Write tutorials and guides

## 🔧 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Linting

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_router.py

# Run linting
black ariadne/
mypy ariadne/

# Run type checking
mypy ariadne/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of your changes"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for all public functions
- Keep line length under 100 characters

### Example Code Style

```python
def simulate_circuit(
    circuit: QuantumCircuit, 
    shots: int = 1000
) -> SimulationResult:
    """
    Simulate a quantum circuit using intelligent routing.
    
    Args:
        circuit: The quantum circuit to simulate
        shots: Number of measurement shots
        
    Returns:
        SimulationResult containing counts and metadata
    """
    router = QuantumRouter()
    return router.simulate(circuit, shots)
```

### File Organization

```
ariadne/
├── __init__.py
├── router.py              # Main router implementation
├── backends/              # Backend implementations
│   ├── __init__.py
│   ├── stim_backend.py
│   ├── qiskit_backend.py
│   └── cuda_backend.py    # Future CUDA implementation
├── route/                 # Circuit analysis
│   ├── __init__.py
│   └── analyze.py
└── converters/            # Circuit conversion utilities
    ├── __init__.py
    └── qiskit_to_stim.py
```

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_router.py
import pytest
from qiskit import QuantumCircuit
from ariadne import simulate, QuantumRouter

class TestQuantumRouter:
    def test_basic_simulation(self):
        """Test basic circuit simulation."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = simulate(circuit, shots=100)
        
        assert result.backend_used is not None
        assert len(result.counts) > 0
        assert result.execution_time > 0
    
    def test_clifford_routing(self):
        """Test that Clifford circuits route to Stim."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        router = QuantumRouter()
        decision = router.select_optimal_backend(circuit)
        
        assert decision.recommended_backend == BackendType.STIM
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test performance characteristics
- **Regression Tests**: Test for known issues

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ariadne

# Run specific test category
pytest -m "not slow"

# Run performance tests
pytest tests/test_performance.py
```

## 📚 Documentation Guidelines

### Docstring Format

```python
def analyze_circuit(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Analyze a quantum circuit and return analysis metrics.
    
    This function performs comprehensive analysis of a quantum circuit
    including entropy calculation, Clifford detection, and complexity
    estimation.
    
    Args:
        circuit: The quantum circuit to analyze
        
    Returns:
        Dictionary containing analysis results:
            - num_qubits: Number of qubits in the circuit
            - depth: Circuit depth
            - is_clifford: Whether circuit contains only Clifford gates
            - clifford_ratio: Ratio of Clifford gates to total gates
            - treewidth_estimate: Estimated treewidth of circuit graph
            
    Raises:
        ValueError: If circuit is empty or invalid
        
    Example:
        >>> from qiskit import QuantumCircuit
        >>> circuit = QuantumCircuit(2)
        >>> circuit.h(0)
        >>> analysis = analyze_circuit(circuit)
        >>> print(analysis['is_clifford'])
        True
    """
```

### README Updates

When adding new features, update the README.md to include:
- New usage examples
- Updated performance benchmarks
- New installation requirements
- New configuration options

## 🚀 CUDA Development Guidelines

### CUDA Backend Structure

```python
# ariadne/backends/cuda_backend.py
class CUDABackend:
    """CUDA-accelerated quantum circuit simulator."""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA backend.
        
        Args:
            device_id: CUDA device ID to use
        """
        self.device_id = device_id
        self.cuda_context = self._init_cuda_context()
    
    def simulate(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Simulate circuit using CUDA kernels."""
        # Implementation here
        pass
```

### CUDA Kernel Guidelines

```cuda
// ariadne/backends/cuda/kernels/state_vector.cu
__global__ void simulate_gate_kernel(
    cuDoubleComplex* state_vector,
    int* gate_matrix,
    int num_qubits,
    int gate_qubits
) {
    // CUDA kernel implementation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel logic
}
```

### CUDA Testing

```python
# tests/test_cuda_backend.py
import pytest
from ariadne.backends import CUDABackend

@pytest.mark.cuda
class TestCUDABackend:
    def test_basic_simulation(self):
        """Test basic CUDA simulation."""
        backend = CUDABackend()
        # Test implementation
        pass
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Error Messages**: Full error traceback

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
```python
# Minimal code example
from ariadne import simulate
# ... code that causes the bug
```

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- Python version: 3.9.0
- OS: macOS 12.0
- Ariadne version: 1.0.0

## Error Messages
```
Traceback (most recent call last):
  ...
```

## Additional Context
Any additional context about the problem.
```

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Implementation Ideas**: Any implementation thoughts

### Feature Request Template

```markdown
## Feature Description
Brief description of the feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches were considered?

## Implementation Ideas
Any thoughts on how to implement this feature?

## Additional Context
Any additional context about the feature request.
```

## 🏷️ Pull Request Guidelines

### PR Title Format

```
Type: Brief description

Examples:
- Add: CUDA backend implementation
- Fix: Memory leak in Stim backend
- Update: Documentation for new API
- Refactor: Circuit analysis algorithms
```

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Performance tests updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or breaking changes documented)
```

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and discussions
- **Email**: For security issues or private matters

## 🎉 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Ariadne! 🚀
