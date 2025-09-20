# Ariadne Examples

This directory contains example scripts demonstrating Ariadne's capabilities.

## Quick Start Examples

### 1. `quickstart.py`
Basic introduction to Ariadne's automatic routing:
- Simple Bell state preparation
- Large Clifford circuits
- General circuits with T gates
- Inspecting routing decisions
- Forcing specific backends

```bash
python examples/quickstart.py
```

### 2. `performance_comparison.py`
Benchmarks comparing Ariadne's intelligent routing vs fixed backends:
- GHZ state circuits (Clifford)
- QFT circuits (non-Clifford)
- Performance visualization

```bash
python examples/performance_comparison.py
```

### 3. `quantum_algorithms.py`
Real quantum algorithms and how Ariadne routes them:
- Grover's search algorithm
- Quantum phase estimation
- Quantum teleportation
- Bernstein-Vazirani algorithm

```bash
python examples/quantum_algorithms.py
```

## Advanced Examples

### CUDA Backend Usage
```python
from ariadne.backends.cuda_backend import CUDABackend

# Check CUDA availability
from ariadne.backends.cuda_backend import is_cuda_available
if is_cuda_available():
    backend = CUDABackend()
    result = backend.simulate(circuit, shots=1000)
```

### Custom Routing Logic
```python
from ariadne import QuantumRouter

# Create router with custom configuration
router = QuantumRouter(config={
    'clifford_threshold': 0.95,  # 95% Clifford gates triggers Stim
    'prefer_gpu': True,           # Prefer GPU backends when available
})
```

### Circuit Analysis
```python
from ariadne.route.analyze import analyze_circuit

analysis = analyze_circuit(circuit)
print(f"Circuit entropy: {analysis['circuit_entropy']}")
print(f"Estimated runtime: {analysis['estimated_runtime']}")
```

## Performance Tips

1. **Let Ariadne choose**: The automatic routing usually picks the best backend
2. **Batch similar circuits**: Process similar circuits together for better performance
3. **Use circuit analysis**: Understand your circuit structure with the analysis tools

## Running All Examples

```bash
# Run all examples
for example in quickstart performance_comparison quantum_algorithms; do
    echo "Running $example.py..."
    python examples/$example.py
    echo
done
```

## Expected Performance

- **Clifford circuits**: 1000-5000× speedup with Stim backend
- **General circuits**: 50× speedup with CUDA backend (GPU required)
- **Large circuits**: Efficient memory usage with tensor network backends

## Need Help?

- Check the [API documentation](../docs/api_reference.md)
- Join our [Discord community](https://discord.gg/shannonlabs)
- Report issues on [GitHub](https://github.com/Shannon-Labs/ariadne/issues)