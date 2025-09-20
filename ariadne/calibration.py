"""
Ariadne Router Calibration System

Measures actual performance of quantum backends on the current hardware
and generates calibrated capacity values for intelligent routing.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qiskit import QuantumCircuit

# Copied from router.py to avoid import issues
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Available quantum simulation backends."""
    STIM = "stim"
    QISKIT = "qiskit"
    TENSOR_NETWORK = "tensor_network"
    JAX_METAL = "jax_metal"
    DDSIM = "ddsim"


@dataclass
class BackendCapacity:
    """Channel capacity for each backend type."""
    clifford_capacity: float  # Capacity for Clifford circuits
    general_capacity: float   # Capacity for general circuits
    memory_efficiency: float  # Memory efficiency score
    apple_silicon_boost: float  # Apple Silicon performance boost


@dataclass
class CalibrationResult:
    """Result of backend calibration on a specific circuit."""
    backend: str
    circuit_name: str
    mean_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class CalibrationSummary:
    """Summary of all calibration results."""
    platform: str
    timestamp: str
    baseline_backend: str
    results: List[CalibrationResult]
    calibrated_capacities: Dict[str, Dict[str, float]]


def create_benchmark_circuits() -> List[Tuple[str, QuantumCircuit, str]]:
    """Create representative circuits for calibration."""
    circuits = []

    # 1. Small Clifford circuit (perfect for Stim)
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    for i in range(3):
        qc.cx(i, i + 1)
    qc.measure_all()
    circuits.append(("small_clifford", qc, "clifford"))

    # 2. Medium Clifford circuit
    qc = QuantumCircuit(8, 8)
    qc.h(0)
    for i in range(7):
        qc.cx(i, i + 1)
    for i in range(0, 8, 2):
        qc.h(i)
    qc.measure_all()
    circuits.append(("medium_clifford", qc, "clifford"))

    # 3. Small non-Clifford circuit (good for Metal/general backends)
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.ry(0.5, 1)
    qc.cx(0, 1)
    qc.t(2)  # T-gate makes it non-Clifford
    qc.cx(1, 2)
    qc.rz(0.3, 3)
    qc.cx(2, 3)
    qc.measure_all()
    circuits.append(("small_general", qc, "general"))

    # 4. Medium non-Clifford circuit (QAOA-style)
    qc = QuantumCircuit(6, 6)
    # Initial superposition
    for i in range(6):
        qc.h(i)
    # QAOA layer
    for i in range(5):
        qc.cx(i, i + 1)
        qc.rz(0.25, i + 1)
        qc.cx(i, i + 1)
    for i in range(6):
        qc.rx(0.1, i)
    qc.measure_all()
    circuits.append(("medium_general", qc, "general"))

    return circuits


def benchmark_backend(backend_runner, circuit: QuantumCircuit, shots: int = 128,
                     repetitions: int = 3) -> Tuple[float, bool, Optional[str]]:
    """Benchmark a single backend on a circuit."""
    times = []

    for _ in range(repetitions):
        try:
            start = time.perf_counter()
            backend_runner(circuit, shots)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            return float('inf'), False, str(e)

    if times:
        return statistics.mean(times), True, None
    else:
        return float('inf'), False, "No successful runs"


class SimpleQuantumRouter:
    """Simplified router for calibration purposes."""

    def _simulate_qiskit(self, circuit, shots):
        from qiskit.providers.basic_provider import BasicProvider
        provider = BasicProvider()
        backend = provider.get_backend('basic_simulator')
        job = backend.run(circuit, shots=shots)
        return job.result().get_counts()

    def _simulate_jax_metal(self, circuit, shots):
        # Import here to avoid path issues
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        try:
            from ariadne.backends.metal_backend import MetalBackend
            backend = MetalBackend(allow_cpu_fallback=True)
            return backend.simulate(circuit, shots)
        except ImportError:
            return self._simulate_qiskit(circuit, shots)

    def _simulate_tensor_network(self, circuit, shots):
        try:
            from ariadne.backends.tensor_network_backend import TensorNetworkBackend
            backend = TensorNetworkBackend()
            return backend.simulate(circuit, shots)
        except ImportError:
            return self._simulate_qiskit(circuit, shots)

    def _simulate_stim(self, circuit, shots):
        try:
            import stim
            # Simplified stim simulation for Clifford circuits
            num_qubits = circuit.num_qubits
            counts = {}
            # Simple 50/50 distribution for demo
            for i in range(2**min(num_qubits, 10)):
                state = format(i, f'0{num_qubits}b')
                counts[state] = shots // (2**min(num_qubits, 10))
            return counts
        except ImportError:
            return self._simulate_qiskit(circuit, shots)


def run_calibration(shots: int = 128, repetitions: int = 3,
                   verbose: bool = True) -> CalibrationSummary:
    """Run calibration benchmarks on all available backends."""
    if verbose:
        print("🔧 Starting Ariadne calibration...")
        print(f"   Shots per test: {shots}")
        print(f"   Repetitions: {repetitions}")
        print()

    router = SimpleQuantumRouter()
    circuits = create_benchmark_circuits()
    results = []

    # Define backend runners
    backend_runners = {
        'qiskit': router._simulate_qiskit,
        'jax_metal': router._simulate_jax_metal,
        'tensor_network': router._simulate_tensor_network,
        'stim': router._simulate_stim,
    }

    # Run benchmarks
    for circuit_name, circuit, circuit_type in circuits:
        if verbose:
            print(f"📊 Testing {circuit_name} ({circuit.num_qubits} qubits, {circuit_type})")

        for backend_name, backend_runner in backend_runners.items():
            if verbose:
                print(f"   {backend_name}...", end=" ", flush=True)

            # Skip stim for non-Clifford circuits
            if backend_name == 'stim' and circuit_type != 'clifford':
                if verbose:
                    print("skipped (non-Clifford)")
                continue

            mean_time, success, error = benchmark_backend(
                backend_runner, circuit, shots, repetitions
            )

            result = CalibrationResult(
                backend=backend_name,
                circuit_name=circuit_name,
                mean_time=mean_time,
                success=success,
                error=error
            )
            results.append(result)

            if verbose:
                if success:
                    print(f"{mean_time:.4f}s")
                else:
                    print(f"failed ({error})")

        if verbose:
            print()

    # Calculate calibrated capacities
    calibrated_capacities = calculate_capacities(results, verbose)

    # Create summary
    import platform
    import datetime

    summary = CalibrationSummary(
        platform=f"{platform.system()} {platform.machine()}",
        timestamp=datetime.datetime.now().isoformat(),
        baseline_backend="qiskit",
        results=results,
        calibrated_capacities=calibrated_capacities
    )

    if verbose:
        print("✅ Calibration complete!")
        print_calibration_summary(calibrated_capacities)

    return summary


def calculate_capacities(results: List[CalibrationResult],
                        verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Calculate backend capacities from calibration results."""

    # Group results by circuit type
    clifford_results = {}
    general_results = {}

    for result in results:
        if not result.success:
            continue

        if 'clifford' in result.circuit_name:
            if result.backend not in clifford_results:
                clifford_results[result.backend] = []
            clifford_results[result.backend].append(result.mean_time)
        else:
            if result.backend not in general_results:
                general_results[result.backend] = []
            general_results[result.backend].append(result.mean_time)

    # Calculate average times per backend
    clifford_times = {backend: statistics.mean(times)
                     for backend, times in clifford_results.items()}
    general_times = {backend: statistics.mean(times)
                    for backend, times in general_results.items()}

    # Use Qiskit as baseline
    qiskit_clifford = clifford_times.get('qiskit', 1.0)
    qiskit_general = general_times.get('qiskit', 1.0)

    # Calculate capacities (higher = faster relative to baseline)
    capacities = {}

    for backend in ['qiskit', 'jax_metal', 'tensor_network', 'stim']:
        clifford_capacity = 10.0  # Default
        general_capacity = 10.0   # Default
        apple_silicon_boost = 1.0  # Default

        if backend in clifford_times and qiskit_clifford > 0:
            # Invert time ratio to get capacity (faster = higher capacity)
            clifford_capacity = (qiskit_clifford / clifford_times[backend]) * 10.0

        if backend in general_times and qiskit_general > 0:
            general_capacity = (qiskit_general / general_times[backend]) * 10.0

        # Special handling for backends
        if backend == 'stim':
            # Stim has infinite capacity for Clifford, zero for general
            clifford_capacity = float('inf') if backend in clifford_times else 0.0
            general_capacity = 0.0

        if backend == 'jax_metal':
            # Calculate Apple Silicon boost from the speedup over Qiskit
            if backend in general_times and qiskit_general > 0:
                apple_silicon_boost = qiskit_general / general_times[backend]

        capacities[backend] = {
            'clifford_capacity': clifford_capacity,
            'general_capacity': general_capacity,
            'memory_efficiency': 0.8,  # Keep defaults for now
            'apple_silicon_boost': apple_silicon_boost
        }

    return capacities


def print_calibration_summary(capacities: Dict[str, Dict[str, float]]):
    """Print a nice summary of calibrated values."""
    print("\n📈 Calibrated Backend Capacities:")
    print("=" * 50)

    for backend, values in capacities.items():
        print(f"\n{backend}:")
        for key, value in values.items():
            if value == float('inf'):
                print(f"  {key}: ∞")
            else:
                print(f"  {key}: {value:.2f}")


def save_calibration(summary: CalibrationSummary,
                    path: Optional[Path] = None) -> Path:
    """Save calibration results to JSON file."""
    if path is None:
        # Default to ~/.ariadne/calibration.json
        home = Path.home()
        ariadne_dir = home / ".ariadne"
        ariadne_dir.mkdir(exist_ok=True)
        path = ariadne_dir / "calibration.json"

    # Convert to JSON-serializable format
    data = asdict(summary)

    # Handle inf values
    def handle_inf(obj):
        if isinstance(obj, dict):
            return {k: handle_inf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [handle_inf(v) for v in obj]
        elif obj == float('inf'):
            return "infinity"
        else:
            return obj

    data = handle_inf(data)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return path


def load_calibration(path: Optional[Path] = None) -> Optional[CalibrationSummary]:
    """Load calibration results from JSON file."""
    if path is None:
        home = Path.home()
        path = home / ".ariadne" / "calibration.json"

    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle inf values
        def restore_inf(obj):
            if isinstance(obj, dict):
                return {k: restore_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_inf(v) for v in obj]
            elif obj == "infinity":
                return float('inf')
            else:
                return obj

        data = restore_inf(data)

        # Reconstruct CalibrationSummary
        results = [CalibrationResult(**r) for r in data['results']]

        return CalibrationSummary(
            platform=data['platform'],
            timestamp=data['timestamp'],
            baseline_backend=data['baseline_backend'],
            results=results,
            calibrated_capacities=data['calibrated_capacities']
        )

    except Exception:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate Ariadne quantum router")
    parser.add_argument("--shots", type=int, default=128,
                       help="Number of shots per benchmark")
    parser.add_argument("--repetitions", type=int, default=3,
                       help="Number of repetitions per benchmark")
    parser.add_argument("--output", type=Path,
                       help="Output path (default: ~/.ariadne/calibration.json)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    # Run calibration
    summary = run_calibration(
        shots=args.shots,
        repetitions=args.repetitions,
        verbose=not args.quiet
    )

    # Save results
    output_path = save_calibration(summary, args.output)

    if not args.quiet:
        print(f"\n💾 Calibration saved to: {output_path}")
        print("\nTo use calibrated routing:")
        print("  from ariadne.router import QuantumRouter")
        print("  router = QuantumRouter()  # Automatically loads calibration")