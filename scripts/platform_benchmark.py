#!/usr/bin/env python3
"""
Platform-specific benchmark runner.

Runs benchmarks appropriate for the current platform and generates reports.
"""

import platform
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ariadne.backends.cuda_backend import is_cuda_available


def detect_platform():
    """Detect current platform and capabilities."""
    system = platform.system()
    processor = platform.processor()
    
    info = {
        'system': system,
        'processor': processor,
        'python': platform.python_version(),
        'machine': platform.machine(),
    }
    
    # Check for CUDA
    if is_cuda_available():
        from ariadne.backends.cuda_backend import get_cuda_info
        info['cuda'] = get_cuda_info()
        info['platform_type'] = 'cuda'
    # Check for Apple Silicon
    elif system == 'Darwin' and 'arm' in platform.machine().lower():
        info['platform_type'] = 'metal'
        info['apple_silicon'] = True
    else:
        info['platform_type'] = 'cpu'
    
    return info


def run_platform_benchmarks():
    """Run benchmarks specific to current platform."""
    platform_info = detect_platform()
    platform_type = platform_info['platform_type']
    
    print(f"Platform detected: {platform_type}")
    print(f"System: {platform_info['system']}")
    print(f"Processor: {platform_info['processor']}")
    
    if platform_type == 'cuda':
        print("\nRunning CUDA benchmarks...")
        from benchmarks.cuda_performance_validation import PerformanceValidator
        
        validator = PerformanceValidator()
        report = validator.generate_report()
        
        # Save CUDA-specific report
        filename = f"cuda_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    elif platform_type == 'metal':
        print("\nRunning Metal/JAX benchmarks...")
        # TODO: Import metal benchmark when available
        print("Metal benchmarks not yet implemented")
        report = {
            'platform': platform_info,
            'status': 'not_implemented',
            'message': 'Metal benchmarks coming soon'
        }
        filename = f"metal_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    else:
        print("\nRunning CPU benchmarks...")
        report = {
            'platform': platform_info,
            'status': 'cpu_only',
            'message': 'No GPU acceleration available'
        }
        filename = f"cpu_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save report
    reports_dir = Path('benchmark_reports')
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / filename
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nBenchmark report saved to: {report_path}")
    
    # Print summary
    if 'clifford_average_speedup' in report:
        print(f"\nPerformance Summary:")
        print(f"  Clifford average speedup: {report['clifford_average_speedup']:.1f}×")
        print(f"  General average speedup: {report.get('general_average_speedup', 'N/A'):.1f}×")
    
    return report


def compare_platforms():
    """Compare benchmark results from different platforms."""
    reports_dir = Path('benchmark_reports')
    if not reports_dir.exists():
        print("No benchmark reports found. Run benchmarks first.")
        return
    
    cuda_reports = list(reports_dir.glob('cuda_benchmark_*.json'))
    metal_reports = list(reports_dir.glob('metal_benchmark_*.json'))
    
    if cuda_reports:
        latest_cuda = sorted(cuda_reports)[-1]
        with open(latest_cuda) as f:
            cuda_data = json.load(f)
        print(f"\nLatest CUDA benchmark: {latest_cuda.name}")
        if 'clifford_average_speedup' in cuda_data:
            print(f"  Clifford: {cuda_data['clifford_average_speedup']:.1f}×")
            print(f"  General: {cuda_data.get('general_average_speedup', 'N/A'):.1f}×")
    
    if metal_reports:
        latest_metal = sorted(metal_reports)[-1]
        with open(latest_metal) as f:
            metal_data = json.load(f)
        print(f"\nLatest Metal benchmark: {latest_metal.name}")
        # Print Metal results when available
    
    if not cuda_reports and not metal_reports:
        print("No platform benchmarks found.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Platform-specific benchmark tool')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare results from different platforms')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_platforms()
    else:
        run_platform_benchmarks()