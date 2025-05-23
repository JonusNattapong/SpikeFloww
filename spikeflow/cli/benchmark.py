"""
SpikeFlow Benchmark CLI Tool
Command-line interface for performance benchmarking
"""

import argparse
import sys
import json
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from benchmarks.performance_benchmark import PerformanceBenchmark

def main():
    """Main CLI entry point for benchmarking"""
    
    parser = argparse.ArgumentParser(
        description='SpikeFlow Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spikeflow-benchmark --quick                    # Quick benchmark
  spikeflow-benchmark --full --output results/   # Full benchmark
  spikeflow-benchmark --model-size large --gpu  # Large model on GPU
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick benchmark (fewer iterations)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true', 
        help='Run comprehensive benchmark suite'
    )
    
    parser.add_argument(
        '--model-size',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Model size for benchmarking'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for benchmarking'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('.'),
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'html'],
        default='json',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Configure device
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Configure benchmark parameters
    if args.quick:
        print("⚡ Running quick benchmark...")
        config = {'n_runs': 10, 'batch_sizes': [1, 8]}
    elif args.full:
        print("🔬 Running comprehensive benchmark...")
        config = {'n_runs': 100, 'batch_sizes': [1, 8, 32, 64]}
    else:
        print("📊 Running standard benchmark...")
        config = {'n_runs': 50, 'batch_sizes': [1, 8, 32]}
    
    # Run benchmark
    try:
        results = benchmark.generate_comparison_report(
            save_path=args.output / f'benchmark_results.{args.format}'
        )
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        print(f"📊 Format: {args.format}")
        
        # Print summary
        print(f"\n📈 Summary:")
        models = list(results['speed'].keys())
        print(f"Models tested: {', '.join(models)}")
        print(f"Device: {device}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()