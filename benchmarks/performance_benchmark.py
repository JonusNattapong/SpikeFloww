"""
SpikeFlow Performance Benchmark Suite
Compare SNN vs traditional NN performance
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import sys
from typing import Dict, List, Tuple
import json

sys.path.append('..')
import spikeflow as sf

class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def benchmark_inference_speed(self, 
                                 models: Dict[str, nn.Module],
                                 input_sizes: List[Tuple[int, ...]],
                                 batch_sizes: List[int] = [1, 8, 32],
                                 n_runs: int = 100) -> Dict[str, Dict]:
        """Benchmark inference speed across different configurations"""
        
        print("🚀 Benchmarking inference speed...")
        speed_results = {}
        
        for model_name, model in models.items():
            model.to(self.device)
            model.eval()
            
            speed_results[model_name] = {}
            
            for input_size in input_sizes:
                for batch_size in batch_sizes:
                    key = f"input_{input_size}_batch_{batch_size}"
                    
                    # Generate test data
                    test_input = torch.randn(batch_size, *input_size).to(self.device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(10):
                            if 'SNN' in model_name:
                                model.reset_state()
                                _ = model(test_input, time_steps=50)
                            else:
                                _ = model(test_input)
                    
                    # Actual benchmark
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(n_runs):
                            if 'SNN' in model_name:
                                model.reset_state()
                                _ = model(test_input, time_steps=50)
                            else:
                                _ = model(test_input)
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    total_time = time.time() - start_time
                    
                    avg_time = total_time / n_runs
                    throughput = batch_size / avg_time
                    
                    speed_results[model_name][key] = {
                        'avg_time_ms': avg_time * 1000,
                        'throughput_samples_per_sec': throughput
                    }
                    
                    print(f"{model_name} - {key}: {avg_time*1000:.2f}ms, {throughput:.1f} samples/sec")
        
        return speed_results
    
    def benchmark_memory_usage(self, 
                              models: Dict[str, nn.Module],
                              input_size: Tuple[int, ...] = (784,),
                              batch_size: int = 32) -> Dict[str, Dict]:
        """Benchmark memory usage"""
        
        print("💾 Benchmarking memory usage...")
        memory_results = {}
        
        for model_name, model in models.items():
            model.to(self.device)
            
            # Reset memory stats
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            # Measure model parameters
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Measure inference memory
            test_input = torch.randn(batch_size, *input_size).to(self.device)
            
            with torch.no_grad():
                if 'SNN' in model_name:
                    model.reset_state()
                    _ = model(test_input, time_steps=50)
                else:
                    _ = model(test_input)
            
            if self.device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
            else:
                # For CPU, use process memory
                process = psutil.Process()
                peak_memory = process.memory_info().rss
            
            memory_results[model_name] = {
                'param_memory_mb': param_memory / (1024 * 1024),
                'peak_memory_mb': peak_memory / (1024 * 1024),
                'memory_efficiency': param_memory / peak_memory if peak_memory > 0 else 0
            }
            
            print(f"{model_name}: Params {param_memory/(1024*1024):.2f}MB, Peak {peak_memory/(1024*1024):.2f}MB")
        
        return memory_results
    
    def benchmark_energy_efficiency(self,
                                   models: Dict[str, nn.Module],
                                   input_size: Tuple[int, ...] = (784,),
                                   n_inferences: int = 1000) -> Dict[str, Dict]:
        """Estimate energy efficiency"""
        
        print("⚡ Benchmarking energy efficiency...")
        energy_results = {}
        
        for model_name, model in models.items():
            model.to(self.device)
            model.eval()
            
            # Count operations
            total_ops = self._count_operations(model, input_size)
            
            # Estimate energy per operation
            if 'SNN' in model_name:
                # SNNs benefit from sparse computation
                sparsity_factor = 0.1  # Assume 10% activity
                energy_per_op = 0.1e-12  # 0.1 pJ per spike operation
                effective_ops = total_ops * sparsity_factor
            else:
                # Traditional neural networks
                energy_per_op = 10e-12  # 10 pJ per MAC operation
                effective_ops = total_ops
            
            # Calculate energy metrics
            energy_per_inference = effective_ops * energy_per_op * 1e9  # Convert to nJ
            total_energy = energy_per_inference * n_inferences / 1e6  # Convert to mJ
            
            energy_results[model_name] = {
                'total_operations': total_ops,
                'effective_operations': effective_ops,
                'energy_per_inference_nj': energy_per_inference,
                'energy_efficiency_ops_per_nj': effective_ops / energy_per_inference if energy_per_inference > 0 else 0,
                'total_energy_mj': total_energy
            }
            
            print(f"{model_name}: {energy_per_inference:.2f} nJ/inference, {total_energy:.2f} mJ total")
        
        return energy_results
    
    def _count_operations(self, model: nn.Module, input_size: Tuple[int, ...]) -> int:
        """Count number of operations in model"""
        total_ops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_ops += module.weight.numel()
            elif isinstance(module, sf.STDPLinear):
                total_ops += module.weight.numel()
            elif isinstance(module, nn.Conv2d):
                # Simplified conv operation count
                total_ops += module.weight.numel() * np.prod(input_size)
        
        return total_ops
    
    def benchmark_accuracy_vs_efficiency(self,
                                        models: Dict[str, nn.Module],
                                        test_loader,
                                        time_steps_range: List[int] = [10, 25, 50, 100]) -> Dict[str, Dict]:
        """Benchmark accuracy vs efficiency trade-offs"""
        
        print("🎯 Benchmarking accuracy vs efficiency...")
        accuracy_results = {}
        
        for model_name, model in models.items():
            model.to(self.device)
            model.eval()
            
            if 'SNN' not in model_name:
                # Standard model - single evaluation
                accuracy = self._evaluate_accuracy(model, test_loader)
                accuracy_results[model_name] = {
                    'accuracy': accuracy,
                    'time_steps': 1,
                    'efficiency_score': accuracy / 1  # Accuracy per time step
                }
            else:
                # SNN model - evaluate across different time steps
                model_results = {}
                for time_steps in time_steps_range:
                    accuracy = self._evaluate_accuracy(model, test_loader, time_steps)
                    efficiency_score = accuracy / time_steps
                    
                    model_results[f'{time_steps}_steps'] = {
                        'accuracy': accuracy,
                        'time_steps': time_steps,
                        'efficiency_score': efficiency_score
                    }
                
                accuracy_results[model_name] = model_results
        
        return accuracy_results
    
    def _evaluate_accuracy(self, model: nn.Module, test_loader, time_steps: int = None) -> float:
        """Evaluate model accuracy"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if data.dim() == 4:  # Image data
                    data = data.view(data.size(0), -1)
                
                if hasattr(model, 'reset_state'):
                    model.reset_state()
                
                if time_steps:
                    # SNN with time steps
                    spike_data = sf.functional.poisson_encoding(data, time_steps)
                    output = model(spike_data, time_steps=time_steps)
                    predictions = output.sum(dim=0).argmax(dim=1)
                else:
                    # Standard NN
                    output = model(data)
                    predictions = output.argmax(dim=1)
                
                correct += (predictions == target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
    
    def generate_comparison_report(self, save_path: str = 'benchmark_report.json'):
        """Generate comprehensive comparison report"""
        
        print("\n📊 Generating benchmark report...")
        
        # Create test models
        models = self._create_test_models()
        
        # Create dummy test data
        test_input_sizes = [(784,), (1024,), (2048,)]
        dummy_loader = self._create_dummy_loader()
        
        # Run all benchmarks
        speed_results = self.benchmark_inference_speed(models, test_input_sizes)
        memory_results = self.benchmark_memory_usage(models)
        energy_results = self.benchmark_energy_efficiency(models)
        accuracy_results = self.benchmark_accuracy_vs_efficiency(models, dummy_loader)
        
        # Compile results
        full_results = {
            'speed': speed_results,
            'memory': memory_results,
            'energy': energy_results,
            'accuracy': accuracy_results,
            'system_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'pytorch_version': torch.__version__
            }
        }
        
        # Save results
        with open(save_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        # Generate visualization
        self._visualize_results(full_results)
        
        print(f"✅ Benchmark report saved to {save_path}")
        
        return full_results
    
    def _create_test_models(self) -> Dict[str, nn.Module]:
        """Create test models for comparison"""
        
        models = {}
        
        # Traditional Neural Network
        models['Traditional_NN'] = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # SpikeFlow SNN
        models['SpikeFlow_SNN'] = sf.create_snn_classifier(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
            synapse_type='LINEAR'  # For fair comparison
        )
        
        # SpikeFlow SNN with STDP
        models['SpikeFlow_STDP'] = sf.create_snn_classifier(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
            synapse_type='STDP'
        )
        
        return models
    
    def _create_dummy_loader(self):
        """Create dummy data loader for testing"""
        dummy_data = torch.randn(100, 784)
        dummy_targets = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
        return torch.utils.data.DataLoader(dataset, batch_size=32)
    
    def _visualize_results(self, results: Dict):
        """Visualize benchmark results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract model names
        model_names = list(results['speed'].keys())
        
        # 1. Inference Speed Comparison
        speeds = []
        for model in model_names:
            # Get average speed across configurations
            model_speeds = [v['avg_time_ms'] for v in results['speed'][model].values()]
            speeds.append(np.mean(model_speeds))
        
        ax1.bar(model_names, speeds)
        ax1.set_ylabel('Average Inference Time (ms)')
        ax1.set_title('Inference Speed Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Memory Usage Comparison
        memory_usage = [results['memory'][model]['peak_memory_mb'] for model in model_names]
        
        ax2.bar(model_names, memory_usage, color='orange')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Energy Efficiency
        energy_per_inference = [results['energy'][model]['energy_per_inference_nj'] for model in model_names]
        
        ax3.bar(model_names, energy_per_inference, color='green')
        ax3.set_ylabel('Energy per Inference (nJ)')
        ax3.set_title('Energy Efficiency Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        # 4. Efficiency Score (placeholder)
        efficiency_scores = [1000/s for s in speeds]  # Simplified efficiency metric
        
        ax4.bar(model_names, efficiency_scores, color='purple')
        ax4.set_ylabel('Efficiency Score (1/latency)')
        ax4.set_title('Overall Efficiency Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run complete benchmark suite"""
    
    print("🔬 SpikeFlow Performance Benchmark Suite")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    results = benchmark.generate_comparison_report()
    
    print("\n📈 Benchmark Summary:")
    print(f"Models tested: {list(results['speed'].keys())}")
    print(f"System: {results['system_info']['device']}")
    print("\nCheck 'benchmark_report.json' and 'benchmark_visualization.png' for detailed results!")


if __name__ == "__main__":
    main()
