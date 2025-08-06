#!/usr/bin/env python3
"""
Utility script to inspect PyTorch model weight types and memory usage
"""

import torch
import argparse
from collections import defaultdict


def inspect_model_weights(model, detailed=False):
    """
    Comprehensive weight type inspection for PyTorch models
    
    Args:
        model: PyTorch model
        detailed: Whether to show detailed layer-by-layer information
    """
    print("="*80)
    print("MODEL WEIGHT TYPE INSPECTION")
    print("="*80)
    
    # Collect information about all parameters
    dtype_info = defaultdict(list)
    total_params = 0
    
    for name, param in model.named_parameters():
        dtype_info[param.dtype].append({
            'name': name,
            'shape': param.shape,
            'numel': param.numel(),
            'requires_grad': param.requires_grad,
            'memory_mb': param.numel() * param.element_size() / (1024 * 1024)
        })
        total_params += param.numel()
    
    # Summary by data type
    print(f"\nSUMMARY:")
    print(f"Total parameters: {total_params:,}")
    print(f"Data types found: {len(dtype_info)}")
    
    print(f"\n{'Data Type':<15} {'Count':<8} {'Parameters':<15} {'Memory (MB)':<12} {'Percentage':<10}")
    print("-" * 70)
    
    for dtype, params in dtype_info.items():
        count = len(params)
        total_elements = sum(p['numel'] for p in params)
        total_memory = sum(p['memory_mb'] for p in params)
        percentage = (total_elements / total_params) * 100
        
        print(f"{str(dtype):<15} {count:<8} {total_elements:<15,} {total_memory:<12.2f} {percentage:<10.1f}%")
    
    # Detailed layer information
    if detailed:
        print(f"\nDETAILED LAYER INFORMATION:")
        print("-" * 80)
        print(f"{'Layer Name':<50} {'Type':<12} {'Shape':<20} {'Elements':<12}")
        print("-" * 80)
        
        for dtype, params in dtype_info.items():
            print(f"\n{str(dtype)} layers:")
            for param in params:
                grad_status = "✓" if param['requires_grad'] else "✗"
                print(f"  {param['name']:<48} {str(dtype):<12} {str(param['shape']):<20} {param['numel']:<12,} ({grad_status})")


def compare_models(*models, model_names=None):
    """
    Compare weight types across multiple models
    
    Args:
        models: Variable number of PyTorch models
        model_names: Optional list of model names
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Collect data for all models
    all_model_data = []
    
    for i, model in enumerate(models):
        dtype_summary = defaultdict(int)
        total_params = 0
        
        for name, param in model.named_parameters():
            dtype_summary[param.dtype] += param.numel()
            total_params += param.numel()
        
        all_model_data.append({
            'name': model_names[i],
            'total_params': total_params,
            'dtype_summary': dict(dtype_summary)
        })
    
    # Find all unique dtypes
    all_dtypes = set()
    for model_data in all_model_data:
        all_dtypes.update(model_data['dtype_summary'].keys())
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Total Params':<15}", end="")
    for dtype in sorted(all_dtypes, key=str):
        print(f"{str(dtype):<15}", end="")
    print()
    print("-" * (35 + len(all_dtypes) * 15))
    
    for model_data in all_model_data:
        print(f"{model_data['name']:<20} {model_data['total_params']:<15,}", end="")
        for dtype in sorted(all_dtypes, key=str):
            count = model_data['dtype_summary'].get(dtype, 0)
            if count > 0:
                print(f"{count:<15,}", end="")
            else:
                print(f"{'0':<15}", end="")
        print()


def get_layer_type_distribution(model):
    """
    Get distribution of different layer types and their weight dtypes
    """
    print("="*80)
    print("LAYER TYPE DISTRIBUTION")
    print("="*80)
    
    layer_types = defaultdict(list)
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            layer_type = type(module).__name__
            layer_types[layer_type].append({
                'name': name,
                'dtype': module.weight.dtype,
                'shape': module.weight.shape,
                'numel': module.weight.numel()
            })
    
    for layer_type, layers in layer_types.items():
        print(f"\n{layer_type} ({len(layers)} layers):")
        dtype_counts = defaultdict(int)
        total_params = 0
        
        for layer in layers:
            dtype_counts[layer['dtype']] += layer['numel']
            total_params += layer['numel']
        
        for dtype, count in dtype_counts.items():
            percentage = (count / total_params) * 100 if total_params > 0 else 0
            print(f"  {str(dtype)}: {count:,} parameters ({percentage:.1f}%)")


def main():
    """
    Example usage of the weight inspection utilities
    """
    parser = argparse.ArgumentParser(description="Inspect PyTorch model weight types")
    parser.add_argument('--detailed', action='store_true', help="Show detailed layer information")
    args = parser.parse_args()
    
    # Example: Create some dummy models with different dtypes
    print("Creating example models...")
    
    # Float32 model
    model_fp32 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).float()
    
    # Float16 model
    model_fp16 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).half()
    
    # Inspect individual models
    print("\nInspecting FP32 model:")
    inspect_model_weights(model_fp32, detailed=args.detailed)
    
    print("\nInspecting FP16 model:")
    inspect_model_weights(model_fp16, detailed=args.detailed)
    
    # Compare models
    print("\nComparing models:")
    compare_models(model_fp32, model_fp16, model_names=["FP32 Model", "FP16 Model"])
    
    # Layer type distribution
    print("\nLayer type distribution (FP32 model):")
    get_layer_type_distribution(model_fp32)


if __name__ == "__main__":
    main()
