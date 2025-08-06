import torch
import yaml
import argparse
from spikingjelly.activation_based import surrogate

from network.SNN_models_simpquant import (
    SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3,
    SQAKD_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3,
    SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3
)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args():
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument('--config_path', type=str, required=False, default="SQAKD_Geoffroy/configuration.yaml", help='Path to the configuration file.')
    args = parser.parse_args()
    return args

def get_model_weight_types(model, model_name="Model"):
    """
    Get the data types of all weights in a PyTorch model
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display purposes
    """
    print(f"\n=== Weight Types for {model_name} ===")
    
    # Get unique data types in the model
    dtypes = set()
    layer_info = []
    
    for name, param in model.named_parameters():
        dtype = param.dtype
        dtypes.add(dtype)
        layer_info.append((name, dtype, param.shape, param.numel()))
    
    # Print summary of data types
    print(f"Data types found: {[str(dt) for dt in dtypes]}")
    print(f"Total parameters: {sum(param.numel() for param in model.parameters())}")
    
    # Print detailed layer information
    print("\nLayer-wise weight types:")
    print("-" * 80)
    print(f"{'Layer Name':<50} {'Data Type':<15} {'Shape':<20} {'Elements':<10}")
    print("-" * 80)
    
    for name, dtype, shape, num_elements in layer_info:
        print(f"{name:<50} {str(dtype):<15} {str(shape):<20} {num_elements:<10}")
    
    return dtypes, layer_info

def check_specific_layer_types(model, layer_types_to_check=None):
    """
    Check specific types of layers (Conv2d, Linear, etc.) for their weight types
    
    Args:
        model: PyTorch model
        layer_types_to_check: List of layer types to check (default: common layer types)
    """
    if layer_types_to_check is None:
        layer_types_to_check = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.Conv3d]
    
    print(f"\n=== Specific Layer Type Analysis ===")
    
    for layer_type in layer_types_to_check:
        matching_layers = []
        for name, module in model.named_modules():
            if isinstance(module, layer_type):
                if hasattr(module, 'weight') and module.weight is not None:
                    matching_layers.append((name, module.weight.dtype, module.weight.shape))
        
        if matching_layers:
            print(f"\n{layer_type.__name__} layers:")
            for name, dtype, shape in matching_layers:
                print(f"  {name}: {dtype} {shape}")

def get_memory_usage_by_dtype(model):
    """
    Calculate memory usage by data type
    
    Args:
        model: PyTorch model
    """
    print(f"\n=== Memory Usage by Data Type ===")
    
    dtype_memory = {}
    dtype_counts = {}
    
    for name, param in model.named_parameters():
        dtype = param.dtype
        num_elements = param.numel()
        
        # Calculate memory usage (in bytes)
        element_size = param.element_size()  # Size of each element in bytes
        memory_bytes = num_elements * element_size
        
        if dtype not in dtype_memory:
            dtype_memory[dtype] = 0
            dtype_counts[dtype] = 0
        
        dtype_memory[dtype] += memory_bytes
        dtype_counts[dtype] += num_elements
    
    total_memory = sum(dtype_memory.values())
    
    print(f"{'Data Type':<15} {'Parameters':<15} {'Memory (MB)':<15} {'Memory (%)':<15}")
    print("-" * 60)
    
    for dtype in dtype_memory:
        memory_mb = dtype_memory[dtype] / (1024 * 1024)  # Convert to MB
        memory_percent = (dtype_memory[dtype] / total_memory) * 100
        print(f"{str(dtype):<15} {dtype_counts[dtype]:<15} {memory_mb:<15.2f} {memory_percent:<15.1f}")
    
    print(f"\nTotal memory: {total_memory / (1024 * 1024):.2f} MB")


def main(config):
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    # training_set = torch.load(config['datasets']['training_set'])
    # validation_set = torch.load(config['datasets']['validation_set'], weights_only=False)
    # test_set = torch.load(config['datasets']['test_set'])
    
    # Load full precision stereospike model
    full_precision_model =  SQAKD_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3(
        input_chans=4, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True,
        multiply_factor=10., surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)
    full_precision_model.load_state_dict(torch.load(config['model']['checkpoint'], map_location=device))

    # Load quantizable model
    quantized_model = SQAKD_v2_QUANTIZABLE_fromZero_feedforward_multiscale_tempo_Matt_NoskipAll_sepConv_SpikeFlowNetLike_v3(
        input_chans=4, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True,
        multiply_factor=10., surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)
    

    print("Models loaded successfully.")
    
    # Analyze weight types for both models
    print("\n" + "="*80)
    print("WEIGHT TYPE ANALYSIS")
    print("="*80)
    
    # Get weight types for full precision model
    fp_dtypes, fp_layer_info = get_model_weight_types(full_precision_model, "Full Precision Model")
    
    # Get weight types for quantized model
    q_dtypes, q_layer_info = get_model_weight_types(quantized_model, "Quantized Model")


    # Check specific layer types
    print("\n" + "="*50)
    print("FULL PRECISION MODEL - Layer Type Analysis")
    print("="*50)
    check_specific_layer_types(full_precision_model)
    
    print("\n" + "="*50)
    print("QUANTIZED MODEL - Layer Type Analysis")
    print("="*50)
    check_specific_layer_types(quantized_model)
    
    # Memory usage analysis
    print("\n" + "="*50)
    print("FULL PRECISION MODEL - Memory Usage")
    print("="*50)
    get_memory_usage_by_dtype(full_precision_model)
    
    print("\n" + "="*50)
    print("QUANTIZED MODEL - Memory Usage")
    print("="*50)
    get_memory_usage_by_dtype(quantized_model)
    
    
if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)
    print("Configuration loaded successfully.")
    main(config=config)

