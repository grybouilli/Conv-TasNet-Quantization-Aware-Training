"""
Model quantization utilities.

This module provides utilities for transforming standard PyTorch models
into quantized versions by replacing layers with their quantized equivalents.
"""

from typing import Optional, Dict, Iterable
import torch
import torch.nn as nn

from hatorch.layers.quantized_conv2d import QuantConv2d
from hatorch.layers.quantized_conv1d import QuantConv1d
from hatorch.layers.quantized_convtranspose1d import QuantConvTranspose1d
from hatorch.layers.quantized_sigmoid import QuantSigmoid
from hatorch.layers.quantized_prelu import QuantPReLU

from hatorch.layers.quantized_conv2d_bn_relu import QuantConvBn2d, QuantConvBnReLU2d, QuantConvReLU2d
from hatorch.layers.quantized_linear import QuantLinear
from hatorch.layers.quantized_relu import QuantReLU
from hatorch.layers.quantized_maxpool2d import QuantMaxPool2d
from hatorch.layers.quantized_averagepool2d import QuantAveragePool2d
from hatorch.layers.quantized_dropout import QuantDropout
from hatorch.layers.quantized_resunit import QuantResUnit
from hatorch.modules.quantize import Quantize
from hatorch.modules.dequantize import Dequantize
from hatorch.transforms.config import LayerQuantConfig, ModelQuantConfig, QuantizerSpec
from hatorch.quantizers.base import BaseQuantizer
from hatorch.modules.quantize_wrapper import QuantizeWrapper
from hatorch.modules.hatorch_quantized_layer import HatorchQuantizedLayer, HatorchQuantizedOperator
from hatorch.quantizers.common._common_quantizer import BaseLearnedQuantizer
from hatorch.quantizers.common.scale_approximation import ScaleApproximation
from hatorch.quantizers.init.lsq_uniform import LsqUniformQuantizer

from hatorch.operators.quantized_add import QuantAdd
from hatorch.operators.quantized_mul import QuantMul

import operator
from torch.fx.passes.shape_prop import ShapeProp


HATORCH_QUANTIZED_LAYERS_CLASS = [QuantConv2d, QuantConv1d, QuantConvTranspose1d, QuantPReLU, QuantSigmoid]


def quantize_model(
    model: nn.Module,
    config: ModelQuantConfig,
    inplace: bool = False,
    add_quantize: bool = True,
    add_dequantize: bool = True,
    example_args: Optional[tuple[torch.Tensor]] = None,
    use_wrapper: bool = False,
    replace_quantizable_operators_using_tracing: bool = False,
    additional_hatorch_quantized_layers_class: list[HatorchQuantizedLayer] | None = None,
) -> QuantizeWrapper:
    """
    Transform a model by replacing standard layers with quantized versions.
    
    Args:
        model: Original PyTorch model
        config: Quantization configuration specifying which quantizers to use
        inplace: Whether to modify the model in-place (default: False)
        add_quantize: Whether to add a Quantize layer at the beginning (default: True)
        add_dequantize: Whether to add a Dequantize layer at the end (default: True)
        use_wrapper: Whether to wrap the entire model in a QuantizeWrapper (default: False)
        replace_quantizable_operators_using_tracing: Whether to replace quantizable operators using tracing (addition, multiplication...) (default: False)

    Returns:
        Quantized model
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    if replace_quantizable_operators_using_tracing:
        if inplace:
            raise ValueError("Cannot use tracing to replace operators in-place")
        if example_args is None:
            raise ValueError("example_args must be provided when using tracing to replace operators")
        model = _tracing_replace_all_quantizable_operators(model, example_args, config)



    # Add Quantize and Dequantize layers directly into the model structure
    if add_quantize or add_dequantize:
        if use_wrapper:
            # Wrap the entire model in a QuantizeWrapper
            if inplace:
                raise ValueError("Cannot use wrapper in-place since it replaces the entire model.")
            model = QuantizeWrapper(model)
        else:
            _inject_quantize_dequantize_layers(model, add_quantize, add_dequantize)
    
    # First pass: identify all quantizable layers to determine which is last
    quantizable_layers = _find_quantizable_layers(model, additional_hatorch_quantized_layers_class=additional_hatorch_quantized_layers_class)
    
    # Recursively replace layers
    _replace_layers_recursive(model, config, "", quantizable_layers, additional_hatorch_quantized_layers_class=additional_hatorch_quantized_layers_class)
    

    # In DDP mode, BatchNorm layers must synchronize running stats across ranks.
    if _should_convert_sync_batchnorm():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    _assign_quantizer_debug_labels(model)
    if config.freeze_approximated_scales:
        freeze_approximated_lsq_scales(model)
    
   
    return model

def is_tensor_node(node):
    return (
        isinstance(node, torch.fx.Node)
        and node.meta["type"] is torch.Tensor
    )

def _tracing_replace_operator_node_by_pattern_with_quantized_module(
    search_patterns: set,
    quantized_module_class: type[HatorchQuantizedOperator],
    gm: torch.fx.GraphModule,
    config: ModelQuantConfig,
):
    """
    Replace nodes in the graph with quantized modules.
    """
    counter = 0

    for n in gm.graph.nodes:
        # If the target matches one of the patterns
        if any(n.target == pattern for pattern in search_patterns):
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            name = f"quant_{quantized_module_class.operator_name}_{counter}"
            layer_config = config.get_config_for_layer(name)

            counter += 1

            quantize_left_activation = is_tensor_node(n.args[0])
            quantize_right_activation = is_tensor_node(n.args[1])
            quant_add = quantized_module_class.from_config(quantize_left_activation=quantize_left_activation, quantize_right_activation=quantize_right_activation, layer_config=layer_config)
            
            gm.add_module(name, quant_add)
            with gm.graph.inserting_after(n):
                new_node = gm.graph.call_module(name, args=n.args, kwargs=n.kwargs)
                # We use the meta field to check if the input args are tensors (and should thus be quantized).
                # We copy the meta field to the newly created node to preserve the information for later nodes that will use this node as arg
                new_node.meta = n.meta.copy()

                print(f"Replacing {n} with {new_node}")
                print(n.args)
                print(n.kwargs)
                n.replace_all_uses_with(new_node)
            # Remove the old node from the graph
            gm.graph.erase_node(n)

def _tracing_replace_all_quantizable_operators(model: nn.Module, example_args: tuple[torch.Tensor], config: ModelQuantConfig) -> nn.Module:
    """
    Replace Tensor operators by quantized versions using torch tracing API

    The following operators will be replaced by their quantized versions:
    - addition (`torch.add`, `operator.add` and `tensor.add`)
    - multiplication (`torch.mul`, `operator.mul` and `tensor.mul`)

    This method should be called on a non-quantized nn.Module (ie: before replacing layers with quantized versions).
    The model will be traced and the operators will be replaced by their quantized versions.
    """
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*example_args)

    add_patterns = set([operator.add, torch.add, "add"])
    mul_patterns = set([operator.mul, torch.mul, "mul"])
    #add_patterns = set([torch.ops.aten.add.Tensor,])
    _tracing_replace_operator_node_by_pattern_with_quantized_module(
        add_patterns,
        QuantAdd,
        gm,
        config,
    )

    # _tracing_replace_operator_node_by_pattern_with_quantized_module(
    #     mul_patterns,
    #     QuantMul,
    #     gm,
    #     config,
    # )

    # Does some checks to make sure the
    # graph is well-formed.
    gm.graph.lint()
    # Don't forget to recompile!
    gm.recompile()

    gm.graph.print_tabular()

    return gm

def _assign_quantizer_debug_labels(model: nn.Module) -> None:
    """Attach stable module-path labels to quantizers for debugging."""
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantizer):
            module.debug_label = name


def _should_convert_sync_batchnorm() -> bool:
    """Return True when distributed training with more than one rank is active."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    return torch.distributed.get_world_size() > 1


def _find_quantizable_layers(
    module: nn.Module,
    prefix: str = "",
    additional_hatorch_quantized_layers_class: list[HatorchQuantizedLayer] | None = None,
) -> list:
    """
    Find all quantizable layers (Conv2d, Linear) in the model.

    Args:
        module: Module to scan
        prefix: Current module path

    Returns:
        List of tuples (full_name, layer_type) for all quantizable layers
    """
    quantizable = []
    children_list = list(module.named_children())

    for i, (name, child) in enumerate(children_list):
        full_name = f"{prefix}.{name}" if prefix else name

        found = False
        for layer_class in HATORCH_QUANTIZED_LAYERS_CLASS + (additional_hatorch_quantized_layers_class or []):
            if isinstance(child, layer_class.layer_non_quantized_class):
                quantizable.append((full_name, layer_class.layer_non_quantized_class.__name__))
                found = True
                break
        if not found:
            if isinstance(child, nn.Linear):
                quantizable.append((full_name, 'Linear'))
            elif hasattr(child, 'children') and len(list(child.children())) > 0:
                quantizable.extend(_find_quantizable_layers(child, full_name, additional_hatorch_quantized_layers_class=additional_hatorch_quantized_layers_class))
        
    return quantizable


def _inject_quantize_dequantize_layers(
    module: nn.Module,
    add_quantize: bool,
    add_dequantize: bool,
) -> None:
    """
    Inject Quantize and Dequantize layers into the model structure.

    Quantize goes at the start of the first Sequential and Dequantize at the
    very end of the model. Intermediate Dequantize -> Quantize barriers are
    intentionally avoided here so quantization metadata can propagate across
    sequential blocks.

    Args:
        module: Module to modify
        add_quantize: Whether to add Quantize layer at the beginning
        add_dequantize: Whether to add Dequantize layer at the end
    """
    children = list(module.named_children())

    if not children:
        return

    # Find all Sequential containers
    sequential_containers = []
    for name, child in children:
        if isinstance(child, nn.Sequential):
            sequential_containers.append((name, child))

    if sequential_containers:
        # Add Quantize to the beginning of the FIRST Sequential
        if add_quantize:
            first_name, first_seq = sequential_containers[0]
            layers = [Quantize()]
            for layer in first_seq.children():
                layers.append(layer)
            setattr(module, first_name, nn.Sequential(*layers))

        # Ensure final output is dequantized
        if add_dequantize:
            # Refresh last child in case we modified earlier children
            last_name, _ = children[-1]
            last_child_current = getattr(module, last_name)
            if isinstance(last_child_current, nn.Sequential):
                last_layers = list(last_child_current.children())
                if not last_layers or not isinstance(last_layers[-1], Dequantize):
                    last_layers.append(Dequantize())
                    setattr(module, last_name, nn.Sequential(*last_layers))
            else:
                # Last child is not Sequential: only append the final dequantize.
                setattr(module, last_name, nn.Sequential(last_child_current, Dequantize()))
    else:
        # No Sequential found - wrap all children
        if add_quantize or add_dequantize:
            layers = []
            if add_quantize:
                layers.append(Quantize())

            children_to_wrap = []
            for name, child in children:
                children_to_wrap.append((name, child))

            for name, child in children_to_wrap:
                layers.append(child)

            if add_dequantize:
                layers.append(Dequantize())

            for name, _ in children_to_wrap:
                delattr(module, name)

            module.add_module('_quantized_sequential', nn.Sequential(*layers))


def _replace_layers_recursive(
    module: nn.Module,
    config: ModelQuantConfig,
    prefix: str = "",
    quantizable_layers: list = None,
    layer_index: list = None,
    additional_hatorch_quantized_layers_class: list[HatorchQuantizedLayer] | None = None,
) -> None:
    """
    Recursively replace layers in a module with quantized versions.

    Detects and fuses common patterns:
    - Conv2d + BatchNorm2d + ReLU -> QuantConvBnReLU2d
    - Conv2d + BatchNorm2d -> QuantConvBn2d
    - Conv2d + ReLU -> QuantConvReLU2d
    - Conv2d -> QuantConv2d
    - Linear -> QuantLinear

    Args:
        module: Module to process
        config: Quantization configuration
        prefix: Current module path (for naming)
        quantizable_layers: List of all quantizable layer names (for tracking first/last)
        layer_index: Mutable list containing current index [idx]
    """
    if quantizable_layers is None:
        quantizable_layers = []
    if layer_index is None:
        layer_index = [0]

    children_list = list(module.named_children())
    skip_next = set()  # Track indices to skip due to fusion

    for i, (name, child) in enumerate(children_list):
        if i in skip_next:
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        # Determine if this is first or last layer
        is_first_layer = (layer_index[0] == 0) if quantizable_layers else True
        is_last_layer = (layer_index[0] == len(quantizable_layers) - 1) if quantizable_layers else False

        # Get configuration for this layer
        layer_config = config.get_config_for_layer(full_name, is_first_layer=is_first_layer, is_last_layer=is_last_layer)

        # Try to detect ConvBlock pattern (from pytorchcv models)
        # ConvBlock structure: conv (Conv2d) + bn (BatchNorm2d) + [optional] activ (ReLU)
        if _is_convblock(child):
            # Replace ConvBlock with fused quantized layer
            quant_layer = _create_quant_from_convblock(
                child,
                layer_config,
                fold_batch_norm=config.fold_batch_norm,
            )
            setattr(module, name, quant_layer)
            layer_index[0] += 1
            continue

        # Try to detect ResUnit pattern (from pytorchcv ResNet models)
        # ResUnit can be BasicBlock (2 convs) or Bottleneck (3 convs)
        if _is_resunit(child):
            total_layers = len(quantizable_layers) if quantizable_layers is not None else 0
            current_idx = layer_index[0]
            # Detect identity conv presence to keep layer indexing aligned
            has_identity_conv = hasattr(child, "identity_conv") and hasattr(child.identity_conv, "conv")

            # Get configs for conv layers in the ResUnit
            conv1_config = config.get_config_for_layer(
                f"{full_name}.body.conv1",
                is_first_layer=is_first_layer,
                is_last_layer=(current_idx == total_layers - 1),
            )
            conv2_config = config.get_config_for_layer(
                f"{full_name}.body.conv2",
                is_first_layer=False,
                is_last_layer=(current_idx + 1 == total_layers - 1),
            )

            # Check if it's a bottleneck (3 convs)
            conv3_config = None
            num_convs = 2
            if hasattr(child.body, 'conv3'):
                conv3_config = config.get_config_for_layer(f"{full_name}.body.conv3",
                                                            is_first_layer=False,
                                                            is_last_layer=(current_idx + 2 == total_layers - 1))
                num_convs = 3
            # Include identity conv in index accounting
            conv_count = num_convs + (1 if has_identity_conv else 0)

            # Create quantized ResUnit (or return unchanged if bottleneck)
            quant_resunit = _create_quant_resunit(
                child,
                conv1_config,
                conv2_config,
                conv3_config,
                scale_approximation=config.scale_approximation,
                fixed_point_bits=config.fixed_point_bits,
                fold_batch_norm=config.fold_batch_norm,
            )
            setattr(module, name, quant_resunit)
            # Count the number of conv layers
            layer_index[0] += conv_count
            # Don't recurse into the ResUnit - it's already been handled
            continue

        fusion_successful = False
        # Try to detect Conv2d fusion patterns
        if isinstance(child, nn.Conv2d):
            # Look ahead for BatchNorm2d and/or ReLU
            next_bn = None
            next_relu = None
            bn_name = None
            relu_name = None

            # Check for BatchNorm after Conv
            if i + 1 < len(children_list):
                next_name, next_child = children_list[i + 1]
                if isinstance(next_child, nn.BatchNorm2d):
                    next_bn = next_child
                    bn_name = next_name

                    # Check for ReLU after BatchNorm
                    if i + 2 < len(children_list):
                        relu_candidate_name, relu_candidate = children_list[i + 2]
                        if isinstance(relu_candidate, nn.ReLU):
                            next_relu = relu_candidate
                            relu_name = relu_candidate_name
                elif isinstance(next_child, nn.ReLU):
                    # Conv + ReLU (no BatchNorm)
                    next_relu = next_child
                    relu_name = next_name

            # Create the appropriate fused layer
            if next_bn is not None and next_relu is not None:
                # Conv + BN + ReLU fusion
                quant_layer = _create_quant_conv_bn_relu(
                    child,
                    next_bn,
                    layer_config,
                    fold_batch_norm=config.fold_batch_norm,
                )
                setattr(module, name, quant_layer)
                # Remove the BN and ReLU layers
                delattr(module, bn_name)
                delattr(module, relu_name)
                skip_next.add(i + 1)
                skip_next.add(i + 2)
                layer_index[0] += 1
                fusion_successful = True
            elif next_bn is not None:
                # Conv + BN fusion
                quant_layer = _create_quant_conv_bn(
                    child,
                    next_bn,
                    layer_config,
                    fold_batch_norm=config.fold_batch_norm,
                )
                setattr(module, name, quant_layer)
                # Remove the BN layer
                delattr(module, bn_name)
                skip_next.add(i + 1)
                layer_index[0] += 1
                fusion_successful = True
            elif next_relu is not None:
                # Conv + ReLU fusion
                quant_layer = _create_quant_conv_relu(child, layer_config)
                setattr(module, name, quant_layer)
                # Remove the ReLU layer
                delattr(module, relu_name)
                skip_next.add(i + 1)
                layer_index[0] += 1
                fusion_successful = True
            else:
                fusion_successful = False

        if not fusion_successful:
            # Replace Linear layers
            if isinstance(child, nn.Linear):
                quant_layer = _create_quant_linear(child, layer_config)
                setattr(module, name, quant_layer)
                layer_index[0] += 1

            # Replace standalone ReLU layers (not fused with Conv)
            elif isinstance(child, nn.ReLU):
                quant_layer = QuantReLU(inplace=child.inplace)
                setattr(module, name, quant_layer)

            # Replace MaxPool2d layers
            elif isinstance(child, nn.MaxPool2d):
                quant_layer = QuantMaxPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    return_indices=child.return_indices,
                    ceil_mode=child.ceil_mode
                )
                setattr(module, name, quant_layer)

            elif isinstance(child, nn.AvgPool2d):
                quant_layer = QuantAveragePool2d(
                    kernel_size= child.kernel_size,
                    stride= child.stride,
                    padding= child.padding,
                    ceil_mode= child.ceil_mode,
                    count_include_pad= child.count_include_pad,
                    divisor_override= child.divisor_override,
                )
                setattr(module, name, quant_layer)

            # Replace Dropout layers
            elif isinstance(child, nn.Dropout):
                quant_layer = QuantDropout(p=child.p, inplace=child.inplace)
                setattr(module, name, quant_layer)
            
            else:
                replacement_successful = False
                for layer_class in HATORCH_QUANTIZED_LAYERS_CLASS + (additional_hatorch_quantized_layers_class or []):
                    if isinstance(child, layer_class.layer_non_quantized_class):
                        quant_layer = layer_class.from_non_quantized_layer(layer=child, layer_config=layer_config)
                        setattr(module, name, quant_layer)
                        layer_index[0] += 1
                        replacement_successful = True
                        break
            
                if not replacement_successful:
                    # Always recursively process child modules (for nested structures)
                    if hasattr(child, 'children') and len(list(child.children())) > 0:
                        _replace_layers_recursive(child, config, full_name, quantizable_layers, layer_index, additional_hatorch_quantized_layers_class=additional_hatorch_quantized_layers_class)

def _create_weight_quantizer(_weight: torch.Tensor, spec: Optional[QuantizerSpec], num_channels: int):
    """Create weight quantizer from specification."""
    if spec is None:
        return None
    return spec.create_quantizer(num_channels)


def _create_quant_conv_bn(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    config: LayerQuantConfig,
    fold_batch_norm: bool = True,
) -> QuantConvBn2d:
    """
    Create a QuantConvBn2d from Conv2d and BatchNorm2d layers.

    Args:
        conv: Original Conv2d layer
        bn: Original BatchNorm2d layer
        config: Layer quantization configuration

    Returns:
        QuantConvBn2d layer
    """
    # Create quantizers from config
    weight_quantizer = _create_weight_quantizer(conv.weight, config.weight_quantizer, conv.out_channels)

    activation_quantizer = None
    if config.activation_quantizer is not None:
        activation_quantizer = config.activation_quantizer.create_quantizer()

    bias_quantizer = None
    if config.bias_quantizer is not None:
        bias_quantizer = config.bias_quantizer

    # Create fused quantized layer
    return QuantConvBn2d.from_conv_bn(
        conv,
        bn,
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer,
        bias_quantizer=bias_quantizer,
        freeze_bn=False,
        fold_batch_norm=fold_batch_norm,
    )


def _create_quant_conv_bn_relu(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    config: LayerQuantConfig,
    fold_batch_norm: bool = True,
) -> QuantConvBnReLU2d:
    """
    Create a QuantConvBnReLU2d from Conv2d, BatchNorm2d, and ReLU layers.

    Args:
        conv: Original Conv2d layer
        bn: Original BatchNorm2d layer
        config: Layer quantization configuration

    Returns:
        QuantConvBnReLU2d layer
    """
    # Create quantizers from config
    weight_quantizer = _create_weight_quantizer(conv.weight, config.weight_quantizer, conv.out_channels)

    activation_quantizer = None
    if config.activation_quantizer is not None:
        activation_quantizer = config.activation_quantizer.create_quantizer()

    bias_quantizer = None
    if config.bias_quantizer is not None:
        bias_quantizer = config.bias_quantizer

    # Create fused quantized layer
    return QuantConvBnReLU2d.from_conv_bn(
        conv,
        bn,
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer,
        bias_quantizer=bias_quantizer,
        freeze_bn=False,
        fold_batch_norm=fold_batch_norm,
    )


def _create_quant_conv_relu(
    conv: nn.Conv2d,
    config: LayerQuantConfig,
) -> QuantConvReLU2d:
    """
    Create a QuantConvReLU2d from Conv2d and ReLU layers.

    Args:
        conv: Original Conv2d layer
        config: Layer quantization configuration

    Returns:
        QuantConvReLU2d layer
    """
    # Create quantizers from config
    weight_quantizer = _create_weight_quantizer(conv.weight, config.weight_quantizer, conv.out_channels)

    activation_quantizer = None
    if config.activation_quantizer is not None:
        activation_quantizer = config.activation_quantizer.create_quantizer()

    bias_quantizer = None
    if config.bias_quantizer is not None:
        bias_quantizer = config.bias_quantizer

    # Create fused quantized layer
    return QuantConvReLU2d.from_conv(
        conv,
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer,
        bias_quantizer=bias_quantizer,
    )


def _is_convblock(module: nn.Module) -> bool:
    """
    Check if a module is a ConvBlock from pytorchcv.

    ConvBlock structure:
        - conv: Conv2d
        - bn: BatchNorm2d
        - [optional] activ: ReLU

    Args:
        module: Module to check

    Returns:
        True if module is a ConvBlock
    """
    # Check for the characteristic structure
    if not (hasattr(module, 'conv') and hasattr(module, 'bn')):
        return False

    # Check types
    if not isinstance(module.conv, nn.Conv2d) or not isinstance(module.bn, nn.BatchNorm2d):
        return False

    return True


def _create_quant_from_convblock(
    convblock: nn.Module,
    config: LayerQuantConfig,
    fold_batch_norm: bool = True,
) -> nn.Module:
    """
    Create a quantized layer from a ConvBlock module.

    Args:
        convblock: Original ConvBlock module
        config: Layer quantization configuration

    Returns:
        Quantized layer (QuantConvBnReLU2d or QuantConvBn2d)
    """
    conv = convblock.conv
    bn = convblock.bn
    has_activ = hasattr(convblock, 'activ') and isinstance(convblock.activ, nn.ReLU)

    # Create quantizers from config
    weight_quantizer = _create_weight_quantizer(conv.weight, config.weight_quantizer, conv.out_channels)

    activation_quantizer = None
    if config.activation_quantizer is not None:
        activation_quantizer = config.activation_quantizer.create_quantizer()

    bias_quantizer = None
    if config.bias_quantizer is not None:
        bias_quantizer = config.bias_quantizer

    # Create appropriate fused quantized layer
    if has_activ:
        return QuantConvBnReLU2d.from_conv_bn(
            conv,
            bn,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
            freeze_bn=False,
            fold_batch_norm=fold_batch_norm,
        )
    else:
        return QuantConvBn2d.from_conv_bn(
            conv,
            bn,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer,
            bias_quantizer=bias_quantizer,
            freeze_bn=False,
            fold_batch_norm=fold_batch_norm,
        )


def _is_resunit(module: nn.Module) -> bool:
    """
    Check if a module is a ResUnit from pytorchcv.

    ResUnit can have two structures:
    1. BasicBlock (2 convs):
        - body: ResBlock
            - conv1: ConvBlock (Conv2d + BatchNorm2d + ReLU)
            - conv2: ConvBlock (Conv2d + BatchNorm2d)
        - activ: ReLU

    2. Bottleneck (3 convs):
        - body: ResBottleneck
            - conv1: ConvBlock (1x1 Conv2d + BatchNorm2d + ReLU)
            - conv2: ConvBlock (3x3 Conv2d + BatchNorm2d + ReLU)
            - conv3: ConvBlock (1x1 Conv2d + BatchNorm2d)
        - activ: ReLU
        - [often has] identity_conv: for dimension matching

    Args:
        module: Module to check

    Returns:
        True if module is a ResUnit (either BasicBlock or Bottleneck)
    """
    # Check for the characteristic structure
    if not hasattr(module, 'body') or not hasattr(module, 'activ'):
        return False

    body = module.body
    # Must have at least conv1 and conv2
    if not hasattr(body, 'conv1') or not hasattr(body, 'conv2'):
        return False

    # Check conv1 has conv, bn, activ
    conv1 = body.conv1
    if not (hasattr(conv1, 'conv') and hasattr(conv1, 'bn') and hasattr(conv1, 'activ')):
        return False

    # Check conv2 - it should have conv, bn, and may or may not have activ
    conv2 = body.conv2
    if not (hasattr(conv2, 'conv') and hasattr(conv2, 'bn')):
        return False

    # Check types for conv1 and conv2
    if not isinstance(conv1.conv, nn.Conv2d) or not isinstance(conv1.bn, nn.BatchNorm2d):
        return False
    if not isinstance(conv2.conv, nn.Conv2d) or not isinstance(conv2.bn, nn.BatchNorm2d):
        return False

    # If conv3 exists (bottleneck), validate it too
    if hasattr(body, 'conv3'):
        conv3 = body.conv3
        if not (hasattr(conv3, 'conv') and hasattr(conv3, 'bn')):
            return False
        if not isinstance(conv3.conv, nn.Conv2d) or not isinstance(conv3.bn, nn.BatchNorm2d):
            return False

    if not isinstance(module.activ, nn.ReLU):
        return False

    return True


def _create_quant_resunit(
    resunit: nn.Module,
    conv1_config: LayerQuantConfig,
    conv2_config: LayerQuantConfig,
    conv3_config: Optional[LayerQuantConfig] = None,
    scale_approximation: ScaleApproximation = ScaleApproximation.NONE,
    fixed_point_bits: int = 8,
    fold_batch_norm: bool = True,
) -> QuantResUnit:
    """
    Create a QuantResUnit from a ResUnit module.

    Handles both BasicBlock (2 convs) and Bottleneck (3 convs) structures.

    Args:
        resunit: Original ResUnit module
        conv1_config: Configuration for first conv layer
        conv2_config: Configuration for second conv layer
        conv3_config: Configuration for third conv layer (bottleneck only)

    Returns:
        QuantResUnit with quantized convolutions
    """
    # Note: Current QuantResUnit only supports 2-conv BasicBlock structure
    # For bottleneck (3 convs), we fall back to leaving it unquantized for now
    # TODO: Implement QuantResUnitBottleneck for 3-conv case

    body = resunit.body
    has_conv3 = hasattr(body, 'conv3')

    if has_conv3:
        # Bottleneck structure (3 convs) - not yet supported
        # Return the original resunit unchanged
        # User can implement QuantResUnitBottleneck separately
        return resunit

    # BasicBlock structure (2 convs) - proceed with quantization
    # Create quantizers for conv1
    conv1_weight_quantizer = None
    if conv1_config.weight_quantizer is not None:
        # Get out_channels from conv1
        out_channels = resunit.body.conv1.conv.out_channels
        conv1_weight_quantizer = _create_weight_quantizer(
            resunit.body.conv1.conv.weight,
            conv1_config.weight_quantizer,
            out_channels,
        )

    conv1_activation_quantizer = None
    if conv1_config.activation_quantizer is not None:
        conv1_activation_quantizer = conv1_config.activation_quantizer.create_quantizer()

    conv1_bias_quantizer = None
    if conv1_config.bias_quantizer is not None:
        conv1_bias_quantizer = conv1_config.bias_quantizer

    # Create quantizers for conv2
    conv2_weight_quantizer = None
    if conv2_config.weight_quantizer is not None:
        # Get out_channels from conv2
        out_channels = resunit.body.conv2.conv.out_channels
        conv2_weight_quantizer = _create_weight_quantizer(
            resunit.body.conv2.conv.weight,
            conv2_config.weight_quantizer,
            out_channels,
        )

    conv2_activation_quantizer = None
    if conv2_config.activation_quantizer is not None:
        conv2_activation_quantizer = conv2_config.activation_quantizer.create_quantizer()

    conv2_bias_quantizer = None
    if conv2_config.bias_quantizer is not None:
        conv2_bias_quantizer = conv2_config.bias_quantizer

    # Create QuantResUnit using factory method, pass conv2_config for identity
    return QuantResUnit.from_resunit(
        resunit,
        conv1_weight_quantizer=conv1_weight_quantizer,
        conv1_activation_quantizer=conv1_activation_quantizer,
        conv1_bias_quantizer=conv1_bias_quantizer,
        conv2_weight_quantizer=conv2_weight_quantizer,
        conv2_activation_quantizer=conv2_activation_quantizer,
        conv2_bias_quantizer=conv2_bias_quantizer,
        freeze_bn=False,
        identity_config=conv2_config,  # Use conv2_config for identity path
        scale_approximation=scale_approximation,
        fixed_point_bits=fixed_point_bits,
        fold_batch_norm=fold_batch_norm,
    )


def _create_quant_linear(
    linear: nn.Linear,
    config: LayerQuantConfig,
) -> QuantLinear:
    """
    Create a QuantLinear from a Linear layer with specified configuration.

    Args:
        linear: Original Linear layer
        config: Layer quantization configuration

    Returns:
        QuantLinear layer
    """
    # Create quantizers from config
    weight_quantizer = _create_weight_quantizer(linear.weight, config.weight_quantizer, linear.out_features)

    activation_quantizer = None
    if config.activation_quantizer is not None:
        activation_quantizer = config.activation_quantizer.create_quantizer()

    bias_quantizer = None
    if config.bias_quantizer is not None:
        bias_quantizer = config.bias_quantizer

    # Create quantized layer
    return QuantLinear.from_linear(
        linear,
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer,
        bias_quantizer=bias_quantizer,
    )


def set_quantizers_to_mode(
    model: nn.Module,
    mode: str,
) -> None:
    """
    Set all quantizers in a model to a specific mode.

    Args:
        model: Model containing quantizers
        mode: Mode to set ('calibration', 'training', 'eval', 'disable', 'enable')
    """
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            if mode == 'calibration':
                module.calibration_mode_(True)
                module.enable()
            elif mode == 'training':
                module.calibration_mode_(False)
                module.enable()
                module.train()
            elif mode == 'eval':
                module.calibration_mode_(False)
                module.enable()
                module.eval()
            elif mode == 'disable':
                module.disable()
            elif mode == 'enable':
                module.enable()
            else:
                raise ValueError(f"Unknown mode: {mode}")

def freeze_model_parameters(model: nn.Module) -> Dict[int, bool]:
    # start by storing the requires_grad state of each parameter in a dict
    param_states = {id(p): p.requires_grad for p in model.parameters()}

    # then set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False

    return param_states

def unfreeze_model_parameters(model: nn.Module, param_states: Dict[int, bool]) -> None:
    # restore the requires_grad state of each parameter from the provided dict
    for param in model.parameters():
        if id(param) in param_states:
            param.requires_grad = param_states[id(param)]

def freeze_quantizer_parameters(model: nn.Module) -> None:
    """
    Freeze all quantizer parameters (e.g., scales) in a model.

    Useful for fine-tuning only the weights while keeping quantization fixed.

    Args:
        model: Model containing quantizers
    """
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_quantizer_parameters(model: nn.Module) -> None:
    """
    Unfreeze all quantizer parameters in a model.

    Args:
        model: Model containing quantizers
    """
    for module in model.modules():
        if isinstance(module, BaseQuantizer):
            for param in module.parameters():
                param.requires_grad = True


def get_lsq_learnable_parameters(
    model: nn.Module,
    include_scale: bool = True,
    include_zero_point: bool = False,
    weights_only: bool = False,
    activations_only: bool = False,
) -> list[nn.Parameter]:
    """
    Collect learnable LSQ quantizer parameters from a model.

    Args:
        model: Model containing LSQ quantizers.
        include_scale: Include learnable LSQ scales.
        include_zero_point: Include learnable LSQ zero-points.
        weights_only: Only include weight quantizers.
        activations_only: Only include activation quantizers.

    Returns:
        List of unique parameters matching the requested filter.
    """
    if weights_only and activations_only:
        raise ValueError("Cannot specify both weights_only and activations_only")

    params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        if not isinstance(module, BaseLearnedQuantizer):
            continue
        if weights_only and module.is_activation:
            continue
        if activations_only and not module.is_activation:
            continue

        candidates: list[nn.Parameter] = []
        if include_scale and isinstance(module.scale, nn.Parameter):
            if module.scale.requires_grad:
                candidates.append(module.scale)

        if include_zero_point and isinstance(module.zero_point, nn.Parameter):
            if module.zero_point.requires_grad:
                candidates.append(module.zero_point)

        for param in candidates:
            if id(param) not in seen:
                seen.add(id(param))
                params.append(param)

    return params


def set_optimizer_weight_decay_for_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
    weight_decay: float = 0.0,
) -> int:
    """
    Move a subset of parameters into optimizer groups with a specific weight decay.

    Args:
        optimizer: Optimizer to update in-place.
        parameters: Parameters to move into dedicated groups.
        weight_decay: Weight decay to set for the moved parameters.

    Returns:
        Number of parameters moved.
    """
    target_ids = {id(param) for param in parameters}
    if not target_ids:
        return 0

    moved_total = 0
    new_groups = []

    for group in list(optimizer.param_groups):
        group_params = list(group["params"])
        selected = [param for param in group_params if id(param) in target_ids]
        if not selected:
            continue

        group["params"] = [param for param in group_params if id(param) not in target_ids]

        new_group = {key: value for key, value in group.items() if key != "params"}
        new_group["params"] = selected
        new_group["weight_decay"] = weight_decay
        new_groups.append(new_group)
        moved_total += len(selected)

    for new_group in new_groups:
        optimizer.add_param_group(new_group)

    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
    return moved_total


def count_quantized_layers(model: nn.Module, additional_hatorch_quantized_layers_class: list[HatorchQuantizedLayer] | None = None) -> Dict[str, int]:
    """
    Count the number of quantized layers in a model.
    
    Args:
        model: Model to analyze
        additional_hatorch_quantized_layers_class: List of additional Hatorch quantized layer classes to count
        
    Returns:
        Dictionary with counts of different quantized layer types
    """
    counts = {
        'QuantConvBn2d': 0,
        'QuantConvReLU2d': 0,
        'QuantConvBnReLU2d': 0,
        'QuantLinear': 0,
        'QuantReLU': 0,
        'QuantMaxPool2d': 0,
        'QuantAveragePool2d': 0,
        'QuantDropout': 0,
        'Quantize': 0,
        'Dequantize': 0,
        'Quantizers': 0,
    }
    
    for module in model.modules():
        if isinstance(module, QuantConvBnReLU2d):
            counts['QuantConvBnReLU2d'] += 1
        elif isinstance(module, QuantConvBn2d):
            counts['QuantConvBn2d'] += 1
        elif isinstance(module, QuantConvReLU2d):
            counts['QuantConvReLU2d'] += 1
        elif isinstance(module, QuantLinear):
            counts['QuantLinear'] += 1
        elif isinstance(module, QuantReLU):
            counts['QuantReLU'] += 1
        elif isinstance(module, QuantMaxPool2d):
            counts['QuantMaxPool2d'] += 1
        elif isinstance(module, QuantAveragePool2d):
            counts['QuantAveragePool2d'] += 1
        elif isinstance(module, QuantDropout):
            counts['QuantDropout'] += 1
        elif isinstance(module, Quantize):
            counts['Quantize'] += 1
        elif isinstance(module, Dequantize):
            counts['Dequantize'] += 1
        elif isinstance(module, BaseQuantizer):
            counts['Quantizers'] += 1
        else:
            for layer_class in HATORCH_QUANTIZED_LAYERS_CLASS + (additional_hatorch_quantized_layers_class or []):
                if isinstance(module, HatorchQuantizedLayer):
                    if not layer_class.__name__ in counts:
                        counts[layer_class.__name__] = 0
                    counts[layer_class.__name__] += 1
                    break
    
    return counts


def print_quantization_summary(model: nn.Module, additional_hatorch_quantized_layers_class: list[HatorchQuantizedLayer] | None = None) -> None:
    """
    Print a summary of quantization configuration in a model.
    
    Args:
        model: Model to summarize
    """
    print("=" * 80)
    print("Quantization Summary")
    print("=" * 80)
    
    counts = count_quantized_layers(model, additional_hatorch_quantized_layers_class=additional_hatorch_quantized_layers_class)
    print(f"\nQuantized Layers:")
    for layer_type, count in counts.items():
        print(f"  {layer_type}: {count}")
    
    print(f"\nQuantizer Details:")
    print("-" * 80)

    quantizable_types = tuple(HATORCH_QUANTIZED_LAYERS_CLASS + (additional_hatorch_quantized_layers_class or []))
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantConvBn2d, QuantConvReLU2d, QuantConvBnReLU2d, QuantLinear)):
            print(f"\n{name} ({type(module).__name__}):")
            print(f"  Weight quantizer: {type(module.weight_quantizer).__name__}")
            print(f"  Activation quantizer: {type(module.activation_quantizer).__name__}")
            print(f"  Bias quantizer: {type(module.bias_quantizer).__name__}")
        elif isinstance(module, quantizable_types):
            print(f"\n{name} ({type(module).__name__}):")
            for attr in dir(module):
                if isinstance(getattr(module, attr), BaseQuantizer):
                    quantizer = getattr(module, attr)
                    print(f"  {attr}: {type(quantizer).__name__}")
    
    print("=" * 80)


def set_scale_approximation(
    model: nn.Module,
    approximation: ScaleApproximation,
    fixed_point_bits: Optional[int] = None,
    weights_only: bool = False,
    activations_only: bool = False,
    skip_first_activation: bool = True,
    skip_output_scale: bool = True,
) -> int:
    """
    Dynamically change the scale approximation method for all LSQ quantizers in a model.
    
    This is useful for switching between training (no approximation) and inference
    (power-of-two or fixed-point approximation) modes.
    
    Args:
        model: Model containing LSQ quantizers
        approximation: Scale approximation method to set
        fixed_point_bits: Number of fractional bits (only used for FIXED_POINT)
        weights_only: Only update weight quantizers (default: False)
        activations_only: Only update activation quantizers (default: False)
        skip_first_activation: Keep the first activation quantizer exact. Use this
            when input/dataset quantization is an external deployment boundary.
        skip_output_scale: Keep the last weight quantizer exact. Use this when
            final output scaling is outside the deployed accelerator graph.
        
    Example:
        >>> # Train with exact scales
        >>> set_scale_approximation(model, ScaleApproximation.NONE)
        >>> trainer.train()
        >>> 
        >>> # Evaluate with power-of-two approximation
        >>> set_scale_approximation(model, ScaleApproximation.POWER_OF_TWO)
        >>> trainer.evaluate()
        >>> 
        >>> # Deploy with fixed-point approximation (8 fractional bits)
        >>> set_scale_approximation(model, ScaleApproximation.FIXED_POINT, fixed_point_bits=8)
    """
    if weights_only and activations_only:
        raise ValueError("Cannot specify both weights_only and activations_only")
    
    updated = 0
    first_activation = _first_lsq_activation_quantizer(model) if skip_first_activation else None
    output_scale_quantizer = _last_lsq_weight_quantizer(model) if skip_output_scale else None
    for module in model.modules():
        if isinstance(module, LsqUniformQuantizer):
            # Filter based on quantizer type
            if weights_only and module.is_activation:
                continue
            if activations_only and not module.is_activation:
                continue
            if module is first_activation or module is output_scale_quantizer:
                module.scale_approximation = ScaleApproximation.NONE
                continue
            
            # Update approximation settings
            module.scale_approximation = approximation
            if fixed_point_bits is not None:
                module.fixed_point_bits = fixed_point_bits
            updated += 1
        elif isinstance(module, QuantResUnit) and not (weights_only or activations_only):
            module.scale_approximation = approximation
            if fixed_point_bits is not None:
                module.fixed_point_bits = fixed_point_bits
            updated += 1
    return updated


def freeze_approximated_lsq_scales(
    model: nn.Module,
    weights_only: bool = False,
    activations_only: bool = False,
) -> int:
    """
    Freeze LSQ scale parameters only for quantizers with active scale approximation.

    This intentionally leaves exact-scale boundary quantizers trainable, such as
    the first input/dataset activation scale when it is kept at NONE.
    """
    if weights_only and activations_only:
        raise ValueError("Cannot specify both weights_only and activations_only")

    frozen = 0
    for module in model.modules():
        if not isinstance(module, LsqUniformQuantizer):
            continue
        if weights_only and module.is_activation:
            continue
        if activations_only and not module.is_activation:
            continue
        if module.scale_approximation == ScaleApproximation.NONE:
            continue
        if not isinstance(module.scale, nn.Parameter):
            continue

        module.scale.requires_grad_(False)
        frozen += 1
    return frozen


def _first_lsq_activation_quantizer(model: nn.Module) -> Optional[LsqUniformQuantizer]:
    for module in model.modules():
        if isinstance(module, LsqUniformQuantizer) and module.is_activation:
            return module
    return None


def _last_lsq_weight_quantizer(model: nn.Module) -> Optional[LsqUniformQuantizer]:
    last = None
    for module in model.modules():
        if isinstance(module, LsqUniformQuantizer) and not module.is_activation:
            last = module
    return last


def freeze_lsq_scales_and_set_power_of_two(
    model: nn.Module,
    weights_only: bool = False,
    activations_only: bool = False,
    skip_first_activation: bool = True,
    skip_output_scale: bool = True,
) -> None:
    """
    Freeze LSQ scale parameters and switch scale approximation to power-of-two.

    Args:
        model: Model containing LSQ quantizers.
        weights_only: Only update weight quantizers (default: False).
        activations_only: Only update activation quantizers (default: False).
        skip_first_activation: Keep the first activation quantizer exact.
        skip_output_scale: Keep the last weight quantizer exact.
    """
    if weights_only and activations_only:
        raise ValueError("Cannot specify both weights_only and activations_only")

    first_activation = _first_lsq_activation_quantizer(model) if skip_first_activation else None
    output_scale_quantizer = _last_lsq_weight_quantizer(model) if skip_output_scale else None
    for module in model.modules():
        if not isinstance(module, LsqUniformQuantizer):
            continue
        if weights_only and module.is_activation:
            continue
        if activations_only and not module.is_activation:
            continue
        if module is first_activation or module is output_scale_quantizer:
            module.scale_approximation = ScaleApproximation.NONE
            continue

        module.scale.requires_grad_(False)
        module.scale_approximation = ScaleApproximation.POWER_OF_TWO


def freeze_batch_norm_stats(model: nn.Module) -> None:
    """
    Freeze BatchNorm running statistics for modules in a model.
    """
    for module in model.modules():
        if hasattr(module, "freeze_bn_stats") and callable(module.freeze_bn_stats):
            module.freeze_bn_stats()
        elif isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            module.eval()


def freeze_folded_batch_norm_stats(model: nn.Module) -> int:
    """
    Freeze running-stat usage only for fused Conv+BN modules with BN folding enabled.

    This leaves standalone/non-folded BatchNorm modules under normal train/eval
    control, while allowing folded QAT paths to stop using batch-stat correction.
    """
    frozen = 0
    for module in model.modules():
        if (
            bool(getattr(module, "fold_batch_norm", False))
            and hasattr(module, "freeze_bn_stats")
            and callable(module.freeze_bn_stats)
        ):
            module.freeze_bn_stats()
            frozen += 1
    return frozen
