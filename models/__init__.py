"""
Model Loading and Initialization for SOTA Small Networks Comparison

This module provides a unified interface for loading and configuring four efficient
CNN architectures, all designed for resource-constrained environments.

Supported Models:
    1. MobileNetV3-Small: ~2.5M params
       - Efficient mobile architecture with inverted residuals
       - Hardware-aware NAS (Neural Architecture Search)
       - Uses Hardswish activation and squeeze-excitation blocks

    2. EfficientNet-B0: ~5.3M params
       - Compound scaling of depth, width, and resolution
       - Mobile inverted bottleneck convolutions (MBConv)
       - Excellent accuracy/efficiency trade-off

    3. ShuffleNetV2 0.5x: ~1.4M params
       - Channel shuffle operations for efficient cross-group communication
       - Designed for low-power devices (10-150 MFLOPs)
       - Focus on memory access cost (MAC) optimization

    4. SqueezeNet 1.1: ~1.2M params
       - Fire modules with squeeze and expand layers
       - Uses 1x1 convolutions to reduce parameters
       - AlexNet-level accuracy with 50x fewer parameters

Transfer Learning Strategy:
    All models load ImageNet pretrained weights and modify only the final
    classification layer to match the target dataset's number of classes.
    This allows the models to leverage features learned on ImageNet while
    adapting to new classification tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm


def get_mobilenetv3(num_classes=1000, pretrained=True):
    """
    Load and configure MobileNetV3-Small model for transfer learning

    MobileNetV3-Small is optimized for mobile devices using:
    - Inverted residual structure with linear bottlenecks
    - Squeeze-and-Excitation blocks for channel attention
    - Hardswish activation function (efficient approximation of Swish)
    - Neural Architecture Search (NAS) for architecture optimization

    Classifier Structure:
        model.classifier is a Sequential containing:
        [0] Linear(in_features, hidden_dim)  # First linear layer
        [1] Hardswish()                       # Efficient activation
        [2] Dropout(p=0.2)                    # Regularization
        [3] Linear(hidden_dim, num_classes)   # Output layer (modified)

    Transfer Learning Modification:
        We modify classifier[3] to match target dataset classes while
        keeping pretrained feature extractor layers frozen initially.

    Args:
        num_classes (int): Number of output classes for target dataset
        pretrained (bool): Load ImageNet pretrained weights

    Returns:
        nn.Module: MobileNetV3-Small model ready for training
    """
    print(f"Loading MobileNetV3-Small (pretrained={pretrained})...")

    # load the model
    if pretrained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # replace the final classifier layer
    # the classifier is a Sequential, we need to change the last layer
    in_features = model.classifier[3].in_features  # last layer in the classifier
    model.classifier[3] = nn.Linear(in_features, num_classes)

    print(f"  Modified final layer: {in_features} -> {num_classes} classes")

    return model


def get_efficientnet(num_classes=1000, pretrained=True):
    """
    Load and configure EfficientNet-B0 model using timm library

    EfficientNet-B0 uses compound scaling to balance:
    - Network depth (number of layers)
    - Network width (number of channels)
    - Input resolution (image size)

    Key Features:
        - Mobile Inverted Bottleneck Convolution (MBConv) blocks
        - Squeeze-and-Excitation optimization
        - Swish activation function
        - Compound coefficient for systematic scaling

    Why timm Library?
        The timm (PyTorch Image Models) library provides robust EfficientNet
        implementations with automatic classifier adaptation. It handles:
        - Correct classifier replacement based on num_classes
        - Proper weight initialization for new classifier layer
        - Consistent interface across model variants

    Args:
        num_classes (int): Number of output classes for target dataset
        pretrained (bool): Load ImageNet pretrained weights

    Returns:
        nn.Module: EfficientNet-B0 model with adapted classifier
    """
    print(f"Loading EfficientNet-B0 (pretrained={pretrained})...")

    # timm automatically replaces the classifier if num_classes is different from 1000
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes
    )

    print(f"  Output classes: {num_classes}")

    return model


def get_shufflenet(num_classes=1000, pretrained=True):
    """
    Load and configure ShuffleNetV2 0.5x (width multiplier 0.5)

    ShuffleNetV2 is specifically designed for mobile devices with limited
    computational budgets (10-150 MFLOPs range).

    Key Features:
        - Channel shuffle operation for efficient information flow
        - Optimized for Memory Access Cost (MAC), not just FLOPs
        - Four design guidelines based on practical speed measurements:
          1. Equal channel width minimizes MAC
          2. Excessive group convolution increases MAC
          3. Network fragmentation reduces parallelism
          4. Element-wise operations are non-negligible

    Architecture:
        - Uses channel split and concatenation for efficiency
        - Depthwise separable convolutions
        - Simple final FC layer: model.fc (Linear layer)

    Width Multiplier 0.5x:
        The 0.5x variant uses half the channels of the base model, making it
        the most lightweight version suitable for extreme resource constraints.

    Args:
        num_classes (int): Number of output classes for target dataset
        pretrained (bool): Load ImageNet pretrained weights

    Returns:
        nn.Module: ShuffleNetV2 0.5x model with modified classifier
    """
    print(f"Loading ShuffleNetV2 0.5x (pretrained={pretrained})...")

    # load shufflenet v2 with 0.5x width multiplier
    if pretrained:
        model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    else:
        model = models.shufflenet_v2_x0_5(weights=None)

    # replace the fc layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    print(f"  Modified final layer: {in_features} -> {num_classes} classes")

    return model


def get_squeezenet(num_classes=1000, pretrained=True):
    """
    Load and configure SqueezeNet 1.1 model

    SqueezeNet achieves AlexNet-level accuracy with 50x fewer parameters
    using Fire modules and aggressive parameter reduction strategies.

    Key Features:
        - Fire modules: Squeeze layer (1x1 conv) + Expand layer (1x1 and 3x3 conv)
        - Strategy 1: Replace 3x3 filters with 1x1 filters
        - Strategy 2: Decrease input channels to 3x3 filters
        - Strategy 3: Downsample late in network (larger activation maps)

    SqueezeNet 1.1 vs 1.0:
        Version 1.1 is optimized with:
        - 2.4x less computation
        - Slightly better accuracy
        - More aggressive use of 1x1 convolutions

    Unique Classifier Structure:
        Unlike other models using Linear layers, SqueezeNet uses:
        model.classifier[1] = Conv2d(in_channels, num_classes, kernel_size=1)
        This maintains the convolutional nature throughout the network and
        performs global average pooling implicitly.

    Args:
        num_classes (int): Number of output classes for target dataset
        pretrained (bool): Load ImageNet pretrained weights

    Returns:
        nn.Module: SqueezeNet 1.1 model with modified classifier
    """
    print(f"Loading SqueezeNet 1.1 (pretrained={pretrained})...")

    # load squeezenet 1.1 (more efficient than 1.0)
    if pretrained:
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    else:
        model = models.squeezenet1_1(weights=None)

    # squeezenet uses a conv layer as final classifier
    # model.classifier[1] is a Conv2d(512, 1000, kernel_size=1)
    # we need to replace it with Conv2d(512, num_classes, kernel_size=1)
    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    print(f"  Modified final conv layer: {in_channels} channels -> {num_classes} classes")

    return model


def get_model(model_name, num_classes=1000, pretrained=True):
    """
    Unified interface to load any of the four models.

    Args:
        model_name: one of ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']
        num_classes: number of output classes for the dataset
        pretrained: whether to load ImageNet pretrained weights

    Returns:
        PyTorch model ready for training/inference
    """

    model_loaders = {
        'mobilenetv3': get_mobilenetv3,
        'efficientnet': get_efficientnet,
        'shufflenet': get_shufflenet,
        'squeezenet': get_squeezenet
    }

    if model_name not in model_loaders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_loaders.keys())}")

    model = model_loaders[model_name](num_classes=num_classes, pretrained=pretrained)

    return model


def count_parameters(model):
    """
    Count and analyze model parameters

    Calculates the total number of parameters and estimates memory footprint.
    This is crucial for comparing model efficiency and deployment feasibility.

    Parameter Count Calculation:
        - Total: All parameters in the model (frozen + trainable)
        - Trainable: Only parameters with requires_grad=True
        - For transfer learning, some layers may be frozen initially

    Memory Estimation:
        Assumes float32 (4 bytes per parameter). Actual memory usage may vary:
        - float16 (half precision): 2 bytes per param
        - int8 (quantized): 1 byte per param
        - Additional memory needed for gradients during training

    Args:
        model (nn.Module): PyTorch model to analyze

    Returns:
        dict: Parameter statistics containing:
            - 'total': Total parameter count
            - 'trainable': Trainable parameter count
            - 'total_mb': Estimated model size in MB (float32)
    """
    # Count total parameters (includes both frozen and trainable)
    total = sum(p.numel() for p in model.parameters())

    # Count only trainable parameters (requires_grad=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 4 / (1024 ** 2)  # float32: 4 bytes per parameter
    }


def print_model_summary(model, model_name):
    """
    Display formatted model statistics summary

    Prints a clean summary table showing model parameters and size.
    Useful for quick comparison during training setup.

    Args:
        model (nn.Module): Model to summarize
        model_name (str): Name of model for display

    Output Format:
        ============================================================
        MODEL SUMMARY: MOBILENETV3
        ============================================================
        Total parameters: 1,620,356
        Trainable parameters: 1,620,356
        Model size: 6.18 MB
        ============================================================
    """
    # Get parameter statistics
    params = count_parameters(model)

    # Print formatted summary
    print("\n" + "=" * 60)
    print(f"MODEL SUMMARY: {model_name.upper()}")
    print("=" * 60)
    print(f"Total parameters: {params['total']:,}")  # Comma-separated thousands
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total_mb']:.2f} MB")
    print("=" * 60 + "\n")
