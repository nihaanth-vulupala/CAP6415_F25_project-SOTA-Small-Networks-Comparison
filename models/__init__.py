"""
Model loading and initialization for SOTA Small Networks Comparison

Handles loading and configuring:
- MobileNetV3-Small
- EfficientNet-B0
- ShuffleNetV2 0.5x
- SqueezeNet 1.1
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm


def get_mobilenetv3(num_classes=1000, pretrained=True):
    """
    Load MobileNetV3-Small model.

    The classifier is at model.classifier which is a Sequential with:
    - Linear layer (in_features depends on architecture)
    - Hardswish
    - Dropout
    - Final Linear layer

    We need to replace the last Linear layer.
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
    Load EfficientNet-B0 using timm library.

    timm makes this easy - just pass num_classes and it handles the classifier replacement.
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
    Load ShuffleNetV2 0.5x model.

    The final layer is model.fc which is a Linear layer.
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
    Load SqueezeNet 1.1 model.

    SqueezeNet has a different structure - it uses a Conv2d as final classifier.
    The classifier is at model.classifier[1] which is a Conv2d layer.
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
    Count the number of trainable parameters in a model.
    Useful for comparing model sizes.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 4 / (1024 ** 2)  # assuming float32, 4 bytes per param
    }


def print_model_summary(model, model_name):
    """
    Print a summary of the model architecture and parameters.
    """
    params = count_parameters(model)

    print("\n" + "=" * 60)
    print(f"MODEL SUMMARY: {model_name.upper()}")
    print("=" * 60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total_mb']:.2f} MB")
    print("=" * 60 + "\n")
