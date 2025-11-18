"""
Test script to verify model loading and forward pass.
Tests all 4 models with all 3 datasets.
"""

import torch
from models import get_model, print_model_summary
from utils.data_loader import get_dataloaders, load_config


def test_model_loading():
    """Test that all models load correctly with different num_classes"""

    print("\n" + "=" * 70)
    print("TESTING MODEL LOADING")
    print("=" * 70)

    models_to_test = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']
    num_classes_list = [100, 120, 102]  # CIFAR-100, Stanford Dogs, Flowers-102

    for model_name in models_to_test:
        print(f"\n{model_name.upper()}")
        print("-" * 70)

        for num_classes in num_classes_list:
            try:
                model = get_model(model_name, num_classes=num_classes, pretrained=False)
                print(f"  {num_classes} classes: OK")
            except Exception as e:
                print(f"  {num_classes} classes: FAILED - {e}")


def test_forward_pass():
    """Test forward pass with actual data from loaders"""

    print("\n" + "=" * 70)
    print("TESTING FORWARD PASS WITH DATA LOADERS")
    print("=" * 70)

    config = load_config()
    datasets = {
        'cifar100': 100,
        'stanford_dogs': 120,
        'flowers102': 102
    }

    models_to_test = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']

    # test each model with each dataset
    for dataset_name, num_classes in datasets.items():
        print(f"\n{dataset_name.upper()} ({num_classes} classes)")
        print("-" * 70)

        # get one batch from the dataset
        try:
            train_loader, _, _ = get_dataloaders(dataset_name, config)
            images, labels = next(iter(train_loader))
            print(f"Loaded batch: {images.shape}, labels: {labels.shape}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            continue

        # test each model with this batch
        for model_name in models_to_test:
            try:
                model = get_model(model_name, num_classes=num_classes, pretrained=False)
                model.eval()

                with torch.no_grad():
                    output = model(images)

                print(f"  {model_name}: output shape {output.shape} - OK")

                # verify output shape
                assert output.shape == (images.shape[0], num_classes), \
                    f"Expected shape ({images.shape[0]}, {num_classes}), got {output.shape}"

            except Exception as e:
                print(f"  {model_name}: FAILED - {e}")


def test_pretrained_loading():
    """Test loading with pretrained weights"""

    print("\n" + "=" * 70)
    print("TESTING PRETRAINED WEIGHTS LOADING")
    print("=" * 70)

    models_to_test = ['mobilenetv3', 'efficientnet', 'shufflenet', 'squeezenet']

    for model_name in models_to_test:
        try:
            print(f"\n{model_name.upper()}")
            model = get_model(model_name, num_classes=100, pretrained=True)
            print_model_summary(model, model_name)
        except Exception as e:
            print(f"FAILED: {e}")


def main():
    print("\n" + "=" * 70)
    print("MODEL VERIFICATION TEST SUITE")
    print("=" * 70)

    # test 1: basic loading
    test_model_loading()

    # test 2: forward pass
    test_forward_pass()

    # test 3: pretrained weights
    test_pretrained_loading()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
