"""
Quick test script to verify data loaders are working
Run this anytime to make sure data loading is functional
"""

from utils.data_loader import get_dataloaders, load_config

def main():
    print("\n" + "="*60)
    print("DATA LOADER VERIFICATION")
    print("="*60)

    config = load_config()
    datasets = ['cifar100', 'stanford_dogs', 'flowers102']

    results = {}

    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()}")
        print("-" * 60)

        try:
            train_loader, val_loader, test_loader = get_dataloaders(dataset_name, config)

            # load one batch from each
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))

            print(f"Train loader: {len(train_loader)} batches")
            print(f"  - Batch shape: {train_batch[0].shape}")
            print(f"Val loader: {len(val_loader)} batches")
            print(f"  - Batch shape: {val_batch[0].shape}")
            print(f"Test loader: {len(test_loader)} batches")
            print(f"  - Batch shape: {test_batch[0].shape}")

            results[dataset_name] = "PASS"

        except Exception as e:
            print(f"FAILED: {e}")
            results[dataset_name] = "FAIL"

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, status in results.items():
        symbol = "PASS" if status == "PASS" else "FAIL"
        print(f"{symbol} {dataset}: {status}")

    all_pass = all(v == "PASS" for v in results.values())
    if all_pass:
        print("\nAll data loaders working correctly!")
    else:
        print("\nSome data loaders failed. Check errors above.")

    return all_pass

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
