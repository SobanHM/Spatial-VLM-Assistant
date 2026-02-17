from datasets import load_dataset, DatasetDict, concatenate_datasets

def download_and_split_nyu():
    print("Starting NYU Depth V2 Download (Unified)")

    # loading all available parts: 1449 labeled images
    ds_train = load_dataset("jagennath-hari/nyuv2", split='train')
    ds_val = load_dataset("jagennath-hari/nyuv2", split='val')
    ds_test = load_dataset("jagennath-hari/nyuv2", split='test')

    # combine images into one dataset
    full_dataset = concatenate_datasets([ds_train, ds_val, ds_test])
    print(f"Total labeled images found: {len(full_dataset)}")

    # shuffle with a fixed seed
    shuffled_ds = full_dataset.shuffle(seed=42)

    # creating splits: Train 1000 images, validate: next 200 images and test remaining 249 images (total is 1449)
    train_data = shuffled_ds.select(range(0, 1000))
    val_data = shuffled_ds.select(range(1000, 1200))
    test_data = shuffled_ds.select(range(1200, 1449))

    print(f"\nSuccessfully Split Data:")
    print(f"Train: {len(train_data)} images")
    print(f"Validate: {len(val_data)} images")
    print(f"Test: {len(test_data)} images")

    return DatasetDict({
        'train': train_data,
        'validation': val_data,
        'test': test_data
    })

if __name__ == "__main__":
    nyu_ds = download_and_split_nyu()
