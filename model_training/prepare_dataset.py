from pathlib import Path
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

# Original dataset location
SOURCE_DIR = Path(
    r"C:\Users\priya\Downloads\archive (8)\gaussian_filtered_images\gaussian_filtered_images"
)

# RetinaGuardAI dataset folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

# Reproducible random split
random.seed(42)

# 70% training, 15% validation, 15% testing
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Source folder -> destination folder
CLASS_MAPPING = {
    "No_DR": "0_No_DR",
    "Mild": "1_Mild",
    "Moderate": "2_Moderate",
    "Severe": "3_Severe",
    "Proliferate_DR": "4_Proliferative_DR",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# CHECK SOURCE DATASET
# ============================================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{SOURCE_DIR}"
    )

print("=" * 60)
print("RetinaGuard AI - Dataset Preparation")
print("=" * 60)

print(f"\nSource dataset:")
print(SOURCE_DIR)

print(f"\nProject dataset:")
print(DATASET_DIR)


# ============================================================
# CREATE DESTINATION FOLDERS
# ============================================================

for split in ["train", "val", "test"]:
    for destination_class in CLASS_MAPPING.values():
        folder = DATASET_DIR / split / destination_class
        folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_images = 0

for source_class, destination_class in CLASS_MAPPING.items():

    source_folder = SOURCE_DIR / source_class

    if not source_folder.exists():
        print(f"\nWARNING: Missing folder: {source_folder}")
        continue

    images = [
        file
        for file in source_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    print(f"\nClass: {source_class}")
    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

    for split_name, split_images in splits.items():

        destination_folder = (
            DATASET_DIR
            / split_name
            / destination_class
        )

        for image_file in split_images:

            destination_file = (
                destination_folder / image_file.name
            )

            shutil.copy2(
                image_file,
                destination_file
            )

    total_images += total


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"\nTotal images processed: {total_images}")

print("\nCreated structure:")

print("""
dataset/
├── train/
│   ├── 0_No_DR/
│   ├── 1_Mild/
│   ├── 2_Moderate/
│   ├── 3_Severe/
│   └── 4_Proliferative_DR/
│
├── val/
│   ├── 0_No_DR/
│   ├── 1_Mild/
│   ├── 2_Moderate/
│   ├── 3_Severe/
│   └── 4_Proliferative_DR/
│
└── test/
    ├── 0_No_DR/
    ├── 1_Mild/
    ├── 2_Moderate/
    ├── 3_Severe/
    └── 4_Proliferative_DR/
""")

print("You can now run the training script.")
