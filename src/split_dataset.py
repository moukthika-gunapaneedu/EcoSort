import os
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE_DIR = Path("data/raw")
TARGET_DIR = Path("data/processed")

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

def make_dirs():
    for split in SPLIT_RATIOS:
        for class_name in os.listdir(SOURCE_DIR):
            class_path = SOURCE_DIR / class_name
            if not class_path.is_dir():
                continue
            split_dir = TARGET_DIR / split / class_name
            os.makedirs(split_dir, exist_ok=True)

def split():
    make_dirs()

    for class_name in os.listdir(SOURCE_DIR):
        class_dir = SOURCE_DIR / class_name
        if not class_dir.is_dir():
            continue

        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        total = len(images)
        if total == 0:
            continue

        train_end = int(total * SPLIT_RATIOS["train"])
        val_end = train_end + int(total * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, files in splits.items():
            for file in files:
                src = class_dir / file
                dst = TARGET_DIR / split_name / class_name / file
                shutil.copy(src, dst)

if __name__ == "__main__":
    split()
    print("Dataset split complete!")
