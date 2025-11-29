import random
import shutil
from pathlib import Path

SOURCE = Path("/workspaces/EcoSort/data/processed/train")
DEST = Path("/workspaces/EcoSort/data/processed_sample")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

DEST.mkdir(parents=True, exist_ok=True)

for cls in classes:
    src_folder = SOURCE / cls
    dst_folder = DEST / cls
    dst_folder.mkdir(parents=True, exist_ok=True)

    images = [f for f in src_folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    sample = random.sample(images, min(3, len(images)))

    for img in sample:
        shutil.copy(img, dst_folder / img.name)

print(f"Processed sample dataset saved to: {DEST}")
