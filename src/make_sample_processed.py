import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Path to processed dataset (train split)
DATA_ROOT = Path("/workspaces/EcoSort/data/processed/train")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Output image path
OUT_PATH = Path("/workspaces/EcoSort/website/sample_processed.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create a grid: 1 sample per class
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for ax, cls in zip(axes, classes):
    folder = DATA_ROOT / cls
    files = [f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not files:
        ax.set_title(f"{cls} (no images)")
        ax.axis("off")
        continue

    img_path = random.choice(files)
    img = Image.open(img_path)

    ax.imshow(img)
    ax.set_title(cls, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.close()

print(f"Sample processed grid saved to: {OUT_PATH}")