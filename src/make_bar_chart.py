import os
from pathlib import Path
import matplotlib.pyplot as plt


DATA_ROOT = Path("/workspaces/EcoSort/data/raw")
 
# Example folder structure:
# data/trashnet/cardboard/
# data/trashnet/glass/
# data/trashnet/metal/
# data/trashnet/paper/
# data/trashnet/plastic/
# data/trashnet/trash/

# ---------------------------------------------
# CLASSES in TrashNet
# ---------------------------------------------
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ---------------------------------------------
# COUNT IMAGES PER CLASS
# ---------------------------------------------
counts = []
for cls in classes:
    folder = DATA_ROOT / cls
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    n = sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    counts.append(n)

print("Image counts:", dict(zip(classes, counts)))

# ---------------------------------------------
# BAR CHART (EcoSort green theme)
# ---------------------------------------------
plt.figure(figsize=(8, 4.5))
ecogreen = "#166534"

bars = plt.bar(classes, counts, color=ecogreen)

plt.title("Number of Images in Each TrashNet Category", fontsize=13)
plt.ylabel("Image Count", fontsize=12)
plt.xlabel("Category", fontsize=12)

# Remove unnecessary chart borders
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Horizontal grid for readability
plt.grid(axis="y", linestyle="--", alpha=0.3)

# Add values on top of each bar
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.015,
        str(count),
        ha="center",
        fontsize=10,
        color="#374151"
    )

plt.tight_layout()

# ---------------------------------------------
# SAVE THE IMAGE TO YOUR WEBSITE FOLDER
# ---------------------------------------------
output_path = Path("/workspaces/EcoSort/website/trash_class_counts.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Chart saved to: {output_path}")
