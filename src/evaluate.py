from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from data_prep import get_dataloaders
from model import build_model


RESULTS_DIR = Path("results")
CONF_MAT_DIR = RESULTS_DIR / "confusion_matrix"
CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = RESULTS_DIR / "model_weights" / "ecosort_cnn_best.pth"


def evaluate_on_test():
    _, _, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model, device = build_model(num_classes=num_classes)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("EcoSort – Confusion Matrix (Test Set)")
    plt.tight_layout()

    out_path = CONF_MAT_DIR / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    evaluate_on_test()