from pathlib import Path

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from data_prep import get_dataloaders
from model import build_model


RESULTS_DIR = Path("results")
CONF_MAT_DIR = RESULTS_DIR / "confusion_matrix"
CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = RESULTS_DIR / "model_weights" / "ecosort_cnn_best.pth"


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path) -> None:
    """Create and save a green-themed confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",        
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("EcoSort CNN – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_on_test() -> None:
    """Evaluate the saved model on the test set and generate metrics."""
    _, _, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model, device = build_model(num_classes=num_classes)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

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

    # Confusion matrix (saved in green theme)
    conf_mat_path = CONF_MAT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        save_path=conf_mat_path,
    )

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print(report)

    # Also save report to disk
    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report)


if __name__ == "__main__":
    evaluate_on_test()
