import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_prep import get_dataloaders
from model import build_model


RESULTS_DIR = Path("results")
WEIGHTS_DIR = RESULTS_DIR / "model_weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    train_loader, val_loader, _, class_names = get_dataloaders()
    num_classes = len(class_names)

    model, device = build_model(num_classes=num_classes, model_type="resnet18")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_path = WEIGHTS_DIR / "ecosort_cnn_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"- {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, best_model_path)
            print(f"New best model saved with val acc {best_val_acc:.4f}")

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
