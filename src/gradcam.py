import os
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_prep import get_dataloaders
from model import build_model


RESULTS_DIR = Path("results")
GRADCAM_DIR = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = RESULTS_DIR / "model_weights" / "ecosort_cnn_best.pth"

# Same normalization as in data_prep.py
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class GradCAM:
    """
    Simple Grad-CAM implementation for EcoSortCNN.
    We attach hooks to the last convolutional layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # forward hook: save activations
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        # backward hook: save gradients
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want grad wrt activations
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class, device):
        """
        input_tensor: (1, 3, H, W)
        target_class: int (class index)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)

        score = output[0, target_class]
        score.backward()

        # activations: (1, C, H', W')
        # gradients:   (1, C, H', W')
        gradients = self.gradients
        activations = self.activations

        # Global average pooling over H',W'
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Normalize CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()  # (H, W)

        return cam

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def denormalize(tensor_batch):
    """
    Undo the normalization for visualization.
    tensor_batch: (B, 3, H, W)
    """
    return tensor_batch * STD.to(tensor_batch.device) + MEAN.to(tensor_batch.device)


def plot_gradcam(image, cam, class_name, pred_name, out_path):
    """
    image: (3, H, W) tensor in [0,1] after denorm
    cam:   (H, W) tensor in [0,1]
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()  # H,W,3
    cam_np = cam.cpu().numpy()                     # H,W

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(img_np)
    ax[0].axis("off")
    ax[0].set_title(f"True: {class_name}\nPred: {pred_name}")

    ax[1].imshow(img_np, alpha=0.5)
    ax[1].imshow(cam_np, cmap="jet", alpha=0.5)
    ax[1].axis("off")
    ax[1].set_title("Grad-CAM")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(num_examples=6):
    # Load data
    _, _, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    # Load model + weights
    model, device = build_model(num_classes=num_classes, model_type="resnet18")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # For ResNet18, last conv layer:
    target_layer = model.backbone.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)


    # Take one batch from test loader
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Denormalize for visualization
    images_denorm = denormalize(images).clamp(0, 1)

    os.makedirs(GRADCAM_DIR, exist_ok=True)

    count = min(num_examples, images.size(0))

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    for i in range(count):
        img = images[i:i+1]          # (1,3,H,W) normalized
        img_vis = images_denorm[i]   # (3,H,W)
        true_idx = labels[i].item()
        pred_idx = preds[i].item()

        # Use predicted class for CAM
        cam = gradcam.generate(img, target_class=pred_idx, device=device)

        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]

        out_path = GRADCAM_DIR / f"example_{i}_true-{true_name}_pred-{pred_name}.png"
        plot_gradcam(img_vis, cam, true_name, pred_name, out_path)
        print(f"Saved Grad-CAM: {out_path}")

    gradcam.close()
    print("Grad-CAM generation complete.")


if __name__ == "__main__":
    main(num_examples=6)
