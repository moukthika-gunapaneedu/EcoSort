# EcoSort

CNN-based waste classification system supporting sustainable recycling.

EcoSort uses a fine-tuned ResNet-18 model to classify images of waste into six material categories:

- cardboard  
- glass  
- metal  
- paper  
- plastic  
- trash  

---

## Features

- Lightweight ResNet-18 backbone  
- Six-class material classification  
- Clear training and evaluation workflow  
- Confusion matrix and classification report  
- Grad-CAM heatmaps to visualize what the model is “looking at”  

---

## Setup

1. Clone the repository  
   - `git clone https://github.com/moukthika-gunapaneedu/EcoSort.git`  
   - `cd EcoSort`  

2. Create and activate a virtual environment  

   - `python -m venv .venv`  
   - On Windows: `.\.venv\Scripts\activate`  
   - On macOS / Linux: `source .venv/bin/activate`  

3. Install dependencies  

   - `pip install -r requirements.txt`  

The code expects the dataset to be arranged into class-specific folders under `data/raw/` (cardboard, glass, metal, paper, plastic, trash). Processed train/val/test splits are stored under `data/processed/`.

---

## Training

Run the training script to fine-tune ResNet-18 on the processed dataset:

- `python src/train.py`

This script:

- Loads the train and validation splits  
- Applies data augmentation and normalization  
- Trains the model for a configurable number of epochs  
- Saves the best checkpoint (based on validation accuracy) to `models/best_model.pt`  

---

## Evaluation

Evaluate the saved checkpoint on the held-out test set:

- `python src/evaluate.py`

This script:

- Loads `models/best_model.pt`  
- Runs inference on the test split  
- Prints a per-class precision/recall/F1 classification report  
- Saves a confusion matrix image to `results/confusion_matrix/confusion_matrix.png`  
- Writes the full classification report to `results/classification_report.txt`  

---

## Grad-CAM Visualizations

To understand what the model focuses on, you can generate Grad-CAM heatmaps:

- `python src/gradcam.py --image <path_to_image>`

The script overlays heatmaps on top of the input image and saves them under `results/gradcam/`, helping you see which regions influenced the prediction.

---

## Project Website

A narrative write-up of the project, including figures and Grad-CAM examples, is available here:

- **EcoSort project page:** https://moukthika-gunapaneedu.github.io/EcoSort/

---

## Tech Stack

- Python  
- PyTorch  
- ResNet-18  
- NumPy, Pandas  
- Matplotlib  

---

## Contact

**Moukthika Gunapaneedu**  

- Email: `moukthikagunapaneedu@gmail.com`  
- LinkedIn: https://www.linkedin.com/in/moukthika-gunapaneedu/  
- Portfolio: https://moukthika.netlify.app  
