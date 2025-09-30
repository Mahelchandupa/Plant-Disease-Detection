# Plant Disease Detection Using Deep Learning

Course: Computer Vision / Deep Learning<br/>
Student: R.A.M.C. Ranweera<br/>
Student ID: YR4COBSCCOMP232P-011<br/>
Submission Date: 30/09/2025<br/>

## Project Overview

> This project implements and compares two deep learning approaches for automated tomato plant disease detection using leaf images from the PlantVillage dataset. The system classifies tomato leaves into four categories: Healthy, Early Blight, Late Blight, and Bacterial Spot.

## Key Results

- Custom CNN Model: 46.2% accuracy
- VGG16 Transfer Learning: 93.1% accuracy
- Successfully exceeded the 60% accuracy requirement
- Generated comprehensive visualizations for analysis

## Project Structure

> This project is organized into 3 separate repositories for better deployment management:

1. Main Repository (Documentation & Training)

```
plant-disease-detection-main/
├── README.md                          
├── CODE_CITATION.md                   # Code attribution documentation
├── requirements.txt                   # Python dependencies for training
├── train_models.ipynb                 # Main training notebook (Google Colab)
├── train_models.py                    # Training script (Python version)
├── visualizations/                    # Generated plots for report
│   ├── dataset_samples.png
│   ├── class_distribution.png
│   ├── training_history.png
│   ├── confusion_matrices.png
│   ├── performance_comparison.png
│   └── sample_predictions.png
├── test_prediction.py                 # Simple testing script
└── report.pdf                         # Final project report
```

2. Backend Repository (API Server)

```
plant-disease-detection-backend/
├── app.py                            # Flask application
├── requirements.txt                   # Backend dependencies
├── models/                           # Trained model files
│   ├── working_cnn_final.keras       # CNN model (6.1 MB)
│   └── working_transfer_final.keras  # Transfer learning model (114.8 MB)
└── .gitignore                        # Ignore model files in git
```

> [!NOTE]
> Note: Model files does not exist in the repo, due to size constraints.

3. Frontend Repository (Web Interface)

```
plant-disease-detection-frontend/
├── index.html                         # Main web page
├── style.css                          # Stylesheet (if separated)
├── script.js                          # JavaScript logic (if separated)
├── assets/                            # Static assets
│   └── sample_images/                # Sample leaf images for testing
│       ├── healthy_sample.jpg
│       ├── early_blight_sample.jpg
│       ├── late_blight_sample.jpg
│       └── bacterial_spot_sample.jpg
└── .gitignore
```

### Repository Links
### GitHub Repositories

 - Main Repository: https://github.com/Mahelchandupa/Plant-Disease-Detection.git
 - Backend Repository: https://github.com/Mahelchandupa/plant-disease-detection-backend
 - Frontend Repository: https://github.com/Mahelchandupa/plant-disease-detection-frontend

### Google Drive (Model Files & Dataset)

 - **Drive Link:** https://drive.google.com/drive/folders/[your-folder-id]

   - Trained Models (working_cnn_final.keras, working_transfer_final.keras)
   - Training Visualizations
   - PlantVillage Dataset
   - Project Report PDF

## Dataset

**Source**: PlantVillage Dataset
**Citation**: Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.

### Dataset Details:

  - Total Images: 800 (200 per class)
  - Image Size: 128×128 pixels
  - Classes: 4 (Healthy, Early Blight, Late Blight, Bacterial Spot)
  - Train/Validation Split: 80/20

### Technologies Used

  - Python 3.10
  - TensorFlow 2.19.0 - Deep learning framework
  - Keras 3.10.0 - High-level neural networks API
  - OpenCV 4.8.1 - Image processing
  - Flask 2.3.3 - Web framework for API
  - NumPy 2.0.2 - Numerical computations
  - Matplotlib & Seaborn - Data visualization

### Installation

### Prerequisites

  - Python 3.10 or higher
  - Google Colab (for training)
  - 8GB RAM minimum
  - GPU recommended for training

### Prerequisites

  - Python 3.10 or higher
  - Google Colab (for training)
  - 8GB RAM minimum
  - GPU recommended for training

## Setup Instructions

1. Main Repository (Training & Documentation)

```
# Clone main repository
git clone https://github.com/Mahelchandupa/plant-disease-detection-main.git
cd plant-disease-detection-main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Backend Repository (API Server)

```
**Clone backend repository**
git clone https://github.com/Mahelchandupa/plant-disease-detection-backend.git
cd plant-disease-detection-backend

**Create virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install dependencies**
pip install -r requirements.txt

**Download model files from Google Drive**
  > **Place them in the models/ directory:**
  >  - working_cnn_final.keras
  >  - working_transfer_final.keras
```

# Model Architectures

### Custom CNN

 - 4 Convolutional blocks (32, 64, 128, 256 filters)
 - Batch Normalization after each Conv layer
 - MaxPooling for dimensionality reduction
 - Global Average Pooling
 - 2 Dense layers (512, 4 neurons)
 - Dropout for regularization (0.25-0.5)
 - Total Parameters: 526,020

### VGG16 Transfer Learning

 - Pre-trained VGG16 base (ImageNet weights)
 - Frozen first 12 layers
 - Fine-tuned last 4 layers
 - Custom classification head
 - Global Average Pooling
 - 3 Dense layers (512, 256, 4 neurons)
 - Total Parameters: 15,113,796

### Training Configuration

 - Optimizer: Adam

   - CNN Learning Rate: 0.001
   - Transfer Learning Rate: 0.0001

 - Loss Function: Categorical Cross-entropy
 - Batch Size: 32
 - Epochs: 25 (with early stopping)
 - Data Augmentation:
    - Rotation: ±15°
    - Width/Height Shift: ±10%
    - Zoom: ±10%
    - Horizontal Flip: Yes