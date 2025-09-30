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

## Repository Structure

> This project is organized into 3 separate repositories for better deployment management:

1. Main Repository (Documentation & Training)

```
plant-disease-detection-main/
├── README.md                   # Project documentation
├── CODE_CITATION.md            # Code attribution
├── train_models.py             # Training script
├── test_prediction.py          # Testing script
├── requirements.txt            # Dependencies
├── visualizations/             # Generated charts
└── report.pdf                  # Final report
```

2. Backend Repository

```
plant-disease-detection-backend/
├── app.py                      # Flask API
├── requirements.txt
└── models/                     # Trained models
```
> [!NOTE]
> Note: Model files does not exist in the repo, due to size constraints.

3. Frontend Repository (Web Interface)

```
plant-disease-detection-frontend/
├── index.html                  # Web interface
├── style.css                   # Styling
├── script.js                   # Frontend logic
└── assets/sample_images/       # Test images
```

## Installation

### Main Repository

```
git clone https://github.com/Mahelchandupa/Plant-Disease-Detection.git
cd plant-disease-detection-main
pip install -r requirements.txt
```

### Backend

```
git clone https://github.com/Mahelchandupa/plant-disease-detection-backend
cd plant-disease-detection-backend
pip install -r requirements.txt
python app.py  # Runs on http://localhost:5000
```

### Frontend

```
git clone https://github.com/Mahelchandupa/plant-disease-detection-frontend
cd plant-disease-detection-frontend
# Open index.html in browser or use: python -m http.server 8000
```

## Quick Start

1. Train models: Run train_models.py or use Google Colab notebook
2. Start backend: python app.py in backend repo
3. Open frontend: Open index.html in browser
4. Upload a tomato leaf image to get predictions

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


# Code Explanation

## Part 1: Data Loading and Preprocessing

### Location: train_models.py lines 50-120

```
def load_working_dataset():
    # Read images from Google Drive
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(selected_classes):
        # Get all image files
        image_files = os.listdir(class_path)
        
        for img_file in image_files:
            img = cv2.imread(img_path)              # Read image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert colors
            img = cv2.resize(img, (128, 128))       # Resize to 128x128
            img = img.astype(np.float32) / 255.0    # Normalize to 0-1
            
            images.append(img)
            labels.append(class_idx)
    
    return np.array(images), to_categorical(labels)
```

**What this does:**

 - Loads images from 4 disease classes
 - Converts BGR (OpenCV format) to RGB
 - Resizes all images to 128x128 pixels (neural networks need consistent input)
 - Normalizes pixel values from 0-255 to 0-1 (helps training)
 - Converts labels to one-hot encoding (e.g., [0,1,0,0] for class 1)

**Why these choices:**

 - 128x128 size balances detail and computational speed
 - Normalization prevents large numbers from dominating training
 - One-hot encoding is standard for multi-class classification

## Part 2: Custom CNN Architecture

### Location: train_models.py lines 180-210

```
def create_working_cnn():
    model = Sequential([
        # Block 1: Basic feature detection
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 2: Detect patterns
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 3: Complex features
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 4: High-level features
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        
        # Classification layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model
```

**Layer-by-layer explanation:**

 - Conv2D(32, (3,3)): Creates 32 filters that scan the image in 3x3 windows to detect edges and basic shapes
 - BatchNormalization(): Normalizes the outputs to speed up training
 - MaxPooling2D(2,2): Reduces image size by half, keeping important features
 - Dropout(0.25): Randomly drops 25% of connections to prevent overfitting

**Progressive feature learning:**

 - 32 filters: Detect edges (vertical, horizontal, diagonal)
 - 64 filters: Detect shapes (circles, rectangles)
 - 128 filters: Detect leaf patterns (veins, spots)
 - 256 filters: Detect disease-specific features

**Why this architecture:**

 - Each block doubles the filters while reducing image size
 - BatchNorm + Dropout prevents overfitting
 - GlobalAveragePooling is better than Flatten for generalization
 - 512 dense neurons allow complex decision making

## Part 3: Transfer Learning with VGG16

### Location: train_models.py lines 215-240

```
def create_working_transfer():
    # Load pre-trained VGG16
    base_model = VGG16(
        weights='imagenet',         # Use pre-trained weights
        include_top=False,          # Remove classification layers
        input_shape=(128, 128, 3)
    )
    
    # Freeze early layers, allow fine-tuning of last 4 layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Add custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')
    ])
    return model
```

**What transfer learning does:**

 - VGG16 was trained on ImageNet (14 million images, 1000 classes)
 - It already learned to detect edges, shapes, textures
 - We "transfer" this knowledge to plant diseases

**Freezing vs fine-tuning:**

 - Freeze first layers: Keep general features (edges, colors)
 - Fine-tune last 4 layers: Adapt to plant-specific patterns
 - Custom head: Learn disease-specific classification

**Why this works better:**

 - Pre-trained features are powerful
 - Don't need millions of images to train from scratch
 - Fine-tuning adapts general features to our specific task

## Part 4: Data Augmentation

### Location: train_models.py lines 260-270

```
train_datagen = ImageDataGenerator(
    rotation_range=15,          # Rotate ±15 degrees
    width_shift_range=0.1,      # Shift left/right 10%
    height_shift_range=0.1,     # Shift up/down 10%
    zoom_range=0.1,             # Zoom in/out 10%
    horizontal_flip=True,       # Flip left-right
    fill_mode='nearest'         # Fill empty pixels
)
```

**Why augmentation is important:**

 - Creates artificial training data
 - Prevents overfitting to specific orientations
 - Makes model robust to real-world variations

**Example transformations:**

 - Original leaf image
 - Same leaf rotated 10 degrees
 - Same leaf flipped horizontally
 - Same leaf zoomed in 5%

## Part 5: Training Process

### Location: train_models.py lines 300-330

```
# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=25,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(patience=7),
        ReduceLROnPlateau(factor=0.2, patience=4)
    ]
)
```

**Training parameters explained:**

 - Adam optimizer: Adaptive learning rate algorithm
 - Learning rate 0.001: How big the update steps are
 - Categorical crossentropy: Loss function for multi-class classification
 - Batch size 32: Process 32 images at once
 - Epochs 25: Maximum 25 complete passes through data

**Callbacks (training helpers):**

 - EarlyStopping: Stop if no improvement for 7 epochs
 - ReduceLROnPlateau: Lower learning rate if stuck

> [!IMPORTANT]
> This prevents overfitting and speeds up training.

## ------------- 

## Part 6: Flask API Backend

### Location: backend/app.py

```
@app.route('/predict/both', methods=['POST'])
def predict_both():
    # Get uploaded image
    file = request.files['image']
    
    # Read and preprocess
    image_data = file.read()
    img_array = preprocess_image(image_data)  # Resize to 128x128, normalize
    
    # Predict with both models
    cnn_pred = cnn_model.predict(img_array)
    transfer_pred = transfer_model.predict(img_array)
    
    # Get predicted class
    cnn_class = CLASS_NAMES[np.argmax(cnn_pred)]
    transfer_class = CLASS_NAMES[np.argmax(transfer_pred)]
    
    # Return results
    return jsonify({
        'cnn': {'class': cnn_class, 'confidence': float(np.max(cnn_pred))},
        'transfer': {'class': transfer_class, 'confidence': float(np.max(transfer_pred))}
    })
```

**API workflow:**

- Receive image file from frontend
- Preprocess to 128x128 and normalize
- Run through both models
- Get predictions and confidence scores 
- Return JSON response to frontend

**Why Flask:**

- Lightweight web framework
- Easy to create API endpoints
- Good for small-scale deployment

## Part 7: Frontend Interface

### Location: frontend/script.js

```
async function makePrediction() {
    // Create form data with image
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    // Send to backend API
    const response = await fetch('http://localhost:5000/predict/both', {
        method: 'POST',
        body: formData
    });
    
    // Get results
    const result = await response.json();
    
    // Display predictions
    displayResults(result);
}

function displayResults(result) {
    // Show CNN prediction
    document.getElementById('cnn-result').textContent = 
        `${result.cnn.class} (${result.cnn.confidence * 100}%)`;
    
    // Show Transfer Learning prediction
    document.getElementById('transfer-result').textContent = 
        `${result.transfer.class} (${result.transfer.confidence * 100}%)`;
}
```
## --------------------

**Frontend flow:**

- User uploads image
- JavaScript sends to backend
- Waits for prediction response
- Displays results with confidence scores

# Key Design Decisions

### Why 128x128 Image Size?

 - Tested 64x64 (too small, lost detail)
 - Tested 224x224 (slower, no accuracy gain)
 - 128x128 is the sweet spot

### Why 4 Classes?

 - More classes = harder to learn
 - These 4 have distinct visual patterns
 - Achieves good accuracy with limited data

### Why VGG16 vs ResNet?

 - Tested both architectures
 - VGG16 performed better on this dataset
 - Simpler architecture, easier to fine-tune

### Why Lower Learning Rate for Transfer?

 - Pre-trained weights are already good
 - Small updates preserve learned features
 - Prevents "forgetting" ImageNet knowledge

# Results Analysis

> [!IMPORTANT]
> ## The 97.5% accuracy from transfer learning shows:

 - Pre-trained features are very useful
 - ImageNet features transfer well to plants
 - Limited training data can still achieve high accuracy

> [!IMPORTANT]
> ## The 45.6% CNN accuracy shows:

 - Training from scratch needs more data
 - 800 images not enough for complex CNN
 - Transfer learning is better for small datasets

# Repository Links

### GitHub Repositories:

 - Main: https://github.com/Mahelchandupa/Plant-Disease-Detection.git
 - Backend: https://github.com/Mahelchandupa/plant-disease-detection-backend
 - Frontend: https://github.com/Mahelchandupa/plant-disease-detection-frontend

### Google Drive:

 - Models & Dataset: https://drive.google.com/drive/folders/1Q75E3OXyooQRljNUWLkTgMz8JyXY-OdQ?usp=sharing

**Last Updated:** 30/09/2025