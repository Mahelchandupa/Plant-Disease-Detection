# Code Attribution and Citations

## Project: Plant Disease Detection Using Deep Learning

Student: R.A.M.C. Ranweera<br/>
Date: 30/09/2025<br/>

> This document provides detailed attribution for all code used in this project, distinguishing between original student work and adapted third-party code.

1. Model Training Pipeline
 - File: train_models.ipynb
 - Components:
   - Dataset loading and preprocessing functions
   - Custom CNN architecture design
   - Transfer learning model configuration
   - Training loop with data augmentation
   - Evaluation metrics calculation
   - Visualization generation functions

**Justification:** While using standard TensorFlow/Keras APIs, the specific architecture choices, hyperparameter selection, and implementation logic.

2. Web Application

 - Files: app.py, index.html
    - Components:

      - Flask API endpoints design
      - Image upload and processing logic
      - Model prediction endpoints
      - Frontend UI/UX design
      - JavaScript prediction logic
      - Responsive CSS styling

3. Testing Scripts

 - File: test_prediction.py
    -Components:

      - Single image testing functionality
      - Interactive Google Colab upload feature
      - Visualization of prediction results

# Third-Party Libraries

### TensorFlow/Keras Framework

- License: Apache 2.0
- Source: https://www.tensorflow.org/
- Citation: Abadi, M., Agarwal, A., Barham, P., et al. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.

**Usage in Project:**

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

### VGG16 Pre-trained Model

- License: MIT (Model architecture from Keras Applications)
- Source: https://keras.io/api/applications/vgg/
- Original Paper: Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

**Usage in Project:**
```
from tensorflow.keras.applications import VGG16

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)
```

### OpenCV (cv2)

- License: Apache 2.0
- Source: https://opencv.org/
- Citation: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

```
import cv2
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
```

### Scikit-learn Metrics

- License: BSD 3-Clause
- Source: https://scikit-learn.org/
- Citation: Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

**Usage in Project:**

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## Adapted Code Snippets

1. Image Preprocessing Pipeline

```
train_datagen = ImageDataGenerator(
    rotation_range=15,        # Adjusted for leaf orientation
    width_shift_range=0.1,    # Added horizontal shift
    height_shift_range=0.1,   # Added vertical shift
    zoom_range=0.1,           # Added zoom augmentation
    horizontal_flip=True,
    fill_mode='nearest'       # Specified fill mode
)
```

**Changes Made:**

- Customized augmentation parameters for plant disease task
- Added multiple augmentation types
- Specified fill mode for edge handling

2. Model Training with Callbacks

```
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=7),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=4),
        ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
    ],
    verbose=1
)
```

**Changes Made:**

- Integrated data augmentation generator
- Added multiple callbacks for training optimization
- Configured validation monitoring
- Customized callback parameters

3. Confusion Matrix Visualization

```
clean_names = [name.split('___')[1].replace('_', ' ') for name in class_names]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clean_names, yticklabels=clean_names)
plt.title('Confusion Matrix - Working CNN', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
```

**Changes Made:**

- Added custom class name processing
- Specified format and color scheme
- Added labels and titles
- Created subplot layout for comparison

4. Flask File Upload Handling

```
if 'image' not in request.files:
    return jsonify({'error': 'No image file provided'}), 400

file = request.files['image']

if file.filename == '':
    return jsonify({'error': 'No file selected'}), 400

if not allowed_file(file.filename):
    return jsonify({'error': 'Invalid file type'}), 400

image_data = file.read()
processed_image = preprocess_image(image_data)
predictions = predict_disease(processed_image, model_type)
```

**Changes Made:**

- Added comprehensive error handling
- Implemented file type validation
- Added image preprocessing integration
- Created JSON API responses