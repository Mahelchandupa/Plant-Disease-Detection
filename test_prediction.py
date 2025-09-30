from google.colab import files
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot']

def test_uploaded_image():
    """Upload and test an image interactively"""
    
    # Upload image
    print("Please upload a tomato leaf image:")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded!")
        return
    
    # Get the uploaded filename
    image_path = list(uploaded.keys())[0]
    print(f"\nUploaded: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Preprocess
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Load model
    print("\nLoading model...")
    MODEL_PATH = '/content/drive/MyDrive/Datasets/PlantVillage/working_models/working_cnn_final.keras'
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Predict
    print("Making prediction...")
    predictions = model.predict(img_batch, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted: {CLASS_NAMES[predicted_idx]}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("\nAll Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15}: {predictions[0][i]*100:5.1f}%")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(img_array)
    ax1.set_title(f'Uploaded Image', fontsize=14)
    ax1.axis('off')
    
    colors = ['green' if i == predicted_idx else 'lightgray' for i in range(4)]
    ax2.barh(CLASS_NAMES, predictions[0] * 100, color=colors)
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Predictions', fontsize=14)
    ax2.set_xlim(0, 100)
    
    # Add confidence text
    ax2.text(confidence*100 + 2, predicted_idx, 
             f'{confidence*100:.1f}%', 
             va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Run the test
test_uploaded_image()