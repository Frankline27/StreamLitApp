from PIL import Image
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = (224, 224)
mri_classifier = tf.keras.models.load_model("efficientnet_model.keras")

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

image_folder = "Test data"

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(image_path)
            preprocessed = preprocess_image(image)
            pred = mri_classifier.predict(preprocessed)[0]  # pred is likely an array

            confidence = float(pred)  # Convert to scalar float

            label = "Non-MRI" if confidence > 0.5 else "MRI"
            print(f"{filename} → Prediction: {label} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
