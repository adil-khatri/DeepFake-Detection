import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
from tensorflow.keras.models import load_model

import tensorflow as tf
print(tf.config.list_physical_devices())
print(tf.config.list_logical_devices())


# Load the model
model = load_model('./Models/model.h5')

# Load and preprocess input data
image_path = './Real_img.jpg'
image = cv2.imread(image_path)  # Load your image
image = cv2.resize(image, (224, 224))  # Resize your image to match the input size of your model
image = image / 255.0  # Normalize your image

# Reshape your image to match the input shape expected by the model
image = np.expand_dims(image, axis=0)  # Add a batch dimension

# Make predictions using the loaded model
predictions = model.predict(image)

# You can then interpret the predictions according to your model's output
# For binary classification models, predictions closer to 1 indicate a positive class, while predictions closer to 0 indicate a negative class
print(predictions)

