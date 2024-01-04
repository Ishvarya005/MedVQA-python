import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np

# Load the saved DenseNet model
model = load_model("denseNet_model.h5")



# Predictions on a sample image
sample_image_path = "D:\\Sem-3\\Intro to Python\\Project\\Augmentation\\testImages\\hdPos3.jpg"
img = image.load_img(sample_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
prediction = model.predict(img_array)
print("Prediction:", prediction)
