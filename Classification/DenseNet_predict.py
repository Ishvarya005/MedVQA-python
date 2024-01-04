import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import classify_question
# Load the saved DenseNet model
model = load_model("denseNet_model.h5")



# Predictions on a sample image
sample_image_path = "C:\\Users\\kavya\\Downloads\\MedVQA-python-master\\Augmentation\\testImages\\hdPos3.jpg"
img = image.load_img(sample_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
prediction = model.predict(img_array)
print("Prediction:", prediction)
p = prediction

def ask_question(int):
    with open("questions.txt", "r") as file:
        questions = file.readlines()
        
    str = input("Enter your Question: ")
    
    que = "do i have hd\n"
    q = classify_question.find_similar_question(str, questions)
    print(q)
    print(que)
    if q.lower() == que:
        if p[0][0] == 1:
            print("Yes, you have HD")
        elif p[0][0] == 0:
            print("No, you dont have HD")
            
    
ask_question(p)
        