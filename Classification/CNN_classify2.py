import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, callbacks
import numpy as np
# Define the paths to your dataset
dataset_path = "D:\\Sem-3\\Intro to Python\\Project\\Augmentation\\Datasets"  # Change this to the path of your dataset
train_data_dir = os.path.join(dataset_path, "TrainingData")

# Use the same directory for both training and test data
test_data_dir = train_data_dir

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data Generator for Training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Data Generator for Training and Validation (using the same directory)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # Since it's binary classification (0 or 1)
    subset='training',  # For training data
    shuffle=True
)

# Data Generator for Validation
val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',  # For validation data
    shuffle=False  # No need to shuffle validation data
)

# Create a base model using pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model on top of the pre-trained base
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and model checkpoint
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=15, mode='max', verbose=1)
model_checkpoint = callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', mode='max', save_best_only=True)

# Train the model with callbacks
model.fit(train_generator, epochs=100, steps_per_epoch=len(train_generator),
          validation_data=val_generator, validation_steps=len(val_generator),
          callbacks=[early_stopping, model_checkpoint])

# Load the best model (if early stopping occurred)
model.load_weights("best_model.h5")

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # No need to shuffle test data
)

eval_result = model.evaluate(test_generator)
print("Test Loss:", eval_result[0])
print("Test Accuracy:", eval_result[1])

# Predictions on a sample image (adjust the path accordingly)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Define the path to the image
sample_image_path = "D:\\Sem-3\\Intro to Python\\Project\\Augmentation\\testImages\\1.jpg"

# Load the image with target size (224, 224) as expected by ResNet50
img = image.load_img(sample_image_path, target_size=(224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand the dimensions to create a batch (batch size = 1)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image for ResNet50 model
img_array = preprocess_input(img_array)

# Now, you can use the preprocessed image in your model

prediction = model.predict(img_array)
print("Prediction:", prediction)
