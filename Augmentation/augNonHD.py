import Augmentor

# Specify the path to your dataset
dataset_path = "D:\Sem-3\Intro to Python\Project\Augmentation\Datasets\HD-Negative"

# Create an Augmentor Pipeline
pipeline = Augmentor.Pipeline(dataset_path ,output_directory="D:\Sem-3\Intro to Python\Project\Augmentation\Datasets\HD-Negative Aug") #creating an augmentor pipeline (a class in the library) object specifying the input path and output path as parameters


# Define augmentation operations

pipeline.flip_left_right(probability=0.5)
pipeline.flip_top_bottom(probability=0.5)
pipeline.rotate_random_90(probability=0.4)  # Random 90-degree rotations
pipeline.zoom_random(probability=0.6, percentage_area=0.95)#this line of code adds a random zoom operation to the Augmentor pipeline with a 50% chance of being applied to each image, and the zoomed images will retain 90% of their original area.

# Intensity changes
pipeline.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.2)# this line of code applies a random brightness adjustment to images in the Augmentor pipeline with a 50% probability. The brightness adjustment is randomly chosen between 80% and 120% of the original brightness.
pipeline.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.2)
pipeline.random_color(probability=0.5, min_factor=0.7, max_factor=1.2)


# Add Gaussian noise
pipeline.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

# Rescaling (adjust min_max_value based on your data)
pipeline.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.2)


# Set the number of augmented samples you want to generate
num_augmented_samples = 1000

# Execute the augmentation process
pipeline.sample(num_augmented_samples)
