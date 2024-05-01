# IEEE-ML-PROJECT-I04
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to the dataset folders in your Google Drive
train_directory = '/content/drive/My Drive/IEEE ML DATASET/TrainIJCNN2013'
test_directory = '/content/drive/My Drive/IEEE ML DATASET/TestIJCNN2013'

# Function to load and preprocess images from subfolders
def load_images_from_subfolder(directory, img_size=(32, 32)):
    images = []
    labels = []
    
    # Create a dictionary to map folder names to integer labels
    label_map = {}
    for i, class_name in enumerate(os.listdir(directory)):
        label_map[class_name] = i
    
    # Iterate over each subfolder (class) in the dataset directory
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        label = label_map[class_name]
        
        if os.path.isdir(class_dir):
            # Iterate over each image file in the class directory
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                
                # Read and preprocess the image
                img = cv2.imread(img_path)
                
                # Check if the image is not empty
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = img / 255.0  # Normalize pixel values

                    # Append the image and its label to the lists
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

# Load images and labels from the training and test subfolders
X_train, y_train = load_images_from_subfolder(train_directory)
X_test, y_test = load_images_from_subfolder(test_directory)

# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
