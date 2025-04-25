import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Read the dataset
df = pd.read_csv('image_dataset_10_classes.csv')

# Create output directories
output_dir = 'preprocessed_images'
os.makedirs(output_dir, exist_ok=True)

# Image preprocessing parameters
IMG_SIZE = (224, 224)  # Standard size for many CNN models
BATCH_SIZE = 32

def preprocess_image(image_path, target_size=IMG_SIZE):
    """Preprocess a single image"""
    try:
        # Open and resize image
        img = Image.open(image_path)
        img = img.resize(target_size)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Create directories for train and validation
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create subdirectories for each class
for label in df['label'].unique():
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# Set up data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# For validation, we only need rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    save_to_dir=train_dir,
    save_format='jpg'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    save_to_dir=val_dir,
    save_format='jpg'
)

# Print class indices
print("\nClass indices:")
print(train_generator.class_indices)

# Print sample counts
print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Visualize some augmented images
def visualize_augmented_images(generator, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        batch = next(generator)
        image = batch[0][0]
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.show()

# Visualize augmented images
print("\nVisualizing augmented images...")
visualize_augmented_images(train_generator)

# Save the preprocessed data information
preprocessed_info = {
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'train_samples': train_generator.samples,
    'val_samples': val_generator.samples,
    'class_indices': train_generator.class_indices
}

import json
with open('preprocessing_info.json', 'w') as f:
    json.dump(preprocessed_info, f, indent=4)

print("\nPreprocessing completed!")
print(f"Preprocessed images saved to: {output_dir}")
print(f"Preprocessing information saved to: preprocessing_info.json") 