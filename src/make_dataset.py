import os
import shutil
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_image_paths_and_labels(base_path):
    image_paths = []
    labels = []
    for class_name in os.listdir(base_path):
        class_dir = os.path.join(base_path, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_name)
    return image_paths, labels

def split_dataset(image_paths, labels, output_dir, test_size=0.15, val_size=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(test_size + val_size), stratify=labels, random_state=42
    )
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=42
    )

    dataset = [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]

    for split_name, X, y in dataset:
        for path, label in zip(X, y):
            dest_dir = os.path.join(output_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))
