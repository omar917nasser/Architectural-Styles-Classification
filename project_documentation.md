# Architectural Style Classification using Machine Learning and Deep Learning

[![GitHub Repository Size](https://img.shields.io/github/repo-size/Seif-Eldin-Omar/Architectural-Styles-Classification)](https://github.com/Seif-Eldin-Omar/Architectural-Styles-Classification)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/Seif-Eldin-Omar/Architectural-Styles-Classification)](https://github.com/Seif-Eldin-Omar/Architectural-Styles-Classification)
![GitHub Followers](https://img.shields.io/github/followers/Seif-Eldin-Omar?style=social)

## Abstract

This project investigates and implements various machine learning and deep learning methodologies for the automated classification of architectural styles from image data. The objective is to build robust models capable of identifying distinct architectural periods and characteristics present in visual content. The repository contains the code and detailed Jupyter notebooks documenting the entire process, from data exploration and preprocessing to the implementation and evaluation of several classification models, including traditional techniques and state-of-the-art deep neural networks.

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
    - [Data Exploration and Analysis](#data-exploration-and-analysis)
    - [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
- [Models Implemented](#models-implemented)
    - [Traditional Machine Learning Models](#traditional-machine-learning-models)
        - [Random Forest](#random-forest)
        - [XGBoost](#xgboost)
    - [Deep Learning Models](#deep-learning-models)
        - [Custom Convolutional Neural Network (CNN)](#custom-convolutional-neural-network-cnn)
        - [VGG16 (Transfer Learning)](#vgg16-transfer-learning)
        - [ResNet50 (Transfer Learning)](#resnet50-transfer-learning)
        - [EfficientNetV2M (Transfer Learning)](#efficientnetv2m-transfer-learning)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Objective

The primary goal of this project is to develop and evaluate machine learning models for the automatic classification of architectural styles from a collection of images. This involves:
- Curating and analyzing a dataset of architectural images across multiple styles.
- Implementing various image preprocessing and feature extraction techniques.
- Training and fine-tuning different classification models, ranging from traditional methods to deep learning architectures.
- Evaluating the performance of each model using standard classification metrics.

## Dataset

The project utilizes a dataset comprising images of various architectural styles.

* **Source:** The dataset is expected to be located at the path `D:\FUCK!!\Pattern\Project\g-images-dataset`. Users will need to adjust this path according to their local setup.
* **Classes:** The dataset is categorized into 10 distinct architectural styles:
    1.  Achaemenid architecture
    2.  American craftsman style
    3.  American Foursquare architecture
    4.  Ancient Egyptian architecture
    5.  Art Deco architecture
    6.  Art Nouveau architecture
    7.  Baroque architecture
    8.  Bauhaus architecture
    9.  Beaux-Arts architecture
    10. Byzantine architecture
* **Volume:** The dataset contains a total of 2185 images.
* **Distribution:** The number of images per class varies, impacting the class balance within the dataset.
* **Splitting:** The dataset is systematically partitioned into training, validation, and testing sets with a distribution of 70%, 15%, and 15% respectively. Stratified splitting is employed to maintain proportional representation of each architectural style across all sets.

## Methodology

The project follows a structured machine learning workflow:

### Data Exploration and Analysis

The `1_data_exploration_and_analysis.ipynb` notebook provides an initial deep dive into the dataset. It includes:
- Analysis of the total number of images and classes.
- Visualization of the class distribution to understand potential imbalances.
- Examination of image characteristics such as dimensions (width and height) and aspect ratios.
- Generation of a `image_dataset.csv` file containing metadata about the images.

### Preprocessing and Feature Engineering

The `2_preprocessing.ipynb` notebook covers essential data preparation steps:
- Loading dataset information from the generated CSV file.
- Creating dedicated directories for storing preprocessed grayscale and RGB images.
- Implementing the stratified split of the dataset into training, validation, and testing subsets.
- Applying various feature extraction techniques crucial for traditional machine learning models, including:
    -   Local Binary Pattern (LBP)
    -   Histogram of Oriented Gradients (HOG)
    -   Gabor Filters
    -   Color Histograms
- Incorporating data augmentation techniques (rotations, shifts, zooms) on the training set to enhance model generalization and mitigate overfitting.
- Scaling the extracted features using `MinMaxScaler`.
- Employing Principal Component Analysis (PCA) for dimensionality reduction on the feature sets, preserving 95% of the variance.
- Saving the preprocessed image data and extracted feature sets in NumPy format (`.npy`) for efficient loading by the models.

## Models Implemented

A variety of machine learning and deep learning models have been implemented and evaluated for the architectural style classification task.

### Traditional Machine Learning Models

These models utilize the features extracted during the preprocessing phase.

#### Random Forest

- **Notebook:** `RandomForest.ipynb`
- **Description:** Implements a RandomForestClassifier on the extracted features (primarily utilizing the PCA-reduced grayscale features).
- **Configuration:** Trained with `n_estimators=200`, `criterion='entropy'`, and `class_weight='balanced'` to address potential class imbalance.
- **Evaluation:** Assessed using accuracy score, classification report, confusion matrix, and ROC AUC curve.

#### XGBoost

- **Notebook:** `xgboost.ipynb`
- **Description:** Utilizes an XGBoostClassifier with the preprocessed and PCA-reduced feature sets (including RGB features).
- **Preprocessing:** Employs `LabelEncoder` for the target variable.
- **Evaluation:** Evaluated using classification report, confusion matrix, and accuracy score.

### Deep Learning Models

These models work directly with image data, often employing transfer learning from pre-trained networks.

#### Custom Convolutional Neural Network (CNN)

- **Notebook:** `CNN.ipynb`
- **Description:** Implementation of a custom-built CNN architecture from scratch.
- **Input Shape:** Designed to process images with dimensions (64, 64, 3).
- **Architecture:** Composed of multiple convolutional and pooling layers, followed by batch normalization and dense layers.
- **Training:** Compiled with the Adam optimizer and utilizes sparse categorical crossentropy loss. Training incorporates Early Stopping and Model Checkpointing callbacks.
- **Evaluation:** Performance is analyzed using a classification report and confusion matrix.

#### VGG16 (Transfer Learning)

- **Notebook:** `VGG16.ipynb`
- **Description:** Leverages the pre-trained VGG16 model by fine-tuning its layers on the architectural dataset.
- **Input Shape:** Configured for an input size of (224, 224, 3).
- **Preprocessing:** Uses the dedicated `preprocess_input` function for VGG16.
- **Training:** Compiled with the Adam optimizer. Training is managed with Early Stopping and Model Checkpointing callbacks.
- **Evaluation:** Evaluated using a classification report and confusion matrix.

#### ResNet50 (Transfer Learning)

- **Notebook:** `RestNet50.ipynb`
- **Description:** Adapts the pre-trained ResNet50 model for the classification task through transfer learning.
- **Input Shape:** Accepts images with dimensions (128, 128, 3).
- **Preprocessing:** Employs the `preprocess_input` function specific to ResNet50.
- **Training:** Trained using the Adam optimizer and sparse categorical crossentropy loss, with Early Stopping and Model Checkpointing callbacks.
- **Evaluation:** Performance is measured using a classification report and confusion matrix.

#### EfficientNetV2M (Transfer Learning)

- **Notebook:** `efficientnet.ipynb`
- **Description:** Applies transfer learning using the advanced EfficientNetV2M architecture.
- **Input Shape:** Processes images of size (128, 128, 3).
- **Preprocessing:** Notably uses the `preprocess_input` function from ResNet50, indicating a potential area for review or a specific design choice.
- **Training:** Compiled with the Adam optimizer and trained with Early Stopping and Model Checkpointing.
- **Evaluation:** Performance is evaluated using a classification report.

## Results

Evaluation of the implemented models was conducted using standard classification metrics, including accuracy, precision, recall, and F1-score, typically presented in classification reports and confusion matrices. While a comprehensive comparison requires running all notebooks, a summary of available results is presented below:

| Model                  | Accuracy | 
| :--------------------- | :------- | 
| Random Forest          | 0.3140   | 
| XGBoost                | 0.42 | 
| Custom CNN             | 0.55 | 
| VGG16 (Transfer Learning) | 0.82 | 
| ResNet50 (Transfer Learning) | 0.74 | 
| EfficientNetV2M (Transfer Learning) | 0.73 |

For other measures see the nootbooks

## Setup and Installation

To replicate the experiments and run the code in this repository, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Seif-Eldin-Omar/Architectural-Styles-Classification.git](https://github.com/Seif-Eldin-Omar/Architectural-Styles-Classification.git)
    cd Architectural-Styles-Classification
    ```

2.  **Set up a Python Environment:** It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Using conda
    conda create -n arch-style-classifier python=3.8  # Or your preferred Python version
    conda activate arch-style-classifier
    ```

3.  **Install Dependencies:** Install the required libraries using pip.
    ```bash
    pip install numpy tensorflow keras scikit-learn matplotlib pandas Pillow seaborn scikit-image xgboost tqdm
    ```
    *Note: Ensure you have a compatible version of TensorFlow installed, preferably with GPU support if available for faster training of deep learning models.*

4.  **Dataset:** Obtain the architectural images dataset and place it in a directory on your system. Update the `base_path` variable in the relevant notebooks (`1_data_exploration_and_analysis.ipynb`, `2_preprocessing.ipynb`, and potentially others if the preprocessed data paths are relative) to point to the location of your dataset. The expected structure is a main directory containing subdirectories for each architectural style.

## Usage

Execute the Jupyter notebooks sequentially to proceed through the project workflow:

1.  **Data Exploration:** Run `1_data_exploration_and_analysis.ipynb` to understand the dataset and generate the necessary CSV file.
2.  **Preprocessing:** Run `2_preprocessing.ipynb` to preprocess the images, extract features, and prepare the data splits. This notebook will create output directories and save processed data files.
3.  **Model Training and Evaluation:** Run the notebooks for each model (`CNN.ipynb`, `efficientnet.ipynb`, `RestNet50.ipynb`, `VGG16.ipynb`, `RandomForest.ipynb`, `xgboost.ipynb`) to train the models and evaluate their performance.

Each notebook contains detailed steps and code within its cells, which can be executed in order.

## Team Members

- Omar Nasser: [https://github.com/omar917nasser](https://github.com/omar917nasser)
- Seif Eldin Omar: [https://github.com/Seif-Eldin-Omar](https://github.com/Seif-Eldin-Omar)
