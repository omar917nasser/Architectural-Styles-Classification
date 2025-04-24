# 🏛️ Architectural Style Classification

This project classifies architectural styles from images using classical ML, CNNs, and transfer learning.

## 📁 Dataset
- [Kaggle Link](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset)
- 10,113 images, 25 architectural styles
- Contains both Google Images and ECCV2014 dataset

## 🧪 Models
- Classical: SVM, XGBoost
- Deep Learning: CNN from scratch
- Transfer Learning: VGG16, ResNet50, EfficientNet

## 📊 Results
- Best model: [insert best model]
- Accuracy: XX%
- F1-Score: XX%

## 📂 Structure
- `notebooks/`: Prototyping and experiments
- `src/`: Clean, modular code
- `data/`: Raw and processed datasets
- `results/`: Plots, metrics, and outputs

## 👨‍💻 Team
- Seif (Team Lead)
- [Other team members]

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/train.py
