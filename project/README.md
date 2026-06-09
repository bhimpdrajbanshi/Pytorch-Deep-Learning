
# PyTorch Custom Dataset and DataLoader

## Project Overview
This project demonstrates a complete PyTorch data pipeline using a custom Kaggle image dataset.
It includes a custom Dataset class, DataLoader, image transformations, and visualization.

## Features
- Custom Dataset class
- Automatic class discovery
- Image preprocessing
- DataLoader batching and shuffling
- Visualization of images and labels

## Project Structure
```text
project/
├── dataset/
├── notebooks/
│   ├── notebook.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
├── README.md
└── requirements.txt
```

Requirements
torch
torchvision
matplotlib
Pillow

Install:

pip install -r requirements.txt

Dataset

Use a Kaggle multi-class Flowers Recognition image dataset with 5 classes.

link : https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download

Transformations
Resize (128x128)
ToTensor()
DataLoader
batch_size=16
shuffle=True
num_workers=2
Expected Output
Dataset size: 4317
Classes: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


## Learning Outcomes
- Custom Dataset creation
- DataLoader usage
- Image preprocessing
- Batch generation
- Deep learning data pipeline

Author

Bhim Prasad Rajbanshi