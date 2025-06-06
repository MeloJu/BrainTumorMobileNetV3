# MobileNetV3 - Brain Tumor Classification on MRI Scans

This repository contains code for training and evaluating a MobileNetV3-based model to classify brain MRI images according to the presence or absence of tumors.

## Objective

Train a binary image classification model using MobileNetV3 to distinguish between:

- MRI images with brain tumors  
- MRI images without brain tumors

## Dataset

- Source: [Kaggle - Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data)  
- Classes: `'no'` (no tumor), `'yes'` (tumor)  
- Image format: `.jpg`  
- Input size: 224x224 pixels

## Preprocessing

- Resize images to 224x224
- Normalize pixel values
- Split into training and validation sets
- Data augmentation using `ImageDataGenerator`

## Model

- Base architecture: `MobileNetV3Large` pre-trained on ImageNet
- Custom head:
  - GlobalAveragePooling2D
  - Dropout (rate = 0.2)
  - Dense layer with 1 unit and sigmoid activation
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metric: Accuracy

## Training

- Epochs: 10  
- Batch size: 32  
- EarlyStopping monitoring `val_loss` with patience of 3 epochs

## Results

- Best validation accuracy: 95.71% at epoch 9  
- Best validation loss: 0.1289 at epoch 9  
- Accuracy gap (train vs validation): 11.97%

## Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.x  
- matplotlib  
- numpy  
- scikit-learn  

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

To train the model:

```bash
python MobileNetV3.ipynb
```

Or open the notebook using Jupyter:

```bash
jupyter notebook MobileNetV3.ipynb
```

## Repository Structure

```
├── MobileNetV3.ipynb         # Main notebook
├── /dataset/                 # Image dataset (not included)
├── /plots/                   # Accuracy and loss charts (optional)
├── README.md
└── requirements.txt          # Optional dependencies file
```

## License

This project is licensed under the [MIT License](LICENSE).
