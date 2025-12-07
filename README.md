# LeNet-MNIST Handwritten Digit Classification

A PyTorch implementation of the LeNet architecture for classifying handwritten digits from the MNIST dataset. This project achieves **98.34% accuracy** on the test set.

## Overview

This project implements a modified LeNet convolutional neural network to classify handwritten digits (0-9) from the MNIST dataset. The implementation includes:

- Custom dataset class for MNIST data preprocessing
- LeNet architecture with adaptive average pooling
- Training pipeline with progress tracking
- Model evaluation and inference capabilities
- Pre-trained model for quick predictions

## Architecture

The LeNet architecture used in this project consists of:

- **Conv1**: 1 → 6 channels, 5×5 kernel, padding=2
- **Pool1**: Average pooling, 2×2 kernel, stride=2
- **Conv2**: 6 → 16 channels, 5×5 kernel
- **Pool2**: Average pooling, 2×2 kernel, stride=2
- **GAP**: Global Average Pooling (AdaptiveAvgPool2d)
- **FC1**: 16 → 120 fully connected layer
- **FC2**: 120 → 84 fully connected layer
- **FC3**: 84 → 10 output layer (one for each digit)

All convolutional and fully connected layers (except the output) use ReLU activation.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow (PIL)
- tqdm

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

## Project Structure

```
LeNet-MNIST/
├── LeNet_MNIST.ipynb           # Main notebook with training and inference code
├── model.pth                   # Pre-trained model weights
├── test.png                    # Sample test image
└── README.md                   # This file
```

## Usage

### Training

The training process is implemented in the Jupyter notebook `LeNet_MNIST.ipynb`. The model is trained for 30 epochs with:
- Batch size: 64
- Optimizer: Adam (learning rate: 0.001)
- Loss function: CrossEntropyLoss
- Data augmentation: Resize, ToTensor, Normalize

### Evaluation

The trained model achieves **98.34% accuracy** on the MNIST test set (10,000 images).

### Inference

Use the `imageClassifier` class (defined in the notebook) to make predictions on new images:

```python
# Run the notebook cells to define the classes, then use:
inference_model = imageClassifier("model.pth")

# Predict on a new image
prediction = inference_model.predict("path/to/your/image.png")
print(f"Predicted digit: {prediction}")
```

The `imageClassifier` class:
- Loads the pre-trained model automatically
- Handles image preprocessing (conversion to grayscale, normalization)
- Returns the predicted digit class (0-9)

## Results

- **Training Loss**: Decreased from 0.0947 to ~0.0483 over 30 epochs
- **Test Accuracy**: **98.34%**
- **Model Size**: Saved as `model.pth`

## Key Features

1. **Custom Dataset Class**: Implements a flexible dataset wrapper for MNIST with custom transforms
2. **GPU Support**: Automatically uses CUDA if available, falls back to CPU
3. **Progress Tracking**: Uses tqdm for training and evaluation progress bars
4. **Model Persistence**: Saves trained model weights for later use
5. **Easy Inference**: Simple API for making predictions on new images

## Notes

- The model uses adaptive average pooling instead of traditional fully connected layers after convolutions, which reduces parameters and improves generalization
- The dataset is automatically downloaded on first run (saved to `./data` directory)
- All images are normalized to [-1, 1] range using mean=0.5 and std=0.5

## License

This project is part of a computer vision course and is provided for educational purposes.

