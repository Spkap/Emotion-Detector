# Emotion Detection using ResNet18

Welcome to the Emotion Detection project! This project leverages the power of deep learning to recognize emotions from facial expressions using a Convolutional Neural Network (CNN) based on the ResNet18 architecture. Whether you're a data scientist, a machine learning enthusiast, or just curious about AI, this project is a great way to explore the fascinating world of emotion recognition.

## Project Highlights

- **Transfer Learning**: Utilizes a pre-trained ResNet18 model, fine-tuned for emotion detection.
- **Data Augmentation**: Enhances model robustness with a variety of transformations.
- **Custom Modifications**: Tailored for grayscale images, with selective layer freezing for efficient learning.
- **Performance Tracking**: Includes detailed metrics and visualizations to monitor model performance.

## Getting Started

### Prerequisites

To get started, you'll need the following Python packages:

```bash
pip install torch torchvision numpy pandas opencv-python Pillow scikit-learn matplotlib seaborn tqdm
```

### Dataset

The model is trained on the FER2013 dataset, which is structured as follows:

```
/kaggle/input/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### Model Architecture

- **Base Model**: ResNet18 with ImageNet weights.
- **Modifications**:
  - First convolutional layer adjusted for single-channel (grayscale) input.
  - Final fully connected layer replaced to output seven emotion classes.
  - Selective layer freezing to retain learned features while adapting to new data.

### Data Augmentation

To improve generalization, the following augmentations are applied:

- Random horizontal flips
- Random rotations up to 10 degrees
- Random affine transformations
- Random sharpness adjustments
- Random autocontrast
- Normalization to standardize input

### Training the Model

The training process involves:

1. **Initialization**: Load and modify the ResNet18 model.
2. **Optimization**: Use AdamW optimizer with differential learning rates for different layers.
3. **Learning Rate Scheduling**: Adjust learning rates based on validation loss.
4. **Gradient Clipping**: Prevent exploding gradients by clipping them.
5. **Early Stopping**: Save the best model based on validation accuracy.

Here's a snippet to kick off training:

```python
model = create_resnet18_model(num_classes=7).to(device)
history = train_and_validate(model, train_loader, test_loader, num_epochs=10, learning_rate=0.0001)
```

### Visualizing Results

After training, visualize the model's performance:

- **Loss and Accuracy Curves**: Track training and validation loss/accuracy over epochs.
- **Confusion Matrix**: Understand model predictions across different emotions.
- **Classification Report**: Detailed precision, recall, and F1-score for each emotion.

### Example Visualizations

- **Training and Validation Loss**: Shows how the model learns over time.
- **Confusion Matrix**: Provides insights into which emotions are often confused.
- **Accuracy Over Time**: Visualizes improvements in model accuracy.

### Performance Metrics

- **Best Training Accuracy**: 74.52%
- **Best test Accuracy**: 66.88%
- **Detailed Classification Report**: Available in the output section of the notebook.

### Future Work

- Explore other architectures like EfficientNet or Vision Transformers.
- Implement real-time emotion detection.
- Enhance data preprocessing and augmentation techniques.
- Experiment with ensemble methods for improved accuracy.

## Conclusion

This project demonstrates the potential of deep learning in emotion recognition. By leveraging transfer learning and data augmentation, the model achieves decent accuracy in classifying emotions from facial expressions.