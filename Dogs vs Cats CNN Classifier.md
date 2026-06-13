Dogs vs Cats CNN Classifier

## Description
A Convolutional Neural Network (CNN) trained to classify images as dogs or cats using TensorFlow/Keras. The model achieves **76.90% accuracy** on the test set.

## Tech Stack
- Python 3
- TensorFlow 2.21 / Keras
- NumPy, Matplotlib
- Algorithm: Deep CNN with multiple convolutional layers

## Dataset
CIFAR-10 dataset (dogs and cats extracted)
- Training: 10,000 images (5000 dogs, 5000 cats)
- Testing: 2,000 images (1000 dogs, 1000 cats)
- Image size: 32x32 pixels

## Model Architecture
Conv2D (32 filters, 3x3) → ReLU

MaxPooling (2x2)

Conv2D (64 filters, 3x3) → ReLU

MaxPooling (2x2)

Conv2D (64 filters, 3x3) → ReLU

Flatten

Dense (64 neurons) → ReLU + Dropout (0.5)

Dense (1 neuron) → Sigmoid (binary classification)

**Total Parameters:** 121,985

## How to Run
```bash
python dogs_vs_cats_cnn.py
```

## Results
- **Test Accuracy: 76.90%**
- **Test Loss: 0.7424**
- Training completed in 15 epochs (~2-3 minutes)
- Accuracy curves saved as `training_history.png`

## Learning Outcomes
✅ CNN architecture design  
✅ Data preprocessing and normalization  
✅ Model training with validation  
✅ Accuracy and loss visualization  
✅ Binary classification with sigmoid activation  

## Author
Rohan Gunjal - MIT-ADT University, Pune  
Slash Mark Internship - Task 2