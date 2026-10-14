# Deep Learning Project Report
## Title
**Image Classification Using a Convolutional Neural Network (CNN)**

## 1. Introduction
Deep learning is a branch of machine learning that uses multi-layer neural networks to learn patterns from data. Image classification is a computer-vision task in which a model assigns an image to one of several categories. This project uses a Convolutional Neural Network (CNN) to classify small colour images.

## 2. Problem Statement
To build and evaluate a deep-learning model that classifies images into ten categories using the CIFAR-10 dataset.

## 3. Objectives
- Understand the basic workflow of a deep-learning project.
- Load and preprocess image data.
- Build a CNN using TensorFlow/Keras.
- Train and evaluate the model.
- Visualize accuracy and sample predictions.

## 4. Dataset
CIFAR-10 contains 60,000 colour images of size 32 × 32 pixels across ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The standard split contains 50,000 training images and 10,000 test images.

## 5. Tools and Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Convolutional Neural Network (CNN)

## 6. Methodology
1. Load the CIFAR-10 dataset.
2. Normalize pixel values from 0–255 to 0–1.
3. Create convolution and pooling layers to learn image features.
4. Use dense layers to classify the extracted features.
5. Train the model using the Adam optimizer and sparse categorical cross-entropy loss.
6. Evaluate the model on test images.
7. Save the trained model and result plots.

## 7. Model Architecture
The model uses three convolutional layers, max-pooling layers, a flattening layer, a dense layer, dropout for regularization, and a ten-unit softmax output layer.

## 8. Results
Run `main.py` to generate the actual test accuracy and the following output files:
- `outputs/accuracy_plot.png`
- `outputs/sample_predictions.png`
- `outputs/cifar10_cnn.keras`

**Important:** Fill in the measured test accuracy after running the code. Results depend on the environment and training settings; no result is claimed before execution.

## 9. Applications
- Photo categorization
- Image-search systems
- Quality inspection
- Basic computer-vision learning and prototyping

## 10. Limitations
The dataset contains small images, and accuracy may be limited by the simple architecture and short training period. Real-world images may differ from CIFAR-10 examples.

## 11. Future Scope
- Train for more epochs and tune hyperparameters.
- Apply data augmentation.
- Compare CNN performance with transfer-learning models.
- Test on a carefully prepared custom image dataset.

## 12. Conclusion
This project demonstrates the core steps of an image-classification pipeline: data preparation, CNN design, training, evaluation, and visualization. It provides a foundation for exploring more advanced computer-vision models.

## References
1. CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. TensorFlow documentation: https://www.tensorflow.org/
3. Keras documentation: https://keras.io/
