# Garbage_classifier
## About Dataset
This dataset contains a balanced collection of images for garbage classification, organized into six classes: plastic, metal, glass, cardboard, paper, and trash, it's made by a really big ensemble for better quality, balance and performance.<br>

All images are standardized: resized to 256x256 pixels, in RGB format, and cleaned of duplicates for quality and consistency.<br>

The dataset includes around 2,300 to 2,500 images per class, making it well-balanced for training machine learning models.<br>

It’s perfect for anyone working on waste sorting or environmental AI projects, providing clean, diverse, and ready-to-use data.<br>

## My approch
## Project Title: Advanced Waste Classification Using a Fine-Tuned MobileNetV2 Architecture with Strategic Regularization
`Objective:` To develop a highly accurate and robust deep learning model for automated garbage classification into six categories (Cardboard, Glass, Metal, Paper, Plastic, Trash), minimizing overfitting to ensure generalizability for real-world application.

## Executive Summary / Abstract
The challenge of automated waste sorting is critical for modern recycling systems. This project tackles the problem of image-based garbage classification by implementing a sophisticated deep learning pipeline. After extensive experimentation with various architectures and techniques, a fine-tuned MobileNetV2 model was developed, achieving a **validation accuracy of 94.78% and a test accuracy of 94.7%** with minimal overfitting. This was accomplished through a combination of aggressive data augmentation, multi-layer regularization strategies (Dropout, Batch Normalization), and dynamic learning rate optimization, demonstrating a highly effective approach for a complex multi-class image recognition task.

**Key Methodology & Technical Implementation**
1) Dataset & Preprocessing:

   a) Utilized the "Garbage Dataset Classification" from Kaggle, containing 13,901 images across       6 classes.

   b) Implemented a robust data cleaning class **(CLEAN_DATA)** to standardize all images to (224,         224, 3) RGB format.

   c) Applied **LabelEncoder** for efficient class label processing.

2) Data Augmentation:

      a) Engineered a comprehensive data augmentation pipeline using **ImageDataGenerator** to  artificially expand the dataset and improve model generalization.               Techniques included:

            i) Rotation (±45°), Width/Height Shifting (10%), Zooming (40%)

            ii) Shear Transformation (15°), Horizontal Flip, Brightness Variation

      b) This was crucial in teaching the model to be invariant to real-world variations in  object orientation, lighting, and placement.

3) Model Architecture:

      a) Base Model: Employed MobileNetV2 pre-trained on ImageNet for powerful feature  extraction, with its weights set as trainable for fine-tuning on the                 specific garbage dataset.

      b) Custom Classifier Head: Added a carefully designed sequence of layers on top of the base model:

            i) GlobalAveragePooling2D to reduce spatial dimensions.

            ii)  Multiple Dense layers (512, 256, 128 units) with ReLU activation.

            iii) Strategic Regularization: Integrated Dropout layers (rates: 0.5, 0.4, 0.3, 0.2) after each dense layer and Batch Normalization to significantly                     reduce overfitting by preventing complex co-adaptations of neurons.

             iv) Final Dense layer with 6 units and softmax activation for classification.

4) Training Strategy & Callbacks:

      1) Optimizer: Adam with an initial learning rate of 0.001.

      2) Callbacks: Implemented a powerful callback suite to automate and optimize training:

            i) EarlyStopping: Halted training after 5 epochs without improvement in validation loss, restoring the best weights to prevent overfitting.
            
            ii) ReduceLROnPlateau: Dynamically reduced the learning rate by a factor of 0.2 upon a plateau in validation loss, allowing for finer weight updates                     (final LR: 1.6e-06).
            
            iii) ModelCheckpoint: Automatically saved the best model based on validation accuracy (val_accuracy).

## Results & Conclusion
The model converged effectively, showing strong and parallel trends in training and validation metrics—a clear indicator that overfitting was successfully mitigated.

1) Final Validation Accuracy: 94.78% (Epoch 29)

2) Final Test Accuracy: 94.7%

3) Final Validation Loss: 0.1833

This project successfully demonstrates the application of transfer learning and advanced regularization techniques to solve a pressing environmental problem. The meticulous approach to combating overfitting—through data augmentation, a carefully tuned network architecture, and intelligent training callbacks—resulted in a model that is not only accurate but also generalizable. This work underscores the potential of deep learning in creating efficient and scalable waste management solutions and reflects a deep understanding of the practical challenges in deploying AI models.

*Keywords: Deep Learning, Computer Vision, Waste Classification, MobileNetV2, Transfer Learning, Regularization, Data Augmentation, TensorFlow, Keras, Overfitting Mitigation.*
