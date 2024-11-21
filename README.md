# FaceMaskDetection

Face Mask Detection


A deep learning project for detecting whether individuals are wearing face masks, using Convolutional Neural Networks (CNNs) and transfer learning techniques.
📋 Table of Contents

    Project Overview
    Technologies Used
    Dataset
    Deep Learning Approach
    Installation
    Usage
    Results
    Contributing
    License

📖 Project Overview

The goal of this project is to classify images as either with a mask or without a mask using deep learning techniques. This is a crucial application in public health monitoring systems, especially during pandemics.

The system leverages convolutional neural networks (CNNs) for feature extraction and classification, ensuring high accuracy and robust performance.
💻 Technologies Used

    Python 3.12
    Libraries and Frameworks:
        TensorFlow/Keras
        OpenCV
        NumPy
        Matplotlib
        Scikit-learn
    Hardware: GPU (for faster training)

📂 Dataset

The dataset contains images of individuals with and without face masks. It includes:

    Labeled data for supervised learning.
    Variations in lighting, angles, and environments to improve model robustness.

Dataset Sources:

    Kaggle Face Mask Dataset or other publicly available repositories.

🧠 Deep Learning Approach
1. Data Preprocessing

    Image Resizing: All images resized to 128x128 pixels.
    Normalization: Pixel values scaled to [0, 1].
    Data Augmentation: Techniques like rotation, flipping, zooming, and brightness adjustments were applied to increase data diversity.

2. Model Architecture

We designed a custom CNN and also leveraged transfer learning with pre-trained models like MobileNetV2 for feature extraction.
Custom CNN Architecture:

    Convolutional Layers: For feature extraction.
    Pooling Layers: To reduce dimensionality.
    Dropout Layers: To prevent overfitting.
    Dense Layers: For classification.

Transfer Learning with MobileNetV2:

    Pre-trained on ImageNet.
    Fine-tuned the final layers for binary classification (Mask/No Mask).

3. Model Training

    Loss Function: Binary Crossentropy
    Optimizer: Adam
    Metrics: Accuracy, Precision, Recall, F1-Score

🛠 Installation

    Clone the repository:

git clone https://github.com/yourusername/FaceMaskDetection.git  
cd FaceMaskDetection  

Install dependencies:

    pip install -r requirements.txt  

    Download the dataset and place it in the data/ folder.

    Optional: If you plan to use a GPU, ensure TensorFlow GPU is installed.

🚀 Usage

    Train the Model:
    Train the deep learning model:

python train.py  

Evaluate the Model:
Test the model on a validation dataset:

python evaluate.py  

Real-Time Detection:
Run the script for real-time face mask detection using a webcam:

    python detect_mask_webcam.py  

    Interactive Notebook:
    Use the Jupyter Notebook FaceMaskDetection.ipynb to explore the training, evaluation, and prediction process.

📊 Results

Visualizations:

    Training Accuracy: 0.9903

    Loss : 0.0273

🤝 Contributing

Contributions are welcome! Follow these steps:

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Commit your changes: git commit -m 'Add feature'.
    Push to the branch: git push origin feature-name.
    Submit a pull request.

📝 License

This project is licensed under the MIT License.
