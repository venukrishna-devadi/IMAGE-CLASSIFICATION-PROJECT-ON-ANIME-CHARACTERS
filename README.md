# Dragon Ball Z Character Classification with Deep Learning

Welcome to the Dragon Ball Z (DBZ) Character Classification project! This repository contains a detailed implementation of a cutting-edge deep learning model aimed at classifying characters from the popular anime series Dragon Ball Z. By leveraging the power of transfer learning and advanced neural network architectures, this project achieves remarkable accuracy in identifying and categorizing DBZ characters.

---

## Project Overview

Anime character classification is a challenging task due to the intricate details and varying visual styles of characters. This project tackles the problem by utilizing state-of-the-art techniques in image recognition and deep learning. Specifically, we trained a Convolutional Neural Network (CNN) based on the EfficientNet B2 architecture, fine-tuned on a curated dataset of Dragon Ball Z characters.

This model demonstrates the potential of AI in niche applications such as content recognition, anime analytics, and fan-based projects.

---

## Key Features

- **Advanced Deep Learning Architecture**: Used the EfficientNet B2 model for its balance of efficiency and performance in image classification tasks.
- **Custom Dataset**: Trained on a custom-built dataset featuring various DBZ characters, including Goku, Vegeta, Piccolo, Trunks, and others.
- **Data Preprocessing**: Applied preprocessing techniques such as resizing, normalization, and augmentation to optimize the dataset for deep learning.
- **Transfer Learning**: Leveraged a pre-trained EfficientNet model and fine-tuned it for the specific task of DBZ character classification.
- **High Accuracy**: Achieved robust performance metrics, showcasing the model's effectiveness in handling real-world data.

---

## Technologies and Skills Utilized

- **Deep Learning Frameworks**: TensorFlow, Keras
- **Programming Language**: Python
- **Key Libraries**: NumPy, Pandas, Matplotlib, OpenCV
- **Techniques**: 
  - Convolutional Neural Networks (CNNs)
  - Transfer Learning
  - Data Augmentation
  - Visualization of Results

---

## Dataset Details

The dataset comprises high-quality images of Dragon Ball Z characters, divided into training and testing sets. Each image is categorized by character, with the dataset preprocessed to meet the input size requirements of the EfficientNet B2 model (224x224 pixels).

Steps in Data Preparation:
1. **Resizing**: Images were resized to 224x224 pixels.
2. **Normalization**: Pixel values were normalized for faster convergence during training.
3. **Augmentation**: Techniques such as rotation, flipping, and scaling were applied to improve generalization.

---

## Model Architecture

### EfficientNet B2
EfficientNet B2 was chosen for its ability to deliver superior accuracy with optimized computational resources. Key architectural elements include:
- **CNN Backbone**: Extracts hierarchical features from images.
- **Transfer Learning**: Fine-tunes a pre-trained model to adapt to the anime domain.
- **Classification Head**: Outputs probabilities for each DBZ character class.

### Custom Enhancements
- Positional Embeddings: Integrated to preserve spatial structure in images.
- Layer Normalization: Used to ensure consistent feature scaling across the model.
- Attention Mechanisms: Incorporated to highlight important image regions, improving recognition accuracy.

---

## Results

The trained model achieves outstanding accuracy in classifying DBZ characters, with robust performance on both training and testing datasets. This validates the effectiveness of the preprocessing pipeline, EfficientNet B2 architecture, and transfer learning strategy.

Applications:
- Automated anime character tagging for fan art and media.
- Content moderation and sorting for anime-based platforms.
- Academic research in image recognition and domain-specific AI.

---

## Installation and Usage

### Prerequisites
- Python 3.7+
- GPU-enabled environment for faster training (recommended).

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dbz-character-classification.git
   cd dbz-character-classification

2. Install Dependencies
   pip install -r requirements.txt

3. Training the Model
    To train the model on your machine, run:
    python train_model.py

4. Predicting Characters
    Use the trained model to classify a character in an image:
    python predict.py --image_path /path/to/your/image.jpg

5. Visualizing Results
    Explore the model's performance metrics and visualizations using the provided Jupyter notebooks:

    notebooks/model_training.ipynb: Training process and insights.
    notebooks/prediction_visualization.ipynb: Visualization of predictions.
