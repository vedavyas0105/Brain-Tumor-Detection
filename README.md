# Brain Tumor Segmentation

## Overview
This project leverages deep learning to detect and segment brain tumors from MRI images. Using PyTorch, it implements a robust pipeline for data preprocessing, model training, and evaluation.

## Process Followed
1. **Data Preprocessing:**
   - Loaded MRI images from Kaggle dataset.
   - Applied normalization and augmentation techniques.
   - Split data into training and testing sets.

2. **Model Implementation:**
   - Used a convolutional neural network (CNN) architecture for tumor classification.
   - Integrated segmentation techniques for precise tumor localization.
   - Employed transfer learning to enhance model performance.

3. **Training & Evaluation:**
   - Used cross-entropy loss for classification.
   - Optimized model using Adam optimizer.
   - Tracked accuracy, loss, and segmentation metrics.

4. **Inference & Results:**
   - Tested the model on unseen MRI scans.
   - Generated segmented tumor regions with high accuracy.
   - Visualized results using Matplotlib.

## Features
- MRI-based tumor classification and segmentation.
- Custom dataset handling with PyTorch.
- CNN-based deep learning model.
- Transfer learning for improved performance.
- Efficient data preprocessing and augmentation.
- Visualization of segmentation outputs.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Brain-Tumor-Segmentation.git
   cd Brain-Tumor-Segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter notebook to preprocess the dataset and train the model.
2. Modify hyperparameters as needed for tuning and evaluation.
3. Use the trained model to perform inference on new MRI scans.

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib

## Results
The model successfully segments brain tumors with high accuracy, aiding in medical diagnostics and research.
