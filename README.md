# Synthetic Medical Image Generation for Prostate Cancer Diagnosis Using DCGAN

## Overview
This repository contains the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** for generating synthetic prostate cancer (PCa) images to address data imbalance and scarcity in medical imaging datasets. The model leverages the **Cancer-Net-PCa dataset** to produce high-resolution synthetic images, which can enhance the performance of deep learning models for prostate cancer diagnosis. The proposed DCGAN consists of a **generator** and a **discriminator**, with a total of **51,859,971 parameters**. The model was trained over **50 epochs**, and evaluation metrics include **discriminator and generator losses**, **classification accuracies**, **Fréchet Inception Distance (FID)**, **precision**, and **recall**.

---

## Key Features
- **Synthetic Image Generation**: Generates high-resolution synthetic prostate cancer images to address data imbalance and scarcity.
- **DCGAN Architecture**: Utilizes a generator and discriminator with **51,859,971 parameters**.
- **Evaluation Metrics**: Includes **FID**, **precision**, **recall**, and **classification accuracies** to assess image quality and model performance.
- **Stable Training**: Demonstrates balanced adversarial learning with **discriminator loss of 0.6953** and **generator loss of 0.6526**.
- **Moderate Perceptual Quality**: Achieves an **FID score of 34.89**, indicating moderate perceptual quality of generated images.

---

## Dataset
The **Cancer-Net-PCa dataset**, sourced from [Kaggle](https://www.kaggle.com/datasets/cancer-net-pca), contains **8,412 instances** across **10 classes** of prostate cancer images. The dataset is used to train the DCGAN for synthetic image generation.

### Dataset Classes
The dataset includes the following classes:
- Class 1: [Class Name]
- Class 2: [Class Name]
- ...
- Class 10: [Class Name]

---

## Methodology
### 1. **Generator Architecture**
The generator is designed to produce high-resolution synthetic images. It consists of **25 layers** and **51,868,035 parameters**. Key components include:
- **Input Layer**: Accepts a latent vector of size `NOISE_DIM = 150`.
- **Dense Layer**: Reshapes the input to a higher-dimensional tensor.
- **Upsampling Layers**: Progressively upsample the image to resolutions of **16x16**, **32x32**, **64x64**, and **128x128**.
- **Output Layer**: Uses a **3x3 convolution** with a **tanh activation function** to produce the final RGB image.

### 2. **Discriminator Architecture**
The discriminator is a **CNN** designed to classify images as real or fake. It consists of **11 layers** and **126,241 parameters**. Key components include:
- **Input Layer**: Accepts images of size **128x128x3**.
- **Convolutional Layers**: Downsample the image using **spectral normalization** and **LeakyReLU activation**.
- **Output Layer**: Uses a **sigmoid activation function** to classify images as real or fake.

### 3. **Training**
The model was trained over **50 epochs** with the following hyperparameters:
- **Batch Size**: 64.
- **Learning Rate**: 0.0001 (discriminator), 0.0006 (generator).
- **Label Smoothing**: 0.8–1.0 for real images, 0.1 for fake images.
- **Steps per Epoch**: 132.

### 4. **Evaluation Metrics**
- **Discriminator and Generator Losses**: Binary cross-entropy losses to measure adversarial learning.
- **Classification Accuracies**: Real and fake image accuracies to assess training balance.
- **Fréchet Inception Distance (FID)**: Measures perceptual quality and diversity of generated images.
- **Precision and Recall**: Evaluates the realism and diversity of generated images.

---

## Results
### Performance Metrics
| Metric                  | Value at Epoch 15 | Trend Over 14 Epochs          | Interpretation                                                                 |
|-------------------------|-------------------|-------------------------------|--------------------------------------------------------------------------------|
| Discriminator Loss      | 0.6953           | Stable competition            | Indicates balanced adversarial learning.                                       |
| Generator Loss          | 0.6526           | Stable competition            | Reflects effective deception of the discriminator.                             |
| Real Image Accuracy     | 52.16%           | Decreased from ~60% to 52.16% | Converges toward 50%, indicating a balanced discriminator.                     |
| Fake Image Accuracy     | 47.63%           | Increased from ~40% to 47.63% | Approaches 50%, suggesting the generator is improving.                         |
| FID Score               | 34.89            | Decreased from ~400 to 35–50  | Significant improvement in perceptual quality, but still moderate.             |
| Precision               | 50.00%           | Sharp drop, then stabilized   | Indicates moderate quality of generated images; half are realistic.            |
| Recall                  | 100.00%          | Immediate rise to 100%        | Suggests the generator captures all real-like features, but may lack diversity. |

### Generated Images
Examples of synthetic images generated by the DCGAN are shown below:
![Generated Images](path/to/generated_images.png)

---

## Usage
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DCGAN-Prostate-Cancer.git
   cd DCGAN-Prostate-Cancer
