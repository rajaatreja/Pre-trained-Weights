# Weights Directory

This directory contains pre-trained and trained weight files for various neural network models used in the SwinCheX project. These weights are essential for reproducing the results of our experiments and for fine-tuning the models for specific tasks.

## Overview

The weights stored in this directory are designed for different neural network architectures, each tailored for specific purposes such as classification, generation, and verification. Below is a detailed description of each weight file, its purpose, and how it can be used.

## Weight Files

### 1. **pretrained_CheXNet_classifier.pth**

- **Description**:
  - This file contains the pre-trained weights for the CheXNet classifier model.
  - CheXNet is a deep learning model trained specifically for chest X-ray classification tasks.

- **Purpose**:
  - Used for detecting multiple thoracic pathologies from chest X-ray images.
  - Provides a strong baseline for transfer learning on medical imaging tasks.

- **Usage**:
  - Load the weights into a compatible CheXNet model architecture.
  - Fine-tune the model on specific datasets if needed.

- **Reference**:
  - For more details on the CheXNet model, see the [CheXNet repository](https://github.com/kaipackhaeuser/PriCheXy-Net).

### 2. **pretrained_generator_prichexy_net.pth**

- **Description**:
  - Contains pre-trained weights for the PriCheXy-Net generator model.
  - PriCheXy-Net is used for generating synthetic chest X-ray images.

- **Purpose**:
  - Useful for data augmentation and generating high-quality synthetic images.
  - Enhances model robustness by providing additional training samples.

- **Usage**:
  - Load into a generator network compatible with PriCheXy-Net architecture.
  - Use the generator to create synthetic images for training or evaluation.

- **Reference**:
  - For more details on the PriCheXy-Net model, see the [PriCheXy-Net repository](https://github.com/kaipackhaeuser/PriCheXy-Net).

### 3. **pretrained_SwinCheX_classifier**

- **Description**:
  - Pre-trained weights for the SwinCheX classifier.
  - SwinCheX uses a Swin Transformer architecture adapted for chest X-ray classification.

- **Purpose**:
  - Achieves state-of-the-art performance on chest X-ray classification tasks.
  - Leverages hierarchical features and shifted windows for efficient computation.

- **Usage**:
  - Load into a SwinCheX classifier model.
  - Suitable for direct evaluation or further fine-tuning on medical datasets.

### 4. **pretrained_verification_SNN_model.pth**

- **Description**:
  - This file contains pre-trained weights for a Siamese Neural Network (SNN) used for verification tasks.
  - SNNs are designed to verify if two input images are similar or belong to the same class.

- **Purpose**:
  - Used in verification tasks such as image matching or similarity detection.
  - Suitable for applications requiring verification of chest X-ray images.

- **Usage**:
  - Load into a Siamese network architecture compatible with SNN tasks.
  - Apply to verification problems in medical imaging domains.

### 5. **trained_generator_PriCheXy_Net_mu_0.01.pth**

- **Description**:
  - Trained weights for the PriCheXy-Net generator with a regularization parameter `mu` set to 0.01.
  - These weights are fine-tuned on specific datasets for improved performance.

- **Purpose**:
  - Generates high-quality synthetic images with enhanced realism.
  - Useful for expanding training datasets and improving model generalization.

- **Usage**:
  - Load into PriCheXy-Net generator with specific settings.
  - Utilize for creating synthetic images with controlled parameters.

- **Reference**:
  - For more details on the PriCheXy-Net model, see the [PriCheXy-Net repository](https://github.com/kaipackhaeuser/PriCheXy-Net).

### 6. **trained_generator_PriSwin_Dis_mu_0.01.pth**

- **Description**:
  - Trained weights for a generator model based on the PriSwin architecture.
  - Includes a regularization parameter `mu` set to 0.01 for fine-tuning.

- **Purpose**:
  - Designed for generating synthetic images while maintaining discriminator properties.
  - Enhances model stability and diversity of generated samples.

- **Usage**:
  - Load into a PriSwin generator network.
  - Employ for generating diverse and realistic synthetic data.

## How to Use

To use these weights in your projects, follow these steps:

1. **Select the appropriate weight file** based on your task requirements.
2. **Load the weights** into the corresponding model architecture using a deep learning framework such as PyTorch or TensorFlow.
3. **Fine-tune or evaluate** the model on your specific dataset or task.
4. **Adjust hyperparameters** and settings to optimize model performance for your application.

## Notes

- Ensure that your environment and model architectures are compatible with the weight files.
- Consider fine-tuning pre-trained models to better suit your specific tasks.
- If using for research or publication, please acknowledge the source of the weights and cite relevant papers.

## Acknowledgments

- **PriCheXy-Net and CheXNet**: Special thanks to the developers of the [PriCheXy-Net](https://github.com/kaipackhaeuser/PriCheXy-Net) repository for providing the foundation for the CheXNet and PriCheXy-Net models used in this project. Their contributions to the field of medical imaging have been invaluable in the development of these models.

