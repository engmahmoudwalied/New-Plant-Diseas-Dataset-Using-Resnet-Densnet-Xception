# **Project Documentation**

## **Overview**

In this project, we trained a model using the dataset available on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). The dataset comprises **38 distinct classes**, and we employed the following models to evaluate its performance:

1. **Xception** (fine-tuning)
2. **DenseNet** (fine-tuning)
3. **ResNet** (trained from scratch)

We utilized **Google Colab** as our virtual environment to leverage its computational power, significantly enhancing the speed and efficiency of the training and evaluation processes.

---

## **ResNet Explanation**

**Reference:** [ResNet Paper (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)

### **Introduction**

ResNet, short for **Residual Network**, was designed to address the vanishing gradient problem encountered in very deep neural networks. Its innovative **residual block** architecture introduces **skip connections**, allowing the network to bypass one or more layers during training.

---

### **Residual Block**

The residual block is the fundamental unit of ResNet. It works by:

- Connecting the activations of a layer to further layers through **skip connections**, which bypass intermediate layers.
- Learning the **residual mapping** (the difference between the input and output) instead of the direct mapping.

Each residual block typically contains:

1. A **1x1 convolution** to reduce dimensionality.
2. A **3x3 convolution** for feature extraction.
3. Another **1x1 convolution** to restore dimensionality.

---

### **ResNet Architecture**

The full architecture consists of:

1. An initial convolutional layer with 64 filters, followed by batch normalization and max pooling.
2. Four layers, each containing multiple **residual blocks** with increasing filter sizes.
3. Adaptive average pooling for fixed-size outputs.
4. A fully connected layer for classification into 38 classes.

---

### **ResNet Variants**

ResNet has several versions depending on the depth:

- **ResNet50:** [3, 4, 6, 3] residual blocks
- **ResNet101:** [3, 4, 23, 3] residual blocks
- **ResNet152:** [3, 8, 36, 3] residual blocks

---

## **Advantages**

1. **Skip Connections:** They prevent performance degradation caused by vanishing gradients in deep networks.
2. **Efficient Training:** ResNet allows for the training of very deep networks (up to 1000 layers) without sacrificing performance.

---

## **Training Details**

- **Dataset:**
    - Training Images: 70,295
    - Validation Images: 17,572
    - Classes: 38
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam

---

## Densenet

---

**Introduction**

DenseNet, or Densely Connected Convolutional Network, represents a significant advancement in the field of convolutional neural networks (CNNs). With its innovative connectivity pattern, DenseNet overcomes several challenges faced by traditional CNN architectures, such as vanishing gradients, redundant feature learning, and inefficient parameter usage. This document provides a comprehensive overview of DenseNet’s architecture, characteristics, variants, advantages, and limitations.

---

**Key Characteristics of DenseNet**

1. **Alleviated Vanishing Gradient Problem**:
    - Dense connections ensure gradients flow directly to earlier layers, mitigating the vanishing gradient issue and facilitating the training of deeper networks.
2. **Improved Feature Propagation**:
    - Each layer receives input from all preceding layers, ensuring direct access to gradients and original inputs. This promotes better feature propagation throughout the network.
3. **Feature Reuse**:
    - By concatenating feature maps from previous layers, DenseNet minimizes redundancy and promotes the reuse of learned features, improving efficiency.
4. **Reduced Parameters**:
    - DenseNet avoids relearning redundant features, resulting in a parameter-efficient architecture despite its extensive connections.

---

**Architecture**

DenseNet’s architecture connects each layer to every other layer in a feed-forward manner. Unlike traditional CNNs that only connect consecutive layers, DenseNet ensures all layers within a block are interconnected.

- **Mathematical Representation**:
    - For a DenseNet with layers, there are direct connections, enhancing information flow and improving gradient propagation.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/1dd993ef-8811-426a-885b-116315f7884b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/ef227b4e-e790-4290-b3c9-fe7645efb67d/image.png)

---

**DenseNet Variants**

1. **DenseNet-121**:
    - Layers: 121
    - Balanced trade-off between computational efficiency and accuracy.
    - Suitable for tasks requiring moderate computational resources.
2. **DenseNet-169**:
    - Layers: 169
    - Offers deeper feature extraction, ideal for complex datasets demanding higher accuracy.
3. **DenseNet-201 and DenseNet-264**:
    - Layers: 201 and 264, respectively.
    - Designed for highly complex tasks requiring extensive feature representation.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/36654511-7632-4178-aa38-d98eb288d6f4/image.png)

---

**Advantages**

1. **Reduced Vanishing Gradient Problem**:
    - Direct connections improve gradient flow, enabling the training of very deep networks.
2. **Feature Reuse**:
    - Access to all preceding layers' feature maps enhances learning efficiency and reduces redundancy.
3. **Fewer Parameters**:
    - DenseNet’s efficient feature reuse minimizes the number of parameters required, compared to traditional CNNs of similar depth.
4. **Improved Accuracy**:
    - DenseNet has demonstrated high performance on benchmarks like ImageNet and CIFAR datasets.

---

**Limitations**

1. **High Memory Consumption**:
    - Dense connections require substantial memory for storing feature maps, limiting its practicality on devices with constrained memory.
2. **Computational Complexity**:
    - Increased connectivity raises computational demands, resulting in longer training times and higher costs.
3. **Risk of Overfitting**:
    - While DenseNet reduces overfitting through feature reuse, it remains susceptible without proper regularization or sufficient training data.

---

**Training Details**

- **Dataset**:
    - Training Images: 70,295
    - Validation Images: 17,572
    - Classes: 38
- **Hyperparameters**:
    - **Loss Function**: categorical_crossentropy
    - **Optimizer**: Adam
    - **Image Size**: 256x256 pixels
    - **Batch Size**: 64
    - **Epochs**: 5

---

**Implementation Highlights**

DenseNet implementations commonly utilize frameworks like TensorFlow or PyTorch. Key implementation steps include dataset preparation, preprocessing, model construction, and training.

1. **Dataset Preparation**:
    - Load images from directories and resize them to the target size.
    - Shuffle the dataset to avoid ordering bias and create batches for efficient processing.
2. **Model Construction**:
    - Define DenseNet blocks with dense connections.
    - Ensure feature maps from all preceding layers are concatenated.
3. **Training**:
    - Optimize the network using Adam optimizer and minimize categorical crossentropy loss.
    - Regularize the model to prevent overfitting, especially on small datasets.

---

**References**

1. “Densely Connected Convolutional Networks.” *arXiv*, 1608.06993v5.
2. GeeksforGeeks: DenseNet Explained.

---

## Xception: Extreme Inception Architecture

## Introduction

Xception, or Extreme Inception, is a deep learning model developed by François Chollet at Google. It enhances the Inception architecture by replacing its inception modules with depthwise separable convolution layers, featuring a total of 36 convolutional layers.

Xception demonstrates slight performance improvements over Inception V3 on the ImageNet dataset but significantly excels on larger datasets, such as one with 350 million images. This scalability makes it highly effective for large-scale image classification tasks.

---

## Key Concepts of Inception and Xception

### The Inception Model

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/7fb2cf44-8e5b-42a3-b8f1-761cc4f9a685/image.png)

Standard convolution layers learn filters in three dimensions: width, height (spatial correlation), and channels (cross-channel correlation). Inception modules improve this by using parallel filters of different sizes (e.g., 1×1, 3×3, 5×5), efficiently dividing spatial and cross-channel tasks.

### Xception Model

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/ef3bd7b0-531b-46e5-b1d0-4a16dd9d7e14/image.png)

Xception takes this concept further by entirely decoupling cross-channel and spatial correlations using depthwise separable convolutions. This separation gives the model its name, "Extreme Inception."

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/3bae6395-2cf3-4918-a754-b2868c2ee866/image.png)

---

## Xception Architecture

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/533b8d19-fce2-4e5a-a988-878f5514a140/image.png)

Xception is divided into three main parts:

### 1. Entry Flow

- **Input:** 299×299 RGB images.
- **Layers:**
    - Two 3×3 convolution layers (32 and 64 filters) with strides of 2×2, followed by ReLU activation.
    - Depthwise separable convolution layers combined with 1×1 convolutions.
    - Max pooling (3×3, stride=2) reduces spatial dimensions.
- **Purpose:** Extract low-level features from the input.

### 2. Middle Flow

- Repeated eight times.
- **Each repetition includes:**
    - Depthwise separable convolution layers (728 filters, 3×3 kernels).
    - ReLU activation.
- **Purpose:** Extract higher-level features progressively.

### 3. Exit Flow

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/69f6deb4-82ab-443f-94dd-dad6a96a49fb/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/45beea02-c606-4c5d-81b9-70c0cad11174/image.png)

- **Layers:**
    - Depthwise separable convolutions (728, 1024, 1536, and 2048 filters, 3×3 kernels).
    - Global average pooling to summarize feature maps.
    - Fully connected layer with logistic regression for classification.
- **Purpose:** Aggregate features and perform final classification.

---

## Detailed Workflow

### 1. Input Images

- Images represent various patterns or objects and are resized to 299×299 pixels.

### 2. Entry Flow

- **Techniques Used:**
    - Separable Convolutions: Decouple spatial and channel correlations.
    - ReLU Activation: Introduce non-linearity.
    - Max Pooling: Downsample feature maps.
- **Output:** Initial feature maps passed to the middle flow.

### 3. Middle Flow

- Stacks multiple separable convolution layers to refine features.
- Retains spatial details while enhancing pattern recognition.
- **Output:** Refined feature maps for the exit flow.

### 4. Exit Flow

- **Processes:**
    - Conv 1×1 to reduce depth without losing spatial dimensions.
    - Depthwise separable convolutions with ReLU and max pooling.
    - Fully connected layers flatten feature maps for classification.
- **Output:** Probability distribution over target classes via the softmax layer.

### 5. Modified Fully Connected Layers (FCL)

- Adapted for specific tasks (e.g., 40-class classification).
- Includes multiple dense layers with ReLU and a final softmax activation.
- **Output:** Predicted label with the highest probability.

---

## Key Components

- **Separable Convolutions (SC):** Lightweight operations for efficient computation.
- **ReLU Activation:** Adds non-linearity.
- **Max Pooling (MP):** Reduces spatial dimensions while preserving dominant features.
- **Fully Connected Layers (FCL):** Maps features to output classes.

---

## Advantages of Xception

1. **Efficient Use of Parameters:**
    - Depthwise separable convolutions reduce trainable parameters.
2. **Improved Feature Learning:**
    - Separately learns spatial and channel-wise features.
3. **Computational Efficiency:**
    - Achieves high performance with reduced computational costs.
4. **Scalability:**
    - Effectively handles larger datasets and complex tasks.

---

## Disadvantages of Xception

1. **Resource Intensive:**
    - High GPU/TPU requirements for training and inference.
2. **Overfitting Risk:**
    - Potential overfitting on small datasets without regularization or augmentation.

---

## Implementation Explanation

### 1. Data Preparation

- **Training Transforms:**
    - Resize to 224×224 pixels.
    - Apply random horizontal flips and rotations.
    - Normalize using ImageNet statistics.
- **Validation Transforms:**
    - Resize and normalize.
- **Dataset Loading:**
    - Use `ImageFolder` for organized datasets.
    - Prepare data batches with `DataLoader`.

### 2. Model Setup

- **Initialization:**
    - Load pre-trained Xception model with `timm.create_model`.
- **Layer Adjustment:**
    - Freeze pre-trained layers and replace the classification layer for new tasks.

### 3. Loss, Optimizer, and Scheduler

- **Loss Function:**
    - CrossEntropyLoss for multi-class classification.
- **Optimizer:**
    - Adam optimizer with a learning rate of 0.001.
- **Scheduler:**
    - Adjusts learning rate every 10 epochs.

### 4. Training Process

- Alternate between training and validation phases each epoch.
- Track metrics: training/validation loss and accuracy.
- Save the best model based on validation accuracy.

---

## Training Summary

- Model: Pretrained Xception with modified FCL.
- Data Augmentation: Random resizing, flipping, and rotation.
- Fine-Tuning: Optimized only for the new dataset's FCL.
- Evaluation: Monitored loss and accuracy on training/validation sets.

---

## References

1. [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9551957)
2. [Viso.ai](https://viso.ai/deep-learning/xception-model/)

---

## Comparison Between 3 models

| Compare on | Resnet | Densnet | Xception |
| --- | --- | --- | --- |
| Over all accuracy | 97% | 97.6% | 95% |

### Roc and Auc Graphs

**Xception**

![roc_auc_Xception.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/8e7291fb-ed95-4eb7-95ad-8013ecfad578/roc_auc_Xception.png)

**Resnet**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/c53b5048-113e-4399-90be-6841caa9dfc7/image.png)

**Densenet**

![RocForDensnet.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/e3f60441-faec-4ceb-9986-396a4759f08e/RocForDensnet.png)

---

### Confiusion Matrix

**Xception**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/5647d116-d71c-4084-b13a-fcf0d625fb16/image.png)

**Densenet**

![CMDens.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/b3dc2590-c93a-4307-8fbd-0c8698d980dd/CMDens.png)

**Resnet**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/57410651-ca59-4f6a-a197-e6b405e3bd20/image.png)

---

### Train and validation Metricis

**DensNet**

![WhatsApp Image 2024-12-19 at 08.32.00_9ac76c17.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/fdc69037-aa00-4b08-adc9-ce969dae301c/WhatsApp_Image_2024-12-19_at_08.32.00_9ac76c17.jpg)

**ResNet**

![newplot.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/976cbad2-fd85-4e96-b637-ae9da33242a8/0b917260-b781-47e4-916f-3c71b5f377b9/newplot.png)

**Xception**

### Report

**Xception**

| **Category** | **Precision** | **Recall** | **F1-Score** | **Support** |
| --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.95 | 0.95 | 0.95 | 504 |
| Apple___Black_rot | 0.96 | 0.99 | 0.97 | 497 |
| Apple___Cedar_apple_rust | 0.97 | 0.95 | 0.96 | 440 |
| Apple___healthy | 0.93 | 0.97 | 0.95 | 502 |
| Blueberry___healthy | 0.95 | 1.00 | 0.97 | 454 |
| Cherry_(including_sour)___Powdery_mildew | 1.00 | 0.98 | 0.99 | 421 |
| Cherry_(including_sour)___healthy | 0.99 | 0.99 | 0.99 | 456 |
| Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot | 0.96 | 0.85 | 0.90 | 410 |
| Corn_(maize)__*Common_rust* | 1.00 | 0.99 | 0.99 | 477 |
| Corn_(maize)___Northern_Leaf_Blight | 0.87 | 0.97 | 0.92 | 477 |
| Corn_(maize)___healthy | 0.99 | 1.00 | 0.99 | 465 |
| Grape___Black_rot | 0.97 | 0.97 | 0.97 | 472 |
| Grape___Esca_(Black_Measles) | 0.97 | 0.98 | 0.98 | 480 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.00 | 0.98 | 0.99 | 430 |
| Grape___healthy | 1.00 | 0.99 | 1.00 | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 0.99 | 1.00 | 1.00 | 503 |
| Peach___Bacterial_spot | 0.98 | 0.96 | 0.97 | 459 |
| Peach___healthy | 0.92 | 0.98 | 0.95 | 432 |
| Pepper,_bell___Bacterial_spot | 0.97 | 0.94 | 0.95 | 478 |
| Pepper,_bell___healthy | 0.95 | 0.95 | 0.95 | 497 |
| Potato___Early_blight | 1.00 | 0.92 | 0.96 | 485 |
| Potato___Late_blight | 0.88 | 0.96 | 0.92 | 485 |
| Potato___healthy | 0.98 | 0.92 | 0.95 | 456 |
| Raspberry___healthy | 0.98 | 0.99 | 0.99 | 445 |
| Soybean___healthy | 0.98 | 0.97 | 0.98 | 505 |
| Squash___Powdery_mildew | 1.00 | 1.00 | 1.00 | 434 |
| Strawberry___Leaf_scorch | 0.99 | 0.98 | 0.99 | 444 |
| Strawberry___healthy | 0.97 | 1.00 | 0.98 | 456 |
| Tomato___Bacterial_spot | 0.91 | 0.94 | 0.92 | 425 |
| Tomato___Early_blight | 0.86 | 0.78 | 0.82 | 480 |
| Tomato___Late_blight | 0.84 | 0.84 | 0.84 | 463 |
| Tomato___Leaf_Mold | 0.91 | 0.91 | 0.91 | 470 |
| Tomato___Septoria_leaf_spot | 0.77 | 0.87 | 0.82 | 436 |
| Tomato___Spider_mites_Two-spotted_spider_mite | 0.87 | 0.84 | 0.85 | 435 |
| Tomato___Target_Spot | 0.88 | 0.77 | 0.82 | 457 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.97 | 0.95 | 0.96 | 490 |
| Tomato___Tomato_mosaic_virus | 0.94 | 0.95 | 0.94 | 448 |
| Tomato___healthy | 0.94 | 0.97 | 0.95 | 481 |
| **Accuracy** |  |  | **0.95** | 17572 |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 17572 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 17572 |

**Resnet**

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
| --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.98 | 0.98 | 0.98 | 504 |
| Apple___Black_rot | 0.99 | 0.99 | 0.99 | 497 |
| Apple___Cedar_apple_rust | 0.95 | 0.99 | 0.97 | 440 |
| Apple___healthy | 0.93 | 0.99 | 0.96 | 502 |
| Blueberry___healthy | 1.00 | 0.93 | 0.96 | 454 |
| Cherry_(including_sour)___Powdery_mildew | 0.98 | 0.99 | 0.98 | 421 |
| Cherry_(including_sour)___healthy | 0.99 | 0.99 | 0.99 | 456 |
| Corn_(maize)___Cercospora_leaf_spot | 0.97 | 0.88 | 0.92 | 410 |
| Corn_(maize)__*Common_rust* | 1.00 | 0.99 | 0.99 | 477 |
| Corn_(maize)___Northern_Leaf_Blight | 0.90 | 0.99 | 0.94 | 477 |
| Corn_(maize)___healthy | 1.00 | 0.99 | 1.00 | 465 |
| Grape___Black_rot | 0.94 | 0.96 | 0.95 | 472 |
| Grape___Esca_(Black_Measles) | 0.93 | 0.95 | 0.94 | 480 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 0.99 | 0.99 | 0.99 | 430 |
| Grape___healthy | 0.99 | 0.99 | 0.99 | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 0.99 | 0.98 | 0.99 | 503 |
| Peach___Bacterial_spot | 0.99 | 0.98 | 0.99 | 459 |
| Peach___healthy | 0.98 | 0.99 | 0.99 | 432 |
| Pepper,_bell___Bacterial_spot | 0.99 | 0.97 | 0.98 | 478 |
| Pepper,_bell___healthy | 0.94 | 0.95 | 0.94 | 497 |
| Potato___Early_blight | 0.92 | 1.00 | 0.96 | 485 |
| Potato___Late_blight | 0.86 | 1.00 | 0.93 | 485 |
| Potato___healthy | 0.98 | 0.97 | 0.98 | 456 |
| Raspberry___healthy | 0.99 | 0.98 | 0.99 | 445 |
| Soybean___healthy | 0.97 | 0.98 | 0.97 | 505 |
| Squash___Powdery_mildew | 0.99 | 1.00 | 0.99 | 434 |
| Strawberry___Leaf_scorch | 0.99 | 0.99 | 0.99 | 444 |
| Strawberry___healthy | 0.99 | 1.00 | 1.00 | 456 |
| Tomato___Bacterial_spot | 0.99 | 0.92 | 0.95 | 425 |
| Tomato___Early_blight | 0.94 | 0.91 | 0.92 | 480 |
| Tomato___Late_blight | 0.99 | 0.78 | 0.87 | 463 |
| Tomato___Leaf_Mold | 1.00 | 0.94 | 0.97 | 470 |
| Tomato___Septoria_leaf_spot | 0.94 | 0.93 | 0.93 | 436 |
| Tomato___Spider_mites | 0.99 | 0.95 | 0.97 | 435 |
| Tomato___Target_Spot | 0.94 | 0.96 | 0.95 | 457 |
| Tomato___Yellow_Leaf_Curl_Virus | 0.93 | 1.00 | 0.96 | 490 |
| Tomato___Tomato_mosaic_virus | 0.98 | 0.99 | 0.99 | 448 |
| Tomato___healthy | 1.00 | 0.98 | 0.99 | 481 |
| **Accuracy** | **0.97** | **0.97** | **0.97** | **17572** |
| **Macro Avg** | **0.97** | **0.97** | **0.97** | **17572** |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **17572** |

**Densenet**

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
| --- | --- | --- | --- | --- |
| Apple___Apple_scab | 0.98 | 1.00 | 0.99 | 218 |
| Apple___Black_rot | 1.00 | 0.99 | 0.99 | 210 |
| Apple___Cedar_apple_rust | 1.00 | 0.93 | 0.97 | 164 |
| Apple___healthy | 1.00 | 0.99 | 1.00 | 188 |
| Blueberry___healthy | 0.99 | 0.99 | 0.99 | 182 |
| Cherry_(including_sour)_Powdery_mildew | 1.00 | 0.99 | 1.00 | 167 |
| Cherry_(including_sour)_healthy | 0.97 | 0.99 | 0.98 | 174 |
| Corn_(maize)_Cercospora_leaf_spot | 0.99 | 0.99 | 0.99 | 150 |
| Corn_(maize)Common_rust | 1.00 | 1.00 | 1.00 | 212 |
| Corn_(maize)_Northern_Leaf_Blight | 0.99 | 0.97 | 0.98 | 179 |
| Corn_(maize)_healthy | 1.00 | 0.98 | 0.99 | 186 |
| Grape___Black_rot | 1.00 | 1.00 | 1.00 | 197 |
| Grape__Esca(Black_Measles) | 1.00 | 1.00 | 1.00 | 211 |
| Grape__Leaf_blight(Isariopsis_Leaf_Spot) | 1.00 | 1.00 | 1.00 | 166 |
| Grape___healthy | 1.00 | 0.99 | 0.99 | 171 |
| Orange__Haunglongbing(Citrus_greening) | 1.00 | 1.00 | 1.00 | 204 |
| Peach___Bacterial_spot | 0.99 | 1.00 | 1.00 | 185 |
| Peach___healthy | 0.98 | 1.00 | 0.99 | 185 |
| Pepper,bell__Bacterial_spot | 0.99 | 0.97 | 0.98 | 175 |
| Pepper,bell__healthy | 0.89 | 1.00 | 0.94 | 161 |
| Potato___Early_blight | 0.98 | 1.00 | 0.99 | 183 |
| Potato___Late_blight | 1.00 | 0.90 | 0.95 | 184 |
| Potato___healthy | 0.98 | 0.96 | 0.97 | 188 |
| Raspberry___healthy | 0.99 | 0.77 | 0.87 | 166 |
| Soybean___healthy | 1.00 | 0.96 | 0.98 | 216 |
| Squash___Powdery_mildew | 0.98 | 1.00 | 0.99 | 168 |
| Strawberry___Leaf_scorch | 0.99 | 1.00 | 0.99 | 171 |
| Strawberry___healthy | 0.78 | 1.00 | 0.88 | 173 |
| Tomato___Bacterial_spot | 1.00 | 0.91 | 0.95 | 171 |
| Tomato___Early_blight | 0.98 | 0.95 | 0.97 | 191 |
| Tomato___Late_blight | 0.96 | 0.97 | 0.96 | 211 |
| Tomato___Leaf_Mold | 1.00 | 0.99 | 1.00 | 180 |
| Tomato___Septoria_leaf_spot | 0.91 | 1.00 | 0.95 | 173 |
| Tomato___Spider_mites | 0.99 | 0.99 | 0.99 | 181 |
| Tomato___Target_Spot | 0.96 | 0.98 | 0.97 | 164 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.97 | 1.00 | 0.99 | 212 |
| Tomato___Tomato_mosaic_virus | 1.00 | 0.99 | 1.00 | 166 |
| Tomato___healthy | 0.99 | 1.00 | 0.99 | 193 |
| **Accuracy** | **0.98** | **0.98** | **0.98** | **6976** |
| **Macro Avg** | **0.98** | **0.98** | **0.98** | **6976** |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **6976** |
