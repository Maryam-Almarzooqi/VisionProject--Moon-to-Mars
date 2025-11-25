# VisionProject--Moon-to-Mars

This repository contains state-of-the-art semantic segmentation models tailored for Lunar Terrian detection (LuSeg) applied to Mars Science Laboratory (MSL) MCAM images. Each model is optimized to segment Martian terrain into multiple classes, including Soil, Bedrock, Sand, and Big Rock, addressing challenges like class imbalance with specialized augmentation and loss functions. The different models are explored to comapre their performance against an aritecture detecated to Lunar Terrian Detection and evlauting how it performs in Mars Terrian Detection.


## All the models have Common Features such as:

Input: 256 x 256 RGB images from MSL MCAM.

Output: Multi-class semantic segmentation mask.

Class imbalance handled via rare-class-aware augmentation.

Loss: Hybrid Focal Dice Loss combining boundary sensitivity (Focal) with overlap optimization (Dice).

## All of the codes are structured in the following Way:

Imports and Setup
Configuration (Hyperparameters & Dataset Paths)

SECTION 1: Data Loading & Splitting

SECTION 2: Identify Rare Classes

SECTION 3: Data Augmentation (Rare-Class Aware)

SECTION 4: Model Architecture

SECTION 5: Loss Functions & Metrics

SECTION 6: Training

SECTION 7: Final Results Summary

SECTION 8: Visualization – Training Graphs

SECTION 9: Visualization – Image Predictions



## The different models we used are:


### LuSeg:
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/3bcb5a57-a77c-40e5-9649-72ef9889ada2" />


The LuSeg_RGB model is a specialized Encoder-Decoder architecture, leveraging a pre-trained ResNet-50 as its feature-rich encoder, specifically designed for semantic segmentation of Mars rover images (MSL MCAM data). The input is a standard 3-channel RGB image. The encoder extracts features through four residual layers, followed by a U-Net style decoder that progressively upsamples these features. A critical component is the use of skip connections—where encoder outputs (e.g., skip_r_4, skip_r_3) are concatenated with the corresponding upsampled features in the decoder. This fusion, followed by a convolution block (decoder_catX), is crucial for combining deep semantic context with fine-grained spatial detail, allowing the model to accurately delineate terrain boundaries (Soil, Bedrock, Sand, Big Rock) for the planetary science application.



### VGG16:
<img width="1024" height="448" alt="image" src="https://github.com/user-attachments/assets/7a46da95-e940-4545-b141-3ed7c299aa5e" />


The VGG16-based U-Net is a semantic segmentation model that adapts the standard U-Net architecture for Mars Science Laboratory (MSL) images by employing the **VGG16** convolutional network as the powerful feature-extracting **encoder** (contracting path). This encoder, leveraging ImageNet pre-trained weights, extracts multi-scale features, while the first two blocks are frozen and the deeper blocks are fine-tuned for terrain classification. The **decoder** (expanding path) uses **UpSampling2D** and convolutional blocks combined with essential **skip connections**—which fuse high-resolution features from the encoder with upsampled decoder features—to accurately localize objects and preserve spatial details. The model handles the MSL dataset's class imbalance using a unique **rare-class-aware augmentation** strategy (prioritizing augmentation for images with rare terrain types) and is trained with a sophisticated **Hybrid Focal Dice Loss** (combining the boundary-focusing power of Focal Loss with the overlap-optimizing Dice Loss) to produce a five-class terrain segmentation mask.




### ResNet50:

<img width="1024" height="448" alt="image" src="https://github.com/user-attachments/assets/25cfdf3d-722d-489c-b6c7-b194c89fb316" />



The **ResNet50-based U-Net** is a deep semantic segmentation network customized for MSL (Mars Science Laboratory) images. It employs a **ResNet50** backbone as its **encoder** (contracting path), which is pre-trained on ImageNet. The earliest layers of the ResNet50 are frozen to maintain robust general feature extraction, while the later layers (the final `conv5_block` groups) are made **trainable** for domain-specific fine-tuning. The network's **decoder** (expanding path) uses standard U-Net operations: **UpSampling2D** and convolutional blocks (Conv2D, LeakyReLU, BatchNorm), connected via powerful **skip connections** to the corresponding encoder outputs (e.g., `conv1_relu`, `conv2_block3_out`, etc.). This structure efficiently captures both deep semantic information and high-resolution spatial details, culminating in a **Softmax** output layer for multi-class terrain prediction. The model is optimized for class imbalance using a **rare-class-aware augmentation** pipeline and the **Hybrid Focal Dice Loss**.


### DeepLabV3Plus:

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/0588991b-ca6f-4935-bf92-73ec2ceb9e5c" />


The **DeepLabV3+** is a highly effective semantic segmentation architecture tailored for the MSL terrain, which enhances the standard segmentation approach by integrating three components: a **ResNet50** backbone (encoder) for robust feature extraction (with the whole network being trainable for fine-tuning); an **ASPP (Atrous Spatial Pyramid Pooling) module** that processes deep features using dilated convolutions to gather multi-scale context; and a sophisticated **decoder** that refines the upsampled ASPP output by combining it with high-resolution **low-level features** extracted early from the ResNet encoder, ensuring sharp boundaries and accurate localization in the final $256\times256$ classification mask. To handle class imbalance typical of the Martian landscape, the model employs a **rare-class-aware augmentation** strategy and is trained using a specialized **Hybrid Focal Dice Loss**.



### DenseNet:

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/6fb4a081-1cc5-42ac-9958-925e68797ad6" />


The **DenseNet121-based U-Net** is a semantic segmentation architecture designed for MSL images that substitutes the standard U-Net encoder with the pre-trained **DenseNet121** model, renowned for its dense connectivity pattern that maximizes feature reuse and minimizes vanishing gradients, with only the deeper blocks (*dense\_block3* and *dense\_block4*) set to trainable for task-specific fine-tuning. This robust feature extraction path is mirrored by a symmetric **U-Net decoder** that utilizes **bilinear upsampling** and multiple **skip connections**—retrieved from key DenseNet layers—to merge fine-grained spatial information with deep semantic context, resulting in accurate boundary localization in the final $256\times256$ classification mask. To address the class imbalance inherent in Martian terrain data, the model incorporates a **rare-class-aware augmentation** strategy and is trained using a composite **Hybrid Focal Dice Loss**. 


## Conclusion

This repository provides a detailed exploration of multiple semantic segmentation architectures for Martian terrain detection (Soil, Bedrock, Sand, Big Rock) using MSL MCAM images. Each model contributes uniquely to addressing the challenges of multi-class segmentation, class imbalance, and accurate boundary delineation:

LuSeg: A specialized encoder-decoder architecture designed for planetary imagery, leveraging ResNet50 as a backbone with skip connections. Excels in combining deep semantic features with fine-grained spatial details, making it highly effective for delineating terrain boundaries.

VGG16 U-Net: Adapts the standard U-Net with a VGG16 encoder, pre-trained on ImageNet. Its strength lies in multi-scale feature extraction and rare-class prioritization, providing robust segmentation in imbalanced datasets while preserving high-resolution spatial information.

ResNet50 U-Net: Uses a ResNet50 backbone with selective fine-tuning, enabling deep feature extraction and domain-specific adaptation. Skip connections and upsampling maintain spatial detail, making it strong at capturing both coarse and fine terrain structures.

DeepLabV3+: Combines a ResNet50 encoder, ASPP module, and refined decoder. Its main contribution is multi-scale context aggregation, which improves boundary accuracy and feature localization, particularly in complex terrain regions.

DenseNet121 U-Net: Leverages dense connectivity for maximum feature reuse and gradient flow. Its contribution is efficient learning from limited data and improved integration of fine-grained spatial details with semantic features, yielding precise segmentation masks.

In summary, this work demonstrates how each architecture contributes uniquely to the problem of Martian terrain segmentation, providing a framework to balance accuracy, boundary precision, and robustness to class imbalance, and offering insights for future planetary surface analysis.



