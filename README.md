# Overview

This project implements a semantic segmentation model for human oocytes, specifically targeting the segmentation of the zona pellucida. The model utilizes a Fully Convolutional Network (FCN) architecture, which replaces traditional fully connected layers in deeper parts of the network with convolutional layers, allowing for end-to-end pixel-wise classification.

This work was initially developed as part of a thesis, with future plans to expand the model to include additional labels for segmenting other parts of the oocyte.
### Features

- Semantic Segmentation: Focused segmentation of human oocytes, specifically the zona pellucida.
- Fully Convolutional Network: Leverages FCN architecture without fully connected layers, making the model efficient and effective for pixel-level tasks.
- Grayscale Images: The input images are resized to 512x512 and reduced to a single channel (grayscale) for simplicity and efficiency.
    
### Model Architecture

The model is based on a Fully Convolutional Network (FCN) architecture, which eliminates fully connected layers in favor of convolutional layers throughout the network. This design allows the model to produce pixel-wise predictions, which are crucial for tasks like semantic segmentation.

Key features of the architecture:

- Input: 512x512 grayscale images.
- Output: Binary mask indicating the presence of the zona pellucida.
- Loss Function: Typically a pixel-wise cross-entropy loss.
- Optimizer: Common choices include Adam or SGD with momentum.

### Future Work

- Additional Labels: The model will be extended to segment additional structures within the oocyte, requiring the use of multi-class segmentation techniques.
- Model Improvements: Exploring advanced architectures like U-Net or DeepLab for improved performance.
- Data Augmentation: Implementing advanced augmentation techniques to increase dataset variability.

### Dataset

The dataset used for training and evaluation consists of images of human oocytes, with manual annotations for the zona pellucida. Due to privacy and ethical considerations, the dataset cannot be shared publicly. However, instructions for replicating the dataset preparation process can be provided upon request.

### Contributing

If you would like to contribute to the project, please fork the repository and create a pull request with your proposed changes. Any contributions towards improving the model, adding new features, or fixing bugs are welcome!

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Contact

For questions or inquiries, please contact John Dimitriadis at thef4llen1994@gmail.com.
