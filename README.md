# Tree Species Classification

This repository contains a Python script (`tree_species (1).py`) that implements tree species classification models using deep learning. The models are designed to predict tree species from images, leveraging convolutional neural networks (CNNs) trained on a custom dataset. The script includes data preprocessing, model training, and prediction functionalities.

## Project Overview

The project aims to classify images of various tree species, such as Amla, Asopalav, Babul, Bamboo, and Pipal, among others. The script performs the following tasks:
- Mounts Google Drive to access the dataset.
- Extracts the dataset from a ZIP file (`Tree_Species.zip`).
- Identifies and removes corrupted and duplicate images.
- Analyzes image sizes and removes outliers (very small or large images).
- Trains three different CNN models for classification.
- Predicts tree species for images in a test folder, providing confidence scores.

## Dataset

The dataset (`Tree_Species.zip`) contains images of different tree species, organized into folders named after each species (e.g., `amla`, `asopalav`, `babul`, `bamboo`, `pipal`). Each folder includes multiple images in `.jpg`, `.jpeg`, or `.png` formats. The dataset is stored in Google Drive and extracted to `/content/drive/MyDrive/Datasets/extracted_data/Tree_Species_Dataset`.

### Data Preprocessing
- **Corrupted Images**: Identified using PIL's `Image.verify()` and removed.
- **Duplicates**: Detected using MD5 hashing and removed, keeping only one copy per duplicate set.
- **Outliers**: Images smaller than 150x150 pixels or larger than 1000x2000 pixels are removed.
- **Image Resizing**: All images are resized to 224x224 pixels for model input.
- **Augmentation**: Applied during training, including rotation, zoom, shear, horizontal flip, and brightness adjustments.

## Architecture

The script implements three distinct CNN models for tree species classification, each with a specific architecture:

1. **EfficientNetB0 Model**:
   - **Base Model**: Pre-trained EfficientNetB0 (from `tensorflow.keras.applications`) with ImageNet weights, excluding the top layer.
   - **Input Shape**: 224x224x3 (RGB images).
   - **Layers**:
     - EfficientNetB0 base (frozen during training).
     - GlobalAveragePooling2D to reduce spatial dimensions.
     - Dropout (0.3) for regularization.
     - Dense layer with 128 units (ReLU activation).
     - Dropout (0.3).
     - Dense output layer with units equal to the number of classes (softmax activation).
   - **Training**:
     - Optimizer: Adam (learning rate = 0.001).
     - Loss: Categorical crossentropy.
     - Metrics: Accuracy.
     - Epochs: 10.
   - **Saved As**: `tree_species_model.h5`.

2. **Basic CNN Model**:
   - **Input Shape**: 224x224x3 (RGB images).
   - **Layers**:
     - Conv2D (32 filters, 3x3 kernel, ReLU activation).
     - MaxPooling2D (2x2 pool size).
     - Conv2D (64 filters, 3x3 kernel, ReLU activation).
     - MaxPooling2D (2x2 pool size).
     - Conv2D (128 filters, 3x3 kernel, ReLU activation).
     - MaxPooling2D (2x2 pool size).
     - Flatten layer.
     - Dense layer with 256 units (ReLU activation).
     - Dropout (0.5).
     - Dense output layer with units equal to the number of classes (softmax activation).
   - **Training**:
     - Optimizer: Adam (default learning rate).
     - Loss: Categorical crossentropy.
     - Metrics: Accuracy.
     - Epochs: 10.
   - **Saved As**: `basic_cnn_tree_species.h5`.

3. **MobileNetV2 Model**:
   - **Base Model**: Pre-trained MobileNetV2 (from `tensorflow.keras.applications`) with ImageNet weights, excluding the top layer.
   - **Input Shape**: 224x224x3 (RGB images).
   - **Layers**:
     - MobileNetV2 base (initially frozen, later fine-tuned with last 20 layers unfrozen).
     - GlobalAveragePooling2D.
     - BatchNormalization.
     - Dense layer with 512 units (ReLU activation, L2 regularization with 0.02).
     - Dropout (0.6).
     - Dense output layer with 31 units (softmax activation, for 31 classes).
   - **Training**:
     - **Stage 1**: Frozen base model.
       - Optimizer: Adam (learning rate = 0.001).
       - Epochs: 20.
       - Callbacks: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6), EarlyStopping (patience=10).
     - **Stage 2**: Fine-tune last 20 layers.
       - Optimizer: Adam (learning rate = 1e-5).
       - Epochs: 50.
       - Same callbacks as Stage 1.
     - Loss: Categorical crossentropy.
     - Metrics: Accuracy.
     - Class weights applied to handle class imbalance.
   - **Saved As**: `improved_cnn_model.h5`.

## Requirements

To run the script, you need the following dependencies:
- Python 3.x
- TensorFlow/Keras (for model training and prediction)
- NumPy
- Pandas
- Matplotlib
- PIL (Pillow, for image processing)
- scikit-learn (for class weights)
- Google Colab (optional, for GPU acceleration)
- Google Drive (for dataset storage)

Install the required Python packages using:
```bash
pip install tensorflow numpy pandas matplotlib pillow scikit-learn
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/tree-species-classification.git
   cd tree-species-classification
   ```

2. **Upload the Dataset**:
   - Place the `Tree_Species.zip` file in your Google Drive under the path `/MyDrive/Datasets/`.
   - Alternatively, modify the script to point to a local dataset directory.

3. **Run the Script**:
   - Open `tree_species (1).py` in a Python environment or Google Colab.
   - If using Colab, ensure you mount your Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Execute the script to:
     - Extract the dataset.
     - Preprocess images (remove corrupted, duplicates, and outliers).
     - Train the three models (EfficientNetB0, Basic CNN, MobileNetV2).
     - Generate predictions for test images.

4. **Model Predictions**:
   - The script includes a `predict_multiple_images` function to classify images in a specified folder (assumes a `predict_tree_species` function, not shown in the script).
   - Example usage:
     ```python
     test_folder = '/content/drive/MyDrive/Datasets/extracted_data/Tree_Species_Dataset/pipal'
     results = predict_multiple_images(test_folder)
     for result in results:
         print(f"Image: {result['image']}")
         print(f"Predicted Tree Species: {result['predicted_class']}")
         print(f"Confidence: {result['confidence']:.2f}%")
     ```

## Results

The models output predictions for each image in the test folder, including:
- The image filename.
- The predicted tree species (e.g., `pipal`, `mango`, `banyan`).
- The confidence score (as a percentage).

Sample output:
```
Image: image10.jpg
Predicted Tree Species: pipal
Confidence: 99.74%

Image: image11.jpg
Predicted Tree Species: mango
Confidence: 44.17%
...
```

Training and validation accuracy/loss curves are plotted for each model to evaluate performance.

## Notes

- The models are saved in HDF5 format (`tree_species_model.h5`, `basic_cnn_tree_species.h5`, `improved_cnn_model.h5`). Consider using the native Keras format (`.keras`) for future compatibility, as recommended by TensorFlow.
- The `predict_tree_species` function referenced in `predict_multiple_images` is not defined in the script. Ensure it is available or implement it to load a model and predict on single images.
- Adjust the `test_folder` path to test on different species.
- The dataset includes a `.git` folder, which is excluded during processing but may be unnecessary. Consider removing it from the ZIP file to reduce size.
- The MobileNetV2 model assumes 31 classes, which should match the number of species in the dataset. Verify class counts to ensure consistency.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, enhancements, or additional features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
