# Tree Species Classification

This project focuses on classifying tree species from images using a convolutional neural network (CNN) built with TensorFlow and Keras. The model is trained on a dataset of tree images and can predict species with a given confidence score.

## Project Overview

The repository contains a Jupyter Notebook (`tree_species.ipynb`) that implements a CNN to classify images of various tree species. The dataset is organized into multiple classes, each representing a different tree species. The model is trained, saved, and used for predictions on new images.

## Dataset

The dataset is sourced from the `Tree_Species_Dataset` folder, containing images of 31 different tree species, including:

- Amla
- Asopalav
- Babul
- Bamboo
- Banyan
- Bili
- Cactus
- Champa
- Coconut
- Garmalo
- Gulmohor
- Gunda
- Jamun
- Kanchan
- Kesudo
- Khajur
- Mango
- Motichanoti
- Neem
- Nilgiri
- Other
- Pilikaren
- Pipal
- Saptaparni
- Shirish
- Simlo
- Sitafal
- Sonmahor
- Sugarcane
- Vad

The dataset is stored in a zipped format (`Tree_Species.zip`) and is extracted for use in the notebook.

## Model Architecture

The CNN model is built using TensorFlow and Keras with the following architecture:

- **Input Layer**: Accepts images of size 224x224 pixels with 3 color channels (RGB).
- **Convolutional Layers**:
  - First layer: 32 filters (3x3), ReLU activation, followed by BatchNormalization and MaxPooling (2x2).
  - Second layer: 64 filters (3x3), ReLU activation, followed by BatchNormalization and MaxPooling (2x2).
  - Third layer: 128 filters (3x3), ReLU activation, followed by BatchNormalization and MaxPooling (2x2).
- **Flattening**: Converts the feature maps to a 1D vector.
- **Fully Connected Layers**:
  - Dense layer with 256 units and ReLU activation, followed by a Dropout layer (0.5).
  - Output layer with 31 units (one for each class) and softmax activation.
- **Optimizer**: Adam with a learning rate of 1e-4.
- **Loss Function**: Categorical crossentropy.
- **Metrics**: Accuracy.

The model is trained for 25 epochs and saved as `improved_cnn_model.h5`.

## Training and Performance

- **Training**: The model is trained using a data generator (`train_generator`) with validation data (`val_generator`).
- **Performance**:
  - Final training accuracy: ~29.82%
  - Final validation accuracy: ~28.52%
  - Final validation loss: ~2.9510

## Usage

### Prerequisites

- Python 3.x
- TensorFlow/Keras
- NumPy
- Google Colab (or equivalent environment with GPU support)

### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Set Up the Dataset**:
   - Place the `Tree_Species.zip` file in the appropriate directory (e.g., `/content/drive/MyDrive/Datasets/`).
   - Extract the dataset using the provided code in the notebook.

3. **Run the Notebook**:
   - Open `tree_species.ipynb` in Jupyter or Google Colab.
   - Execute the cells to mount the drive, extract the dataset, train the model, and make predictions.

4. **Make Predictions**:
   - Use the `predict_tree_species` function to classify new images:
     ```python
     image_path = '/path/to/your/image.jpg'
     predicted_class, confidence = predict_tree_species(image_path)
     print(f'Predicted Tree Species: {predicted_class}')
     print(f'Confidence: {confidence:.2f}%')
     ```

### Example Prediction

For an example image (`image14.jpg`), the model predicted:
- **Species**: Motichanoti
- **Confidence**: 3.32%

## Notes

- The model shows modest performance, likely due to the complexity of the dataset or limited training data. Consider experimenting with data augmentation, transfer learning, or increasing the number of epochs to improve accuracy.
- The `.git` folder appears in the class list, which may be an error in the dataset organization. Ensure it is excluded from training data if not a valid class.
- The model is saved in HDF5 format (`improved_cnn_model.h5`). For better compatibility, consider saving in Keras format (`model.save('my_model.keras')`).

## Future Improvements

- Implement data augmentation to increase dataset diversity.
- Use transfer learning with pre-trained models (e.g., ResNet, VGG) for better feature extraction.
- Address potential issues with the dataset, such as the inclusion of the `.git` folder.
- Fine-tune hyperparameters (e.g., learning rate, number of layers) to improve performance.

## License

This project is licensed under the MIT License.