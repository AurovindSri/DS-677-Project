# Vision Transformer (ViT) for CIFAR-10 Classification

This project implements a Vision Transformer (ViT) model to classify images in the CIFAR-10 dataset. The code includes data preprocessing, training, validation, testing, evaluation metrics, and experiment tracking with MLflow.

## Features

- **Vision Transformer Model**: Based on the `deit_tiny_patch16_224` architecture from the `timm` library.
- **Data Augmentation**: Resize, normalize, and augment CIFAR-10 dataset.
- **Metrics**: Computes accuracy, F1 score, confusion matrix, and ROC curve.
- **Experiment Tracking**: Logs hyperparameters, metrics, and model checkpoints using MLflow.
- **Visualization**: Plots confusion matrix and ROC curves for detailed evaluation.

## Setup

### Prerequisites
- Python 3.8 or later
- PyTorch
- torchvision
- timm
- sklearn
- matplotlib
- seaborn
- numpy
- mlflow

### Installation
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure MLflow for tracking experiments:
   ```bash
   mlflow ui
   ```

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.

The dataset is automatically downloaded from torchvision if not already available.

## Model Architecture
This project uses the `deit_tiny_patch16_224` model from the `timm` library. The model is fine-tuned for 10 classes in the CIFAR-10 dataset. The classification head is replaced with a linear layer corresponding to the number of classes.

## Training Process
1. **Data Loading**: The CIFAR-10 dataset is split into training, validation, and testing sets. Data loaders are created with data augmentation and normalization.
2. **Training**: The model is trained using the Adam optimizer and CrossEntropyLoss.
3. **Validation**: Model performance is evaluated on the validation set after every epoch.
4. **Testing**: After training, the model is tested on the test set, and metrics are computed.
5. **Logging**: Metrics, hyperparameters, and the trained model are logged to MLflow.

## Evaluation Metrics
- **Accuracy**: Percentage of correctly classified images.
- **F1 Score**: Weighted F1 score for multi-class classification.
- **Confusion Matrix**: Visualizes true vs. predicted labels.
- **ROC Curve**: Receiver Operating Characteristic curve for each class with AUC values.

## How to Run
1. Run the main script:
   ```bash
   python main.py
   ```
2. Modify hyperparameters such as learning rate, batch size, and number of epochs in the `main()` function.
3. Results, including metrics and plots, will be logged to MLflow and displayed during execution.

## Outputs
- **Training Logs**: Loss, accuracy, and F1 score for training, validation, and test sets.
- **Confusion Matrix**: Heatmap of classification performance.
- **ROC Curve**: Class-wise ROC curves and AUC scores.
- **MLflow Artifacts**: Trained model, metrics, and hyperparameters.

## Example Logs
```
Epoch 1/15 | LR: 0.001 | Batch Size: 64
Train Loss: 1.4567, Train Acc: 50.23%, Train F1: 0.4912 |
Val Loss: 1.2205, Val Acc: 60.45%, Val F1: 0.6057 |
Test Loss: 1.2104, Test Acc: 60.80%, Test F1: 0.6081
...
```

## Visualization Examples
1. **Confusion Matrix**:
   ![Confusion Matrix](confusion_matrix.png)
2. **ROC Curve**:
   ![ROC Curve](roc_curve.png)

## Customization
- Modify the model architecture by changing the base model in the `TinyViT` class.
- Add additional data augmentations in the `load_data()` function.
- Adjust hyperparameters in the `main()` function.

## References
- [PyTorch Documentation](https://pytorch.org/docs/)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [MLflow Documentation](https://mlflow.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Hugging Face for the `timm` library.
- MLflow for providing a seamless experiment tracking platform.


