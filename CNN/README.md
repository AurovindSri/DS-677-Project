### Deep Learning Experiments

This repository contains code and experiments for systematically tuning hyperparameters in deep learning models. The experiments focus on determining the best configurations for batch size, learning rate (LR), and dropout probability through hyperparameter tuning. The process involves:

1. Creating a predefined hyperparameter space combining all possible configurations.
2. Training and evaluating each configuration on the validation set.
3. Selecting the configuration with the highest validation weighted F1 score as the best.

This approach ensures optimal balance between training stability, model generalization, and regularization effects.

---

## Files and Contents

### DeepLearning_No_dropout.ipynb:
This notebook contains baseline and augmented model experiments with no dropout applied. The following configurations are implemented:

1. **Cell 1:** Baseline model with:
   - Batch size: 128
   - Learning rate (LR): 0.0005

2. **Cell 2:** Horizontal flipping with a probability of 0.5:
   - Batch size: 128
   - LR: 0.0005

3. **Cell 3:** Horizontal flipping (0.5) + Mixup (0.5):
   - Batch size: 128
   - LR: 0.0005

4. **Cell 4:** Horizontal flipping (1.0) + Mixup (0.5):
   - Batch size: 128
   - LR: 0.001

5. **Cell 5:** Horizontal flipping (1.0):
   - Batch size: 64
   - LR: 0.0005

6. **Cell 6:** Horizontal flipping (0.5) + Mixup (0.2):
   - Batch size: 64
   - LR: 0.0005

7. **Cell 7 (Last Cell):** Training with both original and flipped images.

---

### DeepLearning_dropout.ipynb:
This notebook demonstrates the effects of dropout applied in the fully connected layers of a deep learning model. The configuration used is:

- Batch size: 128
- LR: 0.001

---

### CNN_dropout_in_convolution_and_fcs.ipynb:
This notebook implements dropout in both convolutional and fully connected layers of a CNN architecture. The configuration used is:

- Batch size: 128
- LR: 0.001

