#Files and Contents:
-The best batch size, learning rate (LR), and dropout probability are determined through a systematic hyperparameter tuning process. A predefined hyperparameter space is created, combining all possible configurations of these parameters. For each configuration, the model is trained and evaluated on the validation set, tracking the weighted F1 score. The configuration with the highest validation F1 score is selected as the best. This approach ensures optimal balance between training stability, model generalization, and regularization effects.

#DeepLearning_No_dropout.ipynb:
-1.The baseline model, with no dropout, with batch size 128 and LR 0.0005 is in the first cell of the file.
2.In the second cell the code is for Horizontal flipping 0.5; batch size=128, LR=0.0005.
3.In the third cell the code is for Horizontal flip 0.5, Mixup 0.5; Batch-128, LR-0.0005.
4.In the fourth cell the code is for Horizontal Flipping 1.0 and Mixup 0.5; batch size 128, LR 0.001.
5.In the fifth cell the code is for Horizontal flipping (1.0), batch size of 64, LR of 0.0005.
6.In the sixth cell the code is for Horizontal Flipping 0.5 + Mixup 0.2: Batch size: 64, LR: 0.0005.
7.In the last cell the code is for the data of both original and flipped images.

DeepLearning_dropout.ipynb:
In this file the code demonstrates the effect of dropout in fully connected layers of a deep learning model with batch size 128 and LR of 0.001.

CNN_dropout_in_convolution_and_fcs.ipynb:
In this file the code implements and experiments with dropout in convolutional layers and fully connected layers in a CNN architecture with batch size 128, and LR 0.001.
