# Gender Detector
In this project we create a human gender classifier based on convolutional neural network with PyTorch.

## Data Preparation
[Original dataset](https://drive.google.com/file/d/1-HUNDjcmSqdtMCvEkVlI0q43qlkcXBdK/view) structure:

```
└── internship_data
    ├── female
    └── male
```
Analysis of the dataset revealed the following problems:
- presence of duplicates
- class mismatch
- lack of informative signs
- impossibility of visual identification

For validation and testing we will resize and normalize dataset. Large image size is not required therefore we use a 128x128 image. For training we will use additional transformations. Random rotation and horizontal flip will diversify training data.

## Neural network
The assigned task is a classification task. We will use CNN architecture. Since it is necessary to define only two classes, max poolling layers will reduce the number of parameters without unnecessary loss of information. Using Dropout layers will reduce the impact of overfitting during long training. At the output, we use the `log_softmax` function to apply the `NLLLoss` loss function.

## Hyperparameters
After several experiments it was found that `NLLLoss` is more suitable than `CrossEntropy` function for this problem. It gives a more accuracy. `Adam` was chosen as the learning algorithm. It's stable and does a fairly good job of finding a more or less optimal solution. Experiments have shown that the learning rate `2e-4` is optimal for getting out of local minimum.

## Results
The model showed a high rate of accuracy on the test sample. However, the dataset requires a deeper cleanup using clustering algorithms to eliminate the remaining problems. Optimization of the model can be aimed at reducing the depth of the network and using the ensemble of networks. In one of the grayscale tests, the model also performed well. Thus, using a less resource-intensive architecture is possible.

## Instructions
### Training the network
To start training you need (see **gender_detector.ipynb**):
1. If you use google colab then connect google drive with the dataset;
2. Unpack the archive with data (check data_path variable in notebook);
3. Go through the sequential steps in the notebook until loading the model (to the words "*Now let's test the model*").

### Test and using
To test the neural network, you need to load the model (see `.ipynb` after the training stage). Check all imports and variables first.
To check the folder you need to use the command: 
	```
		python3 process.py folder/to/process
	```
Check model.pt file and **model_path** variable in process.py.