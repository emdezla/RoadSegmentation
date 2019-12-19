# Road Segmentation ML Project 2019
##### Emilio Fernández, Tyler Benkley, Frédéric Berdoz

This project presents and explores three different machine learning approaches in order to perform an image segmentation task on satellite images for distinguishing the road. In other words, a binary classifier is implemented in three different ways (Logistic Regression, Convolutional Neural Network and U-net Neural Netwrok) so that it labels each pixel as road or background. 

The dataset was downloaded from [CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation) and our best machine learning approach (U-net neural network) achieved a F1 score of 0.9, which shows the potential of deep learning techniques for image segmentation.


### Description of the files

The following three models have been implemented in three different notebooks:
- `LogisticRegression.ipynb`: classifier based on regularized logistic regression. 
- `CNN.ipnyb`: classifier based on basic convolutional neural network.
- `UNet.ipnyb`: clasiffier based on U-net neural network.

Most of the image processing methods are defined inside the notebooks. However, the following files also contain some utility methods:
- `proj2_helpers.py`: given basic helpers methods
- `mask_to_submission.py`: utilities for submission
- `submission_to_mask.py`: utilities for masking
- `image_processing.py`: utilities for the final run scrip with image processing and deep learning methods

Finally, the following scripts are the ones that can be run:
- `run_with_U-net_training.py`: loads the data, trains the U-Net neural network from the beggining and classifies the test set
- `run.py`:  directly loads the U-net model from a file and performs classification on the test set 


### Setup

The model has been trained using Google Colab, a similar environment to Jupyter-Lab, but providing a free Tesla K80 GPU to run on the cloud, which accelerates substantially the model training.

The following external libraries must be installed to run the project:

- Keras (running on top of TensorFlow): deep learning library used for designing our neural networks
- OpenCV: Computer vision library used for image feature extraction


### How to run







