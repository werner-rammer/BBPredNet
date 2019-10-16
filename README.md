BBPredNet
=========
Werner Rammer, Rupert Seidl, 2019.


Deep Neural Network for predicting bark beetle disturbance using TensorFlow. This is code and data for the publication:

Rammer W, Seidl R, 2019: Harnessing deep learning in ecology: An example predicting bark beetle outbreaks, Front. Plant Sci., doi: [10.3389/fpls.2019.01327](https://www.frontiersin.org/articles/10.3389/fpls.2019.01327)

BBPredNet is an example for the application of Deep Neural Networks (DNNs) for ecological data. The network is trained with data from a 23 year time series of bark beetle outbreaks from the Bavaria National Park, Germany. The network predicts the risk for bark beetle attack for individual 30 m cells based on the distribution of potential host trees and recent bark beetle activity in the vicinity of the cell.


The data used for training and testing are observations from the Bavarian National Park, avaialable here: http://datadryad.org/resource/doi:10.5061/dryad.c5g9s

### Files:

* prepare.data.R: R-Script for data preparation
* bb_model_Experiment1.py: Deep Neural Network used for Experiment 1
* bb_model_Experiment1_keras.py: The DNN for Experiment 1, but using Keras instead of TFLearn
* bb_model_Experiment2.py: Deep Neural Network used for Experiment 2
* training_data.zip: pre-processed training data (prepare.data.R)


### Steps to set up and run the model:

* install Tensorflow (tested with 0.12 and 1.0): https://www.tensorflow.org/install
* install TFLearn (0.3): http://tflearn.org
* download the training data from DRYAD: http://datadryad.org/resource/doi:10.5061/dryad.c5g9s
* run the pre-processing steps (once) or use the data in training_data.zip: prepare.data.R (requires R: https://www.r-project.org/)
* run this script and modify the paths if required (Assumptions: training data is stored in the working directory, TensorFlow logdirectory in /tmp/)
