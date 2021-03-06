#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
BBPredNet
Deep Neural Network for predicting bark beetle disturbance using TensorFlow.
The data used for training and testing are observations from the Bavarian National Park, avaialable here: http://datadryad.org/resource/doi:10.5061/dryad.c5g9s
See also the data preprocessing script prepare.data.R (written in R)

This script trains and executes the model for Experiment 1 in the paper "Harnessing deep learning in ecology: An example predicting bark beetle outbreaks" (https://www.frontiersin.org/articles/10.3389/fpls.2019.01327).
Three years of the data are used for testing (15.7%), the rest for training.

More details in the publication: "Harnessing deep learning in ecology: An example predicting bark beetle outbreaks" (https://www.frontiersin.org/articles/10.3389/fpls.2019.01327).

Steps to set up and run the model:
    * install Tensorflow (tested with 0.12 and 1.0): https://www.tensorflow.org/install
    * install TFLearn (currently 0.3): http://tflearn.org
    * download the training data from DRYAD: http://datadryad.org/resource/doi:10.5061/dryad.c5g9s
    * run the pre-processing steps (once) or use the data in training_data.zip: prepare.data.R (requires R: https://www.r-project.org/)
    * run this script and modify the paths if required (Assumptions: training data is stored in the working directory, TensorFlow logdirectory in /tmp/)
    
Some practical tipps:
    * Use tensorboard for visualizing the training progress (check the tensorboard_verbose parameter of the tflearn.DNN() function )
    * instead of creating a new Python instance, use tf.reset_default_graph() to delete the current TensorFlow graph: this saves the lengthy loading of the training data
@author: Werner Rammer
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from keras.models import  Model
from keras.layers.core import Dense, Dropout,  Flatten,  RepeatVector
from keras.layers.convolutional import Conv2D
from keras.layers import Input,    concatenate,    TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
#import tensorflow as tf

"""  
# Step 1: loading the data
# The data is pre-processed elsewhere (see prepare.data.R)
# Each record in the test/validation data contains:
    * 19x19 image pixels, 0: no host, 1: host, 2: damage last year, 3: damages 2 yrs ago [0..360]
    * observed damage for the focal pixel yes/no (note that the focal pixel in the above image is cleared) [362]
    * the year (1=1989, 2=1990, ...): not used for training [363]
    * the long-term mean annual temperature of the focal cell [361]
    * the precipitation anomaly of the year: not used for training [364]
    * the outbreak stage class (encoded as background=1, culmination=2, gradation=3) [366]
"""

# Data loading and preprocessing
## data layout.
bbeval = np.loadtxt("bbyearseval.txt", dtype='int8')
bbtrain = np.loadtxt("bbyearstrain.txt", dtype='int8')

# load alternative (reversed bbtrain)
bbtrain = np.loadtxt("bbyearstrain_rev.txt", dtype='int8')
bbeval = np.loadtxt("bbevalall.txt", dtype='int8')
"""  
# Step 2: Transformation for training
# the input for training is split up in two streams: input 1 is the image (19x19), input 2 is the climate proxies.
# the target (label) is the observed damage.
"""

# the reshape transform the flat list of 361 numbers to the required format
testX = bbeval[:, 0:361].reshape([-1, 19, 19, 1])
testY = bbeval[:, 362]
#testY = np.eye(2)[testY]
testXclim = bbeval[:, (361,366)]
                
X = bbtrain[:, 0:361].reshape([-1, 19, 19, 1])
Y = bbtrain[:, 362]
#Y = np.eye(2)[Y]
Xclim = bbtrain[:,(361,366)]


## normalize cliamte data: this has a very little impact on the resulsts
testXclim = np.asfarray(testXclim)    
testXclim -= np.mean(Xclim, axis=0)  # use the training data set for this
testXclim /= np.std(Xclim, axis=0)

Xclim = np.asfarray(Xclim)    
Xclim -= np.mean(Xclim, axis=0)  
Xclim /= np.std(Xclim, axis=0)

X = X.astype('float32')
Xclim = Xclim.astype('float32')
testX = testX.astype('float32')
testXclim = testXclim.astype('float32')
Y = Y.astype('float32')
testY = testY.astype('float32')
"""
#############################################################
# Step 3: setup of the DNN
#############################################################
"""
## clear all: this clears all TensorFlow variables and the full network
#tf.reset_default_graph()
K.clear_session()
# Input
imginput =  Input(shape=(19,19,1,), dtype='float32', name='imginput')
climateinput = Input(shape=(2,), dtype='float32', name='climateinput')
## convolutional layers

img = Conv2D(32,5, activation='relu')(imginput)
img = Conv2D(32,5, activation='relu')(img)
img = Conv2D(32,3, activation='relu')(img)
img = Conv2D(48,3, activation='relu')(img)

network = Dense(256, activation='relu')(img)
#network = BatchNormalization()(network)
network = Dropout(0.5)(network)
network = Dense(256, activation='relu')(network)
network = Dropout(0.5)(network)

network = Flatten()(network)

# branch for climate
cbranch = Dense(64, activation='relu')(climateinput)
cbranch = Dense(64, activation='relu')(cbranch)
cbranch = Dense(64, activation='relu')(cbranch)

network = concatenate([network, cbranch])
network = Dense(128, activation='relu')(network)
network = Dropout(0.5)(network)
network = Dense(64, activation='relu')(network)

out = Dense(2, activation="softmax", name="out")(network)

model = Model(inputs=[imginput, climateinput], outputs=[out]) #

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0) # epsilon=1e-08

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])

print(model.summary())


tbCallBack = TensorBoard(log_dir='e:/tmp/tf_bb/v45add') # , histogram_freq=5, write_graph=False, write_images=False

tbReduceLr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.00001)

tbCheckpoint = ModelCheckpoint('snapshots/weights.{epoch:02d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

history = model.fit(x= {'imginput': X, 'climateinput': Xclim}, y=[Y], batch_size=256, epochs=30,
                    validation_data=({'imginput': testX, 'climateinput': testXclim},  testY), callbacks=[tbReduceLr, tbCallBack, tbCheckpoint], initial_epoch=3) #, initial_epoch=20  



# Inputs:
network = input_data(shape=[None, 19, 19, 1], name='input')
in_climate = input_data(shape=[None, 2], name='in_climate') ## temp, outbreak state



# Convolutional layers are fed with the input "image"
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
network = max_pool_2d(network, 2)

network = conv_2d(network, 16, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
network = max_pool_2d(network, 2)

network = conv_2d(network, 16, 3, activation='relu', regularizer="L2", weight_decay=0.0001)
network = max_pool_2d(network, 2)

network = conv_2d(network, 16, 5, activation='relu', regularizer="L2", weight_decay=0.0001)
network = max_pool_2d(network, 2)

# A fully connected layer with regularization (batch normalization and dropout)
network = fully_connected(network, 512, activation='relu')
network = batch_normalization(network)
network = dropout(network, 0.7)

# The climate proxies are processed in a separate "branch" and then merged to the network
cbranch = fully_connected(in_climate, 64, activation='relu', regularizer='L2')
cbranch = fully_connected(cbranch, 64, activation='relu', regularizer='L2')
cbranch = fully_connected(cbranch, 64, activation='relu', regularizer='L2')
network = merge([network, cbranch],'concat') # merge the climate input into the network



# A series of fully connected layers:
network = fully_connected(network, 512, activation='relu')
network = batch_normalization(network)
network = dropout(network, 0.7)

network = fully_connected(network, 512, activation='relu')
network = batch_normalization(network)
network = dropout(network, 0.7)

network = fully_connected(network, 512, activation='relu')
network = batch_normalization(network)
network = dropout(network, 0.7)

network = fully_connected(network, 512, activation='relu')
network = batch_normalization(network)
network = dropout(network, 0.8)

# use a softmax activation which outputs a probability distribution over two classes
network = fully_connected(network, 2, activation='softmax')
adam = tflearn.optimizers.Adam(learning_rate=0.0001, epsilon=1e-06) # use the ADAM optimizer

# the loss function is "categorical cross entropy"
network = regression(network, optimizer=adam,  
                     loss='categorical_crossentropy', name='target')

# set up the DNN. A neat feature: a snapshot is saved whenever the best achieved test accuracy until now is surpassed,
# i.e. the *latest* file in the snapshots folder will be the model with the best accuracy
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logY/', best_checkpoint_path='/tmp/snapshots/', best_val_accuracy=0.96)

# the actual training: the batch_size of 512 is rather high, but showed a good compromise between GPU load and accuracy
# (at least when training with GPU, we used here a single NVidia GTX 1070)
model.fit({'input': X, 'in_climate': Xclim}, {'target': Y}, n_epoch=60, batch_size=512,
           validation_set=({'input': testX,'in_climate': testXclim}, {'target': testY}),
           snapshot_step=None, show_metric=True, run_id='BBPredNet E1')


"""
#############################################################
##############   Steps Post-Training ########################
#############################################################
"""

"""
# Loading/Saving of snapshots
"""
# load the epoch with the best performance on the training set
model = load_model('E:/Daten/Projekte/RESIN/machine_learning/bb_outbreak/\snapshots/weights.03-0.9667.hdf5')

model = load_model('E:/Daten/Projekte/RESIN/machine_learning/bb_outbreak/snapshots/weights.11-0.9671.hdf5')
"""
################ analysis of results ###############
"""

""" Helper function: 
## calculate precision/recall and other metrices for a dataset.
"""
def validationScores( threshold, data, evaldata ):
    nTP=0
    nTN=0
    nFP=0
    nFN=0
    for i in range(0, data.shape[0]):
        predNoKill = data[i]>threshold
        noKill = evaldata[i]==0                         
        if predNoKill:  
            if noKill:                    
                nTN=nTN+1
            else:
                nFN=nFN+1
        else:
            if noKill:
                nFP=nFP+1
            else:
                nTP=nTP+1
    
    accuracy = (nTP + nTN)  / testY.shape[0]
    precision = nTP / (nTP + nFP )
    recall = nTP / ( nTP + nFN )
    F1 = 2*(precision*recall) / (precision + recall)
    # for this particular case: it is desireable, that the total predicted damage is close to the 
    # total observed damage. The 'pred_kill_dev' is 1 in that case.
    pred_kill_dev =  ((nTP + nFP) / (nTP + nFN))
    
    print("     Pred Pos |  Pred Neg")
    print(" %8d (TP)| %8d (FN)" % (nTP, nFN))
    print(" %8d (FP)| %8d (TN)" % (nFP, nTN))
    print("Acc: %5.3f, Precision: %5.3f Recall: %5.3f F1: %5.3f +Kill: %5.3f Threshold: %4.2f" % ( accuracy, precision, recall, F1, pred_kill_dev, threshold) )
    return F1   

"""
####################################################
 Find out the best classification threshold (i.e.)
 at which probability to classify an example
 as 'killed'. 
 To do this, look at the scores for a sample
 of the training set. The optimal threshold is for this 
 case, when the observed damage is equal the predicted 
 damage (and F1 score is highest)
####################################################
"""
# predict part of trainng data
mprnd = np.random.randint(X.shape[0], size=100000)
mpredict_train=np.array([], ndmin=1)
for i in range(0,mprnd.shape[0],10000):
    print( i)
    mpredict_part = model.predict({'imginput': X[mprnd[i:(i+10000)], :], 'climateinput': Xclim[mprnd[i:(i+10000)], :]})      
    mpredict_train = np.append(mpredict_train, np.stack(mpredict_part)[:,0] )


## look at scores between 0.4 and 0.95
for i in range(40, 95):
    validationScores(i/100, mpredict_train, Y[mprnd])

# the best split is found at 0.73 
# 0.56
#################################################################################################
########### predict test data using the threshold determined with training data #################
#################################################################################################

mpredict=np.array([], ndmin=1)
for i in range(0,testX.shape[0],10000):
    print( i)
    mpredict_part = model.predict({'imginput': testX[i:(i+10000), :], 'climateinput': testXclim[i:(i+10000), :]})      
    mpredict = np.append(mpredict, np.stack(mpredict_part)[:,0] )

validationScores(0.5, mpredict, testY)


# save for further analysis, creation of maps, see bb.eval.R
np.savetxt('predict_bb_exp1.txt', mpredict)

