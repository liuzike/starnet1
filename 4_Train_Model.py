
# coding: utf-8

# # Train the StarNet Model
# 
# This notebook takes you through the steps of how to train a StarNet Model
# - Required Python packages: `numpy h5py keras`
# - Required data files: training_data.h5, mean_and_std.npy
# 
# Note: We use tensorflow for the keras backend.

# In[1]:


import numpy as np
import h5py
import random

from keras.models import Model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import HDF5Matrix

datadir = ""
training_set = datadir + 'training_data.h5'
normalization_data = datadir + 'mean_and_std.npy'


# ** Normalization **
# 
# Write a function to normalize the output labels. Each label will be normalized to have approximately have a mean of zero and unit variance.
# 
# NOTE: This is necessary to put output labels on a similar scale in order for the model to train properly, this process is reversed in the test stage to give the output labels their proper units

# In[3]:


mean_and_std = np.load(normalization_data)
mean_labels = mean_and_std[0]
std_labels = mean_and_std[1]

def normalize(labels):
    # Normalize labels
    return (labels-mean_labels) / std_labels


# ** Obtain training data **
# 
# Here we will collect the output labels for the training and cross-validation sets, then normalize each.
# 
# Next we will create an HDF5Matrix for the training and cross-validation input spectra rather than loading them all into memory. This is useful to save RAM when training the model.

# In[4]:


# Define the number of output labels
num_labels = np.load(datadir+'mean_and_std.npy').shape[1]

# Define the number of training spectra
num_train = 41000

# Load labels
with  h5py.File(training_set, 'r') as F:
    y_train = np.hstack((F['TEFF'][0:num_train], F['LOGG'][0:num_train], F['FE_H'][0:num_train]))
    y_cv = np.hstack((F['TEFF'][num_train:], F['LOGG'][num_train:], F['FE_H'][num_train:]))

# Normalize labels
y_train = normalize(y_train)
y_cv = normalize(y_cv)

# Create the spectra training and cv datasets
x_train = HDF5Matrix(training_set, 'spectrum', 
                           start=0, end=num_train)
x_cv = HDF5Matrix(training_set, 'spectrum', 
                           start=num_train, end=None)

# Define the number of output labels
num_labels = y_train.shape[1]

num_fluxes = x_train.shape[1]

print('Each spectrum contains ' + str(num_fluxes) + ' wavelength bins')
print('Training set includes ' + str(x_train.shape[0]) + 
      ' spectra and the cross-validation set includes ' + str(x_cv.shape[0])+' spectra')


# **Build the StarNet model architecture**
# 
# The StarNet architecture is built with:
# - input layer
# - 2 convolutional layers
# - 1 maxpooling layer followed by flattening for the fully connected layer
# - 2 fully connected layers
# - output layer
# 
# First, let's define some model variables.

# In[5]:


# activation function used following every layer except for the output layers
activation = 'relu'

# model weight initializer
initializer = 'he_normal'

# number of filters used in the convolutional layers
num_filters = [4,16]

# length of the filters in the convolutional layers
filter_length = 8

# length of the maxpooling window 
pool_length = 4

# number of nodes in each of the hidden fully connected layers
num_hidden = [256,128]

# number of spectra fed into model at once during training
batch_size = 64

# maximum number of interations for model training
max_epochs = 30

# initial learning rate for optimization algorithm
lr = 0.0007
    
# exponential decay rate for the 1st moment estimates for optimization algorithm
beta_1 = 0.9

# exponential decay rate for the 2nd moment estimates for optimization algorithm
beta_2 = 0.999

# a small constant for numerical stability for optimization algorithm
optimizer_epsilon = 1e-08


# In[6]:


# Input spectra
input_spec = Input(shape=(num_fluxes,), name='starnet_input_x')

# Reshape spectra for CNN layers
cur_in = Reshape((num_fluxes, 1))(input_spec)

# CNN layers
cur_in = Conv1D(kernel_initializer=initializer, activation=activation, 
                padding="same", filters=num_filters[0], kernel_size=filter_length)(cur_in)
cur_in = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[1], kernel_size=filter_length)(cur_in)

# Max pooling layer
cur_in = MaxPooling1D(pool_size=pool_length)(cur_in)

# Flatten the current input for the fully-connected layers
cur_in = Flatten()(cur_in)

# Fully-connected layers
cur_in = Dense(units=num_hidden[0], kernel_initializer=initializer, 
               activation=activation)(cur_in)
cur_in = Dense(units=num_hidden[1], kernel_initializer=initializer, 
               activation=activation)(cur_in)

# Output nodes
output_label = Dense(units=num_labels, activation="linear", 
                     input_dim=num_hidden[1], name='starnet_output_y')(cur_in)

model = Model(input_spec, output_label, name='StarNet')


# **More model techniques**
# * The `Adam` optimizer is the gradient descent algorithm used for minimizing the loss function
# * `EarlyStopping` uses the cross-validation set to test the model following every iteration and stops the training if the cv loss does not decrease by `min_delta` after `patience` iterations
# * `ReduceLROnPlateau` is a form of learning rate decay where the learning rate is decreased by a factor of `factor` if the training loss does not decrease by `epsilon` after `patience` iterations unless the learning rate has reached `min_lr`

# In[7]:


# Default loss function parameters
early_stopping_min_delta = 0.0001
early_stopping_patience = 4
reduce_lr_factor = 0.5
reuce_lr_epsilon = 0.0009
reduce_lr_patience = 2
reduce_lr_min = 0.00008

# loss function to minimize
loss_function = 'mean_squared_error'

# compute mean absolute deviation
metrics = ['mae']


# In[8]:


optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta, 
                                       patience=early_stopping_patience, verbose=2, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon, 
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)


# **Compile model**

# In[10]:


model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
model.summary()


# **Train model**

# In[11]:


model.fit(x_train, y_train, validation_data=(x_cv, y_cv),
          epochs=max_epochs, verbose=1, shuffle='batch',
          callbacks=[early_stopping, reduce_lr])


# **Save model**

# In[12]:


starnet_model = 'starnet_cnn.h5'
model.save(datadir + starnet_model)
print(starnet_model+' saved.')

