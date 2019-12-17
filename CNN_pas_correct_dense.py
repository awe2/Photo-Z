#!/usr/bin/env python
# coding: utf-8

# <h1>import and set up some definitions of classes</h1>

# In[249]:


## needed data packages: derived_data, HP_inputs, LP_inputs, MP_inputs, val_and_test_data_uniform_in_Z.npy',
## train_data_uniform_in_Z.npy', train_data_uniform_in_Z.npy'

##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime
import os
from skimage.transform import resize
import random
from tensorflow.python.keras.utils.data_utils import Sequence
from scipy.ndimage import zoom
from scipy.ndimage import shift
from skimage.transform import rotate
from sklearn import metrics

print(tf.__version__) #make sure we are using TF 2.0.0

print("Num GPUs Available: ", len((tf.config.experimental.list_physical_devices('GPU')))) #check we are using a GPU

tf.compat.v1.disable_eager_execution() #disables eager execution

tf.executing_eagerly() #needs to be false, run above else

data_path = '../../projects/physics/awe2/redshift_estimator_data/'

#input variables
filepath = data_path + 'models/' + "CNN_PAS_correct_dense.hdf5"
sixth_channel_path = data_path + 'derived_data/'
#mo
batch_size = 128 #32 #P used 128
epoch_number = 5 #when ready to test, set to 25
learning_rate = 1e-4 
n_classes = 1
train_from_file = False

params_train = {'dim': (120,120),
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': 5,
          'shuffle': True,
          'augment': False}

params_test = {'dim': (120,120),
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': 5,
          'shuffle': False,
          'augment':False}

#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/48097478
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(120,120), n_channels=3,
                 n_classes=2, shuffle=True, augment=False):
     #   'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augment

    def on_epoch_end(self):
    #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
    #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        X_2 = np.empty((self.batch_size,20), dtype=float)
        y = np.empty((self.batch_size), dtype=float)
    

      # Generate data and perform augmentation
        for i, ID in enumerate(list_IDs_temp):
            
          # Store sample
            if ID[:2] == 'HP':
                X[i,:,:,0:5] = np.load(data_path +'HP_inputs/' + ID + '.npy')
            if ID[:2] == 'LP':
                X[i,:,:,0:5] = np.load(data_path +'LP_inputs/' + ID + '.npy')
            if ID[:2] == 'MP':
                X[i,:,:,0:5] = np.load(data_path +'MP_inputs/' + ID + '.npy')
            
            #I messed up when generating my inputs on the order of the band; I want to train the model in the usual bands:
            
            if ID[:2] == 'HP': #neccessary becasue I am an idiot. This is because I pre-processed data incorrectly.
                #u_filter = X[i,:,:,0]
                #g_filter = X[i,:,:,1]
                i_filter = X[i,:,:,2]
                r_filter = X[i,:,:,3]
                #z_filter = X[i,:,:,4]

                #X[i,:,:,0] = u_filter
                #X[i,:,:,1] = g_filter
                X[i,:,:,2] = r_filter
                X[i,:,:,3] = i_filter
                #X[i,:,:,4] = z_filter
            
            #scale
            X[i,:,:,0:5] = X[i,:,:,0:5] / 255.0
            
            #Augment
            if self.augment == True:
                #flip
                if random.random() > 0.5:
                    X[i,:,:,0:5] = np.flip(X[i,:,:,0:5],0)
                if random.random() > 0.5:
                    X[i,:,:,0:5] = np.flip(X[i,:,:,0:5],1)

                #shift
                if random.random() > 0.5 :
                    X[i,:,:,0:5] = shift(X[i,:,:,0:5], (4,0,0), mode='nearest')
                elif random.random() > 0.5 :
                    X[i,:,:,0:5] = shift(X[i,:,:,0:5], (-4,0,0), mode='nearest')

                if random.random() > 0.5 :
                    X[i,:,:,0:5] = shift(X[i,:,:,0:5], (0,4,0), mode='nearest')
                elif random.random() > 0.5 :
                    X[i,:,:,0:5] = shift(X[i,:,:,0:5], (0,-4,0), mode='nearest')

                #zoom in/out
                zoom_factor = random.uniform(0.75,1.3)
                X[i,:,:,0:5] = clipped_zoom(X[i,:,:,0:5],zoom_factor)

                #rotate
                angle = 45*random.random()
                X[i,:,:,0:5] = rotate(X[i,:,:,0:5], angle=angle, mode='reflect')

            #Load Magnitude array and place in 6th channel
            string = ID.split('_') #list looks like ['HP','00000']
            string = string[0] +'_'+ string[1] +'.npy'
            
            X_2[i,] = np.load(sixth_channel_path+string, allow_pickle=True)
            # Store class
            y[i] = self.labels[ID]
            #print(y[i])
            #determine which class it should be...
            #y[i] = int(round((y[i] / 0.9)*(self.n_classes - 1),0))
            #if y[i] > self.n_classes - 1: #if above 0.4, essentially a far outlier: set to our highest redshift bin.
            #    y[i] = self.n_classes - 1
    
        if self.n_classes > 2:
            return ([X, X_2], keras.utils.to_categorical(y, num_classes=self.n_classes))
        else:
            #print(y)
            return ([X, X_2], y)

    def __len__(self):
    #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    #  'Generate one batch of data'
      # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
###############################################################
###Set up structure of training data###
###############################################################

#targets are generated in a technical notebook, load them in here.
with open(data_path + 'MISC/' + "Z_Spectro.txt","r") as inf:
    targets = eval(inf.read()) #is a dictionary that looks like: {alias:Spectro_Z}
    
train_list = np.load(data_path +'MISC/' + 'train_data_uniform_in_Z.npy').tolist()

data_list = np.load(data_path + 'MISC/' + 'val_and_test_data_uniform_in_Z.npy').tolist()

k = 4500
for i in range(500):
    string = 'LP_{}'.format(str(k + i).zfill(5))
    if string in train_list:
        train_list.remove(string)
    if string in data_list:
        data_list.remove(string)
    
random.seed(1)
random.shuffle(data_list)
index_one = int(len(data_list)/3)
val_list = data_list[0:index_one]
test_list = data_list[index_one::]

partition = {'train':train_list,'validation':val_list,'test':test_list}

#how many steps are in an epoch?
steps_to_take = int(len(train_list)/batch_size)
val_steps_to_take = int(len(val_list)/batch_size)
test_steps_to_take = int(len(test_list)/batch_size)

                #typically be equal to the number of unique samples if your dataset
                #divided by the batch size.

print(steps_to_take)
print(val_steps_to_take)
print(test_steps_to_take)

training_generator = DataGenerator(partition['train'], targets, **params_train) #we want to shuffle these inputs
validation_generator = DataGenerator(partition['validation'], targets, **params_test) #but not the val or test; need order for
test_generator = DataGenerator(partition['test'], targets, **params_test)#analysis; shouldn't matter 'cause not training on them

#############################################################
###model setup, callbacks###
#############################################################

def inception(input_layer, nbS1, nbS2, without_kernel_5=False):
        
    s1_0 = keras.layers.Conv2D(filters=nbS1,
                      kernel_size=1,
                      padding='same',
                      activation=keras.layers.PReLU())(input_layer)

    s2_0 = keras.layers.Conv2D(filters=nbS2,
                      kernel_size=3,
                      padding='same',
                      activation=keras.layers.PReLU())(s1_0)

    s1_2 = keras.layers.Conv2D(filters=nbS1,
                      kernel_size=1,
                      padding='same',
                      activation=keras.layers.PReLU())(input_layer)
         
    pool0 = keras.layers.AveragePooling2D(pool_size=2,
                       strides=1,
                       padding="same")(s1_2)

    if not(without_kernel_5):
        s1_1 = keras.layers.Conv2D(filters=nbS1,
                          kernel_size=1,
                          padding='same',
                          activation=keras.layers.PReLU())(input_layer)

        s2_1 = keras.layers.Conv2D(filters=nbS2,
                          kernel_size=5,
                          padding='same',
                          activation=keras.layers.PReLU())(s1_1)
    
    s2_2 = keras.layers.Conv2D(filters=nbS2,
                      kernel_size=1,
                      padding='same',
                      activation=keras.layers.PReLU())(input_layer)

    if not(without_kernel_5):
        output = keras.layers.concatenate([s2_2, s2_1, s2_0, pool0],
                               axis=3)
    else:
        output = keras.layers.concatenate([s2_2, s2_0, pool0],
                              axis=3)

    return output


def create_model_pasqueet(learning_rate = learning_rate):
    
    input_images = keras.layers.Input(shape=(120,120,5))
    input_derived_data = keras.layers.Input(shape=(20))
    
    conv0 = keras.layers.Conv2D(filters=64,
                                kernel_size=1,
                                padding='same',
                                activation=keras.layers.PReLU())(input_images)
    
    conv0p = keras.layers.AveragePooling2D(pool_size=2,
                                          strides=1)(conv0)
    
    i0 = inception(conv0p, 48, 64)
    i1 = inception(i0, 64, 92)
    i1p = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(i1)
    i2 = inception(i1p, 92, 128)
    i3 = inception(i2, 92, 128)
    i3p = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(i3)
    i4 = inception(i3p, 92, 128, without_kernel_5=True)

    flat = keras.layers.Flatten()(i4)
    concat = keras.layers.concatenate([flat,input_derived_data], axis=1)

    fc0 = keras.layers.Dense(units=1024, activation=keras.layers.ReLU())(concat) #P used 1096
    fc1 = keras.layers.Dense(units=1024, activation=keras.layers.ReLU())(fc0) #P used 1096
    fc2 = keras.layers.Dense(units=params_train['n_classes'], activation=keras.activations.linear)(fc1)
    
    model = keras.Model(inputs=[input_images, input_derived_data], outputs=[fc2])
    
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse')
    return(model)

ModelCheckpointCB = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=1, patience=2, min_lr=1e-8)

keras.backend.clear_session()
model = create_model_pasqueet(learning_rate = learning_rate)

#model.load_weights(filepath)
#######################################################################
#### Train Model#####
######################################################################
hist = model.fit_generator(generator=training_generator,
                    steps_per_epoch=steps_to_take, 
                    epochs=30, #P uses 30
                    initial_epoch=0,
                    verbose=2,
                    callbacks=[ModelCheckpointCB,reduce_lr])

######################################################################
### Predict and test #################################################
######################################################################
#model.load_weights(filepath) 
