import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import subprocess


from tensorflow import keras
from keras import layers, models
from github import Github


### Add lines to import modules as needed

## 

def build_model1():
  model = models.Sequential()
  
  model.add(layers.Conv2D(32,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(64,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(128,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  for i in range(4):
    model.add(layers.Conv2D(32,(3,3), strides=(1,1), padding="same"))
    model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)))

  model.add(layers.Flatten())

  model.add(layers.Dense(128))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10))
  return model

def build_model2():
  model = model = models.Sequential()
  
  model.add(layers.Conv2D(32,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  model.add(layers.SeparableConv2D(64,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  model.add(layers.SeparableConv2D(128,(3,3), strides=(2,2), padding="same"))
  model.add(layers.BatchNormalization())

  for i in range(4):
    model.add(layers.SeparableConv2D(32,(3,3), strides=(1,1), padding="same"))
    model.add(layers.BatchNormalization())
  
  model.add(layers.MaxPooling2D(pool_size=(4,4), strides=(4,4)))

  model.add(layers.Flatten())

  model.add(layers.Dense(128))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10))
  return model

def build_model3():

  inputs = keras.Input(shape=(32,32,3), name="image")

  x = layers.Conv2D(32, (3,3), strides=(2,2), padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  block_1 = x

  x = layers.Conv2D(64, (3,3), strides=(2,2), padding="same")(block_1)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(128, (3,3), strides=(2,2), padding="same")(x)
  x = layers.BatchNormalization()(x)
  block_1 = layers.Conv2D(128, (1,1), strides=(4,4), padding="same")(block_1)
  block_2 = layers.add([x,block_1])
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(128, (3,3), strides=(1,1), padding="same")(block_2)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(128, (3,3), strides=(1,1), padding="same")(x)
  x = layers.BatchNormalization()(x)
  block_2 = layers.Conv2D(128, (3,3), strides=(1,1), padding="same")(block_2)
  block_3 = layers.add([x,block_2])
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(128, (3,3), strides=(1,1), padding="same")(block_3)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(128, (3,3), strides=(1,1), padding="same")(x)
  x = layers.Dropout(0.2)(x)
  x = layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(128)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(10)(x)
  outputs = layers.Dense(10)(x)

  model = keras.Model(inputs, outputs, name="model_3")
  return model

def build_model50k():
  
  inputs = keras.Input(shape=(32,32,3), name="image")

  x = layers.Conv2D(8, (3,3), strides=(2,2), padding="same")(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  block_1 = x

  x = layers.Conv2D(16, (3,3), strides=(2,2), padding="same")(block_1)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(32, (3,3), strides=(2,2), padding="same")(x)
  x = layers.BatchNormalization()(x)
  block_1 = layers.Conv2D(32, (1,1), strides=(4,4), padding="same")(block_1)
  block_2 = layers.add([x,block_1])
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(32, (3,3), strides=(1,1), padding="same")(block_2)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(32, (3,3), strides=(1,1), padding="same")(x)
  x = layers.BatchNormalization()(x)
  block_2 = layers.Conv2D(32, (3,3), strides=(1,1), padding="same")(block_2)
  block_3 = layers.add([x,block_2])
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(32, (3,3), strides=(1,1), padding="same")(block_3)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.2)(x)
  
  x = layers.Conv2D(16, (3,3), strides=(1,1), padding="same")(x)
  x = layers.Dropout(0.2)(x)
  x = layers.MaxPooling2D(pool_size=(4,4), strides=(4,4))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(16)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(10)(x)
  outputs = layers.Dense(10)(x)

  model = keras.Model(inputs, outputs, name="model_sub50k") # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

  # Now separate out a validation set.
  val_frac = 0.1
  num_val_samples = int(len(train_images)*val_frac)
  # choose num_val_samples indices up to the size of train_images, !replace => no repeats
  val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
  trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
  val_images = train_images[val_idxs, :,:,:]
  train_images = train_images[trn_idxs, :,:,:]

  val_labels = train_labels[val_idxs]
  train_labels = train_labels[trn_idxs]

  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()
  val_labels = val_labels.squeeze()

  input_shape  = train_images.shape[1:]
  train_images = train_images / 255.0
  test_images  = test_images  / 255.0
  val_images   = val_images   / 255.0

  seed = 6
  tf.random.set_seed(seed)
  np.random.seed(seed)
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  train_hist_plain = model1.fit(train_images, train_labels, 
                       validation_data=(val_images, val_labels),
                       epochs=50)
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam',
           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  train_hist_plain = model2.fit(train_images, train_labels, 
                       validation_data=(val_images, val_labels),
                       epochs=50)
  
  ### Repeat for model 3 and your best sub-50k params model
  
  model3 = build_model3()
  model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  train_hist_plain = model3.fit(train_images, train_labels, 
                       validation_data=(val_images, val_labels),
                       epochs=50)
  
  model50k = build_model50k()
  model50k.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  train_hist_plain = model50k.fit(train_images, train_labels, 
                       validation_data=(val_images, val_labels),
                       epochs=50)
  
  model50k.save("best_model.h5")