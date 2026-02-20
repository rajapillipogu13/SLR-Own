#https://colab.research.google.com/drive/12lvXOTiU7c9s1fdD2MDuJbadzwhH6DJO#scrollTo=o_dmpiGtZDWd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img("/content/SLR-Own/basedata/train/A/0.jpg")
plt.imshow(img)

cv2.imread("/content/SLR-Own/basedata/train/A/0.jpg").shape

train = ImageDataGenerator(rescale=1/255)
val = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('/content/SLR-Own/basedata/train',
                                          target_size=(128,128),
                                          batch_size = 3,
                                          class_mode = 'categorical')
val_dataset = train.flow_from_directory('/content/SLR-Own/basedata/val',
                                          target_size=(128,128),
                                          batch_size = 3,
                                          class_mode = 'categorical')

train_dataset.class_indices

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Flatten(),
    #
    tf.keras.layers.Dense(512,activation='relu'),
    #
    tf.keras.layers.Dense(32,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
model_fit = model.fit(train_dataset,
                      steps_per_epoch = 8,
                      epochs = 30,
                      validation_data = val_dataset)