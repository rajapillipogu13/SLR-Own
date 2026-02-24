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


#===========================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img("/content/split_dataset/test/Screenshot 2026-02-21 182801.png")
plt.imshow(img)

cv2.imread("/content/split_dataset/train/A/1.jpg").shape
#(128, 128, 3)

train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('/content/split_dataset/train',
                                          target_size=(128,128),
                                          batch_size = 32,
                                          class_mode = 'categorical')
val_dataset = val.flow_from_directory('/content/split_dataset/val',
                                          target_size=(128,128),
                                          batch_size = 32,
                                          class_mode = 'categorical')
#Found 40960 images belonging to 32 classes.
#Found 10240 images belonging to 32 classes.

train_dataset.class_indices
# {'A': 0,
#  'B': 1,
#  'BACKSPACE': 2,
#  'C': 3,
#  'CLOSE': 4,
#  'D': 5,
#  'E': 6,
#  'F': 7,
#  'G': 8,
#  'H': 9,
#  'I': 10,
#  'J': 11,
#  'K': 12,
#  'L': 13,
#  'M': 14,
#  'MINIMIZE': 15,
#  'N': 16,
#  'NEXT': 17,
#  'O': 18,
#  'OK': 19,
#  'P': 20,
#  'Q': 21,
#  'R': 22,
#  'S': 23,
#  'SPACE': 24,
#  'T': 25,
#  'U': 26,
#  'V': 27,
#  'W': 28,
#  'X': 29,
#  'Y': 30,
#  'Z': 31}

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
    tf.keras.layers.GlobalAveragePooling2D(),
    #
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #
    tf.keras.layers.Dense(32,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

model.summary()
model_fit = model.fit(train_dataset,
                      epochs = 20,
                      validation_data = val_dataset,
                      #callbacks = [early_stop]
                      )

dir_path = '/content/split_dataset/test'
for i in os.listdir(dir_path):
  img = image.load_img(dir_path+'//'+i,target_size=(128,128))
  plt.imshow(img)
  plt.show()

  X = image.img_to_array(img)
  X = X/255.0
  X = np.expand_dims(X,axis=0)
  images = np.vstack([X])
  val = model.predict(images)
  predicted_class_index = np.argmax(val[0])
  if predicted_class_index == 0:
    print('A')
  elif predicted_class_index == 1:
    print('B')
  elif predicted_class_index == 2:
    print('Backspace')
  elif predicted_class_index == 3:
    print('C')
  elif predicted_class_index == 4:
    print('Close')
  elif predicted_class_index == 5:
    print('D')
  elif predicted_class_index == 6:
    print('E')
  elif predicted_class_index == 7:
    print('F')
  elif predicted_class_index == 8:
    print('G')
  elif predicted_class_index == 9:
    print('H')
  elif predicted_class_index == 10:
    print('I')
  elif predicted_class_index == 11:
    print('J')
  elif predicted_class_index == 12:
    print('K')
  elif predicted_class_index == 13:
    print('L')
  elif predicted_class_index == 14:
    print('M')
  elif predicted_class_index == 15:
    print('Minimize')
  elif predicted_class_index == 16:
    print('N')
  elif predicted_class_index == 17:
    print('Next')
  elif predicted_class_index == 18:
    print('O')
  elif predicted_class_index == 19:
    print('OK')
  elif predicted_class_index == 20:
    print('P')
  elif predicted_class_index == 21:
    print('Q')
  elif predicted_class_index == 22:
    print('R')
  elif predicted_class_index == 23:
    print('S')
  elif predicted_class_index == 24:
    print('Space')
  elif predicted_class_index == 25:
    print('T')
  elif predicted_class_index == 26:
    print('U')
  elif predicted_class_index == 27:
    print('V')
  elif predicted_class_index == 28:
    print('W')
  elif predicted_class_index == 29:
    print('X')
  elif predicted_class_index == 30:
    print('Y')
  elif predicted_class_index == 31:
    print('Z')
#