#importing required libraries

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50                #importing ResNet for training models
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#used to install unzip 
!sudo apt-get install unzip
#used for unziping dataset 
!unzip /Face.zip -d /tmp

train_path = "/tmp/Train"
valid_path = "/tmp/Test"
#input image size
IMAGE_SIZE = [224, 224]
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#for stoping layers to get trained again
for layer in resnet.layers:
  layer.trainable = False

folders = glob('/tmp/Train/*')

x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=resnet.input, outputs=prediction)
model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#preprocessing of images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/tmp/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/tmp/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=100,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))

#save trained model for future use
import tensorflow as tf

from keras.models import load_model

model.save('transfer.h5')

n=load_model('transfer.h5')

"""These steps are use in colab for mounting drive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()                      
drive = GoogleDrive(gauth)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth 
from oauth2client.client import GoogleCredentials

model.save('transfer.h5')
model_file = drive.CreateFile({'title' : 'transfer.h5'})
model_file.SetContentFile('transfer.h5')
model_file.Upload()

drive.CreateFile({'id': model_file.get('id')}) """

