import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pylab as plt
import os
import time
from keras.models import load_model

noise_dim = 100
num_examples_to_generate = 20

#Seed 재활용
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generating_MODEL(model, test_input):
  predictions = model(test_input, training=False)
  print(predictions.shape[0])
  for k in range(100): #200회 생성을 100번 반복함 총20000개 # Memory Error 때문에 Loop을 두번 돌려야.
    for i in range(predictions.shape[0]):
      data = predictions[i, :, :, :] * 127.5 + 127.5
      data = np.uint8(data)
      imageio.imwrite('C:/~Run Folder~/Result/'
                      'Category_file/ImageName'+str(i+k*num_examples_to_generate)+'.png', data)
      print(i+k*num_examples_to_generate, "completed!")

path = 'C:/Generator path/'
generator = load_model(path+'Generator_model_name.h5')

# Image generation
generating_MODEL(generator,seed)