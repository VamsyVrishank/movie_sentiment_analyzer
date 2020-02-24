import tensorflow_datasets as tfds
import tensorflow.keras as keras
import tensorflow as tf 

pip install -q tensorflow tensorflow-datasets matplotlib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

tfds.disable_progress_bar()



#Loading the Dataset
dataset , info = tfds.load('imdb_reviews/subwords8k' , with_info=True,
                           as_supervised=True)

train_dataset , test_dataset = dataset['train'], dataset['test']

#The encoder object created encodes the text into numbers
encoder = info.features['text'].encoder




BUFFER_SIZE=10000
BATCH_SIZE=64

padded_shapes = ([None],())
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,
                                         padded_shapes=padded_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE,
                                         padded_shapes=padded_shapes)



model = keras.Sequential([keras.layers.Embedding(encoder.vocab_size , 64),
                          keras.layers.Bidirectional(keras.layers.LSTM(64 , return_sequences=True)),
                          keras.layers.Bidirectional(keras.layers.LSTM(32)),
                          keras.layers.Dense(64,activation='relu'),
                          keras.layers.Dense(1,activation='sigmoid')])
						  

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])



def pad_to_size(vec,size):
  zeros= [0]*(size-len(vec))
  vec.extend(zeros)
  return vec


#The sample text will be padded and converted to 64 size sentence
def sample_predict(sentence , pad):
  encoded_sample_pred_text = encoder.encode(sentence)
  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text , 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text , tf.float32)
  prediction = model.predict(tf.expand_dims(encoded_sample_pred_text , 0))

  return predictions


sample_text = ('This movie was awesome,acting was super good too, i would highly suggest you watch it')

predictions = sample_predict(sample_text , pad=True)*100

print(f'Percentage probability of this review being positive is : {predictions}')

