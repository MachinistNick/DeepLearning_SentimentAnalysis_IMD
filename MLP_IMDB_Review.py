# MLP for the IMDB review calculation
from keras.datasets import imdb

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding


#because of below code GPU start working
import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#Above 5 lines of code for GPU 
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#sess = tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True))


from tensorflow.python.client import device_lib
print("COOL")
print( device_lib.list_local_devices())


# load the dataset but only keep the top n words, zero the rest
top_words = 6000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
max_words = 600
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
