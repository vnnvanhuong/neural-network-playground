from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.losses import mean_squared_error
from keras.optimizers import SGD
import tensorflow as tf

import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

input_layer = Input(shape=(28, 28, 1))
dense_1 = Dense(784, activation='relu')(input_layer)
dense_2 = Dense(392, activation='sigmoid')(dense_1)
dense_3 = Dense(196, activation='relu')(dense_2)
dense_4 = Dense(98, activation='sigmoid')(dense_3)
dense_5 = Dense(49, activation='relu')(dense_4)
dense_6 = Dense(98, activation='sigmoid')(dense_5)
dense_7 = Dense(196, activation='relu')(dense_6)
dense_8 = Dense(392, activation='sigmoid')(dense_7)
dense_9 = Dense(784, activation='relu')(dense_8)

model = Model(input_layer, [dense_1,dense_2,dense_3,dense_4,dense_5,dense_6,dense_7,dense_8,dense_9])
model.compile(optimizer=SGD(lr=0.3), loss=mean_squared_error, metrics=['accuracy'])
model.summary()

def plotDense(x, y, layer):
	img = layer[:, :, 0]
	ax[x, y].imshow(img, cmap=plt.cm.binary)
	ax[x, y].set_xlim([0, img.shape[1]])
	ax[x, y].set_ylim([img.shape[0], 0])

for i in range(5):
	img = x_train[i]
	batch_x = tf.expand_dims(input=img, axis=0)
	batch_x = tf.expand_dims(input=batch_x, axis=0)
	prediction = model.predict_on_batch(batch_x)
	[[dense_1],[dense_2],[dense_3],[dense_4],[dense_5],[dense_6],[dense_7],[dense_8],[dense_9]] = prediction

	_, ax = plt.subplots(2, 5, figsize=(15, 7.35))
	ax[0, 0].imshow(img, cmap=plt.cm.binary)
	plotDense(0,1,dense_1)
	plotDense(0,2,dense_2)
	plotDense(0,3,dense_3)
	plotDense(0,4,dense_4)
	plotDense(1,0,dense_5)
	plotDense(1,1,dense_6)
	plotDense(1,2,dense_7)
	plotDense(1,3,dense_8)
	plotDense(1,4,dense_9)	
	plt.show()