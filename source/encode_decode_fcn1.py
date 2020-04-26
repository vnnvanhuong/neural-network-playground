from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[:1000, :]
y = x.reshape((1000,-1))/255

ishape = [28, 28]
input_tensor = Input(shape=ishape)
tensor = Flatten()(input_tensor)
tensor = Dense(128, activation='relu')(tensor)
tensor = Dense(64, activation='relu')(tensor)
tensor = Dense(16, activation='relu')(tensor)
tensor = Dense(64, activation='relu')(tensor)
tensor = Dense(128, activation='relu')(tensor)
output_tensor = Dense(784, activation='sigmoid')(tensor)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)

model.fit(x, y, batch_size=100, epochs=1000)

pred_y = model.predict(x)

plt.figure(figsize=(15,7.35))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.xlabel(y_train[i])
	plt.imshow(pred_y[i].reshape((28,28)), cmap=plt.cm.binary)
plt.show()