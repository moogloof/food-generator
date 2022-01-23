import pickle
import numpy as np
import tensorflow as tf
import keras.models as models
import keras.layers as layers

# Gan training method
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
	gen, dis = gan.layers

	# Train GAN loop
	for epoch in range(n_epochs):
		print(f"\rDoing epoch {epoch}.", end="")
		for bat in dataset:
			# Training discriminator
			noise = tf.random.normal(shape=[batch_size, codings_size])
			gen_images = gen(noise)
			X_fake_and_real = tf.concat([gen_images, bat], axis=0)
			y1 = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)
			dis.trainable = True
			dis.train_on_batch(X_fake_and_real, y1)

			# Training generator
			noise = tf.random.normal(shape=[batch_size, codings_size])
			y2 = tf.constant([[1.0]] * batch_size)
			dis.trainable = False
			gan.train_on_batch(noise, y2)

	print("\r-------------Done-------------")

# Networks
# Generator
gen = models.Sequential([
	layers.Dense(3000, activation="selu"),
	layers.Dense(500, activation="selu"),
	layers.Dense(150 * 150 * 3, activation="sigmoid"),
	layers.Reshape([150, 150, 3])
])

# Discriminator
dis = models.Sequential([
	layers.Flatten(input_shape=[150, 150, 3]),
	layers.Dense(3000, activation="selu"),
	layers.Dense(500, activation="selu"),
	layers.Dense(50, activation="selu"),
	layers.Dense(1, activation="sigmoid")
])

# 0 - fake
# 1 - real

# Full GAN
model = models.Sequential([gen, dis])

# Setup dis
dis.compile(optimizer="rmsprop", loss="binary_crossentropy")
dis.trainable = False

# Setup model
model.compile(loss="binary_crossentropy", optimizer="rmsprop")

with open("data.bin", "rb") as f:
	dataset = np.array(pickle.load(f), dtype="float64")
	print(dataset.shape)
	dataset = np.reshape(dataset, (dataset.shape[0], 150, 150, 3))
	dataset = dataset[:(dataset.shape[0] // 20) * 20]
	dataset = np.reshape(dataset, (dataset.shape[0] // 20, 20, 150, 150, 3)) / 255
	print(dataset)

try:
	train_gan(model, dataset, 20, 3000)
except KeyboardInterrupt:
	pass

# Save model status
model.save("model.h5")

