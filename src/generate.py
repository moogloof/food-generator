import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import random

# Load model
model = tf.keras.models.load_model("model.h5")
gen, _ = model.layers

# Load data
with open("data.bin", "rb") as f:
	dataset = pickle.load(f)
	dataset = np.asarray(dataset[random.randint(0, len(dataset)-1)])
	dataset = np.reshape(dataset, (250, 250))

# Get noise
noise = tf.random.normal(shape=[1, 3000])

# Generated images
gen_image = gen(noise)

# Plot generated images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax1 = axes[0]
ax1.imshow(dataset, cmap="gray")
ax1.set_title("Real")
ax2 = axes[1]
ax2.imshow(gen_image[0], cmap="gray")
ax2.set_title("Generated")
plt.tight_layout()
plt.show()
