import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import random

# Load model
model = tf.keras.models.load_model("model.h5")
gen, _ = model.layers

"""
# Load data
with open("data.bin", "rb") as f:
	dataset = pickle.load(f)
	dataset = np.asarray(dataset)
	dataset_ids = np.random.choice(len(dataset)-1, 5, replace=False)
	dataset = [dataset[i] for i in dataset_ids]
	dataset = np.reshape(dataset, (5, 150, 150, 3))
"""

# Get noise
noise = tf.random.normal(shape=[5, 1000])

# Generated images
gen_image = gen(noise)

# Plot generated images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

"""
for i, ax in enumerate(axes[0, :]):
	ax.imshow(dataset[i], cmap="gray")
	ax.set_title("Real")
"""

for i, ax in enumerate(axes[1, :]):
	ax.imshow(gen_image[i], cmap="gray")
	ax.set_title("Generated")

plt.tight_layout()
plt.show()
