import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

test, train = datos['test'], datos['train']

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255
  return imagenes, etiquetas

train = train.map(normalizar)
test = test.map(normalizar)

#agrgar datos a cahe
train = train.cache()
test = test.cache()

for imagen, etiqueta in train.take(1):
  break
imagen = imagen.numpy().reshape((28, 28))

plt.figure()
plt.imshow(imagen, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

from keras.src.layers.attention.multi_head_attention import activation
#creacion del modelo

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='gelu', name='conv2d_1'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='gelu', name='conv2d_2'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='gelu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=50, activation='gelu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

tamano_lote=32
train_len = 60000
train = train.repeat().shuffle(train_len).batch(tamano_lote)

import math
historial = modelo.fit(
    train,
    epochs=60,
    steps_per_epoch=math.ceil(train_len/tamano_lote)
)

modelo.save('numcatch.keras')