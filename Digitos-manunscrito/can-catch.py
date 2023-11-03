import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Carga el modelo previamente entrenado
model = tf.keras.models.load_model('C:\\Users\\hikop\\Documents\\IA\\Digitos-manunscrito\\numcatch.keras')
# Si has guardado el modelo después de entrenar, puedes cargarlo así:
# model = load_model('path_to_your_model.h5')
def predict_digit(image_path):
    # Carga la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    # Invierte los colores (si es necesario)
    img = 255 - img
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    img = np.invert(img)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    # Redimensiona la imagen a 28x28 píxeles
    img = cv2.resize(img, (28, 28))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()   
    # Normaliza la imagen
    img = img.astype('float32') / 255
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()    
    # Cambia la forma de la imagen para que coincida con el formato de entrada del modelo
    img = np.reshape(img, (1, 28, 28, 1))  
    # Realiza la predicción
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    return digit

# Ruta de la imagen externa (reemplaza 'path_to_your_image' con la ruta correcta)
for i in range(10):
    image_name = os.path.isfile(str(i)+'.png') == True
    if image_name:
        image_path = str(i)+'.png'
        print("Predicted Digit:", predict_digit(image_path))
    else:
        print("No hay imagen")
    #print("Predicted Digit:", predict_digit(image_path))


# Predice el dígito y lo imprime
