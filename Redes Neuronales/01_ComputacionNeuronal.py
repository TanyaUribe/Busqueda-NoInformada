#pip install tensorflow

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Crear un modelo secuencial
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar las imágenes 28x28
    Dense(128, activation='relu'),  # Capa oculta con 128 neuronas y función de activación ReLU
    Dense(10, activation='softmax')  # Capa de salida con 10 neuronas para 10 clases y función de activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')
