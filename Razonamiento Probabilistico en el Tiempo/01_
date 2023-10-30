"""
Los procesos estacionarios en el tiempo, también conocidos como procesos estocásticos estacionarios, 
son modelos utilizados en estadísticas y razonamiento probabilístico para describir datos secuenciales 
que mantienen propiedades estadísticas constantes a lo largo del tiempo. 
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir los parámetros del proceso estacionario
media = 0
desviacion_estandar = 1
n = 100  # Longitud de la secuencia de tiempo

# Generar una secuencia de tiempo de valores aleatorios
np.random.seed(0)  # Establecer una semilla para la reproducibilidad
datos = np.random.normal(loc=media, scale=desviacion_estandar, size=n)

# Visualizar la secuencia de tiempo
plt.plot(datos)
plt.title("Proceso Estacionario en el Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.show()
