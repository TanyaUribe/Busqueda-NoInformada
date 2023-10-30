"""
Los filtros de Kalman son algoritmos utilizados en razonamiento probabilístico
en el tiempo para estimar el estado de un sistema dinámico a partir de mediciones ruidosas.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
dt = 1.0  # Intervalo de tiempo
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado
H = np.array([[1, 0]])  # Matriz de observación
Q = np.array([[0.01, 0], [0, 0.01]])  # Covarianza del proceso (ruido del sistema)
R = np.array([[0.1]])  # Covarianza de la medición (ruido de la observación)

# Condiciones iniciales
x = np.array([0, 0])  # Estado inicial: [posición, velocidad]
P = np.identity(2)  # Covarianza inicial

# Generar una secuencia de observaciones simuladas
observaciones = [0.5, 2.0, 3.5, 5.0, 6.5]

# Listas para almacenar estimaciones de posición y velocidades
estimaciones_posicion = []
estimaciones_velocidad = []

for z in observaciones:
    # Predicción
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Actualización (corrección) utilizando las observaciones
    y = z - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)

    estimaciones_posicion.append(x[0])
    estimaciones_velocidad.append(x[1])

# Visualizar los resultados
tiempo = np.arange(0, len(observaciones) * dt, dt)
plt.figure(figsize=(12, 6))
plt.plot(tiempo, observaciones, label='Observaciones', marker='o', linestyle='--')
plt.plot(tiempo, estimaciones_posicion, label='Estimaciones de Posición')
plt.plot(tiempo, estimaciones_velocidad, label='Estimaciones de Velocidad')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.title('Filtro de Kalman para Estimación de Posición y Velocidad')
plt.grid(True)
plt.show()
