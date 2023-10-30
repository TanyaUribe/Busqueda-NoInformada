"""
La ponderación de verosimilitud, en el contexto del razonamiento probabilístico, 
se utiliza para actualizar la creencia en una hipótesis en función de la evidencia 
observada. A menudo, se aplica en el marco de la inferencia bayesiana. 

Supongamos que tienes una hipótesis H y una evidencia E, y deseas calcular la probabilidad 
posterior P(H|E) en función de las probabilidades a priori y la verosimilitud de la evidencia. 
En este ejemplo, asumiremos una hipótesis binaria (H o no H) y una evidencia binaria (E o no E).
"""

import sympy as sp

# Definir las probabilidades a priori
P_H = 0.3  # Probabilidad a priori de la hipótesis H
P_not_H = 1 - P_H  # Probabilidad a priori de no H

# Definir la verosimilitud de la evidencia dada H y no H
P_E_given_H = 0.8  # Probabilidad de observar E dado que H es verdadera
P_E_given_not_H = 0.2  # Probabilidad de observar E dado que H es falsa

# Calcular la probabilidad a posteriori de H dado E utilizando la regla de Bayes
P_H_given_E = (P_H * P_E_given_H) / ((P_H * P_E_given_H) + (P_not_H * P_E_given_not_H))

# Imprimir el resultado
print("Probabilidad a posteriori de H dado E:", P_H_given_E)
