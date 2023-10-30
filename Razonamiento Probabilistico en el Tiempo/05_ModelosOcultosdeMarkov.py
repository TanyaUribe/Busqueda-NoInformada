"""
Los HMM (modelo oculto de Markov) son modelos utilizados en razonamiento probabilístico 
en el tiempo para modelar secuencias de observaciones con estados ocultos.
"""

#pip install hmmlearn

import numpy as np
from hmmlearn import hmm

# Definir el modelo HMM
model = hmm.MultinomialHMM(n_components=2)  # 2 estados ocultos (soleado y lluvioso)

# Definir las probabilidades iniciales de los estados ocultos
model.startprob_ = np.array([0.7, 0.3])  # Soleado: 70%, Lluvioso: 30%

# Definir las probabilidades de transición entre estados ocultos
model.transmat_ = np.array([[0.8, 0.2],  # Soleado a Soleado: 80%, Soleado a Lluvioso: 20%
                            [0.4, 0.6]])  # Lluvioso a Soleado: 40%, Lluvioso a Lluvioso: 60%

# Definir las probabilidades de emisión (observaciones) en cada estado oculto
model.emissionprob_ = np.array([[0.4, 0.6],  # Soleado: 40% de observar "Caminar", 60% de observar "Correr"
                                [0.7, 0.3]])  # Lluvioso: 70% de observar "Caminar", 30% de observar "Correr"

# Generar una secuencia de observaciones (por ejemplo, caminar o correr) a partir del modelo
X, Z = model.sample(n_samples=10)

# Imprimir la secuencia de observaciones y los estados ocultos
print("Secuencia de Observaciones:", X)
print("Estados Ocultos:", Z)
