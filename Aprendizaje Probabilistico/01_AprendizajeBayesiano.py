"""
El aprendizaje bayesiano es un enfoque que combina el razonamiento probabilístico con el 
aprendizaje automático para modelar la incertidumbre y realizar inferencias sobre datos observados. 
"""

#pip install pymc3

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Datos observados
datos_observados = np.random.normal(5, 2, 100)

# Modelo bayesiano
with pm.Model() as modelo:
    # Prior para la media y la desviación estándar
    media = pm.Normal("media", mu=0, sd=10)
    desviacion_estandar = pm.HalfNormal("desviacion_estandar", sd=10)

    # Likelihood (verosimilitud) de los datos
    likelihood = pm.Normal("likelihood", mu=media, sd=desviacion_estandar, observed=datos_observados)

    # Muestreo de la posterior
    traza = pm.sample(1000, tune=1000, cores=1)

# Visualizar los resultados
pm.traceplot(traza)
plt.show()
