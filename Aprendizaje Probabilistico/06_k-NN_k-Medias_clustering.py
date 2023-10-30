"""
El objetivo del clustering (una técnica de aprendizaje automático) es descubrir 
patrones o estructuras subyacentes en los datos, agrupando objetos que comparten 
características similares. 
k-Means es un algoritmo de clustering ampliamente utilizado. Su objetivo es dividir un 
conjunto de datos en k clusters, donde k es un número predefinido por el usuario. 
k-NN (k-Nearest Neighbors) es un algoritmo de aprendizaje supervisado utilizado tanto 
para clasificación como para regresión. 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Generar datos aleatorios con tres clusters
n_samples = 300
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Visualizar los datos
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Datos de Ejemplo')
plt.show()

# Aplicar k-Means para clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Visualizar los centroides y los clusters encontrados por k-Means
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroides')
plt.title('Clustering con k-Means')
plt.legend()
plt.show()

# Aplicar k-NN para clasificación
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Crear una malla para visualizar las fronteras de decisión de k-NN
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualizar las fronteras de decisión de k-NN
plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Clasificación con k-NN')
plt.show()
