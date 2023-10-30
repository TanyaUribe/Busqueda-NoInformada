import math

# Definir un conjunto de datos (por ejemplo, una lista de números)
data = [10, 15, 20, 25, 30]

# Calcular la media (promedio) de los datos
mean = sum(data) / len(data)

# Calcular la suma de los cuadrados de las diferencias entre cada dato y la media
sum_of_squared_differences = sum((x - mean) ** 2 for x in data)

# Calcular la varianza (promedio de los cuadrados de las diferencias)
variance = sum_of_squared_differences / len(data)

# Calcular la desviación estándar (raíz cuadrada de la varianza)
std_deviation = math.sqrt(variance)

# Imprimir la desviación estándar
print("Desviación estándar:", std_deviation)
