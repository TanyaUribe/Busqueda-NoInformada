"""
La probabilidad a priori es un concepto que se utiliza en estadística y aprendizaje 
automático para representar la probabilidad de un evento antes de observar datos adicionales.
"""

# Probabilidades a priori (suposiciones iniciales)
probabilidad_spam_a_priori = 0.3  # Probabilidad inicial de que un correo sea spam
probabilidad_no_spam_a_priori = 1 - probabilidad_spam_a_priori  # Probabilidad inicial de que un correo no sea spam

# Datos observados (evidencia)
palabras_clave = ["oferta", "ganador", "millones"]  # Ejemplo de palabras clave en el correo

# Calcular probabilidades a posteriori
# Estas son suposiciones simples y pueden mejorarse con datos reales y algoritmos de clasificación.

# Probabilidad condicional de que un correo con palabras clave sea spam
probabilidad_spam_dado_palabras_clave = 0.8

# Probabilidad condicional de que un correo con palabras clave no sea spam
probabilidad_no_spam_dado_palabras_clave = 1 - probabilidad_spam_dado_palabras_clave

# Calcular probabilidades finales usando el teorema de Bayes
probabilidad_spam_final = (probabilidad_spam_a_priori * probabilidad_spam_dado_palabras_clave) / ((probabilidad_spam_a_priori * probabilidad_spam_dado_palabras_clave) + (probabilidad_no_spam_a_priori * probabilidad_no_spam_dado_palabras_clave))
probabilidad_no_spam_final = 1 - probabilidad_spam_final

# Tomar una decisión basada en las probabilidades finales
if probabilidad_spam_final > probabilidad_no_spam_final:
    decision = "Spam"
else:
    decision = "No spam"

print(f"Probabilidad final de spam: {probabilidad_spam_final}")
print(f"Probabilidad final de no spam: {probabilidad_no_spam_final}")
print(f"Decisión: {decision}")
