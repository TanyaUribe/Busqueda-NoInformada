import sympy as sp

# Definir las variables y la función compuesta
x, u = sp.symbols('x u')
f = x**2 * sp.sin(u)

# Calcular la derivada utilizando la regla de cadena
df_dx = sp.diff(f, x)  # Derivada de f respecto a x
df_du = sp.diff(f, u)  # Derivada de f respecto a u

# Imprimir las derivadas
print("Derivada de f respecto a x:")
sp.pprint(df_dx, use_unicode=True)
print("\nDerivada de f respecto a u:")
sp.pprint(df_du, use_unicode=True)
