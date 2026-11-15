import numpy as np

# --- CONTINUOS ---
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e

# --- BINARIOS ---
def onemax(x):
    # Suma de unos (siempre es maximizar, si el usuario elige minimizar lo invertimos)
    return np.sum(x)

def mochila(x, pesos=np.array([2, 3, 4, 5, 9, 7, 1, 6]), valores=np.array([3, 4, 5, 8, 10, 7, 2, 6]), cap=15):
    # Ajusta la dimensión al tamaño de los arreglos
    p = pesos[:len(x)]
    v = valores[:len(x)]
    peso_total = np.sum(x * p)
    valor_total = np.sum(x * v)
    if peso_total > cap:
        return valor_total - 50 * (peso_total - cap) # Penalización por exceso
    return valor_total

# --- DISCRETOS / PERMUTACIONES (TSP) ---
# Matriz de distancias fija de 6 ciudades para pruebas rápidas
DISTANCIAS_TSP = np.array([
    [0, 12, 29, 22, 13, 24],
    [12, 0, 19, 3, 25, 6],
    [29, 19, 0, 21, 23, 28],
    [22, 3, 21, 0, 4, 5],
    [13, 25, 23, 4, 0, 16],
    [24, 6, 28, 5, 16, 0]
])

def tsp(ruta):
    ruta = ruta.astype(int)
    dist = 0
    for i in range(len(ruta)):
        ciudad_actual = ruta[i]
        siguiente_ciudad = ruta[(i + 1) % len(ruta)]
        dist += DISTANCIAS_TSP[ciudad_actual, siguiente_ciudad]
    return dist

CATALOGO_PROBLEMAS = {
    "Continuo": {
        "1": {"nombre": "Sphere", "fn": sphere, "dominio": [-5.12, 5.12], "dim_default": 5, "objetivo_sugerido": "Minimizar"},
        "2": {"nombre": "Rastrigin", "fn": rastrigin, "dominio": [-5.12, 5.12], "dim_default": 5, "objetivo_sugerido": "Minimizar"},
        "3": {"nombre": "Rosenbrock", "fn": rosenbrock, "dominio": [-2.0, 2.0], "dim_default": 5, "objetivo_sugerido": "Minimizar"},
        "4": {"nombre": "Ackley", "fn": ackley, "dominio": [-32.768, 32.768], "dim_default": 5, "objetivo_sugerido": "Minimizar"},
    },
    "Binario": {
        "1": {"nombre": "OneMax", "fn": onemax, "dominio": [0, 1], "dim_default": 15, "objetivo_sugerido": "Maximizar"},
        "2": {"nombre": "Problema de la Mochila", "fn": mochila, "dominio": [0, 1], "dim_default": 8, "objetivo_sugerido": "Maximizar"},
    },
    "Categorico": {
        "1": {"nombre": "TSP (Agente Viajero - 6 Ciudades)", "fn": tsp, "dominio": [0, 5], "dim_default": 6, "objetivo_sugerido": "Minimizar"},
    }
}