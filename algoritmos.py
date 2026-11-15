import numpy as np

def ejecutar_ga(fn_eval, tipo_prob, objetivo, dominio, dim, params):
    pop_size = params["poblacion"]
    gens = params["iteraciones"]
    mut_rate = params["tasa_mutacion"]
    
    # 1. Inicialización según tipo de problema
    if tipo_prob == "Continuo":
        poblacion = np.random.uniform(dominio[0], dominio[1], (pop_size, dim))
    elif tipo_prob == "Binario":
        poblacion = np.random.randint(0, 2, (pop_size, dim))
    elif tipo_prob == "Categorico":
        poblacion = np.array([np.random.permutation(dim) for _ in range(pop_size)])

    historial = []
    sin_mejora = 0
    mejor_global_val = None
    mejor_global_sol = None
    motivo_termino = "Máximo de iteraciones alcanzado"

    for gen in range(1, gens + 1):
        valores_f = np.array([fn_eval(ind) for ind in poblacion])
        
        if objetivo == "Maximizar":
            fitness = valores_f
            idx_mejor = np.argmax(valores_f)
        else:
            fitness = -valores_f
            idx_mejor = np.argmin(valores_f)

        mejor_actual = valores_f[idx_mejor]
        
        # Control de mejor histórico y estancamiento
        if mejor_global_val is None or (objetivo == "Maximizar" and mejor_actual > mejor_global_val) or (objetivo == "Minimizar" and mejor_actual < mejor_global_val):
            mejor_global_val = mejor_actual
            mejor_global_sol = poblacion[idx_mejor].copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        historial.append(float(mejor_global_val))

        # Emitir estado en tiempo real a la terminal
        yield {
            "iteracion": gen,
            "total": gens,
            "mejor_valor": float(mejor_global_val),
            "mejor_sol": mejor_global_sol,
            "historial": historial,
            "terminado": False,
            "motivo": ""
        }

        # Criterio de paro por estancamiento (ej. 30 iteraciones sin mejora)
        if sin_mejora >= 30 and gen > 30:
            motivo_termino = f"Convergencia temprana (sin mejora por {sin_mejora} iteraciones)"
            break

        # --- SELECCIÓN (Torneo) ---
        nueva_pob = []
        for _ in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            ganador = i1 if fitness[i1] > fitness[i2] else i2
            nueva_pob.append(poblacion[ganador].copy())
        nueva_pob = np.array(nueva_pob)

        # --- CRUCE Y MUTACIÓN ADAPTADOS AL TIPO DE PROBLEMA ---
        if tipo_prob == "Continuo":
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < 0.8:
                    alpha = np.random.rand()
                    h1 = alpha * nueva_pob[i] + (1 - alpha) * nueva_pob[i+1]
                    h2 = (1 - alpha) * nueva_pob[i] + alpha * nueva_pob[i+1]
                    nueva_pob[i], nueva_pob[i+1] = h1, h2
            for i in range(pop_size):
                if np.random.rand() < mut_rate:
                    nueva_pob[i] += np.random.normal(0, (dominio[1] - dominio[0]) * 0.05, size=dim)
                    nueva_pob[i] = np.clip(nueva_pob[i], dominio[0], dominio[1])

        elif tipo_prob == "Binario":
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < 0.8 and dim > 1:
                    pt = np.random.randint(1, dim)
                    h1 = np.concatenate([nueva_pob[i][:pt], nueva_pob[i+1][pt:]])
                    h2 = np.concatenate([nueva_pob[i+1][:pt], nueva_pob[i][pt:]])
                    nueva_pob[i], nueva_pob[i+1] = h1, h2
            for i in range(pop_size):
                for j in range(dim):
                    if np.random.rand() < mut_rate:
                        nueva_pob[i][j] = 1 - nueva_pob[i][j] # Bit-flip

        elif tipo_prob == "Categorico":
            # Mutación Swap para mantener permutaciones válidas
            for i in range(pop_size):
                if np.random.rand() < mut_rate and dim > 1:
                    idx1, idx2 = np.random.choice(dim, 2, replace=False)
                    nueva_pob[i][idx1], nueva_pob[i][idx2] = nueva_pob[i][idx2], nueva_pob[i][idx1]

        poblacion = nueva_pob

    # Resultado final
    yield {
        "iteracion": gen,
        "total": gens,
        "mejor_valor": float(mejor_global_val),
        "mejor_sol": mejor_global_sol,
        "historial": historial,
        "terminado": True,
        "motivo": motivo_termino
    }

# Aquí mismo agregas ejecutar_pso, ejecutar_de, ejecutar_aco y ejecutar_ais
# respetando la misma firma y usando `yield` al final de cada iteración.