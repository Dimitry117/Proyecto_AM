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
    elif tipo_prob in ["Categórico", "Categorico", "Discreto"]:
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

        # Criterio de paro por estancamiento
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

        elif tipo_prob in ["Categórico", "Categorico", "Discreto"]:
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


def ejecutar_pso(fn_eval, tipo_prob, objetivo, dominio, dim, params):
    """
    Implementación del algoritmo de Enjambre de Partículas (PSO).
    Especializado en espacios de búsqueda continuos.
    """
    if tipo_prob != "Continuo":
        yield {
            "iteracion": 0, "total": params["iteraciones"],
            "mejor_valor": 0, "mejor_sol": [],
            "historial": [], "terminado": True,
            "motivo": f"Error: PSO no opera nativamente en el tipo {tipo_prob}"
        }
        return

    # 1. Extracción de hiperparámetros
    pop_size = params["particulas"]
    gens = params["iteraciones"]
    w = params["inercia"]
    c1 = params["c1"]
    c2 = params["c2"]

    historial = []
    sin_mejora = 0
    motivo_termino = "Máximo de iteraciones alcanzado"

    # 2. Inicialización
    posiciones = np.random.uniform(dominio[0], dominio[1], (pop_size, dim))
    rango = dominio[1] - dominio[0]
    v_max = 0.2 * rango
    velocidades = np.random.uniform(-v_max, v_max, (pop_size, dim))

    pbest_pos = posiciones.copy()
    pbest_val = np.array([fn_eval(p) for p in posiciones])

    if objetivo == "Maximizar":
        idx_gbest = np.argmax(pbest_val)
    else:
        idx_gbest = np.argmin(pbest_val)
    
    gbest_pos = pbest_pos[idx_gbest].copy()
    gbest_val = pbest_val[idx_gbest]

    # 3. Bucle Principal de Optimización
    for gen in range(1, gens + 1):
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)

        velocidades = (w * velocidades + 
                       c1 * r1 * (pbest_pos - posiciones) + 
                       c2 * r2 * (gbest_pos - posiciones))
        
        velocidades = np.clip(velocidades, -v_max, v_max)
        posiciones = posiciones + velocidades
        posiciones = np.clip(posiciones, dominio[0], dominio[1])
        valores_f = np.array([fn_eval(p) for p in posiciones])
        mejora_global_en_iteracion = False

        for i in range(pop_size):
            mejora_pbest = (valores_f[i] > pbest_val[i]) if objetivo == "Maximizar" else (valores_f[i] < pbest_val[i])
            if mejora_pbest:
                pbest_val[i] = valores_f[i]
                pbest_pos[i] = posiciones[i].copy()
                mejora_gbest = (valores_f[i] > gbest_val) if objetivo == "Maximizar" else (valores_f[i] < gbest_val)
                if mejora_gbest:
                    gbest_val = valores_f[i]
                    gbest_pos = posiciones[i].copy()
                    mejora_global_en_iteracion = True

        if mejora_global_en_iteracion:
            sin_mejora = 0
        else:
            sin_mejora += 1

        historial.append(float(gbest_val))

        yield {
            "iteracion": gen,
            "total": gens,
            "mejor_valor": float(gbest_val),
            "mejor_sol": gbest_pos.copy(),
            "historial": historial,
            "terminado": False,
            "motivo": ""
        }

        if sin_mejora >= 30 and gen > 30:
            motivo_termino = f"Convergencia temprana (sin mejora por {sin_mejora} iteraciones)"
            break

    yield {
        "iteracion": gen,
        "total": gens,
        "mejor_valor": float(gbest_val),
        "mejor_sol": gbest_pos.copy(),
        "historial": historial,
        "terminado": True,
        "motivo": motivo_termino
    }


def ejecutar_aco(fn_eval, tipo_prob, objetivo, dominio, dim, params):
    """
    Implementación del algoritmo de Colonia de Hormigas (ACO).
    Especializado en problemas combinatorios/categóricos (ej. TSP).
    """
    if tipo_prob not in ["Categórico", "Categorico", "Discreto"]:
        yield {
            "iteracion": 0, "total": params["iteraciones"],
            "mejor_valor": 0, "mejor_sol": [],
            "historial": [], "terminado": True,
            "motivo": f"Error: ACO está diseñado para problemas Categóricos, no {tipo_prob}"
        }
        return

    # 1. Extracción de hiperparámetros
    num_hormigas = params.get("hormigas", 20)
    gens = params["iteraciones"]
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 2.0)
    evap = params.get("evaporacion", 0.5)
    q = params.get("q", 100.0)

    if "matriz_distancias" in params:
        distancias = params["matriz_distancias"]
    else:
        distancias = np.ones((dim, dim))
        np.fill_diagonal(distancias, 0)

    distancias_seguras = np.where(distancias == 0, 1e-10, distancias)
    visibilidad = 1.0 / distancias_seguras

    # 2. Inicialización
    feromonas = np.ones((dim, dim)) * 0.1 
    historial = []
    sin_mejora = 0
    motivo_termino = "Máximo de iteraciones alcanzado"
    
    mejor_global_sol = None
    mejor_global_val = -np.inf if objetivo == "Maximizar" else np.inf

    # 3. Bucle Principal de Optimización
    for gen in range(1, gens + 1):
        soluciones_hormigas = []
        valores_hormigas = []

        for h in range(num_hormigas):
            nodo_actual = np.random.randint(0, dim)
            ruta = [nodo_actual]
            nodos_visitados = set(ruta)

            while len(ruta) < dim:
                probabilidades = np.zeros(dim)
                for j in range(dim):
                    if j not in nodos_visitados:
                        tau = feromonas[nodo_actual][j] ** alpha
                        eta = visibilidad[nodo_actual][j] ** beta
                        probabilidades[j] = tau * eta
                
                suma_prob = np.sum(probabilidades)
                if suma_prob > 0:
                    probabilidades = probabilidades / suma_prob
                    siguiente_nodo = np.random.choice(range(dim), p=probabilidades)
                else:
                    nodos_restantes = list(set(range(dim)) - nodos_visitados)
                    siguiente_nodo = np.random.choice(nodos_restantes)
                
                ruta.append(siguiente_nodo)
                nodos_visitados.add(siguiente_nodo)
                nodo_actual = siguiente_nodo

            soluciones_hormigas.append(ruta)
            valor = fn_eval(np.array(ruta))
            valores_hormigas.append(valor)

            mejora = (valor > mejor_global_val) if objetivo == "Maximizar" else (valor < mejor_global_val)
            if mejora:
                mejor_global_val = valor
                mejor_global_sol = ruta.copy()
                sin_mejora = 0

        historial.append(float(mejor_global_val))

        # D. Actualización de Feromonas
        feromonas = (1.0 - evap) * feromonas
        for i in range(num_hormigas):
            ruta = soluciones_hormigas[i]
            valor = valores_hormigas[i]
            delta_tau = q / valor if (objetivo == "Minimizar" and valor > 0) else q * valor
            
            for j in range(dim - 1):
                origen = ruta[j]
                destino = ruta[j+1]
                feromonas[origen][destino] += delta_tau
                feromonas[destino][origen] += delta_tau 
            
            feromonas[ruta[-1]][ruta[0]] += delta_tau
            feromonas[ruta[0]][ruta[-1]] += delta_tau

        if not mejora:
            sin_mejora += 1

        yield {
            "iteracion": gen,
            "total": gens,
            "mejor_valor": float(mejor_global_val),
            "mejor_sol": mejor_global_sol,
            "historial": historial,
            "terminado": False,
            "motivo": ""
        }

        if sin_mejora >= 30 and gen > 30:
            motivo_termino = f"Convergencia temprana (sin mejora por {sin_mejora} iteraciones)"
            break

    yield {
        "iteracion": gen,
        "total": gens,
        "mejor_valor": float(mejor_global_val),
        "mejor_sol": mejor_global_sol,
        "historial": historial,
        "terminado": True,
        "motivo": motivo_termino
    }