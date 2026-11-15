import sys
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from problemas import CATALOGO_PROBLEMAS, DISTANCIAS_TSP
from algoritmos import ejecutar_ga, ejecutar_pso, ejecutar_aco

def pedir_numero(mensaje, tipo_dato=int, minimo=None, maximo=None, default=None):
    """Valida tipo y rango de entrada del usuario."""
    while True:
        entrada = input(f"{mensaje} [Default: {default}]: ").strip()
        if entrada == "" and default is not None:
            return default
        try:
            valor = tipo_dato(entrada)
            if minimo is not None and valor < minimo:
                print(f"  [!] Error: El valor debe ser >= {minimo}.")
                continue
            if maximo is not None and valor > maximo:
                print(f"  [!] Error: El valor debe ser <= {maximo}.")
                continue
            return valor
        except ValueError:
            print(f"  [!] Error: Ingresa un número válido ({tipo_dato.__name__}).")

def configurar_parametros_dinamicos(algo, tipo_prob, prob_info):
    """Pide parámetros específicos según el algoritmo y problema seleccionados."""
    print(f"\n--- 4. Configuración Dinámica ({algo} para {prob_info['nombre']}) ---")
    params = {}
    
    # Dimensión (fija para mochila/TSP de ejemplo, configurable en continuos/OneMax)
    if prob_info["nombre"] in ["TSP (Agente Viajero - 6 Ciudades)", "Problema de la Mochila"]:
        params["dim"] = prob_info["dim_default"]
        print(f"  * Dimensión fijada automáticamente en {params['dim']} para este problema.")
    else:
        params["dim"] = pedir_numero("  * Dimensión del vector (2 a 50)", int, 2, 50, prob_info["dim_default"])

    params["iteraciones"] = pedir_numero("  * Número de iteraciones/generaciones (10 a 1000)", int, 10, 1000, 100)

    if algo == "GA":
        params["poblacion"] = pedir_numero("  * Tamaño de población (10 a 500)", int, 10, 500, 50)
        params["tasa_mutacion"] = pedir_numero("  * Tasa de mutación (0.01 a 1.0)", float, 0.01, 1.0, 0.1)
    elif algo == "PSO":
        params["particulas"] = pedir_numero("  * Número de partículas (10 a 500)", int, 10, 500, 40)
        params["inercia"] = pedir_numero("  * Factor de inercia w (0.1 a 1.2)", float, 0.1, 1.2, 0.7)
        params["c1"] = pedir_numero("  * Coeficiente cognitivo c1 (0.5 a 2.5)", float, 0.5, 2.5, 1.5)
        params["c2"] = pedir_numero("  * Coeficiente social c2 (0.5 a 2.5)", float, 0.5, 2.5, 1.5)
    elif algo == "DE":
        params["poblacion"] = pedir_numero("  * Tamaño de población NP (10 a 500)", int, 10, 500, 50)
        params["F"] = pedir_numero("  * Factor de escala F (0.1 a 2.0)", float, 0.1, 2.0, 0.8)
        params["CR"] = pedir_numero("  * Probabilidad de cruce CR (0.0 a 1.0)", float, 0.0, 1.0, 0.9)
    elif algo == "ACO":
        params["hormigas"] = pedir_numero("  * Número de hormigas (5 a 200)", int, 5, 200, 30)
        params["evaporacion"] = pedir_numero("  * Tasa de evaporación rho (0.01 a 0.99)", float, 0.01, 0.99, 0.5)
        params["alfa"] = pedir_numero("  * Importancia de feromona alfa (0.5 a 5.0)", float, 0.5, 5.0, 1.0)
    elif algo == "AIS":
        params["anticuerpos"] = pedir_numero("  * Tamaño de población de anticuerpos (10 a 300)", int, 10, 300, 40)
        params["factor_clonacion"] = pedir_numero("  * Factor de clonación beta (0.1 a 2.0)", float, 0.1, 2.0, 0.5)

    return params

def graficar_y_mostrar_solucion(tipo_prob, prob_info, resultado):
    """Genera la curva de convergencia y la representación visual de la solución."""
    sol = resultado["mejor_sol"]

    # --- PROTECCIÓN PARA GRÁFICAS VACÍAS ---
    # Evita que el programa colapse si el algoritmo abortó y devolvió una solución vacía
    if len(sol) == 0:
        print("\n [!] No hay datos para graficar (solución vacía debida a incompatibilidad).")
        return
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Gráfica 1: Curva de convergencia
    axes[0].plot(resultado["historial"], color="royalblue", linewidth=2)
    axes[0].set_title(f"Convergencia - {prob_info['nombre']}")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Mejor Fitness")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Gráfica 2: Representación visual de la solución
    if tipo_prob == "Continuo":
        axes[1].bar(range(len(sol)), sol, color="teal")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Valores del Vector Solución")
        axes[1].set_xlabel("Dimensión (x_i)")
        axes[1].set_ylabel("Valor")
    elif tipo_prob == "Binario":
        axes[1].imshow([sol], cmap="Greys", aspect="auto")
        axes[1].set_yticks([])
        axes[1].set_xticks(range(len(sol)))
        axes[1].set_title("Vector Binario (Negro = 1, Blanco = 0)")
    elif tipo_prob == "Categorico":
        # Visualización de ruta TSP en círculo
        n = len(sol)
        angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angulos), np.sin(angulos)])
        
        ruta_cerrada = np.append(sol, sol[0]).astype(int)
        axes[1].plot(coords[ruta_cerrada, 0], coords[ruta_cerrada, 1], 'o-', color='crimson', markersize=10)
        for idx, (cx, cy) in enumerate(coords):
            axes[1].text(cx * 1.12, cy * 1.12, f"C{idx}", fontsize=11, ha='center', va='center', fontweight='bold')
        axes[1].set_title("Ruta Óptima Encontrada (TSP)")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def exportar_resultados(algo, prob_nombre, objetivo, params, resultado):
    """Exporta el reporte final a JSON o CSV."""
    op = input("\n¿Deseas exportar los resultados? (1: JSON | 2: CSV | Enter: Omitir): ").strip()
    if op not in ["1", "2"]:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Manejo seguro si la solución está vacía
    sol_lista = resultado["mejor_sol"].tolist() if isinstance(resultado["mejor_sol"], np.ndarray) else resultado["mejor_sol"]

    if op == "1":
        nombre_archivo = f"resultado_{algo}_{timestamp}.json"
        datos = {
            "algoritmo": algo,
            "problema": prob_nombre,
            "objetivo": objetivo,
            "parametros": params,
            "mejor_fitness": resultado["mejor_valor"],
            "mejor_solucion": sol_lista,
            "iteraciones_ejecutadas": resultado["iteracion"],
            "motivo_termino": resultado["motivo"],
            "historial_convergencia": resultado["historial"]
        }
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        print(f"  [OK] Guardado exitosamente en: {nombre_archivo}")

    elif op == "2":
        nombre_archivo = f"resultado_{algo}_{timestamp}.csv"
        with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Algoritmo", algo])
            writer.writerow(["Problema", prob_nombre])
            writer.writerow(["Objetivo", objetivo])
            writer.writerow(["Mejor_Fitness", resultado["mejor_valor"]])
            writer.writerow(["Mejor_Solucion", str(sol_lista)])
            writer.writerow(["Motivo_Termino", resultado["motivo"]])
            writer.writerow([])
            writer.writerow(["Iteracion", "Mejor_Fitness_Acumulado"])
            for idx, val in enumerate(resultado["historial"], start=1):
                writer.writerow([idx, val])
        print(f"  [OK] Guardado exitosamente en: {nombre_archivo}")

def main():
    while True:
        print("\n" + "=" * 60)
        print("  ENTORNO INTERACTIVO PARA ALGORITMOS METAHEURÍSTICOS")
        print("=" * 60)

        # 1. Tipo de Problema
        print("\n--- 1. Tipo de Problema ---")
        print("1. Problema Continuo (Sphere, Rastrigin, Rosenbrock, Ackley)")
        print("2. Problema Binario (OneMax, Mochila)")
        print("3. Problema Categórico / Discreto (TSP)")
        op_tipo = input("Selecciona una opción (1-3): ").strip()
        mapa_tipos = {"1": "Continuo", "2": "Binario", "3": "Categorico"}
        
        if op_tipo not in mapa_tipos:
            print("  [!] Opción inválida.")
            continue
        tipo_prob = mapa_tipos[op_tipo]

        # 1.5 Selección del problema específico
        print(f"\n--- Submenú: Problemas de tipo {tipo_prob} ---")
        sub_catalogo = CATALOGO_PROBLEMAS[tipo_prob]
        for k, v in sub_catalogo.items():
            print(f"{k}. {v['nombre']} (Sugerido: {v['objetivo_sugerido']})")
        op_prob = input("Elige el problema específico: ").strip()
        if op_prob not in sub_catalogo:
            print("  [!] Problema inválido.")
            continue
        prob_info = sub_catalogo[op_prob]

        # 2. Selección del Algoritmo
        print("\n--- 2. Selección del Algoritmo Metaheurístico ---")
        print("1. Algoritmo Genético (GA)")
        print("2. Colonia de Hormigas (ACO)")
        print("3. Enjambre de Partículas (PSO)")
        print("4. Sistema Inmune Artificial (AIS)")
        print("5. Evolución Diferencial (DE)")
        op_algo = input("Selecciona un algoritmo (1-5): ").strip()
        mapa_algos = {"1": "GA", "2": "ACO", "3": "PSO", "4": "AIS", "5": "DE"}
        algo = mapa_algos.get(op_algo, "GA")

        if algo in ["PSO", "DE"] and tipo_prob != "Continuo":
            print(f"\n  [AVISO] {algo} opera nativamente en espacios continuos. Se recomienda GA o ACO para {tipo_prob}.")

        # 3. Objetivo
        print("\n--- 3. Objetivo de Optimización ---")
        print(f"1. Minimizar | 2. Maximizar [Enter para sugerido: {prob_info['objetivo_sugerido']}]")
        op_obj = input("Elige una opción: ").strip()
        if op_obj == "1":
            objetivo = "Minimizar"
        elif op_obj == "2":
            objetivo = "Maximizar"
        else:
            objetivo = prob_info["objetivo_sugerido"]

        # 4. Configuración dinámica
        params = configurar_parametros_dinamicos(algo, tipo_prob, prob_info)

        # 5. Ejecución con visualización en tiempo real
        print("\n" + "-" * 50)
        print(f"Ejecutando {algo} en {prob_info['nombre']} ({objetivo})...")

        if algo == "GA":
            generador = ejecutar_ga(prob_info["fn"], tipo_prob, objetivo, prob_info["dominio"], params["dim"], params)
        elif algo == "PSO":
            generador = ejecutar_pso(prob_info["fn"], tipo_prob, objetivo, prob_info["dominio"], params["dim"], params)
        elif algo == "ACO":
            # --- CORRECCIÓN --- 
            # Inyectamos la matriz de distancias importada para que las hormigas puedan usarla
            if "TSP" in prob_info["nombre"]:
                params["matriz_distancias"] = DISTANCIAS_TSP
            generador = ejecutar_aco(prob_info["fn"], tipo_prob, objetivo, prob_info["dominio"], params["dim"], params)
        else:
            print(f"  [!] Nota: Conecta aquí tu función ejecutar_{algo.lower()}() en algoritmos.py.")
            continue

        resultado_final = None
        for estado in generador:
            sys.stdout.write(f"\r  Progreso: Iteración {estado['iteracion']}/{estado['total']} | Mejor Fitness: {estado['mejor_valor']:.6f}")
            sys.stdout.flush()
            time.sleep(0.02)
            resultado_final = estado

        # 6. Reporte Final
        print("\n\n" + "=" * 50)
        print(" RESULTADOS DE LA OPTIMIZACIÓN")
        print("=" * 50)
        print(f" * Mejor Fitness : {resultado_final['mejor_valor']:.6f}")
        # Manejo seguro por si devuelve una lista en lugar de array
        sol_print = np.round(resultado_final['mejor_sol'], 4) if isinstance(resultado_final['mejor_sol'], np.ndarray) else resultado_final['mejor_sol']
        print(f" * Mejor Solución: {sol_print}")
        print(f" * Motivo Término: {resultado_final['motivo']}")
        print("=" * 50)

        # 7. Gráficas y Exportación
        graficar_y_mostrar_solucion(tipo_prob, prob_info, resultado_final)
        exportar_resultados(algo, prob_info["nombre"], objetivo, params, resultado_final)

        if input("\n¿Deseas realizar otra ejecución? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main()