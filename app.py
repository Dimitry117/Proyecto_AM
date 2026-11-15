import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import io
import time
from problemas import CATALOGO_PROBLEMAS
from algoritmos import ejecutar_ga  # Aquí importarás luego ejecutar_pso, ejecutar_de, ejecutar_aco, ejecutar_ais

# Configuración general de la página
st.set_page_config(page_title="Entorno Metaheurístico", layout="wide")

# ==========================================
# 1. CONTROL DE ESTADO (Para botones Ejecutar, Detener, Reiniciar, Regresar)
# ==========================================
if "paso" not in st.session_state:
    st.session_state.paso = 1  # 1: Selección, 2: Configuración y Ejecución
if "ejecutando" not in st.session_state:
    st.session_state.ejecutando = False
if "historial_estados" not in st.session_state:
    st.session_state.historial_estados = []
if "config_usada" not in st.session_state:
    st.session_state.config_usada = {}

# Matriz de compatibilidad (Sección 4 del PDF: deshabilitar o explicar combinaciones inválidas)
COMPATIBILIDAD = {
    "GA": ["Continuo", "Binario", "Categorico"],
    "PSO": ["Continuo"],
    "DE": ["Continuo"],
    "ACO": ["Categorico", "Binario"],
    "AIS": ["Continuo", "Binario"]
}

st.title("🧬 Entorno Interactivo para Algoritmos Metaheurísticos")
st.markdown("---")

# ==========================================
# PANTALLA 1: SELECCIÓN DE PROBLEMA Y ALGORITMO
# ==========================================
if st.session_state.paso == 1:
    st.header("Paso 1: Selección de Problema y Algoritmo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tipo_prob = st.selectbox(
            "1. Tipo de Problema",
            ["Continuo", "Binario", "Categorico"],
            help="Continuo: vectores reales | Binario: ceros y unos | Categórico: permutaciones/rutas"
        )
        
        opciones_prob = {v["nombre"]: k for k, v in CATALOGO_PROBLEMAS[tipo_prob].items()}
        nombre_prob = st.selectbox("2. Problema Específico", list(opciones_prob.keys()))
        prob_key = opciones_prob[nombre_prob]
        prob_info = CATALOGO_PROBLEMAS[tipo_prob][prob_key]

    with col2:
        algo = st.selectbox(
            "3. Algoritmo Metaheurístico",
            ["GA", "ACO", "PSO", "AIS", "DE"],
            format_func=lambda x: {
                "GA": "Algoritmo Genético (GA)",
                "ACO": "Colonia de Hormigas (ACO)",
                "PSO": "Enjambre de Partículas (PSO)",
                "AIS": "Sistema Inmune Artificial (AIS)",
                "DE": "Evolución Diferencial (DE)"
            }[x]
        )
        
        objetivo = st.radio(
            "4. Objetivo de Optimización",
            ["Minimizar", "Maximizar"],
            index=0 if prob_info["objetivo_sugerido"] == "Minimizar" else 1,
            horizontal=True
        )

    # Validación de compatibilidad (Sección 4 y 6 del PDF)
    es_compatible = tipo_prob in COMPATIBILIDAD[algo]
    if not es_compatible:
        st.error(
            f"⚠️ **Combinación Incompatible:** El algoritmo **{algo}** estándar no opera sobre representaciones de tipo **{tipo_prob}**. "
            f"Este algoritmo admite únicamente: {', '.join(COMPATIBILIDAD[algo])}."
        )
    else:
        st.success(f"✅ Combinación válida: **{algo}** aplicándose a **{nombre_prob}** ({objetivo}).")

    if st.button("Continuar a Configuración ➡️", disabled=not es_compatible, type="primary"):
        st.session_state.seleccion = {
            "tipo_prob": tipo_prob,
            "prob_info": prob_info,
            "algo": algo,
            "objetivo": objetivo
        }
        st.session_state.paso = 2
        st.rerun()

# ==========================================
# PANTALLA 2: CONFIGURACIÓN DINÁMICA Y EJECUCIÓN
# ==========================================
elif st.session_state.paso == 2:
    sel = st.session_state.seleccion
    tipo_prob = sel["tipo_prob"]
    prob_info = sel["prob_info"]
    algo = sel["algo"]
    objetivo = sel["objetivo"]

    # Botón Regresar (Sección 6 del PDF)
    if st.button("⬅️ Regresar a Selección"):
        st.session_state.paso = 1
        st.session_state.ejecutando = False
        st.session_state.historial_estados = []
        st.rerun()

    st.header(f"Paso 2: Configuración y Ejecución ({algo} - {prob_info['nombre']})")

    # --- BARRA LATERAL: PARÁMETROS DINÁMICOS Y REPRODUCIBILIDAD ---
    with st.sidebar:
        st.subheader("⚙️ Parámetros del Problema")
        semilla = st.number_input("Semilla Aleatoria (Seed)", min_value=0, max_value=99999, value=42, help="Garantiza experimentos reproducibles (Sección 9).")
        
        if prob_info["nombre"] in ["TSP (Agente Viajero - 6 Ciudades)", "Problema de la Mochila"]:
            dim = prob_info["dim_default"]
            st.info(f"Dimensión fija para este problema: **{dim}**")
        else:
            dim = st.number_input("Dimensión del vector", min_value=2, max_value=50, value=prob_info["dim_default"])

        lim_inf, lim_sup = prob_info["dominio"]
        if tipo_prob == "Continuo":
            col_l1, col_l2 = st.columns(2)
            lim_inf = col_l1.number_input("Límite Inferior", value=float(lim_inf))
            lim_sup = col_l2.number_input("Límite Superior", value=float(lim_sup))
            if lim_inf >= lim_sup:
                st.error("El límite inferior debe ser menor al superior.")

        st.subheader(f"🎛️ Parámetros de {algo}")
        iteraciones = st.slider("Número de Iteraciones / Generaciones", 10, 500, 100)
        params = {"iteraciones": iteraciones, "dim": dim, "semilla": semilla}

        # Controles dinámicos según algoritmo (Sección 5 del PDF)
        if algo == "GA":
            params["poblacion"] = st.slider("Tamaño de Población", 10, 300, 50)
            params["tasa_mutacion"] = st.slider("Tasa de Mutación", 0.01, 1.0, 0.10)
        elif algo == "PSO":
            params["particulas"] = st.slider("Número de Partículas", 10, 300, 40)
            params["inercia"] = st.slider("Inercia (w)", 0.1, 1.2, 0.7)
            params["c1"] = st.slider("Coeficiente Cognitivo (c1)", 0.5, 2.5, 1.5)
            params["c2"] = st.slider("Coeficiente Social (c2)", 0.5, 2.5, 1.5)
        elif algo == "DE":
            params["poblacion"] = st.slider("Tamaño de Población (NP)", 10, 300, 50)
            params["F"] = st.slider("Factor de Mutación (F)", 0.1, 2.0, 0.8)
            params["CR"] = st.slider("Tasa de Cruza (CR)", 0.0, 1.0, 0.9)
        elif algo == "ACO":
            params["hormigas"] = st.slider("Número de Hormigas", 5, 200, 30)
            params["evaporacion"] = st.slider("Tasa de Evaporación (rho)", 0.01, 0.99, 0.5)
            params["alfa"] = st.slider("Influencia de Feromona (alfa)", 0.5, 5.0, 1.0)
        elif algo == "AIS":
            params["anticuerpos"] = st.slider("Número de Anticuerpos", 10, 200, 40)
            params["factor_clonacion"] = st.slider("Tasa de Clonación (beta)", 0.1, 2.0, 0.5)

    # --- BOTONES DE CONTROL (Ejecutar, Detener, Reiniciar - Sección 6) ---
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    datos_validos = (tipo_prob != "Continuo") or (lim_inf < lim_sup)

    if b1.button("▶️ Ejecutar", disabled=not datos_validos, type="primary"):
        np.random.seed(semilla)
        st.session_state.ejecutando = True
        st.session_state.historial_estados = []
        st.session_state.config_usada = {"algoritmo": algo, "problema": prob_info["nombre"], "objetivo": objetivo, **params}

    if b2.button("⏹️ Detener"):
        st.session_state.ejecutando = False

    if b3.button("🔄 Reiniciar"):
        st.session_state.ejecutando = False
        st.session_state.historial_estados = []
        st.rerun()

    # Contenedores dinámicos para visualización en tiempo real (Sección 7)
    barra_progreso = st.empty()
    metricas_cont = st.empty()
    graficas_cont = st.empty()

    # --- BUCLE DE EJECUCIÓN EN TIEMPO REAL ---
    if st.session_state.ejecutando:
        if algo == "GA":
            generador = ejecutar_ga(prob_info["fn"], tipo_prob, objetivo, [lim_inf, lim_sup], dim, params)
        else:
            st.warning(f"Conecta aquí `ejecutar_{algo.lower()}()` desde `algoritmos.py` para correr la búsqueda.")
            st.session_state.ejecutando = False
            st.stop()

        for estado in generador:
            if not st.session_state.ejecutando:
                estado["motivo"] = "Detenido manualmente por el usuario"
                st.session_state.historial_estados.append(estado)
                break
            
            st.session_state.historial_estados.append(estado)
            
            # Actualizar barra y métricas en tiempo real
            prog = estado["iteracion"] / estado["total"]
            barra_progreso.progress(prog, text=f"Iteración {estado['iteracion']} de {estado['total']}")
            
            with metricas_cont.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Iteración Actual", f"{estado['iteracion']} / {estado['total']}")
                m2.metric("Mejor Valor Objetivo", f"{estado['mejor_valor']:.6f}")
                m3.metric("Objetivo", objetivo)

            time.sleep(0.01) # Pausa mínima para refrescar pantalla

        st.session_state.ejecutando = False

    # --- VISUALIZACIÓN DE RESULTADOS Y LOS 4 MOMENTOS CLAVE (Secciones 7 y 8) ---
    if st.session_state.historial_estados:
        estados = st.session_state.historial_estados
        final = estados[-1]
        total_exec = len(estados)

        st.success(f"🏁 **Ejecución finalizada** | Motivo de término: **{final.get('motivo', 'Completado')}**")

        # Selector de los 4 momentos obligatorios (Iteración 0, ~1/3, ~2/3, Final)
        idx_0 = 0
        idx_1_3 = max(0, total_exec // 3)
        idx_2_3 = max(0, (2 * total_exec) // 3)
        idx_fin = total_exec - 1

        momentos = {
            f"Iteración Inicial (Iter {estados[idx_0]['iteracion']})": idx_0,
            f"Aprox. 1/3 del proceso (Iter {estados[idx_1_3]['iteracion']})": idx_1_3,
            f"Aprox. 2/3 del proceso (Iter {estados[idx_2_3]['iteracion']})": idx_2_3,
            f"Iteración Final (Iter {estados[idx_fin]['iteracion']})": idx_fin,
        }

        momento_sel = st.select_slider(
            "⏱️ Inspeccionar momentos de la evolución (Sección 7):",
            options=list(momentos.keys()),
            value=list(momentos.keys())[-1]
        )
        estado_ver = estados[momentos[momento_sel]]

        # Gráficas lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 1. Convergencia hasta el momento seleccionado
        axes[0].plot(estado_ver["historial"], color="royalblue", linewidth=2)
        axes[0].set_title(f"Convergencia hasta Iteración {estado_ver['iteracion']}")
        axes[0].set_xlabel("Iteración")
        axes[0].set_ylabel("Fitness")
        axes[0].grid(True, alpha=0.4)

        # 2. Representación visual de la solución en ese momento
        sol = estado_ver["mejor_sol"]
        if tipo_prob == "Continuo":
            axes[1].bar(range(len(sol)), sol, color="teal")
            axes[1].set_title("Vector Solución (Continuo)")
        elif tipo_prob == "Binario":
            axes[1].imshow([sol], cmap="Greys", aspect="auto")
            axes[1].set_yticks([])
            axes[1].set_title("Vector Binario (Negro=1, Blanco=0)")
        elif tipo_prob == "Categorico":
            n = len(sol)
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            coords = np.column_stack([np.cos(ang), np.sin(ang)])
            ruta = np.append(sol, sol[0]).astype(int)
            axes[1].plot(coords[ruta, 0], coords[ruta, 1], 'o-', color='crimson')
            for i, (cx, cy) in enumerate(coords):
                axes[1].text(cx*1.12, cy*1.12, f"C{i}", ha='center', va='center', fontweight='bold')
            axes[1].set_title("Ruta TSP")
            axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

        # Reporte detallado y configuración utilizada (Sección 8)
        with st.expander("📋 Ver Reporte Completo y Configuración Utilizada", expanded=True):
            c_rep1, c_rep2 = st.columns(2)
            c_rep1.write("**Mejor Solución Encontrada:**")
            c_rep1.code(np.round(final["mejor_sol"], 5).tolist())
            c_rep1.write(f"**Fitness Final:** `{final['mejor_valor']:.6f}`")
            c_rep1.write(f"**Iteraciones Ejecutadas:** `{final['iteracion']}`")
            
            c_rep2.write("**Configuración Utilizada:**")
            c_rep2.json(st.session_state.config_usada)

        # Botones de Exportación CSV y JSON (Sección 8)
        col_exp1, col_exp2, _ = st.columns([1, 1, 3])
        
        resumen_json = json.dumps({
            "configuracion": st.session_state.config_usada,
            "mejor_fitness": final["mejor_valor"],
            "mejor_solucion": final["mejor_sol"].tolist(),
            "motivo_termino": final.get("motivo", ""),
            "historial": final["historial"]
        }, indent=4)

        col_exp1.download_button(
            label="📥 Exportar JSON",
            data=resumen_json,
            file_name=f"resultado_{algo}_{prob_info['nombre']}.json",
            mime="application/json"
        )

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Iteracion", "Mejor_Fitness"])
        for i, val in enumerate(final["historial"], 1):
            writer.writerow([i, val])

        col_exp2.download_button(
            label="📥 Exportar CSV",
            data=csv_buffer.getvalue(),
            file_name=f"convergencia_{algo}_{prob_info['nombre']}.csv",
            mime="text/csv"
        )