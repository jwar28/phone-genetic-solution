import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import io
import time
import json
from datetime import datetime
import plotly.express as px

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Solución Set Cover - Telefonía",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Solución al Problema de Set Cover para Telefonía")
st.write("""
Esta aplicación resuelve el problema de cobertura óptima de antenas para una empresa de telefonía.
El objetivo es cubrir a 500 clientes utilizando la menor cantidad de antenas posible, considerando el costo de cada antena.
""")

# Función para cargar los datos desde archivos subidos
@st.cache_data
def load_data(coverage_file, cost_file):
    # Cargar la matriz de cobertura
    coverage_data = pd.read_csv(coverage_file, header=None)
    coverage_matrix = coverage_data.values
    
    # Cargar los costos de las antenas
    if cost_file.name.endswith('.xlsx'):
        costs_df = pd.read_excel(cost_file, header=0)
        if costs_df.shape[1] > costs_df.shape[0]:
            costs_df = costs_df.T
    else:
        costs_df = pd.read_csv(cost_file, header=0)
    
    # Extraer la columna de costos
    costs = costs_df.iloc[1:, 0].values
    
    # Verificar dimensiones
    if len(costs) != coverage_matrix.shape[1]:
        st.error(f"Error: El número de costos ({len(costs)}) no coincide con el número de antenas ({coverage_matrix.shape[1]})")
        return None, None
    
    return coverage_matrix, costs

# Función de fitness optimizada
def evaluate_fitness(individual, coverage_matrix, costs):
    try:
        # Convertir individual a array numpy para operaciones vectoriales
        selected = np.array(individual, dtype=bool)
        
        # Calcular costo total usando operaciones vectoriales
        total_cost = np.sum(costs[selected])
        
        # Calcular cobertura usando operaciones vectoriales
        covered = np.any(coverage_matrix[:, selected], axis=1)
        
        # Penalización si no se cubren todos los clientes
        if not np.all(covered):
            return float('inf'),
        
        return float(total_cost),
    except Exception as e:
        st.error(f"Error en la evaluación del fitness: {str(e)}")
        return float('inf'),

# Función principal del algoritmo genético optimizada
def genetic_algorithm(coverage_matrix, costs, pop_size=100, n_gen=50, crossover_prob=0.7, mutation_prob=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n_antennas = coverage_matrix.shape[1]
    
    # Configurar el algoritmo genético
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=n_antennas)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_fitness, 
                     coverage_matrix=coverage_matrix, costs=costs)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Crear población inicial
    pop = toolbox.population(n=pop_size)
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Hall of Fame
    hof = tools.HallOfFame(1)
    
    # Ejecutar algoritmo
    result, logbook = algorithms.eaSimple(pop, toolbox, 
                                        cxpb=crossover_prob, 
                                        mutpb=mutation_prob,
                                        ngen=n_gen, 
                                        stats=stats, 
                                        halloffame=hof,
                                        verbose=True)
    
    return result, logbook, hof

# Función para calcular métricas de eficiencia
def calculate_efficiency_metrics(selected, coverage_matrix, costs):
    selected = np.array(selected, dtype=bool)
    coverage_per_antenna = np.sum(coverage_matrix[:, selected], axis=0)
    cost_per_client = costs[selected] / coverage_per_antenna
    
    return {
        'coverage_per_antenna': coverage_per_antenna,
        'cost_per_client': cost_per_client,
        'total_cost': np.sum(costs[selected]),
        'total_coverage': np.sum(coverage_per_antenna),
        'num_antennas': np.sum(selected)
    }

# Función para guardar solución
def save_solution(solution_data):
    if 'saved_solutions' not in st.session_state:
        st.session_state.saved_solutions = []
    
    st.session_state.saved_solutions.append(solution_data)

# Interfaz principal
def main():
    # Sección para cargar archivos
    with st.expander("Cargar archivos de datos", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            coverage_file = st.file_uploader("Archivo de matriz de cobertura (CSV)", type=["csv"])
        
        with col2:
            cost_file = st.file_uploader("Archivo de costos de antenas (XLSX/CSV)", type=["xlsx", "csv"])

    if coverage_file and cost_file:
        try:
            with st.spinner("Cargando datos y preparando modelo..."):
                coverage_matrix, costs = load_data(coverage_file, cost_file)
            
            if coverage_matrix is not None and costs is not None:
                st.success(f"Datos cargados correctamente: {coverage_matrix.shape[0]} clientes y {coverage_matrix.shape[1]} antenas potenciales")
                
                # Botón para verificar costos
                if st.button("Verificar primeros 5 costos"):
                    st.write("Primeros 5 costos:")
                    costs_df = pd.DataFrame({
                        'Índice': range(1, 6),
                        'Costo': costs[:5]
                    })
                    st.dataframe(costs_df, use_container_width=True)
                
                # Parámetros del algoritmo
                st.subheader("Parámetros del Algoritmo Genético")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    pop_size = st.slider("Tamaño de población", 50, 500, 100)
                
                with col2:
                    n_generations = st.slider("Número de generaciones", 10, 200, 50)
                
                with col3:
                    crossover_prob = st.slider("Probabilidad de cruce", 0.1, 0.9, 0.7)
                    mutation_prob = st.slider("Probabilidad de mutación", 0.01, 0.5, 0.2)
                
                with col4:
                    seed = st.number_input("Semilla aleatoria", value=42, help="Usar la misma semilla para reproducir resultados")
                
                if st.button("Resolver el problema de Set Cover"):
                    with st.spinner("Ejecutando algoritmo genético..."):
                        start_time = time.time()
                        result, logbook, hof = genetic_algorithm(
                            coverage_matrix, 
                            costs, 
                            pop_size=pop_size, 
                            n_gen=n_generations,
                            crossover_prob=crossover_prob,
                            mutation_prob=mutation_prob,
                            seed=seed
                        )
                        execution_time = time.time() - start_time
                    
                    # Obtener mejor solución
                    best_solution = hof[0]
                    
                    # Calcular métricas usando operaciones vectoriales
                    selected = np.array(best_solution, dtype=bool)
                    covered = np.any(coverage_matrix[:, selected], axis=1)
                    
                    total_cost = np.sum(costs[selected])
                    num_antennas_used = np.sum(selected)
                    num_clients_covered = np.sum(covered)
                    
                    # Mostrar resultados
                    st.header("Resultados del Algoritmo Genético")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total de antenas usadas", f"{int(num_antennas_used)}/{coverage_matrix.shape[1]}")
                    col2.metric("Costo total", f"${float(total_cost):,.2f}")
                    col3.metric("Clientes cubiertos", f"{int(num_clients_covered)}/{coverage_matrix.shape[0]}")
                    col4.metric("Tiempo de ejecución", f"{float(execution_time):.2f} segundos")

                    # Gráfico de torta de antenas utilizadas vs no utilizadas
                    st.subheader("Distribución de Antenas")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    labels = ['Antenas Utilizadas', 'Antenas No Utilizadas']
                    sizes = [num_antennas_used, coverage_matrix.shape[1] - num_antennas_used]
                    colors = ['#5cb85c', '#d9534f']
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    # Gráfico de evolución del fitness
                    st.subheader("Evolución del costo durante la optimización")
                    
                    # Obtener datos del logbook
                    gen = logbook.select("gen")
                    avg_values = [stats['avg'] for stats in logbook]
                    min_values = [stats['min'] for stats in logbook]

                    # Rellenar avg_values con np.nan si es más corto que min_values
                    if len(avg_values) < len(min_values):
                        avg_values += [np.nan] * (len(min_values) - len(avg_values))
                    elif len(min_values) < len(avg_values):
                        min_values += [np.nan] * (len(avg_values) - len(min_values))

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(gen, avg_values, label='Costo promedio', alpha=0.6, s=50, color='#5cb85c')
                    ax.scatter(gen, min_values, label='Costo mínimo', alpha=0.6, s=50, color='#d9534f')
                    ax.set_xlabel("Generación")
                    ax.set_ylabel("Costo")
                    ax.legend()
                    ax.grid(True)
                    
                    st.pyplot(fig)

                    # Tabla de comparación entre costo promedio y costo mínimo
                    comparacion_df = pd.DataFrame({
                        'Generación': gen,
                        'Costo promedio': avg_values,
                        'Costo mínimo': min_values
                    })
                    # Reemplazar np.nan e Infinity por 'inf' en la tabla
                    comparacion_df = comparacion_df.replace([np.nan, np.inf, float('inf'), 'Infinity'], 'inf')
                    st.subheader("Comparación de costos por generación")
                    st.dataframe(comparacion_df, use_container_width=True)
                    
                    # Análisis de eficiencia de costo
                    st.subheader("Análisis de Eficiencia de Costo")
                    efficiency_metrics = calculate_efficiency_metrics(selected, coverage_matrix, costs)

                    # Gráfico interactivo con Plotly
                    selected_indices = np.where(selected)[0] + 1
                    fig = px.scatter(
                        x=efficiency_metrics['coverage_per_antenna'],
                        y=efficiency_metrics['cost_per_client'],
                        hover_name=selected_indices,
                        labels={
                            'x': 'Número de Clientes Cubiertos por Antena',
                            'y': 'Costo por Cliente'
                        },
                        title='Relación Costo-Cobertura por Antena',
                    )
                    fig.update_traces(marker=dict(size=12, color='#5cb85c', line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar antenas seleccionadas con métricas de eficiencia
                    st.subheader("Antenas seleccionadas")
                    
                    # Crear DataFrame con las antenas seleccionadas y sus métricas
                    selected_indices = np.where(selected)[0] + 1
                    if len(selected_indices) > 0:
                        selected_antennas_df = pd.DataFrame({
                            'Antena': selected_indices,
                            'Costo': costs[selected_indices - 1],
                            'Clientes Cubiertos': efficiency_metrics['coverage_per_antenna'],
                            'Costo por Cliente': efficiency_metrics['cost_per_client']
                        })
                        
                        # Ordenar por costo descendente
                        selected_antennas_df = selected_antennas_df.sort_values(by='Costo', ascending=False)
                        
                        # Mostrar tabla
                        st.dataframe(selected_antennas_df, use_container_width=True)
                        
                        # Opción para descargar la solución
                        solution_df = pd.DataFrame({
                            'Antena': range(1, coverage_matrix.shape[1] + 1),
                            'Seleccionada': best_solution,
                            'Costo': costs
                        })
                        
                        csv = solution_df.to_csv(index=False)
                        st.download_button(
                            label="Descargar solución como CSV",
                            data=csv,
                            file_name="solucion_set_cover.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No se encontraron antenas seleccionadas en la solución.")
                
        except Exception as e:
            st.error(f"Error al procesar los datos: {str(e)}")
    else:
        st.info("Por favor, sube los archivos de cobertura (CSV) y costos (Excel) para comenzar.")

    # Formulación matemática del problema
    with st.expander("Formulación matemática del Set Cover"):
        st.write("""
        ## Modelo de Programación Lineal: Set Cover
        
        ### Variables de decisión:
        - $x_j = 1$ si la antena $j$ es seleccionada, $0$ en caso contrario
        
        ### Función objetivo:
        Minimizar el costo total de las antenas seleccionadas:
        
        $$\min \sum_{j=1}^{n} c_j x_j$$
        
        donde $c_j$ es el costo de la antena $j$.
        
        ### Restricciones:
        Cada cliente debe ser cubierto por al menos una antena:
        
        $$\sum_{j \in N_i} x_j \geq 1, \quad \forall i \in \{1, 2, \ldots, m\}$$
        
        donde $N_i$ es el conjunto de antenas que cubren al cliente $i$.
        
        ### Dominio de las variables:
        $$x_j \in \{0, 1\}, \quad \forall j \in \{1, 2, \ldots, n\}$$
        
        ### Algoritmo Genético:
        Para resolver este problema, utilizamos un algoritmo genético con:
        - Representación binaria: cada gen representa una antena
        - Función de fitness: costo total + penalización por clientes no cubiertos
        - Operadores genéticos: cruce de dos puntos y mutación flip bit
        - Selección por torneo
        """)

    # Barra lateral con información
    st.sidebar.title("Información")
    st.sidebar.info("""
    Esta aplicación resuelve el problema de optimización de ubicación de antenas para una empresa de telefonía utilizando un modelo de Set Cover.

    **Objetivo:** Minimizar el costo total de las antenas seleccionadas asegurando que todos los clientes sean cubiertos.

    **Método:** Algoritmo genético para encontrar una solución aproximada al problema de Set Cover.

    **Archivos necesarios:**
    - Matriz de cobertura (CSV): Define qué clientes cubre cada antena.
    - Costos de antenas (XLSX/CSV): Costos asociados a cada antena.
    """)

    st.sidebar.subheader("Sobre el modelo")
    st.sidebar.write("""
    El problema de Set Cover es NP-duro, por lo que utilizamos un algoritmo genético para encontrar una solución aproximada de buena calidad en un tiempo razonable.

    El algoritmo genético evoluciona una población de soluciones candidatas a lo largo de varias generaciones, aplicando selección natural, cruce y mutación para mejorar progresivamente la calidad de las soluciones.
    """)

if __name__ == "__main__":
    main()
