import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import random
from deap import base, creator, tools, algorithms

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

# Función para cargar y procesar los datos
@st.cache_data
def load_data(coverage_file, cost_file):
    # Cargar la matriz de cobertura
    coverage_data = pd.read_csv(coverage_file, header=None)
    coverage_matrix = coverage_data.values
    
    # Cargar los costos de las antenas
    if cost_file.name.endswith('.xlsx'):
        costs_df = pd.read_excel(cost_file)
    else:
        costs_df = pd.read_csv(cost_file)
    
    # Extraer la columna de costos
    costs = costs_df.iloc[:, 1].values
    
    return coverage_matrix, costs

# Sección para cargar archivos
with st.expander("Cargar archivos de datos", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        coverage_file = st.file_uploader("Archivo de matriz de cobertura (CSV)", type=["csv"])
    
    with col2:
        cost_file = st.file_uploader("Archivo de costos de antenas (XLSX/CSV)", type=["xlsx", "csv"])

# Función para resolver el problema con algoritmo genético
def solve_set_cover_with_ga(coverage_matrix, costs, pop_size=100, n_generations=50, crossover_prob=0.7, mutation_prob=0.2):
    n_antennas = coverage_matrix.shape[1]
    n_clients = coverage_matrix.shape[0]
    
    # Definir el problema como minimización
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Atributo: 0 o 1 (antena no usada o usada)
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Individuo: lista de 0s y 1s de longitud n_antennas
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_antennas)
    
    # Población: lista de individuos
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Función de evaluación
    def evaluate(individual):
        selected_antennas = np.array(individual)
        
        # Verificar cobertura de todos los clientes
        covered_clients = np.zeros(n_clients, dtype=bool)
        for i in range(n_antennas):
            if selected_antennas[i] == 1:
                covered_clients = covered_clients | (coverage_matrix[:, i] == 1)
        
        # Calcular costo total
        total_cost = np.sum(selected_antennas * costs)
        
        # Penalización si no todos los clientes están cubiertos
        coverage_penalty = 1000000 * (n_clients - np.sum(covered_clients))
        
        return total_cost + coverage_penalty,
    
    toolbox.register("evaluate", evaluate)
    
    # Operadores genéticos
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Inicializar población
    pop = toolbox.population(n=pop_size)
    
    # Estadísticas para seguimiento
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Ejecutar algoritmo genético
    hof = tools.HallOfFame(1)
    result, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, 
                               ngen=n_generations, stats=stats, halloffame=hof, verbose=False)
    
    # Obtener la mejor solución
    best_solution = hof[0]
    
    # Verificar cobertura y calcular métricas
    selected_antennas = np.array(best_solution)
    covered_clients = np.zeros(n_clients, dtype=bool)
    for i in range(n_antennas):
        if selected_antennas[i] == 1:
            covered_clients = covered_clients | (coverage_matrix[:, i] == 1)
    
    # Calcular métricas de la solución
    total_cost = np.sum(selected_antennas * costs)
    num_antennas_used = np.sum(selected_antennas)
    num_clients_covered = np.sum(covered_clients)
    
    # Obtener índices de las antenas seleccionadas
    selected_antenna_indices = np.where(selected_antennas == 1)[0]
    
    return {
        'solution': best_solution,
        'total_cost': total_cost,
        'num_antennas_used': num_antennas_used,
        'num_clients_covered': num_clients_covered,
        'selected_antenna_indices': selected_antenna_indices + 1,  # +1 para índices desde 1
        'log': log
    }

# Procesar los datos y resolver el problema
if coverage_file and cost_file:
    try:
        with st.spinner("Cargando datos y preparando modelo..."):
            coverage_matrix, costs = load_data(coverage_file, cost_file)
        
        st.success(f"Datos cargados correctamente: {coverage_matrix.shape[0]} clientes y {coverage_matrix.shape[1]} antenas potenciales")
        
        # Mostrar parámetros del algoritmo genético
        st.subheader("Parámetros del Algoritmo Genético")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pop_size = st.slider("Tamaño de población", 50, 500, 100)
        
        with col2:
            n_generations = st.slider("Número de generaciones", 10, 200, 50)
        
        with col3:
            crossover_prob = st.slider("Probabilidad de cruce", 0.1, 0.9, 0.7)
            mutation_prob = st.slider("Probabilidad de mutación", 0.01, 0.5, 0.2)
        
        # Botón para iniciar la resolución
        if st.button("Resolver el problema de Set Cover"):
            with st.spinner("Ejecutando algoritmo genético..."):
                start_time = time.time()
                solution_info = solve_set_cover_with_ga(
                    coverage_matrix, 
                    costs, 
                    pop_size=pop_size, 
                    n_generations=n_generations, 
                    crossover_prob=crossover_prob, 
                    mutation_prob=mutation_prob
                )
                execution_time = time.time() - start_time
            
            # Mostrar resultados
            st.header("Resultados del Algoritmo Genético")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total de antenas usadas", f"{solution_info['num_antennas_used']}/{coverage_matrix.shape[1]}")
            col2.metric("Costo total", f"${solution_info['total_cost']:,.2f}")
            col3.metric("Clientes cubiertos", f"{solution_info['num_clients_covered']}/{coverage_matrix.shape[0]}")
            col4.metric("Tiempo de ejecución", f"{execution_time:.2f} segundos")
            
            # Gráfico de evolución del fitness
            st.subheader("Evolución del costo durante la optimización")
            
            # Extraer datos de log para gráfico
            gen = range(len(solution_info['log']))
            avg_values = [record['avg'] for record in solution_info['log']]
            min_values = [record['min'] for record in solution_info['log']]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(gen, avg_values, label='Costo promedio')
            ax.plot(gen, min_values, label='Costo mínimo')
            ax.set_xlabel("Generación")
            ax.set_ylabel("Costo")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Mostrar antenas seleccionadas
            st.subheader("Antenas seleccionadas")
            
            # Crear un DataFrame con las antenas seleccionadas y sus costos
            selected_antennas_df = pd.DataFrame({
                'Antena': solution_info['selected_antenna_indices'],
                'Costo': costs[solution_info['selected_antenna_indices'] - 1]
            })
            
            # Ordenar por costo descendente
            selected_antennas_df = selected_antennas_df.sort_values(by='Costo', ascending=False)
            
            # Mostrar las antenas seleccionadas en una tabla
            st.dataframe(selected_antennas_df, use_container_width=True)
            
            # Visualización de la proporción entre antenas usadas y no usadas
            st.subheader("Proporción de antenas utilizadas")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie([solution_info['num_antennas_used'], coverage_matrix.shape[1] - solution_info['num_antennas_used']],
                  labels=['Utilizadas', 'No utilizadas'],
                  autopct='%1.1f%%',
                  colors=['#5cb85c', '#d9534f'])
            ax.axis('equal')
            
            st.pyplot(fig)
            
            # Opción para descargar la solución
            solution_df = pd.DataFrame({
                'Antena': range(1, coverage_matrix.shape[1] + 1),
                'Seleccionada': solution_info['solution'],
                'Costo': costs
            })
            
            csv = solution_df.to_csv(index=False)
            st.download_button(
                label="Descargar solución como CSV",
                data=csv,
                file_name="solucion_set_cover.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")

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