import matplotlib
matplotlib.use('Agg') # Usar el backend 'Agg' para no mostrar ventanas de matplotlib

from flask import Flask, render_template, jsonify
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq
import random
import io, base64, time
from collections import deque
import copy

app = Flask(__name__)

# === ESTADO GLOBAL ===
# Almacena el estado inicial del juego para reinicios
initial_state = {}
# Almacena el estado actual de cada algoritmo (A*, BFS, DFS, GA)
game_states = {
    'A*': {}, 'BFS': {}, 'DFS': {}, 'GA': {}
}

# Direcciones posibles (Arriba, Abajo, Izquierda, Derecha)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# Mapeo de índices a direcciones para el AG
DIRS_MAP = {
    0: (-1, 0), # Arriba
    1: (1, 0),  # Abajo
    2: (0, -1), # Izquierda
    3: (0, 1)   # Derecha
}
filas, columnas = 10, 10 # Dimensiones del tablero

# --- Parámetros para el Algoritmo Genético ---
POPULATION_SIZE = 500      # Tamaño de la población de individuos
GENERATIONS = 3000         # Número de generaciones a evolucionar
CHROMOSOME_LENGTH = 100    # Longitud del cromosoma (secuencia de movimientos)
MUTATION_RATE = 0.1        # Probabilidad de mutación por gen
ELITISM_COUNT = 50         # Número de los mejores individuos que pasan directamente a la siguiente generación
TOUR_SIZE = 5              # Tamaño del torneo para la selección de padres

# --- Funciones auxiliares para el Algoritmo Genético (GLOBAL SCOPE) ---

def create_individual():
    """Crea un individuo (cromosoma) con movimientos aleatorios."""
    return [random.randint(0, 3) for _ in range(CHROMOSOME_LENGTH)]

def select_parent(population_with_fitness, tour_size=TOUR_SIZE):
    """Selecciona un padre usando selección por torneo."""
    tournament = random.sample(population_with_fitness, min(tour_size, len(population_with_fitness)))
    return max(tournament, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    """Realiza el cruce de dos padres para crear dos hijos."""
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, rate=MUTATION_RATE):
    """Muta un individuo cambiando aleatoriamente algunos de sus genes."""
    for i in range(len(individual)):
        if random.random() < rate:
            individual[i] = random.randint(0, 3) # Cambia el gen a una dirección aleatoria
    return individual

# --- Función de Aptitud para el Algoritmo Genético ---
def evaluate_fitness(individual, initial_snake_eval, initial_apples_eval, initial_obstacles_eval):
    """
    Evalúa la aptitud de un individuo (secuencia de movimientos).
    Simula el camino del individuo y calcula una puntuación basada en:
    - Evitar colisiones (penalización muy alta)
    - Comer manzanas (recompensa muy alta)
    - Acercarse a las manzanas (recompensa)
    - Tomar pasos válidos (recompensa)
    - Manzanas no comidas (penalización alta)
    """
    sim_snake = list(initial_snake_eval) # Copia la serpiente inicial para la simulación
    sim_apples = sorted(list(initial_apples_eval), key=lambda a: (a[0], a[1])) 
    sim_obstacles = list(initial_obstacles_eval)
    
    head = sim_snake[-1] 
    score = 0
    apples_eaten_in_sim = 0
    
    score += 100 

    if sim_apples:
        score -= heuristica(head, sim_apples[0]) * 50 

    for i, move_idx in enumerate(individual):
        dx, dy = DIRS_MAP[move_idx]
        next_head = (head[0] + dx, head[1] + dy)

        if not (0 <= next_head[0] < filas and 0 <= next_head[1] < columnas):
            score -= 1000000 
            break
        
        if next_head in sim_obstacles:
            score -= 1000000 
            break
        
        if next_head in sim_snake[:-1] and next_head != sim_snake[0]: 
            score -= 1000000 
            break

        sim_snake.append(next_head)
        head = next_head

        apple_consumed_this_step = False
        if sim_apples and next_head in sim_apples:
            score += 5000000 
            sim_apples.remove(next_head)
            apples_eaten_in_sim += 1
            apple_consumed_this_step = True
            score += len(sim_snake) * 500 
        
        if not apple_consumed_this_step:
            sim_snake.pop(0) 
        
        if sim_apples: 
            current_target_apple = min(sim_apples, key=lambda apple: heuristica(head, apple))
            if len(sim_snake) >= 2: 
                dist_before = heuristica(sim_snake[-2], current_target_apple) 
                dist_after = heuristica(sim_snake[-1], current_target_apple)  
                score += (dist_before - dist_after) * 300 
        
        score += 50 

    if sim_apples:
        remaining_apples_penalty = sum(heuristica(head, apple) for apple in sim_apples)
        score -= remaining_apples_penalty * 1000 
    
    score += apples_eaten_in_sim * 10000000 

    return max(1, score) 

# Función de heurística (Distancia de Manhattan) para A*
def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Función para crear el mapa del juego para visualización
def crear_mapa(filas, columnas, serpiente, manzanas, obstaculos):
    mapa = [[' ' for _ in range(columnas)] for _ in range(filas)]
    for x, y in serpiente:
        mapa[x][y] = 'S'
    for mx, my in manzanas:
        mapa[mx][my] = 'M'
    for ox, oy in obstaculos:
        mapa[ox][oy] = 'X'
    return mapa

# Función para obtener la imagen del mapa actual para el frontend
def get_map_image(state):
    mapa = crear_mapa(
        filas, columnas,
        state['serpiente'],
        state['manzanas'],
        state['obstaculos']
    )
    fig, ax = plt.subplots(figsize=(6,6)) 
    for i in range(filas):
        for j in range(columnas):
            color = 'white'
            if (i, j) in state['serpiente']:
                color = 'darkgreen' if (i,j) == state['serpiente'][-1] else 'limegreen'
            elif mapa[i][j] == 'M':
                color = 'red' 
            elif mapa[i][j] == 'X':
                color = 'gray' 
            elif (i, j) in state.get('current_path', []):
                # Pintar el camino solo si es del algoritmo actual y el juego no ha terminado
                # y si el camino tiene elementos y el índice está dentro de los límites
                if state['current_path'] and not state['game_over'] and state['path_idx'] <= state['current_path'].index((i,j)):
                    color = 'cyan' 
            ax.add_patch(patches.Rectangle((j, filas-1-i), 1, 1, facecolor=color, edgecolor='black'))
    
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal') 
    
    buf = io.BytesIO()
    try:
        plt.savefig(buf, format='png')
        buf.seek(0) 
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') 
        return img_base64
    except Exception as e:
        print(f"Error al guardar la imagen o codificar en base64: {e}")
        return ''
    finally:
        plt.close(fig) 

# Función para inicializar el estado del juego para todos los algoritmos
def init_game():
    global initial_state
    serpiente = [(random.randint(0, filas-1), random.randint(0, columnas-1))]

    manzanas = []
    while len(manzanas) < 5:
        mx, my = random.randint(0, filas-1), random.randint(0, columnas-1)
        if (mx, my) not in serpiente and (mx, my) not in manzanas:
            manzanas.append((mx, my))

    obstaculos = []
    while len(obstaculos) < 10:
        ox, oy = random.randint(0, filas-1), random.randint(0, columnas-1)
        if (ox, oy) not in serpiente and (ox, oy) not in manzanas and (ox, oy) not in obstaculos:
            obstaculos.append((ox, oy))
    
    initial_state = {
        'serpiente': serpiente.copy(),
        'manzanas': manzanas.copy(),
        'obstaculos': obstaculos.copy()
    }
    
    for key in game_states:
        game_states[key] = {
            'serpiente': serpiente.copy(),
            'manzanas': manzanas.copy(),
            'obstaculos': obstaculos.copy(),
            'current_path': [], 
            'path_idx': 0,      
            'game_over': False, 
            'steps': 0,         
            'time': 0,          
            'current_position': serpiente[-1], 
            'open_nodes': [],   
            'closed_nodes': [], 
            'apples_eaten': 0   
        }

# Función para calcular el camino para un algoritmo dado (A*, BFS, DFS, GA)
def compute_path(algorithm):
    state = game_states[algorithm]
    serp_sim_current = state['serpiente'][:] 
    manzanas_actuales_sim = state['manzanas'][:]
    obstaculos_actuales_sim = state['obstaculos'][:]

    cabeza = serp_sim_current[-1] 
    
    start_time = time.time() 

    if not manzanas_actuales_sim:
        state['game_over'] = True
        state['current_path'] = []
        return

    found_path = [] 

    if algorithm == 'A*':
        objetivo = min(manzanas_actuales_sim, key=lambda apple: heuristica(cabeza, apple))
        
        abierta = [(heuristica(cabeza, objetivo), 0, cabeza, [])]
        visitado = set() 
        open_set = set([cabeza]) 
        
        while abierta:
            f, g, actual, camino = heapq.heappop(abierta) 
            if actual in visitado:
                continue
            visitado.add(actual)
            open_set.discard(actual)
            
            if actual == objetivo:
                found_path = camino
                break 
            
            for dx, dy in DIRS:
                nx, ny = actual[0]+dx, actual[1]+dy
                vecino = (nx, ny)
                if 0 <= nx < filas and 0 <= ny < columnas and \
                   vecino not in obstaculos_actuales_sim and \
                   vecino not in serp_sim_current and \
                   vecino not in visitado:
                    heapq.heappush(abierta, (g+1+heuristica(vecino, objetivo), g+1, vecino, camino+[vecino]))
                    open_set.add(vecino)
        state['open_nodes'] = list(open_set) 
        state['closed_nodes'] = list(visitado)

    elif algorithm == 'BFS':
        objetivo = min(manzanas_actuales_sim, key=lambda apple: heuristica(cabeza, apple))
        queue = deque([(cabeza, [])]) 
        visitado = set()
        while queue:
            actual, camino = queue.popleft()
            if actual in visitado: continue
            visitado.add(actual)
            if actual == objetivo:
                found_path = camino
                break
            for dx, dy in DIRS:
                nx, ny = actual[0]+dx, actual[1]+dy
                vecino = (nx, ny)
                if 0 <= nx < filas and 0 <= ny < columnas and \
                   vecino not in obstaculos_actuales_sim and \
                   vecino not in serp_sim_current and \
                   vecino not in visitado:
                    queue.append((vecino, camino+[vecino]))

    elif algorithm == 'DFS':
        objetivo = min(manzanas_actuales_sim, key=lambda apple: heuristica(cabeza, apple))
        stack = [(cabeza, [])] 
        visitado = set()
        while stack:
            actual, camino = stack.pop()
            if actual in visitado: continue
            visitado.add(actual)
            if actual == objetivo:
                found_path = camino
                break
            for dx, dy in reversed(DIRS): 
                nx, ny = actual[0] + dx, actual[1] + dy
                vecino = (nx, ny)
                if 0 <= nx < filas and 0 <= ny < columnas and \
                   vecino not in obstaculos_actuales_sim and \
                   vecino not in serp_sim_current and \
                   vecino not in visitado:
                    stack.append((vecino, camino + [vecino]))

    elif algorithm == 'GA':
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        best_individual_path = [] 
        best_apples_eaten_global = -1 

        for gen in range(GENERATIONS):
            fitness_scores = [evaluate_fitness(ind, copy.deepcopy(state['serpiente']), copy.deepcopy(state['manzanas']), copy.deepcopy(state['obstaculos'])) for ind in population]
            
            population_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            population = [ind for ind, fitness in population_with_fitness]

            current_best_individual = population[0]
            current_path_sim = []
            sim_snake_temp = list(state['serpiente']) 
            sim_apples_temp = list(state['manzanas']) # Correctly defined here
            sim_obstacles_temp = list(state['obstaculos'])
            head_sim_temp = sim_snake_temp[-1]
            apples_eaten_current_sim = 0
            
            for move_idx in current_best_individual:
                dx, dy = DIRS_MAP[move_idx]
                next_head = (head_sim_temp[0] + dx, head_sim_temp[1] + dy)

                if not (0 <= next_head[0] < filas and 0 <= next_head[1] < columnas) or \
                   next_head in sim_obstacles_temp or \
                   (next_head in sim_snake_temp[:-1] and next_head != sim_snake_temp[0]):
                    break 

                current_path_sim.append(next_head)
                sim_snake_temp.append(next_head)
                head_sim_temp = next_head

                apple_eaten_in_this_step_sim = False
                if sim_apples_temp and next_head in sim_apples_temp: # Corrected to sim_apples_temp
                    sim_apples_temp.remove(next_head)
                    apples_eaten_current_sim += 1
                    apple_eaten_in_this_step_sim = True
                
                if not apple_eaten_in_this_step_sim:
                    sim_snake_temp.pop(0)

            if apples_eaten_current_sim > best_apples_eaten_global:
                best_apples_eaten_global = apples_eaten_current_sim
                best_individual_path = current_path_sim[:]
                state['apples_eaten'] = best_apples_eaten_global 
            elif apples_eaten_current_sim == best_apples_eaten_global:
                 if len(current_path_sim) > 0 and (len(current_path_sim) < len(best_individual_path) or not best_individual_path):
                     best_individual_path = current_path_sim[:]

            # If the best individual ate all apples, stop the GA
            if best_apples_eaten_global == len(initial_state['manzanas']) and len(initial_state['manzanas']) > 0:
                break 
            
            new_population = population[:ELITISM_COUNT]

            while len(new_population) < POPULATION_SIZE:
                parent1 = select_parent(population_with_fitness)
                parent2 = select_parent(population_with_fitness)
                
                child1, child2 = crossover(parent1, parent2)
                
                new_population.append(mutate(child1))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(mutate(child2))
            
            population = new_population 
        
        state['current_path'] = best_individual_path 
        state['steps'] = len(best_individual_path) 

        # The game is over for GA only if all apples are eaten for this specific algorithm instance
        if state['apples_eaten'] == len(initial_state['manzanas']) and len(initial_state['manzanas']) > 0:
            state['game_over'] = True # Mark as True when all apples are eaten
        else:
            state['game_over'] = False # Continue if not all apples are eaten

    if not found_path and algorithm != 'GA': # A*, BFS, DFS might not find a path
        state['game_over'] = True 
    elif algorithm != 'GA': 
        state['current_path'] = found_path 
        
    end_time = time.time() 
    state['time'] = round(end_time - start_time, 4) 
    state['current_position'] = state['serpiente'][-1] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset')
def reset():
    """Reinicia el estado de todos los juegos y calcula los caminos iniciales."""
    init_game() 
    data = {}
    for algo in ['A*', 'BFS', 'DFS', 'GA']:
        game_states[algo]['game_over'] = False 
        game_states[algo]['apples_eaten'] = 0 
        game_states[algo]['path_idx'] = 0 
        compute_path(algo) 
        state = game_states[algo] 
        data[algo] = {
            'img_data': get_map_image(state), 
            'time': state['time'],           
            'steps': state['steps'],         
            'game_over': state['game_over'], 
            'current_position': state.get('current_position', None),
            'open_nodes': state.get('open_nodes', []) if algo == 'A*' else [],
            'closed_nodes': state.get('closed_nodes', []) if algo == 'A*' else [],
            'apples_eaten': state.get('apples_eaten', 0)
        }
    return jsonify(data) 

@app.route('/step')
def step():
    """
    Avanza un único paso para cada algoritmo.
    """
    data = {}
    for algo in ['A*', 'BFS', 'DFS', 'GA']:
        state = game_states[algo]
        
        # Si el juego ya terminó para este algoritmo, no hacer más pasos
        if state['game_over']:
            data[algo] = { 
                'img_data': get_map_image(state),
                'time': state['time'],
                'steps': state['steps'],
                'game_over': True,
                'current_position': state.get('current_position', None),
                'open_nodes': state.get('open_nodes', []) if algo == 'A*' else [],
                'closed_nodes': state.get('closed_nodes', []) if algo == 'A*' else [],
                'apples_eaten': state.get('apples_eaten', 0)
            }
            continue 

        # Si no hay un camino actual o se llegó al final del camino, recalcular
        if not state['current_path'] or state['path_idx'] >= len(state['current_path']):
            # Solo recalcular si aún quedan manzanas
            if state['manzanas']:
                compute_path(algo)
                state['path_idx'] = 0 # Reiniciar el índice para el nuevo camino
            else:
                state['game_over'] = True # No quedan manzanas, juego terminado
                
        # Si el juego sigue en curso y hay un camino
        if not state['game_over'] and state['current_path'] and state['path_idx'] < len(state['current_path']):
            next_pos = state['current_path'][state['path_idx']]
            
            collision = False
            # Verificar colisiones antes de mover la serpiente
            if not (0 <= next_pos[0] < filas and 0 <= next_pos[1] < columnas):
                collision = True 
            elif next_pos in state['obstaculos']:
                collision = True 
            elif len(state['serpiente']) > 1 and next_pos in state['serpiente'][:-1] and next_pos != state['serpiente'][0]:
                collision = True
            
            if collision:
                state['game_over'] = True
            else:
                state['serpiente'].append(next_pos) 
                state['current_position'] = next_pos 
                if next_pos in state['manzanas']:
                    state['manzanas'].remove(next_pos) 
                    state['apples_eaten'] += 1 
                    # El camino actual se invalida al comer una manzana, forzando un recalculo
                    state['current_path'] = [] 
                    state['path_idx'] = 0
                    if not state['manzanas']: 
                        state['game_over'] = True
                else:
                    state['serpiente'].pop(0) 
                
                state['path_idx'] += 1 # Avanzar el índice del camino si no hubo colisión
                state['steps'] += 1 # Incrementar el contador de pasos

        # Asegurar que el estado de game_over se actualice si no hay más manzanas
        if not state['manzanas'] and not state['game_over']:
            state['game_over'] = True

        data[algo] = {
            'img_data': get_map_image(state),
            'time': state['time'],
            'steps': state['steps'],
            'game_over': state['game_over'],
            'current_position': state.get('current_position', None),
            'open_nodes': state.get('open_nodes', []) if algo == 'A*' else [],
            'closed_nodes': state.get('closed_nodes', []) if algo == 'A*' else [],
            'apples_eaten': state.get('apples_eaten', 0)
        }
    return jsonify(data) 

if __name__ == '__main__':
    init_game() 
    app.run(debug=True)