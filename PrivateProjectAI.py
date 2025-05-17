import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import heapq
import time
from collections import deque
import random
import threading
import copy
from functools import partial
import math
import sys
import numpy as np

# --- Core Logic ---
ROWS, COLS = 3, 3
start_state_default = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
#start_state_default = [[0, 1, 2], [4, 5, 3], [7, 8, 6]]
#start_state_default = [[0, 1, 2], [3, 4, 5], [7, 8, 6]]
#start_state_default = [[1, 2, 3], [4, 5, 6], [0, 8, 7]]
#start_state_default = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
goal_state_default = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
current_start_state = copy.deepcopy(start_state_default)
current_goal_state = copy.deepcopy(goal_state_default)


def get_neighbors(state):
    """Returns a list of (neighbor_state, cost) for the given state (list[list[int]]). Used in state space search."""
    zero_pos = find_zero(state)
    if zero_pos is None: return []
    r, c = zero_pos
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            new_state = [row[:] for row in state] # Deep copy
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            neighbors.append((new_state, 1))
    return neighbors

def heuristic(state, goal):
    """Calculates the Manhattan distance heuristic."""
    distance = 0
    goal_positions = {}
    for r in range(ROWS):
        for c in range(COLS):
            val = goal[r][c]
            if val != 0:
                goal_positions[val] = (r, c)
    for r in range(ROWS):
        for c in range(COLS):
            val = state[r][c]
            if val != 0 and val in goal_positions:
                goal_r, goal_c = goal_positions[val]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

ACTION_LIST = ['RIGHT', 'LEFT', 'DOWN', 'UP']

def state_to_tuple(state):
    """Chuyển ma trận 3x3 thành tuple."""
    return tuple(val for row in state for val in row)

def find_zero(state):
    """Tìm vị trí (row, col) của ô trống (0)."""
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                return r, c
    return None

def tuple_to_state(state):
    if isinstance(state, (list, tuple)) and len(state) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in state):
        return [list(row) for row in state]
    if isinstance(state, (list, tuple)) and len(state) == 9:
        return [list(state[i*3:(i+1)*3]) for i in range(3)]
    raise ValueError(f"Invalid state format: {state}, expected a 3x3 matrix or a tuple of 9 elements")

def is_solvable(state, goal_state):
    """Kiểm tra xem trạng thái có thể giải được so với trạng thái mục tiêu."""
    flat_state = [val for row in state for val in row if val != 0]
    goal_flat_state = [val for row in goal_state for val in row if val != 0]
    inversions = sum(1 for i in range(len(flat_state)) for j in range(i + 1, len(flat_state)) if flat_state[i] > flat_state[j])
    goal_inversions = sum(1 for i in range(len(goal_flat_state)) for j in range(i + 1, len(goal_flat_state)) if goal_flat_state[i] > goal_flat_state[j])
    blank_row = find_zero(state)[0]
    goal_blank_row = find_zero(goal_state)[0]
    return (inversions % 2 == goal_inversions % 2) and ((blank_row % 2) == (goal_blank_row % 2))

def generate_initial_belief_set(start_state):
    initial_belief_state_tuples = {
        state_to_tuple([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
    
    }
    belief_set = [tuple_to_state(state_tuple) for state_tuple in initial_belief_state_tuples]
    return belief_set

def apply_action(state, action):
    state_matrix = tuple_to_state(state)
    blank_pos = find_zero(state_matrix)
    if blank_pos is None:
        return None

    r, c = blank_pos
    new_r, new_c = r, c

    if action == 'UP':
        new_r = r - 1
    elif action == 'DOWN':
        new_r = r + 1
    elif action == 'LEFT':
        new_c = c - 1
    elif action == 'RIGHT':
        new_c = c + 1
    else:
        return None

    if 0 <= new_r < 3 and 0 <= new_c < 3:
        state_matrix[r][c] = state_matrix[new_r][new_c]
        state_matrix[new_r][new_c] = 0
        return state_matrix
    return None

def bfs_belief_search(start_state, goal_state):
    print("Running Sensorless Search (BFS on Belief Space)...")
    
    # Sử dụng trạng thái mặc định cho belief set
    belief_start_state = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
    initial_belief_set = generate_initial_belief_set(state_to_tuple(belief_start_state))
    
    print("Initial belief set:")
    for state in initial_belief_set:
        print(state)
    print()

    goal_matrix = tuple_to_state(goal_state)
    goal_tuple = tuple(state_to_tuple(goal_matrix))
    visited = set()
    queue = deque([(initial_belief_set, [])])  # (belief_set, actions)
    belief_state_count = 0
    start_time = time.time()  # Bắt đầu đo thời gian

    while queue:
        current_belief_set, actions = queue.popleft()
        belief_state_count += 1

        belief_set_tuple = tuple(sorted([tuple(state_to_tuple(state)) for state in current_belief_set]))
        if belief_set_tuple in visited:
            continue
        visited.add(belief_set_tuple)

        print(f"BFS Belief: Exploring belief state {belief_state_count}, size={len(current_belief_set)}, current actions={actions}")
        print("Current belief set:")
        for state in current_belief_set:
            print(state)

        all_goal = all(tuple(state_to_tuple(state)) == goal_tuple for state in current_belief_set)
        if all_goal:
            print(f"Sensorless Search (BFS): Found a plan with actions: {actions}")
            
            # Cập nhật current_start_state thành trạng thái mặc định của belief_start_state
            global current_start_state
            current_start_state = copy.deepcopy(belief_start_state)
            
            # Cập nhật lưới Start trên GUI
            gui.update_grid(gui.initial_canvas, current_start_state, gui.START_END_TILE_COLOR)
            
            return actions, time.time() - start_time, belief_state_count  # Tính thời gian thực tế

        for action in ACTION_LIST:
            print(f"Trying action: {action}")
            new_belief_set = set()
            action_failed = False

            for state in current_belief_set:
                print(f"Applying {action} to {state}")
                new_state = apply_action(state, action)
                if new_state is None:
                    print(f"Action {action} failed for state {state}")
                    action_failed = True
                    break
                print(f"Result: {new_state}")
                new_belief_set.add(tuple(state_to_tuple(new_state)))

            if not action_failed:
                new_belief_set = [tuple_to_state(state) for state in new_belief_set]
                print(f"New belief set after {action}: {new_belief_set}")
                queue.append((new_belief_set, actions + [action]))
            print()

    print("Sensorless Search (BFS): Failed to find a plan.")
    return None, time.time() - start_time, belief_state_count  # Tính thời gian thực tế

def get_observation(state):
    """Get the observation from the state."""
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)
    return None

def filter_belief_set(belief_set, observation):
    """Filter the belief set based on the observation."""
    filtered_belief_set = []
    for state in belief_set:
        if get_observation(state) == observation:
            filtered_belief_set.append(state)
    return filtered_belief_set

def bfs_partially_observable_search(start_state, goal_state):
    """BFS run on belief space with observations."""
    print("Running Partially Observable Search (BFS on Belief Space)...")

    # Sử dụng trạng thái mặc định cho belief set
    belief_start_state = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]
    initial_belief_set = generate_initial_belief_set(state_to_tuple(belief_start_state))
    
    print("Initial belief set:")
    for state in initial_belief_set:
        print(state)
    print()

    goal_matrix = tuple_to_state(goal_state)
    goal_tuple = tuple(state_to_tuple(goal_matrix))
    visited = set()
    queue = deque([(initial_belief_set, [], None)])  # (belief_set, actions, last_observation)
    belief_state_count = 0
    start_time = time.time()

    while queue:
        current_belief_set, actions, last_observation = queue.popleft()
        belief_state_count += 1

        # Chuyển tập niềm tin thành tuple để kiểm tra visited
        belief_set_tuple = tuple(sorted([tuple(state_to_tuple(state)) for state in current_belief_set]))
        if belief_set_tuple in visited:
            continue
        visited.add(belief_set_tuple)

        print(f"BFS Belief: Exploring belief state {belief_state_count}, size={len(current_belief_set)}, current actions={actions}, last observation={last_observation}")
        print("Current belief set:")
        for state in current_belief_set:
            print(state)

        # Kiểm tra xem tất cả trạng thái trong tập niềm tin có đạt mục tiêu không
        all_goal = all(tuple(state_to_tuple(state)) == goal_tuple for state in current_belief_set)
        if all_goal:
            print(f"Partially Observable Search (BFS): Found a plan with actions: {actions}")
            
            # Cập nhật current_start_state thành trạng thái mặc định
            global current_start_state
            current_start_state = copy.deepcopy(belief_start_state)
            
            # Cập nhật GUI (giả định GUI tồn tại)
            gui.update_grid(gui.initial_canvas, current_start_state, gui.START_END_TILE_COLOR)
            
            return actions, time.time() - start_time, belief_state_count

        for action in ACTION_LIST:
            print(f"Trying action: {action}")
            new_belief_set = set()

            # Áp dụng hành động cho từng trạng thái trong tập niềm tin
            for state in current_belief_set:
                new_state = apply_action(state, action)
                if new_state is not None:
                    new_belief_set.add(tuple(state_to_tuple(new_state)))

            if not new_belief_set:
                continue

            new_belief_set = [tuple_to_state(state) for state in new_belief_set]
            print(f"New belief set after {action}: {new_belief_set}")

            
            possible_observations = set(get_observation(state) for state in new_belief_set)
            for observation in possible_observations:
                
                filtered_belief_set = filter_belief_set(new_belief_set, observation)
                if filtered_belief_set:  # Chỉ thêm nếu tập niềm tin không rỗng
                    print(f"Observation {observation} -> Filtered belief set: {filtered_belief_set}")
                    queue.append((filtered_belief_set, actions + [action], observation))

    print("Partially Observable Search (BFS): Failed to find a plan.")
    return None, time.time() - start_time, belief_state_count

def dfs(start, goal):
    """Depth-First Search (State Space)."""
    print("Running DFS (State Space)...")
    stack = [(start, [])]
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while stack:
        state, path = stack.pop()
        nodes_visited_count += 1
        if state == goal:
            print(f"DFS: Solution found of length {len(path)}.")
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in reversed(get_neighbors(state)):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                stack.append((neighbor, path + [state]))
    print("DFS: Failed to find a solution.")
    return None, time.time() - start_time, nodes_visited_count

def bfs(start, goal):
    """Breadth-First Search (State Space)."""
    print("Running BFS (State Space)...")
    queue = deque([(start, [])])
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while queue:
        state, path = queue.popleft()
        nodes_visited_count += 1
        if state == goal:
            print(f"BFS: Solution found of length {len(path)}.")
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in get_neighbors(state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, path + [state]))
    print("BFS: Failed to find a solution.")
    return None, time.time() - start_time, nodes_visited_count

def ucs(start, goal):
    """Uniform Cost Search (State Space)."""
    print("Running UCS (State Space)...")
    pq = [(0, start, [])] # (cost, state, path)
    visited_costs = {state_to_tuple(start): 0}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        cost, state, path = heapq.heappop(pq)
        nodes_visited_count += 1
        state_tuple = state_to_tuple(state)

        if cost > visited_costs.get(state_tuple, float('inf')): continue

        if state == goal:
            print(f"UCS: Solution found of length {len(path)}.")
            return path + [state], time.time() - start_time, nodes_visited_count

        for neighbor, step_cost in get_neighbors(state):
            new_cost = cost + step_cost
            neighbor_tuple = state_to_tuple(neighbor)
            if new_cost < visited_costs.get(neighbor_tuple, float('inf')):
                visited_costs[neighbor_tuple] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [state]))

    print("UCS: Failed to find a solution.")
    return None, time.time() - start_time, nodes_visited_count

def a_star(start, goal):
    """A* Search (State Space) using Manhattan distance heuristic."""
    print("Running A* (State Space)...")
    start_h = heuristic(start, goal)
    pq = [(start_h, 0, start, [])] # (f_cost, g_cost, state, path)
    g_costs = {state_to_tuple(start): 0}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        f_cost, g_cost, state, path = heapq.heappop(pq)
        nodes_visited_count += 1
        state_tuple = state_to_tuple(state)

        if g_cost > g_costs.get(state_tuple, float('inf')): continue

        if state == goal:
            print(f"A*: Solution found of length {len(path)}.")
            return path + [state], time.time() - start_time, nodes_visited_count

        for neighbor, step_cost in get_neighbors(state):
            new_g_cost = g_cost + step_cost
            neighbor_tuple = state_to_tuple(neighbor)
            if new_g_cost < g_costs.get(neighbor_tuple, float('inf')):
                g_costs[neighbor_tuple] = new_g_cost
                new_h_cost = heuristic(neighbor, goal)
                new_f_cost = new_g_cost + new_h_cost
                heapq.heappush(pq, (new_f_cost, new_g_cost, neighbor, path + [state]))

    print("A*: Failed to find a solution.")
    return None, time.time() - start_time, nodes_visited_count

def greedy_best_first(start, goal):
    """Greedy Best-First Search (State Space) using Manhattan distance heuristic."""
    print("Running Greedy Best-First Search (State Space)...")
    pq = [(heuristic(start, goal), start, [])] # (h_cost, state, path)
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        _, state, path = heapq.heappop(pq)
        nodes_visited_count += 1

        if state == goal:
            print(f"Greedy: Solution found of length {len(path)}.")
            return path + [state], time.time() - start_time, nodes_visited_count

        for neighbor, _ in get_neighbors(state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor, path + [state]))

    print("Greedy: Failed to find a solution.")
    return None, time.time() - start_time, nodes_visited_count

def ids(start, goal):
    print("Running IDS (State Space)...")
    start_time = time.time()
    goal_tuple = state_to_tuple(goal) 

    total_nodes_visited = 0
    depth_limit = 0

    max_possible_depth = 50

    while depth_limit <= max_possible_depth:

        nodes_visited_this_depth = 0 
        stack = deque([(start, [copy.deepcopy(start)], 0, {state_to_tuple(start)})])

        while stack:

            current_state, path, current_depth_in_this_iteration, visited_in_path = stack.pop()
            nodes_visited_this_depth += 1 
            if current_state == goal:
                print(f"IDS: Solution found at depth {current_depth_in_this_iteration}.")
                total_nodes_visited += nodes_visited_this_depth 

                return path, time.time() - start_time, total_nodes_visited


            if current_depth_in_this_iteration >= depth_limit:
                continue

            for neighbor_matrix, _ in reversed(get_neighbors(current_state)):
                neighbor_tuple = state_to_tuple(neighbor_matrix)

                if neighbor_tuple not in visited_in_path:

                    new_visited_in_path = visited_in_path.copy()
                    new_visited_in_path.add(neighbor_tuple)

                    stack.append((neighbor_matrix, path + [neighbor_matrix], current_depth_in_this_iteration + 1, new_visited_in_path))
        depth_limit += 1
    print(f"IDS: Max possible depth ({max_possible_depth}) reached without finding a solution.")
    return None, time.time() - start_time, total_nodes_visited 


def ida_star_search(node, g_cost, threshold, path, goal, visited_in_path, nodes_visited_list):
    """Recursive helper for IDA*."""
    nodes_visited_list[0] += 1
    node_tuple = state_to_tuple(node)
    f_cost = g_cost + heuristic(node, goal)

    if f_cost > threshold: return f_cost, None, False

    if node == goal: return True, path + [node], True

    min_next_threshold = float('inf')
    visited_in_path_copy = visited_in_path.copy()
    visited_in_path_copy.add(node_tuple)

    for next_state, step_cost in get_neighbors(node):
         next_state_tuple = state_to_tuple(next_state)
         if next_state_tuple not in visited_in_path_copy: # Avoid cycles in path
            result, found_path, found_goal_recursive = ida_star_search(
                next_state, g_cost + step_cost, threshold, path + [node], goal,
                visited_in_path_copy, nodes_visited_list
            )

            if found_goal_recursive: return True, found_path, True
            if result is not True and result < min_next_threshold:
                min_next_threshold = result

    return min_next_threshold, None, False

def ida_star(start, goal):
    """Iterative Deepening A* Search (State Space)."""
    print("Running IDA* (State Space)...")
    threshold = heuristic(start, goal)
    total_nodes_visited = 0
    start_time = time.time()

    original_recursion_limit = sys.getrecursionlimit()
    new_limit = 5000
    limit_changed = False
    if new_limit > original_recursion_limit:
        try:
            sys.setrecursionlimit(new_limit)
            limit_changed = True
        except Exception as e: print(f"Could not set recursion limit for IDA*: {e}")

    try:
        while True:
            nodes_visited_list = [0]
            result, solution_path, found_goal = ida_star_search(
                start, 0, threshold, [], goal, set(), nodes_visited_list
            )
            total_nodes_visited += nodes_visited_list[0]

            if found_goal:
                print(f"IDA*: Solution found with threshold {threshold}.")
                return solution_path, time.time() - start_time, total_nodes_visited
            if result == float('inf'):
                 print("IDA*: Search space exhausted.")
                 break
            if threshold > 1000: # Safety break for threshold
                 print(f"IDA* threshold limit ({threshold}) reached.")
                 break
            threshold = result # Increase threshold

    except RecursionError:
        print(f"IDA* Failed: Maximum recursion depth exceeded ({sys.getrecursionlimit()}).")
        return None, time.time() - start_time, total_nodes_visited
    except Exception as e:
        print(f"IDA* encountered an error: {e}")
        return None, time.time() - start_time, total_nodes_visited
    finally:
        if limit_changed: sys.setrecursionlimit(original_recursion_limit)

    print("IDA*: Failed to find a solution.")
    return None, time.time() - start_time, total_nodes_visited


# --- Local Search Algorithms ---
# These algorithms do not guarantee optimality or completeness.

def simple_hill_climbing(start, goal):
    """Simple Hill Climbing (State Space)."""
    print("Running Simple Hill Climbing...")
    current = start
    path = [copy.deepcopy(current)]
    nodes_evaluated = 0
    start_time = time.time()
    current_h = heuristic(current, goal)
    max_iterations = 10000

    while current != goal and nodes_evaluated < max_iterations:
        nodes_evaluated += 1
        neighbors = get_neighbors(current)
        if not neighbors: break

        next_state = None
        best_neighbor_h = current_h

        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            if neighbor_h < best_neighbor_h:
                next_state = neighbor_state
                best_neighbor_h = neighbor_h
                break

        if next_state is None:
            if current != goal: print("Simple Hill Climbing stuck.")
            break

        current = next_state
        path.append(copy.deepcopy(current))
        current_h = best_neighbor_h

    if nodes_evaluated >= max_iterations and current != goal:
        print(f"Simple Hill Climbing reached max iterations ({max_iterations}).")
    elif current == goal:
        print(f"Simple Hill Climbing: Goal reached.")

    time_taken = time.time() - start_time
    return path, time_taken, nodes_evaluated


def steepest_ascent_hill_climbing(start, goal):
    """Steepest Ascent Hill Climbing (State Space)."""
    print("Running Steepest Ascent Hill Climbing...")
    current = start
    path = [copy.deepcopy(current)]
    nodes_evaluated = 0
    start_time = time.time()
    current_h = heuristic(current, goal)
    max_iterations = 10000

    while current != goal and nodes_evaluated < max_iterations:
        nodes_evaluated += 1
        neighbors = get_neighbors(current)
        if not neighbors: break

        best_neighbor = None
        best_h = current_h

        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbor = neighbor_state

        if best_neighbor is None or best_h >= current_h:
             if current != goal: print("Steepest Ascent stuck.")
             break

        current = best_neighbor
        current_h = best_h
        path.append(copy.deepcopy(current))

    if nodes_evaluated >= max_iterations and current != goal:
        print(f"Steepest Ascent Hill Climbing reached max iterations ({max_iterations}).")
    elif current == goal:
        print(f"Steepest Ascent: Goal reached.")

    time_taken = time.time() - start_time
    return path, time_taken, nodes_evaluated


def stochastic_hill_climbing(start, goal, max_iterations=10000):
    """Random Hill Climbing (State Space)."""
    print("Running Random Hill Climbing...")
    current = start
    path = [copy.deepcopy(current)]
    nodes_evaluated = 0
    start_time = time.time()
    current_h = heuristic(current, goal)

    while current != goal and nodes_evaluated < max_iterations:
        nodes_evaluated += 1
        neighbors = get_neighbors(current)
        if not neighbors: break

        evaluated_neighbors = []
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            evaluated_neighbors.append((neighbor_state, neighbor_h))

        better_neighbors = [(n_state, n_h) for n_state, n_h in evaluated_neighbors if n_h < current_h]

        if better_neighbors:
            chosen_neighbor, chosen_h = random.choice(better_neighbors)
            current = chosen_neighbor
            current_h = chosen_h
            path.append(copy.deepcopy(current))
        else:
            if current != goal: print("Random Hill Climbing stuck (no better neighbors).")
            break

    if nodes_evaluated >= max_iterations and current != goal:
        print(f"Random Hill Climbing reached max iterations ({max_iterations}).")
    elif current == goal:
        print(f"Random Hill Climbing: Goal reached.")

    time_taken = time.time() - start_time
    return path, time_taken, nodes_evaluated


def simulated_annealing(start_state, goal_state, initial_temp=100.0, cooling_rate=0.99, min_temp=0.1, max_iterations_per_temp=100):
    """Simulated Annealing (State Space)."""
    print("Running Simulated Annealing...")
    start_time = time.time()
    nodes_evaluated = 0
    current_state = copy.deepcopy(start_state)
    current_energy = heuristic(current_state, goal_state)
    best_state = copy.deepcopy(current_state)
    best_energy = current_energy
    path_taken = [copy.deepcopy(current_state)]

    temp = initial_temp
    while temp > min_temp:
        if current_state == goal_state: break

        for _ in range(max_iterations_per_temp):
             if current_state == goal_state: break

             neighbors = get_neighbors(current_state)
             if not neighbors: break

             next_state, _ = random.choice(neighbors)
             nodes_evaluated += 1

             next_energy = heuristic(next_state, goal_state)
             delta_E = next_energy - current_energy

             accept = False
             if delta_E < 0: accept = True
             else:
                 if temp > 1e-9:
                    try: probability = math.exp(-delta_E / temp)
                    except (OverflowError, ValueError): probability = 0
                    if random.random() < probability: accept = True

             if accept:
                 current_state = next_state
                 current_energy = next_energy
                 path_taken.append(copy.deepcopy(current_state))
                 if current_energy < best_energy:
                     best_energy = current_energy
                     best_state = copy.deepcopy(current_state)

        if current_state == goal_state or not neighbors: break
        temp *= cooling_rate

    time_taken = time.time() - start_time
    if current_state == goal_state:
         try:
            goal_index = path_taken.index(goal_state)
            path_taken = path_taken[:goal_index + 1]
         except ValueError: pass
         print(f"SA finished. Goal reached. Final Energy: {current_energy}.")
    else:
        print(f"SA finished. Best energy reached: {best_energy}. Final Temp: {temp:.3f}. Goal reached: {current_state == goal_state}")

    return path_taken, time_taken, nodes_evaluated


def beam_search(start_state, goal_state, beam_width=5):
    start_time = time.time()
    nodes_expanded = 0
    start_h = heuristic(start_state, goal_state)
    beam = [(start_h, start_state, [start_state])]
    visited = {state_to_tuple(start_state)}
    while beam:
        candidates = []
        nodes_expanded += len(beam)
        for h_val, current_state, path in beam:
            for neighbor, _ in get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    neighbor_h = heuristic(neighbor, goal_state)
                    new_path = path + [neighbor]
                    if neighbor == goal_state:
                        time_taken = time.time() - start_time
                        print(f"Beam Search: Goal found! Beam width={beam_width}")
                        return new_path, time_taken, nodes_expanded
                    candidates.append((neighbor_h, neighbor, new_path))
        if not candidates:
            print(f"Beam Search: Failed - No candidates generated. Beam width={beam_width}")
            break
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    time_taken = time.time() - start_time
    print(f"Beam Search: Failed - Beam became empty. Beam width={beam_width}")
    return None, time_taken, nodes_expanded

def get_successors(state):
    """Returns a list of successor states for AND-OR search."""
    zero_pos = find_zero(state)
    if zero_pos is None:
        return []
    r, c = zero_pos
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    successors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            new_state = [row[:] for row in state]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            successors.append(new_state)
    return successors

def is_valid_state(state):
    """Kiểm tra xem trạng thái có hợp lệ không (chứa đủ các số từ 0 đến 8 đúng một lần)."""
    flat_state = [state[i][j] for i in range(ROWS) for j in range(COLS)]
    return sorted(flat_state) == list(range(ROWS * COLS))

def and_or_search(state, goal_state, get_successors, node_type='OR', visited=None, find_all_solutions=False, max_depth=20, max_nodes=1000, timeout=5):
    
    """AND-OR search with depth, node, and timeout limits."""
    if visited is None:
        visited = set()
    start_time = time.time()
    nodes_visited = [0]  # Use list to modify in recursive calls
    all_solutions = [] if find_all_solutions else None

    def recursive_or_and_search(state, goal_state, get_successors, node_type, visited, depth):
        nonlocal nodes_visited
        nodes_visited[0] += 1

        # Kiểm tra giới hạn
        if time.time() - start_time > timeout:
            print("AND-OR Search timeout.")
            return None
        if depth > max_depth:
            return None
        if nodes_visited[0] > max_nodes:
            return None

        state_tuple = state_to_tuple(state)
        if state_tuple in visited:
            return None
        visited.add(state_tuple)

        # Kiểm tra trạng thái mục tiêu
        if state == goal_state and is_valid_state(state):
            return [state]

        if node_type == 'OR':
            for next_state in get_successors(state):
                if is_solvable(next_state, goal_state):  # Chỉ tiếp tục với trạng thái có thể giải được
                    result = recursive_or_and_search(next_state, goal_state, get_successors, 'AND', visited.copy(), depth + 1)
                    if result:
                        return [state] + result
        else:  # node_type == 'AND'
            all_results = [state]
            successors = get_successors(state)
            if not successors:
                return None
            for next_state in successors:
                if is_solvable(next_state, goal_state):  # Chỉ tiếp tục với trạng thái có thể giải được
                    result = recursive_or_and_search(next_state, goal_state, get_successors, 'OR', visited.copy(), depth + 1)
                    if not result:
                        return None
                    all_results += result
            return all_results

        return None

    # Kiểm tra trạng thái đầu vào
    if not is_valid_state(state) or not is_valid_state(goal_state):
        print("Error: Invalid initial or goal state.")
        return None, time.time() - start_time, nodes_visited[0]

    if not is_solvable(state, goal_state):
        print("Error: Initial state is not solvable.")
        return None, time.time() - start_time, nodes_visited[0]

    result = recursive_or_and_search(state, goal_state, get_successors, node_type, visited.copy(), 0)
    if result:
        if find_all_solutions:
            all_solutions.append(result)
            visited.discard(state_to_tuple(result[-1]))
        else:
            return result, time.time() - start_time, nodes_visited[0]
    else:
        return all_solutions, time.time() - start_time, nodes_visited[0] if find_all_solutions else None


def calculate_fitness(final_state, goal_state, is_valid_path):
    """Calculates fitness based on inverse Manhattan distance to goal."""
    if not is_valid_path: return 0.0
    h = heuristic(final_state, goal_state)
    if h == float('inf'): return 0.0
    return 1.0 / (1.0 + h)

def create_initial_population(size, start_state, max_initial_len=20):
    """Creates initial population of random action sequences."""
    population = []
    attempts = 0
    max_attempts = size * 10
    start_state_tuple = state_to_tuple(start_state)

    while len(population) < size and attempts < max_attempts:
        attempts += 1
        seq_len = random.randint(1, max_initial_len)
        sequence = [random.choice(ACTION_LIST) for _ in range(seq_len)]  # Sửa lỗi: Thêm `range` để lặp
        # Simple check: ensure at least the first move is valid from the *known* start state
        if sequence and apply_action(start_state_tuple, sequence[0]) is not None:
            population.append(sequence)

    while len(population) < size:  # Ensure population size
        population.append([random.choice(ACTION_LIST)])

    return population

def simulate_path(action_sequence, start_state, max_len=60):
    """Simulates executing an action sequence."""
    current_state = copy.deepcopy(start_state)
    path = [copy.deepcopy(current_state)]

    for i, action in enumerate(action_sequence):
        if i >= max_len: return current_state, path, False

        next_state = apply_action(current_state, action)

        if next_state is None: return current_state, path, False

        current_state = next_state
        path.append(copy.deepcopy(current_state))

    return current_state, path, True

def tournament_selection(population_with_fitness, tournament_size=5):
    """Selects an individual using tournament selection."""
    if not population_with_fitness: return None
    tournament = random.sample(population_with_fitness, min(tournament_size, len(population_with_fitness)))
    tournament.sort(key=lambda x: x[0], reverse=True)
    return tournament[0][1] # Return the action sequence

def crossover(parent1, parent2):
    """Performs single-point crossover."""
    if not parent1 or not parent2 or min(len(parent1), len(parent2)) < 2:
        return parent1[:], parent2[:]
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(sequence, mutation_rate=0.1):
    """Mutates an action sequence."""
    mutated_sequence = sequence[:]
    if random.random() < mutation_rate:
        mutation_type = random.choice(['change', 'insert', 'delete'])
        if mutation_type == 'change' and mutated_sequence:
            idx = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence[idx] = random.choice(ACTION_LIST)
        elif mutation_type == 'insert':
            idx = random.randint(0, len(mutated_sequence))
            mutated_sequence.insert(idx, random.choice(ACTION_LIST))
        elif mutation_type == 'delete' and len(mutated_sequence) > 0:
            idx = random.randint(0, len(mutated_sequence) - 1)
            del mutated_sequence[idx]
    return mutated_sequence

def genetic_algorithm(start_state, goal_state, population_size=200, max_generations=50,
                      mutation_rate=0.1, tournament_size=5, elitism_count=5, max_sim_len=100):
    """Genetic Algorithm for finding a puzzle solution (State Space)."""
    print("Running Genetic Algorithm...")
    start_time = time.time()
    # GA operates on a single start state, not a belief set in this implementation.
    population = create_initial_population(population_size, start_state, max_initial_len=math.ceil(max_sim_len / 2))
    best_overall_fitness = -1.0
    best_overall_sim_path = [copy.deepcopy(start_state)]

    generations_run = 0
    total_evaluations = 0

    for generation in range(max_generations):
        generations_run += 1
        population_with_fitness = []

        for individual_sequence in population:
            total_evaluations += 1
            final_state, sim_path, is_valid = simulate_path(individual_sequence, start_state, max_sim_len)
            fitness = calculate_fitness(final_state, goal_state, is_valid)
            population_with_fitness.append((fitness, individual_sequence, sim_path))

            if is_valid and final_state == goal_state:
                print(f"GA: Goal found in generation {generation}!")
                time_taken = time.time() - start_time
                return sim_path, time_taken, total_evaluations

        population_with_fitness.sort(key=lambda x: x[0], reverse=True)

        if population_with_fitness[0][0] > best_overall_fitness:
             best_overall_fitness = population_with_fitness[0][0]
             best_overall_sim_path = population_with_fitness[0][2]

        #print(f"Gen {generation}: Best Fitness={population_with_fitness[0][0]:.4f}, Avg Fitness={sum(f for f,_,_ in population_with_fitness)/len(population_with_fitness):.4f}")

        next_generation = []
        for i in range(elitism_count):
            if i < len(population_with_fitness):
                next_generation.append(population_with_fitness[i][1])

        while len(next_generation) < population_size:
            parent1 = tournament_selection(population_with_fitness, tournament_size)
            parent2 = tournament_selection(population_with_fitness, tournament_size)

            if parent1 is None or parent2 is None:
                 next_generation.extend(create_initial_population(1, start_state, max_initial_len=5))
                 continue

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)

        population = next_generation

    time_taken = time.time() - start_time
    print(f"GA: Max generations reached. Best fitness: {best_overall_fitness:.4f}")
    return best_overall_sim_path, time_taken, total_evaluations

def is_consistent(state, row, col, value):
        """Checks if placing `value` at (row, col) is consistent with constraints."""
        if value != 0:
            if value in state:
                return False
            if col > 0 and state[row, col - 1] != 0 and value != state[row, col - 1] + 1:
                return False
            if row > 0 and state[row - 1, col] != 0 and value != state[row - 1, col] + 3:
                return False
        return True

def backtracking_csp_solve(start_state, goal_state):
    """Solves the 8-puzzle problem using CSP with backtracking."""
    np.random.seed(42)

    def is_complete(state):
        """Checks if the state is complete (all cells filled except one)."""
        return np.count_nonzero(state) == 8

    def find_unassigned(state):
        """Finds the first unassigned cell (value 0)."""
        for r in range(ROWS):
            for c in range(COLS):
                if state[r, c] == 0:
                    return r, c
        return None, None

    def recursive_backtracking(state, all_states):
        """Recursive backtracking function."""
        if is_complete(state):
            all_states.append(state.copy().tolist())  # Convert to list for compatibility
            return True

        row, col = find_unassigned(state)
        if row is None:
            return False

        values = list(range(1, 9))  # Possible values (1-8)
        np.random.shuffle(values)  # Shuffle the order of values to observe backtracking
        for value in values:
            if is_consistent(state, row, col, value):
                state[row, col] = value
                all_states.append(state.copy().tolist())  # Save intermediate state as list

                if recursive_backtracking(state, all_states):
                    return True

                state[row, col] = 0  # Backtrack
                all_states.append(state.copy().tolist())  # Save state after backtracking

        return False

    all_states = []
    initial_state = np.zeros((ROWS, COLS), dtype=int)
    recursive_backtracking(initial_state, all_states)
    return all_states, len(all_states), len(all_states)

def ac3_csp_solve(start_state, goal_state):
    """AC3 algorithm to solve the 8-puzzle problem using CSP."""

    def revise(state, domains, xi, xj):
        """Revises the domain of xi to ensure consistency with xj."""
        revised = False
        for x in domains[xi][:]:
            if not any(is_consistent(state, *xi, x) for x in domains[xj]):
                domains[xi].remove(x)
                revised = True
        return revised

    def ac3(state, domains, arcs):
        """AC3 algorithm to enforce arc consistency."""
        while arcs:
            xi, xj = arcs.popleft()
            if revise(state, domains, xi, xj):
                if not domains[xi]:
                    return False
                for xk in [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != xi and (r, c) != xj]:
                    arcs.append((xk, xi))
        return True

    def backtrack(state, domains, all_states):
        """Backtracking search to find a solution after enforcing arc consistency."""
        def is_complete(state):
            """Checks if the state is complete (all cells filled except one)."""
            return np.count_nonzero(state) == 8

        def find_unassigned(state):
            """Finds the first unassigned cell (value 0)."""
            for r in range(ROWS):
                for c in range(COLS):
                    if state[r, c] == 0:
                        return r, c
            return None, None

        if is_complete(state):
            all_states.append(state.copy().tolist())  # Convert to list for compatibility
            return True

        row, col = find_unassigned(state)
        if row is None:
            return False

        values = domains[(row, col)]  # Use the domain values for the current cell
        for value in values:
            if is_consistent(state, row, col, value):
                state[row, col] = value
                all_states.append(state.copy().tolist())  # Save intermediate state as list

                if backtrack(state, domains, all_states):
                    return True

                state[row, col] = 0  # Backtrack
                all_states.append(state.copy().tolist())  # Save state after backtracking

        return False

    # Initialize domains and arcs
    domains = {(r, c): list(range(1, 9)) for r in range(ROWS) for c in range(COLS)}
    arcs = deque([(xi, xj) for xi in domains for xj in domains if xi != xj])

    # Initialize state and all_states
    initial_state = np.zeros((ROWS, COLS), dtype=int)
    all_states = []

    # Run AC3 to enforce arc consistency
    if ac3(initial_state, domains, arcs):
        # Use backtracking to find a solution
        backtrack(initial_state, domains, all_states)

    return all_states, len(all_states), len(all_states)

def generate_and_test(start_state, goal_state, max_attempts=100000, max_states_to_store=500):
    """Generate and Test algorithm with CSP constraints."""
    print("Running Generate and Test with CSP constraints (showing all states)...")
    ROWS, COLS = 3, 3
    attempts = 0
    start_time = time.time()
    all_states = []  # Store all generated states (valid and invalid)

    def is_consistent(state, row, col, value):
        """Checks CSP constraints: uniqueness, +1 horizontal, +3 vertical."""
        if value != 0:
            # Check uniqueness, excluding the current cell
            flat_state = [state[r][c] for r in range(ROWS) for c in range(COLS) if (r, c) != (row, col)]
            if value in flat_state:
                return False
            if col > 0 and state[row][col - 1] != 0 and value != state[row][col - 1] + 1:
                return False
            if row > 0 and state[row - 1][col] != 0 and value != state[row - 1][col] + 3:
                return False
        return True

    def is_solvable(state):
        """Check if the state is solvable (based on inversion count)."""
        flat = [val for row in state for val in row]
        inversions = sum(1 for i in range(len(flat)) for j in range(i + 1, len(flat))
                         if flat[i] != 0 and flat[j] != 0 and flat[i] > flat[j])
        return inversions % 2 == 0

    def is_valid_goal(state):
        """Check if goal_state satisfies the constraints."""
        for r in range(ROWS):
            for c in range(COLS):
                if state[r][c] != 0:
                    if c > 0 and state[r][c-1] != 0 and state[r][c] != state[r][c-1] + 1:
                        return False
                    if r > 0 and state[r-1][c] != 0 and state[r][c] != state[r-1][c] + 3:
                        return False
        return True

    def is_complete_state(state):
        """Check if the state contains exactly one of each number from 0 to 8."""
        flat_state = [val for row in state for val in row]
        return sorted(flat_state) == list(range(9))

    def heuristic_value(value, row, col, goal_state, state):
        """Heuristic to prioritize values matching goal_state."""
        score = 0
        if row < len(goal_state) and col < len(goal_state[0]):
            score += 0 if value == goal_state[row][col] else 1
        return score

    def fill_state(state, used_numbers, cell_index, temp_states):
        """Randomized backtracking to fill the state, storing all states."""
        nonlocal attempts
        if attempts >= max_attempts:
            print(f"Debug: Reached max attempts ({max_attempts})")
            return False

        if cell_index == ROWS * COLS:
            if is_complete_state(state):
                if len(all_states) < max_states_to_store:
                    all_states.append([row[:] for row in state])
                if [row[:] for row in state] == goal_state:
                    if len(all_states) < max_states_to_store or all_states[-1] != state:
                        all_states.append([row[:] for row in state])
                    print(f"Debug: Goal state found: {state} after {attempts} attempts")
                    return True
                else:
                    print(f"Debug: Complete state (not goal): {state}")
            return False

        row, col = divmod(cell_index, COLS)
        values = [num for num in range(9) if num not in used_numbers]
        values.sort(key=lambda v: heuristic_value(v, row, col, goal_state, state))
        random.shuffle(values)  # Shuffle all values

        for num in values:
            print(f"Debug: Trying value {num} at position ({row}, {col})")
            state[row][col] = num
            used_numbers.add(num)
            if len(all_states) < max_states_to_store:
                all_states.append([row[:] for row in state])
            attempts += 1

            if is_consistent(state, row, col, num):
                temp_states.append([row[:] for row in state])
                if fill_state(state, used_numbers, cell_index + 1, temp_states):
                    return True
                temp_states.pop()
            else:
                print(f"Debug: Value {num} at ({row}, {col}) failed consistency check")

            state[row][col] = 0
            used_numbers.remove(num)
            if len(all_states) < max_states_to_store:
                all_states.append([row[:] for row in state])

        return False

    # Validate inputs
    start_state = np.array(start_state).tolist() if isinstance(start_state, np.ndarray) else start_state
    goal_state = np.array(goal_state).tolist() if isinstance(goal_state, np.ndarray) else goal_state
    if not (isinstance(start_state, list) and isinstance(goal_state, list) and
            len(start_state) == ROWS and len(goal_state) == ROWS and
            all(len(row) == COLS for row in start_state + goal_state)):
        print(f"Invalid input: Expected {ROWS}x{COLS} matrices")
        return [], 0, 0

    # Check if goal_state is valid and solvable
    if not is_valid_goal(goal_state):
        print("Invalid goal_state: Does not satisfy +1 horizontal or +3 vertical constraints")
        return [], 0, 0
    if not is_solvable(goal_state):
        print("Invalid goal_state: Unsolvable based on inversion count")
        return [], 0, 0

    # Initialize state and variables
    initial_state = [[0] * COLS for _ in range(ROWS)]
    used_numbers = set()
    temp_states = []

    # Run randomized backtracking
    found = fill_state(initial_state, used_numbers, 0, temp_states)
    if not found and temp_states:
        if len(all_states) < max_states_to_store:
            all_states.extend(temp_states)

    print(f"Random Generate and Test: {'Goal state found' if found else 'Failed to find goal state'} after {attempts} attempts.")
    return all_states, time.time() - start_time, attempts

def q_learning(start_state, goal_state, episodes=20000, alpha=0.1, gamma=0.95, epsilon=1.0, 
               epsilon_decay=0.999, min_epsilon=0.05, max_steps_per_episode=10000):
    """Q-Learning for solving the 8-puzzle with improved exploration and reward structure."""
    print("Running Q-Learning...")
    q_table = {}  # Q-table: {(state, action): Q-value}
    start_time = time.time()

    def state_to_tuple(state):
        """Convert state to tuple for Q-table keys."""
        return tuple(tuple(row) for row in state)

    def get_q_value(state, action):
        """Get Q-value, initializing with optimistic value for unvisited pairs."""
        return q_table.get((state_to_tuple(state), action), 10.0)  # Optimistic initialization

    def set_q_value(state, action, value):
        q_table[(state_to_tuple(state), action)] = value

    def choose_action(state, epsilon_now):
        """Choose action using epsilon-greedy strategy."""
        if random.random() < epsilon_now:
            return random.choice(ACTION_LIST)  # Explore
        q_values = {action: get_q_value(state, action) for action in ACTION_LIST}
        max_q = max(q_values.values())
        return random.choice([a for a, q in q_values.items() if q == max_q])  # Exploit

    def calculate_reward(state, next_state, prev_heuristic):
        """Calculate reward based on progress toward the goal."""
        if next_state == goal_state:
            return 100.0  # High reward for reaching the goal
        elif next_state is None:
            return -10.0  # Penalty for invalid moves
        else:
            current_h = heuristic(next_state, goal_state)
            # Reward progress toward the goal
            delta_h = prev_heuristic - current_h
            return delta_h * 0.5 - 1.0  # Reward heuristic improvement, penalize each step

    states_visited = 0
    for episode in range(episodes):
        current_state = copy.deepcopy(start_state)
        steps = 0
        visited_in_episode = set()  # Track states to detect cycles
        prev_h = heuristic(current_state, goal_state)
        total_reward = 0  # Track total reward for the episode

        if episode % 500 == 0:
            print(f"Episode {episode}/{episodes}, Q-table size: {len(q_table)}, Epsilon: {epsilon:.3f}")

        while current_state != goal_state and steps < max_steps_per_episode:
            state_tuple = state_to_tuple(current_state)
            visited_in_episode.add(state_tuple)

            action = choose_action(current_state, epsilon)
            next_state = apply_action(current_state, action)

            reward = calculate_reward(current_state, next_state, prev_h)
            total_reward += reward  # Accumulate reward for the episode

            if next_state is None:
                next_state = current_state  # Stay in place for invalid moves
            else:
                prev_h = heuristic(next_state, goal_state)

            # Update Q-value
            max_next_q = max(get_q_value(next_state, a) for a in ACTION_LIST)
            current_q = get_q_value(current_state, action)
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            set_q_value(current_state, action, new_q)

            current_state = next_state
            steps += 1

            # Early termination if stuck in a cycle
            if len(visited_in_episode) > 100 and steps > 100:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        # Print total reward for the episode
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
        states_visited += len(visited_in_episode)

    # Extract the best path
    path = [copy.deepcopy(start_state)]
    current_state = copy.deepcopy(start_state)
    steps = 0
    visited_states = set([state_to_tuple(current_state)])
    max_path_steps = max_steps_per_episode * 2  # Allow longer path extraction

    while current_state != goal_state and steps < max_path_steps:
        action = choose_action(current_state, 0)  # Greedy policy
        next_state = apply_action(current_state, action)
        if next_state is None or state_to_tuple(next_state) in visited_states:
            break  # Avoid cycles or invalid moves
        path.append(copy.deepcopy(next_state))
        current_state = next_state
        visited_states.add(state_to_tuple(current_state))
        steps += 1

    time_taken = time.time() - start_time

    print(f"Q-Learning: Visited {states_visited} unique states, Q-table size: {len(q_table)}")
    if current_state == goal_state:
        print(f"Q-Learning: Goal reached in {steps} steps.")
        return path, time_taken, len(q_table)
    else:
        print("Q-Learning: Failed to find a solution.")
        return path, time_taken, len(q_table)

# --- GUI Class ---
class PuzzleGUI:
    def __init__(self, master):
        self.master = master
        master.title("8-PUZZLE SOLVER VISUALIZATION")
        self.current_displayed_state = copy.deepcopy(current_start_state)
        master.geometry("1000x750")
        master.configure(bg="#FFFFFF")

        self.title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.grid_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self.detail_font = tkfont.Font(family="Helvetica", size=12)
        self.grid_label_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        self.BG_COLOR = "#FFFFFF"
        self.START_END_TILE_COLOR = "#60D060"
        self.CURRENT_TILE_COLOR = "#9370DB"
        self.EMPTY_TILE_COLOR = "#F5F5F5"
        self.GRID_BORDER_COLOR = "#A0A0A0"
        self.TEXT_COLOR_ON_TILE = "#FFFFFF"
        self.TEXT_COLOR_LABEL = "#FF0000"
        self.TEXT_COLOR_CURRENT_LABEL = "#8A2BE2"
        self.DETAIL_BG = "#FFFFFF"
        self.DETAIL_BORDER = "#000000"
        self.DETAIL_TEXT_COLOR = "#000000"
        self.BUTTON_FG_COLOR = "#FFFFFF"
        self.BUTTON_COLORS = [
            "#FF5757", "#42D6A4", "#FF9F43", "#9370DB", "#FFD700",
            "#32CD32", "#FF4500", "#00BFFF", "#FF1493", "#778899",
            "#FF6347", "#20B2AA", "#FFA07A", "#D2691E", "#4682B4",
            "#90EE90", "#ADD8E6", "#FFB6C1", "#8B008B", "#4B0082"
        ]

        self.animation_running = False
        self.stop_animation_flag = False
        self.animation_thread = None
        self.animation_speed = tk.DoubleVar(value=0.6)
        self.current_step_count = tk.IntVar(value=0)
        self.algorithm_status_text = tk.StringVar(value="Status: Idle")

        self.tile_size = 60
        self.grid_padding = 4
        self.canvas_width = COLS * self.tile_size + (COLS + 1) * self.grid_padding
        self.canvas_height = ROWS * self.tile_size + (ROWS + 1) * self.grid_padding

        self.top_control_frame = tk.Frame(master, bg=self.BG_COLOR, pady=10)
        self.top_control_frame.pack(side=tk.TOP, fill=tk.X)
        self.main_display_frame = tk.Frame(master, bg=self.BG_COLOR, padx=20, pady=10)
        self.main_display_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        algo_frame = tk.Frame(self.top_control_frame, bg=self.BG_COLOR)
        algo_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        tk.Label(algo_frame, text="Algorithms:", font=self.button_font, bg=self.BG_COLOR).pack(side=tk.LEFT, padx=(0,5), anchor='n')

        self.buttons = []
        button_labels = ["DFS", "BFS", "UCS", "A*", "Greedy", "IDS", "IDA*",
                        "SimpleHC", "SteepestHC", "StochasticHC", "SA", "Beam", "AOSerach",
                        "Sensorless", "POsearch",  
                        "GA", "Backtracking", "AC3", "G&T", "Q-Learning"]
        beam_width_default = 5

        algo_map = {
            "DFS": dfs,
            "BFS": bfs,
            "UCS": ucs,
            "A*": a_star,
            "Greedy": greedy_best_first,
            "IDS": ids,
            "IDA*": ida_star,
            "SimpleHC": simple_hill_climbing,
            "SteepestHC": steepest_ascent_hill_climbing,
            "StochasticHC": stochastic_hill_climbing,
            "SA": simulated_annealing,
            "Beam": partial(beam_search, beam_width=beam_width_default),
            "AOSerach": partial(and_or_search, get_successors=get_successors),
            "Sensorless": bfs_belief_search,
            "POsearch": bfs_partially_observable_search,
            "GA": genetic_algorithm,
            "Backtracking": backtracking_csp_solve,
            "AC3": ac3_csp_solve,
            "G&T": generate_and_test,
            "Q-Learning": q_learning
        }

        button_container = tk.Frame(algo_frame, bg=self.BG_COLOR)
        button_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        max_buttons_per_row = 8
        current_row_frame = None

        for i, label in enumerate(button_labels):
            if i % max_buttons_per_row == 0:
                current_row_frame = tk.Frame(button_container, bg=self.BG_COLOR)
                current_row_frame.pack(fill=tk.X, pady=1)

            color_index = i % len(self.BUTTON_COLORS)
            color = self.BUTTON_COLORS[color_index]

            algo = algo_map[label]
            cmd = partial(self.run_algorithm_thread, algo, label)

            button = tk.Button(current_row_frame, text=label, font=self.button_font,
                            bg=color, fg=self.BUTTON_FG_COLOR, width=12, height=1,
                            command=cmd, relief=tk.RAISED, borderwidth=2)
            button.pack(side=tk.LEFT, padx=2, pady=1)
            self.buttons.append(button)

        speed_stop_frame = tk.Frame(self.top_control_frame, bg=self.BG_COLOR)
        speed_stop_frame.pack(side=tk.RIGHT, padx=10, anchor='n')
        tk.Label(speed_stop_frame, text="Speed:", font=self.button_font, bg=self.BG_COLOR).pack(side=tk.LEFT, padx=(0,2))
        self.speed_slider = tk.Scale(speed_stop_frame, from_=0.1, to=1.5, resolution=0.1,
                                    orient=tk.HORIZONTAL, variable=self.animation_speed,
                                    bg=self.BG_COLOR, length=100, showvalue=0)
        self.speed_slider.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(speed_stop_frame, text="Stop", font=self.button_font,
                                    bg="#DC143C", fg=self.BUTTON_FG_COLOR, width=5, height=1,
                                    command=self.stop_animation, relief=tk.RAISED, borderwidth=2,
                                    state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.top_grids_frame = tk.Frame(self.main_display_frame, bg=self.BG_COLOR)
        self.top_grids_frame.pack(pady=10, anchor='n', fill=tk.X, expand=True)

        self.bottom_frame = tk.Frame(self.main_display_frame, bg=self.BG_COLOR)
        self.bottom_frame.pack(pady=10, anchor='n', fill=tk.X, expand=True)

        self.initial_canvas_frame = self.create_grid_canvas_with_label(self.top_grids_frame, "Start", self.TEXT_COLOR_LABEL)
        self.goal_canvas_frame = self.create_grid_canvas_with_label(self.top_grids_frame, "End", self.TEXT_COLOR_LABEL)
        self.current_canvas_frame = self.create_grid_canvas_with_label(self.bottom_frame, "Current", self.TEXT_COLOR_CURRENT_LABEL)

        self.initial_canvas = self.initial_canvas_frame.winfo_children()[1]
        self.goal_canvas = self.goal_canvas_frame.winfo_children()[1]
        self.current_canvas = self.current_canvas_frame.winfo_children()[1]

        self.initial_canvas_frame.pack(side=tk.LEFT, padx=30, expand=True)
        self.goal_canvas_frame.pack(side=tk.LEFT, padx=30, expand=True)
        self.current_canvas_frame.pack(side=tk.LEFT, padx=30, expand=True)

        self.detail_frame = tk.Frame(self.bottom_frame, bg=self.DETAIL_BG, bd=1, relief=tk.SOLID,
                                    width=200, height=self.canvas_height + self.grid_label_font.metrics('linespace') + 5)
        self.detail_frame.pack(side=tk.LEFT, padx=30, pady=0, anchor='n', fill=tk.Y)
        self.detail_frame.pack_propagate(False)

        tk.Label(self.detail_frame, text="Result Details", font=self.grid_label_font, bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR).pack(pady=5)
        self.status_display_label = tk.Label(self.detail_frame, textvariable=self.algorithm_status_text, font=self.detail_font,
                                bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR, wraplength=180)
        self.status_display_label.pack(pady=5, padx=10, anchor='w')
        self.steps_display_label = tk.Label(self.detail_frame, text="Path Steps: -", font=self.detail_font,
                                bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR, wraplength=180)
        self.steps_display_label.pack(pady=5, padx=10, anchor='w')
        self.time_display_label = tk.Label(self.detail_frame, text="Time: -", font=self.detail_font,
                                bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR, wraplength=180)
        self.time_display_label.pack(pady=5, padx=10, anchor='w')
        self.nodes_display_label = tk.Label(self.detail_frame, text="Count: -", font=self.detail_font,
                                bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR, wraplength=180)
        self.nodes_display_label.pack(pady=5, padx=10, anchor='w')

        print(f"Updating initial grid with: {current_start_state}")
        self.update_grid(self.initial_canvas, current_start_state, self.START_END_TILE_COLOR)
        print(f"Updating goal grid with: {current_goal_state}")
        self.update_grid(self.goal_canvas, current_goal_state, self.START_END_TILE_COLOR)
        self.update_grid(self.current_canvas, self.current_displayed_state, self.CURRENT_TILE_COLOR)


    def create_grid_canvas_with_label(self, parent_frame, label_text, label_color):
        frame = tk.Frame(parent_frame, bg=self.BG_COLOR)
        label = tk.Label(frame, text=label_text, font=self.grid_label_font, bg=self.BG_COLOR, fg=label_color)
        label.pack(pady=(0, 5))
        canvas = tk.Canvas(frame, width=self.canvas_width, height=self.canvas_height,
                           bg=self.BG_COLOR, highlightthickness=0)
        canvas.pack()
        return frame

    def update_grid(self, canvas, state, tile_base_color):
        canvas.delete("all")
        if state is None:
            print("State is None, skipping update")
            return
        state_list = tuple_to_state(state)
        print(f"State list: {state_list}")  # Log để kiểm tra
        if len(state_list) != 3 or any(len(row) != 3 for row in state_list):
            raise ValueError(f"Invalid state format: {state_list}, expected 3x3 matrix")
        for r in range(ROWS):
            for c in range(COLS):
                val = state_list[r][c]
                x0 = c * self.tile_size + (c + 1) * self.grid_padding
                y0 = r * self.tile_size + (r + 1) * self.grid_padding
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size

                if val == 0:
                    canvas.create_rectangle(x0, y0, x1, y1, fill=self.EMPTY_TILE_COLOR, outline=self.GRID_BORDER_COLOR, width=1)
                else:
                    canvas.create_rectangle(x0, y0, x1, y1, fill=tile_base_color, outline=self.GRID_BORDER_COLOR, width=1)
                    canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(val),
                                    font=self.grid_font, fill=self.TEXT_COLOR_ON_TILE)

    def update_detail_box(self, status=None, steps=None, time_val=None, count_val=None):
        """Updates details box. Count can be nodes, beliefs, or generations/evaluations."""
        if status is not None:
            self.algorithm_status_text.set(f"Status: {status}")
        if steps is not None:
            self.steps_display_label.config(text=f"Path Steps: {steps}" if steps != -1 else "Path Steps: -")
        if time_val is not None:
            self.time_display_label.config(text=f"Time: {time_val:.3f}s" if isinstance(time_val, (int, float)) else f"Time: {time_val}")
        if count_val is not None:
            # Determine label based on current status or algorithm type
            current_status_text = self.algorithm_status_text.get()
            belief_algorithms = ["Running Sensorless", "Running Partially Observable", "Plan Found", "Belief"]
            is_belief_algorithm = any(keyword in current_status_text for keyword in belief_algorithms)

            if is_belief_algorithm:
                label = "Beliefs"
            elif "Running Genetic Algorithm" in current_status_text or "(max iterations/generations)." in current_status_text:
                label = "Evaluations"  # Or Generations
            else:
                label = "Nodes"

            if isinstance(count_val, (int, float)):
                count_str = f"{int(count_val)}" if count_val == int(count_val) else f"{count_val:.2f}"
                self.nodes_display_label.config(text=f"{label}: {count_str}")
            else:
                self.nodes_display_label.config(text=f"{label}: {count_val}")

    def set_buttons_state(self, state):
        for button in self.buttons:
            button.config(state=state)
        self.stop_button.config(state=tk.NORMAL if state == tk.DISABLED else tk.DISABLED)
        self.speed_slider.config(state=state)

    def run_algorithm_thread(self, algorithm_func, algo_name):
        if self.animation_running:
            messagebox.showwarning("Busy", "An animation or algorithm is already running.")
            return

        self.set_buttons_state(tk.DISABLED)
        self.animation_running = True
        self.stop_animation_flag = False

        self.current_step_count.set(0)
        self.update_detail_box(status=f"Running {algo_name}... Please wait.", steps="-", time_val="-", count_val="-")
        self.update_grid(self.current_canvas, current_start_state, self.CURRENT_TILE_COLOR)
        self.master.update()

        def target():
            start_time = time.time()
            try:
                result = algorithm_func(copy.deepcopy(current_start_state), copy.deepcopy(current_goal_state))
                self.master.after(0, self._handle_algorithm_result, result, algo_name, time.time() - start_time)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.master.after(0, self._handle_algorithm_error, algo_name, e, time.time() - start_time)

        self.animation_thread = threading.Thread(target=target, daemon=True)
        self.animation_thread.start()

    def _handle_algorithm_result(self, result, algo_name, reported_time_taken):
        solution_data = None
        actual_time_taken = reported_time_taken
        count_info = "-"
        solution_found_msg = None
        path_to_animate = None # Path of states for animation

        if result is not None and len(result) == 3:
             solution_data, time_from_algo, count_info = result
             actual_time_taken = time_from_algo if isinstance(time_from_algo, (int, float)) else reported_time_taken
        else:
             self.update_detail_box(status=f"Error: Bad result format from {algo_name}")
             if result is not None:
                  messagebox.showerror("Algorithm Result Error", f"Algorithm {algo_name} returned unexpected value: {result}")
             self.animation_running = False
             self.set_buttons_state(tk.NORMAL)
             return

        # Update count display *after* determining algorithm type from algo_name
        self.update_detail_box(time_val=actual_time_taken, count_val=count_info)


        if solution_data is not None:
            if isinstance(solution_data, list) and solution_data:
                if isinstance(solution_data[0], (list, tuple)):
                     # It's a path (list of states/tuples) - State Space or GA path
                     solution_path = [tuple_to_state(s) for s in solution_data]
                     path_steps = len(solution_path) - 1 if len(solution_path) > 0 else 0
                     is_goal_reached = False
                     if len(solution_path) > 0:
                        is_goal_reached = (solution_path[-1] == current_goal_state)

                     self.update_detail_box(status="Solution Found!" if is_goal_reached else "Path Found", steps=path_steps)
                     solution_found_msg = f"{algo_name} finished.\nPath length: {path_steps}.\nGoal reached: {is_goal_reached}."
                     if algo_name == "GA" and not is_goal_reached:
                          solution_found_msg = f"{algo_name} finished (max iterations/generations).\nBest path found has {path_steps} steps.\nGoal reached: {is_goal_reached}."


                     if path_steps >= 0: path_to_animate = solution_path
                     else: solution_found_msg += "\nInvalid path."; path_to_animate = None

                elif isinstance(solution_data[0], str):
                     # It's an action sequence (list of actions) - from Belief Space / Sensorless
                     action_sequence = solution_data
                     path_steps = len(action_sequence)
                     self.update_detail_box(status="Plan Found!", steps=path_steps)

                     # Simulate the plan from the current start state for animation
                     # This is for visualization; Sensorless doesn't actually know the initial state.
                     final_state_sim, path_sim, sim_valid = simulate_path(action_sequence, current_start_state, max_len=path_steps + 1) # Simulate up to path length
                     is_goal_reached_sim = sim_valid and (final_state_sim == current_goal_state)

                     solution_found_msg = f"{algo_name} found a plan of length {path_steps} actions."
                     if algo_name in ["bfs_belief_search"]: # Mention belief space for these
                          solution_found_msg += "\n(Plan guarantees goal if starting from initial belief set.)"
                          solution_found_msg += f"\nSimulating plan from current Start state reaches goal: {is_goal_reached_sim}"

                     path_to_animate = path_sim

                else:
                     self.update_detail_box(status=f"Result from {algo_name} unclear.")
                     solution_found_msg = f"{algo_name} finished but returned unexpected data type ({type(solution_data)})."
                     path_to_animate = None
            else:
                 self.update_detail_box(status=f"Result from {algo_name} unclear.")
                 solution_found_msg = f"{algo_name} finished but returned empty data."
                 path_to_animate = None

        else: # solution_data is None
            self.update_detail_box(status="No Solution Found", steps="-")
            solution_found_msg = f"{algo_name} did not find a solution."
            path_to_animate = None

        if solution_found_msg:
             messagebox.showinfo("Algorithm Result", solution_found_msg)

        if path_to_animate and len(path_to_animate) > 0:
             self._animate_solution(deque(path_to_animate))
        else:
             self.animation_running = False
             self.set_buttons_state(tk.NORMAL)


    def _handle_algorithm_error(self, algo_name, error, reported_time_taken):
        self.update_detail_box(status=f"Error in {algo_name}", time_val=reported_time_taken, steps="-", count_val="-")
        messagebox.showerror("Algorithm Error", f"An error occurred during {algo_name}:\n{str(error)}\nPlease check the console for details.")
        self.animation_running = False
        self.set_buttons_state(tk.NORMAL)


    def _animate_solution(self, solution_path_deque):
        if self.stop_animation_flag:
            self.update_detail_box(status="Animation Stopped")
            self.animation_running = False
            self.set_buttons_state(tk.NORMAL)
            self.stop_animation_flag = False
            return

        if not solution_path_deque:
            is_goal_reached = (self.current_displayed_state == current_goal_state)
            final_status = "Animation Finished (Goal Reached)" if is_goal_reached else "Animation Finished (Goal Not Reached)"
            self.update_detail_box(status=final_status)
            self.animation_running = False
            self.set_buttons_state(tk.NORMAL)
            return

        current_step_state = solution_path_deque.popleft()
        self.current_displayed_state = current_step_state
        step_num_to_display = self.current_step_count.get()

        self.update_grid(self.current_canvas, current_step_state, self.CURRENT_TILE_COLOR)
        self.update_detail_box(steps=self.current_step_count.get())
        self.current_step_count.set(step_num_to_display + 1)


        delay_ms = int(self.animation_speed.get() * 1000)
        self.master.after(delay_ms, self._animate_solution, solution_path_deque)

    def stop_animation(self):
        if self.animation_running:
            self.stop_animation_flag = True
            
# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = PuzzleGUI(root)
    root.mainloop()