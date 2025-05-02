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

# --- Core Logic ---
ROWS, COLS = 3, 3
start_state_default = [[2, 6, 5], [0, 8, 7], [4, 3, 1]]
goal_state_default = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
current_start_state = copy.deepcopy(start_state_default)
current_goal_state = copy.deepcopy(goal_state_default)

# --- Helper Functions ---
def find_zero(state):
    """Finds the position (row, col) of the empty tile (0). Works for list or tuple states."""
    for i, row in enumerate(state):
        try:
            row_list = list(row) if isinstance(row, tuple) else row
            j = row_list.index(0)
            return i, j
        except ValueError:
            continue
    return None

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

def state_to_tuple(state):
    """Converts a state (list[list[int]] or tuple[tuple[int]]) to a tuple (tuple[tuple[int]]) for hashing."""
    if isinstance(state, tuple): return state
    return tuple(tuple(row) for row in state)

def tuple_to_state(state_tuple):
    """Converts a state tuple (tuple[tuple[int]]) to a list (list[list[int]])."""
    if isinstance(state_tuple, list): return state_tuple
    return [list(row) for row in state_tuple]

def heuristic(state, goal):
    """Calculates the Manhattan distance heuristic. Works for list or tuple states."""
    distance = 0
    goal_list = tuple_to_state(goal) if isinstance(goal, tuple) else goal
    goal_positions = {}
    for r in range(ROWS):
        for c in range(COLS):
            val = goal_list[r][c]
            if val != 0:
                goal_positions[val] = (r, c)

    state_list = tuple_to_state(state) if isinstance(state, tuple) else state

    for r in range(ROWS):
        for c in range(COLS):
            val = state_list[r][c]
            if val != 0 and val in goal_positions:
                goal_r, goal_c = goal_positions[val]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

# --- Action Definitions ---
ACTION_MAP = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}
ACTION_LIST = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def apply_action(state, action_name):
    """
    Applies an action to a state (list or tuple).
    Returns the resulting state (same type) or None if invalid.
    Handles state conversion internally.
    """
    is_tuple = isinstance(state, tuple)
    state_list = tuple_to_state(state)

    zero_pos = find_zero(state_list)
    if zero_pos is None: return None

    r, c = zero_pos
    if action_name not in ACTION_MAP: return None

    dr, dc = ACTION_MAP[action_name]
    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS:
        new_state_list = [row[:] for row in state_list]
        new_state_list[r][c], new_state_list[nr][nc] = new_state_list[nr][nc], new_state_list[r][c]
        return state_to_tuple(new_state_list) if is_tuple else new_state_list
    else:
        return None

initial_belief_state_tuples = {
    state_to_tuple(start_state_default), # The actual default start state
    state_to_tuple([[2, 6, 5], [8, 7, 0], [4, 3, 1]]), # Zero at (1,2)
    state_to_tuple([[2, 6, 0], [8, 7, 5], [4, 3, 1]]), # Zero at (0,2)
    state_to_tuple([[2, 6, 5], [8, 0, 7], [4, 3, 1]]), # Zero at (1,1)
    state_to_tuple([[2, 0, 5], [8, 6, 7], [4, 3, 1]]), # Zero at (0,1)
    state_to_tuple([[0, 2, 5], [8, 6, 7], [4, 3, 1]]), # Zero at (0,0)
    state_to_tuple([[2, 6, 5], [4, 8, 7], [0, 3, 1]]), # Zero at (2,0)
    state_to_tuple([[2, 6, 5], [4, 3, 7], [8, 0, 1]]), # Zero at (2,1)
    state_to_tuple([[2, 6, 5], [4, 3, 1], [8, 7, 0]]), # Zero at (2,2)
    state_to_tuple([[0, 6, 5], [2, 3, 1], [4, 8, 7]]), # Zero at (0,0)
    state_to_tuple([[2, 0, 5], [6, 3, 1], [4, 8, 7]]), # Zero at (0,1)
    state_to_tuple([[2, 5, 0], [6, 3, 1], [4, 8, 7]]), # Zero at (0,2)
    state_to_tuple([[2, 5, 6], [0, 3, 1], [4, 8, 7]]), # Zero at (1,0)
    state_to_tuple([[2, 5, 6], [4, 3, 1], [0, 8, 7]]), # Zero at (2,0)
    state_to_tuple([[2, 5, 6], [4, 3, 1], [8, 7, 0]]), # Zero at (2,1)
    state_to_tuple([[2, 5, 6], [4, 0, 1], [8, 3, 7]]), # Zero at (1,1)
    state_to_tuple([[2, 5, 6], [4, 3, 1], [7, 0, 8]]), # Zero at (2,2)
    state_to_tuple([[2, 5, 6], [4, 3, 1], [7, 8, 0]]), # Zero at (2,2)
}

def sensorless_search_bfs(start_state_dummy, goal_state):
    """
    Sensorless Search using BFS on Belief Space.
    """
    print("Running Sensorless Search (BFS on Belief Space)...")
    initial_belief_state_set = initial_belief_state_tuples
    start_time = time.time()
    belief_states_explored = 0
    goal_state_tuple = state_to_tuple(goal_state)
    initial_belief_frozenset = frozenset(initial_belief_state_set)

    if not initial_belief_frozenset:
        print("Initial belief state set is empty.")
        return None, 0.0, 0
    if len(initial_belief_frozenset) == 1 and goal_state_tuple in initial_belief_frozenset:
         print("Goal reached immediately (initial belief is the goal belief state).")
         return [], 0.0, 1 # 0 actions needed

    queue = deque([(initial_belief_frozenset, [])])
    visited = {initial_belief_frozenset}

    while queue:
        current_belief_frozenset, actions = queue.popleft()
        belief_states_explored += 1

        for action in ACTION_LIST:
            next_belief_state_set = set()
            possible_for_all = True
            # Apply action to *every* state in the current belief
            for state_tuple in current_belief_frozenset:
                next_state_tuple = apply_action(state_tuple, action) # apply_action handles tuple
                if next_state_tuple is None:
                    # If action is invalid for *any* state, it's invalid for the belief
                    possible_for_all = False
                    break
                next_belief_state_set.add(next_state_tuple)

            # Only proceed if the action was valid for all states AND it resulted in a new belief state
            if possible_for_all and next_belief_state_set:
                next_belief_frozenset = frozenset(next_belief_state_set)

                # Check if this next belief state has been visited before
                if next_belief_frozenset not in visited:
                    visited.add(next_belief_frozenset)
                    new_actions = actions + [action]
                    queue.append((next_belief_frozenset, new_actions))

                    # Check goal condition: the new belief state contains *only* the goal state
                    if len(next_belief_frozenset) == 1 and goal_state_tuple in next_belief_frozenset:
                        time_taken = time.time() - start_time
                        print(f"Sensorless (BFS): Plan found of length {len(new_actions)}.")
                        return new_actions, time_taken, belief_states_explored

    time_taken = time.time() - start_time
    print("Sensorless (BFS): Failed to find a plan.")
    return None, time_taken, belief_states_explored


def dfs_belief_search(start_state_dummy, goal_state):
    print("Running DFS on Belief Space (Sensorless)...")
    initial_belief_state_set = initial_belief_state_tuples
    start_time = time.time()
    belief_states_explored = 0
    goal_state_tuple = state_to_tuple(goal_state)
    initial_belief_frozenset = frozenset(initial_belief_state_set)

    if not initial_belief_frozenset: return None, 0.0, 0
    if len(initial_belief_frozenset) == 1 and goal_state_tuple in initial_belief_frozenset:
        print("Goal reached immediately (initial belief is the goal belief state).")
        return [], 0.0, 1

    stack = [(initial_belief_frozenset, [])]
    visited = {initial_belief_frozenset}

    while stack:
        current_belief_frozenset, actions = stack.pop()
        belief_states_explored += 1

        # Iterate actions in reverse for typical DFS exploration order
        for action in reversed(ACTION_LIST):
            next_belief_state_set = set()
            possible_for_all = True
            for state_tuple in current_belief_frozenset:
                next_state_tuple = apply_action(state_tuple, action)
                if next_state_tuple is None:
                    possible_for_all = False
                    break
                next_belief_state_set.add(next_state_tuple)

            if possible_for_all and next_belief_state_set:
                next_belief_frozenset = frozenset(next_belief_state_set)
                if next_belief_frozenset not in visited:
                    visited.add(next_belief_frozenset)
                    new_actions = actions + [action]
                    stack.append((next_belief_frozenset, new_actions))

                    if len(next_belief_frozenset) == 1 and goal_state_tuple in next_belief_frozenset:
                        time_taken = time.time() - start_time
                        print(f"Sensorless (DFS): Plan found of length {len(new_actions)}.")
                        return new_actions, time_taken, belief_states_explored

    time_taken = time.time() - start_time
    print("Sensorless (DFS): Failed to find a plan.")
    return None, time_taken, belief_states_explored

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

def dls(current_state, goal, depth_limit, path, visited_in_path, nodes_visited_list):
    """Recursive helper for Limited Depth Search."""
    nodes_visited_list[0] += 1
    current_tuple = state_to_tuple(current_state)

    if current_state == goal: return path + [current_state], True
    if depth_limit == 0: return None, False

    visited_in_path_copy = visited_in_path.copy()
    visited_in_path_copy.add(current_tuple)

    for neighbor, _ in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple not in visited_in_path_copy:
             result_path, found_goal = dls(
                 neighbor, goal, depth_limit - 1, path + [current_state],
                 visited_in_path_copy, nodes_visited_list
             )
             if found_goal: return result_path, True

    return None, False

def ids(start, goal):
    """Iterative Deepening Depth-First Search (State Space)."""
    print("Running IDS (State Space)...")
    depth = 0
    total_nodes_visited = 0
    start_time = time.time()

    original_recursion_limit = sys.getrecursionlimit()
    new_limit = 5000
    limit_changed = False
    if new_limit > original_recursion_limit:
        try:
            sys.setrecursionlimit(new_limit)
            limit_changed = True
        except Exception as e: print(f"Could not set recursion limit for IDS: {e}")

    try:
        while True:
            nodes_visited_list = [0]
            result_path, found_goal = dls(start, goal, depth, [], set(), nodes_visited_list)
            total_nodes_visited += nodes_visited_list[0]

            if found_goal:
                print(f"IDS: Solution found at depth {depth}.")
                return result_path, time.time() - start_time, total_nodes_visited
            if depth > 30: # Safety break
                 print(f"IDS depth limit {depth} reached.")
                 break
            depth += 1
    except RecursionError:
        print(f"IDS Failed: Maximum recursion depth exceeded ({sys.getrecursionlimit()}).")
        return None, time.time() - start_time, total_nodes_visited
    finally:
        if limit_changed: sys.setrecursionlimit(original_recursion_limit)

    print("IDS: Failed to find a solution.")
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


def random_hill_climbing(start, goal, max_iterations=10000):
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


# --- CSP-style Backtracking ---
def _backtracking_recursive_csp_style(current_state, goal_state, path, visited_in_path, nodes_visited_list, start_time, time_limit):
    """Recursive helper for CSP-style Backtracking."""
    nodes_visited_list[0] += 1
    current_tuple = state_to_tuple(current_state)

    if time.time() - start_time > time_limit: return "Timeout", False
    if current_state == goal_state: return path, True

    visited_in_path_copy = visited_in_path.copy()
    visited_in_path_copy.add(current_tuple)

    for neighbor, _ in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple not in visited_in_path_copy:
            result, found_goal = _backtracking_recursive_csp_style(
                neighbor, goal_state, path + [neighbor], visited_in_path_copy,
                nodes_visited_list, start_time, time_limit
            )
            if found_goal or result == "Timeout": return result, found_goal

    return None, False

def backtracking_search_csp(start, goal, time_limit=30):
    """Backtracking Search (CSP Style) to find a path from start to goal."""
    print("Running Backtracking Search (CSP Style)...")
    start_time = time.time()
    nodes_visited_list = [0]
    path_init = [copy.deepcopy(start)]
    visited_init = set()

    original_recursion_limit = sys.getrecursionlimit()
    new_limit = 5000
    limit_changed = False
    if new_limit > original_recursion_limit:
        try:
            sys.setrecursionlimit(new_limit)
            limit_changed = True
        except Exception as e: print(f"Could not set recursion limit: {e}")

    solution_path = None
    status = "Unknown"
    try:
        result, found_goal = _backtracking_recursive_csp_style(
            start, goal, path_init, visited_init, nodes_visited_list, start_time, time_limit
        )

        if result == "Timeout": status = "Timeout"
        elif found_goal:
            status = "Solution Found"
            solution_path = result
        else: status = "Not Found"

    except RecursionError:
        print(f"Backtracking Search (CSP Style) Failed: Max recursion depth exceeded ({sys.getrecursionlimit()}).")
        status = "Recursion Limit Exceeded"
        solution_path = None
    except Exception as e:
        print(f"Backtracking Search (CSP Style) Error: {e}")
        status = f"Error: {e}"
        solution_path = None
    finally:
        if limit_changed: sys.setrecursionlimit(original_recursion_limit)

    time_taken = time.time() - start_time
    print(f"Backtracking Search (CSP Style) completed with status: {status}")
    return solution_path, time_taken, nodes_visited_list[0]

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
        sequence = [random.choice(ACTION_LIST) for _ in range(seq_len)]
        # Simple check: ensure at least the first move is valid from the *known* start state
        if sequence and apply_action(start_state_tuple, sequence[0]) is not None:
             population.append(sequence)

    while len(population) < size: # Ensure population size
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

def genetic_algorithm(start_state, goal_state, population_size=100, max_generations=100,
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
        # Ordered: State Space first, then Belief Space, then Others
        button_labels = ["DFS", "BFS", "UCS", "A*", "Greedy", "IDS", "IDA*",
                         "SimpleHC", "SteepestHC", "RandomHC", "SA", "Beam",
                         "BFS_Belief", "DFS_Belief",
                         "GA", "Backtracking"] # GA and Backtracking don't fit neatly into state/belief
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
            "RandomHC": random_hill_climbing,
            "SA": simulated_annealing,
            "Beam": partial(beam_search, beam_width=beam_width_default), # Use partial for beam width
            "BFS_Belief": sensorless_search_bfs, # Sensorless calls BFS Belief
            "DFS_Belief": dfs_belief_search,
            "GA": genetic_algorithm,
            "Backtracking": backtracking_search_csp
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

        self.update_grid(self.initial_canvas, current_start_state, self.START_END_TILE_COLOR)
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
        if state is None: return
        state_list = tuple_to_state(state)

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
             label = "Count" # Default label
             if "Running Sensorless" in current_status_text or "Plan Found" in current_status_text or "Belief" in current_status_text:
                  label = "Beliefs"
             elif "Running Genetic Algorithm" in current_status_text or "(max iterations/generations)." in current_status_text:
                  label = "Evaluations" # Or Generations
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
        self.update_detail_box(status=f"Running {algo_name}...", steps="-", time_val="-", count_val="-")
        self.update_grid(self.current_canvas, current_start_state, self.CURRENT_TILE_COLOR)
        self.master.update()

        def target():
            start_time = time.time()
            try:
                # Pass current_start_state and current_goal_state.
                # Belief search algorithms will use goal_state but ignore start_state,
                # operating on the initial_belief_state_tuples instead.
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
                     if algo_name in ["Sensorless", "DFS_Belief"]: # Mention belief space for these
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
        messagebox.showerror("Algorithm Error", f"An error occurred during {algo_name}:\n{error}")
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