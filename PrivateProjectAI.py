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
    for i, row in enumerate(state):
        try:
            j = row.index(0)
            return i, j
        except ValueError:
            continue
    return None

def get_neighbors(state):
    zero_pos = find_zero(state)
    if zero_pos is None: return []
    r, c = zero_pos
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            new_state = [row[:] for row in state]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            neighbors.append((new_state, 1))
    return neighbors

def state_to_tuple(state):
    if isinstance(state, tuple): return state
    return tuple(tuple(row) for row in state)

def heuristic(state, goal):
    distance = 0
    goal_positions = {goal[r][c]: (r, c) for r in range(ROWS) for c in range(COLS) if goal[r][c] != 0}
    for r in range(ROWS):
        for c in range(COLS):
            val = state[r][c]
            if val != 0:
                if val in goal_positions:
                    goal_r, goal_c = goal_positions[val]
                    distance += abs(r - goal_r) + abs(c - goal_c) # Corrected Manhattan distance
                else:
                    return float('inf')
    return distance

initial_belief_state_tuples = {
    state_to_tuple([[2, 6, 5], [8, 7, 0], [4, 3, 1]]),
    state_to_tuple([[2, 6, 0], [8, 7, 5], [4, 3, 1]]),
    state_to_tuple([[2, 6, 5], [8, 0, 7], [4, 3, 1]])
}
# --- Action Definitions ---
ACTION_MAP = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}
ACTION_LIST = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def apply_action(state_tuple, action_name):
    """
    Áp dụng một hành động lên một trạng thái (dạng tuple).
    Trả về trạng thái kết quả (dạng tuple) hoặc None nếu hành động không hợp lệ.
    """
    state_list = [list(row) for row in state_tuple]
    zero_pos = find_zero(state_list)
    if zero_pos is None: return None
    r, c = zero_pos

    if action_name not in ACTION_MAP: return None
    dr, dc = ACTION_MAP[action_name]
    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS:
        new_state_list = [row[:] for row in state_list]
        new_state_list[r][c], new_state_list[nr][nc] = new_state_list[nr][nc], new_state_list[r][c]
        return state_to_tuple(new_state_list)
    else:
        return None

# --- Belief Space Search Algorithms ---
# Chỉ giữ lại BFS và DFS trên không gian niềm tin

def dfs_belief_search(start_state, goal_state): # DFS on Belief Space
    """Tìm kiếm DFS trên không gian niềm tin."""
    initial_belief_state_set = initial_belief_state_tuples # Bỏ qua start_state bên trong
    start_time = time.time()
    belief_states_explored = 0
    goal_state_tuple = state_to_tuple(goal_state)
    initial_belief_frozenset = frozenset(initial_belief_state_set)
    stack = [(initial_belief_frozenset, [])]
    visited = {initial_belief_frozenset}

    while stack:
        current_belief_frozenset, actions = stack.pop()
        belief_states_explored += 1

        if len(current_belief_frozenset) == 1 and goal_state_tuple in current_belief_frozenset:
            time_taken = time.time() - start_time
            print(f"DFS Belief Search: Goal reached!")
            return actions, time_taken, belief_states_explored

        for action in reversed(ACTION_LIST):
            next_belief_state_set = set()
            possible_for_all = True
            for state_tuple in current_belief_frozenset:
                next_state_tuple = apply_action(state_tuple, action)
                if next_state_tuple is None:
                    possible_for_all = False
                    break
                else:
                    next_belief_state_set.add(next_state_tuple)

            if possible_for_all and next_belief_state_set:
                next_belief_frozenset = frozenset(next_belief_state_set)
                if next_belief_frozenset not in visited:
                    visited.add(next_belief_frozenset)
                    stack.append((next_belief_frozenset, actions + [action]))

    time_taken = time.time() - start_time
    print("DFS Belief Search: Failed - Stack empty.")
    return None, time_taken, belief_states_explored

def bfs_belief_search(start_state, goal_state): # BFS on Belief Space (Sensorless)
    """Tìm kiếm không cảm biến (BFS trên không gian niềm tin)."""
    initial_belief_state_set = initial_belief_state_tuples # Bỏ qua start_state bên trong
    start_time = time.time()
    belief_states_explored = 0
    goal_state_tuple = state_to_tuple(goal_state)
    initial_belief_frozenset = frozenset(initial_belief_state_set)
    queue = deque([(initial_belief_frozenset, [])])
    visited = {initial_belief_frozenset}

    while queue:
        current_belief_frozenset, actions = queue.popleft()
        belief_states_explored += 1

        if len(current_belief_frozenset) == 1 and goal_state_tuple in current_belief_frozenset:
            time_taken = time.time() - start_time
            print(f"BFS Belief Search: Goal reached!")
            return actions, time_taken, belief_states_explored

        for action in ACTION_LIST:
            next_belief_state_set = set()
            possible_for_all = True
            for state_tuple in current_belief_frozenset:
                next_state_tuple = apply_action(state_tuple, action)
                if next_state_tuple is None:
                    possible_for_all = False
                    break
                else:
                    next_belief_state_set.add(next_state_tuple)

            if possible_for_all and next_belief_state_set:
                next_belief_frozenset = frozenset(next_belief_state_set)
                if next_belief_frozenset not in visited:
                    visited.add(next_belief_frozenset)
                    queue.append((next_belief_frozenset, actions + [action]))

    time_taken = time.time() - start_time
    print("BFS Belief Search: Failed - Queue empty.")
    return None, time_taken, belief_states_explored

# --- Search Algorithms ---
def dfs(start, goal):
    stack = [(start, [])]
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while stack:
        state, path = stack.pop()
        nodes_visited_count += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in reversed(get_neighbors(state)):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                stack.append((neighbor, path + [state]))
    return None, time.time() - start_time, nodes_visited_count

def bfs(start, goal):
    queue = deque([(start, [])])
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while queue:
        state, path = queue.popleft()
        nodes_visited_count += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in get_neighbors(state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                queue.append((neighbor, path + [state]))
    return None, time.time() - start_time, nodes_visited_count

def ucs(start, goal):
    pq = [(0, start, [])]
    visited_costs = {state_to_tuple(start): 0}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        cost, state, path = heapq.heappop(pq)
        nodes_visited_count += 1
        state_tuple = state_to_tuple(state)
        if cost > visited_costs.get(state_tuple, float('inf')):
            continue
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, step_cost in get_neighbors(state):
            new_cost = cost + step_cost
            neighbor_tuple = state_to_tuple(neighbor)
            if new_cost < visited_costs.get(neighbor_tuple, float('inf')):
                visited_costs[neighbor_tuple] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [state]))
    return None, time.time() - start_time, nodes_visited_count

def a_star(start, goal):
    start_h = heuristic(start, goal)
    pq = [(start_h, 0, start, [])]
    visited_costs = {state_to_tuple(start): 0}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        _, cost, state, path = heapq.heappop(pq)
        nodes_visited_count += 1
        state_tuple = state_to_tuple(state)
        if cost > visited_costs.get(state_tuple, float('inf')):
             continue
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, step_cost in get_neighbors(state):
            new_cost = cost + step_cost
            neighbor_tuple = state_to_tuple(neighbor)
            if new_cost < visited_costs.get(neighbor_tuple, float('inf')):
                visited_costs[neighbor_tuple] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (priority, new_cost, neighbor, path + [state]))
    return None, time.time() - start_time, nodes_visited_count

def greedy_best_first(start, goal):
    pq = [(heuristic(start, goal), start, [])]
    visited = {state_to_tuple(start)}
    nodes_visited_count = 0
    start_time = time.time()
    while pq:
        _, state, path = heapq.heappop(pq)
        nodes_visited_count += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in get_neighbors(state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor, path + [state]))
    return None, time.time() - start_time, nodes_visited_count

def dls(current_state, goal, depth_limit, path, visited_in_path):
    current_tuple = state_to_tuple(current_state)
    nodes_visited_count = 1
    if current_state == goal:
          return path + [current_state], nodes_visited_count, True
    if depth_limit == 0:
        return None, nodes_visited_count, False
    visited_in_path.add(current_tuple)
    for neighbor, _ in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        if neighbor_tuple not in visited_in_path:
            result, nodes_count, found_goal = dls(neighbor, goal, depth_limit - 1, path + [current_state], visited_in_path.copy())
            nodes_visited_count += nodes_count
            if found_goal:
                return result, nodes_visited_count, True
    return None, nodes_visited_count, False

def ids(start, goal):
    depth = 0
    total_nodes_visited = 0
    start_time = time.time()
    while True:
        result, nodes_count, found_goal = dls(start, goal, depth, [], set())
        total_nodes_visited += nodes_count
        if found_goal:
            return result, time.time() - start_time, total_nodes_visited
        if depth > 30:
             print(f"IDS depth limit {depth} reached.")
             break
        depth += 1
    return None, time.time() - start_time, total_nodes_visited

def ida_star_search(node, g_cost, threshold, path, goal, visited_in_path):
    nodes_visited_count = 1
    f_cost = g_cost + heuristic(node, goal)
    if f_cost > threshold:
        return f_cost, None, nodes_visited_count
    if node == goal:
        return True, path + [node], nodes_visited_count
    min_next_threshold = float('inf')
    node_tuple = state_to_tuple(node)
    visited_in_path.add(node_tuple)
    for next_state, step_cost in get_neighbors(node):
         next_state_tuple = state_to_tuple(next_state)
         if next_state_tuple not in visited_in_path:
            result, found_path, nodes_count = ida_star_search(
                next_state, g_cost + step_cost, threshold, path + [node], goal, visited_in_path.copy()
            )
            nodes_visited_count += nodes_count
            if result is True:
                return True, found_path, nodes_visited_count
            if result < min_next_threshold:
                min_next_threshold = result
    return min_next_threshold, None, nodes_visited_count

def ida_star(start, goal):
    threshold = heuristic(start, goal)
    total_nodes_visited = 0
    start_time = time.time()
    while True:
        result, solution_path, nodes_count = ida_star_search(start, 0, threshold, [], goal, set())
        total_nodes_visited += nodes_count
        if result is True:
            return solution_path, time.time() - start_time, total_nodes_visited
        if result == float('inf'):
             print("IDA* search space exhausted, no solution found.")
             break
        if threshold > 100:
             print(f"IDA* threshold limit ({threshold}) reached.")
             break
        threshold = result
    return None, time.time() - start_time, total_nodes_visited

def simple_hill_climbing(start, goal):
    current = start
    path = [current]
    nodes_visited_count = 0
    start_time = time.time()
    current_h = heuristic(current, goal)
    while current != goal:
        nodes_visited_count += 1
        neighbors = get_neighbors(current)
        if not neighbors: break
        next_state = None
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count +=1
            if neighbor_h < current_h:
                next_state = neighbor_state
                current_h = neighbor_h
                break
        if next_state is None:
            if current != goal: print("Simple Hill Climbing got stuck at a local optimum.")
            break
        current = next_state
        path.append(current)
    return path, time.time() - start_time, nodes_visited_count

def steepest_ascent_hill_climbing(start, goal):
    current = start
    path = [current]
    nodes_visited_count = 0
    start_time = time.time()
    current_h = heuristic(current, goal)
    while current != goal:
        nodes_visited_count += 1
        neighbors = get_neighbors(current)
        if not neighbors: break
        best_neighbor = None
        best_h = current_h
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count += 1
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbor = neighbor_state
        if best_neighbor is None or best_h >= current_h:
             if current != goal: print("Steepest Ascent got stuck at a local optimum.")
             break
        current = best_neighbor
        current_h = best_h
        path.append(current)
    return path, time.time() - start_time, nodes_visited_count

def random_hill_climbing(start, goal, max_iterations=10000):
    current = start
    path = [current]
    nodes_visited_count = 0
    start_time = time.time()
    current_h = heuristic(current, goal)
    iterations = 0
    while current != goal and iterations < max_iterations:
        nodes_visited_count += 1
        iterations += 1
        neighbors = get_neighbors(current)
        if not neighbors: break
        evaluated_neighbors = []
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count += 1
            evaluated_neighbors.append((neighbor_state, neighbor_h))
        better_neighbors = [(n_state, n_h) for n_state, n_h in evaluated_neighbors if n_h < current_h]
        if better_neighbors:
            chosen_neighbor, chosen_h = random.choice(better_neighbors)
            current = chosen_neighbor
            current_h = chosen_h
            path.append(current)
        else:
            if current != goal: print("Random Hill Climbing got stuck (no better neighbors).")
            break
    if iterations >= max_iterations and current != goal:
        print(f"Random Hill Climbing reached max iterations ({max_iterations}) without finding goal.")
    return path, time.time() - start_time, nodes_visited_count

def simulated_annealing(start_state, goal_state, initial_temp=100.0, cooling_rate=0.99, min_temp=0.1, max_iterations_per_temp=100):
    start_time = time.time()
    nodes_visited = 0
    current_state = copy.deepcopy(start_state)
    current_energy = heuristic(current_state, goal_state)
    best_state = copy.deepcopy(current_state)
    best_energy = current_energy
    path_taken = [copy.deepcopy(current_state)]
    temp = initial_temp
    while temp > min_temp:
        for _ in range(max_iterations_per_temp):
            if current_state == goal_state:
                time_taken = time.time() - start_time
                print(f"SA: Goal found! Temp={temp:.2f}")
                return path_taken, time_taken, nodes_visited
            neighbors = get_neighbors(current_state)
            if not neighbors: break
            next_state, _ = random.choice(neighbors)
            nodes_visited += 1
            next_energy = heuristic(next_state, goal_state)
            delta_E = next_energy - current_energy
            accept = False
            if delta_E < 0:
                accept = True
            else:
                if temp > 1e-9:
                    probability = math.exp(-delta_E / temp)
                    if random.random() < probability:
                        accept = True
            if accept:
                current_state = next_state
                current_energy = next_energy
                path_taken.append(copy.deepcopy(current_state))
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = copy.deepcopy(current_state)
        temp *= cooling_rate
        if not neighbors: break
    time_taken = time.time() - start_time
    print(f"SA: Finished. Best energy reached: {best_energy}. Final Temp: {temp:.3f}")
    return path_taken, time_taken, nodes_visited

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

def ao_search(start_state, goal_state):
    start_time = time.time()
    nodes_visited = 0
    start_h = heuristic(start_state, goal_state)
    pq = [(start_h, 0, start_state, [start_state])]
    g_costs = {state_to_tuple(start_state): 0}
    while pq:
        f_cost_current, g_cost_current, current_state, path_current = heapq.heappop(pq)
        nodes_visited += 1
        if current_state == goal_state:
            time_taken = time.time() - start_time
            return path_current, time_taken, nodes_visited
        current_state_tuple = state_to_tuple(current_state)
        if g_cost_current > g_costs.get(current_state_tuple, float('inf')):
             continue
        for neighbor, step_cost in get_neighbors(current_state):
            new_g_cost = g_cost_current + step_cost
            neighbor_tuple = state_to_tuple(neighbor)
            if new_g_cost < g_costs.get(neighbor_tuple, float('inf')):
                g_costs[neighbor_tuple] = new_g_cost
                new_h_cost = heuristic(neighbor, goal_state)
                new_f_cost = new_g_cost + new_h_cost
                heapq.heappush(pq, (new_f_cost, new_g_cost, neighbor, path_current + [neighbor]))
    time_taken = time.time() - start_time
    return None, time_taken, nodes_visited

ACTION_MAP = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}
ACTION_LIST = ['UP', 'DOWN', 'LEFT', 'RIGHT']

def apply_action(state, action_name):
    zero_pos = find_zero(state)
    if zero_pos is None: return None
    r, c = zero_pos

    if action_name not in ACTION_MAP: return None
    dr, dc = ACTION_MAP[action_name]

    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS:
        state_list = [list(row) for row in state]
        state_list[r][c], state_list[nr][nc] = state_list[nr][nc], state_list[r][c]
        return state_to_tuple(state_list)
    else:
        return None


def sensorless_search(start_state, goal_state):
    initial_belief_state_set = initial_belief_state_tuples
    start_time = time.time()
    belief_states_explored = 0
    goal_state_tuple = state_to_tuple(goal_state)
    initial_belief_frozenset = frozenset(initial_belief_state_set)
    queue = deque([(initial_belief_frozenset, [])])
    visited = {initial_belief_frozenset}

    while queue:
        current_belief_frozenset, actions = queue.popleft()
        belief_states_explored += 1

        if len(current_belief_frozenset) == 1 and goal_state_tuple in current_belief_frozenset:
            time_taken = time.time() - start_time
            print(f"Sensorless Search: Goal reached! (Belief state contains only goal)")
            return actions, time_taken, belief_states_explored

        for action in ACTION_LIST:
            next_belief_state_set = set()
            possible_for_all = True
            for state_tuple in current_belief_frozenset:
                next_state_tuple = apply_action(state_tuple, action)
                if next_state_tuple is None:
                    possible_for_all = False
                    break
                else:
                    next_belief_state_set.add(next_state_tuple)

            if possible_for_all and next_belief_state_set:
                next_belief_frozenset = frozenset(next_belief_state_set)
                if next_belief_frozenset not in visited:
                    visited.add(next_belief_frozenset)
                    queue.append((next_belief_frozenset, actions + [action]))

    time_taken = time.time() - start_time
    print("Sensorless Search: Failed - Queue empty.")
    return None, time_taken, belief_states_explored

def simulate_path(action_sequence, start_state, max_len=50):
    current_state = copy.deepcopy(start_state)
    path = [copy.deepcopy(current_state)]
    visited_sim = {state_to_tuple(current_state)}

    for i, action in enumerate(action_sequence):
        if i >= max_len:
             return current_state, path, False

        next_state_tuple = apply_action(state_to_tuple(current_state), action)

        if next_state_tuple is None:
            return current_state, path, False

        if next_state_tuple in visited_sim:
            return tuple(list(r) for r in next_state_tuple), path + [tuple(list(r) for r in next_state_tuple)], False

        current_state = [list(row) for row in next_state_tuple]
        path.append(copy.deepcopy(current_state))
        visited_sim.add(next_state_tuple)

    return current_state, path, True

def _backtracking_recursive_csp_style(current_state, goal_state, path, visited_in_path, nodes_visited_list, start_time, time_limit):
    """Hàm đệ quy theo phong cách CSP Backtracking."""
    nodes_visited_list[0] += 1
    current_tuple = state_to_tuple(current_state)

    # --- Kiểm tra điều kiện dừng ---
    # 1. Giới hạn thời gian
    if time.time() - start_time > time_limit:
        return "Timeout"

    # 2. Đã đạt đích (tương đương assignment is complete và hợp lệ)
    if current_state == goal_state:
        return path # Trả về lời giải (đường đi)

    # --- Lựa chọn và khám phá ---
    # "Chọn biến chưa gán" tương đương với việc chọn trạng thái tiếp theo từ hàng xóm.
    # "Duyệt qua miền giá trị" tương đương với việc lặp qua các hàng xóm.

    # Thêm trạng thái hiện tại vào tập visited của đường đi này
    visited_in_path.add(current_tuple)

    for neighbor, _ in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)

        # "Kiểm tra ràng buộc": Chỉ đi tiếp nếu hàng xóm chưa có trong đường đi hiện tại
        if neighbor_tuple not in visited_in_path:
            # "Thêm vào assignment" & "Gọi đệ quy"
            # Truyền bản sao của visited_in_path để backtracking hoạt động đúng
            result = _backtracking_recursive_csp_style(
                neighbor, goal_state, path + [neighbor], visited_in_path.copy(),
                nodes_visited_list, start_time, time_limit
            )

            # Nếu lời gọi đệ quy thành công (tìm thấy đường đi hoặc bị timeout)
            if result is not None: # result có thể là path (list) hoặc "Timeout" (str)
                return result # Truyền kết quả lên

    # --- Backtrack ---
    # Nếu vòng lặp kết thúc mà không tìm thấy giải pháp từ các hàng xóm
    # "remove from assignment" xảy ra ngầm khi hàm trả về và stack frame bị hủy
    # Không cần xóa khỏi visited_in_path vì đã truyền bản sao
    return None # Tương đương "failure"

# Hàm chính gọi backtracking
def backtracking_search_csp(start, goal, time_limit=30):
    """
    Hàm chính khởi tạo Backtracking Search theo phong cách CSP.
    Trả về (path, time_taken, nodes_visited) hoặc (None, time_taken, nodes_visited).
    """
    start_time = time.time()
    nodes_visited_list = [0] # Dùng list để truyền tham chiếu
    path_init = [start]
    visited_init = set() # Visited cho đường đi ban đầu

    # Tăng giới hạn đệ quy (Rất quan trọng cho backtracking sâu)
    original_recursion_limit = sys.getrecursionlimit()
    # Ước lượng một giới hạn đủ lớn, ví dụ 5000 hoặc hơn nếu cần
    new_limit = 5000
    limit_changed = False
    if new_limit > original_recursion_limit:
        try:
            sys.setrecursionlimit(new_limit)
            limit_changed = True
            print(f"Temporarily increasing recursion limit to {new_limit}")
        except Exception as e:
            print(f"Could not set recursion limit: {e}. Using default: {original_recursion_limit}")

    solution_path = None
    status = "Unknown"
    try:
        result = _backtracking_recursive_csp_style(
            start, goal, path_init, visited_init, nodes_visited_list, start_time, time_limit
        )

        if result == "Timeout":
            print(f"Backtracking Search (CSP Style): Timeout after {time_limit} seconds.")
            status = "Timeout"
        elif isinstance(result, list): # Nếu kết quả là một đường đi
            print(f"Backtracking Search (CSP Style): Goal found!")
            solution_path = result
            status = "Solution Found"
        else: # Kết quả là None (không tìm thấy)
             print(f"Backtracking Search (CSP Style): Failed - Goal not found or not reachable.")
             status = "Not Found"

    except RecursionError:
        print(f"Backtracking Search (CSP Style): Failed - Maximum recursion depth exceeded (limit was {sys.getrecursionlimit()}).")
        solution_path = None
        status = "Recursion Limit Exceeded"
    except Exception as e:
        print(f"Backtracking Search (CSP Style): An unexpected error occurred: {e}")
        solution_path = None
        status = f"Error: {e}"
    finally:
        # Luôn khôi phục giới hạn đệ quy ban đầu
        if limit_changed:
             sys.setrecursionlimit(original_recursion_limit)
             print(f"Recursion limit restored to {original_recursion_limit}")

    time_taken = time.time() - start_time
    print(f"Backtracking Search (CSP Style) completed with status: {status}")
    return solution_path, time_taken, nodes_visited_list[0]


def calculate_fitness(final_state, goal_state, is_valid_path):
    if not is_valid_path:
        return 0.0

    h = heuristic(final_state, goal_state)
    if h == float('inf'):
        return 0.0
    fitness = 1.0 / (1.0 + h)
    return fitness

def create_initial_population(size, start_state, max_initial_len=15):
    population = []
    attempts = 0
    max_attempts = size * 5

    while len(population) < size and attempts < max_attempts:
        attempts += 1
        seq_len = random.randint(1, max_initial_len)
        sequence = [random.choice(ACTION_LIST) for _ in range(seq_len)]

        first_state_tuple = apply_action(state_to_tuple(start_state), sequence[0])
        if first_state_tuple is not None:
            population.append(sequence)

    while len(population) < size:
         population.append([random.choice(ACTION_LIST)])

    return population

def tournament_selection(population_with_fitness, tournament_size=3):
    if not population_with_fitness: return None
    tournament = random.sample(population_with_fitness, min(tournament_size, len(population_with_fitness)))
    tournament.sort(key=lambda x: x[0], reverse=True)
    return tournament[0][1]

def crossover(parent1, parent2):
    if not parent1 or not parent2:
        return parent1[:], parent2[:]

    min_len = min(len(parent1), len(parent2))
    if min_len < 2:
        return parent1[:], parent2[:]

    point = random.randint(1, min_len - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(sequence, mutation_rate=0.1):
    if not sequence: return sequence

    mutated_sequence = sequence[:]
    if random.random() < mutation_rate:
        mutation_type = random.choice(['change', 'insert', 'delete'])

        if mutation_type == 'change' and mutated_sequence:
            idx = random.randint(0, len(mutated_sequence) - 1)
            mutated_sequence[idx] = random.choice(ACTION_LIST)
        elif mutation_type == 'insert':
            idx = random.randint(0, len(mutated_sequence))
            mutated_sequence.insert(idx, random.choice(ACTION_LIST))
        elif mutation_type == 'delete' and len(mutated_sequence) > 1:
            idx = random.randint(0, len(mutated_sequence) - 1)
            del mutated_sequence[idx]

    return mutated_sequence


def genetic_algorithm(start_state, goal_state, population_size=100, max_generations=50,
                      mutation_rate=0.1, tournament_size=5, elitism_count=2, max_sim_len=60):
    start_time = time.time()
    population = create_initial_population(population_size, start_state)
    best_overall_fitness = -1.0
    best_overall_sim_path = [start_state]

    generations_run = 0
    for generation in range(max_generations):
        generations_run += 1
        population_with_fitness = []
        current_best_fitness_in_gen = -1.0
        goal_found = False

        for individual in population:
            final_state, sim_path, is_valid = simulate_path(individual, start_state, max_sim_len)
            fitness = calculate_fitness(final_state, goal_state, is_valid)
            population_with_fitness.append((fitness, individual, sim_path))

            if is_valid and final_state == goal_state:
                print(f"GA: Goal found in generation {generation}!")
                time_taken = time.time() - start_time
                return sim_path, time_taken, generation + 1

            if fitness > best_overall_fitness:
                 best_overall_fitness = fitness
                 best_overall_sim_path = sim_path

            current_best_fitness_in_gen = max(current_best_fitness_in_gen, fitness)


        print(f"Generation {generation}: Best Fitness = {current_best_fitness_in_gen:.4f}")

        population_with_fitness.sort(key=lambda x: x[0], reverse=True)

        next_generation = []

        for i in range(elitism_count):
            if i < len(population_with_fitness):
                next_generation.append(population_with_fitness[i][1])

        while len(next_generation) < population_size:
            parent1 = tournament_selection(population_with_fitness, tournament_size)
            parent2 = tournament_selection(population_with_fitness, tournament_size)

            if parent1 is None or parent2 is None:
                 next_generation.extend(create_initial_population(1, start_state))
                 continue

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)

        population = next_generation

    time_taken = time.time() - start_time
    print(f"GA: Max generations reached. Best fitness achieved: {best_overall_fitness:.4f}")
    return best_overall_sim_path, time_taken, generations_run

# --- GUI Class ---
class PuzzleGUI:
    def __init__(self, master):
        self.master = master
        master.title("8-PUZZLE SOLVER VISUALIZATION")
        self.current_displayed_state = copy.deepcopy(current_start_state)
        master.geometry("850x700")
        master.configure(bg="#FFFFFF")

        self.title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.grid_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self.detail_font = tkfont.Font(family="Helvetica", size=12)
        self.grid_label_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        self.BG_COLOR = "#FFFFFF"
        self.CONTROL_BG_COLOR = "#F0F0F0"
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
            "#FF6347", "#20B2AA", "#FFA07A", "#D2691E", "#4682B4" # Added more colors
        ]

        self.animation_running = False
        self.stop_animation_flag = False
        self.animation_thread = None
        self.animation_speed = tk.DoubleVar(value=0.6)
        self.current_step_count = tk.IntVar(value=0)

        self.tile_size = 60
        self.grid_padding = 4
        self.canvas_width = COLS * self.tile_size + (COLS + 1) * self.grid_padding
        self.canvas_height = ROWS * self.tile_size + (ROWS + 1) * self.grid_padding

        self.top_control_frame = tk.Frame(master, bg=self.BG_COLOR, pady=10)
        self.top_control_frame.pack(side=tk.TOP, fill=tk.X)
        self.main_display_frame = tk.Frame(master, bg=self.BG_COLOR, padx=20, pady=10)
        self.main_display_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        algo_frame = tk.Frame(self.top_control_frame, bg=self.BG_COLOR)
        algo_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(algo_frame, text="Algorithms:", font=self.button_font, bg=self.BG_COLOR).pack(side=tk.LEFT, padx=(0,5))

        self.buttons = []
        button_labels = ["DFS", "BFS", "UCS", "A*", "Greedy", "IDS", "IDA*",
                         "SimpleHC", "SteepestHC", "RandomHC", "SA", "Beam", "AO_Search", "Sensorless", "GA", "DFS_Belief", "BFS_Belief", "Backtracking"]
        beam_width_default = 5
        beam_search_partial = partial(beam_search, beam_width=beam_width_default)

        algorithms = [
            dfs, bfs, ucs, a_star, greedy_best_first, ids, ida_star,
            simple_hill_climbing, steepest_ascent_hill_climbing, random_hill_climbing,
            simulated_annealing, beam_search_partial, ao_search, sensorless_search, genetic_algorithm, bfs_belief_search, dfs_belief_search, backtracking_search_csp
        ]

        button_container = tk.Frame(algo_frame, bg=self.BG_COLOR)
        button_container.pack(side=tk.LEFT)
        max_buttons_per_row = 8
        current_row_frame = None

        for i, (label, algo) in enumerate(zip(button_labels, algorithms)):
            if i % max_buttons_per_row == 0:
                current_row_frame = tk.Frame(button_container, bg=self.BG_COLOR)
                current_row_frame.pack(fill=tk.X)
            color_index = i % len(self.BUTTON_COLORS)
            color = self.BUTTON_COLORS[color_index]
            cmd = partial(self.run_algorithm_thread, algo, label)
            button = tk.Button(current_row_frame, text=label, font=self.button_font,
                               bg=color, fg=self.BUTTON_FG_COLOR, width=10, height=1,
                               command=cmd, relief=tk.RAISED, borderwidth=2)
            button.pack(side=tk.LEFT, padx=3, pady=2)
            self.buttons.append(button)

        speed_stop_frame = tk.Frame(self.top_control_frame, bg=self.BG_COLOR)
        speed_stop_frame.pack(side=tk.RIGHT, padx=10)
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
        self.top_grids_frame.pack(pady=10, anchor='n')
        self.bottom_frame = tk.Frame(self.main_display_frame, bg=self.BG_COLOR)
        self.bottom_frame.pack(pady=10, anchor='n')

        self.initial_canvas_frame = self.create_grid_canvas_with_label(self.top_grids_frame, "Start", self.TEXT_COLOR_LABEL)
        self.goal_canvas_frame = self.create_grid_canvas_with_label(self.top_grids_frame, "End", self.TEXT_COLOR_LABEL)
        self.current_canvas_frame = self.create_grid_canvas_with_label(self.bottom_frame, "Current", self.TEXT_COLOR_CURRENT_LABEL)

        self.initial_canvas = self.initial_canvas_frame.winfo_children()[1]
        self.goal_canvas = self.goal_canvas_frame.winfo_children()[1]
        self.current_canvas = self.current_canvas_frame.winfo_children()[1]

        self.initial_canvas_frame.pack(side=tk.LEFT, padx=30)
        self.goal_canvas_frame.pack(side=tk.LEFT, padx=30)
        self.current_canvas_frame.pack(side=tk.LEFT, padx=30)

        self.detail_frame = tk.Frame(self.bottom_frame, bg=self.DETAIL_BG, bd=1, relief=tk.SOLID,
                                     width=150, height=self.canvas_height)
        self.detail_frame.pack(side=tk.LEFT, padx=30, pady=(self.grid_label_font.metrics('linespace') + 5, 0), anchor='n')
        self.detail_frame.pack_propagate(False)

        tk.Label(self.detail_frame, text="Detail", font=self.grid_label_font, bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR).pack(pady=5)
        self.steps_display_label = tk.Label(self.detail_frame, text="Steps: 0", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
        self.steps_display_label.pack(pady=10, padx=10, anchor='w')
        self.time_display_label = tk.Label(self.detail_frame, text="Time: -", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
        self.time_display_label.pack(pady=5, padx=10, anchor='w')
        self.nodes_display_label = tk.Label(self.detail_frame, text="Nodes: -", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
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
        for r in range(ROWS):
            for c in range(COLS):
                val = state[r][c]
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

    def update_detail_box(self, steps=None, time_val=None, nodes_val=None, status=None):
        if steps is not None:
            self.steps_display_label.config(text=f"Steps: {steps}")
        if time_val is not None:
             self.time_display_label.config(text=f"Time: {time_val:.3f}s" if isinstance(time_val, float) else f"Time: {time_val}")
        if nodes_val is not None:
            self.nodes_display_label.config(text=f"Nodes: {nodes_val}")

    def set_buttons_state(self, state):
        for button in self.buttons:
            button.config(state=state)
        self.stop_button.config(state=tk.NORMAL if state == tk.DISABLED else tk.DISABLED)

    def run_algorithm_thread(self, algorithm_func, algo_name):
        if self.animation_running:
            messagebox.showwarning("Busy", "An animation is already running.")
            return

        self.set_buttons_state(tk.DISABLED)
        self.animation_running = True
        self.stop_animation_flag = False
        self.current_step_count.set(0)
        self.update_detail_box(steps=0, time_val="-", nodes_val="-")
        self.update_grid(self.current_canvas, current_start_state, self.CURRENT_TILE_COLOR)
        self.master.update()

        def target():
            try:
                result = algorithm_func(current_start_state, current_goal_state)
                self.master.after(0, self._handle_algorithm_result, result, algo_name)
            except Exception as e:
                 print(f"Error in algorithm thread ({algo_name}): {e}")
                 import traceback
                 traceback.print_exc()
                 self.master.after(0, self._handle_algorithm_error, algo_name, e)

        self.animation_thread = threading.Thread(target=target, daemon=True)
        self.animation_thread.start()

    def _handle_algorithm_result(self, result, algo_name):
        generations_or_nodes = 0
        solution_path_or_actions = None
        time_taken = 0

        if result is None:
            print(f"Warning: Algorithm {algo_name} returned None.")
            solution_path_or_actions, time_taken, generations_or_nodes = None, 0, 0
        else:
            solution_path_or_actions, time_taken, generations_or_nodes = result

        if algo_name == "GA":
            self.update_detail_box(time_val=time_taken, nodes_val=f"{generations_or_nodes} gen")
        elif algo_name == "Sensorless":
             self.update_detail_box(time_val=time_taken, nodes_val=f"{generations_or_nodes} beliefs")
        else:
            self.update_detail_box(time_val=time_taken, nodes_val=generations_or_nodes)


        if algo_name == "Sensorless":
            action_sequence = solution_path_or_actions
            if action_sequence is not None:
                print(f"{algo_name} found action sequence: {' -> '.join(action_sequence)}")
                self.update_detail_box(steps=len(action_sequence))
                self.update_grid(self.current_canvas, current_goal_state, self.CURRENT_TILE_COLOR)
                messagebox.showinfo("Result", f"{algo_name} found a plan!\nActions ({len(action_sequence)}): {', '.join(action_sequence)}")
            else:
                print(f"{algo_name} did not find a plan.")
                self.update_detail_box(steps="-")
                messagebox.showinfo("Result", f"{algo_name} did not find a plan.")

            self.animation_running = False
            self.set_buttons_state(tk.NORMAL)


        elif solution_path_or_actions and isinstance(solution_path_or_actions, list) and len(solution_path_or_actions) > 0:
            solution_path = solution_path_or_actions
            is_goal_reached = False
            if solution_path[-1] == current_goal_state:
                 is_goal_reached = True

            path_steps = len(solution_path) - 1

            print(f"{algo_name} finished. Path length: {path_steps+1}. Goal reached: {is_goal_reached}.")
            self.update_detail_box(steps=path_steps)

            if path_steps >= 0:
                 if is_goal_reached:
                      msg = f"{algo_name} found solution."
                 elif algo_name == "GA":
                      msg = f"{algo_name} finished.\nMax generations reached.\nDisplaying best path found."
                 else:
                      msg = f"{algo_name} finished.\nGoal reached: {is_goal_reached}\nDisplaying path found."
                 messagebox.showinfo("Result", msg)
                 self._animate_solution(solution_path)
            else:
                 messagebox.showinfo("Result", f"{algo_name} did not produce a valid path.")
                 self.animation_running = False
                 self.set_buttons_state(tk.NORMAL)

        else:
            print(f"{algo_name} did not find a solution path.")
            self.update_detail_box(steps="-")
            messagebox.showinfo("Result", f"{algo_name} did not find a solution path.")
            self.animation_running = False
            self.set_buttons_state(tk.NORMAL)

    def _handle_algorithm_error(self, algo_name, error):
        messagebox.showerror("Algorithm Error", f"An error occurred during {algo_name}:\n{error}")
        self.animation_running = False
        self.set_buttons_state(tk.NORMAL)

    def _animate_solution(self, solution_path):
        if not solution_path or self.stop_animation_flag:
            is_goal_reached = (self.current_displayed_state == current_goal_state)
            if self.stop_animation_flag:
                print("Animation stopped by user.")
            elif not is_goal_reached:
                print("Animation finished (did not reach goal).")
            else:
                print("Animation finished (goal reached).")
            self.animation_running = False
            self.set_buttons_state(tk.NORMAL)
            return

        step_num_to_display = self.current_step_count.get()
        current_step_state = solution_path.pop(0)
        self.current_displayed_state = current_step_state

        self.update_grid(self.current_canvas, current_step_state, self.CURRENT_TILE_COLOR)
        self.update_detail_box(steps=step_num_to_display)
        self.current_step_count.set(step_num_to_display + 1)

        delay_ms = int(self.animation_speed.get() * 1000)
        self.master.after(delay_ms, self._animate_solution, solution_path)

    def stop_animation(self):
        if self.animation_running:
            self.stop_animation_flag = True

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = PuzzleGUI(root)
    root.mainloop()
  