import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import heapq
import time
from collections import deque
import random
import threading # To run algorithms without freezing the GUI
import copy # For deep copying states
from functools import partial # To pass arguments to button commands
import math
# --- Core Algorithm Logic ---

ROWS, COLS = 3, 3

# Define states (can be modified if needed)
start_state_default = [[2, 6, 5], [8, 7, 0], [4, 3, 1]]
goal_state_default = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Global variables for current states
current_start_state = copy.deepcopy(start_state_default)
current_goal_state = copy.deepcopy(goal_state_default)
current_displayed_state = copy.deepcopy(start_state_default) # State shown in the 'Current State' grid

# --- Helper Functions ---

def find_zero(state):
    """Finds the row and column of the empty tile (0)."""
    for i, row in enumerate(state):
        try:
            j = row.index(0)
            return i, j
        except ValueError:
            continue
    return None # Should not happen in a valid puzzle state

def get_neighbors(state):
    """Generates neighbor states by moving the empty tile."""
    zero_pos = find_zero(state)
    if zero_pos is None: return [] # Should not happen
    r, c = zero_pos
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            # Create a deep copy to avoid modifying the original state indirectly
            new_state = [row[:] for row in state]
            # Swap the empty tile with the adjacent tile
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            # Store the new state and the cost (usually 1 for puzzle moves)
            neighbors.append((new_state, 1))
    return neighbors

def state_to_tuple(state):
    """Converts a list-of-lists state into a tuple of tuples for hashing."""
    # Check if already a tuple (might happen in some intermediate steps)
    if isinstance(state, tuple):
        return state
    return tuple(tuple(row) for row in state)

def heuristic(state, goal):
    """Calculates the Manhattan distance heuristic."""
    distance = 0
    # Create goal position lookup only once if goal state doesn't change often,
    # but for flexibility, we recreate it here.
    goal_positions = {goal[r][c]: (r, c) for r in range(ROWS) for c in range(COLS) if goal[r][c] != 0}

    for r in range(ROWS):
        for c in range(COLS):
            val = state[r][c]
            if val != 0:
                if val in goal_positions:
                    goal_r, goal_c = goal_positions[val]
                    distance += abs(r - goal_r) + abs(c - goal_c)
                else:
                    print(f"Warning: Tile {val} from current state not found in goal state!")
                    return float('inf') # Or raise an error
    return distance

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

    return None, time.time() - start_time, nodes_visited_count # No solution found

def ucs(start, goal):
    # Priority queue: (cost, state, path_so_far)
    pq = [(0, start, [])]
    # Keep track of visited states and their *minimum* cost found so far
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

            # If neighbor is unvisited or we found a cheaper path to it
            if new_cost < visited_costs.get(neighbor_tuple, float('inf')):
                visited_costs[neighbor_tuple] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [state]))

    return None, time.time() - start_time, nodes_visited_count # No solution found

def a_star(start, goal):
    # Priority queue: (f_cost, g_cost, state, path_so_far)
    start_h = heuristic(start, goal)
    pq = [(start_h, 0, start, [])]
    visited_costs = {state_to_tuple(start): 0} # Store g_cost
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
        _, state, path = heapq.heappop(pq) # Heuristic value only used for ordering
        nodes_visited_count += 1

        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited_count
        for neighbor, _ in get_neighbors(state):
            neighbor_tuple = state_to_tuple(neighbor)
            if neighbor_tuple not in visited:
                 # Mark visited when adding to queue to prevent cycles and redundant exploration
                 # in basic Greedy Search.
                visited.add(neighbor_tuple)
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor, path + [state]))

    return None, time.time() - start_time, nodes_visited_count # No solution found

def dls(current_state, goal, depth_limit, path, visited_in_path):
    """Depth Limited Search helper for IDS. visited_in_path tracks nodes in the current path to prevent cycles."""
    current_tuple = state_to_tuple(current_state)
    nodes_visited_count = 1 # Count the current node
    if current_state == goal:
          return path + [current_state], nodes_visited_count, True 
    if depth_limit == 0:
        return None, nodes_visited_count, False 
    visited_in_path.add(current_tuple) 
    any_remaining = False 
    for neighbor, _ in get_neighbors(current_state):
        neighbor_tuple = state_to_tuple(neighbor)
        # Avoid cycles within the current DLS path
        if neighbor_tuple not in visited_in_path:
            result, nodes_count, found_goal = dls(neighbor, goal, depth_limit - 1, path + [current_state], visited_in_path.copy()) # Pass copy for backtracking
            nodes_visited_count += nodes_count
            if found_goal: 
                return result, nodes_visited_count, True
    return None, nodes_visited_count, False 

def ids(start, goal):
    """Iterative Deepening Search."""
    depth = 0
    total_nodes_visited = 0
    start_time = time.time()
    while True:
        result, nodes_count, found_goal = dls(start, goal, depth, [], set())
        total_nodes_visited += nodes_count # Accumulate nodes from each DLS iteration

        if found_goal: # Goal found
            return result, time.time() - start_time, total_nodes_visited
        if depth > 30: 
             print(f"IDS depth limit {depth} reached.")
             break
        depth += 1

    return None, time.time() - start_time, total_nodes_visited

def ida_star_search(node, g_cost, threshold, path, goal, visited_in_path):
    """Recursive search function for IDA*."""
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
         if next_state_tuple not in visited_in_path: # Avoid cycles in the current path
            result, found_path, nodes_count = ida_star_search(
                next_state, g_cost + step_cost, threshold, path + [node], goal, visited_in_path.copy() # Pass copy
            )
            nodes_visited_count += nodes_count
            if result is True:
                return True, found_path, nodes_visited_count
            if result < min_next_threshold:
                min_next_threshold = result
    return min_next_threshold, None, nodes_visited_count 

def ida_star(start, goal):
    """Iterative Deepening A* Search."""
    threshold = heuristic(start, goal)
    total_nodes_visited = 0
    start_time = time.time()

    while True:
        result, solution_path, nodes_count = ida_star_search(start, 0, threshold, [], goal, set())
        total_nodes_visited += nodes_count # Accumulate nodes

        if result is True: # Solution found
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
        nodes_visited_count += 1 # Count current state evaluation
        neighbors = get_neighbors(current)
        if not neighbors: break # Isolated state

        next_state = None
        # Find the first neighbor that is strictly better
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count +=1 # Count neighbor evaluation
            if neighbor_h < current_h:
                next_state = neighbor_state # Found a better one
                current_h = neighbor_h     # Update the heuristic benchmark
                break # Take the first improvement

        # If no better neighbor was found, we are at a local optimum (or goal)
        if next_state is None:
            if current != goal: print("Simple Hill Climbing got stuck at a local optimum.")
            break

        # Move to the better neighbor
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
        nodes_visited_count += 1 # Count current state evaluation
        neighbors = get_neighbors(current)
        if not neighbors: break

        best_neighbor = None
        best_h = current_h # Initialize with current heuristic value

        # Find the neighbor with the absolute minimum heuristic value among all neighbors
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count += 1 # Count neighbor evaluation
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbor = neighbor_state

        # If the absolutely best neighbor is not strictly better than current, stuck.
        if best_neighbor is None or best_h >= current_h:
             if current != goal: print("Steepest Ascent got stuck at a local optimum.")
             break

        # Move to the best neighbor found
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
        nodes_visited_count += 1 # Count current evaluation
        iterations += 1
        neighbors = get_neighbors(current)
        if not neighbors: break

        # Evaluate all neighbors first
        evaluated_neighbors = []
        for neighbor_state, _ in neighbors:
            neighbor_h = heuristic(neighbor_state, goal)
            nodes_visited_count += 1 # Count neighbor evaluation
            evaluated_neighbors.append((neighbor_state, neighbor_h))

        # Filter neighbors that are strictly better
        better_neighbors = [(n_state, n_h) for n_state, n_h in evaluated_neighbors if n_h < current_h]

        if better_neighbors:
            # Choose randomly among the better options
            chosen_neighbor, chosen_h = random.choice(better_neighbors)
            current = chosen_neighbor
            current_h = chosen_h
            path.append(current)
        else:
            # No better neighbors, stuck (or already at goal)
            if current != goal: print("Random Hill Climbing got stuck (no better neighbors).")
            break # Stop if no improvement possible

    if iterations >= max_iterations and current != goal:
        print(f"Random Hill Climbing reached max iterations ({max_iterations}) without finding goal.")

    # Return path even if stuck or max iterations reached
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
            if not neighbors:
                break
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
        if not neighbors:
            break
    time_taken = time.time() - start_time
    print(f"SA: Finished. Best energy reached: {best_energy}. Final Temp: {temp:.3f}")
    return path_taken, time_taken, nodes_visited

def beam_search(start_state, goal_state, beam_width=5): # Đặt beam_width mặc định
    start_time = time.time()
    nodes_expanded = 0 # Đếm số nút được lấy ra từ beam để mở rộng

    start_h = heuristic(start_state, goal_state)
    # Beam lưu các tuple: (heuristic_value, state, path_list)
    beam = [(start_h, start_state, [start_state])]
    # Visited lưu các tuple trạng thái để tránh lặp
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
    return None, time_taken, nodes_expanded # Trả về None cho path
class PuzzleGUI:
    def __init__(self, master):
        self.master = master
        master.title("8-PUZZLE SOLVER")
        self.current_displayed_state = copy.deepcopy(current_start_state)
        # Adjust size to fit grids, controls, and detail
        master.geometry("850x700") # May need tweaking
        master.configure(bg="#FFFFFF") # White background

        # --- Fonts ---
        self.title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.grid_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self.detail_font = tkfont.Font(family="Helvetica", size=12)
        self.grid_label_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

        # --- Colors (Matching Image) ---
        self.BG_COLOR = "#FFFFFF" # White background overall
        self.CONTROL_BG_COLOR = "#F0F0F0" # Light grey for control panel area
        self.START_END_TILE_COLOR = "#60D060" # Greenish
        self.CURRENT_TILE_COLOR = "#9370DB" # Purpleish (MediumPurple)
        self.EMPTY_TILE_COLOR = "#F5F5F5" # Very light grey for empty
        self.GRID_BORDER_COLOR = "#A0A0A0" # Grey border for tiles
        self.TEXT_COLOR_ON_TILE = "#FFFFFF" # White text on colored tiles
        self.TEXT_COLOR_LABEL = "#FF0000" # Red for Start/End labels
        self.TEXT_COLOR_CURRENT_LABEL = "#8A2BE2" # BlueViolet for Current label
        self.DETAIL_BG = "#FFFFFF"
        self.DETAIL_BORDER = "#000000"
        self.DETAIL_TEXT_COLOR = "#000000"
        self.BUTTON_FG_COLOR = "#FFFFFF"
        self.BUTTON_COLORS = [ # Using colors from previous version for buttons
            "#FF5757", "#42D6A4", "#FF9F43", "#9370DB", "#FFD700",
            "#32CD32", "#FF4500", "#00BFFF", "#FF1493", "#778899",
        ]

        self.animation_running = False
        self.stop_animation_flag = False
        self.animation_thread = None
        self.animation_speed = tk.DoubleVar(value=0.6) # Default animation speed
        self.current_step_count = tk.IntVar(value=0) # Variable for step display

        self.tile_size = 60 # Pixel size for each tile
        self.grid_padding = 4 # Padding around tiles
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
                         "SimpleHC", "SteepestHC", "RandomHC", "SA", "Beam"] 
        algorithms = [dfs, bfs, ucs, a_star, greedy_best_first, ids, ida_star,
                      simple_hill_climbing, steepest_ascent_hill_climbing, random_hill_climbing, simulated_annealing, beam_search]

        button_container = tk.Frame(algo_frame, bg=self.BG_COLOR) # Container for wrapping buttons
        button_container.pack(side=tk.LEFT)
        max_buttons_per_row = 5
        current_row_frame = None

        for i, (label, algo) in enumerate(zip(button_labels, algorithms)):
            if i % max_buttons_per_row == 0:
                current_row_frame = tk.Frame(button_container, bg=self.BG_COLOR)
                current_row_frame.pack(fill=tk.X)

            color = self.BUTTON_COLORS[i % len(self.BUTTON_COLORS)]
            cmd = partial(self.run_algorithm_thread, algo, label)
            button = tk.Button(current_row_frame, text=label, font=self.button_font,
                               bg=color, fg=self.BUTTON_FG_COLOR, width=10, height=1, # Smaller buttons
                               command=cmd, relief=tk.RAISED, borderwidth=2)
            button.pack(side=tk.LEFT, padx=3, pady=2)
            self.buttons.append(button)

        speed_stop_frame = tk.Frame(self.top_control_frame, bg=self.BG_COLOR)
        speed_stop_frame.pack(side=tk.RIGHT, padx=10)

        tk.Label(speed_stop_frame, text="Speed:", font=self.button_font, bg=self.BG_COLOR).pack(side=tk.LEFT, padx=(0,2))
        self.speed_slider = tk.Scale(speed_stop_frame, from_=0.1, to=1.5, resolution=0.1,
                                     orient=tk.HORIZONTAL, variable=self.animation_speed,
                                     bg=self.BG_COLOR, length=100, showvalue=0) # No value shown on slider itself
        self.speed_slider.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(speed_stop_frame, text="Stop", font=self.button_font,
                                     bg="#DC143C", fg=self.BUTTON_FG_COLOR, width=5, height=1,
                                     command=self.stop_animation, relief=tk.RAISED, borderwidth=2,
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.top_grids_frame = tk.Frame(self.main_display_frame, bg=self.BG_COLOR)
        self.top_grids_frame.pack(pady=10, anchor='n') # Pack at the top
        self.bottom_frame = tk.Frame(self.main_display_frame, bg=self.BG_COLOR)
        self.bottom_frame.pack(pady=10, anchor='n') # Pack below top grids

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
                                     width=150, height=self.canvas_height) # Match height roughly
        self.detail_frame.pack(side=tk.LEFT, padx=30, pady=(self.grid_label_font.metrics('linespace') + 5, 0), anchor='n') # Align top near Current label
        self.detail_frame.pack_propagate(False) 

        tk.Label(self.detail_frame, text="Detail", font=self.grid_label_font, bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR).pack(pady=5)
        self.step_label = tk.Label(self.detail_frame, textvariable=self.current_step_count, font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)

        self.steps_display_label = tk.Label(self.detail_frame, text="Steps: 0", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
        self.steps_display_label.pack(pady=10, padx=10, anchor='w')

        self.time_display_label = tk.Label(self.detail_frame, text="Time: -", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
        self.time_display_label.pack(pady=5, padx=10, anchor='w')

        self.nodes_display_label = tk.Label(self.detail_frame, text="Nodes: -", font=self.detail_font,
                                   bg=self.DETAIL_BG, fg=self.DETAIL_TEXT_COLOR)
        self.nodes_display_label.pack(pady=5, padx=10, anchor='w')


        # --- Initial Display ---
        self.update_grid(self.initial_canvas, current_start_state, self.START_END_TILE_COLOR)
        self.update_grid(self.goal_canvas, current_goal_state, self.START_END_TILE_COLOR)
        self.update_grid(self.current_canvas, current_displayed_state, self.CURRENT_TILE_COLOR)


    def create_grid_canvas_with_label(self, parent_frame, label_text, label_color):
        """Creates a frame containing a label and a canvas for a grid."""
        frame = tk.Frame(parent_frame, bg=self.BG_COLOR)
        label = tk.Label(frame, text=label_text, font=self.grid_label_font, bg=self.BG_COLOR, fg=label_color)
        label.pack(pady=(0, 5))
        canvas = tk.Canvas(frame, width=self.canvas_width, height=self.canvas_height,
                           bg=self.BG_COLOR, highlightthickness=0) # No border around canvas itself
        canvas.pack()
        return frame # Return the container frame

    def update_grid(self, canvas, state, tile_base_color):
        """Draws the puzzle state onto the given canvas."""
        canvas.delete("all") # Clear previous drawing
        if state is None: return

        for r in range(ROWS):
            for c in range(COLS):
                val = state[r][c]
                x0 = c * self.tile_size + (c + 1) * self.grid_padding
                y0 = r * self.tile_size + (r + 1) * self.grid_padding
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size

                # Draw tile background
                if val == 0:
                    # Draw empty tile (just background or a subtle rectangle)
                     canvas.create_rectangle(x0, y0, x1, y1, fill=self.EMPTY_TILE_COLOR, outline=self.GRID_BORDER_COLOR, width=1)
                else:
                    # Draw numbered tile
                    canvas.create_rectangle(x0, y0, x1, y1, fill=tile_base_color, outline=self.GRID_BORDER_COLOR, width=1)
                    canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(val),
                                       font=self.grid_font, fill=self.TEXT_COLOR_ON_TILE)

    def update_detail_box(self, steps=None, time_val=None, nodes_val=None, status=None):
        """Updates the information in the Detail box."""
        if steps is not None:
            self.steps_display_label.config(text=f"Steps: {steps}")
        if time_val is not None:
             self.time_display_label.config(text=f"Time: {time_val:.3f}s" if isinstance(time_val, float) else f"Time: {time_val}")
        if nodes_val is not None:
            self.nodes_display_label.config(text=f"Nodes: {nodes_val}")
        # Could add a status label here too if needed

    def set_buttons_state(self, state):
        """Enable or disable all algorithm buttons and stop button."""
        for button in self.buttons:
            button.config(state=state)
        # Enable stop button only when animation is running (buttons are disabled)
        self.stop_button.config(state=tk.NORMAL if state == tk.DISABLED else tk.DISABLED)


    def run_algorithm_thread(self, algorithm_func, algo_name):
        """Runs the selected algorithm in a separate thread."""
        if self.animation_running:
            messagebox.showwarning("Busy", "An animation is already running.")
            return

        self.set_buttons_state(tk.DISABLED)
        self.animation_running = True # Dùng flag này cho cả quá trình tính toán và animation
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
                 print(f"Error in algorithm thread: {e}")
                 self.master.after(0, self._handle_algorithm_error, algo_name, e)

        self.animation_thread = threading.Thread(target=target, daemon=True)
        self.animation_thread.start()

    def _handle_algorithm_result(self, result, algo_name):
        """Handles the result from the algorithm thread."""


        if result is None:
            solution_path, time_taken, nodes_visited = None, 0, 0
            print(f"Warning: Algorithm {algo_name} returned None.")
        else:
             solution_path, time_taken, nodes_visited = result

        self.update_detail_box(time_val=time_taken, nodes_val=nodes_visited)

        if solution_path and len(solution_path) > 0:
            # Check if goal was actually reached (for algorithms like Hill Climbing)
            is_goal_reached = (solution_path[-1] == current_goal_state)
            if is_goal_reached:
                 print(f"{algo_name} found solution.")
                 self._animate_solution(solution_path) # Start animation
            else:
                 print(f"{algo_name} finished but did not reach the goal state.")
                 self._animate_solution(solution_path) # Animate the partial path
        else:
            # No solution path found or empty path returned
            print(f"{algo_name} did not find a solution.")
            messagebox.showinfo("Result", f"{algo_name} did not find a solution path.")
            self.animation_running = False # Ensure flag is reset
            self.set_buttons_state(tk.NORMAL)


    def _handle_algorithm_error(self, algo_name, error):
        """Handles errors occurring in the algorithm thread."""
        messagebox.showerror("Algorithm Error", f"An error occurred during {algo_name}:\n{error}")
        self.animation_running = False # Reset flag
        self.set_buttons_state(tk.NORMAL)


    def _animate_solution(self, solution_path):
        global current_displayed_state # <<< CHUYỂN LÊN ĐÂY
        """Animates the solution path step-by-step."""
        # Base case: Stop if path is empty or stop flag is set
        if not solution_path or self.stop_animation_flag:
            # Bây giờ có thể đọc biến global này một cách an toàn
            is_goal_reached = (current_displayed_state == current_goal_state)
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

        current_step_state = solution_path.pop(0) # Get the next state

        # Gán giá trị cho biến global (đã được khai báo ở trên)
        current_displayed_state = current_step_state

        # Update GUI
        self.update_grid(self.current_canvas, current_step_state, self.CURRENT_TILE_COLOR)
        self.update_detail_box(steps=step_num_to_display)

        # Increment step count *after* displaying the current step's state
        self.current_step_count.set(step_num_to_display + 1)

        # Schedule the next step
        delay_ms = int(self.animation_speed.get() * 1000)
        # Pass the *remaining* path to the next call
        self.master.after(delay_ms, self._animate_solution, solution_path)

    def stop_animation(self):
        """Sets the flag to stop the current animation."""
        if self.animation_running: 
            self.stop_animation_flag = True
if __name__ == "__main__":
    root = tk.Tk()
    gui = PuzzleGUI(root)
    root.mainloop()