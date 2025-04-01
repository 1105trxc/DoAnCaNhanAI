import pygame
import heapq
import time
from collections import deque

WIDTH, HEIGHT = 1600, 900
ROWS, COLS = 3, 3
TILE_SIZE = (WIDTH - 200) // (COLS + 9)  # Space for buttons on the left
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 80

# Define states
start_state = [[2, 6, 5], [8, 7, 0], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 60, bold=True)
button_font = pygame.font.SysFont(None, 30, bold=True)
info_font = pygame.font.SysFont(None, 30, bold=True)

# Colors
BG_COLOR = (245, 245, 250)  # Light gray
TILE_COLOR = (65, 105, 225)  # Royal blue
GOAL_COLOR = (50, 205, 50)   # Green (from BUTTON_COLORS[5])
BORDER_COLOR = (25, 25, 112)  # Dark blue
TEXT_COLOR = (255, 255, 255)  # White
BUTTON_COLORS = [
    (255, 87, 87),    # Light red
    (66, 214, 164),   # Turquoise
    (255, 159, 67),   # Orange
    (147, 112, 219),  # Medium purple
    (255, 215, 0),    # Gold
    (50, 205, 50),     # Green
    (255, 69, 0),      # Red
    (0, 191, 255),     # Deep sky blue
    (255, 20, 147),    # Deep pink
    (0, 0, 0) ,
    (113,218,255)        # Black
]

def draw_grid(state, x_offset, y_offset, label, tile_color=TILE_COLOR):
    # Draw background for the grid
    pygame.draw.rect(win, (230, 230, 250), (x_offset, y_offset, TILE_SIZE * 3, TILE_SIZE * 3 + 50))
    text_label = font.render(str(label), True, BORDER_COLOR)
    win.blit(text_label, (x_offset + TILE_SIZE, y_offset + 10))
    
    for r in range(ROWS):
        for c in range(COLS):
            if state[r][c] != 0:
                pygame.draw.rect(win, tile_color,
                               (x_offset + c * TILE_SIZE, y_offset + r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE),
                               border_radius=15)
                text = font.render(str(state[r][c]), True, TEXT_COLOR)
                text_rect = text.get_rect(center=(x_offset + c * TILE_SIZE + TILE_SIZE // 2,
                                                y_offset + r * TILE_SIZE + 50 + TILE_SIZE // 2))
                win.blit(text, text_rect)
            pygame.draw.rect(win, BORDER_COLOR,
                           (x_offset + c * TILE_SIZE, y_offset + r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE),
                           3, border_radius=15)

def draw_buttons():
    button_labels = ["DFS", "BFS", "UCS", "A*", "IDS", "Greedy", "IDA*", "Simple Hill Climbing","Steepest Ascent Hill Climbing"]
    for i, label in enumerate(button_labels):
        pygame.draw.rect(win, BUTTON_COLORS[i],
                        (10, 10 + i * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=12)
        text_surface = button_font.render(label, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(10 + BUTTON_WIDTH // 2,
                                                10 + i * (BUTTON_HEIGHT + 10) + BUTTON_HEIGHT // 2))
        win.blit(text_surface, text_rect)

# Search algorithms (unchanged)
def find_zero(state):
    for i, row in enumerate(state):
        if 0 in row:
            return i, row.index(0)

def get_neighbors(state):
    r, c = find_zero(state)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            new_state = [row[:] for row in state]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            neighbors.append((new_state, 1))
    return neighbors

def dfs(start, goal):
    stack = [(start, [])]
    visited = set()
    nodes_visited = 0
    start_time = time.time()
    while stack:
        state, path = stack.pop()
        nodes_visited += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        for neighbor, _ in get_neighbors(state):
            if str(neighbor) not in visited:
                stack.append((neighbor, path + [state]))
    return [], time.time() - start_time, nodes_visited

def bfs(start, goal):
    queue = deque([(start, [])])
    visited = set()
    nodes_visited = 0
    start_time = time.time()
    while queue:
        state, path = queue.popleft()
        nodes_visited += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        for neighbor, _ in get_neighbors(state):
            if str(neighbor) not in visited:
                queue.append((neighbor, path + [state]))
    return [], time.time() - start_time, nodes_visited

def ucs(start, goal):
    pq = [(0, start, [])]
    visited = set()
    nodes_visited = 0
    start_time = time.time()
    while pq:
        cost, state, path = heapq.heappop(pq)
        nodes_visited += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        for neighbor, step_cost in get_neighbors(state):
            if str(neighbor) not in visited:
                heapq.heappush(pq, (cost + step_cost, neighbor, path + [state]))
    return [], time.time() - start_time, nodes_visited

def heuristic(state):
    return sum(abs(r - goal_r) + abs(c - goal_c) 
            for r, row in enumerate(state) 
            for c, val in enumerate(row) 
            if val and (goal_r := (val - 1) // COLS) is not None and (goal_c := (val - 1) % COLS) is not None)

#A* = UCS + Greedy
def a_star(start, goal):
    pq = [(heuristic(start), 0, start, [])]
    visited = set()
    nodes_visited = 0
    start_time = time.time()
    while pq:
        _, cost, state, path = heapq.heappop(pq)
        nodes_visited += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        for neighbor, step_cost in get_neighbors(state):
            if str(neighbor) not in visited:
                heapq.heappush(pq, (cost + step_cost + heuristic(neighbor), cost + step_cost, neighbor, path + [state]))
    return [], time.time() - start_time, nodes_visited

def greedy_best_first(start, goal):
    pq = [(heuristic(start), start, [])]
    visited = set()
    nodes_visited = 0
    start_time = time.time()
    while pq:
        _, state, path = heapq.heappop(pq)
        nodes_visited += 1
        if state == goal:
            return path + [state], time.time() - start_time, nodes_visited
        state_str = str(state)
        if state_str in visited:
            continue
        visited.add(state_str)
        for neighbor, _ in get_neighbors(state):
            if str(neighbor) not in visited:
                heapq.heappush(pq, (heuristic(neighbor), neighbor, path + [state]))
    return [], time.time() - start_time, nodes_visited

def dls(state, goal, depth, path, visited):
    if depth == 0:
        return None
    if state == goal:
        return path + [state]
    state_str = str(state)
    if state_str in visited:
        return None
    visited.add(state_str)
    for neighbor, _ in get_neighbors(state):
        result = dls(neighbor, goal, depth - 1, path + [state], visited)
        if result:
            return result
    return None

def ids(start, goal):
    depth = 0
    start_time = time.time()
    nodes_visited = 0
    while True:
        visited = set()
        result = dls(start, goal, depth, [], visited)
        nodes_visited += len(visited)
        if result:
            return result, time.time() - start_time, nodes_visited
        depth += 1

def ida_star(start, goal):
    """Thuật toán IDA* tìm đường đi từ start đến goal"""
    
    nodes_visited = 0  # Biến đếm số nút đã duyệt

    def search(node, g, threshold, path):
        """Hàm tìm kiếm đệ quy"""
        nonlocal nodes_visited
        nodes_visited += 1  # Tăng số nút đã duyệt
        
        f = g + heuristic(node)
        if f > threshold:
            return f, None
        if node == goal:
            return True, path + [node]  # Trả về đường đi
        
        min_threshold = float('inf')
        best_path = None

        for next_state, cost in get_neighbors(node):  
            temp, new_path = search(next_state, g + cost, threshold, path + [node])
            if temp is True:
                return True, new_path  # Trả về đường đi nếu tìm thấy
            if temp < min_threshold:
                min_threshold = temp
                best_path = new_path
        
        return min_threshold, best_path

    start_time = time.time()  # Bắt đầu đo thời gian
    threshold = heuristic(start)

    while True:
        result, solution_path = search(start, 0, threshold, [])
        if result is True:
            time_taken = time.time() - start_time  # Tính thời gian chạy
            return solution_path, time_taken, nodes_visited  # Trả về danh sách trạng thái hợp lệ
        if result == float('inf'):
            return None, time.time() - start_time, nodes_visited  # Trả về None nếu không tìm thấy đường đi
        threshold = result

def simple_hill_climbing(start, goal):
    current = start
    path = [current]  
    nodes_visited = 0  
    start_time = time.time()  

    while current != goal:
        neighbors = get_neighbors(current)  
        nodes_visited += 1  

        # Tìm trạng thái có heuristic nhỏ nhất
        next_state = min(neighbors, key=lambda x: heuristic(x[0]), default=None)

        # Nếu không có trạng thái tốt hơn, dừng lại (mắc kẹt)
        if not next_state or heuristic(next_state[0]) >= heuristic(current):
            break

        # Di chuyển đến trạng thái tốt hơn
        current = next_state[0]
        path.append(current)

    return path, time.time() - start_time, nodes_visited

def steepest_ascent_hill_climbing(start, goal):
    current = start
    path = [current]
    nodes_visited = 0
    start_time = time.time()

    while current != goal:
        neighbors = get_neighbors(current)
        nodes_visited += 1

        # Tìm trạng thái có heuristic nhỏ nhất
        best_neighbor = min(neighbors, key=lambda x:heuristic(x[0]), default=None)

        # Nếu không có trạng thái tốt hơn, dừng lại (mắc kẹt)
        if best_neighbor is None or heuristic(best_neighbor[0]) >= heuristic(current):
            break

        # Di chuyển đến trạng thái tốt nhất
        current = best_neighbor[0]
        path.append(current)

    return path, time.time() - start_time, nodes_visited

def run_algorithm(algorithm, speed):
    win.fill(BG_COLOR)
    result = algorithm(start_state, goal_state)

    if result is None or result[0] is None:  # Nếu không tìm thấy đường đi
        info_text = f"{algorithm.__name__.upper()}|No Solution Found"
        info_surface = info_font.render(info_text, True, BORDER_COLOR)
        win.blit(info_surface, (BUTTON_WIDTH + 30 + TILE_SIZE * 8, TILE_SIZE * 4 + 50))
        pygame.display.flip()
        return  # Thoát hàm ngay

    solution, time_taken, nodes_visited = result
    clock = pygame.time.Clock()

    for state in solution:
        win.fill(BG_COLOR)
        # First row: Initial State and Goal State
        draw_grid(start_state, BUTTON_WIDTH + 30, 0, "Initial State", TILE_COLOR)
        draw_grid(goal_state, BUTTON_WIDTH + 30 + TILE_SIZE * 4, 0, "Goal State", GOAL_COLOR)
        # Second row: Current State
        draw_grid(state, BUTTON_WIDTH + 30, TILE_SIZE * 4, "Current State", GOAL_COLOR)
        draw_buttons()
        pygame.display.flip()
        clock.tick(speed)
    
    # Hiển thị thông tin thuật toán
    info_text = f"{algorithm.__name__.upper()}|Time: {time_taken:.3f}s|Steps: {len(solution)-1}"
    info_surface = info_font.render(info_text, True, BORDER_COLOR)
    win.blit(info_surface, (BUTTON_WIDTH + 30 + TILE_SIZE * 8, TILE_SIZE * 4 + 50))
    pygame.display.flip()
def run_algorithm2(algorithm, speed):
    win.fill(BG_COLOR)
    result = algorithm(start_state, goal_state)

    if not result or not isinstance(result, tuple) or len(result) < 3 or not result[0]:
        info_text = f"{algorithm.__name__.upper()}|No Solution Found"
        info_surface = info_font.render(info_text, True, BORDER_COLOR)
        win.blit(info_surface, (BUTTON_WIDTH + 30 + TILE_SIZE * 8, TILE_SIZE * 4 + 50))
        pygame.display.flip()
        return  # Thoát hàm ngay

    solution, time_taken, nodes_visited = result
    clock = pygame.time.Clock()

    for state in solution:
        win.fill(BG_COLOR)
        # First row: Initial State and Goal State
        draw_grid(start_state, BUTTON_WIDTH + 30, 0, "Initial State", TILE_COLOR)
        draw_grid(goal_state, BUTTON_WIDTH + 30 + TILE_SIZE * 4, 0, "Goal State", GOAL_COLOR)
        # Second row: Current State
        draw_grid(state, BUTTON_WIDTH + 30, TILE_SIZE * 4, "Current State", GOAL_COLOR)
        draw_buttons()
        pygame.display.flip()
        clock.tick(speed)

    # Hiển thị thông tin thuật toán
    info_text = f"{algorithm.__name__.upper()}|Time: {time_taken:.3f}s|Steps: {len(solution)-1}"
    info_surface = info_font.render(info_text, True, BORDER_COLOR)
    win.blit(info_surface, (BUTTON_WIDTH + 30 + TILE_SIZE * 8, TILE_SIZE * 4 + 50))
    pygame.display.flip()


def main():
    run = True
    speed = 1  # Speed for animation
    
    win.fill(BG_COLOR)
    # First row: Initial State and Goal State
    draw_grid(start_state, BUTTON_WIDTH + 30, 0, "Initial State", TILE_COLOR)
    draw_grid(goal_state, BUTTON_WIDTH + 30 + TILE_SIZE * 4, 0, "Goal State", GOAL_COLOR)
    # Second row: Current State (starts as Initial State)
    draw_grid(start_state, BUTTON_WIDTH + 30, TILE_SIZE * 4, "Current State", GOAL_COLOR)
    draw_buttons()
    pygame.display.flip()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < BUTTON_WIDTH:  # Check if click is in button area
                    button_index = y // (BUTTON_HEIGHT + 10)
                    if button_index == 0:
                        run_algorithm(dfs, speed)
                    elif button_index == 1:
                        run_algorithm(bfs, speed)
                    elif button_index == 2:
                        run_algorithm(ucs, speed)
                    elif button_index == 3:
                        run_algorithm(a_star, speed)
                    elif button_index == 4:
                        run_algorithm(ids, speed)
                    elif button_index == 5:
                        run_algorithm(greedy_best_first, speed)
                    elif button_index == 6:
                        run_algorithm(ida_star, speed)   
                    elif button_index == 7:
                        run_algorithm2(simple_hill_climbing, speed) 
                    elif button_index == 8:
                        run_algorithm2(steepest_ascent_hill_climbing, speed)
    
    pygame.quit()

if __name__ == "__main__":
    main()