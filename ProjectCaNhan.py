import pygame
import heapq
import time
from collections import deque

WIDTH, HEIGHT = 1500, 800
ROWS, COLS = 3, 3
TILE_SIZE = (WIDTH - 200) // (COLS + 4)  # Giảm để dành chỗ cho nút bên trái
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 80

start_state = [[2, 6, 5], [8, 7, 0], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 60, bold=True)
button_font = pygame.font.SysFont(None, 30, bold=True)
info_font = pygame.font.SysFont(None, 30, bold=True)

# Màu sắc mới
BG_COLOR = (245, 245, 250)  # Xám nhạt
TILE_COLOR = (65, 105, 225)  # Xanh hoàng gia
BORDER_COLOR = (25, 25, 112)  # Xanh đậm
TEXT_COLOR = (255, 255, 255)  # Trắng
BUTTON_COLORS = [
    (255, 87, 87),    # Đỏ nhạt
    (66, 214, 164),   # Xanh ngọc
    (255, 159, 67),   # Cam
    (147, 112, 219),  # Tím trung bình
    (255, 215, 0),    # Vàng
    (50, 205, 50)     # Xanh lá
]

def draw_grid(state, x_offset, label):
    # Vẽ nền cho lưới
    pygame.draw.rect(win, (230, 230, 250), (x_offset, 0, WIDTH // 2, HEIGHT))
    text_label = font.render(str(label), True, BORDER_COLOR)
    win.blit(text_label, (x_offset + TILE_SIZE, 10))
    
    for r in range(ROWS):
        for c in range(COLS):
            if state[r][c] != 0:
                pygame.draw.rect(win, TILE_COLOR,
                               (x_offset + c * TILE_SIZE, r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE),
                               border_radius=15)
                text = font.render(str(state[r][c]), True, TEXT_COLOR)
                text_rect = text.get_rect(center=(x_offset + c * TILE_SIZE + TILE_SIZE // 2,
                                                r * TILE_SIZE + 50 + TILE_SIZE // 2))
                win.blit(text, text_rect)
            pygame.draw.rect(win, BORDER_COLOR,
                           (x_offset + c * TILE_SIZE, r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE),
                           3, border_radius=15)

def draw_buttons():
    button_labels = ["DFS", "BFS", "UCS", "A*", "IDS", "Greedy"]
    for i, label in enumerate(button_labels):
        pygame.draw.rect(win, BUTTON_COLORS[i],
                        (10, 10 + i * (BUTTON_HEIGHT + 10), BUTTON_WIDTH, BUTTON_HEIGHT),
                        border_radius=12)
        text_surface = button_font.render(label, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(10 + BUTTON_WIDTH // 2,
                                                10 + i * (BUTTON_HEIGHT + 10) + BUTTON_HEIGHT // 2))
        win.blit(text_surface, text_rect)
    pygame.display.update()

# Các hàm tìm kiếm giữ nguyên như code gốc
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

def run_algorithm(algorithm, speed):
    win.fill(BG_COLOR)
    solution, time_taken, nodes_visited = algorithm(start_state, goal_state)
    clock = pygame.time.Clock()
    
    for state in solution:
        win.fill(BG_COLOR)
        draw_grid(state, BUTTON_WIDTH + 30, "Current")
        draw_grid(goal_state, WIDTH // 2 + BUTTON_WIDTH + 30, "Goal")
        draw_buttons()
        pygame.display.flip()
        clock.tick(speed)
    
    # Hiển thị thông tin
    info_text = f"Time: {time_taken:.2f}s | Nodes: {nodes_visited}"
    info_surface = info_font.render(info_text, True, BORDER_COLOR)
    win.blit(info_surface, (BUTTON_WIDTH + 40, HEIGHT - 50))
    pygame.display.flip()

def main():
    run = True
    speed = 1  # Tăng tốc độ lên một chút
    
    win.fill(BG_COLOR)
    draw_grid(start_state, BUTTON_WIDTH + 30, "Start")
    draw_grid(goal_state, WIDTH // 2 + BUTTON_WIDTH + 30, "Goal")
    draw_buttons()
    pygame.display.flip()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < BUTTON_WIDTH:  # Kiểm tra click trong vùng nút
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
    
    pygame.quit()

if __name__ == "__main__":
    main()