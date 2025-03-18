import pygame
import heapq
import time
from collections import deque

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 3, 3
TILE_SIZE = WIDTH // (COLS + 3)  # Giảm kích thước ô
BUTTON_HEIGHT = 100
INFO_HEIGHT = 50

start_state = [[2, 6, 5], [8, 7, 0], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def sort_start_state():
    global start_state
    flat_list = sorted(sum(start_state, []))
    flat_list.remove(0)
    flat_list.append(0)
    
    temp_state = sum(start_state, [])  # Lấy danh sách phẳng của trạng thái ban đầu
    
    for i in range(len(temp_state)):  # Duyệt qua từng phần tử để sắp xếp chậm
        temp_state[i] = flat_list[i]
        start_state = [temp_state[j * COLS:(j + 1) * COLS] for j in range(ROWS)]
        
        win.fill((255, 255, 255))
        draw_grid(start_state, 30, "Start")
        draw_grid(goal_state, WIDTH // 2 + 30, "Goal")
        draw_buttons()  
        pygame.display.flip()
        time.sleep(0.3)  # Tạo hiệu ứng chậm (0.3 giây mỗi bước)


pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 60, bold=True)  # Giảm kích thước chữ
button_font = pygame.font.SysFont(None, 30, bold=True)
info_font = pygame.font.SysFont(None, 30, bold=True)

def draw_grid(state, x_offset, label):
    pygame.draw.rect(win, (220, 220, 220), (x_offset, 0, WIDTH // 2, HEIGHT - BUTTON_HEIGHT))
    text_label = font.render(str(label), True, (0, 0, 0))
    win.blit(text_label, (x_offset + TILE_SIZE, 10))
    for r in range(ROWS):
        for c in range(COLS):
            if state[r][c] != 0:
                pygame.draw.rect(win, (0, 0, 255), (x_offset + c * TILE_SIZE, r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE), border_radius=10)
                text = font.render(str(state[r][c]), True, (255, 255, 255))
                text_rect = text.get_rect(center=(x_offset + c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + 50 + TILE_SIZE // 2))
                win.blit(text, text_rect)
            pygame.draw.rect(win, (0, 0, 0), (x_offset + c * TILE_SIZE, r * TILE_SIZE + 50, TILE_SIZE, TILE_SIZE), 3, border_radius=10)
    
def draw_buttons():
    button_labels = ["DFS", "BFS", "UCS", "A*", "IDS", "Greedy"]
    button_colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (200, 0, 200), (0, 200, 200)]
    for i, label in enumerate(button_labels):
        pygame.draw.rect(win, button_colors[i], (i * (WIDTH // 6), HEIGHT - BUTTON_HEIGHT, WIDTH // 6, BUTTON_HEIGHT), border_radius=10)
        text_surface = button_font.render(label, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(i * (WIDTH // 6) + (WIDTH // 12), HEIGHT - BUTTON_HEIGHT + 50))
        win.blit(text_surface, text_rect)
    pygame.display.update()

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
    solution, _, nodes_visited = algorithm(start_state, goal_state)
    clock = pygame.time.Clock()
    start_time = time.time()
    for state in solution:
        time_elapsed = time.time() - start_time  # Cập nhật thời gian liên tục
        clock.tick(speed)
        draw_grid(state, time_elapsed, nodes_visited)


def main():
    run = True
    speed = 1
    draw_grid(start_state, 0, 0)
    draw_buttons()
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y >= HEIGHT - BUTTON_HEIGHT:
                    if x < WIDTH // 4:
                        run_algorithm(dfs, speed)
                    elif x < 2 * WIDTH // 4:
                        run_algorithm(bfs, speed)
                    elif x < 3 * WIDTH // 4:
                        run_algorithm(ucs, speed)
                    else:
                        run_algorithm(a_star, speed)
                    draw_buttons()
    pygame.quit()

main()


