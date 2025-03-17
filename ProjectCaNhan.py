import pygame
import heapq
import time
from collections import deque

WIDTH, HEIGHT = 600, 750
ROWS, COLS = 3, 3
TILE_SIZE = WIDTH // COLS
BUTTON_HEIGHT = 100
INFO_HEIGHT = 50

start_state = [[2, 6, 5], [8, 7, 0], [4, 3, 1]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 100)
button_font = pygame.font.SysFont(None, 50)
info_font = pygame.font.SysFont(None, 30)

def draw_grid(state, time_elapsed, nodes_visited):
    win.fill((255, 255, 255))
    for r in range(ROWS):
        for c in range(COLS):
            if state[r][c] != 0:
                pygame.draw.rect(win, (0, 0, 255), (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                text = font.render(str(state[r][c]), True, (255, 255, 255))
                win.blit(text, (c * TILE_SIZE + TILE_SIZE // 3, r * TILE_SIZE + TILE_SIZE // 3))
            pygame.draw.rect(win, (0, 0, 0), (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)
    
    info_text = f"Time: {time_elapsed:.4f} sec | Nodes: {nodes_visited}"
    win.blit(info_font.render(info_text, True, (0, 0, 0)), (20, HEIGHT - BUTTON_HEIGHT - INFO_HEIGHT))
    pygame.display.flip()

def draw_buttons():
    pygame.draw.rect(win, (200, 0, 0), (0, HEIGHT - BUTTON_HEIGHT, WIDTH // 4, BUTTON_HEIGHT))
    pygame.draw.rect(win, (0, 200, 0), (WIDTH // 4, HEIGHT - BUTTON_HEIGHT, WIDTH // 4, BUTTON_HEIGHT))
    pygame.draw.rect(win, (0, 0, 200), (2 * WIDTH // 4, HEIGHT - BUTTON_HEIGHT, WIDTH // 4, BUTTON_HEIGHT))
    pygame.draw.rect(win, (200, 200, 0), (3 * WIDTH // 4, HEIGHT - BUTTON_HEIGHT, WIDTH // 4, BUTTON_HEIGHT))
    win.blit(button_font.render("DFS", True, (255, 255, 255)), (WIDTH // 8 - 20, HEIGHT - BUTTON_HEIGHT + 30))
    win.blit(button_font.render("BFS", True, (255, 255, 255)), (3 * WIDTH // 8 - 20, HEIGHT - BUTTON_HEIGHT + 30))
    win.blit(button_font.render("UCS", True, (255, 255, 255)), (5 * WIDTH // 8 - 20, HEIGHT - BUTTON_HEIGHT + 30))
    win.blit(button_font.render("A*", True, (255, 255, 255)), (7 * WIDTH // 8 - 20, HEIGHT - BUTTON_HEIGHT + 30))
    pygame.display.flip()

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
