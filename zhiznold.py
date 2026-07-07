import pygame
from random import randint, choice
from copy import deepcopy
import math

RES = WIDTH, HEIGHT = 1400, 800
TILE = 10
W, H = WIDTH // TILE, HEIGHT // TILE
FPS = -1

cols = ["blue", "purple", "red", "orange", "gray", "yellow", "pink", "green"]

pygame.init()
surface = pygame.display.set_mode(RES)
clock = pygame.time.Clock()

next_field = [[0 for i in range(W)] for j in range(H)]
current_field = [[1 if not (int(math.sin(i) * 1000) * int(math.cos(j) * 1000)) % 2 else 0 for i in range(W)] for j in range(H)]

def check_cell(current_filed, x, y):
    count = 0
    for j in range(y - 1, y + 2):
        for i in range(x - 1, x + 2):
            if current_filed[j][i]:
                count += 1

    if current_filed[y][x]:
        count -= 1
        if count == 2 or count == 3:
            return 1
        return 0

    else:
        if count == 3:
            return 1
        return 0

#current_field[10][5] = 1
#current_field[11][5] = 1
#current_field[10][6] = 1
#current_field[11][6] = 1

#current_field[8][10] = 1
#current_field[9][10] = 1
#current_field[10][10] = 1

#current_field[20][5] = 1
#current_field[21][6] = 1
#current_field[21][7] = 1
#current_field[20][7] = 1
#current_field[19][7] = 1

#current_field[25][20] = 1
#current_field[26][20] = 1
#current_field[25][21] = 1
#current_field[26][21] = 1
#current_field[27][22] = 1
#current_field[28][22] = 1
#current_field[27][23] = 1
#current_field[28][23] = 1

while True:

    surface.fill(pygame.Color('black'))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    [pygame.draw.line(surface, pygame.Color('dimgray'), (x, 0), (x, HEIGHT)) for x in range(0, WIDTH, TILE)]
    [pygame.draw.line(surface, pygame.Color('dimgray'), (0, y), (WIDTH, y)) for y in range(0, HEIGHT, TILE)]

    for x in range(1, W - 1):
        for y in range(1, H - 1):
            if current_field[y][x]:
                pygame.draw.rect(surface, pygame.Color(choice(cols)), (x*TILE + 2, y*TILE + 2, TILE - 2, TILE - 2))
            next_field[y][x] = check_cell(current_field, x, y)

    current_field = deepcopy(next_field)


    pygame.display.flip()
    clock.tick(FPS)
