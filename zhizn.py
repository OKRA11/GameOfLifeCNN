import pygame
from random import randint, choice
from copy import deepcopy
import math
import numpy as np

RES = WIDTH, HEIGHT = 1400, 800
TILE = 10
W, H = WIDTH // TILE, HEIGHT // TILE
FPS = -1

cols = ["blue", "purple", "red", "orange", "gray", "yellow", "magenta", "green"]

pygame.init()
surface = pygame.display.set_mode(RES)
clock = pygame.time.Clock()

my_list = [[1 if not (int(math.sin(i) * 1000) * int(math.cos(j) * 1000)) % 2 else 0 for i in range(W)] for j in range(H)]

next_field = [[0 for i in range(W)] for j in range(H)]
current_field = np.array(my_list)

def get_next_field(f):
    neighbours = (np.roll(f, 1 ,axis=0) + np.roll(f, -1 ,axis=0) +
                  np.roll(f, -1 ,axis=1) + np.roll(f, 1 ,axis=1) +
                  np.roll(np.roll(f, 1, axis=0), 1, axis=1) +
                  np.roll(np.roll(f, -1, axis=0), 1, axis=1) +
                  np.roll(np.roll(f, -1, axis=0), -1, axis=1) +
                  np.roll(np.roll(f, 1, axis=0), -1, axis=1))

    next_field = np.logical_or(
        np.logical_and(f == 1, np.logical_or(neighbours == 2, neighbours == 3)),
                       np.logical_and(f == 0, neighbours == 3)).astype(int)
    return next_field



while True:

    surface.fill(pygame.Color('black'))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    [pygame.draw.line(surface, pygame.Color('dimgray'), (x, 0), (x, HEIGHT)) for x in range(0, WIDTH, TILE)]
    [pygame.draw.line(surface, pygame.Color('dimgray'), (0, y), (WIDTH, y)) for y in range(0, HEIGHT, TILE)]

    life_cell = np.argwhere(current_field == 1)

    for y, x in life_cell:
        pygame.draw.rect(surface, pygame.Color(choice(cols)), (x*TILE + 2, y*TILE + 2, TILE - 2, TILE - 2))

    current_field = get_next_field(current_field)


    pygame.display.flip()
    clock.tick(FPS)
