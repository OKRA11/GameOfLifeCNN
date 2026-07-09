import math
import random
import os
from tqdm import tqdm
import numpy as np

SAMPLES = 10_000
W, H = 40, 40
DATA_PATH = "dataset3"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def get_next_field(f):
    neighbours = (
        np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
        np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
        np.roll(np.roll(f, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(f, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(f, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(f, -1, axis=0), -1, axis=1)
    )

    next_f = np.logical_or(np.logical_and(f == 1, np.logical_or(neighbours == 2, neighbours == 3)),
                           np.logical_and(f == 0, neighbours == 3)).astype(np.float32)
    return next_f

all_inputs = np.empty((SAMPLES, H, W), dtype=np.float32)
all_targets = np.empty((SAMPLES, H, W), dtype=np.float32)

print("Generating data...")
for i in tqdm(range(SAMPLES)):
    current_field = np.random.choice([0, 1], size=(H, W), p=[0.67, 0.33]).astype(np.float32)
    target = get_next_field(current_field)

    all_inputs[i] = current_field
    all_targets[i] = target

np.save(os.path.join(DATA_PATH, "inputs.npy"), all_inputs)
np.save(os.path.join(DATA_PATH, "targets.npy"), all_targets)

print("Done.")
print(f"{all_inputs.shape}")
