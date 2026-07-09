import numpy as np

def generate_highlife_data(n_samples=10000, size=40, live_ratio=0.2):
    inputs = np.random.choice([0, 1], size=(n_samples, size, size), p=[1-live_ratio, live_ratio]).astype(np.float32)
    
    n = 0
    n += np.roll(inputs, 1, axis=1)
    n += np.roll(inputs, -1, axis=1)
    n += np.roll(inputs, 1, axis=2)
    n += np.roll(inputs, -1, axis=2)
    n += np.roll(np.roll(inputs, 1, axis=1), 1, axis=2)
    n += np.roll(np.roll(inputs, 1, axis=1), -1, axis=2)
    n += np.roll(np.roll(inputs, -1, axis=1), 1, axis=2)
    n += np.roll(np.roll(inputs, -1, axis=1), -1, axis=2)
    
    birth = (n == 3) | (n == 6)
    survive = (inputs == 1) & ((n == 2) | (n == 3))
    
    targets = (birth | survive).astype(np.float32)
    
    return inputs, targets

print("Генерация данных HighLife...")
new_inputs, new_targets = generate_highlife_data(live_ratio=0.2)
np.save('inputs.npy', new_inputs)
np.save('targets.npy', new_targets)
print("Готово! Файлы сохранены.")
