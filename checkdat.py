import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = "datasethighless"
inputs = np.load(os.path.join(DATA_PATH, 'inputs.npy'))
targets = np.load(os.path.join(DATA_PATH, 'targets.npy'))

print(f"Всего примеров в базе: {len(inputs)}")
print(f"Размер одного поля: {inputs[0].shape}")


idx = np.random.randint(0, len(inputs))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(inputs[idx], cmap='binary')
plt.title(f"Пример {idx}: Кадр сейчас")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(targets[idx], cmap='binary')
plt.title("Кадр через 1 шаг (Ответ)")
plt.axis('off')

plt.show()
