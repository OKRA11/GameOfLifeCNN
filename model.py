import torch
import torch.nn as nn

class GameOfLifeCNN(nn.Module):
    def __init__(self):
        super(GameOfLifeCNN, self).__init__()
        # Conv3x3 сохраняет размер поля за счет padding=1
        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.relu = nn.ReLU()
        # Conv1x1 для финального решения по каждой клетке
        self.conv2 = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

class LogisticRegressionModel(nn.Module):
    """Линейная модель без понимания пространственных связей (Baseline)."""
    def __init__(self, size=64):
        super(LogisticRegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        # Вход: 64*64, Выход: 64*64. Каждый пиксель — отдельная фича.
        self.linear = nn.Linear(size * size, size * size)
        self.sigmoid = nn.Sigmoid()
        self.size = size

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.view(-1, 1, self.size, self.size)
