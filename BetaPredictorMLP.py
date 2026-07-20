import torch
import torch.nn as nn

class BetaPredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Входной слой: 3 признака (n, mu, sigma) -> 64 скрытых нейрона
        self.fc1 = nn.Linear(3, 64)
        self.relu1 = nn.ReLU()
        
        # Скрытый слой: 64 нейрона -> 64 нейрона
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        
        # Выходной слой: 64 нейрона -> 1 значение (угол beta)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # Пропускаем данные через слои
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x