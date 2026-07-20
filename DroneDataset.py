import torch
from torch.utils.data import Dataset
import pandas as pd

class DroneDataset(Dataset):
    def __init__(self, csv_file):
        # 1. Загружаем таблицу с данными (например, с помощью pandas)
        self.data = pd.read_csv(csv_file)
        
        # 2. Выделяем признаки (X) и то, что предсказываем (y)
        # Предполагаем, что колонки называются так:
        # Делим на 1000.0, чтобы числа были от 0 до 1
        self.X = self.data[['n', 'mu', 'sigma']].values / 1000.0
        self.y = self.data['best_beta'].values
        
    def __len__(self):
        # 3. Метод сообщает PyTorch, сколько всего строк в нашей таблице
        return len(self.data)
    
    def __getitem__(self, idx):
        # 4. Вытаскиваем одну строку по индексу и превращаем в PyTorch-тензор
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        # unsqueeze(0) превращает скалярное число (например, 0.32) в массив из одного элемента [0.32]
        # Это нужно, чтобы размерности совпали с выходом сети
        return features, target.unsqueeze(0)