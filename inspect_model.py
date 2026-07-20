import torch

# 1. Загружаем сохраненный файл
weights = torch.load('trained_beta_predictor.pth')

# 2. Выводим названия всех слоев сети и размеры их матриц
print("Содержимое файла .pth:")
print("-" * 30)
for layer_name, matrix in weights.items():
    print(f"Слой: {layer_name: <15} | Размер матрицы: {list(matrix.size())}")