import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Импортируем ваши классы из других файлов
from Supervised_Learning.DroneDataset import DroneDataset
from Supervised_Learning.BetaPredictorMLP import BetaPredictorMLP

def train_model():
    # 1. Загрузка данных
    print("Загрузка датасета...")
    dataset = DroneDataset('drone_dataset.csv')
    
    # Разделяем данные: 80% на обучение, 20% на проверку (тестирование)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoader разбивает данные на "батчи" (пачки по 32 строки) для ускорения обучения
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Инициализация нейросети
    model = BetaPredictorMLP()
    
    # 3. Настройки обучения
    # MSELoss (Mean Squared Error) - стандартная функция ошибки для задач регрессии (предсказания числа)
    criterion = nn.MSELoss() 
    # Оптимизатор Adam - алгоритм, который будет обновлять веса нейросети
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    epochs = 300 # Сколько раз сеть полностью просмотрит датасет
    
    print("Начинаем обучение...")
    
    # 4. Главный цикл обучения
    for epoch in range(epochs):
        model.train() # Переводим модель в режим тренировки
        running_loss = 0.0
        
        for features, targets in train_loader:
            optimizer.zero_grad()               # Сбрасываем старые вычисления
            
            outputs = model(features)           # Сеть пытается предсказать угол
            loss = criterion(outputs, targets)  # Считаем, насколько она ошиблась
            
            loss.backward()                     # Вычисляем направление работы над ошибками
            optimizer.step()                    # Обновляем "синапсы" (веса) сети
            
            running_loss += loss.item()
            
        # Каждые 10 эпох выводим статистику
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка (Loss): {avg_loss:.4f}")
            
    # 5. Сохраняем "мозг" на жесткий диск
    torch.save(model.state_dict(), 'trained_beta_predictor.pth')
    print("Обучение завершено! Веса модели сохранены в файл 'trained_beta_predictor.pth'")

if __name__ == "__main__":
    train_model()