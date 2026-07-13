import math
import random
from models import Request, Drone
import numpy as np

class Environment:
    AOV = math.pi/4 # <-- CHANGE AOV HERE
    extr_left_distance = 0.0
    extr_right_distance = 0.0
    def __init__(self):
        self.drone = Drone() 
        self.requests = []
        
    def spawn_requests(self, nb_requests, mu, sig):
        self.requests = [Request(random.gauss(mu,sig),self.AOV) for _ in range(nb_requests)]
        # self.requests.sort(key=lambda req: abs(req.x))
        # data = np.array(self.requests)
        # normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        # self.requests = normalized.tolist()       #mb doesnt work

    # def OPT(self):
    #         L = min([req.x for req in self.requests])
    #         R = max([req.x for req in self.requests])

    #         c = 1/math.tan(self.drone.alpha)
            
    #         return 1/2 * math.sqrt(((R - L)**2)*c**2 + (R + L)**2)
    
    def OPT(self):
        # На самом деле функция возвращает opt_cost (идеальную дистанцию), 
        # а не само ратио, но название можно оставить.
        L = min([req.x for req in self.requests])
        R = max([req.x for req in self.requests])
        c = math.tan(self.drone.alpha)

        # 1. Полет в геометрический центр кластера
        x1 = (L + R) / 2.0
        y1 = (R - L) / (2.0 * c)
        dist1 = math.dist((0, 0), (x1, y1))

        # 2. Полет по правому краю (срез угла, спасает при mu=1000)
        x2 = R / (1 + c**2)
        y2 = (c * R) / (1 + c**2)
        dist2 = math.dist((0, 0), (x2, y2))
        valid2 = (x2 - c * y2 <= L) and (y2 >= 0) 

        # 3. Полет по левому краю (срез угла, спасает при mu= -1000)
        x3 = L / (1 + c**2)
        y3 = -(c * L) / (1 + c**2)
        dist3 = math.dist((0, 0), (x3, y3))
        valid3 = (x3 + c * y3 >= R) and (y3 >= 0)

        # Собираем все физически возможные пути покрытия
        valid_distances = [dist1]
        if valid2: valid_distances.append(dist2)
        if valid3: valid_distances.append(dist3)

        # Оптимум - это самый дешевый из возможных путей
        opt_cost = min(valid_distances)
        
        return opt_cost if opt_cost > 0 else 0.0001