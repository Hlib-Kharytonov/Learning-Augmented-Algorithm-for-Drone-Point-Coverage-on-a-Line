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

    def competitive_ratio(self):
            L = min([req.x for req in self.requests])
            R = max([req.x for req in self.requests])
            c = math.tan(self.drone.alpha)
            
            opt_x = (L + R) / 2.0
            opt_y = (R - L) / (2.0 * c)
            
            opt_cost = math.dist((0, 0), (opt_x, opt_y))
            
            if opt_cost == 0:
                opt_cost = 0.0001
            return opt_cost