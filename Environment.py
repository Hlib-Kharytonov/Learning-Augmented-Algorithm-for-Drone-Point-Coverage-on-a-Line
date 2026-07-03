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
        self.requests.sort(key=lambda req: abs(req.x))
        # data = np.array(self.requests)
        # normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        # self.requests = normalized.tolist()       #mb doesnt work