import math
import random
import matplotlib.pyplot as plt

#random.seed(42)

class Request:
    
    def __init__(self,x, feas_cone):
        self.x= x
        self.feas_cone=feas_cone 

class Drone:
    def __init__(self, alpha):
        self.x = self.y = 0.0
        self.alpha = alpha
        self.total_distance = 0.0
        self.min_x_seen = 0.0
        self.max_x_seen = 0.0
        self.movement_track=[(0,0)]

        
    def reset(self):
        self.x = self.y = 0.0
        self.total_distance = 0.0
        self.min_x_seen = 0.0
        self.max_x_seen = 0.0
        self.movement_track=[(0,0)]
        
    def get_coverage_radius(self):
        return self.y * math.tan(self.alpha)


    def move_straight_up(self, target_y ):
        
        if target_y > self.y:
            distance_flown = target_y - self.y
            self.total_distance += distance_flown
            self.y = target_y
            self.movement_track.append((0,self.y))
            print(f"Drone moved to ({self.x:.2f},{self.y:.2f}), total distance: {self.total_distance:.2f}")
            
        else:
            print("drone didn't move")


    def straight_up_algorithm(self, target_x):
        horizontalDist = abs(self.x - target_x)
        requiredY = horizontalDist / math.tan(self.alpha)
        self.move_straight_up(requiredY) 


    def move_greedy(self, target_x, target_y):
        
        if target_y >= self.y:
            distance_flown = math.dist((self.x, self.y),(target_x,target_y))
            self.total_distance +=distance_flown
            self.x = target_x
            self.y = target_y
            self.movement_track.append((self.x,self.y))
            print(f"Drone moved to ({self.x:.2f},{self.y:.2f}). total distance: {self.total_distance:.2f}")
        
        else:
            print("drone didn't moove")
    
    def greedy_algorithm(self,target_x):
        self.min_x_seen = min(self.min_x_seen, target_x)
        self.max_x_seen = max(self.max_x_seen, target_x)
        
        apex_x = (self.min_x_seen + self.max_x_seen)/2.0
        
        distance_to_edge = (self.max_x_seen - self.min_x_seen)/2.0
        requiredY = distance_to_edge / math.tan(self.alpha)

        self.move_greedy(apex_x, requiredY)





class Environment:
    AOV = math.pi/4 # <-- CHANGE AOV HERE
    extr_left_distance = 0.0
    extr_right_distance = 0.0
    def __init__(self):
        self.drone = Drone(self.AOV) 
        self.requests = []
        
    def spawn_requests(self, nb_requests, mu, sig):
        self.requests = [Request(random.gauss(mu,sig),self.AOV) for _ in range(nb_requests)]
        # self.requests.sort(key=lambda req: abs(req.x))




if __name__ == "__main__":
    print("=== test simulation ===")
    field = Environment()
    field.spawn_requests(nb_requests=100, mu=100.0, sig=100.0)
    
    print(f"{len(field.requests)} requests has been generated. Their coordinatest:") # test the order of points
    for i, req in enumerate(field.requests):
       print(f"request {i+1}: x = {req.x:.2f}")
       
       
    print("=== STRAIGHT-UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.straight_up_algorithm(req.x)
        
    n=len(field.requests)
    plt.plot([r.x for r in field.requests],[0]*n,"ko")
    x,y = zip(*field.drone.movement_track)
    
    plt.plot(x,y,"r-")
    # plt.show()
        
        
    field.drone.reset()
    print("=== GREEDY algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.greedy_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    plt.plot(x,y,"b-")
    plt.show()
    # plt.savefig('my_simulation.png')
    
    
