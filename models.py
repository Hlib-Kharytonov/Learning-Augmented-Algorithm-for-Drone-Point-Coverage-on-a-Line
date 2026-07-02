import math

class Request:
    
    def __init__(self,x, feas_cone):
        self.x= x
        self.feas_cone=feas_cone 

class Drone:
    def __init__(self):
        self.x = self.y = 0.0
        
        self.alpha = math.pi/4          #you can change your AOV here, but you need qs well to change corresponding beta ange just below
        self.beta = math.pi/9.666		#as our drone has fixed  AOV of pi/4 
        
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


    def move_zigzag(self, target_x, target_y):
        
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

        self.move_zigzag(apex_x, requiredY)
        
    def calculate_dist_beta(self,r):
        return (math.tan(self.alpha)+(1+2*r)*math.tan(self.beta)) / (((math.tan(self.alpha)+math.tan(self.beta))**2)*math.cos(self.beta))
        
        
    def beta_hedge_algorithm(self, target_x):
        
        self.min_x_seen = min(self.min_x_seen,target_x)
        self.max_x_seen = max(self.max_x_seen,target_x)
        
        current_coverage = self.get_coverage_radius()
        
        # Если L и R уже внутри текущего конуса, не двигаемся
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return
        
        r = abs(self.max_x_seen - self.min_x_seen)
        dist = self.calculate_dist_beta(r)
        
        apex_x = (self.min_x_seen + self.max_x_seen) / 2.0
        direction = 1 if apex_x > self.x else -1
        
        x_X0_U_Zprim = dist*math.cos(math.pi/2-self.beta)
        y_X0_U_Zprim = dist*math.sin(math.pi/2-self.beta)
        
        x = self.x + (direction * x_X0_U_Zprim)
        y = self.y + y_X0_U_Zprim
        
        self.move_zigzag(x,y)
     
        
