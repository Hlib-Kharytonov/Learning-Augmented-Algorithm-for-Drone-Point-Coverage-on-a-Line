import Environment
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("=== test simulation ===")
    field = Environment.Environment()
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
    
    field.drone.reset()
    print("=== BETA-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.beta_hedge_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    plt.plot(x,y,"g-")
    plt.show()
    # plt.savefig('my_simulation.png')
    
    
