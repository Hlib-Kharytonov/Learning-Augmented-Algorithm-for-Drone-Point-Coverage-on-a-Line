import Environment
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

if __name__ == "__main__":
    random.seed(17)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6.5), dpi=300)   
    print("=== test simulation ===")
    field = Environment.Environment()
    field.spawn_requests(nb_requests=500, mu=100.0, sig=100.0)
    
    print(f"{len(field.requests)} requests has been generated. Their coordinatest:") # test the order of points
    for i, req in enumerate(field.requests):
       print(f"request {i+1}: x = {req.x:.2f}")
       
    print("=== STRAIGHT-UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.straight_up_algorithm(req.x)
        
    n=len(field.requests)
    ax3.plot([r.x for r in field.requests],[0]*n,"ko",label="Targets")
    x,y = zip(*field.drone.movement_track)
    
    dist_straight = field.drone.total_distance
    
    ax3.plot(x,y, label="STRAIGHT-UP")

        
        
    field.drone.reset()
    print("=== GREEDY algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.greedy_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_greedy = field.drone.total_distance
    
    ax3.plot(x,y, label="GREEDY")
    
    field.drone.reset()
    print("=== BETA-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.beta_hedge_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_beta = field.drone.total_distance
    
    ax3.plot(x,y, label="BETA-HEDGE")
    
    
    
    field.drone.reset()
    print("=== LEARNING BETA UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.learning_beta_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LBA = field.drone.total_distance
    
    ax3.plot(x,y, label="LEARNING STRAIGHT UP")
    
    
    
    field.drone.reset()
    print("=== LEARNING GREEDY UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.learning_greedy_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LGA = field.drone.total_distance
    
    ax3.plot(x,y, label="LEARNING GREEDY UP")


    field.drone.reset()
    print("===  MWU algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.MWU_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_MWU = field.drone.total_distance
    
    ax3.plot(x,y, label="MWU")
    # 
    # print("\n" + "="*30)
    # print("results::")
    # print(f"STRAIGHT-UP:  {dist_straight:.2f}")
    # print(f"GREEDY:       {dist_greedy:.2f}")
    # print(f"BETA-HEDGE:   {dist_beta:.2f}")
    # print(f"LEARNING BETA UP: {dist_LBA:.2f}")
    # print(f"LEARNING GREEDY UP: {dist_LGA:.2f}")
    # print("="*30 + "\n")
    
    
    final_x = field.drone.x
    final_y = field.drone.y

    coverage_radius = final_y * math.tan(field.drone.alpha)

    ax3.plot([final_x, final_x - coverage_radius], [final_y, 0], "g--", alpha=0.5)
    ax3.plot([final_x, final_x + coverage_radius], [final_y, 0], "g--", alpha=0.5)
    ax3.set_title(r'trajectories of the drone for different algorithms')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')


#======================================== Mu dependence test =========================================

    stats = {
        "STRAIGHT-UP": [],
        "GREEDY": [],
        "BETA-HEDGE": [],
        "LEARNING BETA UP": [],
        "LEARNING GREEDY UP": [],
        "MWU": [] }
    
    mu_samples = [0, 10, 100, 500, 650]
    k = 100 

    for mu in mu_samples:
        sigma_stats = {algo: 0 for algo in stats.keys()}
        total_OPT = 0

        for i in range(k):
            field = Environment.Environment()
            field.spawn_requests(nb_requests=500, mu=mu, sig=100)

            OPT = field.OPT()
            
            
            # --- STRAIGHT-UP ---
            for req in field.requests:
                field.drone.straight_up_algorithm(req.x)
            sigma_stats["STRAIGHT-UP"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- GREEDY ---
            for req in field.requests:
                field.drone.greedy_algorithm(req.x)
            sigma_stats["GREEDY"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- BETA-HEDGE ---
            for req in field.requests:
                field.drone.beta_hedge_algorithm(req.x)
            sigma_stats["BETA-HEDGE"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- LEARNING BETA UP ---
            for req in field.requests:
                field.drone.learning_beta_up_algorithm(req.x)
            sigma_stats["LEARNING BETA UP"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- LEARNING GREEDY UP ---
            for req in field.requests:
                field.drone.learning_greedy_up_algorithm(req.x)
            sigma_stats["LEARNING GREEDY UP"] += field.drone.total_distance/OPT
            field.drone.reset()

            # --- MWU ---
            for req in field.requests:
                field.drone.MWU_algorithm(req.x)
            sigma_stats["MWU"] += field.drone.total_distance/OPT
            field.drone.reset()

        for algo in stats.keys():
            stats[algo].append(sigma_stats[algo] / k) 

    
    df = pd.DataFrame(stats, index=mu_samples)
    df.plot(ax=ax1, marker='o')

    ax1.set_title(r'dependence of mu on distance flown, sigma=100')
    ax1.set_xlabel(r'Esperance($\mu$)')
    ax1.set_ylabel('Competitive Ratio')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')

    fig.suptitle(r'Effectiveness of Algorithms', fontsize=14, y=0.95)
    fig.suptitle('Effectiveness of Algorithms ($k=20$ instances, $N=500$ targets)', fontsize=16, y=1.02)


#======================================== SIGMA dependence test =========================================
    stats = {
        "STRAIGHT-UP": [],
        "GREEDY": [],
        "BETA-HEDGE": [],
        "LEARNING BETA UP": [],
        "LEARNING GREEDY UP": [],
        "MWU": []}
    
    sigma_samples = [10, 50, 100, 150, 300]

    for sigma in sigma_samples:
        sigma_stats = {algo: 0 for algo in stats.keys()}
        

        for i in range(k):
            field = Environment.Environment()
            field.spawn_requests(nb_requests=500, mu=100, sig=sigma)
            OPT = field.OPT()
            
            # --- STRAIGHT-UP ---
            for req in field.requests:
                field.drone.straight_up_algorithm(req.x)
            sigma_stats["STRAIGHT-UP"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- GREEDY ---
            for req in field.requests:
                field.drone.greedy_algorithm(req.x)
            sigma_stats["GREEDY"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- BETA-HEDGE ---
            for req in field.requests:
                field.drone.beta_hedge_algorithm(req.x)
            sigma_stats["BETA-HEDGE"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- LEARNING BETA UP ---
            for req in field.requests:
                field.drone.learning_beta_up_algorithm(req.x)
            sigma_stats["LEARNING BETA UP"] += field.drone.total_distance/OPT
            field.drone.reset()
            
            # --- LEARNING GREEDY UP ---
            for req in field.requests:
                field.drone.learning_greedy_up_algorithm(req.x)
            sigma_stats["LEARNING GREEDY UP"] += field.drone.total_distance/OPT
            field.drone.reset()

            for req in field.requests:
                field.drone.MWU_algorithm(req.x)
            sigma_stats["MWU"] += field.drone.total_distance/OPT
            field.drone.reset()

        for algo in stats.keys():
            stats[algo].append(sigma_stats[algo] / k)
        
    
    df = pd.DataFrame(stats, index=sigma_samples)
    df.plot(ax=ax2, marker='o')

    ax2.set_title(r'dependence of sigma on distance flown, mu=100')
    ax2.set_xlabel(r'Esperance($\sigma$)')
    ax2.set_ylabel('Competitive Ratio')

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')

    # plt.show()

    plt.tight_layout()
    plt.savefig("high_res_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig("high_res_plots.pdf", dpi=300, bbox_inches='tight')
    # plt.show()
    
    
