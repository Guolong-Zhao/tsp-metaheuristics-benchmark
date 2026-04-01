import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

# 计算路径成本
def tour_cost(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]

def load_tsplib(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coords = []
    reading_coords = False
    for line in lines:
        line = line.strip()
        # 忽略空行
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        if line == "EOF":
            break
        if reading_coords:
            parts = line.split()
            if len(parts) < 3:  # 如果行不完整，则跳过
                continue
            try:
                # 获取坐标信息，通常是两个浮动数值
                coords.append([float(parts[1]), float(parts[2])])
            except ValueError:
                continue  # 如果有格式错误，跳过此行
    
    coords = np.array(coords)
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))
    return coords, dist_matrix


# 局部搜索 - 2-opt
def two_opt(tour, dist_matrix):
    best_tour = tour[:]
    best_cost = tour_cost(tour, dist_matrix)
    for i in range(len(tour) - 1):
        for j in range(i + 2, len(tour)):
            new_tour = tour[:]
            new_tour[i:j+1] = reversed(tour[i:j+1])  # 反转部分路径
            new_cost = tour_cost(new_tour, dist_matrix)
            if new_cost < best_cost:
                best_tour, best_cost = new_tour, new_cost
    return best_tour, best_cost

# 初始化路径 - 最近邻算法 (Heuristic)
def nearest_neighbor(dist_matrix):
    n_cities = len(dist_matrix)
    visited = [False] * n_cities
    start = random.randint(0, n_cities - 1)  # 随机起点
    tour = [start]
    visited[start] = True

    for _ in range(n_cities - 1):
        last_city = tour[-1]
        nearest_city = min(
            (i for i in range(n_cities) if not visited[i]),
            key=lambda i: dist_matrix[last_city, i]
        )
        tour.append(nearest_city)
        visited[nearest_city] = True

    return tour

# 改进的模拟退火算法
def simulated_annealing_advanced(dist_matrix, initial_temp=1000, cooling_rate=0.995, iterations=1000, two_opt_enabled=True, plot_progress=True):
    # 使用启发式算法生成初始解
    current_solution = nearest_neighbor(dist_matrix)
    current_cost = tour_cost(current_solution, dist_matrix)

    best_solution = current_solution[:]
    best_cost = current_cost
    
    temp = initial_temp
    cost_progress = [current_cost]  # 记录每次迭代的最优成本
    
    for i in range(iterations):
        # 随机交换两个城市
        i, j = sorted(random.sample(range(len(dist_matrix)), 2))
        new_solution = current_solution[:]
        new_solution[i:j] = reversed(new_solution[i:j])  # 交换部分路径
        new_cost = tour_cost(new_solution, dist_matrix)
        
        # 如果新解更好或者通过模拟退火准则接受新解
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
            current_solution, current_cost = new_solution, new_cost
            if new_cost < best_cost:
                best_solution, best_cost = new_solution[:], new_cost
        
        # 局部搜索 - 2-opt (在每迭代周期后应用)
        if two_opt_enabled:
            best_solution, best_cost = two_opt(best_solution, dist_matrix)
        
        # 温度下降
        temp *= cooling_rate
        
        # 记录最优解进展
        cost_progress.append(best_cost)
        
    # if plot_progress:
    #     plt.plot(cost_progress)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Best Cost")
    #     plt.title("Simulated Annealing Progress")
    #     plt.show()
    
    return best_solution, best_cost, cost_progress

# 示例运行代码
if __name__ == "__main__":
    coords, dist_matrix = load_tsplib("att48.tsp")

    OPTIMAL = 33523
    runs = 5
    all_costs = []
    all_convergences = []
    all_times = []

    for run in range(runs):
        print(f"\n=== Run {run + 1} ===")
        
        # 随机种子初始化（基于系统时间）
        random.seed(None)
        np.random.seed(None)

        start_time = time.time()

        best_sol, best_cost, convergence = simulated_annealing_advanced(
            dist_matrix,
            iterations=1000,
            initial_temp=1000,
            cooling_rate=0.995,
            two_opt_enabled=True,
            plot_progress=False
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        all_costs.append(best_cost)
        all_convergences.append(convergence)
        all_times.append(elapsed_time)

        print(f"Run {run + 1} Best Cost: {best_cost:.2f} — Gap: {best_cost - OPTIMAL:.2f} — Time: {elapsed_time:.2f} seconds")

    # 统计汇总
    mean_cost = np.mean(all_costs)
    std_dev = np.std(all_costs)
    best_run_cost = min(all_costs)
    avg_time = np.mean(all_times)

    print(f"\nSummary over {runs} runs:")
    print(f"Best Cost: {best_run_cost:.2f}")
    print(f"Average Cost: {mean_cost:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Average Time per Run: {avg_time:.2f} seconds")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    for idx, convergence in enumerate(all_convergences):
        log_diff = [np.log10((c - OPTIMAL + 1e-8)) for c in convergence]
        plt.plot(log_diff, label=f'Run {idx + 1}')

    plt.xlabel("Iteration")
    plt.ylabel("log10(Tour Cost - p)")
    plt.title("Simulated Annealing Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()