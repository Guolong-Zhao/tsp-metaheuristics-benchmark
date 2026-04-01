import time
import numpy as np
import random
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------ JIT 加速函数 ------------------

@njit(fastmath=True)
def tour_cost_jit(tour, dist):
    cost = 0.0
    n = tour.shape[0]
    for i in range(n-1):
        cost += dist[tour[i], tour[i+1]]
    cost += dist[tour[-1], tour[0]]
    return cost

@njit(fastmath=True)
def two_opt_jit(tour, dist, max_swap=20):
    best = tour.copy()
    best_cost = tour_cost_jit(best, dist)
    n = tour.shape[0]
    improved = True
    while improved:
        improved = False
        # 限制 j−i <= max_swap，避免全 O(n^2) 搜索
        for i in range(n - 2):
            for j in range(i+2, min(n, i+max_swap)):
                new_tour = best.copy()
                # 逆序片段
                new_tour[i+1:j+1] = best[j:i:-1]
                c = tour_cost_jit(new_tour, dist)
                if c < best_cost:
                    best = new_tour
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
    return best

# ------------------ 工具函数 ------------------

def load_tsplib(filename):
    coords = []
    with open(filename, 'r') as f:
        reading = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if not reading or line == "EOF":
                continue
            parts = line.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    coords = np.array(coords)
    # 欧几里得距离矩阵
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return coords, dist

def nearest_neighbor_tour(start, dist):
    n = dist.shape[0]
    unv = set(range(n))
    tour = [start]
    unv.remove(start)
    cur = start
    while unv:
        nxt = min(unv, key=lambda j: dist[cur, j])
        tour.append(nxt)
        unv.remove(nxt)
        cur = nxt
    return tour

def pmx_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    c1 = [-1]*size; c2 = [-1]*size
    c1[a:b+1] = p1[a:b+1]
    c2[a:b+1] = p2[a:b+1]
    m1 = {p1[i]: p2[i] for i in range(a, b+1)}
    m2 = {p2[i]: p1[i] for i in range(a, b+1)}
    for i in range(size):
        if c1[i] == -1:
            g = p2[i]
            while g in m1:
                g = m1[g]
            c1[i] = g
        if c2[i] == -1:
            g = p1[i]
            while g in m2:
                g = m2[g]
            c2[i] = g
    return c1, c2

def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]

# ------------------ 优化后的 GA 主函数 ------------------

def genetic_algorithm_optimized(
    dist_matrix,
    pop_size=100,
    mutation_rate=0.05,
    generations=300,
    tournament_size=5,
    elitism_k=5,
    nn_ratio=0.3,
    two_opt_freq=10,
    two_opt_top=0.1
):
    n = dist_matrix.shape[0]

    # 1. 初始化种群：nn_ratio 比例用最近邻，剩余随机
    pop = []
    nn_cnt = int(pop_size * nn_ratio)
    for _ in range(nn_cnt):
        pop.append(nearest_neighbor_tour(random.randrange(n), dist_matrix))
    for _ in range(pop_size - nn_cnt):
        pop.append(random.sample(range(n), n))

    best_costs = []

    for gen in tqdm(range(generations), desc="GA 进化"):
        # 2. 评估并排序
        arr_pop = [np.array(t) for t in pop]
        costs = [tour_cost_jit(t, dist_matrix) for t in arr_pop]
        idx = np.argsort(costs)
        pop = [pop[i] for i in idx]
        best_costs.append(costs[idx[0]])

        # 3. 精英保留
        new_pop = pop[:elitism_k]

        # 4. 交叉 + 变异 生成子代
        while len(new_pop) < pop_size:
            # 锦标赛选择
            cands = random.sample(pop, tournament_size)
            p1 = min(cands, key=lambda t: tour_cost_jit(np.array(t), dist_matrix))
            cands = random.sample(pop, tournament_size)
            p2 = min(cands, key=lambda t: tour_cost_jit(np.array(t), dist_matrix))

            c1, c2 = pmx_crossover(p1, p2)
            if random.random() < mutation_rate:
                swap_mutation(c1)
            if random.random() < mutation_rate:
                swap_mutation(c2)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

        # 5. 限频 2-opt：每 two_opt_freq 代对前 two_opt_top 比例个体做一次
        if gen % two_opt_freq == 0:
            top_n = int(pop_size * two_opt_top)
            for i in range(elitism_k + top_n):
                pop[i] = two_opt_jit(np.array(pop[i]), dist_matrix).tolist()

    # 6. 绘制收敛曲线
    # plt.plot(best_costs)
    # plt.xlabel("代数")
    # plt.ylabel("最优成本")
    # plt.title("GA + JIT + 限频 2-opt 收敛")
    # plt.show()

    best = min(pop, key=lambda t: tour_cost_jit(np.array(t), dist_matrix))
    best_cost = tour_cost_jit(np.array(best), dist_matrix)
    return best, best_cost, best_costs

# ------------------ 主程序 ------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    coords, dm = load_tsplib("att48.tsp")

    true_opt = 33523
    num_runs = 5
    runtimes = []
    all_best_costs = []
    all_convergence_curves = []

    for run in range(num_runs):
        print(f"\n运行第 {run+1} 次")
        t0 = time.time()
        best_sol, best_c, best_costs = genetic_algorithm_optimized(
            dm,
            pop_size=100,
            mutation_rate=0.05,
            generations=300,
            tournament_size=5,
            elitism_k=5,
            nn_ratio=0.3,
            two_opt_freq=10,
            two_opt_top=0.1
        )
        t1 = time.time()
        runtime = t1 - t0
        runtimes.append(runtime)
        print(f"Run {run+1} 时长: {runtime:.2f} 秒")
        
        all_best_costs.append(best_c)
        print(f"Run {run+1} 最优成本: {best_c:.2f}（比最优多 {best_c - true_opt:.2f}）")

        # log scale 差值记录
        convergence_log = []
        for c in best_costs:
            diff = (c - true_opt)
            convergence_log.append(np.log10(diff + 1e-8))  # 避免 log(0)
        all_convergence_curves.append(convergence_log)
    print("\n每次运行时长 (秒)", [f"{t:.2f}" for t in runtimes])
    # 总结统计信息
    mean_c = np.mean(all_best_costs)
    std_c = np.std(all_best_costs)
    print(f"\n5 次运行平均成本: {mean_c:.2f}")
    print(f"标准差: {std_c:.2f}")

    # 绘制收敛图（Figure 1 风格）
    plt.figure(figsize=(10, 6))
    for i, curve in enumerate(all_convergence_curves):
        plt.plot(curve, label=f"Run {i+1}")
    plt.xlabel("Generation")
    plt.ylabel("log10(Tour Cost - Optimum)")
    plt.title("GA Convergence Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
