import numpy as np
import random
import matplotlib.pyplot as plt
import time

# --- 1. Load TSPLIB File ---
def load_tsplib(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coords = []
    reading_coords = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        if line == "EOF":
            break
        if reading_coords:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                coords.append([float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    coords = np.array(coords)
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))
    return coords, dist_matrix

# --- 2. Tour Utilities ---
def tour_cost(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]

def two_opt(tour, dist_matrix):
    improved = True
    best = tour
    best_cost = tour_cost(tour, dist_matrix)
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue
                new_tour = best[:]
                new_tour[i:j] = best[j - 1:i - 1:-1]
                new_cost = tour_cost(new_tour, dist_matrix)
                if new_cost < best_cost:
                    best = new_tour
                    best_cost = new_cost
                    improved = True
        tour = best
    return best

# --- 3. Ant Colony Optimization (with history recording) ---
def ant_colony_optimization_2opt(
    dist_matrix,
    n_ants=50,
    n_iterations=100,
    alpha=1.0,
    beta=5.0,
    evaporation_rate=0.5,
    pheromone_min=1e-4,
    pheromone_max=10.0
):
    n_cities = len(dist_matrix)
    pheromone = np.ones((n_cities, n_cities))
    best_tour = None
    best_cost = float("inf")
    convergence_curve = []

    for iteration in range(n_iterations):
        all_tours = []
        for _ in range(n_ants):
            tour = []
            unvisited = set(range(n_cities))
            current = random.randint(0, n_cities - 1)
            tour.append(current)
            unvisited.remove(current)

            while unvisited:
                probabilities = []
                for city in unvisited:
                    pher = pheromone[current][city] ** alpha
                    heuristic = (1.0 / dist_matrix[current][city]) ** beta
                    probabilities.append(pher * heuristic)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = random.choices(list(unvisited), weights=probabilities, k=1)[0]
                tour.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            improved_tour = two_opt(tour, dist_matrix)
            cost = tour_cost(improved_tour, dist_matrix)
            all_tours.append((improved_tour, cost))

        # Pheromone evaporation
        pheromone *= (1 - evaporation_rate)

        # Best of this iteration
        best_iteration_tour, best_iteration_cost = min(all_tours, key=lambda x: x[1])
        for i in range(n_cities):
            a, b = best_iteration_tour[i], best_iteration_tour[(i + 1) % n_cities]
            pheromone[a][b] += 1.0 / best_iteration_cost
            pheromone[b][a] += 1.0 / best_iteration_cost

        pheromone = np.clip(pheromone, pheromone_min, pheromone_max)

        # Update best overall
        if best_iteration_cost < best_cost:
            best_tour = best_iteration_tour
            best_cost = best_iteration_cost

        convergence_curve.append(best_cost)

    return best_tour, best_cost, convergence_curve

# --- 4. Plot Convergence Curve ---
def plot_convergence(convergence_data, optimum_cost, title="ACO Convergence", log_scale=False):
    plt.figure(figsize=(10, 6))
    for run_idx, run_curve in enumerate(convergence_data):
        if log_scale:
            diff = [np.log10(c - optimum_cost + 1e-6) for c in run_curve]
            plt.plot(diff, label=f"Run {run_idx+1}")
        else:
            plt.plot(run_curve, label=f"Run {run_idx+1}")
    plt.title(title)
    plt.xlabel("Iteration")
    ylabel = "log10(Tour Cost - Optimum)" if log_scale else "Best Tour Cost"
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 5. Run 5 Times for Evaluation ---
if __name__ == "__main__":
    coords, dist_matrix = load_tsplib("att48.tsp")
    optimum = 33523
    results = []
    convergence_data = []

    for run in range(5):
        print(f"\n=== Run {run+1} ===")
        start = time.time()
        best_tour, best_cost, convergence_curve = ant_colony_optimization_2opt(dist_matrix)
        end = time.time()
        error = best_cost - optimum
        print(f"Run {run+1} — Cost: {best_cost:.2f} | Error: {error:.2f} | Time: {end - start:.2f}s")
        results.append(best_cost)
        convergence_data.append(convergence_curve)

    # --- Stats ---
    avg_cost = np.mean(results)
    std_dev = np.std(results)
    min_cost = np.min(results)
    print("\n=== Summary ===")
    print(f"Average cost: {avg_cost:.2f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print(f"Best found: {min_cost:.2f} vs Optimum: {optimum}")

    # --- Plot ---
    plot_convergence(convergence_data, optimum, log_scale=True)
