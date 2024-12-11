from SimulatedAnnealing.SA import SA_TSP, SA_Func_Optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import os
import time
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Run the experiments for intelligent optimization algorithms.')
parser.add_argument('--prob', type=str, required=True, help='The problem to solve, chosen from \"func\" and \"TSP\".')
parser.add_argument('--alg', type=str, required=True, help='The algorithm to run, chosen from \"SA\" and \"GA\".')
parser.add_argument('--config', type=str, default='SimulatedAnnealing/config/params.yaml', help='The path to the configuration file.')

args = parser.parse_args()
if not args.prob in ['func', 'TSP']:
    raise ValueError('The problem must be chosen from \"func\" and \"TSP\".')
if not args.alg in ['SA', 'GA']:
    raise ValueError('The algorithm must be chosen from \"SA\" and \"GA\".')
if args.alg == 'GA':
    args.config = 'GeneticAlgorithm/config/params.yaml'

exp_dir = 'exp/' + args.prob + '/' + args.alg + '/' + time.strftime('%Y-%m-%d_%H-%M-%S')
config_file = args.config

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
os.system('cp ' + config_file + ' ' + exp_dir)

# run the experiments
EXP_RUNS = 20
overall_solutions = []
overall_costs = []
overall_times = []
overall_iters = []
overall_all_solutions = []
overall_all_costs = []

print('Running the experiments...')
if args.prob == 'func':
    def func(x, n):
        # Rastrigin Function
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    if args.alg == 'SA':
        solver = SA_Func_Optim(config_file, func)
    elif args.alg == 'GA':
        pass
        # TODO
        # solver = GA_Func_Optim(config_file, func)

elif args.prob == 'TSP':
    if args.alg == 'SA':
        solver = SA_TSP(config_file)
    elif args.alg == 'GA':
        pass
        # TODO
        # solver = GA_TSP(config_file)

for i in tqdm.tqdm(range(EXP_RUNS)):
    solution, cost, time_consumed, iter, all_solutions, all_costs = solver.run(debug=True)

    overall_solutions.append(solution)
    overall_costs.append(cost)
    overall_times.append(time_consumed)
    overall_iters.append(iter)
    overall_all_solutions.append(all_solutions)
    overall_all_costs.append(all_costs)
print('Done!')

# calculate the statistics
print('Calculating the statistics...')
overall_costs = np.array(overall_costs)
overall_times = np.array(overall_times)
overall_iters = np.array(overall_iters)

best_idx = np.argmin(overall_costs)
worst_idx = np.argmax(overall_costs)
average_cost = np.mean(overall_costs)
variance_cost = np.var(overall_costs)
average_time = np.mean(overall_times)
average_iter = np.mean(overall_iters)
print('Done!')

# save the log file
print('Saving the log file...')
log_file = exp_dir + '/log.txt'
with open(log_file, 'w') as file:
    file.write('Best cost: {}\n'.format(overall_costs[best_idx]))
    file.write('Worst cost: {}\n'.format(overall_costs[worst_idx]))
    file.write('Average cost: {}\n'.format(average_cost))
    file.write('Variance of cost: {}\n'.format(variance_cost))
    file.write('Average time consumed: {}\n'.format(average_time))
    file.write('Average iterations: {}\n'.format(average_iter))
print('Done!')

# save the best solution figure
if args.prob == 'TSP':
    print('Saving the best solution figure...')
    best_solution = overall_solutions[best_idx]
    fig_file = exp_dir + '/best_solution.png'
    solver.visualize(best_solution, True, fig_file)
    print('Done!')

# save the cost curve of the best run
print('Saving the cost curve of the best run...')
best_all_costs = overall_all_costs[best_idx]
plt.figure(1)
plt.clf()
plt.plot(best_all_costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over iterations')
plt.savefig(exp_dir + '/cost_curve.png')
print('Done!')

# save the video of the best run
if args.prob == 'TSP' or (args.prob == 'func' and solver.dim == 2):
    print('Saving the video of the best run...')
    best_all_solutions = overall_all_solutions[best_idx]
    fig = plt.figure(2)
    plt.clf()
    if args.prob == 'TSP':
        for i in range(solver.num_cities):
            city = solver.cities[i]
            plt.plot(city[0], city[1], 'ro')
        lines = []
        for i in range(solver.num_cities):
            inext = (i + 1) % solver.num_cities
            city1 = solver.cities[best_all_solutions[0][i]]
            city2 = solver.cities[best_all_solutions[0][inext]]
            line, = plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-')
            lines.append(line)

        def update(frame):
            for i in range(solver.num_cities):
                inext = (i + 1) % solver.num_cities
                city1 = solver.cities[best_all_solutions[frame][i]]
                city2 = solver.cities[best_all_solutions[frame][inext]]
                lines[i].set_data([city1[0], city2[0]], [city1[1], city2[1]])
            return lines
    elif args.prob == 'func':
        steps = 100
        x = np.linspace(solver.lower_bound, solver.upper_bound, steps)
        y = np.linspace(solver.lower_bound, solver.upper_bound, steps)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(steps):
            for j in range(steps):
                Z[i, j] = solver.func(np.array([X[i, j], Y[i, j]]), solver.dim)
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Function value (z)')
        pt, = plt.plot(best_all_solutions[0][0], best_all_solutions[0][0], 'ro')

        def update(frame):
            pt.set_data(best_all_solutions[frame][0], best_all_solutions[frame][1])
            return pt,

    ani = FuncAnimation(fig, update, frames=len(best_all_solutions), blit=True)
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(exp_dir + '/best_solution_found.mp4', writer=writer)
    print('Done!')

# All done
print(f'All done! The results are saved in {exp_dir}.')
