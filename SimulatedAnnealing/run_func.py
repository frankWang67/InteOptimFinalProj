from SimulatedAnnealing.SA import SA_Func_Optim
import numpy as np
import matplotlib.pyplot as plt

config_file = 'SimulatedAnnealing/config/params.yaml'

def func(x, n):
    # Rastrigin Function
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

solver = SA_Func_Optim(config_file, func)
solution, cost, time_consumed, iter, all_solutions, all_costs = solver.run(debug=True)

print('Best solution:', solution)
print('Best cost:', cost)
print('Time consumed:', time_consumed)
print('Iterations:', iter)

plt.figure(1)
plt.plot(all_costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over iterations')
plt.show()