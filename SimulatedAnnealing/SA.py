import yaml
import numpy as np
import time
import matplotlib.pyplot as plt

# Abstract base class for Simulated Annealing
class SA_Base:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config['base']
        self.temperature = self.params['initial_temperature']
        self.alpha = self.params['cooling_rate']
        self.max_iter = self.params['max_iterations']
        self.early_stop_iter = self.params['early_stop_iterations']
        
    def initialize_solution(self):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def evaluate(self, solution):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def neighbor(self, solution):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def run(self, debug=False, show_tsp=False):
        current_solution = self.initialize_solution()
        best_solution = current_solution
        current_cost = self.evaluate(current_solution)
        best_cost = current_cost

        no_update_iters = 0
        start_time = time.time()

        if debug:
            all_solutions = [current_solution]
            all_costs = [current_cost]
        
        for i in range(self.max_iter):
            if no_update_iters >= self.early_stop_iter:
                break

            new_solution = self.neighbor(current_solution)
            new_cost = self.evaluate(new_solution)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / self.temperature):
                current_solution = new_solution
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
                    no_update_iters = 0
                    if debug:
                        all_solutions.append(best_solution)
            else:
                no_update_iters += 1

            if debug:
                all_costs.append(best_cost)
                if show_tsp and isinstance(self, SA_TSP):
                    self.visualize(best_solution, False)
            
            if self.temperature > 1e-3:
                self.temperature *= self.alpha

        time_consumed = time.time() - start_time
        
        if debug:
            return best_solution, best_cost, time_consumed, i+1, all_solutions, all_costs
        else:
            return best_solution, best_cost, time_consumed, i+1

# Class for optimizing functions
class SA_Func_Optim(SA_Base):
    def __init__(self, config_file, func):
        super().__init__(config_file)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config['func_optim']
        self.dim = self.params['dimension']
        self.lower_bound = self.params['lower_bound']
        self.upper_bound = self.params['upper_bound']
        self.step_size = self.params['step_size']

        if not callable(func):
            raise ValueError('The argument `func` must be a function.')
        if func.__code__.co_argcount != 2:
            raise ValueError('The function must have 2 arguments, which are the input and its dimension, respectively.')
        self.func = func

    def initialize_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
    
    def evaluate(self, solution):
        return self.func(solution, self.dim)
    
    def neighbor(self, solution):
        neighbor = solution + np.random.uniform(-self.step_size, self.step_size, size=self.dim)
        return np.clip(neighbor, self.lower_bound, self.upper_bound)

# Class for solving TSP problems
class SA_TSP(SA_Base):
    def __init__(self, config_file):
        super().__init__(config_file)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config['TSP']
        tsp_file = self.params['tsp_file']
        self.load_tsp(tsp_file)
    
    def load_tsp(self, tsp_file):
        with open(tsp_file, 'r') as file:
            lines = file.readlines()
        self.num_cities = int(lines[0])
        self.cities = [tuple(map(int, line.split())) for line in lines[1:]]
        self.dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                city1 = self.cities[i]
                city2 = self.cities[j]
                self.dist_matrix[i, j] = np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
                self.dist_matrix[j, i] = self.dist_matrix[i, j]
    
    def initialize_solution(self):
        return np.random.permutation(self.num_cities)
    
    def evaluate(self, solution):
        total_distance = 0
        for i in range(self.num_cities):
            inext = (i + 1) % self.num_cities
            total_distance += self.dist_matrix[solution[i], solution[inext]]
        return total_distance
    
    def neighbor(self, solution):
        a, b = np.random.choice(self.num_cities, size=2, replace=False)
        new_solution = solution.copy()
        new_solution[a], new_solution[b] = new_solution[b], new_solution[a]
        return new_solution
    
    def visualize(self, solution, save_fig=True, save_file='tsp_solution.png', wait=True):
        plt.figure(1)
        plt.clf()
        for i in range(self.num_cities):
            inext = (i + 1) % self.num_cities
            city1 = self.cities[solution[i]]
            city2 = self.cities[solution[inext]]
            plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-')
        for i in range(self.num_cities):
            city = self.cities[i]
            plt.plot(city[0], city[1], 'ro')
        plt.title('TSP solution')
        if save_fig:
            plt.savefig(save_file)
        else:
            if wait:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.05)
