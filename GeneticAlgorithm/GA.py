import yaml
import numpy as np
import time
import matplotlib.pyplot as plt

# Abstract base class for Genetic Algorithm
class GA_Base:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config['base']
        self.population_size = self.params['population_size']
        self.max_iter = self.params['max_iterations']
        self.mutation_rate = self.params['mutation_rate']
        self.crossover_rate = self.params['crossover_rate']
        self.elite_size = self.params['elite_size']
        self.early_stop_iter = self.params['early_stop_iterations']
    
    def initialize_population(self):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def evaluate(self, individual):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def mutate(self, individual):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def crossover(self, parent1, parent2):
        raise NotImplementedError('This method must be implemented by the subclass')
    
    def select(self, population, fitness):
        # Roulette Wheel Selection
        fitness_sum = np.sum(fitness)
        probabilities = fitness / fitness_sum
        return population[np.random.choice(len(population), p=probabilities)]
    
    def run(self, debug=False, show_tsp=False):
        population = self.initialize_population()
        fitness = np.array([self.evaluate(ind) for ind in population])
        best_idx = np.argmax(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        no_update_iters = 0
        start_time = time.time()
        
        if debug:
            all_costs = [-best_fitness]
            all_solutions = [best_individual]
        
        for i in range(self.max_iter):
            if no_update_iters >= self.early_stop_iter:
                break
            
            new_population = []
            if self.elite_size > 0.0:
                elite_count = int(self.elite_size * self.population_size)
                elite_indices = np.argsort(fitness)[-elite_count:]
                elites = [population[j] for j in elite_indices]
                new_population.extend(elites)
            
            while len(new_population) < self.population_size:
                parent1 = self.select(population, fitness)
                parent2 = self.select(population, fitness)
                if np.random.rand() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                new_population.append(self.mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self.mutate(child2))
            
            population = new_population
            fitness = np.array([self.evaluate(ind) for ind in population])
            max_fitness = fitness.max()
            if max_fitness > best_fitness:
                best_individual = population[np.argmax(fitness)]
                best_fitness = max_fitness
                no_update_iters = 0
                if debug:
                    all_solutions.append(best_individual)
            else:
                no_update_iters += 1
            
            if debug:
                all_costs.append(-best_fitness)
                if show_tsp and isinstance(self, GA_TSP):
                    self.visualize(best_individual, False)
        
        time_consumed = time.time() - start_time

        if debug:
            cost = best_fitness * -1
            return best_individual, cost, time_consumed, i + 1, all_solutions, all_costs
        else:
            return best_individual, best_fitness, i + 1

# Class for optimizing functions
class GA_Func_Optim(GA_Base):
    def __init__(self, config_file, func):
        super().__init__(config_file)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config['func_optim']
        self.dim = self.params['dimension']
        self.lower_bound = self.params['lower_bound']
        self.upper_bound = self.params['upper_bound']
        self.step_size = self.params['step_size']
        self.func = func

    def initialize_population(self):
        return [
            np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for _ in range(self.population_size)
        ]
    
    def evaluate(self, individual):
        return -self.func(individual, self.dim)  # Negative for maximizing the fitness
    
    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = individual + np.random.uniform(-self.step_size, self.step_size, self.dim)
            return np.clip(mutation, self.lower_bound, self.upper_bound)
        return individual
    
    def crossover(self, parent1, parent2):
        if self.dim == 1:
            child1, child2 =  parent1, parent2
        elif self.dim == 2:
            child1 = np.array([parent1[0], parent2[1]])
            child2 = np.array([parent2[0], parent1[1]])
        else:
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

# Class for solving TSP problems
class GA_TSP(GA_Base):
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
    
    def initialize_population(self):
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]
    
    def evaluate(self, individual):
        total_distance = sum(
            self.dist_matrix[individual[i], individual[(i + 1) % self.num_cities]]
            for i in range(self.num_cities)
        )
        return -total_distance  # Negative for maximizing the fitness
    
    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            a, b = np.random.choice(self.num_cities, size=2, replace=False)
            mutated = individual.copy()
            mutated[a], mutated[b] = mutated[b], mutated[a]
            return mutated
        return individual
    
    def crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(range(self.num_cities), 2, replace=False))
        child1 = np.zeros(self.num_cities, dtype=int) - 1
        child1[start:end] = parent1[start:end]
        for city in parent2:
            if city not in child1:
                idx = np.where(child1 == -1)[0][0]
                child1[idx] = city
        child2 = np.zeros(self.num_cities, dtype=int) - 1
        child2[start:end] = parent2[start:end]
        for city in parent1:
            if city not in child2:
                idx = np.where(child2 == -1)[0][0]
                child2[idx] = city
        return child1, child2
    
    def visualize(self, individual, save_fig=True, save_file='tsp_solution.png', wait=True):
        plt.figure(1)
        plt.clf()
        for i in range(self.num_cities):
            inext = (i + 1) % self.num_cities
            city1 = self.cities[individual[i]]
            city2 = self.cities[individual[inext]]
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