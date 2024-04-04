import numpy as np

# Define parameters
num_products = 5
max_production_capacity = 100
selling_price = np.array([10, 20, 15, 25, 30])
man_hour_consumption = np.array([2, 4, 3, 5, 6])
demand = np.array([50, 30, 40, 20, 10])

# Genetic Algorithm parameters
population_size = 50
mutation_rate = 0.1
generations = 100

# Define the initial population
population = np.random.randint(0, max_production_capacity + 1, size=(population_size, num_products))

# Define the fitness function
def calculate_fitness(individual):
    total_profit = np.sum((selling_price - man_hour_consumption) * np.minimum(individual, demand))
    return total_profit

# Define the selection function (tournament selection)
def tournament_selection(population, fitness_values):
    selected_indices = np.random.choice(len(population), size=2, replace=False)
    selected_fitness = fitness_values[selected_indices]
    selected_individuals = population[selected_indices]
    return selected_individuals[np.argmax(selected_fitness)]

# Define the crossover function (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_products)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Define the mutation function
def mutate(individual):
    mutation_indices = np.random.choice(num_products, size=int(mutation_rate * num_products), replace=False)
    individual[mutation_indices] = np.random.randint(0, max_production_capacity + 1, size=len(mutation_indices))
    return individual

# Main Genetic Algorithm loop
for generation in range(generations):
    # Calculate fitness for each individual in the population
    fitness_values = np.array([calculate_fitness(individual) for individual in population])

    # Select parents using tournament selection
    parents = np.array([tournament_selection(population, fitness_values) for _ in range(population_size)])

    # Create the next generation using crossover and mutation
    new_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    population = np.array(new_population)

# Find the best individual in the final population
best_individual = population[np.argmax([calculate_fitness(individual) for individual in population])]

print("Best production plan:", best_individual)
print("Maximized profit:", calculate_fitness(best_individual))
