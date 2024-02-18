import random


target_string = "EMIL_BJORNESET*566119"
valid_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789*"

Population_Size = 100
Mutation_Rate = 0.02
Crossover_Rate = 0.8
Max_Generations = 100000

def generate_random_string(length):
    return ''.join(random.choice(valid_characters) for _ in range(length))

def calculate_fitness(candidate):
    return sum(1 for a, b in zip(candidate, target_string) if a == b)

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child, mutation_rate):
    mutated_child = ''.join(
        c if random.random() > mutation_rate else random.choice(valid_characters)
        for c in child
    )
    return mutated_child

def genetic_algorithm(target_string, population_size, mutation_rate, crossover_rate, max_generations):
    population = [generate_random_string(len(target_string)) for _ in range(population_size)]
    global fitness_scores
    for generation in range(max_generations):
        fitness_scores = [calculate_fitness(candidate) for candidate in population]

        if max(fitness_scores) == len(target_string):
            print(f"Found solution in generation {generation}: {population[fitness_scores.index(max(fitness_scores))]}")
            break
        best_fit_index = fitness_scores.index(max(fitness_scores))
        best_fit_candidate = population[best_fit_index]
        print(f"Generation {generation}: Best Fit - {best_fit_candidate}, Fitness - {fitness_scores[best_fit_index]}")

        parents = random.choices(population, weights=fitness_scores, k=2)
        if random.random() < crossover_rate:
            child = crossover(parents[0], parents[1])
            child = mutate(child, mutation_rate)
            min_fitness_index = fitness_scores.index(min(fitness_scores))
            population[min_fitness_index] = child

    else:
        print("Maximum generations reached. Solution not found.")


genetic_algorithm(target_string,Population_Size, Mutation_Rate, Crossover_Rate, Max_Generations)
