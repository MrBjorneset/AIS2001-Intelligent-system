import random
import matplotlib.pyplot as plt


targetString = "EMIL_BJORNESET*566119"
validCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789*"

populationSize = 400
mutationRate = 0.01
crossoverRate = 1
maxGenerations = 100000

def generate_random_string(length):
    return ''.join(random.choice(validCharacters) for _ in range(length))

def calculate_fitness(candidate):
    return sum(1 for a, b in zip(candidate, targetString) if a == b)

def crossover(parent1, parent2):
    crossoverPoint = random.randint(0, len(parent1) - 1)
    child = parent1[:crossoverPoint] + parent2[crossoverPoint:]
    return child

def mutate(child, mutationRate):
    mutatedChild = ''.join(
        c if random.random() > mutationRate else random.choice(validCharacters)
        for c in child
    )
    return mutatedChild

def genetic_algorithm(targetString, populationSize, mutationRate, crossoverRate, maxGenerations):
    population = [generate_random_string(len(targetString)) for _ in range(populationSize)]
    bestFitnesses = []
    global fitnesScores
    for generation in range(maxGenerations):
        fitnesScores = [calculate_fitness(candidate) for candidate in population]

        if max(fitnesScores) == len(targetString):
            print(f"Found solution in generation {generation}: {population[fitnesScores.index(max(fitnesScores))]}")
            break
        bestFixIndex = fitnesScores.index(max(fitnesScores))
        bestFitCandidate = population[bestFixIndex]
        print(f"Generation {generation}: Best Fit - {bestFitCandidate}, Fitness - {fitnesScores[bestFixIndex]}")

        bestFitnesses.append(max(fitnesScores))
        parents = random.choices(population, weights=fitnesScores, k=2)
        if random.random() < crossoverRate:
            child = crossover(parents[0], parents[1])
            child = mutate(child, mutationRate)
            min_fitness_index = fitnesScores.index(min(fitnesScores))
            population[min_fitness_index] = child

    else:
        print("Maximum generations reached. Solution not found.")
    return bestFitnesses

bestFitnesses = genetic_algorithm(targetString,populationSize, mutationRate, crossoverRate, maxGenerations)

# Plot the best fitness values
plt.plot(bestFitnesses)
plt.title('Best Fitness Values Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()