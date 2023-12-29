import random
from deap import base, creator, tools, algorithms
import numpy as np
import math
from sipo_model import three_bit_sipo
from params import param_ranges
from scipy.integrate import odeint
from sipo_test_and_draw import calc_input, calc_clear, t_end, N, n, KM


# Define the objective function
def evaluate(individual):
    # Unpack the parameters from the individual

    alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd = individual
    params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n, delta1, KM)

    Y0 = np.array([0] * 18)
    T = np.linspace(0, t_end, N)

    # Run the model simulation
    Y = odeint(three_bit_sipo, Y0, T, args=(params_ff, calc_input, calc_clear))
    Y_reshaped = np.split(Y, Y.shape[1], 1)

    # Extract Q1, Q2, Q3
    Q1 = Y_reshaped[2]
    Q2 = Y_reshaped[6]
    Q3 = Y_reshaped[10]

    # Calculate the objective: maximizing the range (max - min)
    range_Q1 = np.max(Q1) - np.min(Q1)
    range_Q2 = np.max(Q2) - np.min(Q2)
    range_Q3 = np.max(Q3) - np.min(Q3)

    # Since we are using a GA that minimizes the objective, return the negative sum of ranges
    score = -1.0 * (range_Q1 + range_Q2 + range_Q3)
    if math.isnan(score):
        return (10000000000,)
    print(score)
    return (score,)


# Define the boundaries of the parameters
bounds = [
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # alpha1
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # alpha2
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # alpha3
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # alpha4
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # delta1
    (
        param_ranges["protein_production"]["min"],
        param_ranges["protein_production"]["max"],
    ),  # delta2
    (
        param_ranges["Kd"]["min"],
        param_ranges["Kd"]["max"],
    ),  # Kd
]

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the individual and population
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    lambda: [random.uniform(*bounds[i]) for i in range(len(bounds))],
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=16, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=20)

# Parameter for the GA
population = toolbox.population(n=1000)
NGEN = 30
CXPB, MUTPB = 0.1, 0.25

# Run the GA
for gen in range(NGEN):
    print(gen)
    offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

top_individual = tools.selBest(population, k=1)[0]
print("Best Individual: ", top_individual)

print("\n".join([str(x) for x in top_individual]))
