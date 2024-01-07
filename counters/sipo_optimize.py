import random
from deap import base, creator, tools, algorithms
import numpy as np
import math
from sipo_model import three_bit_sipo
from params import param_ranges
from scipy.integrate import odeint
from sipo_test_and_draw import calc_input, calc_clear, t_end, N, n, KM
from tqdm import tqdm
from scipy.signal import find_peaks


def calculate_avg_frequency(data):
    # Find the indices of the peaks in the data
    peaks, _ = find_peaks(data)

    if len(peaks) == 0:
        return None

    freq = 1 / (t_end / len(peaks))

    return freq


def calculate_avg_diff(data):
    peaks, _ = find_peaks(data)
    lows, _ = find_peaks(-data)

    peak_values = data[peaks]
    low_values = data[lows]

    if len(peaks) == 0 or len(lows) == 0:
        return None

    return (np.mean(peak_values) - np.mean(low_values)) / 200


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

    freqs = [
        calculate_avg_frequency(np.squeeze(Q1)),
        calculate_avg_frequency(np.squeeze(Q2)),
        calculate_avg_frequency(np.squeeze(Q3)),
    ]
    ampls = [
        calculate_avg_diff(np.squeeze(Q1)),
        calculate_avg_diff(np.squeeze(Q2)),
        calculate_avg_diff(np.squeeze(Q3)),
    ]

    if None in freqs or None in ampls:
        return (10000,)

    freq_score = np.sum(freqs)
    ampl_score = np.sum([x for x in ampls])

    # score = (1 - freq_score) + ampl_score
    score = (1 - freq_score) * ampl_score

    print(freq_score, ampl_score, score)
    return (-score,)


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
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)


# Parameter for the GA
population = toolbox.population(n=100)
NGEN = 60
CXPB, MUTPB = 0.1, 0.25

best_overall = None
best_overall_score = 10000

# Run the GA
for gen in tqdm(range(NGEN), desc="Evolving Generations"):
    offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)

    # Adding tqdm to the inner loop
    fits = [
        toolbox.evaluate(ind) for ind in tqdm(offspring, desc=f"Evaluating Gen {gen}")
    ]

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

    top_individual = tools.selBest(population, k=1)[0]
    top_individual_score = evaluate(top_individual)[0]

    print(top_individual_score)

    with open("best_overall.txt", "a") as f:
        f.write(f"best overall from gen {gen} with score {top_individual_score}")
        f.write("\n")
        f.write("\n".join([str(x) for x in top_individual]))
        f.write("\n")
        f.write("\n")

    if best_overall is None or top_individual_score < best_overall_score:
        best_overall = top_individual
        best_overall_score = top_individual_score


print()
print()
print("Top overall individual:", best_overall_score)
print("\n".join([str(x) for x in best_overall]))
print()
