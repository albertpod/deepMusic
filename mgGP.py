"""
Genetic component of the program. Supports mutation, crossover and elitism, generates initial random population.
"""
import random

import numpy as np
import fitness as fit
from musicGeneration import data
from musicGraph import *
segment_size = 3
nb_node = 20

bar = data[:][0]
beat = data[:][1]
x = data[:][2]
y = data[:][3]
z = data[:][4]


def mutate(individual, probability=0.05):
    # Probabilty could be changed, or could be applied to 5% of the population, instead of 5% of each values.
    for _ in individual:
        if random.uniform(0.0, 1.0) < probability:
            fi_gene = random.randint(1, len(individual) - 1)
            individual[fi_gene] = random.randint(0, int(len(individual) / segment_size))
    return individual


def crossover(ind_f, ind_s, min_size=12):
    # Precautions regarding this functions need not be taken, for the function array_to_graph makes sure it always works
    child = []
    xover_f, xover_s = 0, len(ind_s)
    while xover_f + (len(ind_s) - xover_s) < min_size:
        xover_f = random.randint(1, len(ind_f))
        xover_f_seg = xover_f % segment_size

        xover_s = random.randint(1, len(ind_s))
        xover_s_seg = xover_s % segment_size

        xover_s += (xover_f_seg - xover_s_seg)
        if xover_s >= len(ind_s):
            xover_s -= segment_size

    for i in range(0, xover_f): child.append(ind_f[i])

    for i in range(xover_s, len(ind_s)): child.append(ind_s[i])

    return child


def create_population(amount):
    dump = []
    for k in range(amount):
        dump.append(MusicGraph(inputs={"X": x, "Y": y, "Z": z, "beat": beat, "bar": bar},
                outputs=["output1", "output2", "output3"],
                internal_nodes_n=nb_node, connect=True))
    return dump


def elitism(ancestors, f_pop, proportion=0.2):
    """f_pop = []
    for individual in ancestors:
        f_pop.append(fitness(individual))"""
    elected = [ANCESTORS for (F_POP, ANCESTORS) in sorted(zip(f_pop, ancestors), key=lambda x: x[0], reverse=True)]
    return elected[:int(len(elected)*proportion)], sorted(f_pop, reverse=True)[:int(len(elected)*proportion)]


def get_statistics(population):
    best, best_f, worst_f, mean_f, best_size = None, 0, 100, 0, 0
    for individual in population:
        f_ind = [0] # fitness(individual) TODO: use fitness
        if f_ind > best_f:
            best = individual
            best_f = f_ind
            best_size = len(f_ind)  # Im not sure about this
        if f_ind < worst_f:
            worst_f = f_ind
        mean_f += f_ind
    mean_f = mean_f / len(population)
    print(best, best_f, worst_f, mean_f, best_size)


def evolve(population, f_pop, generations=100):
    for i in range(generations):
        # elitism
        new_population, new_f_pop = elitism(population, f_pop)
        tmp = new_population[:]
        for k in range(len(tmp), 100):
            parents = random.sample(tmp, 2)
            # crossover
            new_population.append(crossover(parents[0], parents[1]))
            # mutation
            new_population[k] = mutate(new_population[k])
        population = new_population[:]
    return population


#best = elitism(create_population(100))
