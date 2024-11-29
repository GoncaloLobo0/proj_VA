import random
import numpy as np
from neural_net import MLP
from deap.tools import selTournament, mutGaussian, cxTwoPoint, cxOnePoint

class Population():
    def __init__(self, pop_size, input_size, hidden_size, output_size, dataloader):
        self.individuals = [MLP(input_size, hidden_size, output_size) for _ in range(pop_size)]
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

        for individual in self.individuals:
            individual.fitness = 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pop_size = pop_size

        # Default hyperparameters
        self.tourn_size = int(pop_size * 0.2)
        self.tourn_k = int(pop_size * 0.15)

        self.mut_mu = 0
        self.mut_sigma = 0.01
        self.mut_indpb = 0.0005

        self.cxpb = 0.05

        # Starts by evaluating the first generation
        self.evaluatePopulation()

    # Updates the fitness of the population
    def evaluatePopulation(self):
        try:
            X, y = next(self.data_iter)

        except StopIteration:
            self.data_iter = iter(self.dataloader)
            X, y = next(self.data_iter)

        for i, individual in enumerate(self.individuals):
            self.individuals[i].fitness = 0
            self.individuals[i].fitness = individual.calculate_fitness(X, y)

    def select_tournament(self, k, tournsize):
        individuals = selTournament(self.individuals, k, tournsize, "fitness")

        self.individuals = individuals

    def mutate_gaussian(self, mu, sigma, indpb):
        for individual in self.individuals:
            ind_weights = individual.get_parameters_numpy()
            mutGaussian(ind_weights, mu, sigma, indpb)

            individual.load_parameters_numpy(ind_weights)

    # Generational replacement crossover with probability
    def crossover_twopoint_genrep(self, cxpb):
        for i in range(1, len(self.individuals), 2):
            if random.random() < cxpb:

                ind1 = self.individuals[i-1]
                ind2 = self.individuals[i]
                ind1_weights = self.individuals[i-1].get_parameters_numpy()
                ind2_weights = self.individuals[i].get_parameters_numpy()

                # Inplace crossover
                cxTwoPoint(ind1_weights, ind2_weights)

                ind1.load_parameters_numpy(ind1_weights)
                ind2.load_parameters_numpy(ind2_weights)

    # Steady state crossover (creates a number of offspring)
    def crossover_twopoint_steady(self, n_offspring):
        offspring_count = 0

        while offspring_count < n_offspring:
            ind1, ind2 = random.sample(self.individuals, 2)
            ind1_weigths, ind2_weights = ind1.get_parameters_numpy(), ind2.get_parameters_numpy()

            # Inplace crossover
            cxTwoPoint(ind1_weigths, ind2_weights)

            # Only one offspring added to the pop
            offspring1 = MLP(self.input_size, self.hidden_size, self.output_size)
            #offspring2 = MLP(self.input_size, self.hidden_size, self.output_size)

            offspring1.load_parameters_numpy(ind1_weigths)
            #offspring2.load_parameters_numpy(ind2_weights)

            self.individuals.append(offspring1)
            #self.individuals.append(offspring2)

            offspring_count += 1

    def advance_generation(self):
        self.select_tournament(self.pop_size, self.tourn_size)
        
        self.crossover_twopoint_genrep(self.cxpb)
        self.mutate_gaussian(self.mut_mu, self.mut_sigma, self.mut_indpb)

        self.evaluatePopulation()

    def get_best_individual(self):
        return max(self.individuals, key=lambda ind:ind.fitness)