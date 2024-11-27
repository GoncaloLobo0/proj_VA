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

            individual.fitness = 0

    def crossover_twopoint(self, cxpb):
        for i in range(1, len(self.individuals), 2):
            if random.random() < cxpb:

                ind1 = self.individuals[i-1]
                ind2 = self.individuals[i]
                ind1_weights = self.individuals[i-1].get_parameters_numpy()
                ind2_weights = self.individuals[i].get_parameters_numpy()

                cxTwoPoint(ind1_weights, ind2_weights)
                
                print(ind2_weights)

                ind1.load_parameters_numpy(ind1_weights)
                ind2.load_parameters_numpy(ind2_weights)

                self.individuals[i - 1].fitness = 0
                self.individuals[i].fitness = 0