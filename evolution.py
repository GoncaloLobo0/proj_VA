import copy
import random
import numpy as np
from neural_net import MLP
from deap.tools import selTournament, mutGaussian, cxTwoPoint
from concurrent.futures import ThreadPoolExecutor, as_completed

class Population():
    def __init__(self, pop_size, input_size, hidden_size, output_size, dataloader,
                 tourn_size=None, mut_mu=0, mut_sigma=0.1, mut_indpb=None, cxpb=0.6, evaluate_at_start=True):
        self.individuals = [MLP(input_size, hidden_size, output_size) for _ in range(pop_size)]
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.data = next(self.data_iter)

        for individual in self.individuals:
            individual.fitness = 0
            individual.accuracy = 0
            
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mut_mu = mut_mu
        self.mut_sigma = mut_sigma
        self.mut_indpb = mut_indpb
        self.cxpb = cxpb
        
        if tourn_size is None:
            self.tourn_size = int(pop_size*0.1)
            
        if mut_indpb is None:
            self.mut_indpb = ((input_size*hidden_size)+(hidden_size*output_size))*1e-7

        if evaluate_at_start:
            # Starts by evaluating the first generation
            self.evaluatePopulation()

    # Updates the fitness of the population
    def evaluatePopulation(self):
        """
        try:
            X, y = next(self.data_iter)

        except StopIteration:
            self.data_iter = iter(self.dataloader)
            X, y = next(self.data_iter)
            """
        X, y = self.data

        for i, individual in enumerate(self.individuals):
            self.individuals[i].fitness, self.individuals[i].accuracy = individual.calculate_fitness(X, y)
    
    # Chatgpt
    def evaluatePopulationMulti(self):
        X, y = self.data

        # Define a function to calculate fitness for an individual
        def calculate_individual_fitness(individual):
            return individual.calculate_fitness(X, y)

        with ThreadPoolExecutor() as executor:
            # Submit tasks for each individual
            future_to_individual = {executor.submit(calculate_individual_fitness, ind): i for i, ind in enumerate(self.individuals)}

            for future in as_completed(future_to_individual):
                i = future_to_individual[future]
                fitness, accuracy = future.result()
                self.individuals[i].fitness = fitness
                self.individuals[i].accuracy = accuracy

    def select_tournament(self, k, tournsize):
        individuals = selTournament(self.individuals, k, tournsize, "fitness")

        self.individuals = individuals

    def mutate_gaussian(self, mu, sigma, indpb):
        for individual in self.individuals:
            ind_weights = individual.get_parameters_numpy()
            mutGaussian(ind_weights, mu, sigma, indpb)

            individual.load_parameters_numpy(ind_weights)
            
    def mutate_gaussianBetter(self, mu, sigma, indpb):
        for individual in self.individuals:
            # Get the individual’s weights (consider caching it directly if possible)
            ind_weights = individual.get_parameters_numpy()

            # Apply mutation directly in-place to avoid unnecessary copying
            mask = np.random.rand(*ind_weights.shape) < indpb
            ind_weights[mask] += np.random.normal(mu, sigma, size=mask.sum())

            # Directly set the mutated weights back to the individual
            individual.load_parameters_numpy(ind_weights)

    # Generational replacement crossover with probability
    def crossover_twopoint_genrep(self, cxpb):
        indices = list(range(len(self.individuals)))
        random.shuffle(indices)
        
        for i in range(0, len(indices) - 1, 2):  # Step by 2 for pairs
            if random.random() < cxpb:
                ind1 = self.individuals[indices[i]]
                ind2 = self.individuals[indices[i + 1]]

                ind1_weights = ind1.get_parameters_numpy()
                ind2_weights = ind2.get_parameters_numpy()

                # Perform in-place two-point crossover
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
        elite  = self.get_elite(3)
        #print("tourn")
        self.select_tournament(self.pop_size - len(elite), self.tourn_size)
        #print("cx")
        
        
        self.crossover_twopoint_genrep(self.cxpb)
        #print("mut")
        self.mutate_gaussianBetter(self.mut_mu, self.mut_sigma, self.mut_indpb)
        
        # Remove random individuals and replace with the elite
        indices_to_remove = random.sample(range(len(self.individuals)), len(elite))
        for index in sorted(indices_to_remove, reverse=True):
            del self.individuals[index]
        self.individuals.extend(elite)

        #print("evaluate")
        self.evaluatePopulationMulti()

    def get_best_individual(self):
        return max(self.individuals, key=lambda ind:ind.fitness)
    
    def get_elite(self, n):
        
        elite = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)[:n]
        elite_copies = [copy.deepcopy(ind) for ind in elite]
        return elite_copies