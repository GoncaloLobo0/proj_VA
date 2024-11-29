import numpy as np
from neural_net import MLP
from evolution import Population
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import copy

def get_and_load_params_test():
    test = True
    
    mlp1 = MLP(3, 7, 2)
    params1 = mlp1.get_parameters_numpy()

    mlp2 = MLP(3, 7, 2)
    params2 = mlp2.get_parameters_numpy()

    # The parameters should be different here
    test = test and not np.array_equal(params1, params2)

    mlp2.load_parameters_numpy(params1)
    params2 = mlp2.get_parameters_numpy()
    
    # The parameters should be the same here
    test = test and np.array_equal(params1, params2)

    return test

def evaluate_population_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=10, input_size=28*28, hidden_size=5, output_size=10, dataloader=train_loader)

    pop.evaluatePopulation()

    for i in pop.individuals:
        print(i.fitness)

def select_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=10, input_size=28*28, hidden_size=5, output_size=10, dataloader=train_loader)

    print(len(pop.individuals))

    pop.evaluatePopulation()
    pop.select_tournament(3, 6)

    print(len(pop.individuals))


def crossover_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=4, input_size=1, hidden_size=1, output_size=2, dataloader=train_loader)

    
    before1 = copy.deepcopy(pop.individuals[0].get_parameters_numpy())
    before2 = copy.deepcopy(pop.individuals[1].get_parameters_numpy())
    before3 = copy.deepcopy(pop.individuals[2].get_parameters_numpy())
    before4 = copy.deepcopy(pop.individuals[3].get_parameters_numpy())
    
    print("-------------------------------------------")
    pop.crossover_twopoint(0.5)
    
    after1 = pop.individuals[0].get_parameters_numpy()
    after2 = pop.individuals[1].get_parameters_numpy()
    after3 = pop.individuals[2].get_parameters_numpy()
    after4 = pop.individuals[3].get_parameters_numpy()

    print(np.array_equal(before1, after1))
    print(np.array_equal(before2, after2))
    print(np.array_equal(before3, after3))
    print(np.array_equal(before4, after4))

def simple_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=2, input_size=1, hidden_size=2, output_size=2, dataloader=train_loader)

    pop.crossover_twopoint(1)

def mutate_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=2, input_size=1, hidden_size=2, output_size=2, dataloader=train_loader)

    print(pop.individuals[0].get_parameters_numpy())

    pop.mutate_gaussian(0, 0.1, 0.7)

    print(pop.individuals[0].get_parameters_numpy())


def select_and_crossover_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=100, input_size=1, hidden_size=2, output_size=2, dataloader=train_loader)

    print(len(pop.individuals))
    pop.select_tournament(50, 10)

    print(len(pop.individuals))
    pop.crossover_twopoint_steady(20)

    print(len(pop.individuals))

def advance_gen_test():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    dataset_size = len(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=dataset_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = Population(pop_size=400, input_size=28*28, hidden_size=200, output_size=10, dataloader=train_loader)

    print(len(pop.individuals))
    print(pop.get_best_individual().fitness)
    
    for _ in range(1,6000):
        pop.advance_generation()
        print(pop.get_best_individual().fitness)
        
    print(len(pop.individuals))
    print(pop.get_best_individual().fitness)


advance_gen_test()