import numpy as np
from neural_net import MLP

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


print(get_and_load_params_test())