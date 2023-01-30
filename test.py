import numpy as np


def random_weights(neural_network_structure, minimum=0, maximum=1):
    weights_list = []

    for i in range(len(neural_network_structure) - 1):
        weights = np.random.uniform(minimum, maximum,
                                    (neural_network_structure[i + 1], neural_network_structure[i] + 1))
        weights_list.append(weights)

    return weights_list


def f(x):
    y = x
    return y


def neuron_value(value_arr, weight_arr, act_func):
    x = np.dot(value_arr, np.transpose(weight_arr))
    res = act_func(x)
    return res


def neuron_value_list(input_data, weight_list, activation_functions_list):
    number_of_iterations = len(activation_functions_list)

    value_list = []

    for i in range(number_of_iterations):
        print(i)
        if i == 0:
            value = np.append(activation_functions_list[i](input_data), 1)
            value_list.append(value)
        elif i == number_of_iterations - 1:
            value = neuron_value(value, weight_list[i - 1], activation_functions_list[i])
            value_list.append(value)
        else:
            value = np.append(neuron_value(value, weight_list[i - 1], activation_functions_list[i]), 1)
            value_list.append(value)

    return value_list


c = np.array([-2, -2, 2])
if c.any() < 0:
    c = 0

print(c)
