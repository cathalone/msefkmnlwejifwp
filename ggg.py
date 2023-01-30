import numpy as np


def neuron(value_arr, weight_arr):
    res = np.dot(value_arr, np.transpose(weight_arr))
    return res


def backpropogation(weight1, real_res, neuron_res, val):
    weight1_6 = np.transpose(weight1)

    delta_res_xy = (real_res - neuron_res) * (1 - neuron_res)
    print(delta_res_xy)

    grad_w_1_6 = np.matmul(np.transpose(np.array([delta_res_xy])), np.array([val]))

    delta_w_1_6 = 0.1 * grad_w_1_6

    return delta_w_1_6


weight1 = np.array([[1, 2, 3], [4, 5, 6]])
real_res = np.array([34, 67])
val = np.array([22, 67, 1])

neuron_res = neuron(val, weight1)
print(neuron_res)

print(backpropogation(weight1, real_res, neuron_res, val))

print(neuron_res, real_res)
