import numpy as np


def backpropagation(weight_list, value_list, expected_results):
    education_speed = 0.000001

    weight_list_reversed = weight_list[::-1]
    value_list_reversed = value_list[::-1]

    delta_w_list = []

    for i in range(len(value_list_reversed)):
        if i == 0:
            delta = expected_results - value_list_reversed[i]
        else:
            gradient = np.matmul(np.transpose(np.array([delta])), np.array([value_list_reversed[i]]))
            delta_w = education_speed * gradient
            delta_w_list.append(delta_w)
            delta = np.sum(delta * np.transpose(weight_list_reversed[i - 1]), axis=1)[:-1]

    return delta_w_list



weights1 = np.random.uniform(0, 1, (4, 3))
weights2 = np.random.uniform(0, 1, (2, 5))
w = [weights1, weights2]

val1 = np.array([23, 56, 1])
val2 = np.array([12, 23, 34, 56, 1])
val3 = np.array([45, 67])
v = [val1, val2, val3]

res = np.array([56,78])

print(backpropagation(w,v,res))