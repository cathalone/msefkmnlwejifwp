import numpy as np
import pygame as pg


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



def f(x):
    y = x
    return y


def error_func(obtained, expected):
    res = 0
    for i in range(len(obtained)):
        res += (obtained[i] - expected[i]) ** 2
        res /= len(obtained)
        return res


def backpropagation_linear(weight_list, value_list, expected_results):
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


sc = pg.display.set_mode((600, 600))

clock = pg.time.Clock()

col0 = (0, 0, 255)

col1 = (255, 0, 0)
col2 = (0, 255, 0)

fps = 10000

c = 1
press = []

weights1 = np.random.uniform(0, 1, (2, 3))
weights2 = np.random.uniform(0, 1, (2, 3))

weight = [weights1, weights2]

while c:
    for i in pg.event.get():
        if i.type == pg.QUIT:
            c = 0
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 1:
                pg.draw.circle(sc, col1, i.pos, 10)
                press.append(i.pos)
                print(weights1, weights2)

                pg.display.update()
            if i.button == 3:
                fps = 2

                pg.display.update()

    if len(press) == 2:
        test = np.random.randint(1, 600, 2)

        sc.fill((0, 0, 0))

        pg.draw.circle(sc, col2, test, 10)

        y1 = press[0][1]
        y2 = press[1][1]
        x1 = press[0][0]
        x2 = press[1][0]

        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1

        pg.draw.aaline(sc, (255, 255, 255), [0, b], [600, k * 600 + b])

        k2 = -1 / k

        b2 = test[1] - k2 * test[0]

        x_cross = (b2 - b) / (k - k2)
        y_cross = k * x_cross + b

        x_res = 2 * x_cross - test[0]
        y_res = 2 * y_cross - test[1]

        real_res = np.array([x_res, y_res])

        pg.draw.circle(sc, (255, 255, 255), real_res, 10)


        # --------------------------------------------------------------------------------------

        val = np.array([test[0], test[1], 100])

        hiden_layer = np.append(neuron_value(val, weights1, f), 100)

        neuro_res = neuron_value(hiden_layer, weights2, f)

        pg.draw.circle(sc, col0, neuro_res, 10)

        value = [val, hiden_layer, neuro_res]

        ad = backpropagation_linear(weight, value, real_res)

        weights1+=ad[1]

        weights2+=ad[0]

        print(error_func(neuro_res, real_res))

        pg.display.update()

    if len(press) > 2:
        sc.fill((0, 0, 0))

        y1 = press[0][1]
        y2 = press[1][1]
        x1 = press[0][0]
        x2 = press[1][0]

        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1

        pg.draw.aaline(sc, (255, 255, 255), [0, b], [600, k * 600 + b])

        k2 = -1 / k

        b2 = press[len(press) - 1][1] - k2 * press[len(press) - 1][0]

        x_cross = (b2 - b) / (k - k2)
        y_cross = k * x_cross + b

        x_res = 2 * x_cross - press[len(press) - 1][0]
        y_res = 2 * y_cross - press[len(press) - 1][1]

        real_res = [x_res, y_res]

        pg.draw.circle(sc, (255, 255, 255), real_res, 10)
        pg.draw.circle(sc, col1, press[len(press) - 1], 10)

        val = np.array([press[len(press) - 1][0], press[len(press) - 1][1], 100])

        hiden_layer = np.append(neuron_value(val, weights1, f), 100)

        neuro_res = neuron_value(hiden_layer, weights2, f)

        pg.draw.circle(sc, col0, neuro_res, 10)

        pg.display.update()



    clock.tick(fps)