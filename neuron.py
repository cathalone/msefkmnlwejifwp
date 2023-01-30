import numpy as np
import pygame as pg


def neuron(value_arr, weight_arr, act_func):
    x = np.dot(value_arr, np.transpose(weight_arr))
    res = act_func(x)
    return res


def f(x):
    y = x
    return y


def error_func(obtained, expected):
    res = 0
    for i in range(len(obtained)):
        res += (obtained[i] - expected[i]) ** 2
        res /= len(obtained)
        return res


sc = pg.display.set_mode((600, 600))

clock = pg.time.Clock()

col0 = (0, 0, 255)

col1 = (255, 0, 0)
col2 = (0, 255, 0)

c = 1
press = []


weights = np.random.uniform(0, 2, (2, 3))

while c:
    for i in pg.event.get():
        if i.type == pg.QUIT:
            c = 0
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 1:

                pg.draw.circle(sc, col1, i.pos, 10)
                press.append(i.pos)

                pg.display.update()

    if len(press) == 2:
        test = np.random.randint(0, 600, 2)

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

        real_res = [x_res, y_res]

        pg.draw.circle(sc, (50, 50, 50), real_res, 10)

        # --------------------------------------------------------------------------------------

        val = np.array([test[0], test[1], 1])

        neuro_res = neuron(val, weights, f)


        pg.draw.circle(sc, col0, neuro_res, 10)

        adj1 = 2 * (np.dot(val, np.transpose(weights[0]))-x_res)*val
        weights[0] -= 0.0000001 * adj1

        adj2 = 2 * (np.dot(val, np.transpose(weights[1]))-y_res)*val
        weights[1] -= 0.0000001 * adj2

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

        real_res = np.array([x_res, y_res])

        pg.draw.circle(sc, (50, 50, 50), real_res, 10)
        pg.draw.circle(sc, col1, press[len(press) - 1], 10)

        val = np.array([press[len(press) - 1][0], press[len(press) - 1][1], 1])
        neuro_res = neuron(val, weights, f)

        pg.draw.circle(sc, col0, neuro_res, 10)

        pg.display.update()

    clock.tick(500)
