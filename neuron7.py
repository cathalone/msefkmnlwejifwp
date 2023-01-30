import numpy as np
import pygame as pg
import neural_network as nn


sc = pg.display.set_mode((600, 600))

clock = pg.time.Clock()

col0 = (0, 0, 255)

col1 = (255, 0, 0)
col2 = (0, 255, 0)

fps = 10000

c = 1
press = []

neuron_struct = nn.NeuralNetwork([2, 2, 2], ['identity', 'identity', 'identity'], 0, 1, 1)
bias = 1
education_speed = 0.001

while c:
    for i in pg.event.get():
        if i.type == pg.QUIT:
            c = 0
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 1:
                pg.draw.circle(sc, col1, i.pos, 10)
                press.append(i.pos)

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

        real_res = np.array([x_res, y_res])/600

        pg.draw.circle(sc, (255, 255, 255), real_res*600, 10)

        # --------------------------------------------------------------------------------------

        input_data = np.array([test[0], test[1]])/600

        neuron_result = neuron_struct.check(input_data)

        pg.draw.circle(sc, col0, neuron_result*600, 10)

        neuron_struct.train([input_data], [real_res], 1, education_speed)

        print(nn.error_func(neuron_result, real_res))

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

        val = np.array([press[len(press) - 1][0], press[len(press) - 1][1]])/600

        neuro_res = neuron_struct.check(val)

        pg.draw.circle(sc, col0, neuro_res*600, 10)

        pg.display.update()

    clock.tick(fps)
