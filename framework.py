import math
from secondGen import *

import numpy as np
import matplotlib.pyplot as plt
import copy

import keyboard
if __name__ == "__main__":
    N = 1024
    x = [math.sin(ix * 2.0 * math.pi / float(N)) for ix in range(N)]
    origin = x[:]   # copy x
    wlt = SecondGen(x, 2, 'periodic', 0.01) # data, order, boundary, threshold

    print("Press arrow or ctrl (esc for exit)")
    print(">>>", wlt.print())
    while True:
        if keyboard.read_key() == "up":
            if not wlt.level < 0 : continue
            wlt.forward()
            print(">F>", wlt.print())
        if keyboard.read_key() == "down":
            wlt.backward()
            print("<B<", wlt.print())
        if keyboard.read_key() == "ctrl":
            print("... plotting")
            wlt.show(origin)
        if keyboard.read_key() == "right":
            if wlt.order < len(x): wlt.order += 1
            print("... order =", wlt.order)
        if keyboard.read_key() == "left":
            if wlt.order > 1: wlt.order -= 1
            print("... order =", wlt.order)
        if keyboard.read_key() == "backspace":
            wlt.x = origin[:]
            wlt.order = 2
            wlt.level = 0
            print("RES", wlt.print())
        if keyboard.read_key() == "esc":
            break

        if keyboard.read_key() == "t":
            print("... testing:")
            wlt = SecondGenAB()
            func = lambda x: (np.sin(10. * x) + np.exp(- 0.25 * ((x - 5.) ** 2))).astype(np.float64)

            cvals = (10 ** np.arange(-1., -7., -1.)).astype(np.float64)  # 10**np.arange(2, )
            err_list = []
            nz_list = []
            param_j = 10

            for c0, cval in enumerate(cvals):
                y_ = func(np.arange(0., 10.001, 10. / 4096.)).astype(np.float64)
                # y_ = np.arange(0., 10.001, 10./ 4096.).astype(np.float64)
                ans_, nz = wlt.forward_wt(copy.deepcopy(y_), param_j, 3, 3, cval)
                y_rest = wlt.inverse_wt(copy.deepcopy(ans_), param_j, 3, 3)

                nz_list.append(copy.deepcopy(nz))
                err_list.append(copy.deepcopy(np.max(np.abs(y_ - y_rest))))
                print(c0, len(cvals), err_list[-1])

            plt.figure(figsize=(12, 25))

            nz_list = np.array(nz_list).astype(float)
            err_list = np.array(err_list).astype(float)

            plt.subplot(211)
            plt.plot(cvals, err_list)
            plt.scatter(cvals, err_list)

            plt.grid()
            plt.yscale("log")
            plt.xscale("log")

            plt.subplot(212)
            plt.plot(nz_list, err_list)
            plt.scatter(nz_list, err_list)

            plt.plot(nz_list, cvals)
            plt.scatter(nz_list, cvals)

            plt.plot(nz_list, 1000000 * nz_list ** (-4))
            plt.scatter(nz_list, 1000000 * nz_list ** (-4))

            plt.legend(["err(nz)", "cvals(nz)", "nz**(-4) (nz)"])

            plt.grid()

            plt.yscale("log")
            plt.xscale("log")

            plt.show()

