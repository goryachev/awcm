import base
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial


class SecondGen(base.wavelet):
    def f(x: float) -> None:
        xxx = np.arange(0.0, 2.0 * math.pi, math.pi * 0.5)
        poly = lagrange(xxx, [y ** 3 for y in xxx])
        print(Polynomial(poly).coef)
        yyy = Polynomial(poly).coef

    def wgh(self, k0: int, n_l: int, n_h: int) -> np.float64:
        array = range(n_l, n_h + 1)
        i0 = array[array != k0]
        return np.prod(np.float64(i0) + 0.5) / np.prod(np.float64(i0) - np.float64(k0))

    def get_stencil(self, k: int, parity : int) -> list:
        N = len(self.x)
        g_stencil = []
        l_stencil = []
        if self.boundary == 'periodic':
            for ki in range(-(self.order + 1) // 2, self.order - (self.order + 1) // 2 + 1):
                g_stencil.append( # (k + ki + N) % N \
                                    (k + (2 * ki + parity) + N) % N)
                l_stencil.append(2 * ki + parity)
        if self.boundary == 'fantom':
            pass
        #else:  # 'nonperiodic'
        #    for ki in range(-(self.order+1)//2, self.order-(self.order+1)//2+1) \
        #                if (k + (2 * ki + parity) >= 0) and (k + (2 * ki + parity) < N) \
        #    l_stencil.append( k + (2 * ki + parity) # k + ki  )
        #    g_stencil.append()[]
        return l_stencil, g_stencil

    def transform(self, coef: np.float64, isOdd: int) -> None:
        for k in range(isOdd, len(self.x), 2):  # steps over ODD elements
            # self.x[k] += coef * (self.x[(k-1+len(self.x))%len(self.x)] + self.x[(k+1+len(self.x))%len(self.x)]); continue
            l_stencil, g_stencil = self.get_stencil(k, isOdd)
            for ki in g_stencil:
                self.x[k] += coef * self.x[ki] * self.wgh(ki, l_stencil[0], l_stencil[-1])

    def forward(self) -> None:
        self.move_level(1)  # self.x = list(map(lambda ix: self.predict(-1.0, ix), self.x[::2]))
        self.transform(-1.0, 0)   # predict
        self.transform(0.5, 1)   # update
        self.label = "forward transform"

    def backward(self) -> None:
        self.transform(-0.5, 1)  # update
        self.transform(1.0, 0)    # predict
        self.move_level(-1)
        self.label = "backward transform"

'''
    Alexey Buzovkin implementation
'''
import copy     # TODO: must be removed
class SecondGenAB:
    def wgh(self, k0, n_l, n_h):
        array = np.arange(n_l, n_h + 1)
        i0 = array[array != k0]
        return np.prod(i0 + 0.5) / np.prod(i0 - k0)

    def range_Corr(self, Ind, len, step, edge, order):
        N_range = np.arange(-int((order + 1) / 2), order - int((order + 1) / 2) + 1).astype(np.float32)
        N_range_corr = Ind + step * (2 * N_range + 1)
        N_range_corr = N_range_corr + (max(edge, N_range_corr[0]) - N_range_corr[0])
        N_range_corr = N_range_corr + (min(len - 1 - edge, N_range_corr[-1]) - N_range_corr[-1])
        N_range_corr = ((N_range_corr[N_range_corr >= 0] - Ind) / step - 1) / 2
        return N_range_corr

    '''
    PREDICT or UPDATE transform
    Predict::   e = step and coef = (+/-)1.0
    Update ::   e = 0    and coef = (+/-)0.5 
    '''
    def transform(self, y, step, e, order, coef):
        len_y = len(y)
        for i in np.arange(e, len_y, 2 * step):  # C_ind:
            range_corr = self.range_Corr(i, len_y, step, step - e, order)
            for k in range(len(range_corr)):
                y[i] = y[i] + coef * y[int(i + (2 * range_corr[k] + 1) * step)] * \
                       self.wgh(range_corr[k], range_corr[0], range_corr[-1])

    def forward_wt(self, y, jlvl, order_p, order_u, cv):
        y0 = copy.deepcopy(y)
        #   plt.figure(figsize=(14, 14))
        for j in range(jlvl - 1, 0, -1):
            s = 2 ** (jlvl - j - 1)
            self.transform(y, s, s, order_p, -1.0)  # predict
            self.transform(y, s, 0, order_u, 0.5)  # update
            # 1/0
        # plt.hist(y, bins=30, alpha=0.35)
        # y[np.abs(y) < cv] = 0.
        # plt.hist(y, bins=30, alpha=0.35)
        # plt.show()
        non_zero = np.sum(np.abs(y) >= cv * np.max(np.abs(y0)))
        y[np.abs(y) < cv * np.max(np.abs(y0))] = 0.
        return y, non_zero

    def inverse_wt(self, y, jlvl, order_p, order_u):
        for j in range(1, jlvl):
            s = 2 ** (jlvl - j - 1)
            self.transform(y, s, 0, order_u, -0.5)  # update
            self.transform(y, s, s, order_p, 1.0)  # predict
        return y

