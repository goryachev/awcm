import math
import base
import numpy as np

class nonAdaptive(base.wavelet):
    def forward(self):
        self.x = list(map(self.functor, self.x))
        self.label = "forward transform"

    def backward(self):
        self.label = "backward transform"

def f(x: float) -> float:
    return math.sin(x)

if __name__ == "__main__":
    x = np.arange(0.0, 2.0*math.pi, 0.5/math.pi)
    nonAW = nonAdaptive(f, x)
    nonAW.forward()
    nonAW.show()