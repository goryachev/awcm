from math import sin
import base

class nonAdaptive(base.wavelet):
    def forward(self, x):
        print(self.functor(x))

    def backward(self, x):
        pass

def f(x: float) -> float:
    return sin(x)

if __name__ == "__main__":
    nonAW = nonAdaptive(f)
    nonAW.forward(0.0)