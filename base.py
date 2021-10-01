from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class wavelet(ABC):
    """Base class for all wavelets"""
    def __init__(self, func, x):
        self.functor = func
        self.x = x
        self.label = ''

    @classmethod
    @abstractmethod
    def forward(self):
        pass

    @classmethod
    @abstractmethod
    def backward(self):
        pass

    def show(self):
        plt.plot(self.x)
        plt.ylabel(self.label)
        plt.grid()
        plt.show()