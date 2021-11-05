from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math

class wavelet(ABC):
    """Base class for all wavelets"""
    def __init__(self, x, order=2, boundary='periodic', threshold=-1):
        self.x = x
        self.label = ''
        self.boundary = boundary
        self.order = order
        self.level = 0

    @classmethod
    @abstractmethod
    def forward(self):
        pass

    @classmethod
    @abstractmethod
    def backward(self):
        pass

    def move_level(self, shift: int) -> None:
      #  if not self.x or len(self.x) % 2: return
        if not shift: return
        if shift < 0:
            self.x = self.x[::2][:]
            shift += 1
            self.level -= 1
        if shift > 0:
            self.x = [item for tup in zip(self.x, [0.0 for i in range(len(self.x))]) for item in tup][:]
            shift -= 1
            self.level += 1
        self.move_level(shift)

    def show(self, origin=[]) -> None:
        plt.grid()
        plt.plot(self.x)
        plt.legend(['tranformed'])
        plt.gca().axes.xaxis.set_ticklabels([]) # turn off X-tick marks
        if not self.level > 0 and origin:
            plt.plot(origin[::math.ceil(2**(-self.level))])
            plt.legend(['tranformed', 'original'])
        plt.xlabel('x')
        plt.ylabel(self.label)
        plt.show()

    def print(self) -> str:
        return "j = " + str(self.level) + ", order = " + str(self.order) + ", array_size = " + str(len(self.x))
