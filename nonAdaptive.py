from abc import ABC, abstractmethod

class wavelet(ABC):
    """Base class for all wavelets"""
    @classmethod
    @abstractmethod
    def forward(self):
        pass

    @classmethod
    @abstractmethod
    def backward(self):
        pass


class nonAdaptive(wavelet):
    def forward(self):
        pass

    def backward(self):
        pass

if __name__ == "__main__":
    nonAW = nonAdaptive()
    nonAW.forward()