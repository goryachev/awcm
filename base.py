from abc import ABC, abstractmethod

class wavelet(ABC):
    """Base class for all wavelets"""
    def __init__(self, func):
        self.functor = func

    @classmethod
    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    @abstractmethod
    def backward(self, x):
        pass