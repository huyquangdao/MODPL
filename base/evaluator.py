from abc import ABC, abstractmethod


class Evaluator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        raise NotImplementedError("Must implement")

    @abstractmethod
    def report(self):
        raise NotImplementedError("Must implement")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("Must implement")
