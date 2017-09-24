from abc import ABCMeta, abstractmethod


class train_abc(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        """ Training code
        """
        pass
