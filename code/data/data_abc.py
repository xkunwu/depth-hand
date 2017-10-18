from abc import ABCMeta, abstractmethod


class data_abc():
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_data(self):
        """ Training code
        """
        pass
