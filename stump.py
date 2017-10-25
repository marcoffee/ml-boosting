import numpy as np
import abc


positive = '+'
negative = '-'

class Stump (object, metaclass = abc.ABCMeta):

    @abc.abstractmethod
    def fit (self, data, labels):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate (self, inp, positive, negative):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__ (self):
        raise NotImplementedError

    @abc.abstractmethod
    def __deepcopy__ (self, _):
        raise NotImplementedError

class Decision (Stump):

    def __init__ (self, column, value):
        self.__column = column
        self.__value = value

    def fit (self, data, labels):
        data = data[ :, self.column] == self.value
        return np.logical_xor(labels, data)

    def evaluate (self, data, positive = positive, negative = negative):
        result = np.tile(negative, ( data.shape[0], 1 ))
        result[data[ :, self.column ] == self.value] = positive
        return result

    def __str__ (self):
        return '[ {} = {} ]'.format(self.column, self.value)

    def __deepcopy__ (self, _):
        return Decision(self.column, self.value)

    @property
    def column (self):
        return self.__column

    @property
    def value (self):
        return self.__value

class Positive (Stump):

    def fit (self, _, labels):
        return ~labels

    def evaluate (self, inp, positive = positive, negative = negative):
        return np.tile(positive, ( inp.shape[0], 1 ))

    def __deepcopy__ (self, _):
        return Positive()

    def __str__ (self):
        return 'Positive'

class Negative (Stump):

    def fit (self, _, labels):
        return labels.copy()

    def evaluate (self, inp, positive = positive, negative = negative):
        return np.tile(positive, ( inp.shape[0], 1 ))

    def __deepcopy__ (self, _):
        return Negative()

    def __str__ (self):
        return 'Negative'
