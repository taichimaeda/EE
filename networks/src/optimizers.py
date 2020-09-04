from enum import Enum
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta


class Optimizers(Enum):
    """ enum class handling all the optimizer classes """
    ADADELTA = Adadelta
    ADAGRAD = Adagrad
    ADAM = Adam
    ADAMAX = Adamax
    MOMENTUM = SGD
    NADAM = Nadam
    NESTEROV = SGD
    RMSPROP = RMSprop
    SGD = SGD

    @classmethod
    def exists(cls, name):
        """
        check if the specified optimizer class exists

        :param name: name of the optimizer
        :type name: str
        :rtype: bool
        """
        return True if name.upper() in cls.__members__ else False

    @classmethod
    def get(cls, name, params=None):
        """
        get instance of optimizer class by name

        :param name: name of the optimizer
        :type name: str
        :param params: hyperparamters for optimizer
        :type params: dict
        :return: instance of the optimizer class
        """
        if name == 'nesterov':
            params['nesterov'] = True

        optimizer = cls[name.upper()].value(**params)
        return optimizer
