import numpy as np
from enum import Enum


class Adadelta:
    def __init__(self, benchmark, learning_rate, rho, epsilon):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.beta = rho
        self.eps = epsilon
        self.G = 0.0
        self.D = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.G = self.beta * self.G + (1 - self.beta) * grads ** 2.0
        d = - (self.D + self.eps) ** 0.5 / (self.G + self.eps) ** 0.5 * grads
        self.D = self.beta * self.D + (1 - self.beta) * d ** 2.0
        coords += self.lr * d
        return coords


class Adagrad:
    def __init__(self, benchmark, learning_rate, epsilon, initial_accumulator_value):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.eps = epsilon
        self.G = initial_accumulator_value

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.G += grads ** 2.0
        coords -= self.lr / (self.G + self.eps) ** 0.5 * grads
        return coords


class Adam:
    def __init__(self, benchmark, learning_rate, beta_1, beta_2, epsilon):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = epsilon
        self.t = 0.0
        self.m = 0.0
        self.v = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.t += 1.0
        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grads
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * grads ** 2.0
        m_bc = self.m / (1.0 - self.beta_1 ** self.t)
        v_bc = self.v / (1.0 - self.beta_2 ** self.t)
        coords -= self.lr * m_bc / (v_bc + self.eps) ** 0.5
        return coords


class Adamax:
    def __init__(self, benchmark, learning_rate, beta_1, beta_2, epsilon):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = epsilon
        self.t = 0.0
        self.m = 0.0
        self.v = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.t += 1.0
        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grads
        self.v = np.maximum(self.beta_2 * self.v, np.absolute(grads))
        m_bc = self.m / (1.0 - self.beta_1 ** self.t)
        coords -= self.lr * m_bc / (self.v + self.eps) ** 0.5
        return coords


class Momentum:
    def __init__(self, benchmark, learning_rate, momentum):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.gamma = momentum
        self.v = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.v = self.gamma * self.v + self.lr * grads
        coords -= self.v
        return coords


class Nadam:
    def __init__(self, benchmark, learning_rate, beta_1, beta_2, epsilon):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = epsilon
        self.t = 0.0
        self.m = 0.0
        self.v = 0.0

    def update(self, coords):
        # TODO: find out more
        #  https://medium.com/konvergen/modifying-adam-to-use-nesterov-accelerated-gradients-nesterov-accelerated-adaptive-moment-67154177e1fd
        grads = self.benchmark.grads(coords)
        self.t += 1.0
        self.m = self.beta_1 * self.m + (1.0 - self.beta_1) * grads
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * grads ** 2.0
        m_bc = self.m / (1.0 - self.beta_1 ** self.t)
        v_bc = self.v / (1.0 - self.beta_2 ** self.t)
        m_t = self.beta_1 * m_bc + (1 - self.beta_1) * grads
        coords -= self.lr * m_t / (v_bc + self.eps) ** 0.5
        return coords


class Nesterov:
    def __init__(self, benchmark, learning_rate, momentum):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.gamma = momentum
        self.v = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords - self.gamma * self.v)
        self.v = self.gamma * self.v + self.lr * grads
        coords -= self.v
        return coords


class RMSprop:
    def __init__(self, benchmark, learning_rate, rho, epsilon):
        self.benchmark = benchmark
        self.lr = learning_rate
        self.beta = rho
        self.eps = epsilon
        self.G = 0.0

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        self.G = self.beta * self.G + (1 - self.beta) * grads ** 2.0
        coords -= self.lr / (self.G + self.eps) ** 0.5 * grads
        return coords


class SGD:
    def __init__(self, benchmark, learning_rate):
        self.benchmark = benchmark
        self.lr = learning_rate

    def update(self, coords):
        grads = self.benchmark.grads(coords)
        coords -= self.lr * grads
        return coords


class Optimizers(Enum):
    """ enum class handling all the optimizer classes """
    ADADELTA = Adadelta
    ADAGRAD = Adagrad
    ADAM = Adam
    ADAMAX = Adamax
    MOMENTUM = Momentum
    NADAM = Nadam
    NESTEROV = Nesterov
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
    def get(cls, name, benchmark=None, params=None):
        """
        get instance of optimizer class by name

        :param name: name of the optimizer
        :type name: str
        :param benchmark: instace of the benchmark class
        :param params: hyperparamters for optimizer
        :type params: dict
        :return: instance of the optimizer class
        """
        return cls[name.upper()].value(benchmark, **params)

    @classmethod
    def list_all(cls):
        """
        get name list of all the optimizer classes

        :return: name list of all the optimizer classes
        :rtype: list[str]
        """
        return [member.lower() for member in sorted(list(cls.__members__.keys()))]
