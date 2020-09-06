from enum import Enum
import numpy as np


class Ackley:
    def __init__(self):
        self.optimum = (0.0, 0.0)

    def grads(self, coords):
        x, y = coords
        def d(t):
            t1 = 4.0 * y * np.exp(-0.1 * (x ** 2.0 + y ** 2.0))
            t2 = 6.28 * t * np.sin(6.28 * t ** 2.0) * np.exp(0.5 * (np.cos(6.28 * x ** 2.0) + np.cos(6.28 * y ** 2.0)))
            return t1 + t2
        grads = np.array([d(x), d(y)])
        return grads


class Bukin:
    def __init__(self):
        self.optimum = (-10.0, 1.0)

    def grads(self, coords):
        x, y = coords
        dx = (0.01 * (x + 10.0)) / (np.abs(x + 10.0) + 1e-7) - (0.5 * (y - 0.01 * x)) / (
                    (np.abs(y - 0.01 * x) ** (3.0 / 2.0)) + 1e-7)
        dy = (50.0 * (y - 0.01 * x)) / (np.abs(y - 0.01 * x) ** (3.0 / 2.0) + 1e-7)
        grads = np.array([dx, dy])
        return grads


class Eclipse:
    def __init__(self):
        self.optimum = (0.0, 0.0)

    def grads(self, coords):
        x, y = coords
        dx, dy = 0.2 * x, 2.0 * y
        grads = np.array([dx, dy])
        return grads


class Levi:
    def __init__(self):
        self.optimum = (1.0, 1.0)

    def grads(self, coords):
        x, y = coords
        dx = 2.0 * (x - 1.0) * (np.sin(3.0 * np.pi * y) ** 2.0 + 1.0) + 6.0 * np.pi * np.sin(3.0 * np.pi * x) * np.cos(
            3.0 * np.pi * x)
        dy = 6.0 * np.pi * (x - 1.0) ** 2 * np.sin(3.0 * np.pi * y) * np.cos(3.0 * np.pi * y) + 2.0 * (y - 1.0) * (
                    np.sin(2.0 * np.pi * y) ** 2.0 + 1.0) \
             + 4.0 * np.pi * (y - 1.0) ** 2.0 * np.sin(2.0 * np.pi * y) * np.cos(2.0 * np.pi * y)
        grads = np.array([dx, dy])
        return grads


class Rastrigin:
    def __init__(self):
        self.optimum = (0.0, 0.0)

    def grads(self, coords):
        x, y = coords
        def d(t):
            return 2.0 * (t + 10.0 * np.pi * np.sin(2.0 * np.pi * t))
        grads = np.array([d(x), d(y)])
        return grads


class Rosenbrock:
    def __init__(self):
        self.optimum = (1.0, 1.0)

    def grads(self, coords):
        x, y = coords
        dx = 200.0 * x ** 3.0 + x * (1.0 - 200 * y) - 1.0
        dy = 100.0 * (y - x ** 2.0)
        grads = np.array([dx, dy])
        return grads


class Benchmarks(Enum):
    """ enum class handling all the benchmark classes """
    ECLIPSE = Eclipse
    ACKLEY = Ackley
    ROSENBROCK = Rosenbrock
    RASTRIGIN = Rastrigin
    LEVI = Levi
    BUKIN = Bukin

    @classmethod
    def exists(cls, name):
        """
        check if the specified benchmark class exists

        :param name: name of the benchmark
        :type name: str
        :rtype: bool
        """
        return True if name.upper() in cls.__members__ else False

    @classmethod
    def get(cls, name):
        """
        get instance of benchmark class by name

        :param name: name of the benchmark
        :type name: str
        :return: instance of the benchmark class
        """
        return cls[name.upper()].value()

    @classmethod
    def list_all(cls):
        """
        get name list of all the benchmark classes

        :return: name list of all the benchmark classes
        :rtype: list[str]
        """
        return [member.lower() for member in sorted(list(cls.__members__.keys()))]


