from enum import Enum
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import imdb
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Singleton:
    """
    avoid loading datasets multiple times
    """
    def __new__(cls, *args):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args)
        return cls._instance


class Cifar10(Singleton):
    def __init__(self):
        self.num_classes = 10
        self.image_shape = (32, 32, 3)
        self.data = cifar10.load_data()

    def get_batch(self):
        """ get entire data from the dataset """
        (x_train, y_train), (x_test, y_test) = self.data
        x_train, x_test = [self.preprocess(d) for d in (x_train, x_test)]
        y_train, y_test = [self.preprocess(d, is_label=True) for d in (y_train, y_test)]

        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, data, is_label=False):
        """ this normalises data and converts labels into one hot encoding """
        if is_label:
            data = to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255.0
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data


class FashionMnist(Singleton):
    def __init__(self):
        self.num_classes = 10
        self.image_shape = (28, 28, 1)
        self.data = fashion_mnist.load_data()

    def get_batch(self):
        """ get entire data from the dataset """
        (x_train, y_train), (x_test, y_test) = self.data
        x_train, x_test = [self.preprocess(d) for d in (x_train, x_test)]
        y_train, y_test = [self.preprocess(d, is_label=True) for d in (y_train, y_test)]

        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, data, is_label=False):
        """ this normalises data and converts labels into one hot encoding """
        if is_label:
            data = to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255.0
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data


class IMDB(Singleton):
    def __init__(self):
        self.num_words = 5000
        self.max_review_len = 500
        self.embedding_vec_len = 32
        self.data = imdb.load_data(num_words=self.num_words)

    def get_batch(self):
        """ get entire data from the dataset """
        (x_train, y_train), (x_test, y_test) = self.data
        x_train = pad_sequences(x_train, self.max_review_len)
        x_test = pad_sequences(x_test, self.max_review_len)

        return (x_train, y_train), (x_test, y_test)


class Mnist(Singleton):
    def __init__(self):
        self.num_classes = 10
        self.image_shape = (28, 28, 1)
        self.data = mnist.load_data()

    def get_batch(self):
        """ get entire data from the dataset """
        (x_train, y_train), (x_test, y_test) = self.data
        x_train, x_test = [self.preprocess(d) for d in (x_train, x_test)]
        y_train, y_test = [self.preprocess(d, is_label=True) for d in (y_train, y_test)]

        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, data, is_label=False):
        """ this normalises data and converts labels into one hot encoding """
        if is_label:
            data = to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255.0
            shape = (data.shape[0],) + self.image_shape
            data = data.reshape(shape)

        return data


class Datasets(Enum):
    """ enum class handling all the dataset classes """
    MNIST = Mnist
    FASHION_MNIST = FashionMnist
    CIFAR10 = Cifar10
    IMDB = IMDB

    @classmethod
    def exists(cls, name):
        """
        check if the specified dataset class exists

        :param name: name of the dataset
        :type name: str
        :rtype: bool
        """
        return True if name.upper() in cls.__members__ else False

    @classmethod
    def get(cls, name):
        """
        get instance of dataset class by name

        :param name: name of the dataset
        :type name: str
        :return: instance of the dataset class
        """
        return cls[name.upper()].value()
