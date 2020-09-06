from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from enum import Enum
import functools


def get_lenet(dataset):
    """
    function which returns lenet based model

    this requires instance of the dataset class in argument for accessing some information

    :param dataset: instance of the dataset class
    :return: instance of the squential model
    """
    model = Sequential()

    # 3 convolution layers
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=dataset.image_shape, activation='relu'))

    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2 dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dataset.num_classes, activation='softmax'))

    return model


def get_lstm(dataset):
    """
    function which returns lstm based model

    this requires instance of the dataset class in argument for accessing some information

    :param dataset: instance of the dataset class
    :return: instance of the squential model
    """
    model = Sequential()

    model.add(Embedding(input_dim=dataset.num_words, input_length=dataset.max_review_len, output_dim=dataset.embedding_vec_len))
    model.add(Dropout(0.2))

    model.add(LSTM(100))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    return model


class Models(Enum):
    """
    enum class handling all the model functions

    functions as enum values are considered to be method definitions
    so use `partial()` to avoid it
    """
    LENET = functools.partial(get_lenet)
    LSTM = functools.partial(get_lstm)

    @classmethod
    def exists(cls, name):
        """
        check if the specified model fuction exists

        :param name: name of the model
        :type name: str
        :rtype: bool
        """
        return True if name.upper() in cls.__members__ else False

    @classmethod
    def get(cls, name, dataset):
        """
        get the return value of the model function by name

        :param name: name of the model
        :type name: str
        :param dataset: instance of the dataset class
        :return: instance of the model
        """
        return cls[name.upper()].value(dataset)
