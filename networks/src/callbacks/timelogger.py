import csv
from datetime import datetime
from tensorflow.keras.callbacks import Callback


class TimeLogger(Callback):
    """
    callback class for recording time performance of the learning

    this records the time taken for each epoch and at the end of the training
    dumps the data in the first row of `time.csv`
    """
    def __init__(self, filename):
        """
        :param filename: path to store the log
        :type filename: str
        """
        super().__init__()
        self.filename = filename
        self.time_history = []
        self.start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        self.time_history.append((datetime.now() - self.start).total_seconds())

    def on_train_end(self, logs=None):
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            header = ['epoch', 'time']
            writer.writerow(header)
            for i in range(len(self.time_history)):
                row = [(i + 1), self.time_history[i]]
                writer.writerow(row)
