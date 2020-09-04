from tensorflow.keras.callbacks import Callback


class Hyperdash(Callback):
    """
    callback class for hyperdash app

    this allows you to check the learning progress on smartphones
    """
    def __init__(self, entries, exp):
        """
        :param entries: metrics of the learning
        :type entries: list[str]
        :param exp: hyperdash experiment object
        """
        super().__init__()

        self.exp = exp
        self.entries = entries

    def on_epoch_end(self, epoch, logs=None):
        # update experiment progress
        for entry in self.entries:
            log = logs.get(entry)
            if log is not None:
                self.exp.metric(entry, log)
