import tensorflow as tf
from tensorflow import keras
import sys
import subprocess
import functools
from datetime import datetime
from datetime import timezone
from datetime import timedelta


class Logger:
    def __init__(self):
        """ create and initialise log """
        # set log path
        self.path = f'../logs/' \
                    + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S') + '.txt'

        # get system info from shell
        def execute(*args):
            return subprocess.run(args, stdout=subprocess.PIPE).stdout.decode()

        header = f'sys.version\n{sys.version}'
        header += f'\n\ntf.__version__\n{tf.__version__}'
        header += f'\n\nkeras.__version__\n{keras.__version__}'
        header += '\n\n' + execute('cat', '/etc/issue')
        header += execute('lshw', '-short')
        header += ' \n' + execute('nvidia-smi')

        with open(self.path, 'w') as f:
            f.write(header)

        self.args = None

    def read(self, func):
        """ decorator function for reading argument info of the target function """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # get arugments after `self`
            self.args = ' '.join(map(str, args[1:]))

            # execute func
            func(*args, **kwargs)
        return wrapper

    def write(self, func):
        """ decorator function for writing logs """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # leave log before
            print(f'started {self.args}')
            with open(self.path, 'a') as f:
                f.write(f'\n\n{self.args}')
                f.write(f'\nstarted at ' + datetime.now(timezone(timedelta(hours=2))).strftime('%Y/%m/%d %H:%M:%S:%f'))

            # execute func
            func(*args, **kwargs)

            # leave log after
            print(f'finished {self.args}')
            with open(self.path, 'a') as f:
                f.write(f'\nfinished at ' + datetime.now(timezone(timedelta(hours=2))).strftime('%Y/%m/%d %H:%M:%S:%f'))
        return wrapper
