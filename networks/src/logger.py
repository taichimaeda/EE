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
        # list of info used in writing logs
        self.info_history = []

        # set log path
        self.log_path = f'../logs/log_'\
                        + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S') + '.txt'

        # prepare log header
        log_header = f'sys.version\n{sys.version}'
        log_header += f'\n\ntf.__version__\n{tf.__version__}'
        log_header += f'\n\nkeras.__version__\n{keras.__version__}'

        # get system info from shell
        def exec_command(*args):
            return subprocess.run(args, stdout=subprocess.PIPE).stdout.decode()

        log_header += '\n\n' + exec_command('cat', '/etc/issue')
        log_header += exec_command('lshw', '-short')
        log_header += ' \n' + exec_command('nvidia-smi')

        with open(self.log_path, 'w') as f:
            f.write(log_header)

    def read(self, func):
        """ decorator function for reading argument info of the target function """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # get arugments after `self`
            info = ' '.join(map(str, args[1:]))
            self.info_history.append(info)

            # execute func
            func(*args, **kwargs)

        return wrapper

    def write(self, func):
        """ decorator function for writing logs """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # leave log before
            print(f'started {self.info_history[-1]}')
            with open(self.log_path, 'a') as f:
                f.write(f'\n\n{self.info_history[-1]}')
                f.write(f'\nstarted at ' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S:%f'))

            # execute func
            func(*args, **kwargs)

            # leave log after
            print(f'finished {self.info_history[-1]}')
            with open(self.log_path, 'a') as f:
                f.write(f'\nfinished at ' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S:%f'))

        return wrapper
