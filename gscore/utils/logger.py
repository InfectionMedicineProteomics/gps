import logging
import sys

from enum import Enum

NAME = ''
LEVEL = ''

class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @classmethod
    def get_level(cls, name=''):
        return cls[name].value

## TODO: Only emit based on log level

class Logger(logging.Logger):

    def __init__(self, name='', level=''):

        super().__init__(name)

        self.setLevel(
            LogLevel.get_level(level)
        )

        file_handler = logging.FileHandler(
            f"{name}.log"
        )

        stdout_handler = logging.StreamHandler(
            sys.stdout
        )

        self.addHandler(file_handler)
        
        self.addHandler(stdout_handler)