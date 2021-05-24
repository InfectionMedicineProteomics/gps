
class PassArgs:

    def __init__(self, args=dict()):
        self._args = args

    def __getattr__(self, key):
        return self._args[key]