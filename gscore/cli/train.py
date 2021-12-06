import argparse

class Train:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "train"

    def __call__(self, args: argparse.Namespace):

        print("executing score command")

    def build_subparser(self,
                        subparser
    ):

        self.parser = subparser.add_parser(
            self.name,
            help="Training a scoring model from input data"
        )

        self.parser.add_argument(
            '-i',
            '--input',
            dest='input_files',
            help='OSW files for training scorer',
            nargs='+'
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"