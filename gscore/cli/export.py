import argparse

from gscore.parsers import queries
from gscore.parsers.osw import OSWFile


class Export:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "export"

    def __call__(self, args: argparse.Namespace):

        if args.export_method == "training-data":

            with OSWFile(args.input) as osw_file:

                print(f"Parsing {args.input}")

                precursors = osw_file.parse_to_precursors(
                    query=queries.SelectPeakGroups.FETCH_ALL_DENOIZED_DATA
                )

                print(f"Filtering and writing output.")

                precursors.dump_training_data(
                    args.output
                )

    def build_subparser(self,
                        subparser
                        ):

        self.parser = subparser.add_parser(
            self.name,
            help="Filter peakgroups and export training data."
        )

        self.parser.add_argument(
            '-i',
            '--input',
            dest='input',
            help='OSW file to filter and export training data',
            type=str
        )

        self.parser.add_argument(
            "--input-files",
            dest="input_files",
            nargs="+",
            help="List of files to export to quant matrix"
        )

        self.parser.add_argument(
            '--export-method',
            dest='export_method',
            choices=[
                'tric-formatted', 'comprehensive', 'peptide', 'protein', 'pyprophet', 'training-data'
            ],
            default='comprehensive',
            help='Which format to export results'
        )

        self.parser.add_argument(
            '-o',
            '--output',
            dest='output',
            help="Output file "
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):
        return f"<Export> {self.name}"