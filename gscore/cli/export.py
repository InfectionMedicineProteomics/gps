import argparse
from typing import Union

from gscore.chromatograms import Chromatograms
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile
from gscore.parsers.sqmass import SqMassFile


class Export:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "export"

    def __call__(self, args: argparse.Namespace):

        print(f"Parsing {args.input}")

        osw_file = OSWFile(args.input)

        precursors = osw_file.parse_to_precursors(
            query=queries.SelectPeakGroups.FETCH_PREC_RECORDS
        )

        use_chromatograms: bool = False

        if args.chromatogram_file:

            print("Parsing Chromatograms...")

            chromatogram_file = SqMassFile(args.chromatogram_file)

            chromatograms = chromatogram_file.parse()

            print("Matching chromatograms with precursors...")

            precursors.set_chromatograms(chromatograms)

            use_chromatograms = True


        print(f"Filtering and writing output.")

        precursors.dump_training_data(
            args.output,
            filter_field=args.filter_field,
            filter_value=args.filter_value,
            use_relateive_intensities=args.use_relative_intensities
        )

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name, help="Filter peakgroups and export training data."
        )

        self.parser.add_argument(
            "-i",
            "--input",
            dest="input",
            help="OSW file to filter and export training data",
            type=str,
        )

        self.parser.add_argument(
            "--filter-field",
            dest="filter_field",
            choices=["PROBABILITY", "VOTE_PERCENTAGE"],
            default="VOTE_PERCENTAGE",
            help="Field to filter the peakgroups for export as training data.",
        )

        self.parser.add_argument(
            "--filter-value",
            dest="filter_value",
            type=float,
            default=1.0,
            help="Set value by which the target peakgroups are filtered on the filter-field",
        )

        self.parser.add_argument("-o", "--output", dest="output", help="Output file ")

        self.parser.add_argument(
            "-c",
            "--chromatogram-file",
            dest="chromatogram_file",
            help="File containing chromatograms associated with the peakgroups.",
            default=""
        )

        self.parser.add_argument(
            "--use-relative-intensities",
            dest="use_relative_intensities",
            help="Scale each chromatogram to use relative intensities.",
            action="store_true"
        )

        self.parser.add_argument(
            "--include-score-columns",
            dest="include_score_columns",
            help="Include VOTE_PERCENTAGE and PROBABILITY columns as sub-scores.",
            action="store_true"
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):
        return f"<Export> {self.name}"
