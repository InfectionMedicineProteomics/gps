import argparse
from typing import Union, Any

from gscore.chromatograms import Chromatograms
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile
from gscore.parsers.sqmass import SqMassFile


class Export:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "export"

    def __call__(self, args: argparse.Namespace) -> None:

        print(f"Parsing {args.input}")

        osw_file = OSWFile(args.input)

        precursors = osw_file.parse_to_precursors(
            query=queries.SelectPeakGroups.FETCH_TRAINING_RECORDS
        )

        use_chromatograms: bool = False

        if args.chromatogram_file:

            print("Parsing Chromatograms...")

            chromatogram_file = SqMassFile(args.chromatogram_file)

            chromatograms = chromatogram_file.parse()

            print("Matching chromatograms with precursors...")

            precursors.set_chromatograms(chromatograms)

            use_chromatograms = True

        print("Denoising...")

        precursors.denoise(
            num_folds=args.num_folds,
            num_classifiers=args.num_classifiers,
            num_threads=args.threads,
            vote_percentage=args.vote_percentage,
        )

        print("Writing scores to OSW file...")

        osw_file.add_score_records(precursors)

        print(f"Filtering and writing output.")

        if args.no_filter:

            precursors.dump_training_data(
                args.output, filter_field="PROBABILITY", filter_value=0.0
            )

        else:

            precursors.dump_training_data(
                args.output,
                filter_field=args.filter_field,
                filter_value=args.filter_value,
            )

    def build_subparser(self, subparser: Any) -> None:

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
            default="",
        )

        self.parser.add_argument(
            "--num-classifiers",
            dest="num_classifiers",
            help="The number of ensemble learners used to denoise each fold",
            default=10,
            type=int,
        )

        self.parser.add_argument(
            "--num-folds",
            dest="num_folds",
            help="The number of folds used to denoise the target labels",
            default=10,
            type=int,
        )

        self.parser.add_argument(
            "--vote-percentage",
            dest="vote_percentage",
            help="The minimum probability needed to be counted as a positive vote",
            default=0.8,
            type=float,
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            help="The number of threads to use",
            default=10,
            type=int,
        )

        self.parser.add_argument(
            "--no-filter",
            dest="no_filter",
            help="Do not filter exported data.",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:
        return f"<Export> {self.name}"
