import argparse
from typing import Any

from gps.parsers.osw import OSWFile
from gps.parsers.queries import SelectPeakGroups


class Denoise:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "denoise"

    def __call__(self, args: argparse.Namespace) -> None:

        print(f"Processing file {args.input}")

        osw_file = OSWFile(args.input)

        precursors = osw_file.parse_to_precursors(
            query=SelectPeakGroups.FETCH_FEATURES_REDUCED
        )

        print("Denoising...")

        precursors.denoise(
            num_folds=args.num_folds,
            num_classifiers=args.num_classifiers,
            num_threads=args.threads,
            vote_percentage=args.vote_percentage,
        )

        print("Writing scores to OSW file...")

        osw_file.add_score_records(precursors)

        print("Done!")

    def build_subparser(self, subparser: Any) -> None:

        self.parser = subparser.add_parser(
            self.name, help="Commands to denoise OSW files"
        )

        self.parser.add_argument("-i", "--input", help="OSW file to process", type=str)

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
            "--threads",
            dest="threads",
            help="The number of threads to use",
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

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:

        return f"<Denoise> {self.name}"
