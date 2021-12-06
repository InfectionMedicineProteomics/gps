import argparse

from gscore.parsers.osw import OSWFile
from gscore.parsers.queries import SelectPeakGroups

class Score:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "score"

    def __call__(self, args: argparse.Namespace):

        print(f"Processing file {args.input}")

        with OSWFile(args.input) as osw_conn:

            precursors = osw_conn.parse_to_precursors(
                query=SelectPeakGroups.FETCH_FEATURES
            )

            print("Denoising...")

            precursors.denoise(
                num_folds=args.num_folds,
                num_classifiers=args.num_classifiers,
                num_threads=args.threads,
                vote_threshold=args.vote_threshold
            )

            print("Writing scores to OSW file...")

            osw_conn.add_score_records(precursors)

            print("Done!")


    def build_subparser(self,
                        subparser
    ):

        self.parser = subparser.add_parser(
            self.name,
            help="Commands to score and denoise OSW files"
        )

        self.parser.add_argument(
            "-i",
            "--input",
            help="OSW file to process",
            type=str
        )

        self.parser.add_argument(
            "--denoise-only",
            dest="denoise_only",
            help=(
                "Set this flag if you want to only denoise the data, and not calculate the q-values. "
                "This is done if you are training a new model."
            ),
            default=False,
            action="store_true"
        )

        self.parser.add_argument(
            '--num-classifiers',
            dest='num_classifiers',
            help='The number of ensemble learners used to denoise each fold',
            default=10,
            type=int
        )

        self.parser.add_argument(
            '--num-folds',
            dest='num_folds',
            help='The number of folds used to denoise the target labels',
            default=10,
            type=int
        )

        self.parser.add_argument(
            '--threads',
            dest='threads',
            help='The number of threads to use',
            default=10,
            type=int
        )

        self.parser.add_argument(
            "--vote-threshold",
            dest="vote_threshold",
            help="The minimum probability needed to be counted as a positive vote",
            default=0.8,
            type=float
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Score> {self.name}"