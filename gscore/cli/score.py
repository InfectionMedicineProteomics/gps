import argparse

from gscore.parsers.osw import OSWFile
from gscore.parsers.queries import SelectPeakGroups
from gscore.parsers.sqmass import SqMassFile


class Score:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "score"

    def __call__(self, args: argparse.Namespace):

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

        if args.chromatogram_file:

            print("Parsing Chromatograms...")

            chromatogram_file = SqMassFile(args.chromatogram_file)

            chromatograms = chromatogram_file.parse()

            print("Matching chromatograms with precursors...")

            precursors.set_chromatograms(chromatograms)

        print("Scoring...")

        precursors.score_run(
            model_path=args.scoring_model,
            scaler_path=args.scaler,
            threads=args.threads,
            gpus=args.gpus,
            use_relative_intensities=args.use_relative_intensities,
        )

        print("Calculating Q Values")

        precursors.calculate_q_values(sort_key="d_score", use_decoys=True)

        print("Updating Q Values in file")

        osw_file.add_score_and_q_value_records(
            precursors
        )

        print("Done!")

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name, help="Commands to score and denoise OSW files"
        )

        self.parser.add_argument("-i", "--input", help="OSW file to process", type=str)

        self.parser.add_argument(
            "-c",
            "--chromatogram-file",
            dest="chromatogram_file",
            help="File containing chromatograms associated with the peakgroups.",
            type=str,
            default="",
        )

        self.parser.add_argument(
            "--scoring-model",
            dest="scoring_model",
            help="Path to scoring model to apply to data.",
            type=str,
        )

        self.parser.add_argument(
            "--scaler",
            dest="scaler",
            help="Path to scaler to transform data.",
            type=str,
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
            default=0.5,
            type=float,
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            help="The number of threads to use",
            default=1,
            type=int,
        )

        self.parser.add_argument(
            "--gpus",
            dest="gpus",
            type=int,
            help="Number of GPUs to use to train model.",
            default=1,
        )

        self.parser.add_argument(
            "--use-interpolated-chroms",
            dest="use_interpolated_chroms",
            help="Export interpolated chromatograms of a uniform length.",
            action="store_true",
        )

        self.parser.add_argument(
            "--use-relative-intensities",
            dest="use_relative_intensities",
            help="Scale each chromatogram to use relative intensities.",
            action="store_true",
        )

        self.parser.add_argument(
            "--include-score-columns",
            dest="include_score_columns",
            help="Include VOTE_PERCENTAGE and PROBABILITY columns as sub-scores.",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Score> {self.name}"
