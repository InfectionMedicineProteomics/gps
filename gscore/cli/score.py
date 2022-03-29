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
            encoder_path=args.chromatogram_encoder,
            threads=args.threads,
            gpus=args.gpus,
            use_relative_intensities=args.use_relative_intensities,
            chromatogram_only=args.use_only_chromatogram_features,
        )

        print("Calculating Q Values")

        precursors.calculate_q_values(sort_key="d_score", decoy_free=args.decoy_free)

        if args.output:

            precursors.write_tsv(file_path=args.output, ranked=1)

        else:

            print("Updating Q Values in file")

            osw_file.add_score_and_q_value_records(precursors)

        print("Done!")

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name, help="Commands to score and denoise OSW files"
        )

        self.parser.add_argument("-i", "--input", help="OSW file to process", type=str)

        self.parser.add_argument("-o", "--output", help="Output TSV file.", type=str)

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
            required=True,
        )

        self.parser.add_argument(
            "--scaler",
            dest="scaler",
            help="Path to scaler to transform data.",
            type=str,
            default="",
            required=True,
        )

        self.parser.add_argument(
            "--chromatogram-encoder",
            dest="chromatogram_encoder",
            help="Path to trained encoder model.",
            type=str,
            default="",
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
            "--use-only-chromatogram-features",
            dest="use_only_chromatogram_features",
            action="store_true",
            help="Use only features from the deepchrom model",
        )

        self.parser.add_argument(
            "--include-score-columns",
            dest="include_score_columns",
            help="Include VOTE_PERCENTAGE and PROBABILITY columns as sub-scores.",
            action="store_true",
        )

        self.parser.add_argument(
            "--decoy-free",
            dest="decoy_free",
            help="Use the second ranked target peakgroups as decoys for modelling the scores and calculating q-values",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Score> {self.name}"
