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

        already_scored = False

        with OSWFile(args.input) as osw_conn:

            if "ghost_score_table" in osw_conn:

                already_scored = True

                precursors = osw_conn.parse_to_precursors(
                    query=SelectPeakGroups.FETCH_ALL_DENOIZED_DATA
                )

            else:

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

            print("Scoring...")

            precursors.score_run(
                model_path=args.scoring_model,
                scaler_path=args.scaler
            )

            print("Calculating Q Values")

            precursors.calculate_q_values(
                sort_key="d_score",
                use_decoys=True
            )

            print("Updating Q Values in PQP file")

            if not already_scored:

                osw_conn.add_score_and_q_value_records(
                    precursors
                )

            else:

                osw_conn.update_q_value_records(
                    precursors
                )

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
            "--apply-model",
            dest="apply_model",
            help=(
                "Set this flag if you want to apply a trained scoring model to the data"
            ),
            default=False,
            action="store_true"
        )

        self.parser.add_argument(
            "--scoring-model",
            dest="scoring_model",
            help="Path to scoring model to apply to data.",
            type=str
        )

        self.parser.add_argument(
            "--scaler",
            dest="scaler",
            help="Path to scaler to transform data.",
            type=str
        )

        self.parser.add_argument(
            '--threads',
            dest='threads',
            help='The number of threads to use',
            default=10,
            type=int
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Score> {self.name}"
