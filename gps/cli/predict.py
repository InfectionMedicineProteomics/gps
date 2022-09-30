import argparse

from typing import Any

import numpy as np

from gps.parsers.osw import OSWFile
from gps.parsers.queries import SelectPeakGroups
from gps.precursors import Precursors


class Predict:
    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "predict"

    def __call__(self, args: argparse.Namespace) -> None:

        print(f"Processing file {args.input}")

        osw_file = OSWFile(args.input)

        precursors: Precursors = osw_file.parse_to_precursors(
            query=SelectPeakGroups.FETCH_TRAINING_RECORDS
        )

        print("Predicting Peakgroups")

        precursors.predict_peakgroups(
            model_path=args.scoring_model,
            scaler_path=args.scaler,
            threads=args.threads,
            method=args.method,
        )

        labels = []

        for peakgroup in precursors.filter_peakgroups(
            rank=1, filter_key="PEAKGROUP_SCORE"
        ):

            if peakgroup.peakgroup_prediction == 1.0:

                labels.append(peakgroup.target)

        labels = np.array(labels)

        print(f"Targets: {labels[labels == 1].shape}")
        print(f"Decoys: {labels[labels == 0].shape}")

        print("Writing output")

        precursors.write_tsv(file_path=args.output, write_predicted=True)

    def build_subparser(self, subparser: Any) -> None:

        self.parser = subparser.add_parser(
            self.name, help="Commands to predict potential peakgroups from OSW files"
        )

        self.parser.add_argument("-i", "--input", help="OSW file to process", type=str)

        self.parser.add_argument("-o", "--output", help="Output TSV file.", type=str)

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
            "--method", dest="method", default="standard", type=str
        )

        self.parser.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:

        return f"<Score> {self.name}"
