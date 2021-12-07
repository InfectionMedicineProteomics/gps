import argparse
from collections import Counter

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import (
    shuffle,
    class_weight
)

from gscore.utils import ml
from gscore.scaler import Scaler
from gscore.scorer import (
    SGDScorer,
    AdaBoostSGDScorer,
    GradientBoostingScorer,
    BalancedBaggingScorer,
    EasyEnsembleScorer,
    XGBoostScorer
)

class Train:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "train"

    def __call__(self, args: argparse.Namespace):

        print("Building scoring model...")

        combined_data = []
        combined_labels = []

        print("Loading data...")

        for input_file in args.input_files:

            scores, labels = ml.get_training_data_from_npz(
                input_file
            )

            combined_data.append(scores)
            combined_labels.append(labels)

        combined_data = np.concatenate(combined_data)
        combined_labels = np.concatenate(combined_labels)

        scaler = Scaler()

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_data,
            combined_labels,
            test_size=0.2,
            shuffle=True
        )

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(training_labels),
            y=training_labels.ravel()
        )

        counter = Counter(training_labels.ravel())
        scale_pos_weight = counter[0] / counter[1]

        scorer = XGBoostScorer(scale_pos_weight=scale_pos_weight)

        training_data = scaler.fit_transform(training_data)

        print("Training model...")

        scorer.fit(training_data, training_labels.ravel())

        testing_data = scaler.transform(testing_data)

        print("Evaluating model...")

        roc = scorer.evaluate(testing_data, testing_labels)

        print(f"Model ROC-AUC: {roc}")

        scorer.save(args.model_output)
        scaler.save(args.scaler_output)


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
            help='NPZ files for training scorer',
            nargs='+'
        )

        self.parser.add_argument(
            "--model-output",
            dest="model_output",
            help="Output path for scoring model."
        )

        self.parser.add_argument(
            "--scaler-output",
            dest="scaler_output",
            help="Output path for scoring scaler."
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"