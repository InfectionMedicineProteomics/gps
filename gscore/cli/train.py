import argparse
from collections import Counter

import numpy as np

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import class_weight  # type: ignore

from gscore import preprocess
from gscore.models.deep_chrom_feature_classifier import DeepChromFeatureScorer
from gscore.models.deep_chromatogram_classifier import DeepChromModel, DeepChromScorer
from gscore.scaler import Scaler
from gscore.scorer import XGBoostScorer


class Train:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "train"

    def __call__(self, args: argparse.Namespace):

        print("Building scoring model...")

        combined_labels = []
        combined_chromatograms = []

        print("Loading data...")

        for input_file in args.input_files:

            training_data_npz = preprocess.get_training_data_from_npz(input_file)

            labels = training_data_npz["labels"]
            chromatograms = training_data_npz["chromatograms"]

            combined_chromatograms.append(chromatograms)
            combined_labels.append(labels)

        combined_chromatograms = np.concatenate(combined_chromatograms)
        combined_labels = np.concatenate(combined_labels)

        target_label_count = combined_labels[combined_labels == 1.0].size
        decoy_label_count = combined_labels[combined_labels == 0.0].size

        print(f"Target Peakgroups: {target_label_count}, Decoy Peakgroups: {decoy_label_count}")

        if args.train_deep_chromatogram_model:

            self.train_deep_model(
                combined_chromatograms,
                combined_labels,
                args.model_output,
                args.threads,
                args.gpus,
                args.epochs,
            )

    def train_deep_model(self, combined_chromatograms, combined_labels, model_output, threads, gpus, epochs):

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_chromatograms, combined_labels, test_size=0.2, shuffle=True
        )

        model = DeepChromScorer(
            threads=threads,
            max_epochs=epochs,
            gpus=gpus
        )

        print("Training model...")

        model.fit(
            data=training_data,
            labels=training_labels
        )

        print("Saving model...")

        model.save(
            model_output
        )

        print("Testing model...")

        roc_auc = model.evaluate(testing_data, testing_labels)

        print(f"ROC-AUC: {roc_auc}")


    def train_model(self, combined_data, combined_labels, model_output, scaler_output):

        scaler = Scaler()

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_data, combined_labels, test_size=0.2, shuffle=True
        )

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(training_labels),
            y=training_labels.ravel(),
        )

        counter: Counter = Counter(training_labels.ravel())
        scale_pos_weight = counter[0] / counter[1]

        scorer = XGBoostScorer(scale_pos_weight=scale_pos_weight)

        training_data = scaler.fit_transform(training_data)

        print("Training model...")

        scorer.fit(training_data, training_labels.ravel())

        testing_data = scaler.transform(testing_data)

        print("Evaluating model...")

        roc = scorer.evaluate(testing_data, testing_labels)

        print(f"Model ROC-AUC: {roc}")

        scorer.save(model_output)
        scaler.save(scaler_output)

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name, help="Training a scoring model from input data"
        )

        self.parser.add_argument(
            "-i",
            "--input",
            dest="input_files",
            help="NPZ files for training scorer",
            nargs="+",
        )

        self.parser.add_argument(
            "--model-output", dest="model_output", help="Output path for scoring model."
        )

        self.parser.add_argument(
            "--scaler-output",
            dest="scaler_output",
            help="Output path for scoring scaler.",
        )

        self.parser.add_argument(
            "--train-deep-chromatogram-model",
            dest="train_deep_chromatogram_model",
            action="store_true",
            help="Flag to indicate that a deep learning model should be trained on raw chromatograms."
        )

        self.parser.add_argument(
            "--include-score-columns",
            dest="include_score_columns",
            help="Include VOTE_PERCENTAGE and PROBABILITY columns as sub-scores.",
            action="store_true"
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            type=int,
            help="Number of threads/workers to use to train model.",
            default=1
        )

        self.parser.add_argument(
            "--gpus",
            dest="gpus",
            type=int,
            help="Number of GPUs to use to train model.",
            default=1
        )


        self.parser.add_argument(
            "--epochs",
            dest="epochs",
            type=int,
            help="Number of Epochs to use to train deep chrom model.",
            default=1
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"
