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
        combined_scores = []

        print("Loading data...")

        for input_file in args.input_files:

            training_data_npz = preprocess.get_training_data_from_npz(input_file)

            labels = training_data_npz["labels"]
            chromatograms = training_data_npz["chromatograms"]
            scores = training_data_npz["scores"]

            combined_chromatograms.append(chromatograms)
            combined_labels.append(labels)
            combined_scores.append(scores)

        combined_chromatograms = np.concatenate(combined_chromatograms)
        combined_labels = np.concatenate(combined_labels)
        combined_scores = np.concatenate(combined_scores)

        target_label_count = combined_labels[combined_labels == 1.0].size
        decoy_label_count = combined_labels[combined_labels == 0.0].size

        print(
            f"Target Peakgroups: {target_label_count}, Decoy Peakgroups: {decoy_label_count}"
        )

        if args.train_deep_chromatogram_model:

            print("Fitting chromatogram encoder.")

            chromatogram_encoder = self.train_deep_model(
                combined_chromatograms,
                combined_labels,
                args.chromatogram_encoder_output,
                args.threads,
                args.gpus,
                args.epochs,
            )

            print("Augmenting scores with learned chromatogram embeddings.")

            if args.use_only_chromatogram_features:

                print("Using only chromatogram features.")

            combined_scores = self.augment_score_columns(
                combined_chromatograms,
                combined_scores,
                chromatogram_encoder,
                args.use_only_chromatogram_features,
            )

        if args.chromatogram_encoder_input:

            print("Using pretrained encoder model.")

            chromatogram_encoder = DeepChromScorer(
                max_epochs=1, gpus=args.gpus, threads=args.threads
            )  # type: DeepChromScorer

            chromatogram_encoder.load(args.chromatogram_encoder_input)

            if args.use_only_chromatogram_features:
                print("Using only chromatogram features.")

            combined_scores = self.augment_score_columns(
                combined_chromatograms,
                combined_scores,
                chromatogram_encoder,
                args.use_only_chromatogram_features,
            )

        print("Training scoring model.")

        self.train_model(
            combined_data=combined_scores,
            combined_labels=combined_labels,
            model_output=args.model_output,
            scaler_output=args.scaler_output,
        )

    def augment_score_columns(
        self,
        combined_chromatograms,
        combined_scores,
        chromatogram_encoder,
        chromatogram_only=False,
    ):

        chromatogram_embeddings = chromatogram_encoder.encode(combined_chromatograms)

        if chromatogram_only:

            return chromatogram_embeddings

        else:

            return np.concatenate((combined_scores, chromatogram_embeddings), axis=1)

    def train_deep_model(
        self,
        combined_chromatograms,
        combined_labels,
        model_output,
        threads,
        gpus,
        epochs,
    ):

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_chromatograms, combined_labels, test_size=0.2, shuffle=True
        )

        model = DeepChromScorer(
            threads=threads,
            max_epochs=epochs,
            gpus=gpus,
        )

        print("Training model...")

        model.fit(data=training_data, labels=training_labels)

        print("Saving model...")

        model.save(model_output)

        print("Testing model...")

        roc_auc = model.evaluate(testing_data, testing_labels)

        print(f"ROC-AUC: {roc_auc}")

        return model

    def train_model(self, combined_data, combined_labels, model_output, scaler_output):

        scaler = Scaler()

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_data, combined_labels, test_size=0.2, shuffle=True
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
            "--chromatogram-encoder-input",
            dest="chromatogram_encoder_input",
            help="Input path for chromatogram encoder model.",
            default="",
        )

        self.parser.add_argument(
            "--chromatogram-encoder-output",
            dest="chromatogram_encoder_output",
            help="Output path for chromatogram encoder model.",
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
            help="Flag to indicate that a deep learning model should be trained on raw chromatograms.",
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
            "--threads",
            dest="threads",
            type=int,
            help="Number of threads/workers to use to train model.",
            default=1,
        )

        self.parser.add_argument(
            "--gpus",
            dest="gpus",
            type=int,
            help="Number of GPUs to use to train model.",
            default=1,
        )

        self.parser.add_argument(
            "--epochs",
            dest="epochs",
            type=int,
            help="Number of Epochs to use to train deep chrom model.",
            default=1,
        )

        self.parser.add_argument(
            "--early-stopping",
            dest="early_stopping",
            type=int,
            help="Number of epochs with unchanged error to stop training.",
            default=25,
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"
