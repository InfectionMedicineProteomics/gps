import argparse
from collections import Counter
from typing import Any, List, Dict
from csv import DictReader, DictWriter

from concurrent.futures import ProcessPoolExecutor
import itertools
from subprocess import Popen, PIPE, STDOUT

from pathlib import Path

import numpy as np
import typing

from sklearn.model_selection import train_test_split

from gps import preprocess
from gps.models.deep_chromatogram_classifier import DeepChromScorer
from gps.scaler import Scaler
from gps.scorer import XGBoostScorer


def read_percolator_files(pin_file: str) -> List[Dict[str, Any]]:
    percolator_records = []

    print(f"Processing {pin_file}")

    with open(pin_file) as pin_file_h:
        reader = DictReader(pin_file_h, delimiter="\t")

        for row in reader:
            row["id"] = f"{Path(pin_file).name}_{row['id']}"

            percolator_records.append(row)

    return percolator_records


def train_percolator_model(args: argparse.Namespace) -> None:

    with ProcessPoolExecutor(max_workers=args.threads) as pool:

        percolator_records = pool.map(read_percolator_files, args.input_files)

    percolator_records = list(itertools.chain.from_iterable(percolator_records))

    print(len(percolator_records))

    field_names = list(percolator_records[0].keys())

    print(f"Writing {args.percolator_output}")

    with open(args.percolator_output, "w") as perc_output_h:

        writer = DictWriter(perc_output_h, fieldnames=field_names, delimiter="\t")

        writer.writeheader()

        for row in percolator_records:
            writer.writerow(row)

    print("Training percolator model...")

    pin_path = Path(args.percolator_output)

    with Popen(
        [
            args.percolator_exe,
            args.percolator_output,
            "--results-psms",
            f"{pin_path.parent}/{pin_path.name}_results_psms.tsv",
            "--decoy-results-psms",
            f"{pin_path.parent}/{pin_path.name}_decoy_results_psms.tsv",
            "--protein-decoy-pattern",
            "DECOY_",
            "--num-threads",
            str(args.threads),
            "--only-psms",
            "--weights",
            args.model_output,
        ],
        stdout=PIPE,
        stderr=STDOUT,
    ) as process:

        for line in iter(process.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))


def augment_score_columns(
    combined_chromatograms: np.ndarray,
    combined_scores: np.ndarray,
    chromatogram_encoder: DeepChromScorer,
    chromatogram_only: bool = False,
) -> np.ndarray:

    chromatogram_embeddings = chromatogram_encoder.encode(combined_chromatograms)

    if chromatogram_only:

        return chromatogram_embeddings

    else:

        return np.concatenate((combined_scores, chromatogram_embeddings), axis=1)


def train_deep_model(
    combined_chromatograms: np.ndarray,
    combined_labels: np.ndarray,
    model_output: str,
    threads: int,
    gpus: int,
    epochs: int,
) -> DeepChromScorer:

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


def train_model(
    combined_data: np.ndarray,
    combined_labels: np.ndarray,
    model_output: str,
    scaler_output: str,
    no_split: bool = False,
) -> None:

    scaler = Scaler()

    if no_split:

        training_data = combined_data
        training_labels = combined_labels

    else:

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            combined_data, combined_labels, test_size=0.2, shuffle=True
        )

    counter: typing.Counter[int] = Counter(training_labels.ravel())
    scale_pos_weight = counter[0] / counter[1]

    scorer = XGBoostScorer(scale_pos_weight=scale_pos_weight)

    training_data = scaler.fit_transform(training_data)

    print("Training model...")

    scorer.fit(training_data, training_labels.ravel())

    if not no_split:

        testing_data = scaler.transform(testing_data)

        print("Evaluating model...")

        roc = scorer.evaluate(testing_data, testing_labels)

        print(f"Model ROC-AUC: {roc}")

    scorer.save(model_output)
    scaler.save(scaler_output)


def train_gps_model(args: argparse.Namespace) -> None:

    print("Building scoring model...")

    combined_labels_list = []
    combined_scores_list = []

    print("Loading data...")

    for input_file in args.input_files:
        training_data_npz = preprocess.get_training_data_from_npz(input_file)

        labels = training_data_npz["labels"]
        scores = training_data_npz["scores"]

        combined_labels_list.append(labels)
        combined_scores_list.append(scores)

    combined_labels = np.concatenate(combined_labels_list)
    combined_scores = np.concatenate(combined_scores_list)

    target_label_count = combined_labels[combined_labels == 1.0].size
    decoy_label_count = combined_labels[combined_labels == 0.0].size

    print(
        f"Target Peakgroups: {target_label_count}, Decoy Peakgroups: {decoy_label_count}"
    )

    print("Training scoring model.")

    train_model(
        combined_data=combined_scores,
        combined_labels=combined_labels,
        model_output=args.model_output,
        scaler_output=args.scaler_output,
        no_split=args.nosplit,
    )


class Train:
    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "train"

    def __call__(self, args: argparse.Namespace) -> None:

        if args.train_percolator_model:

            train_percolator_model(args)

        else:

            train_gps_model(args)

    def build_subparser(self, subparser: Any) -> None:

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
            "--percolator-output",
            dest="percolator_output",
            help="Output path for percolator training data.",
        )

        self.parser.add_argument(
            "--train-percolator-model",
            dest="train_percolator_model",
            help="flag to indicate Percolator should be used to train the model",
            action="store_true",
        )

        self.parser.add_argument(
            "--percolator-exe", dest="percolator_exe", help="Percolator exe file path"
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            type=int,
            help="Number of threads/workers to use to train model.",
            default=1,
        )

        self.parser.add_argument(
            "--nosplit",
            dest="nosplit",
            help="Do not split out test set and evaluate the model",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:

        return f"<Train> {self.name}"
