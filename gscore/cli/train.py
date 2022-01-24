import argparse
from collections import Counter

import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils import class_weight  # type: ignore

from gscore import preprocess
from gscore.models.deep_chromatogram_classifier import DeepChromModel
from gscore.scaler import Scaler
from gscore.scorer import XGBoostScorer

import torch

from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import Trainer

class Train:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "train"

    def __call__(self, args: argparse.Namespace):

        print("Building scoring model...")

        combined_data = []
        combined_labels = []
        combined_chromatograms = []

        print("Loading data...")

        for input_file in args.input_files:

            training_data_npz = preprocess.get_training_data_from_npz(input_file)


            if "chromatograms" in training_data_npz:

                scores = training_data_npz["scores"]
                labels = training_data_npz["labels"]

                chromatograms = torch.from_numpy(training_data_npz["chromatograms"]).type(torch.FloatTensor)
                labels = torch.from_numpy(training_data_npz["labels"]).type(torch.FloatTensor)
                scores = torch.from_numpy(training_data_npz["scores"]).type(torch.FloatTensor)

                combined_chromatograms.append(chromatograms)
                combined_data.append(scores)
                combined_labels.append(labels)

            else:

                scores = training_data_npz["scores"]
                labels = training_data_npz["labels"]

                combined_data.append(scores)
                combined_labels.append(labels)

        if args.train_deep_chromatogram_model:

            combined_data = torch.cat(combined_data, 0)
            combined_labels = torch.cat(combined_labels, 0)
            combined_chromatograms = torch.cat(combined_chromatograms, 0)

            self.train_deep_model(
                combined_chromatograms,
                combined_labels,
                args.model_output,
                args.threads
            )

        else:

            combined_data = np.concatenate(combined_data)
            combined_labels = np.concatenate(combined_labels)

            self.train_model(
                combined_data,
                combined_labels,
                args.model_output,
                args.scaler_output
            )

    def train_deep_model(self, combined_chromatograms, combined_labels, model_output, threads):

        chromatogram_dataset = TensorDataset(combined_chromatograms, combined_labels)

        train_length = int(0.7 * len(chromatogram_dataset))

        test_length = len(chromatogram_dataset) - train_length

        train_dataset, test_dataset = torch.utils.data.random_split(
            chromatogram_dataset,
            (train_length, test_length)
        )

        validation_length = int(0.1 * len(train_dataset))

        train_length = len(train_dataset) - validation_length

        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset,
            (train_length, validation_length)
        )

        print(f"Training dataset: {train_length}, Validation dataset: {validation_length}, Testing dataset: {test_length}")

        chromatogram_dataloader = DataLoader(
            train_dataset,
            batch_size=100,
            shuffle=True,
            num_workers=threads
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=100,
            num_workers=10
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=100,
            num_workers=10
        )

        model = DeepChromModel()

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=1000,
            gpus=1,
            callbacks=[
                lr_monitor,
                early_stop_callback
            ]
        )

        print("Training model...")

        trainer.fit(
            model,
            train_dataloaders=chromatogram_dataloader,
            val_dataloaders=validation_dataloader
        )

        print("Testing model...")

        trainer.test(dataloaders=test_dataloader)

        print("Saving model...")

        trainer.save_checkpoint(
            model_output
        )


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
            "--threads",
            dest="threads",
            type=int,
            help="Number of threads/workers to use to train model.",
            default=1
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"
