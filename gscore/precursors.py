from __future__ import annotations

from collections import Counter
from csv import DictWriter
from typing import List, Dict, Union, Tuple, Optional

import numpy as np
from sklearn.metrics import precision_score, recall_score  # type: ignore
from sklearn.utils import shuffle, class_weight  # type: ignore


from gscore import preprocess
from gscore.chromatograms import Chromatogram
from gscore.models.deep_chromatogram_classifier import DeepChromScorer
from gscore.scaler import Scaler
from gscore.denoiser import BaggedDenoiser
from gscore.fdr import ScoreDistribution, DecoyCounter
from gscore.models.base_model import Scorer

from typing import TYPE_CHECKING

from gscore.scorer import XGBoostScorer

if TYPE_CHECKING:
    from gscore.peakgroups import PeakGroup


class Precursor:

    id: str
    charge: int
    decoy: int
    target: int
    q_value: float
    peakgroups: List[PeakGroup]
    scores: Dict[str, float]
    modified_sequence: str
    unmodified_sequence: str
    protein_accession: str
    mz: float
    chromatograms: Union[dict[str, Chromatogram], None]
    probability: float

    def __init__(
        self,
        precursor_id="",
        charge=0,
        decoy=0,
        q_value=None,
        modified_sequence="",
        unmodified_sequence="",
        protein_accession="",
        mz=0.0,
    ):

        self.id = precursor_id
        self.charge = charge
        self.decoy = decoy
        self.target = abs(decoy - 1)
        self.q_value = q_value
        self.peakgroups = []
        self.scores = dict()
        self.modified_sequence = modified_sequence
        self.unmodified_sequence = unmodified_sequence
        self.protein_accession = protein_accession
        self.mz = mz
        self.chromatograms = None

    def set_chromatograms(self, chromatograms: Dict[str, Chromatogram]):

        self.chromatograms = chromatograms

    def get_chromatograms(self, min_rt: float, max_rt: float) -> np.ndarray:

        chrom_list = []

        if self.chromatograms:

            for idx, chromatogram in enumerate(self.chromatograms.values()):

                if idx == 0:

                    chrom_list.append(
                        chromatogram.scaled_rts(min_val=min_rt, max_val=max_rt)
                    )

                chrom_list.append(
                    chromatogram.normalized_intensities(add_min_max=(0.0, 10.0))
                )

        return np.array(chrom_list)

    def get_peakgroup(self, rank: int, key: str, reverse: bool = False) -> PeakGroup:

        if key == "Q_VALUE":

            self.peakgroups.sort(key=lambda x: x.q_value, reverse=reverse)

        if key == "D_SCORE":

            self.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

        else:

            self.peakgroups.sort(key=lambda x: x.scores[key], reverse=reverse)

        rank = rank - 1

        return self.peakgroups[rank]


class Precursors:

    precursors: Dict[str, Precursor]

    def __init__(self):

        self.precursors = dict()

    def __contains__(self, item):

        return item in self.precursors

    def __iter__(self):

        for precursor_id, precursor in self.precursors.items():
            yield precursor

    def add_peakgroup(self, precursor_id: str, peakgroup: PeakGroup) -> None:

        self.precursors[precursor_id].peakgroups.append(peakgroup)

    def keys(self):

        return self.precursors.keys()

    def __setitem__(self, key: str, value: Precursor):

        self.precursors[key] = value

    def __getitem__(self, item: str) -> Precursor:

        return self.precursors[item]

    def get_rt_bounds(self) -> Tuple[float, float]:

        rts = list()

        for precursor in self.precursors.values():

            for peakgroup in precursor.peakgroups:
                rts.append(peakgroup.end_rt)
                rts.append(peakgroup.start_rt)

        return (np.min(rts), np.max(rts))

    def set_chromatograms(self, chromatograms):

        out_of_bounds = 0

        for precursor in self:

            precursor_chromatograms = chromatograms.get(
                f"{precursor.mz}_{precursor.charge}", precursor.unmodified_sequence
            )

            if precursor_chromatograms:

                for peakgroup in precursor.peakgroups:

                    peakgroup_chromatograms = dict()

                    start_rt = peakgroup.start_rt
                    end_rt = peakgroup.end_rt

                    new_rt_steps = np.linspace(start_rt, end_rt, 25)

                    for key, chrom in precursor_chromatograms.items():

                        peakgroup_chromatogram = Chromatogram(
                            type="peakgroup",
                            chrom_id=key,
                            precursor_mz=peakgroup.mz,
                            intensities=chrom.interpolated_chromatogram(new_rt_steps),
                            rts=new_rt_steps,
                            start_rt=peakgroup.start_rt,
                            end_rt=peakgroup.end_rt,
                        )

                        peakgroup_chromatograms[key] = peakgroup_chromatogram

                        peakgroup.chromatograms = peakgroup_chromatograms

    def get_peakgroups_by_list(
        self,
        precursor_list: np.ndarray,
        rank: int = 0,
        score_key: str = "",
        reverse: bool = True,
        return_all: bool = False,
    ) -> List[PeakGroup]:

        peakgroups = list()

        for precursor_key in precursor_list:

            precursor = self.precursors[precursor_key]

            if not return_all:

                peakgroup = precursor.get_peakgroup(
                    rank=1, key=score_key, reverse=reverse
                )

                peakgroups.append(peakgroup)

            else:

                for peakgroup in precursor.peakgroups:
                    peakgroups.append(peakgroup)

        return peakgroups

    def get_target_peakgroups_by_rank(
        self, rank: int, score_key: str = "", reverse: bool = True
    ):

        filtered_peakgroups = []

        for precursor in self.precursors.values():

            precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=reverse)

            if (rank) <= len(precursor.peakgroups):

                peakgroup = precursor.peakgroups[rank - 1]

                if peakgroup.target == 1:

                    filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def filter_target_peakgroups(
        self, rank: int, filter_key: str = "PROBABILITY", value: float = 0.0
    ) -> List[PeakGroup]:

        filtered_peakgroups = []

        rank = rank - 1

        for precursor in self.precursors.values():

            if filter_key == "PROBABILITY":

                precursor.peakgroups.sort(key=lambda x: x.probability, reverse=True)

            elif filter_key == "D_SCORE":

                precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

            elif filter_key == "TRUE_TARGET_SCORE":

                precursor.peakgroups.sort(
                    key=lambda x: x.true_target_score, reverse=True
                )

            if rank + 1 <= len(precursor.peakgroups):

                peakgroup = precursor.peakgroups[rank]

                if peakgroup.target == 1 and peakgroup.vote_percentage >= value:

                    filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def get_decoy_peakgroups(
        self, filter_field="PROBABILITY", use_second_ranked: bool = False
    ) -> List[PeakGroup]:

        filtered_peakgroups = []

        for precursor in self.precursors.values():

            if filter_field == "PROBABILITY":

                precursor.peakgroups.sort(key=lambda x: x.probability, reverse=True)

            elif filter_field == "D_SCORE":

                precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

            if use_second_ranked and len(precursor.peakgroups) > 1:

                peakgroup = precursor.peakgroups[1]

                if peakgroup.target == 1:
                    peakgroup.target = 0
                    peakgroup.decoy = 1

                    filtered_peakgroups.append(peakgroup)

            else:

                if precursor.target == 0:
                    peakgroup = precursor.peakgroups[0]

                    filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def get_all_peakgroups(self) -> List[PeakGroup]:

        all_peakgroups = []

        for precursor in self.precursors.values():

            for peakgroup in precursor.peakgroups:
                all_peakgroups.append(peakgroup)

        return all_peakgroups

    def denoise(
        self,
        num_folds: int,
        num_classifiers: int,
        num_threads: int,
        vote_percentage: float,
        verbose: bool = False,
        base_estimator=None,
    ) -> Precursors:

        precursor_folds = preprocess.get_precursor_id_folds(
            list(self.keys()), num_folds
        )

        total_recall = []
        total_precision = []

        for fold_num, precursor_fold_ids in enumerate(precursor_folds):

            if verbose:
                print(f"Processing fold {fold_num + 1}...")

            training_precursors = preprocess.get_training_data(
                folds=precursor_folds, fold_num=fold_num
            )

            # TODO: Pick better way to initially rank features
            training_data_targets = self.get_peakgroups_by_list(
                precursor_list=training_precursors,
                rank=1,
                score_key="VAR_XCORR_SHAPE_WEIGHTED",
                reverse=True,
            )

            (
                peakgroup_scores,
                peakgroup_labels,
                peakgroup_indices,
            ) = preprocess.reformat_data(peakgroups=training_data_targets)

            train_data, train_labels = shuffle(
                peakgroup_scores, peakgroup_labels, random_state=42
            )

            scaler = Scaler()

            train_data = scaler.fit_transform(train_data)

            n_samples = int(len(train_data) * 1.0)

            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(train_labels),
                y=train_labels.ravel(),
            )

            if base_estimator:

                denoizer = BaggedDenoiser(
                    base_estimator=base_estimator,
                    max_samples=n_samples,
                    n_estimators=num_classifiers,
                    n_jobs=num_threads,
                    random_state=42,
                    class_weights=class_weights,
                )

            else:

                denoizer = BaggedDenoiser(
                    max_samples=n_samples,
                    n_estimators=num_classifiers,
                    n_jobs=num_threads,
                    random_state=42,
                    class_weights=class_weights,
                )

            denoizer.fit(train_data, train_labels.ravel())

            peakgroups_to_score = self.get_peakgroups_by_list(
                precursor_list=precursor_fold_ids, return_all=True
            )

            testing_scores, testing_labels, testing_keys = preprocess.reformat_data(
                peakgroups=peakgroups_to_score
            )

            testing_scores = scaler.transform(testing_scores)

            class_index = np.where(denoizer.classes_ == 1.0)[0][0]

            vote_percentages = denoizer.vote(testing_scores, threshold=vote_percentage)

            probabilities = denoizer.predict_proba(testing_scores)[:, class_index]

            true_target_scores = denoizer.decision_function(testing_scores)

            if verbose:
                print(
                    "Updating peakgroups", len(probabilities), len(peakgroups_to_score)
                )

            for idx, peakgroup in enumerate(peakgroups_to_score):

                peakgroup.true_target_score = true_target_scores[idx]

                peakgroup.probability = probabilities[idx]

                peakgroup.vote_percentage = vote_percentages[idx]

            validation_data = self.get_peakgroups_by_list(
                precursor_list=precursor_fold_ids,
                rank=1,
                score_key="VAR_XCORR_SHAPE_WEIGHTED",
                reverse=True,
            )

            val_scores, val_labels, _ = preprocess.reformat_data(
                peakgroups=validation_data
            )

            val_scores = scaler.transform(val_scores)

            fold_precision = precision_score(
                y_pred=denoizer.predict(val_scores), y_true=val_labels.ravel()
            )

            fold_recall = recall_score(
                y_pred=denoizer.predict(val_scores), y_true=val_labels.ravel()
            )

            total_recall.append(fold_recall)
            total_precision.append(fold_precision)

            if verbose:
                print(
                    f"Fold {fold_num + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
                )

        print(
            f"Mean recall: {np.mean(total_recall)}, Mean precision: {np.mean(total_precision)}"
        )

        return self

    def score_run(
        self,
        model_path: str,
        scaler_path: str,
        encoder_path: str = None,
        threads: int = 10,
        gpus: int = 1,
        use_relative_intensities: bool = False,
        chromatogram_only: bool = False,
    ):

        scoring_model: Optional[Scorer]

        pipeline = Scaler()

        pipeline.load(scaler_path)

        all_peakgroups = self.get_all_peakgroups()

        (
            all_data_labels,
            all_data_indices,
            all_chromatograms,
            all_scores,
        ) = preprocess.reformat_chromatogram_data(
            all_peakgroups,
            use_relative_intensities=use_relative_intensities,
            training=False,
        )

        if encoder_path:

            chromatogram_encoder = DeepChromScorer(
                max_epochs=1, gpus=gpus, threads=threads
            )  # type: DeepChromScorer

            chromatogram_encoder.load(encoder_path)

            chromatogram_embeddings = chromatogram_encoder.encode(all_chromatograms)

            if chromatogram_only:

                print("Only using chromatograms.")

                all_scores = chromatogram_embeddings

            else:

                all_scores = np.concatenate(
                    (all_scores, chromatogram_embeddings), axis=1
                )

        counter: Counter = Counter(all_data_labels.ravel())
        scale_pos_weight = counter[0] / counter[1]

        scoring_model = Scorer()

        scoring_model.load(model_path)

        all_scores = pipeline.transform(all_scores)

        model_scores = scoring_model.score(all_scores)

        model_probabilities = scoring_model.probability(all_scores)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.d_score = model_scores[idx].item()

            peakgroup.probability = model_probabilities[idx].item()

        return self

    def calculate_q_values(self, sort_key: str, decoy_free: bool = False, count_decoys=True, num_threads: int = 10):

        target_peakgroups = self.get_target_peakgroups_by_rank(
            rank=1, score_key=sort_key, reverse=True
        )

        if not decoy_free:

            decoy_peakgroups = self.get_decoy_peakgroups(filter_field="D_SCORE")

        else:

            print("Decoy free scoring")

            decoy_peakgroups = self.get_target_peakgroups_by_rank(
                rank=2, score_key=sort_key, reverse=True
            )

            decoy_free_labels = []

            for peakgroup in decoy_peakgroups:

                peakgroup.target = 0
                peakgroup.decoy = 1

        if count_decoys:

            all_peakgroups = self.get_all_peakgroups()

            all_data_scores, all_data_labels = preprocess.reformat_distribution_data(
                all_peakgroups
            )

            q_values = DecoyCounter(num_threads=num_threads).calc_q_values(
                all_data_scores,
                all_data_labels
            )

        else:

            modelling_peakgroups = target_peakgroups + decoy_peakgroups

            scores, labels = preprocess.reformat_distribution_data(modelling_peakgroups)

            self.score_distribution = ScoreDistribution(scale=False, smooth=False)

            self.score_distribution.fit(scores, labels)

            all_peakgroups = self.get_all_peakgroups()

            all_data_scores, all_data_labels = preprocess.reformat_distribution_data(
                all_peakgroups
            )

            q_values = self.score_distribution.calculate_q_values(all_data_scores)

        for idx, peakgroup in enumerate(all_peakgroups):
            peakgroup.q_value = q_values[idx].item()

        return self

    def dump_training_data(
        self,
        file_path: str,
        filter_field: str,
        filter_value: float,
        use_relateive_intensities: bool = False,
    ) -> None:

        positive_labels = self.filter_target_peakgroups(
            rank=1, filter_key=filter_field, value=filter_value
        )

        negative_labels = self.get_decoy_peakgroups(use_second_ranked=False)

        combined = positive_labels + negative_labels

        (
            all_data_labels,
            all_data_indices,
            all_chromatograms,
            all_scores,
        ) = preprocess.reformat_chromatogram_data(
            combined, use_relative_intensities=use_relateive_intensities
        )

        with open(file_path, "wb") as npfh:
            np.savez(
                npfh,
                labels=all_data_labels,
                chromatograms=all_chromatograms,
                scores=all_scores,
            )

    def write_tsv(self, file_path: str = "", ranked: int = 1) -> None:

        field_names = [
            "UnmodifiedSequence",
            "ModifiedSequence",
            "Charge",
            "Protein",
            "Decoy",
            "RT",
            "Intensity",
            "QValue",
            "DScore",
            "Probability",
        ]

        with open(file_path, "w") as out_file:

            csv_writer = DictWriter(out_file, delimiter="\t", fieldnames=field_names)

            csv_writer.writeheader()

            for precursor in self:

                precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

                peakgroups = precursor.peakgroups[:ranked]

                for peakgroup in peakgroups:

                    record = {
                        "UnmodifiedSequence": precursor.unmodified_sequence,
                        "ModifiedSequence": precursor.modified_sequence,
                        "Charge": precursor.charge,
                        "Protein": precursor.protein_accession,
                        "Decoy": precursor.decoy,
                        "RT": peakgroup.retention_time,
                        "Intensity": peakgroup.intensity,
                        "QValue": peakgroup.q_value,
                        "DScore": peakgroup.d_score,
                        "Probability": peakgroup.probability,
                    }

                    csv_writer.writerow(record)
