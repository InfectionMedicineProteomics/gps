from __future__ import annotations

import importlib.resources
from csv import DictWriter
from typing import List, Dict, Union, Tuple, Optional, Generator, KeysView, Any

import numpy as np

from sklearn.metrics import precision_score, recall_score
from sklearn.utils import shuffle, class_weight

from gps import preprocess
from gps.chromatograms import Chromatogram, Chromatograms
from gps.models.deep_chromatogram_classifier import DeepChromScorer
from gps.scaler import Scaler
from gps.denoiser import BaggedDenoiser
from gps.fdr import ScoreDistribution, DecoyCounter
from gps.models.base_model import Scorer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gps.peakgroups import PeakGroup


class Precursor:
    id: str
    charge: int
    decoy: int
    target: int
    q_value: Optional[float]
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
        precursor_id: str = "",
        charge: int = 0,
        decoy: int = 0,
        q_value: Optional[float] = None,
        modified_sequence: str = "",
        unmodified_sequence: str = "",
        protein_accession: str = "",
        mz: float = 0.0,
    ) -> None:

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

    def set_chromatograms(self, chromatograms: Dict[str, Chromatogram]) -> None:

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
    pit: float

    def __init__(self) -> None:

        self.precursors = dict()
        self.pit = 1.0

    def __contains__(self, item: str) -> bool:

        return item in self.precursors

    def __iter__(self) -> Generator[Precursor, None, None]:

        for precursor_id, precursor in self.precursors.items():
            yield precursor

    def add_peakgroup(self, precursor_id: str, peakgroup: PeakGroup) -> None:

        self.precursors[precursor_id].peakgroups.append(peakgroup)

    def keys(self) -> KeysView[str]:

        return self.precursors.keys()

    def __setitem__(self, key: str, value: Precursor) -> None:

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

    def set_chromatograms(self, chromatograms: Chromatograms) -> None:

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
        precursor_list: Union[List[str], np.ndarray],
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
    ) -> List[PeakGroup]:

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

            elif filter_key == "CHROMATOGRAM_SCORE":

                precursor.peakgroups.sort(
                    key=lambda x: x.chromatogram_score, reverse=True
                )

            if rank + 1 <= len(precursor.peakgroups):

                peakgroup = precursor.peakgroups[rank]

                if peakgroup.target == 1 and peakgroup.vote_percentage >= value:
                    filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def filter_peakgroups(
        self,
        rank: int,
        filter_key: str = "PROBABILITY",
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

            elif filter_key == "CHROMATOGRAM_SCORE":

                precursor.peakgroups.sort(
                    key=lambda x: x.chromatogram_score, reverse=True
                )

            elif filter_key == "PEAKGROUP_SCORE":

                precursor.peakgroups.sort(key=lambda x: x.peakgroup_score, reverse=True)

            if rank + 1 <= len(precursor.peakgroups):

                peakgroup = precursor.peakgroups[rank]

                filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def get_decoy_peakgroups(
        self, filter_field: str = "PROBABILITY", use_second_ranked: bool = False
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
        base_estimator: Any = None,
        use_only_spectra_scores: bool = False,
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
            ) = preprocess.reformat_data(
                peakgroups=training_data_targets,
                use_only_spectra_scores=use_only_spectra_scores,
            )

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
                peakgroups=peakgroups_to_score,
                use_only_spectra_scores=use_only_spectra_scores,
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

                peakgroup.true_target_probability = probabilities[idx]

                peakgroup.vote_percentage = vote_percentages[idx]

            validation_data = self.get_peakgroups_by_list(
                precursor_list=precursor_fold_ids,
                rank=1,
                score_key="VAR_XCORR_SHAPE_WEIGHTED",
                reverse=True,
            )

            val_scores, val_labels, _ = preprocess.reformat_data(
                peakgroups=validation_data,
                use_only_spectra_scores=use_only_spectra_scores,
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

    def export_pin(
        self,
        pin_output_file: str,
        export_initial_pin: bool = False,
        use_sub_scores: bool = True,
        rank: int = 1,
    ):

        flagged_score_columns = self.flag_score_columns()

        print(flagged_score_columns)

        peakgroup_records = list()
        chromatograms = list()
        peptide_protein_ids = list()

        for precursor in self:

            precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

            if export_initial_pin:

                peakgroups = precursor.peakgroups

            else:

                peakgroups = precursor.peakgroups[:rank]

            for peakgroup in peakgroups:

                peakgroup_record: Dict[str, Union[str, int, float]] = {
                    "id": f"{precursor.modified_sequence}_{precursor.charge}_{peakgroup.idx}",
                    "label": 1 if peakgroup.target else -1,
                    "scannr": peakgroup.idx,
                }

                if use_sub_scores:

                    peakgroup_record.update(
                        peakgroup.get_score_columns(flagged_score_columns)
                    )

                peptide_protein_ids.append(
                    {
                        "peptide": precursor.modified_sequence,
                        "proteinId1": precursor.protein_accession,
                    }
                )

                peakgroup_records.append(peakgroup_record)

        for peakgroup_idx, peakgroup_record in enumerate(peakgroup_records):

            peakgroup_record.update(peptide_protein_ids[peakgroup_idx])

        field_names = list(peakgroup_records[0].keys())

        with open(pin_output_file, "w") as out_file:

            csv_writer = DictWriter(out_file, delimiter="\t", fieldnames=field_names)

            csv_writer.writeheader()

            for peakgroup_record in peakgroup_records:
                csv_writer.writerow(peakgroup_record)

    def flag_score_columns(self) -> List[str]:

        score_names = (
            self.precursors[list(self.precursors.keys())[0]]
            .peakgroups[0]
            .get_sub_score_column_names()
        )

        score_columns = {score_name: list() for score_name in score_names}

        all_peakgroups = self.get_all_peakgroups()

        flagged_columns = list()

        for peakgroup in all_peakgroups:

            scores = peakgroup.scores

            for score_name in score_names:
                score_columns[score_name].append(scores[score_name])

        for score_column, score_values in score_columns.items():

            score_array = np.array(score_values, dtype=np.float64)

            if np.all(score_array == 0) or np.all(np.isnan(score_array)):
                flagged_columns.append(score_column)

        return flagged_columns

    def score_run(
        self,
        model_path: str,
        scaler_path: str,
        threads: int = 10,
        weight_scores: bool = False,
        use_only_spectra_scores: bool = False,
    ) -> Precursors:

        scoring_model: Union[Scorer]

        all_peakgroups = self.get_all_peakgroups()

        (all_scores, all_data_labels, all_data_indices,) = preprocess.reformat_data(
            all_peakgroups, use_only_spectra_scores=use_only_spectra_scores
        )

        scoring_model = Scorer()

        if not model_path:

            model_path = importlib.resources.path("gps.models.trained", "scoring.model")
            scaler_path = importlib.resources.path("gps.models.trained", "scoring.scaler")

        scoring_model.load(model_path)

        pipeline = Scaler()

        pipeline.load(scaler_path)

        all_scores = pipeline.transform(all_scores)

        model_scores = scoring_model.score(all_scores)

        if weight_scores:

            target_probabilities = preprocess.get_probability_vector(all_peakgroups)

            model_scores = np.exp(target_probabilities) * model_scores

        model_probabilities = scoring_model.probability(all_scores)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.d_score = model_scores[idx].item()

            peakgroup.probability = model_probabilities[idx].item()

        return self

    def predict_chromatograms(
        self, model_path: str = "", threads: int = 10, gpus: int = 1
    ) -> Precursors:

        all_peakgroups = self.get_all_peakgroups()

        (
            all_data_labels,
            all_data_indices,
            all_chromatograms,
            all_scores,
        ) = preprocess.reformat_chromatogram_data(
            all_peakgroups,
            training=False,
        )

        scoring_model = DeepChromScorer(max_epochs=1, gpus=gpus, threads=threads)

        scoring_model.load(model_path)

        model_scores = scoring_model.score(all_chromatograms)

        model_predictions = 1.0 / (1.0 + np.exp(-model_scores))

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.chromatogram_prediction = model_predictions[idx].item()
            peakgroup.chromatogram_score = model_scores[idx].item()

        return self

    def predict_peakgroups(
        self,
        model_path: str = "",
        scaler_path: str = "",
        method: str = "",
        threads: int = 10,
        gpus: int = 1,
    ) -> Precursors:

        all_peakgroups = self.get_all_peakgroups()

        if method == "standard":

            (all_scores, all_data_labels, all_data_indices) = preprocess.reformat_data(
                all_peakgroups
            )

            scoring_model = Scorer()

            scoring_model.load(model_path)

            pipeline = Scaler()

            pipeline.load(scaler_path)

            all_scores = pipeline.transform(all_scores)

            model_scores = scoring_model.score(all_scores)

            model_predictions = scoring_model.predict(all_scores)

            for idx, peakgroup in enumerate(all_peakgroups):
                peakgroup.peakgroup_prediction = model_predictions[idx].item()
                peakgroup.peakgroup_score = model_scores[idx].item()

        elif method == "denoise":

            self.denoise(
                num_folds=10,
                num_classifiers=10,
                num_threads=threads,
                vote_percentage=0.5,
                verbose=True,
                use_only_spectra_scores=True,
            )

            for idx, peakgroup in enumerate(all_peakgroups):

                peakgroup.peakgroup_prediction = peakgroup.vote_percentage
                peakgroup.peakgroup_score = peakgroup.true_target_score

        return self

    def estimate_pit(self) -> float:

        all_peakgroups = self.filter_target_peakgroups(
            rank=1, filter_key="D_SCORE", value=0.0
        )

        true_target_peakgroups = []

        false_target_peakgroups = []

        for peakgroup in all_peakgroups:

            if peakgroup.vote_percentage >= 1.0:

                true_target_peakgroups.append(peakgroup)

            elif peakgroup.vote_percentage < 1.0:

                false_target_peakgroups.append(peakgroup)

        decoy_peakgroups = self.get_decoy_peakgroups(
            filter_field="D_SCORE", use_second_ranked=False
        )

        self.pit = len(false_target_peakgroups) / len(decoy_peakgroups)

        return self.pit

    def calculate_q_values(
        self,
        sort_key: str,
        decoy_free: bool = False,
        count_decoys: bool = True,
        num_threads: int = 10,
        pit: float = 1.0,
        debug: bool = False,
    ) -> np.ndarray:

        target_peakgroups = self.get_target_peakgroups_by_rank(
            rank=1, score_key=sort_key, reverse=True
        )

        if not decoy_free:

            decoy_peakgroups = self.get_decoy_peakgroups(filter_field="D_SCORE")

        else:

            print("Decoy free scoring")

            all_decoy_peakgroups = self.get_target_peakgroups_by_rank(
                rank=2, score_key=sort_key, reverse=True
            )

            decoy_peakgroups = []

            for decoy in all_decoy_peakgroups:

                if decoy.vote_percentage < 1.0:
                    decoy_peakgroups.append(decoy)

            for peakgroup in decoy_peakgroups:
                peakgroup.target = 0
                peakgroup.decoy = 1

        if count_decoys:

            all_peakgroups = self.get_all_peakgroups()

            all_data_scores, all_data_labels = preprocess.reformat_distribution_data(
                all_peakgroups
            )

            q_values = DecoyCounter(num_threads=num_threads, pit=pit).calc_q_values(
                all_data_scores, all_data_labels
            )

        else:

            modelling_peakgroups = target_peakgroups + decoy_peakgroups

            scores, labels = preprocess.reformat_distribution_data(modelling_peakgroups)

            self.score_distribution = ScoreDistribution(pit=pit)

            self.score_distribution.fit(scores, labels)

            all_peakgroups = self.get_all_peakgroups()

            all_data_scores, all_data_labels = preprocess.reformat_distribution_data(
                all_peakgroups
            )

            q_values = self.score_distribution.calculate_q_values(all_data_scores)

        for idx, peakgroup in enumerate(all_peakgroups):
            peakgroup.q_value = q_values[idx].item()

        return q_values

    def dump_training_data(
        self, file_path: str, filter_field: str, filter_value: float
    ) -> None:

        positive_labels = self.filter_target_peakgroups(
            rank=1, filter_key=filter_field, value=filter_value
        )

        negative_labels = self.get_decoy_peakgroups(use_second_ranked=False)

        combined = positive_labels + negative_labels

        (all_scores, all_data_labels, all_data_indices) = preprocess.reformat_data(
            combined
        )

        with open(file_path, "wb") as npfh:
            np.savez(
                npfh,
                labels=all_data_labels,
                scores=all_scores,
            )

    def write_tsv(
        self,
        file_path: str = "",
        ranked: int = 1,
        write_predicted: bool = False,
        write_percolator: bool = False,
    ) -> None:

        if write_predicted:

            field_names = [
                "PrecursorMz",
                "UnmodifiedSequence",
                "ModifiedSequence",
                "Charge",
                "Protein",
                "Decoy",
                "RT",
                "Intensity",
                "PeakgroupPrediction",
                "PeakgroupScore",
            ]

            with open(file_path, "w") as out_file:

                csv_writer = DictWriter(
                    out_file, delimiter="\t", fieldnames=field_names
                )

                csv_writer.writeheader()

                for precursor in self:

                    precursor.peakgroups.sort(
                        key=lambda x: x.peakgroup_score, reverse=True
                    )

                    peakgroups = precursor.peakgroups[:ranked]

                    for peakgroup in peakgroups:

                        record = {
                            "PrecursorMz": precursor.mz,
                            "UnmodifiedSequence": precursor.unmodified_sequence,
                            "ModifiedSequence": precursor.modified_sequence,
                            "Charge": precursor.charge,
                            "Protein": precursor.protein_accession,
                            "Decoy": precursor.decoy,
                            "RT": peakgroup.retention_time,
                            "Intensity": peakgroup.intensity,
                            "PeakgroupPrediction": peakgroup.peakgroup_prediction,
                            "PeakgroupScore": peakgroup.peakgroup_score,
                        }

                        csv_writer.writerow(record)

        else:

            field_names = [
                "PrecursorMz",
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
                "Rank",
                "VotePercentage",
                "TargetProbability",
            ]

            with open(file_path, "w") as out_file:

                csv_writer = DictWriter(
                    out_file, delimiter="\t", fieldnames=field_names
                )

                csv_writer.writeheader()

                for precursor in self:

                    peakgroups = []

                    if write_percolator:

                        for peakgroup in precursor.peakgroups:

                            if peakgroup.top_scoring:

                                peakgroups.append(peakgroup)

                    else:

                        precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

                        peakgroups = precursor.peakgroups[:ranked]

                    for rank_idx, peakgroup in enumerate(peakgroups):
                        rank = rank_idx + 1

                        record = {
                            "PrecursorMz": precursor.mz,
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
                            "Rank": rank,
                            "VotePercentage": peakgroup.vote_percentage,
                            "TargetProbability": peakgroup.true_target_probability,
                        }

                        csv_writer.writerow(record)
