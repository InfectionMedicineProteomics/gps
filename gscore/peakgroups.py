from __future__ import annotations

from typing import List, Dict, Union, Tuple

import numpy as np

from sklearn.utils import shuffle, class_weight  # type: ignore
from sklearn.metrics import precision_score, recall_score  # type: ignore

from gscore import preprocess
from gscore.chromatograms import Chromatograms, Chromatogram
from gscore.models.deep_chromatogram_classifier import DeepChromScorer
from gscore.scaler import Scaler
from gscore.denoiser import BaggedDenoiser
from gscore.scorer import Scorer
from gscore.fdr import ScoreDistribution


class PeakGroup:
    ghost_score_id: str
    idx: str
    mz: float
    retention_time: float
    intensity: float
    decoy: int
    target: int
    delta_rt: float
    start_rt: float
    end_rt: float
    chromatograms: Union[Dict[str, Chromatogram], None]

    def __init__(
        self,
        ghost_score_id="",
        idx="",
        mz=0.0,
        rt=0.0,
        q_value=None,
        intensity=0.0,
        decoy=0,
        delta_rt=0.0,
        start_rt=0.0,
        end_rt=0.0,
    ):

        self.ghost_score_id = ghost_score_id
        self.idx = idx

        self.retention_time = rt
        self.intensity = intensity
        self.q_value = q_value

        self.mz = mz
        self.intensity = intensity

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.scores = dict()

        self.delta_rt = delta_rt
        self.start_rt = start_rt
        self.end_rt = end_rt

        self.chromatograms = None

        self.scaled_rt_start = 0.0
        self.scaled_rt_apex = 0.0
        self.scaled_rt_end = 0.0

    def __repr__(self):

        return f"{self.mz=} {self.retention_time=} {self.decoy=} {self.scores=}"

    def get_chromatogram_rt_array(self, interpolated=False, num_rt_steps=25):

        chromatogram = list(self.chromatograms.values())[0]

        if interpolated:

            return chromatogram.interpolated_rt(num_steps=25)

        return chromatogram.rts

    def get_chromatogram_intensity_arrays(self, scaled=False, interpolated=False, num_rt_steps=25,
                                          use_relative_intensities=False):

        intensities = list()

        for chromatogram in self.chromatograms.values():

            if scaled:

                intensities.append(chromatogram.scaled_intensities(min=0.0, max=10.0))

            if interpolated:

                intensities.append(
                    chromatogram.interpolated_intensities(num_steps=num_rt_steps)
                )

            else:

                intensities.append(chromatogram.intensities)


        intensities = np.array(intensities)

        if use_relative_intensities:

            if intensities.max() != 0.0:

                intensities = intensities / intensities.max()

        return intensities[intensities.mean(axis=1).argsort()]


    def add_score_column(self, key, value):

        self.scores[key] = value

    def get_sub_score_column_array(self, include_score_columns=False):

        score_values = list()

        if include_score_columns:

            for score_column, score_value in self.scores.items():

                #if score_column not in ["VOTE_PERCENTAGE", "vote_percentage"]:

                score_values.append(score_value)

        else:

            for score_column, score_value in self.scores.items():

                if score_column not in ["probability", "vote_percentage"]:

                    score_values.append(score_value)

        return np.asarray(score_values, dtype=np.double)


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

    def __init__(
        self,
        precursor_id="",
        charge=0,
        decoy=0,
        q_value=None,
        modified_sequence="",
        unmodified_sequence="",
        protein_accession="",
        mz=0.0
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

    def set_chromatograms(self, chromatograms: dict[str, Chromatogram]):

        self.chromatograms = chromatograms

    def get_chromatograms(self, min_rt: float, max_rt: float) -> np.ndarray:

        chrom_list = []

        if self.chromatograms:

            for idx, chromatogram in enumerate(self.chromatograms.values()):

                if idx == 0:

                    chrom_list.append(chromatogram.scaled_rts(min_val=min_rt, max_val=max_rt))

                chrom_list.append(chromatogram.normalized_intensities(add_min_max=(0.0, 10.0)))

        return np.array(chrom_list)


    def get_peakgroup(self, rank: int, key: str, reverse: bool = False) -> PeakGroup:

        self.peakgroups.sort(key=lambda x: x.scores[key], reverse=reverse)

        rank = rank - 1

        return self.peakgroups[rank]


class Protein:

    protein_accession: str
    decoy: int
    target: int
    q_value: float
    d_score: float
    scores: Dict[str, float]

    def __init__(self, protein_accession="", decoy=0, q_value=0.0, d_score=0.0):

        self.protein_accession = protein_accession

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value

        self.d_score = d_score

        self.scores = dict()

    @property
    def identifier(self):

        return self.protein_accession


class Proteins:

    proteins: Dict[str, Protein]

    def __init__(self):

        self.proteins = dict()

    def __contains__(self, item):

        return item in self.proteins

    def __setitem__(self, key: str, protein: Protein):

        self.proteins[key] = protein

    def __getitem__(self, item: str) -> Protein:

        return self.proteins[item]

    def __iter__(self):

        for protein_accession, protein in self.proteins.items():

            yield protein


class Peptide:

    sequence: str
    modified_sequence: str
    decoy: int
    target: int
    q_value: float
    d_score: float

    def __init__(
        self,
        sequence="",
        modified_sequence: str = "",
        decoy: int = 0,
        q_value: float = 0.0,
        d_score: float = 0.0,
    ):

        self.sequence = sequence
        self.modified_sequence = modified_sequence

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value
        self.d_score = d_score

    @property
    def identifier(self):

        return self.modified_sequence


class Peptides:

    peptides: Dict[str, Peptide]

    def __init__(self):

        self.peptides = dict()

    def __contains__(self, item):

        return item in self.peptides

    def __setitem__(self, key: str, peptide: Peptide):

        self.peptides[key] = peptide

    def __getitem__(self, key: str) -> Peptide:

        return self.peptides[key]

    def __iter__(self):

        for modified_peptide_sequence, peptide in self.peptides.items():

            yield peptide


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

        for precursor in self:

            precursor_chromatograms = chromatograms.get(precursor)

            if precursor_chromatograms:

                for peakgroup in precursor.peakgroups:

                    peakgroup_chromatograms = dict()

                    for key, chrom in precursor_chromatograms.items():

                        start_rt = peakgroup.start_rt
                        end_rt = peakgroup.end_rt

                        indices = np.where((chrom.rts > start_rt) & (chrom.rts < end_rt))

                        if not indices[0].any():
                            indices = np.where((chrom.rts - start_rt) < 0.5)

                        indices = np.insert(indices, 0, indices[0][0] - 1)

                        if indices[-1] + 1 < chrom.rts.size:
                            indices = np.append(indices, indices[-1] + 1)

                        peakgroup_chromatogram = Chromatogram(
                            type="peakgroup",
                            chrom_id=key,
                            precursor_mz=peakgroup.mz,
                            intensities=chrom.intensities[indices],
                            rts=chrom.rts[indices],
                            start_rt=peakgroup.start_rt,
                            end_rt=peakgroup.end_rt
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

        rank = rank - 1

        for precursor in self.precursors.values():

            precursor.peakgroups.sort(
                key=lambda x: x.scores[score_key], reverse=reverse
            )

            peakgroup = precursor.peakgroups[rank]

            if peakgroup.target == 1:

                filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups

    def filter_target_peakgroups(
        self, rank: int, sort_key: str, filter_key: str, value: float
    ) -> List[PeakGroup]:

        filtered_peakgroups = []

        rank = rank - 1

        for precursor in self.precursors.values():

            precursor.peakgroups.sort(key=lambda x: x.scores[sort_key], reverse=True)

            peakgroup = precursor.peakgroups[rank]

            try:

                if peakgroup.target == 1 and peakgroup.scores[filter_key] >= value:

                    filtered_peakgroups.append(peakgroup)

            except KeyError as e:

                print(filter_key)

                print("[WARNING] peakgroup found without correct subscore")

                print(peakgroup.scores)

                raise e

        return filtered_peakgroups

    def get_decoy_peakgroups(
        self, sort_key: str, use_second_ranked: bool = False
    ) -> List[PeakGroup]:

        filtered_peakgroups = []

        for precursor in self.precursors.values():

            precursor.peakgroups.sort(key=lambda x: x.scores[sort_key], reverse=True)

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

            if verbose:

                print(
                    "Updating peakgroups", len(probabilities), len(peakgroups_to_score)
                )

            for idx, peakgroup in enumerate(peakgroups_to_score):

                peakgroup.scores["probability"] = probabilities[idx]

                peakgroup.scores["vote_percentage"] = vote_percentages[idx]

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

    def score_run(self, model_path: str, scaler_path: str,
                  use_chromatograms: bool, threads: int,
                  gpus: int, use_relative_intensities: bool, use_interpolated_chroms: bool):


        if scaler_path:

            scoring_model = Scorer()

            scoring_model.load(model_path)

            pipeline = Scaler()

            pipeline.load(scaler_path)

        all_peakgroups = self.get_all_peakgroups()

        if use_chromatograms:

            scoring_model = DeepChromScorer(
                max_epochs=1,
                gpus=gpus,
                threads=threads
            )

            scoring_model.load(model_path)

            _, _, _, all_data = preprocess.reformat_chromatogram_data(
                all_peakgroups,
                include_scores=[],
                use_relative_intensities=use_relative_intensities,
                use_interpolated_chroms=use_interpolated_chroms,
                training=False
            )

        else:

            all_data, _, _ = preprocess.reformat_data(
                all_peakgroups, include_score_columns=True
            )

            all_data = pipeline.transform(all_data)

        model_scores = scoring_model.score(all_data)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.scores["d_score"] = model_scores[idx].item()

        return self

    def calculate_q_values(self, sort_key: str, use_decoys: bool = True):

        target_peakgroups = self.get_target_peakgroups_by_rank(
            rank=1, score_key=sort_key, reverse=True
        )

        if use_decoys:

            decoy_peakgroups = self.get_decoy_peakgroups(sort_key=sort_key)

        else:

            decoy_peakgroups = self.get_target_peakgroups_by_rank(
                rank=2, score_key=sort_key, reverse=True
            )

        modelling_peakgroups = target_peakgroups + decoy_peakgroups

        scores, labels = preprocess.reformat_distribution_data(
            modelling_peakgroups, score_column=sort_key
        )

        self.score_distribution = ScoreDistribution()

        self.score_distribution.fit(scores, labels)

        all_peakgroups = self.get_all_peakgroups()

        all_data_scores, all_data_labels = preprocess.reformat_distribution_data(
            all_peakgroups, score_column=sort_key
        )

        q_values = self.score_distribution.calculate_q_vales(all_data_scores)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.scores["q_value"] = q_values[idx].item()

        return self

    def dump_training_data(
        self, file_path: str, filter_field: str, filter_value: float, use_chromatograms: bool,
            use_interpolated_chroms:bool = False, use_relateive_intensities:bool = False
    ) -> None:

        positive_labels = self.filter_target_peakgroups(
            rank=1, sort_key="PROBABILITY", filter_key=filter_field, value=filter_value
        )

        negative_labels = self.get_decoy_peakgroups(
            sort_key="PROBABILITY", use_second_ranked=False
        )

        combined = positive_labels + negative_labels

        if use_chromatograms:

            all_data_scores, all_data_labels, all_data_indices, all_chromatograms = preprocess.reformat_chromatogram_data(
                combined,
                include_scores=["PROBABILITY"],
                use_relative_intensities=use_relateive_intensities,
                use_interpolated_chroms=use_interpolated_chroms
            )

            with open(file_path, "wb") as npfh:
                np.savez(npfh,
                         scores=all_data_scores,
                         labels=all_data_labels,
                         chromatograms=all_chromatograms
                         )

        else:

            all_data_scores, all_data_labels, all_data_indices = preprocess.reformat_data(
                combined, include_score_columns=True
            )

            with open(file_path, "wb") as npfh:

                np.savez(
                    npfh,
                    scores=all_data_scores,
                    labels=all_data_labels
                )


if __name__ == '__main__':
    from gscore.parsers import sqmass
    from gscore.parsers.osw import OSWFile
    from gscore.parsers import queries

    sqmass_file_path = "/home/aaron/projects/ghost/data/spike_in/chromatograms/AAS_P2009_167.sqMass"
    osw_file_path = "/home/aaron/projects/ghost/data/spike_in/openswath/AAS_P2009_167.osw"

    with sqmass.SqMassFile(sqmass_file_path) as sqmass_file:
        chromatograms = sqmass_file.parse()

    with OSWFile(osw_file_path) as osw_file:
        precursors = osw_file.parse_to_precursors(
            query=queries.SelectPeakGroups.FETCH_CHROMATOGRAM_TRAINING_RECORDS
        )

    precursors.set_chromatograms(chromatograms)