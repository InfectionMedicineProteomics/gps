from __future__ import annotations

from typing import List, Dict

import numpy as np
import networkx as nx
from sklearn.linear_model import SGDClassifier

from sklearn.utils import shuffle, class_weight
from sklearn.metrics import precision_score, recall_score

from gscore.utils import ml
from gscore.scaler import Scaler
from gscore.denoiser import BaggedDenoiser
from gscore.scorer import Scorer
from gscore.fdr import ScoreDistribution

from cleanlab.classification import LearningWithNoisyLabels

from joblib import dump

class PeakGroup:
    ghost_score_id: str
    idx: str
    mz: float
    rt: float
    intensity: float
    decoy: int
    target: int
    delta_rt: float
    start_rt: float
    end_rt: float

    def __init__(self, ghost_score_id='', idx='', mz=0.0, rt=0.0, q_value=None, intensity=0.0, decoy=0,
                 delta_rt = 0.0, start_rt = 0.0, end_rt = 0.0):

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

        self.scaled_rt_start = 0.0
        self.scaled_rt_apex = 0.0
        self.scaled_rt_end = 0.0

    def __repr__(self):

        return f"{self.mz=} {self.retention_time=} {self.decoy=} {self.scores=}"

    def add_score_column(self, key, value):

        self.scores[key] = value

    def get_sub_score_column_array(self, include_score_columns=False):

        score_values = list()

        if include_score_columns:

            for score_value in self.scores.values():

                score_values.append(score_value)

        else:

            for score_column, score_value in self.scores.items():

                if score_column not in ['probability', 'vote_percentage']:

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
    protein_accession: str

    def __init__(self, precursor_id = '', charge=0, decoy=0, q_value=None, modified_sequence="", protein_accession=""):

        self.id = precursor_id
        self.charge = charge
        self.decoy = decoy
        self.target = abs(decoy - 1)
        self.q_value = q_value
        self.peakgroups = []
        self.scores = dict()
        self.modified_sequence = modified_sequence
        self.protein_accession = protein_accession

    def get_peakgroup(self, rank: int, key: str, reverse: bool = False) -> PeakGroup:

        self.peakgroups.sort(
            key=lambda x: x.scores[key],
            reverse=reverse
        )

        rank = rank - 1

        return self.peakgroups[rank]


class Protein:

    protein_accession: str
    decoy: int

    def __init__(self, protein_accession='', decoy=0, q_value=None, d_score: float = None):

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

    def __init__(self, sequence='', modified_sequence: str = "", decoy: int = 0, q_value: float = None, d_score: float = None):

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

    def get_peakgroups_by_list(self, precursor_list: List[str], rank: int = 0, score_key: str = '',
                               reverse: bool = True, return_all: bool = False) -> List[PeakGroup]:

        peakgroups = list()

        for precursor_key in precursor_list:

            precursor = self.precursors[precursor_key]

            if not return_all:

                peakgroup = precursor.get_peakgroup(
                    rank=1,
                    key=score_key,
                    reverse=reverse
                )

                peakgroups.append(peakgroup)

            else:

                for peakgroup in precursor.peakgroups:

                    peakgroups.append(peakgroup)

        return peakgroups

    def get_target_peakgroups_by_rank(self, rank: int, score_key: str = '', reverse: bool = True):

        filtered_peakgroups = []

        rank = rank - 1

        for precursor in self.precursors.values():

            precursor.peakgroups.sort(key=lambda x: x.scores[score_key], reverse=reverse)

            peakgroup = precursor.peakgroups[rank]

            if peakgroup.target == 1:

                filtered_peakgroups.append(peakgroup)

        return filtered_peakgroups


    def filter_target_peakgroups(self, rank: int, sort_key: str, filter_key: str, value: float) -> List[PeakGroup]:

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

    def get_decoy_peakgroups(self, sort_key: str, use_second_ranked: bool = False) -> List[PeakGroup]:

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

    def denoise(self,
                num_folds: int,
                num_classifiers: int,
                num_threads: int,
                vote_percentage: float,
                verbose: bool = False,
                base_estimator = None) -> Precursors:

        precursor_folds = ml.get_precursor_id_folds(list(self.keys()), num_folds)

        total_recall = []
        total_precision = []

        for fold_num, precursor_fold_ids in enumerate(precursor_folds):

            if verbose:

                print(f"Processing fold {fold_num + 1}...")

            training_precursors = ml.get_training_data(
                folds=precursor_folds,
                fold_num=fold_num
            )


            #TODO: Pick better way to initially rank features
            training_data_targets = self.get_peakgroups_by_list(
                precursor_list=training_precursors,
                rank=1,
                score_key='VAR_XCORR_SHAPE_WEIGHTED',
                reverse=True
            )

            peakgroup_scores, peakgroup_labels, peakgroup_indices = ml.reformat_data(
                peakgroups=training_data_targets
            )

            train_data, train_labels = shuffle(
                peakgroup_scores,
                peakgroup_labels,
                random_state=42
            )

            scaler = Scaler()

            train_data = scaler.fit_transform(train_data)

            n_samples = int(len(train_data) * 1.0)

            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(train_labels),
                y=train_labels.ravel()
            )

            if base_estimator:

                denoizer = BaggedDenoiser(
                    base_estimator=base_estimator,
                    max_samples=n_samples,
                    n_estimators=num_classifiers,
                    n_jobs=num_threads,
                    random_state=42,
                    class_weights=class_weights
                )

            else:

                denoizer = BaggedDenoiser(
                    max_samples=n_samples,
                    n_estimators=num_classifiers,
                    n_jobs=num_threads,
                    random_state=42,
                    class_weights=class_weights
                )

            denoizer.fit(
                train_data,
                train_labels.ravel()
            )

            peakgroups_to_score = self.get_peakgroups_by_list(
                precursor_list=precursor_fold_ids,
                return_all=True
            )

            testing_scores, testing_labels, testing_keys = ml.reformat_data(
                peakgroups=peakgroups_to_score
            )

            testing_scores = scaler.transform(
                testing_scores
            )

            class_index = np.where(
                denoizer.classes_ == 1.0
            )[0][0]

            vote_percentages = denoizer.vote(
                testing_scores,
                threshold=vote_percentage
            )

            probabilities = denoizer.predict_proba(
                testing_scores
            )[:, class_index]


            if verbose:

                print("Updating peakgroups", len(probabilities), len(peakgroups_to_score))

            for idx, peakgroup in enumerate(peakgroups_to_score):

                peakgroup.scores['probability'] = probabilities[idx]

                peakgroup.scores['vote_percentage'] = vote_percentages[idx]


            validation_data = self.get_peakgroups_by_list(
                precursor_list=precursor_fold_ids,
                rank=1,
                score_key='VAR_XCORR_SHAPE_WEIGHTED',
                reverse=True
            )

            val_scores, val_labels, _ = ml.reformat_data(
                peakgroups=validation_data
            )

            val_scores = scaler.transform(
                val_scores
            )

            fold_precision = precision_score(
                y_pred=denoizer.predict(val_scores),
                y_true=val_labels.ravel()
            )

            fold_recall = recall_score(
                y_pred=denoizer.predict(val_scores),
                y_true=val_labels.ravel()
            )

            total_recall.append(fold_recall)
            total_precision.append(fold_precision)

            if verbose:

                print(
                    f"Fold {fold_num + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
                )

        print(f"Mean recall: {np.mean(total_recall)}, Mean precision: {np.mean(total_precision)}")

        return self

    def score_run(self, model_path: str, scaler_path: str):

        scoring_model = Scorer()

        scoring_model.load(model_path)

        pipeline = Scaler()

        pipeline.load(scaler_path)

        all_peakgroups = self.get_all_peakgroups()

        all_data_scores, all_data_labels, all_data_indices = ml.reformat_data(
            all_peakgroups,
            include_score_columns=True
        )

        all_data_scores = pipeline.transform(all_data_scores)

        model_scores = scoring_model.score(all_data_scores)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.scores['d_score'] = model_scores[idx].item()

        return self

    def calculate_q_values(self, sort_key: str, use_decoys: bool = True):

        target_peakgroups = self.get_target_peakgroups_by_rank(
            rank=1,
            score_key=sort_key,
            reverse=True
        )

        if use_decoys:

            decoy_peakgroups = self.get_decoy_peakgroups(
                sort_key=sort_key
            )

        else:

            decoy_peakgroups = self.get_target_peakgroups_by_rank(
                rank=2,
                score_key=sort_key,
                reverse=True
            )

        modelling_peakgroups = target_peakgroups + decoy_peakgroups

        scores, labels = ml.reformat_distribution_data(
            modelling_peakgroups,
            score_column=sort_key
        )

        self.score_distribution = ScoreDistribution()

        self.score_distribution.fit(
            scores,
            labels
        )

        all_peakgroups = self.get_all_peakgroups()

        all_data_scores, all_data_labels = ml.reformat_distribution_data(
            all_peakgroups,
            score_column=sort_key
        )

        q_values = self.score_distribution.calculate_q_vales(all_data_scores)

        for idx, peakgroup in enumerate(all_peakgroups):

            peakgroup.scores['q_value'] = q_values[idx].item()

        return self

    def dump_training_data(self, file_path: str, filter_field: str, filter_value: float) -> None:

        positive_labels = self.filter_target_peakgroups(
            rank=1,
            sort_key="PROBABILITY",
            filter_key=filter_field,
            value=filter_value
        )

        negative_labels = self.get_decoy_peakgroups(
            sort_key='PROBABILITY',
            use_second_ranked=False
        )

        combined = positive_labels + negative_labels

        all_data_scores, all_data_labels, all_data_indices = ml.reformat_data(
            combined,
            include_score_columns=True
        )

        with open(file_path, 'wb') as npfh:

            np.savez(npfh, x=all_data_scores, y=all_data_labels)


def get_decoy_peakgroups(graph: nx.Graph, sort_key: str, use_second_ranked: bool = False) -> List[PeakGroup]:

    filtered_peakgroups = []

    for node, node_data in graph.nodes(data=True):

        if node_data["bipartite"] == "precursor":

            precursor_data = node_data['data']

            precursor_data.peakgroups.sort(key=lambda x: x.scores[sort_key], reverse=True)

            if use_second_ranked and len(precursor_data.peakgroups) > 1:

                peakgroup = precursor_data.peakgroups[1]

                if peakgroup.target == 1:

                    peakgroup.target = 0
                    peakgroup.decoy = 1

                    filtered_peakgroups.append(peakgroup)

            else:

                peakgroup = precursor_data.peakgroups[0]

                filtered_peakgroups.append(peakgroup)

    return filtered_peakgroups

def get_all_peakgroups(graph: nx.Graph) -> List[PeakGroup]:

    all_peakgroups = []

    for node, node_data in graph.nodes(data=True):

        if node_data['bipartite'] == "precursor":

            precursor_data = node_data['data']

            for peakgroup in precursor_data.peakgroups:

                all_peakgroups.append(peakgroup)

    return all_peakgroups


def filter_target_peakgroups(graph: nx.Graph, rank: int, sort_key: str, filter_key: str, value: float) -> List[PeakGroup]:

    filtered_peakgroups = []

    rank = rank - 1

    for node, node_data in graph.nodes(data=True):

        if node_data["bipartite"] == "precursor":

            precursor_data = node_data['data']

            precursor_data.peakgroups.sort(key=lambda x: x.scores[sort_key], reverse=True)

            peakgroup = precursor_data.peakgroups[rank]

            if peakgroup.target == 1 and peakgroup.scores[filter_key] >= value:

                filtered_peakgroups.append(peakgroup)

    return filtered_peakgroups


def get_peakgroups_by_list(graph: nx.Graph, node_list: List[str], rank: int = 0, score_key: str = '', reverse: bool = True, return_all: bool = False) -> List[PeakGroup]:

    nodes_by_rank = list()

    for node_key in node_list:

        node = graph.nodes[node_key]

        if not return_all:

            peakgroup = node['data'].get_peakgroup(
                rank = 1,
                key = score_key,
                reverse = reverse
            )

            nodes_by_rank.append(peakgroup)

        else:

            peakgroups = node['data'].peakgroups

            nodes_by_rank.extend(peakgroups)

    return nodes_by_rank



def get_score_array(graph, node_list, score_column=''):

    score_array = list()

    for node in graph.iter(keys=node_list):

        if node.color == 'peakgroup':

            score = node.data.scores[score_column]
            
            score_array.append(score)

        else:

            print(node.color, node.data.scores, node.key, node._edges)

    return np.array(score_array)


def calc_score_grouped_by_level(graph, function=None, level='', score_column='', new_column_name=''):

    for node in graph.iter(color=level):

        scores = list()

        for key in node.get_edges():

            scores.append(
                graph[key].data.scores[score_column]
            )

        scores = np.array(scores, dtype=np.float64)

        score = function(scores)

        #TODO: This is not optimal. Should instead set at parent node level
        for key in node.get_edges():

            graph[key].data.scores[new_column_name] = score
