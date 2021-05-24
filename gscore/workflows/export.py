import os
import pathlib

from collections.abc import MutableMapping

import numpy as np

from gscore.parsers.osw import osw
from gscore.parsers.osw import queries


from gscore.utils.connection import Connection
from gscore.cli.common import PassArgs
from gscore.workflows import build_global_model
from gscore.peakgroups import (
    apply_scoring_model,
    Graph,
    PeakGroup,
    Peptide,
    Protein
)
from gscore.exportdata import (
    ExportData,
    PeptideExportRecord
)


class Peptides(MutableMapping):

    def __init__(self, data=dict()):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def apply_q_value_cutoff(self, q_value_method=''):

        if q_value_method == 'sample':

            pass

        elif q_value_method == 'peptide':

            peptide_d_scores = list()

            for _, peptide in self._data.items():

                samples = [sample for _, sample in peptide.sample_data.items()]

                samples.sort(
                    key=lambda sample: sample.d_score,
                    reverse=True
                )


class SampleOutputRecord:

    def __init__(self, mz=0.0, rt=0.0, q_value=0.0, intensity=0.0, d_score=0.0):
        self.mz = mz
        self.rt = rt
        self.q_value = q_value
        self.intensity = intensity
        self.d_score = d_score


class PeptideOutputRecord:

    def __init__(self, peptide_sequence='', charge=0, protein_name='', decoy='', peptide_id=''):
        self.peptide_sequence = peptide_sequence
        self.charge = charge
        self.protein_name = protein_name
        self.target = decoy
        self.peptide_id =peptide_id

        self.sample_data = dict()

    def mz(self):

        sample_mzs = list()

        for sample_name, sample in self.sample_data.items():

            sample_mzs.append(
                sample.mz
            )

        return np.mean(sample_mzs)

    def rt(self):

        sample_rts = list()

        for sample_name, sample in self.sample_data.items():

            sample_rts.append(
                sample.rt
            )

        return np.mean(sample_rts)

    def min_q_value(self):

        sample_q_values = list()

        for sample_name, sample in self.sample_data.items():

            sample_q_values.append(
                sample.q_value
            )

        return np.min(sample_q_values)



def build_peptide_output(args):

    input_files = list()
    input_sample_names = list()

    for input_file in args.input_files:

        input_files.append(input_file.name)

        input_sample_names.append(
            os.path.basename(
                input_file.name
            )
        )

    peptides = Peptides()

    for input_file in input_files:
        with Connection(input_file) as conn:

            for sql_index in CreateIndex.ALL_INDICES:
                conn.run_raw_sql(sql_index)

            for record in conn.iterate_records(
                SelectPeakGroups.FETCH_TRIC_EXPORT_DATA
            ):

                peptide_key = record['modified_peptide_name']

                if peptide_key not in peptides:

                    peptides[peptide_key] = PeptideOutputRecord(
                        peptide_sequence=peptide_key,
                        charge=record['charge'],
                        protein_name=record['protein_name'],
                        decoy=record['decoy']
                    )

                    sample_name = os.path.basename(
                        input_file
                    )

                    #TODO: Add option for MS1 or MS2 intensity

                    intensity = float(record['ms2_intensity'])

                    sample_data = SampleOutputRecord(
                        mz=float(record['mz']),
                        rt=float(record['rt']),
                        q_value=float(record['m_score']),
                        intensity=intensity,
                        d_score=float(record['weighted_d_score'])
                    )

                    peptides[peptide_key].sample_data[sample_name] = sample_data

    return peptides

def process_input_osws(args, peptide_score_distribution=None, protein_score_distribution=None):

    sample_graphs = dict()

    for osw_path in args.input_files:

        sample_name = pathlib.Path(osw_path).name

        print(sample_name)

        sample_graph = osw.fetch_peakgroup_graph(
            osw_path,
            osw_query=queries.SelectPeakGroups.FETCH_PEPTIDE_MATRIX_EXPORT_DATA,
            peakgroup_weight_column=args.score_column,
            peptide_index=['modified_peptide_sequence', 'charge']
        )

        if peptide_score_distribution:

            apply_scoring_model(
                sample_graph,
                'peptide',
                peptide_score_distribution,
                args.score_column
            )

        if protein_score_distribution:

            apply_scoring_model(
                sample_graph,
                'protein',
                protein_score_distribution,
                args.score_column
            )

        sample_graphs[sample_name] = sample_graph

    return sample_graphs

def prepare_export_data(args, export_graph, sample_graphs):

    export_data = ExportData()

    for protein_node in export_graph.iter(color='protein'):

        for peptide_node in protein_node.iter_edges(export_graph):

            peptide_key = f"{peptide_node.data.modified_sequence}_{peptide_node.data.charge}"

            peptide_export_record = PeptideExportRecord(
                sequence=peptide_node.data.modified_sequence,
                charge=peptide_node.data.charge,
                decoy=peptide_node.data.decoy,
                protein_accession=protein_node.data.protein_accession
            )

            for sample_key, sample_graph in sample_graphs.items():

                if peptide_key in sample_graph:

                    peptide_node = sample_graph[peptide_key]

                    highest_ranked_peakgroup_key = peptide_node.get_edge_by_ranked_weight()

                    peakgroup_node = sample_graph[highest_ranked_peakgroup_key]

                    if peakgroup_node.data.scores['q_value'] <= args.max_peakgroup_q_value:

                        sample_peptide = peakgroup_node.data

                    else:

                        sample_peptide = None

                else:

                    sample_peptide = None

                peptide_export_record[sample_key] = sample_peptide

            export_data[peptide_key] = peptide_export_record

    return export_data

def main(args, logger):

    if args.export_method == 'tric-formatted':

        peak_groups = osw.fetch_peak_groups(
            host=args.input_osw_file,
            query=queries.SelectPeakGroups.FETCH_TRIC_EXPORT_DATA,
            preprocess=False
        )

        highest_scoring = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['m_score'],
            ascending=True
        )   

        highest_scoring.to_csv(
            args.output_tsv_file,
            sep='\t',
            index=False
        )

    elif args.export_method == 'comprehensive':

        peptide_model_args = PassArgs(
            {
                'input_files': args.input_files,
                'model_output': args.model_output,
                'scoring_level': 'peptide',
                'use_decoys': args.use_decoys,
                'score_column': args.score_column
            }
        )

        peptide_score_distribution = build_global_model.main(peptide_model_args)

        protein_model_args = PassArgs(
            {
                'input_files': args.input_files,
                'model_output': args.model_output,
                'scoring_level': 'protein',
                'use_decoys': args.use_decoys,
                'score_column': args.score_column
            }
        )

        protein_score_distribution = build_global_model.main(protein_model_args)

        sample_graphs = process_input_osws(
            args,
            peptide_score_distribution=peptide_score_distribution,
            protein_score_distribution=protein_score_distribution
        )

        export_graph = osw.fetch_export_graph(
            args.input_files,
            osw_query=queries.SelectPeakGroups.FETCH_PEPTIDE_MATRIX_EXPORT_DATA
        )

        export_data = prepare_export_data(
            args,
            export_graph,
            sample_graphs
        )

        sample_names = [pathlib.Path(osw_path).name for osw_path in args.input_files]

        export_data.write(
            path=args.output_file,
            sample_names=sample_names
        )
