import numpy as np

from collections.abc import MutableMapping
from csv import DictWriter

import pathlib

from gscore.parsers import osw, queries


import pickle


def process_pyprophet_export_data(osw_graphs, args):

    export_data = ExportData()

    for protein_graph in osw_graphs:

        sample = pathlib.Path(protein_graph.graph['file_name']).name

        export_data.samples.append(sample)

        for node, node_data in protein_graph.nodes(data=True):

            if node_data["bipartite"] == "precursor":

                precursor_data = protein_graph.nodes(data=True)[node]['data']

                precursor_data.peakgroups.sort(key=lambda x: x.q_value, reverse=False)

                peakgroup = precursor_data.peakgroups[0]

                if peakgroup.q_value <= args.max_peakgroup_q_value:

                    precursor_id = f"{precursor_data.modified_sequence}_{precursor_data.charge}"

                    protein_ids = []

                    decoy = 0

                    protein_q_value = 0.0

                    peptide_q_value = precursor_data.q_value

                    for protein_node in protein_graph.neighbors(node):

                        protein_ids.append(protein_node)

                        protein_data = protein_graph.nodes(data=True)[protein_node]['data']

                        protein_q_value = protein_data.q_value

                        if protein_data.decoy == 1:
                            decoy = 1

                    if precursor_id not in export_data:
                        precursor_export_record = PrecursorExportRecord(

                            modified_sequence=precursor_data.modified_sequence,
                            charge=precursor_data.charge,
                            decoy=decoy,
                            protein_accession=protein_ids,
                            rt=peakgroup.retention_time,
                            protein_q_value=protein_q_value,
                            peptide_q_value=peptide_q_value

                        )

                        export_data[precursor_id] = precursor_export_record

                    export_data[precursor_id].protein_accession.extend(protein_ids)

                    export_data[precursor_id].add_sample(
                        sample_key=sample,
                        peakgroup_record=peakgroup
                    )

    return export_data


def process_input_osws(args, peptide_score_distribution=None, protein_score_distribution=None):

    sample_graphs = dict()

    for osw_path in args.input_files:

        sample_name = pathlib.Path(osw_path).name

        print(sample_name)

        sample_graph, _ = osw.fetch_peakgroup_graph(
            osw_path,
            query=queries.SelectPeakGroups.FETCH_PEPTIDE_MATRIX_EXPORT_DATA,
            peakgroup_weight_column="d_score",
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

    for protein_node in export_graph.get_nodes('protein'):

        protein_node = export_graph[protein_node]

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

    elif args.export_method == 'pyprophet':

        print("Parsing OSW files.")

        query = queries.SelectPeakGroups.FETCH_PYPROPHET_SCORED_DATA_FOR_EXPORT

        osw_graphs = osw.fetch_export_graphs(args.input_pyprophet_osw_files, query)

        print("Processing export data.")

        export_data = process_pyprophet_export_data(osw_graphs, args)

        print("Writing output file.")
        export_data.write(path=args.output_file)

    elif args.export_method == 'comprehensive':

        # peptide_model_args = PassArgs(
        #     {
        #         'input_files': args.input_files,
        #         'model_output': args.model_output,
        #         'scoring_level': 'peptide',
        #         'use_decoys': args.use_decoys,
        #         'score_column': args.score_column,
        #         'true_target_cutoff': args.true_target_cutoff,
        #         'false_target_cutoff': args.false_target_cutoff
        #     }
        # )
        #
        # peptide_score_distribution = build_global_model.main(peptide_model_args)
        #
        # protein_model_args = PassArgs(
        #     {
        #         'input_files': args.input_files,
        #         'model_output': args.model_output,
        #         'scoring_level': 'protein',
        #         'use_decoys': args.use_decoys,
        #         'score_column': args.score_column,
        #         'true_target_cutoff': args.true_target_cutoff,
        #         'false_target_cutoff': args.false_target_cutoff
        #     }
        # )
        #
        # protein_score_distribution = build_global_model.main(protein_model_args)

        if args.peptide_score_distribution:

            with open(args.peptide_score_distribution, 'rb') as pkl:
                peptide_score_distribution = pickle.load(pkl)

        if args.protein_score_distribution:
            with open(args.protein_score_distribution, 'rb') as pkl:
                protein_score_distribution = pickle.load(pkl)

        else:

            protein_score_distribution = None

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
            sample_names=sample_names,
            quant_type=args.quant_type
        )


class PrecursorExportRecord:

    def __init__(
            self,
            modified_sequence: str,
            charge: int,
            decoy: bool,
            protein_accession: list,
            rt: float,
            protein_q_value: float,
            peptide_q_value: float
    ):

        self.modified_sequence = modified_sequence
        self.charge = charge
        self.decoy = decoy
        self.protein_accession = protein_accession
        self.protein_q_value = protein_q_value
        self.peptide_q_value = peptide_q_value
        self.samples = dict()

    def add_sample(self, sample_key, peakgroup_record):

        self.samples[sample_key] = peakgroup_record

    def get_sample_intensity(self, sample_key=''):

        if sample_key in self.samples:

            return self.samples[sample_key].intensity

        else:

            return np.NaN

    def retention_time(self):

        rts = list()

        for sample_key, data in self.samples.items():
            rts.append(data.retention_time)

        return np.median(rts)

class PeptideExportRecord:

    def __init__(self, sequence='', charge=0, decoy=0, protein_accession=''):

        self.sequence = sequence
        self.charge = charge
        self.decoy = decoy
        self.protein_accession = protein_accession

        self.sample_data = dict()

    def __getitem__(self, key):

        return self.sample_data[key]

    def __setitem__(self, key, value):

        self.sample_data[key] = value

    def __len__(self):

        num_samples = 0

        for sample_key, sample_data in self.sample_data.items():

            if sample_data:
                num_samples += 1

        return num_samples

    def __iter__(self):

        return iter(self.sample_data)

    def mz(self):

        mzs = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                mzs.append(data.mz)

        return np.median(mzs)

    def rt(self):

        rts = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                rts.append(data.rt)

        return np.median(rts)

    def q_value(self, level=''):

        q_values = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                q_values.append(
                    data.scores[f"{level}_q_value"]
                )

        try:

            return np.min(q_values)

        except ValueError:

            print(q_values)

            print(self.sequence, self.sample_data)

            raise

    def add_sample(self, sample_key, peakgroup_record):

        self.sample_data[sample_key] = peakgroup_record

    def get_sample_intensity(self, sample_key='', intensity_type=''):

        if self.sample_data[sample_key]:

            if intensity_type == 'ms1':

                return self.sample_data[sample_key].ms1_intensity

            elif intensity_type == 'ms2':

                return self.sample_data[sample_key].ms2_intensity

        else:

            return 0.0


class ExportData(MutableMapping):

    def __init__(self, data=dict()):

        self._data = data
        self.samples = []

    def __getitem__(self, key):

        return self._data[key]

    def __setitem__(self, key, value):

        self._data[key] = value

    def __delitem__(self, key):

        del self._data[key]

    def __len__(self):

        return len(self._data)

    def __contains__(self, item):

        return item in self._data

    def __iter__(self):

        for peptide_key, record in self._data.items():

            export_record = {
                'PeptideSequence': record.modified_sequence,
                'Charge': record.charge,
                'Decoy': record.decoy,
                'Protein': ';'.join(list(set(record.protein_accession))),
                'RetentionTime': record.retention_time(),
                'PeptideQValue': record.peptide_q_value,
                'ProteinQValue': record.protein_q_value
            }

            for sample in self.samples:

                export_record[sample] = record.get_sample_intensity(
                    sample_key=sample
                )

            yield export_record

    def items(self):

        return self._data.items()

    def write(self, path='', quant_type=''):

        with open(path, 'w') as outfile:

            first_line = True

            for record in self:

                if first_line:

                    fieldnames = list(record.keys())

                    writer = DictWriter(
                        outfile,
                        fieldnames=fieldnames
                    )

                    writer.writeheader()

                    first_line = False

                else:

                    writer.writerow(record)
