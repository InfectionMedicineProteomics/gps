import pathlib

from gscore.parsers import osw, queries

from gscore.cli.common import PassArgs
from gscore.workflows import build_global_model
from gscore.peakgroups import (
    apply_scoring_model
)
from gscore.exportdata import (
    ExportData,
    PeptideExportRecord,
    PrecursorExportRecord
)

import pickle



def process_pyprophet_export_data(osw_graphs):

    export_data = ExportData()

    for protein_graph in osw_graphs:

        sample = pathlib.Path(protein_graph.graph['file_name']).name

        export_data.samples.append(sample)

        for node, node_data in protein_graph.nodes(data=True):

            if node_data["bipartite"] == "precursor":

                precursor_data = protein_graph.nodes(data=True)[node]['data']

                precursor_data.peakgroups.sort(key=lambda x: x.q_value, reverse=False)

                peakgroup = precursor_data.peakgroups[0]

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

        export_data = process_pyprophet_export_data(osw_graphs)

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
