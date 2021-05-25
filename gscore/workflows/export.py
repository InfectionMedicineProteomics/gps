import pathlib

from gscore.parsers.osw import osw
from gscore.parsers.osw import queries


from gscore.cli.common import PassArgs
from gscore.workflows import build_global_model
from gscore.peakgroups import (
    apply_scoring_model
)
from gscore.exportdata import (
    ExportData,
    PeptideExportRecord
)


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
                'score_column': args.score_column,
                'true_target_cutoff': args.true_target_cutoff,
                'false_target_cutoff': args.false_target_cutoff
            }
        )

        peptide_score_distribution = build_global_model.main(peptide_model_args)

        protein_model_args = PassArgs(
            {
                'input_files': args.input_files,
                'model_output': args.model_output,
                'scoring_level': 'protein',
                'use_decoys': args.use_decoys,
                'score_column': args.score_column,
                'true_target_cutoff': args.true_target_cutoff,
                'false_target_cutoff': args.false_target_cutoff
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
