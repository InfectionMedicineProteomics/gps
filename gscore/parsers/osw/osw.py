from gscore.utils.connection import Connection

from gscore.parsers.osw.queries import (
    CreateIndex
)

from gscore.peakgroups import (
    Graph,
    PeakGroup,
    Peptide,
    Protein
)

def fetch_peakgroup_graph(osw_path, osw_query, peakgroup_weight_column='var_xcorr_shape_weighted'):

    graph = Graph()

    with Connection(osw_path) as conn:

        for sql_index in CreateIndex.ALL_INDICES:
            conn.run_raw_sql(sql_index)

        for record in conn.iterate_records(osw_query):

            if str(record['protein_accession']) not in graph:
                protein = Protein(
                    key=str(record['protein_accession']),
                    decoy=record['protein_decoy']
                )

                graph.add_node(
                    key=protein.protein_accession,
                    data=protein,
                    color='protein'
                )

            if str(record['transition_group_id']) not in graph:
                peptide = Peptide(
                    key=str(record['transition_group_id']),
                    sequence=record['peptide_sequence'],
                    modified_sequence=record['modified_peptide_sequence'],
                    charge=int(record['charge']),
                    decoy=int(record['protein_decoy'])
                )

                graph.add_node(
                    key=peptide.key,
                    data=peptide,
                    color='peptide'
                )

                graph.add_edge(
                    node_from=str(record['protein_accession']),
                    node_to=peptide.key
                )

            if str(record['feature_id']) not in graph:

                peakgroup = PeakGroup(
                    key=str(record['feature_id']),
                    mz=record['mz'],
                    rt=record['rt'],
                    ms2_intensity=record['ms2_integrated_intensity'],
                    ms1_intensity=record['ms1_integrated_intensity']
                )

                for column_name, column_value in record.items():
                    column_name = column_name.lower()

                    if column_name.startswith('var_'):
                        peakgroup.add_sub_score_column(
                            key=column_name,
                            value=float(column_value)
                        )

                    elif column_name in ['vote_percentage', 'probability', 'logit_probability']:

                        peakgroup.add_score_column(
                            key=column_name,
                            value=float(column_value)
                        )

                    elif column_name == 'ghost_score_id':

                        peakgroup.add_ghost_score_id(str(column_value))

                graph.add_node(
                    key=peakgroup.key,
                    data=peakgroup,
                    color='peakgroup'
                )

                if peakgroup_weight_column in ['vote_percentage', 'probability', 'logit_probability']:

                    graph.add_edge(
                        node_from=str(record['transition_group_id']),
                        node_to=peakgroup.key,
                        weight=peakgroup.scores[peakgroup_weight_column],
                        directed=False
                    )

                else:

                    graph.add_edge(
                        node_from=str(record['transition_group_id']),
                        node_to=peakgroup.key,
                        weight=peakgroup.sub_scores[peakgroup_weight_column],
                        directed=False
                    )

    return graph


def fetch_peptide_level_global_graph(osw_paths, osw_query, score_column):

    full_graph = Graph()

    for osw_path in osw_paths:

        graph = Graph()

        print(f'processing {osw_path}')

        with Connection(osw_path) as conn:

            for sql_index in CreateIndex.ALL_INDICES:
                conn.run_raw_sql(sql_index)

            for record in conn.iterate_records(osw_query):

                protein_key = record['protein_accession']
                peptide_key = str(record['transition_group_id'])
                peakgroup_key = str(record['feature_id'])

                if protein_key not in full_graph:
                    protein = Protein(
                        key=protein_key,
                        decoy=record['protein_decoy']
                    )

                    full_graph.add_node(
                        key=protein_key,
                        data=protein,
                        color='protein'
                    )

                if protein_key not in graph:
                    protein = Protein(
                        key=protein_key,
                        decoy=record['protein_decoy']
                    )

                    graph.add_node(
                        key=protein_key,
                        data=protein,
                        color='protein'
                    )

                if peptide_key not in full_graph:
                    peptide = Peptide(
                        key=peptide_key,
                        sequence=record['peptide_sequence'],
                        modified_sequence=record['modified_peptide_sequence'],
                        charge=int(record['charge']),
                        decoy=int(record['protein_decoy'])
                    )

                    full_graph.add_node(
                        key=peptide_key,
                        data=peptide,
                        color='peptide'
                    )

                    full_graph.add_edge(
                        node_from=protein_key,
                        node_to=peptide_key
                    )

                if peptide_key not in graph:
                    peptide = Peptide(
                        key=peptide_key,
                        sequence=record['peptide_sequence'],
                        modified_sequence=record['modified_peptide_sequence'],
                        charge=int(record['charge']),
                        decoy=int(record['protein_decoy'])
                    )

                    graph.add_node(
                        key=peptide_key,
                        data=peptide,
                        color='peptide'
                    )

                    graph.add_edge(
                        node_from=protein_key,
                        node_to=peptide_key
                    )

                if peakgroup_key not in graph:

                    peakgroup = PeakGroup(
                        key=peakgroup_key,
                        mz=record['mz'],
                        rt=record['rt'],
                        ms2_intensity=record['ms2_integrated_intensity'],
                        ms1_intensity=record['ms1_integrated_intensity']
                    )

                    for column_name, column_value in record.items():

                        if column_name in ['vote_percentage', 'probability', 'logit_probability', 'weighted_d_score',
                                           'd_score']:
                            peakgroup.add_score_column(
                                key=column_name,
                                value=float(column_value)
                            )

                    graph.add_node(
                        key=peakgroup_key,
                        data=peakgroup,
                        color='peakgroup'
                    )

                    score_value = float(record[score_column])

                    graph.add_edge(
                        node_from=peptide_key,
                        node_to=peakgroup_key,
                        weight=score_value,
                        directed=False
                    )

        highest_scoring_peak_groups = graph.query_nodes(
            color='peptide',
            rank=1
        )

        for peakgroup in graph.iter(keys=highest_scoring_peak_groups):
            full_graph.add_node(
                key=peakgroup.key,
                data=peakgroup.data,
                color='peakgroup'
            )

            peptide_key = list(peakgroup._edges.keys())[0]

            full_graph.add_edge(
                node_from=peptide_key,
                node_to=peakgroup.key,
                weight=peakgroup.data.scores[score_column],
                directed=False
            )

    return full_graph