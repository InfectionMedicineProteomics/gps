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
