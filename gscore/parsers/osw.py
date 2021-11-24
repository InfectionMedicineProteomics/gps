from gscore.utils.connection import Connection

from gscore.parsers.queries import (
    CreateIndex,
    SelectPeakGroups
)

from gscore.peakgroups import (
    PeakGroup,
    Precursor,
    Protein
)

from gscore.parsers import queries

from gscore.datastructures.graph import Graph

import networkx as nx

from typing import List, Dict


def update_score_records(osw_file: str, records: List[Dict]):

    with Connection(osw_file) as conn:

        conn.drop_table(
            'ghost_score_table'
        )

        conn.create_table(
            query=queries.CreateTable.CREATE_GHOSTSCORE_TABLE
        )

        conn.add_records(
            table_name='ghost_score_table',
            records=records
        )

def fetch_export_graphs(osw_files, query):

    osw_graphs = []

    for osw_file in osw_files:

        print(f"Parsing {osw_file}")

        protein_graph = nx.Graph(file_name=osw_file)

        with Connection(osw_file) as conn:

            for record in conn.iterate_records(query):

                if record['protein_accession'] not in protein_graph:

                    protein = Protein(
                        protein_accession=record['protein_accession'],
                        decoy=int(record['decoy']),
                        q_value=record['global_protein_q_value']
                    )

                    protein_graph.add_nodes_from(
                        [record['protein_accession']],
                        data=protein,
                        bipartite='protein'
                    )

                else:

                    prec_id = f"{record['modified_sequence']}_{record['charge']}"

                    if prec_id not in protein_graph:
                        precursor = Precursor(
                            sequence=record['modified_sequence'],
                            modified_sequence=record['modified_sequence'],
                            charge=record['charge'],
                            q_value=record['global_peptide_q_value']
                        )

                        protein_graph.add_nodes_from(
                            [prec_id],
                            data=precursor,
                            bipartite='precursor'
                        )

                        protein_graph.add_edges_from(
                            [
                                (record['protein_accession'], prec_id)
                            ]
                        )

                    peakgroup = PeakGroup(
                        rt=record['retention_time'],
                        intensity=record['intensity'],
                        q_value=record['peakgroup_q_value']
                    )

                    protein_graph.nodes[prec_id]['data'].peakgroups.append(peakgroup)

        osw_graphs.append(protein_graph)

    return osw_graphs

def fetch_peakgroup_data(osw_path: str, osw_query: str):

    protein_graph = nx.Graph(file_name=osw_path)

    with Connection(osw_path) as conn:

        for sql_index in CreateIndex.ALL_INDICES:
            conn.run_raw_sql(sql_index)

        for record in conn.iterate_records(osw_query):

            precursor_id = f"{record['modified_sequence']}_{record['charge']}"

            if record['protein_accession'] not in protein_graph:

                protein = Protein(
                    protein_accession=record['protein_accession'],
                    decoy=int(record['decoy'])
                )

                protein_graph.add_nodes_from(
                    [record['protein_accession']],
                    data=protein,
                    bipartite='protein'
                )

            if precursor_id not in protein_graph:

                precursor = Precursor(
                    sequence=record['modified_sequence'],
                    modified_sequence=record['modified_sequence'],
                    charge=record['charge'],
                    decoy=record['decoy']
                )

                protein_graph.add_nodes_from(
                    [precursor_id],
                    data=precursor,
                    bipartite='precursor'
                )

                protein_graph.add_edges_from(
                    [
                        (record['protein_accession'], precursor_id)
                    ]
                )

            peakgroup = PeakGroup(
                idx=record['feature_id'],
                mz=record['mz'],
                rt=record['retention_time'],
                ms2_intensity=record['ms2_intensity'],
                ms1_intensity=record['ms1_intensity'],
                decoy=record['decoy'],
                start_rt=record['left_width'],
                end_rt=record['right_width'],
                delta_rt=record['delta_rt'],
            )

            if 'ghost_score_id' in record:

                peakgroup.ghost_score_id = record['ghost_score_id']

            for column_name, column_value in record.items():

                column_name = column_name.lower()

                if column_name.startswith('var_'):

                    peakgroup.add_sub_score_column(
                        key=column_name,
                        value=float(column_value)
                    )

                elif column_name in [
                    'vote_percentage',
                    'probability',
                    'logit_probability',
                    'd_score',
                    'weighted_d_score',
                    'q_value',
                    'transition_mass_dev_score',
                    'precursor_mass_dev_score'
                ]:

                    peakgroup.add_score_column(
                        key=column_name,
                        value=float(column_value)
                    )

            protein_graph.nodes[precursor_id]['data'].peakgroups.append(
                peakgroup
            )

    return protein_graph

# def fetch_pyprophet_export_graph(osw_path, query):
#
#     graph = Graph()
#
#     with Connection(osw_path) as conn:
#
#         for record in conn.iterate_records(query):
#
#             protein_key = record['protein_accession']
#
#             precursor_key = record['precursor_id']
#
#             peakgroup_key = record['feature_id']
#
#             if protein_key not in graph:
#
#                 protein = Protein(
#
#                     key=protein_key,
#                     color='protein',
#                     protein_accession=str(record['protein_accession']),
#                     decoy=int(record['decoy'])
#
#                 )
#
#                 protein.scores['q-value'] = float(record['global_protein_q_value'])
#
#                 graph.add_node(
#                     protein.key,
#                     protein,
#                     "protein"
#                 )
#
#             if peptide_key not in graph:
#
#                 peptide = Peptide(
#                     key=peptide_key,
#                     color='peptide',
#                     modified_sequence=record['modified_sequence'],
#                     charge=int(record['charge']),
#                     decoy=int(record['decoy'])
#                 )
#
#                 peptide.scores['q-value'] = float(record['global_peptide_q_value'])
#
#                 graph.add_node(
#                     peptide.key,
#                     peptide,
#                     "peptide"
#                 )
#
#                 graph.add_edge(
#                     protein_key,
#                     peptide_key,
#                     0.0,
#                     False
#                 )
#
#             if peakgroup_key not in graph:
#
#                 peakgroup = PeakGroup(
#                     key=peakgroup_key,
#                     color='peakgroup',
#                     mz=float(record['mz']),
#                     rt=float(record['rt']),
#                     ms2_intensity=float(record['ms2_integrated_intensity']),
#                     ms1_intensity=float(record['ms1_integrated_intensity']),
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 if "delta_rt" in record:
#                     peakgroup.start_rt = record['left_width']
#                     peakgroup.end_rt = record['right_width']
#                     peakgroup.delta_rt = record['delta_rt']
#
#                 for column_name, column_value in record.items():
#
#                     column_name = column_name.lower()
#
#                     if column_name.startswith('var_'):
#                         peakgroup.add_sub_score_column(
#                             column_name,
#                             float(column_value)
#                         )
#
#                     elif column_name in ['vote_percentage', 'probability', 'logit_probability', 'd_score',
#                                          'weighted_d_score', 'q_value']:
#
#                         if column_value != None:
#
#                             column_value = float(column_value)
#
#                         else:
#
#                             ## TODO: fix bug where probability is not calculated for all peakgroups
#                             none_peak_groups.append(record)
#
#                         peakgroup.add_score_column(
#                             column_name,
#                             column_value
#                         )
#
#                     # elif column_name == 'ghost_score_id':
#                     #
#                     #     peakgroup.add_ghost_score_id(str(column_value))
#
#                 graph.add_node(
#                     peakgroup.key,
#                     peakgroup,
#                     "peakgroup"
#                 )
#
#                 if peakgroup_weight_column in [
#                     'vote_percentage',
#                     'probability',
#                     'logit_probability',
#                     'd_score',
#                     'weighted_d_score',
#                     'q_value'
#                 ]:
#
#                     graph.add_edge(
#                         node_from=peptide_key,
#                         node_to=peakgroup_key,
#                         weight=peakgroup.scores[peakgroup_weight_column],
#                         bidirectional=True
#                     )
#
#                 else:
#
#                     graph.add_edge(
#                         node_from=peptide_key,
#                         node_to=peakgroup_key,
#                         weight=peakgroup.sub_scores[peakgroup_weight_column],
#                         bidirectional=True
#                     )
#
#
#
# def fetch_peakgroup_graph(osw_path, use_decoys=False, query=None, peakgroup_weight_column='var_xcorr_shape_weighted', peptide_index='transition_group_id'):
#
#     graph = Graph()
#
#     none_peak_groups = list()
#
#     with Connection(osw_path) as conn:
#
#         for sql_index in CreateIndex.ALL_INDICES:
#             conn.run_raw_sql(sql_index)
#
#         if query:
#
#             query = query
#
#         else:
#
#             if use_decoys:
#
#                 query = SelectPeakGroups.FETCH_UNSCORED_PEAK_GROUPS
#
#             else:
#
#                 query = SelectPeakGroups.FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE
#
#         for record in conn.iterate_records(query):
#
#             protein_key = str(record['protein_accession'])
#
#             if isinstance(peptide_index, list):
#
#                 indices = list()
#
#                 for index in peptide_index:
#                     indices.append(
#                         str(record[index])
#                     )
#
#                 peptide_key = '_'.join(indices)
#
#             else:
#
#                 peptide_key = str(record[peptide_index])
#
#             peakgroup_key = str(record['feature_id'])
#
#             if protein_key not in graph:
#                 protein = Protein(
#
#                     key=protein_key,
#                     color='protein',
#                     protein_accession=str(record['protein_accession']),
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 graph.add_node(
#                     protein.key,
#                     protein,
#                     "protein"
#                 )
#
#             if peptide_key not in graph:
#                 peptide = Peptide(
#                     key=peptide_key,
#                     color='peptide',
#                     sequence=record['peptide_sequence'],
#                     modified_sequence=record['modified_peptide_sequence'],
#                     charge=int(record['charge']),
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 graph.add_node(
#                     peptide.key,
#                     peptide,
#                     "peptide"
#                 )
#
#                 graph.add_edge(
#                     protein_key,
#                     peptide_key,
#                     0.0,
#                     False
#                 )
#
#             if peakgroup_key not in graph:
#
#                 peakgroup = PeakGroup(
#                     key=peakgroup_key,
#                     color='peakgroup',
#                     mz=float(record['mz']),
#                     rt=float(record['rt']),
#                     ms2_intensity=float(record['ms2_integrated_intensity']),
#                     ms1_intensity=float(record['ms1_integrated_intensity']),
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 if "delta_rt" in record:
#                     peakgroup.start_rt = record['left_width']
#                     peakgroup.end_rt = record['right_width']
#                     peakgroup.delta_rt = record['delta_rt']
#
#                 for column_name, column_value in record.items():
#
#                     column_name = column_name.lower()
#
#                     if column_name.startswith('var_'):
#                         peakgroup.add_sub_score_column(
#                             column_name,
#                             float(column_value)
#                         )
#
#                     elif column_name in ['vote_percentage', 'probability', 'logit_probability', 'd_score',
#                                          'weighted_d_score', 'q_value']:
#
#                         if column_value != None:
#
#                             column_value = float(column_value)
#
#                         else:
#
#                             ## TODO: fix bug where probability is not calculated for all peakgroups
#                             none_peak_groups.append(record)
#
#                         peakgroup.add_score_column(
#                             column_name,
#                             column_value
#                         )
#
#
#                     # elif column_name == 'ghost_score_id':
#                     #
#                     #     peakgroup.add_ghost_score_id(str(column_value))
#
#                 graph.add_node(
#                     peakgroup.key,
#                     peakgroup,
#                     "peakgroup"
#                 )
#
#                 if peakgroup_weight_column in [
#                     'vote_percentage',
#                     'probability',
#                     'logit_probability',
#                     'd_score',
#                     'weighted_d_score',
#                     'q_value'
#                 ]:
#
#                     graph.add_edge(
#                         node_from=peptide_key,
#                         node_to=peakgroup_key,
#                         weight=peakgroup.scores[peakgroup_weight_column],
#                         bidirectional=True
#                     )
#
#                 else:
#
#                     graph.add_edge(
#                         node_from=peptide_key,
#                         node_to=peakgroup_key,
#                         weight=peakgroup.sub_scores[peakgroup_weight_column],
#                         bidirectional=True
#                     )
#
#     return graph, none_peak_groups
#
#
# def fetch_peptide_level_global_graph(osw_paths, osw_query, score_column):
#
#     full_graph = Graph()
#
#     for osw_path in osw_paths:
#
#         graph = Graph()
#
#         print(f'processing {osw_path}')
#
#         with Connection(osw_path) as conn:
#
#             for sql_index in CreateIndex.ALL_INDICES:
#                 conn.run_raw_sql(sql_index)
#
#             for record in conn.iterate_records(osw_query):
#
#                 protein_key = record['protein_accession']
#                 peptide_key = str(record['transition_group_id'])
#                 peakgroup_key = str(record['feature_id'])
#
#                 protein = Protein(
#                     key=protein_key,
#                     color="protein",
#                     decoy=record['protein_decoy']
#                 )
#
#                 peptide = Peptide(
#                     key=peptide_key,
#                     color="peptide",
#                     sequence=record['peptide_sequence'],
#                     modified_sequence=record['modified_peptide_sequence'],
#                     charge=int(record['charge']),
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 peakgroup = PeakGroup(
#                     key=peakgroup_key,
#                     color="peakgroup",
#                     mz=record['mz'],
#                     rt=record['rt'],
#                     ms2_intensity=record['ms2_integrated_intensity'],
#                     ms1_intensity=record['ms1_integrated_intensity'],
#                     decoy=int(record['protein_decoy'])
#                 )
#
#                 for column_name, column_value in record.items():
#
#                     if column_name in ['vote_percentage', 'probability', 'logit_probability', 'weighted_d_score',
#                                        'd_score']:
#                         peakgroup.add_score_column(
#                             key=column_name,
#                             value=float(column_value)
#                         )
#
#
#                 if protein_key not in full_graph:
#
#                     full_graph.add_node(
#                         key=protein_key,
#                         data=protein,
#                         color='protein'
#                     )
#
#                 if protein_key not in graph:
#
#                     graph.add_node(
#                         key=protein_key,
#                         data=protein,
#                         color='protein'
#                     )
#
#                 if peptide_key not in full_graph:
#
#                     full_graph.add_node(
#                         key=peptide_key,
#                         data=peptide,
#                         color='peptide'
#                     )
#
#                     full_graph.add_edge(
#                         node_from=protein_key,
#                         node_to=peptide_key
#                     )
#
#                 if peptide_key not in graph:
#
#                     graph.add_node(
#                         key=peptide_key,
#                         data=peptide,
#                         color='peptide'
#                     )
#
#                     graph.add_edge(
#                         node_from=protein_key,
#                         node_to=peptide_key
#                     )
#
#                 if peakgroup_key not in graph:
#
#                     graph.add_node(
#                         key=peakgroup_key,
#                         data=peakgroup,
#                         color='peakgroup'
#                     )
#
#                     score_value = float(record[score_column])
#
#                     graph.add_edge(
#                         node_from=peptide_key,
#                         node_to=peakgroup_key,
#                         weight=score_value,
#                         directed=False
#                     )
#
#         highest_scoring_peak_groups = graph.query_nodes(
#             color='peptide',
#             rank=1
#         )
#
#         for peakgroup_key in highest_scoring_peak_groups:
#             full_graph.add_node(
#                 key=peakgroup.key,
#                 data=peakgroup.data,
#                 color='peakgroup'
#             )
#
#             peptide_key = list(peakgroup._edges.keys())[0]
#
#             full_graph.add_edge(
#                 node_from=peptide_key,
#                 node_to=peakgroup.key,
#                 weight=peakgroup.data.scores[score_column],
#                 directed=False
#             )
#
#     return full_graph
#
# def fetch_protein_level_global_graph(osw_paths, osw_query, score_column):
#
#     full_graph = Graph()
#
#     for osw_path in osw_paths:
#
#         graph = Graph()
#
#         print(f'processing {osw_path}')
#
#         with Connection(osw_path) as conn:
#
#             for sql_index in CreateIndex.ALL_INDICES:
#                 conn.run_raw_sql(sql_index)
#
#             for record in conn.iterate_records(osw_query):
#
#                 protein_key = record['protein_accession']
#                 peakgroup_key = str(record['feature_id'])
#
#                 if protein_key not in full_graph:
#                     protein = Protein(
#                         key=protein_key,
#                         decoy=record['protein_decoy']
#                     )
#
#                     full_graph.add_node(
#                         key=protein_key,
#                         data=protein,
#                         color='protein'
#                     )
#
#                 if protein_key not in graph:
#                     protein = Protein(
#                         key=protein_key,
#                         decoy=record['protein_decoy']
#                     )
#
#                     graph.add_node(
#                         key=protein_key,
#                         data=protein,
#                         color='protein'
#                     )
#
#                 if peakgroup_key not in graph:
#
#                     peakgroup = PeakGroup(
#                         key=peakgroup_key,
#                         mz=record['mz'],
#                         rt=record['rt'],
#                         ms2_intensity=record['ms2_integrated_intensity'],
#                         ms1_intensity=record['ms1_integrated_intensity']
#                     )
#
#                     for column_name, column_value in record.items():
#
#                         if column_name in ['vote_percentage', 'probability', 'logit_probability', 'weighted_d_score',
#                                            'd_score']:
#                             peakgroup.add_score_column(
#                                 key=column_name,
#                                 value=float(column_value)
#                             )
#
#                     graph.add_node(
#                         key=peakgroup_key,
#                         data=peakgroup,
#                         color='peakgroup'
#                     )
#
#                     score_value = float(record[score_column])
#
#                     graph.add_edge(
#                         node_from=protein_key,
#                         node_to=peakgroup_key,
#                         weight=score_value,
#                         directed=False
#                     )
#
#         highest_scoring_peak_groups = graph.query_nodes(
#             color='protein',
#             rank=1
#         )
#
#         for peakgroup in graph.iter(keys=highest_scoring_peak_groups):
#             full_graph.add_node(
#                 key=peakgroup.key,
#                 data=peakgroup.data,
#                 color='peakgroup'
#             )
#
#             protein_key = list(peakgroup._edges.keys())[0]
#
#             full_graph.add_edge(
#                 node_from=protein_key,
#                 node_to=peakgroup.key,
#                 weight=peakgroup.data.scores[score_column],
#                 directed=False
#             )
#
#     return full_graph
#
# def fetch_export_graph(osw_paths, osw_query):
#
#     graph = Graph()
#
#     for osw_path in osw_paths:
#
#         with Connection(osw_path) as conn:
#
#             for sql_index in CreateIndex.ALL_INDICES:
#                 conn.run_raw_sql(sql_index)
#
#             for record in conn.iterate_records(osw_query):
#
#                 protein_key = record['protein_accession']
#                 peptide_key = str(record['transition_group_id'])
#
#                 if protein_key not in graph:
#
#                     protein = Protein(
#                         key=protein_key,
#                         color='protein',
#                         protein_accession=str(record['protein_accession']),
#                         decoy=int(record['protein_decoy'])
#                     )
#
#                     graph.add_node(
#                         protein_key,
#                         protein,
#                         'protein'
#                     )
#
#                 if peptide_key not in graph:
#
#                     peptide = Peptide(
#                         key=peptide_key,
#                         color='peptide',
#                         sequence=record['peptide_sequence'],
#                         modified_sequence=record['modified_peptide_sequence'],
#                         charge=int(record['charge']),
#                         decoy=int(record['protein_decoy'])
#                     )
#
#                     graph.add_node(
#                         peptide_key,
#                         peptide,
#                         'peptide'
#                     )
#
#                     graph.add_edge(
#                         node_from=protein_key,
#                         node_to=peptide_key
#                     )
#
#     return graph