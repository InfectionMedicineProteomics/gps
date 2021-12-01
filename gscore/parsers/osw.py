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

from gscore.peakgroups import Precursors, Precursor, PeakGroup


import sqlite3


class Queries:
    ADD_RECORD = """INSERT INTO {table_name} ({list_fields}) VALUES ({list_values});"""

    UPDATE_RECORD = """UPDATE {table_name} SET {update_values} WHERE {key_field} = {record_id};"""

    CREATE_INDEX = """CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_list});"""


OSW_FEATURE_QUERY = \
"""
SELECT *,
       RUN_ID || '_' || PRECURSOR_ID AS GROUP_ID
FROM FEATURE_MS2
INNER JOIN
  (SELECT RUN_ID,
          ID,
          PRECURSOR_ID,
          EXP_RT
   FROM FEATURE) AS FEATURE ON FEATURE_ID = FEATURE.ID
INNER JOIN
  (SELECT ID,
          CHARGE AS PRECURSOR_CHARGE,
          DECOY
   FROM PRECURSOR) AS PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
INNER JOIN
  (SELECT PRECURSOR_ID AS ID,
          COUNT(*) AS TRANSITION_COUNT
   FROM TRANSITION_PRECURSOR_MAPPING
   INNER JOIN TRANSITION ON TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID = TRANSITION.ID
   WHERE DETECTING==1
   GROUP BY PRECURSOR_ID) AS VAR_TRANSITION_SCORE ON FEATURE.PRECURSOR_ID = VAR_TRANSITION_SCORE.ID
ORDER BY RUN_ID,
         PRECURSOR.ID ASC,
         FEATURE.EXP_RT ASC;
"""

FETCH_MS1_SCORES = \
"""
SELECT
    *
FROM FEATURE_MS1
WHERE FEATURE_ID = {value};
"""

class OSWPeakgroup:

    transition_group_id: str
    precursor_id: str
    feature_id: str
    mz: float
    charge: int
    decoy: int
    target: int
    peptide_sequence: str
    modified_sequence: str
    protein_accession: str
    rt_start: float
    rt_apex: float
    rt_end: float
    scores: Dict[str, float]

    def __init__(self,
        transition_group_id: str,
        precursor_id: str,
        feature_id: str,
        mz: float,
        charge: int,
        decoy: int,
        target: int,
        peptide_sequence: str,
        modified_sequence: str,
        protein_accession: str,
        rt_start: float,
        rt_apex: float,
        rt_end: float
    ):
        self.transition_group_id = transition_group_id
        self.precursor_id = precursor_id
        self.feature_id = feature_id
        self.mz = mz
        self.charge = charge
        self.decoy = decoy
        self.target = target
        self.peptide_sequence = peptide_sequence
        self.modified_sequence = modified_sequence
        self.protein_accession = protein_accession
        self.rt_start = rt_start
        self.rt_apex = rt_apex
        self.rt_end = rt_end
        self.scores = Dict[str, float]



class OSWFile:

    def __init__(self, db_path):

        self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)

        for sql_index in CreateIndex.ALL_INDICES:

            self.run_raw_sql(sql_index)

        self.conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def iterate_subscore_records(self, query):

        cursor = self.conn.cursor()

        cursor.execute(query)

        for row in cursor:

            if len(row) > 0:

                record = {column: value for column, value in zip(row.keys(), row)}

                # osw_peakgroup = OSWPeakgroup(
                #     transition_group_id=record['transition_group_id'],
                #     precursor_id=record['precursor_id'],
                #     feature_id=record['feature_id'],
                #     mz=record['mz'],
                #     charge=record['charge'],
                #     decoy=record['decoy'],
                #     target=abs(record['decoy'] - 1),
                #     peptide_sequence=record['peptide_sequence'],
                #     modified_sequence=record['modified_sequence'],
                #     protein_accession=record['protein_accession'],
                #     rt_start=record['rt_start'],
                #     rt_apex=record['rt_apex'],
                #     rt_end=record['rt_end']
                # )
                #
                # osw_peakgroup.scores = {key: value for key, value in record.items() if key.startswith("VAR") and value is not None}

                yield record

    def fetch_subscore_records(self, query: str) -> Precursors:

        precursors = Precursors()

        score_columns = dict()

        for record in self.iterate_subscore_records(query):

            if record['precursor_id'] not in precursors:
                precursor = Precursor(
                    sequence=record['peptide_sequence'],
                    modified_sequence=record['modified_sequence'],
                    charge=record['charge'],
                    decoy=record['decoy']
                )

                precursors[record['precursor_id']] = precursor

            peakgroup = PeakGroup(
                idx=record['feature_id'],
                mz=record['mz'],
                start_rt=record['rt_start'],
                rt=record['rt_apex'],
                end_rt=record['rt_end'],
                decoy=record['decoy']
            )

            for key, value in record.items():

                if key.startswith("VAR") and value is not None:

                    peakgroup.scores[key] = value

                elif key in ['probability', 'vote_percentage']:

                    peakgroup.scores[key] = value

            precursors.add_peakgroup(record['precursor_id'], peakgroup)

        return precursors



    def fetch_all_records(self, query):

        cursor = self.conn.cursor()

        cursor.execute(query)

        fetching_records = True

        records = []

        while fetching_records:

            record_batch = cursor.fetchmany(10000)

            if record_batch:

                records.extend(record_batch)

            else:

                fetching_records = False

        formatted_records = []

        for row in records:

            record = {column: value for column, value in zip(row.keys(), row)}

            formatted_records.append(record)

        return formatted_records

    def fetch_feature_subscores(self, query: str, use_ms1_scores: bool = False) -> List[OSWPeakgroup]:

        records = self.fetch_all_records(query)

        if use_ms1_scores:

            ms1_scores = self.fetch_all_records(query=FETCH_MS1_SCORES)

            ms1_scores = {record['FEATURE_ID']: record for record in ms1_scores}

        for record in records:

            if use_ms1_scores:

                ms1_record_scores = {f"{key}_MS1": value for key, value in ms1_scores[record['feature_id']].items() if
                                     key.startswith("VAR")}

                record.update(ms1_record_scores)

        return records


    def add_records(self, table_name='', records=[]):

        input_records = list()

        for record_num, record in enumerate(records):

            field_names = list()
            field_values = list()

            for field_name, field_value in record.items():
                field_names.append(field_name)

                field_values.append(field_value)

            input_records.append(tuple(field_values))

            if record_num % 1000 == 0 and record_num > 0:
                holders = ','.join('?' * len(field_names))

                add_record_query = Queries.ADD_RECORD.format(
                    table_name=table_name,
                    list_fields=','.join(field_names),
                    list_values=holders
                )

                cursor = self.conn.cursor()

                cursor.executemany(add_record_query, input_records)

                self.conn.commit()

                input_records = list()

        if input_records:
            holders = ','.join('?' * len(field_names))

            add_record_query = Queries.ADD_RECORD.format(
                table_name=table_name,
                list_fields=','.join(field_names),
                list_values=holders
            )

            cursor = self.conn.cursor()

            cursor.executemany(add_record_query, input_records)

            self.conn.commit()

            input_records = list()

    def update_records(self, table_name='', key_field='', records={}):

        """
        records = dict()

        Key in records is the record id
        """

        self.run_raw_sql(
            Queries.CREATE_INDEX.format(
                index_name=f"idx_{key_field}_{table_name}",
                table_name=table_name,
                column_list=key_field
            )

        )

        input_records = list()

        for record_num, (record_id, record) in enumerate(records.items()):

            field_values = list()

            field_names = list()

            for field_name, field_value in record.items():
                field_values.append(field_value)

                field_names.append(field_name)

            field_values.append(record_id)

            input_records.append(
                field_values
            )

            if record_num % 1000 == 0 and record_num > 0:

                if len(field_names) == 1:

                    formatted_update_string_holder = f"{field_names[0]}=?"

                else:

                    formatted_update_string_holder = '=?, '.join(
                        field_names
                    )

                update_record_query = Queries.UPDATE_RECORD.format(
                    table_name=table_name,
                    update_values=formatted_update_string_holder,
                    key_field=key_field,
                    record_id='?'
                )

                cursor = self.conn.cursor()

                try:

                    cursor.executemany(
                        update_record_query,
                        input_records
                    )

                except sqlite3.OperationalError as e:

                    print(update_record_query)
                    print(input_records)
                    raise e

                self.conn.commit()

                input_records = list()

        if input_records:

            if len(field_names) == 1:

                formatted_update_string_holder = f"{field_names[0]}=?"

            else:

                formatted_update_string_holder = '=?, '.join(
                    field_names
                )

            update_record_query = Queries.UPDATE_RECORD.format(
                table_name=table_name,
                update_values=formatted_update_string_holder,
                key_field=key_field,
                record_id='?'
            )

            cursor = self.conn.cursor()

            try:

                cursor.executemany(
                    update_record_query,
                    input_records
                )

            except sqlite3.OperationalError as e:

                print(update_record_query)
                print(input_records)
                raise e

            self.conn.commit()

            input_records = list()

    def run_raw_sql(self, sql):

        cursor = self.conn.cursor()

        cursor.execute(sql)

    def create_table(self, query):

        self.run_raw_sql(query)

    def drop_table(self, table_name):

        query = f"drop table if exists {table_name};"

        cursor = self.conn.cursor()

        cursor.execute(query)


def update_score_records(osw_file: str, records: List[Dict]):

    with OSWFile(osw_file) as conn:

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

    #protein_graph = nx.Graph(file_name=osw_path)

    protein_graph = Graph()

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

                protein_graph.add_node(
                    key=record['protein_accession'],
                    node=protein,
                    color='protein'
                )

            if precursor_id not in protein_graph:

                precursor = Precursor(
                    sequence=record['modified_sequence'],
                    modified_sequence=record['modified_sequence'],
                    charge=record['charge'],
                    decoy=record['decoy']
                )

                protein_graph.add_node(
                    key=precursor_id,
                    node=precursor,
                    color='precursor'
                )

                protein_graph.add_edge(
                    node_from=record['protein_accession'],
                    node_to=precursor_id,
                    bidirectional=True
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

            protein_graph[precursor_id].peakgroups.append(
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