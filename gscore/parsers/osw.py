from gscore.utils.connection import Connection

from gscore.parsers.queries import (
    CreateIndex,
    SelectPeakGroups
)

from gscore.peakgroups import (
    PeakGroup,
    Precursor,
    Protein, Proteins
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

        self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def iterate_records(self, query):

        cursor = self.conn.cursor()

        cursor.execute(query)

        for row in cursor:

            if len(row) > 0:

                record = {column: value for column, value in zip(row.keys(), row)}

                yield record

    def parse_to_proteins(self, query: str) -> Proteins:

        proteins = Proteins()

        for record in self.iterate_records(query):

            if record['protein_accession'] not in proteins:

                if not record['d_score']:

                    print("is none")

                protein = Protein(
                    protein_accession=record['protein_accession'],
                    decoy=record['decoy'],
                    q_value=record['q_value'],
                    d_score=record['d_score']
                )

                proteins[record['protein_accession']] = protein

        return proteins

    def parse_to_precursors(self, query: str) -> Precursors:

        precursors = Precursors()

        score_columns = dict()

        for record in self.iterate_records(query):

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

                elif key == "ghost_score_id":

                    peakgroup.ghost_score_id = value


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

        index_query = Queries.CREATE_INDEX.format(
            index_name=f"idx_{key_field}_{table_name}",
            table_name=table_name,
            column_list=key_field
        )

        self.run_raw_sql(
            index_query
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

                    formatted_field_names = [f"{field_name}=?" for field_name in field_names]

                    formatted_update_string_holder = ', '.join(
                        formatted_field_names
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
                    #print(input_records)
                    raise e

                self.conn.commit()

                input_records = list()

        if input_records:

            if len(field_names) == 1:

                formatted_update_string_holder = f"{field_names[0]}=?"

            else:

                formatted_field_names = [f"{field_name}=?" for field_name in field_names]

                formatted_update_string_holder = ', '.join(
                    formatted_field_names
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


    def add_score_records(self, precursors):

        records = list()

        for precursor in precursors.precursors.values():

            for peakgroup in precursor.peakgroups:
                record = {
                    'feature_id': peakgroup.idx,
                    'probability': peakgroup.scores['probability'],
                    'vote_percentage': peakgroup.scores['vote_percentage']
                }

                records.append(record)

        self.drop_table(
            'ghost_score_table'
        )

        self.create_table(
            query=queries.CreateTable.CREATE_GHOSTSCORE_TABLE
        )

        self.add_records(
            table_name='ghost_score_table',
            records=records
        )

    def update_q_value_records(self, precursors):

        records = dict()

        for precursor in precursors.precursors.values():

            for peakgroup in precursor.peakgroups:

                records[peakgroup.ghost_score_id] = {
                    'd_score': peakgroup.scores['d_score'],
                    'q_value': peakgroup.scores['q_value']
                }

        # self.drop_table(
        #     'ghost_score_table'
        # )

        # self.create_table(
        #     queries.CreateTable.CREATE_GHOSTSCORE_TABLE
        # )

        self.update_records(
            table_name='ghost_score_table',
            key_field="ghost_score_id",
            records=records
        )
