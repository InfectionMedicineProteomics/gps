from gps.parsers.queries import CreateIndex

from gps.peakgroups import PeakGroup

from gps.proteins import Protein, Proteins
from gps.peptides import Peptide, Peptides
from gps.precursors import Precursor, Precursors

from gps.parsers import queries

from gps.parsers.sqlite_file import SQLiteFile


from typing import Dict, List, Any

import sqlite3


class Queries:
    ADD_RECORD = """INSERT INTO {table_name} ({list_fields}) VALUES ({list_values});"""

    UPDATE_RECORD = (
        """UPDATE {table_name} SET {update_values} WHERE {key_field} = {record_id};"""
    )

    CREATE_INDEX = (
        """CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_list});"""
    )


OSW_FEATURE_QUERY = """
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

FETCH_MS1_SCORES = """
SELECT
    *
FROM FEATURE_MS1
WHERE FEATURE_ID = {value};
"""


class OSWFile:
    def __init__(self, db_path: str, set_indices: bool = False) -> None:

        self.db_path = db_path
        self.sqlite_file = SQLiteFile(db_path)

        if set_indices:

            with self.sqlite_file as sqlite_file:

                for sql_index in CreateIndex.ALL_INDICES:

                    sqlite_file.run_raw_sql(sql_index)

                if "GHOST_SCORE_TABLE" in self.sqlite_file:

                    sqlite_file.run_raw_sql(CreateIndex.CREATE_GHOST_SCORE_IDX)

    def parse_to_proteins(self, query: str) -> Proteins:

        proteins = Proteins()

        with self.sqlite_file as sqlite_file:

            for record in sqlite_file.iterate_records(query):

                if record["PROTEIN_ACCESSION"] not in proteins:

                    protein = Protein(
                        protein_accession=record["PROTEIN_ACCESSION"],
                        decoy=record["DECOY"],
                        q_value=record["Q_VALUE"],
                        d_score=record["D_SCORE"],
                    )

                    proteins[record["PROTEIN_ACCESSION"]] = protein

                else:

                    if (
                        record["D_SCORE"]
                        > proteins[record["PROTEIN_ACCESSION"]].d_score
                    ):

                        protein = Protein(
                            protein_accession=record["PROTEIN_ACCESSION"],
                            decoy=record["DECOY"],
                            q_value=record["Q_VALUE"],
                            d_score=record["D_SCORE"],
                            probability=record["PROBABILITY"],
                        )

                        proteins[record["PROTEIN_ACCESSION"]] = protein

        return proteins

    def parse_to_peptides(self, query: str) -> Peptides:

        peptides = Peptides()

        with self.sqlite_file as sqlite_file:

            for record in sqlite_file.iterate_records(query):

                if record["MODIFIED_SEQUENCE"] not in peptides:

                    peptide = Peptide(
                        sequence=record["UNMODIFIED_SEQUENCE"],
                        modified_sequence=record["MODIFIED_SEQUENCE"],
                        decoy=record["DECOY"],
                        q_value=record["Q_VALUE"],
                        d_score=record["D_SCORE"],
                    )

                    peptides[record["MODIFIED_SEQUENCE"]] = peptide

                else:

                    if (
                        record["D_SCORE"]
                        > peptides[record["MODIFIED_SEQUENCE"]].d_score
                    ):

                        peptide = Peptide(
                            sequence=record["UNMODIFIED_SEQUENCE"],
                            modified_sequence=record["MODIFIED_SEQUENCE"],
                            decoy=record["DECOY"],
                            q_value=record["Q_VALUE"],
                            d_score=record["D_SCORE"],
                            probability=record["PROBABILITY"],
                        )

                        peptides[record["MODIFIED_SEQUENCE"]] = peptide

        return peptides

    def parse_to_precursors(
        self, query: str, pyprophet_scored: bool = False
    ) -> Precursors:

        with self.sqlite_file as sqlite_file:

            precursors = Precursors()

            for record in sqlite_file.iterate_records(query):

                if record["PRECURSOR_ID"] not in precursors:

                    precursor = Precursor(
                        precursor_id=record["PRECURSOR_ID"],
                        charge=record["CHARGE"],
                        decoy=record["DECOY"],
                        mz=record["MZ"],
                        modified_sequence=record["MODIFIED_SEQUENCE"],
                        unmodified_sequence=record["UNMODIFIED_SEQUENCE"],
                        protein_accession=record.get("PROTEIN_ACCESSION", ""),
                    )

                    precursors[record["PRECURSOR_ID"]] = precursor

                peakgroup = PeakGroup(
                    idx=record["FEATURE_ID"],
                    mz=record["MZ"],
                    start_rt=record["RT_START"],
                    rt=record["RT_APEX"],
                    end_rt=record["RT_END"],
                    decoy=record["DECOY"],
                    ghost_score_id=record.get("GHOST_SCORE_ID", ""),
                    intensity=record.get("AREA_INTENSITY", 0.0),
                    probability=record.get("PROBABILITY", 0.0),
                    vote_percentage=record.get("VOTE_PERCENTAGE", 0.0),
                    q_value=record.get("QVALUE", 0.0)
                    if pyprophet_scored
                    else record.get("Q_VALUE", 0.0),
                    d_score=record.get("SCORE", 0.0)
                    if pyprophet_scored
                    else record.get("D_SCORE", 0.0),
                    scores={
                        score_col: score_value
                        for score_col, score_value in record.items()
                        if score_col.startswith("VAR_")
                    },
                )

                precursors.add_peakgroup(record["PRECURSOR_ID"], peakgroup)

        return precursors

    def fetch_all_records(self, query: str) -> List[Dict[str, Any]]:

        with self.sqlite_file as sqlite_file:

            cursor = sqlite_file.conn.cursor()

            cursor.arraysize = 1000

            cursor.execute(query)

            fetching_records = True

            records = []

            while fetching_records:

                record_batch = cursor.fetchmany()

                if record_batch:

                    records.extend(record_batch)

                else:

                    fetching_records = False

            # formatted_records = []
            #
            # for row in records:
            #
            #     record = {column: value for column, value in zip(row.keys(), row)}
            #
            #     formatted_records.append(record)

        return records

    def add_score_records(self, precursors: Precursors) -> None:

        records = list()

        for precursor in precursors.precursors.values():

            for peakgroup in precursor.peakgroups:
                record = {
                    "FEATURE_ID": peakgroup.idx,
                    "PROBABILITY": peakgroup.probability,
                    "VOTE_PERCENTAGE": peakgroup.vote_percentage,
                }

                records.append(record)

        with self.sqlite_file as sqlite_file:

            sqlite_file.drop_table("GHOST_SCORE_TABLE")

            sqlite_file.create_table(query=queries.CreateTable.CREATE_GHOSTSCORE_TABLE)

            sqlite_file.add_records(table_name="GHOST_SCORE_TABLE", records=records)

    def add_score_and_q_value_records(self, precursors: Precursors) -> None:

        records = list()

        for precursor in precursors.precursors.values():

            for peakgroup in precursor.peakgroups:

                record = {
                    "feature_id": peakgroup.idx,
                    "d_score": peakgroup.d_score,
                    "q_value": peakgroup.q_value,
                    "probability": peakgroup.probability,
                }

                records.append(record)

        with self.sqlite_file as sqlite_file:

            sqlite_file.drop_table("ghost_score_table")

            sqlite_file.create_table(query=queries.CreateTable.CREATE_GHOSTSCORE_TABLE)

            sqlite_file.add_records(table_name="ghost_score_table", records=records)
