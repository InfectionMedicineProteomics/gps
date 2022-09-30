import sqlite3


CREATE_SCHEMA = [
    """
CREATE TABLE IF NOT EXISTS VERSION(
    ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS GENE(
    ID INTEGER PRIMARY KEY,
    GENE_NAME TEXT NOT NULL,
    DECOY INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PEPTIDE_GENE_MAPPING(
    PEPTIDE_ID INT NOT NULL,
    GENE_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PROTEIN(
    ID INTEGER PRIMARY KEY,
    PROTEIN_ACCESSION TEXT NOT NULL,
    DECOY INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PEPTIDE_PROTEIN_MAPPING(
    PEPTIDE_ID INT NOT NULL,
    PROTEIN_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PEPTIDE(
    ID INTEGER PRIMARY KEY,
    UNMODIFIED_SEQUENCE TEXT NOT NULL,
    MODIFIED_SEQUENCE TEXT NOT NULL,
    DECOY INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PRECURSOR_PEPTIDE_MAPPING(
    PRECURSOR_ID INT NOT NULL,
    PEPTIDE_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS COMPOUND(
    ID INTEGER PRIMARY KEY,
    COMPOUND_NAME TEXT NOT NULL,
    SUM_FORMULA TEXT NOT NULL,
    SMILES TEXT NOT NULL,
    ADDUCTS TEXT NOT NULL,
    DECOY INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PRECURSOR_COMPOUND_MAPPING(
    PRECURSOR_ID INT NOT NULL,
    COMPOUND_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS PRECURSOR(
    ID INTEGER PRIMARY KEY,
    TRAML_ID TEXT NULL,
    GROUP_LABEL TEXT NULL,
    PRECURSOR_MZ REAL NOT NULL,
    CHARGE INT NULL,
    LIBRARY_INTENSITY REAL NULL,
    LIBRARY_RT REAL NULL,
    LIBRARY_DRIFT_TIME REAL NULL,
    DECOY INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS TRANSITION_PRECURSOR_MAPPING(
    TRANSITION_ID INT NOT NULL,
    PRECURSOR_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS TRANSITION_PEPTIDE_MAPPING(
    TRANSITION_ID INT NOT NULL,
    PEPTIDE_ID INT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS TRANSITION(
    ID INTEGER PRIMARY KEY,
    TRAML_ID TEXT NULL,
    PRODUCT_MZ REAL NOT NULL,
    CHARGE INT NULL,
    TYPE CHAR(255) NULL,
    ANNOTATION TEXT NULL,
    ORDINAL INT NULL,
    DETECTING INT NOT NULL,
    IDENTIFYING INT NOT NULL,
    QUANTIFYING INT NOT NULL,
    LIBRARY_INTENSITY REAL NULL,
    DECOY INT NOT NULL
);""",
]


class PQPFile:

    ADD_RECORD = """INSERT INTO {table_name} ({list_fields}) VALUES ({list_values});"""

    UPDATE_RECORD = (
        """UPDATE {table_name} SET {update_values} WHERE {key_field} = {record_id};"""
    )

    CREATE_INDEX = (
        """CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_list});"""
    )

    def __init__(self, file_path: str = ""):

        self.db_path = file_path

    def __enter__(self):

        self.conn = sqlite3.connect(self.db_path)

        self.conn.row_factory = sqlite3.Row

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.conn.close()

    def create_tables(self):

        for create_table_query in CREATE_SCHEMA:

            self.create_table(create_table_query)

    def get_genes(self):

        gene_query = """
        SELECT
            *
        FROM GENE;
        """

        genes = []

        for gene in self.iterate_records(gene_query):

            genes.append(gene)

        return genes

    def get_proteins(self):

        protein_query = """
        SELECT
            *
        FROM PROTEIN;
        """

        proteins = []

        for protein in self.iterate_records(protein_query):

            proteins.append(protein)

        return proteins

    def get_peptides(self):

        peptide_query = """
        SELECT
            *
        FROM PEPTIDE;
        """

        peptides = []

        for peptide in self.iterate_records(peptide_query):

            peptides.append(peptide)

        return peptides

    def get_precursors(self):

        precursor_query = """
        SELECT
            *
        FROM PRECURSOR;
        """

        precursors = []

        for precursor in self.iterate_records(precursor_query):

            precursors.append(precursor)

        return precursors

    def get_annotated_precursors(self):

        query = """
        SELECT
            *
        FROM PRECURSOR
        JOIN PRECURSOR_PEPTIDE_MAPPING on PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        JOIN PEPTIDE on PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        JOIN PEPTIDE_PROTEIN_MAPPING on PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        JOIN PROTEIN on PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID;
        """

        records = []

        for record in self.iterate_records(query):

            records.append(record)

        return records

    def get_annotated_transitions(self):

        query = """
        SELECT
            TRANSITION.TRAML_ID TRANSITION_TRAML_ID,
            PRECURSOR.TRAML_ID PRECURSOR_TRAML_ID,
            PRECURSOR.PRECURSOR_MZ,
            PRECURSOR.CHARGE PRECURSOR_CHARGE,
            PRECURSOR.LIBRARY_RT,
            TRANSITION.CHARGE PRODUCT_CHARGE,
            TRANSITION.PRODUCT_MZ,
            TRANSITION.TYPE,
            TRANSITION.ANNOTATION,
            TRANSITION.LIBRARY_INTENSITY,
            TRANSITION.ORDINAL,
            PROTEIN.PROTEIN_ACCESSION,
            PEPTIDE.UNMODIFIED_SEQUENCE,
            PEPTIDE.MODIFIED_SEQUENCE
        FROM TRANSITION
        JOIN TRANSITION_PRECURSOR_MAPPING on TRANSITION.ID = TRANSITION_PRECURSOR_MAPPING.TRANSITION_ID
        JOIN PRECURSOR on TRANSITION_PRECURSOR_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        JOIN PRECURSOR_PEPTIDE_MAPPING on PRECURSOR.ID = PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID
        JOIN PEPTIDE on PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        JOIN PEPTIDE_PROTEIN_MAPPING on PEPTIDE.ID = PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID
        JOIN PROTEIN on PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID = PROTEIN.ID;
        """

        records = []

        for record in self.iterate_records(query):
            records.append(record)

        return records

    def get_transitions(self):

        transition_query = """
        SELECT
            *
        FROM TRANSITION;
        """

        transitions = []

        for transition in self.iterate_records(transition_query):

            transitions.append(transition)

        return transitions

    def get_precursor_peptide_mapping(self):

        query = """
        SELECT
            *
        FROM PRECURSOR_PEPTIDE_MAPPING;
        """

        records = []

        for record in self.iterate_records(query):

            records.append(record)

        return records

    def get_peptide_protein_mapping(self):

        query = """
        SELECT
            *
        FROM PEPTIDE_PROTEIN_MAPPING;
        """

        records = []

        for record in self.iterate_records(query):
            records.append(record)

        return records

    def get_transition_precursor_mapping(self):

        query = """
        SELECT
            *
        FROM TRANSITION_PRECURSOR_MAPPING;
        """

        records = []

        for record in self.iterate_records(query):
            records.append(record)

        return records

    def iterate_records(self, query: str):

        with self:

            cursor = self.conn.cursor()

            cursor.execute(query)

            for row in cursor:

                if len(row) > 0:

                    record = {column: value for column, value in zip(row.keys(), row)}

                    yield record

    def add_records(self, table_name="", records=[]):

        with self:

            input_records = list()

            for record_num, record in enumerate(records):

                field_names = list()
                field_values = list()

                for field_name, field_value in record.items():
                    field_names.append(field_name)

                    field_values.append(field_value)

                input_records.append(tuple(field_values))

                if record_num % 1000 == 0 and record_num > 0:
                    holders = ",".join("?" * len(field_names))

                    add_record_query = self.ADD_RECORD.format(
                        table_name=table_name,
                        list_fields=",".join(field_names),
                        list_values=holders,
                    )

                    cursor = self.conn.cursor()

                    cursor.executemany(add_record_query, input_records)

                    self.conn.commit()

                    input_records = list()

            if input_records:
                holders = ",".join("?" * len(field_names))

                add_record_query = self.ADD_RECORD.format(
                    table_name=table_name,
                    list_fields=",".join(field_names),
                    list_values=holders,
                )

                cursor = self.conn.cursor()

                cursor.executemany(add_record_query, input_records)

                self.conn.commit()

                input_records = list()

    def update_records(self, table_name="", key_field="", records={}):

        """
        records = dict()

        Key in records is the record id
        """

        index_query = self.CREATE_INDEX.format(
            index_name=f"idx_{key_field}_{table_name}",
            table_name=table_name,
            column_list=key_field,
        )

        self.run_raw_sql(index_query)

        input_records = list()

        for record_num, (record_id, record) in enumerate(records.items()):

            field_values = list()

            field_names = list()

            for field_name, field_value in record.items():
                field_values.append(field_value)

                field_names.append(field_name)

            field_values.append(record_id)

            input_records.append(field_values)

            if record_num % 1000 == 0 and record_num > 0:

                if len(field_names) == 1:

                    formatted_update_string_holder = f"{field_names[0]}=?"

                else:

                    formatted_field_names = [
                        f"{field_name}=?" for field_name in field_names
                    ]

                    formatted_update_string_holder = ", ".join(formatted_field_names)

                update_record_query = self.UPDATE_RECORD.format(
                    table_name=table_name,
                    update_values=formatted_update_string_holder,
                    key_field=key_field,
                    record_id="?",
                )

                cursor = self.conn.cursor()

                try:

                    cursor.executemany(update_record_query, input_records)

                except sqlite3.OperationalError as e:

                    print(update_record_query)
                    # print(input_records)
                    raise e

                self.conn.commit()

                input_records = list()

        if input_records:

            if len(field_names) == 1:

                formatted_update_string_holder = f"{field_names[0]}=?"

            else:

                formatted_field_names = [
                    f"{field_name}=?" for field_name in field_names
                ]

                formatted_update_string_holder = ", ".join(formatted_field_names)

            update_record_query = self.UPDATE_RECORD.format(
                table_name=table_name,
                update_values=formatted_update_string_holder,
                key_field=key_field,
                record_id="?",
            )

            cursor = self.conn.cursor()

            try:

                cursor.executemany(update_record_query, input_records)

            except sqlite3.OperationalError as e:

                print(update_record_query)
                print(input_records)
                raise e

            self.conn.commit()

    def run_raw_sql(self, sql):

        with self:

            cursor = self.conn.cursor()

            cursor.execute(sql)

    def create_table(self, query):

        self.run_raw_sql(query)

    def drop_table(self, table_name):

        query = f"drop table if exists {table_name};"

        cursor = self.conn.cursor()

        cursor.execute(query)

    def __contains__(self, table_name: str) -> bool:

        query = (
            """SELECT name FROM sqlite_master "
            f"WHERE type='table' AND name='{table_name}';"""
        ).format(table_name=table_name)

        cursor = self.conn.cursor()

        cursor.execute(query)

        result = cursor.fetchone()

        return bool(result)
