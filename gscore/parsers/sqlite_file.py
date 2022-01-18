import sqlite3


class SQLiteFile:


    ADD_RECORD = """INSERT INTO {table_name} ({list_fields}) VALUES ({list_values});"""

    UPDATE_RECORD = (
        """UPDATE {table_name} SET {update_values} WHERE {key_field} = {record_id};"""
    )

    CREATE_INDEX = (
        """CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_list});"""
    )


    def __init__(self, file_path : str = ""):

        self.db_path = file_path

        self.conn = sqlite3.connect(file_path)

        self.conn.row_factory = sqlite3.Row

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print(exc_type, exc_val, exc_tb)

        self.conn.close()

    def iterate_records(self, query: str):

        cursor = self.conn.cursor()

        cursor.execute(query)

        for row in cursor:

            if len(row) > 0:

                record = {column: value for column, value in zip(row.keys(), row)}

                yield record

    def add_records(self, table_name="", records=[]):

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
