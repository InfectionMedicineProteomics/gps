from __future__ import annotations

from typing import Dict

import numpy as np

from gscore.chromatograms import Chromatogram, decode_data, ChromatogramDataType, Chromatograms
from gscore.parsers.sqlite_file import SQLiteFile


FETCH_PEPTIDE_CHROMATOGRAM = """
    select
        PRECURSOR.PEPTIDE_SEQUENCE,
        PRECURSOR.CHARGE,
        PRECURSOR.ISOLATION_TARGET PRECURSOR_ISOLATION_TARGET,
        CHROMATOGRAM.NATIVE_ID,
        DATA.COMPRESSION,
        DATA.DATA_TYPE,
        DATA.DATA,
        PRODUCT.ISOLATION_TARGET PRODUCT_ISOLATION_TARGET
    from precursor
    join CHROMATOGRAM on PRECURSOR.CHROMATOGRAM_ID = CHROMATOGRAM.ID
    join DATA on CHROMATOGRAM.ID = DATA.CHROMATOGRAM_ID
    join PRODUCT on CHROMATOGRAM.ID = PRODUCT.CHROMATOGRAM_ID;
    """


class SqMassFile(SQLiteFile):


    def __init__(self, file_path : str):

        super().__init__(file_path)

    def parse(self, level: str = "ms2"):

        chromatograms = Chromatograms()

        chromatogram_records = list()

        for record in self.iterate_records(FETCH_PEPTIDE_CHROMATOGRAM):

            chromatogram_records.append(record)

        for record in chromatogram_records:

            precursor_id = f"{record['PRECURSOR_ISOLATION_TARGET']}_{record['CHARGE']}"

            if precursor_id not in chromatograms:

                chromatograms[precursor_id] = dict()

            if level == "ms2":

                if "Precursor" not in record["NATIVE_ID"]:

                    if record["NATIVE_ID"] not in chromatograms[precursor_id]:


                        chromatogram_record = Chromatogram(
                            type="fragment",
                            chrom_id=record["NATIVE_ID"],
                            precursor_mz=record["PRECURSOR_ISOLATION_TARGET"],
                            mz=record["PRODUCT_ISOLATION_TARGET"],
                            charge=record["CHARGE"],
                            peptide_sequence=record["PEPTIDE_SEQUENCE"]
                        )

                        chromatograms[precursor_id][
                            chromatogram_record.id
                        ] = chromatogram_record

                    data_type = record["DATA_TYPE"]

                    decoded_data = decode_data(
                        chromatogram_data=record["DATA"],
                        compression_type=record["COMPRESSION"]
                    )

                    if data_type == ChromatogramDataType.RT.value:

                        chromatograms[precursor_id][record["NATIVE_ID"]].rts = np.array(
                            decoded_data
                        )

                    elif data_type == ChromatogramDataType.INT.value:

                        chromatograms[precursor_id][record["NATIVE_ID"]].intensities = np.array(
                            decoded_data
                        )

        return chromatograms
