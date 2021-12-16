from gscore.parsers.connection import Connection

from .queries import ChromatogramQueries


def fetch_chromatograms(chromatogram_file_path: str = ""):

    chromatogram_records = list()

    with Connection(chromatogram_file_path) as conn:

        for record in conn.iterate_records(
            ChromatogramQueries.FETCH_PEPTIDE_CHROMATOGRAM
        ):

            chromatogram_records.append(record)

    return chromatogram_records
