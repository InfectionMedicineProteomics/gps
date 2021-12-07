from csv import DictWriter
from collections.abc import MutableMapping

import numpy as np

from gscore.peakgroups import PeakGroup

class PrecursorExportRecord:

    def __init__(
            self,
            modified_sequence: str,
            charge: int,
            decoy: bool,
            protein_accession: str,
            protein_q_value: float = None,
            peptide_q_value: float = None
    ):

        self.modified_sequence = modified_sequence
        self.charge = charge
        self.decoy = decoy
        self.protein_accession = protein_accession
        self.protein_q_value = protein_q_value
        self.peptide_q_value = peptide_q_value
        self.samples = dict()

    def __getitem__(self, item: str):

        return self.samples[item]

    def __contains__(self, item):

        return item in self.samples

    def add_sample(self, sample_key, peakgroup: PeakGroup):

        self.samples[sample_key] = peakgroup

    def get_sample_intensity(self, sample_key='', quant_level: str = ""):

        if sample_key in self.samples:

            if quant_level == "ms1":

                return self.samples[sample_key].ms1_intensity

            elif quant_level == "ms2":

                return self.samples[sample_key].ms2_intensity
        else:

            return np.NaN

    def retention_time(self):

        rts = list()

        for sample_key, data in self.samples.items():
            rts.append(data.retention_time)

        return np.median(rts)


class PrecursorExport(MutableMapping):

    def __init__(self, data=None, quant_level: str = "ms2", max_q_value: float = 0.05):

        if data is None:
            data = dict()

        self._data = data
        self.samples = []
        self.quant_level = quant_level
        self.max_q_value = max_q_value

    def add_sample(self, sample_name: str) -> None:

        self.samples.append(sample_name)

    def __getitem__(self, key):

        return self._data[key]

    def __setitem__(self, key, value):

        self._data[key] = value

    def __delitem__(self, key):

        del self._data[key]

    def __len__(self):

        return len(self._data)

    def __contains__(self, item):

        return item in self._data

    def __iter__(self):

        for peptide_key, record in self._data.items():

            export_record = {
                'PeptideSequence': record.modified_sequence,
                'Charge': record.charge,
                'Decoy': record.decoy,
                'Protein': record.protein_accession,
                'RetentionTime': record.retention_time(),
                'PeptideQValue': record.peptide_q_value,
                'ProteinQValue': record.protein_q_value
            }

            for sample in self.samples:

                if sample in record:

                    if record[sample].scores['q_value'] <= self.max_q_value:

                        export_record[sample] = record.get_sample_intensity(
                            sample_key=sample,
                            quant_level=self.quant_level
                        )

                    else:

                        export_record[sample] = np.nan

                else:

                    export_record[sample] = np.nan

            yield export_record

    def items(self):

        return self._data.items()

    def write(self, path=''):

        with open(path, 'w') as outfile:

            first_line = True

            for record in self:

                if first_line:

                    fieldnames = list(record.keys())

                    writer = DictWriter(
                        outfile,
                        fieldnames=fieldnames,
                        delimiter="\t"
                    )

                    writer.writeheader()

                    first_line = False

                else:

                    writer.writerow(record)


if __name__ == '__main__':
    import glob

    from gscore.parsers import osw
    from gscore.parsers import queries

    from gscore.fdr import GlobalDistribution

    osw_files = glob.glob("/home/aaron/projects/aki/data/sample_specific/osw/*.osw")

    global_protein_model = GlobalDistribution.load(
        "/home/aaron/projects/aki/data/sample_specific/models/protein.model"
    )

    global_peptide_model = GlobalDistribution.load(
        "/home/aaron/projects/aki/data/sample_specific/models/peptide.model"
    )

    from pathlib import Path

    export = PrecursorExport(max_q_value=0.05)

    for osw_file in osw_files[:10]:

        print(f"{osw_file}")

        sample_name = Path(osw_file).stem

        export.add_sample(sample_name)

        with osw.OSWFile(osw_file) as osw_conn:

            precursors = osw_conn.parse_to_precursors(
                query=queries.SelectPeakGroups.FETCH_ALL_SCORED_DATA
            )

            for precursor in precursors:

                precursor_id = f"{precursor.modified_sequence}_{precursor.charge}"

                if precursor_id not in export:

                    export[precursor_id] = PrecursorExportRecord(
                        modified_sequence=precursor.modified_sequence,
                        charge=precursor.charge,
                        decoy=precursor.decoy,
                        protein_accession=precursor.protein_accession,
                        protein_q_value=global_protein_model.get_q_value(precursor.protein_accession),
                        peptide_q_value=global_peptide_model.get_q_value(precursor.modified_sequence)
                    )

                peakgroup = precursor.get_peakgroup(rank=1, key="q_value")

                export[precursor_id].add_sample(
                    sample_key=sample_name,
                    peakgroup=peakgroup
                )

    export.write("/home/aaron/projects/aki/data/sample_specific/gscore/211207_sample_specific.tsv")

