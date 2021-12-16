from csv import DictWriter
from collections.abc import MutableMapping
from typing import Dict, List

import numpy as np

from gscore.peakgroups import PeakGroup


class PrecursorExportRecord:

    modified_sequence: str
    charge: int
    decoy: bool
    protein_accession: str
    protein_q_value: float
    peptide_q_value: float
    samples: Dict[str, PeakGroup]

    def __init__(
        self,
        modified_sequence: str,
        charge: int,
        decoy: bool,
        protein_accession: str,
        protein_q_value: float = 0.0,
        peptide_q_value: float = 0.0,
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

    def get_sample_intensity(self, sample_key=""):

        if sample_key in self.samples:

            return self.samples[sample_key].intensity

        else:

            return np.NaN

    def retention_time(self):

        rts = list()

        for sample_key, data in self.samples.items():
            rts.append(data.retention_time)

        return np.median(rts)


class PrecursorExport(MutableMapping):

    _data: Dict[str, PrecursorExportRecord]
    samples: List[str]
    max_q_value: float

    def __init__(self, data=None, max_q_value=0.05):

        if data is None:
            data = dict()

        self._data = data
        self.samples = []
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
                "PeptideSequence": record.modified_sequence,
                "Charge": record.charge,
                "Decoy": record.decoy,
                "Protein": record.protein_accession,
                "RetentionTime": record.retention_time(),
                "PeptideQValue": record.peptide_q_value,
                "ProteinQValue": record.protein_q_value,
            }

            for sample in self.samples:

                if sample in record:

                    if record[sample].scores["Q_VALUE"] <= self.max_q_value:

                        export_record[sample] = record.get_sample_intensity(
                            sample_key=sample
                        )

                    else:

                        export_record[sample] = np.nan

                else:

                    export_record[sample] = np.nan

            yield export_record

    def items(self):

        return self._data.items()

    def write(self, path=""):

        with open(path, "w") as outfile:

            first_line = True

            for record in self:

                if first_line:

                    fieldnames = list(record.keys())

                    writer = DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")

                    writer.writeheader()

                    first_line = False

                else:

                    writer.writerow(record)