from csv import DictWriter

from typing import Dict, List, Union, ItemsView, Generator, Any

import numpy as np

from gps.peakgroups import PeakGroup


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

    def __getitem__(self, item: str) -> PeakGroup:

        return self.samples[item]

    def __contains__(self, item: str) -> bool:

        return item in self.samples

    def add_sample(self, sample_key: str, peakgroup: PeakGroup) -> None:

        self.samples[sample_key] = peakgroup

    def get_sample_intensity(self, sample_key: str = "") -> Any:

        if sample_key in self.samples:

            return self.samples[sample_key].intensity

        else:

            return np.NaN

    def retention_time(self) -> float:

        rts = list()

        for sample_key, data in self.samples.items():
            rts.append(data.retention_time)

        return float(np.median(rts))


class PrecursorExport:

    _data: Dict[str, PrecursorExportRecord]
    samples: List[str]
    max_q_value: float

    def __init__(
        self,
        data: Union[Dict[str, PrecursorExportRecord], None] = None,
        max_q_value: float = 0.05,
    ):

        if data is None:
            self._data = dict()
        else:
            self._data = data
        self.samples = []
        self.max_q_value = max_q_value

    def add_sample(self, sample_name: str) -> None:

        self.samples.append(sample_name)

    def __getitem__(self, key: str) -> PrecursorExportRecord:

        return self._data[key]

    def __setitem__(self, key: str, value: PrecursorExportRecord) -> None:

        self._data[key] = value

    def __delitem__(self, key: str) -> None:

        del self._data[key]

    def __len__(self) -> int:

        return len(self._data)

    def __contains__(self, item: object) -> bool:

        return item in self._data

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:

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

                    if record[sample].q_value <= self.max_q_value:

                        export_record[sample] = record.get_sample_intensity(
                            sample_key=sample
                        )

                    else:

                        export_record[sample] = np.nan

                else:

                    export_record[sample] = np.nan

            yield export_record

    def items(self) -> ItemsView[str, PrecursorExportRecord]:

        return self._data.items()

    def write(self, path: str = "") -> None:

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
