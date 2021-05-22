from collections.abc import MutableMapping
from csv import DictWriter


class PeptideExportRecord:

    def __init__(self, sequence='', charge=0, decoy=0, protein_accession=''):

        self.sequence = sequence
        self.charge = charge
        self.decoy = decoy
        self.protein_accession = protein_accession

        self.sample_data = dict()

    def __getitem__(self, key):

        return self.sample_data[key]

    def __setitem__(self, key, value):

        self.sample_data[key] = value

    def __len__(self):

        num_samples = 0

        for sample_key, sample_data in self.sample_data.items():

            if sample_data:
                num_samples += 1

        return num_samples

    def __iter__(self):

        return iter(self.sample_data)

    def mz(self):

        mzs = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                mzs.append(data.mz)

        return np.median(mzs)

    def rt(self):

        rts = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                rts.append(data.rt)

        return np.median(rts)

    def q_value(self, level=''):

        q_values = list()

        for sample_key, data in self.sample_data.items():

            if self.sample_data[sample_key]:
                q_values.append(
                    data.scores[f"{level}_q_value"]
                )

        try:

            return np.min(q_values)

        except ValueError:

            print(q_values)

            print(self.sequence, self.sample_data)

            raise

    def get_sample_intensity(self, sample_key='', intensity_type=''):

        if self.sample_data[sample_key]:

            if intensity_type == 'ms1':

                return self.sample_data[sample_key].ms1_intensity

            elif intensity_type == 'ms2':

                return self.sample_data[sample_key].ms2_intensity

        else:

            return 0.0


class ExportData(MutableMapping):

    def __init__(self, data=dict()):

        self._data = data

    def __getitem__(self, key):

        return self._data[key]

    def __setitem__(self, key, value):

        self._data[key] = value

    def __delitem__(self, key):

        del self._data[key]

    def __len__(self):

        return len(self._data)

    def __iter__(self, sample_names=[], quant_type=''):

        for peptide_key, record in self._data.items():

            if len(record) > 0:

                export_record = {
                    'sequence': record.sequence,
                    'charge': record.charge,
                    'decoy': record.decoy,
                    'protein': record.protein_accession,
                    'mz': record.mz(),
                    'rt': record.rt(),
                    'peptide_q_value': record.q_value(level='peptide'),
                    'protein_q_value': record.q_value(level='protein')
                }

                for sample in sample_names:
                    export_record[sample] = record.get_sample_intensity(
                        sample_key=sample,
                        intensity_type=quant_type
                    )

                yield export_record

    def items(self):

        return self._data.items()

    def write(self, path='', sample_names=[], quant_type=''):

        with open(path, 'w') as outfile:

            first_line = True

            for record in self.__iter__(sample_names=sample_names, quant_type=quant_type):

                if first_line:

                    fieldnames = list(record.keys())

                    writer = DictWriter(
                        outfile,
                        fieldnames=fieldnames
                    )

                    writer.writeheader()

                    first_line = False

                else:

                    writer.writerow(record)
