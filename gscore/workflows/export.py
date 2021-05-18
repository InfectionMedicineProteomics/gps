import os

from collections.abc import MutableMapping

import numpy as np

from gscore.parsers.osw import osw
from gscore.parsers.osw.queries import (
    SelectPeakGroups,
    CreateIndex
)

from gscore.utils.connection import Connection


class Peptides(MutableMapping):

    def __init__(self):
        self._data = dict()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def apply_q_value_cutoff(self, q_value_method=''):

        if q_value_method == 'sample':

            pass

        elif q_value_method == 'peptide':

            peptide_d_scores = list()

            for _, peptide in self._data.items():

                samples = [sample for _, sample in peptide.sample_data.items()]

                samples.sort(
                    key=lambda sample: sample.d_score,
                    reverse=True
                )


class SampleOutputRecord:

    def __init__(self, mz=0.0, rt=0.0, q_value=0.0, intensity=0.0, d_score=0.0):
        self.mz = mz
        self.rt = rt
        self.q_value = q_value
        self.intensity = intensity
        self.d_score = d_score


class PeptideOutputRecord:

    def __init__(self, peptide_sequence='', charge=0, protein_name='', decoy='', peptide_id=''):
        self.peptide_sequence = peptide_sequence
        self.charge = charge
        self.protein_name = protein_name
        self.target = decoy
        self.peptide_id =peptide_id

        self.sample_data = dict()

    def mz(self):

        sample_mzs = list()

        for sample_name, sample in self.sample_data.items():

            sample_mzs.append(
                sample.mz
            )

        return np.mean(sample_mzs)

    def rt(self):

        sample_rts = list()

        for sample_name, sample in self.sample_data.items():

            sample_rts.append(
                sample.rt
            )

        return np.mean(sample_rts)

    def min_q_value(self):

        sample_q_values = list()

        for sample_name, sample in self.sample_data.items():

            sample_q_values.append(
                sample.q_value
            )

        return np.min(sample_q_values)



def build_peptide_output(args):

    input_files = list()
    input_sample_names = list()

    for input_file in args.input_files:

        input_files.append(input_file.name)

        input_sample_names.append(
            os.path.basename(
                input_file.name
            )
        )

    peptides = Peptides()

    for input_file in input_files:
        with Connection(input_file) as conn:

            for sql_index in CreateIndex.ALL_INDICES:
                conn.run_raw_sql(sql_index)

            for record in conn.iterate_records(
                SelectPeakGroups.FETCH_TRIC_EXPORT_DATA
            ):

                peptide_key = record['modified_peptide_name']

                if peptide_key not in peptides:

                    peptides[peptide_key] = PeptideOutputRecord(
                        peptide_sequence=peptide_key,
                        charge=record['charge'],
                        protein_name=record['protein_name'],
                        decoy=record['decoy']
                    )

                    sample_name = os.path.basename(
                        input_file
                    )

                    #TODO: Add option for MS1 or MS2 intensity

                    intensity = float(record['ms2_intensity'])

                    sample_data = SampleOutputRecord(
                        mz=float(record['mz']),
                        rt=float(record['rt']),
                        q_value=float(record['m_score']),
                        intensity=intensity,
                        d_score=float(record['weighted_d_score'])
                    )

                    peptides[peptide_key].sample_data[sample_name] = sample_data

    return peptides



def main(args, logger):

    if args.export_method == 'tric-formatted':

        peak_groups = osw.fetch_peak_groups(
            host=args.input_osw_file,
            query=SelectPeakGroups.FETCH_TRIC_EXPORT_DATA,
            preprocess=False
        )

        highest_scoring = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['m_score'],
            ascending=True
        )   

        highest_scoring.to_csv(
            args.output_tsv_file,
            sep='\t',
            index=False
        )

    elif args.export_method == 'peptide':

        peptide_output = build_peptide_output(args)


