from enum import Enum

from pynumpress import (
    decode_slof,
    decode_linear,
    decode_pic
)
import zlib

import numpy as np

from gscore.parsers.sqmass import fetch_chromatograms

import numpy as np

class CompressionType(Enum):
    NO = 0
    ZLIB = 1
    NP_LINEAR = 2
    NP_SLOF = 3
    NP_PIC = 4
    NP_LINEAR_ZLIB = 5
    NP_SLOF_ZLIB = 6
    NP_PIC_ZLIB = 7


class ChromatogramDataType(Enum):
    MZ = 0
    INT = 1
    RT = 2


class Chromatogram:

    def __init__(self):
        self.type = ""
        self.id = ""
        self.precursor_mz = 0.0
        self.mz = 0.0
        self.rts = None
        self.intensities = None
        self.charge = 0

    def _mean_intensity(self):

        return np.mean(self.intensities)

    def _stdev_intensity(self):

        return np.std(self.intensities)

    def normalized_intensities(self, add_min_max: tuple =None):

        if not self.intensities.any():

            return self.intensities

        normalized_values = (self.intensities - self._mean_intensity()) / self._stdev_intensity()

        if self._stdev_intensity() == 0.0:

            print(self._stdev_intensity())
            print(self.intensities)

        if add_min_max:

            min_val, max_val = add_min_max

            normalized_values = min_val + (
                    ((normalized_values - normalized_values.min()) * (max_val - min_val)) / (normalized_values.max() - normalized_values.min())
            )

        return normalized_values

    def scaled_rts(self, min_val, max_val, a=0.0, b=100.0):

        return a + (
            ((self.rts - min_val) * (b - a)) / (max_val - min_val)
        )




class Chromatograms:

    def __init__(self):
        self.chromatogram_records = dict()

    def __contains__(self, item):

        return item in self.chromatogram_records

    def items(self):

        return self.chromatogram_records.items()

    def __getitem__(self, key):

        return self.chromatogram_records[key]

    def __get_chromatogram_lengths(self):

        chrom_lengths = set()

        for peptide_id, peptide_chromatograms in self.chromatogram_records.items():

            for record_id, record in peptide_chromatograms.items():

                if record.type != "precursor":
                    chrom_length = len(record.rts)

                    chrom_lengths.add(chrom_length)
                    break

        return chrom_lengths


    def min_chromatogram_length(self):

        return min(self.__get_chromatogram_lengths())

    def max_chromatogram_length(self):

        return max(self.__get_chromatogram_lengths())

    @staticmethod
    def decode_data(chromatogram_data, compression_type):

        decoded_data = list()

        if compression_type == CompressionType.NP_LINEAR_ZLIB.value:

            decompressed = bytearray(
                zlib.decompress(
                    chromatogram_data
                )
            )

            if len(decompressed) > 0:
                decoded_data = decode_linear(decompressed)
            else:
                decoded_data = [0]

        elif compression_type == CompressionType.NP_SLOF_ZLIB.value:

            decompressed = bytearray(
                zlib.decompress(
                    chromatogram_data
                )
            )

            if len(decompressed) > 0:
                decoded_data = decode_slof(decompressed)
            else:
                decoded_data = [0]

        return decoded_data


    @classmethod
    def from_sqmass_file(
            cls,
            file_path: str):

        chromatogram_records = fetch_chromatograms(file_path)

        chromatograms = Chromatograms()

        chromatogram_ids = dict()

        for record in chromatogram_records:

            peptide_id = f"{record['PEPTIDE_SEQUENCE']}_{record['CHARGE']}"

            if peptide_id not in chromatogram_ids:

                chromatogram_ids[peptide_id] = dict()


            if record['NATIVE_ID'] not in chromatogram_ids[peptide_id]:

                chromatogram_record = Chromatogram()

                chromatogram_record.id = record['NATIVE_ID']

                if "Precursor" in record['NATIVE_ID']:
                    chromatogram_record.type = "precursor"
                else:
                    chromatogram_record.type = "fragment"

                chromatogram_record.mz = record['PRODUCT_ISOLATION_TARGET']
                chromatogram_record.precursor_mz = record['PRECURSOR_ISOLATION_TARGET']
                chromatogram_record.charge = int(record['CHARGE'])

                chromatogram_ids[peptide_id][chromatogram_record.id] = chromatogram_record

            data_type = record['DATA_TYPE']

            decoded_data = Chromatograms.decode_data(
                chromatogram_data=record['DATA'],
                compression_type=record['COMPRESSION']
            )

            if data_type == ChromatogramDataType.RT.value:
                chromatogram_ids[peptide_id][record['NATIVE_ID']].rts = np.asarray(decoded_data)
            elif data_type == ChromatogramDataType.INT.value:
                chromatogram_ids[peptide_id][record['NATIVE_ID']].intensities = np.asarray(decoded_data)


        chromatograms.chromatogram_records = chromatogram_ids

        return chromatograms


if __name__ == '__main__':


    from gscore.parsers import osw, queries

    osw_path = '/home/aaron/projects/ghost/data/spike_in/openswath/AAS_P2009_172.osw'
    chrom_path = '/home/aaron/projects/ghost/data/spike_in/openswath/chromatograms/AAS_P2009_172.sqMass'

    peakgroup_graph = osw.fetch_chromatogram_training_data(
        osw_path=osw_path,
        osw_query=queries.SelectPeakGroups.FETCH_TRAIN_CHROMATOGRAM_SCORING_DATA
    )

    highest_ranking = peakgroup_graph.query_nodes(
        color='peptide',
        rank=1,
        query=f"var_xcorr_shape_weighted >= 0.15"
    )

    chromatograms = Chromatograms().from_sqmass_file(chrom_path)


    chrom_lengths = list()
    counts = dict()
    for peptide_id, peptide_chromatograms in chromatograms.items():

        for record_id, record in peptide_chromatograms.items():

            if record.type != "precursor":

                chrom_length = len(record.rts)

                if chrom_length not in counts:

                    counts[chrom_length] = 1
                else:
                    counts[chrom_length] += 1
                chrom_lengths.append(chrom_length)
                break