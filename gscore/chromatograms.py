from enum import Enum
from typing import Dict

from pynumpress import decode_slof, decode_linear, decode_pic  # type: ignore

import torch
from torch.utils.data import Dataset

import zlib

import numpy as np

from scipy.interpolate import interp1d


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

    def __init__(self,
                 type: str = "",
                 chrom_id: str = "",
                 precursor_mz: float = 0.0,
                 mz: float = 0.0,
                 rts: np.ndarray = np.array([]),
                 intensities: np.ndarray = np.array([]),
                 charge: int = 0,
                 peptide_sequence: str = "",
                 start_rt: float = 0.0,
                 end_rt: float = 0.0):
        self.type = type
        self.id = chrom_id
        self.precursor_mz = precursor_mz
        self.mz = mz
        self.rts = rts
        self.intensities = intensities
        self.charge = charge
        self.peptide_sequence = peptide_sequence
        self.start_rt = start_rt
        self.end_rt = end_rt

    def _mean_intensity(self):

        return np.mean(self.intensities)

    def _stdev_intensity(self):

        return np.std(self.intensities)

    def normalized_intensities(self, add_min_max: tuple = None):

        if not self.intensities.any():

            return self.intensities

        normalized_values = (
            self.intensities - self._mean_intensity()
        ) / self._stdev_intensity()

        if self._stdev_intensity() == 0.0:

            print(self._stdev_intensity())
            print(self.intensities)

        if add_min_max:

            min_val, max_val = add_min_max

            normalized_values = min_val + (
                ((normalized_values - normalized_values.min()) * (max_val - min_val))
                / (normalized_values.max() - normalized_values.min())
            )

        return normalized_values

    def scaled_intensities(self, min: float, max: float):

        min_intensity = self.intensities.min()
        max_intensity = self.intensities.max()

        scaled_intensities = min + (
            ((self.intensities - min_intensity) * (max - min)) / (max_intensity - min_intensity)
        )

        return np.nan_to_num(scaled_intensities, nan=0.0)


    def scaled_rts(self, min_val, max_val, a=0.0, b=100.0):

        return a + (((self.rts - min_val) * (b - a)) / (max_val - min_val))

    def interpolated_intensities(self, num_steps):

        spline = interp1d(
            x=self.rts,
            y=self.intensities
        )

        start_rt = self.start_rt
        end_rt = self.end_rt

        if start_rt < self.rts.min():

            start_rt = self.rts.min()

        if end_rt > self.rts.max():

            end_rt = self.rts.max()

        new_rt_steps = np.linspace(
            start_rt,
            end_rt,
            num_steps
        )

        try:
            return spline(new_rt_steps)
        except ValueError as e:
            print(spline(new_rt_steps))
            raise e

    def interpolated_rt(self, num_steps):

        return np.linspace(
            self.start_rt,
            self.end_rt,
            num_steps
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

    def __setitem__(self, key, value):

        self.chromatogram_records[key] = value

    def get(self, precursor):

        chromatograms = self.chromatogram_records[f"{precursor.mz}_{precursor.charge}"]

        return_dict = dict()

        for key, chromatogram in chromatograms.items():

            if chromatogram.peptide_sequence == precursor.unmodified_sequence:

                return_dict[key] = chromatogram

        if len(return_dict) > 6:

            return_dict = None

        return return_dict


    def __get_chromatogram_lengths(self):

        chrom_lengths = set()

        for chromatograms in self.chromatogram_records.values():

            for record_id, record in chromatograms.items():

                if record.type != "precursor":
                    chrom_length = len(record.rts)

                    chrom_lengths.add(chrom_length)
                    break

        return chrom_lengths

    def rt_span(self):

        return np.array(self.__get_chromatogram_lengths())

    def min_chromatogram_length(self):

        return min(self.__get_chromatogram_lengths())

    def max_chromatogram_length(self):

        return max(self.__get_chromatogram_lengths())


    # @classmethod
    # def from_sqmass_file(cls, file_path: str):
    #
    #     chromatogram_records = fetch_chromatograms(file_path)
    #
    #     chromatograms = Chromatograms()
    #
    #     chromatogram_ids: Dict[str, Dict[str, Chromatogram]] = dict()
    #
    #     for record in chromatogram_records:
    #
    #         peptide_id = f"{record['PEPTIDE_SEQUENCE']}_{record['CHARGE']}"
    #
    #         if peptide_id not in chromatogram_ids:
    #
    #             chromatogram_ids[peptide_id] = dict()
    #
    #         if record["NATIVE_ID"] not in chromatogram_ids[peptide_id]:
    #
    #             chromatogram_record = Chromatogram()
    #
    #             chromatogram_record.id = record["NATIVE_ID"]
    #
    #             if "Precursor" in record["NATIVE_ID"]:
    #                 chromatogram_record.type = "precursor"
    #             else:
    #                 chromatogram_record.type = "fragment"
    #
    #             chromatogram_record.mz = record["PRODUCT_ISOLATION_TARGET"]
    #             chromatogram_record.precursor_mz = record["PRECURSOR_ISOLATION_TARGET"]
    #             chromatogram_record.charge = int(record["CHARGE"])
    #
    #             chromatogram_ids[peptide_id][
    #                 chromatogram_record.id
    #             ] = chromatogram_record
    #
    #         data_type = record["DATA_TYPE"]
    #
    #         decoded_data = Chromatograms.decode_data(
    #             chromatogram_data=record["DATA"], compression_type=record["COMPRESSION"]
    #         )
    #
    #         if data_type == ChromatogramDataType.RT.value:
    #             chromatogram_ids[peptide_id][record["NATIVE_ID"]].rts = np.asarray(
    #                 decoded_data
    #             )
    #         elif data_type == ChromatogramDataType.INT.value:
    #             chromatogram_ids[peptide_id][
    #                 record["NATIVE_ID"]
    #             ].intensities = np.asarray(decoded_data)
    #
    #     chromatograms.chromatogram_records = chromatogram_ids
    #
    #     return chromatograms


class ChromatogramDataset(Dataset):
    def __init__(self, peakgroups, chromatograms, peakgroup_graph):

        self.peakgroups = peakgroups

        print("Scaling Retention Times...")

        peakgroup_graph.scale_peakgroup_retention_times()

        self.chromatograms = chromatograms

        self.peakgroup_graph = peakgroup_graph
        self.min_chromatogram_length = chromatograms.min_chromatogram_length()

        self.interfering_chroms = []

    def __len__(self):

        return len(self.peakgroups)

    def __getitem__(self, idx):

        peakgroup = self.peakgroups[idx]

        peptide_id = ""

        for edge_node in peakgroup.iter_edges(self.peakgroup_graph):

            if edge_node.color == "peptide":
                peptide_id = f"{edge_node.sequence}_{edge_node.charge}"

        peakgroup_boundaries = np.array(
            [
                peakgroup.scaled_rt_start,
                peakgroup.scaled_rt_apex,
                peakgroup.scaled_rt_end,
            ],
            dtype=np.float64,
        )

        label = peakgroup.target

        transition_chromatograms = list()

        rt_steps = list()
        chrom_ids = list()

        for native_id, chromatogram_records in self.chromatograms[peptide_id].items():

            precursor_difference = abs(chromatogram_records.precursor_mz - peakgroup.mz)
            #             print(precursor_difference)

            rt_min = chromatogram_records.rts.min()
            rt_max = chromatogram_records.rts.max()

            if precursor_difference == 0.0:

                if chromatogram_records.type == "fragment":

                    if not rt_steps:
                        scaled_chrom_rt = chromatogram_records.scaled_rts(
                            min_val=self.peakgroup_graph.min_rt_val,
                            max_val=self.peakgroup_graph.max_rt_val,
                        )

                        rt_steps = [
                            scaled_chrom_rt[chrom_idx]
                            for chrom_idx in range(self.min_chromatogram_length)
                        ]

                        transition_chromatograms.append(np.asarray(rt_steps))

                    transition_chromatogram = list()

                    normalized_intensities = (
                        chromatogram_records.normalized_intensities(
                            add_min_max=(0.0, 10.0)
                        )
                    )

                    if np.isfinite(normalized_intensities).all():

                        for chrom_idx in range(self.min_chromatogram_length):

                            norm_intensity = normalized_intensities[chrom_idx]

                            if np.isnan(norm_intensity):
                                norm_intensity = 0.0

                            transition_chromatogram.append(
                                normalized_intensities[chrom_idx]
                            )
                    else:

                        for chrom_idx in range(self.min_chromatogram_length):
                            transition_chromatogram.append(0)

                    transition_chromatograms.append(np.asarray(transition_chromatogram))

                    chrom_ids.append(native_id)

        chromatograms_transformed = list()

        if len(transition_chromatograms) > 7:
            peptide_node = self.peakgroup_graph[peakgroup.get_edges()[0]]

            self.interfering_chroms.append(
                [
                    chrom_ids,
                    len(transition_chromatograms),
                    peptide_id,
                    idx,
                    peakgroup.mz,
                    peptide_node.modified_sequence,
                ]
            )

            transition_chromatograms = transition_chromatograms[:7]

        for row_transform in zip(*transition_chromatograms):
            chromatogram_row = np.asarray(row_transform, dtype="double")

            chromatograms_transformed.append(chromatogram_row)

        chromatograms_transformed = torch.tensor(
            chromatograms_transformed, dtype=torch.double
        )

        return (
            torch.tensor(peakgroup_boundaries, dtype=torch.double),
            chromatograms_transformed.double().T,
            torch.tensor(label).double(),
        )


def decode_data(chromatogram_data, compression_type):

    decoded_data = list()

    if compression_type == CompressionType.NP_LINEAR_ZLIB.value:

        decompressed = bytearray(zlib.decompress(chromatogram_data))

        if len(decompressed) > 0:
            decoded_data = decode_linear(decompressed)
        else:
            decoded_data = [0]

    elif compression_type == CompressionType.NP_SLOF_ZLIB.value:

        decompressed = bytearray(zlib.decompress(chromatogram_data))

        if len(decompressed) > 0:
            decoded_data = decode_slof(decompressed)
        else:
            decoded_data = [0]

    return decoded_data