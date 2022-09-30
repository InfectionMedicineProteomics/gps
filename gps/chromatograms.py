from typing import Dict, ItemsView, Optional, Set, Tuple

import numpy as np

from scipy.interpolate import interp1d


class Chromatogram:

    type: str
    chrom_id: str
    precursor_mz: float
    mz: float
    rts: np.ndarray
    intensities: np.ndarray
    charge: int
    peptide_sequence: str
    start_rt: float
    end_rt: float

    def __init__(
        self,
        type: str = "",
        chrom_id: str = "",
        precursor_mz: float = 0.0,
        mz: float = 0.0,
        rts: np.ndarray = np.array([]),
        intensities: np.ndarray = np.array([]),
        charge: int = 0,
        peptide_sequence: str = "",
        start_rt: float = 0.0,
        end_rt: float = 0.0,
        # calculate_interpolated_function: bool=False
    ):
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

        # if calculate_interpolated_function:
        #
        #     self.interpolated_chromatogram = interp1d(
        #         x=rts,
        #         y=intensities,
        #         bounds_error=False,
        #         fill_value=0.0,
        #         assume_sorted=True
        #     )

    def interpolated_chromatogram(self, rts: np.ndarray) -> np.ndarray:

        # TODO: Investigate if the left and right extrapolation should change

        return np.interp(rts, self.rts, self.intensities)

    def _mean_intensity(self) -> np.ndarray:

        return np.mean(self.intensities)

    def _stdev_intensity(self) -> np.ndarray:

        return np.std(self.intensities)

    def normalized_intensities(
        self, add_min_max: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:

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

    def scaled_intensities(self, min: float, max: float) -> np.ndarray:

        min_intensity = self.intensities.min()
        max_intensity = self.intensities.max()

        scaled_intensities = min + (
            ((self.intensities - min_intensity) * (max - min))
            / (max_intensity - min_intensity)
        )

        return np.nan_to_num(scaled_intensities, nan=0.0)

    def scaled_rts(
        self, min_val: float, max_val: float, a: float = 0.0, b: float = 100.0
    ) -> np.ndarray:

        return a + (((self.rts - min_val) * (b - a)) / (max_val - min_val))

    def interpolated_intensities(self, num_steps: int) -> np.ndarray:

        spline = interp1d(x=self.rts, y=self.intensities)

        start_rt = self.start_rt
        end_rt = self.end_rt

        if start_rt < self.rts.min():

            start_rt = self.rts.min()

        if end_rt > self.rts.max():

            end_rt = self.rts.max()

        new_rt_steps = np.linspace(start_rt, end_rt, num_steps)

        try:
            return spline(new_rt_steps)
        except ValueError as e:
            print(spline(new_rt_steps))
            raise e

    def interpolated_rt(self, num_steps: int) -> np.ndarray:

        return np.linspace(self.start_rt, self.end_rt, num_steps)


class Chromatograms:

    chromatogram_records: Dict[str, Dict[str, Chromatogram]]

    def __init__(self) -> None:

        self.chromatogram_records = dict()

    def __contains__(self, item: str) -> bool:

        return item in self.chromatogram_records

    def items(self) -> ItemsView[str, Dict[str, Chromatogram]]:

        return self.chromatogram_records.items()

    def __getitem__(self, key: str) -> Dict[str, Chromatogram]:

        return self.chromatogram_records[key]

    def __setitem__(self, key: str, value: Dict[str, Chromatogram]) -> None:

        self.chromatogram_records[key] = value

    def get(
        self, precursor_id: str, unmodified_sequence: str
    ) -> Optional[Dict[str, Chromatogram]]:

        chromatograms = self.chromatogram_records[precursor_id]

        return_dict = dict()

        for key, chromatogram in chromatograms.items():

            if chromatogram.peptide_sequence == unmodified_sequence:

                return_dict[key] = chromatogram

        # TODO: check if this can be removed
        if len(return_dict) > 6:

            return None

        return return_dict

    def __get_chromatogram_lengths(self) -> Set[int]:

        chrom_lengths = set()

        for chromatograms in self.chromatogram_records.values():

            for record_id, record in chromatograms.items():

                if record.type != "precursor":
                    chrom_length = len(record.rts)

                    chrom_lengths.add(chrom_length)
                    break

        return chrom_lengths

    def rt_span(self) -> np.ndarray:

        return np.array(self.__get_chromatogram_lengths())

    def min_chromatogram_length(self) -> int:

        return min(self.__get_chromatogram_lengths())

    def max_chromatogram_length(self) -> int:

        return max(self.__get_chromatogram_lengths())
