from __future__ import annotations

from pynumpress import decode_slof, decode_linear, decode_pic

from gps.chromatograms import Chromatogram, Chromatograms
import pyopenms

# FETCH_PEPTIDE_CHROMATOGRAM = """
#     select
#         PRECURSOR.PEPTIDE_SEQUENCE,
#         PRECURSOR.CHARGE,
#         PRECURSOR.ISOLATION_TARGET PRECURSOR_ISOLATION_TARGET,
#         CHROMATOGRAM.NATIVE_ID,
#         DATA.COMPRESSION,
#         DATA.DATA_TYPE,
#         DATA.DATA,
#         PRODUCT.ISOLATION_TARGET PRODUCT_ISOLATION_TARGET
#     from precursor
#     join CHROMATOGRAM on PRECURSOR.CHROMATOGRAM_ID = CHROMATOGRAM.ID
#     join DATA on CHROMATOGRAM.ID = DATA.CHROMATOGRAM_ID
#     join PRODUCT on CHROMATOGRAM.ID = PRODUCT.CHROMATOGRAM_ID;
#     """
#
# class CompressionType(Enum):
#     NO = 0
#     ZLIB = 1
#     NP_LINEAR = 2
#     NP_SLOF = 3
#     NP_PIC = 4
#     NP_LINEAR_ZLIB = 5
#     NP_SLOF_ZLIB = 6
#     NP_PIC_ZLIB = 7
#
#
# class ChromatogramDataType(Enum):
#     MZ = 0
#     INT = 1
#     RT = 2
#
# def decode_data(chromatogram_data, compression_type):
#
#     decoded_data = list()
#
#     if compression_type == CompressionType.NP_LINEAR_ZLIB.value:
#
#         decompressed = bytearray(zlib.decompress(chromatogram_data))
#
#         if len(decompressed) > 0:
#             decoded_data = decode_linear(decompressed)
#         else:
#             decoded_data = [0]
#
#     elif compression_type == CompressionType.NP_SLOF_ZLIB.value:
#
#         decompressed = bytearray(zlib.decompress(chromatogram_data))
#
#         if len(decompressed) > 0:
#             decoded_data = decode_slof(decompressed)
#         else:
#             decoded_data = [0]
#
#     return decoded_data


class SqMassFile:
    def __init__(self, file_path: str):

        self.file_path = file_path

    def parse(self, level: str = "ms2") -> Chromatograms:

        chromatograms = pyopenms.MSExperiment()
        sqmass_file = pyopenms.SqMassFile()
        sq_mass_config = pyopenms.SqMassConfig()
        sq_mass_config.write_full_meta = True

        sqmass_file.setConfig(sq_mass_config)

        sqmass_file.load(self.file_path, chromatograms)

        chromatogram_records = Chromatograms()

        for chromatogram in chromatograms.getChromatograms():

            retention_times, intensities = chromatogram.get_peaks()

            native_id = chromatogram.getNativeID()

            precursor = chromatogram.getPrecursor()

            precursor_id = f"{precursor.getMZ()}_{precursor.getCharge()}"

            if precursor_id not in chromatogram_records:

                chromatogram_records[precursor_id] = dict()

            if level == "ms2":

                if "Precursor" not in native_id:

                    chromatogram_records[precursor_id][native_id] = Chromatogram(
                        type="fragment",
                        chrom_id=native_id,
                        precursor_mz=precursor.getMZ(),
                        charge=precursor.getCharge(),
                        peptide_sequence=precursor.getMetaValue("peptide_sequence"),
                        rts=retention_times,
                        intensities=intensities,
                    )

        return chromatogram_records
