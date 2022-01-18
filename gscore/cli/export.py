import argparse
from typing import Union

from gscore.chromatograms import Chromatograms
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile
from gscore.parsers.sqmass import SqMassFile


class Export:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "export"

    def __call__(self, args: argparse.Namespace):

        with OSWFile(args.input) as osw_file:

            print(f"Parsing {args.input}")

            use_chromatograms: bool = False

            if args.chromatogram_file:

                precursors = osw_file.parse_to_precursors(
                    query=queries.SelectPeakGroups.FETCH_CHROMATOGRAM_TRAINING_RECORDS
                )

                print("Parsing Chromatograms...")

                with SqMassFile(args.chromatogram_file) as chrom_file:

                    chromatograms = chrom_file.parse()

                print("Matching chromatograms with precursors...")

                for precursor in precursors:

                    precursor.set_chromatograms(
                        chromatograms=chromatograms.get(precursor)
                    )

                use_chromatograms = True

            else:

                precursors = osw_file.parse_to_precursors(
                    query=queries.SelectPeakGroups.FETCH_DENOIZED_REDUCED
                )

            print(f"Filtering and writing output.")

            precursors.dump_training_data(
                args.output,
                filter_field=args.filter_field,
                filter_value=args.filter_value,
                use_chromatograms=use_chromatograms,
                max_chrom_length=chromatograms.max_chromatogram_length()
            )

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name, help="Filter peakgroups and export training data."
        )

        self.parser.add_argument(
            "-i",
            "--input",
            dest="input",
            help="OSW file to filter and export training data",
            type=str,
        )

        self.parser.add_argument(
            "--filter-field",
            dest="filter_field",
            choices=["PROBABILITY", "VOTE_PERCENTAGE"],
            default="VOTE_PERCENTAGE",
            help="Field to filter the peakgroups for export as training data.",
        )

        self.parser.add_argument(
            "--filter-value",
            dest="filter_value",
            type=float,
            default=1.0,
            help="Set value by which the target peakgroups are filtered on the filter-field",
        )

        self.parser.add_argument("-o", "--output", dest="output", help="Output file ")

        self.parser.add_argument(
            "-c",
            "--chromatogram-file",
            dest="chromatogram_file",
            help="File containing chromatograms associated with the peakgroups.",
            default=""
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):
        return f"<Export> {self.name}"
