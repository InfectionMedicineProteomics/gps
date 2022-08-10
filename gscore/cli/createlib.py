import argparse
import csv

from typing import Any

from gscore.parsers.pqp_file import PQPFile


def combine_predicted_peakgroups(predicted_peakgroup_files):
    predicted_peakgroups = dict()

    for predicted_peakgroup_file in predicted_peakgroup_files:

        with open(predicted_peakgroup_file) as predicted_file_h:

            reader = csv.DictReader(predicted_file_h, delimiter="\t")

            for row in reader:

                if row["Decoy"] == "0" and row["PeakgroupPrediction"] == "1.0":

                    record = {
                        "UnmodifiedSequence": row["UnmodifiedSequence"],
                        "ModifiedSequence": row["ModifiedSequence"],
                        "Charge": int(row["Charge"]),
                        "Protein": row["Protein"],
                        "Decoy": int(row["Decoy"]),
                        "RT": float(row["RT"])
                    }

                    precursor_id = f"{row['ModifiedSequence']}_{row['Charge']}"

                    if precursor_id not in predicted_peakgroups:
                        predicted_peakgroups[precursor_id] = record

    return list(predicted_peakgroups.values())


def create_new_library(new_library_path, library_annotated_transitions, used_precursors):

    library_transitions = []

    for transition in library_annotated_transitions:

        precursor_id = f"{transition['MODIFIED_SEQUENCE']}_{transition['PRECURSOR_CHARGE']}"

        if precursor_id in used_precursors:

            transition_record = {
                "PrecursorMz": transition["PRECURSOR_MZ"],
                "ProductMz": transition["PRODUCT_MZ"],
                "LibraryIntensity": transition["LIBRARY_INTENSITY"],
                "RetentionTime": transition["LIBRARY_RT"],
                "ProteinId": transition["PROTEIN_ACCESSION"],
                "PeptideSequence": transition["UNMODIFIED_SEQUENCE"],
                "ModifiedPeptideSequence": transition["MODIFIED_SEQUENCE"],
                "PrecursorCharge": transition["PRECURSOR_CHARGE"],
                "ProductCharge": transition["PRODUCT_CHARGE"],
                "FragmentType": transition["TYPE"],
                "FragmentSeriesNumber": transition["ORDINAL"],
                "Annotation": transition["ANNOTATION"]
            }

            library_transitions.append(transition_record)

    return library_transitions

def write_csv_library(file_path, library_transitions):

    with open(file_path, "w") as library_h:

        writer = csv.DictWriter(
            library_h,
            fieldnames=list(library_transitions[0].keys()),
            delimiter="\t"
        )

        writer.writeheader()

        for library_transition in library_transitions:

            writer.writerow(library_transition)

class CreateLib:
    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:
        self.name = "createlib"

    def __call__(self, args: argparse.Namespace) -> None:
        predicted_peakgroups = combine_predicted_peakgroups(args.input_files)

        print(predicted_peakgroups[0])

        print("Parsing spectral library.")

        pqp_file = PQPFile(args.pqp)

        library_annotated_transitions = pqp_file.get_annotated_transitions()

        used_precursors = set(
            [f"{row['ModifiedSequence']}_{row['Charge']}" for row in predicted_peakgroups]
        )

        print("Subsetting library")

        library_transitions = create_new_library(
            args.output,
            library_annotated_transitions,
            used_precursors
        )

        print("Writing new library")

        write_csv_library(
            args.output,
            library_transitions
        )

    def build_subparser(self, subparser: Any) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help="Combine multiple predicted runs into PQP spectral library.",
        )

        self.parser.add_argument(
            "--input-files",
            dest="input_files",
            nargs="+",
            help="List of files to compile",
        )

        self.parser.add_argument("-o", "--output", dest="output", help="Output file spectral library")

        self.parser.add_argument(
            "--pqp",
            dest="pqp",
            help="Base spectral library to extract fragments from"
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:
        return f"<Export> {self.name}"
