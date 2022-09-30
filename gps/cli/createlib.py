import argparse
import csv
import operator

from typing import Any

import numpy as np

from gps.parsers.pqp_file import PQPFile


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
                        "RT": float(row["RT"]),
                    }

                    precursor_id = f"{row['ModifiedSequence']}_{row['Charge']}"

                    if precursor_id not in predicted_peakgroups:
                        predicted_peakgroups[precursor_id] = record

    return list(predicted_peakgroups.values())


def get_scored_precursors(scored_precursors_file):

    scored_precursors = list()

    with open(scored_precursors_file) as scored_file_h:

        reader = csv.DictReader(scored_file_h, delimiter="\t")

        for row in reader:

            if row["Decoy"] == "0":

                if "QValue" in row:

                    record = {
                        "UnmodifiedSequence": row["UnmodifiedSequence"],
                        "ModifiedSequence": row["ModifiedSequence"],
                        "Charge": int(row["Charge"]),
                        "Protein": row["Protein"],
                        "Decoy": int(row["Decoy"]),
                        "RT": float(row["RT"]),
                        "Intensity": float(row["Intensity"]),
                        "QValue": float(row["QValue"]),
                        "DScore": float(row["DScore"]),
                        "Probability": float(row["Probability"]),
                    }

                elif "PeakgroupPrediction" in row:

                    record = {
                        "UnmodifiedSequence": row["UnmodifiedSequence"],
                        "ModifiedSequence": row["ModifiedSequence"],
                        "Charge": int(row["Charge"]),
                        "Protein": row["Protein"],
                        "Decoy": int(row["Decoy"]),
                        "RT": float(row["RT"]),
                        "PeakgroupPrediction": float(row["PeakgroupPrediction"]),
                        "PeakgroupScore": float(row["PeakgroupScore"]),
                    }

                scored_precursors.append(record)

    return scored_precursors


def calculate_rt_bins(scored_precursors, num_rt_bins):

    if "QValue" in scored_precursors[0]:

        retention_times = np.array(
            [row["RT"] for row in scored_precursors if row["QValue"] <= 0.15]
        )

    elif "PeakgroupPrediction" in scored_precursors[0]:

        retention_times = np.array(
            [
                row["RT"]
                for row in scored_precursors
                if row["PeakgroupPrediction"] == 1.0
            ]
        )

    retention_times.sort()

    hist_values, rt_edges = np.histogram(retention_times, bins=num_rt_bins)

    return rt_edges


def select_rt_precursors(rt_bins, scored_precursors, num_peptides_per_bin):

    selected_rt_peptides = []

    retention_times = np.array([float(row["RT"]) for row in scored_precursors])

    for i in range(1, len(rt_bins)):

        rt_left_bin = rt_bins[i - 1]
        rt_right_bin = rt_bins[i]

        rt_bin_indices = np.argwhere(
            (rt_left_bin <= retention_times) & (retention_times <= rt_right_bin)
        )

        rt_bin_peptides = [scored_precursors[i] for i in rt_bin_indices.ravel()]

        if rt_bin_peptides:

            if "DScore" in rt_bin_peptides[0]:

                rt_bin_peptides.sort(key=operator.itemgetter("DScore"), reverse=True)

            elif "PeakgroupScore" in rt_bin_peptides[0]:

                rt_bin_peptides.sort(
                    key=operator.itemgetter("PeakgroupScore"), reverse=True
                )

            selected_precursors_for_bin = rt_bin_peptides[:num_peptides_per_bin]

            selected_rt_peptides.extend(selected_precursors_for_bin)

    return selected_rt_peptides


def create_new_library(
    new_library_path, library_annotated_transitions, used_precursors
):
    library_transitions = []

    for transition in library_annotated_transitions:

        precursor_id = (
            f"{transition['MODIFIED_SEQUENCE']}_{transition['PRECURSOR_CHARGE']}"
        )

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
                "Annotation": transition["ANNOTATION"],
            }

            library_transitions.append(transition_record)

    return library_transitions


def write_csv_library(file_path, library_transitions):
    with open(file_path, "w") as library_h:
        writer = csv.DictWriter(
            library_h, fieldnames=list(library_transitions[0].keys()), delimiter="\t"
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

        print("Parsing spectral library.")

        pqp_file = PQPFile(args.pqp)

        library_annotated_transitions = pqp_file.get_annotated_transitions()

        used_precursors = set()

        if args.library_type == "standard":

            predicted_peakgroups = combine_predicted_peakgroups(args.input)

            used_precursors = set(
                [
                    f"{row['ModifiedSequence']}_{row['Charge']}"
                    for row in predicted_peakgroups
                ]
            )

        elif args.library_type == "rt":

            print("Selecting RT precursors")

            scored_precursors = get_scored_precursors(args.input[0])

            rt_edges = calculate_rt_bins(scored_precursors, args.num_rt_bins)

            selected_rt_precursors = select_rt_precursors(
                rt_edges, scored_precursors, args.num_peptides_per_bin
            )

            used_precursors = set(
                [
                    f"{row['ModifiedSequence']}_{row['Charge']}"
                    for row in selected_rt_precursors
                ]
            )

        print("Subsetting library")

        library_transitions = create_new_library(
            args.output, library_annotated_transitions, used_precursors
        )

        print("Writing new library")

        write_csv_library(args.output, library_transitions)

    def build_subparser(self, subparser: Any) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help="Combine multiple predicted runs into PQP spectral library.",
        )

        self.parser.add_argument(
            "--input",
            dest="input",
            nargs="+",
            help="List of files to compile",
        )

        self.parser.add_argument(
            "-o", "--output", dest="output", help="Output file spectral library"
        )

        self.parser.add_argument(
            "--pqp", dest="pqp", help="Base spectral library to extract fragments from"
        )

        self.parser.add_argument(
            "--library-type", dest="library_type", default="standard", type=str
        )

        self.parser.add_argument(
            "--num-rt-bins", dest="num_rt_bins", type=int, default=25
        )

        self.parser.add_argument(
            "--num-peptides-per-bin", dest="num_peptides_per_bin", type=int, default=5
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:
        return f"<Export> {self.name}"
