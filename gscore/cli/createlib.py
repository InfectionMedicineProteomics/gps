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
    protein_records = dict()
    peptide_protein_mapping = dict()
    peptide_records = dict()
    precursor_peptide_mapping = dict()
    precursor_records = dict()
    transition_precursor_mapping = dict()
    transition_records = dict()

    for transition in library_annotated_transitions:

        precursor_id = f"{transition['MODIFIED_SEQUENCE']}_{transition['CHARGE']}"

        if precursor_id in used_precursors:

            if transition["PROTEIN_ACCESSION"] not in protein_records:
                protein_records[transition["PROTEIN_ACCESSION"]] = {
                    "ID": transition["PROTEIN_ID"],
                    "PROTEIN_ACCESSION": transition["PROTEIN_ACCESSION"],
                    "DECOY": transition["DECOY"]
                }

            peptide_protein_id = f"{transition['PEPTIDE_ID']}_{transition['PROTEIN_ID']}"

            if peptide_protein_id not in peptide_protein_mapping:
                peptide_protein_mapping[peptide_protein_id] = {
                    "PEPTIDE_ID": transition["PEPTIDE_ID"],
                    "PROTEIN_ID": transition["PROTEIN_ID"]
                }

            if transition["MODIFIED_SEQUENCE"] not in peptide_records:
                peptide_records[transition["MODIFIED_SEQUENCE"]] = {
                    "ID": transition["PEPTIDE_ID"],
                    "UNMODIFIED_SEQUENCE": transition["UNMODIFIED_SEQUENCE"],
                    "MODIFIED_SEQUENCE": transition["MODIFIED_SEQUENCE"],
                    "DECOY": transition["DECOY"]
                }

            precursor_peptide_id = f"{transition['PRECURSOR_ID']}_{transition['PEPTIDE_ID']}"

            if precursor_peptide_id not in precursor_peptide_mapping:
                precursor_peptide_mapping[precursor_peptide_id] = {
                    "PRECURSOR_ID": transition["PRECURSOR_ID"],
                    "PEPTIDE_ID": transition["PEPTIDE_ID"]
                }

            if precursor_id not in precursor_records:
                precursor_records[precursor_id] = {
                    "ID": transition["PRECURSOR_ID"],
                    "TRAML_ID": transition["PRECURSOR_TRAML_ID"],
                    "PRECURSOR_MZ": transition["PRECURSOR_MZ"],
                    "CHARGE": transition["PRECURSOR_CHARGE"],
                    "LIBRARY_RT": transition["LIBRARY_RT"],
                    "DECOY": transition["DECOY"]
                }

            transition_precursor_id = f"{transition['TRANSITION_ID']}_{transition['PRECURSOR_ID']}"

            if transition_precursor_id not in transition_precursor_mapping:
                transition_precursor_mapping[transition_precursor_id] = {
                    "TRANSITION_ID": transition["TRANSITION_ID"],
                    "PRECURSOR_ID": transition["PRECURSOR_ID"]
                }

            if transition["TRANSITION_ID"] not in transition_records:
                transition_records[transition["TRANSITION_ID"]] = {
                    "ID": transition["TRANSITION_ID"],
                    "TRAML_ID": transition["TRANSITION_TRAML_ID"],
                    "PRODUCT_MZ": transition["PRODUCT_MZ"],
                    "CHARGE": transition["TRANSITION_CHARGE"],
                    "TYPE": transition["TYPE"],
                    "ANNOTATION": transition["ANNOTATION"],
                    "ORDINAL": transition["ORDINAL"],
                    "DETECTING": transition["DETECTING"],
                    "IDENTIFYING": transition["IDENTIFYING"],
                    "QUANTIFYING": transition["QUANTIFYING"],
                    "LIBRARY_INTENSITY": transition["LIBRARY_INTENSITY"],
                    "DECOY": transition["DECOY"]
                }

    protein_records = protein_records.values()
    peptide_protein_mapping = peptide_protein_mapping.values()
    peptide_records = peptide_records.values()
    precursor_peptide_mapping = precursor_peptide_mapping.values()
    precursor_records = precursor_records.values()
    transition_precursor_mapping = transition_precursor_mapping.values()
    transition_records = transition_records.values()

    new_library = PQPFile(new_library_path)
    new_library.create_tables()

    gene_records = [
        {
            "gene_name": "NA",
            "decoy": 0
        }
    ]

    version_record = [
        {
            "id": 3
        }
    ]

    new_library.add_records("VERSION", version_record)
    new_library.add_records("GENE", gene_records)

    peptide_gene_map_ids = []

    added_genes = new_library.get_genes()

    gene_id = added_genes[0]["ID"]

    for peptide_protein_mapping_record in peptide_protein_mapping:
        peptide_gene_map_ids.append(
            {
                "peptide_id": peptide_protein_mapping_record["PEPTIDE_ID"],
                "gene_id": gene_id
            }
        )

    new_library.add_records("PEPTIDE_GENE_MAPPING", peptide_gene_map_ids)

    new_library.add_records("PROTEIN", protein_records)
    new_library.add_records("PEPTIDE", peptide_records)
    new_library.add_records("PRECURSOR", precursor_records)
    new_library.add_records("TRANSITION", transition_records)
    new_library.add_records("PEPTIDE_PROTEIN_MAPPING", peptide_protein_mapping)
    new_library.add_records("PRECURSOR_PEPTIDE_MAPPING", precursor_peptide_mapping)
    new_library.add_records("TRANSITION_PRECURSOR_MAPPING", transition_precursor_mapping)

    print(len(precursor_records))
    print(len(peptide_records))
    print(len(protein_records))


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

        create_new_library(
            args.output,
            library_annotated_transitions,
            used_precursors
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
