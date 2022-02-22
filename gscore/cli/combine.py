import argparse

from pathlib import Path

from gscore.combiner import PrecursorExport, PrecursorExportRecord
from gscore.fdr import GlobalDistribution
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile


class Combine:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "combine"

    def __call__(self, args: argparse.Namespace):

        export = PrecursorExport(max_q_value=args.max_peakgroup_q_value)

        global_protein_model = GlobalDistribution.load(args.protein_model)
        global_peptide_model = GlobalDistribution.load(args.peptide_model)

        for input_file in args.input_files:

            print(f"Parsing {input_file}...")

            sample_name = Path(input_file).stem

            export.add_sample(sample_name)

            osw_file = OSWFile(input_file)

            precursors = osw_file.parse_to_precursors(
                query=queries.SelectPeakGroups.FETCH_PRECURSORS_FOR_EXPORT_REDUCED
            )

            for precursor in precursors:

                precursor_id = f"{precursor.modified_sequence}_{precursor.charge}"

                if precursor_id not in export:

                    export[precursor_id] = PrecursorExportRecord(
                        modified_sequence=precursor.modified_sequence,
                        charge=precursor.charge,
                        decoy=precursor.decoy,
                        protein_accession=precursor.protein_accession,
                        protein_q_value=global_protein_model.get_q_value(
                            precursor.protein_accession
                        ),
                        peptide_q_value=global_peptide_model.get_q_value(
                            precursor.modified_sequence
                        ),
                    )

                peakgroup = precursor.get_peakgroup(rank=1, key="D_SCORE")

                export[precursor_id].add_sample(
                    sample_key=sample_name, peakgroup=peakgroup
                )

        print(f"Writing {args.output}...")

        export.write(args.output)

    def build_subparser(self, subparser):

        self.parser = subparser.add_parser(
            self.name,
            help="Combine multiple scored runs into a single quantification matrix.",
        )

        self.parser.add_argument(
            "--input-files",
            dest="input_files",
            nargs="+",
            help="List of files to export to quant matrix",
        )

        self.parser.add_argument(
            "--peptide-model",
            dest="peptide_model",
            type=str,
            help="Path to the peptide scoring model",
        )

        self.parser.add_argument(
            "--protein-model",
            dest="protein_model",
            type=str,
            help="Path to the protein scoring model",
        )

        self.parser.add_argument("-o", "--output", dest="output", help="Output file ")

        self.parser.add_argument(
            "--max-peakgroup-q-value",
            dest="max_peakgroup_q_value",
            help="Max q-value for peakgroup inclusion in the output matrix",
            type=float,
            default=0.05,
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):
        return f"<Export> {self.name}"
