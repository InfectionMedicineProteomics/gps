from csv import DictReader

from gps.peakgroups import PeakGroup
from gps.peptides import Peptides, Peptide
from gps.precursors import Precursors, Precursor
from gps.proteins import Proteins, Protein


class ScoreFile:
    def __init__(self, file_path: str) -> None:

        self.file_path = file_path

    def parse_to_peptides(self) -> Peptides:

        peptides = Peptides()

        with open(self.file_path) as score_file:

            reader = DictReader(score_file, delimiter="\t")

            for record in reader:

                if record["ModifiedSequence"] not in peptides:

                    peptide = Peptide(
                        sequence=record["UnmodifiedSequence"],
                        modified_sequence=record["ModifiedSequence"],
                        decoy=int(record["Decoy"]),
                        q_value=float(record["QValue"]),
                        d_score=float(record["DScore"]),
                    )

                    peptides[record["ModifiedSequence"]] = peptide

                elif (
                    float(record["DScore"])
                    > peptides[record["ModifiedSequence"]].d_score
                ):

                    peptide = Peptide(
                        sequence=record["UnmodifiedSequence"],
                        modified_sequence=record["ModifiedSequence"],
                        decoy=int(record["Decoy"]),
                        q_value=float(record["QValue"]),
                        d_score=float(record["DScore"]),
                    )

                    peptides[record["ModifiedSequence"]] = peptide

        return peptides

    def parse_to_proteins(self) -> Proteins:

        proteins = Proteins()

        with open(self.file_path) as score_file:

            reader = DictReader(score_file, delimiter="\t")

            for record in reader:

                if record["Protein"] not in proteins:

                    protein = Protein(
                        protein_accession=record["Protein"],
                        decoy=int(record["Decoy"]),
                        q_value=float(record["QValue"]),
                        d_score=float(record["DScore"]),
                    )

                    proteins[record["Protein"]] = protein

                elif float(record["DScore"]) > proteins[record["Protein"]].d_score:

                    protein = Protein(
                        protein_accession=record["Protein"],
                        decoy=int(record["Decoy"]),
                        q_value=float(record["QValue"]),
                        d_score=float(record["DScore"]),
                    )

                    proteins[record["Protein"]] = protein

        return proteins

    def parse_to_precursors(self) -> Precursors:

        precursors = Precursors()

        with open(self.file_path) as score_file:

            reader = DictReader(score_file, delimiter="\t")

            for idx, record in enumerate(reader):

                precursor_id = f"{record['ModifiedSequence']}_{record['Charge']}"

                if precursor_id not in precursors:

                    precursor = Precursor(
                        precursor_id=precursor_id,
                        charge=int(record["Charge"]),
                        decoy=int(record["Decoy"]),
                        mz=float(record.get("PrecursorMz", 0.0)),
                        modified_sequence=record["ModifiedSequence"],
                        unmodified_sequence=record["UnmodifiedSequence"],
                        protein_accession=record.get("Protein", ""),
                    )

                    precursors[precursor_id] = precursor

                peakgroup = PeakGroup(
                    idx=str(idx),
                    mz=float(record.get("PrecursorMz", 0.0)),
                    rt=float(record["RT"]),
                    decoy=int(record["Decoy"]),
                    intensity=float(record.get("Intensity", 0.0)),
                    probability=float(record.get("Probability", 0.0)),
                    q_value=float(record.get("QValue", 0.0)),
                    d_score=float(record.get("DScore", 0.0)),
                )

                precursors.add_peakgroup(precursor_id, peakgroup)

        return precursors
