from typing import Dict


class Peptide:

    sequence: str
    modified_sequence: str
    decoy: int
    target: int
    q_value: float
    d_score: float

    def __init__(
        self,
        sequence="",
        modified_sequence: str = "",
        decoy: int = 0,
        q_value: float = 0.0,
        d_score: float = 0.0,
    ):

        self.sequence = sequence
        self.modified_sequence = modified_sequence

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value
        self.d_score = d_score

    @property
    def identifier(self):

        return self.modified_sequence


class Peptides:

    peptides: Dict[str, Peptide]

    def __init__(self):

        self.peptides = dict()

    def __contains__(self, item: str):

        return item in self.peptides

    def __setitem__(self, key: str, peptide: Peptide):

        self.peptides[key] = peptide

    def __getitem__(self, key: str) -> Peptide:

        return self.peptides[key]

    def __iter__(self):

        for modified_peptide_sequence, peptide in self.peptides.items():

            yield peptide
