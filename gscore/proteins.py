from typing import Dict


class Protein:

    protein_accession: str
    decoy: int
    target: int
    q_value: float
    d_score: float
    probability: float
    scores: Dict[str, float]

    def __init__(self, protein_accession="", decoy=0, q_value=0.0, d_score=0.0, probability=0.0):

        self.protein_accession = protein_accession

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value

        self.d_score = d_score

        self.probability = probability

        self.scores = dict()

    @property
    def identifier(self):

        return self.protein_accession


class Proteins:

    proteins: Dict[str, Protein]

    def __init__(self):

        self.proteins = dict()

    def __contains__(self, item: str):

        return item in self.proteins

    def __setitem__(self, key: str, protein: Protein):

        self.proteins[key] = protein

    def __getitem__(self, item: str) -> Protein:

        return self.proteins[item]

    def __iter__(self):

        for protein_accession, protein in self.proteins.items():

            yield protein
