from typing import Dict, Generator


class Protein:

    protein_accession: str
    decoy: int
    target: int
    q_value: float
    d_score: float
    probability: float
    scores: Dict[str, float]

    def __init__(
        self,
        protein_accession: str = "",
        decoy: int = 0,
        q_value: float = 0.0,
        d_score: float = 0.0,
        probability: float = 0.0,
    ):

        self.protein_accession = protein_accession

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value

        self.d_score = d_score

        self.probability = probability

        self.scores = dict()

    @property
    def identifier(self) -> str:

        return self.protein_accession


class Proteins:

    proteins: Dict[str, Protein]

    def __init__(self) -> None:

        self.proteins = dict()

    def __contains__(self, item: str) -> bool:

        return item in self.proteins

    def __setitem__(self, key: str, protein: Protein) -> None:

        self.proteins[key] = protein

    def __getitem__(self, item: str) -> Protein:

        return self.proteins[item]

    def __iter__(self) -> Generator[Protein, None, None]:

        for protein_accession, protein in self.proteins.items():

            yield protein
