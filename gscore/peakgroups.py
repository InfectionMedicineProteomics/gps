import numpy as np

class Protein:

    protein_accession: str
    decoy: int

    def __init__(self, protein_accession='', decoy=0, q_value=None):

        self.protein_accession = protein_accession

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value

        self.scores = dict()


class Precursor:
    sequence: str
    modified_sequence: str
    charge: int
    decoy: int
    target: int

    def __init__(self, sequence='', modified_sequence='', charge=0, decoy=0, q_value=None):

        self.sequence = sequence
        self.modified_sequence = modified_sequence
        self.charge = charge
        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value

        self.peakgroups = []

        self.scores = dict()


class PeakGroup:
    mz: float
    rt: float
    ms2_intensity: float
    ms1_intensity: float
    decoy: int
    target: int
    delta_rt: float
    start_rt: float
    end_rt: float

    def __init__(self, mz=0.0, rt=0.0, intensity=None, q_value=None, ms2_intensity=0.0, ms1_intensity=0.0, decoy=0):

        self.retention_time = rt
        self.intensity = intensity
        self.q_value = q_value

        self.mz = mz
        self.ms2_intensity = ms2_intensity
        self.ms1_intensity = ms1_intensity
        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.sub_scores = dict()
        self.scores = dict()

        self.delta_rt = 0.0
        self.start_rt = 0.0
        self.end_rt = 0.0

        self.scaled_rt_start = 0.0
        self.scaled_rt_apex = 0.0
        self.scaled_rt_end = 0.0

    def add_score_column(self, key, value):

        self.scores[key] = value

    def add_sub_score_column(self, key, value):

        self.sub_scores[key] = value

    def get_score_columns(self):

        score_columns = list()

        for score_column in self.sub_scores.keys():
            score_columns.append(score_column)

        return score_columns

    def get_sub_score_column_array(self, include_score_columns=True):

        score_values = list()

        for score_value in self.sub_scores.values():
            score_values.append(score_value)

        if include_score_columns:

            for score_value in self.scores.values():
                score_values.append(score_value)

        return np.asarray(score_values, dtype=np.double)



def get_score_array(graph, node_list, score_column=''):

    score_array = list()

    for node in graph.iter(keys=node_list):

        if node.color == 'peakgroup':

            score = node.data.scores[score_column]
            
            score_array.append(score)

        else:

            print(node.color, node.data.scores, node.key, node._edges)

    return np.array(score_array)


def calc_score_grouped_by_level(graph, function=None, level='', score_column='', new_column_name=''):

    for node in graph.iter(color=level):

        scores = list()

        for key in node.get_edges():

            scores.append(
                graph[key].data.scores[score_column]
            )

        scores = np.array(scores, dtype=np.float64)

        score = function(scores)

        #TODO: This is not optimal. Should instead set at parent node level
        for key in node.get_edges():

            graph[key].data.scores[new_column_name] = score


def apply_scoring_model(graph, level, model, score_column):

    for peakgroup_node in graph.get_nodes("peakgroup"):

        peakgroup_node = graph[peakgroup_node]

        score = peakgroup_node.scores["d_score"]

        peakgroup_node.scores[f"{level}_q_value"] = model.calc_q_value(score)
