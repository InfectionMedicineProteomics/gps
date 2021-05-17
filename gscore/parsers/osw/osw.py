import pandas as pd

from gscore.utils.connection import Connection
from gscore.peakgroups import PeakGroupList

from gscore.parsers.osw.queries import (
    CreateIndex
)

from gscore.datastructures import (
    Graph,
    PeakGroup,
    Peptide,
    Protein
)

def fetch_peakgroup_graph(osw_path, osw_query):

    graph = Graph()

    with Connection(osw_path) as conn:

        for sql_index in CreateIndex.ALL_INDICES:
            conn.run_raw_sql(sql_index)

        for record in conn.iterate_records(osw_query):

            if str(record['protein_accession']) not in graph:
                protein = Protein(
                    key=str(record['protein_accession']),
                    decoy=record['protein_decoy']
                )

                graph.add_node(
                    key=protein.protein_accession,
                    data=protein,
                    color='protein'
                )

            if str(record['transition_group_id']) not in graph:
                peptide = Peptide(
                    key=str(record['transition_group_id']),
                    sequence=record['peptide_sequence'],
                    modified_sequence=record['modified_peptide_sequence'],
                    charge=int(record['charge']),
                    decoy=int(record['protein_decoy'])
                )

                graph.add_node(
                    key=peptide.key,
                    data=peptide,
                    color='peptide'
                )

                graph.add_edge(
                    node_from=str(record['protein_accession']),
                    node_to=peptide.key
                )

            if str(record['feature_id']) not in graph:

                peakgroup = PeakGroup(
                    key=str(record['feature_id']),
                    mz=record['mz'],
                    rt=record['rt'],
                    ms2_intensity=record['ms2_integrated_intensity'],
                    ms1_intensity=record['ms1_integrated_intensity']
                )

                for column_name, column_value in record.items():
                    column_name = column_name.lower()

                    if column_name.startswith('var_'):
                        peakgroup.add_sub_score_column(
                            key=column_name,
                            value=float(column_value)
                        )

                graph.add_node(
                    key=peakgroup.key,
                    data=peakgroup,
                    color='peakgroup'
                )

                graph.add_edge(
                    node_from=str(record['transition_group_id']),
                    node_to=peakgroup.key,
                    weight=peakgroup.sub_scores['var_xcorr_shape_weighted'],
                    directed=False
                )

    return graph



def add_target_column(df):
    df.loc[df['decoy'] == 0, 'target'] = 1
    df.loc[df['decoy'] == 1, 'target'] = 0

    return df


def preprocess_data(osw_df):
    osw_df = osw_df.dropna(how='all', axis=1)

    osw_df.columns = map(str.lower, osw_df.columns)

    osw_df = add_target_column(osw_df)

    return osw_df


def fetch_peak_groups(host='', query='', preprocess=True):
    peak_groups = list()

    with Connection(host) as conn:

        for record in conn.iterate_records(query):
            peak_groups.append(record)

    print(len(peak_groups))

    if preprocess:

        peak_groups = preprocess_data(
            pd.DataFrame(peak_groups)
        )

    else:

        peak_groups = pd.DataFrame(
            peak_groups
        )

    print(peak_groups.columns)

    split_peak_groups = dict(
        tuple(
            peak_groups.groupby('transition_group_id')
        )
    )

    peak_groups = PeakGroupList(split_peak_groups)

    return peak_groups
