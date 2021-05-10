import pandas as pd

from gscore.utils.connection import Connection
from gscore.peakgroups import PeakGroupList

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