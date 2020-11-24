import pandas as pd

from gscore.osw.connection import OSWConnection


class PeakGroup:
    
    def __init__(self, peak_group):
        
        self.peak_group = peak_group
        
    def sort(self, rerank_keys=[], ascending=True):
        
        self.peak_group = self.peak_group.sort_values(
            by=rerank_keys,
            ascending=ascending
        )


class PeakGroupList:
    
    def __init__(self, peak_groups):
        
        self.peak_groups = list()
        
        for _, group in peak_groups.items():
            self.peak_groups.append(PeakGroup(group))
            
        self.sort_key = None
        
    @property
    def ml_features(self):

        ml_features = [
            col for col in self.peak_groups[0].peak_group.columns
            if col.startswith('var')
        ]
        
        return ml_features
    
    def rerank_groups(self, rerank_keys=[], ascending=True):
        
        for peak_group in self.peak_groups:
            peak_group.sort(rerank_keys=rerank_keys, ascending=ascending)
            
        self.sort_key = rerank_keys
        
    
    def select_peak_group(self, rank=None, rerank_keys=[], ascending=True, return_all=False):
        
        if return_all:
            return pd.concat(
                [peak_group.peak_group for peak_group in self.peak_groups]
            )
        
        if rerank_keys != self.sort_key:
            
            self.rerank_groups(
                rerank_keys=rerank_keys,
                ascending=ascending
            )
        
        if isinstance(rank, list):
            rank = [r - 1 for r in rank]
        else:
            rank = rank - 1
        
        highest_ranking = list()

        for peak_group in self.peak_groups:
            
            try:
                highest_ranking.append(
                    peak_group.peak_group.iloc[rank]
                ) 
            except IndexError:
                pass

        return pd.DataFrame(highest_ranking)

    def select_proteotypic_peptides(self, rerank_keys=[]):
        
        all_peptides = self.select_peak_group(
            rank=1,
            rerank_keys=rerank_keys,
            ascending=False
        )

        all_peptides['peptide_sequence_charge'] = all_peptides.apply(
            lambda row: '{}_{}'.format(row['peptide_sequence'], row['charge']),
            axis=1
        )

        proteotypic_counts = pd.DataFrame(
            all_peptides['peptide_sequence_charge'].value_counts(),
        ).reset_index()

        proteotypic_counts.columns = ['peptide_charge', 'count']

        proteotypic_peptides = list(
            proteotypic_counts[
                proteotypic_counts['count'] == 1
            ]['peptide_charge']
        )

        return proteotypic_peptides

    def select_protein_groups(self, protein_group_column='protein_accession', rerank_keys=[]):

        all_peptides = self.select_peak_group(
            rank=1,
            rerank_keys=rerank_keys,
            ascending=False
        )

        protein_groups = all_peptides.groupby(
            protein_group_column
        )

        protein_groups = [
            group for _, group in protein_groups
        ]

        return protein_groups



def add_target_column(df):

    df.loc[df['decoy'] == 0, 'target'] = 1
    df.loc[df['decoy'] == 1, 'target'] = 0
    
    return df

def preprocess_data(osw_df):
    
    osw_df = osw_df.dropna(how='all', axis=1)
    
    osw_df.columns = map(str.lower, osw_df.columns)
    
    osw_df = add_target_column(osw_df)
    
    return osw_df


def fetch_peak_groups(host='', query='',):

    peak_groups = list()

    with OSWConnection(host) as conn:

        for record in conn.iterate_records(query):

            peak_groups.append(record)

    peak_groups = preprocess_data(
        pd.DataFrame(peak_groups)
    )
    
    split_peak_groups = dict(
        tuple(
            peak_groups.groupby('transition_group_id')
        )
    )

    peak_groups = PeakGroupList(split_peak_groups)

    return peak_groups
