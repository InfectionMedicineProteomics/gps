from gscore.parsers.osw import osw
from gscore.parsers.osw.queries import (
    SelectPeakGroups
)

def main(args, logger):

    if args.export_method == 'tric-formatted':

        peak_groups = osw.fetch_peak_groups(
            host=args.input_osw_file,
            query=SelectPeakGroups.FETCH_TRIC_EXPORT_DATA,
            preprocess=False
        )

        highest_scoring = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['m_score'],
            ascending=True
        )   

        highest_scoring.to_csv(
            args.output_tsv_file,
            sep='\t',
            index=False
        )

    elif args.export_method == 'peptide':

        pass