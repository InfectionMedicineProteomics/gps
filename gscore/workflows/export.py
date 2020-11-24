import pandas as pd
import numpy as np

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_EXPORT_DATA
)

def main(args, logger):

    if args.export_method == 'tric-formatted':
        peak_groups = fetch_peak_groups(
            host=args.input_osw_file,
            query=FETCH_EXPORT_DATA
        )

        highest_scoring = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['m_score'],
            ascending=True
        )   

        highest_scoring.to_csv(
            args.output_tsv_file,
            sep='\t'
        )