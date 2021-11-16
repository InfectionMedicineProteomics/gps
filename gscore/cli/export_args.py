import argparse

def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input_osw_file',
        help='Scored OSW files to export'
    )

    parser.add_argument(
        '--input-pyprophet-files',
        dest='input_pyprophet_osw_files',
        nargs='+',
        help='PyProphet scored OSW files for export and global cutoff'
    )

    parser.add_argument(
        '--max-peakgroup-q-value',
        dest='max_peakgroup_q_value',
        help="Max q-value for peakgroup inclusion in the output matrix",
        type=float,
        default=0.05
    )

    parser.add_argument(
        '-o',
        '--output-file',
        dest='output_file',
        help='Exported csv file',
        default=''
    )

    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbosity_level',
        help='Level of verbosity to use, corresponds to python log levels',
        choices=[
            'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
        ],
        default='INFO'
    )

    parser.add_argument(
        '-m',
        '--export-method',
        dest='export_method',
        choices=[
            'tric-formatted', 'comprehensive', 'peptide', 'protein', 'pyprophet'
        ],
        default='comprehensive',
        help='Which format to export results'
    )

    return parser