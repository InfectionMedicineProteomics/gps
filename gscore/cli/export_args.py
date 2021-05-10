import argparse

def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input_osw_file',
        help='Scored OSW files to export'
    )

    parser.add_argument(
        '-o',
        '--outfile',
        dest='output_tsv_file',
        help='Exported tsv file',
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
            'tric-formatted', 'scored', 'peptide', 'protein'
        ],
        help='Which format to export results'
    )

    parser.add_argument(
        '-f',
        '--input-files',
        dest='input_files',
        help='Indicate multiple input files to ',
        type=argparse.FileType('r'),
        nargs='+'
    )

    return parser