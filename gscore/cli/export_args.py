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

    parser.add_argument(
        '-f',
        '--input-files',
        dest='input_files',
        help='Indicate multiple input files to ',
        nargs='+'
    )

    parser.add_argument(
        '--quant-type',
        dest='quant_type',
        help='Whether to use MS1 or MS2 based quantification from swath data',
        choices=[
            'ms1', 'ms2'
        ],
        default='ms2'
    )

    parser.add_argument(
        '--max-peakgroup-q-value',
        dest='max_peakgroup_q_value',
        help='What q-value allowed for inclusion of peakgroup in the matrix',
        type=float,
        default=0.05
    )

    parser.add_argument(
        '-om',
        '--outmodel',
        dest='model_output',
        help='Specify output path for the scoring model',
        default=''
    )

    parser.add_argument(
        '-d',
        '--use-decoys',
        dest='use_decoys',
        help=(
            'Set this flag if you wish to use DECOYS as false targets for scoring. '
            'The default is (False), or decoy-free mode'
        ),
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '-sc',
        '--score-column',
        dest='score_column',
        help='Level of verbosity to use, corresponds to python log levels',
        choices=[
            'weighted_d_score', 'd_score'
        ],
        default='d_score'
    )

    parser.add_argument(
        '--true-target-cutoff',
        dest='true_target_cutoff',
        help='What q-value allowed for inclusion of true target peakgroup in the matrix',
        type=float,
        default=0.8
    )

    parser.add_argument(
        '--false-target-cutoff',
        dest='false_target_cutoff',
        help='What q-value allowed for inclusion of false target peakgroup in the matrix',
        type=float,
        default=0.5
    )

    return parser