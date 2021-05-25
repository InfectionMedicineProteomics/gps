import argparse

def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input_files',
        help='OSW files for training scorer',
        nargs='+'
    )

    parser.add_argument(
        '-o',
        '--outmodeldir',
        dest='output_directory',
        help='Specify output directory for storing the static model',
        default=''
    )

    parser.add_argument(
        '-m',
        '--model-name',
        dest='model_name',
        help='Specify model name, used for naming the scaler and ',
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
        '--true-target-cutoff',
        dest='true_target_cutoff',
        help='What q-value allowed for inclusion of peakgroup in the matrix',
        type=float,
        default=0.8
    )

    parser.add_argument(
        '--false-target-cutoff',
        dest='false_target_cutoff',
        help='What q-value allowed for inclusion of peakgroup in the matrix',
        type=float,
        default=0.8
    )

    parser.add_argument(
        '--vote-threshold',
        dest='vote_threshold',
        help='The probability threshold to consider for a positive vote',
        default=0.8,
        type=float
    )

    return