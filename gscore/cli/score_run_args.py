

def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input',
        help='Scored OSW files for q value calculation'
    )

    parser.add_argument(
        '-o',
        '--outfile',
        dest='output',
        help='Specify output directory',
        default=''
    )

    parser.add_argument(
        '-m',
        '--model-path',
        dest='model_path',
        help='Specify model path',
    )

    parser.add_argument(
        '-s',
        '--scaler-path',
        dest='scaler_path',
        help='Specify scaler name',
    )

    parser.add_argument(
        '-nc',
        '--num-classifiers',
        dest='num_classifiers',
        help='The number of ensemble learners used to denoise each fold',
        default=100,
        type=int
    )

    parser.add_argument(
        '-f',
        '--num-folds',
        dest='num_folds',
        help='The number of folds used to denoise the target labels',
        default=10,
        type=int
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
        '-t',
        '--threads',
        dest='threads',
        help='The number of threads used to denoise the target labels',
        default=10,
        type=int
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

    return parser