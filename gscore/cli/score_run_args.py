

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

    denoising_classifier_group = parser.add_argument_group()

    denoising_classifier_group.add_argument(
        '-nc',
        '--num-classifiers',
        dest='num_classifiers',
        help='The number of ensemble learners used to denoise each fold',
        default=100,
        type=int
    )

    denoising_classifier_group.add_argument(
        '-f',
        '--num-folds',
        dest='num_folds',
        help='The number of folds used to denoise the target labels',
        default=10,
        type=int
    )

    denoising_classifier_group.add_argument(
        '--vote-threshold',
        dest='vote_threshold',
        help='The probability threshold to consider for a positive vote',
        default=0.8,
        type=float
    )

    denoising_classifier_group.add_argument(
        '-sc',
        '--score-column',
        dest='score_column',
        help='Level of verbosity to use, corresponds to python log levels',
        choices=[
            'weighted_d_score', 'd_score'
        ],
        default='d_score'
    )

    denoising_classifier_group.add_argument(
        '-t',
        '--threads',
        dest='threads',
        help='The number of threads used to denoise the target labels',
        default=10,
        type=int
    )

    denoise_only_group = parser.add_argument_group()

    denoise_only_group.add_argument(
        '-dn',
        '--denoise-only',
        dest='denoise_only',
        help=(
            'Set this flag if you want to only denoise the data, and not calculate the q-values. '
            'This is done if you are training a new model.'
        ),
        default=False,
        action='store_true'
    )

    apply_static_model_group = parser.add_argument_group()

    apply_static_model_group.add_argument(
        '-m',
        '--model-path',
        dest='model_path',
        help='Specify model path.',
    )

    apply_static_model_group.add_argument(
        '-s',
        '--scaler-path',
        dest='scaler_path',
        help='Specify scaler name',
    )

    global_scoring_model_group = parser.add_argument_group()

    global_scoring_model_group.add_argument(
        '--apply-scoring-model',
        dest='apply_scoring_model',
        help=(
            'Set this flag if you want to only apply a global scoring model to a dataset. '
            'This is done if you want to parallelize the scoring of other already scored data.'
        ),
        default=False,
        action='store_true'
    )

    global_scoring_model_group.add_argument(
        '--scoring-model-path',
        dest='scoring_model_path',
        help='Specify scoring model path.',
    )

    return parser