def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input_files',
        help='Scored files for building model',
        nargs='+'
    )

    parser.add_argument(
        '-o',
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
        '-s',
        '--scoring-level',
        dest='scoring_level',
        help=(
            'Indicates if a peptide or protein level global scoring model should be used. '
            'The default is (peptide), for peptide level scoring. Change to protein if '
            'protein level scoring is desired.'
        ),
        choices=[
            'peptide', 'protein'
        ],
        default='peptide'
    )

    parser.add_argument(
        '-sc',
        '--score-column',
        dest='score_column',
        help='Level of verbosity to use, corresponds to python log levels',
        choices=[
            'weighted_d_score', 'd_score'
        ],
        default='weighted_d_score'
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

    return