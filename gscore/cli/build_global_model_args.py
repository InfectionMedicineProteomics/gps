def parse_args(parser):

    parser.add_argument(
        '-i',
        '--input',
        dest='input_files',
        help='Scored files for building model',
        nargs='+'
    )

    parser.add_argument(
        '--peakgroup-model',
        dest='peakgroup_model_output',
        help='Specify output path for the scoring model',
        default=''
    )

    parser.add_argument(
        '--peptide-model',
        dest='peptide_model_output',
        help='Specify output path for the scoring model',
        default=''
    )

    parser.add_argument(
        '--protein-model',
        dest='protein_model_output',
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
        default=0.9
    )

    parser.add_argument(
        '--false-target-cutoff',
        dest='false_target_cutoff',
        help='What q-value allowed for inclusion of peakgroup in the matrix',
        type=float,
        default=0.9
    )

    return