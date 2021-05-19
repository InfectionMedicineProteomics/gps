#!/usr/bin/env python

SAMPLES, = glob_wildcards(f"{config['osw_dir']}/{{samples}}.osw")

rule all:
    input:
        placeholder_output = expand(f"{config['osw_dir']}/scored/210519/{{sample}}", sample=SAMPLES)

rule score_osw_runs:
    input:
        osw_file = f"{config['osw_dir']}/{{sample}}.osw"
    output:
        placeholder_output = f"{config['osw_dir']}/scored/210519/{{sample}}"
    params:
        scoring_model = config['scoring_model']['path'],
        scoring_scaler = config['scoring_model']['scaler'],
        log_level = config['gscore']['scorerun']['log_level'],
       	num_classifiers = config['gscore']['scorerun']['num_classifiers'],
        num_folds = config['gscore']['scorerun']['num_folds']
    threads:
        config['gscore']['scorerun']['threads']
    run:
        shell(
            "gscore scorerun "
            "-i {input.osw_file} "
            "-m {params.scoring_model} "
            "-s {params.scoring_scaler} "
            "--num-classifiers {params.num_classifiers} "
            "--num-folds {params.num_folds} "
            "--threads {threads} "
            "-v {params.log_level}"
        )

        shell(
            "touch {output.placeholder_output}"
        )