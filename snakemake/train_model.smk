#!/usr/bin/env python

SAMPLES, = glob_wildcards(f"{config['training_dir']}/{{samples}}.osw")

rule all:
    input:
        f"{config['model_dir']}/{config['model_name']}_trained",
        f"{config['model_dir']}/{config['model_name']}.scaler.pkl"
        
rule denoise_target_labels:
    input:
        osw_file = f"{config['training_dir']}/{{sample}}.osw"
    output:
        placeholder_output = f"{config['training_dir']}/denoised/{{sample}}"
    params:
        num_classifiers = 100,
        num_folds = 20
    threads:
        10
    run:
        shell(
            "gscore scorerun "
            "--denoise-only "
            "-i {input.osw_file} "
            "--num-classifiers {params.num_classifiers} "
            "--num-folds {params.num_folds} "
            "--threads {threads} "
            "-v NOTSET"
        )
        
        shell(
            "touch {output.placeholder_output}"
        )


rule train_model:
    input:
        placeholder_input = expand(f"{config['training_dir']}/denoised/{{sample}}", sample=SAMPLES),
        osw_files = expand(f"{config['training_dir']}/{{sample}}.osw", sample=SAMPLES)
    output:
        placeholder_output = f"{config['model_dir']}/{config['model_name']}_trained",
        scaler = f"{config['model_dir']}/{config['model_name']}.scaler.pkl"
    params:
        model_dir = config['model_dir'],
        model_name = config['model_name'],
        log_level = config['log_level']
    run:
        shell(
            "gscore buildscorer "
            "-i {input.osw_files} "
            "--outmodeldir {params.model_dir} "
            "--model-name {params.model_name} "
            "-v {params.log_level}"
        )

        shell(
            "touch {output.placeholder_output}"
        )
