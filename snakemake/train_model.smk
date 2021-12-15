SAMPLES, = glob_wildcards(f"{config['base_file_paths']['osw']}/{{sample}}.osw")


rule all:
    input:
        scoring_model=f"{config['base_file_paths']['results']}/models/{config['gscore']['model_name']}",
        scaler=f"{config['base_file_paths']['results']}/models/{config['gscore']['scaler_name']}"

rule gscore_denoise:
    input:
        osw_file=f"{config['base_file_paths']['osw']}/{{sample}}.osw"
    output:
        placeholder=f"{config['base_file_paths']['results']}/denoised/{{sample}}",
    params:
        num_classifiers=10,
        num_folds=10,
        vote_percentage=0.5
    threads:
        4
    shell:
        (
            "gscore denoise "
            "-i {input.osw_file} "
            "--num-classifiers {params.num_classifiers} "
            "--num-folds {params.num_folds} "
            "--vote-percentage {params.vote_percentage} "
            "--threads {threads} &&"

            "touch {output.placeholder}"
        )

rule gscore_export_training_data:
    input:
        placeholder=f"{config['base_file_paths']['results']}/denoised/{{sample}}",
        osw_file=f"{config['base_file_paths']['osw']}/{{sample}}.osw"
    output:
        npz=f"{config['base_file_paths']['results']}/training_data/{{sample}}.npz"
    params:
        export_method="training-data"
    threads:
        1
    shell:
        (
            "gscore export "
            "--export-method {params.export_method} "
            "--input {input.osw_file} "
            "--output {output.npz}"
        )

rule gscore_train_model:
    input:
        npz=expand(f"{config['base_file_paths']['results']}/training_data/{{sample}}.npz",sample=SAMPLES)
    output:
        scoring_model=f"{config['base_file_paths']['results']}/models/{config['gscore']['model_name']}",
        scaler=f"{config['base_file_paths']['results']}/models/{config['gscore']['scaler_name']}"
    threads:
        12
    shell:
        (
            "gscore train "
            "--input {input.npz} "
            "--model-output {output.scoring_model} "
            "--scaler-output {output.scaler}"
        )