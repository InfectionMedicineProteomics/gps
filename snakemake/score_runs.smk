SAMPLES, = glob_wildcards(f"{config['base_file_paths']['osw']}/{{samples}}.osw")

rule all:
    input:
        results = f"{config['base_file_paths']['results']}/quantification/{config['gscore']['combine']['output']}"

rule score_osw_runs:
    input:
        osw_file = f"{config['base_file_paths']['osw']}/{{sample}}.osw"
    output:
        placeholder = f"{config['base_file_paths']['results']}/scored/{{sample}}",
    params:
        scoring_model = config['scoring_model']['path'],
        scoring_scaler = config['scoring_model']['scaler'],
       	num_classifiers = config['gscore']['score']['num_classifiers'],
        num_folds = config['gscore']['score']['num_folds']
    threads:
        config['gscore']['score']['threads']
    run:
        shell(
            "gscore score "
            "--input {input.osw_file} "
            "--scoring-model {params.scoring_model} "
            "--scaler {params.scoring_scaler} "
            "--num-classifiers {params.num_classifiers} "
            "--num-folds {params.num_folds} "
            "--threads {threads} && "
            
            "touch {output.placeholder}"
        )

rule gscore_build_peptide_model:
    input:
        score_placeholder = expand(f"{config['base_file_paths']['results']}/scored/{{sample}}", sample=SAMPLES),
        osws = expand(f"{config['base_file_paths']['osw']}/{{sample}}.osw", sample=SAMPLES)
    output:
        peptide_model = f"{config['base_file_paths']['results']}/models/peptide.model"
    shell:
        (
            "gscore build "
            "--level peptide "
            "-i {input.osws} "
            "--output {output.peptide_model}"
        )

rule gscore_build_protein_model:
    input:
        score_placeholder = expand(f"{config['base_file_paths']['results']}/scored/{{sample}}", sample=SAMPLES),
        osws = expand(f"{config['base_file_paths']['osw']}/{{sample}}.osw", sample=SAMPLES)
    output:
        protein_model = f"{config['base_file_paths']['results']}/models/protein.model"
    shell:
        (
            "gscore build "
            "--level protein "
            "-i {input.osws} "
            "--output {output.protein_model}"
        )

rule gscore_combine:
    input:
        osws = expand(f"{config['base_file_paths']['osw']}/{{sample}}.osw",sample=SAMPLES),
        peptide_model= f"{config['base_file_paths']['results']}/models/peptide.model",
        protein_model= f"{config['base_file_paths']['results']}/models/protein.model"
    output:
        results = f"{config['base_file_paths']['results']}/quantification/{config['gscore']['combine']['output']}"
    params:
        max_peakgroup_q_value = config['gscore']['combine']['max_peakgroup_q_value']
    shell:
        (
            "gscore combine "
            "--input {input.osws} "
            "--peptide-model {input.peptide_model} "
            "--protein-model {input.protein_model} "
            "--output {output.results} "
            "--max-peakgroup-q-value {params.max_peakgroup_q_value}"
        )
