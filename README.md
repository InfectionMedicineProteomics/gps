# GPS: Machine learning based generalized peakgroup scoring
This is a python package for scoring SWATH-MS data using static generalizable machine learning models trained on large curated datasets. Current support is for OpenSwath, but could be expanded to other tools quite easily.

## Installation
The recommended way to install GPS is into a virtual environment to make sure that all dependencies work correctly.

This can be done using your method of choice. The following demonstration will be using miniconda
```commandline
conda create -n gpsenv -c conda-forge python=3.10 pip

conda activate gpsenv
```

With your environment activated, you can then install via pip

```commandline
pip install gps
```

GPS is now installed and ready to use!

## Usage

### Scoring individual files

GPS is very easy to use. To get started scoring a processed file, you simple run the score command:

```commandline
gps score --input extracted_peakgroups.osw --output extracted_peakgroups.scored.tsv
```

This command will take in output from OpenSwath, score the extracted peakgroups, and write a tsv file with the q-values, scores, probabilities, etc.

To increase the number of identifications at a particular q-value cutoff, you can enable PIT estimation and correction. This will use a novel denoising algorithm to estimate the false target probability distribution of the target labels, and weight the decoys during q-value calculation.

```commandline
gps score --input extracted_peakgroups.osw --output extracted_peakgroups.scored.tsv --estimate-pit
```

You can also make use of multiple cores using the ```--threads``` option

```commandline
gps score --input extracted_peakgroups.osw --output extracted_peakgroups.scored.tsv --estimate-pit --threads 10
```

### Controlling global peptide and protein FDR

Once all individual files are scored using GPS, it is very straightforward to build models to control the global levels of peptides and proteins in the analysis.

You can specify the level of the model to build using the ```--level``` cli option

```commandline
gps build --level peptide --input *.scored.tsv --output peptide.model --estimate-pit
```
The above command will take all scored files in at once using wildcard command line options, build a peptide level model, and estimate the global PIT for q-value correction

To build a protein level model, you only need to change the ```--level``` option.

```commandline
gps build --level protein --input *.scored.tsv --output protein.model --estimate-pit
```

### Combining scored files into a quantitative matrix for downstream analysis

Once all files have been scored and the global models have been built, GPS can combined all files into a quantiative matrix for convinient downstream analysis.

```commandline
gps combine --input-files *.scored.tsv --peptide-model peptide.model --protein-model protein.model --output quantitative_matrix.tsv --max-peakgroup-q-value 0.01
```

The maximum q-value for the included precursors can be indicated if you would like to be more, or less, lenient on the identifications that you include in your final quantitative_matrix
