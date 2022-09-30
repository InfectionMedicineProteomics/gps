import argparse
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import Any, Dict, Union
from csv import DictReader, DictWriter

from gps.parsers.osw import OSWFile
from gps.parsers.queries import SelectPeakGroups
from gps.precursors import Precursors

MatchScoreType = Dict[str, Dict[str, Dict[str, Any]]]


def score_with_percolator(args: argparse.Namespace) -> MatchScoreType:

    pin_path = Path(args.output)

    target_results = f"{pin_path.parent}/{pin_path.name}_results_psms.tsv"
    decoy_results = f"{pin_path.parent}/{pin_path.name}_decoy_results_psms.tsv"

    with Popen(
        [
            args.percolator_exe,
            args.output,
            "--results-psms",
            target_results,
            "--decoy-results-psms",
            decoy_results,
            "--protein-decoy-pattern",
            "DECOY_",
            "--num-threads",
            str(args.threads),
            "--only-psms",
        ],
        stdout=PIPE,
        stderr=STDOUT,
    ) as process:

        for line in iter(process.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))

    target_records = []

    with open(target_results) as target_results_h:

        reader = DictReader(target_results_h, delimiter="\t")

        for row in reader:
            row["Decoy"] = 0

            target_records.append(row)

    decoy_records = []

    with open(decoy_results) as decoy_results:

        reader = DictReader(decoy_results, delimiter="\t")

        for row in reader:
            row["Decoy"] = 1

            decoy_records.append(row)

    records = target_records + decoy_records

    grouped_records = dict()

    for record in records:

        sequence, charge, peakgroup_id = record["PSMId"].split("_")

        precursor_id = f"{sequence}_{charge}"

        if precursor_id not in grouped_records:

            grouped_records[precursor_id] = dict()

        grouped_records[precursor_id][peakgroup_id] = record

    return grouped_records


def apply_percolator_weights(args: argparse.Namespace) -> MatchScoreType:

    pin_path = Path(args.pin)

    target_results = f"{pin_path.parent}/{pin_path.name}_results_psms.tsv"
    decoy_results = f"{pin_path.parent}/{pin_path.name}_decoy_results_psms.tsv"

    with Popen(
        [
            args.percolator_exe,
            args.pin,
            "--results-psms",
            target_results,
            "--decoy-results-psms",
            decoy_results,
            "--protein-decoy-pattern",
            "DECOY_",
            "--num-threads",
            str(args.threads),
            "--only-psms",
            "--init-weights",
            args.percolator_weights,
            "--static",
        ],
        stdout=PIPE,
        stderr=STDOUT,
    ) as process:

        for line in iter(process.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))

    target_records = []

    with open(target_results) as target_results_h:

        reader = DictReader(target_results_h, delimiter="\t")

        for row in reader:
            row["Decoy"] = 0

            target_records.append(row)

    decoy_records = []

    with open(decoy_results) as decoy_results:

        reader = DictReader(decoy_results, delimiter="\t")

        for row in reader:
            row["Decoy"] = 1

            decoy_records.append(row)

    records = target_records + decoy_records

    grouped_records = dict()

    for record in records:

        sequence, charge, peakgroup_id = record["PSMId"].split("_")

        precursor_id = f"{sequence}_{charge}"

        if precursor_id not in grouped_records:
            grouped_records[precursor_id] = dict()

        grouped_records[precursor_id][peakgroup_id] = record

    return grouped_records


def update_precusors(precursors: Precursors, grouped_records: MatchScoreType) -> None:
    for precursor in precursors.precursors.values():

        precursor_id = f"{precursor.modified_sequence}_{precursor.charge}"

        scored_peakgroups = grouped_records[precursor_id]

        for peakgroup in precursor.peakgroups:

            peakgroup_id = str(peakgroup.idx)

            if peakgroup_id in scored_peakgroups:

                scored_peakgroup = scored_peakgroups[peakgroup_id]

                peakgroup.d_score = float(scored_peakgroup["score"])
                peakgroup.q_value = float(scored_peakgroup["q-value"])
                peakgroup.probability = float(scored_peakgroup["posterior_error_prob"])
                peakgroup.top_scoring = True


def export_initial_pin(
    precursors: Precursors,
    pin_output_file: str,
):
    flagged_score_columns = precursors.flag_score_columns()

    print(flagged_score_columns)

    peakgroup_records = list()
    peptide_protein_ids = list()

    for precursor in precursors:

        precursor.peakgroups.sort(key=lambda x: x.d_score, reverse=True)

        peakgroups = precursor.peakgroups

        for peakgroup in peakgroups:
            peakgroup_record: Dict[str, Union[str, int, float]] = {
                "id": f"{precursor.modified_sequence}_{precursor.charge}_{peakgroup.idx}",
                "label": 1 if peakgroup.target else -1,
                "scannr": peakgroup.idx,
            }

            peakgroup_record.update(peakgroup.get_score_columns(flagged_score_columns))

            peptide_protein_ids.append(
                {
                    "peptide": precursor.modified_sequence,
                    "proteinId1": precursor.protein_accession,
                }
            )

            peakgroup_records.append(peakgroup_record)

    for peakgroup_idx, peakgroup_record in enumerate(peakgroup_records):
        peakgroup_record.update(peptide_protein_ids[peakgroup_idx])

    field_names = list(peakgroup_records[0].keys())

    with open(pin_output_file, "w") as out_file:

        csv_writer = DictWriter(out_file, delimiter="\t", fieldnames=field_names)

        csv_writer.writeheader()

        for peakgroup_record in peakgroup_records:
            csv_writer.writerow(peakgroup_record)


class Score:
    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "score"

    def __call__(self, args: argparse.Namespace) -> None:

        print(f"Processing file {args.input}")

        osw_file = OSWFile(args.input)

        precursors = osw_file.parse_to_precursors(
            query=SelectPeakGroups.FETCH_FEATURES_REDUCED
        )

        if args.percolator_weights:

            print("Exporting PIN.")

            precursors.export_pin(args.pin, export_initial_pin=False)

            scored_peakgroups = apply_percolator_weights(args)

            update_precusors(precursors, scored_peakgroups)

            precursors.write_tsv(file_path=args.output, ranked=1, write_percolator=True)

        else:

            if args.decoy_free:
                print("Denoising.")

                precursors.denoise(
                    num_folds=args.num_folds,
                    num_classifiers=args.num_classifiers,
                    num_threads=args.threads,
                    vote_percentage=args.vote_percentage,
                )

            if args.weight_scores:
                print("Denoising.")

                precursors.denoise(
                    num_folds=args.num_folds,
                    num_classifiers=args.num_classifiers,
                    num_threads=args.threads,
                    vote_percentage=args.vote_percentage,
                )

            if args.estimate_pit:

                print("Estimating PIT.")

                precursors.denoise(
                    num_folds=args.num_folds,
                    num_classifiers=args.num_classifiers,
                    num_threads=args.threads,
                    vote_percentage=args.vote_percentage,
                )

                pit = precursors.estimate_pit()

                if pit > 1.0:

                    print(f"PIT estimate to be: {pit}, but set to 1.0")

                    pit = 1.0

                else:

                    print(f"PIT estimate to be: {pit}")

            else:

                pit = 1.0

            print("Scoring. . .")

            if args.percolator_output:

                print("Writing percolator output...")

                precursors.export_pin(args.output, export_initial_pin=True)

                print("Rescoring with percolator to find best candidates...")

                scored_peakgroups = score_with_percolator(args)

                update_precusors(precursors, scored_peakgroups)

                precursors.export_pin(args.output)

            else:

                precursors.score_run(
                    model_path=args.scoring_model,
                    scaler_path=args.scaler,
                    threads=args.threads,
                    weight_scores=args.weight_scores,
                )

                print("Calculating Q Values")

                if args.count_decoys:

                    print("Counting decoys.")

                else:

                    print("Estimating score distributions.")

                q_values = precursors.calculate_q_values(
                    sort_key="d_score",
                    decoy_free=args.decoy_free,
                    count_decoys=args.count_decoys,
                    num_threads=args.threads,
                    pit=pit,
                    debug=args.debug,
                )

                num_below_threshold = q_values[q_values <= 0.01].shape[0]

                print(num_below_threshold)

                if args.output:

                    if args.decoy_free:

                        precursors.write_tsv(file_path=args.output, ranked=2)

                    else:

                        precursors.write_tsv(file_path=args.output, ranked=1)

                else:

                    print("Updating Q Values in file")

                    osw_file.add_score_and_q_value_records(precursors)

        print("Done!")

    def build_subparser(self, subparser: Any) -> None:

        self.parser = subparser.add_parser(
            self.name, help="Commands to score and denoise OSW files"
        )

        self.parser.add_argument("-i", "--input", help="OSW file to process", type=str)

        self.parser.add_argument("-o", "--output", help="Output TSV file.", type=str)

        self.parser.add_argument(
            "--percolator-output",
            dest="percolator_output",
            help="Export data in PIN format.",
            action="store_true",
        )

        self.parser.add_argument(
            "--percolator-weights",
            dest="percolator_weights",
            help="Use percolator weights to score files with path to weights.",
            default="",
        )

        self.parser.add_argument(
            "--pin", dest="pin", help="PIN input file.", default=""
        )

        self.parser.add_argument(
            "--percolator-exe", dest="percolator_exe", help="Percolator exe file path"
        )

        self.parser.add_argument(
            "--scoring-model",
            dest="scoring_model",
            help="Path to scoring model to apply to data.",
            type=str,
            default=""
        )

        self.parser.add_argument(
            "--scaler",
            dest="scaler",
            help="Path to scaler to transform data.",
            type=str,
            default="",
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            help="The number of threads to use",
            default=1,
            type=int,
        )

        self.parser.add_argument(
            "--gpus",
            dest="gpus",
            type=int,
            help="Number of GPUs to use to train model.",
            default=1,
        )

        self.parser.add_argument(
            "--count-decoys",
            dest="count_decoys",
            help="Count decoys to calculate q-values for each peakgroup",
            action="store_true",
            default=False,
        )

        self.parser.add_argument(
            "--estimate-pit",
            dest="estimate_pit",
            help="Use an ensemble denoising process to estimate the percentage of incorrect targets",
            action="store_true",
            default=False,
        )

        self.parser.add_argument(
            "--decoy-free",
            dest="decoy_free",
            help="Use the second ranked target peakgroups as decoys for modelling the scores and calculating q-values",
            action="store_true",
        )

        self.parser.add_argument(
            "--num-classifiers",
            dest="num_classifiers",
            help="The number of ensemble learners used to denoise each fold",
            default=10,
            type=int,
        )

        self.parser.add_argument(
            "--num-folds",
            dest="num_folds",
            help="The number of folds used to denoise the target labels",
            default=10,
            type=int,
        )

        self.parser.add_argument(
            "--vote-percentage",
            dest="vote_percentage",
            help="The minimum probability needed to be counted as a positive vote",
            default=0.5,
            type=float,
        )

        self.parser.add_argument(
            "--weight-scores",
            dest="weight_scores",
            help="Use sample-specific data to weight the static score.",
            action="store_true",
        )

        self.parser.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:

        return f"<Score> {self.name}"
