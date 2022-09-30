import argparse
from typing import Union, Any

from gps.fdr import GlobalDistribution
from gps.parsers import queries
from gps.parsers.osw import OSWFile
from gps.parsers.score_file import ScoreFile
from gps.peptides import Peptides
from gps.proteins import Proteins


class Build:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self) -> None:

        self.name = "build"

    def __call__(self, args: argparse.Namespace) -> None:

        global_distribution = GlobalDistribution(
            count_decoys=args.count_decoys, num_threads=args.threads
        )

        for input_file in args.input_files:

            print(f"Processing {input_file}...")

            groups: Union[Peptides, Proteins]

            if input_file.lower().endswith(".tsv"):

                gps_file = ScoreFile(input_file)

                if args.level == "protein":

                    groups = gps_file.parse_to_proteins()

                elif args.level == "peptide":

                    groups = gps_file.parse_to_peptides()

            elif input_file.lower().endswith("osw"):

                osw_file = OSWFile(input_file)

                if args.level == "protein":

                    groups = osw_file.parse_to_proteins(
                        query=queries.SelectPeakGroups.BUILD_GLOBAL_MODEL_QUERY
                    )

                elif args.level == "peptide":

                    groups = osw_file.parse_to_peptides(
                        query=queries.SelectPeakGroups.BUILD_GLOBAL_MODEL_QUERY
                    )

            print(f"Comparing {args.level} level scores...")

            for group in groups:

                if group.identifier not in global_distribution:

                    global_distribution[group.identifier] = group

                else:

                    global_distribution.compare_score(group.identifier, group)

        if args.estimate_pit:

            print("Estimating PIT...")

            pit = global_distribution.estimate_pit()

            print(f"PIT estimated to be: {pit}")

        print("Fitting distribution...")
        global_distribution.fit()

        num_below_001 = global_distribution.q_values[
            global_distribution.q_values <= 0.01
        ].shape[0]

        print(f"Features below 0.01: {num_below_001}")

        print(f"Saving model: {args.output}")

        global_distribution.save(args.output)

    def build_subparser(self, subparser: Any) -> None:

        self.parser = subparser.add_parser(
            self.name, help="Build q-value scoring models for different contexts."
        )

        self.parser.add_argument(
            "-i",
            "--input",
            dest="input_files",
            help="NPZ files for training scorer",
            nargs="+",
        )

        self.parser.add_argument(
            "--level",
            dest="level",
            choices=["peptide", "protein"],
            help="Level to build the scoring distribution.",
        )

        self.parser.add_argument(
            "--output", dest="output", help="Output path for scoring model."
        )

        self.parser.add_argument(
            "--estimate-pit",
            dest="estimate_pit",
            help="Use an ensemble denoising process to estimate the percentage of incorrect targets",
            action="store_true",
            default=False,
        )

        self.parser.add_argument(
            "--count-decoys",
            dest="count_decoys",
            help="Count decoys to calculate q-values for each peakgroup",
            action="store_true",
            default=False,
        )

        self.parser.add_argument(
            "--threads",
            dest="threads",
            help="The number of threads to use",
            default=1,
            type=int,
        )

        self.parser.set_defaults(run=self)

    def __repr__(self) -> str:

        return f"<Build> {self.name}"
