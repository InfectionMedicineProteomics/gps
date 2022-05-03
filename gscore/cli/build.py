import argparse

from gscore.fdr import GlobalDistribution
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile
from gscore.parsers.score_file import ScoreFile


class Build:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "build"

    def __call__(self, args: argparse.Namespace):

        global_distribution = GlobalDistribution(
            count_decoys=args.count_decoys, num_threads=args.threads
        )

        for input_file in args.input_files:

            print(f"Processing {input_file}...")

            if input_file.lower().endswith(".tsv"):

                gscore_file = ScoreFile(input_file)

                if args.level == "protein":

                    groups = gscore_file.parse_to_proteins()

                elif args.level == "peptide":

                    groups = gscore_file.parse_to_peptides()

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

    def build_subparser(self, subparser):

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

    def __repr__(self):

        return f"<Build> {self.name}"
