import argparse

from gscore.fdr import GlobalDistribution
from gscore.parsers import queries
from gscore.parsers.osw import OSWFile


class Build:

    name: str
    parser: argparse.ArgumentParser

    def __init__(self):

        self.name = "build"

    def __call__(self, args: argparse.Namespace):

        global_distribution = GlobalDistribution()

        for input_file in args.input_files:

            print(f"Processing {input_file}...")

            with OSWFile(input_file) as osw_conn:

                if args.level == "protein":

                    groups = osw_conn.parse_to_proteins(
                        query=queries.SelectPeakGroups.FETCH_ALL_SCORED_DATA
                    )

                elif args.level == "peptide":

                    groups = osw_conn.parse_to_peptides(
                        query=queries.SelectPeakGroups.FETCH_ALL_SCORED_DATA
                    )

                print(f"Comparing {args.level} level scores...")

                for group in groups:

                    if group.identifier not in global_distribution:

                        global_distribution[group.identifier] = group

                    else:

                        global_distribution.compare_score(
                            group.identifier,
                            group
                        )

        print("Fitting distribution...")
        global_distribution.fit()

        print(f"Saving model: {args.output}")

        global_distribution.save(args.output)


    def build_subparser(self,
                        subparser
    ):

        self.parser = subparser.add_parser(
            self.name,
            help="Build q-value scoring models for different contexts."
        )

        self.parser.add_argument(
            '-i',
            '--input',
            dest='input_files',
            help='NPZ files for training scorer',
            nargs='+'
        )

        self.parser.add_argument(
            "--level",
            dest="level",
            choices=[
                "peptide", "protein"
            ],
            help="Level to build the scoring distribution."
        )

        self.parser.add_argument(
            "--output",
            dest="output",
            help="Output path for scoring model."
        )

        self.parser.set_defaults(run=self)

    def __repr__(self):

        return f"<Train> {self.name}"