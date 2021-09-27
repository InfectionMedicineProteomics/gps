import numpy as np
import tensorflow as tf


class ChromatogramDataGenerator:

    def __init__(self, peak_group_graph, precursor_list, chromatogram_data, use_decoy=False, num_transitions=7):
        self.peak_group_graph = peak_group_graph
        self.precursor_list = precursor_list
        self.chromatogram_data = chromatogram_data
        self.use_decoy = use_decoy
        self.min_chrom_length = chromatogram_data.min_chromatogram_length()
        self.num_transitions = num_transitions

    def generator(self):

        for precursor_id in self.precursor_list:

            precursor_node = self.peak_group_graph[precursor_id]

            peptide_id = f"{precursor_node.data.modified_sequence}_{precursor_node.data.charge}"

            if peptide_id in self.chromatogram_data:

                top_ranked_peakgroup_key = precursor_node.get_edge_by_ranked_weight(
                    rank=1
                )

                peakgroup_data = self.peak_group_graph[top_ranked_peakgroup_key].data

                peakgroup_boundaries = np.asarray(
                    [
                        peakgroup_data.start_rt,
                        peakgroup_data.rt,
                        peakgroup_data.end_rt
                    ],
                    dtype='float64'
                ).reshape((3, 1))

                transition_chromatograms = list()

                rt_steps = list()

                for chromatogram_id, chromatogram_record in self.chromatogram_data[peptide_id].items():

                    if chromatogram_record.type != "precursor":

                        if not rt_steps:

                            for rt_chromatogram_idx in range(self.min_chrom_length):

                                rt_steps.append(
                                    chromatogram_record.rts[rt_chromatogram_idx]
                                )

                            transition_chromatograms.append(rt_steps)

                        transition_chromatogram = list()

                        for chrom_idx in range(self.min_chrom_length):

                            transition_chromatogram.append(
                                chromatogram_record.intensities[chrom_idx]
                            )

                        transition_chromatograms.append(transition_chromatogram)

                transformed_chromatograms = list()

                for row_transform in zip(transformed_chromatograms):

                    chromatogram_row = np.asarray(row_transform, dtype='float64')

                    transformed_chromatograms.append(chromatogram_row)

                yield (
                    tf.constant(peakgroup_boundaries),
                    tf.constant(np.asarray(transformed_chromatograms))
                ), tf.constant(precursor_node.data.target, dtype=tf.float64)


    def create_dataset(self):

        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                (
                    tf.TensorSpec(
                        shape=(3, 1),
                        dtype=tf.float64,
                        name="query_features"
                    ),
                    tf.TensorSpec(
                        shape=(self.min_chrom_length, self.num_transitions + 1),
                        dtype=tf.float64,
                        name="chromatogram_features"
                    )
                ),
                tf.TensorSpec(
                    shape=(),
                    dtype=tf.float64,
                    name="target"
                )
            )
        )

        return dataset.shuffle(
            10,
            reshuffle_each_iteration=True
        ).batch(32).prefetch(1)
