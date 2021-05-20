import pickle
import os

import numpy as np
import tensorflow as tf

from sklearn.utils import (
    shuffle,
    class_weight
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline

from gscore.parsers.osw import (
    osw,
    queries
)
from gscore import (
    peakgroups,
    scorer
)


def main(args, logger):

    input_files = args.input_files

    all_sample_data = list()
    all_sample_labels = list()

    for input_path in input_files:

        logger.info(f"Processing {input_path}")

        base_filename = os.path.basename(input_path)

        print(f"parsing {base_filename}")

        peakgroup_graph = osw.fetch_peakgroup_graph(
            osw_path=input_path,
            osw_query=queries.SelectPeakGroups.FETCH_VOTED_DATA_DECOY_FREE,
            peakgroup_weight_column='probability'
        )

        true_targets = peakgroup_graph.query_nodes(
            color='peptide',
            rank=1,
            query="probability > 0.8"
        )

        for node in peakgroup_graph.iter(keys=true_targets):

            probability = node.data.scores['probability']

            node.data.add_sub_score_column(
                key='probability',
                value=probability
            )

        false_targets = peakgroup_graph.query_nodes(
            color='peptide',
            rank=1,
            query="vote_percentage < 0.5"
        )

        for node in peakgroup_graph.iter(keys=false_targets):

            probability = node.data.scores['probability']

            node.data.add_sub_score_column(
                key='probability',
                value=probability
            )

        target_scores, target_labels, target_indices = peakgroups.parse_scores_labels_index(
            graph=peakgroup_graph,
            node_keys=true_targets,
            is_decoy=False,
            include_score_columns=False
        )

        false_target_scores, false_target_labels, false_target_indices = peakgroups.parse_scores_labels_index(
            graph=peakgroup_graph,
            node_keys=false_targets,
            is_decoy=True,
            include_score_columns=False
        )

        scores = np.concatenate(
            [
                target_scores,
                false_target_scores
            ]
        )

        score_labels = np.concatenate(
            [
                target_labels,
                false_target_labels
            ]
        )

        score_indices = np.concatenate(
            [
                target_indices,
                false_target_indices
            ]
        )

        sample_data, sample_labels, _ = shuffle(
            scores, score_labels, score_indices,
            random_state=42
        )

        all_sample_data.append(sample_data)
        all_sample_labels.append(sample_labels)

    all_data = np.concatenate(all_sample_data)
    all_labels = np.concatenate(all_sample_labels)

    training_data, testing_data, training_labels, testing_labels = train_test_split(
        all_data, all_labels,
        test_size=0.1,
        shuffle=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    pipeline = Pipeline([
        ('standard_scaler', RobustScaler()),
        ('min_max_scaler', MinMaxScaler())
    ])

    training_data = pipeline.fit_transform(training_data)

    dense_model = scorer.TargetScoringModel(
        input_dim=training_data.shape[1:]
    )

    dense_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            'accuracy'
        ],

    )

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(training_labels),
        y=training_labels.ravel()
    )

    dense_history = dense_model.fit(
        training_data,
        training_labels,
        epochs=10,
        validation_split=0.10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True
            )
        ],
        batch_size=32,
        shuffle=True,
        class_weight=dict(
            enumerate(class_weights)
        )
    )

    testing_data = pipeline.transform(testing_data)

    dense_model.evaluate(
        testing_data,
        testing_labels
    )

    print('saving trained model and scaler')

    dense_model.save(
        f'{args.output_directory}/{args.model_name}',
        save_format='tf'
    )

    scaler_file_path = f'{args.output_directory}/{args.model_name}.scaler.pkl'

    with open(scaler_file_path, 'wb') as pkl:
        pickle.dump(pipeline, pkl)
