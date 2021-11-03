import pickle
import os

import numpy as np

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

from gscore.parsers import osw, queries
from gscore import (
    peakgroups,
    scorer
)

from gscore.utils import ml


def main(args, logger):

    input_files = args.input_files

    all_sample_data = list()
    all_sample_labels = list()

    for i, input_path in enumerate(input_files):

        logger.info(f"Processing {input_path}")

        base_filename = os.path.basename(input_path)

        print(f"parsing {base_filename}")

        peakgroup_graph, _ = osw.fetch_peakgroup_graph(
            osw_path=input_path,
            query=queries.SelectPeakGroups.FETCH_TRAIN_CHROMATOGRAM_SCORING_DATA
        )

        positive_labels = peakgroup_graph.filter_ranked_peakgroups(
            rank=1,
            score_column='probability',
            value=0.85,
            user_operator='>',
            target=1
        )

        negative_labels = peakgroup_graph.get_ranked_peakgroups(
            rank=1,
            target=0
        )

        combined_data = positive_labels + negative_labels

        all_sample_data.append(combined_data)

    all_data = np.concatenate(all_sample_data)

    training_data, testing_data = train_test_split(
        all_data,
        test_size=0.2,
        shuffle=True
    )

    training_data_scores, training_data_labels, training_data_indices = ml.reformat_data(
        training_data,
        include_score_columns=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    pipeline = Pipeline([
        ('standard_scaler', RobustScaler()),
        ('min_max_scaler', MinMaxScaler())
    ])

    training_data_scores = pipeline.fit_transform(training_data_scores)

    dense_model = scorer.TargetScoringModel(
        input_dim=training_data_scores.shape[1:]
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
        classes=np.unique(training_data_labels),
        y=training_data_labels.ravel()
    )

    print(class_weights)

    dense_history = dense_model.fit(
        training_data_scores,
        training_data_labels,
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

    testing_data_scores, testing_data_labels, testing_data_indices = ml.reformat_data(
        testing_data,
        include_score_columns=True
    )

    testing_data_scores = pipeline.transform(testing_data_scores)

    dense_model.evaluate(
        testing_data_scores,
        testing_data_labels
    )

    print('saving trained model and scaler')

    dense_model.save(
        f'{args.output_directory}/{args.model_name}',
        save_format='tf'
    )

    scaler_file_path = f'{args.output_directory}/{args.model_name}.scaler.pkl'

    with open(scaler_file_path, 'wb') as pkl:
        pickle.dump(pipeline, pkl)
