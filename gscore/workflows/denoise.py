import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score


from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)

from gscore.models.denoiser import (
    BaggedDenoiser
)

SCORE_COLUMNS = [
    'var_massdev_score_ms1',
    'var_isotope_correlation_score_ms1',
    'var_isotope_overlap_score_ms1',
    'var_xcorr_coelution_contrast_ms1',
    'var_xcorr_coelution_combined_ms1',
    'var_xcorr_shape_contrast_ms1',
    'var_xcorr_shape_combined_ms1',
    'var_bseries_score',
    'var_dotprod_score',
    'var_intensity_score',
    'var_isotope_correlation_score',
    'var_isotope_overlap_score',
    'var_library_corr',
    'var_library_dotprod',
    'var_library_manhattan',
    'var_library_rmsd',
    'var_library_rootmeansquare',
    'var_library_sangle',
    'var_log_sn_score',
    'var_manhattan_score',
    'var_massdev_score',
    'var_massdev_score_weighted',
    'var_norm_rt_score',
    'var_xcorr_coelution',
    'var_xcorr_coelution_weighted',
    'var_xcorr_shape',
    'var_xcorr_shape_weighted',
    'var_yseries_score'
]

def split_peak_groups_into_folds(peakgroup_data, num_folds):

    shuffled_peak_groups = peakgroup_data.sample(frac=1)

    split_data = np.array(
        shuffled_peak_groups,
        num_folds
    )

    split_data = split_data[::-1]

    return split_data


def denoise(all_peak_groups, noisey_target_labels, num_folds, num_classifiers, threads):

    split_data = split_peak_groups_into_folds(
        peakgroup_data=noisey_target_labels,
        num_folds=num_folds
    )

    print("Denoising target labels")

    scored_data = list()

    for idx, fold_data in enumerate(split_data):

        training_data = pd.concat(
            [df for i, df in enumerate(split_data) if i != idx]
        )

        full_pipeline = Pipeline([
            ('standard_scaler', RobustScaler()),
            ('min_max_scaler', MinMaxScaler())
        ])

        swath_training_prepared = training_data.copy()

        swath_training_prepared[SCORE_COLUMNS] = full_pipeline.fit_transform(
            swath_training_prepared[SCORE_COLUMNS]
        )

        print(
            f"Number of labels: \n",
            swath_training_prepared.target.value_counts()
        )

        n_samples = int(len(swath_training_prepared) * 1.0)

        denoizer = BaggedDenoiser(
            max_samples=n_samples,
            n_estimators=num_classifiers,
            threads=threads,
            random_state=idx
        )

        denoizer.fit(
            swath_training_prepared[SCORE_COLUMNS],
            swath_training_prepared['target']
        )

        group_ids = list(fold_data['transition_group_id'])

        left_out_peak_groups = all_peak_groups.loc[
            all_peak_groups['transition_group_id'].isin(group_ids)
        ].copy()

        left_out_peak_groups_transformed = left_out_peak_groups.copy()

        left_out_peak_groups_transformed[SCORE_COLUMNS] = full_pipeline.transform(
            left_out_peak_groups[SCORE_COLUMNS]
        )

        left_out_peak_groups['vote_percentage'] = denoizer.vote(
            left_out_peak_groups_transformed[SCORE_COLUMNS]
        )

        class_index = np.where(
            denoizer.classes_ == 1.0
        )[0][0]

        left_out_peak_groups['probability'] = denoizer.predict_proba(
            left_out_peak_groups_transformed[SCORE_COLUMNS]
        )[:, class_index]

        left_out_peak_groups['logit_probability'] = np.log(
            (
                    left_out_peak_groups['probability'] / (1 - left_out_peak_groups['probability'])
            )
        )

        fold_data[SCORE_COLUMNS] = full_pipeline.transform(
            fold_data[SCORE_COLUMNS]
        )

        fold_precision = precision_score(
            denoizer.predict(
                fold_data[SCORE_COLUMNS]
            ),
            fold_data['target']
        )

        fold_recall = recall_score(
            denoizer.predict(
                fold_data[SCORE_COLUMNS]
            ),
            fold_data['target']
        )

        print(
            f"Fold {idx + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
        )

        untransformed_peak_groups = all_peak_groups.loc[
            all_peak_groups['transition_group_id'].isin(group_ids)
        ].copy()

        untransformed_peak_groups['vote_percentage'] = left_out_peak_groups['vote_percentage']
        untransformed_peak_groups['probability'] = left_out_peak_groups['probability']
        untransformed_peak_groups['logit_probability'] = left_out_peak_groups['logit_probability']

        scored_data.append(untransformed_peak_groups)

    scored_data = pd.concat(
        scored_data,
        ignore_index=True
    )

    return scored_data