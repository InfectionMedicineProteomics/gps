class CreateTable:

    CREATE_GHOSTSCORE_TABLE = (
        """
        CREATE TABLE IF NOT EXISTS ghost_score_table (
            ghost_score_id INTEGER PRIMARY KEY,
            feature_id INTEGER not null,
            vote_percentage REAL,
            probability REAL,
            d_score REAL,
            q_value REAL,
            peptide_q_value REAL,
            protein_q_value REAL,
            weighted_d_score REAL,
            FOREIGN KEY (feature_id)
                REFERENCES FEATURE (id)
        )
        """
    )

class CreateIndex:

    CREATE_PROTEIN_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_protein
        ON PROTEIN(id);
        """
    )

    CREATE_PRECURSOR_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_precursor_id
        ON precursor(id);
        """
    )

    CREATE_PREC_PEP_MAPPING_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_prec_pep_mapping
        ON PRECURSOR_PEPTIDE_MAPPING(precursor_id);
        """
    )

    CREATE_PROT_PEP_MAPPING_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_prot_pep_mapping
        ON PEPTIDE_PROTEIN_MAPPING(peptide_id);
        """
    )

    CREATE_PEPTIDE_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_peptide
        ON PEPTIDE(id);
        """
    )

    CREATE_FEATURE_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_feature_precursor
        ON FEATURE(precursor_id);
        """
    )

    CREATE_FEATURE_MS2_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_feature_feature_ms2
        ON FEATURE_MS2(feature_id)
        """
    )

    CREATE_FEATURE_MS1_IDX = (
        """
        CREATE INDEX IF NOT EXISTS idx_feature_feature_ms1
        ON FEATURE_MS1(feature_id)
        """
    )

    ALL_INDICES = [
        CREATE_PRECURSOR_IDX,
        CREATE_FEATURE_IDX,
        CREATE_PROTEIN_IDX,
        CREATE_PEPTIDE_IDX,
        CREATE_FEATURE_MS1_IDX,
        CREATE_FEATURE_MS2_IDX,
        CREATE_PREC_PEP_MAPPING_IDX,
        CREATE_PROT_PEP_MAPPING_IDX
    ]



class ChromatogramQueries:

    FETCH_PEPTIDE_CHROMATOGRAM = (
        """
        select
            PRECURSOR.PEPTIDE_SEQUENCE,
            PRECURSOR.CHARGE,
            PRECURSOR.ISOLATION_TARGET PRECURSOR_ISOLATION_TARGET,
            CHROMATOGRAM.NATIVE_ID,
            DATA.COMPRESSION,
            DATA.DATA_TYPE,
            DATA.DATA,
            PRODUCT.ISOLATION_TARGET PRODUCT_ISOLATION_TARGET
        from precursor
        join CHROMATOGRAM on PRECURSOR.CHROMATOGRAM_ID = CHROMATOGRAM.ID
        join DATA on CHROMATOGRAM.ID = DATA.CHROMATOGRAM_ID
        join PRODUCT on CHROMATOGRAM.ID = PRODUCT.CHROMATOGRAM_ID;
        """
    )

class SelectPeakGroups:

    BASE_COLUMNS = (
        """
        feature.run_id run_id,
        precursor.id transition_group_id,
        peptide.MODIFIED_SEQUENCE || '_' || precursor.CHARGE as peptide_charge,
        feature.id feature_id,
        feature.exp_rt rt,
        precursor.PRECURSOR_MZ mz,
        precursor.CHARGE charge,
        precursor.DECOY decoy,
        peptide.UNMODIFIED_SEQUENCE peptide_sequence,
        peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
        protein.PROTEIN_ACCESSION protein_accession,
        protein.decoy protein_decoy,
        ms1.AREA_INTENSITY ms1_integrated_intensity,
        ms1.APEX_INTENSITY ms1_apex_intensity,
        ms2.AREA_INTENSITY ms2_integrated_intensity,
        feature.LEFT_WIDTH left_width,
        feature.RIGHT_WIDTH right_width
        """
    )

    SCORE_COLUMNS = (
        """    
        ms1.var_massdev_score var_massdev_score_ms1,
        ms1.var_isotope_correlation_score var_isotope_correlation_score_ms1,
        ms1.var_isotope_overlap_score var_isotope_overlap_score_ms1,
        ms1.var_xcorr_coelution_contrast var_xcorr_coelution_contrast_ms1,
        ms1.var_xcorr_coelution_combined var_xcorr_coelution_combined_ms1,
        ms1.var_xcorr_shape_contrast var_xcorr_shape_contrast_ms1,
        ms1.var_xcorr_shape_combined var_xcorr_shape_combined_ms1,
        ms2.VAR_BSERIES_SCORE,
        ms2.VAR_DOTPROD_SCORE,
        ms2.VAR_INTENSITY_SCORE,
        ms2.VAR_ISOTOPE_CORRELATION_SCORE,
        ms2.VAR_ISOTOPE_OVERLAP_SCORE,
        ms2.VAR_LIBRARY_CORR,
        ms2.VAR_LIBRARY_DOTPROD,
        ms2.VAR_LIBRARY_MANHATTAN,
        ms2.VAR_LIBRARY_RMSD,
        ms2.VAR_LIBRARY_ROOTMEANSQUARE,
        ms2.VAR_LIBRARY_SANGLE,
        ms2.VAR_LOG_SN_SCORE,
        ms2.VAR_MANHATTAN_SCORE,
        ms2.VAR_MASSDEV_SCORE,
        ms2.VAR_MASSDEV_SCORE_WEIGHTED,
        ms2.VAR_NORM_RT_SCORE,
        ms2.VAR_XCORR_COELUTION,
        ms2.VAR_XCORR_COELUTION_WEIGHTED,
        ms2.VAR_XCORR_SHAPE,
        ms2.VAR_XCORR_SHAPE_WEIGHTED,
        ms2.VAR_YSERIES_SCORE
        """
    )

    FETCH_UNSCORED_PEAK_GROUPS = (
        """
        select
            {base_columns},
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id 
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """
    ).format(
        base_columns=BASE_COLUMNS,
        score_columns=SCORE_COLUMNS)

    FETCH_DECOY_PEAK_GROUPS = (
        """
        select
            {base_columns},
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id 
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        where precursor.DECOY == 1
        order by precursor.id;
        """
    )

    FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE = (
        """
        select
            {base_columns},
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id 
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        where precursor.DECOY == 0
        order by transition_group_id;
        """
    ).format(
        base_columns=BASE_COLUMNS,
        score_columns=SCORE_COLUMNS)

    FETCH_VOTED_DATA_DECOY_FREE = (
        """
        select
            {base_columns},
            {score_columns},
            gst.vote_percentage,
            gst.probability,
            gst.ghost_score_id
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        where precursor.DECOY == 0
        order by precursor.id;
        """
    ).format(
        base_columns=BASE_COLUMNS,
        score_columns=SCORE_COLUMNS)

    FETCH_VOTED_DATA = (
        """
        select
                {base_columns},
                {score_columns},
                gst.vote_percentage,
                gst.vote_score
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """
    ).format(
        base_columns=BASE_COLUMNS,
        score_columns=SCORE_COLUMNS)

    FETCH_SCORED_DATA_DECOY_FREE = (
        """
        select
            {base_columns},
            gst.vote_percentage,
            gst.probability,
            gst.d_score,
            gst.weighted_d_score,
            gst.ghost_score_id
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        where precursor.DECOY == 0
        order by precursor.id;
        """
    ).format(base_columns=BASE_COLUMNS)

    FETCH_SCORED_DATA_ALL = (
        """
        select
            {base_columns},
            ms2.VAR_XCORR_SHAPE var_xcorr_shape,
            gst.vote_percentage,
            gst.vote_score,
            gst.d_score,
            gst.alt_d_score,
            gst.ghost_score_id
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """
    ).format(base_columns=BASE_COLUMNS)

    FETCH_TRIC_EXPORT_DATA = (
        """
        select
            feature.run_id run_id,
            precursor.id transition_group_id,
            feature.id id,
            feature.exp_rt RT,
            feature.norm_rt iRT,
            feature.delta_rt delta_rt,
            precursor.PRECURSOR_MZ mz,
            precursor.CHARGE Charge,
            precursor.DECOY decoy,
            peptide.UNMODIFIED_SEQUENCE Sequence,
            peptide.MODIFIED_SEQUENCE FullPeptideName,
            protein.PROTEIN_ACCESSION ProteinName,
            protein.decoy protein_decoy,
            ms1.AREA_INTENSITY aggr_prec_Peak_Area,
            ms1.APEX_INTENSITY aggr_prec_Peak_Apex,
            ms2.AREA_INTENSITY Intensity,
            feature.LEFT_WIDTH leftWidth,
            feature.RIGHT_WIDTH rightWidth,
            gst.m_score m_score,
            gst.weighted_d_score d_score
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """
    )

    FETCH_PEPTIDE_MATRIX_EXPORT_DATA = (
        """
        select
            feature.run_id run_id,
            precursor.id transition_group_id,
            feature.id feature_id,
            feature.exp_rt rt,
            feature.norm_rt irt,
            feature.delta_rt delta_rt,
            precursor.PRECURSOR_MZ mz,
            precursor.CHARGE charge,
            precursor.DECOY decoy,
            peptide.UNMODIFIED_SEQUENCE peptide_sequence,
            peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
            protein.PROTEIN_ACCESSION protein_accession,
            protein.decoy protein_decoy,
            ms1.AREA_INTENSITY ms1_integrated_intensity,
            ms1.APEX_INTENSITY ms1_apex_intensity,
            ms2.AREA_INTENSITY ms2_integrated_intensity,
            feature.LEFT_WIDTH left_width,
            feature.RIGHT_WIDTH right_width,
            gst.q_value,
            gst.d_score
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join ghost_score_table as gst on gst.feature_id = feature.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """
    )

    FETCH_EXPERIMENT_WIDE_PRECURSOR_DATA = (
        """
        select
            precursor.id transition_group_id,
            precursor.PRECURSOR_MZ mz,
            precursor.CHARGE charge,
            precursor.DECOY decoy,
            peptide.UNMODIFIED_SEQUENCE peptide_sequence,
            peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
            protein.PROTEIN_ACCESSION protein_accession,
            protein.decoy protein_decoy
        from precursor
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        group by PRECURSOR.ID
        order by PRECURSOR.ID;
        """
    )

    FETCH_TRAIN_CHROMATOGRAM_SCORING_DATA = (
        """
        select
            {base_columns},
            feature.delta_rt delta_rt,
            ms2.VAR_MASSDEV_SCORE transition_mass_dev_score,
            ms1.VAR_MASSDEV_SCORE precursor_mass_dev_score,
            gst.probability,
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        left join ghost_score_table as gst on gst.feature_id = feature.id
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """.format(
            base_columns=BASE_COLUMNS,
            score_columns=SCORE_COLUMNS
        )
    )

    FETCH_ALL_SCORED_DATA = (
        """
        select
            {base_columns},
            feature.delta_rt delta_rt,
            ms2.VAR_MASSDEV_SCORE transition_mass_dev_score,
            ms1.VAR_MASSDEV_SCORE precursor_mass_dev_score,
            gst.probability,
            gst.d_score,
            gst.q_value,
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        left join ghost_score_table as gst on gst.feature_id = feature.id
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """.format(
            base_columns=BASE_COLUMNS,
            score_columns=SCORE_COLUMNS
        )
    )

    FETCH_ALL_UNSCORED_DATA = (
        """
        select
            {base_columns},
            feature.delta_rt delta_rt,
            ms2.VAR_MASSDEV_SCORE transition_mass_dev_score,
            ms1.VAR_MASSDEV_SCORE precursor_mass_dev_score,
            {score_columns}
        from precursor 
        inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
        inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
        inner join feature on feature.precursor_id = precursor.id
        left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
        left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
        inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
        inner join protein as protein on protein.id = pep_prot_map.protein_id
        order by precursor.id;
        """.format(
            base_columns=BASE_COLUMNS,
            score_columns=SCORE_COLUMNS
        )
    )
