class CreateTable:

    CREATE_GHOSTSCORE_TABLE = """
        CREATE TABLE IF NOT EXISTS GHOST_SCORE_TABLE (
            GHOST_SCORE_ID INTEGER PRIMARY KEY,
            FEATURE_ID INTEGER NOT NULL,
            VOTE_PERCENTAGE REAL,
            PROBABILITY REAL,
            D_SCORE REAL,
            Q_VALUE REAL,
            FOREIGN KEY (FEATURE_ID)
                REFERENCES FEATURE (ID)
        )
        """


class CreateIndex:

    CREATE_PROTEIN_IDX = """
        CREATE INDEX IF NOT EXISTS idx_protein
        ON PROTEIN(id);
        """

    CREATE_PRECURSOR_IDX = """
        CREATE INDEX IF NOT EXISTS idx_precursor_id
        ON precursor(id);
        """

    CREATE_PREC_PEP_MAPPING_IDX = """
        CREATE INDEX IF NOT EXISTS idx_prec_pep_mapping
        ON PRECURSOR_PEPTIDE_MAPPING(precursor_id);
        """

    CREATE_PROT_PEP_MAPPING_IDX = """
        CREATE INDEX IF NOT EXISTS idx_prot_pep_mapping
        ON PEPTIDE_PROTEIN_MAPPING(peptide_id);
        """

    CREATE_PEPTIDE_IDX = """
        CREATE INDEX IF NOT EXISTS idx_peptide
        ON PEPTIDE(id);
        """

    CREATE_FEATURE_IDX = """
        CREATE INDEX IF NOT EXISTS idx_feature_precursor
        ON FEATURE(precursor_id);
        """

    CREATE_FEATURE_MS2_IDX = """
        CREATE INDEX IF NOT EXISTS idx_feature_feature_ms2
        ON FEATURE_MS2(feature_id)
        """

    CREATE_FEATURE_MS1_IDX = """
        CREATE INDEX IF NOT EXISTS idx_feature_feature_ms1
        ON FEATURE_MS1(feature_id)
        """

    CREATE_GHOST_SCORE_IDX = """
        CREATE INDEX IF NOT EXISTS idx_ghost_score_feature
        ON GHOST_SCORE_TABLE(FEATURE_ID)
        """

    ALL_INDICES = [
        CREATE_PRECURSOR_IDX,
        CREATE_FEATURE_IDX,
        CREATE_PROTEIN_IDX,
        CREATE_PEPTIDE_IDX,
        CREATE_FEATURE_MS1_IDX,
        CREATE_FEATURE_MS2_IDX,
        CREATE_PREC_PEP_MAPPING_IDX,
        CREATE_PROT_PEP_MAPPING_IDX,
    ]


class SelectPeakGroups:

    FETCH_SCORED_PYPROPHET_RECORDS = """
                SELECT
                    *
                FROM FEATURE_MS2
                INNER JOIN(
                    SELECT
                        ID FEATURE_ID,
                        PRECURSOR_ID PRECURSOR_ID,
                        EXP_RT RT_APEX,
                        LEFT_WIDTH RT_START,
                        RIGHT_WIDTH RT_END
                    from FEATURE
                ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
                INNER JOIN (
                    SELECT
                        ID,
                        CHARGE,
                        PRECURSOR_MZ MZ,
                        DECOY
                    FROM PRECURSOR
                ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN (
                    SELECT
                        ID,
                        MODIFIED_SEQUENCE,
                        UNMODIFIED_SEQUENCE
                    FROM PEPTIDE
                ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
                INNER JOIN (
                    SELECT
                        ID,
                        PROTEIN_ACCESSION
                    FROM PROTEIN
                ) PROTEIN ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
                LEFT JOIN (
                    SELECT
                        FEATURE_ID,
                        SCORE,
                        PEP,
                        PVALUE,
                        QVALUE
                    FROM SCORE_MS2
                ) SCORE_MS2 ON SCORE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
                ORDER BY PRECURSOR.ID ASC,
                         FEATURE.RT_APEX ASC
                """

    FETCH_TRAINING_RECORDS = """
                SELECT
                    *
                FROM FEATURE_MS2
                INNER JOIN(
                    SELECT
                        ID FEATURE_ID,
                        PRECURSOR_ID PRECURSOR_ID,
                        EXP_RT RT_APEX,
                        LEFT_WIDTH RT_START,
                        RIGHT_WIDTH RT_END
                    from FEATURE
                ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
                INNER JOIN (
                    SELECT
                        ID,
                        CHARGE,
                        PRECURSOR_MZ MZ,
                        DECOY
                    FROM PRECURSOR
                ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN (
                    SELECT
                        ID,
                        MODIFIED_SEQUENCE,
                        UNMODIFIED_SEQUENCE
                    FROM PEPTIDE
                ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
                INNER JOIN (
                    SELECT
                        ID,
                        PROTEIN_ACCESSION
                    FROM PROTEIN
                ) PROTEIN ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
                ORDER BY PRECURSOR.ID ASC,
                         FEATURE.RT_APEX ASC
                """

    FETCH_PREC_RECORDS = """
                SELECT
                    *
                FROM FEATURE_MS2
                INNER JOIN(
                    SELECT
                        ID FEATURE_ID,
                        PRECURSOR_ID PRECURSOR_ID,
                        EXP_RT RT_APEX,
                        LEFT_WIDTH RT_START,
                        RIGHT_WIDTH RT_END
                    from FEATURE
                ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
                INNER JOIN (
                    SELECT
                        ID,
                        CHARGE,
                        PRECURSOR_MZ MZ,
                        DECOY
                    FROM PRECURSOR
                ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
                INNER JOIN (
                    SELECT
                        ID,
                        MODIFIED_SEQUENCE,
                        UNMODIFIED_SEQUENCE
                    FROM PEPTIDE
                ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
                INNER JOIN (
                    SELECT
                        GHOST_SCORE_ID,
                        FEATURE_ID,
                        PROBABILITY,
                        VOTE_PERCENTAGE
                    FROM GHOST_SCORE_TABLE
                ) GHOST_SCORE_TABLE ON FEATURE.FEATURE_ID = GHOST_SCORE_TABLE.FEATURE_ID
                ORDER BY PRECURSOR.ID ASC,
                         FEATURE.RT_APEX ASC
                """

    FETCH_CHROMATOGRAM_TRAINING_RECORDS = """
            SELECT
                *
            FROM FEATURE_MS2
            INNER JOIN(
                SELECT
                    ID FEATURE_ID,
                    PRECURSOR_ID PRECURSOR_ID,
                    EXP_RT RT_APEX,
                    LEFT_WIDTH RT_START,
                    RIGHT_WIDTH RT_END
                from FEATURE
            ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
            INNER JOIN (
                SELECT
                    ID,
                    CHARGE,
                    PRECURSOR_MZ MZ,
                    DECOY
                FROM PRECURSOR
            ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN (
                SELECT
                    FEATURE_ID MS1_FEATURE_ID,
                    AREA_INTENSITY AREA_INTENSITY_MS1,
                    APEX_INTENSITY APEX_INTENSITY_MS1,
                    VAR_MASSDEV_SCORE VAR_MASSDEV_SCORE_MS1,
                    VAR_MI_SCORE VAR_MI_SCORE_MS1,
                    VAR_MI_CONTRAST_SCORE VAR_MI_CONTRAST_SCORE_MS1,
                    VAR_MI_COMBINED_SCORE VAR_MI_COMBINED_SCORE_MS1,
                    VAR_ISOTOPE_CORRELATION_SCORE VAR_ISOTOPE_CORRELATION_SCORE_MS1,
                    VAR_ISOTOPE_OVERLAP_SCORE VAR_ISOTOPE_OVERLAP_SCORE_MS1,
                    VAR_IM_MS1_DELTA_SCORE,
                    VAR_XCORR_COELUTION VAR_XCORR_COELUTION_MS1,
                    VAR_XCORR_COELUTION_CONTRAST VAR_XCORR_COELUTION_CONTRAST_MS1,
                    VAR_XCORR_COELUTION_COMBINED VAR_XCORR_COELUTION_COMBINED_MS1,
                    VAR_XCORR_SHAPE VAR_XCORR_SHAPE_MS1,
                    VAR_XCORR_SHAPE_CONTRAST VAR_XCORR_SHAPE_CONTRAST_MS1,
                    VAR_XCORR_SHAPE_COMBINED VAR_XCORR_SHAPE_COMBINED_MS1
                FROM FEATURE_MS1
            ) FEATURE_MS1 ON FEATURE.FEATURE_ID = FEATURE_MS1.MS1_FEATURE_ID
            INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
            INNER JOIN (
                SELECT
                    ID,
                    MODIFIED_SEQUENCE,
                    UNMODIFIED_SEQUENCE
                FROM PEPTIDE
            ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
            INNER JOIN (
                SELECT
                    GHOST_SCORE_ID,
                    PROBABILITY,
                    VOTE_PERCENTAGE,
                    FEATURE_ID
                FROM GHOST_SCORE_TABLE
            ) GHOST_SCORE_TABLE ON FEATURE.FEATURE_ID = GHOST_SCORE_TABLE.FEATURE_ID
            ORDER BY PRECURSOR.ID ASC,
                     FEATURE.RT_APEX ASC
            """

    FETCH_DENOIZED_REDUCED = """
        SELECT
            *
        FROM FEATURE_MS2
        INNER JOIN(
            SELECT
                ID FEATURE_ID,
                PRECURSOR_ID PRECURSOR_ID,
                EXP_RT RT_APEX,
                LEFT_WIDTH RT_START,
                RIGHT_WIDTH RT_END
            from FEATURE
        ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
        INNER JOIN (
            SELECT
                ID,
                CHARGE,
                PRECURSOR_MZ MZ,
                DECOY
            FROM PRECURSOR
        ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                FEATURE_ID MS1_FEATURE_ID,
                AREA_INTENSITY AREA_INTENSITY_MS1,
                APEX_INTENSITY APEX_INTENSITY_MS1,
                VAR_MASSDEV_SCORE VAR_MASSDEV_SCORE_MS1,
                VAR_MI_SCORE VAR_MI_SCORE_MS1,
                VAR_MI_CONTRAST_SCORE VAR_MI_CONTRAST_SCORE_MS1,
                VAR_MI_COMBINED_SCORE VAR_MI_COMBINED_SCORE_MS1,
                VAR_ISOTOPE_CORRELATION_SCORE VAR_ISOTOPE_CORRELATION_SCORE_MS1,
                VAR_ISOTOPE_OVERLAP_SCORE VAR_ISOTOPE_OVERLAP_SCORE_MS1,
                VAR_IM_MS1_DELTA_SCORE,
                VAR_XCORR_COELUTION VAR_XCORR_COELUTION_MS1,
                VAR_XCORR_COELUTION_CONTRAST VAR_XCORR_COELUTION_CONTRAST_MS1,
                VAR_XCORR_COELUTION_COMBINED VAR_XCORR_COELUTION_COMBINED_MS1,
                VAR_XCORR_SHAPE VAR_XCORR_SHAPE_MS1,
                VAR_XCORR_SHAPE_CONTRAST VAR_XCORR_SHAPE_CONTRAST_MS1,
                VAR_XCORR_SHAPE_COMBINED VAR_XCORR_SHAPE_COMBINED_MS1
            FROM FEATURE_MS1
        ) FEATURE_MS1 ON FEATURE.FEATURE_ID = FEATURE_MS1.MS1_FEATURE_ID
        INNER JOIN (
            SELECT
                GHOST_SCORE_ID,
                PROBABILITY,
                VOTE_PERCENTAGE,
                FEATURE_ID
            FROM GHOST_SCORE_TABLE
        ) GHOST_SCORE_TABLE ON FEATURE.FEATURE_ID = GHOST_SCORE_TABLE.FEATURE_ID
        ORDER BY PRECURSOR.ID ASC,
                 FEATURE.RT_APEX ASC
        """

    BUILD_GLOBAL_MODEL_QUERY = """
        SELECT
            FEATURE_MS2.FEATURE_ID,
            FEATURE.PRECURSOR_ID,
            MODIFIED_SEQUENCE,
            UNMODIFIED_SEQUENCE,
            CHARGE,
            DECOY,
            PROTEIN_ACCESSION,
            Q_VALUE,
            D_SCORE,
            PROBABILITY
        FROM FEATURE_MS2
        INNER JOIN(
            SELECT
                ID FEATURE_ID,
                PRECURSOR_ID PRECURSOR_ID,
                EXP_RT RT_APEX,
                LEFT_WIDTH RT_START,
                RIGHT_WIDTH RT_END
            from FEATURE
        ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
        INNER JOIN (
            SELECT
                ID,
                CHARGE,
                PRECURSOR_MZ MZ,
                DECOY
            FROM PRECURSOR
        ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                FEATURE_ID MS1_FEATURE_ID,
                AREA_INTENSITY AREA_INTENSITY_MS1,
                APEX_INTENSITY APEX_INTENSITY_MS1
            FROM FEATURE_MS1
        ) FEATURE_MS1 ON FEATURE.FEATURE_ID = FEATURE_MS1.MS1_FEATURE_ID
        INNER JOIN (
            SELECT
                GHOST_SCORE_ID,
                PROBABILITY,
                VOTE_PERCENTAGE,
                FEATURE_ID,
                Q_VALUE,
                D_SCORE
            FROM GHOST_SCORE_TABLE
        ) GHOST_SCORE_TABLE ON FEATURE.FEATURE_ID = GHOST_SCORE_TABLE.FEATURE_ID
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                ID,
                MODIFIED_SEQUENCE,
                UNMODIFIED_SEQUENCE
            FROM PEPTIDE
        ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN (
            SELECT
                ID,
                PROTEIN_ACCESSION
            FROM PROTEIN
        ) PROTEIN ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
        ORDER BY PRECURSOR.ID ASC,
                 FEATURE.RT_APEX ASC
    
    """

    FETCH_PRECURSORS_FOR_EXPORT_REDUCED = """
        SELECT
            FEATURE_MS2.FEATURE_ID,
            FEATURE.PRECURSOR_ID,
            RT_APEX,
            RT_START,
            RT_END,
            CHARGE,
            MZ,
            DECOY,
            MODIFIED_SEQUENCE,
            UNMODIFIED_SEQUENCE,
            PROTEIN_ACCESSION,
            GHOST_SCORE_ID,
            PROBABILITY,
            VOTE_PERCENTAGE,
            FEATURE_MS2.AREA_INTENSITY,
            Q_VALUE,
            D_SCORE,
            PROBABILITY
        FROM FEATURE_MS2
        INNER JOIN(
            SELECT
                ID FEATURE_ID,
                PRECURSOR_ID PRECURSOR_ID,
                EXP_RT RT_APEX,
                LEFT_WIDTH RT_START,
                RIGHT_WIDTH RT_END
            from FEATURE
        ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
        INNER JOIN (
            SELECT
                ID,
                CHARGE,
                PRECURSOR_MZ MZ,
                DECOY
            FROM PRECURSOR
        ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                FEATURE_ID MS1_FEATURE_ID,
                AREA_INTENSITY AREA_INTENSITY_MS1,
                APEX_INTENSITY APEX_INTENSITY_MS1
            FROM FEATURE_MS1
        ) FEATURE_MS1 ON FEATURE.FEATURE_ID = FEATURE_MS1.MS1_FEATURE_ID
        INNER JOIN (
            SELECT
                GHOST_SCORE_ID,
                PROBABILITY,
                VOTE_PERCENTAGE,
                FEATURE_ID,
                Q_VALUE,
                D_SCORE
            FROM GHOST_SCORE_TABLE
        ) GHOST_SCORE_TABLE ON FEATURE.FEATURE_ID = GHOST_SCORE_TABLE.FEATURE_ID
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                ID,
                MODIFIED_SEQUENCE,
                UNMODIFIED_SEQUENCE
            FROM PEPTIDE
        ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN (
            SELECT
                ID,
                PROTEIN_ACCESSION
            FROM PROTEIN
        ) PROTEIN ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
        ORDER BY PRECURSOR.ID ASC,
                 FEATURE.RT_APEX ASC
        """

    FETCH_FEATURES_REDUCED = """
        SELECT
            *
        FROM FEATURE_MS2
        INNER JOIN(
            SELECT
                ID FEATURE_ID,
                PRECURSOR_ID PRECURSOR_ID,
                EXP_RT RT_APEX,
                LEFT_WIDTH RT_START,
                RIGHT_WIDTH RT_END
            from FEATURE
            ) FEATURE ON FEATURE_MS2.FEATURE_ID = FEATURE.FEATURE_ID
        INNER JOIN (
            SELECT
                ID,
                CHARGE,
                PRECURSOR_MZ MZ,
                DECOY
            FROM PRECURSOR
            ) PRECURSOR ON FEATURE.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                FEATURE_ID MS1_FEATURE_ID,
                AREA_INTENSITY AREA_INTENSITY_MS1,
                APEX_INTENSITY APEX_INTENSITY_MS1
            FROM FEATURE_MS1
        ) FEATURE_MS1 ON FEATURE.FEATURE_ID = FEATURE_MS1.MS1_FEATURE_ID
        INNER JOIN PRECURSOR_PEPTIDE_MAPPING ON PRECURSOR_PEPTIDE_MAPPING.PRECURSOR_ID = PRECURSOR.ID
        INNER JOIN (
            SELECT
                ID,
                MODIFIED_SEQUENCE,
                UNMODIFIED_SEQUENCE
            FROM PEPTIDE
        ) PEPTIDE ON PEPTIDE.ID = PRECURSOR_PEPTIDE_MAPPING.PEPTIDE_ID
        INNER JOIN PEPTIDE_PROTEIN_MAPPING ON PEPTIDE_PROTEIN_MAPPING.PEPTIDE_ID = PEPTIDE.ID
        INNER JOIN (
            SELECT
                ID,
                PROTEIN_ACCESSION
            FROM PROTEIN
        ) PROTEIN ON PROTEIN.ID = PEPTIDE_PROTEIN_MAPPING.PROTEIN_ID
        ORDER BY PRECURSOR.ID ASC,
                 FEATURE.RT_APEX ASC
        """
