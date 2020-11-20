base_columns = """
precursor.id transition_group_id,
feature.id feature_id,
feature.exp_rt exp_rt,
feature.norm_rt norm_rt,
feature.delta_rt delta_rt,
precursor.PRECURSOR_MZ mz,
precursor.CHARGE charge,
precursor.DECOY decoy,
peptide.UNMODIFIED_SEQUENCE peptide_sequence,
peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
protein.PROTEIN_ACCESSION protein_accession,
protein.decoy protein_decoy,
ms1.AREA_INTENSITY area_intensity,
ms1.APEX_INTENSITY apex_intensity,
ms2.AREA_INTENSITY ms2_area_intensity,
ms2.TOTAL_AREA_INTENSITY total_area_intensity,
ms1.var_massdev_score,
ms1.var_isotope_correlation_score,
ms1.var_isotope_overlap_score,
ms1.var_xcorr_coelution_contrast,
ms1.var_xcorr_coelution_combined,
ms1.var_xcorr_shape_contrast,
ms1.var_xcorr_shape_combined,
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

peak_group_precursor_joins = """
inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
inner join feature on feature.precursor_id = precursor.id 
left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
inner join protein as protein on protein.id = pep_prot_map.protein_id
"""

FETCH_UNSCORED_PEAK_GROUPS = """
select
    precursor.id transition_group_id,
    feature.id feature_id,
    feature.exp_rt exp_rt,
    feature.norm_rt norm_rt,
    feature.delta_rt delta_rt,
    precursor.PRECURSOR_MZ mz,
    precursor.CHARGE charge,
    precursor.DECOY decoy,
    peptide.UNMODIFIED_SEQUENCE peptide_sequence,
    peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
    protein.PROTEIN_ACCESSION protein_accession,
    protein.decoy protein_decoy,
    ms1.AREA_INTENSITY area_intensity,
    ms1.APEX_INTENSITY apex_intensity,
    ms2.AREA_INTENSITY ms2_area_intensity,
    ms2.TOTAL_AREA_INTENSITY total_area_intensity,
    ms1.var_massdev_score,
    ms1.var_isotope_correlation_score,
    ms1.var_isotope_overlap_score,
    ms1.var_xcorr_coelution_contrast,
    ms1.var_xcorr_coelution_combined,
    ms1.var_xcorr_shape_contrast,
    ms1.var_xcorr_shape_combined,
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
from precursor 
inner join PRECURSOR_PEPTIDE_MAPPING as pre_pep_map on pre_pep_map.precursor_id = precursor.id
inner join peptide as peptide on peptide.id = pre_pep_map.peptide_id
inner join feature on feature.precursor_id = precursor.id 
left join feature_ms2 as ms2 on ms2.feature_id = feature.id 
left join feature_ms1 as ms1 on ms1.feature_id = feature.id 
inner join PEPTIDE_PROTEIN_MAPPING as pep_prot_map on pep_prot_map.peptide_id = peptide.id
inner join protein as protein on protein.id = pep_prot_map.protein_id
order by transition_group_id;
"""

FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE = """
select
    precursor.id transition_group_id,
    feature.id feature_id,
    feature.exp_rt exp_rt,
    feature.norm_rt norm_rt,
    feature.delta_rt delta_rt,
    precursor.PRECURSOR_MZ mz,
    precursor.CHARGE charge,
    precursor.DECOY decoy,
    peptide.UNMODIFIED_SEQUENCE peptide_sequence,
    peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
    protein.PROTEIN_ACCESSION protein_accession,
    protein.decoy protein_decoy,
    ms1.AREA_INTENSITY area_intensity,
    ms1.APEX_INTENSITY apex_intensity,
    ms2.AREA_INTENSITY ms2_area_intensity,
    ms2.TOTAL_AREA_INTENSITY total_area_intensity,
    ms1.var_massdev_score,
    ms1.var_isotope_correlation_score,
    ms1.var_isotope_overlap_score,
    ms1.var_xcorr_coelution_contrast,
    ms1.var_xcorr_coelution_combined,
    ms1.var_xcorr_shape_contrast,
    ms1.var_xcorr_shape_combined,
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


CREATE_GHOSTSCORE_TABLE = """CREATE TABLE IF NOT EXISTS ghost_score_table (
    ghost_score_id INTEGER PRIMARY KEY,
    feature_id INTEGER not null,
    vote_percentage REAL,
    m_score REAL,
    d_score REAL,
    alt_d_score REAL,
    FOREIGN KEY (feature_id)
        REFERENCES FEATURE (id)
)
"""


FETCH_VOTED_DATA = """
select
    precursor.id transition_group_id,
    feature.id feature_id,
    feature.exp_rt exp_rt,
    feature.norm_rt norm_rt,
    feature.delta_rt delta_rt,
    precursor.PRECURSOR_MZ mz,
    precursor.CHARGE charge,
    precursor.DECOY decoy,
    peptide.UNMODIFIED_SEQUENCE peptide_sequence,
    peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
    protein.PROTEIN_ACCESSION protein_accession,
    protein.decoy protein_decoy,
    ms1.AREA_INTENSITY area_intensity,
    ms1.APEX_INTENSITY apex_intensity,
    ms2.AREA_INTENSITY ms2_area_intensity,
    ms2.TOTAL_AREA_INTENSITY total_area_intensity,
    ms1.var_massdev_score,
    ms1.var_isotope_correlation_score,
    ms1.var_isotope_overlap_score,
    ms1.var_xcorr_coelution_contrast,
    ms1.var_xcorr_coelution_combined,
    ms1.var_xcorr_shape_contrast,
    ms1.var_xcorr_shape_combined,
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
    ms2.VAR_YSERIES_SCORE,
    gst.vote_percentage
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
order by transition_group_id;
"""

FETCH_SCORED_DATA = """
select
    precursor.id transition_group_id,
    feature.id feature_id,
    feature.exp_rt exp_rt,
    feature.norm_rt norm_rt,
    feature.delta_rt delta_rt,
    precursor.PRECURSOR_MZ mz,
    precursor.CHARGE charge,
    precursor.DECOY decoy,
    peptide.UNMODIFIED_SEQUENCE peptide_sequence,
    peptide.MODIFIED_SEQUENCE modified_peptide_sequence,
    protein.PROTEIN_ACCESSION protein_accession,
    protein.decoy protein_decoy,
    ms1.AREA_INTENSITY area_intensity,
    ms1.APEX_INTENSITY apex_intensity,
    ms2.AREA_INTENSITY ms2_area_intensity,
    ms2.TOTAL_AREA_INTENSITY total_area_intensity,
    ms2.VAR_XCORR_SHAPE var_xcorr_shape,
    gst.vote_percentage,
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
where precursor.DECOY == 0
order by transition_group_id;
"""