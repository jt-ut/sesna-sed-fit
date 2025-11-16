"""
Gutermuth et al. (2009) Classification Scoring Functions
========================================================

Vectorized implementations of color-cut criteria for astronomical source classification.

Author: Generated for probabilistic YSO classification
Date: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Zero-point fluxes for magnitude conversion (mJy)
FLUX_ZERO = {
    'J': 1594.0,      # 2MASS J-band
    'H': 1024.0,      # 2MASS H-band
    'KS': 666.7,      # 2MASS Ks-band
    '3_6': 280.9,     # IRAC [3.6]
    '4_5': 179.7,     # IRAC [4.5]
    '5_8': 115.0,     # IRAC [5.8]
    '8_0': 64.13,     # IRAC [8.0]
    '24': 7.17        # MIPS [24]
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Sigmoid function for smooth criterion scoring.
    
    Maps distance from threshold to score in [0, 1]:
    - Large positive x → score ≈ 1 (strongly passes)
    - x = 0 → score = 0.5 (at boundary)
    - Large negative x → score ≈ 0 (strongly fails)
    
    Parameters
    ----------
    x : float or array
        Distance from threshold (in units of scale parameter)
    
    Returns
    -------
    score : float or array
        Value in [0, 1]
    """
    # Convert pandas Series to numpy array to avoid indexing issues
    if hasattr(x, 'values'):
        x = x.values
    
    with np.errstate(invalid='ignore'):  # Suppress warnings for NaN operations
        result = 1.0 / (1.0 + np.exp(-x))
    
    # Explicitly ensure NaN inputs produce NaN outputs
    result = np.where(np.isnan(x), np.nan, result)
    
    return result


def flux_to_mag_batch(fnu_df: pd.DataFrame, 
                      origin_df: pd.DataFrame,
                      bands: Optional[list] = None) -> pd.DataFrame:
    """
    Convert flux (mJy) to magnitude for multiple bands (vectorized).
    
    Respects ORIGIN_FNU flags:
    - ORIGIN == 1: Valid detection, convert to magnitude
    - ORIGIN > 1: Non-detection/upper limit, set to NaN
    
    Parameters
    ----------
    fnu_df : pd.DataFrame
        Flux values in mJy (N sources × M bands)
    origin_df : pd.DataFrame
        Origin flags (N sources × M bands)
        1 = valid detection, >1 = upper limit
    bands : list, optional
        List of band names to convert. If None, uses all bands in FLUX_ZERO.
    
    Returns
    -------
    mags_df : pd.DataFrame
        Magnitudes (N sources × M bands), NaN where not detected
    """
    if bands is None:
        bands = list(FLUX_ZERO.keys())
    
    mags_df = pd.DataFrame(index=fnu_df.index)
    
    for band in bands:
        if band not in fnu_df.columns:
            # Band not in data, skip
            continue
        
        # Mask for valid detections
        valid = (origin_df[band] == 1).values
        
        # Convert to magnitude: mag = -2.5 * log10(flux / flux_zero)
        mags_df[band] = np.where(
            valid,
            -2.5 * np.log10(fnu_df[band].values / FLUX_ZERO[band]),
            np.nan
        )
    
    return mags_df


def flux_to_mag_err_batch(fnu_df: pd.DataFrame,
                          sigma_fnu_df: pd.DataFrame,
                          origin_df: pd.DataFrame,
                          bands: Optional[list] = None) -> pd.DataFrame:
    """
    Convert flux uncertainties to magnitude uncertainties (vectorized).
    
    Uses the standard error propagation formula:
    σ_mag = 1.0857 × (σ_flux / flux)
    
    Parameters
    ----------
    fnu_df : pd.DataFrame
        Flux values in mJy
    sigma_fnu_df : pd.DataFrame
        Flux uncertainties in mJy
    origin_df : pd.DataFrame
        Origin flags (1 = valid)
    bands : list, optional
        List of band names to convert
    
    Returns
    -------
    mag_errs_df : pd.DataFrame
        Magnitude uncertainties, NaN where not detected
    """
    if bands is None:
        bands = list(FLUX_ZERO.keys())
    
    mag_errs_df = pd.DataFrame(index=fnu_df.index)
    
    for band in bands:
        if band not in fnu_df.columns:
            continue
        
        valid = (origin_df[band] == 1).values
        
        # σ_mag = 1.0857 × (σ_flux / flux)
        mag_errs_df[band] = np.where(
            valid,
            1.0857 * (sigma_fnu_df[band].values / fnu_df[band].values),
            np.nan
        )
    
    return mag_errs_df


def compute_color_uncertainty(mag_err_1: np.ndarray, 
                               mag_err_2: np.ndarray) -> np.ndarray:
    """
    Compute uncertainty in color (mag1 - mag2).
    
    Uncertainties add in quadrature:
    σ(color) = sqrt(σ_mag1² + σ_mag2²)
    
    Parameters
    ----------
    mag_err_1, mag_err_2 : array
        Magnitude uncertainties
    
    Returns
    -------
    color_err : array
        Color uncertainty
    """
    return np.sqrt(mag_err_1**2 + mag_err_2**2)


def aggregate_criterion_scores_v1_1(score_array: np.ndarray) -> np.ndarray:
    """
    Aggregate multiple criterion scores using cumulative contributions (v1.1).
    
    Strategy (Framework v1.1):
    1. Start at neutral point (0.5)
    2. Each non-NaN criterion contributes (score - 0.5) / N_total
    3. Return NaN if no criteria were evaluable
    
    This preserves evidence direction:
    - If 1/5 criteria pass strongly (score=1.0), final = 0.5 + (1.0-0.5)/5 = 0.6 (evidence FOR)
    - Old method would give 1.0 × sqrt(0.2) = 0.45 (evidence AGAINST - wrong!)
    
    Parameters
    ----------
    score_array : np.ndarray
        Array of shape (N_sources, N_criteria) with criterion scores in [0, 1] or NaN
    
    Returns
    -------
    final_scores : np.ndarray
        Aggregated scores in [0, 1] with 0.5 = neutral, or NaN
    
    Notes
    -----
    See scoring_framework_reference.md v1.1 for complete documentation.
    """
    n_sources, n_criteria = score_array.shape
    
    # Start at neutral
    aggregate = np.full(n_sources, 0.5)
    
    # Count evaluable criteria per source
    n_evaluable = np.sum(~np.isnan(score_array), axis=1)
    
    # Add cumulative contributions
    for i in range(n_criteria):
        criterion_scores = score_array[:, i]
        valid = ~np.isnan(criterion_scores)
        
        # Contribution: (score - 0.5) / N_total
        # Each criterion can contribute ±0.5/N to final score
        contribution = (criterion_scores - 0.5) / n_criteria
        aggregate = np.where(valid, aggregate + contribution, aggregate)
    
    # Set to NaN where no criteria were evaluable
    final_scores = np.where(n_evaluable == 0, np.nan, aggregate)
    
    return final_scores


# ==============================================================================
# PHASE 1: PAH GALAXY SCORING
# ==============================================================================

def compute_pah_scores_batch(fnu_df: pd.DataFrame,
                             sigma_fnu_df: pd.DataFrame,
                             origin_df: pd.DataFrame,
                             debug_idx: Optional[int] = None) -> pd.Series:
    """
    Compute PAH galaxy scores for all sources (Phase 1 criteria).
    
    Updated to Framework v1.1:
    - Uses measurement-specific uncertainties in sigmoid
    - Uses cumulative contribution aggregation
    
    PAH galaxies are identified by strong 8μm emission from polycyclic aromatic
    hydrocarbon (PAH) features. Criteria from Gutermuth et al. (2009) Appendix A.1.
    
    Criteria evaluated:
    1.1: [4.5] - [5.8] < 1.05 × ([5.8] - [8.0] - 1)
    1.2: [4.5] - [5.8] < 1.05
    1.3: [5.8] - [8.0] > 1
    1.4: [4.5] > 11.5 (faint magnitude limit)
    1.5: [3.6] - [5.8] constraint (conditional on [4.5] - [8.0])
    
    Parameters
    ----------
    fnu_df : pd.DataFrame
        Flux table (N × 8) with columns: J, H, KS, 3_6, 4_5, 5_8, 8_0, 24
        Units: mJy
    sigma_fnu_df : pd.DataFrame
        Flux uncertainty table (same structure as fnu_df)
        Units: mJy
    origin_df : pd.DataFrame
        Origin flag table (same structure, 1=valid detection, >1=upper limit)
    debug_idx : int, optional
        If provided, print intermediate values for this source index
    
    Returns
    -------
    scores : pd.Series
        PAH galaxy scores in [0, 1], or NaN if insufficient photometry
        Index matches input DataFrames
    
    Notes
    -----
    Required bands: [4.5], [5.8], [8.0] (minimum)
    Optional bands: [3.6] (for criterion 1.5)
    
    Score interpretation:
    - score > 0.7: Strong PAH galaxy candidate
    - score 0.4-0.7: Moderate evidence
    - score < 0.4: Unlikely to be PAH galaxy
    """
    n_sources = len(fnu_df)
    
    # Convert fluxes to magnitudes for all potentially useful bands
    # Note: Not all bands are required for all criteria - they're used opportunistically
    # Missing bands → NaN for criteria that need them
    possible_bands = ['3_6', '4_5', '5_8', '8_0']
    mags = flux_to_mag_batch(fnu_df, origin_df, bands=possible_bands)
    
    # Convert flux uncertainties to magnitude uncertainties (v1.1)
    mag_errs = flux_to_mag_err_batch(fnu_df, sigma_fnu_df, origin_df, bands=possible_bands)
    
    # Ensure all possible bands exist as columns (NaN if missing from data)
    # This allows color computations to work without KeyError
    for band in possible_bands:
        if band not in mags.columns:
            mags[band] = np.nan
        if band not in mag_errs.columns:
            mag_errs[band] = np.nan
    
    # Compute colors
    color_45_58 = mags['4_5'] - mags['5_8']  # [4.5] - [5.8]
    color_58_80 = mags['5_8'] - mags['8_0']  # [5.8] - [8.0]
    color_45_80 = mags['4_5'] - mags['8_0']  # [4.5] - [8.0]
    color_36_58 = mags['3_6'] - mags['5_8']  # [3.6] - [5.8]
    
    # Compute color uncertainties (v1.1)
    sigma_45_58 = compute_color_uncertainty(mag_errs['4_5'], mag_errs['5_8'])
    sigma_58_80 = compute_color_uncertainty(mag_errs['5_8'], mag_errs['8_0'])
    sigma_45_80 = compute_color_uncertainty(mag_errs['4_5'], mag_errs['8_0'])
    sigma_36_58 = compute_color_uncertainty(mag_errs['3_6'], mag_errs['5_8'])
    
    # -------------------------------------------------------------------------
    # Criterion 1.1: [4.5] - [5.8] < 1.05 × ([5.8] - [8.0] - 1)
    # -------------------------------------------------------------------------
    # Left side: [4.5] - [5.8], uncertainty = sigma_45_58
    # Right side: 1.05 × ([5.8] - [8.0] - 1), uncertainty = 1.05 × sigma_58_80
    # Combined uncertainty for (LHS - RHS)
    sigma_rhs_1_1 = 1.05 * sigma_58_80
    sigma_1_1 = np.sqrt(sigma_45_58**2 + sigma_rhs_1_1**2)
    
    threshold_1_1 = 1.05 * (color_58_80 - 1.0)
    difference_1_1 = color_45_58 - threshold_1_1
    # Want LESS THAN: score high when difference is negative
    score_1_1 = sigmoid(-difference_1_1 / sigma_1_1)
    
    # -------------------------------------------------------------------------
    # Criterion 1.2: [4.5] - [5.8] < 1.05
    # -------------------------------------------------------------------------
    sigma_1_2 = sigma_45_58
    score_1_2 = sigmoid(-(color_45_58 - 1.05) / sigma_1_2)
    
    # -------------------------------------------------------------------------
    # Criterion 1.3: [5.8] - [8.0] > 1
    # -------------------------------------------------------------------------
    sigma_1_3 = sigma_58_80
    score_1_3 = sigmoid((color_58_80 - 1.0) / sigma_1_3)
    
    # -------------------------------------------------------------------------
    # Criterion 1.4: [4.5] > 11.5 (faint sources)
    # -------------------------------------------------------------------------
    sigma_1_4 = mag_errs['4_5']
    score_1_4 = sigmoid((mags['4_5'] - 11.5) / sigma_1_4)
    
    # -------------------------------------------------------------------------
    # Criterion 1.5: [3.6] - [5.8] constraint (conditional)
    # -------------------------------------------------------------------------
    # Two branches depending on [4.5] - [8.0]:
    # If [4.5] - [8.0] > 1:
    #     [3.6] - [5.8] < 1.4 - (1.2/2) × ([4.5] - [8.0] - 1)
    #     RHS uncertainty: 0.6 × sigma_45_80
    # Else:
    #     [3.6] - [5.8] < 1.4
    #     RHS uncertainty: 0 (constant)
    
    # Compute RHS and its uncertainty
    mask_red = (color_45_80 > 1.0).values
    threshold_1_5 = np.where(
        mask_red,
        1.4 - 0.6 * (color_45_80 - 1.0),  # Red branch
        1.4                                 # Blue branch
    )
    sigma_rhs_1_5 = np.where(
        mask_red,
        0.6 * sigma_45_80,  # Red branch has uncertainty
        0.0                  # Blue branch is constant
    )
    
    # Combined uncertainty
    sigma_1_5 = np.sqrt(sigma_36_58**2 + sigma_rhs_1_5**2)
    
    # Score: want LESS THAN threshold
    score_1_5 = sigmoid(-(color_36_58 - threshold_1_5) / sigma_1_5)
    
    # -------------------------------------------------------------------------
    # Aggregate scores (v1.1)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    score_array = np.column_stack([
        score_1_1,
        score_1_2,
        score_1_3,
        score_1_4,
        score_1_5
    ])
    
    # DIAGNOSTIC: Check score_array contents
    if debug_idx is not None:
        print(f"\nDIAGNOSTIC - score_array for debug source:")
        print(f"  score_array shape: {score_array.shape}")
        print(f"  score_array[{debug_idx}]: {score_array[debug_idx]}")
        print(f"  Individual scores at [{debug_idx}]:")
        print(f"    score_1_1[{debug_idx}]: {score_1_1[debug_idx]}")
        print(f"    score_1_2[{debug_idx}]: {score_1_2[debug_idx]}")
        print(f"    score_1_3[{debug_idx}]: {score_1_3[debug_idx]}")
        print(f"    score_1_4[{debug_idx}]: {score_1_4[debug_idx]}")
        print(f"    score_1_5[{debug_idx}]: {score_1_5[debug_idx]}")
    
    final_scores = aggregate_criterion_scores_v1_1(score_array)
    
    # Note: v1.1 aggregation uses cumulative contributions starting at 0.5.
    # This preserves evidence direction even with partial photometry:
    # - If 1/5 criteria evaluable and passes: 0.5 + (1.0-0.5)/5 = 0.6 (evidence FOR)
    # - Missing criteria don't contribute (neutral)
    # - Returns NaN only if zero criteria were evaluable
    
    # -------------------------------------------------------------------------
    # Debug output
    # -------------------------------------------------------------------------
    if debug_idx is not None:
        idx = debug_idx
        print(f"\n{'='*70}")
        print(f"DEBUG: PAH Score for source {idx}")
        print(f"{'='*70}")
        print(f"Magnitudes:")
        print(f"  [3.6] = {mags['3_6'].iloc[idx]:.3f}")
        print(f"  [4.5] = {mags['4_5'].iloc[idx]:.3f}")
        print(f"  [5.8] = {mags['5_8'].iloc[idx]:.3f}")
        print(f"  [8.0] = {mags['8_0'].iloc[idx]:.3f}")
        print(f"\nColors:")
        print(f"  [4.5] - [5.8] = {color_45_58.iloc[idx]:.3f}")
        print(f"  [5.8] - [8.0] = {color_58_80.iloc[idx]:.3f}")
        print(f"  [4.5] - [8.0] = {color_45_80.iloc[idx]:.3f}")
        print(f"  [3.6] - [5.8] = {color_36_58.iloc[idx]:.3f}")
        print(f"\nCriterion Scores:")
        print(f"  1.1: {score_1_1[idx]:.3f}  ([4.5]-[5.8] < 1.05×([5.8]-[8.0]-1))")
        print(f"  1.2: {score_1_2[idx]:.3f}  ([4.5]-[5.8] < 1.05)")
        print(f"  1.3: {score_1_3[idx]:.3f}  ([5.8]-[8.0] > 1)")
        print(f"  1.4: {score_1_4[idx]:.3f}  ([4.5] > 11.5)")
        print(f"  1.5: {score_1_5[idx]:.3f}  ([3.6]-[5.8] conditional)")
        n_valid = np.sum(~np.isnan(score_array[idx]))
        print(f"\nAggregation (v1.1 - cumulative contributions):")
        print(f"  Valid criteria: {n_valid}/5")
        # Show contributions
        contributions = []
        for i in range(5):
            if not np.isnan(score_array[idx, i]):
                contrib = (score_array[idx, i] - 0.5) / 5
                contributions.append(f"{contrib:+.3f}")
            else:
                contributions.append("  NaN")
        print(f"  Contributions: [{', '.join(contributions)}]")
        sum_contrib = sum([float(c) for c in contributions if c != "  NaN"])
        print(f"  Sum: {sum_contrib:+.3f}")
        print(f"  Final score (0.5 + sum): {final_scores[idx]:.3f}")
        print(f"{'='*70}\n")
    
    return pd.Series(final_scores, index=fnu_df.index, name='PAH_score')


# ==============================================================================
# PHASE 1: AGN SCORING
# ==============================================================================

def compute_agn_scores_batch(fnu_df: pd.DataFrame,
                             sigma_fnu_df: pd.DataFrame,
                             origin_df: pd.DataFrame,
                             debug_idx: Optional[int] = None) -> pd.Series:
    """
    Compute AGN scores for all sources (Phase 1 criteria).
    
    Updated to Framework v1.1:
    - Uses measurement-specific uncertainties in sigmoid
    - Uses cumulative contribution aggregation for all 6 criteria
    - Treats all criteria equally (no product/max logic)
    
    AGN (Active Galactic Nuclei) are identified by power-law continuum emission
    that produces specific color-magnitude signatures.
    
    Six criteria evaluated (Gutermuth et al. 2009):
        1.6:  [4.5] - [8.0] < 0.5
        1.7:  [4.5] > 13.5 + ([4.5] - [8.0] - 2.3)/0.4
        1.8:  [4.5] > 13.5
        1.9:  [4.5] > 14 + ([4.5] - [8.0] - 0.5)
        1.10: [4.5] > 14.5 - ([4.5] - [8.0] - 1.2)/0.3
        1.11: [4.5] > 14.5
    
    Parameters
    ----------
    fnu_df : pd.DataFrame
        Flux table (mJy)
    sigma_fnu_df : pd.DataFrame
        Flux uncertainty table (mJy)
    origin_df : pd.DataFrame
        Origin flag table (1=valid)
    debug_idx : int, optional
        Print debug info for this source
    
    Returns
    -------
    scores : pd.Series
        AGN scores in [0, 1] with 0.5 = neutral, or NaN if insufficient photometry
    
    Notes
    -----
    Required bands: [4.5], [8.0]
    """
    n_sources = len(fnu_df)
    
    # Convert fluxes to magnitudes
    possible_bands = ['4_5', '8_0']
    mags = flux_to_mag_batch(fnu_df, origin_df, bands=possible_bands)
    
    # Convert flux uncertainties to magnitude uncertainties (v1.1)
    mag_errs = flux_to_mag_err_batch(fnu_df, sigma_fnu_df, origin_df, bands=possible_bands)
    
    # Ensure bands exist (NaN if missing)
    for band in possible_bands:
        if band not in mags.columns:
            mags[band] = np.nan
        if band not in mag_errs.columns:
            mag_errs[band] = np.nan
    
    # Compute colors and magnitudes
    color_45_80 = mags['4_5'] - mags['8_0']
    mag_45 = mags['4_5']
    
    # Compute measurement uncertainties (v1.1)
    sigma_45_80 = compute_color_uncertainty(mag_errs['4_5'], mag_errs['8_0'])
    sigma_45 = mag_errs['4_5']
    
    # -------------------------------------------------------------------------
    # Criterion 1.6: [4.5] - [8.0] < 0.5
    # -------------------------------------------------------------------------
    sigma_1_6 = sigma_45_80
    score_1_6 = sigmoid(-(color_45_80 - 0.5) / sigma_1_6)
    
    # -------------------------------------------------------------------------
    # Criterion 1.7: [4.5] > 13.5 + ([4.5] - [8.0] - 2.3)/0.4
    # -------------------------------------------------------------------------
    threshold_1_7 = 13.5 + (color_45_80 - 2.3) / 0.4
    sigma_threshold_1_7 = sigma_45_80 / 0.4
    sigma_1_7 = np.sqrt(sigma_45**2 + sigma_threshold_1_7**2)
    score_1_7 = sigmoid((mag_45 - threshold_1_7) / sigma_1_7)
    
    # -------------------------------------------------------------------------
    # Criterion 1.8: [4.5] > 13.5
    # -------------------------------------------------------------------------
    sigma_1_8 = sigma_45
    score_1_8 = sigmoid((mag_45 - 13.5) / sigma_1_8)
    
    # -------------------------------------------------------------------------
    # Criterion 1.9: [4.5] > 14 + ([4.5] - [8.0] - 0.5)
    # -------------------------------------------------------------------------
    threshold_1_9 = 14.0 + (color_45_80 - 0.5)
    sigma_threshold_1_9 = sigma_45_80
    sigma_1_9 = np.sqrt(sigma_45**2 + sigma_threshold_1_9**2)
    score_1_9 = sigmoid((mag_45 - threshold_1_9) / sigma_1_9)
    
    # -------------------------------------------------------------------------
    # Criterion 1.10: [4.5] > 14.5 - ([4.5] - [8.0] - 1.2)/0.3
    # -------------------------------------------------------------------------
    threshold_1_10 = 14.5 - (color_45_80 - 1.2) / 0.3
    sigma_threshold_1_10 = sigma_45_80 / 0.3
    sigma_1_10 = np.sqrt(sigma_45**2 + sigma_threshold_1_10**2)
    score_1_10 = sigmoid((mag_45 - threshold_1_10) / sigma_1_10)
    
    # -------------------------------------------------------------------------
    # Criterion 1.11: [4.5] > 14.5
    # -------------------------------------------------------------------------
    sigma_1_11 = sigma_45
    score_1_11 = sigmoid((mag_45 - 14.5) / sigma_1_11)
    
    # -------------------------------------------------------------------------
    # Aggregate scores (v1.1)
    # -------------------------------------------------------------------------
    score_array = np.column_stack([
        score_1_6,
        score_1_7,
        score_1_8,
        score_1_9,
        score_1_10,
        score_1_11
    ])
    
    final_scores = aggregate_criterion_scores_v1_1(score_array)
    
    # Note: v1.1 aggregation treats all 6 criteria equally using cumulative contributions.
    # This replaces the old product/max logic which enforced binary "all/none" constraints.
    
    # -------------------------------------------------------------------------
    # Debug output
    # -------------------------------------------------------------------------
    if debug_idx is not None:
        idx = debug_idx
        print(f"\n{'='*70}")
        print(f"DEBUG: AGN Score for source {idx}")
        print(f"{'='*70}")
        print(f"Magnitudes:")
        print(f"  [4.5] = {mag_45.iloc[idx]:.3f}")
        print(f"  [8.0] = {mags['8_0'].iloc[idx]:.3f}")
        print(f"\nColor:")
        print(f"  [4.5] - [8.0] = {color_45_80.iloc[idx]:.3f}")
        print(f"\nCriterion Scores:")
        print(f"  1.6:  {score_1_6[idx]:.3f}  ([4.5]-[8.0] < 0.5)")
        print(f"  1.7:  {score_1_7[idx]:.3f}  ([4.5] > 13.5 + ...)")
        print(f"  1.8:  {score_1_8[idx]:.3f}  ([4.5] > 13.5)")
        print(f"  1.9:  {score_1_9[idx]:.3f}  ([4.5] > 14 + ...)")
        print(f"  1.10: {score_1_10[idx]:.3f}  ([4.5] > 14.5 - ...)")
        print(f"  1.11: {score_1_11[idx]:.3f}  ([4.5] > 14.5)")
        
        n_valid = np.sum(~np.isnan(score_array[idx]))
        print(f"\nAggregation (v1.1 - cumulative contributions):")
        print(f"  Valid criteria: {n_valid}/6")
        
        # Show contributions
        contributions = []
        for i in range(6):
            if not np.isnan(score_array[idx, i]):
                contrib = (score_array[idx, i] - 0.5) / 6
                contributions.append(f"{contrib:+.3f}")
            else:
                contributions.append("  NaN")
        print(f"  Contributions: [{', '.join(contributions)}]")
        sum_contrib = sum([float(c) for c in contributions if c != "  NaN"])
        print(f"  Sum: {sum_contrib:+.3f}")
        print(f"  Final score (0.5 + sum): {final_scores[idx]:.3f}")
        print(f"{'='*70}\n")
    
    return pd.Series(final_scores, index=fnu_df.index, name='AGN_score')


# ==============================================================================
# PHASE 2: CLASS I YSO SCORING
# ==============================================================================

def compute_class_i_scores_batch(fnu_df: pd.DataFrame,
                                  sigma_fnu_df: pd.DataFrame,
                                  origin_df: pd.DataFrame,
                                  debug_idx: Optional[int] = None) -> pd.Series:
    """
    Compute Class I YSO scores for all sources (Phase 2A criteria).
    
    Updated to Framework v1.1:
    - Uses measurement-specific uncertainties in sigmoid
    - Uses cumulative contribution aggregation for all 4 criteria
    - Properly handles Gutermuth's uncertainty-adjusted thresholds
    
    Class I YSOs are young stellar objects with significant infrared excess from
    circumstellar disks/envelopes. Identified by IR colors in IRAC bands.
    
    Four criteria evaluated (Gutermuth et al. 2009 Phase 2A):
        2.1: [4.5] - [8.0] - σ > 0.5
        2.2: [3.6] - [5.8] - σ > 0.35
        2.3: [3.6] - [5.8] + σ < 3.5×([4.5] - [8.0] - σ) + 0.5  [IF [4.5]-[8.0] ≤ 0.5]
        2.4: [3.6] - [4.5] - σ > 0.15
    
    Parameters
    ----------
    fnu_df : pd.DataFrame
        Flux table (mJy)
    sigma_fnu_df : pd.DataFrame
        Flux uncertainty table (mJy)
    origin_df : pd.DataFrame
        Origin flag table (1=valid)
    debug_idx : int, optional
        Print debug info for this source
    
    Returns
    -------
    scores : pd.Series
        Class I scores in [0, 1] with 0.5 = neutral, or NaN if insufficient photometry
    
    Notes
    -----
    Required bands: [3.6], [4.5], [5.8], [8.0]
    
    Gutermuth's σ terms represent conservative statistical requirements.
    We interpret "color - σ > threshold" as "color > threshold + σ",
    which maps Gutermuth's boundary to our neutral point (score = 0.5).
    """
    n_sources = len(fnu_df)
    
    # Convert fluxes to magnitudes
    possible_bands = ['3_6', '4_5', '5_8', '8_0']
    mags = flux_to_mag_batch(fnu_df, origin_df, bands=possible_bands)
    
    # Convert flux uncertainties to magnitude uncertainties (v1.1)
    mag_errs = flux_to_mag_err_batch(fnu_df, sigma_fnu_df, origin_df, bands=possible_bands)
    
    # Ensure bands exist (NaN if missing)
    for band in possible_bands:
        if band not in mags.columns:
            mags[band] = np.nan
        if band not in mag_errs.columns:
            mag_errs[band] = np.nan
    
    # Compute colors
    color_45_80 = mags['4_5'] - mags['8_0']
    color_36_58 = mags['3_6'] - mags['5_8']
    color_36_45 = mags['3_6'] - mags['4_5']
    
    # Compute color uncertainties (v1.1)
    sigma_45_80 = compute_color_uncertainty(mag_errs['4_5'], mag_errs['8_0'])
    sigma_36_58 = compute_color_uncertainty(mag_errs['3_6'], mag_errs['5_8'])
    sigma_36_45 = compute_color_uncertainty(mag_errs['3_6'], mag_errs['4_5'])
    
    # -------------------------------------------------------------------------
    # Criterion 2.1: [4.5] - [8.0] - σ > 0.5
    # -------------------------------------------------------------------------
    # Gutermuth's "color - σ > threshold" means "color > threshold + σ"
    # This ensures sources at Gutermuth's boundary get score = 0.5 (neutral)
    threshold_2_1 = 0.5 + sigma_45_80
    score_2_1 = sigmoid((color_45_80 - threshold_2_1) / sigma_45_80)
    
    # -------------------------------------------------------------------------
    # Criterion 2.2: [3.6] - [5.8] - σ > 0.35
    # -------------------------------------------------------------------------
    threshold_2_2 = 0.35 + sigma_36_58
    score_2_2 = sigmoid((color_36_58 - threshold_2_2) / sigma_36_58)
    
    # -------------------------------------------------------------------------
    # Criterion 2.3: [3.6] - [5.8] + σ < 3.5×([4.5] - [8.0] - σ) + 0.5
    # -------------------------------------------------------------------------
    # Only evaluated if [4.5] - [8.0] <= 0.5
    # This criterion has σ terms on both sides - evaluate as written
    
    condition_2_3 = color_45_80 <= 0.5
    
    # LHS: [3.6] - [5.8] + σ (deterministic addition of uncertainty)
    lhs_2_3 = color_36_58 + sigma_36_58
    
    # RHS: 3.5 × ([4.5] - [8.0] - σ) + 0.5
    rhs_2_3 = 3.5 * (color_45_80 - sigma_45_80) + 0.5
    
    # Want LHS < RHS (difference < 0)
    difference_2_3 = lhs_2_3 - rhs_2_3
    
    # Uncertainty of difference through error propagation:
    # LHS uncertainty: σ([3.6] - [5.8]) = sigma_36_58
    # RHS uncertainty: σ(3.5 × [4.5] - 3.5 × [8.0]) = 3.5 × sigma_45_80
    # Combined: sqrt(sigma_36_58² + (3.5 × sigma_45_80)²)
    sigma_diff_2_3 = np.sqrt(sigma_36_58**2 + (3.5 * sigma_45_80)**2)
    
    # Want difference < 0, so negate for sigmoid
    score_2_3_value = sigmoid(-difference_2_3 / sigma_diff_2_3)
    
    # Apply conditional: only evaluate if color_45_80 <= 0.5, else NaN
    score_2_3 = np.where(condition_2_3, score_2_3_value, np.nan)
    
    # -------------------------------------------------------------------------
    # Criterion 2.4: [3.6] - [4.5] - σ > 0.15
    # -------------------------------------------------------------------------
    # Distinguishes Class I from Class II
    threshold_2_4 = 0.15 + sigma_36_45
    score_2_4 = sigmoid((color_36_45 - threshold_2_4) / sigma_36_45)
    
    # -------------------------------------------------------------------------
    # Aggregate scores (v1.1)
    # -------------------------------------------------------------------------
    score_array = np.column_stack([
        score_2_1,
        score_2_2,
        score_2_3,
        score_2_4
    ])
    
    final_scores = aggregate_criterion_scores_v1_1(score_array)
    
    # Note: v1.1 aggregation treats all 4 criteria equally using cumulative contributions.
    # Criterion 2.3 is conditional and will be NaN if [4.5]-[8.0] > 0.5.
    
    # -------------------------------------------------------------------------
    # Debug output
    # -------------------------------------------------------------------------
    if debug_idx is not None:
        idx = debug_idx
        print(f"\n{'='*70}")
        print(f"DEBUG: Class I Score for source {idx}")
        print(f"{'='*70}")
        print(f"Magnitudes:")
        print(f"  [3.6] = {mags['3_6'].iloc[idx]:.3f}")
        print(f"  [4.5] = {mags['4_5'].iloc[idx]:.3f}")
        print(f"  [5.8] = {mags['5_8'].iloc[idx]:.3f}")
        print(f"  [8.0] = {mags['8_0'].iloc[idx]:.3f}")
        print(f"\nColors:")
        print(f"  [4.5] - [8.0] = {color_45_80.iloc[idx]:.3f} ± {sigma_45_80.iloc[idx]:.3f}")
        print(f"  [3.6] - [5.8] = {color_36_58.iloc[idx]:.3f} ± {sigma_36_58.iloc[idx]:.3f}")
        print(f"  [3.6] - [4.5] = {color_36_45.iloc[idx]:.3f} ± {sigma_36_45.iloc[idx]:.3f}")
        print(f"\nCriterion Scores:")
        print(f"  2.1: {score_2_1[idx]:.3f}  ([4.5]-[8.0] - σ > 0.5)")
        print(f"      Threshold: {threshold_2_1.iloc[idx]:.3f}")
        print(f"  2.2: {score_2_2[idx]:.3f}  ([3.6]-[5.8] - σ > 0.35)")
        print(f"      Threshold: {threshold_2_2.iloc[idx]:.3f}")
        print(f"  2.3: {score_2_3[idx]:.3f}  (contamination filter, conditional)")
        if not np.isnan(score_2_3[idx]):
            print(f"      LHS: {lhs_2_3.iloc[idx]:.3f}, RHS: {rhs_2_3.iloc[idx]:.3f}")
        print(f"  2.4: {score_2_4[idx]:.3f}  ([3.6]-[4.5] - σ > 0.15)")
        print(f"      Threshold: {threshold_2_4.iloc[idx]:.3f}")
        
        n_valid = np.sum(~np.isnan(score_array[idx]))
        print(f"\nAggregation (v1.1 - cumulative contributions):")
        print(f"  Valid criteria: {n_valid}/4")
        
        # Show contributions
        contributions = []
        for i in range(4):
            if not np.isnan(score_array[idx, i]):
                contrib = (score_array[idx, i] - 0.5) / 4
                contributions.append(f"{contrib:+.3f}")
            else:
                contributions.append("  NaN")
        print(f"  Contributions: [{', '.join(contributions)}]")
        sum_contrib = sum([float(c) for c in contributions if c != "  NaN"])
        print(f"  Sum: {sum_contrib:+.3f}")
        print(f"  Final score (0.5 + sum): {final_scores[idx]:.3f}")
        print(f"{'='*70}\n")
    
    return pd.Series(final_scores, index=fnu_df.index, name='ClassI_score')


# ==============================================================================
# MAIN EXECUTION (for testing)
# ==============================================================================

if __name__ == "__main__":
    print("Gutermuth Scoring Functions - Test Mode")
    print("="*70)
    print("\nThis module provides vectorized scoring functions.")
    print("Import and use with your HDF5 data:")
    print("\n  from gutermuth_scoring import compute_pah_scores_batch")
    print("  scores = compute_pah_scores_batch(fnu, sigma_fnu, origin)")
    print("\nAvailable functions:")
    print("  - compute_pah_scores_batch()      [Phase 1: PAH galaxies]")
    print("  - compute_agn_scores_batch()      [Phase 1: AGN]")
    print("  - compute_class_i_scores_batch()  [Phase 2: Class I YSOs]")
    print("  - (more to come: Class II, Transition Disk, etc.)")