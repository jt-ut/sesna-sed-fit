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


def aggregate_criterion_scores(score_array: np.ndarray,
                                apply_completeness_penalty: bool = True) -> np.ndarray:
    """
    Aggregate multiple criterion scores into a single class score.
    
    Strategy:
    1. Compute mean of all non-NaN criterion scores
    2. Apply completeness penalty: score × sqrt(fraction_evaluated)
    3. Return NaN if no criteria were evaluable
    
    Parameters
    ----------
    score_array : np.ndarray
        Array of shape (N_sources, N_criteria) with criterion scores
    apply_completeness_penalty : bool, default=True
        If True, penalize sources with fewer evaluable criteria
    
    Returns
    -------
    final_scores : np.ndarray
        Aggregated scores (length N_sources)
    """
    # Count non-NaN scores per source
    n_valid = np.sum(~np.isnan(score_array), axis=1)
    n_total = score_array.shape[1]
    
    # Mean of valid scores
    avg_scores = np.nanmean(score_array, axis=1)
    
    if apply_completeness_penalty:
        # Penalize incomplete evaluations
        completeness = n_valid / n_total
        final_scores = avg_scores * np.sqrt(completeness)
    else:
        final_scores = avg_scores
    
    # Set to NaN where no criteria were evaluable
    final_scores = np.where(n_valid == 0, np.nan, final_scores)
    
    return final_scores


# ==============================================================================
# PHASE 1: PAH GALAXY SCORING
# ==============================================================================

def compute_pah_scores_batch(fnu_df: pd.DataFrame,
                             sigma_fnu_df: pd.DataFrame,
                             origin_df: pd.DataFrame,
                             scale: float = 0.15,
                             apply_completeness_penalty: bool = True,
                             debug_idx: Optional[int] = None) -> pd.Series:
    """
    Compute PAH galaxy scores for all sources (Phase 1 criteria).
    
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
    scale : float, default=0.15
        Sigmoid scale parameter (in magnitudes). Controls boundary softness.
        Typical: 0.1-0.2 for photometric uncertainties in IRAC bands.
    apply_completeness_penalty : bool, default=True
        If True, penalize sources with fewer evaluable criteria via sqrt(completeness)
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
    
    # Convert fluxes to magnitudes
    required_bands = ['3_6', '4_5', '5_8', '8_0']
    mags = flux_to_mag_batch(fnu_df, origin_df, bands=required_bands)
    
    # Compute colors
    color_45_58 = mags['4_5'] - mags['5_8']  # [4.5] - [5.8]
    color_58_80 = mags['5_8'] - mags['8_0']  # [5.8] - [8.0]
    color_45_80 = mags['4_5'] - mags['8_0']  # [4.5] - [8.0]
    color_36_58 = mags['3_6'] - mags['5_8']  # [3.6] - [5.8]
    
    # -------------------------------------------------------------------------
    # Criterion 1.1: [4.5] - [5.8] < 1.05 × ([5.8] - [8.0] - 1)
    # -------------------------------------------------------------------------
    threshold_1_1 = 1.05 * (color_58_80 - 1.0)
    # Want [4.5]-[5.8] to be LESS than threshold, so:
    # distance = -(color - threshold) / scale
    # Positive distance → passes criterion
    score_1_1 = sigmoid(-(color_45_58 - threshold_1_1) / scale)
    
    # -------------------------------------------------------------------------
    # Criterion 1.2: [4.5] - [5.8] < 1.05
    # -------------------------------------------------------------------------
    score_1_2 = sigmoid(-(color_45_58 - 1.05) / scale)
    
    # -------------------------------------------------------------------------
    # Criterion 1.3: [5.8] - [8.0] > 1
    # -------------------------------------------------------------------------
    score_1_3 = sigmoid((color_58_80 - 1.0) / scale)
    
    # -------------------------------------------------------------------------
    # Criterion 1.4: [4.5] > 11.5 (faint sources)
    # -------------------------------------------------------------------------
    # Want magnitude to be GREATER than 11.5 (fainter)
    # distance = (mag - threshold) is positive when mag > threshold (fainter)
    score_1_4 = sigmoid((mags['4_5'] - 11.5) / scale)
    
    # -------------------------------------------------------------------------
    # Criterion 1.5: [3.6] - [5.8] constraint (conditional)
    # -------------------------------------------------------------------------
    # Two branches depending on [4.5] - [8.0]:
    # If [4.5] - [8.0] > 1:
    #     [3.6] - [5.8] < 1.4 - (1.2/2) × ([4.5] - [8.0] - 1)
    # Else:
    #     [3.6] - [5.8] < 1.4
    
    mask_red = (color_45_80 > 1.0).values
    
    # Branch 1: Red sources ([4.5] - [8.0] > 1)
    threshold_1_5_red = 1.4 - (1.2/2.0) * (color_45_80 - 1.0)
    score_1_5_red = sigmoid(-(color_36_58 - threshold_1_5_red) / scale)
    
    # Branch 2: Blue sources ([4.5] - [8.0] <= 1)
    score_1_5_blue = sigmoid(-(color_36_58 - 1.4) / scale)
    
    # Select appropriate branch
    score_1_5 = np.where(mask_red, score_1_5_red, score_1_5_blue)
    
    # -------------------------------------------------------------------------
    # Aggregate scores
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
    
    final_scores = aggregate_criterion_scores(
        score_array,
        apply_completeness_penalty=apply_completeness_penalty
    )
    
    # Note: We do NOT require all bands to be present. The aggregation function
    # handles partial photometry correctly by:
    # 1. Computing mean of available (non-NaN) criterion scores
    # 2. Applying completeness penalty: score × sqrt(n_evaluable / n_total)
    # 3. Returning NaN only if zero criteria were evaluable
    # This allows scoring with partial IRAC coverage.
    
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
        print(f"\nAggregation:")
        print(f"  Valid criteria: {n_valid}/5")
        print(f"  Mean score: {np.nanmean(score_array[idx]):.3f}")
        print(f"  Completeness: {n_valid/5:.3f}")
        print(f"  Final score: {final_scores[idx]:.3f}")
        print(f"{'='*70}\n")
    
    return pd.Series(final_scores, index=fnu_df.index, name='PAH_score')


# ==============================================================================
# PHASE 1: AGN SCORING
# ==============================================================================

def compute_agn_scores_batch(fnu_df: pd.DataFrame,
                             sigma_fnu_df: pd.DataFrame,
                             origin_df: pd.DataFrame,
                             scale: float = 0.15,
                             apply_completeness_penalty: bool = True,
                             debug_idx: Optional[int] = None) -> pd.Series:
    """
    Compute AGN scores for all sources (Phase 1 criteria).
    
    AGN (Active Galactic Nuclei) are identified by power-law continuum emission
    that produces specific color-magnitude signatures.
    
    Two groups of criteria:
    Group A (all must be satisfied):
        1.6: [4.5] - [8.0] < 0.5
        1.7: [4.5] > 13.5 + ([4.5] - [8.0] - 2.3)/0.4
        1.8: [4.5] > 13.5
    
    Group B (any one satisfied):
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
    scale : float, default=0.15
        Sigmoid scale parameter
    apply_completeness_penalty : bool, default=True
        Apply sqrt(completeness) penalty
    debug_idx : int, optional
        Print debug info for this source
    
    Returns
    -------
    scores : pd.Series
        AGN scores in [0, 1], or NaN if insufficient photometry
    
    Notes
    -----
    Required bands: [4.5], [8.0]
    
    Scoring logic:
    - Group A score: product of all three criteria (must all pass)
    - Group B score: max of three criteria (any one can pass)
    - Final score: max(Group A, Group B)
    """
    n_sources = len(fnu_df)
    
    # Convert fluxes to magnitudes
    required_bands = ['4_5', '8_0']
    mags = flux_to_mag_batch(fnu_df, origin_df, bands=required_bands)
    
    # Compute color
    color_45_80 = mags['4_5'] - mags['8_0']
    mag_45 = mags['4_5']
    
    # -------------------------------------------------------------------------
    # Group A: All three criteria must be satisfied
    # -------------------------------------------------------------------------
    
    # Criterion 1.6: [4.5] - [8.0] < 0.5
    score_1_6 = sigmoid(-(color_45_80 - 0.5) / scale)
    
    # Criterion 1.7: [4.5] > 13.5 + ([4.5] - [8.0] - 2.3)/0.4
    # Want FAINT sources (mag > threshold)
    threshold_1_7 = 13.5 + (color_45_80 - 2.3) / 0.4
    score_1_7 = sigmoid((mag_45 - threshold_1_7) / scale)
    
    # Criterion 1.8: [4.5] > 13.5
    # Want FAINT sources (mag > threshold)
    score_1_8 = sigmoid((mag_45 - 13.5) / scale)
    
    # Group A: Product (all must pass)
    score_group_A = score_1_6 * score_1_7 * score_1_8
    
    # -------------------------------------------------------------------------
    # Group B: Any one criterion can be satisfied
    # -------------------------------------------------------------------------
    
    # Criterion 1.9: [4.5] > 14 + ([4.5] - [8.0] - 0.5)
    # Want FAINT sources (mag > threshold)
    threshold_1_9 = 14.0 + (color_45_80 - 0.5)
    score_1_9 = sigmoid((mag_45 - threshold_1_9) / scale)
    
    # Criterion 1.10: [4.5] > 14.5 - ([4.5] - [8.0] - 1.2)/0.3
    # Want FAINT sources (mag > threshold)
    threshold_1_10 = 14.5 - (color_45_80 - 1.2) / 0.3
    score_1_10 = sigmoid((mag_45 - threshold_1_10) / scale)
    
    # Criterion 1.11: [4.5] > 14.5
    # Want FAINT sources (mag > threshold)
    score_1_11 = sigmoid((mag_45 - 14.5) / scale)
    
    # Group B: Max (any one can pass)
    score_group_B = np.maximum.reduce([score_1_9, score_1_10, score_1_11])
    
    # -------------------------------------------------------------------------
    # Final score: max of two groups
    # -------------------------------------------------------------------------
    final_scores = np.maximum(score_group_A, score_group_B)
    
    # Note: AGN scoring computes two independent scores (Group A and Group B)
    # and takes the maximum. If photometry is missing, both groups will be NaN
    # and the final score will naturally be NaN. No explicit check needed.
    
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
        print(f"\nGroup A Scores (all must pass):")
        print(f"  1.6: {score_1_6[idx]:.3f}  ([4.5]-[8.0] < 0.5)")
        print(f"  1.7: {score_1_7[idx]:.3f}  ([4.5] > 13.5 + ...)")
        print(f"  1.8: {score_1_8[idx]:.3f}  ([4.5] > 13.5)")
        print(f"  Group A product: {score_group_A[idx]:.3f}")
        print(f"\nGroup B Scores (any one can pass):")
        print(f"  1.9:  {score_1_9[idx]:.3f}  ([4.5] > 14 + ...)")
        print(f"  1.10: {score_1_10[idx]:.3f}  ([4.5] > 14.5 - ...)")
        print(f"  1.11: {score_1_11[idx]:.3f}  ([4.5] > 14.5)")
        print(f"  Group B max: {score_group_B[idx]:.3f}")
        print(f"\nFinal Score: {final_scores[idx]:.3f}")
        print(f"{'='*70}\n")
    
    return pd.Series(final_scores, index=fnu_df.index, name='AGN_score')


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
    print("  - compute_pah_scores_batch()")
    print("  - compute_agn_scores_batch()")
    print("  - (more to come: Class I, Class II, etc.)")