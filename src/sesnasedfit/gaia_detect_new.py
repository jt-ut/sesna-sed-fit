"""
Gaia Detection Probability Analysis

Functions for computing Gaia detection probabilities based on SED fitting results.
This module applies distance scaling and extinction to model SEDs, convolves them 
with the Gaia G-band filter, and determines detection probabilities.

The Gaia G-band filter (GaiaDR3) is pre-loaded as GAIA_G_FILTER for convenience.

Key Functions
-------------
- compute_detection_prob: Compute detection probability for a single source
- compute_detection_probs_batch: Batch process all sources
- _compute_detection_prob_single: Helper for parallel processing

Pre-loaded Data
---------------
- GAIA_G_FILTER: Gaia G-band filter (wavelength, transmission, zeropoint)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .sed_utils import scale_sed_distance_extinction, convolve_sed_with_filter, flux_to_magnitude


## ======== Load Gaia's G-Band filter into a module constant ========
def _load_gaia_g_filter(passband_file, zeropoint_file, system='VEGAMAG'):
    """
    Load Gaia G-band filter transmission curve and zero point.
    
    Parameters
    ----------
    passband_file : str
        Path to passband.dat file
    zeropoint_file : str
        Path to zeropt.dat file
    system : str, optional
        Photometric system: 'VEGAMAG' or 'AB' (default: 'VEGAMAG')
    
    Returns
    -------
    gaia_filter : dict
        Dictionary with keys:
        - 'wavelength': Wavelength in microns (array)
        - 'transmission': G-band transmission curve (array)
        - 'zeropoint': Zero point magnitude (scalar)
        - 'system': Photometric system used (str)
    
    Examples
    --------
    gaia_filter = load_gaia_g_filter('passband.dat', 'zeropt.dat')
    # Access components:
    gaia_filter['wavelength']
    gaia_filter['transmission']
    gaia_filter['zeropoint']
    """
    # Load passband data
    # Columns: lambda(nm), GPb, e_GPb, BPPb, e_BPPb, RPPb, e_RPPb
    data = np.loadtxt(passband_file)
    wavelength_nm = data[:, 0]
    g_transmission = data[:, 1]
    
    # Filter out undefined values (99.99)
    valid = g_transmission < 90.0
    wavelength_nm = wavelength_nm[valid]
    g_transmission = g_transmission[valid]
    
    # Convert wavelength to microns
    wavelength_micron = wavelength_nm / 1000.0
    
    # Load zero point
    with open(zeropoint_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if system in line:
                parts = line.split()
                zeropoint = float(parts[0])  # G band zero point
                break
    
    return {
        'wavelength': wavelength_micron,
        'transmission': g_transmission,
        'zeropoint': zeropoint,
        'system': system
    }


# Pre-load Gaia G-band filter data (GaiaDR3)
_GAIA_FILTER_DIR = Path(__file__).parent / 'data' / 'GaiaEDR3_passbands_zeropoints_version2'
GAIA_G_FILTER = _load_gaia_g_filter(_GAIA_FILTER_DIR / 'passband.dat', _GAIA_FILTER_DIR / 'zeropt.dat', system='AB')
# Clean up temporary variables
del _GAIA_FILTER_DIR

def compute_detection_prob(model_fluxes, wavelengths, av_law, fitresult, 
                           gaia_filter=GAIA_G_FILTER, is_galaxy=False, 
                           n_samples=5, return_per_model=False):
    """
    Compute Gaia detection probability for model(s) given fit results for a source.
    
    Uses chi2-weighted sampling of (sc_best, av_best) pairs from fit results,
    scales ALL models simultaneously, computes G-band magnitudes, and determines detection rate.
    
    Parameters
    ----------
    model_fluxes : np.ndarray
        Model SED fluxes in LINEAR units (mJy)
        Shape: (N_models, N_wavelengths) or (N_wavelengths,) for single model
    wavelengths : np.ndarray
        Wavelength grid in microns
    av_law : np.ndarray
        Extinction law (A_lambda / A_V) for each wavelength
    fitresult : pandas.DataFrame
        Fit results for one source (top K fits for this source)
        Must contain columns: 'sc_best', 'av_best', 'chi2'
        For galaxy models, must also contain: 'imp_flux_H'
    gaia_filter : dict or tuple, optional
        Gaia filter data (default: GAIA_G_FILTER). Can be:
        - dict with keys 'wavelength', 'transmission', 'zeropoint'
        - tuple of (wavelength, transmission, zeropoint)
    is_galaxy : bool, optional
        If True, multiply scaled flux by imp_flux_H from sampled fit (default: False)
    n_samples : int, optional
        Number of Monte Carlo samples (default: 5)
    return_per_model : bool, optional
        If True, return detection probability for each model (default: True)
        If False, return overall detection probability (fraction of models detected)
    
    Returns
    -------
    detection_prob : np.ndarray or float
        If return_per_model=True: Detection probability for each model, shape (N_models,)
        If return_per_model=False: Overall detection probability (scalar)
        For single model input, always returns scalar
    
    Notes
    -----
    Detection criterion: Gaia G < 20.7 mag (EDR3 limit)
    
    Algorithm:
    1. Sample (sc_best, av_best, [h_band]) from fit results using chi2 weights
    2. Apply these SAME parameters to ALL models simultaneously
    3. Convolve all scaled models with Gaia G filter
    4. Check which models would be detected (G < 20.7)
    5. Repeat n_samples times and compute detection fraction per model
    """
    # Handle single model case
    single_model = (model_fluxes.ndim == 1)
    if single_model:
        model_fluxes = model_fluxes.reshape(1, -1)
    
    n_models = model_fluxes.shape[0]
    
    # Extract fit parameters
    if isinstance(fitresult, dict):
        chi2 = np.array(fitresult['chi2'])
        sc_best_vals = np.array(fitresult['sc_best'])
        av_best_vals = np.array(fitresult['av_best'])
        if is_galaxy:
            h_band_vals = np.array(fitresult['imp_flux_H'])
    else:
        chi2 = fitresult['chi2'].values
        sc_best_vals = fitresult['sc_best'].values
        av_best_vals = fitresult['av_best'].values
        if is_galaxy:
            h_band_vals = fitresult['imp_flux_H'].values
    
    # Convert chi2 to sampling weights
    weights = np.exp(-chi2 / 2.0)
    weights /= weights.sum()
    
    # Parse gaia_filter (handle both dict and tuple formats)
    if isinstance(gaia_filter, dict):
        filter_wav = gaia_filter['wavelength']
        filter_trans = gaia_filter['transmission']
        zeropoint = gaia_filter['zeropoint']
    else:
        # Legacy tuple format: (wavelength, transmission, zeropoint)
        filter_wav, filter_trans, zeropoint = gaia_filter
    
    # Count detections per model across all samples
    detection_count = np.zeros(n_models)
    
    for _ in range(n_samples):
        # Sample ONE (sc_best, av_best, [h_band]) tuple for this iteration
        fit_idx = np.random.choice(len(chi2), p=weights)
        sc_best = sc_best_vals[fit_idx]
        av_best = av_best_vals[fit_idx]
        
        # Scale ALL models at once (vectorized operation)
        flux_scaled_all = scale_sed_distance_extinction(model_fluxes, sc_best, av_best, av_law)
        # Shape: (N_models, N_wavelengths)
        
        # If galaxy model: multiply ALL models by H-band flux from this fit
        if is_galaxy:
            h_band_flux = h_band_vals[fit_idx]
            flux_scaled_all = flux_scaled_all * h_band_flux
        
        # Compute G-band flux for ALL models at once
        flux_g_all = convolve_sed_with_filter(wavelengths, flux_scaled_all, filter_wav, filter_trans)
        # Shape: (N_models,)
        
        # Convert to magnitudes for all models
        mag_g_all = flux_to_magnitude(flux_g_all, zeropoint)
        # Shape: (N_models,)
        
        # Check detection for all models (boolean array)
        detected = (mag_g_all < 20.7)
        # Shape: (N_models,) boolean
        
        # Increment detection count
        detection_count += detected
    
    # Compute detection probabilities
    detection_probs = detection_count / n_samples
    
    # Return based on parameters
    if single_model:
        # Single model always returns scalar
        return detection_probs[0]
    elif return_per_model:
        # Return per-model probabilities
        return detection_probs
    else:
        # Return overall probability (mean across models)
        return detection_probs.mean()

def _compute_detection_prob_single(source_id, fitresult, model_fluxes, wavelengths,
                                   av_law, gaia_filter=GAIA_G_FILTER, is_galaxy=False, 
                                   n_samples=5):
    """
    Process a single source to compute detection probability.
    
    This is a helper function for parallel processing.
    
    Parameters
    ----------
    source_id : str or int
        Source identifier
    fitresult : pandas.DataFrame
        Fit results for one source (top K fits for this source)
    model_fluxes : np.ndarray
        Model SED fluxes, shape (N_models, N_wavelengths)
    wavelengths : np.ndarray
        Wavelength grid in microns
    av_law : np.ndarray
        Extinction law (A_lambda / A_V) for each wavelength
    gaia_filter : dict, optional
        Gaia filter data (default: GAIA_G_FILTER)
    is_galaxy : bool, optional
        If True, use H-band scaling for galaxy models (default: False)
    n_samples : int, optional
        Number of Monte Carlo samples (default: 5)
    
    Returns
    -------
    result : dict
        Dictionary with keys: 'id', 'pDetect', 'model_cat', 'n_fits'
        If computation fails, 'pDetect' will be NaN
    """
    # Get model category
    if 'model_cat' in fitresult.columns:
        model_cat = fitresult['model_cat'].iloc[0]
    else:
        model_cat = 'gal' if is_galaxy else 'yso'
    
    n_fits = len(fitresult)
    
    # Compute detection probability with exception handling
    try:
        p_detect = compute_detection_prob(
            model_fluxes, wavelengths, av_law, fitresult, gaia_filter,
            is_galaxy=is_galaxy, n_samples=n_samples, return_per_model=False
        )
    except Exception as e:
        p_detect = np.nan
    
    return {
        'id': source_id,
        'pDetect': p_detect,
        'model_cat': model_cat,
        'n_fits': n_fits
    }

def compute_detection_probs_batch(fitresults, model_fluxes, wavelengths, 
                                  av_law, gaia_filter=GAIA_G_FILTER, 
                                  is_galaxy=False, n_samples=5,
                                  max_rank=5, n_workers=1, verbose=True):
    """
    Compute Gaia detection probabilities for all sources in fit results.
    
    Loops over each source, computes overall detection probability, and returns
    a DataFrame with results.
    
    Parameters
    ----------
    fitresults : pandas.DataFrame
        Fit results dataframe for all sources from io.load_batchfit_dir()
        Must contain columns: 'id', 'fit_rank', 'chi2', 'sc', 'av'
        For galaxy models, must also contain: 'imp_flux_H'
    model_fluxes : np.ndarray
        Model SED fluxes, shape (N_models, N_wavelengths)
    wavelengths : np.ndarray
        Wavelength grid in microns
    av_law : np.ndarray
        Extinction law (A_lambda / A_V) for each wavelength
    gaia_filter : dict or tuple, optional
        Gaia filter data (dict with 'wavelength', 'transmission', 'zeropoint' keys
        or tuple of (wavelength, transmission, zeropoint)).
        Defaults to GAIA_G_FILTER (pre-loaded Gaia G-band filter).
    is_galaxy : bool, optional
        If True, use H-band scaling for galaxy models (default: False)
    n_samples : int, optional
        Number of Monte Carlo samples per source (default: 5)
    max_rank : int, optional
        Maximum fit rank to use (default: 5)
    n_workers : int, optional
        Number of parallel workers. Use 1 for serial processing (default: 1),
        -1 for all available cores, or specify number of cores to use.
    verbose : bool, optional
        Print progress (default: True)
    
    Returns
    -------
    results : pandas.DataFrame
        DataFrame with columns:
        - 'id': Source ID (same as in fitresults)
        - 'pDetect': Detection probability (0-1)
        - 'model_cat': Model category (YSO, gal, etc.)
        - 'n_fits': Number of fits used (should be <= max_rank)
    
    Examples
    --------
    # For YSO sources (using default GAIA_G_FILTER)
    results = compute_detection_probs_batch(
        fitresults, yso_fluxes, wavelengths, av_law,
        is_galaxy=False, n_samples=5
    )
    
    # For galaxy sources (using default GAIA_G_FILTER)
    results = compute_detection_probs_batch(
        fitresults, gal_fluxes, wavelengths, av_law,
        is_galaxy=True, n_samples=5
    )
    
    # Or explicitly pass a custom filter
    results = compute_detection_probs_batch(
        fitresults, yso_fluxes, wavelengths, av_law, 
        gaia_filter=custom_filter, is_galaxy=False
    )
    """
    # Filter to requested fit ranks and rename columns ONCE
    fitres_filtered = fitresults[fitresults['fit_rank'] <= max_rank].copy()
    fitres_filtered['sc_best'] = fitres_filtered['sc']
    fitres_filtered['av_best'] = fitres_filtered['av']
    
    # **KEY OPTIMIZATION**: Group by source ID once instead of filtering N times
    grouped = fitres_filtered.groupby('id', sort=False)
    source_ids = list(grouped.groups.keys())
    n_sources = len(source_ids)
    
    if verbose:
        print(f"Computing detection probabilities for {n_sources} sources...")
        print(f"Model type: {'Galaxy' if is_galaxy else 'YSO/SPS'}")
        print(f"N_samples per source: {n_samples}")
        print(f"N_models: {model_fluxes.shape[0]}")
        print(f"N_workers: {n_workers if n_workers != -1 else 'all cores'}")
    
    # **FIX FOR MEMMAP PICKLING**: Convert any memory-mapped arrays to regular arrays
    # This ensures they can be pickled and sent to worker processes
    model_fluxes = np.array(model_fluxes)
    wavelengths = np.array(wavelengths)
    av_law = np.array(av_law)
    
    # Handle gaia_filter - convert to dict format and ensure arrays are not memmaps
    if isinstance(gaia_filter, dict):
        gaia_filter = {
            'wavelength': np.array(gaia_filter['wavelength']),
            'transmission': np.array(gaia_filter['transmission']),
            'zeropoint': gaia_filter['zeropoint']
        }
    else:
        # Assume tuple format (wavelength, transmission, zeropoint)
        gaia_filter = {
            'wavelength': np.array(gaia_filter[0]),
            'transmission': np.array(gaia_filter[1]),
            'zeropoint': gaia_filter[2]
        }
    
    # Choose serial or parallel processing
    if n_workers == 1:
        # Serial processing
        results_list = []
        for i, source_id in enumerate(source_ids):
            source_fits = grouped.get_group(source_id)
            result = _compute_detection_prob_single(
                source_id, source_fits, model_fluxes, wavelengths,
                av_law, gaia_filter, is_galaxy, n_samples
            )
            results_list.append(result)
            
            # Progress update
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_sources} sources...")
    else:
        # Parallel processing
        from joblib import Parallel, delayed
        from tqdm import tqdm
        
        if verbose:
            print(f"  Starting parallel processing...")
        
        # Use loky backend (multiprocessing) for true parallelism across CPU cores
        # DataFrames will be pickled once per worker process (not per task)
        results_list = Parallel(n_jobs=n_workers, backend='multiprocessing', verbose=0)(
            delayed(_compute_detection_prob_single)(
                source_id, grouped.get_group(source_id), 
                model_fluxes, wavelengths, av_law, gaia_filter, 
                is_galaxy, n_samples
            )
            for source_id in tqdm(source_ids, disable=not verbose, 
                                  desc="Processing sources", unit="source")
        )
    
    if verbose:
        print(f"✓ Completed all {n_sources} sources")
    
    # Convert list of dicts to DataFrame
    results = pd.DataFrame(results_list)
    
    if verbose:
        print(f"\nSummary statistics:")
        print(f"  Mean pDetect: {results['pDetect'].mean():.3f}")
        print(f"  Median pDetect: {results['pDetect'].median():.3f}")
        print(f"  Min pDetect: {results['pDetect'].min():.3f}")
        print(f"  Max pDetect: {results['pDetect'].max():.3f}")
    
    return results














