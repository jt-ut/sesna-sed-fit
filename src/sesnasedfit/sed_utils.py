"""
SED Utility Functions

General-purpose functions for manipulating and analyzing spectral energy 
distributions (SEDs). These utilities handle:
- Scaling SEDs by distance and extinction
- Convolving SEDs with filter transmission curves
- Converting flux measurements to magnitudes
"""

import numpy as np


def scale_sed_distance_extinction(flux, sc_best, av_best, av_law):
    """
    Scale SED by distance and extinction.
    
    Applies transformations in log space internally, returns linear flux:
    log_flux_scaled = log_flux + sc_best * (-2) + av_best * av_law
    
    Parameters
    ----------
    flux : np.ndarray
        Flux in LINEAR units. Can be:
        - 1D array (n_wavelengths,): Single SED
        - 2D array (n_seds, n_wavelengths): Multiple SEDs
    sc_best : float
        Fitted distance scaling parameter (log10 of distance in units of 10 pc)
    av_best : float
        Fitted extinction (A_V) value in magnitudes
    av_law : np.ndarray
        Extinction law pattern (1D array of length n_wavelengths), pre-scaled 
        with -1/2.5 factor. This should be obtained from Extinction.get_av(wavelengths), 
        which returns -0.4 * [A_lambda / A_V]. Do NOT use raw A_lambda / A_V values.
    
    Returns
    -------
    flux_scaled : np.ndarray
        Scaled flux in LINEAR units (same shape as input flux)
    
    Raises
    ------
    ValueError
        If flux and av_law dimensions are incompatible
    TypeError
        If sc_best or av_best are not scalars
    
    Notes
    -----
    The brightness scaling term (sc_best * -2) encodes the inverse-square law
    for flux dimming with distance: F ∝ 1/d^2, where sc_best = log10(d/10pc).
    
    The extinction term (av_best * av_law) applies wavelength-dependent extinction,
    where av_law already includes the magnitude-to-log-flux conversion factor (-1/2.5).
    """
    # Validate that sc_best and av_best are scalars
    if not np.isscalar(sc_best):
        raise TypeError(f"sc_best must be a scalar, got array with shape {np.shape(sc_best)}")
    if not np.isscalar(av_best):
        raise TypeError(f"av_best must be a scalar, got array with shape {np.shape(av_best)}")
    
    # Validate av_law is 1D
    av_law = np.asarray(av_law)
    if av_law.ndim != 1:
        raise ValueError(f"av_law must be 1D array, got shape {av_law.shape}")
    
    # Validate flux dimensions
    flux = np.asarray(flux)
    if flux.ndim == 1:
        # Single SED - must match av_law length
        if len(flux) != len(av_law):
            raise ValueError(f"For 1D flux, length must match av_law length. "
                           f"Got flux length {len(flux)}, av_law length {len(av_law)}")
    elif flux.ndim == 2:
        # Multiple SEDs - last axis must match av_law length
        if flux.shape[1] != len(av_law):
            raise ValueError(f"For 2D flux, second dimension must match av_law length. "
                           f"Got flux shape {flux.shape}, av_law length {len(av_law)}")
    else:
        raise ValueError(f"flux must be 1D or 2D array, got shape {flux.shape}")
    
    # Convert to log space
    log_flux = np.log10(flux)
    
    # Apply scaling in log space
    # Brightness scaling term (distance dimming)
    brightness_term = sc_best * (-2.0)
    
    # Extinction term (already has -1/2.5 factor from Extinction.get_av())
    extinction_term = av_best * av_law
    
    log_flux_scaled = log_flux + brightness_term + extinction_term
    
    # Convert back to linear flux
    flux_scaled = 10.0 ** log_flux_scaled
    
    return flux_scaled


# def convolve_sed_with_filter(sed_wav, sed_flux, filter_wav, filter_trans):
#     """
#     Convolve SED(s) with a filter transmission curve.
    
#     Parameters
#     ----------
#     sed_wav : np.ndarray
#         SED wavelength grid in microns (length N_wav)
#     sed_flux : np.ndarray
#         SED flux in LINEAR units
#         Can be 1D (single SED) or 2D (multiple SEDs, shape N_seds x N_wav)
#     filter_wav : np.ndarray
#         Filter wavelength grid in microns
#     filter_trans : np.ndarray
#         Filter transmission curve (normalized to peak = 1)
    
#     Returns
#     -------
#     convolved_flux : float or np.ndarray
#         Convolved flux(es) in same units as input fluxes
#         Returns float if input is 1D, array if input is 2D
#     """
#     # Interpolate filter transmission onto SED wavelength grid
#     filter_interp = np.interp(sed_wav, filter_wav, filter_trans, 
#                                left=0.0, right=0.0)
    
#     # Handle 1D vs 2D input
#     if sed_flux.ndim == 1:
#         # Single SED
#         numerator = np.trapz(sed_flux * filter_interp * sed_wav, sed_wav)
#         denominator = np.trapz(filter_interp * sed_wav, sed_wav)
#     else:
#         # Multiple SEDs (shape: N_seds x N_wav)
#         numerator = np.trapz(sed_flux * filter_interp[np.newaxis, :] * sed_wav[np.newaxis, :], 
#                             sed_wav, axis=1)
#         denominator = np.trapz(filter_interp * sed_wav, sed_wav)
    
#     convolved_flux = numerator / denominator
    
#     return convolved_flux

def convolve_sed_with_filter(sed_wav, sed_flux, filter_wav, filter_trans):
    """
    Convolve SED(s) with a filter transmission curve to compute synthetic photometry.
    
    Uses the photon-counting formula appropriate for CCD and semiconductor detectors:
        flux_convolved = ∫(F_λ · T_λ · λ · dλ) / ∫(T_λ · λ · dλ)
    
    The wavelength weighting (λ) accounts for the fact that photon-counting detectors
    respond to photon number rather than photon energy. For energy-counting detectors
    (bolometers), use a different formula without the λ weighting.
    
    Parameters
    ----------
    sed_wav : np.ndarray
        SED wavelength grid in microns (length N_wav)
    sed_flux : np.ndarray
        SED flux in LINEAR units (e.g., mJy, erg/s/cm²/Å)
        Can be 1D (single SED) or 2D (multiple SEDs, shape N_seds x N_wav)
    filter_wav : np.ndarray
        Filter wavelength grid in microns
    filter_trans : np.ndarray
        Filter transmission curve. Normalization does not matter as the 
        transmission appears in both numerator and denominator and cancels out.
        Can be normalized to peak = 1, or provided as raw transmission values.
    
    Returns
    -------
    convolved_flux : float or np.ndarray
        Convolved flux(es) in same units as input sed_flux
        Returns float if input is 1D, array if input is 2D
    
    Notes
    -----
    This function is appropriate for photon-counting detectors including:
    - CCDs (e.g., Gaia, optical surveys)
    - InfraRed arrays (e.g., 2MASS J/H/K bands)
    - InSb detectors (e.g., Spitzer IRAC 3.6, 4.5 μm)
    - Si:As detectors (e.g., Spitzer IRAC 5.8, 8.0 μm, MIPS 24 μm)
    
    For energy-counting detectors (bolometers), a different formula without
    the λ weighting factor should be used.
    
    The filter transmission curve is interpolated onto the SED wavelength grid
    before integration. Wavelengths outside the filter range are assigned zero
    transmission.
    
    Examples
    --------
    >>> # Single SED
    >>> wav = np.linspace(0.3, 1.0, 100)  # microns
    >>> flux = np.ones(100) * 1000  # mJy
    >>> filter_wav = np.linspace(0.4, 0.7, 50)
    >>> filter_trans = np.ones(50)  # Flat filter
    >>> result = convolve_sed_with_filter(wav, flux, filter_wav, filter_trans)
    
    >>> # Multiple SEDs
    >>> flux_array = np.ones((10, 100)) * 1000  # 10 SEDs
    >>> results = convolve_sed_with_filter(wav, flux_array, filter_wav, filter_trans)
    >>> results.shape
    (10,)
    """
    # Interpolate filter transmission onto SED wavelength grid
    # Set transmission to 0 outside the filter wavelength range
    filter_interp = np.interp(sed_wav, filter_wav, filter_trans, 
                               left=0.0, right=0.0)
    
    # Handle 1D vs 2D input
    if sed_flux.ndim == 1:
        # Single SED
        numerator = np.trapz(sed_flux * filter_interp * sed_wav, sed_wav)
        denominator = np.trapz(filter_interp * sed_wav, sed_wav)
    else:
        # Multiple SEDs (shape: N_seds x N_wav)
        # Use broadcasting to apply filter to all SEDs at once
        numerator = np.trapz(sed_flux * filter_interp[np.newaxis, :] * sed_wav[np.newaxis, :], 
                            sed_wav, axis=1)
        denominator = np.trapz(filter_interp * sed_wav, sed_wav)
    
    convolved_flux = numerator / denominator
    
    return convolved_flux


def flux_to_magnitude(flux_mJy, zeropoint):
    """
    Convert flux to AB magnitude.
    
    Parameters
    ----------
    flux_mJy : float or np.ndarray
        Flux in milli-Janskys
    zeropoint : float
        Magnitude zero point for the filter
    
    Returns
    -------
    magnitude : float or np.ndarray
        AB magnitude
    """
    # Avoid log of zero/negative
    with np.errstate(divide='ignore', invalid='ignore'):
        magnitude = -2.5 * np.log10(flux_mJy) + zeropoint
    
    return magnitude

