"""
I/O utilities for SESNA SEDFIT package.

This module provides functions for reading and formatting Spitzer Extended Solar 
Neighborhood Archive (SESNA) catalog data for SED (Spectral Energy Distribution) 
fitting. It handles IPAC table format parsing, data reformatting for SED fitting 
tools, and conversion between HDF5 and ASCII formats.

Functions
---------
read_catalog_ipac1_txt
    Parse IPAC table format catalog files with fixed-width columns.
format_SESNA_SEDFIT
    Reformat SESNA catalog data into HDF5 format suitable for SED fitting.
read_SESNA_SEDFIT_ascii
    Convert HDF5 catalog data to ASCII format for SED fitting input.
read_fitparm
    Parse SED fitting parameter files.
read_extinction_info
    Parse extinction law information files.
"""

import pandas as pd
import numpy as np
import faiss
import os
import warnings
from tables import NaturalNameWarning
import re
from scipy.interpolate import interp1d
import h5py
from pathlib import Path
# For parallel results loading 
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial



def read_catalog_ipac1_txt(path_catalog_ipac1_txt, select_columns=None):
    """
    Read an IPAC table format catalog file with pipe-delimited fixed-width columns.
    
    IPAC format catalogs have a specific structure:
    - Line 1: Comment line with creation timestamp (starts with backslash)
    - Line 2: Column names separated by pipes (|)
    - Line 3: Data types (CHAR, DOUBLE, FLOAT, INT)
    - Line 4: Units for each column
    - Line 5: Null/missing value indicators
    - Line 6+: Data rows
    
    The pipes in line 2 define field boundaries for all subsequent lines.
    
    Parameters
    ----------
    path_catalog_ipac1_txt : str
        Path to the IPAC1 format catalog text file.
    select_columns : list of str, optional
        If provided, only extract these columns from the catalog. Column names
        must match those in the file header exactly.
    
    Returns
    -------
    lines : pandas.DataFrame
        DataFrame containing the catalog data with columns as specified in the
        file header (or subset if select_columns provided).
    hdr : pandas.DataFrame
        Metadata DataFrame with column names as index and the following columns:
        - DTYPES: Python data type functions (str, int, float)
        - UNITS: Physical units for each column
        - NAVALS: Null/missing value indicators for each column
    
    Examples
    --------
    >>> data, header = read_catalog_ipac1_txt('catalog_ipac1.txt')
    >>> data, header = read_catalog_ipac1_txt('catalog_ipac1.txt', 
    ...                                        select_columns=['SESNA_NAME', 'ra', 'dec'])
    """
    # Open file, read all lines
    with open(path_catalog_ipac1_txt, 'r') as f:
        lines = f.readlines()

    # First line is a "\created <timestamp>" comment, delete it
    del lines[0]
    
    # Next line has pipes ("|") separating column names.
    # Find the location of the pipes, which sets field delimeters for every subsequent line,
    # strip out the column names, and delete them
    pipelocs = np.array([m.start() for m in re.finditer('\|', lines[0])])
    field_from = pipelocs[:-1] + 1
    field_to = pipelocs[1:]
    field_zip = lambda: zip(field_from, field_to)
    def field_stripper(l):
        return [l[i:j].strip() for (i,j) in field_zip()]
    COLUMNS = field_stripper(lines[0])
    del lines[0]

    # If we are only import selected columns, must modify the field_from and field_to lists to reflect sub-selection
    if select_columns is not None:
        select_idx = [COLUMNS.index(i) for i in select_columns]
        field_from = [field_from[i] for i in select_idx]
        field_to = [field_to[i] for i in select_idx]
        COLUMNS = [COLUMNS[i] for i in select_idx]

    # Next line contains column data types: CHAR, DOUBLE, FLOAT, INT
    # Strip them out, then map to Python datatypes: str(), float(), float(), int()
    DTYPES = field_stripper(lines[0])
    dtype_mapper = {'CHAR': str, 'DOUBLE': float, 'FLOAT': float, 'INT': int}
    DTYPES = [dtype_mapper[i] for i in DTYPES]
    del lines[0]

    # Next line contains units.
    # Strip them out
    UNITS = field_stripper(lines[0])
    del lines[0]
    
    # Next line defines the dummy values representing missing values in each field.
    # Strip them out, then convert them to their respective datatypes
    NAVALS = field_stripper(lines[0])
    NAVALS = [j(i) if i!='null' else None for (i,j) in zip(NAVALS, DTYPES)]
    del lines[0]

    # Put all this meta-information into a header dictionary
    hdr = pd.DataFrame({'COLUMNS': COLUMNS, 'DTYPES': DTYPES, 'UNITS':UNITS, 'NAVALS':NAVALS})
    hdr.set_index('COLUMNS',inplace=True)
    hdr.index.names = [None]
    
    lines = [[s[i:(j+1)].strip() for i,j in zip(field_from, field_to)] for s in lines]
    ## Convert string values to their respective datatypes,
    ## Then transform to dataframe
    lines = [[fxn(val) if val!='null' else None for val, fxn in zip(l, hdr['DTYPES'])] for l in lines]
    lines = pd.DataFrame(columns = hdr.index, data = lines)

    return lines, hdr


# def format_SESNA_SEDFIT(path_catalog_ipac1_txt, distance_range_kpc, outpath_hdf5=None):
#     """
#     Format SESNA catalog data for SED fitting and save to HDF5.
    
#     This function reads a SESNA IPAC1 format catalog, extracts photometry from
#     2MASS, IRAC, and MIPS bands, handles missing data by using survey upper 
#     bounds or nearest-neighbor DCOMP90 values, and saves the formatted data
#     to an HDF5 file suitable for SED fitting.
    
#     Processing steps:
#     1. Extract multi-band photometry (J, H, KS, IRAC 3.6-8.0, MIPS 24)
#     2. Handle missing 2MASS fluxes by substituting survey upper bounds
#     3. Handle missing IRAC/MIPS fluxes by substituting DCOMP90 values
#     4. Use spatial nearest neighbors for missing DCOMP90 values
#     5. Save formatted tables to HDF5
    
#     Parameters
#     ----------
#     path_catalog_ipac1_txt : str
#         Path to the input SESNA catalog in IPAC1 text format.
#     distance_range_kpc : tuple of float
#         (min_distance, max_distance) in kiloparsecs. This is stored as metadata
#         indicating the distance range to the molecular cloud region.
#     outpath_hdf5 : str, optional
#         Directory path for output HDF5 file. If None, does not write to disk
#         and instead returns the data as a dictionary. If provided, writes HDF5
#         file with name <catalog>_SEDFIT_INPUT.hdf5.
    
#     Returns
#     -------
#     dict or None
#         If outpath_hdf5 is None, returns dictionary containing:
#         - ID: Source identifiers (SESNA names with 'SESNA ' prefix removed)
#         - ROW: Row indices in the catalog
#         - COORDS: RA and DEC coordinates
#         - CLASS: Source classifications
#         - AK: A_K extinction values
#         - FNU: Flux densities in Jy for each band
#         - SIGMA_FNU: Flux uncertainties in Jy
#         - ORIGIN_FNU: Origin codes for each flux measurement
#         - DISTANCE_RANGE_KPC: Distance range metadata
#         - NSOURCES: Total number of sources
        
#         If outpath_hdf5 is provided, writes HDF5 file and returns None.
    
#     Notes
#     -----
#     ORIGIN_FNU codes indicate data provenance:
#         1: Valid observed flux measurement
#         2: Global survey upper bound (2MASS bands only)
#         90: DCOMP90 value from same source
#         91: DCOMP90 value from nearest spatial neighbor
    
#     For ORIGIN_FNU = 91, additional columns DCOMP90_NN_<band> store the 
#     row index of the neighbor from which the value was taken.
    
#     Examples
#     --------
#     >>> format_SESNA_SEDFIT('catalog_ipac1.txt', (0.1, 2.0))
#     >>> format_SESNA_SEDFIT('catalog_ipac1.txt', (0.1, 2.0), 
#     ...                     outpath_hdf5='/path/to/output/')
#     """
#     ## Define the columns of the catalog file that we are importing
#     # Temporarily define the filter names, these are used in various contexts
#     filters_2MASS = ['J','H','KS']
#     filters_IRAC = ['3_6','4_5','5_8','8_0']
#     filters_MIPS = ['24']

#     # Build the columns we're extracting from the text file
#     select_columns = ['SESNA_NAME','ra','dec']
#     for filter in filters_2MASS + filters_IRAC + filters_MIPS:
#         select_columns.extend(['FNU_'+filter, 'SIGMA_FNU_'+filter])
#     for filter in filters_IRAC + filters_MIPS:
#         select_columns.append('DCOMP90_FNU_'+filter)
#     select_columns.append('CLASS')
#     select_columns.append('AK')

#     ## Open the catalog file, import the selected columns
#     print("Importing catalog %s ... " % path_catalog_ipac1_txt, end = '', flush=True)
#     df, hdr = read_catalog_ipac1_txt(path_catalog_ipac1_txt, select_columns=select_columns)
#     print("done")


#     ## Check for duplicate names
#     isdup = df['SESNA_NAME'].duplicated()
#     if sum(isdup) > 0:
#         print("Removing %d duplicate sources ... " % sum(isdup), end = '', flush=True)
#         df = df[~isdup]
#         print("done")


#     ## Extract the SESNA_NAME, COORDS, CLASS, AK variables into their own dataframes, drop them from the original dataframe
#     print("-- Extracting SESNA_NAME table ... ", end = '', flush=True)
#     ID = df['SESNA_NAME'].copy()
#     ID = ID.str.replace("SESNA ", "")
#     ID = ID.str.strip()

#     print("done")
#     print("-- Extracting COORDS table ... ", end = '', flush=True)
#     COORDS = df[['ra','dec']].copy()
#     COORDS.columns = ['RA', 'DEC']
#     print("done")
#     print("-- Extracting CLASS table ... ", end = '', flush=True)
#     CLASS = df['CLASS'].copy()
#     print("done")
#     print("-- Extracting AK table ... ", end = '', flush=True)
#     AK = df['AK'].copy()
#     print("done")
#     # Drop these columns from the imported data frame, just to save memory
#     df.drop(['SESNA_NAME','ra','dec','CLASS','AK'], axis=1, inplace=True)

#     ## Define the catalog row number
#     print("-- Building ROW table ... ", end = '', flush=True)
#     ROW = pd.DataFrame(np.arange(0, df.shape[0]), columns=['ROWINDEX'])
#     print("done")


#     ## Extract the (raw) FNU values into a FNU data frame
#     print("-- Extracting FNU table ... ", end = '', flush=True)
#     extract_cols = ['FNU_' + i for i in filters_2MASS+filters_IRAC+filters_MIPS]
#     FNU = df[extract_cols].copy()
#     FNU.columns = filters_2MASS + filters_IRAC + filters_MIPS
#     df.drop(extract_cols, axis=1, inplace=True)
#     print("done")


#     ## Extract the (raw) SIGMA_FNU values into a FNU data frame
#     print("-- Extracting SIGMA_FNU table ... ", end = '', flush=True)
#     extract_cols = ['SIGMA_FNU_' + i for i in filters_2MASS+filters_IRAC+filters_MIPS]
#     SIGMA_FNU = df[extract_cols].copy()
#     SIGMA_FNU.columns = filters_2MASS + filters_IRAC + filters_MIPS
#     df.drop(extract_cols, axis=1, inplace=True)
#     print("done")

#     ## Initialize the ORIGIN table.
#     ## This is used to record the type of measurement stored in the FNU table.
#     ## It has columns J,H,K,...,24 that contain integer codes:
#     ##   = 1: means FNU[i,<band>] is a valid observed flux measurement
#     ##   = 2: means FNU[i,<band>] was replaced by a global survey upper bound.
#     ##        NOTE: This can only occur for the 2MASS bands J,H,K.
#     ##   = 90: means the value in FNU[i,<band>] is the corresponding DCOMP90_<band> value for source i
#     ##   = 91: means the value in FNU[i,<band>] is the DCOMP90_<band> value for the 1st within-catalog nearest neighbor
#     ##         (as measured by RA/DEC coords) that contained a valid DCOMP90_<band> measurement.
#     ## Additionally, it has columns DCOMP90_NN_<band>, containing the integer row index from which the DCOMP90 value was pulled,
#     ## when this occurs. Here, <band> can only be one of the IRAC or MIPS wavelengths, because the 2MASS J,H,K are only plugged with
#     ## their upper bounds. Thus, if ORIGIN[i,'3_6'] = 91 (indicating that source i's FNU[i,'3_6'] value contains the DCOMP90_FNU_3_6 value from
#     ## i's nearest-spatial-neighbor, ORIGIN[i, 'DCOMP90_NN_3_6'] contains the row index (in the same catalog) from where this value was pulled.
#     ## This is just bookkeeping, in case we need to investigate from where the values for a specific source came.
#     ##
#     ## To start, initialize the J through 24 columns = 1.
#     ## Below, we will do searching to determine where FNU is invalid, and overwrite these columns appropriately (=1, 2, 90, or 91)
#     ORIGIN_FNU = pd.DataFrame(data = int(1), index=range(df.shape[0]), columns=filters_2MASS + filters_IRAC + filters_MIPS, dtype=int)
#     ## Now, for the IRAC/MIPS bands, add integer columns to store the DCOMP90 nearest-neighbor index.
#     ## My convention is -1 indicates no neighbor information was used.
#     ## Anything >= 0 is the row index from which nearest-neighbor DCOMP90 info was taken.
#     for filter in filters_IRAC + filters_MIPS:
#         ORIGIN_FNU['DCOMP90_NN_' + filter] = int(-1)

#     ## ** Loop over each filter, and modify the FNU, SIGMA_FNU, and ORIGIN_FNU tables accordingly

#     ## Process the 2MASS filters
#     UB_2MASS = {'J':0.763, 'H':0.934, 'KS':1.27} # hard-coded values from Rob for upper bounds of 2mass
#     for filter in filters_2MASS:
#         print("Processing filter %s ... " % filter, end = '', flush=True)
#         # Determine where the flux is missing
#         isna = (FNU[filter] == hdr['NAVALS'].loc['FNU_' + filter])
#         # Overwrite missing fluxes with their global bounds
#         FNU.loc[isna, filter] = UB_2MASS[filter]
#         # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
#         # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
#         SIGMA_FNU.loc[isna, filter] = 0.99 # 1.0
#         # Set the origin = 2 (global upper bounds are used)
#         ORIGIN_FNU.loc[isna,filter] = 2
#         print("done")

#     ## Process the IRAC+MIPS filters

#     for filter in filters_IRAC + filters_MIPS:
#         print("-- Processing FNU + SIGMA_FNU + ORIGIN_FNU for filter %s ... " % filter, end = '', flush=True)
        
#         # Determine where flux and dcomp90 values are invalid for this band
#         fluxnaval = hdr['NAVALS'].loc['FNU_' + filter]
#         isna_flux = FNU[filter] == fluxnaval
#         dcompnaval = hdr['NAVALS'].loc['DCOMP90_FNU_' + filter]
#         isna_dcomp = df['DCOMP90_FNU_' + filter] == dcompnaval

#         ## Now work through each case, and make the appropriate changes to FNU, SIGMA_FNU, ORIGIN_FNU

#         ## Case 0: FNU exists, so we use it.
#         # This case is already taken care of above, since we just copied the existing flux & sigma values over
#         # into these tables, and initialized the ORIGIN table to = 1 everywhere

#         ## Case 1: FNU is missing, but a DCOMP90 value exists to plug it
#         isna_w_dcomp = isna_flux & (~isna_dcomp)
#         FNU.loc[isna_w_dcomp, filter] = df.loc[isna_w_dcomp, 'DCOMP90_FNU_' + filter].to_numpy()
#         # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
#         # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
#         SIGMA_FNU.loc[isna_w_dcomp, filter] = 0.99 # 1.0
#         # Set the origin = 90 (existing dcomp upper bound is used)
#         ORIGIN_FNU.loc[isna_w_dcomp,filter] = 90

#         ## Case 2: FNU is missing, but a DCOMP90 value DOES NOT exist to plug it.
#         isna_wo_dcomp = isna_flux & isna_dcomp
#         # We need to find the nearest spatial neighbor of this source that does have a valid DCOMP90 value for this band.
#         # We use the FAISS library to find these nearest neighbors
#         faiss_index = faiss.IndexFlatL2(2) # 2, because we're searching the length=2 (RA,DEC) vector
#         faiss_index.add(COORDS.loc[~isna_dcomp, ['RA','DEC']])
#         _, nhbs = faiss_index.search(COORDS.loc[isna_wo_dcomp,['RA','DEC']], 1)
#         nhbs = nhbs.reshape(-1)
#         # Now that nearest neighbors are found, use their values to plug the missing FNU value
#         FNU.loc[isna_wo_dcomp,filter] = df.loc[df[~isna_dcomp].index[nhbs], 'DCOMP90_FNU_' + filter].to_numpy()
#         # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
#         # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
#         SIGMA_FNU.loc[isna_wo_dcomp, filter] = 0.99 #1.0
#         # Set the origin = 91 (nearest neighbor dcomp upper bound is used),
#         # and record the row index of the nearest neighbor
#         ORIGIN_FNU.loc[isna_wo_dcomp, filter] = int(91)
#         ORIGIN_FNU.loc[isna_wo_dcomp, 'DCOMP90_NN_'+filter] = df[~isna_dcomp].index[nhbs].to_numpy()
#         print("done")


#     ## Format the (sorted) input distance range and number of sources in this catalog as dataframes
#     distance_range_kpc = pd.Series(distance_range_kpc, name='DISTANCE_RANGE_KPC').sort_values().to_frame()
#     NSOURCES = pd.Series([ID.shape[0]], name='NSOURCES').to_frame()

#     ## Ready to return. If an output file path was given, we will construct an hdf5 file with tables
#     ## whose names match the relevant info we formatted.
#     if outpath_hdf5 is not None:
#         # Create path if it doesn't exist
#         if not os.path.exists(outpath_hdf5):
#             os.makedirs(outpath_hdf5, exist_ok=True)
#         # Build output filename for hdf5 file
#         hdf5_fn = os.path.basename(path_catalog_ipac1_txt).strip(".txt")+"_SEDFIT_INPUT.hdf5"
#         hdf5_fullpath = os.path.join(outpath_hdf5, hdf5_fn)
#         warnings.filterwarnings('ignore', category=NaturalNameWarning)
#         # Open hdf5 file, add tables
#         with pd.HDFStore(hdf5_fullpath) as store:
#             store.put('ID', ID, format='table', data_columns=True)
#             store.put('COORDS', COORDS, format='table', data_columns=True)
#             store.put('CLASS', CLASS, format='table', data_columns=True)
#             store.put('AK', AK, format='table', data_columns=True)
#             store.put('ROW', ROW, format='table', data_columns=True)
#             store.put('FNU', FNU, format='table', data_columns=True)
#             store.put('SIGMA_FNU', SIGMA_FNU, format='table', data_columns=True)
#             store.put('ORIGIN_FNU', ORIGIN_FNU, format='table', data_columns=True)
#             store.put('DISTANCE_RANGE_KPC', distance_range_kpc, format='table', data_columns=True)
#             store.put('NSOURCES', NSOURCES, format='table', data_columns=True)
#         print("-- HDF5 file saved %s ... " % hdf5_fullpath)
#         return
#     else:
#         out = {"ID": ID, "COORDS": COORDS,
#         "CLASS": CLASS, "AK": AK, "ROW": ROW, "FNU": FNU, "SIGMA_FNU": SIGMA_FNU, "ORIGIN_FNU": ORIGIN_FNU,
#         "DISTANCE_RANGE_KPC": distance_range_kpc, "NSOURCES": NSOURCES}
#         return out

def format_SESNA_SEDFIT(path_catalog_ipac1_txt, distance_range_kpc, outpath_hdf5=None):
    """
    Format SESNA catalog data for SED fitting and save to HDF5.
    
    This function reads a SESNA IPAC1 format catalog, extracts photometry from
    2MASS, IRAC, and MIPS bands, handles missing data by using survey upper 
    bounds or nearest-neighbor DCOMP90 values, and saves the formatted data
    to an HDF5 file suitable for SED fitting.
    
    Processing steps:
    1. Extract multi-band photometry (J, H, KS, IRAC 3.6-8.0, MIPS 24)
    2. Handle missing 2MASS fluxes by substituting survey upper bounds
    3. Handle missing IRAC/MIPS fluxes by substituting DCOMP90 values
    4. Use spatial nearest neighbors for missing DCOMP90 values
    5. Save formatted tables to HDF5
    
    Parameters
    ----------
    path_catalog_ipac1_txt : str
        Path to the input SESNA catalog in IPAC1 text format.
    distance_range_kpc : tuple of float
        (min_distance, max_distance) in kiloparsecs. This is stored as metadata
        indicating the distance range to the molecular cloud region.
    outpath_hdf5 : str, optional
        Directory path for output HDF5 file. If None, does not write to disk
        and instead returns the data as a dictionary. If provided, writes HDF5
        file with name <catalog>_SEDFIT_INPUT.hdf5.
    
    Returns
    -------
    dict or None
        If outpath_hdf5 is None, returns dictionary containing:
        - ID: Source identifiers (SESNA names with 'SESNA ' prefix removed)
        - ROW: Row indices in the catalog
        - COORDS: RA and DEC coordinates
        - CLASS: Source classifications
        - AK: A_K extinction values
        - FNU: Flux densities in Jy for each band
        - SIGMA_FNU: Flux uncertainties in Jy
        - ORIGIN_FNU: Origin codes for each flux measurement
        - DISTANCE_RANGE_KPC: Distance range metadata
        - NSOURCES: Total number of sources
        
        If outpath_hdf5 is provided, writes HDF5 file and returns None.
    
    Notes
    -----
    ORIGIN_FNU codes indicate data provenance:
        1: Valid observed flux measurement
        2: Global survey upper bound (2MASS bands only)
        90: DCOMP90 value from same source
        91: DCOMP90 value from nearest spatial neighbor
    
    For ORIGIN_FNU = 91, additional columns DCOMP90_NN_<band> store the 
    row index of the neighbor from which the value was taken.
    
    Examples
    --------
    >>> format_SESNA_SEDFIT('catalog_ipac1.txt', (0.1, 2.0))
    >>> format_SESNA_SEDFIT('catalog_ipac1.txt', (0.1, 2.0), 
    ...                     outpath_hdf5='/path/to/output/')
    """
    ## Define the columns of the catalog file that we are importing
    # Temporarily define the filter names, these are used in various contexts
    filters_2MASS = ['J','H','KS']
    filters_IRAC = ['3_6','4_5','5_8','8_0']
    filters_MIPS = ['24']

    # Build the columns we're extracting from the text file
    select_columns = ['SESNA_NAME','ra','dec']
    for filter in filters_2MASS + filters_IRAC + filters_MIPS:
        select_columns.extend(['FNU_'+filter, 'SIGMA_FNU_'+filter])
    for filter in filters_IRAC + filters_MIPS:
        select_columns.append('DCOMP90_FNU_'+filter)
    select_columns.append('CLASS')
    select_columns.append('AK')

    ## Open the catalog file, import the selected columns
    print("Importing catalog %s ... " % path_catalog_ipac1_txt, end = '', flush=True)
    df, hdr = read_catalog_ipac1_txt(path_catalog_ipac1_txt, select_columns=select_columns)
    print("done")


    ## Check for duplicate names
    isdup = df['SESNA_NAME'].duplicated()
    if sum(isdup) > 0:
        print("Removing %d duplicate sources ... " % sum(isdup), end = '', flush=True)
        df = df[~isdup]
        print("done")


    ## Extract the SESNA_NAME, COORDS, CLASS, AK variables into their own dataframes, drop them from the original dataframe
    print("-- Extracting SESNA_NAME table ... ", end = '', flush=True)
    ID = df['SESNA_NAME'].copy()
    ID = ID.str.replace("SESNA ", "")
    ID = ID.str.strip()

    print("done")
    print("-- Extracting COORDS table ... ", end = '', flush=True)
    COORDS = df[['ra','dec']].copy()
    COORDS.columns = ['RA', 'DEC']
    print("done")
    print("-- Extracting CLASS table ... ", end = '', flush=True)
    CLASS = df['CLASS'].copy()
    print("done")
    print("-- Extracting AK table ... ", end = '', flush=True)
    AK = df['AK'].copy()
    print("done")
    # Drop these columns from the imported data frame, just to save memory
    df.drop(['SESNA_NAME','ra','dec','CLASS','AK'], axis=1, inplace=True)

    ## Define the catalog row number
    print("-- Building ROW table ... ", end = '', flush=True)
    ROW = pd.DataFrame(np.arange(0, df.shape[0]), index=df.index, columns=['ROWINDEX'])
    print("done")


    ## Extract the (raw) FNU values into a FNU data frame
    print("-- Extracting FNU table ... ", end = '', flush=True)
    extract_cols = ['FNU_' + i for i in filters_2MASS+filters_IRAC+filters_MIPS]
    FNU = df[extract_cols].copy()
    FNU.columns = filters_2MASS + filters_IRAC + filters_MIPS
    df.drop(extract_cols, axis=1, inplace=True)
    print("done")


    ## Extract the (raw) SIGMA_FNU values into a FNU data frame
    print("-- Extracting SIGMA_FNU table ... ", end = '', flush=True)
    extract_cols = ['SIGMA_FNU_' + i for i in filters_2MASS+filters_IRAC+filters_MIPS]
    SIGMA_FNU = df[extract_cols].copy()
    SIGMA_FNU.columns = filters_2MASS + filters_IRAC + filters_MIPS
    df.drop(extract_cols, axis=1, inplace=True)
    print("done")

    ## Initialize the ORIGIN table.
    ## This is used to record the type of measurement stored in the FNU table.
    ## It has columns J,H,K,...,24 that contain integer codes:
    ##   = 1: means FNU[i,<band>] is a valid observed flux measurement
    ##   = 2: means FNU[i,<band>] was replaced by a global survey upper bound.
    ##        NOTE: This can only occur for the 2MASS bands J,H,K.
    ##   = 90: means the value in FNU[i,<band>] is the corresponding DCOMP90_<band> value for source i
    ##   = 91: means the value in FNU[i,<band>] is the DCOMP90_<band> value for the 1st within-catalog nearest neighbor
    ##         (as measured by RA/DEC coords) that contained a valid DCOMP90_<band> measurement.
    ## Additionally, it has columns DCOMP90_NN_<band>, containing the integer row index from which the DCOMP90 value was pulled,
    ## when this occurs. Here, <band> can only be one of the IRAC or MIPS wavelengths, because the 2MASS J,H,K are only plugged with
    ## their upper bounds. Thus, if ORIGIN[i,'3_6'] = 91 (indicating that source i's FNU[i,'3_6'] value contains the DCOMP90_FNU_3_6 value from
    ## i's nearest-spatial-neighbor, ORIGIN[i, 'DCOMP90_NN_3_6'] contains the row index (in the same catalog) from where this value was pulled.
    ## This is just bookkeeping, in case we need to investigate from where the values for a specific source came.
    ##
    ## To start, initialize the J through 24 columns = 1.
    ## Below, we will do searching to determine where FNU is invalid, and overwrite these columns appropriately (=1, 2, 90, or 91)
    ORIGIN_FNU = pd.DataFrame(data = int(1), index=df.index, columns=filters_2MASS + filters_IRAC + filters_MIPS, dtype=int)
    ## Now, for the IRAC/MIPS bands, add integer columns to store the DCOMP90 nearest-neighbor index.
    ## My convention is -1 indicates no neighbor information was used.
    ## Anything >= 0 is the row index from which nearest-neighbor DCOMP90 info was taken.
    for filter in filters_IRAC + filters_MIPS:
        ORIGIN_FNU['DCOMP90_NN_' + filter] = int(-1)

    ## ** Loop over each filter, and modify the FNU, SIGMA_FNU, and ORIGIN_FNU tables accordingly

    ## Process the 2MASS filters
    UB_2MASS = {'J':0.763, 'H':0.934, 'KS':1.27} # hard-coded values from Rob for upper bounds of 2mass
    for filter in filters_2MASS:
        print("Processing filter %s ... " % filter, end = '', flush=True)
        # Determine where the flux is missing
        isna = FNU[filter] == hdr['NAVALS'].loc['FNU_' + filter]
        # Overwrite missing fluxes with their global bounds
        FNU.loc[isna, filter] = UB_2MASS[filter]
        # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
        # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
        SIGMA_FNU.loc[isna, filter] = 0.99 # 1.0
        # Set the origin = 2 (global upper bounds are used)
        ORIGIN_FNU.loc[isna, filter] = 2
        print("done")

    ## Process the IRAC+MIPS filters

    for filter in filters_IRAC + filters_MIPS:
        print("-- Processing FNU + SIGMA_FNU + ORIGIN_FNU for filter %s ... " % filter, end = '', flush=True)
        
        # Determine where flux and dcomp90 values are invalid for this band
        fluxnaval = hdr['NAVALS'].loc['FNU_' + filter]
        isna_flux = FNU[filter] == fluxnaval
        dcompnaval = hdr['NAVALS'].loc['DCOMP90_FNU_' + filter]
        isna_dcomp = df['DCOMP90_FNU_' + filter] == dcompnaval

        ## Now work through each case, and make the appropriate changes to FNU, SIGMA_FNU, ORIGIN_FNU

        ## Case 0: FNU exists, so we use it.
        # This case is already taken care of above, since we just copied the existing flux & sigma values over
        # into these tables, and initialized the ORIGIN table to = 1 everywhere

        ## Case 1: FNU is missing, but a DCOMP90 value exists to plug it
        isna_w_dcomp = isna_flux & (~isna_dcomp)
        FNU.loc[isna_w_dcomp, filter] = df.loc[isna_w_dcomp, 'DCOMP90_FNU_' + filter].to_numpy()
        # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
        # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
        SIGMA_FNU.loc[isna_w_dcomp, filter] = 0.99 # 1.0
        # Set the origin = 90 (existing dcomp upper bound is used)
        ORIGIN_FNU.loc[isna_w_dcomp,filter] = 90

        ## Case 2: FNU is missing, but a DCOMP90 value DOES NOT exist to plug it.
        isna_wo_dcomp = isna_flux & isna_dcomp
        # We need to find the nearest spatial neighbor of this source that does have a valid DCOMP90 value for this band.
        # We use the FAISS library to find these nearest neighbors
        faiss_index = faiss.IndexFlatL2(2) # 2, because we're searching the length=2 (RA,DEC) vector
        faiss_index.add(COORDS.loc[~isna_dcomp, ['RA','DEC']])
        _, nhbs = faiss_index.search(COORDS.loc[isna_wo_dcomp,['RA','DEC']], 1)
        nhbs = nhbs.reshape(-1)
        # Now that nearest neighbors are found, use their values to plug the missing FNU value
        FNU.loc[isna_wo_dcomp,filter] = df.loc[df[~isna_dcomp].index[nhbs], 'DCOMP90_FNU_' + filter].to_numpy()
        # Change the flux sigma = 1.0, required by sedfitter when bounds are given instead of fluxes
        # 1.0 tells sedfitter that the upper bound is hard, 0.9 would mean less hard, etc.
        SIGMA_FNU.loc[isna_wo_dcomp, filter] = 0.99 #1.0
        # Set the origin = 91 (nearest neighbor dcomp upper bound is used),
        # and record the row index of the nearest neighbor
        ORIGIN_FNU.loc[isna_wo_dcomp, filter] = int(91)
        ORIGIN_FNU.loc[isna_wo_dcomp, 'DCOMP90_NN_'+filter] = df[~isna_dcomp].index[nhbs].to_numpy()
        print("done")


    ## Format the (sorted) input distance range and number of sources in this catalog as dataframes
    distance_range_kpc = pd.Series(distance_range_kpc, name='DISTANCE_RANGE_KPC').sort_values().to_frame()
    NSOURCES = pd.Series([ID.shape[0]], name='NSOURCES').to_frame()

    ## Ready to return. If an output file path was given, we will construct an hdf5 file with tables
    ## whose names match the relevant info we formatted.
    if outpath_hdf5 is not None:
        # Create path if it doesn't exist
        if not os.path.exists(outpath_hdf5):
            os.makedirs(outpath_hdf5, exist_ok=True)
        # Build output filename for hdf5 file
        hdf5_fn = os.path.basename(path_catalog_ipac1_txt).strip(".txt")+"_SEDFIT_INPUT.hdf5"
        hdf5_fullpath = os.path.join(outpath_hdf5, hdf5_fn)
        warnings.filterwarnings('ignore', category=NaturalNameWarning)
        # Open hdf5 file, add tables
        with pd.HDFStore(hdf5_fullpath) as store:
            store.put('ID', ID, format='table', data_columns=True)
            store.put('COORDS', COORDS, format='table', data_columns=True)
            store.put('CLASS', CLASS, format='table', data_columns=True)
            store.put('AK', AK, format='table', data_columns=True)
            store.put('ROW', ROW, format='table', data_columns=True)
            store.put('FNU', FNU, format='table', data_columns=True)
            store.put('SIGMA_FNU', SIGMA_FNU, format='table', data_columns=True)
            store.put('ORIGIN_FNU', ORIGIN_FNU, format='table', data_columns=True)
            store.put('DISTANCE_RANGE_KPC', distance_range_kpc, format='table', data_columns=True)
            store.put('NSOURCES', NSOURCES, format='table', data_columns=True)
        print("-- HDF5 file saved %s ... " % hdf5_fullpath)
        return
    else:
        out = {"ID": ID, "COORDS": COORDS,
        "CLASS": CLASS, "AK": AK, "ROW": ROW, "FNU": FNU, "SIGMA_FNU": SIGMA_FNU, "ORIGIN_FNU": ORIGIN_FNU,
        "DISTANCE_RANGE_KPC": distance_range_kpc, "NSOURCES": NSOURCES}
        return out

def read_SESNA_SEDFIT_ascii(path_catalog_hdf5, startindex=None, endindex=None):
    """
    Read SESNA SED fitting input data from HDF5 file and format as ASCII strings.
    
    Reads formatted SESNA catalog data from HDF5 and converts each source to
    an ASCII string format expected by SED fitting tools. Each source's data
    includes ID, coordinates, validity flags, and interleaved flux/uncertainty
    pairs for all photometric bands.
    
    Parameters
    ----------
    path_catalog_hdf5 : str
        Path to the HDF5 file containing the catalog data.
        Expected filename format: <region>_catalog_ipac1_SEDFIT_INPUT.hdf5
    startindex : int, optional
        Starting row index (inclusive). If None, starts from 0.
    endindex : int, optional
        Ending row index (inclusive). If None, reads to end of file.
        Will be capped at the last available row if it exceeds catalog size.
    
    Returns
    -------
    list of dict
        List of dictionaries, one per source, each containing:
        - 'SOURCE_ASCII': Space-separated ASCII string with source data
        - 'AK': A_K extinction value (float)
        - 'CATALOG': Region name extracted from filename (str)
        - 'NVALID': Number of valid detections (int)
    
    Notes
    -----
    The SOURCE_ASCII format is space-delimited:
    1. ID (source name)
    2. RA coordinate
    3. DEC coordinate
    4. 8 flag values (1 if ORIGIN_FNU==1, else 3 if ORIGIN_FNU>1)
       for bands: J, H, KS, 3_6, 4_5, 5_8, 8_0, 24
    5. 16 flux values (interleaved FNU, SIGMA_FNU pairs)
       for bands: J, H, KS, 3_6, 4_5, 5_8, 8_0, 24
    
    Where flags are:
        1 = valid detection (ORIGIN_FNU == 1)
        3 = upper bound or substitute value (ORIGIN_FNU > 1)
    
    Examples
    --------
    >>> data = read_SESNA_SEDFIT_ascii('Serpens_catalog_ipac1_SEDFIT_INPUT.hdf5')
    >>> data = read_SESNA_SEDFIT_ascii('catalog.hdf5', startindex=0, endindex=999)
    >>> print(data[0]['SOURCE_ASCII'])
    J163000-242100 247.5000 -24.3500 1 1 1 3 3 3 3 3 0.763 0.05 0.934 0.06 ...
    """
    # Extract region name from filename
    # Format: <region>_catalog_ipac1_SEDFIT_INPUT.hdf5
    filename = os.path.basename(path_catalog_hdf5)
    # Remove the suffix
    region_name = filename.replace('_catalog_ipac1_SEDFIT_INPUT.hdf5', '')
    
    # Define band order
    bands = ['J', 'H', 'KS', '3_6', '4_5', '5_8', '8_0', '24']
    
    # Open HDF5 file and read tables
    with pd.HDFStore(path_catalog_hdf5, 'r') as store:
        # Determine row range
        if startindex is None:
            startindex = 0
        
        # Get total number of rows from ID table
        n_total_rows = store.get('NSOURCES')['NSOURCES'][0]
        
        if endindex is None:
            endindex = n_total_rows - 1
        else:
            # Cap endindex at the last available row
            endindex = min(endindex, n_total_rows - 1)
        
        # Load tables with the determined range (stop is exclusive in pandas, so add 1)
        ID = store.select('ID', start=startindex, stop=endindex+1)
        
        # Load other tables with the same row range
        COORDS = store.select('COORDS', start=startindex, stop=endindex+1)
        AK = store.select('AK', start=startindex, stop=endindex+1)
        FNU = store.select('FNU', start=startindex, stop=endindex+1)
        SIGMA_FNU = store.select('SIGMA_FNU', start=startindex, stop=endindex+1)
        ORIGIN_FNU = store.select('ORIGIN_FNU', start=startindex, stop=endindex+1)
    
    # Prepare output list
    output_data = []
    
    # Process each row
    n_rows = len(ID)
    for i in range(n_rows):
        # Extract ID (handle both single column DataFrame and Series)
        if isinstance(ID, pd.DataFrame):
            source_id = str(ID.iloc[i, 0])
        else:
            source_id = str(ID.iloc[i])
        
        # Extract coordinates
        ra = COORDS.iloc[i]['RA']
        dec = COORDS.iloc[i]['DEC']
        
        # Extract AK value (handle both DataFrame and Series)
        if isinstance(AK, pd.DataFrame):
            ak_value = float(AK.iloc[i, 0])
        else:
            ak_value = float(AK.iloc[i])
        
        # Build ASCII string components
        components = [source_id, str(ra), str(dec)]
        
        # Add flags (1 if ORIGIN_FNU==1, else 3 if ORIGIN_FNU>1)
        flags = []
        n_detections = 0
        for band in bands:
            origin_val = ORIGIN_FNU.iloc[i][band]
            flag = 1 if origin_val == 1 else 3
            flags.append(str(flag))
            if flag == 1:
                n_detections += 1
        components.extend(flags)
        
        # Add interleaved flux pairs (FNU, SIGMA_FNU for each band)
        for band in bands:
            fnu_val = FNU.iloc[i][band]
            sigma_val = SIGMA_FNU.iloc[i][band]
            components.append(str(fnu_val))
            components.append(str(sigma_val))
        
        # Create ASCII string
        ascii_string = ' '.join(components)
        
        # Add to output list
        output_data.append({
            'SOURCE_ASCII': ascii_string,
            'AK': ak_value,
            'CATALOG': region_name,
            'NVALID': n_detections
        })
    
    return output_data



def read_fit_parm(filename):
    """
    Read a SED fitting parameter file with keyword-value pairs.
    
    Parses a plain text parameter file where each line contains a keyword
    followed by its value(s). Lines starting with '#' are treated as comments.
    Special handling is provided for multi-value and repeated keywords.
    
    Additionally, if 'path_input_hdf5' is present in the parameter file,
    reads the DISTANCE_RANGE_KPC data from that HDF5 file and adds it to the
    returned dictionary as 'distance_range_kpc'. This allows the distance
    data to be loaded once in the main process rather than by each worker
    process, avoiding HDF5 file locking issues in parallel processing.
    
    Parameters
    ----------
    filename : str
        Path to the parameter file.
    
    Returns
    -------
    dict
        Dictionary with keyword as key and value(s) as value. Special cases:
        - 'model_dir': Can appear multiple times, returns list of paths
        - 'av_range': Returns list with format-specific structure:
            - ['absolute', min_val, max_val] for absolute ranges
            - ['catalog', offset_val] for catalog-relative ranges
        - 'apertures', 'filters': Split into lists of floats
        - 'nkeep': Converted to integer
        - 'distance_range_kpc': Automatically loaded from path_input_hdf5 if present
        - All other keywords: Single string value
    
    Examples
    --------
    >>> params = read_fitparm('fit_params.txt')
    >>> params['model_dir']
    ['/path/to/models1', '/path/to/models2']
    >>> params['av_range']
    ['absolute', 0.0, 30.0]
    >>> params['filters']
    [3.6, 4.5, 5.8, 8.0, 24.0]
    >>> params['nkeep']
    5
    >>> params['distance_range_kpc']
    [0.1, 2.0]
    
    Notes
    -----
    Example parameter file format:
        # Comment line
        data_dir /path/to/data
        path_input_hdf5 /path/to/catalog.hdf5
        model_dir /path/to/models1
        model_dir /path/to/models2
        av_range absolute 0.0 30.0
        filters 3.6 4.5 5.8 8.0 24.0
        apertures 1.0 2.0 3.0
        nkeep 5
    """
    params = {}
    
    # Keywords that should have their values split into lists
    list_keywords = ['apertures', 'filters']
    
    # Keywords that should be converted to integers
    int_keywords = ['nkeep']
    
    # Keywords that should be converted to floats
    float_keywords = []  # Add any float keywords here if needed
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split(None, 1)
            
            if len(parts) == 2:
                key = parts[0]
                value = parts[1]
                
                # Handle multiple model_dir entries
                if key == 'model_dir':
                    if key not in params:
                        params[key] = []
                    params[key].append(value)
                
                # Handle av_range specially
                elif key == 'av_range':
                    value_parts = value.split()
                    if value_parts[0] == 'absolute':
                        # Absolute range: ['absolute', min, max]
                        params[key] = ['absolute', float(value_parts[1]), float(value_parts[2])]
                    elif value_parts[0] == 'catalog':
                        # Relative to catalog: ['catalog', offset]
                        params[key] = ['catalog', float(value_parts[1])]
                
                # Handle list keywords
                elif key in list_keywords:
                    value_parts = value.split()
                    try:
                        params[key] = [float(v) for v in value_parts]
                    except ValueError:
                        params[key] = value_parts
                
                # Handle integer keywords
                elif key in int_keywords:
                    params[key] = int(value)
                
                # Handle float keywords
                elif key in float_keywords:
                    params[key] = float(value)
                
                else:
                    # Keep as single string
                    params[key] = value
    
    # If path_input_hdf5 exists, read distance range from it
    if 'path_input_hdf5' in params:
        with pd.HDFStore(params['path_input_hdf5'], 'r') as store:
            distance_range_kpc = store.get('DISTANCE_RANGE_KPC')['DISTANCE_RANGE_KPC'].to_list()
            params['distance_range_kpc'] = distance_range_kpc
    
    return params

def read_extinction_parm(info_file):
    """
    Read an extinction law information file.
    
    Parses a metadata file describing an extinction law, including the path
    to the extinction curve data file and information about how to parse it.
    
    Parameters
    ----------
    info_file : str
        Path to the .info file containing extinction law metadata.
    
    Returns
    -------
    dict
        Dictionary containing extinction law information:
        - 'name': Short-hand name of the extinction law (str)
        - 'path': Path to the extinction .par file (str)
        - 'colidx_wav': Column index for wavelength data (int)
        - 'colidx_extinction': Column index for extinction/opacity data (int)
        - 'AV_over_AK': A_V/A_K ratio if present in file (float, optional)
    
    Examples
    --------
    >>> info = read_extinction_info('extinction_law.info')
    >>> print(info['name'])
    'Draine2003'
    >>> print(info['path'])
    '/path/to/extinction_curves/draine2003.par'
    >>> print(info['AV_over_AK'])
    8.9
    
    Notes
    -----
    Example .info file format:
        # Extinction law information
        name Draine2003
        path /path/to/extinction_curves/draine2003.par
        colidx_wav 0
        colidx_extinction 1
        AV_over_AK 8.9
    """
    info = {}
    
    with open(info_file, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            # Parse key-value pairs
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                
                if key == 'name':
                    info['name'] = value
                elif key == 'path':
                    info['path'] = value
                elif key == 'colidx_wav':
                    info['colidx_wav'] = int(value)
                elif key == 'colidx_extinction':
                    info['colidx_extinction'] = int(value)
                elif key == 'AV_over_AK':
                    info['AV_over_AK'] = float(value)
    
    return info


def build_batchresults_fileroot(fitparm, startindex, endindex):
    """
    Build the file root (full path without extension) for batch results.
    
    Constructs the standardized base filepath used across all batch result files
    (HDF5 results, log files, failure lists, etc.) to ensure consistent naming.
    Includes the output directory path but not the file extension.
    
    Parameters
    ----------
    fitparm : dict
        Fit parameters dictionary containing:
        - 'dir_output': Output directory path
        - 'catalog': Catalog name (e.g., 'Serpens', 'Ophiuchus')
        - 'model_category': Model category (e.g., 'YSO', 'Galaxy')
    startindex : int
        Starting source index (inclusive) for this batch.
    endindex : int
        Ending source index (inclusive) for this batch.
    
    Returns
    -------
    str
        Full path without extension:
        {dir_output}/{catalog}_SEDFIT-{model_category}_{startindex:07d}-{endindex:07d}
    
    Examples
    --------
    >>> fitparm = {'dir_output': '/data/results',
    ...            'catalog': 'Serpens', 
    ...            'model_category': 'YSO'}
    >>> fileroot = build_batchresults_fileroot(fitparm, 0, 9999)
    >>> print(fileroot)
    /data/results/Serpens_SEDFIT-YSO_0000000-0009999
    
    >>> # Use with different extensions
    >>> hdf5_file = f"{fileroot}.hdf5"
    >>> log_file = f"{fileroot}.log"
    >>> fail_file = f"{fileroot}.fitfail"
    
    Notes
    -----
    This function provides a single source of truth for the batch file naming
    convention. All batch-related files should use this fileroot with their
    appropriate extensions appended.
    """
    filename = f"{fitparm['catalog']}_SEDFIT-{fitparm['model_category']}_{startindex:07d}-{endindex:07d}"
    return os.path.join(fitparm['dir_output'], filename)

def save_batchfit_results_hdf5(arrays, fitparm, startindex, endindex):
    """
    Save aggregated fit results to HDF5 file.
    
    Writes numpy arrays of fit results to an HDF5 file with compression.
    File naming and location are determined by fitparm dictionary entries.
    Each array is stored as a separate HDF5 dataset with the same key name.
    
    Parameters
    ----------
    arrays : dict
        Dictionary of numpy arrays from aggregate_fitresults_to_arrays.
        Expected keys:
        - 'ID': shape (nsources,), dtype string
        - 'NVALID': shape (nsources,), dtype int
        - 'MODEL_CAT': shape (nsources,), dtype string
        - 'CHI2': shape (nsources, nkeep), dtype float
        - 'AV': shape (nsources, nkeep), dtype float
        - 'SC': shape (nsources, nkeep), dtype float
        - 'MODEL_DIR': shape (nsources, nkeep), dtype string
        - 'MODEL_NAME': shape (nsources, nkeep), dtype string
        - 'MODEL_ID': shape (nsources, nkeep), dtype int
        - 'MODEL_FLUX': shape (nsources, nkeep, nbands), dtype float
        - 'IMPUTED_FLUX': shape (nsources, nkeep, nbands), dtype float
        
        Note: CATALOG dataset is automatically added from fitparm['catalog'].
    fitparm : dict
        Fit parameters dictionary containing:
        - 'dir_output': Output directory path
        - 'catalog': Catalog name (e.g., 'Serpens', 'Ophiuchus')
        - 'model_category': Model category (e.g., 'YSO', 'Galaxy')
    startindex : int
        Starting source index (inclusive) for this batch.
    endindex : int
        Ending source index (inclusive) for this batch.
    
    Returns
    -------
    str
        Full path to the created HDF5 file.
    
    Notes
    -----
    Output filename format:
        {catalog}_SEDFIT-{model_category}_{startindex:07d}-{endindex:07d}.hdf5
    
    Indices are padded to 7 digits to accommodate catalogs with up to 10 million
    sources and ensure proper alphabetical sorting of batch files.
    
    Example filenames:
        Serpens_SEDFIT-YSO_0000000-0009999.hdf5
        Ophiuchus_SEDFIT-YSO_0010000-0019999.hdf5
        Serpens_SEDFIT-YSO_0990000-0999999.hdf5
    
    All datasets are stored with gzip compression (level 4 by default) to
    reduce file size while maintaining reasonable read/write performance.
    
    Examples
    --------
    >>> arrays = aggregate_fitresults_to_arrays(fitresultlist)
    >>> fitparm = {'dir_output': '/data/results', 
    ...            'catalog': 'Serpens', 
    ...            'model_category': 'YSO'}
    >>> filepath = save_batchfit_results_hdf5(arrays, fitparm, 
    ...                                        startindex=0, endindex=9999)
    >>> print(filepath)
    /data/results/Serpens_SEDFIT-YSO_0000000-0009999.hdf5
    
    Reading back the data:
    >>> import h5py
    >>> with h5py.File(filepath, 'r') as f:
    ...     chi2 = f['CHI2'][:]  # Load all chi2 values
    ...     flux = f['MODEL_FLUX'][0, :, :]  # Load fluxes for first source
    """
    # Construct filepath using fileroot (includes directory path)
    fileroot = build_batchresults_fileroot(fitparm, startindex, endindex)
    filepath = f"{fileroot}.hdf5"
    
    # Create output directory if it doesn't exist
    os.makedirs(fitparm['dir_output'], exist_ok=True)
    
    # Write arrays to HDF5 file
    with h5py.File(filepath, 'w') as f:
        # Save the catalog name as a dataset (repeated for each source)
        nsources = len(arrays['ID'])
        catalog_array = np.full(nsources, fitparm['catalog'], dtype='S50')
        f.create_dataset('CATALOG', data=catalog_array, compression='gzip', compression_opts=4)
        
        for key, arr in arrays.items():
            # Handle string arrays - h5py doesn't support numpy unicode strings
            if arr.dtype.kind == 'U':  # Unicode string
                # Convert to object array of byte strings for h5py compatibility
                arr_bytes = arr.astype('S')
                f.create_dataset(key, data=arr_bytes, compression='gzip', compression_opts=4)
            else:
                # Numeric arrays work fine
                f.create_dataset(key, data=arr, compression='gzip', compression_opts=4)
    
    return filepath


# def load_batchfit_results_as_dataframe(filepath):
#     """
#     Load batch fit results and return as long-format DataFrame.
    
#     Converts HDF5 fit results into a pandas DataFrame with one row per fit.
#     Uses efficient numpy operations for fast loading and reshaping.
    
#     Parameters
#     ----------
#     filepath : str
#         Path to HDF5 file created by save_batchfit_results_hdf5.
    
#     Returns
#     -------
#     pandas.DataFrame
#         Long-format DataFrame with one row per fit, columns organized as:
        
#         Catalog information:
#         - catalog: Catalog/region name
        
#         Source information:
#         - id: Source identifier string
#         - nvalid: Number of valid detections
        
#         Fit information:
#         - model_cat: Model category
#         - fit_rank: Fit ranking (1=best, 2=second best, etc.)
#         - chi2: Chi-squared value
#         - av: Visual extinction A_V
#         - sc: Scale factor
#         - model_dir: Model directory path
#         - model_name: Model name
#         - model_id: Model ID
        
#         Model fluxes (one column per band):
#         - model_flux_J, model_flux_H, model_flux_Ks
#         - model_flux_3_6, model_flux_4_5, model_flux_5_8, model_flux_8_0
#         - model_flux_24
        
#         Imputed fluxes (one column per band):
#         - imp_flux_J, imp_flux_H, imp_flux_Ks
#         - imp_flux_3_6, imp_flux_4_5, imp_flux_5_8, imp_flux_8_0
#         - imp_flux_24
    
#     Examples
#     --------
#     >>> df = load_batchfit_results_as_dataframe('Serpens_SEDFIT-YSO_0000000-0009999.hdf5')
#     >>> print(df.shape)
#     (50000, 27)  # 10000 sources × 5 fits = 50000 rows, 27 columns
#     >>> 
#     >>> # Get only best fits
#     >>> best_fits = df[df['fit_rank'] == 1]
#     >>> 
#     >>> # Get all good fits (chi2 < 50)
#     >>> good_fits = df[df['chi2'] < 50]
#     >>> 
#     >>> # Access specific source's fits
#     >>> source_fits = df[df['id'] == 'J033021.04+582049.3']
#     """
#     # Band names in order
#     band_names = ['J', 'H', 'Ks', '3_6', '4_5', '5_8', '8_0', '24']
    
#     # Load all arrays from HDF5
#     with h5py.File(filepath, 'r') as f:
#         arrays = {key: f[key][:] for key in f.keys()}
    
#     # Decode byte strings to regular strings
#     string_keys = ['CATALOG', 'ID', 'MODEL_CAT', 'MODEL_DIR', 'MODEL_NAME']
#     for key in string_keys:
#         if key in arrays and arrays[key].dtype.kind == 'S':  # Byte string
#             arrays[key] = np.char.decode(arrays[key], 'utf-8')
    
#     nsources, nkeep, nbands = arrays['MODEL_FLUX'].shape
#     n_rows = nsources * nkeep
    
#     # Create fit_rank array efficiently (starting at 1)
#     fit_ranks = np.tile(np.arange(1, nkeep + 1), nsources)
    
#     # Build DataFrame dict with flattened/repeated arrays
#     # Organized in groups: source info, fit info, model fluxes, imputed fluxes
#     data = {}
    
#     # Catalog name (first column)
#     data['catalog'] = np.repeat(arrays['CATALOG'], nkeep)
    
#     # Source information
#     data['id'] = np.repeat(arrays['ID'], nkeep)
#     data['nvalid'] = np.repeat(arrays['NVALID'], nkeep)
    
#     # Fit information
#     data['model_cat'] = np.repeat(arrays['MODEL_CAT'], nkeep)
#     data['fit_rank'] = fit_ranks
#     data['chi2'] = arrays['CHI2'].ravel()
#     data['av'] = arrays['AV'].ravel()
#     data['sc'] = arrays['SC'].ravel()
#     data['model_dir'] = arrays['MODEL_DIR'].ravel()
#     data['model_name'] = arrays['MODEL_NAME'].ravel()
#     data['model_id'] = arrays['MODEL_ID'].ravel()
    
#     # Reshape 3D flux arrays to 2D for easier column extraction
#     model_flux_2d = arrays['MODEL_FLUX'].reshape(n_rows, nbands)
#     imputed_flux_2d = arrays['IMPUTED_FLUX'].reshape(n_rows, nbands)
    
#     # Add model flux columns (one per band)
#     for band_idx, band_name in enumerate(band_names):
#         data[f'model_flux_{band_name}'] = model_flux_2d[:, band_idx]
    
#     # Add imputed flux columns (one per band)
#     for band_idx, band_name in enumerate(band_names):
#         data[f'imp_flux_{band_name}'] = imputed_flux_2d[:, band_idx]
    
#     return pd.DataFrame(data)

def save_batch_failures(failed_sources, fitparm, startindex, endindex):
    """
    Save list of failed sources to text file.
    
    Creates a simple text file listing sources that failed to fit, including
    error type and message for debugging. Uses the same fileroot naming
    convention as other batch files with .fitfail extension.
    
    Parameters
    ----------
    failed_sources : list of dict
        List of failure records, each containing:
        - 'source_id': Source identifier string
        - 'error_type': Exception type name (e.g., 'ValueError')
        - 'error_message': Exception message string
    fitparm : dict
        Fit parameters dictionary containing:
        - 'dir_output': Output directory path
        - 'catalog': Catalog name
        - 'model_category': Model category
    startindex : int
        Starting source index for this batch.
    endindex : int
        Ending source index for this batch.
    
    Returns
    -------
    str or None
        Path to failure log file if any failures occurred, None if no failures.
    
    Examples
    --------
    >>> fitresultlist, failed_sources = fit.fit_batch_parallel(srclist, ...)
    >>> if len(failed_sources) > 0:
    ...     fail_path = io.save_batch_failures(failed_sources, fitparm, 0, 9999)
    ...     print(f"Failure log: {fail_path}")
    
    Notes
    -----
    Output filename format:
        {catalog}_SEDFIT-{model_category}_{startindex:07d}-{endindex:07d}.fitfail
    
    File format:
        # Header with metadata
        # One line per failure: SOURCE_ID | ERROR_TYPE | ERROR_MESSAGE
    
    Example file content:
        # Fit failures for Serpens batch 0-9999
        # Total failures: 3
        # Format: SOURCE_ID | ERROR_TYPE | ERROR_MESSAGE
        #==============================================================================
        
        J032600.97+582132.5            | ValueError           | invalid flux value in band 3
        J032701.23+581045.8            | RuntimeError         | model grid interpolation failed
    """
    # Return None if no failures
    if len(failed_sources) == 0:
        return None
    
    # Construct filepath using fileroot
    fileroot = build_batchresults_fileroot(fitparm, startindex, endindex)
    filepath = f"{fileroot}.fitfail"
    
    # Ensure output directory exists
    os.makedirs(fitparm['dir_output'], exist_ok=True)
    
    # Write failure log
    with open(filepath, 'w') as f:
        # Header
        f.write(f"# Fit failures for {fitparm['catalog']} batch {startindex}-{endindex}\n")
        f.write(f"# Total failures: {len(failed_sources)}\n")
        f.write(f"# Format: SOURCE_ID | ERROR_TYPE | ERROR_MESSAGE\n")
        f.write("#" + "="*78 + "\n\n")
        
        # Failure records
        for fail in failed_sources:
            f.write(f"{fail['source_id']:<30s} | {fail['error_type']:<20s} | {fail['error_message']}\n")
    
    return filepath

# def load_batchfit_dir(dirpaths):
#     """
#     Load and concatenate batch fit results from one or more directories.
    
#     Searches directories for HDF5 files matching the pattern:
#     <region>_SEDFIT-<model>_<start>-<end>.hdf5
    
#     Loads files in ascending order by start index and concatenates results.
    
#     Parameters
#     ----------
#     dirpaths : str or list of str
#         Path to directory (or list of directory paths) containing HDF5 result files.
#         Files should match pattern: *_SEDFIT-*_NNNNNNN-NNNNNNN.hdf5
    
#     Returns
#     -------
#     pandas.DataFrame
#         Concatenated DataFrame containing all results from all files in all
#         directories, preserving ascending order by start index within each
#         directory, then across directories in the order provided.
    
#     Examples
#     --------
#     >>> # Load from single directory
#     >>> df = load_batchfit_dir('/path/to/results')
#     >>> 
#     >>> # Load from multiple directories
#     >>> df = load_batchfit_dir(['/path/to/serpens', '/path/to/aquila'])
#     >>> 
#     >>> # Check what got loaded
#     >>> print(df['catalog'].unique())
#     """
#     # Convert single path to list for uniform processing
#     if isinstance(dirpaths, (str, Path)):
#         dirpaths = [dirpaths]
    
#     # Pattern to match filenames and extract start index
#     # Format: <region>_SEDFIT-<model>_<start>-<end>.hdf5
#     pattern = re.compile(r'.*_SEDFIT-.*_(\d{7})-\d{7}\.hdf5$')
    
#     all_dfs = []
    
#     for dirpath in dirpaths:
#         dirpath = Path(dirpath)
        
#         if not dirpath.exists():
#             raise FileNotFoundError(f"Directory not found: {dirpath}")
        
#         if not dirpath.is_dir():
#             raise NotADirectoryError(f"Not a directory: {dirpath}")
        
#         # Find all matching HDF5 files and extract their start indices
#         files_with_indices = []
#         for filepath in dirpath.glob('*.hdf5'):
#             match = pattern.match(filepath.name)
#             if match:
#                 start_index = int(match.group(1))
#                 files_with_indices.append((start_index, filepath))
        
#         if not files_with_indices:
#             print(f"Warning: No matching HDF5 files found in {dirpath}")
#             continue
        
#         # Sort by start index (ascending)
#         files_with_indices.sort(key=lambda x: x[0])
        
#         # Load each file and collect dataframes
#         for start_index, filepath in files_with_indices:
#             print(f"Loading {filepath.name}...")
#             df = load_batchfit_results_as_dataframe(str(filepath))
#             all_dfs.append(df)
    
#     # Concatenate all dataframes
#     if not all_dfs:
#         raise ValueError("No data loaded. Check that directories contain matching HDF5 files.")
    
#     result = pd.concat(all_dfs, axis=0, ignore_index=True)
#     print(f"\nLoaded {len(all_dfs)} files, total {len(result):,} rows")
    
#     return result

def load_batchfit_results_as_dataframe(filepath, max_fit_rank=None):
    """
    Load batch fit results and return as long-format DataFrame.
    
    Converts HDF5 fit results into a pandas DataFrame with one row per fit.
    Uses efficient numpy operations for fast loading and reshaping.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file created by save_batchfit_results_hdf5.
    max_fit_rank : int, optional
        Maximum fit rank to include. If provided, only fits with 
        fit_rank <= max_fit_rank are returned. If None (default), 
        all fit ranks are returned.
    
    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with one row per fit, columns organized as:
        
        Catalog information:
        - catalog: Catalog/region name
        
        Source information:
        - id: Source identifier string
        - nvalid: Number of valid detections
        
        Fit information:
        - model_cat: Model category
        - fit_rank: Fit ranking (1=best, 2=second best, etc.)
        - chi2: Chi-squared value
        - av: Visual extinction A_V
        - sc: Scale factor
        - model_dir: Model directory path
        - model_name: Model name
        - model_id: Model ID
        
        Model fluxes (one column per band):
        - model_flux_J, model_flux_H, model_flux_Ks
        - model_flux_3_6, model_flux_4_5, model_flux_5_8, model_flux_8_0
        - model_flux_24
        
        Imputed fluxes (one column per band):
        - imp_flux_J, imp_flux_H, imp_flux_Ks
        - imp_flux_3_6, imp_flux_4_5, imp_flux_5_8, imp_flux_8_0
        - imp_flux_24
    
    Examples
    --------
    >>> # Load all fits
    >>> df = load_batchfit_results_as_dataframe('Serpens_SEDFIT-YSO_0000000-0009999.hdf5')
    >>> print(df.shape)
    (50000, 27)  # 10000 sources × 5 fits = 50000 rows, 27 columns
    >>> 
    >>> # Load only best fits
    >>> best_fits = load_batchfit_results_as_dataframe('Serpens_SEDFIT-YSO_0000000-0009999.hdf5', 
    ...                                                  max_fit_rank=1)
    >>> print(best_fits.shape)
    (10000, 27)  # 10000 sources × 1 fit = 10000 rows
    >>> 
    >>> # Load top 3 fits
    >>> top3 = load_batchfit_results_as_dataframe('Serpens_SEDFIT-YSO_0000000-0009999.hdf5',
    ...                                            max_fit_rank=3)
    """
    # Band names in order
    band_names = ['J', 'H', 'Ks', '3_6', '4_5', '5_8', '8_0', '24']
    
    # Load all arrays from HDF5
    with h5py.File(filepath, 'r') as f:
        arrays = {key: f[key][:] for key in f.keys()}
    
    # Decode byte strings to regular strings
    string_keys = ['CATALOG', 'ID', 'MODEL_CAT', 'MODEL_DIR', 'MODEL_NAME']
    for key in string_keys:
        if key in arrays and arrays[key].dtype.kind == 'S':  # Byte string
            arrays[key] = np.char.decode(arrays[key], 'utf-8')
    
    nsources, nkeep, nbands = arrays['MODEL_FLUX'].shape
    
    # Determine which fit ranks to include
    if max_fit_rank is None:
        # Include all ranks
        ranks_to_include = nkeep
    else:
        # Include only ranks up to max_fit_rank
        ranks_to_include = min(max_fit_rank, nkeep)
    
    n_rows = nsources * ranks_to_include
    
    # Create fit_rank array efficiently (starting at 1)
    fit_ranks = np.tile(np.arange(1, ranks_to_include + 1), nsources)
    
    # Build DataFrame dict with flattened/repeated arrays
    # Organized in groups: source info, fit info, model fluxes, imputed fluxes
    data = {}
    
    # Catalog name (first column)
    data['catalog'] = np.repeat(arrays['CATALOG'], ranks_to_include)
    
    # Source information
    data['id'] = np.repeat(arrays['ID'], ranks_to_include)
    data['nvalid'] = np.repeat(arrays['NVALID'], ranks_to_include)
    
    # Fit information
    data['model_cat'] = np.repeat(arrays['MODEL_CAT'], ranks_to_include)
    data['fit_rank'] = fit_ranks
    # Slice arrays to only include requested ranks
    data['chi2'] = arrays['CHI2'][:, :ranks_to_include].ravel()
    data['av'] = arrays['AV'][:, :ranks_to_include].ravel()
    data['sc'] = arrays['SC'][:, :ranks_to_include].ravel()
    data['model_dir'] = arrays['MODEL_DIR'][:, :ranks_to_include].ravel()
    data['model_name'] = arrays['MODEL_NAME'][:, :ranks_to_include].ravel()
    data['model_id'] = arrays['MODEL_ID'][:, :ranks_to_include].ravel()
    
    # Reshape 3D flux arrays to 2D for easier column extraction
    # Only include the requested ranks
    model_flux_2d = arrays['MODEL_FLUX'][:, :ranks_to_include, :].reshape(n_rows, nbands)
    imputed_flux_2d = arrays['IMPUTED_FLUX'][:, :ranks_to_include, :].reshape(n_rows, nbands)
    
    # Add model flux columns (one per band)
    for band_idx, band_name in enumerate(band_names):
        data[f'model_flux_{band_name}'] = model_flux_2d[:, band_idx]
    
    # Add imputed flux columns (one per band)
    for band_idx, band_name in enumerate(band_names):
        data[f'imp_flux_{band_name}'] = imputed_flux_2d[:, band_idx]
    
    return pd.DataFrame(data)

# def load_batchfit_dir(dirpaths, max_fit_rank=None):
#     """
#     Load and concatenate batch fit results from one or more directories.
    
#     Searches directories for HDF5 files matching the pattern:
#     <region>_SEDFIT-<model>_<start>-<end>.hdf5
    
#     Loads files in ascending order by start index and concatenates results.
    
#     Parameters
#     ----------
#     dirpaths : str or list of str
#         Path to directory (or list of directory paths) containing HDF5 result files.
#         Files should match pattern: *_SEDFIT-*_NNNNNNN-NNNNNNN.hdf5
#     max_fit_rank : int, optional
#         Maximum fit rank to include. If provided, only fits with 
#         fit_rank <= max_fit_rank are returned. If None (default), 
#         all fit ranks are returned.
    
#     Returns
#     -------
#     pandas.DataFrame
#         Concatenated DataFrame containing all results from all files in all
#         directories, preserving ascending order by start index within each
#         directory, then across directories in the order provided.
    
#     Examples
#     --------
#     >>> # Load all fits from single directory
#     >>> df = load_batchfit_dir('/path/to/results')
#     >>> 
#     >>> # Load only best fits from multiple directories
#     >>> df = load_batchfit_dir(['/path/to/serpens', '/path/to/aquila'], 
#     ...                         max_fit_rank=1)
#     >>> 
#     >>> # Load top 3 fits
#     >>> df = load_batchfit_dir('/path/to/results', max_fit_rank=3)
#     >>> 
#     >>> # Check what got loaded
#     >>> print(df['catalog'].unique())
#     >>> print(df['fit_rank'].unique())
#     """
#     # Convert single path to list for uniform processing
#     if isinstance(dirpaths, (str, Path)):
#         dirpaths = [dirpaths]
    
#     # Pattern to match filenames and extract start index
#     # Format: <region>_SEDFIT-<model>_<start>-<end>.hdf5
#     pattern = re.compile(r'.*_SEDFIT-.*_(\d{7})-\d{7}\.hdf5$')
    
#     all_dfs = []
    
#     for dirpath in dirpaths:
#         dirpath = Path(dirpath)
        
#         if not dirpath.exists():
#             raise FileNotFoundError(f"Directory not found: {dirpath}")
        
#         if not dirpath.is_dir():
#             raise NotADirectoryError(f"Not a directory: {dirpath}")
        
#         # Find all matching HDF5 files and extract their start indices
#         files_with_indices = []
#         for filepath in dirpath.glob('*.hdf5'):
#             match = pattern.match(filepath.name)
#             if match:
#                 start_index = int(match.group(1))
#                 files_with_indices.append((start_index, filepath))
        
#         if not files_with_indices:
#             print(f"Warning: No matching HDF5 files found in {dirpath}")
#             continue
        
#         # Sort by start index (ascending)
#         files_with_indices.sort(key=lambda x: x[0])
        
#         # Load each file and collect dataframes
#         for start_index, filepath in files_with_indices:
#             print(f"Loading {filepath.name}...")
#             df = load_batchfit_results_as_dataframe(str(filepath), max_fit_rank=max_fit_rank)
#             all_dfs.append(df)
    
#     # Concatenate all dataframes
#     if not all_dfs:
#         raise ValueError("No data loaded. Check that directories contain matching HDF5 files.")
    
#     result = pd.concat(all_dfs, axis=0, ignore_index=True)
    
#     rank_info = f" (fit_rank <= {max_fit_rank})" if max_fit_rank is not None else " (all ranks)"
#     print(f"\nLoaded {len(all_dfs)} files{rank_info}, total {len(result):,} rows")
    
#     return result



def load_batchfit_dir(dirpaths, max_fit_rank=None, n_workers=None, verbose=True):
    """
    Load and concatenate batch fit results from one or more directories.
    
    Parameters
    ----------
    dirpaths : str or list of str
        Path to directory (or list of directory paths) containing HDF5 result files.
    max_fit_rank : int, optional
        Maximum fit rank to include. If None, all fit ranks are returned.
    n_workers : int, optional
        Number of parallel workers. If None, uses all CPU cores. Set to 1 for sequential.
    verbose : bool, optional
        Print progress information.
    """
    if isinstance(dirpaths, (str, Path)):
        dirpaths = [dirpaths]
    
    pattern = re.compile(r'.*_SEDFIT-.*_(\d{7})-\d{7}\.hdf5$')
    
    # Collect all files with their start indices
    all_files_with_indices = []
    for dirpath in dirpaths:
        dirpath = Path(dirpath)
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Not a directory: {dirpath}")
        
        for filepath in dirpath.glob('*.hdf5'):
            match = pattern.match(filepath.name)
            if match:
                start_index = int(match.group(1))
                all_files_with_indices.append((start_index, filepath))
    
    if not all_files_with_indices:
        raise ValueError("No matching HDF5 files found")
    
    # Sort by start index
    all_files_with_indices.sort(key=lambda x: x[0])
    
    if verbose:
        print(f"Found {len(all_files_with_indices)} files to load")
    
    # Pre-allocate list for results (preserves order)
    results = [None] * len(all_files_with_indices)
    
    if n_workers == 1:
        # Sequential loading
        for i, (start_index, filepath) in enumerate(all_files_with_indices):
            if verbose:
                print(f"Loading {filepath.name} ({i+1}/{len(all_files_with_indices)})...")
            results[i] = load_batchfit_results_as_dataframe(str(filepath), max_fit_rank=max_fit_rank)
    else:
        # Parallel loading
        if verbose:
            print(f"Loading in parallel with {n_workers or 'auto'} workers...")
        
        load_func = partial(load_batchfit_results_as_dataframe, max_fit_rank=max_fit_rank)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Map each future to its index in the results list
            future_to_idx = {
                executor.submit(load_func, str(filepath)): i
                for i, (start_index, filepath) in enumerate(all_files_with_indices)
            }
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                filepath = all_files_with_indices[idx][1]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if verbose:
                        print(f"Loaded {filepath.name} ({completed}/{len(all_files_with_indices)})")
                except Exception as exc:
                    print(f"Error loading {filepath.name}: {exc}")
                    raise
    
    # Single concatenation at the end
    if verbose:
        print(f"\nConcatenating {len(results)} dataframes...")
    
    final_result = pd.concat(results, axis=0, ignore_index=True)
    
    if verbose:
        print(f"Final result: {len(final_result):,} rows")
        if 'catalog' in final_result.columns:
            catalogs = final_result['catalog'].unique()
            print(f"Catalogs: {catalogs}")
    
    return final_result
    """Load files in parallel with chunked concatenation."""
    load_func = partial(load_batchfit_results_as_dataframe, max_fit_rank=max_fit_rank)
    
    if chunk_size is None:
        # Load all in parallel, concatenate at end
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(load_func, str(filepath)): (start_index, filepath)
                for start_index, filepath in files_with_indices
            }
            
            results_dict = {}
            completed = 0
            for future in as_completed(future_to_index):
                start_index, filepath = future_to_index[future]
                try:
                    df = future.result()
                    results_dict[start_index] = df
                    completed += 1
                    if verbose:
                        print(f"Loaded {filepath.name} ({completed}/{len(files_with_indices)})")
                except Exception as exc:
                    print(f"Error loading {filepath.name}: {exc}")
                    raise
        
        all_dfs = [results_dict[start_index] for start_index, _ in files_with_indices]
        return pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Chunked parallel loading
    result_chunks = []
    
    # Process files in chunks
    num_chunks = (len(files_with_indices) + chunk_size - 1) // chunk_size
    for chunk_idx, chunk_start in enumerate(range(0, len(files_with_indices), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(files_with_indices))
        chunk_files = files_with_indices[chunk_start:chunk_end]
        
        if verbose:
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} "
                  f"(files {chunk_start+1}-{chunk_end})...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(load_func, str(filepath)): (start_index, filepath)
                for start_index, filepath in chunk_files
            }
            
            chunk_results_dict = {}
            completed = 0
            for future in as_completed(future_to_index):
                start_index, filepath = future_to_index[future]
                try:
                    df = future.result()
                    chunk_results_dict[start_index] = df
                    completed += 1
                    if verbose:
                        print(f"  Loaded {filepath.name} ({completed}/{len(chunk_files)}, "
                              f"{len(df):,} rows)")
                except Exception as exc:
                    print(f"  Error loading {filepath.name}: {exc}")
                    raise
        
        # Concatenate this chunk in the correct order
        chunk_dfs = [chunk_results_dict[start_index] for start_index, _ in chunk_files]
        chunk_df = pd.concat(chunk_dfs, axis=0, ignore_index=True)
        
        if verbose:
            print(f"  Concatenated chunk {chunk_idx + 1} into {len(chunk_df):,} rows")
            if 'catalog' in chunk_df.columns:
                cats = chunk_df['catalog'].unique()
                print(f"  Catalogs in this chunk: {cats}")
        
        # Append BEFORE deleting anything
        result_chunks.append(chunk_df)
        
        # Now delete to free memory - but DON'T delete chunk_df since it's in the list
        del chunk_dfs, chunk_results_dict
    
    # Final concatenation
    if verbose:
        print(f"\nPerforming final concatenation of {len(result_chunks)} chunks...")
        print(f"Chunk shapes: {[len(chunk) for chunk in result_chunks]}")
    
    final_result = pd.concat(result_chunks, axis=0, ignore_index=True)
    
    if verbose:
        print(f"Final concatenation complete: {len(final_result):,} rows")
    
    return final_result
    """Load files in parallel with chunked concatenation."""
    load_func = partial(load_batchfit_results_as_dataframe, max_fit_rank=max_fit_rank)
    
    if chunk_size is None:
        # Load all in parallel, concatenate at end
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(load_func, str(filepath)): (start_index, filepath)
                for start_index, filepath in files_with_indices
            }
            
            results_dict = {}
            completed = 0
            for future in as_completed(future_to_index):
                start_index, filepath = future_to_index[future]
                try:
                    df = future.result()
                    results_dict[start_index] = df
                    completed += 1
                    if verbose:
                        print(f"Loaded {filepath.name} ({completed}/{len(files_with_indices)})")
                except Exception as exc:
                    print(f"Error loading {filepath.name}: {exc}")
                    raise
        
        all_dfs = [results_dict[start_index] for start_index, _ in files_with_indices]
        return pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Chunked parallel loading
    result_chunks = []
    
    # Process files in chunks
    num_chunks = (len(files_with_indices) + chunk_size - 1) // chunk_size
    for chunk_idx, chunk_start in enumerate(range(0, len(files_with_indices), chunk_size)):
        chunk_end = min(chunk_start + chunk_size, len(files_with_indices))
        chunk_files = files_with_indices[chunk_start:chunk_end]
        
        if verbose:
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} "
                  f"(files {chunk_start+1}-{chunk_end})...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(load_func, str(filepath)): (start_index, filepath)
                for start_index, filepath in chunk_files
            }
            
            chunk_results_dict = {}
            completed = 0
            for future in as_completed(future_to_index):
                start_index, filepath = future_to_index[future]
                try:
                    df = future.result()
                    chunk_results_dict[start_index] = df
                    completed += 1
                    if verbose:
                        print(f"  Loaded {filepath.name} ({completed}/{len(chunk_files)}, "
                              f"{len(df):,} rows)")
                except Exception as exc:
                    print(f"  Error loading {filepath.name}: {exc}")
                    raise
        
        # Concatenate this chunk in the correct order
        chunk_dfs = [chunk_results_dict[start_index] for start_index, _ in chunk_files]
        chunk_df = pd.concat(chunk_dfs, axis=0, ignore_index=True)
        result_chunks.append(chunk_df)
        if verbose:
            print(f"  Concatenated chunk {chunk_idx + 1} into {len(chunk_df):,} rows")
            if 'catalog' in chunk_df.columns:
                cats = chunk_df['catalog'].unique()
                print(f"  Catalogs in this chunk: {cats}")
        
        # Explicitly delete to free memory
        del chunk_dfs, chunk_results_dict
    
    # Final concatenation
    if verbose:
        print(f"\nPerforming final concatenation of {len(result_chunks)} chunks...")
    final_result = pd.concat(result_chunks, axis=0, ignore_index=True)
    
    return final_result