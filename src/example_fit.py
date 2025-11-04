import pandas as pd 
from sesnasedfit import io, fit 

## Inputs 
path_fitparm = "/Users/jtaylor/Dropbox/Research/SESNA_SEDFit_v2/fit_parm/yso/fit_yso_AFGL 490.parm"
source_startindex = 0 
source_endindex = 100

## Read parameter files
fitparm = io.read_fit_parm(path_fitparm) 
extparm = io.read_extinction_parm(fitparm['path_extinction_parm'])

## Read in list of source IDs from startindex to endindex 
srclist = io.read_SESNA_SEDFIT_ascii(fitparm['path_input_hdf5'], source_startindex, source_endindex)

## Load fitter object(s)
fitterlist = fit.load_fitterobj(fitparm, extparm, default_av_range=[0., 100.])

## Work out how many sources we are fitting in this loop 
nsources_to_fit = source_endindex - source_startindex + 1 
fitresultlist = [None]*nsources_to_fit
source_curindex = source_startindex
nsources_fitted = 0 

## Fitting loop 
for src in srclist: # Iterate through the dataframe rows
    
    if (nsources_fitted % 10) == 0:
        time = pd.Timestamp.now()
        print(f"Fitting source at index:{source_curindex:10.0f} at time: {time}")

    fitresultlist[nsources_fitted] = fit.fit_source(src, fitterlist, fitparm, extparm, fitparm['nkeep'])

    # Increment 
    source_curindex += 1
    nsources_fitted += 1
    
## Saving 
fitarr = fit.aggregate_fitresults_to_arrays(fitresultlist)
io.save_batchfit_results_hdf5(fitarr, fitparm, startindex=source_startindex, endindex=source_endindex)

## Loading 
h5_filepath = io.build_batchresults_filepath(fitparm, source_startindex, source_endindex)
loadres = io.load_batchfit_results_as_dataframe(h5_filepath)

def get_deep_size(obj, seen=None):
    """Recursively calculates the size of an object and its contents."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Already counted
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_deep_size(key, seen) for key in obj)
        size += sum(get_deep_size(value, seen) for value in obj.values())
    # Add more conditions for other container types if needed

    return size