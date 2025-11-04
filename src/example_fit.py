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
## Saving 
fitarr = fit.aggregate_fitresults_to_arrays(fitresultlist)
io.save_batchfit_results_hdf5(fitarr, fitparm, startindex=source_startindex, endindex=source_endindex)

## Loading 
h5_filepath = io.build_batchresults_fileroot(fitparm, source_startindex, source_endindex) + ".hdf5"
loadres = io.load_batchfit_results_as_dataframe(h5_filepath)




def get_deep_size(obj, seen=None):
    """Recursively calculates size of object and its contents."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) 
                   for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    
    return size

size_bytes = get_deep_size(fitterlist)
size_mb = size_bytes / (1024**2)
print(f"Deep size: {size_mb:.2f} MB")



from sedfitter.source import Source
s = Source.from_ascii(srclist[0]['SOURCE_ASCII'])
print(dir(s))

info = fitterlist[0].fit(s)
