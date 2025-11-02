from sesnasedfit import io, fit 

## Inputs 
path_fitparm = "/Users/jtaylor/Dropbox/Research/SESNA_SEDFit_v2/fit_parm/yso/fit_yso_AFGL 490.parm"
startindex = 0 
endindex = 999

## Read parameter files
fitparm = io.read_fit_parm(path_fitparm) 
extparm = io.read_extinction_parm(fitparm['path_extinction_parm'])

## Read in list of source IDs from startindex to endindex 
srclist = io.read_SESNA_SEDFIT_ascii(fitparm['path_input_hdf5'], startindex, endindex)

# Parallel fit
fitresultlist, failed_sources = fit.fit_batch_parallel(srclist, fitparm, extparm, startindex, endindex, n_workers=10)

# Save successful fits
if len(fitresultlist) > 0:
    arrays = fit.aggregate_fitresults_to_arrays(fitresultlist)
    io.save_batchfit_results_hdf5(arrays, fitparm, startindex, endindex)

# Save failure log
if len(failed_sources) > 0:
    fail_path = io.save_batch_failures(failed_sources, fitparm, startindex, endindex)
    print(f"Failure log saved: {fail_path}")