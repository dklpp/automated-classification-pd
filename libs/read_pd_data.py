from scipy.io import loadmat

import numpy as np
import mne, glob 
import pandas as pd

def readMatData(subjectType, subjectNum, trials='trial_norm', verbose=False):
    """
    Input:
    - subjectType: 'ctrl' or 'pd'
    - subjectNumb: number of subject in a group. Min: 0, Max: 19
    - trials: 'trial_norm' default or 'trial' (not normalized)
    - verbose: print the shape of a signal
    Output:
    - signal: pd.DataFrame, columns=channels, rows=signals
    - final_t: nd.array, time-vector
    - final_y: nd.array, signal-vector
    """
    print('Reading files started..')
    pd_list = glob.glob('data/PD_broad/*.mat')
    ctrl_list = glob.glob('data/_CTRL_PD_broad/*.mat')
    if subjectType == 'pd':
        matlabfile = pd_list[subjectNum] 
        if verbose:
            print('Loading file...', pd_list[subjectNum])
    elif subjectType == 'ctrl':
        matlabfile = ctrl_list[subjectNum]
        if verbose:
            print('Loading file...', ctrl_list[subjectNum])

    loadedfile = loadmat(matlabfile)
    
    dat = loadedfile['dati_bf']
    t = dat['time'].flatten()[0].flatten()
    y = dat[trials].flatten()[0].flatten() 
    
    #print('Timepoints are divided into #timepoints:', len(t))
    all_timepoints = [t[i] for i in range(len(t))] # for the case when vectors are separated in several
    final_t = np.hstack(all_timepoints).reshape(-1)
    
    #print('Signal is divided into #signals:', len(y))
    all_signals = [y[i] for i in range(len(y))]
    final_y = np.hstack(all_signals)
        
    sign = pd.DataFrame(data=final_y.T, index=final_t)
    if verbose:
        print('Signal is of shape:', sign.shape)
    
    return sign, final_t, final_y