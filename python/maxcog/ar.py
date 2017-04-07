#
# ...

## Imports

import numpy as np
import sklearn.linear_model as lm

# Kludge to make independent of pbar
try:
    from pbar import pbar
except ImportError:
    pbar = lambda x : x


## Functions

def _kernel_gaussian ( x, h ):
    return np.exp( -(x ** 2) / h )

def _tvdbn_kernel ( ts, t0, h ):
    s = np.sum( _kernel_gaussian( ts, h ) )
    return (1/s) * _kernel_gaussian( ts - t0, h )

def tvdbn_labeled( X, **kwargs ):
    pass

def tvdbn ( X,
            step = 1,
            predict_lag = 1,
            reg_weight = 1.0,
            kernel_width = 4.0,
            kernel_thresh = 0.001,
            show_pbar = False ):
    ''' ... '''
    
    # Shortcuts
    T = X.shape[0]
    N = X.shape[1]
    ts = np.arange( T )
    
    # Pre-allocate output
    t_out = np.arange( 1, T, step )
    ret = np.zeros( (N, N, t_out.shape[0]) )
    
    # TODO Correct mapping between lambda and alpha??
    model = lm.Ridge( alpha = reg_weight )
    
    t_iterator = pbar( range( t_out.shape[0] ) ) if show_pbar else range( t_out.shape[0] )
    for i_t in t_iterator:
        t = t_out[i_t]

        # Compute weighting kernel
        cur_kernel = _tvdbn_kernel( ts, t, kernel_width )
        # Create slicing vectors for present and past, using threshold to cut fat
        kernel_slice = np.nonzero( cur_kernel[predict_lag:] > kernel_thresh )[0]
        kernel_slice_prev = kernel_slice - predict_lag
        
        for ch in range( N ):
            # "All other channels"
            ch_slice = np.arange( N ) != ch
            
            # Perform regression
            # TODO Don't know why I have to double-index like this
            model.fit( X[kernel_slice_prev,:][:,ch_slice],
                       X[kernel_slice,:][:,ch],
                       sample_weight = cur_kernel[kernel_slice] )
            
            # Add regression output to return value
            params = model.coef_
            ret[ch, ch_slice, i_t] = params
    
    return ret