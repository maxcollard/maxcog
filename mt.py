#
# ...

## Imports

import numpy as np
import scipy.fftpack

import maxcog.mtm_partial


## Functions

def mt_csd ( x, y, nw = 3.0, n_tapers = 6, fs = 1000.0 ):
    '''mt_csd - ...'''
    
    if not len( x ) == len( y ):
        return None
    
    h_tapers, sum_tapers = maxcog.mtm_partial._other_dpss_method( len( x ), nw, n_tapers )
    
    ret = np.zeros( x.shape )
    
    for i in range( n_tapers ):
        xf_cur = scipy.fftpack.fft( x * h_tapers[i] )
        yf_cur = scipy.fftpack.fft( y * h_tapers[i] )
        ret = ret + conj( xf_cur ) * yf_cur
    
    return ret * ( fs / n_tapers )

def mt_psd ( x, **params ):
    '''mt_psd - ...'''
    return mt_csd( x, x, **params )

def mt_coh ( x, y, **params ):
    # TODO: Doesn't work very well ...
    '''mt_coh -  ...'''
    num = np.abs( mt_csd( x, y, **params ) ) ** 2
    den = mt_psd( x, **params ) * mt_psd( y, **params )
    return num / den