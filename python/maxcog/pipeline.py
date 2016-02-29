##!/usr/bin/env python

#==============================================================================#
# maxcog
# v. 0.0.1
# 
# Home of the LabeledArray.
# 
# 2016 - Maxwell J. Collard
#==============================================================================#

"""Functions for common tasks performed on LabeledArrays."""


## === IMPORTS === ##


# See __init__.py

from maxcog import LabeledArray

from collections import OrderedDict

import numpy                        as np
import scipy.signal                 as sig
import scipy.stats.distributions    as dist


## === MAIN === ##


# TODO Add unit tests

if __name__ == '__main__':
    main()

def main ():
    pass


## === SIGNAL PROCESSING === ##
# Generalizes common methods from scipy, numpy, etc. to work with LabeledArrays.


# TODO Should modify these to work on individual trials, to avoid scipy's terrible memory management

def filtfilt_labeled ( b, a, x, **kwargs ):
    '''Labeled analogue of scipy.signal.filtfilt; runs along x's 'time' axis.

    Inputs:
    b, a - Filter coefficients passed to filtfilt
    x - LabeledArray containing the data; must have a 'time' axis
    **kwargs - (Optional) Keyword arguments to pass to filtfilt

    Output is a LabeledArray with the same axes as x.
    '''

    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )
    
    # Filter along time axis
    ret_array = sig.filtfilt( b, a, x.array,
                              axis = x.axis_index( 'time' ),
                              **kwargs )
    # Maintain same axes as original
    ret_axes = OrderedDict( x.axes )
    
    return LabeledArray( array = ret_array, axes = ret_axes )


def decimate_labeled ( x, q, **kwargs ):
    '''Labeled analogue of scipy.signal.decimate; runs along x's 'time' axis.

    Inputs:
    x - LabeledArray containing the data; must have a 'time' axis
    q - Decimation factor passed to decimate
    **kwargs - (Optional) Keyword arguments passed to decimate

    Output is a LabeledArray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
    
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )

    time_axis = x.axis_index( 'time' )
    
    # Decimate
    ret_array = sig.decimate( x.array, q,
                              axis = time_axis,
                              **kwargs )
    
    # Form new axes
    ret_axes = OrderedDict( x.axes )
    ret_time = np.linspace( x.axes['time'][0], x.axes['time'][-1], ret_array.shape[time_axis] )
    ret_axes['time'] = ret_time
    
    return LabeledArray( array = ret_array, axes = ret_axes )

def timefreq_fft ( x, **kwargs ):
    '''Labeled analogue of scipy.signal.spectrogram; runs along x's 'time' axis.

    Inputs:
    x - LabeledArray containing the data; must have a 'time' axis, and probably
        shouldn't have a 'frequency' axis
    **kwargs - (Optional) Keyword arguments passed to spectrogram

    Output is a LabeledArray with the same axes as x except for:
        'time' - Determined by the window properties; uses the values returned
            by spectrogram
        'frequency' - Added based on the values returned by spectrogram
    '''
    
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )

    time_axis = x.axis_index( 'time' )
    other_axes = [ k
                   for k in x.axes.keys()
                   if not k == 'time' ]
    
    ## Compute spectrograms
    
    # Test with first sample along each non-time axis
    test_slice = functools.reduce( lambda a, b : a + b,
                                  tuple( zip( other_axes,
                                              ( 0 for axis in other_axes ) ) ) )
    f_spec, t_spec, ret_test = sig.spectrogram( x[test_slice], **kwargs )
    
    # Form new axes
    ret_axes = OrderedDict( x.axes )
    ret_axes['time'] = t_spec + x.axes['time'][0]
    ret_axes.move_to_end( 'time', last = True )
    ret_axes['frequency'] = f_spec
    ret_axes.move_to_end( 'frequency', last = False )
    # Allocate full result
    ret = LabeledArray( axes = ret_axes )
    
    # Compute for each trial
    for x_cur, i in x.iter_over( other_axes, return_index = True ):
        cur_slice = functools.reduce( lambda a, b : a + b,
                                      tuple( zip( other_axes,
                                                  i ) ) )
        f_cur, t_cur, ret_cur = sig.spectrogram( x_cur.array, **kwargs )
        ret[cur_slice] = ret_cur
    
    return ret

def map_labeled ( f, x ):
    '''Maps each element of the LabeledArray x through a function f.

    Inputs:
    x - LabeledArray containing the data to act on
    f - A one-argument function that works elementwise on a numpy ndarray

    Output is a LabeledArray with the same axes as x.
    '''

    ret_array = f( x.array )
    ret_axes = OrderedDict( x.axes )
    return LabeledArray( array = ret_array, axes = ret_axes )

def mean_labeled ( x, axis = None ):
    '''Labeled analogue of np.mean, running along the specified axis(es).

    Inputs:
    x - LabeledArray containing the data to average
    axis - (Optional) A string or sequence of strings, specifying the name(s) of
        the axis(es) to average over
        Default: Runs over the last axis in x.axes.keys()

    Output is a LabeledArray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
    
    ret_axes = OrderedDict( x.axes )
    
    # Default: Use last axis
    if axis is None:
        axis = list( ret_axes.keys() )[-1]
    # Promote strings to lists for convenience
    if isinstance( axis, str ):
        axis = [axis]

    # Strip out the axes we average over from the return value
    for ax in axis:
        ret_axes.pop( ax, None )
    
    axis_idx = tuple( x.axis_index( ax )
                      for ax in axis )
    ret_array = np.mean( x.array, axis = axis_idx )
    
    return LabeledArray( array = ret_array, axes = ret_axes )

def std_labeled ( x, axis = None ):
    '''Labeled analogue of np.std, running along the specified axis(es).

    Inputs:
    x - LabeledArray containing the data
    axis - (Optional) A string or sequence of strings, specifying the name(s) of
        the axis(es) to take the standard deviation over
        Default: Runs over the last axis in x.axes.keys()

    Output is a LabeledArray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
    
    ret_axes = OrderedDict( x.axes )
    
    # Default: Use last axis
    if axis is None:
        axis = list( ret_axes.keys() )[-1]
    # Promote strings to lists for convenience
    if isinstance( axis, str ):
        axis = [axis]

    # Strip out the axes we average over from the return value
    for ax in axis:
        ret_axes.pop( ax, None )
    
    axis_idx = tuple( x.axis_index( ax )
                      for ax in axis )
    ret_array = np.std( x.array, axis = axis_idx )
    
    return LabeledArray( array = ret_array, axes = ret_axes )

def baseline_normalize ( x, window, return_baseline = False ):
    '''Normalizes the data in x using the mean and standard deviation from the
    time points within window, aggregated across the 'trial' axis if present.
    Each non-'trial' feature in x is normalized individually.

    Inputs:
    x - LabeledArray containing data to be normalized. Must have a 'time' axis
    window - 2-tuple specifying the start and end time of the baseline window.
        This is used for LabeledArray slicing, so it's a "right-open" interval,
        i.e. [start, end)
    return_baseline - (Optional) If True, will also return the baseline mean
        and standard deviation
        Default: False

    Output depends upon the context:
        If return_baseline is False, output is a LabeledArray with the same axes
            as x.
        If return_baseline is True, output is a tuple:
                (x_normalized, baseline_mu, baseline_sig)
            where baseline_mu and baseline_sig are LabeledArrays with the same
            axes as x except for 'time' (and 'trial' if present), which are
            collapsed when forming the baseline distributions.
    '''

    # Slice out baseline time window
    baseline_data = x['time', window, 'labeled']
    
    axis_combine = ( 'time', 'trial' ) if 'trial' in x.axes.keys() else ('time',)
        
    baseline_mu = mean_labeled( baseline_data, axis = axis_combine )
    baseline_sig = std_labeled( baseline_data, axis = axis_combine )
    
    # Allocate zeros
    ret = LabeledArray( axes = x.axes )
    # Apply normalization to data
    for xi, i in x.iter_over( axis_combine, return_index = True ):
        # TODO Kludge
        if not isinstance( i, collections.Sequence ):
            i = (i,)
        i_slice = tuple( functools.reduce( lambda a, b : a + b, zip( axis_combine, i ) ) )
        ret[i_slice] = ( xi.array - baseline_mu.array ) / baseline_sig.array
    
    if return_baseline:
        return ret, baseline_mu, baseline_sig
    else:
        return ret

















