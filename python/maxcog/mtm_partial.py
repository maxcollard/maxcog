### RIPPED CODE:
# Taken from
# http://janroman.dhis.org/AFI/Python/spectrum/src/spectrum/mtm.py

import numpy as np
#import ctypes
#from ctypes import *
#import ctypes as C
import platform
import os
#from ctypes import POINTER
#import os
#from os.path import join as pj
#from spectrum.tools import nextpow2
#from pylab import semilogy

def _other_dpss_method(N, NW, Kmax):
    """Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N. 

    See dpss function that is the official version. This version is indepedant 
    of the C code and relies on Scipy function. However, it is slower by a factor 3

    Tridiagonal form of DPSS calculation from:

    """
    # here we want to set up an optimization problem to find a sequence
    # whose energy is maximally concentrated within band [-W,W].
    # Thus, the measure lambda(T,W) is the ratio between the energy within
    # that band, and the total energy. This leads to the eigen-system
    # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
    # eigenvalue is the sequence with maximally concentrated energy. The
    # collection of eigenvectors of this system are called Slepian sequences,
    # or discrete prolate spheroidal sequences (DPSS). Only the first K,
    # K = 2NW/dt orders of DPSS will exhibit good spectral concentration
    # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]
    
    # Here I set up an alternative symmetric tri-diagonal eigenvalue problem
    # such that
    # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
    # and the first off-diangonal = t(N-t)/2, t=[1,2,...,N-1]
    # [see Percival and Walden, 1993]
    from scipy import linalg as la
    Kmax = int(Kmax)
    W = float(NW)/N
    ab = np.zeros((2,N), 'd')
    nidx = np.arange(N)
    ab[0,1:] = nidx[1:]*(N-nidx[1:])/2.
    ab[1] = ((N-1-2*nidx)/2.)**2 * np.cos(2*np.pi*W)
    # only calculate the highest Kmax-1 eigenvectors
    l,v = la.eig_banded(ab, select='i', select_range=(N-Kmax, N-1))
    dpss = v.transpose()[::-1]

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2*i] *= -1
    fix_skew = (dpss[1::2,1] < 0)
    for i, f in enumerate(fix_skew):
        if f:
            dpss[2*i+1] *= -1

    # Now find the eigenvalues of the original 
    # Use the autocovariance sequence technique from Percival and Walden, 1993
    # pg 390
    # XXX : why debias false? it's all messed up o.w., even with means
    # on the order of 1e-2
    acvs = _autocov(dpss, debias=False) * N
    r = 4*W*np.sinc(2*W*nidx)
    r[0] = 2*W
    eigvals = np.dot(acvs, r)
    return dpss, eigvals


def _autocov(s, **kwargs):
    """Returns the autocovariance of signal s at all lags.

    Adheres to the definition
    sxx[k] = E{S[n]S[n+k]} = cov{S[n],S[n+k]}
    where E{} is the expectation operator, and S is a zero mean process
    """
    # only remove the mean once, if needed
    debias = kwargs.pop('debias', True)
    axis = kwargs.get('axis', -1)
    if debias:
        s = remove_bias(s, axis)
    kwargs['debias'] = False
    return _crosscov(s, s, **kwargs)


def _crosscov(x, y, axis=-1, all_lags=False, debias=True):
    """Returns the crosscovariance sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters


    x: ndarray
    y: ndarray
    axis: time axis

    all_lags: {True/False}
    whether to return all nonzero lags, or to clip the length of s_xy
    to be the length of x and y. If False, then the zero lag covariance
    is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    debias: {True/False}
    Always removes an estimate of the mean along the axis, unless
    told not to.


    cross covariance is defined as
    sxy[k] := E{X[t]*Y[t+k]}, where X,Y are zero mean random processes
    """
    if x.shape[axis] != y.shape[axis]:
        raise ValueError(
            'crosscov() only works on same-length sequences for now'
            )
    if debias:
        x = _remove_bias(x, axis)
        y = _remove_bias(y, axis)
    slicing = [slice(d) for d in x.shape]
    slicing[axis] = slice(None,None,-1)
    sxy = _fftconvolve(x, y[tuple(slicing)], axis=axis, mode='full')
    N = x.shape[axis]
    sxy /= N
    if all_lags:
        return sxy
    slicing[axis] = slice(N-1,2*N-1)
    return sxy[tuple(slicing)]
    
def _crosscorr(x, y, **kwargs):
    """
    Returns the crosscorrelation sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters


    x: ndarray
    y: ndarray
    axis: time axis
    all_lags: {True/False}
    whether to return all nonzero lags, or to clip the length of r_xy
    to be the length of x and y. If False, then the zero lag correlation
    is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Notes


    cross correlation is defined as
    rxy[k] := E{X[t]*Y[t+k]}/(E{X*X}E{Y*Y})**.5,
    where X,Y are zero mean random processes. It is the noramlized cross
    covariance.
    """
    sxy = _crosscov(x, y, **kwargs)
    # estimate sigma_x, sigma_y to normalize
    sx = np.std(x)
    sy = np.std(y)
    return sxy/(sx*sy)

def _remove_bias(x, axis):
    "Subtracts an estimate of the mean from signal x at axis"
    padded_slice = [slice(d) for d in x.shape]
    padded_slice[axis] = np.newaxis
    mn = np.mean(x, axis=axis)
    return x - mn[tuple(padded_slice)]


def _fftconvolve(in1, in2, mode="full", axis=None):
    """ Convolve two N-dimensional arrays using FFT. See convolve.

    This is a fix of scipy.signal.fftconvolve, adding an axis argument and
    importing locally the stuff only needed for this function
    
    """
    #Locally import stuff only required for this:
    from scipy.fftpack import fftn, fft, ifftn, ifft
    from scipy.signal.signaltools import _centered
    from numpy import array, product


    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))

    if axis is None:
        size = s1+s2-1
        fslice = tuple([slice(0, int(sz)) for sz in size])
    else:
        equal_shapes = s1==s2
        # allow equal_shapes[axis] to be False
        equal_shapes[axis] = True
        assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
        size = s1[axis]+s2[axis]-1
        fslice = [slice(l) for l in s1]
        fslice[axis] = slice(0, int(size))
        fslice = tuple(fslice)

    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    if axis is None:
        IN1 = fftn(in1,fsize)
        IN1 *= fftn(in2,fsize)
        ret = ifftn(IN1)[fslice].copy()
    else:
        IN1 = fft(in1,fsize,axis=axis)
        IN1 *= fft(in2,fsize,axis=axis)
        ret = ifft(IN1,axis=axis)[fslice].copy()
    if not complex_result:
        del IN1
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if product(s1,axis=0) > product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret,osize)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1)