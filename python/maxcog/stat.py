#
# ...

## === IMPORTS === ##


# See __init__.py


import numpy                        as np
import scipy.signal                 as sig
import scipy.stats.distributions    as dist

import numpy.linalg as linalg
import scipy.sparse as sparse

from maxcog.pipeline import baseline_normalize


## === BASICS === ##


# Helper functions for modulation_classic
def _t_stat ( mu0, sig0, n0, mu1, sig1, n1 ):
    return ( ( mu1 - mu0 ) /
             np.sqrt( ( sig0 ** 2 / n0 ) + ( sig1 ** 2 / n1 ) ) )
def _nu_stat ( sig0, n0, sig1, n1 ):
    return ( ( ( sig1 ** 2 / n1 ) + ( sig0 ** 2 / n0 ) ) ** 2 /
             ( ( sig1 ** 4 / ( n1 ** 2 * (n1 - 1) ) ) + ( sig0 ** 4 / ( n0 ** 2 * (n0 - 1) ) ) ) )
def _p_stat_z ( z ):
    return ( 1.0 - dist.norm.cdf( z ) ) / 2.0
def _p_stat_t ( t, nu ):
    return ( 1.0 - dist.t.cdf( t, nu ) ) / 2.0

def modulation_classic ( x, baseline_window, z_test = False ):
    '''Computes pointwise t-statistics for modulation against baseline
    fluctuations. Each non-trial feature of x is tested separately.

    REMARK: All data is normalized to the baseline mean and standard deviation
    prior to testing.

    Inputs:
    x - LabeledArray containing the data. Must have a 'time' and 'trial' axis.
    baseline_window - 2-tuple specifying the time window to use for constructing
        the baseline distributions. This interval is "right-open" in this
        implementation; i.e., [start, stop).
    z_test - (Optional) If True, p-values will be computed using a normal
        approximation to the t-distribution, which improves performance. This
        approximation is valid when the estimated nu (df) parameter is ~> 20;
        this is generally the case for large trial-count datasets.

    Output is a 3-tuple,
        (x_mu, x_sig, x_p)
    where:
    x_mu - The means of the baseline-normalized data across trials
    x_sig - The standard deviations of the baseline-normalized data across trials
    x_p - The p-values for the observed t-statistics under the null hypothesis
        that the true x_mu is zero.
    '''

    x_norm, baseline_mu, baseline_sig = baseline_normalize( x,
                                                            window = baseline_window,
                                                            return_baseline = True )
    
    # Compute mean and SD across trials
    x_mu = mean_labeled( x_norm, axis = 'trial' )
    x_sig = std_labeled( x_norm, axis = 'trial' )
    
    # Allocate statistics
    x_p = maxcog.pipeline.LabeledArray( axes = x_mu.axes )
    
    n_trials = len( x.axes['trial'] )
    n_baseline = n_trials * len( x['time', baseline_window, 'labeled'].axes['time'] )
    
    # Compute statistics
    for x_mu_cur, i in x_mu.iter_over( 'time', return_index = True ):
        x_sig_cur = x_sig['time', i, 'labeled']
        
        x_t_cur = _t_stat( baseline_mu.array, baseline_sig.array, n_baseline,
                           x_mu_cur.array, x_sig_cur.array, n_trials )
        if z_test:
            x_p_cur = _p_stat_z( x_t_cur )
        else:
            x_nu_cur = _nu_stat( baseline_sig.array, n_baseline, x_sig_cur.array, n_trials )
            x_p_cur = _p_stat_t( x_t_cur, x_nu_cur )
        
        x_p['time', i] = x_p_cur
    
    return x_mu, x_sig, x_p


## === LOCAL REGRESSION === ##
# TODO Stuff in here is not yet ready for primetime.


# Helper functions
def _kernel_tricube( x, h ):
    return (70.0 / 81.0) * (np.abs( x / h ) < 1) * (1 - np.abs(x / h)**3)**3


def locregress_weights ( x, span = 0.2, order = 1, fast_interior = True ):
    '''DO NOT USE'''

    # Cache some constants
    n = x.shape[0] # Number of input points
    d = x.shape[1] # Dimensionality of fit
    span_samples = np.ceil( span * n ) # Number of samples in the span

    # Preallocate sparse return value
    ret = sparse.lil_matrix( (n, n) )

    # Apply intuition that, for regularly spaced grids, weights on the interior
    # are just shifted versions of the same kernel
    if fast_interior:

        # Find "center" point
        x_center = 0.5 * ( np.max( x, axis = 0 ) + np.min( x, axis = 0 ) )
        dx_center = x - np.tile( x_center, (n, 1) )
        dx_norm = linalg.norm( dx_center, axis = 1 )
        idx_center = np.argmin( dx_norm )

        # Find critical kernel width
        x_star = x[idx_center, :]
        dx = x - np.tile( x_star, (n, 1) )
        dx_norm = linalg.norm( dx, axis = 1 )
        dx_norm = np.sort( dx_norm, axis = 0 )
        h_interior = dx_norm[span_samples + 1]

        # Find bounding box for where we can use interior kernel
        x_interior_min = np.min( x, axis = 0 ) + h_interior
        x_interior_max = np.max( x, axis = 0 ) - h_interior

        # Lambda for interior kernel
        kh_interior = lambda x_v : _kernel_tricube( linalg.norm( x_v, axis = 1 ), h_interior )

        # Compute weight matrix for the interior
        w_vec = kh_interior( dx )
        w_interior_slice = np.where( w_vec > 0 )
        w_vec = w_vec[w_interior_slice]
        w = np.diag( w_vec )

        # Include quadratic terms
        if order > 1:
            for j in range( d ):
                for k in range( j, d ):
                    dx = np.c_[dx, dx[:, j] * dx[:, k]]
        # Include affine terms
        dx = np.c_[np.ones( (n, 1) ), dx]
        # Slice out only relevant items
        dx = dx[w_interior_slice]

        e1 = np.zeros( (dx.shape[1], 1) )
        e1[0] = 1

        a2 = np.dot( dx.T, w )
        a1 = np.dot( a2, dx )

        ret_interior = np.dot( e1.T, linalg.solve( a1, a2 ) )
        # Cache the index offsets for quick copying
        offset_interior = w_interior_slice - idx_center

    # Compute weight vectors at each point

    for i in pbar( range( n ), task = 'locregress_weights' ):

        x0 = x[i, :]

        # Apply interior heuristic
        if fast_interior:

            # Check if we're in the interior
            is_interior = np.logical_and( np.logical_not( np.any( x0 <= x_interior_min ) ),
                                          np.logical_not( np.any( x0 >= x_interior_max ) ) )
            if is_interior:
                # Copy known interior weights
                ret[i, i + offset_interior] = ret_interior
                continue

        # Not in interior; have to manually compute

        # For debugging purposes, screw that lol
        #continue

        # Determine input-space deltas
        dx = x - np.tile( x0, (n, 1) )

        # Determine correct kernel width
        dx_norm = linalg.norm( dx, axis = 1 )
        dx_norm = np.sort( dx_norm, axis = 0 )
        h = dx_norm[span_samples + 1]

        # Lambda for interior kernel
        kh = lambda x_v : _kernel_tricube( linalg.norm( x_v, axis = 1 ), h )

        # Calculate weight matrix
        w_vec = kh( dx )
        w_slice = np.where( w_vec > 0 )
        w_vec = w_vec[w_slice]
        w = np.diag( w_vec )

        # Include quadratic terms
        if order > 1:
            for j in range( d ):
                for k in range( j, d ):
                    dx = np.c_[dx, dx[:, j] * dx[:, k]]
        # Include affine terms
        dx = np.c_[np.ones( (n, 1) ), dx]
        # Slice out only relevant items
        dx = dx[w_slice]

        e1 = np.zeros( (dx.shape[1], 1) )
        e1[0] = 1

        a2 = np.dot( dx.T, w )
        a1 = np.dot( a2, dx )

        ret[i, w_slice] = np.dot( e1.T, linalg.solve( a1, a2 ) )

    return ret.tcsr()

import time

def locregress_eval ( y, L,
    alpha = 0.05,
    L_norms = None,
    nu = None,
    nu_twiddle = None,
    return_stats = False ):
    '''DO NOT USE'''

    stat_dict = {}

    n_y = y.size

    # TODO Hella stupid
    if L_norms is None:
        L_norms = np.sum( (L * L).toarray(), axis = 1 )
    if nu is None:
        nu = np.trace( L.toarray() )
    if nu_twiddle is None:
        nu_twiddle = np.trace( L.dot( L ).toarray() )
    stat_dict['L_norms'] = L_norms
    stat_dict['nu'] = nu
    stat_dict['nu_twiddle'] = nu_twiddle

    # Apply linear smoother
    y_hat = np.reshape( L.dot( y.flatten() ), y.shape )
    t_end = time.clock()

    # Compute error metrics
    sse = np.sum( (y_hat - y) ** 2 )
    stat_dict['sse'] = sse
    s2_hat = sse / ( n_y - (2 * nu) + nu_twiddle )
    stat_dict['s2_hat'] = s2_hat

    # Compute error bounds
    p_star = alpha / n_y
    z_crit = dist.norm.ppf( 1.0 - ( p_star / 2.0 ) )

    y_err = np.reshape( z_crit * np.sqrt( s2_hat * L_norms ), y.shape )

    if return_stats:
        return y_hat, y_err, stat_dict

    return y_hat, y_err


def locregress_weights_old ( x, span = 0.2, order = 1, use_pbar = False ):
    '''DO NOT USE'''
    # Shortcuts
    N = len( x )
    n_thresh = ceil( span * N )
    
    # Pre-allocate
    ret = np.zeros( (N, N) )
    
    method = 2
    
    if use_pbar:
        iterator = pbar( range( N ) )
    else:
        iterator = range( N )
    
    for i in iterator:
        x0 = x[i]
        dx = x - x0
        
        # Choose adaptive kernel width
        # TODO Do without sorting
        h_thresh = sort( abs( dx ) )[n_thresh]
        
        if method == 1:
            # Old method
            w = _kernel_tricube( dx, h_thresh )
            S1 = sum( w * dx )
            S2 = sum( w * (dx ** 2) )

            den = sum( w * S2 - w * dx * S1 )

            for j in range( N ):
                ret[i,j] = (1 / den) * ( w[j] * S2 - w[j] * dx[j] * S1 )
                
        if method == 2:
            # New method
            w = _kernel_tricube( dx, h_thresh )
            w_slice = w != 0

            e1 = np.zeros( (order + 1, 1) )
            e1[0] = 1.0

            bigW = np.diag( w[w_slice] )

            bigX = np.ones( (N, order + 1) )
            bigX[:, 1] = dx
            if order > 1:
                bigX[:, 2] = dx ** 2
            bigX = bigX[w_slice, :]

            A2 = np.dot( bigX.T, bigW )
            A1 = np.dot( A2, bigX )

            ret[i, w_slice] = np.dot( np.dot( e1.T, np.linalg.inv( A1 ) ), A2 )
        
    return ret






