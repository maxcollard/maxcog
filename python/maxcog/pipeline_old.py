#
# ...

# Imports

import time, unittest, collections, itertools, functools
from collections import OrderedDict

import numpy                        as np

import scipy.signal                 as sig
import scipy.stats.distributions    as dist

import matplotlib.pyplot            as plt
import matplotlib.cm                as cm

try:
    from pbar import pbar
except:
    pbar = lambda x, task = None : x


# Testing

def test ():

    n_ch = 64
    n_t = 1000

    x = np.random.randn( n_ch, n_t )

    x_ch = [ 'AMIC' + str(i) for i in range( 1, n_ch + 1 ) ]
    x_t = np.linspace( 0, 1, n_t )
    x_ax = {'channel': x_ch, 'time': x_t}

    x_arr = LabeledArray( x, x_ax )
    y_arr = x_arr[ 'channel', :5 ]
    print( y_arr )


# Classes







class FeatureExtractor ( object ):
    '''FeatureExtractor - ...'''

    def __init__ ( self ):
        
        pass

    def extract ( self, input ):
        '''extract - ...'''

        pass


# ...


class Feature ( object ):
    '''Feature - ...'''

    def __init__ ( self ):

        # Axes
        self.axes = OrderedDict()

        # Data
        self.data = None


    def values ( self ):
        pass

    def compute ( self, fm ):
        pass

    def window_indices ( self, window, axis = 'time' ):
        '''window_indices - ...'''
        return np.where( np.logical_and( self.axes[axis] > window[0], self.axes[axis] < window[1] ) )


class TrialFeature ( Feature ):
    '''TrialFeature - ...'''

    def __init__ ( self, compute_axes, compute_trial ):
        # Call superclass constructor
        super().__init__( self )

        # Absorb lambdas
        self.compute_axes = compute_axes
        self.compute_trial = compute_trial


    def compute ( self, in_data ):

        self.axes = self.compute_axes( in_data, self.axes )

        for data_trial in in_data():
            yield self.compute_trial( data_trial )


class NormalizeFeature ( Feature ):
    '''NormalizeFeature - ...'''

    def __init__ ( self, null_window ):
        # Call superclass constructor
        super().__init__( self )

        self.null_window = null_window

    #def 


class FunctionalMap ( object ):
    '''FunctionalMap - ...'''

    def __init__ ( self, eeg, events,
        aux_channels = [], bad_channels = [],
        window = [-1.0, 3.0],
        baseline_window = [-1.0, -0.5] ):

        # Data sources
        self.eeg        = eeg
        self.events     = events

        self.aux_channels       = aux_channels
        self.bad_channels       = bad_channels
        self.all_channels       = self.eeg.get_labels()
        self.good_channels      = [ s for s in self.all_channels
                                    if s not in self.bad_channels ]

        # Windows
        self.window             = window
        self.baseline_window    = baseline_window

        # Analysis dict
        self._features  = {}


    @property
    def features ( self ):
        '''features - ...'''

        return self._features

    def compute_feature( self, feature, name, input_name = 'raw' ):
        '''compute_feature - ...'''

        self._features[name] = feature
        feature.compute( self )


    def _raw_indices ( self, event, window = Default ):
        '''_window_idx - ...'''

        window = self.window if window is Default else window

        return np.where( np.logical_and(
            self._features[feature].axes['time'] > window[0],
            self._features[feature].axes['time'] < window[1] ) )

    def _parse_trials ( self, trials ):
        '''_parse_trials - ...'''

        ret = make_event()

        if not trials:
            # Add all events
            return self.events

        # TODO other cases

    def _raw_data_for_event ( self, event, show_all = False ):
        '''_raw_data_for_event - ...'''

        t_slice = self._raw_indices( event )
        ch_slice = self.all_channels if show_all else self.good_channels

        return self.eeg[t_slice, ch_slice]

    def raw_data ( self, trials = None ):
        '''raw_data - ...'''

        cur_events = self._parse_trials( trials )

        for event in cur_events:
            yield self._raw_data_for_event( event )


# Routines

def sr ( eeg, lock_events, ch,
        aux_ch = [], bad_ch = [],
        decimation = 1,
        t_buffer = 0.5, t_pre = 1.0, t_safe = 0.1, t_post = 3.0,
        spec_dict = {},
        plt_width = 8, plt_height = 6, plt_cmax = 12 ):

    '''sr - ...'''

    # Basics
    fs_raw = eeg.get_rate()
    fs = fs_raw / decimation

    all_ch      = eeg.get_labels()
    exclude_ch  = aux_ch + bad_ch
    car_ch      = [ s for s in all_ch if s not in exclude_ch ]

    n_t_raw = fs_raw * ( t_pre + t_post + 2 * t_buffer )
    n_trials = len( lock_events )


    ## Preprocessing

    # CAR
    eeg.auto_CAR( exclude_ch )


    ## Trial separation

    make_t_slice = lambda x : range( int( round( x - fs_raw * ( t_pre + t_buffer ) ) ),
                                     int( round( x + fs_raw * ( t_post + t_buffer ) ) ) )
    
    # Test decimation to get bounds
    event_start = fs * ( t_pre + t_buffer ) + 1
    t_slice_raw = make_t_slice( event_start )
    #tv_raw = np.linspace( -(t_pre + t_buffer), (t_post + t_buffer), len( t_slice ) )
    cur_data_raw = np.squeeze( eeg[t_slice_raw, [ch]] )
    if decimation > 1:
        cur_data = sig.decimate( cur_data_raw, decimation )
    else:
        cur_data = cur_data_raw
    n_t = len( cur_data )

    # Test spectrogram to get bounds
    f_spec, t_spec, spec_test = sig.spectrogram( np.squeeze( cur_data ),
        axis = 0,
        fs = fs,
        **spec_dict )
    t_spec = t_spec - ( t_pre + t_buffer )
    
    n_t_spec = len( t_spec )
    n_f_spec = len( f_spec )


    # Preallocate total data
    big_data = np.zeros( (n_t_spec, n_f_spec, n_trials) )

    i_trial = 0
    n_eta = 10     # TODO Magic

    for event_code, event_start, event_duration in pbar( lock_events, task = 'Filtering' ):

        #print( '.', end = '' )

        t_slice_raw = make_t_slice( event_start )

        cur_data_raw = np.squeeze( eeg[t_slice_raw, [ch]] )
        if decimation > 1:
            cur_data = sig.decimate( cur_data_raw, decimation,
                axis = 0 )
        else:
            cur_data = cur_data_raw
        
        tmp1, tmp2, cur_data_spec = sig.spectrogram( np.squeeze( cur_data ),
            axis = 0,
            fs = fs,
            **spec_dict )
        
        big_data[:, :, i_trial] = cur_data_spec.T
        i_trial = i_trial + 1


    ## Aggregate baseline

    baseline_mean = np.zeros( (n_f_spec) )
    baseline_std = np.zeros( (n_f_spec) )

    t_slice_baseline = np.where( np.logical_and( t_spec > -t_pre, t_spec < -t_safe ) )
    n_baseline = len( t_slice_baseline ) * n_trials

    for i in range( n_f_spec ):
        baseline_mean[i] = np.mean( np.log( np.ravel( big_data[t_slice_baseline, i, :] ) ) )
        baseline_std[i] = np.std( np.log( np.ravel( big_data[t_slice_baseline, i, :] ) ) )


    ## Normalization

    big_data_mean = np.mean( np.log( big_data ),
        axis = 2 )
    big_data_std = np.std( np.log( big_data ),
        axis = 2 )
    big_data_mean_norm = np.zeros( big_data_mean.shape )
    for i in range( n_f_spec ):
        big_data_mean_norm[:, i] = (big_data_mean[:, i] - baseline_mean[i]) / ( baseline_std[i] / np.sqrt( n_trials ) )

    
    ## Display

    fig, ax = plt.subplots( figsize = (plt_width, plt_height) )
    
    big_data_thresh = np.abs( big_data_mean_norm ) > 3.0
    
    im = ax.imshow( big_data_thresh.T * big_data_mean_norm.T,
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'none',
        cmap = cm.Spectral_r,
        extent = (t_spec[0], t_spec[-1], f_spec[0], f_spec[-1]),
        clim = (-plt_cmax, plt_cmax) )

    ax.set_xlim( -t_pre, t_post )
    ax.set_ylim( 0, 2000 )
    
    #ax.set_yticks( range( n_ch ) )
    #ax.set_yticklabels( car_ch )

    ax.grid( True )
    ax.set_title( ch )

    fig.colorbar( im )

    plt.show()

    return big_data_mean_norm.T


def fm ( eeg, lock_events,
        aux_ch = [], bad_ch = [],
        taps_filt = 200, f_filt = [70.0, 110.0], decimation = 1,
        t_buffer = 0.5, t_pre = 1.0, t_safe = 0.1, t_post = 3.0,
        stat_method = 'bins', alpha = 0.05, nu_max = 100,
        plt_width = 8, plt_height_per = 0.15, plt_cmax = 12 ):
    
    '''fm - ...'''


    # Basics
    fs_raw = eeg.get_rate()
    fs = fs_raw / decimation

    all_ch      = eeg.get_labels()
    exclude_ch  = aux_ch + bad_ch
    car_ch      = [ s for s in all_ch if s not in exclude_ch ]

    n_t_raw = fs_raw * ( t_pre + t_post + 2 * t_buffer )
    n_ch = len( car_ch )
    n_trials = len( lock_events )


    ## Preprocessing

    # CAR
    eeg.auto_CAR( exclude_ch )

    # Filtering & trial separation

    #print( 'Filtering:' )

    # Construct FIR BP filter parameters
    a_filt = np.array( [1.0] )
    b_filt = sig.firwin( taps_filt, f_filt,
        nyq = fs / 2.0,
        pass_zero = False )

    # Test decimation to get bounds
    if decimation > 1:
        event_start = fs * ( t_pre + t_buffer ) + 1
        t_slice = range( int( round( event_start - fs_raw * ( t_pre + t_buffer ) ) ),
                         int( round( event_start + fs_raw * ( t_post + t_buffer ) ) ) )
        #tv_raw = np.linspace( -(t_pre + t_buffer), (t_post + t_buffer), len( t_slice ) )
        cur_data_raw = eeg[t_slice, [car_ch[0]]]
        cur_data = sig.decimate( cur_data_raw, decimation, axis = 0 )
        n_t = len( cur_data )
    else:
        n_t = n_t_raw

    # Preallocate total data
    big_data = np.zeros( (n_t, n_ch, n_trials) )

    i_trial = 0
    n_eta = 10     # TODO Magic

    for event_code, event_start, event_duration in pbar( lock_events, task = 'Filtering' ):

        #print( '.', end = '' )

        t_slice = range( int( round( event_start - fs_raw * ( t_pre + t_buffer ) ) ),
                         int( round( event_start + fs_raw * ( t_post + t_buffer ) ) ) )

        timer_start = time.clock()

        cur_data_raw = eeg[t_slice, car_ch]
        if decimation > 1:
            cur_data = sig.decimate( cur_data_raw, decimation,
                axis = 0 )
        else:
            cur_data = cur_data_raw

        #print( cur_data_raw.shape )
        #print( cur_data.shape )

        cur_data_filt = sig.filtfilt( b_filt, a_filt, cur_data,
            axis = 0 )
        cur_data_env = np.abs( sig.hilbert( cur_data_filt,
            axis = 0 ) )

        big_data[:, :, i_trial] = cur_data_env

        timer_end = time.clock()
        i_trial = i_trial + 1

        # Show ETA
        if False and not ( i_trial % n_eta ):

            dt = timer_end - timer_start # s
            dmin, dsec = dt // 60, dt % 60

            eta = dt * ( n_trials - i_trial )
            emin, esec = eta // 60, eta % 60
            
            print( ' ETA: {:01.0f}m {:04.1f}s'.format( emin, esec ) )

    #print( 'Done.' )


    ## Aggregate baseline

    baseline_mean = np.zeros( (n_ch) )
    baseline_std = np.zeros( (n_ch) )

    t_slice_baseline = range( int( round( fs * t_buffer ) ),
        int( round( fs * ( t_buffer + t_pre - t_safe ) ) ) )
    n_baseline = len( t_slice_baseline ) * n_trials

    for i in range( n_ch ):
        baseline_mean[i] = np.mean( np.log( np.ravel( big_data[t_slice_baseline, i, :] ) ) )
        baseline_std[i] = np.std( np.log( np.ravel( big_data[t_slice_baseline, i, :] ) ) )


    ## Log-transform and baseline normalize data

    if False:
        big_data_norm = np.zeros( big_data.shape )
        for i_ch in range( n_ch ):
            for i_trial in range( n_trials ):
                big_data_norm[:, i_ch, i_trial] = ( ( np.log( big_data[:, i_ch, i_trial] ) - baseline_mean[i_ch] ) /
                    ( baseline_std[i] / np.sqrt( n_trials ) ) )


    ## Statistics

    #print( 'Statistics:' )

    if stat_method == 'bins':

        big_data_mean = np.mean( np.log( big_data ),
            axis = 2 )
        big_data_std = np.std( np.log( big_data ),
            axis = 2 )
        big_data_mean_norm = np.zeros( big_data_mean.shape )
        for i in range( n_ch ):
            big_data_mean_norm[:, i] = (big_data_mean[:, i] - baseline_mean[i]) / ( baseline_std[i] / np.sqrt( n_trials ) )

        n_tests = fs * t_post

        alpha_pointwise     = alpha / 2.0
        alpha_global        = alpha_pointwise / n_tests

        t_crit_cache = [ dist.t.ppf( 1 - alpha_global, nu ) for nu in range( 0, nu_max ) ]
        z_crit = dist.norm.ppf( 1 - alpha_global )
        t_crit = lambda nu : t_crit_cache[nu] if nu < nu_max else z_crit

        t_stat = lambda t, ch: ( ( big_data_mean[t, ch] - baseline_mean[ch] ) /
                                 np.sqrt( ( big_data_std[t, ch] ** 2 / n_trials ) +
                                       ( baseline_std[ch] ** 2 / n_baseline ) ) )
        nu_stat = lambda t, ch: ( ( ( big_data_std[t, ch] ** 2 / n_trials ) +
                                    ( baseline_std[ch] ** 2 / n_baseline ) ) ** 2 /
                                  ( ( big_data_std[t, ch] ** 4 / ( n_trials ** 2 * (n_trials - 1) ) ) +
                                    ( baseline_std[ch] ** 4 / ( n_baseline ** 2 * (n_baseline - 1) ) ) ) )

        big_data_t          = np.zeros( big_data_mean.shape )
        big_data_nu         = np.zeros( big_data_mean.shape )
        big_data_t_crit     = np.zeros( big_data_mean.shape )

        for i_ch in pbar( range( n_ch ), task = 'Statisticsing' ):
            for t in range( int( n_t ) ):
                big_data_t[t, i_ch]         = t_stat( t, i_ch )
                big_data_nu[t, i_ch]        = nu_stat( t, i_ch )
                big_data_t_crit[t, i_ch]    = t_crit( int( np.floor( big_data_nu[t, i_ch] ) ) )

        big_data_thresh = np.abs( big_data_t ) > big_data_t_crit

    #print( 'Done.' )


    ## Construct FM object



    ## Display

    plt_height = plt_height_per * n_ch

    fig, ax = plt.subplots( figsize = (plt_width, plt_height) )
    
    im = ax.imshow( big_data_thresh.T * big_data_mean_norm.T,
        aspect = 'auto',
        interpolation = 'none',
        cmap = cm.Spectral_r,
        extent = (-(t_pre + t_buffer), t_post + t_buffer,
                    -0.5, n_ch - 0.5 ),
        clim = (-plt_cmax, plt_cmax) )

    ax.set_xlim( -t_pre, t_post )
    
    ax.set_yticks( range( n_ch ) )
    ax.set_yticklabels( car_ch )

    ax.grid( True )

    fig.colorbar( im )

    plt.show()

    
    return big_data_mean_norm.T



























