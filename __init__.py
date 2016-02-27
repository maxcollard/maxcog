#
# ...


## === IMPORTS === ##


import time
import unittest
import collections
import itertools
import functools

from collections import OrderedDict

import numpy                        as np
import scipy.signal                 as sig
import scipy.stats.distributions    as dist


# TODO Kludgey way to make things work without dependencies for testing

try:
    from pbar import pbar
except:
    def pbar( x, task = None ): return x

try:
    from h5eeg import H5EEGEvents
    EVENT_DTYPE = H5EEGEvents.EVENT_DTYPE
except:
    EVENT_DTYPE = [('name', '<U32'), ('start_idx', '<i4'), ('duration', '<i4')]


## === MAIN === ##


# TODO Add unit tests

if __name__ == '__main__':
    main()

def main ():
    pass


## === MISCELLANEOUS DOODADS === ##


# Default sentinel object.
# To be used with "if arg is Default" pattern.
# (Enables passing arg as None to be non-Default behavior.)
Default = object()


def unpack_channels ( *packed, formatter = None ):
    '''Turns ('ABC', [1, 2]) or 'ABC', [1, 2] into ['ABC1', 'ABC2']

    Inputs:
    *packed - Tuples ('ABC', [1,2]), or sequence in format 'ABC', [1,2], 'DEF',
        [3,4], ...
    formatter - (optional) Lambda of form (string, int) -> string that
        determines how a (name, number) pair should be formatted.
        Default is equivalent to
            lambda name, chan : name + '{:02d}'.format( chan )

    Output is a list.
    '''

    # TODO Re-do; this is pretty bad implementation ...
    
    # Handle un-tupled case
    if not type( packed[0] ) == tuple:
        # Expecting sequence of form 'ABC', [1,2], ...
        # Must be even length.
        if len( packed ) % 2:
            raise RuntimeError( 'Definition list must be even length.' )
        else:
            packed = [ (packed[2*i], packed[2*i+1]) for i in range( 0, len(packed)//2 ) ]
    
    # Create default formatter
    if formatter is None:
        def formatter( name, chan ): return name + '{:02d}'.format( chan )
    
    def tuple_to_list( name, chans ): return [ formatter( name, x ) for x in chans ]
    def flatten( x ): return [ i for j in x for i in j ]
    
    ret_lists = [ tuple_to_list( name, chans ) for name, chans in packed ]
    return flatten( ret_lists )


def make_event ( name = None, start_idx = 0, duration = 0 ):
    '''Conveniently creates an event compatible with H5EEG's format'''
    if name == None:
        return np.array( [], dtype = EVENT_DTYPE )
    return np.array( [(name, start_idx, duration)], dtype = EVENT_DTYPE )[0]


## === CLASSES === ##


class EventFrame ( object ):
    '''Provides an abstraction of the notion of "locking to an event"

    EventFrame relies upon a H5EEGDataset to serve as a data source. It has two
    primary modes of operation: trials, which returns a generator that extracts
    each trial individually, and extract, which pulls 
    '''

    def __init__ ( self, eeg, events, window ):
        '''Initializes an EventFrame for given data, locking events, and
        extraction window

        Inputs:
        eeg - H5EEGDataset to serve as the data source
        events - ndarray of h5eeg event records (i.e.,
            dtype = H5EEGEvents.EVENT_DTYPE) to serve as t=0 for the frame
        window - extraction window of form (start, stop), in seconds
        '''

        self.eeg        = eeg
        self.events     = events
        self.window     = window

    def _t_slice ( self, x ):
        '''Convenience method to generate a slicing array for a window centered
        around a particular sample, x
        '''
        return range( int( round( x + self.eeg.get_rate() * self.window[0] ) ),
                      int( round( x + self.eeg.get_rate() * self.window[1] ) ) )

    def _axes ( self, single_trial = False ):
        '''Convenience method to create an axes object compatible with
        LabeledArray

        Inputs:
        single_trial - (Optional) If True, the trial axis is omitted
            Default: False
        '''

        ret = OrderedDict()
        ret['time'] = np.linspace( self.window[0],self.window[1], len( self._t_slice( 0 ) ) )
        ret['channel'] = self.eeg.get_labels()
        if not single_trial:
            ret['trial'] = np.arange( 1, len( self.events ) + 1 )
        return ret


    def trials ( self, channels = slice( None ) ):
        '''Returns a generator that pulls out the data from each event into a
        LabeledArray

        Inputs:
        channels - (Optional) Subset of channels to be extracted, in a format
            compatible with H5EEGDataset's slicing
            Default: All channels; i.e., slice(None)

        Output is a generator; each returned item is a LabeledArray of signature
        ('time', 'channel')
        '''

        # TODO Figure out how to get length caching to work
        # Cache length
        #ret.__length_hint__ = lambda : len( self.events )

        return ( LabeledArray( self.eeg[self._t_slice( e['start_idx'] ), channels],
                               self._axes( single_trial = True ) )
                 for e in self.events )

    def extract ( self ):
        '''Pulls out the data from all events into a LabeledArray. Each event is
        placed into its own entry along the 'trial' axis of the returned array.

        Output is a LabeledArray of signature ('time', 'channel', 'trial')
        '''

        # TODO Get this to work without copying ...

        # Preallocate return value
        ret_axes = self._axes()
        ret_array = np.zeros( tuple( map( len, ret_axes.values() ) ) )
        i = 0
        for event in pbar( self.events ):
            ret_array[:, :, i] = self.eeg[self._t_slice( event['start_idx'] ), :]
            i += 1

        return LabeledArray( ret_array, ret_axes )

class LabeledArray ( object ):
    '''Encapsulates a data array and a "labeling" of the array, which includes
    names for each dimension (axis), and annotations ("ticks") at each entry of
    each dimension.

    The underlying data for a LabeledArray is a numpy ndarray combined with an
    OrderedDict, which specifies the name of each axis of the ndarray (keys) and
    an annotation for each slot along that axis (values; think "tick marks").

    LabeledArray allows the nature of the data being manipulated to be
    abstracted away from the particular organization of the underlying data,
    making "data shepherding" considerably easier. It also ensures that
    dimension labels "tag along" with the data they represent, so there's no
    guesswork when it comes to, for example, time/frequency spacing or channel
    names.

    This labeling facilitates nifty things like
        Slicing:
            best_data = data['channel', 'RTG32', 'time', (-0.5, 0.5), 'labeled']
        Iteration:
            for trial_data in data.iter_over( 'trial' ):
                ...
        General fun abstraction:
            plt.set_ylim( data.extent['frequency'] )
    '''

    def __init__ ( self, array = Default, axes = Default ):
        '''Initializes a LabeledArray with (a) given data and axes labelings,
        (b) blank data suited to a given axes labeling, or (c) data copied from
        another LabeledArray.

        Inputs:
        array - A numpy ndarray containing the data
        axes - An OrderedDict containing the axis annotations
            axes.keys() should be strings, and should have the same number of
                elements as array has dimensions
            axes.values() should be a sequence or array, and should have the
                same length as the respective dimension of array
            Equivalently,
                array.shape == np.array( [ len( axes[k] ) for k in axes.keys() ] )

        This constructor has a couple handy special cases:

        LabeledArray( array = la ), where la is a LabeledArray, *copies* the
            underlying array and axes of la into the new LabeledArray
        LabeledArray( axes = x ) allocates a zero array whose size is determined
            by x
        '''

        self.array = None
        self.axes = OrderedDict()

        if array is Default:
            if axes is not Default:
                self.array = np.zeros( tuple( map( len, axes.values() ) ) )
        else:
            if isinstance( array, LabeledArray ):
                # Optimal NP deep copy?
                self.array = np.empty_like( array.array )
                self.array[:] = array.array

                self.axes = OrderedDict( array.axes )

            else:
                self.array = array

        if axes is not Default:
            self.axes = OrderedDict( axes )

    @property
    def shape ( self ):
        '''Gives the length of each dimension of the underlying ndarray, as
        implemented by numpy. Returns None if the underlying array is None.
        '''

        if self.array is None:
            return None

        return self.array.shape

    @property
    def extent ( self ):
        '''Gives a dictionary mapping each axis name to a tuple
            (first, last),
        where first and last are the first and last annotations along that
        axis, respectively.

        This is particularly useful, for example, when determining bounds for
        plotting; e.g.,
            plt.set_xlim( data.extent['time'] )
        '''
        return { axis_key : ( self.axes[axis_key][0], self.axes[axis_key][-1] )
                 for axis_key in self.axes.keys() }

    def axis_index( self, axis ):
        '''Returns the index of the specified axis in the underlying array.'''
        return list( x.axes.keys() ).index( 'time' )

    def to_array ( self, order = Default ):
        '''Returns the underlying array, re-ordered as desired.

        Inputs:
        order - (Optional) Specifies the desired axis order of the returned
            array. If order is not exhaustive, to_array will pull the axes in
            order to the front, and leave the remaining axes in their original
            relative ordering. Can be a sequence or a single string.
            Default: Return the underlying array in its original axis ordering;
                equivalent to
                    order = self.axes.keys()

        Output is an ndarray equivalent in shape to the LabeledArray's
        underlying array up to permutation.
        '''
        
        ret = self.array
        ret_axes = list( self.axes.keys() )

        if not ( order is Default ):

            # Lets us just put a single string into order.
            if isinstance( order, str ):
                order = [order]

            i = 0
            for axis in order:
                if axis in ret_axes:
                    i_axis = ret_axes.index( axis )
                    ret = np.swapaxes( ret, i, i_axis )
                    ret_axes[i], ret_axes[i_axis] = ret_axes[i_axis], ret_axes[i]
                    i += 1

        return ret

    def iter_over ( self, axis, return_index = False ):
        '''Creates a generator that iterates along a specified axis or sequence
        of axes.

        Inputs:
        axis - The name(s) of the axis(es) over which to iterate. Can be a
            string (single axis) or sequence (multiple axes)
        return_index - (Optional) If True, the returned generator will also
            include the current index (or indices) in the iteration
            Default: False

        Output is a generator, whose form depends on the context:
        
        If return_index is False, the generator yields a LabeledArray with the
            sliced data for that iteration.
        If return_index is True and only one axis is iterated over, it yields a
            tuple
                (la, idx)
            where la is the LabeledArray from above, and idx is the current
            index in the iteration.
        If return_index is True and multiple axes are iterated over, it yields a
            tuple as above, but in which idx is itself a tuple
                (la, (i0, i1, ... in))
            where ij is the index along the jth iterated axis.

        When multiple axes are iterated over, the order/nesting of the iteration
        is determined by itertools.product.
        '''

        # Always wrap axis in a list, for convenience
        if isinstance( axis, str ):
            axis = [ axis ]

        # Intermediate method for generating the sliced LabeledArray returned
        # by the generator for a given index
        def _slicerate ( idx ):
            cs, _ = self._compound_slice( dict( zip( axis, idx ) ) )
            ret_arr = np.squeeze( self.array[cs] )
            ret_axes = OrderedDict( { k: v for (k, v) in self.axes.items()
                                            if not (k in axis) } )
            return LabeledArray( ret_arr, ret_axes )

        product_indices = itertools.product( *[ range( len( self.axes[x] ) ) for x in axis ] )
        
        if return_index:
            if len( axis ) == 1:
                return ( (_slicerate( idx ), idx[0]) for idx in product_indices )
            else:
                return ( (_slicerate( idx ), idx) for idx in product_indices )
        else:
            return ( _slicerate( idx ) for idx in product_indices )

    def combine_axes ( self, axes, key_delimeter = '**' ):
        '''Creates a new LabeledArray by collapsing the designated axes into a
        single "compound axis".

        Inputs:
        axes - A list of the names of the axes to combine
        key_delimeter - (Optional) The string to be used when joining the keys
            of the individual axes; i.e.,
                combined_axis = key_delimeter.join( axes )
            Default: '**'

        If self is an N-dimensional LabeledArray, the output is a LabeledArray
        of dimension
            N - ( len(axes) - 1 )
        The compound axis' key is determined joining axes through key_delimeter.
        Its values are tuples whose entries were the values from the respective
        original axes. For example, combining 'time': [t0, t1, t2, ... tn] and
        'frequency': [f0, f1, f2, ... fn] creates
            'time**frequency': [ (t0, f0), (t0, f1), ... (t1, f0), ... (tn, fn) ]
        '''

        axes_idx = [ list( self.axes.keys() ).index( x ) for x in axes ]
        axes_len = [ len( self.axes[x] ) for x in axes ]

        other_axes = [ x for x in self.axes if not x in axes ]

        new_len = functools.reduce( (lambda a, b: a * b), axes_len )
        new_shape = tuple( len( self.axes[x] ) for x in other_axes )
        new_shape = new_shape + (new_len,)

        # Create new axes
        ret_axes = OrderedDict( self.axes )
        # Remove axes being combined
        for x in axes:
            ret_axes.pop( x, None )
        # Add new combined axis at end
        combined_axis = key_delimeter.join( axes )
        ret_axes[combined_axis] = np.array( list(
            itertools.product( *[ self.axes[x] for x in axes ] ) ) )
        
        # Construct new array
        ret_array = np.zeros( new_shape )
        # Iterator over all compound indices of preserved axes
        product_idx = itertools.product( *[ range( len( self.axes[x] ) ) for x in other_axes ] )
        # For each compound index ...
        for i in product_idx:
            # Produce a slice description
            slice_desc = tuple( functools.reduce( lambda a, b: a + b, zip( other_axes, i ) ) )
            cur_arr = self[slice_desc]
            ret_array[i] = cur_arr.flatten()

        return LabeledArray( array = ret_array, axes = ret_axes )

    def _parse_pairs ( self, slice_list ):
        '''Convenience method for parsing the contents of a compound slice list.
        Determines whether there is a 'labeled' at the end of slice_list, and
        turns the
            'key', value, ...
        list in the initial segment of slice_list into a {'key': value} dict.

        Output is a tuple
            (key_dict, labeled)
        where:
        key_dict - The {'key': value} dict constructed from slice_list
        labeled - True if 'labeled' appears at the end of slice_list;
            False otherwise
        '''

        # TODO Should validate that odd terms are strings

        labeled = False
        if len( slice_list ) % 2:
            if slice_list[-1] == 'labeled':
                labeled = True
            else:
                raise IndexError( "Key-value slice must have even number of terms, or end with 'labeled'." )
        
        return ( { slice_list[i]: slice_list[i+1]
                   for i in range( 0, len( slice_list ) - 1, 2 ) },
            labeled )

    def _compound_slice ( self, key_dict ):
        '''Convenience method that generates a tuple that numpy knows how to use
        to slice the underlying ndarray, given a {'key': value} dict generated
        by _parse_pairs.
        '''

        # Generates an appropriate slice-capable object for a particular axis x
        def _handle_axis ( x ):

            ret = key_dict[x]

            # Axis labels are strings
            if isinstance( self.axes[x][0], str ):
                # Single string query
                if isinstance( ret, str ):
                    ret = list( self.axes[x] ).index( ret )
                # Multiple string query
                elif isinstance( ret, collections.Sequence ):
                    if isinstance( ret[0], str ):
                        ret = [ list( self.axes[x] ).index( r ) for r in ret ]

            # Query is a 2-tuple -- Range
            elif isinstance( ret, tuple ):
                if len( ret ) == 2:
                    # Yields a ndarray for slicing, which can get hairy with multiples.
                    #ret = np.where( np.logical_and( self.axes[x] >= ret[0],
                    #    self.axes[x] < ret[1] ) )[0]

                    # Just operate as a slice
                    ret_where = np.where( np.logical_and( self.axes[x] >= ret[0],
                        self.axes[x] < ret[1] ) )[0]
                    if len( ret) == 0:
                        ret = slice( 0, 0 )
                    else:
                        ret = slice( ret_where[0], ret_where[-1] + 1 )

            # TODO Raise an error for incompatible objects

            return ret

        # Determines whether numpy kills the axis named x when it's sliced;
        # used for determining the axes of the newly returned 
        def _is_killed ( x ):

            # TODO This is *Hella* slow. Should re-write if LabeledArray slicing becomes performance-critical
            full_d = len( self.shape )
            test_slice = tuple( _handle_axis( axis_key )
                                if (axis_key == x and axis_key in key_dict)
                                else slice( None )
                                for axis_key in self.axes.keys() )
            slice_d = len( self.array[test_slice].shape )

            return (full_d > slice_d)

        return ( tuple( _handle_axis( axis )
                        if axis in key_dict
                        else slice( None )
                        for axis in self.axes ),
            [ axis for axis in self.axes if _is_killed( axis ) ] )

    def __setitem__ ( self, key, new_value ):
        '''Alters the data in the underlying ndarray, using the same slicing
        conventions as in __getitem__.

        Attempting to put a square peg into a round hole will hopefully generate
        numpy errors.'''

        # TODO Should probably do some of my own error checking, but ... eh

        # Slicing in form 'key0', slice0, 'key1', slice1 ..., ['labeled']
        if isinstance( key, tuple ):
            # Make dict from pairs
            key_dict, _ = self._parse_pairs( key )
            # Slice appropriately
            set_slice, _ = self._compound_slice( key_dict )
            self.array[ set_slice ] = value

        # TODO Other forms of slicing?

        # TODO Raise an error for incompatible slice types?

    def __getitem__ ( self, key ):
        '''Magic Matlab-style key-value pair data slicing.

        The slice should be of the form
            'key0', slice0, 'key1', slice1, ... ['labeled']
        where:
        'keyI' - Name of an axis in this LabeledArray's axes to slice along
        sliceI - A "slice-capable" object (see below)
        'labeled' - (Optional) Flag in the terminal position that determines
            the format of the returned result. If present, slicing returns a
            LabeledArray; if absent, slicing returns a numpy ndarray.

        REMARK: Remember that the default behavior, for performance reasons, is
        to extract the slice from the underlying data and discard the
        annotations. If you need labels, just add 'labeled' at the end of the
        slice.

        "Slice-capable" objects include:
            ! A string or sequence of strings, when the axis in question is
                annotated with strings; e.g.,
                    data['channel', ('LTG34', 'ainp2')]
            ! A 2-tuple of numbers, which is interpreted as a range in the same
                units as the axis annotations; e.g.,
                    data['time', (-0.5, 0.5)]
                This is done in a "half-open" way; that is, slicing with
                    'x', (low, high)
                serves up values where the low <= 'x' < high.
            - Anything that you could slice a 1d numpy array with, including
                Python slice objects (e.g., :)
        '''
        # Slicing in form 'key0', slice0, 'key1', slice1 ..., ['labeled']
        if isinstance( key, tuple ):
            # Make dict from pairs
            key_dict, labeled = self._parse_pairs( key )
            # Slice appropriately
            ret_slice, killed_axes = self._compound_slice( key_dict )
            if labeled:
                good_keys = [ axis_key for axis_key in self.axes.keys()
                    if not ( axis_key in killed_axes ) ]
                good_values = ( self.axes[axis_key] for axis_key in good_keys )
                good_slices = ( ret_slice[i] for i, axis_key in enumerate( self.axes.keys() )
                    if not ( axis_key in killed_axes ) )

                return LabeledArray( array = self.array[ret_slice],
                                     axes = OrderedDict( zip( good_keys,
                                                         map( lambda x,y : x[y], good_values, good_slices ) ) ) )
            else:
                return self.array[ret_slice]

        # TODO Other forms of slicing?

        # TODO Raise an error for incompatible slice types?
