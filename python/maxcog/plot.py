#!/usr/bin/env python

#==============================================================================#
# maxcog.plot
# v. 0.0.1
#
# Functions for making pretty plots.
# 
# 2016 - Maxwell J. Collard
#==============================================================================#

"""Functions for generating pretty plots."""


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

import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.cm        as cm


## === STANDARD PLOTTERS === ##

# Plotters are functions that take in a plotting environment and data, and plot
# the data appropriately. They should have a signature
#   def _plotter ( ax, data, err = None, **kwargs ):
#       ...
#       return None


def plotter_trace ( axis = 'time', xlim = None, ylim = None ):
    
    # Create a plotter function to return
    def _plotter ( ax, data, err = None, **kwargs ):
        
        if not ( err is None ):
            ax.fill_between( data.axes[axis],
                             data.array - err.array,
                             data.array + err.array,
                             facecolor = 'black',
                             alpha = 0.2 )
        
        default_kwargs = { 'linewidth': 2 }
        
        for k in default_kwargs:
            if not ( k in kwargs ):
                kwargs[k] = default_kwargs[k]
        
        ax.plot( data.axes[axis],
                 data.array,
                 'k-',
                 **kwargs )
        
        ax.grid( color = '#cccccc',
                 linestyle = '-' )
        ax.set_axisbelow( True )
        #ax.tick_params( axis = 'x',
        #                colors = '#cccccc' )
        
        the_xlim = ax.get_xlim() if xlim is None else xlim
        #ax.plot( xlim, np.array( [0, 0] ), 'k--' )
        ax.set_xlim( the_xlim )
        
        the_ylim = ax.get_ylim() if ylim is None else ylim
        ax.plot( np.array( [0, 0] ),
                 the_ylim,
                 'g-',
                 linewidth = 1.5 )
        ax.set_ylim( the_ylim )
        
        if not ( err is None ):
            data_sig = 2 * ( ( data.array - err.array ) > 0 ) - 1
            ax.fill_between( data.axes[axis],
                             the_ylim[0] - 1,
                             the_ylim[0] + 2 * (the_ylim[1] - the_ylim[0]) * data_sig,
                             facecolor = 'red',
                             alpha = 0.2,
                             linewidth = 0 )

            data_sig = 2 * ( ( data.array + err.array ) < 0 ) - 1
            ax.fill_between( data.axes[axis],
                             the_ylim[0] - 1,
                             the_ylim[0] + 2 * (the_ylim[1] - the_ylim[0]) * data_sig,
                             facecolor = 'blue',
                             alpha = 0.2,
                             linewidth = 0 )
        
    return _plotter

def plotter_raster ( axis, xlim = None, ylim = None, clim = None ):
    
    # Create a plotter function to return
    def _plotter ( ax, data, err = None, **kwargs ):
        
        # For convenience, wrap axis in an array if it's one string
        the_axis = [ axis ] if isinstance( axis, str ) else axis
        # For convenience, if we only have one axis, add on another
        if len( the_axis ) == 1:
            the_axis += [ next( x for x in data.axes if not (x == the_axis[0]) ) ]
        
        plt_array = data.to_array( the_axis )
        plt_extent = data.extent[axis[1]] + data.extent[axis[0]]
        
        default_kwargs = { 'aspect': 'auto',
                           'interpolation': 'none',
                           'origin': 'lower',
                           'cmap': cm.Spectral_r,
                           'clim': clim }
        
        for k in default_kwargs:
            if not (k in kwargs):
                kwargs[k] = default_kwargs[k]
        
        ax.imshow( plt_array,
                   extent = plt_extent,
                   **kwargs )

        the_xlim = ax.get_xlim() if xlim is None else xlim
        # TODO Kludge?
        if the_axis[0] == 'time':
            ax.plot( the_xlim, np.array( [0, 0] ), 'g-', linewidth = 1.5 )
        ax.set_xlim( the_xlim )

        the_ylim = ax.get_ylim() if ylim is None else ylim
        # TODO Kludge?
        if the_axis[1] == 'time':
            ax.plot( np.array( [0, 0] ), the_ylim, 'g-', linewidth = 1.5 )
        ax.set_ylim( the_ylim )
    
    return _plotter


## === THE MASTER PLOT === ##


def grid_plot ( data, grids,
                err = None,
                plotter = plotter_trace(),
                figsize = (24, 8),
                grid_data_axis = 'channel',
                grid_left = 0.05,
                grid_sep = 0.02,
                grid_names = True,
                xlabel = 'Time (s)',
                ylabel = '',
                origin_label = '0',
                **kwargs ):

    ## ...
    if isinstance( grid_data_axis, str ):
        grid_data_axis = [ grid_data_axis ]
    
    ## Parse grids
    # If too shallow, pop it up
    depth = lambda L: isinstance( L, list ) and max( map( depth, L ) ) + 1
    grids_depth = depth( grids )
    if grids_depth == 1:
        raise # TODO Reshape as square.
    elif grids_depth == 2:
        grids = [ grids ]
    
    # ... For now, assume in the correct format.
    grid_sizes = [ ( len( g ), len( g[0] ) ) for g in grids ]
    
    total_cols = sum( [ gsz[1] for gsz in grid_sizes ] )
    col_width = ( 1.0 - ( grid_left + len( grids ) * grid_sep ) ) / total_cols
    
    grid_widths = [ col_width * gsz[1] for gsz in grid_sizes ]
    grid_lefts = list( itertools.accumulate( [ grid_left ] + grid_widths,
                                             lambda x, y : x + y + grid_sep ) )[:-1]
    
    f = plt.figure( figsize = figsize )
    
    border_space = 0.05
    grid_specs = [ gridspec.GridSpec( *gsz ) for gsz in grid_sizes ]
    for i in range( len( grid_specs ) ):
        grid_specs[i].update( left = grid_lefts[i],
                              right = grid_lefts[i] + grid_widths[i],
                              wspace = border_space,
                              hspace = border_space )
    
    first_valid_1d = lambda L : next( ( i for i, j in enumerate( L )
                                        if not (j is None) ),
                                      None )
    first_valid_2d = lambda M : next( ( (i, j) for i, j in enumerate( map( first_valid_1d, M ) )
                                        if not (j is None) ),
                                      None )
   
    anchor_idxs = list( map( first_valid_2d, grids ) )
    anchors = list( map( lambda spec, anchor_idx : f.add_subplot( spec[anchor_idx] ),
                                grid_specs, anchor_idxs ) )
    
    plt_axes = list( map( lambda spec, grid, anchor, anchor_idx :
                                      [ [ None if grid[i][j] is None
                                          else ( f.add_subplot( spec[i, j],
                                                                sharex = anchor,
                                                                sharey = anchor )
                                            if not (i, j) == anchor_idx
                                            else anchor )
                                          for j in range( spec.get_geometry()[1] ) ]
                                        for i in range( spec.get_geometry()[0] ) ],
                                 grid_specs, grids, anchors, anchor_idxs ) )
    
    ## Make all plots
    
    # TODO Just add this to maxcog
    def _tupelize ( x ):
        if isinstance( x, str ):
            return (x,)
        return x
    
    for i_grid in range( len( plt_axes ) ):
        
        grid_axes = plt_axes[i_grid]
        
        for i in range( len( grid_axes ) ):
            for j in range( len( grid_axes[0] ) ):
                
                ax = grid_axes[i][j]
                if ax is None:
                    continue
                
                cur_slice = functools.reduce( lambda x, y : x + y,
                                              zip( grid_data_axis,
                                                   _tupelize( grids[i_grid][i][j] ) ) )
                cur_slice += ('labeled',)
                
                if err is None:
                    plotter( ax,
                             data[cur_slice],
                             **kwargs )
                else:
                    plotter( ax,
                             data[cur_slice],
                             err[cur_slice],
                             **kwargs )
                
                # Channel names
                if grid_names:
                    text_bump = 0.03
                    text_size = 10
                    
                    text_pos_x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * text_bump
                    text_pos_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * text_bump
                    
                    ax.text( text_pos_x, text_pos_y, grids[i_grid][i][j],
                             verticalalignment = 'top',
                             horizontalalignment = 'left',
                             fontsize = text_size )
    
    # Clean up axis labels
    plt.setp( [a.get_xticklabels() for a in f.axes], visible=False )
    plt.setp( [a.get_yticklabels() for a in f.axes], visible=False )
    
    # Thicken plot borders
    border_width = 1.0
    [ [ i.set_linewidth( border_width )
        for i in a.spines.values() ]
      for a in f.axes ]
    [ [ i.set_color( '#cccccc' )
        for i in a.spines.values() ]
      for a in f.axes ]
    
    f.canvas.draw()
    
    # Put tick labels on lower-left plot of each grid
    for ll_ax in [ grid_axes[-1][0] for grid_axes in plt_axes ]:
        # Swap out label for x origin.
        ll_ax.set_xticklabels( [ origin_label if ll_ax.get_xticks()[i] == 0
                                 else ll_ax.get_xticklabels()[i].get_text()
                                 for i in range( len( ll_ax.get_xticks() ) ) ] )
        # Turn on visibility for x origin and axis ends.
        plt.setp( [ ll_ax.get_xticklabels()[i]
                    for i in range( -1, len( ll_ax.get_xticks() - 1 ) )
                    if i in [-1, 0] or ll_ax.get_xticks()[i] == 0 ],
                  visible = True )
        # Turn on visibility for y.
        plt.setp( ll_ax.get_yticklabels(), visible = True )

    # Put axis labels on first grid only.
    ax = plt_axes[0][-1][0]
    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )

    plt.show()














