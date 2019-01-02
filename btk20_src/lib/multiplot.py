#!/usr/bin/python

#====================================================================#
# Author: Uwe Mayer                                                  #
# Date: 2004-10-06                                                   #
#                                                                    #
# Creates multiple plots, supports lightweight animations for        #
# debugging and presentation purposes.                               #
#====================================================================#


import Gnuplot
from Gnuplot.utils import float_array, write_array
from Gnuplot.PlotItems import _InlineFileItem, _FIFOFileItem

from Numeric import *
from select import select
from types import *


def a2str(vector):
    """converts an array to a string that can be used with gnuplot

    By setting a Gnuplot.Data object's attribute <content> the the
    result of this output, updates the Gnuplot internal data; a
    succsequent replot() or plot() command will show the new data..
    """
    # convert to a large float array 
    vector = asarray(vector, Float64)

    # use Gnuplot's way of conversion
    # (taken from Gnuplot.PlotItems.Data function)
    dim = len(shape(vector))
    if (dim == 1): vector = vector[:,NewAxis]
    f = StringIO()
    write_array(f, vector)

    return f.getvalue()



class FeaturePlot(_InlineFileItem):
    """plot feature vectors, possibly as animations"""

    def __init__(self, arg, **kwargs):
        """create and return a _FileItem representing the data from *args

        If passed a single array with one dimension then each point is
        plotted against its index.

        If passed a function the plot is initialised with a zero-plot.
        Upon calling the update() method the plot contents is updated.

        If <None> is passed as source argument the plot is initialised
        with a zero-plot and receives its values by calling the update()
        method with a vector as argument.

        Last but not least a generator can be passed as data source. The
        plot is initialised with a zero-plot. Upon calling the updated()
        method the plot contents is updated.
        """
        self.getFeature = lambda: arg   # feature generator 
        content = None
        
        #-- process types of parameters        
        if (type(arg) in [FunctionType, MethodType, BuiltinFunctionType]):
            self.getFeature = arg
            content = "0\n"

        elif (type(arg) is GeneratorType):
            self.getFeature = arg.next
            content = "0\n"

        elif (arg == None):
            self.getFeature = lambda: "0\n"
            content = "0\n"
            
        elif (type(arg) in [ArrayType, ListType]):            
            content = a2str(arg)

        else:
            raise TypeError("don't know how to convert \"%s\" to a plotable format"%(str(type(arg)),))
        
        #-- output the content into a string:     
        _InlineFileItem.__init__(self, content, **kwargs)


    def update(self, vector=None):
        """updates the internal value to the contents of <vector>"""
        if (type(vector) in [ArrayType, ListType]):            
            self.content = a2str(vector)
        else:
            self.content = a2str(self.getFeature())



#======================================================================#
# Multiplot simplifies multiple plot creation                          #
#======================================================================#
class Multiplot:
    """Aids in creating animations with multiple plots.

    Acts as a 2D array of plots and displays them in a Gnuplot
    grid. However, if there is only one plot Multiplot does not
    switch to Gnuplot multiplot mode.
    The different plots must be created by-hand. This is good,
    when you want to create animations simply update the plot's
    variable \"content\" and assign it the new data (see a2str)
    and call the Multiplot instance again to update it.
    """
    def __init__(self, dim=(0,0), *args, **kwargs):
        """Create multiplot of <dim> dimensionality.

        Param: dim  2-tuple with (<num-rows>,<num-cols>)
        """
        assert(type(dim) is TupleType)
        assert(len(dim) == 2)

        self.rows = dim[0]
        self.cols = dim[1]
        self.is_firstplot = True        # nothing plotted yet
        self.prev_x_instance = None     # previous x-instance

        self.p = []                     # list of plots
        for y in range(dim[0]):
            self.p.append([])           # add new row
            for x in range(dim[1]):
                self.p[y].append([])    # add new col

        self.g = Gnuplot.Gnuplot(*args, **kwargs)

    def __getitem__(self, key):
        """return item at (row,col)"""
        assert(type(key) is TupleType)
        assert(len(key) == 2)
        return self.p[key[0]][key[1]]

    def __setitem__(self, key, value):
        """set value (a Gnuplot.PlotItem) at key (row,col)"""
        assert(type(key) is TupleType)
        assert(len(key) == 2)

        # convert single value to a list of values
        if (not type(value) in [ListType, TupleType]): value = [value]
        
        # assert listed items are Gnuplot.PlotItems
        assert(len(filter(lambda a: not isinstance(a, Gnuplot.PlotItem),value)) == 0)

        # auto-extend rows
        updateCols = False              # when updating rows, update cols, too
        if (key[0] >= self.rows):
            self.p.extend([[] for _ in range(key[0] -self.rows+1)]) # append appropriate number of cols
            self.rows = key[0]+1
            updateCols = True           # if updated rows, update cols, too

        # auto-extend cols
        if (key[1] >= self.cols) or updateCols:
            for row in range(self.rows): # check and update all cols to same width
                self.p[row].extend([[] for _ in range(key[1] -len(self.p[row]) +1)])
            self.cols = key[1] +1

        # update settings
        self.is_multiplot = not (self.rows == self.cols == 1)
        self.y_size = 1.0/self.rows
        self.x_size = 1.0/self.cols
        self.p[key[0]][key[1]] = value  # finally set value

    def __call__(self, *args):
        self.g(*args)

    def __calcXDim(self, row,col):
        """calculates the columns for a plot"""
        # a plot takes multiple horizontal spaces if it contains the
        # same instance
        cur_plot = self.p[row][col]
        count = 0                       # count parallel plot instances

        while ((col < self.cols) and (cur_plot == self.p[row][col])):
            count += 1
            col += 1
        return count
        
    def __calcYDim(self, row,col):
        """calculates the rows for a plot"""
        # a plot takes multiple vertical spaces if it contains the
        # same instance
        cur_plot = self.p[row][col]
        count = 0                       # count parallel plot instances
        
        while ((row >= 0) and
               (cur_plot == self.p[row][col])):
            count += 1
            row -= 1
        return count

    def __hasPrevPlot(self, row,col):
        """tests wether plot at [row,col] """

    def plot(self, delay=0):
        """plots the current configuration of plots"""
        # no multiplot
        if (not self.is_multiplot):
            # on first plot do normal plotting
            if (self.is_firstplot):
                self.is_firstplot = False
                if (len(self.p[0][0]) != 0): self.g.plot(*self.p[0][0])
            # next time only refresh
            else:
                self.g.refresh()

        # multiplot
        else:
            self.g('set multiplot')     # auto clears the plotting area

            y_span = [0 for x in range(self.cols)] # number of identical previous plots in y-direction

            pos = [0.0, 0.0] # (x,y) positions, starting from the bottom, left
            for row in range(self.rows-1, -1, -1): # reverse processing
                row_el = self.p[row]
                x_span = 0              # number of same previous plots in x-direction

                for col in range(self.cols):
                    col_el = row_el[col]
                    
                    # if a previous plot takes this space, too
                    if (x_span == 0) and (y_span[col] == 0):
                        x_span = self.__calcXDim(row,col) # x-span for this plot
                        y_span[col] = self.__calcYDim(row,col) # y-span for this plot

                        # prevent top-right corner overlaps:
                        # if current plot span > 1 and the column span of the next block
                        # (if column exists) is > 1 then current plot may not span over
                        # next block
                        if ((x_span > 1) and (col < self.cols) and (y_span[col+1] > 1)):
                            x_span = 1                            

                        # consider simultaneous XY spanning:
                        # if current plot span > 1 and current column span > 1 then
                        # the the next x-span columns have the same y-span, too
                        if (x_span > 1) and (y_span[col] > 1): # consider multi-fields
                            for i in range(x_span): y_span[col+i] = y_span[col]
                            
                        self.g('set size '+str(self.x_size*x_span)+','+str(self.y_size*y_span[col]))
                        self.g('set origin '+str(pos[0])+','+str(pos[1]))
                        if (len(col_el) != 0): # plot if there's something to plot
                            self.g.plot(*col_el)
                            
                    # address next column
                    pos[0] += self.x_size
                    if (x_span > 0): x_span -= 1
                    if (y_span[col] > 0): y_span[col] -= 1

                pos[0] = 0.0            # start at left border again
                pos[1] += self.y_size   # address next row (direction: upwards)

            # clear first-plot flag (not used in multiplot)
            if (self.is_firstplot):
                self.is_firstplot = False

        # finally wait no of ms
        select([],[],[],delay)



#-- test -----------------------------------------------------------------------
if (__name__ == "__main__"):
    from btk.signalGen import WaveFeature

    WINLEN = 46

    sin1_plot = FeaturePlot(iter(WaveFeature(WINLEN, frequency=2.05)), with='lines 1')

    sinIter2 = iter(WaveFeature(WINLEN, frequency=4.05))
    sin2_plot = FeaturePlot(sinIter2, with='lines 2')

    cosIter1 = iter(WaveFeature(WINLEN, frequency=2.1, f=cos))
    cos1_plot = FeaturePlot(cosIter1, with='lines 3')

    cosIter2 = iter(WaveFeature(WINLEN, frequency=4.1, f=cos))
    cos2_plot = FeaturePlot(cosIter2, with='lines 4')

    centerIter = iter(WaveFeature(WINLEN, frequency=3.1, f=tan, typecode=Float64))
    center_plot = FeaturePlot(centerIter, with='lines 5')

    G = Multiplot((6,5), persist=1)
    G('set grid')
    G('set yrange [-100:100]')
    G[(0,1)] = cos1_plot
    
    G[(1,0)] = cos2_plot
    G[(1,1)] = cos2_plot
    G[(1,2)] = sin2_plot

    G[(2,0)] = sin1_plot
    G[(3,0)] = sin1_plot
    G[(4,0)] = sin1_plot

    G[(2,1)] = center_plot
    G[(2,2)] = center_plot
    G[(2,3)] = center_plot
    G[(3,1)] = center_plot
    G[(3,2)] = center_plot
    G[(3,3)] = center_plot
    G[(4,1)] = center_plot
    G[(4,2)] = center_plot
    G[(4,3)] = center_plot

    G[(5,1)] = cos1_plot
    G[(5,2)] = cos1_plot

    G[(0,3)] = cos1_plot
    G[(0,4)] = cos1_plot
    G[(1,3)] = cos1_plot
    G[(1,4)] = cos1_plot
    G[(2,4)] = cos1_plot #<-- top right corner creates overlaps
    G[(3,4)] = cos1_plot

    for _ in range(10):
        sin1_plot.update()
        sin2_plot.update()

        cos1_plot.update(cosIter1.next())
        cos2_plot.update(cosIter2.next())

        center_plot.update(centerIter.next())

        G.plot(0.3)
        
