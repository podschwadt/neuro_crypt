import numpy as np
import matplotlib.pylab as plt
import os
plt.ion()
# plt.show()


class Metric( object ):
    
    def __init__( self, style ):
        self.style = style
        self.values = []


class ProgressPlot( object ):
    
    def __init__( self, steps, y_lim=(0,1.1)):
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.1, 0.1, 0.6, 0.75])
        self.ax.set_ylim( y_lim )
        # plt.xlim( (0, steps) )
        self.x = np.arange( steps )
        self.metrics = {}
        self.is_init = False

    def add( self, name, style ):
        self.metrics[ name ] = Metric( style )

    def init( self ):
        for m in self.metrics:
            metric = self.metrics[ m ]
            self.ax.plot( self.x[ :len( metric.values ) ], metric.values, metric.style, label=m )
              
        self.ax.legend( bbox_to_anchor=(1.04, 0.5), loc="center left" )
        # self.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        self.is_init = True

    def update( self, **kwargs ):
        if not self.is_init:
            self.init()
        # self.fig.clear()
        for m in kwargs:
            metric = self.metrics[ m ]
            metric.values.append( kwargs[ m ] )
            self.ax.plot( self.x[ :len( metric.values ) ], metric.values, metric.style, label=m )
        self.ax.axhline( y=0.5, color='k', linestyle=':', label='0.5', alpha=0.5 )
        self.ax.axhline( y=0.7, color='y', linestyle=':', label='at least beat this', alpha=0.5 )              
        # plt.legend( bbox_to_anchor=(1.04, 0.5), loc="center left" )
        plt.pause( 0.00001 )

    def save( self, file, dir=None ):
        plt.savefig( os.path.join( dir, file ) )
    
    def block( self) :
        plt.show( block=True )




