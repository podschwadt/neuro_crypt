import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools


parameters = [ 'fbirn' , 'abide', 'cobre', 'oasis' ]
def plot( metric='f1' ):
    row = 0
    col = 0
    fig, axs = plt.subplots(2, 2)
    for data_name in parameters:
        # single site
        ax = axs[ row, col ]
        ax.set_ylim( (0.5,1.1) )
        ax.set_title( data_name.upper() )
        plots = []
        single_site = pickle.load( open( '{}_singel_site.pik'.format( data_name ), 'rb' ) )
        ticks = []
        for i, d in enumerate( single_site ):
            plots.append( d[ 'test' ][ metric ] )
            ticks.append( 'site ' + str( i + 1 ) )
        
        # f1_ss = list( itertools.chain.from_iterable( [ d[ 'test' ][ metric ] for d in single_site ] ) )

        # pooled
        pooled = pickle.load( open( '{}_pooled.pik'.format( data_name ), 'rb' ) )
        f1_pooled = list( itertools.chain.from_iterable( [ d[ 'test' ][ metric ] for d in pooled ] ) )
        plots.append( f1_pooled )
        # f1_pooled = [ np.mean( d[ 'test' ][ 'f1' ] ) for d in pooled ] 

        # distributed     
        dist = pickle.load( open( '{}_neurcrypt.pik'.format( data_name ), 'rb' ) )
        print( dist )
        if metric != 'f1' and isinstance( dist, list ): # legacy
            raise RuntimeError( 'legacy pickles only have f1' )
        if isinstance( dist, dict ): # legacy
            dist = dist[ metric ]
        print( dist )
        # dist = [ np.mean( d ) for d in dist ]
        dist = list( itertools.chain.from_iterable( dist ) )
        plots.append( dist )

        ax.boxplot( plots, showfliers=False )
        ax.set_xticks( list( range( 1, len( plots ) + 1 ) ) )
        ticks.append( 'pooled' )
        ticks.append( 'Neurocrypt' )
        ax.set_xticklabels( ticks ) 
        ax.set_ylabel( metric.upper() )
        ax.grid(True)
        row += 1
        if row > 1:
            col += 1
            row = 0

    plt.tight_layout()
    plt.savefig( '1dcnns_{}.pdf'.format( metric ) )
    plt.show()

plot( 'auc' )