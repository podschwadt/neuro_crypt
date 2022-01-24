import h5py
import os
import numpy as np
from torch.utils.data import Dataset

data_dir = 'data'
data_files = { 'abide': 'ABIDE.h5', 
                'cobre': 'COBRE.h5',
                'fbirn': 'FBIRN.h5', 
                'oasis': 'OASIS.h5' }
test_size = 0.5

class MyDataset( Dataset ):

    def __init__( self, x, y ):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__( self, index ):
        return self.x[ index ], self.y[ index ]

    def __len__( self ):
        return self.x.shape[ 0 ]
    
def build_cross_validation_folds( x, y, folds=5, seed=17 ):
    """
    returns a list of `folds` pairs: [ ( x_train, y_train ), ( x_val, y_val ) ]
    """
    # validation data
    # shuffel data
    rng = np.random.RandomState( seed=17 )
    num_samples = x.shape[ 0 ]
    idx = np.arange( num_samples )
    rng.shuffle( idx )
    # number of samples in the fold
    n_vs = int( num_samples / folds )
    
    data_folds = []
    for i in range( folds ):
        val_idx = idx[ i * n_vs : ( i + 1 ) * n_vs ]
        train_idx = list( set( idx ) - set( val_idx ) )
        data_folds.append( ( ( x[ train_idx ], y[ train_idx ] ), ( x[ val_idx ], y[ val_idx ] ) )  )

    return data_folds


def load( dataset, as_torch_dataset=True, test_size=0.5, seed=39 ):
    """
    return: training , test dataset
    """
    path = os.path.join( data_dir, data_files[ dataset ] )
    print( 'loading from:', path )
    f = h5py.File( path, 'r' )
    print( 'h5 content:', list( f.keys() ) )
    for key in f.keys():
        print( key, f[ key ].shape, f[ key ].dtype )
    x = f[ 'data' ][()]
    y = f[ 'labels' ][()]

    rng = np.random.RandomState( seed=seed )
    idx = np.arange( x.shape[ 0 ] )
    rng.shuffle( idx )
    x = x[ idx ]
    y = y[ idx ]

    # mi = x.min( axis=(0, 2), keepdims=True )
    # mx = x.max( axis=(0, 2), keepdims=True )
    # x = (x - mi ) / ( mx - mi )

    split = int( x.shape[ 0 ] * test_size )
    if as_torch_dataset:
        return MyDataset( x[ split : ], y[ split : ] ), MyDataset( x[ :split ], y[ :split ] )
    else:
        return ( x[ split : ], y[ split : ] ), ( x[ :split ], y[ :split ] )

def load_abide():
    return load( 'abide' )

def load_cobre():
    return load( 'cobre' )

def load_fbirn():
    return load( 'fbirn' )

def load_oasis():
    return load( 'oasis' )


# x, y = load_abide()
# print( 'x', np.min( x ), np.max( x ) )
# print( 'x', 'mean', np.mean( x, axis=1 ) )
# print( 'x', 'var', np.var( x, axis=1 ) )
# print( 'y', np.min( y ), np.max( y ) )

if __name__ == "__main__":
    for key in data_files:
        print( key )
        ( x, y ), _ = load( key, as_torch_dataset=False, test_size=0.0 )
        print( x.shape )
        print( '0s', sum( y ), '1s', abs( sum( y - 1 ) )  )


