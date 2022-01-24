from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import roc_auc_score
from data import load
import numpy as np

(x_train, y_train), (x_test, y_test)  = load( 'fbirn', as_torch_dataset=False, test_size=0.3 )
samples = np.zeros( ( x_train.shape[ 0 ], x_train.shape[ 1 ] * x_train.shape[ 1 ] - x_train.shape[ 1 ] ) )
for i, x in enumerate( x_train ):  
    c =  np.corrcoef( x )
    n = 0
    for j in range( c.shape[ 0 ] ):
        for k in range( 0, j ):
            samples[ i, n ] = c[ j,k ]
            n += 1
x_train = samples

samples = np.zeros( ( x_test.shape[ 0 ], x_test.shape[ 1 ] * x_test.shape[ 1 ] - x_test.shape[ 1 ] ) )
for i, x in enumerate( x_test ):  
    c =  np.corrcoef( x )
    n = 0
    for j in range( c.shape[ 0 ] ):
        for k in range( 0, j ):
            samples[ i, n ] = c[ j,k ]
            n += 1

x_test = samples
print( x_train.shape, y_train.shape )
clf = MLP( hidden_layer_sizes= (50, 25) ) # , (50, 25) 
clf.fit( x_train, y_train )

print( clf.get_params( deep=False ) )
print( clf.score(x_test, y_test) )
print( clf.predict_proba( x_test )[:,1] ) 

print( roc_auc_score( y_test, clf.predict_proba( x_test )[:,1] ) )