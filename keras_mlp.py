import tensorflow as tf
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
model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Dense( 50, activation='relu', input_shape=x_train.shape[ 1: ] ) )
model.add( tf.keras.layers.Dense( 20, activation='relu' ) )
model.add( tf.keras.layers.Dense( 1, activation='sigmoid' ) )

model.compile( optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'accuracy', 'AUC' ] )
model.fit( x_train, y_train, epochs=200 )
print( model.metrics )
print( model.evaluate(x_test, y_test) )
print( roc_auc_score( y_test, model.predict( x_test ).reshape( -1 ) ) )
