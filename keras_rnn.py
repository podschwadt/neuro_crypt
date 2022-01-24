import tensorflow as tf
from data import load, build_cross_validation_folds
import numpy as np
import plot
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load data
(x_train, y_train), _ = load( 'fbirn', as_torch_dataset=False, test_size=0 )
print( x_train.shape )
x_train = np.moveaxis( x_train, 1 , 2 )
print( x_train.shape )
data_folds = build_cross_validation_folds( x_train, y_train, folds=10 )

train_auc = []
test_auc = []
for i, ( (x_train, y_train), ( x_val, y_val ) ) in enumerate( data_folds ):
    print( 'running fold:', i )
    # plotter = plot.ProgressPlot( 256 )
    # plotter.add( 'loss', 'g-' )
    # plotter.add( 'auc', 'm-' )
    # plotter.add( 'val_loss', 'g--' )
    # plotter.add( 'val_auc', 'm--' )

    # cb = tf.keras.callbacks.LambdaCallback( on_epoch_end=lambda epoch, logs: plotter.update( **logs ) )
    early_stopping = tf.keras.callbacks.EarlyStopping( patience=128, restore_best_weights=True ) 

    l1 = 0.03
    # build model
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Conv1D( 4, kernel_size=3, strides=2, activation='relu', input_shape=x_train.shape[ 1: ], kernel_regularizer=tf.keras.regularizers.l1( l1), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )
    model.add( tf.keras.layers.MaxPool1D( pool_size=15, strides=12 ) )
    # model.add( tf.keras.layers.Dropout( 0.5 ) )
    model.add( tf.keras.layers.Flatten() )
    model.add( tf.keras.layers.Dense( 1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1( l1 ), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )

    model.compile( optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )
    # model.summary()

    model.fit( x_train, y_train, batch_size=32, epochs=256, validation_split=0.2, callbacks=[ early_stopping ], verbose=0 )

    train_auc.append( model.evaluate( x_train, y_train, verbose=0  )[ 1 ] )
    test_auc.append( model.evaluate( x_val, y_val, verbose=0 )[ 1 ] )
    print( 'auc train {:.3f}, test {:.3f}:'.format( train_auc[ - 1 ], test_auc[ -1 ] ) )

print( '\tmean\tstd\tmin\tmax' )
print( 'train\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}:'.format( np.mean( train_auc ), np.std( train_auc ), np.min( train_auc ), np.max( train_auc ) ) )
print( 'test\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}:'.format( np.mean( test_auc ), np.std( test_auc ), np.min( test_auc ), np.max( test_auc ) ) )


