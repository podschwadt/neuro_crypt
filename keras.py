import tensorflow as tf
from data import load, build_cross_validation_folds, data_files
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


# constants
data_name = 'fbirn'
n_filters = [ 6 ] # number of filters to try,  
auc_threshold = 0.75


def print_fn( *values, file=None ):
    if file is not None:
        if not hasattr( file, 'write' ):
            with open( file, 'a+' ) as f:
                print( *values, file=f  )
        else:
            print( *values, file=file  )
    print( *values )

# load data
(x_train, y_train), _ = load( data_name, as_torch_dataset=False, test_size=0 )
print( x_train.shape )
x_train = np.moveaxis( x_train, 1 , 2 )
print( x_train.shape )
data_folds = build_cross_validation_folds( x_train, y_train, folds=10 )
train_mean = []
train_std = []
test_mean = []
test_std = []
print( data_name )
for filters in n_filters:
    train_auc = []
    test_auc = []
    print_fn( 'filters', filters, file=data_name +'.log' )
    for i, ( (x_train, y_train), ( x_test, y_test ) ) in enumerate( data_folds ):
        x_val = x_train[ : int( x_train.shape[ 0 ] * 0.2 ) ] 
        y_val = y_train[ : int( y_train.shape[ 0 ] * 0.2 ) ]

        x_train = x_train[ int( x_train.shape[ 0 ] * 0.2 ): ] 
        y_train = y_train[ int( y_train.shape[ 0 ] * 0.2 ): ]


        while True:
            early_stopping = tf.keras.callbacks.EarlyStopping( patience=128, restore_best_weights=True ) 

            l1 = 0.03
            # build model
            model = tf.keras.models.Sequential()
            model.add( tf.keras.layers.Conv1D( filters, kernel_size=3, strides=2, activation='relu', input_shape=x_train.shape[ 1: ], kernel_regularizer=tf.keras.regularizers.l1( l1), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )
            model.add( tf.keras.layers.MaxPool1D( pool_size=15, strides=12 ) )
            model.add( tf.keras.layers.Flatten() )
            model.add( tf.keras.layers.Dense( 1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1( l1 ), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )

            model.compile( optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )
            # model.summary()

            model.fit( x_train, y_train, batch_size=32, epochs=256, validation_data=(x_val,y_val), callbacks=[ early_stopping ], verbose=0 )
            val_auc = model.evaluate( x_val, y_val, verbose=0  )[ 1 ]
            if val_auc > auc_threshold:
                break

            print( 'restarting fold {} val auc was {:.3f}'.format( i, val_auc ) )
            tf.keras.backend.clear_session()

        train_auc.append( model.evaluate( x_train, y_train, verbose=0  )[ 1 ] )
        test_auc.append( model.evaluate( x_test, y_test, verbose=0 )[ 1 ] )


    mean = np.mean( train_auc )
    std = np.std( train_auc )
    train_mean.append( mean )
    train_std.append( std )
    print_fn( '\tmean\tstd\tmin\tmax', file=data_name +'.log' ) 
    print_fn( 'train\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format( mean, std , np.min( train_auc ), np.max( train_auc ) ),  file=data_name +'.log' ) 
    mean = np.mean( test_auc )
    std = np.std( test_auc )
    test_mean.append( mean )
    test_std.append( std )
    print_fn( 'test\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(  mean, std, np.min( test_auc ), np.max( test_auc ) ),  file=data_name +'.log' ) 
    print_fn( 'params\t{}'.format( tf.python.keras.utils.layer_utils.count_params( model.trainable_weights ) ),  file=data_name +'.log' ) 

plt.ioff()
plt.ylim( (0,1) )
plt.errorbar( x=n_filters, y=train_mean, yerr=train_std, label='train', fmt='go', ecolor='k' )
plt.errorbar( x=n_filters, y=test_mean, yerr=test_std, label='test', fmt='mo', ecolor='k' )
plt.title( data_name +' k-fold cross validation (k=10)' )
plt.xlabel( '# Filters' )
plt.ylabel( 'AUC' )
plt.grid(True)
plt.legend()


plt.savefig( data_name + '.pdf'  ) 
plt.clf()