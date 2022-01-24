import tensorflow as tf
from data import load, build_cross_validation_folds, data_files
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
from utils import XavierUniformWithGain


n_sites = 4
threshold = 0.8

def test_single_site( model, site_id, x_data, y_data, names=None ):
    """
    Test a single site model on the data of all other sites.

    model: keras model of the site
    site_id: int id of the site the model belongs to
    x_data: list of np.ndarrays contaiing the site data
    y_data: list of np.ndarrays contaiing the site labels
    """
    # load train and validation data
    x_train = x_data[ site_id ]
    y_train = y_data[ site_id ]
    # create test data
    x_test = [ x_data[ i ] for i in range( len( x_data ) ) if i != site_id ]
    y_test = [ y_data[ i ] for i in range( len( y_data ) ) if i != site_id ]

    result_dict = {}
    acc = []
    f1 = []
    auc = []

    for x, y_true, key in zip( [ x_train, x_test ], [ y_train, y_test ], [ 'train', 'test' ] ):
        if key == 'test':
            for _x, _y in zip( x, y_true ):
                y = model.predict( _x ).reshape( -1 )       
                acc.append( accuracy_score( _y , y > 0.5 ) )
                f1.append( f1_score( _y, y > 0.5 ) )
                auc.append( roc_auc_score( _y, y ) )
            d = {
                'acc': acc,
                'f1': f1,
                'auc': auc
            }
        else:
            y = model.predict( x ).reshape( -1 )
            d = {
                'acc': accuracy_score( y_true, y > 0.5),
                'f1': f1_score( y_true, y > 0.5 ),
                'auc': roc_auc_score( y_true, y ) 
            }
        result_dict[ key ] = d
    return result_dict

 

def print_fn( *values, file=None ):
    if file is not None:
        if not hasattr( file, 'write' ):
            with open( file, 'a+' ) as f:
                print( *values, file=f  )
        else:
            print( *values, file=file  )
    print( *values )


        


def validation_split( x_train, y_train, split=0.2 ):
    x_val = x_train[ : int( x_train.shape[ 0 ] * split ) ] 
    y_val = y_train[ : int( y_train.shape[ 0 ] * split ) ]

    x_train = x_train[ int( x_train.shape[ 0 ] * split ): ] 
    y_train = y_train[ int( y_train.shape[ 0 ] * split ): ]

    return (x_train, y_train), (x_val, y_val)

def train_and_eval( data_name, n_filters, l1=0.03 ):
    # constants

    # load data
    (x_train, y_train),_  = load( data_name, as_torch_dataset=False, test_size=0.0 )
    x_train = np.moveaxis( x_train, 1 , 2 )
    print( 'train data:', x_train.shape )
    

    def train( x_train, y_train, seed=65440, gain=1.0 ):
        (x_train, y_train), (x_val, y_val) = validation_split( x_train, y_train )
        early_stopping = tf.keras.callbacks.EarlyStopping( patience=128, restore_best_weights=True ) 
        # init = tf.keras.initializers.GlorotUniform( seed=1324 )

        while True:
            # build model
            model = tf.keras.models.Sequential()
            init = XavierUniformWithGain(  gain=0.8,  seed=np.random.randint( seed ) )
            model = tf.keras.models.Sequential()
            model.add( tf.keras.layers.Conv1D( n_filters, kernel_size=3, strides=2, activation='relu', input_shape=x_train.shape[ 1: ], kernel_initializer=init, 
                                                kernel_regularizer=tf.keras.regularizers.l1( l1), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )
            model.add( tf.keras.layers.MaxPool1D( pool_size=15, strides=12 ) )
            model.add( tf.keras.layers.Flatten() )
            model.add( tf.keras.layers.Dense( 1, activation='sigmoid', kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l1( l1 ), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )

            # opt = tf.keras.optimizers.SGD()
            model.compile( optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )
            model.fit( x_train, y_train, batch_size=32, epochs=256, validation_data=(x_val,y_val), callbacks=[ early_stopping ], verbose=0 )
    
            val_auc = roc_auc_score( y_val,  model.predict( x_val ).reshape( -1 ) )
            if val_auc >= threshold:
                break
            print( 'restarting at: ', val_auc )  
        # model.compile( optimizer=tf.keras.optimizers.Adam(  learning_rate=0.0001 ), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )
        # model.fit( x_train, y_train, epochs=128, batch_size=32, verbose=0 )

        return model


    # print_fn( '+++++++++++', file=data_name +'.log' )

    plt.ioff()
    plt.ylim( (0,1.1) )

    # setup the data 
    # split into n sites
    names = [ 'site' + str( i + 1 ) for i in range( n_sites ) ]
    train_data =  np.array_split( x_train, n_sites )
    train_labels = np.array_split( y_train, n_sites )

    # train a model for each site:
    print( 'single site' )
    site_models = [ train( train_data[ i ], train_labels[ i ]  ) for i in range( n_sites ) ]
    # eval the models
    eval_site_models = [ test_single_site( site_models[ i ], i, train_data, train_labels, names=None ) for i in range( n_sites ) ]
    pickle.dump( eval_site_models, open( data_name + '_singel_site.pik', 'wb' ) )
    print( eval_site_models )


    #pooled cross validation
    def cross_validate( holdout_id, x_data, y_data ):
        print( 'holde out fold', holdout_id )
        # prep data
        x_folds = [ x_data[ i ] for i in range( len( x_data ) ) if i != holdout_id ] 
        y_folds = [ y_data[ i ] for i in range( len( y_data ) ) if i != holdout_id ] 
        x_test = x_data[ holdout_id ]
        y_test = y_data[ holdout_id ]
        # train and eval models

        result_dict = { 'train': { 'acc': [], 'f1': [], 'auc': [] }, 
                        'test': { 'acc': [], 'f1': [], 'auc': [] } }
        for x, y in zip( x_folds, y_folds ):
            model = train( x, y ) 
            # load train and validation data
            x_train = x
            y_train = y

            for x, y_true, key in zip( [ x_train, x_test ], [ y_train, y_test ], [ 'train', 'test' ] ):
                y = model.predict( x ).reshape( -1 ) 
                result_dict[ key ][ 'acc' ].append( accuracy_score( y_true, y > 0.5 ) )
                result_dict[ key ][ 'f1' ].append( f1_score( y_true, y > 0.5 ) )
                result_dict[ key ][ 'auc' ].append( roc_auc_score( y_true, y ) )
        return result_dict

    eval_pooled_models = [ cross_validate( i, train_data, train_labels ) for i in range( n_sites ) ]
    pickle.dump( eval_pooled_models, open( data_name + '_pooled.pik', 'wb' ) )
    print( eval_pooled_models )

# values determined in earlier experiments with cv
parameters = [ ('fbirn', 6, 0.003 ) ] #, ('abide', 12, 0.001 ), ('cobre', 8) ,('oasis', 14) ] # ( dataset, n_filters, l1  )

if __name__ == '__main__':
    for params in parameters:
        train_and_eval( *params )

    import plot_results