import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import csv
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import syft as sy
import data as D
import pickle
from utils import XavierUniformWithGain

plot = False
if plot:
    from plot import ProgressPlot

threshold = 0.7
n_sites = 4
glabal_iterations = 64
local_iterations = 2
parameters = [                          # ( name, n_filters, l1 )
                ('fbirn', 16, 0.003) , 
                # ('abide', 12, 0.1 ), 
                # ('cobre', 8  ,0.1), 
                # ('oasis', 12, 0.1) 
            ]



def validation_split( x_train, y_train, split=0.2 ):
    x_val = x_train[ : int( x_train.shape[ 0 ] * split ) ] 
    y_val = y_train[ : int( y_train.shape[ 0 ] * split ) ]

    x_train = x_train[ int( x_train.shape[ 0 ] * split ): ] 
    y_train = y_train[ int( y_train.shape[ 0 ] * split ): ]

    return (x_train, y_train), (x_val, y_val)

def get_model_gradeint(model, t_x, t_y, reg=None):
    # print( t_x.shape, t_y.shape)
    with tf.GradientTape() as tape:
        y_pred = model( t_x )  # forward-propagation
        loss = model.loss( y_true=t_y, y_pred=y_pred )  # calculate loss
        if reg is not None:
            loss += reg( model.layers[ 0 ].kernel ) + reg( model.layers[ 0 ].kernel )
            loss += reg( model.layers[ 3 ].kernel ) + reg( model.layers[ 3 ].kernel )
    gradients = tape.gradient( loss, model.trainable_weights )  # back-propagation
    # print( 'grads', len(gradients), gradients[0].shape )

    return gradients

  
for name, n_filters, l1 in parameters:

    (x_train, y_train), _ = D.load( name, as_torch_dataset=False, test_size=0.0 )
    x_train = np.moveaxis( x_train, 1 , 2 )
    print( 'train data:', x_train.shape )
        
    sample_shape = x_train.shape[ 1: ]

    x = np.array_split( x_train, n_sites )
    y = np.array_split( y_train, n_sites )
    data_folds = [ d for d in zip( x,y ) ]
    print(  )
    print( 'data', data_folds[0][0].shape )
    print( 'data', data_folds[0][1].shape )

    def create_model( seed=65440 ):
        # l1 = 0.5
        # l1 = 0.3
        # init = tf.keras.initializers.GlorotUniform( seed=np.random.randint( 0, 65440 ) )
        init = XavierUniformWithGain( gain=0.6, seed=seed )
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.Conv1D( n_filters, kernel_size=3, strides=2, activation='relu', input_shape=sample_shape, kernel_initializer=init, 
                                            kernel_regularizer=tf.keras.regularizers.l1( l1), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )
        model.add( tf.keras.layers.MaxPool1D( pool_size=15, strides=12 ) )
        model.add( tf.keras.layers.Flatten() )
        model.add( tf.keras.layers.Dense( 1, activation='sigmoid', kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l1( l1 ), bias_regularizer=tf.keras.regularizers.l1( l1 ) ) )

        # model.compile( optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )
        model.compile( optimizer=tf.keras.optimizers.Adam(  learning_rate=0.0001 ), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[ 'AUC' ] )

        return model
      # setup sites and workers
    hook = sy.TorchHook(torch)
    site_workers = [ sy.VirtualWorker( hook, id="site" + str( i ) ) for i in range( n_sites ) ]
    secure_worker = sy.VirtualWorker( hook, id="secure_worker" )

    for i, site in enumerate( site_workers ):
        wrkrs = [ w for j,w in enumerate( site_workers ) if j != i ]
        site.add_workers( wrkrs )
        site.add_worker( secure_worker )
    secure_worker.add_workers( site_workers )

    def train_distributed( data_folds, workers, holdout_id ):
        p_ids = list( range( len( workers ) ) ) # participant ids
        p_ids.remove( holdout_id )
        print( 'training on', p_ids, 'testing on', holdout_id )
        p_workers = [ w for i, w in enumerate( workers) if i != holdout_id ]

        rnd = np.random.default_rng( )
        l1_reg = tf.keras.regularizers.l1( l1 )
        while True:
            if plot:
                progress_plt = ProgressPlot( glabal_iterations )
                styles = [ 'g-', 'r-', 'c-' ]
            # create models 
            models = {}
            seed = rnd.integers( 8456963854 )
            for i,id in enumerate( p_ids ):
                models[ id ] = create_model( seed=seed )
                if plot:
                    progress_plt.add( str( id ), styles[ i ] )

            # train loop
            for e in range( glabal_iterations ):
                # get gradients for each site
                grads = {}
                for id in models:
                    model = models[ id ]
                    # get the gradient info for the site 
                    (x_train, y_train), _ = validation_split( data_folds[ id ][0], data_folds[ id ][ 1 ] )
                    model.fit( x_train, y_train, epochs=local_iterations, verbose=0 )
                    grads[ id ] = get_model_gradeint( model, x_train, y_train, reg=l1_reg )

                # secret share the grad info
                ptr_list = [ singleGradientSplit( grads[ i ], p_workers ) for i in p_ids ]
                # calcuclate the sum
                grad_sums = []
                # do one round by hand to init the list
                for i in range( len( ptr_list[ 0 ] ) ): 
                    grad_sums.append( ptr_list[ 0 ][ i ] + ptr_list[ 1 ][ i ] )

                # add the rest of the sites
                for i in range( 2, len( ptr_list ) ):
                    for j in range( len( ptr_list[ 0 ] ) ): 
                        grad_sums[ j ] += ptr_list[ i ][ j ] 

                # print( [ g.shape for g in grad_sums ] )
                grad_sums = [ itm.get().data.float().numpy() / 1000000 / len( p_ids ) for itm in grad_sums ]
                # print( [ g.shape for g in grad_sums ] )

                # update model weights
                train_f1 = {}
                for id in models:
                    model = models[ id ]
                    # get the gradient info for the site data
                    model.optimizer.apply_gradients( zip( grad_sums, model.trainable_variables ) )
                    _, (x_val, y_val) = validation_split( data_folds[ id ][0], data_folds[ id ][ 1 ] )
                    train_f1[ str( id ) ] = roc_auc_score( y_val, model.predict( x_val ).reshape( -1 ) )
                
                if plot:
                    progress_plt.update( **train_f1 )
            #training is done
            # this is where the restarting happens
            stop = True
            for id in models:
                model = models[ id ]
                _, (x_val, y_val) = validation_split( data_folds[ id ][0], data_folds[ id ][ 1 ] )
                val_auc = roc_auc_score( y_val, model.predict( x_val ).reshape( -1 ) )
                if val_auc <= threshold:
                    print( 'restarting at: ', val_auc ) 
                    stop = False
                    break
            if not stop:
                tf.keras.backend.clear_session()
                continue # restarting

            f1 = []
            auc = []
            for _, model in models.items():
                y_pred = model.predict( data_folds[ holdout_id ][ 0 ] ).reshape( -1 )
                f1.append( f1_score( data_folds[ holdout_id ][ 1 ], y_pred  > 0.5 )  )
                auc.append( roc_auc_score( data_folds[ holdout_id ][ 1 ], y_pred ) )
            return f1, auc # while loop ends

    def print_fn( *values, file=None ):
        if file is not None:
            if not hasattr( file, 'write' ):
                with open( file, 'a+' ) as f:
                    print( *values, file=f  )
            else:
                print( *values, file=file  )
        print( *values )

    def singleGradientSplit( singleGradinent, sites ):
        rtnList =[]
        for itm in singleGradinent:
            rtnList.append( torch.from_numpy(itm.numpy() *1000000).share( *sites ) )
        return rtnList

    print_fn( '', file= name + '.log' )
    f1s = []
    aucs = []
    for fold_id in range( n_sites ):
        f1, auc = train_distributed( data_folds, site_workers, holdout_id=fold_id )
        print_fn( 'fold {}\t'.format( fold_id ), f1, auc, file= name + '.log' )
        f1s.append( f1 )
        aucs.append( auc )
    tf.keras.backend.clear_session()
    pickle.dump( { 'f1':f1s, 'auc': aucs },  open( name + '_neurcrypt.pik', 'wb' ) )

import plot_results