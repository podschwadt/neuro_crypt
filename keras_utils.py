import tensorflow as tf
import numpy as np
import plot
from sklearn.metrics import roc_auc_score



class StupidCallback(tf.keras.callbacks.Callback ):
    """
    Do i really need this?
    """

    def __init__( self, model ):
        self.model = model
        self.plotter = plot.ProgressPlot( 256 )
        plotter.add( 'loss', 'g-' )
        # plotter.add( 'acc', 'r-' )
        plotter.add( 'auc', 'm-' )
        plotter.add( 'val_loss', 'g--' )
        # plotter.add( 'val_acc', 'r--' )
        plotter.add( 'val_auc', 'm--' )
    
    def on_epoch_end( self, epochs, logs=None ): 
        # train auc
        y = np.argmax( self.model.predict( x_train ), axis=1 )
        train_auc = roc_auc_score( y_train, y )

        # val auc
        y = np.argmax( self.model.predict( x_val ), axis=1 )
        val_auc = roc_auc_score( y_val, y )

        plotter.update( auc=train_auc, val_auc=val_auc, **logs )