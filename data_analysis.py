import data as D
from polyssifier import poly
import numpy as np


( data, labels ), _ = D.load( 'cobre',  as_torch_dataset=False, test_size=0.0 )
print( 'data', data.shape )
samples = np.zeros( ( data.shape[ 0 ], data.shape[ 1 ] * data.shape[ 1 ] - data.shape[ 1 ] ) )
print( 'samples', samples.shape )
for i, x in enumerate( data ):  
    # print( 'i', i )
    # print( 'x', x.shape )
    c =  np.corrcoef( x )
    # print( 'c', c.shape )
    n = 0
    for j in range( c.shape[ 0 ] ):
        for k in range( 0, j ):
            # print( j, k )
            samples[ i, n ] = c[ j,k ]
            n += 1

print( 'samples:', samples.shape )



if __name__ == '__main__':
    pass
    # from multiprocessing import freeze_support
    # freeze_support()


    exclude = [
        'Nearest Neighbors',
        'Linear SVM',
        'RBF SVM',
        'Decision Tree',
        'Random Forest',
        'Logistic Regression',
        'Naive Bayes',
        'Voting Classifier',
        'SVM',
        'Voting' ]

    # Run analysis
    report = poly( samples, labels, n_folds=5, verbose=1, scoring='auc', exclude=exclude  )
    # # Plot results

    report.plot_scores()
    report.plot_features( ntop=1 )
    # report.coefficients