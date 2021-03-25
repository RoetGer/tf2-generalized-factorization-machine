import tensorflow as tf

def compute_feature_card(X):
    '''Returns number of unique values per column'''
    
    nunique_vals = [
        tf.unique(X[:,i]).y.shape[0] 
            for i in tf.range(X.shape[1])
    ]
    return tf.cast(nunique_vals, dtype=tf.int32)