import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def mean_squared_error( Y , y_pred ):
    return tf.reduce_mean( tf.square( y_pred - Y ) )

def mean_squared_error_deriv( Y , y_pred ):
    return tf.reshape( tf.reduce_mean( 2 * ( y_pred - Y ) ) , [ 1 , 1 ] )

def h ( X , weights , bias ):
    return tf.tensordot( X , weights , axes=1 ) + bias

path='/home/ava/codeRep/archive/Admission_Predict.csv'
num_epochs = 10
batch_size = 10
learning_rate = 0.001
data= pd.read_csv(path)

X = pd.DataFrame(data,columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research' ],dtype=np.double)/data.values.max()
Y = pd.DataFrame(data, columns= [ 'Chance of Admit ' ] ,dtype=np.double)
train_features , test_features ,train_labels, test_labels = train_test_split( X , Y , test_size=0.2 )
num_samples = X.shape[0]
dataset = tf.data.Dataset.from_tensor_slices(( X , Y )) 
dataset = dataset.shuffle( 500 ).repeat( num_epochs ).batch( batch_size )
iterator = dataset.__iter__()
num_features = X.shape[1]
weights = tf.random.uniform( (num_features , 1) ,minval=0,maxval=1 ) 
bias = 0

epochs_plot = list()
loss_plot = list()

for i in range( num_epochs ) :
    
    epoch_loss = list()
    for b in range( int(num_samples/batch_size) ):
        x_batch , y_batch = iterator.get_next()
        x_batch=tf.cast(x_batch,np.float32)
        y_batch=tf.cast(y_batch,np.float32)
        output = h( x_batch, weights , bias ) 

        loss = epoch_loss.append( mean_squared_error( y_batch , output ).numpy() )
    
        dJ_dH = mean_squared_error_deriv( tf.cast(y_batch,np.float32) , output)
        dH_dW = x_batch
        dJ_dW = tf.reduce_mean( dJ_dH * dH_dW )
        dJ_dB = tf.reduce_mean( dJ_dH )
    
        weights -= ( learning_rate * dJ_dW )
        bias -= ( learning_rate * dJ_dB ) 
        
    loss = np.array( epoch_loss ).mean()
    epochs_plot.append( i + 1 )
    loss_plot.append( loss ) 
    
    print( 'Loss is {}'.format( loss ) ) 


print('ji')    