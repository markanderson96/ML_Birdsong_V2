from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Permute, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GRU, LSTM

def CRNN2D(shape, n_classes):
    n_layers = 3
    n_filters = [32, 64, 128] 
    kernel_size = (3, 3)
    activation = 'relu'
    pool_size = [(2, 2), (4, 2), (4, 2)]

    input_shape = (shape[0], shape[1], shape[2])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # create sequential model
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
    
    # first conv layer with shape params
    model.add(Conv2D(n_filters[0], kernel_size=kernel_size, padding='same'))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # other convolutional layers (for loop)
    for layer in range(n_layers - 1):
        model.add(Conv2D(n_filters[layer+1], kernel_size=kernel_size, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channel_axis))
        model.add(MaxPooling2D(pool_size=pool_size[layer+1], strides=pool_size[layer+1]))
        model.add(Dropout(0.1))

    # reshape for R layers
    # flip frequency and time axes
    model.add(Flatten())
    #model.add(Permute((time_axis, frequency_axis, channel_axis)))
    #resize_shape = model.output_shape[2] * model.output_shape[3]
    #model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    #model.add(GRU(32, return_sequences=True))
    #model.add(GRU(32, return_sequences=False))
    #model.add(Dropout(0.3))

    # Dense Layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l=0.0001)))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model
