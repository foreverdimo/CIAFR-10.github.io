# CIAFR-10  image-classfication project

Implemented by Yiyang Zhang 1800013111

Using **ResNet-50** model

About CIAFR-10 Datasets: 
[Datasets](https://www.cs.toronto.edu/~kriz/cifar.html )                         
[References](https://en.wikipedia.org/wiki/CIFAR-10)

## Import modules


```python
#import required packages#
import pickle
import numpy as np 
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, Input, Flatten, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

```

    Using TensorFlow backend.
    

## Load Datasets


```python
# If you have already downloaded the dataset and unpackaged it, make file_local true

file_local = True

#load dataset#
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data():
    
    #load train data
    X_train = []
    Y_train = []
    for i in range(1,6):
        train_batch = unpickle("data_batch_"+ str(i))
        X_orig = train_batch[b"data"]
        Y_orig = train_batch[b"labels"]
        X_processed = X_orig.reshape((10000,3,32,32)).transpose(0,2,3,1).astype('float32')
        Y_processed = to_categorical(np.array(Y_orig),10)
        X_train.append(X_processed)
        Y_train.append(Y_processed)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    
    #load test data
    test_batch = unpickle("test_batch")
    X_orig = test_batch[b"data"]
    Y_orig = test_batch[b"labels"]
    X_test = X_orig.reshape((10000,3,32,32)).transpose(0,2,3,1).astype('float32')
    Y_test = to_categorical(np.array(Y_orig),10)
    
    return X_train, Y_train, X_test, Y_test
    
if file_local :  
    X_train, Y_train, X_test, Y_test = load_data()

else :
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    Y_train = to_categorical(np.array(Y_train),10)
    Y_test = to_categorical(np.array(Y_test),10)
    
    
(M, n_H, n_W, n_C) = X_train.shape
input_shape = (n_H,n_W,n_C)


print ("data loading completed")
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


```

    data loading completed
    number of training examples = 50000
    number of test examples = 10000
    X_train shape: (50000, 32, 32, 3)
    Y_train shape: (50000, 10)
    X_test shape: (10000, 32, 32, 3)
    Y_test shape: (10000, 10)
    

## Build ResNet-50 Model

The following figure describes in detail the architecture of this  network
<img src="resnet_kiank.png" style="width:850px;height:150px;">

And firstly We will impletemt convolutional residual block and identity residual block
<img src="idblock3_kiank.png" style="width:650px;height:150px;">                       
<caption><center> Identity block </center></caption>

<img src="convblock_kiank.png" style="width:650px;height:150px;">                       
<caption><center> convolutional block </center></caption>



```python
def id_block(X, f, kernel_channels ,activation = 'relu'):
    
    X_shortcut = X
    
    F1,F2,F3 = kernel_channels

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation(activation)(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation(activation)(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)

    X = Add()([X,X_shortcut])
    X = Activation(activation)(X)
    
    return X


def conv_block(X, f, kernel_channels, strides, activation = 'relu'):
    
    X_shortcut = X
    
    F1,F2,F3 = kernel_channels
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (strides,strides), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation(activation)(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation(activation)(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (strides,strides), padding = 'valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut )

    X = Add()([X,X_shortcut])
    X = Activation(activation)(X)
    
    return X



def ResNet50(Input_shape = (32, 32, 3), classes = 10):
    
    X_input = Input(Input_shape)
    X = ZeroPadding2D((1, 1))(X_input)
    
    #stage 1
    X = Conv2D(64, (3, 3), strides = (2, 2), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    #stage 2
    
    X = conv_block(X , 3 , kernel_channels = [64, 64, 256], strides = 1)
    X = id_block(X, 3, kernel_channels = [64, 64, 256])
    X = id_block(X, 3, kernel_channels = [64, 64, 256])
    
    #stage 3
    
    X = conv_block(X, 3, kernel_channels = [128, 128, 512], strides = 2)
    X = id_block(X, 3, kernel_channels = [128, 128, 512])
    X = id_block(X, 3, kernel_channels = [128, 128, 512])
    X = id_block(X, 3, kernel_channels = [128, 128, 512])
    
    #stage 4
    
    X = conv_block(X, 3, kernel_channels = [256, 256, 1024], strides = 2)
    X = id_block(X, 3, kernel_channels = [256, 256, 1024])
    X = id_block(X, 3, kernel_channels = [256, 256, 1024])
    #X = id_block(X, 3, kernel_channels = [256, 256, 1024])
    #X = id_block(X, 3, kernel_channels = [256, 256, 1024])
    #X = id_block(X, 3, kernel_channels = [256, 256, 1024])
    
    #stage 5
    
    #X = conv_block(X, 3, kernel_channels = [512, 512, 2048], strides = 2)
    #X = id_block(X, 3, kernel_channels = [512, 512, 2048])
    #X = id_block(X, 3, kernel_channels = [512, 512, 2048])
    
    X = AveragePooling2D(pool_size = (2, 2))(X)
    
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(classes, activation='softmax')(X)
    
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model


```

## Train the model


```python
model = ResNet50(input_shape, classes= 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 200, batch_size = 256)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```

    Epoch 1/200
    50000/50000 [==============================] - 25s 495us/step - loss: 1.6875 - accuracy: 0.4302s - l
    Epoch 2/200
    50000/50000 [==============================] - 18s 370us/step - loss: 1.0912 - accuracy: 0.6119
    Epoch 3/200
    50000/50000 [==============================] - 19s 372us/step - loss: 0.8783 - accuracy: 0.6904s - loss: 0.8779 - accu
    Epoch 4/200
    50000/50000 [==============================] - 19s 375us/step - loss: 0.7256 - accuracy: 0.7444
    Epoch 5/200
    50000/50000 [==============================] - 19s 374us/step - loss: 0.6015 - accuracy: 0.7894
    Epoch 6/200
    50000/50000 [==============================] - 19s 375us/step - loss: 0.5069 - accuracy: 0.8236
    Epoch 7/200
    50000/50000 [==============================] - 19s 376us/step - loss: 0.4209 - accuracy: 0.8531
    Epoch 8/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.3496 - accuracy: 0.8776
    Epoch 9/200
    50000/50000 [==============================] - 19s 377us/step - loss: 0.2966 - accuracy: 0.8969
    Epoch 10/200
    50000/50000 [==============================] - 19s 377us/step - loss: 0.2379 - accuracy: 0.9174
    Epoch 11/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.2019 - accuracy: 0.9297
    Epoch 12/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.1793 - accuracy: 0.9367s
    Epoch 13/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.1479 - accuracy: 0.9485
    Epoch 14/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.1376 - accuracy: 0.9523
    Epoch 15/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.1274 - accuracy: 0.9557
    Epoch 16/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.1254 - accuracy: 0.9549
    Epoch 17/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.1010 - accuracy: 0.9651
    Epoch 18/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0926 - accuracy: 0.9676
    Epoch 19/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0942 - accuracy: 0.9669
    Epoch 20/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0794 - accuracy: 0.9731
    Epoch 21/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0776 - accuracy: 0.9731
    Epoch 22/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.0768 - accuracy: 0.9735
    Epoch 23/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0686 - accuracy: 0.9762
    Epoch 24/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0819 - accuracy: 0.9723
    Epoch 25/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0667 - accuracy: 0.9771
    Epoch 26/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0549 - accuracy: 0.9808
    Epoch 27/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0639 - accuracy: 0.9782
    Epoch 28/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0679 - accuracy: 0.9771
    Epoch 29/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0547 - accuracy: 0.9818
    Epoch 30/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0655 - accuracy: 0.9770
    Epoch 31/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.0610 - accuracy: 0.9792
    Epoch 32/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0516 - accuracy: 0.9830
    Epoch 33/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0570 - accuracy: 0.9802
    Epoch 34/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0501 - accuracy: 0.9830s - l
    Epoch 35/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0523 - accuracy: 0.9824
    Epoch 36/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0494 - accuracy: 0.9828
    Epoch 37/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0500 - accuracy: 0.9837
    Epoch 38/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0468 - accuracy: 0.9839s - loss: 0.0465 
    Epoch 39/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0471 - accuracy: 0.9841
    Epoch 40/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0447 - accuracy: 0.9845
    Epoch 41/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0446 - accuracy: 0.9848
    Epoch 42/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0490 - accuracy: 0.9836
    Epoch 43/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0416 - accuracy: 0.9863
    Epoch 44/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0371 - accuracy: 0.9876
    Epoch 45/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0389 - accuracy: 0.9865
    Epoch 46/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0375 - accuracy: 0.9870
    Epoch 47/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0513 - accuracy: 0.9826
    Epoch 48/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0331 - accuracy: 0.9883
    Epoch 49/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0362 - accuracy: 0.9875
    Epoch 50/200
    50000/50000 [==============================] - 19s 378us/step - loss: 0.0420 - accuracy: 0.9859
    Epoch 51/200
    50000/50000 [==============================] - 20s 398us/step - loss: 0.0332 - accuracy: 0.9890s - loss: 0.0330 - accu
    Epoch 52/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0265 - accuracy: 0.9916
    Epoch 53/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0311 - accuracy: 0.9894
    Epoch 54/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0374 - accuracy: 0.9876
    Epoch 55/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0364 - accuracy: 0.9878
    Epoch 56/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0361 - accuracy: 0.9880
    Epoch 57/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0349 - accuracy: 0.9884
    Epoch 58/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0307 - accuracy: 0.9896
    Epoch 59/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0347 - accuracy: 0.9882
    Epoch 60/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0260 - accuracy: 0.9911
    Epoch 61/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0303 - accuracy: 0.9893
    Epoch 62/200
    50000/50000 [==============================] - 20s 391us/step - loss: 0.0334 - accuracy: 0.9891
    Epoch 63/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.0309 - accuracy: 0.9900
    Epoch 64/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0252 - accuracy: 0.9916
    Epoch 65/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0302 - accuracy: 0.9899
    Epoch 66/200
    50000/50000 [==============================] - 20s 393us/step - loss: 0.0306 - accuracy: 0.9895
    Epoch 67/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0426 - accuracy: 0.9859
    Epoch 68/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0229 - accuracy: 0.9921
    Epoch 69/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0277 - accuracy: 0.9905
    Epoch 70/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0207 - accuracy: 0.9927
    Epoch 71/200
    50000/50000 [==============================] - 19s 386us/step - loss: 0.0246 - accuracy: 0.9916
    Epoch 72/200
    50000/50000 [==============================] - 20s 393us/step - loss: 0.0282 - accuracy: 0.9905
    Epoch 73/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0241 - accuracy: 0.9918s - loss: 0
    Epoch 74/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0237 - accuracy: 0.9921
    Epoch 75/200
    50000/50000 [==============================] - 19s 381us/step - loss: 0.0288 - accuracy: 0.9900
    Epoch 76/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.0199 - accuracy: 0.9931
    Epoch 77/200
    50000/50000 [==============================] - 19s 386us/step - loss: 0.0246 - accuracy: 0.9919
    Epoch 78/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0276 - accuracy: 0.9906
    Epoch 79/200
    50000/50000 [==============================] - 20s 409us/step - loss: 0.0176 - accuracy: 0.9944
    Epoch 80/200
    50000/50000 [==============================] - 23s 456us/step - loss: 0.0246 - accuracy: 0.9914
    Epoch 81/200
    50000/50000 [==============================] - 21s 427us/step - loss: 0.0213 - accuracy: 0.9925
    Epoch 82/200
    50000/50000 [==============================] - 21s 420us/step - loss: 0.0184 - accuracy: 0.9937
    Epoch 83/200
    50000/50000 [==============================] - 21s 412us/step - loss: 0.0209 - accuracy: 0.9936
    Epoch 84/200
    50000/50000 [==============================] - 21s 412us/step - loss: 0.0244 - accuracy: 0.9917
    Epoch 85/200
    50000/50000 [==============================] - 20s 407us/step - loss: 0.0230 - accuracy: 0.9919
    Epoch 86/200
    50000/50000 [==============================] - 20s 405us/step - loss: 0.0222 - accuracy: 0.9925
    Epoch 87/200
    50000/50000 [==============================] - 21s 410us/step - loss: 0.0166 - accuracy: 0.9945
    Epoch 88/200
    50000/50000 [==============================] - 20s 408us/step - loss: 0.0168 - accuracy: 0.9941
    Epoch 89/200
    50000/50000 [==============================] - 21s 416us/step - loss: 0.0285 - accuracy: 0.9906
    Epoch 90/200
    50000/50000 [==============================] - 20s 398us/step - loss: 0.0230 - accuracy: 0.9922s - loss: 0.0227 - accura
    Epoch 91/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0137 - accuracy: 0.9955
    Epoch 92/200
    50000/50000 [==============================] - 20s 410us/step - loss: 0.0200 - accuracy: 0.9935
    Epoch 93/200
    50000/50000 [==============================] - 21s 416us/step - loss: 0.0197 - accuracy: 0.9933
    Epoch 94/200
    50000/50000 [==============================] - 21s 417us/step - loss: 0.0189 - accuracy: 0.9935
    Epoch 95/200
    50000/50000 [==============================] - 22s 441us/step - loss: 0.0222 - accuracy: 0.9929
    Epoch 96/200
    50000/50000 [==============================] - 21s 419us/step - loss: 0.0196 - accuracy: 0.9934s - loss: 0.0195 - accuracy: 
    Epoch 97/200
    50000/50000 [==============================] - 21s 428us/step - loss: 0.0167 - accuracy: 0.9946
    Epoch 98/200
    50000/50000 [==============================] - 21s 424us/step - loss: 0.0205 - accuracy: 0.9933
    Epoch 99/200
    50000/50000 [==============================] - 21s 422us/step - loss: 0.0167 - accuracy: 0.9945
    Epoch 100/200
    50000/50000 [==============================] - 21s 426us/step - loss: 0.0247 - accuracy: 0.9918
    Epoch 101/200
    50000/50000 [==============================] - 20s 409us/step - loss: 0.0184 - accuracy: 0.9941
    Epoch 102/200
    50000/50000 [==============================] - 21s 418us/step - loss: 0.0146 - accuracy: 0.9954
    Epoch 103/200
    50000/50000 [==============================] - 21s 422us/step - loss: 0.0150 - accuracy: 0.9951
    Epoch 104/200
    50000/50000 [==============================] - 20s 397us/step - loss: 0.0158 - accuracy: 0.9948
    Epoch 105/200
    50000/50000 [==============================] - 20s 398us/step - loss: 0.0217 - accuracy: 0.9925
    Epoch 106/200
    50000/50000 [==============================] - 20s 395us/step - loss: 0.0140 - accuracy: 0.9952s - loss: 0.014
    Epoch 107/200
    50000/50000 [==============================] - 19s 388us/step - loss: 0.0132 - accuracy: 0.9955
    Epoch 108/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0231 - accuracy: 0.9922
    Epoch 109/200
    50000/50000 [==============================] - 20s 398us/step - loss: 0.0130 - accuracy: 0.9958
    Epoch 110/200
    50000/50000 [==============================] - 20s 399us/step - loss: 0.0183 - accuracy: 0.9941
    Epoch 111/200
    50000/50000 [==============================] - 20s 391us/step - loss: 0.0172 - accuracy: 0.9940
    Epoch 112/200
    50000/50000 [==============================] - 20s 396us/step - loss: 0.0169 - accuracy: 0.9945
    Epoch 113/200
    50000/50000 [==============================] - 20s 402us/step - loss: 0.0124 - accuracy: 0.9961
    Epoch 114/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0112 - accuracy: 0.9962
    Epoch 115/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0205 - accuracy: 0.9933
    Epoch 116/200
    50000/50000 [==============================] - 20s 399us/step - loss: 0.0144 - accuracy: 0.9954
    Epoch 117/200
    50000/50000 [==============================] - 20s 402us/step - loss: 0.0143 - accuracy: 0.9952
    Epoch 118/200
    50000/50000 [==============================] - 19s 390us/step - loss: 0.0128 - accuracy: 0.9954
    Epoch 119/200
    50000/50000 [==============================] - 20s 402us/step - loss: 0.0228 - accuracy: 0.9925
    Epoch 120/200
    50000/50000 [==============================] - 20s 405us/step - loss: 0.0107 - accuracy: 0.9962
    Epoch 121/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0145 - accuracy: 0.9955
    Epoch 122/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0178 - accuracy: 0.9940
    Epoch 123/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0161 - accuracy: 0.9948
    Epoch 124/200
    50000/50000 [==============================] - 19s 386us/step - loss: 0.0123 - accuracy: 0.9960
    Epoch 125/200
    50000/50000 [==============================] - 20s 392us/step - loss: 0.0145 - accuracy: 0.9949
    Epoch 126/200
    50000/50000 [==============================] - 20s 400us/step - loss: 0.0104 - accuracy: 0.9963
    Epoch 127/200
    50000/50000 [==============================] - 20s 396us/step - loss: 0.0170 - accuracy: 0.9943
    Epoch 128/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0160 - accuracy: 0.9950
    Epoch 129/200
    50000/50000 [==============================] - 19s 386us/step - loss: 0.0192 - accuracy: 0.9937
    Epoch 130/200
    50000/50000 [==============================] - 20s 395us/step - loss: 0.0077 - accuracy: 0.9974
    Epoch 131/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0041 - accuracy: 0.9985
    Epoch 132/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0206 - accuracy: 0.9934
    Epoch 133/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0125 - accuracy: 0.9958
    Epoch 134/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0093 - accuracy: 0.9970
    Epoch 135/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0142 - accuracy: 0.9951
    Epoch 136/200
    50000/50000 [==============================] - 19s 385us/step - loss: 0.0157 - accuracy: 0.9948
    Epoch 137/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0148 - accuracy: 0.9948
    Epoch 138/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0131 - accuracy: 0.9957
    Epoch 139/200
    50000/50000 [==============================] - 20s 393us/step - loss: 0.0097 - accuracy: 0.9967
    Epoch 140/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0150 - accuracy: 0.9949
    Epoch 141/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0172 - accuracy: 0.9941
    Epoch 142/200
    50000/50000 [==============================] - 19s 388us/step - loss: 0.0133 - accuracy: 0.9955
    Epoch 143/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0102 - accuracy: 0.9968
    Epoch 144/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0106 - accuracy: 0.9965s - l
    Epoch 145/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0110 - accuracy: 0.9964
    Epoch 146/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0123 - accuracy: 0.9958
    Epoch 147/200
    50000/50000 [==============================] - 19s 382us/step - loss: 0.0100 - accuracy: 0.9968
    Epoch 148/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0082 - accuracy: 0.9972
    Epoch 149/200
    50000/50000 [==============================] - 19s 380us/step - loss: 0.0130 - accuracy: 0.9956
    Epoch 150/200
    50000/50000 [==============================] - 19s 379us/step - loss: 0.0148 - accuracy: 0.9953
    Epoch 151/200
    50000/50000 [==============================] - 20s 392us/step - loss: 0.0093 - accuracy: 0.9969
    Epoch 152/200
    50000/50000 [==============================] - 20s 409us/step - loss: 0.0110 - accuracy: 0.9965
    Epoch 153/200
    50000/50000 [==============================] - 21s 418us/step - loss: 0.0123 - accuracy: 0.9959
    Epoch 154/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0097 - accuracy: 0.9965
    Epoch 155/200
    50000/50000 [==============================] - 19s 390us/step - loss: 0.0137 - accuracy: 0.9960
    Epoch 156/200
    50000/50000 [==============================] - 20s 392us/step - loss: 0.0092 - accuracy: 0.9971
    Epoch 157/200
    50000/50000 [==============================] - 20s 391us/step - loss: 0.0111 - accuracy: 0.9963
    Epoch 158/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0149 - accuracy: 0.9954
    Epoch 159/200
    50000/50000 [==============================] - 20s 393us/step - loss: 0.0102 - accuracy: 0.9967
    Epoch 160/200
    50000/50000 [==============================] - 20s 401us/step - loss: 0.0067 - accuracy: 0.9978
    Epoch 161/200
    50000/50000 [==============================] - 19s 383us/step - loss: 0.0108 - accuracy: 0.9967
    Epoch 162/200
    50000/50000 [==============================] - 19s 390us/step - loss: 0.0078 - accuracy: 0.9975
    Epoch 163/200
    50000/50000 [==============================] - 20s 400us/step - loss: 0.0122 - accuracy: 0.9960
    Epoch 164/200
    50000/50000 [==============================] - 20s 393us/step - loss: 0.0145 - accuracy: 0.9954s - l
    Epoch 165/200
    50000/50000 [==============================] - 20s 390us/step - loss: 0.0114 - accuracy: 0.9963
    Epoch 166/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0120 - accuracy: 0.9959
    Epoch 167/200
    50000/50000 [==============================] - 19s 388us/step - loss: 0.0081 - accuracy: 0.9973s - loss: 0.0080 - 
    Epoch 168/200
    50000/50000 [==============================] - 19s 388us/step - loss: 0.0117 - accuracy: 0.9961
    Epoch 169/200
    50000/50000 [==============================] - 19s 384us/step - loss: 0.0091 - accuracy: 0.9971
    Epoch 170/200
    50000/50000 [==============================] - 20s 397us/step - loss: 0.0078 - accuracy: 0.9976s - loss: 0.0075 
    Epoch 171/200
    50000/50000 [==============================] - 19s 389us/step - loss: 0.0123 - accuracy: 0.9961
    Epoch 172/200
    50000/50000 [==============================] - 20s 399us/step - loss: 0.0093 - accuracy: 0.9967
    Epoch 173/200
    50000/50000 [==============================] - 20s 392us/step - loss: 0.0323 - accuracy: 0.9904
    Epoch 174/200
    50000/50000 [==============================] - 22s 449us/step - loss: 0.0083 - accuracy: 0.9971
    Epoch 175/200
    50000/50000 [==============================] - 20s 407us/step - loss: 0.0059 - accuracy: 0.9983
    Epoch 176/200
    50000/50000 [==============================] - 20s 408us/step - loss: 0.0100 - accuracy: 0.9968
    Epoch 177/200
    50000/50000 [==============================] - 20s 401us/step - loss: 0.0165 - accuracy: 0.9949s - los
    Epoch 178/200
    50000/50000 [==============================] - 19s 387us/step - loss: 0.0123 - accuracy: 0.9959
    Epoch 179/200
    50000/50000 [==============================] - 21s 416us/step - loss: 0.0056 - accuracy: 0.9982
    Epoch 180/200
    50000/50000 [==============================] - 21s 410us/step - loss: 0.0093 - accuracy: 0.9970
    Epoch 181/200
    50000/50000 [==============================] - 20s 406us/step - loss: 0.0050 - accuracy: 0.9980
    Epoch 182/200
    50000/50000 [==============================] - 21s 421us/step - loss: 0.0053 - accuracy: 0.9982
    Epoch 183/200
    50000/50000 [==============================] - 20s 403us/step - loss: 0.0148 - accuracy: 0.9955
    Epoch 184/200
    50000/50000 [==============================] - 21s 412us/step - loss: 0.0134 - accuracy: 0.9959
    Epoch 185/200
    50000/50000 [==============================] - 21s 416us/step - loss: 0.0081 - accuracy: 0.9974s - loss: 0.0081 - accura
    Epoch 186/200
    50000/50000 [==============================] - ETA: 0s - loss: 0.0050 - accuracy: 0.9984 ETA: 0s - loss: 0.0051 - accu - 21s 414us/step - loss: 0.0050 - accuracy: 0.9984
    Epoch 187/200
    50000/50000 [==============================] - 22s 444us/step - loss: 0.0084 - accuracy: 0.9974
    Epoch 188/200
    50000/50000 [==============================] - 23s 457us/step - loss: 0.0102 - accuracy: 0.9968
    Epoch 189/200
    50000/50000 [==============================] - 22s 437us/step - loss: 0.0077 - accuracy: 0.9976
    Epoch 190/200
    50000/50000 [==============================] - 21s 420us/step - loss: 0.0083 - accuracy: 0.9972
    Epoch 191/200
    50000/50000 [==============================] - 21s 410us/step - loss: 0.0118 - accuracy: 0.9959s -
    Epoch 192/200
    50000/50000 [==============================] - 20s 409us/step - loss: 0.0071 - accuracy: 0.9977
    Epoch 193/200
    50000/50000 [==============================] - 20s 394us/step - loss: 0.0061 - accuracy: 0.9981
    Epoch 194/200
    50000/50000 [==============================] - 20s 410us/step - loss: 0.0063 - accuracy: 0.9980
    Epoch 195/200
    50000/50000 [==============================] - 20s 396us/step - loss: 0.0090 - accuracy: 0.9969
    Epoch 196/200
    50000/50000 [==============================] - 20s 399us/step - loss: 0.0080 - accuracy: 0.9973
    Epoch 197/200
    50000/50000 [==============================] - 20s 405us/step - loss: 0.0115 - accuracy: 0.9963s - l
    Epoch 198/200
    50000/50000 [==============================] - 21s 421us/step - loss: 0.0095 - accuracy: 0.9969
    Epoch 199/200
    50000/50000 [==============================] - 20s 400us/step - loss: 0.0063 - accuracy: 0.9977
    Epoch 200/200
    50000/50000 [==============================] - 20s 400us/step - loss: 0.0100 - accuracy: 0.9968
    10000/10000 [==============================] - 5s 523us/step
    Loss = 1.8030616061210631
    Test Accuracy = 0.7542999982833862
    


```python

```
