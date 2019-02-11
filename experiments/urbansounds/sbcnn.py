
# From https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from sklearn import metrics 

# to run this code, you'll need to load the following data: 
# train_x, train_y
# valid_x, valid_y
# test_x, test_y
# see http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/ for details

# data dimension parameters 
frames = 41
bands = 60
num_channels = 2
num_labels = test_y.shape[1]

# this model implements the 5 layer CNN described in https://arxiv.org/pdf/1608.04363.pdf
# be aware, there are 2 main differences: 
# the input is 60x41 data frames with 2 channels => (60,41,2) tensors 
# the paper seems to report using 128x128 data frames (with no mention of channels)
# the paper also uses a receptive field size of 5x5 - as our input is smaller, I'm using 3x3

f_size = 3

model = Sequential()

# Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the shape (24,1,f,f). 
# This is followed by (4,2) max-pooling over the last two dimensions and a ReLU activation function.
model.add(Convolution2D(24, f_size, f_size, border_mode='same', input_shape=(bands, frames, num_channels)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

# Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48,24,f,f). 
# Like L1 this is followed by (4,2) max-pooling and a ReLU activation function.
model.add(Convolution2D(48, f_size, f_size, border_mode='same'))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

# Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48, 48, f, f). 
# This is followed by a ReLU but no pooling.
model.add(Convolution2D(48, f_size, f_size, border_mode='valid'))
model.add(Activation('relu'))

# flatten output into a single dimension, let Keras do shape inference
model.add(Flatten())

# Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
model.add(Dense(64, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 5 - an output layer with one output unit per class, with L2 penalty, 
# followed by a softmax activation function
model.add(Dense(num_labels, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('softmax'))


# create a SGD optimiser
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

# a stopping function should the validation loss stop improving
earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

# compile and fit model, reduce epochs if you want a result faster
# the validation set is used to identify parameter settings (epoch) that achieves 
# the highest classification accuracy
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[earlystop], batch_size=32, nb_epoch=50)

# finally, evaluate the model using the withheld test dataset

# determine the ROC AUC score 
y_prob = model.predict_proba(test_x, verbose=0)
y_pred = np_utils.probas_to_classes(y_prob)
y_true = np.argmax(test_y, 1)
roc = metrics.roc_auc_score(test_y, y_prob)
print "ROC:", round(roc,3)

# determine the classification accuracy
score, accuracy = model.evaluate(test_x, test_y, batch_size=32)
print("\nAccuracy = {:.2f}".format(accuracy))

