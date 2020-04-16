import matplotlib.pyplot as plt
import numpy as np
import scipy


import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
#import theano as th
#import theano.tensor as T
import keras
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,Conv2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras


import keras.backend as K

class Cat():
    def __init__(self):
        print('init')
        self.f_BPSK = np.fromfile(open("/root/workspace/capston_folder/after_channel.dat"), dtype=np.float32)
        print('start cat!')

    def generate(self, samp_len):
        self.samp_len = samp_len
        self.f_BPSK = np.fromfile(open("/root/workspace/capston_folder/after_channel.dat"), dtype=np.float32)

        print(type(self.f_BPSK))
        print(self.f_BPSK.shape)

        self.saved_I_data = []
        self.saved_Q_data = []

        for i, v in enumerate(self.f_BPSK[:self.samp_len*500]):    
            if i % 2 == 0:
                self.saved_I_data.append(v)    
            else:
                self.saved_Q_data.append(v)

        self.saved_I_data = np.reshape(self.saved_I_data, (-1,self.samp_len))
        self.saved_Q_data = np.reshape(self.saved_Q_data, (-1,self.samp_len))
        self.test_BPSK=[]

        self.test_BPSK=np.reshape(self.test_BPSK, (-1,2,self.samp_len))
        self.test_BPSK = np.hstack([self.saved_I_data, self.saved_Q_data])
        self.test_BPSK=np.reshape(self.test_BPSK, (-1,2,self.samp_len))
        self.dict_test={}
        self.dict_test[('BPSK')]=self.test_BPSK;

        #print("dic_test :", self.dict_test)            
        print(len(self.saved_I_data)) # 25000
        print(len(self.saved_Q_data)) # 25000       

    def draw_graph(self,num):
        for i in range(self.test_BPSK.shape[0]):
            if i >= num:
                break
            self.saved_i_data = []
            self.saved_q_data = []
            for j in range(self.test_BPSK.shape[2]):
                self.saved_i_data.append(self.test_BPSK[i][0][j])
                self.saved_q_data.append(self.test_BPSK[i][1][j])
            plt.scatter(self.saved_i_data, self.saved_q_data)
            plt.grid(b=True)
            plt.show()



    def train_test(self):
        np.random.seed(2017)
        self.n_examples = self.test_BPSK.shape[0]
        self.n_train = int(self.n_examples * 0.5)
        self.train_idx = np.random.choice(range(0,self.n_examples), size=self.n_train, replace=False)
        self.test_idx = list(set(range(0,self.n_examples))-set(self.train_idx))
        self.X_train = self.test_BPSK[self.train_idx]
        self.X_test =  self.test_BPSK[self.test_idx]
        print(self.X_test)
            
        def to_onehot_test(yy):
            yy1 = np.zeros([len(yy), max(yy)+1])
            yy1[np.arange(len(yy)),yy] = 1
            return yy1
        def to_onehot_train(yy, max_value):
            self.max_value = max_value
            yy1 = np.zeros([len(yy), self.max_value])
            yy1[np.arange(len(yy)),yy] = 1
            return yy1
    
        self.mods = ['bpsk','qpsk','8psk','pam4','qam16','qam64','gfsk','cpfsk']
        self.yn_examples = self.test_BPSK.shape[0]
        self.yn_train = int(self.yn_examples*0.5)
        #self.a = np.random.choice(range(0,8), size=self.yn_train, replace=True)
        #self.b = np.random.choice(range(0,8), size=self.yn_train, replace=True)
        self.a = np.ones(self.X_train.shape[0], dtype='int')
        self.b = np.ones(self.X_train.shape[0], dtype='int')
        print(self.b)
        self.map_a = map(lambda x:self.mods.index(self.mods[x]), self.a)
        self.map_b = map(lambda x: self.mods.index(self.mods[x]), self.b)
        print(self.a)
        print(self.map_a)
        print(self.map_b)
        self.max_v = max(self.map_a) + 1
        self.Y_train = to_onehot_train(self.map_a,8)
        self.Y_test = to_onehot_train(self.map_b,8)
        print(self.Y_train)
        print(self.Y_train.shape[1])
        print(self.Y_test)
        print('X_test.shape : ', self.X_test.shape)
        print('Y_train.shape : ',self.Y_train.shape)
        print('train_test_complete!!!')
        self.in_shp = list(self.X_train.shape[1:])
        print(self.X_train.shape, self.in_shp)

    def cnn_model(self):

        self.dr = 0.5
        self.model = models.Sequential()
        self.model.add(Reshape(self.in_shp+[1], input_shape=self.in_shp))
        self.model.add(ZeroPadding2D((0,2)))
        self.model.add(Conv2D(64, (1,4), activation="relu"))
        self.model.add(Dropout(self.dr))
        self.model.add(ZeroPadding2D((0,2)))
        self.model.add(Conv2D(64, (2,4), activation="relu"))
        self.model.add(Dropout(self.dr))
        self.model.add(Conv2D(128, (1,8), activation="relu"))
        self.model.add(Dropout(self.dr))
        self.model.add(Conv2D(128, (1,8), activation="relu"))
        self.model.add(Dropout(self.dr))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(self.dr))
        self.model.add(Dense(8, activation='softmax'))
        self.model.add(Reshape([8]))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def training_session(self, epoch, batch_size):
        self.nb_epoch = epoch
        self.batch_size = batch_size

        self.filepath = 'weight_4layers.wts.h5'
        self.history = self.model.fit(self.X_train,
        self.Y_train,
        batch_size=self.batch_size,
        epochs=self.nb_epoch,
        verbose=2,
        validation_data=(self.X_test, self.Y_test),
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
            ])

    def score(self, batch_size):
        self.batch_size = batch_size
        self.score = self.model.evaluate(self.X_test, self.Y_test, verbose=1, batch_size = self.batch_size)
        print("self.score : ", self.score)

    def training_performance(self):
        plt.figure()
        plt.title('training performance')
        plt.plot(self.history.epoch, self.history.history['loss'], label='train loss + error')
        plt.plot(self.history.epoch, self.history.history['val_loss'], label='val_error')
        plt.legend()

    def load_model(self, url):
        from keras.models import load_model
        self.url = url
        self.pre_model =load_model(url)
        self.pre_model.summary()



    def plot_matrix(self):
        # Plot confusion matrix
        def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        self.test_Y_hat = self.model.predict(self.X_test, batch_size=1024)
        print(self.test_Y_hat)
        self.conf = np.zeros([len(self.mods),len(self.mods)])
        self.confnorm = np.zeros([len(self.mods),len(self.mods)])
        for i in range(0,self.X_test.shape[0]):
            j = list(self.Y_test[i,:]).index(1)
            k = int(np.argmax(self.test_Y_hat[i,:]))
            self.conf[j,k] = self.conf[j,k] + 1

        for i in range(0,len(self.mods)):
            self.confnorm[i,:] = self.conf[i,:] / np.sum(self.conf[i,:])
        print(self.confnorm)
        plot_confusion_matrix(self.confnorm, labels=self.mods)
