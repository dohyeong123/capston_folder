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
import numpy as np
import math


import keras.backend as K

class Cat():
    def __init__(self):
        print('init')
        self.f_BPSK = np.fromfile(open("/root/workspace/capston_folder/after_channel.dat"), dtype=np.float32)
        print('start cat!')

    def generate(self, samp_len, mod_names):
        self.mod_names = mod_names
        self.samp_len = samp_len
        self.f_BPSK = np.fromfile(open("/root/workspace/capston_folder/after_channel.dat"), dtype=np.float32)
        print(self.f_BPSK.shape)

        self.pre_I = []
        self.pre_Q = []
        ## shaping (?,2,128)
        for i, v in enumerate(self.f_BPSK[:self.samp_len*1400]):    
            if i % 2 == 0:
                self.pre_I.append(v)    
            else:
                self.pre_Q.append(v)
        
        self.pre_I = np.reshape(self.pre_I, (-1,self.samp_len))
        self.pre_Q = np.reshape(self.pre_Q, (-1,self.samp_len))
        
        ##IQ_data normalization
        self.isquared=np.power(self.pre_I, 2.0)
        self.qsquared=np.power(self.pre_Q, 2.0)
        self.energy=np.sqrt(self.isquared+self.qsquared)
        self.post_I=np.zeros((self.energy.shape[0], self.energy.shape[1]))
        self.post_Q=np.zeros((self.energy.shape[0], self.energy.shape[1]))
        self.total_energy=0
        for i in range(0, self.energy.shape[0]):
            for j in range (0, self.energy.shape[1]):
                self.total_energy=self.total_energy+self.energy[i][j]
            self.post_I[i]=self.pre_I[i]/self.total_energy
            self.post_Q[i]=self.pre_Q[i]/self.total_energy
            self.total_energy=0
        
        self.test_BPSK=[]
        self.test_BPSK=np.reshape(self.test_BPSK, (-1,2,self.samp_len))
        self.test_BPSK = np.hstack([self.post_I, self.post_Q])
        self.test_BPSK=np.reshape(self.test_BPSK, (-1,2,self.samp_len))
        print(" test_mod : ", self.test_BPSK.shape)
        self.size=(1, 2, self.test_BPSK.shape[2])
        self.data_zero=np.zeros(self.size)
        self.dict_test={}
        
        ## make dictionary type dataset
        self.dict_test[('QAM16')]=self.data_zero
        self.dict_test[('BPSK')]=self.data_zero
        self.dict_test[('8PSK')]=self.data_zero
        self.dict_test[('CPFSK')]=self.data_zero
        self.dict_test[('GFSK')]=self.data_zero
        self.dict_test[('PAM4')]=self.data_zero
        self.dict_test[('QPSK')]=self.data_zero
        self.dict_test[('QAM64')]=self.data_zero
        self.dict_test[('AM-DSB')]=self.data_zero
        self.dict_test[('AM-SSB')]=self.data_zero
        self.dict_test[('WBFM')]=self.data_zero
        self.dict_test[(self.mod_names)]=self.test_BPSK

    def train_test(self):
        self.Xd_test=self.dict_test
        self.mods_new = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
        self.X_new = []
        self.lbl_new = []
        print("mod : ", self.mods_new)
        for mod in self.mods_new:    
            self.X_new.append(self.Xd_test[(mod)])
            for i in range(self.Xd_test[(mod)].shape[0]):
                self.lbl_new.append((mod))
        
        self.X_new = np.vstack(self.X_new)
        np.random.seed(2017)
        self.n_examples = self.X_new.shape[0]
        self.n_train = int(self.n_examples)
        self.test_idx_new = np.random.choice(range(0,self.n_examples), size=self.n_train, replace=False)
        self.X_test_new = self.X_new[self.test_idx_new]

        def to_onehot(yy):
            yy1 = np.zeros([len(yy), max(yy)+1])
            yy1[np.arange(len(yy)),yy] = 1
            return yy1

        self.Y_test_new= to_onehot(list(map(lambda x: self.mods_new.index(self.lbl_new[x]), self.test_idx_new)))
        self.in_shp = list(self.X_test_new.shape[1:])

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
        self.model.add(Dense(11, activation='softmax'))
        self.model.add(Reshape([11]))
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
        self.score = self.pre_model.evaluate(self.X_test_new, self.Y_test_new, verbose=1, batch_size = self.batch_size)
        print("self.score : ", self.score)

    def load_model(self, url):
        from keras.models import load_model
        self.url = url
        self.pre_model =load_model(url)
        self.pre_model.summary()



    def plot_matrix(self):
        ## Define Plot confusion matrix
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

        self.test_Y_hat = self.pre_model.predict(self.X_test_new, batch_size=400)
        self.conf = np.zeros([len(self.mods_new),len(self.mods_new)])
        self.confnorm = np.zeros([len(self.mods_new),len(self.mods_new)])
        for i in range(0,self.X_test_new.shape[0]):
            j = list(self.Y_test_new[i,:]).index(1)
            k = int(np.argmax(self.test_Y_hat[i,:]))
            self.conf[j,k] = self.conf[j,k] + 1

        for i in range(0,len(self.mods_new)):
            self.confnorm[i,:] = self.conf[i,:] / np.sum(self.conf[i,:])
        ## express accuracy by 11*11
        #print(self.confnorm)

        ## express 11*11 confusion matrix
        #plot_confusion_matrix(self.confnorm, labels=self.mods_new)
        
        cnt = 0
        for test in range(len(self.mods_new)):
            if self.mods_new[test] == self.mod_names:
                print("True_mode : ",self.mod_names)
                max_value = 0
                max_index = 0
                for test_2 in range(len(self.mods_new)):
                    if max_value < self.confnorm[cnt][test_2]:
                        max_value = self.confnorm[cnt][test_2]
                        max_index = test_2        
                print("max_value : ",max_value)
                print("predict_mode : ",self.mods_new[max_index])
                max_value=0
            else :
                cnt += 1