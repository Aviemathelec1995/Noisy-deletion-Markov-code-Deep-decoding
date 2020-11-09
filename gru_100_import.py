import numpy as np
import tensorflow as tf
import keras
from keras import models
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.layers.wrappers import  Bidirectional
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import LSTM, GRU, SimpleRNN
from keras.engine.topology import Layer
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Reshape
import keras.layers.core
f1=Bidirectional(GRU(name='bidirectional_1', units=100, activation='tanh', return_sequences=True, dropout=1.0))
f2= BatchNormalization(name='batch_normalization_1')
f3=Bidirectional(GRU(name='bidirectional_2', units=100, activation='tanh', return_sequences=True, dropout=1.0))
f4=BatchNormalization(name='batch_normalization_2')
f5=Bidirectional(GRU(name='bidirectional_3', units=100, activation='tanh', return_sequences=True, dropout=1.0))
f6=BatchNormalization(name='batch_normalization_3')
f7=Dense(1,activation='sigmoid')
f8=Dense(1,activation='softmax')
f10=Dense(1,activation='tanh')
f11=Lambda(lambda x: (x-0.5)*2)
r=Sequential()
r.add(f1)
r.add(f2)
r.add(f3)
r.add(f4)
r.add(f5)
r.add(f6)
r.add(f7)

optimizer= keras.optimizers.adam(lr=0.02,clipnorm=1.)
r.compile(optimizer=optimizer,loss='binary_crossentropy')

f11=Bidirectional(GRU(name='bidirectional_1', units=99, activation='tanh', return_sequences=True, dropout=0.1))
f21= BatchNormalization(name='batch_normalization_1')
f31=Bidirectional(GRU(name='bidirectional_2', units=99, activation='tanh', return_sequences=True, dropout=0.1))
f41=BatchNormalization(name='batch_normalization_2')
f51=Bidirectional(GRU(name='bidirectional_3', units=99, activation='tanh', dropout=0.1))
f61=BatchNormalization(name='batch_normalization_3')
f71=Dense(100,activation='sigmoid')
r99=Sequential()
r99.add(f11)
r99.add(f21)
r99.add(f31)
r99.add(f41)
r99.add(f51)
r99.add(f61)
r99.add(f71)
r99.compile(optimizer=optimizer,loss='binary_crossentropy')


f12=Bidirectional(GRU(name='bidirectional_1', units=98, activation='tanh', return_sequences=True, dropout=1.0))
f22= BatchNormalization(name='batch_normalization_1')
f32=Bidirectional(GRU(name='bidirectional_2', units=98, activation='tanh', return_sequences=True, dropout=1.0))
f42=BatchNormalization(name='batch_normalization_2')
f52=Bidirectional(GRU(name='bidirectional_3', units=98, activation='tanh', dropout=1.0))
f62=BatchNormalization(name='batch_normalization_3')
f72=Dense(100,activation='sigmoid')
r98 =Sequential()
r98.add(f12)
r98.add(f22)
r98.add(f32)
r98.add(f42)
r98.add(f52)
r98.add(f62)
r98.add(f72)
r98.compile(optimizer=optimizer,loss='binary_crossentropy')


f110=Bidirectional(GRU(name='bidirectional_1', units=90, activation='tanh', return_sequences=True, dropout=1.0))
f210= BatchNormalization(name='batch_normalization_1')
f310=Bidirectional(GRU(name='bidirectional_2', units=90, activation='tanh', return_sequences=True, dropout=1.0))
f410=BatchNormalization(name='batch_normalization_2')
f510=Bidirectional(GRU(name='bidirectional_3', units=90, activation='tanh', dropout=1.0))
f610=BatchNormalization(name='batch_normalization_3')
f710=Dense(100,activation='sigmoid')
r90 =Sequential()
r90.add(f110)
r90.add(f210)
r90.add(f310)
r90.add(f410)
r90.add(f510)
r90.add(f610)
r90.add(f710)
r90.compile(optimizer=optimizer,loss='binary_crossentropy')

