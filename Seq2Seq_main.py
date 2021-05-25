# Importing Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  models, optimizers, layers, activations,callbacks,optimizers
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, GRU, Dense, Embedding,AdditiveAttention,Dropout,Concatenate
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from matplotlib.font_manager import FontProperties
from IPython.display import HTML as html_print
from IPython.display import display
import matplotlib.cm as cm
from time import sleep
from IPython.display import clear_output
import numpy as np
import random
import codecs
!pip install pybind11
!pip install fastwer
import fastwer
import math
from math import log
import csv

#=========================================================================================================================================================#
# Creating Classs
class NLP:
  xTrain = []
  yTrain = []
  xValid = []
  yValid = []
  xTest = []
  yTest =[]

#=========================================================================================================================================================#
# Initialisation    
  def __init__(self):

      self.SOS = '<'
      self.EOS = '>'
      self.trainingExamples = len(self.xTrain)
      self.validExamples = len(self.xValid)
      self.batchSize = 32      #[32,64]
      self.architecture = 'RNN' #['LSTM','RNN','GRU']
      
      self.dropout = 0.2       #[0,0.1,0.2]
      self.epochs = 1   
      self.beam_search = 1     #[1,3,5]

      self.isAttention = True  #[True,False]
      self.encoderLayers = 1   #[1,2,3]
      self.decoderLayers = 1   #[1,2,3]
      self.hiddenSize = 20    #[32,64,128,256]
      self.embeddingSize = 32  #[32,64,128,256]
      self.optimizer = 'Nadam' #['Adam','Nadam']

#=========================================================================================================================================================#
# Preparing charater level dictionary for input and output
  def getDictionary(self):
      
      enText = self.xTrain
      hiText = self.yTrain

      self.charText_hi = set(''.join(hiText))
      self.charText_hi.add(' ')
      self.charText_en = set(''.join(enText))
      self.charText_en.add(' ')

      self.int2char_hi = dict(enumerate(self.charText_hi))
      self.int2char_en = dict(enumerate(self.charText_en))

      self.char2int_hi = {char: ind for ind, char in self.int2char_hi.items()}
      self.char2int_en = {char: ind for ind, char in self.int2char_en.items()}

      self.vocabSize_hi = len(self.int2char_hi)
      self.vocabSize_en = len(self.int2char_en)

      self.inTrainMaxlen = len(max(self.xTrain, key=len))
      self.outTrainMaxlen = len(max(self.yTrain, key=len))

#=========================================================================================================================================================#
# Vectorizing word based on character level tokens
  def word2vec(self,dataset):
      
      if (dataset == 'train'):
          enText = self.xTrain
          hiText = self.yTrain

      elif(dataset == 'valid'):
          enText = self.xValid
          hiText = self.yValid 
      else:
          enText = self.xTest
          hiText = self.yTest

      
      self.encoderInput = np.zeros((len(enText), self.inTrainMaxlen), dtype="float32")
      self.decoderInput = np.zeros((len(hiText), self.outTrainMaxlen), dtype="float32")
      self.decoderOutput = np.zeros((len(hiText), self.outTrainMaxlen, self.vocabSize_hi), dtype="float32")

      for i, (x, y) in enumerate(zip(enText, hiText)):
          for t, char in enumerate(x):
              self.encoderInput[i, t] = self.char2int_en[char]
              
          self.encoderInput[i, t + 1 :] = self.char2int_en[" "]

          for t, char in enumerate(y):
              self.decoderInput[i, t] = self.char2int_hi[char]
              if t > 0:
                  self.decoderOutput[i, t - 1, self.char2int_hi[char]] = 1.0
                  
          self.decoderInput[i, t + 1 :] = self.char2int_hi[" "]
          self.decoderOutput[i, t:, self.char2int_hi[" "]] = 1.0  

#=========================================================================================================================================================#
# Passing architecture type
  def toggleFunc(self):
      if self.architecture == 'GRU':
          return GRU
      elif self.architecture == 'RNN':
          return SimpleRNN
      elif self.architecture == 'LSTM':
          return LSTM
      else:
          print('Please enter correct architecture')
          exit()



#=========================================================================================================================================================#
# Defining Model architecture using Encoder and Decoder layers
  def encoderDecoderModels(self):

      encoderInputs = Input(shape=(None,),name = 'encoder_input')
      encoderEmbedding = layers.Embedding(input_dim=self.vocabSize_en, output_dim=self.embeddingSize,mask_zero=True,name = 'encoder_embedding')(encoderInputs) 
      
      print(self.architecture)
      if (self.architecture == 'LSTM'):
          for i in range(self.encoderLayers):
            encoderEmbedding, state_h, state_c = self.toggleFunc()(self.hiddenSize, return_state=True,return_sequences=True, dropout=self.dropout,name ='encoder_layer'+str(i))(encoderEmbedding)
            encoderStates = [state_h, state_c]
      
      else:     
          for i in range(self.encoderLayers):
            encoderEmbedding, state_h = self.toggleFunc()(self.hiddenSize, return_state=True,return_sequences=True, dropout=self.dropout,name ='encoder_layer'+str(i))(encoderEmbedding)
            encoderStates = [state_h]   

      encoderOutputs = encoderEmbedding

    #=====================================================================================================================================================#
    # Defining Decoder layer  
