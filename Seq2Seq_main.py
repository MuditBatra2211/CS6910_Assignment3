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
      decoderInputs = Input(shape=(None,),name = 'decoder_input')
      decoderEmbedding = layers.Embedding(input_dim=self.vocabSize_hi, output_dim=self.embeddingSize,name = 'decoder_embedding')(decoderInputs)
      
      if (self.architecture == 'LSTM'):
        for i in range(self.decoderLayers):
          if (i == 0):
            decoderEmbedding,decoder_state_h,decoder_state_c = self.toggleFunc()(self.hiddenSize, return_sequences=True, return_state=True,dropout=self.dropout,name ='decoder_layer'+str(i))(decoderEmbedding,initial_state=encoderStates)
          else:
            decoderEmbedding,decoder_state_h,decoder_state_c = self.toggleFunc()(self.hiddenSize, return_sequences=True, return_state=True,dropout=self.dropout,name ='decoder_layer'+str(i))(decoderEmbedding,initial_state=None)      
          decoderStates = [decoder_state_h, decoder_state_c]  
      else:
        for i in range(self.decoderLayers):
          if (i == 0):
            decoderEmbedding,decoder_state_h = self.toggleFunc()(self.hiddenSize, return_sequences=True, return_state=True,dropout=self.dropout,name ='decoder_layer'+str(i))(decoderEmbedding,initial_state=encoderStates)
          else:
            decoderEmbedding,decoder_state_h= self.toggleFunc()(self.hiddenSize, return_sequences=True, return_state=True,dropout=self.dropout,name ='decoder_layer'+str(i))(decoderEmbedding,initial_state=None)      
          decoderStates = [decoder_state_h] 

      decoderOutputs = decoderEmbedding

      if self.isAttention:
        attContext, attWeights = AdditiveAttention(name = 'Attention')([decoderOutputs, encoderOutputs],return_attention_scores=True)
        decoderOutputs = Concatenate(name = 'Concatenate')([decoderOutputs, attContext])
      
      decoderOutputs = Dropout(self.dropout)(decoderOutputs,training = True)
      decoderDense = Dense(self.vocabSize_hi, activation="softmax",name = 'decoder_dense')
      decoderOutputs = decoderDense(decoderOutputs)
      
      model = Model([encoderInputs, decoderInputs], decoderOutputs)
      return model
#=========================================================================================================================================================#
# Creating Object  
NLP = NLP()

#=========================================================================================================================================================#
# Preprocessing Data  
def getInputData(filename):
  hiText,enText = [],[]
  with codecs.open(filename, encoding='utf-8') as f:
    for row in f:
      hiWord, enWord, _ = row.split("\t")
      hiWord = NLP.SOS + hiWord + NLP.EOS
      hiText.append(hiWord)
      enText.append(enWord)  
  return enText,hiText

x_Train,y_Train = getInputData('train.txt')
xValid,yValid = getInputData('valid.txt')
xTest,yTest = getInputData('test.txt')
#=========================================================================================================================================================#
# Shuffling Train Data
temp = list(zip(x_Train, y_Train))
random.shuffle(temp)
xTrain, yTrain = zip(*temp)

NLP.xTrain = xTrain
NLP.yTrain = yTrain
NLP.xValid = xValid
NLP.yValid = yValid
NLP.xTest = xTest
NLP.yTest = yTest

NLP.getDictionary()
NLP.word2vec('train')
trainEncoderInput,trainDecoderInput,trainDecoderOutput = NLP.encoderInput,NLP.decoderInput,NLP.decoderOutput
NLP.word2vec('valid')
validEncoderInput,validDecoderInput,validDecoderOutput = NLP.encoderInput,NLP.decoderInput,NLP.decoderOutput
NLP.word2vec('test')
testEncoderInput,testDecoderInput,testDecoderOutput = NLP.encoderInput,NLP.decoderInput,NLP.decoderOutput
model = NLP.encoderDecoderModels()
plot_model(model, to_file='model.png', show_shapes=True)
#=========================================================================================================================================================#
# Training model
model.compile(optimizer=NLP.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
callback = EarlyStopping(monitor='val_loss', patience=5)
model.fit([trainEncoderInput, trainDecoderInput],
    trainDecoderOutput,
    batch_size=NLP.batchSize,
    epochs=NLP.epochs,
    validation_data=([validEncoderInput, validDecoderInput],validDecoderOutput),
    callbacks=[callback]
)
#=========================================================================================================================================================#
# Inference Model 
def inferenceModel(model):
  encoderInputs = model.input[0]
  if (NLP.architecture=='LSTM'):
      encoderOutputs, state_h_enc, state_c_enc = model.layers[NLP.encoderLayers+3].output
      encoderStates = [state_h_enc, state_c_enc]
  else:
      encoderOutputs, state_h_enc = model.layers[NLP.encoderLayers+3].output
      encoderStates = [state_h_enc]
  encoder_model = Model(encoderInputs, [encoderOutputs, encoderStates])

  decoderInputs = model.input[1]

  attEncoderOutput = Input(shape=(NLP.inTrainMaxlen,NLP.hiddenSize,), name="Encoder Output for Attention")
  decoderInputEmbedding = model.layers[NLP.encoderLayers+2](decoderInputs)
  decoderOutputs = decoderInputEmbedding
  inStatesDecoder = []
  outStatesDecoder = []
  
  decoderStartingLayer = NLP.encoderLayers+4
  attnNumber = decoderStartingLayer+NLP.decoderLayers
  for i in range(NLP.decoderLayers):
    
    if (NLP.architecture=='LSTM'):

        state_h_dec = Input(shape=(NLP.hiddenSize,))
        state_c_dec = Input(shape=(NLP.hiddenSize,))
        inStatesDecoder.append([state_h_dec, state_c_dec])          
        decoderOutputs,state_h_dec,state_c_dec = model.layers[decoderStartingLayer+i](decoderOutputs, initial_state = inStatesDecoder[-1])
        outStatesDecoder.append([state_h_dec, state_c_dec])
    
    else:
        state_h_dec = Input(shape=(NLP.hiddenSize,))
        inStatesDecoder.append([state_h_dec])
        decoderOutputs,state_h_dec= model.layers[decoderStartingLayer+i](decoderOutputs, initial_state = inStatesDecoder[i])
        outStatesDecoder.append([state_h_dec])
    
  if NLP.isAttention:
    attContext, attWeights = model.layers[attnNumber]([decoderOutputs, attEncoderOutput], return_attention_scores=True)
    decoderOutputs = model.layers[attnNumber+1]([decoderOutputs, attContext])
    attnNumber += 2
  
  # Making Dropout and Dense layer
  decoderOutputs = model.layers[attnNumber](decoderOutputs, training=False)
  decoderOutputs = model.layers[attnNumber+1](decoderOutputs)

  if NLP.isAttention:
    decoder_model = Model([decoderInputs, attEncoderOutput, inStatesDecoder],[decoderOutputs, outStatesDecoder, attWeights])
  else:
    decoder_model = Model([decoderInputs, inStatesDecoder], [decoderOutputs, outStatesDecoder])

  return encoder_model, decoder_model
#=========================================================================================================================================================#
# Decoding particular sequences that are given from test data set

def decodeSequence(beam_search,inputSeq):
  encoderOutputs, encoderStates= encoder_model.predict(inputSeq)
  if (NLP.architecture == 'LSTM'):
    initialState = [[np.zeros((1,NLP.hiddenSize))], [np.zeros((1,NLP.hiddenSize))]]
  else:
    initialState = [np.zeros((1,NLP.hiddenSize))]

  decoderStates = [encoderStates] + [initialState]*(NLP.decoderLayers-1)
  outputSeq = np.array(NLP.char2int_hi['<'],ndmin=2)

  decodedSeq = ""
  attWeightsList = []
  decoded_prob =[]
  decodedChar = ""
  outputSequences =[]
  scores = []
  sequences = [[list(), 0.0]]

  while not (decodedChar == NLP.EOS or (len(decodedSeq) > NLP.outTrainMaxlen)):
    if NLP.isAttention:
      decodedTokens, decoderStates, attWeights = decoder_model.predict([outputSeq, encoderOutputs, decoderStates])
      attWeightsList.append(attWeights)
    else:
      decodedTokens, decoderStates = decoder_model.predict([outputSeq, decoderStates])

    decodedTokenNumber = np.argmax(decodedTokens[0, -1, :])
    decodedChar = NLP.int2char_hi[decodedTokenNumber]

    decoded_prob.append(decodedTokens[0, 0])

    decodedSeq += decodedChar
    outputSeq[0,0] = decodedTokenNumber
#=========================================================================================================================================================#
# To predict the all words in given test data

def testDataPrediction(testEncoderInput):
  ref = []
  hypo = [] 
  beam_search = NLP.beam_search
 
  for seqIndex in range(10):

    referenceSeq = yTest[seqIndex]
    temp1 = referenceSeq.replace('>','')
    referenceLabel = temp1.replace('<','')
    
    inputSeq = (testEncoderInput[seqIndex : seqIndex + 1]).reshape(1,NLP.inTrainMaxlen)
    if NLP.isAttention:
      decodedSeq, _ = decodeSequence(beam_search,inputSeq)
    else:
      decodedSeq = decodeSequence(beam_search,inputSeq)
    decodedLabel = decodedSeq.replace('>','')
    hypo.append(decodedLabel)


    if ((seqIndex%1) == 0 ):
        print("-----")
        print("Input Sequence:", xTest[seqIndex])
        print("Decoded Sequence:", decodedLabel)
        print("Reference Sequence:", referenceLabel)
      
    ref.append(referenceLabel)
  return ref,hypo
  for row in decoded_prob:
    all_candidates = list()
    for i in range(len(sequences)):
      seq, score = sequences[i]
      for j in range(len(row)):
        candidate = [seq + [j], score - log(row[j])]
        all_candidates.append(candidate)

    ordered = sorted(all_candidates, key=lambda tup:tup[1])
    # select k best
    sequences = ordered[:beam_search]
  for i in range(beam_search):
    for j in range(len(sequences[i][0])):
      sequences[i][0][j] = NLP.int2char_hi[sequences[i][0][j]]
    temp = sequences[i][0]
    str_temp = "".join(temp)
    str_temp = str_temp.replace('>','')
    outputSequences.append(str_temp)   
    scores.append(sequences[i][1])
  print(outputSequences,scores)
  if (NLP.isAttention == True):
    return outputSequences[0],attWeightsList
  else:
    return outputSequences[0]
#=========================================================================================================================================================#
# Computing Word Error rate and Character Error rate
def computeCerWer(ref,hypo):
  
  WER = fastwer.score(hypo, ref)
  print('WER : ',WER)
  CER = fastwer.score(hypo, ref, char_level=True)
  print('CER : ', CER)
  return WER, CER

#=========================================================================================================================================================#
# Joining full pipeline
encoder_model,decoder_model = inferenceModel(model)
ref,hypo = testDataPrediction(testEncoderInput)
WER,CER = computeCerWer(ref,hypo)
# decoding particular word and plotting it's attention heatmap

def decode_word(word):
  encoderInput_1 = np.zeros((1, NLP.inTrainMaxlen), dtype="float32")
  for t, char in enumerate(word):
      encoderInput_1[0, t] = NLP.char2int_en[char]
      
  encoderInput_1[0, t + 1 :] = NLP.char2int_en[" "]
  return encoderInput_1

number = 2                                   # number of words to decode
temp = list(zip(xTest, yTest))
random.shuffle(temp)          
xTest, yTest= zip(*temp)

random_word_list = []
decodedLabel_list = []
referenceLabel_list =[]

for i in range(number):
  random_word = xTest[i]
  xx_attn = decode_word(random_word)
  inputSeq = xx_attn.reshape(1,NLP.inTrainMaxlen)
  if NLP.isAttention:
    decodedSeq, attnWeights = decodeSequence(NLP.beam_search,inputSeq)
  else:
    decodedSeq = decodeSequence(inputSeq)
  decodedLabel = decodedSeq.replace('>','')
  

  referenceSeq = yTest[i]
  temp1 = referenceSeq.replace('>','')
  referenceLabel = temp1.replace('<','')

  decodedLabel_list.append(decodedLabel) 
  referenceLabel_list.append(referenceLabel)
  random_word_list.append(random_word)

  print("-----")
  print("Input Word:",random_word)
  print("Decoded Word:", decodedLabel)
  print("Refernce Word: ", referenceLabel)
#=========================================================================================================================================================#
# plotting attention heatmap

  len_input = len(random_word)
  attnWeights = attnWeights[0:len(attnWeights)-1]
  attnWeights = np.array([ element[0,0,:len_input] for element in attnWeights ])

  font_prop = FontProperties(fname="Akshar Unicode Regular.ttf", size=25)

  fig, ax = plt.subplots()
  en_char = [ch for ch in random_word]
  hi_char = [ch for ch in decodedLabel]
  plt.imshow(attnWeights, cmap='viridis', interpolation='bilinear')
  ax.set_xticks(range(len(random_word)))
  ax.set_yticks(range(len(decodedLabel)))
  ax.set_xticklabels(en_char,size = 25)
  ax.set_yticklabels(hi_char, fontproperties=font_prop)
  plt.colorbar()
  plt.show()
  plt.savefig('%s.png' %(random_word))

# Tried Connectivity question but not not completed 
'''
 # get html element
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)
	
# print html
def print_color(t):
	display(html_print(''.join([cstr(ti, color=ci) for ti,ci in t])))

# get appropriate color for value
def get_clr(value):
	colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
	value = int((value * 100) / 5)
	return colors[value]

# sigmoid function
def sigmoid(x):
	z = 1/(1 + np.exp(-x)) 
	return z
  
#Computing Gradient for findding connectivity at each step
def encoderOutputFromEmbedding(model, x):
  if (NLP.architecture=='LSTM'):
      encoderOutputs, state_h_enc, state_c_enc = model.layers[NLP.encoderLayers+3](x)
      encoderStates = [state_h_enc, state_c_enc]
  else:
      encoderOutputs, state_h_enc = model.layers[NLP.encoderLayers+3](x)
      encoderStates = [state_h_enc]
  return encoderOutputs,encoderStates

def DecoderOutput(model, x, hidden, enc_out=None):
  decoderInputEmbedding = model.layers[NLP.encoderLayers+2](x)
  decoderOutputs = decoderInputEmbedding
  inStatesDecoder = []
  outStatesDecoder = []
  
  decoderStartingLayer = NLP.encoderLayers+4
  attnNumber = decoderStartingLayer+NLP.decoderLayers
  attEncoderOutput = Input(shape=(NLP.inTrainMaxlen,NLP.hiddenSize,))
  inStatesDecoder = hidden 
  for i in range(NLP.decoderLayers):
  
    if (NLP.architecture=='LSTM'):
     
      decoderOutputs,state_h_dec,state_c_dec = model.layers[decoderStartingLayer+i](decoderOutputs, initial_state = inStatesDecoder[-1])
      outStatesDecoder.append([state_h_dec, state_c_dec])
      inStatesDecoder.append([state_h_dec, state_c_dec]) 
    
    else:

      decoderOutputs,state_h_dec= model.layers[decoderStartingLayer+i](decoderOutputs, initial_state = inStatesDecoder[-1])
      outStatesDecoder.append([state_h_dec])
      inStatesDecoder.append([state_h_dec])

  if NLP.isAttention:
    attContext, attWeights = model.layers[attnNumber]([decoderOutputs, attEncoderOutput], return_attention_scores=True)
    decoderOutputs = model.layers[attnNumber+1]([decoderOutputs, attContext])

  

  return decoderOutputs,outStatesDecoder[-1],attWeights
#=========================================================================================================================================================#
import tensorflow as tf
from keras import backend as K
import math
from math import log
# tf.compat.v1.disable_eager_execution()


seqIndex = 1
inputSeq = (testEncoderInput[seqIndex : seqIndex + 1]).reshape(1,NLP.inTrainMaxlen)
decodedChar = ""
decodedSeq = ""
gradient_list =[]
if (NLP.architecture == 'LSTM'):
  initialState = [[np.zeros((1,NLP.hiddenSize))], [np.zeros((1,NLP.hiddenSize))]]
else:
  initialState = [np.zeros((1,NLP.hiddenSize))]
encoderEmbedding = model.layers[2](inputSeq)
# embed_array = encoderEmbedding.numpy()
# print(type(embed_array))
# encoderEmbedding = tf.convert_to_tensor(embed_array)
with tf.GradientTape(persistent=True, watch_accessed_variables =False) as tape:
  tape.watch(encoderEmbedding)
  encOut, encState = encoderOutputFromEmbedding(model, encoderEmbedding)

  decState = [encState] + [initialState]*(NLP.decoderLayers-1)
  decoderInputs = np.array(NLP.char2int_hi['<'],ndmin=2)

  while not (decodedChar == NLP.EOS or (len(decodedSeq) > NLP.outTrainMaxlen)):
    print('entered')
    decOut, decState,_ = DecoderOutput(model, decoderInputs, dec_state, enc_out)
    decOut = tf.make_tensor_proto(decOut)
    decoderOutputs_lstm = model.layers[-2](decOut, training=False)
    decodeTokens = model.layers[-1](decoderOutputs_lstm)
    
    gradient_list.append(tape.gradient(decOut, encoderEmbedding))
    decodeTokens = np.argmax(decodeTokens[0, -1, :])
    decodedChar = NLP.int2char_hi[decodeTokens]
    decodedSeq += decodedChar
    outputSeq[0,0] = decodeTokens
'''
