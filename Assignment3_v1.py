# Importing Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  models, optimizers, layers, activations,callbacks,optimizers
from tensorflow.keras.optimizers import Adam,Nadam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, GRU, Dense, Embedding,AdditiveAttention,Dropout,Concatenate
import matplotlib.pyplot as plt
import numpy as np
import random
import codecs
!pip install pybind11
!pip install fastwer
import fastwer
%pip install wandb -q
import wandb
from wandb.keras import WandbCallback

#======================================================================================#
def train():
  default_configs = {
    "embedding_size": 16, 
    "hidden_size": 32, 
    "encoder_layers": 1, 
    "decoder_layers": 1, 
    "architecture": "LSTM",
    "isAttention": True,
    "dropout": 0, 
    "batch_size": 32,
    "optimizer" : "Adam",
    "epochs" : 50
    }
  
  run = wandb.init(project='CS6910_Assignment_3', config=default_configs,settings=wandb.Settings(start_method="fork"))
  config = wandb.config
  
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
        self.batchSize = config.batch_size
        self.architecture = config.architecture
        
        self.dropout = config.dropout
        self.epochs = config.epochs

        self.isAttention = config.isAttention
        self.encoderLayers = config.encoder_layers
        self.decoderLayers = config.decoder_layers
        self.hiddenSize = config.hidden_size
        self.embeddingSize = config.embedding_size
        self.optimizer = config.optimizer
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
  if (NLP.isAttention == False):
    wandb.run.name = str(NLP.architecture)+"_el"+str(NLP.encoderLayers)+"_dl"+str(NLP.decoderLayers)+"_embedSize"+str(NLP.embeddingSize)+"_hs"+str(NLP.hiddenSize)+"_dropout"+str(NLP.dropout)+"_bs"+str(NLP.batchSize)+"_"+str(NLP.optimizer)
  else:
    wandb.run.name = "Attn_"+ str(NLP.architecture)+"_el"+str(NLP.encoderLayers)+"_dl"+str(NLP.decoderLayers)+"_embedSize"+str(NLP.embeddingSize)+"_hs"+str(NLP.hiddenSize)+"_dropout"+str(NLP.dropout)+"_bs"+str(NLP.batchSize)+"_"+str(NLP.optimizer)

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

#=========================================================================================================================================================#
# Training model
  model.compile(optimizer=NLP.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
  callback = EarlyStopping(monitor='val_loss', patience=5)
  model.fit([trainEncoderInput, trainDecoderInput],
      trainDecoderOutput,
      batch_size=NLP.batchSize,
      epochs=NLP.epochs,
      validation_data=([validEncoderInput, validDecoderInput],validDecoderOutput),
      callbacks=[callback,WandbCallback()]
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
# Decoding test sequences
  def decodeSequence(inputSeq):

    encoderOutputs, encoderStates= encoder_model.predict(inputSeq)
    if (NLP.architecture == 'LSTM'):
      initialState = [[np.zeros((1,NLP.hiddenSize))], [np.zeros((1,NLP.hiddenSize))]]
    else:
      initialState = [np.zeros((1,NLP.hiddenSize))]

    decoderStates = [encoderStates] + [initialState]*(NLP.decoderLayers-1)
    outputSeq = np.array(NLP.char2int_hi['<'],ndmin=2)

    decodedSeq = ""
    attWeightsList = []
    decodedChar = ""
    while not (decodedChar == NLP.EOS or (len(decodedSeq) > NLP.outTrainMaxlen)):
      if NLP.isAttention:
        decodedTokens, decoderStates, attWeights = decoder_model.predict([outputSeq, encoderOutputs, decoderStates])
        attWeightsList.append(attWeights)
      else:
        decodedTokens, decoderStates = decoder_model.predict([outputSeq, decoderStates])

      decodedTokenNumber = np.argmax(decodedTokens[0, -1, :])
      decodedChar = NLP.int2char_hi[decodedTokenNumber]
      decodedSeq += decodedChar
      outputSeq[0,0] = decodedTokenNumber
    if NLP.isAttention:
      return decodedSeq, attWeightsList
    else:
      return decodedSeq

#=========================================================================================================================================================#
# Predicting test data output after decoding sequences
  def testDataPrediction(testEncoderInput):
    ref = []
    hypo = [] 
    for seqIndex in range(len(testEncoderInput)):

        inputSeq = (testEncoderInput[seqIndex : seqIndex + 1]).reshape(1,NLP.inTrainMaxlen)
        if NLP.isAttention:
          decodedSeq, _ = decodeSequence(inputSeq)
        else:
          decodedSeq = decodeSequence(inputSeq)
        decodedLabel = decodedSeq.replace('>','')
        hypo.append(decodedLabel)

        referenceSeq = yTest[seqIndex]
        temp1 = referenceSeq.replace('>','')
        referenceLabel = temp1.replace('<','')

        if ((seqIndex%50) == 0 ):
            print("-----")
            print("Input Sequence:", xTest[seqIndex])
            print("Decoded Sequence:", decodedLabel)
            print("Reference Sequence:", referenceLabel)
        
        ref.append(referenceLabel)
    return ref,hypo

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

#=========================================================================================================================================================#
# Logging into wandb
  wandb.log({'Char Accuracy':(100-CER),
            'Word Accuracy':(100-WER)})

#=========================================================================================================================================================#
# Config file for sweep
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'Char Accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [50]
        },
        'batch_size':{
            'values':[32,64]
        },
        'Embedding_size': {
            'values': [16,32,64,128,256]
        },
        'encoder_layers': {
            'values': [1,2,3]
        },'decoder_layers':{
            'values': [1,2,3]
        },
        'hidden_size': {
            'values': [16,32,64,128,256]
        },'architecture':{
            'values': ['RNN','LSTM','GRU']
        },'optimizer':{
            'values': ['Adam','Nadam']
        },'dropout': {
            'values': [0,0.05,0.1,0.2]
        }
    }
}
