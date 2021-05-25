# CS6910_Assignment3
# Authors: EE19S015: Mudit Batra, EE20S008: Vrunda Sukhadia
Sequence to Sequence Model for Transliteration.

## Description of Files
1. This file was kept kept for practice "Assignment3_v1.py"
2. To replicate results with different configurations refer "Seq2Seq_main.py"

### "Seq2Seq_main.py"
The dataset has been given here:<br/>
Training Examples: 'train.txt'<br/>
Validation Examples: 'valid.txt'<br/>
Test Examples: 'test.txt'<br/>

To replicate the results download all the files along with Dataset.<br/><br/>
If you want to replicate results for particular configuration then no need to go with "Seq2Seq_main.py" file.We have seperatley defined all the hyperparameter
configuration in the "init block" so you can directly change the parameters in the "init block" using below description:

```
  def __init__(self):

      self.SOS = '<'
      self.EOS = '>'
      self.trainingExamples = len(self.xTrain)
      self.validExamples = len(self.xValid)
      self.batchSize = 32      #[32,64]
      self.architecture = 'RNN' #['LSTM','RNN','GRU']
      
      self.dropout = 0.2       #[0,0.1,0.2]
      self.epochs = 50         #   
      self.beam_search = 1     #[1,3,5]

      self.isAttention = True  #[True,False]
      self.encoderLayers = 1   #[1,2,3]
      self.decoderLayers = 1   #[1,2,3]
      self.hiddenSize = 20    #[32,64,128,256]
      self.embeddingSize = 32  #[32,64,128,256]
      self.optimizer = 'Nadam' #['Adam','Nadam']
```
All the Parameters that can be tuned are provided in comments<br/>
Note: To run Attention based models, keep self.isAttention = True

### "Results"
In this folder all results have been submitted.<br/>
1. Attention best model.txt : Results of all test Examples decoded by the Attentionbased best model.
2. Without attention best model.txt : Results of all test Examples decoded by the test model withou attention
3. Beam Search Results without attention.csv : Results of all test Examples decoded by the best model without attention with Beam width of 3
4. Attention Heatmap.jpg : Attention heatmap results on best atten based model

### [Check out our project workspace]()
### [Check out project detailed report]()
