

```python
import os
import pickle

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model as plot
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model

# Hyper params
MAXLEN = 100
MAX_FEATURES = 200000
BATCH_SIZE = 32
MAX_NB_WORDS=1000
EPOCHS=50

MODEL_ARCH_FOLDER = 'model_architecture/'
MODEL_CHECKPOINT_FOLDER = 'checkpoint/'
TOKENIZER = 'tokenizer.pkl'
MODEL_NAME = 'weights-improvement-.hdf5'
print('Hyper params set')
import os
os.chdir('C:\\Users\\Jameel shaik\\Documents\\Projects\\Personalized Medicine Redefining Cancer Treatment')
source= 'C:\\Users\\Jameel shaik\\Documents\\Projects\\Personalized Medicine Redefining Cancer Treatment'

#train_variant = pd.read_csv(source+"/training_variants")

# Load train
data_train = pd.read_csv(source+'/training_text', sep='\|\|', engine='python', 
                      skiprows=1, names=['ID', 'Text']).set_index('ID').reset_index()
#data_train=data_train.drop(['|'],axis=1)
print('Training Data Loaded')

# Load variants
data_training_variants = pd.read_csv(source+'/training_variants')

print('Variants loaded')

```

    Using TensorFlow backend.


    Hyper params set
    Training Data Loaded
    Variants loaded



```python
!pip install keras
```

    Requirement already satisfied: keras in c:\program files\anaconda3\lib\site-packages
    Requirement already satisfied: pyyaml in c:\program files\anaconda3\lib\site-packages (from keras)
    Requirement already satisfied: theano in c:\program files\anaconda3\lib\site-packages (from keras)
    Requirement already satisfied: six in c:\program files\anaconda3\lib\site-packages (from keras)
    Requirement already satisfied: numpy>=1.9.1 in c:\program files\anaconda3\lib\site-packages (from theano->keras)
    Requirement already satisfied: scipy>=0.14 in c:\program files\anaconda3\lib\site-packages (from theano->keras)



```python
data_training_variants.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>FAM58A</td>
      <td>Truncating Mutations</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CBL</td>
      <td>W802*</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>CBL</td>
      <td>Q249E</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>CBL</td>
      <td>N454D</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>CBL</td>
      <td>L399V</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Merge X and Y
data_train_merge = pd.merge(data_train,data_training_variants,how='inner',on='ID')
print('Data merged')

# Dense layer can be encoded to 9. Add 1 to prediction class
data_train_merge['Class'] = data_train_merge['Class']-1
print('output variable differenced')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

print('Tokenizing...')
tokenizer.fit_on_texts(data_train_merge['Text'])


print('Converting text to sequences...')
sequences_train = tokenizer.texts_to_sequences(data_train_merge['Text'])

word_index = tokenizer.word_index

print('Preparing data...')
x = sequence.pad_sequences(sequences_train, maxlen=MAXLEN)
y = np.array(data_train_merge['Class'])

y_binary = to_categorical(y)

print('Split train and test...')
rng = np.random.RandomState(42)
n_samples = len(x)
indices = np.arange(n_samples)
rng.shuffle(indices)
x_shuffled = x[indices]
y_shuffled = y[indices]

x_train = x_shuffled[:int(n_samples*0.8)]
x_test = x_shuffled[int(n_samples*0.8):]

y_train = y_shuffled[:int(n_samples*0.8)]
y_test = y_shuffled[int(n_samples*0.8):]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Build model...')
model = Sequential()
model.add(Embedding(MAX_FEATURES, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='sigmoid'))

model_pretrained_path = MODEL_CHECKPOINT_FOLDER + MODEL_NAME
if os.path.exists(model_pretrained_path):
    model = load_model(model_pretrained_path)
else:
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    json_string = model.to_json()
    
print('Train...')

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_test, y_test))
```

    Data merged
    output variable differenced
    Tokenizing...
    Converting text to sequences...
    Preparing data...
    Split train and test...
    Build model...
    Train...
    Train on 111 samples, validate on 28 samples
    Epoch 1/50
    111/111 [==============================] - 14s - loss: 2.1918 - acc: 0.2613 - val_loss: 2.1834 - val_acc: 0.3571
    Epoch 2/50
    111/111 [==============================] - 4s - loss: 2.1631 - acc: 0.4685 - val_loss: 2.1569 - val_acc: 0.3214
    Epoch 3/50
    111/111 [==============================] - 4s - loss: 2.0951 - acc: 0.3964 - val_loss: 2.0432 - val_acc: 0.3214
    Epoch 4/50
    111/111 [==============================] - 4s - loss: 1.7891 - acc: 0.3964 - val_loss: 1.9861 - val_acc: 0.3214
    Epoch 5/50
    111/111 [==============================] - 4s - loss: 1.6305 - acc: 0.3964 - val_loss: 2.0161 - val_acc: 0.3214
    Epoch 6/50
    111/111 [==============================] - 4s - loss: 1.5828 - acc: 0.3964 - val_loss: 2.0515 - val_acc: 0.3214
    Epoch 7/50
    111/111 [==============================] - 4s - loss: 1.5607 - acc: 0.3964 - val_loss: 2.0574 - val_acc: 0.3214
    Epoch 8/50
    111/111 [==============================] - 4s - loss: 1.5450 - acc: 0.3964 - val_loss: 2.0521 - val_acc: 0.3214
    Epoch 9/50
    111/111 [==============================] - 4s - loss: 1.5427 - acc: 0.3964 - val_loss: 2.0549 - val_acc: 0.3214
    Epoch 10/50
    111/111 [==============================] - 3s - loss: 1.5348 - acc: 0.3964 - val_loss: 2.0812 - val_acc: 0.3214
    Epoch 11/50
    111/111 [==============================] - 4s - loss: 1.5242 - acc: 0.3964 - val_loss: 2.0905 - val_acc: 0.3214
    Epoch 12/50
    111/111 [==============================] - 3s - loss: 1.5191 - acc: 0.3964 - val_loss: 2.1123 - val_acc: 0.3214
    Epoch 13/50
    111/111 [==============================] - 3s - loss: 1.5024 - acc: 0.3964 - val_loss: 2.1080 - val_acc: 0.3214
    Epoch 14/50
    111/111 [==============================] - 3s - loss: 1.4946 - acc: 0.3964 - val_loss: 2.1054 - val_acc: 0.3214
    Epoch 15/50
    111/111 [==============================] - 3s - loss: 1.4825 - acc: 0.3964 - val_loss: 2.1026 - val_acc: 0.3214
    Epoch 16/50
    111/111 [==============================] - 3s - loss: 1.4713 - acc: 0.3964 - val_loss: 2.1023 - val_acc: 0.3214
    Epoch 17/50
    111/111 [==============================] - 3s - loss: 1.4431 - acc: 0.3964 - val_loss: 2.0912 - val_acc: 0.3214
    Epoch 18/50
    111/111 [==============================] - 3s - loss: 1.4131 - acc: 0.3964 - val_loss: 2.0738 - val_acc: 0.3214
    Epoch 19/50
    111/111 [==============================] - 3s - loss: 1.3758 - acc: 0.3964 - val_loss: 2.0618 - val_acc: 0.3214
    Epoch 20/50
    111/111 [==============================] - 3s - loss: 1.3446 - acc: 0.3964 - val_loss: 2.0375 - val_acc: 0.3214
    Epoch 21/50
    111/111 [==============================] - 3s - loss: 1.3161 - acc: 0.3964 - val_loss: 2.0315 - val_acc: 0.3214
    Epoch 22/50
    111/111 [==============================] - 3s - loss: 1.2714 - acc: 0.3964 - val_loss: 2.0585 - val_acc: 0.3214
    Epoch 23/50
    111/111 [==============================] - 4s - loss: 1.2114 - acc: 0.3964 - val_loss: 2.0801 - val_acc: 0.3214
    Epoch 24/50
    111/111 [==============================] - 4s - loss: 1.1805 - acc: 0.3964 - val_loss: 2.0807 - val_acc: 0.3214
    Epoch 25/50
    111/111 [==============================] - 4s - loss: 1.1367 - acc: 0.3964 - val_loss: 2.0777 - val_acc: 0.3214
    Epoch 26/50
    111/111 [==============================] - 4s - loss: 1.1028 - acc: 0.3964 - val_loss: 2.0795 - val_acc: 0.3214
    Epoch 27/50
    111/111 [==============================] - 4s - loss: 1.0448 - acc: 0.3964 - val_loss: 2.0934 - val_acc: 0.3214
    Epoch 28/50
    111/111 [==============================] - 4s - loss: 1.0033 - acc: 0.3964 - val_loss: 2.0233 - val_acc: 0.3214
    Epoch 29/50
    111/111 [==============================] - 4s - loss: 1.0083 - acc: 0.3964 - val_loss: 2.0095 - val_acc: 0.3214
    Epoch 30/50
    111/111 [==============================] - 4s - loss: 0.9611 - acc: 0.3964 - val_loss: 2.0485 - val_acc: 0.3214
    Epoch 31/50
    111/111 [==============================] - 4s - loss: 0.9176 - acc: 0.3964 - val_loss: 2.0297 - val_acc: 0.3214
    Epoch 32/50
    111/111 [==============================] - 4s - loss: 0.9278 - acc: 0.3964 - val_loss: 2.1209 - val_acc: 0.3214
    Epoch 33/50
    111/111 [==============================] - 4s - loss: 0.8651 - acc: 0.4054 - val_loss: 2.1556 - val_acc: 0.3571
    Epoch 34/50
    111/111 [==============================] - 4s - loss: 0.8431 - acc: 0.4505 - val_loss: 2.1020 - val_acc: 0.3571
    Epoch 35/50
    111/111 [==============================] - 3s - loss: 0.7903 - acc: 0.4865 - val_loss: 2.1466 - val_acc: 0.3571
    Epoch 36/50
    111/111 [==============================] - 3s - loss: 0.7672 - acc: 0.5045 - val_loss: 2.2207 - val_acc: 0.3571
    Epoch 37/50
    111/111 [==============================] - 3s - loss: 0.7371 - acc: 0.4955 - val_loss: 2.0951 - val_acc: 0.3571
    Epoch 38/50
    111/111 [==============================] - 4s - loss: 0.7470 - acc: 0.5045 - val_loss: 2.1422 - val_acc: 0.3571
    Epoch 39/50
    111/111 [==============================] - 4s - loss: 0.7000 - acc: 0.4955 - val_loss: 2.2601 - val_acc: 0.3571
    Epoch 40/50
    111/111 [==============================] - 3s - loss: 0.6690 - acc: 0.5315 - val_loss: 2.2017 - val_acc: 0.3571
    Epoch 41/50
    111/111 [==============================] - 3s - loss: 0.6376 - acc: 0.5856 - val_loss: 2.0367 - val_acc: 0.3929
    Epoch 42/50
    111/111 [==============================] - 3s - loss: 0.6271 - acc: 0.6757 - val_loss: 2.1254 - val_acc: 0.4286
    Epoch 43/50
    111/111 [==============================] - 4s - loss: 0.5887 - acc: 0.7748 - val_loss: 2.1620 - val_acc: 0.4286
    Epoch 44/50
    111/111 [==============================] - 4s - loss: 0.5405 - acc: 0.8378 - val_loss: 2.1074 - val_acc: 0.4643
    Epoch 45/50
    111/111 [==============================] - 4s - loss: 0.5025 - acc: 0.8739 - val_loss: 1.9829 - val_acc: 0.5000
    Epoch 46/50
    111/111 [==============================] - 4s - loss: 0.4354 - acc: 0.8919 - val_loss: 1.9443 - val_acc: 0.5000
    Epoch 47/50
    111/111 [==============================] - 4s - loss: 0.3952 - acc: 0.9099 - val_loss: 1.8854 - val_acc: 0.5000
    Epoch 48/50
    111/111 [==============================] - 3s - loss: 0.3433 - acc: 0.8919 - val_loss: 1.9065 - val_acc: 0.4643
    Epoch 49/50
    111/111 [==============================] - 3s - loss: 0.3218 - acc: 0.8919 - val_loss: 2.0085 - val_acc: 0.4643
    Epoch 50/50
    111/111 [==============================] - 4s - loss: 0.3134 - acc: 0.9099 - val_loss: 1.7983 - val_acc: 0.4643





    <keras.callbacks.History at 0x1128e7f0>


