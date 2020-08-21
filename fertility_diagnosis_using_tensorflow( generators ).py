#!/usr/bin/env python
# coding: utf-8

# In[80]:


import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import DataFrame


# In[89]:


#finite data genrator function
def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        # say n = 1, batch_size = 10 then features[10:20], labels[10:20]
        yield (features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])

#infinite data generator function
def get_generator_cyclic2(features, labels, batch_size = 1):
    while True:
        for i in range(int(len(features)/batch_size)):
            batch_features = features[i*batch_size: (i+1)*batch_size]
            batch_labels   = labels[i*batch_size:(i+1)*batch_size]
            yield batch_features, batch_labels
    
def get_generator_cyclic(features, labels, batch_size = 1):
    while True:
        for i in range(int(len(features)/batch_size)):
            batch_features = features[i*batch_size: (i+1)*batch_size]
            batch_labels   = labels[i*batch_size:(i+1)*batch_size]
            yield batch_features, batch_labels
            
#building model            
def my_model(input_shape, output_shape):
    model_input = Input(input_shape)
    batch_1 = BatchNormalization(momentum=0.8)(model_input)
    dense_1 = Dense(100, activation='relu')(batch_1)
    batch_2 = BatchNormalization(momentum=0.8)(dense_1)
    output = Dense(1, activation='sigmoid')(batch_2)
    model = Model([model_input], output)
    return model


 
#training_evalutig_model
def train_and_evaluate_model(training_features, training_lables,validation_features, validation_lables,
                             input_shape,output_shape,batch_size, train_steps, epochs):
    model = my_model(input_shape, output_shape)
    #print("model type=", type(model1))
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
    
    
    train_generator_cyclic     = get_generator_cyclic(training_features, training_labels, batch_size=batch_size)
    validation_generator_cyclic = get_generator_cyclic(validation_features, validation_labels, batch_size= 30)
    
    print("======= training ============ ")
    model.fit_generator(train_generator_cyclic,  steps_per_epoch = train_steps, epochs = epochs, 
                        validation_data = validation_generator_cyclic, validation_steps = 1)
    
    print("======= evaluating ============ ")
    validation_generator = get_generator(validation_features, validation_labels, batch_size=30)
    model.evaluate(validation_generator)


# In[93]:


from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization

#dataset available at https://archive.ics.uci.edu/ml/datasets/Fertility from UC Irvine.
headers = ['Season', 'Age', 'Diseases', 'Trauma', 'Surgery', 'Fever', 'Alcohol', 'Smoking', 'Sitting', 'Output']
fertility = pd.read_csv('data/fertility_diagnosis.txt', delimiter=',', header=None, names=headers)
fertility['Output'] = fertility['Output'].map(lambda x : 0.0 if x=='N' else 1.0)
fertility = fertility.astype('float32')
# Shuffle

fertility = fertility.sample(frac=1).reset_index(drop=True)

#one-hot-encoding the season column
fertility = pd.get_dummies(fertility, prefix='Season', columns=['Season'])

#moving the label(output) col to last index
fertility.columns = [col for col in fertility.columns if col != 'Output'] + ['Output']

fertility = fertility.to_numpy()


#train,validation split
training = fertility[0:70]
validation = fertility[70:100]
training_features = training[:,0:-1]
training_labels = training[:,-1]
validation_features = validation[:,0:-1]
validation_labels = validation[:,-1]

input_shape = (12,)
output_shape = (1,)

batch_size = 5
train_steps = len(training) // batch_size
epochs = 3

train_and_evaluate_model(training_features, training_labels,validation_features, validation_labels,
                          input_shape,output_shape,batch_size, train_steps, epochs) 


# In[74]:





# In[75]:



    


# In[ ]:





# In[ ]:





# In[ ]:




